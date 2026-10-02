//! Chest and hopper block entities: container storage, the open/close
//! menu choreography, hopper transfers, comparator fill levels, and the
//! block-entity data on the wire (chunk `block_entities` arrays plus the
//! per-position sync packet).
//!
//! Registry ids below come from the pinned vanilla jar's data-generator
//! reports, the same source as `pins/blocks.json` and the item table.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{bail, Context, Result};
use doppel_protocol::{write_varint, Reader};

use super::{prop_dir, prop_value, ConnId, Game};
use crate::inventory::{
    apply_click, encode_container_set_content, item_name, ContainerClick, ItemStack, MenuSession,
    MenuSlots, PlayerInventory, ABSOLUTE_MAX_STACK_SIZE, PACKET_CONTAINER_SET_CONTENT,
};
use doppel_world::chunk_codec::WireBlockEntity;

// ---------------------------------------------------------------------
// Wire ids and registry ids (26.3 data-generator reports)
// ---------------------------------------------------------------------

/// `block_entity_data`: BlockPos, block-entity type VarInt, one trusted
/// NBT compound.
pub const PACKET_BLOCK_ENTITY_DATA: i32 = 0x06;
/// `block_event`: BlockPos, u8, u8, block registry VarInt (the chest lid
/// uses event id 1 with the viewer count as the second byte).
pub const PACKET_BLOCK_EVENT: i32 = 0x07;
/// `container_close` (clientbound): containerId VarInt.
pub const PACKET_CONTAINER_CLOSE: i32 = 0x11;
/// `open_screen`: containerId VarInt, menu type VarInt, title component.
pub const PACKET_OPEN_SCREEN: i32 = 0x3c;

/// `minecraft:menu` registry ids.
pub const MENU_GENERIC_9X3: i32 = 2;
pub const MENU_GENERIC_9X6: i32 = 5;
pub const MENU_HOPPER: i32 = 16;

/// `minecraft:block_entity_type` registry ids.
pub const BE_TYPE_CHEST: u32 = 1;
pub const BE_TYPE_TRAPPED_CHEST: u32 = 2;
pub const BE_TYPE_HOPPER: u32 = 18;

/// `minecraft:block` registry ids (block_event carries the block id, not
/// the block-state id).
pub const BLOCK_CHEST: i32 = 245;
pub const BLOCK_TRAPPED_CHEST: i32 = 523;
pub const BLOCK_HOPPER: i32 = 530;

pub const CHEST_SIZE: usize = 27;
pub const HOPPER_SIZE: usize = 5;
/// The hopper transfer cooldown, in game ticks.
pub const HOPPER_COOLDOWN: i32 = 8;

type Pos = (i32, i32, i32);

// ---------------------------------------------------------------------
// Container storage
// ---------------------------------------------------------------------

/// One block entity's item storage.
#[derive(Clone, Debug, Default)]
pub struct Container {
    pub slots: Vec<Option<ItemStack>>,
}

impl Container {
    pub fn new(size: usize) -> Container {
        Container {
            slots: vec![None; size],
        }
    }

    pub fn size(&self) -> usize {
        self.slots.len()
    }

    pub fn get(&self, slot: usize) -> Option<ItemStack> {
        self.slots.get(slot).cloned().flatten()
    }

    /// Writes a slot: the count clamps to min(99, the stack's own max).
    pub fn set(&mut self, slot: usize, stack: Option<ItemStack>) {
        if let Some(cell) = self.slots.get_mut(slot) {
            *cell = stack
                .map(|mut s| {
                    let cap = s.max_stack_size().min(ABSOLUTE_MAX_STACK_SIZE);
                    s.set_count(s.count().min(cap));
                    s
                })
                .filter(|s| !s.is_empty());
        }
    }

    pub fn is_empty(&self) -> bool {
        self.slots.iter().flatten().all(|s| s.count() <= 0)
    }

    /// Every slot occupied and at its own max: the "rejects all inserts"
    /// state.
    pub fn is_full(&self) -> bool {
        self.slots.iter().all(|s| {
            s.as_ref()
                .is_some_and(|st| st.count() >= st.max_stack_size().min(ABSOLUTE_MAX_STACK_SIZE))
        })
    }

    /// The comparator fill signal over the whole container:
    /// floor(fill * 14) + (fill > 0), each stack measured against
    /// min(99, its own max stack).
    pub fn signal(&self) -> i32 {
        if self.slots.is_empty() {
            return 0;
        }
        let mut fill = 0.0f32;
        for s in self.slots.iter().flatten() {
            fill += s.count() as f32 / s.max_stack_size().min(ABSOLUTE_MAX_STACK_SIZE) as f32;
        }
        fill /= self.slots.len() as f32;
        (fill * 14.0).floor() as i32 + i32::from(fill > 0.0)
    }
}

/// Which container family a block position hosts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeKind {
    Chest,
    TrappedChest,
    Hopper,
}

impl BeKind {
    fn from_block(name: &str) -> Option<BeKind> {
        Some(match name {
            "minecraft:chest" => BeKind::Chest,
            "minecraft:trapped_chest" => BeKind::TrappedChest,
            "minecraft:hopper" => BeKind::Hopper,
            _ => return None,
        })
    }

    fn container_size(self) -> usize {
        match self {
            BeKind::Hopper => HOPPER_SIZE,
            _ => CHEST_SIZE,
        }
    }

    fn be_type_id(self) -> u32 {
        match self {
            BeKind::Chest => BE_TYPE_CHEST,
            BeKind::TrappedChest => BE_TYPE_TRAPPED_CHEST,
            BeKind::Hopper => BE_TYPE_HOPPER,
        }
    }

    fn block_id(self) -> i32 {
        match self {
            BeKind::Chest => BLOCK_CHEST,
            BeKind::TrappedChest => BLOCK_TRAPPED_CHEST,
            BeKind::Hopper => BLOCK_HOPPER,
        }
    }
}

/// One live block entity.
#[derive(Clone, Debug)]
pub struct BlockEntityData {
    pub kind: BeKind,
    pub container: Container,
    /// Chest viewer count (the lid payload on the wire).
    pub open_count: i32,
    /// Hopper transfer cooldown, decremented every tick; <= 0 means due.
    pub cooldown: i32,
    /// The tick this hopper last ticked at (transfer chain ordering).
    pub ticked_at: u64,
}

impl BlockEntityData {
    fn new(kind: BeKind) -> BlockEntityData {
        BlockEntityData {
            kind,
            container: Container::new(kind.container_size()),
            open_count: 0,
            cooldown: 0,
            ticked_at: 0,
        }
    }
}

/// Game-side container state.
#[derive(Default)]
pub struct ContainersState {
    pub block_entities: BTreeMap<Pos, BlockEntityData>,
    /// Chest lid block events queued for this tick's event phase:
    /// (pos, viewer count).
    lid_events: Vec<(Pos, i32)>,
    /// Block entities whose data changed this tick (per-tick sync batch).
    dirty: BTreeSet<Pos>,
}

// ---------------------------------------------------------------------
// Open menus
// ---------------------------------------------------------------------

/// Which container an open menu addresses.
#[derive(Clone, Debug)]
pub(crate) enum OpenKind {
    /// A chest grid: one or two halves (the first half is the menu's
    /// slots 0..27) and the row count.
    Chest {
        halves: Vec<Pos>,
        rows: usize,
    },
    Hopper {
        pos: Pos,
    },
}

pub(crate) struct OpenMenu {
    pub id: i32,
    pub kind: OpenKind,
    pub session: MenuSession,
}

fn menu_positions(kind: &OpenKind) -> Vec<Pos> {
    match kind {
        OpenKind::Chest { halves, .. } => halves.clone(),
        OpenKind::Hopper { pos } => vec![*pos],
    }
}

fn container_slots_of(kind: &OpenKind) -> usize {
    match kind {
        OpenKind::Chest { rows, .. } => rows * 9,
        OpenKind::Hopper { .. } => HOPPER_SIZE,
    }
}

/// Menu slot -> (container position, slot inside that container).
/// Player inventory slots map to None.
fn container_slot_at(kind: &OpenKind, menu_slot: usize) -> Option<(Pos, usize)> {
    match kind {
        OpenKind::Chest { halves, rows } => {
            if menu_slot >= rows * 9 {
                return None;
            }
            let (half, sub) = (menu_slot / CHEST_SIZE, menu_slot % CHEST_SIZE);
            Some((*halves.get(half)?, sub))
        }
        OpenKind::Hopper { pos } => (menu_slot < HOPPER_SIZE).then_some((*pos, menu_slot)),
    }
}

fn player_menu_get(inv: &PlayerInventory, local: usize) -> Option<ItemStack> {
    if local < 27 {
        inv.get(9 + local)
    } else if local < 36 {
        inv.get(local - 27)
    } else {
        None
    }
}

fn player_menu_set(inv: &mut PlayerInventory, local: usize, stack: Option<ItemStack>) {
    if local < 27 {
        inv.set(9 + local, stack);
    } else if local < 36 {
        inv.set(local - 27, stack);
    }
}

/// The click-engine view of a container menu: the container grid plus the
/// player inventory behind it.
struct ContainerMenuView<'a> {
    state: &'a mut ContainersState,
    inventory: &'a mut PlayerInventory,
    kind: &'a OpenKind,
}

impl MenuSlots for ContainerMenuView<'_> {
    fn menu_size(&self) -> usize {
        container_slots_of(self.kind) + 36
    }

    fn menu_get(&self, slot: usize) -> Option<ItemStack> {
        match container_slot_at(self.kind, slot) {
            Some((pos, sub)) => self
                .state
                .block_entities
                .get(&pos)
                .and_then(|be| be.container.get(sub)),
            None => player_menu_get(self.inventory, slot - container_slots_of(self.kind)),
        }
    }

    fn menu_set(&mut self, slot: usize, stack: Option<ItemStack>) {
        match container_slot_at(self.kind, slot) {
            Some((pos, sub)) => {
                if let Some(be) = self.state.block_entities.get_mut(&pos) {
                    be.container.set(sub, stack);
                }
            }
            None => player_menu_set(self.inventory, slot - container_slots_of(self.kind), stack),
        }
    }

    fn slot_max_stack(&self, _slot: usize, stack: &ItemStack) -> i32 {
        ABSOLUTE_MAX_STACK_SIZE.min(stack.max_stack_size())
    }

    fn slot_may_place(&self, _slot: usize) -> bool {
        true
    }

    fn may_pick_all(&self, _slot: usize) -> bool {
        true
    }

    fn quick_move_bounds(&self, slot: usize) -> (usize, usize, bool) {
        let grid = container_slots_of(self.kind);
        if slot < grid {
            (grid, self.menu_size(), true)
        } else {
            (0, grid, false)
        }
    }

    fn swap_get(&self, container_slot: usize) -> Option<ItemStack> {
        self.inventory.get(container_slot)
    }

    fn swap_set(&mut self, container_slot: usize, stack: Option<ItemStack>) {
        self.inventory.set(container_slot, stack)
    }

    fn insert_into_inventory(&mut self, stack: ItemStack) -> Option<ItemStack> {
        self.inventory.add(stack)
    }
}

fn push_container_slots(
    out: &mut Vec<Option<ItemStack>>,
    state: &ContainersState,
    kind: &OpenKind,
) {
    for slot in 0..container_slots_of(kind) {
        let stack = container_slot_at(kind, slot).and_then(|(pos, sub)| {
            state
                .block_entities
                .get(&pos)
                .and_then(|be| be.container.get(sub))
        });
        out.push(stack);
    }
}

fn push_player_slots(out: &mut Vec<Option<ItemStack>>, inv: &PlayerInventory) {
    for local in 0..36 {
        out.push(player_menu_get(inv, local));
    }
}

fn menu_snapshot(
    state: &ContainersState,
    inv: &PlayerInventory,
    kind: &OpenKind,
) -> Vec<Option<ItemStack>> {
    let mut slots = Vec::with_capacity(container_slots_of(kind) + 36);
    push_container_slots(&mut slots, state, kind);
    push_player_slots(&mut slots, inv);
    slots
}

fn menu_wire(kind: &OpenKind) -> (i32, Vec<u8>) {
    match kind {
        OpenKind::Chest { halves, .. } => {
            if halves.len() == 2 {
                (
                    MENU_GENERIC_9X6,
                    translatable_title("container.chestDouble"),
                )
            } else {
                (MENU_GENERIC_9X3, translatable_title("container.chest"))
            }
        }
        OpenKind::Hopper { .. } => (MENU_HOPPER, translatable_title("container.hopper")),
    }
}

/// A translatable component as network NBT: an unnamed root compound
/// holding one `translate` string.
fn translatable_title(key: &str) -> Vec<u8> {
    let mut out = vec![0x0a];
    nbt_name(&mut out, 8, "translate");
    nbt_string(&mut out, key);
    out.push(0x00);
    out
}

// ---------------------------------------------------------------------
// Block-entity NBT
// ---------------------------------------------------------------------

/// The block entity's data NBT: the item list plus the hopper transfer
/// cooldown, as an unnamed root compound.
/// NOTE(containers): item component patches are dropped until component
/// NBT lands; items outside the curated table fall back to air.
fn be_nbt(be: &BlockEntityData) -> Vec<u8> {
    let occupied = be.container.slots.iter().filter(|s| s.is_some()).count();
    let mut out = vec![0x0a]; // TAG_Compound root
    nbt_name(&mut out, 9, "Items");
    out.push(0x0a); // list element type
    out.extend_from_slice(&(occupied as i32).to_be_bytes());
    for (slot, stack) in be.container.slots.iter().enumerate() {
        let Some(stack) = stack else { continue };
        nbt_name(&mut out, 1, "Slot");
        out.push(slot as u8);
        nbt_name(&mut out, 8, "id");
        nbt_string(&mut out, item_name(stack.item()).unwrap_or("minecraft:air"));
        nbt_name(&mut out, 3, "count");
        out.extend_from_slice(&stack.count().to_be_bytes());
        out.push(0x00); // end of the element compound
    }
    if be.kind == BeKind::Hopper {
        nbt_name(&mut out, 3, "TransferCooldown");
        out.extend_from_slice(&be.cooldown.to_be_bytes());
    }
    out.push(0x00); // end of the root compound
    out
}

/// One named NBT tag header (type byte + u16-be name length + name).
fn nbt_name(out: &mut Vec<u8>, tag: u8, name: &str) {
    out.push(tag);
    out.extend_from_slice(&(name.len() as u16).to_be_bytes());
    out.extend_from_slice(name.as_bytes());
}

fn nbt_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u16).to_be_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Parses a serverbound container_close body: one VarInt container id.
pub fn parse_container_close(body: &[u8]) -> Result<i32> {
    let mut r = Reader::new(body);
    let id = r.read_varint().context("container id")?;
    if r.remaining() != 0 {
        bail!("trailing bytes in container_close");
    }
    Ok(id)
}

// ---------------------------------------------------------------------
// Transfer targets
// ---------------------------------------------------------------------

/// A container position resolved for transfers: a chest pair merges into
/// one 54-slot view (the first half is the right-hand chest).
#[derive(Clone, Copy, Debug)]
enum ContainerRef {
    Single(Pos),
    Double(Pos, Pos),
}

fn cref_positions(r: &ContainerRef) -> Vec<Pos> {
    match r {
        ContainerRef::Single(p) => vec![*p],
        ContainerRef::Double(a, b) => vec![*a, *b],
    }
}

fn cref_size(state: &ContainersState, r: &ContainerRef) -> usize {
    match r {
        ContainerRef::Single(p) => state
            .block_entities
            .get(p)
            .map_or(0, |be| be.container.size()),
        ContainerRef::Double(_, _) => CHEST_SIZE * 2,
    }
}

fn cref_get(state: &ContainersState, r: &ContainerRef, slot: usize) -> Option<ItemStack> {
    let (pos, sub) = match r {
        ContainerRef::Single(p) => (*p, slot),
        ContainerRef::Double(a, b) => {
            if slot < CHEST_SIZE {
                (*a, slot)
            } else {
                (*b, slot - CHEST_SIZE)
            }
        }
    };
    state
        .block_entities
        .get(&pos)
        .and_then(|be| be.container.get(sub))
}

fn cref_set(state: &mut ContainersState, r: &ContainerRef, slot: usize, stack: Option<ItemStack>) {
    let (pos, sub) = match r {
        ContainerRef::Single(p) => (*p, slot),
        ContainerRef::Double(a, b) => {
            if slot < CHEST_SIZE {
                (*a, slot)
            } else {
                (*b, slot - CHEST_SIZE)
            }
        }
    };
    if let Some(be) = state.block_entities.get_mut(&pos) {
        be.container.set(sub, stack);
    }
}

/// The horizontal step of a facing prop (default east).
fn facing_step(facing: &str) -> (i32, i32) {
    match facing {
        "north" => (0, -1),
        "south" => (0, 1),
        "west" => (-1, 0),
        _ => (1, 0),
    }
}

/// The 3D step of a facing prop (hoppers also face down).
fn facing_step3(facing: &str) -> (i32, i32, i32) {
    match facing {
        "down" => (0, -1, 0),
        "north" => (0, 0, -1),
        "south" => (0, 0, 1),
        "west" => (-1, 0, 0),
        _ => (1, 0, 0),
    }
}

/// True for the common full-solid blocks (the redstone-conductor set is
/// curated; air, the redstone families, and the containers themselves are
/// not conductors).
fn is_redstone_conductor(name: &str) -> bool {
    matches!(
        name,
        "minecraft:stone"
            | "minecraft:granite"
            | "minecraft:polished_granite"
            | "minecraft:diorite"
            | "minecraft:polished_diorite"
            | "minecraft:andesite"
            | "minecraft:polished_andesite"
            | "minecraft:deepslate"
            | "minecraft:cobbled_deepslate"
            | "minecraft:grass_block"
            | "minecraft:dirt"
            | "minecraft:cobblestone"
            | "minecraft:oak_planks"
            | "minecraft:oak_log"
            | "minecraft:obsidian"
            | "minecraft:bedrock"
    )
}

/// The packed BlockPos form shared with the block-update packet.
fn pack_block_pos(pos: Pos) -> i64 {
    (((pos.0 as i64) & 0x3ff_ffff) << 38)
        | (((pos.2 as i64) & 0x3ff_ffff) << 12)
        | ((pos.1 as i64) & 0xfff)
}

// ---------------------------------------------------------------------
// Game-thread hooks
// ---------------------------------------------------------------------

impl Game {
    /// Keeps the block-entity map in step with the block at a position
    /// after a successful write: containers appear on placement and are
    /// discarded (contents dropped) on removal.
    pub(super) fn sync_block_entity(&mut self, x: i32, y: i32, z: i32) {
        let pos = (x, y, z);
        let kind = self
            .get_block(x, y, z)
            .and_then(|(n, _)| BeKind::from_block(&n));
        let existing = self.containers.block_entities.get(&pos).map(|be| be.kind);
        match (kind, existing) {
            (Some(k), None) => {
                self.containers
                    .block_entities
                    .insert(pos, BlockEntityData::new(k));
                self.update_chunk_be_entry(pos, true);
            }
            (Some(k), Some(old)) if k != old => {
                self.destroy_block_entity(pos);
                self.containers
                    .block_entities
                    .insert(pos, BlockEntityData::new(k));
                self.update_chunk_be_entry(pos, true);
            }
            (None, Some(_)) => self.destroy_block_entity(pos),
            _ => {}
        }
    }

    /// Removes a block entity: viewers' menus close, the queued lid
    /// events and sync entries go, and the contents drop with the block.
    fn destroy_block_entity(&mut self, pos: Pos) {
        if !self.containers.block_entities.contains_key(&pos) {
            return;
        }
        let conns: Vec<ConnId> = self
            .players
            .iter()
            .filter(|(_, p)| {
                p.menu
                    .as_ref()
                    .is_some_and(|m| menu_positions(&m.kind).contains(&pos))
            })
            .map(|(c, _)| *c)
            .collect();
        for conn in conns {
            self.close_menu(conn, true, false);
        }
        self.containers.block_entities.remove(&pos);
        self.containers.lid_events.retain(|(p, _)| *p != pos);
        self.containers.dirty.remove(&pos);
        self.update_chunk_be_entry(pos, false);
    }

    /// Rewrites (or removes) the cached chunk's block-entity entry for a
    /// position so later chunk sends carry the live data.
    fn update_chunk_be_entry(&mut self, pos: Pos, present: bool) {
        let packed_xz = ((pos.0.rem_euclid(16) as u8) << 4) | pos.2.rem_euclid(16) as u8;
        let entry = present.then(|| {
            let be = self.containers.block_entities.get(&pos)?;
            let mut tag = vec![0x01]; // nullable-NBT presence byte
            tag.extend_from_slice(&be_nbt(be));
            Some(WireBlockEntity {
                packed_xz,
                y: pos.1 as i16,
                ty: be.kind.be_type_id(),
                tag: Some(tag),
            })
        });
        let (cx, cz) = (pos.0.div_euclid(16), pos.2.div_euclid(16));
        let Some(chunk) = self.chunks.get_mut(&(cx, cz)) else {
            return;
        };
        chunk
            .wire
            .block_entities
            .retain(|be| be.packed_xz != packed_xz || be.y != pos.1 as i16);
        if let Some(e) = entry.flatten() {
            chunk.wire.block_entities.push(e);
        }
        chunk.version += 1;
    }

    /// The double-chest pairing: LEFT connects clockwise of the facing,
    /// RIGHT counter-clockwise; the partner is the same chest kind, the
    /// opposite type, and the same facing. The RIGHT half orders first in
    /// a merged container.
    fn chest_partner(&self, pos: Pos) -> Option<Pos> {
        let (name, props) = self.get_block(pos.0, pos.1, pos.2)?;
        if BeKind::from_block(&name) == Some(BeKind::Hopper) {
            return None;
        }
        let ty = prop_value(&props, "type");
        if ty != "left" && ty != "right" {
            return None;
        }
        let facing = prop_dir(&props);
        let (fx, fz) = facing_step(facing);
        // Horizontal clockwise: north -> east -> south -> west.
        let (cx, cz) = if ty == "left" { (-fz, fx) } else { (fz, -fx) };
        let npos = (pos.0 + cx, pos.1, pos.2 + cz);
        let (nname, nprops) = self.get_block(npos.0, npos.1, npos.2)?;
        let nty = prop_value(&nprops, "type");
        if nname != name || nty == ty || nty == "single" || prop_dir(&nprops) != facing {
            return None;
        }
        Some(npos)
    }

    /// The menu/transfer halves for a chest position: the pair (right
    /// half first) or the single chest. NOTE(containers): a type prop
    /// left stale by direct state edits degrades to the single view, the
    /// same fallback the reference's pairing check produces.
    fn chest_menu_halves(&self, pos: Pos) -> Vec<Pos> {
        let own_type = self
            .get_block(pos.0, pos.1, pos.2)
            .map(|(_, p)| prop_value(&p, "type").to_string());
        let partner = self
            .chest_partner(pos)
            .filter(|p| self.containers.block_entities.contains_key(p));
        match (partner, own_type.as_deref()) {
            (Some(p), Some("left")) => vec![p, pos],
            (Some(p), _) => vec![pos, p],
            (None, _) => vec![pos],
        }
    }

    /// A chest with a redstone conductor above refuses to open and reads
    /// as an empty comparator source.
    fn chest_blocked(&self, x: i32, y: i32, z: i32) -> bool {
        self.get_block(x, y + 1, z)
            .is_some_and(|(n, _)| is_redstone_conductor(&n))
    }

    /// The transfer-accessible container at a position (transfers ignore
    /// the chest-blocked rule); a chest pair merges into one view.
    fn resolve_container_ref(&self, pos: Pos) -> Option<ContainerRef> {
        let (name, _) = self.get_block(pos.0, pos.1, pos.2)?;
        match BeKind::from_block(&name)? {
            BeKind::Hopper => Some(ContainerRef::Single(pos)),
            _ => {
                if !self.containers.block_entities.contains_key(&pos) {
                    return None;
                }
                let partner = self
                    .chest_partner(pos)
                    .filter(|p| self.containers.block_entities.contains_key(p));
                match partner {
                    Some(p) => {
                        let props = self.get_block(pos.0, pos.1, pos.2)?.1;
                        if prop_value(&props, "type") == "left" {
                            Some(ContainerRef::Double(p, pos))
                        } else {
                            Some(ContainerRef::Double(pos, p))
                        }
                    }
                    None => Some(ContainerRef::Single(pos)),
                }
            }
        }
    }

    /// The comparator's container fill signal at a block position, or
    /// None when the block is not a live container. Chests read as their
    /// merged pair unless blocked from above.
    pub(super) fn container_analog_output(&self, x: i32, y: i32, z: i32) -> Option<i32> {
        let pos = (x, y, z);
        let (name, _) = self.get_block(x, y, z)?;
        match BeKind::from_block(&name)? {
            BeKind::Hopper => {
                let be = self.containers.block_entities.get(&pos)?;
                Some(be.container.signal())
            }
            _ => {
                if self.chest_blocked(x, y, z) {
                    return Some(0);
                }
                let r = self.resolve_container_ref(pos)?;
                let mut fill = 0.0f32;
                let mut total = 0usize;
                for p in cref_positions(&r) {
                    match self.containers.block_entities.get(&p) {
                        Some(be) => {
                            for s in be.container.slots.iter().flatten() {
                                fill += s.count() as f32
                                    / s.max_stack_size().min(ABSOLUTE_MAX_STACK_SIZE) as f32;
                            }
                            total += be.container.slots.len();
                        }
                        None => total += CHEST_SIZE,
                    }
                }
                if total == 0 {
                    return Some(0);
                }
                let fill = fill / total as f32;
                Some((fill * 14.0).floor() as i32 + i32::from(fill > 0.0))
            }
        }
    }

    /// `opencontainer x y z` (harness driver): opens the container menu
    /// at a block position for one player.
    pub(super) fn open_container(&mut self, conn: ConnId, x: i32, y: i32, z: i32) {
        let Some((name, _)) = self.get_block(x, y, z) else {
            return;
        };
        let kind = match BeKind::from_block(&name) {
            Some(BeKind::Hopper) => OpenKind::Hopper { pos: (x, y, z) },
            Some(_) => {
                let halves = self.chest_menu_halves((x, y, z));
                // A blocked chest refuses to open; both halves of a pair
                // must be clear.
                if halves.iter().any(|p| self.chest_blocked(p.0, p.1, p.2)) {
                    return;
                }
                let rows = if halves.len() == 2 { 6 } else { 3 };
                OpenKind::Chest { halves, rows }
            }
            None => return,
        };
        let anchor = match &kind {
            OpenKind::Chest { halves, .. } => halves[0],
            OpenKind::Hopper { pos } => *pos,
        };
        if !self.containers.block_entities.contains_key(&anchor) {
            return;
        }
        // Opening replaces an already-open menu (server close first).
        self.close_menu(conn, true, false);
        let Some(p) = self.players.get_mut(&conn) else {
            return;
        };
        p.container_counter = p.container_counter % 100 + 1;
        let id = p.container_counter;
        if let OpenKind::Chest { halves, .. } = &kind {
            for half in halves.clone() {
                if let Some(be) = self.containers.block_entities.get_mut(&half) {
                    be.open_count += 1;
                    self.containers.lid_events.push((half, be.open_count));
                }
            }
        }
        let (menu_id, title) = menu_wire(&kind);
        let mut body = Vec::with_capacity(title.len() + 8);
        write_varint(&mut body, id);
        write_varint(&mut body, menu_id);
        body.extend_from_slice(&title);
        self.send(conn, PACKET_OPEN_SCREEN, &body);
        if let Some(p) = self.players.get_mut(&conn) {
            p.menu = Some(OpenMenu {
                id,
                kind,
                session: MenuSession::default(),
            });
        }
        self.broadcast_menu(conn);
    }

    /// A container_click against an open container menu: the shared click
    /// engine runs over the container-grid view, the touched containers
    /// sync (data + comparator neighbors), and the client gets a full
    /// set_content resync.
    pub(crate) fn container_menu_clicked(&mut self, conn: ConnId, click: &ContainerClick) {
        let kind = {
            let Some(p) = self.players.get(&conn) else {
                return;
            };
            let Some(menu) = p.menu.as_ref() else {
                return;
            };
            if menu.id != click.container_id {
                return;
            }
            menu.kind.clone()
        };
        let positions = menu_positions(&kind);
        {
            // Split borrows: the click engine writes the player's slots
            // and the container grid in one pass.
            let Game {
                players,
                containers,
                ..
            } = self;
            let Some(p) = players.get_mut(&conn) else {
                return;
            };
            let creative = p.inv.creative;
            let Some(menu) = p.menu.as_mut() else {
                return;
            };
            let mut view = ContainerMenuView {
                state: containers,
                inventory: &mut p.inv.inventory,
                kind: &kind,
            };
            apply_click(&mut menu.session, &mut view, creative, click);
        }
        for pos in positions {
            self.container_changed(pos);
        }
        self.broadcast_menu(conn);
    }

    /// Ends a player's open menu: the carried stack returns to the
    /// inventory (or drops when the player is leaving), chest viewer
    /// counts decrement with lid events, and the close packet goes out
    /// when the server initiated the close.
    pub(super) fn close_menu(&mut self, conn: ConnId, send_close: bool, discard_carried: bool) {
        let menu = self.players.get_mut(&conn).and_then(|p| p.menu.take());
        let Some(menu) = menu else {
            return;
        };
        if !discard_carried {
            if let Some(carried) = menu.session.carried {
                if let Some(p) = self.players.get_mut(&conn) {
                    // Overflow would drop; no item entities yet.
                    let _ = p.inv.inventory.add(carried);
                }
            }
        }
        if let OpenKind::Chest { halves, .. } = &menu.kind {
            for half in halves.clone() {
                if let Some(be) = self.containers.block_entities.get_mut(&half) {
                    be.open_count = (be.open_count - 1).max(0);
                    let count = be.open_count;
                    self.containers.lid_events.push((half, count));
                }
            }
        }
        if send_close {
            let mut body = Vec::with_capacity(2);
            write_varint(&mut body, menu.id);
            self.send(conn, PACKET_CONTAINER_CLOSE, &body);
        }
    }

    /// Serverbound container_close: a matching id ends the menu silently
    /// (the client already closed its side).
    pub(super) fn client_closed_container(&mut self, conn: ConnId, container_id: i32) {
        let matches = self
            .players
            .get(&conn)
            .and_then(|p| p.menu.as_ref())
            .is_some_and(|m| m.id == container_id);
        if matches {
            self.close_menu(conn, false, false);
        }
    }

    /// Full resync of an open container menu: one set_content over the
    /// whole menu-slot list plus the carried stack.
    fn broadcast_menu(&mut self, conn: ConnId) {
        let body = {
            let Game {
                players,
                containers,
                ..
            } = self;
            let Some(p) = players.get_mut(&conn) else {
                return;
            };
            let Some(menu) = p.menu.as_mut() else {
                return;
            };
            let state_id = menu.session.next_state_id();
            let carried = menu.session.carried.clone();
            let id = menu.id;
            let kind = menu.kind.clone();
            let slots = menu_snapshot(containers, &p.inv.inventory, &kind);
            encode_container_set_content(id, state_id, &slots, carried.as_ref())
        };
        self.send(conn, PACKET_CONTAINER_SET_CONTENT, &body);
    }

    /// The block-entity phase after block events: chest lid events fire,
    /// open menus validate their distance, and hoppers transfer.
    pub(super) fn tick_containers(&mut self) {
        self.fire_lid_events();
        self.check_menu_distances();
        self.tick_hoppers();
    }

    fn fire_lid_events(&mut self) {
        let events = std::mem::take(&mut self.containers.lid_events);
        for (pos, count) in events {
            let Some((name, _)) = self.get_block(pos.0, pos.1, pos.2) else {
                continue;
            };
            let Some(kind) = BeKind::from_block(&name) else {
                continue;
            };
            if kind == BeKind::Hopper {
                continue;
            }
            let mut body = Vec::with_capacity(16);
            body.extend_from_slice(&pack_block_pos(pos).to_be_bytes());
            body.push(1); // the lid event id
            body.push(count.clamp(0, 255) as u8);
            write_varint(&mut body, kind.block_id());
            self.send_to_chunk(pos, PACKET_BLOCK_EVENT, &body);
        }
    }

    /// Open menus close once the player leaves interaction range
    /// (survival block reach 4.5 + the still-valid buffer 4.0).
    fn check_menu_distances(&mut self) {
        const REACH: f64 = 8.5;
        let mut stale: Vec<ConnId> = Vec::new();
        for (conn, p) in &self.players {
            let Some(menu) = &p.menu else {
                continue;
            };
            let near = menu_positions(&menu.kind).iter().any(|pos| {
                let dx = p.x - (pos.0 as f64 + 0.5);
                let dy = p.y - (pos.1 as f64 + 0.5);
                let dz = p.z - (pos.2 as f64 + 0.5);
                dx * dx + dy * dy + dz * dz <= REACH * REACH
            });
            if !near {
                stale.push(*conn);
            }
        }
        for conn in stale {
            self.close_menu(conn, true, false);
        }
    }

    /// The hopper transfer tick: the cooldown decrements first, then when
    /// due (and enabled) one item pushes into the facing container before
    /// one pulls from above; any move resets the 8gt cooldown.
    fn tick_hoppers(&mut self) {
        let hoppers: Vec<Pos> = self
            .containers
            .block_entities
            .iter()
            .filter(|(_, be)| be.kind == BeKind::Hopper)
            .map(|(pos, _)| *pos)
            .collect();
        for pos in hoppers {
            {
                let Some(be) = self.containers.block_entities.get_mut(&pos) else {
                    continue;
                };
                be.cooldown -= 1;
                be.ticked_at = self.tick;
                if be.cooldown > 0 {
                    continue;
                }
                be.cooldown = 0;
            }
            let Some((name, props)) = self.get_block(pos.0, pos.1, pos.2) else {
                continue;
            };
            if name != "minecraft:hopper" || prop_value(&props, "enabled") != "true" {
                continue;
            }
            let facing = prop_dir(&props);
            let mut changed = false;
            let has_items = self
                .containers
                .block_entities
                .get(&pos)
                .is_some_and(|be| !be.container.is_empty());
            if has_items {
                changed = self.hopper_push(pos, facing);
            }
            if !self.hopper_inventory_full(pos) {
                changed |= self.hopper_pull(pos);
            }
            if changed {
                if let Some(be) = self.containers.block_entities.get_mut(&pos) {
                    be.cooldown = HOPPER_COOLDOWN;
                }
                self.container_changed(pos);
            }
        }
    }

    /// Push one item into the facing container; a target that cannot take
    /// it leaves the hopper untouched.
    fn hopper_push(&mut self, pos: Pos, facing: &str) -> bool {
        let step = facing_step3(facing);
        let target_pos = (pos.0 + step.0, pos.1 + step.1, pos.2 + step.2);
        let Some(target) = self.resolve_container_ref(target_pos) else {
            return false;
        };
        if self.cref_is_full(&target) {
            return false;
        }
        for slot in 0..HOPPER_SIZE {
            let Some(item) = self
                .containers
                .block_entities
                .get(&pos)
                .and_then(|be| be.container.get(slot))
            else {
                continue;
            };
            let original = item.clone();
            self.hopper_take_one(pos, slot);
            let leftover = self.insert_into_container(&target, item.with_count(1), Some(pos));
            if leftover.is_none() {
                for t in cref_positions(&target) {
                    self.container_changed(t);
                }
                return true;
            }
            // The item did not fit: restore the untouched stack.
            if let Some(be) = self.containers.block_entities.get_mut(&pos) {
                be.container.set(slot, Some(original));
            }
        }
        false
    }

    /// Pull one item from the container above into the hopper.
    fn hopper_pull(&mut self, pos: Pos) -> bool {
        let source_pos = (pos.0, pos.1 + 1, pos.2);
        let Some(source) = self.resolve_container_ref(source_pos) else {
            return false;
        };
        for slot in 0..cref_size(&self.containers, &source) {
            let Some(item) = cref_get(&self.containers, &source, slot) else {
                continue;
            };
            let original = item.clone();
            let rest = original.count() - 1;
            cref_set(
                &mut self.containers,
                &source,
                slot,
                (rest > 0).then(|| original.with_count(rest)),
            );
            let leftover = self.insert_into_container(
                &ContainerRef::Single(pos),
                item.with_count(1),
                Some(source_pos),
            );
            if leftover.is_none() {
                for s in cref_positions(&source) {
                    self.container_changed(s);
                }
                return true;
            }
            cref_set(&mut self.containers, &source, slot, Some(original));
        }
        false
    }

    /// The hopper's own inventory rejecting pulls: every slot at max.
    fn hopper_inventory_full(&self, pos: Pos) -> bool {
        self.containers
            .block_entities
            .get(&pos)
            .is_some_and(|be| be.container.is_full())
    }

    fn cref_is_full(&self, r: &ContainerRef) -> bool {
        cref_positions(r).iter().all(|p| {
            self.containers
                .block_entities
                .get(p)
                .is_some_and(|be| be.container.is_full())
        })
    }

    /// Takes one item out of a hopper slot.
    fn hopper_take_one(&mut self, pos: Pos, slot: usize) {
        if let Some(be) = self.containers.block_entities.get_mut(&pos) {
            if let Some(s) = be.container.get(slot) {
                let rest = s.count() - 1;
                be.container
                    .set(slot, (rest > 0).then(|| s.with_count(rest)));
            }
        }
    }

    /// Merges a stack into the first matching slot of a
    /// container, returning the leftover. An insert into a hopper that
    /// was empty resets its cooldown to 8, minus 1 when the source hopper
    /// already ticked this tick (the chain ordering rule).
    fn insert_into_container(
        &mut self,
        target: &ContainerRef,
        mut stack: ItemStack,
        from: Option<Pos>,
    ) -> Option<ItemStack> {
        let was_empty = cref_positions(target).iter().all(|p| {
            self.containers
                .block_entities
                .get(p)
                .is_none_or(|be| be.container.is_empty())
        });
        let from_info = from.and_then(|f| {
            self.containers
                .block_entities
                .get(&f)
                .map(|be| (be.kind == BeKind::Hopper, be.ticked_at))
        });
        for slot in 0..cref_size(&self.containers, target) {
            match cref_get(&self.containers, target, slot) {
                None => {
                    cref_set(&mut self.containers, target, slot, Some(stack));
                    if was_empty {
                        self.hopper_cooldown_reset(target, from_info);
                    }
                    return None;
                }
                Some(occupant) if ItemStack::same_item_same_components(&occupant, &stack) => {
                    let max = occupant.max_stack_size().min(ABSOLUTE_MAX_STACK_SIZE);
                    let room = max - occupant.count();
                    if room <= 0 {
                        continue;
                    }
                    let transfer = room.min(stack.count());
                    cref_set(
                        &mut self.containers,
                        target,
                        slot,
                        Some(occupant.with_count(occupant.count() + transfer)),
                    );
                    stack.shrink(transfer);
                    if stack.is_empty() {
                        if was_empty {
                            self.hopper_cooldown_reset(target, from_info);
                        }
                        return None;
                    }
                }
                _ => {}
            }
        }
        Some(stack)
    }

    /// The receiving hopper's cooldown after an insert while it was
    /// empty: 8, or 7 when the source hopper already ticked this tick
    /// (the transfer-chain ordering). A longer cooldown is not shortened.
    fn hopper_cooldown_reset(&mut self, target: &ContainerRef, from_info: Option<(bool, u64)>) {
        let ContainerRef::Single(tp) = target else {
            return;
        };
        let Some(be) = self.containers.block_entities.get_mut(tp) else {
            return;
        };
        if be.kind != BeKind::Hopper || be.cooldown > HOPPER_COOLDOWN {
            return;
        }
        let skip = match from_info {
            Some((true, t)) if be.ticked_at >= t => 1,
            _ => 0,
        };
        be.cooldown = HOPPER_COOLDOWN - skip;
    }

    /// A container's contents changed: queue the block-entity data sync
    /// and update the horizontal comparator neighbors (the analog feed,
    /// including the look-through-a-conductor read two blocks out).
    fn container_changed(&mut self, pos: Pos) {
        if !self.containers.block_entities.contains_key(&pos) {
            return;
        }
        self.containers.dirty.insert(pos);
        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, nz) = (pos.0 + dx, pos.2 + dz);
            let Some((name, props)) = self.get_block(nx, pos.1, nz) else {
                continue;
            };
            if name == "minecraft:comparator" {
                self.update_comparator(nx, pos.1, nz, &props);
            } else if is_redstone_conductor(&name) {
                let (bx, bz) = (nx + dx, nz + dz);
                if let Some((bname, bprops)) = self.get_block(bx, pos.1, bz) {
                    if bname == "minecraft:comparator" {
                        self.update_comparator(bx, pos.1, bz, &bprops);
                    }
                }
            }
        }
    }

    /// Tick-end sync for changed block entities: refresh the cached chunk
    /// entry and push one data packet per position to the chunk's
    /// viewers.
    pub(super) fn flush_block_entities(&mut self) {
        if self.containers.dirty.is_empty() {
            return;
        }
        let dirty: Vec<Pos> = std::mem::take(&mut self.containers.dirty)
            .into_iter()
            .collect();
        for pos in dirty {
            let Some(be) = self.containers.block_entities.get(&pos) else {
                continue;
            };
            let ty = be.kind.be_type_id();
            let nbt = be_nbt(be);
            self.update_chunk_be_entry(pos, true);
            let mut body = Vec::with_capacity(nbt.len() + 12);
            body.extend_from_slice(&pack_block_pos(pos).to_be_bytes());
            write_varint(&mut body, ty as i32);
            body.extend_from_slice(&nbt);
            self.send_to_chunk(pos, PACKET_BLOCK_ENTITY_DATA, &body);
        }
    }

    fn send_to_chunk(&mut self, pos: Pos, id: i32, body: &[u8]) {
        let key = (pos.0.div_euclid(16), pos.2.div_euclid(16));
        let viewers = self.viewers.get(&key).cloned().unwrap_or_default();
        for v in viewers {
            self.send(v, id, body);
        }
    }

    /// Hopper enablement: any neighbor signal disables the hopper (a
    /// client-only state write, like the reference's flag-2 update).
    pub(super) fn update_hopper(&mut self, x: i32, y: i32, z: i32, props: &str) {
        let enabled = !self.hopper_powered(x, y, z);
        let want = if enabled { "true" } else { "false" };
        if prop_value(props, "enabled") == want {
            return;
        }
        let new_props = doppel_world::registry::BlockRegistry::with_prop(props, "enabled", want);
        let spec = format!("minecraft:hopper[{new_props}]");
        if let Some(state) = self.resolve_state(&spec) {
            self.set_block(x, y, z, state, false);
        }
    }

    fn hopper_powered(&self, x: i32, y: i32, z: i32) -> bool {
        for d in 0u8..6 {
            let (dx, dy, dz) = super::dir_step(d);
            let (sx, sy, sz) = (x + dx, y + dy, z + dz);
            if self.signal_toward(x, y, z, d, sx, sy, sz) {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
impl Game {
    fn teleport_player_for_test(&mut self, conn: ConnId, x: f64, y: f64, z: f64) {
        if let Some(p) = self.players.get_mut(&conn) {
            p.x = x;
            p.y = y;
            p.z = z;
        }
    }

    fn container_get_for_test(&self, pos: Pos, slot: usize) -> Option<ItemStack> {
        self.containers
            .block_entities
            .get(&pos)
            .and_then(|be| be.container.get(slot))
    }

    fn container_set_for_test(&mut self, pos: Pos, slot: usize, stack: Option<ItemStack>) {
        if let Some(be) = self.containers.block_entities.get_mut(&pos) {
            be.container.set(slot, stack);
        }
        // The game paths always notify on a container write; tests read
        // the comparator feed through the same channel.
        self.container_changed(pos);
    }

    fn hopper_cooldown_for_test(&self, pos: Pos) -> Option<i32> {
        self.containers
            .block_entities
            .get(&pos)
            .filter(|be| be.kind == BeKind::Hopper)
            .map(|be| be.cooldown)
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Game, Inbound, Outbound};
    use crate::inventory::item_id;
    use crate::WireChunk;

    /// A game with three synthetic all-air chunks around the origin and
    /// one viewer whose outbound frames we read back. A stone layer at
    /// y=99 floors the world (comparator support).
    fn harness() -> (Game, std::sync::mpsc::Receiver<Outbound>) {
        let (_tx, rx) = std::sync::mpsc::channel::<Inbound>();
        let mut g = Game::new(rx, None, None);
        assert!(g.registry_for_test(), "pins/blocks.json not found");
        let wire = |x: i32| {
            let mut w = WireChunk {
                x,
                z: 0,
                heightmaps: Vec::new(),
                sections: Vec::new(),
                block_entities: Vec::new(),
                light: Default::default(),
            };
            for sy in 0..24 {
                let block_states = if sy == 10 {
                    let mut longs = vec![0u64; 256];
                    for (l, slot) in longs.iter_mut().enumerate() {
                        for j in 0..16 {
                            let i = l * 16 + j;
                            let v: u64 = if (i >> 8) == 3 { 1 } else { 0 };
                            *slot |= v << (j * 4);
                        }
                    }
                    doppel_world::chunk_codec::Container::Palette {
                        bits: 4,
                        entries: vec![0, 1],
                        longs,
                    }
                } else {
                    doppel_world::chunk_codec::Container::Single(0)
                };
                w.sections.push(doppel_world::chunk_codec::WireSection {
                    non_empty: if sy == 10 { 256 } else { 0 },
                    fluid: 0,
                    block_states,
                    biomes: doppel_world::chunk_codec::Container::Single(0),
                });
            }
            w
        };
        for cx in [-1, 0, 1, 2] {
            g.seed_chunk_for_test(cx, 0, wire(cx));
        }
        let (tx_out, rx_out) = std::sync::mpsc::channel::<Outbound>();
        g.join_viewer_for_test(0, &[(-1, 0), (0, 0), (1, 0), (2, 0)], tx_out);
        (g, rx_out)
    }

    fn cmd(g: &mut Game, s: &str) {
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts[0] == "tick" {
            g.handle(Inbound::TickStep {
                conn: 0,
                steps: parts[2].parse().unwrap(),
            });
            return;
        }
        g.handle(Inbound::Setblock {
            conn: 0,
            x: parts[1].parse().unwrap(),
            y: parts[2].parse().unwrap(),
            z: parts[3].parse().unwrap(),
            name: parts[4].to_string(),
        });
    }

    /// Drains every queued frame.
    fn frames(rx: &std::sync::mpsc::Receiver<Outbound>) -> Vec<(i32, Vec<u8>)> {
        let mut out = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            if let Outbound::Frame { id, body } = frame {
                out.push((id, body));
            }
        }
        out
    }

    fn find_frame(fs: &[(i32, Vec<u8>)], id: i32) -> Option<Vec<u8>> {
        fs.iter()
            .find(|(fid, _)| *fid == id)
            .map(|(_, body)| body.clone())
    }

    fn read_varint(body: &[u8], o: &mut usize) -> i64 {
        let mut v: i64 = 0;
        let mut sh = 0u32;
        while *o < body.len() {
            let b = body[*o];
            *o += 1;
            v |= i64::from(b & 0x7f) << sh;
            sh += 7;
            if b & 0x80 == 0 {
                break;
            }
        }
        v
    }

    fn stone() -> Option<ItemStack> {
        stones(64)
    }

    fn stones(n: i32) -> Option<ItemStack> {
        Some(ItemStack::new(item_id("minecraft:stone").unwrap(), n))
    }

    #[test]
    fn chest_open_close_broadcasts_lid_events() {
        let (mut g, rx) = harness();
        cmd(
            &mut g,
            "setblock 10 100 10 minecraft:chest[facing=north,type=single,waterlogged=false]",
        );
        g.tick_once_for_test();
        frames(&rx);
        g.teleport_player_for_test(0, 10.5, 100.5, 10.5);
        g.handle(Inbound::OpenContainer {
            conn: 0,
            x: 10,
            y: 100,
            z: 10,
        });
        let fs = frames(&rx);
        // open_screen: containerId 1, menu generic_9x3 (2), then the bare
        // NBT title compound (one "translate" string).
        let open = find_frame(&fs, PACKET_OPEN_SCREEN).expect("open_screen");
        assert_eq!(open[0], 1);
        assert_eq!(open[1], MENU_GENERIC_9X3 as u8);
        let mut title = vec![0x0a, 0x08, 0x00, 0x09];
        title.extend_from_slice(b"translate");
        title.extend_from_slice(&[0x00, 0x0f]);
        title.extend_from_slice(b"container.chest");
        title.push(0x00);
        assert_eq!(&open[2..], &title[..], "translatable title component");
        // set_content: 63 slots, stateId 1.
        let content = find_frame(&fs, PACKET_CONTAINER_SET_CONTENT).expect("set_content");
        let (cid, sid, slots, carried) =
            crate::inventory::decode_container_set_content(&content).unwrap();
        assert_eq!((cid, sid), (1, 1));
        assert_eq!(slots.len(), 27 + 36);
        assert!(slots.iter().all(Option::is_none));
        assert_eq!(carried, None);
        // The lid event fires in the next tick's event phase.
        g.tick_once_for_test();
        let fs = frames(&rx);
        let ev = find_frame(&fs, PACKET_BLOCK_EVENT).expect("block_event");
        let packed = i64::from_be_bytes(ev[0..8].try_into().unwrap());
        assert_eq!(packed, pack_block_pos((10, 100, 10)));
        assert_eq!(ev[8], 1, "lid event id");
        assert_eq!(ev[9], 1, "viewer count");
        let mut o2 = 10usize;
        assert_eq!(read_varint(&ev, &mut o2), BLOCK_CHEST as i64);
        // Client close: no close packet, the count returns to 0.
        g.handle(Inbound::ContainerClose {
            conn: 0,
            container_id: 1,
        });
        let fs = frames(&rx);
        assert!(find_frame(&fs, PACKET_CONTAINER_CLOSE).is_none());
        g.tick_once_for_test();
        let fs = frames(&rx);
        let ev = find_frame(&fs, PACKET_BLOCK_EVENT).expect("close event");
        assert_eq!(ev[9], 0, "viewer count back to zero");
        // A mismatched id does nothing.
        g.handle(Inbound::ContainerClose {
            conn: 0,
            container_id: 7,
        });
    }

    #[test]
    fn double_chest_merges_and_routes_clicks() {
        let (mut g, rx) = harness();
        // Facing north: right connects counter-clockwise (west), left
        // clockwise (east).
        cmd(
            &mut g,
            "setblock 10 100 10 minecraft:chest[facing=north,type=right,waterlogged=false]",
        );
        cmd(
            &mut g,
            "setblock 9 100 10 minecraft:chest[facing=north,type=left,waterlogged=false]",
        );
        g.tick_once_for_test();
        frames(&rx);
        g.teleport_player_for_test(0, 10.5, 100.5, 10.5);
        g.handle(Inbound::OpenContainer {
            conn: 0,
            x: 10,
            y: 100,
            z: 10,
        });
        let fs = frames(&rx);
        let open = find_frame(&fs, PACKET_OPEN_SCREEN).expect("open_screen");
        assert_eq!(open[1], MENU_GENERIC_9X6 as u8, "double chest is 9x6");
        let content = find_frame(&fs, PACKET_CONTAINER_SET_CONTENT).expect("set_content");
        let (_, _, slots, _) = crate::inventory::decode_container_set_content(&content).unwrap();
        assert_eq!(slots.len(), 54 + 36);
        // Give the player a stack and quick-move it into the chest.
        g.handle(Inbound::Give {
            conn: 0,
            item: "minecraft:stone".to_string(),
            count: 64,
        });
        frames(&rx);
        // Player hotbar container 0 = menu slot 54 + 27 = 81.
        g.handle(Inbound::ContainerClick {
            conn: 0,
            click: click_raw(1, 2, 81, 0, crate::inventory::ClickKind::QuickMove),
        });
        let fs = frames(&rx);
        let content = find_frame(&fs, PACKET_CONTAINER_SET_CONTENT).expect("set_content");
        let (_, _, slots, _) = crate::inventory::decode_container_set_content(&content).unwrap();
        assert_eq!(
            slots[0].as_ref().map(crate::inventory::ItemStack::count),
            Some(64),
            "quick-move lands in the first grid slot"
        );
        assert_eq!(
            g.container_get_for_test((10, 100, 10), 0)
                .as_ref()
                .map(crate::inventory::ItemStack::count),
            Some(64),
            "menu slot 0 is the right half (the opened chest)"
        );
        // Lift from slot 0 and place into slot 27 (the left half).
        g.handle(Inbound::ContainerClick {
            conn: 0,
            click: click_raw(1, 3, 0, 0, crate::inventory::ClickKind::Pickup),
        });
        frames(&rx);
        g.handle(Inbound::ContainerClick {
            conn: 0,
            click: click_raw(1, 4, 27, 0, crate::inventory::ClickKind::Pickup),
        });
        let fs = frames(&rx);
        let content = find_frame(&fs, PACKET_CONTAINER_SET_CONTENT).expect("set_content");
        let (_, _, slots, _) = crate::inventory::decode_container_set_content(&content).unwrap();
        assert_eq!(
            slots[27].as_ref().map(crate::inventory::ItemStack::count),
            Some(64)
        );
        assert_eq!(
            g.container_get_for_test((9, 100, 10), 0)
                .as_ref()
                .map(crate::inventory::ItemStack::count),
            Some(64),
            "menu slot 27 routes into the left half's slot 0"
        );
        assert!(g.container_get_for_test((10, 100, 10), 0).is_none());
        // Opening the LEFT half yields the same merged order.
        g.handle(Inbound::ContainerClose {
            conn: 0,
            container_id: 1,
        });
        g.handle(Inbound::OpenContainer {
            conn: 0,
            x: 9,
            y: 100,
            z: 10,
        });
        let fs = frames(&rx);
        let content = find_frame(&fs, PACKET_CONTAINER_SET_CONTENT).expect("set_content");
        let (_, _, slots, _) = crate::inventory::decode_container_set_content(&content).unwrap();
        assert_eq!(
            slots[27].as_ref().map(crate::inventory::ItemStack::count),
            Some(64),
            "the merged view is identical from either half"
        );
    }

    #[test]
    fn hopper_pulls_from_chest_above_at_8gt() {
        let (mut g, rx) = harness();
        cmd(
            &mut g,
            "setblock 10 101 10 minecraft:chest[facing=north,type=single,waterlogged=false]",
        );
        cmd(
            &mut g,
            "setblock 10 100 10 minecraft:hopper[enabled=true,facing=down]",
        );
        g.tick_once_for_test();
        frames(&rx);
        g.container_set_for_test((10, 101, 10), 0, stone());
        // T1: the hopper pulls one item.
        g.tick_once_for_test();
        assert_eq!(
            g.container_get_for_test((10, 100, 10), 0)
                .as_ref()
                .map(crate::inventory::ItemStack::count),
            Some(1)
        );
        assert_eq!(
            g.container_get_for_test((10, 101, 10), 0)
                .as_ref()
                .map(crate::inventory::ItemStack::count),
            Some(63)
        );
        // Exactly 8gt between transfers: ticks +1..+7 hold, +8 moves.
        for i in 1..8 {
            g.tick_once_for_test();
            assert_eq!(
                g.container_get_for_test((10, 101, 10), 0)
                    .as_ref()
                    .map(crate::inventory::ItemStack::count),
                Some(63),
                "no transfer before the 8gt cooldown, tick +{i}"
            );
        }
        g.tick_once_for_test();
        assert_eq!(
            g.container_get_for_test((10, 101, 10), 0)
                .as_ref()
                .map(crate::inventory::ItemStack::count),
            Some(62),
            "the 8th tick transfers again"
        );
        // The changed containers broadcast block_entity_data.
        let fs = frames(&rx);
        assert!(
            fs.iter().any(|(id, _)| *id == PACKET_BLOCK_ENTITY_DATA),
            "block_entity_data frames: {:?}",
            fs.iter().map(|(id, _)| *id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn hopper_pushes_into_chest_below() {
        let (mut g, _rx) = harness();
        cmd(
            &mut g,
            "setblock 10 100 10 minecraft:chest[facing=north,type=single,waterlogged=false]",
        );
        cmd(
            &mut g,
            "setblock 10 101 10 minecraft:hopper[enabled=true,facing=down]",
        );
        g.tick_once_for_test();
        g.container_set_for_test((10, 101, 10), 2, stone());
        g.tick_once_for_test();
        assert_eq!(
            g.container_get_for_test((10, 100, 10), 0)
                .as_ref()
                .map(crate::inventory::ItemStack::count),
            Some(1),
            "push lands in the chest's first slot"
        );
        assert_eq!(
            g.container_get_for_test((10, 101, 10), 2)
                .as_ref()
                .map(crate::inventory::ItemStack::count),
            Some(63)
        );
    }

    #[test]
    fn hopper_chain_gets_the_one_tick_skip() {
        let (mut g, _rx) = harness();
        cmd(
            &mut g,
            "setblock 10 100 10 minecraft:hopper[enabled=true,facing=down]",
        );
        cmd(
            &mut g,
            "setblock 10 101 10 minecraft:hopper[enabled=true,facing=down]",
        );
        g.tick_once_for_test();
        g.container_set_for_test((10, 101, 10), 0, stone());
        // The lower hopper (sorted first) ticks before the upper one: it
        // pulls one item, then the upper hopper's own tick pushes one more
        // down in the same gt. The pull itself counts as a transfer, so
        // its cooldown reset overwrites the skip: both sit at 8.
        g.tick_once_for_test();
        assert_eq!(
            g.container_get_for_test((10, 100, 10), 0)
                .as_ref()
                .map(crate::inventory::ItemStack::count),
            Some(2),
            "one item pulled plus one pushed in the same tick"
        );
        assert_eq!(g.hopper_cooldown_for_test((10, 100, 10)), Some(8));
        assert_eq!(g.hopper_cooldown_for_test((10, 101, 10)), Some(8));
        // The skip survives only when the receiver moved nothing itself:
        // the facing hopper ticks first (empty, nothing above), then the
        // west-facing source pushes into it while empty, landing cooldown
        // 7 - one tick short of the full cycle.
        let (mut g2, _rx2) = harness();
        cmd(
            &mut g2,
            "setblock 9 99 10 minecraft:chest[facing=north,type=single,waterlogged=false]",
        );
        cmd(
            &mut g2,
            "setblock 9 100 10 minecraft:hopper[enabled=true,facing=down]",
        );
        cmd(
            &mut g2,
            "setblock 10 100 10 minecraft:hopper[enabled=true,facing=west]",
        );
        g2.tick_once_for_test();
        g2.container_set_for_test((10, 100, 10), 0, stone());
        g2.tick_once_for_test();
        assert_eq!(
            g2.hopper_cooldown_for_test((9, 100, 10)),
            Some(7),
            "the push into an idle hopper carries the one-tick skip"
        );
        assert_eq!(
            g2.container_get_for_test((9, 100, 10), 0)
                .as_ref()
                .map(crate::inventory::ItemStack::count),
            Some(1)
        );
        for i in 1..7 {
            g2.tick_once_for_test();
            assert!(
                g2.container_get_for_test((9, 99, 10), 0).is_none(),
                "the shortened cycle flushes on the 7th tick, not the 8th (+{i})"
            );
        }
        g2.tick_once_for_test();
        assert_eq!(
            g2.container_get_for_test((9, 99, 10), 0)
                .as_ref()
                .map(crate::inventory::ItemStack::count),
            Some(1),
            "the skip lands the flush one tick early"
        );
        // A chest-to-hopper pull carries no skip: the cooldown resets to
        // the full 8.
        let (mut g3, _rx3) = harness();
        cmd(
            &mut g3,
            "setblock 11 101 10 minecraft:chest[facing=north,type=single,waterlogged=false]",
        );
        cmd(
            &mut g3,
            "setblock 11 100 10 minecraft:hopper[enabled=true,facing=down]",
        );
        g3.tick_once_for_test();
        g3.container_set_for_test((11, 101, 10), 0, stone());
        g3.tick_once_for_test();
        assert_eq!(g3.hopper_cooldown_for_test((11, 100, 10)), Some(8));
    }

    #[test]
    fn full_target_is_a_noop_without_cooldown() {
        let (mut g, _rx) = harness();
        cmd(
            &mut g,
            "setblock 10 100 10 minecraft:chest[facing=north,type=single,waterlogged=false]",
        );
        cmd(
            &mut g,
            "setblock 10 101 10 minecraft:hopper[enabled=true,facing=down]",
        );
        g.tick_once_for_test();
        for slot in 0..CHEST_SIZE {
            g.container_set_for_test((10, 100, 10), slot, stone());
        }
        g.container_set_for_test((10, 101, 10), 0, stone());
        g.tick_once_for_test();
        g.tick_once_for_test();
        // Nothing moved and the cooldown never engages: the hopper keeps
        // retrying every tick.
        assert_eq!(g.hopper_cooldown_for_test((10, 101, 10)), Some(0));
        assert_eq!(
            g.container_get_for_test((10, 100, 10), 0)
                .as_ref()
                .map(crate::inventory::ItemStack::count),
            Some(64)
        );
        assert_eq!(
            g.container_get_for_test((10, 101, 10), 0)
                .as_ref()
                .map(crate::inventory::ItemStack::count),
            Some(64)
        );
        // Freeing one slot lets the push through.
        g.container_set_for_test((10, 100, 10), 26, None);
        g.tick_once_for_test();
        assert_eq!(
            g.container_get_for_test((10, 100, 10), 26)
                .as_ref()
                .map(crate::inventory::ItemStack::count),
            Some(1)
        );
    }

    #[test]
    fn comparator_reads_container_fill() {
        let (mut g, _rx) = harness();
        cmd(
            &mut g,
            "setblock 10 100 10 minecraft:chest[facing=north,type=single,waterlogged=false]",
        );
        cmd(
            &mut g,
            "setblock 10 100 11 minecraft:comparator[facing=north,mode=compare,powered=false]",
        );
        g.tick_once_for_test();
        g.tick_once_for_test();
        // The stored output: the placement default reads as 0 (an empty
        // chest never turns the comparator stale, so no toggle fires).
        let read = |g: &Game| {
            g.comparator_outputs
                .get(&(10, 100, 11))
                .copied()
                .unwrap_or(0)
        };
        assert_eq!(read(&g), 0);
        // One item in one slot: fill 1/64/27 -> floor(14/27) + 1 = 1.
        g.container_set_for_test((10, 100, 10), 0, stones(1));
        g.tick_once_for_test();
        g.tick_once_for_test();
        assert_eq!(read(&g), 1);
        // 13 full slots: fill 13/27 -> floor(6.74) + 1 = 7.
        g.container_set_for_test((10, 100, 10), 0, None);
        for slot in 0..13 {
            g.container_set_for_test((10, 100, 10), slot, stone());
        }
        g.tick_once_for_test();
        g.tick_once_for_test();
        assert_eq!(read(&g), 7);
        // All 27 slots full: fill 1 -> 15.
        for slot in 13..CHEST_SIZE {
            g.container_set_for_test((10, 100, 10), slot, stone());
        }
        g.tick_once_for_test();
        g.tick_once_for_test();
        assert_eq!(read(&g), 15);
        // A conductor above the chest blocks the comparator read; the
        // next chest change re-evaluates the neighbor comparator.
        cmd(&mut g, "setblock 10 101 10 minecraft:stone");
        g.tick_once_for_test();
        g.container_set_for_test((10, 100, 10), 0, None);
        g.tick_once_for_test();
        g.tick_once_for_test();
        assert_eq!(read(&g), 0, "blocked chest reads as empty");
    }

    #[test]
    fn block_entity_data_and_chunk_entries() {
        let (mut g, rx) = harness();
        cmd(
            &mut g,
            "setblock 10 100 10 minecraft:chest[facing=north,type=single,waterlogged=false]",
        );
        g.tick_once_for_test();
        g.container_set_for_test((10, 100, 10), 0, stones(3));
        g.tick_once_for_test();
        let fs = frames(&rx);
        let be = find_frame(&fs, PACKET_BLOCK_ENTITY_DATA).expect("block_entity_data");
        let packed = i64::from_be_bytes(be[0..8].try_into().unwrap());
        assert_eq!(packed, pack_block_pos((10, 100, 10)));
        let mut o = 8usize;
        assert_eq!(read_varint(&be, &mut o), BE_TYPE_CHEST as i64);
        // NBT root compound, Items list, one stack.
        assert_eq!(be[o], 0x0a);
        assert!(
            be[o + 1..].starts_with(&[0x09, 0x00, 0x05, b'I', b't', b'e', b'm', b's']),
            "Items list tag"
        );
        assert!(
            be.windows(15).any(|w| w == b"minecraft:stone"),
            "the item name rides in the NBT"
        );
        // The cached chunk carries the entry for later sends.
        let chunk = g.chunks.get(&(0, 0)).expect("chunk cached");
        assert_eq!(chunk.wire.block_entities.len(), 1);
        let entry = &chunk.wire.block_entities[0];
        assert_eq!(entry.ty, BE_TYPE_CHEST);
        assert_eq!(entry.packed_xz, (10 << 4) | 10);
        assert_eq!(entry.y, 100);
        // Removing the block discards the entry.
        cmd(&mut g, "setblock 10 100 10 minecraft:air");
        g.tick_once_for_test();
        let chunk = g.chunks.get(&(0, 0)).expect("chunk cached");
        assert!(
            chunk.wire.block_entities.is_empty(),
            "breaking the chest drops its block entity"
        );
        assert!(g.container_get_for_test((10, 100, 10), 0).is_none());
    }

    #[test]
    fn breaking_chest_closes_menu_and_discards() {
        let (mut g, rx) = harness();
        cmd(
            &mut g,
            "setblock 10 100 10 minecraft:chest[facing=north,type=single,waterlogged=false]",
        );
        g.tick_once_for_test();
        g.teleport_player_for_test(0, 10.5, 100.5, 10.5);
        g.handle(Inbound::OpenContainer {
            conn: 0,
            x: 10,
            y: 100,
            z: 10,
        });
        frames(&rx);
        g.container_set_for_test((10, 100, 10), 0, stone());
        cmd(&mut g, "setblock 10 100 10 minecraft:air");
        let fs = frames(&rx);
        // Server-initiated close packet.
        let close = find_frame(&fs, PACKET_CONTAINER_CLOSE).expect("close packet");
        assert_eq!(close[0], 1);
        assert!(g.container_get_for_test((10, 100, 10), 0).is_none());
        // Re-placing yields a fresh, empty container.
        cmd(
            &mut g,
            "setblock 10 100 10 minecraft:chest[facing=north,type=single,waterlogged=false]",
        );
        g.tick_once_for_test();
        assert!(g.container_get_for_test((10, 100, 10), 0).is_none());
    }

    #[test]
    fn hopper_disables_under_power() {
        let (mut g, _rx) = harness();
        cmd(
            &mut g,
            "setblock 10 101 10 minecraft:chest[facing=north,type=single,waterlogged=false]",
        );
        cmd(
            &mut g,
            "setblock 10 100 10 minecraft:hopper[enabled=true,facing=down]",
        );
        cmd(
            &mut g,
            "setblock 11 100 10 minecraft:lever[face=floor,facing=north,powered=true]",
        );
        g.tick_once_for_test();
        g.tick_once_for_test();
        assert_eq!(
            g.block_label_for_test(10, 100, 10),
            "minecraft:hopper[enabled=false,facing=down]",
            "the powered neighbor disables the hopper"
        );
        g.container_set_for_test((10, 101, 10), 0, stone());
        g.tick_once_for_test();
        assert!(
            g.container_get_for_test((10, 100, 10), 0).is_none(),
            "a disabled hopper transfers nothing"
        );
        // Power off: transfers resume.
        cmd(
            &mut g,
            "setblock 11 100 10 minecraft:lever[face=floor,facing=north,powered=false]",
        );
        g.tick_once_for_test();
        g.tick_once_for_test();
        g.tick_once_for_test();
        assert!(
            g.container_get_for_test((10, 100, 10), 0).is_some(),
            "an enabled hopper transfers again"
        );
    }

    #[test]
    fn blocked_chest_refuses_to_open() {
        let (mut g, rx) = harness();
        cmd(
            &mut g,
            "setblock 10 100 10 minecraft:chest[facing=north,type=single,waterlogged=false]",
        );
        cmd(&mut g, "setblock 10 101 10 minecraft:stone");
        g.tick_once_for_test();
        g.teleport_player_for_test(0, 10.5, 100.5, 10.5);
        g.handle(Inbound::OpenContainer {
            conn: 0,
            x: 10,
            y: 100,
            z: 10,
        });
        let fs = frames(&rx);
        assert!(
            find_frame(&fs, PACKET_OPEN_SCREEN).is_none(),
            "a blocked chest does not open"
        );
    }

    #[test]
    fn distance_closes_open_menus() {
        let (mut g, rx) = harness();
        cmd(
            &mut g,
            "setblock 10 100 10 minecraft:chest[facing=north,type=single,waterlogged=false]",
        );
        g.tick_once_for_test();
        g.teleport_player_for_test(0, 10.5, 100.5, 10.5);
        g.handle(Inbound::OpenContainer {
            conn: 0,
            x: 10,
            y: 100,
            z: 10,
        });
        frames(&rx);
        // Far away: the next tick closes the menu server-side.
        g.teleport_player_for_test(0, 0.5, 0.5, 0.5);
        g.tick_once_for_test();
        let fs = frames(&rx);
        assert!(find_frame(&fs, PACKET_CONTAINER_CLOSE).is_some());
    }

    #[test]
    fn hopper_pulls_merge_double_chest_above() {
        let (mut g, _rx) = harness();
        cmd(
            &mut g,
            "setblock 10 101 10 minecraft:chest[facing=north,type=right,waterlogged=false]",
        );
        cmd(
            &mut g,
            "setblock 9 101 10 minecraft:chest[facing=north,type=left,waterlogged=false]",
        );
        cmd(
            &mut g,
            "setblock 10 100 10 minecraft:hopper[enabled=true,facing=down]",
        );
        g.tick_once_for_test();
        // An item in the left half (merged slot 27): the hopper's pull
        // scans the pair, not just the chest above it.
        g.container_set_for_test((9, 101, 10), 0, stones(2));
        g.tick_once_for_test();
        assert_eq!(
            g.container_get_for_test((10, 100, 10), 0)
                .as_ref()
                .map(crate::inventory::ItemStack::count),
            Some(1)
        );
        assert_eq!(
            g.container_get_for_test((9, 101, 10), 0)
                .as_ref()
                .map(crate::inventory::ItemStack::count),
            Some(1)
        );
    }

    fn click_raw(
        container_id: i32,
        state_id: i32,
        slot: i16,
        button: i8,
        kind: crate::inventory::ClickKind,
    ) -> crate::inventory::ContainerClick {
        crate::inventory::ContainerClick {
            container_id,
            state_id,
            slot_num: slot,
            button_num: button,
            kind,
            changed_slots: Vec::new(),
            carried: crate::inventory::HashedStack::default(),
        }
    }
}
