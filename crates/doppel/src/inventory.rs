//! Item stacks, the player inventory, and container clicks (Minecraft 26.3).
//!
//! Registry numeric ids (items, data component types, menus) are
//! runtime-assigned by load
//! order; the values used below were extracted from the pinned vanilla
//! jar's own data generator (`--reports`), the same source as
//! `pins/blocks.json`, and hold for the pinned 26.3 build only.
//!
//! Scope: the always-open player inventory menu (containerId 0) plus the
//! shared click engine (`MenuSlots`): container menus (containers.rs)
//! implement the same trait over their own storage.

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{bail, Context, Result};
use doppel_protocol::{write_varint, Reader};

use crate::game::{ConnId, Game};

// ---------------------------------------------------------------------
// Wire packet ids (clientbound play state unless noted)
// ---------------------------------------------------------------------

/// `container_set_content`: the join capture carries this
/// id with the 46-slot player inventory body (see `join_set_content_golden`).
pub const PACKET_CONTAINER_SET_CONTENT: i32 = 0x12;
/// `container_set_data`. 26.3 registration order; unused until a
/// menu with data slots exists.
pub const PACKET_CONTAINER_SET_DATA: i32 = 0x13;
/// `container_set_slot`. 26.3 registration order, adjacent to the confirmed
/// 0x12 anchor.
pub const PACKET_CONTAINER_SET_SLOT: i32 = 0x14;
/// `set_cursor_item` (carried stack, no containerId). 26.3 registration
/// order - wire-verify against the oracle before relying on it.
pub const PACKET_SET_CURSOR_ITEM: i32 = 0x62;
/// `set_held_slot`: the join capture sends `6b 00`
/// immediately after the abilities burst.
pub const PACKET_SET_HELD_SLOT: i32 = 0x6b;
/// `set_player_inventory`. 26.3 registration order - wire-verify.
pub const PACKET_SET_PLAYER_INVENTORY: i32 = 0x6e;

/// Serverbound `container_click`.
pub const SERVERBOUND_CONTAINER_CLICK: i32 = 0x12;
/// Serverbound `set_carried_item` (hotbar select, i16 slot). Derived from
/// the 26.3 registration order: the 26.2 tail of the protocol notes said
/// 0x35, and 26.3's inserted `punch` at 0x2e shifts everything after it +1.
/// TODO(inventory): wire-verify with the oracle before treating as pinned.
pub const SERVERBOUND_SET_CARRIED_ITEM: i32 = 0x36;

/// Serverbound `container_close`, for completeness. 26.3 registration order.
pub const SERVERBOUND_CONTAINER_CLOSE: i32 = 0x13;

// ---------------------------------------------------------------------
// Data components
// ---------------------------------------------------------------------

/// `minecraft:data_component_type` registry ids (26.3 generator order).
/// Only the payload shapes Doppel can currently decode are listed; the
/// names are comments, the numbers are what goes on the wire.
pub mod component {
    pub const CUSTOM_DATA: i32 = 0;
    pub const MAX_STACK_SIZE: i32 = 1;
    pub const MAX_DAMAGE: i32 = 2;
    pub const DAMAGE: i32 = 3;
    pub const UNBREAKABLE: i32 = 4;
    pub const CUSTOM_NAME: i32 = 6;
    pub const ITEM_NAME: i32 = 9;
    pub const RARITY: i32 = 12;
    pub const REPAIR_COST: i32 = 19;
}

/// One component value in a patch. `Unit` and `VarInt` cover the scalar
/// components (and round-trip); `Bytes` passes a pre-encoded payload
/// through on encode only - decoding it back requires the component's own
/// stream codec, which arrives with full component support.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComponentValue {
    /// Zero-byte payload (`unbreakable`, ...).
    Unit,
    /// VAR_INT payload (`max_stack_size`, `damage`, `rarity`, ...).
    VarInt(i32),
    /// Pre-encoded payload (encode-only passthrough).
    Bytes(Vec<u8>),
}

/// A `DataComponentPatch`: added entries (type id + value) and the set of
/// component types removed relative to the item's prototype. Encode order
/// is all added entries first, then all removed ids.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ComponentPatch {
    pub added: Vec<(i32, ComponentValue)>,
    pub removed: Vec<i32>,
}

impl ComponentPatch {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }

    fn encode(&self, buf: &mut Vec<u8>) {
        write_varint(buf, self.added.len() as i32);
        for (ty, value) in &self.added {
            write_varint(buf, *ty);
            match value {
                ComponentValue::Unit => {}
                ComponentValue::VarInt(v) => write_varint(buf, *v),
                ComponentValue::Bytes(bytes) => buf.extend_from_slice(bytes),
            }
        }
        write_varint(buf, self.removed.len() as i32);
        for ty in &self.removed {
            write_varint(buf, *ty);
        }
    }

    /// Decodes a patch. Every added component's payload must be decodable
    /// by the small table below; unknown component types error instead of
    /// guessing (a wrong skip would desync the whole stream).
    fn decode(r: &mut Reader) -> Result<ComponentPatch> {
        let mut patch = ComponentPatch::default();
        let added = read_count(r, 256, "component patch")?;
        for _ in 0..added {
            let ty = r.read_varint().context("component type id")?;
            let value = match ty {
                component::MAX_STACK_SIZE
                | component::MAX_DAMAGE
                | component::DAMAGE
                | component::RARITY
                | component::REPAIR_COST => {
                    ComponentValue::VarInt(r.read_varint().context("component value")?)
                }
                component::UNBREAKABLE => ComponentValue::Unit,
                other => bail!("no decoder for data component type {other}"),
            };
            patch.added.push((ty, value));
        }
        let removed = read_count(r, 256, "removed components")?;
        for _ in 0..removed {
            patch
                .removed
                .push(r.read_varint().context("removed component type id")?);
        }
        Ok(patch)
    }
}

/// Reads a VarInt collection size, bounded like vanilla's codec caps.
fn read_count(r: &mut Reader, max: usize, what: &'static str) -> Result<usize> {
    let n = r.read_varint().context(what)?;
    if n < 0 || n as usize > max {
        bail!("{what} count {n} out of bounds (max {max})");
    }
    Ok(n as usize)
}

// ---------------------------------------------------------------------
// ItemStack
// ---------------------------------------------------------------------

/// A stack: count, item registry id, and the component patch relative to
/// the item's prototype. An empty slot is `None` (vanilla's EMPTY sentinel).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ItemStack {
    count: i32,
    item: i32,
    patch: ComponentPatch,
}

/// Vanilla's absolute cap on any stack count (`ABSOLUTE_MAX_STACK_SIZE`).
pub const ABSOLUTE_MAX_STACK_SIZE: i32 = 99;
/// The prototype default when the patch
/// carries no `max_stack_size`. NOTE(inventory): per-item prototypes need
/// the item registry pin; until then every item defaults to 64, so
/// unstackable items (swords, ...) merge like stackables on click paths.
pub const DEFAULT_MAX_STACK_SIZE: i32 = 64;

impl ItemStack {
    pub fn new(item: i32, count: i32) -> ItemStack {
        ItemStack {
            count,
            item,
            patch: ComponentPatch::default(),
        }
    }

    pub fn with_patch(mut self, patch: ComponentPatch) -> ItemStack {
        self.patch = patch;
        self
    }

    pub fn count(&self) -> i32 {
        self.count
    }

    pub fn item(&self) -> i32 {
        self.item
    }

    pub fn patch(&self) -> &ComponentPatch {
        &self.patch
    }

    pub fn is_empty(&self) -> bool {
        self.count <= 0
    }

    /// `getMaxStackSize`: the patch's `max_stack_size` when present, else
    /// the item default (see `DEFAULT_MAX_STACK_SIZE` note).
    pub fn max_stack_size(&self) -> i32 {
        for (ty, value) in &self.patch.added {
            if *ty == component::MAX_STACK_SIZE {
                if let ComponentValue::VarInt(v) = value {
                    return (*v).clamp(1, ABSOLUTE_MAX_STACK_SIZE);
                }
            }
        }
        DEFAULT_MAX_STACK_SIZE
    }

    pub fn with_count(&self, count: i32) -> ItemStack {
        let mut out = self.clone();
        out.count = count;
        out
    }

    pub fn set_count(&mut self, count: i32) {
        self.count = count;
    }

    /// `split`: a copy of the first `n` items (capped at the count); the
    /// source keeps the remainder.
    pub fn split(&mut self, n: i32) -> ItemStack {
        let real = n.min(self.count).max(0);
        self.count -= real;
        self.with_count(real)
    }

    pub fn grow(&mut self, n: i32) {
        self.count += n;
    }

    pub fn shrink(&mut self, n: i32) {
        self.count -= n;
    }

    /// `isSameItemSameComponents`.
    pub fn same_item_same_components(a: &ItemStack, b: &ItemStack) -> bool {
        a.item == b.item && a.patch == b.patch
    }

    /// `ItemStack.matches`: count plus item+components equality.
    pub fn matches(a: &ItemStack, b: &ItemStack) -> bool {
        a.count == b.count && ItemStack::same_item_same_components(a, b)
    }
}

/// `ItemStack.OPTIONAL_STREAM_CODEC`: count <= 0 (encoded as a single
/// VarInt 0) means the slot is empty; otherwise count, item id, patch.
pub fn encode_item_stack(buf: &mut Vec<u8>, stack: Option<&ItemStack>) {
    let Some(stack) = stack.filter(|s| !s.is_empty()) else {
        write_varint(buf, 0);
        return;
    };
    write_varint(buf, stack.count);
    write_varint(buf, stack.item);
    stack.patch.encode(buf);
}

/// Decodes an optional ItemStack.
pub fn decode_item_stack(r: &mut Reader) -> Result<Option<ItemStack>> {
    let count = r.read_varint().context("item count")?;
    if count <= 0 {
        return Ok(None);
    }
    let item = r.read_varint().context("item id")?;
    let patch = ComponentPatch::decode(r)?;
    Ok(Some(ItemStack { count, item, patch }))
}

// ---------------------------------------------------------------------
// HashedStack (client click prediction payloads)
// ---------------------------------------------------------------------

/// A `HashedStack`: the client's post-click belief about a slot, carrying
/// CRC32C component hashes instead of values. Doppel decodes these for
/// validation but does not yet apply them as predictions (see the NOTE in
/// `Game::container_clicked`); the hashes are never item data.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HashedStack {
    pub present: bool,
    pub item: i32,
    pub count: i32,
    /// (component type id, CRC32C hash).
    pub added: Vec<(i32, i32)>,
    pub removed: Vec<i32>,
}

impl HashedStack {
    /// `HashedStack.STREAM_CODEC` = optional(ActualItem): a boolean prefix,
    /// then item VarInt + count VarInt + the hashed patch map (added
    /// {type, i32 hash} pairs first, then the removed type set).
    pub fn encode(&self, buf: &mut Vec<u8>) {
        if !self.present {
            buf.push(0);
            return;
        }
        buf.push(1);
        write_varint(buf, self.item);
        write_varint(buf, self.count);
        write_varint(buf, self.added.len() as i32);
        for (ty, hash) in &self.added {
            write_varint(buf, *ty);
            buf.extend_from_slice(&hash.to_be_bytes());
        }
        write_varint(buf, self.removed.len() as i32);
        for ty in &self.removed {
            write_varint(buf, *ty);
        }
    }

    pub fn decode(r: &mut Reader) -> Result<HashedStack> {
        let mut out = HashedStack::default();
        if r.read_u8().context("hashed stack present flag")? == 0 {
            return Ok(out);
        }
        out.present = true;
        out.item = r.read_varint().context("hashed stack item")?;
        out.count = r.read_varint().context("hashed stack count")?;
        for _ in 0..read_count(r, 256, "hashed patch additions")? {
            let ty = r.read_varint().context("hashed component type")?;
            let hash = read_i32be(r)?;
            out.added.push((ty, hash));
        }
        for _ in 0..read_count(r, 256, "hashed patch removals")? {
            out.removed
                .push(r.read_varint().context("hashed removed component")?);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------
// Item registry ids
// ---------------------------------------------------------------------

/// Curated name -> item registry id table for the pinned 26.3 build,
/// extracted from the vanilla data generator's registries report (same
/// flow as `pins/blocks.json`). The full table lands as a pin when the
/// oracle's registry extraction grows an items pass; `ItemTable::from_pairs`
/// is the injection point.
const ITEM_IDS: &[(&str, i32)] = &[
    ("minecraft:air", 0),
    ("minecraft:stone", 1),
    ("minecraft:granite", 2),
    ("minecraft:polished_granite", 3),
    ("minecraft:diorite", 4),
    ("minecraft:polished_diorite", 5),
    ("minecraft:andesite", 6),
    ("minecraft:polished_andesite", 7),
    ("minecraft:deepslate", 8),
    ("minecraft:cobbled_deepslate", 9),
    ("minecraft:grass_block", 54),
    ("minecraft:dirt", 55),
    ("minecraft:cobblestone", 62),
    ("minecraft:oak_planks", 63),
    ("minecraft:glass", 231),
    ("minecraft:oak_log", 163),
    ("minecraft:oak_slab", 341),
    ("minecraft:crafting_table", 405),
    ("minecraft:furnace", 407),
    ("minecraft:chest", 404),
    ("minecraft:trapped_chest", 852),
    ("minecraft:ender_chest", 513),
    ("minecraft:barrel", 1506),
    ("minecraft:hopper", 834),
    ("minecraft:dispenser", 835),
    ("minecraft:dropper", 836),
    ("minecraft:shulker_box", 656),
    ("minecraft:torch", 395),
    ("minecraft:carved_pumpkin", 431),
    ("minecraft:redstone", 824),
    ("minecraft:redstone_torch", 825),
    ("minecraft:lever", 839),
    ("minecraft:observer", 833),
    ("minecraft:repeater", 827),
    ("minecraft:comparator", 828),
    ("minecraft:piston", 829),
    ("minecraft:sticky_piston", 830),
    ("minecraft:slime_block", 831),
    ("minecraft:honey_block", 832),
    ("minecraft:coal", 1010),
    ("minecraft:diamond", 1012),
    ("minecraft:iron_ingot", 1018),
    ("minecraft:gold_ingot", 1022),
    ("minecraft:stick", 1060),
    ("minecraft:arrow", 1009),
    ("minecraft:bow", 1008),
    ("minecraft:saddle", 949),
    ("minecraft:elytra", 974),
    ("minecraft:wolf_armor", 1004),
    ("minecraft:golden_apple", 1100),
    ("minecraft:leather_helmet", 1068),
    ("minecraft:leather_chestplate", 1069),
    ("minecraft:leather_leggings", 1070),
    ("minecraft:leather_boots", 1071),
    ("minecraft:diamond_sword", 1050),
    ("minecraft:diamond_pickaxe", 1052),
    ("minecraft:netherite_sword", 1055),
];

/// An item name -> registry id table. Built from the curated vanilla
/// subset by default; a future items pin constructs one with
/// `from_pairs`.
pub struct ItemTable {
    by_name: HashMap<String, i32>,
}

impl ItemTable {
    pub fn from_pairs<'a>(pairs: impl IntoIterator<Item = (&'a str, i32)>) -> ItemTable {
        ItemTable {
            by_name: pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        }
    }

    /// The curated vanilla-26.3 subset.
    pub fn vanilla_subset() -> ItemTable {
        ItemTable::from_pairs(ITEM_IDS.iter().copied())
    }

    /// Resolves an item name, applying the `minecraft:` default namespace
    /// like vanilla's id parser.
    pub fn id_of(&self, name: &str) -> Option<i32> {
        if name.contains(':') {
            self.by_name.get(name).copied()
        } else {
            self.by_name.get(&format!("minecraft:{name}")).copied()
        }
    }
}

fn item_table() -> &'static ItemTable {
    static TABLE: OnceLock<ItemTable> = OnceLock::new();
    TABLE.get_or_init(ItemTable::vanilla_subset)
}

/// Resolves an item name against the default table.
pub fn item_id(name: &str) -> Option<i32> {
    item_table().id_of(name)
}

/// Reverse lookup: the registry name of a curated item id (block-entity
/// NBT writes item names, not ids).
pub fn item_name(id: i32) -> Option<&'static str> {
    ITEM_IDS.iter().find(|(_, v)| *v == id).map(|(k, _)| *k)
}

// ---------------------------------------------------------------------
// Player inventory (container-slot space)
// ---------------------------------------------------------------------

/// Container-slot layout: 0-8 hotbar, 9-35 main, 36-39 armor
/// FEET/LEGS/CHEST/HEAD, 40 offhand, 41 body, 42 saddle.
pub const INVENTORY_SIZE: usize = 36;
pub const SLOT_FEET: usize = 36;
pub const SLOT_LEGS: usize = 37;
pub const SLOT_CHEST: usize = 38;
pub const SLOT_HEAD: usize = 39;
pub const SLOT_OFFHAND: usize = 40;
pub const SLOT_BODY_ARMOR: usize = 41;
pub const SLOT_SADDLE: usize = 42;
/// `getContainerSize()`: 36 item slots + 7 equipment slots.
pub const TOTAL_SLOTS: usize = 43;

/// The player's inventory in raw container-slot space.
#[derive(Clone, Debug)]
pub struct PlayerInventory {
    slots: Vec<Option<ItemStack>>,
    selected: u8,
}

impl Default for PlayerInventory {
    fn default() -> Self {
        PlayerInventory {
            slots: vec![None; TOTAL_SLOTS],
            selected: 0,
        }
    }
}

impl PlayerInventory {
    pub fn get(&self, slot: usize) -> Option<ItemStack> {
        self.slots.get(slot).cloned().flatten()
    }

    pub fn set(&mut self, slot: usize, stack: Option<ItemStack>) {
        if let Some(cell) = self.slots.get_mut(slot) {
            *cell = stack.filter(|s| !s.is_empty());
        }
    }

    pub fn selected(&self) -> u8 {
        self.selected
    }

    /// Hotbar indices only (vanilla rejects anything else).
    pub fn set_selected(&mut self, slot: u8) {
        if slot < 9 {
            self.selected = slot;
        }
    }

    /// `add(-1, stack)` simplified: merge into partial stacks (selected
    /// hotbar slot, offhand, then 0-35), then fill the first empty slot
    /// (0-35, hotbar first). Returns whatever did not fit.
    pub fn add(&mut self, stack: ItemStack) -> Option<ItemStack> {
        let mut stack = stack;
        let order = {
            let mut order: Vec<usize> = Vec::with_capacity(TOTAL_SLOTS);
            order.push(self.selected as usize);
            order.push(SLOT_OFFHAND);
            order.extend(0..INVENTORY_SIZE);
            order
        };
        for slot in order {
            let Some(occupant) = self.get(slot) else {
                continue;
            };
            if !ItemStack::same_item_same_components(&occupant, &stack) {
                continue;
            }
            let room = occupant.max_stack_size() - occupant.count();
            if room > 0 {
                let moved = room.min(stack.count());
                self.set(slot, Some(occupant.with_count(occupant.count() + moved)));
                stack.shrink(moved);
            }
            if stack.is_empty() {
                return None;
            }
        }
        for slot in 0..INVENTORY_SIZE {
            if self.get(slot).is_none() {
                let placed = stack.split(stack.max_stack_size());
                self.set(slot, Some(placed));
                if stack.is_empty() {
                    return None;
                }
            }
        }
        Some(stack)
    }
}

// ---------------------------------------------------------------------
// The inventory menu (containerId 0) view
// ---------------------------------------------------------------------

/// Slot count of the player's own `InventoryMenu`: result,
/// 2x2 craft grid, 4 armor, 27 main, 9 hotbar, offhand.
pub const INVENTORY_MENU_SIZE: usize = 46;

/// Menu slot index -> backing container slot. `None` for the crafting view
/// (result + 2x2 grid): those slots have no container backing yet - the
/// extension point crafting plugs into.
pub fn menu_to_container(menu_slot: usize) -> Option<usize> {
    match menu_slot {
        // Result + craft grid 1..=4.
        0..=4 => None,
        // Armor: HEAD, CHEST, LEGS, FEET -> container 39, 38, 37, 36.
        5..=8 => Some(SLOT_HEAD - (menu_slot - 5)),
        // Main inventory, identity.
        9..=35 => Some(menu_slot),
        // Hotbar.
        36..=44 => Some(menu_slot - 36),
        // Offhand.
        45 => Some(SLOT_OFFHAND),
        _ => None,
    }
}

/// Container slot -> the menu slot presenting it (inverse of the above,
/// for the crafting-free slots).
pub fn container_to_menu(container_slot: usize) -> Option<usize> {
    match container_slot {
        0..=8 => Some(36 + container_slot),
        9..=35 => Some(container_slot),
        SLOT_FEET => Some(8),
        SLOT_LEGS => Some(7),
        SLOT_CHEST => Some(6),
        SLOT_HEAD => Some(5),
        SLOT_OFFHAND => Some(45),
        _ => None,
    }
}

/// Slot max stack: armor menu slots cap at 1 (`ArmorSlot`), everything
/// else inherits the container default 99; the effective cap is against
/// the stack's own max as well.
fn inventory_slot_max_stack(menu_slot: usize, stack: &ItemStack) -> i32 {
    let slot_cap = if (5..=8).contains(&menu_slot) {
        1
    } else {
        ABSOLUTE_MAX_STACK_SIZE
    };
    slot_cap.min(stack.max_stack_size())
}

/// `Slot.mayPlace` for the inventory menu: only container-backed slots
/// accept items. NOTE(inventory): equippability checks (`ArmorSlot`
/// `isEquippableInSlot`, curses) arrive with equipment data.
fn inventory_slot_may_place(menu_slot: usize) -> bool {
    menu_to_container(menu_slot).is_some()
}

/// Reads a menu slot's stack (empty for the crafting view).
fn menu_slot_get(inv: &PlayerInventory, menu_slot: usize) -> Option<ItemStack> {
    menu_to_container(menu_slot).and_then(|c| inv.get(c))
}

/// Writes a menu slot (no-op for the crafting view).
pub(crate) fn menu_slot_set(inv: &mut PlayerInventory, menu_slot: usize, stack: Option<ItemStack>) {
    if let Some(c) = menu_to_container(menu_slot) {
        inv.set(c, stack);
    }
}

// ---------------------------------------------------------------------
// Serverbound container_click
// ---------------------------------------------------------------------

/// `ContainerInput` wire ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClickKind {
    Pickup,
    QuickMove,
    Swap,
    Clone,
    Throw,
    QuickCraft,
    PickupAll,
}

impl ClickKind {
    pub fn from_id(id: i32) -> Option<ClickKind> {
        Some(match id {
            0 => ClickKind::Pickup,
            1 => ClickKind::QuickMove,
            2 => ClickKind::Swap,
            3 => ClickKind::Clone,
            4 => ClickKind::Throw,
            5 => ClickKind::QuickCraft,
            6 => ClickKind::PickupAll,
            _ => return None,
        })
    }
}

/// A parsed `ServerboundContainerClickPacket`. The `changed_slots` and
/// `carried` HashedStacks are the client's predicted post-click state;
/// Doppel validates their shape but does not apply them as predictions.
#[derive(Clone, Debug)]
pub struct ContainerClick {
    pub container_id: i32,
    pub state_id: i32,
    pub slot_num: i16,
    pub button_num: i8,
    pub kind: ClickKind,
    pub changed_slots: Vec<(i16, HashedStack)>,
    pub carried: HashedStack,
}

/// Parses a serverbound container_click body (after the packet id).
pub fn parse_container_click(body: &[u8]) -> Result<ContainerClick> {
    let mut r = Reader::new(body);
    let container_id = r.read_varint().context("container id")?;
    let state_id = r.read_varint().context("state id")?;
    let slot_num = read_i16(&mut r)?;
    let button_num = r.read_u8().context("button num")? as i8;
    let input = r.read_varint().context("container input")?;
    let kind = ClickKind::from_id(input).context("unknown container input id")?;
    let mut changed_slots = Vec::new();
    for _ in 0..read_count(&mut r, 128, "changed slots")? {
        let slot = read_i16(&mut r)?;
        let stack = HashedStack::decode(&mut r)?;
        changed_slots.push((slot, stack));
    }
    let carried = HashedStack::decode(&mut r)?;
    if r.remaining() != 0 {
        bail!("{} trailing bytes in container_click", r.remaining());
    }
    Ok(ContainerClick {
        container_id,
        state_id,
        slot_num,
        button_num,
        kind,
        changed_slots,
        carried,
    })
}

/// Parses a serverbound set_carried_item body: one i16 hotbar slot.
/// One set_creative_mode_slot push: the menu slot and the stack (None
/// slot means the client dropped the picked stack outside).
pub struct CreativeSlotSet {
    pub slot: i16,
    pub stack: Option<ItemStack>,
}

pub fn parse_set_creative_slot(body: &[u8]) -> Result<CreativeSlotSet> {
    let mut r = Reader::new(body);
    let slot = read_i16(&mut r)?;
    // The nullable stack codec: a presence boolean, then the stack.
    let present = r.read_u8().context("stack presence")?;
    let stack = if present != 0 {
        decode_item_stack(&mut r)?
    } else {
        None
    };
    if r.remaining() != 0 {
        bail!("trailing bytes in set_creative_mode_slot");
    }
    Ok(CreativeSlotSet { slot, stack })
}

pub fn parse_set_carried_item(body: &[u8]) -> Result<i16> {
    let mut r = Reader::new(body);
    let slot = read_i16(&mut r)?;
    if r.remaining() != 0 {
        bail!("trailing bytes in set_carried_item");
    }
    Ok(slot)
}

// ---------------------------------------------------------------------
// Click semantics (vanilla `AbstractContainerMenu.doClick`, inventory menu)
// ---------------------------------------------------------------------

/// Per-menu click session: the cursor stack, the quick-craft drag
/// machine, and the menu sync state id. One per open menu; the player's
/// inventory menu keeps its session inside `PlayerInvState`.
#[derive(Default)]
pub struct MenuSession {
    /// The stack held on the cursor.
    pub carried: Option<ItemStack>,
    quickcraft: QuickCraftState,
    state_id: i32,
}

impl MenuSession {
    /// The menu sync counter, wrapping at 0x7FFF (`incrementStateId`).
    pub(crate) fn next_state_id(&mut self) -> i32 {
        self.state_id = (self.state_id + 1) & 0x7fff;
        self.state_id
    }
}

/// Per-player inventory state: the inventory itself, the always-open
/// inventory menu's session (containerId 0), and the materials flag
/// gating CLONE / clone-drags.
#[derive(Default)]
pub struct PlayerInvState {
    pub inventory: PlayerInventory,
    /// The inventory menu's click session.
    pub session: MenuSession,
    /// `hasInfiniteMaterials` (creative). No gamemode system yet, so this
    /// starts false; the harness flips it via `set_creative_for_test`.
    pub creative: bool,
    /// Slots changed outside a click (a placement's spent stack, a drop)
    /// awaiting the per-tick menu broadcast.
    pub pending_sync: std::collections::BTreeSet<usize>,
}

/// The slot access a menu's click engine runs against: menu slot ids
/// mapped onto slot storage plus the menu-specific placement rules.
/// `PlayerInventory` implements the player side shared by every menu;
/// container menus (containers.rs) implement the container-grid side.
pub trait MenuSlots {
    /// Total menu slot count; valid menu ids are `0..menu_size()`.
    fn menu_size(&self) -> usize;
    fn menu_get(&self, slot: usize) -> Option<ItemStack>;
    fn menu_set(&mut self, slot: usize, stack: Option<ItemStack>);
    /// The effective stack cap for a placement into this slot.
    fn slot_max_stack(&self, slot: usize, stack: &ItemStack) -> i32;
    /// `Slot.mayPlace`.
    fn slot_may_place(&self, slot: usize) -> bool;
    /// `canTakeItemForPickAll`.
    fn may_pick_all(&self, slot: usize) -> bool;
    /// QUICK_MOVE destination range (exclusive end) and walk order.
    fn quick_move_bounds(&self, slot: usize) -> (usize, usize, bool);
    /// SWAP partner access in player-inventory container space (hotbar
    /// 0..9, offhand 40).
    fn swap_get(&self, container_slot: usize) -> Option<ItemStack>;
    fn swap_set(&mut self, container_slot: usize, stack: Option<ItemStack>);
    /// Best-effort insert into the player inventory (SWAP overflow).
    fn insert_into_inventory(&mut self, stack: ItemStack) -> Option<ItemStack>;
}

impl MenuSlots for PlayerInventory {
    fn menu_size(&self) -> usize {
        INVENTORY_MENU_SIZE
    }

    fn menu_get(&self, slot: usize) -> Option<ItemStack> {
        menu_slot_get(self, slot)
    }

    fn menu_set(&mut self, slot: usize, stack: Option<ItemStack>) {
        menu_slot_set(self, slot, stack)
    }

    fn slot_max_stack(&self, slot: usize, stack: &ItemStack) -> i32 {
        inventory_slot_max_stack(slot, stack)
    }

    fn slot_may_place(&self, slot: usize) -> bool {
        inventory_slot_may_place(slot)
    }

    fn may_pick_all(&self, slot: usize) -> bool {
        slot != 0
    }

    /// QUICK_MOVE bounds for the inventory menu: result/craft/armor/
    /// offhand sources move over menu 9..45 (main + hotbar, the offhand
    /// excluded as a destination), main and hotbar swap.
    fn quick_move_bounds(&self, slot: usize) -> (usize, usize, bool) {
        match slot {
            // Result slot: into the inventory, from the end.
            0 => (9, 45, true),
            // Craft grid + armor: into the inventory.
            1..=8 => (9, 45, false),
            // Main <-> hotbar.
            9..=35 => (36, 45, false),
            36..=44 => (9, 36, false),
            // Offhand: into the inventory.
            _ => (9, 45, false),
        }
    }

    fn swap_get(&self, container_slot: usize) -> Option<ItemStack> {
        self.get(container_slot)
    }

    fn swap_set(&mut self, container_slot: usize, stack: Option<ItemStack>) {
        self.set(container_slot, stack)
    }

    fn insert_into_inventory(&mut self, stack: ItemStack) -> Option<ItemStack> {
        self.add(stack)
    }
}

#[derive(Default, Debug)]
struct QuickCraftState {
    status: u8,
    kind: u8,
    slots: Vec<usize>,
}

impl QuickCraftState {
    fn reset(&mut self) {
        self.status = 0;
        self.slots.clear();
    }
}

/// Applies one click to an open menu. `doClick` semantics: special slot
/// -999 (outside) drops the carried stack; unknown container ids are
/// rejected by the caller before this runs.
pub fn apply_click<S: MenuSlots + ?Sized>(
    session: &mut MenuSession,
    slots: &mut S,
    creative: bool,
    click: &ContainerClick,
) {
    match click.kind {
        ClickKind::Pickup => {
            if is_click_button(click.button_num) {
                apply_pickup_or_drop(session, slots, click.slot_num, click.button_num == 0);
            }
        }
        ClickKind::QuickMove => {
            if is_click_button(click.button_num) {
                apply_quick_move(slots, click.slot_num);
            }
        }
        ClickKind::Swap => {
            let button = click.button_num as i32;
            if (0..9).contains(&button) || button == 40 {
                apply_swap(slots, click.slot_num, button);
            }
        }
        ClickKind::Clone => apply_clone(session, slots, creative, click.slot_num),
        ClickKind::Throw => apply_throw(session, slots, click.slot_num, click.button_num),
        ClickKind::QuickCraft => {
            apply_quick_craft(session, slots, creative, click.slot_num, click.button_num)
        }
        ClickKind::PickupAll => apply_pickup_all(session, slots, click.slot_num, click.button_num),
    }
}

fn is_click_button(button: i8) -> bool {
    button == 0 || button == 1
}

fn valid_menu_slot<S: MenuSlots + ?Sized>(slots: &S, slot_num: i16) -> Option<usize> {
    let slot = usize::try_from(slot_num).ok()?;
    (slot < slots.menu_size()).then_some(slot)
}

/// `canItemQuickReplace(slot, stack, ignoreSize=true)`: the slot is empty
/// or holds the same item+components.
fn can_quick_replace(slot_stack: Option<&ItemStack>, stack: &ItemStack) -> bool {
    match slot_stack {
        None => true,
        Some(occupant) => ItemStack::same_item_same_components(occupant, stack),
    }
}

/// PICKUP (and the -999 outside drop): primary/secondary button semantics
/// per `doClick`.
fn apply_pickup_or_drop<S: MenuSlots + ?Sized>(
    session: &mut MenuSession,
    slots: &mut S,
    slot_num: i16,
    primary: bool,
) {
    if slot_num == -999 {
        // Outside drop: primary drops the whole carried stack, secondary
        // one item. No item entities yet - the drop leaves the inventory
        // (NOTE(inventory): entity sync is the follow-up).
        if let Some(mut carried) = session.carried.take() {
            if !primary {
                carried.shrink(1);
                if !carried.is_empty() {
                    session.carried = Some(carried);
                }
            }
        }
        return;
    }
    let Some(menu_slot) = valid_menu_slot(slots, slot_num) else {
        return;
    };
    let clicked = slots.menu_get(menu_slot);
    match (clicked, session.carried.clone()) {
        (None, None) => {}
        (None, Some(carried)) => {
            let amount = if primary { carried.count() } else { 1 };
            session.carried = safe_insert(slots, menu_slot, carried, amount);
        }
        (Some(clicked), None) => {
            let amount = if primary {
                clicked.count()
            } else {
                (clicked.count() + 1) / 2
            };
            let taken = take_from_slot(slots, menu_slot, amount);
            session.carried = taken.filter(|s| !s.is_empty());
        }
        (Some(clicked), Some(carried)) => {
            if ItemStack::same_item_same_components(&clicked, &carried) {
                let amount = if primary { carried.count() } else { 1 };
                session.carried = safe_insert(slots, menu_slot, carried, amount);
            } else if carried.count() <= slots.slot_max_stack(menu_slot, &carried) {
                slots.menu_set(menu_slot, Some(carried));
                session.carried = Some(clicked);
            }
        }
    }
}

/// `Slot.safeInsert`: moves up to `amount` of `input` into the slot,
/// returning the carried leftover.
fn safe_insert<S: MenuSlots + ?Sized>(
    slots: &mut S,
    menu_slot: usize,
    input: ItemStack,
    amount: i32,
) -> Option<ItemStack> {
    if input.is_empty() || !slots.slot_may_place(menu_slot) {
        return Some(input);
    }
    let mut input = input;
    let occupant = slots.menu_get(menu_slot);
    let occupant_count = occupant.as_ref().map_or(0, ItemStack::count);
    let transferable = amount
        .min(input.count())
        .min(slots.slot_max_stack(menu_slot, &input) - occupant_count);
    if transferable <= 0 {
        return Some(input);
    }
    match occupant {
        None => {
            let placed = input.split(transferable);
            slots.menu_set(menu_slot, Some(placed));
        }
        Some(occupant) => {
            input.shrink(transferable);
            let mut grown = occupant;
            grown.grow(transferable);
            slots.menu_set(menu_slot, Some(grown));
        }
    }
    (!input.is_empty()).then_some(input)
}

/// `Slot.tryRemove`/`removeItem`: takes up to `amount` out of the slot.
fn take_from_slot<S: MenuSlots + ?Sized>(
    slots: &mut S,
    menu_slot: usize,
    amount: i32,
) -> Option<ItemStack> {
    let occupant = slots.menu_get(menu_slot)?;
    let taken = occupant.with_count(amount.min(occupant.count()));
    let rest = occupant.count() - taken.count();
    slots.menu_set(menu_slot, (rest > 0).then(|| occupant.with_count(rest)));
    (!taken.is_empty()).then_some(taken)
}

/// QUICK_MOVE: the menu's destination bounds plus the doClick repeat loop.
/// NOTE(inventory): equipment routing (armor/offhand preference for
/// equippable items) needs the equipment registry; moves use the plain
/// ranges.
fn apply_quick_move<S: MenuSlots + ?Sized>(slots: &mut S, slot_num: i16) {
    let Some(menu_slot) = valid_menu_slot(slots, slot_num) else {
        return;
    };
    while let Some(current) = slots.menu_get(menu_slot) {
        let (start, end, backwards) = slots.quick_move_bounds(menu_slot);
        let moved = move_stack_to(slots, menu_slot, start, end, backwards);
        if !moved {
            break;
        }
        let still_same = slots
            .menu_get(menu_slot)
            .is_some_and(|after| ItemStack::same_item_same_components(&after, &current));
        if !still_same {
            break;
        }
    }
}

/// `moveItemStackTo`: merge the stack at `menu_slot` into same-item stacks
/// in the menu range first, then into empty placeable slots. `backwards`
/// walks the range from the end.
fn move_stack_to<S: MenuSlots + ?Sized>(
    slots: &mut S,
    menu_slot: usize,
    start: usize,
    end: usize,
    backwards: bool,
) -> bool {
    let Some(mut stack) = slots.menu_get(menu_slot) else {
        return false;
    };
    let mut changed = false;
    let order: Vec<usize> = if backwards {
        (start..end).rev().collect()
    } else {
        (start..end).collect()
    };
    if stack.max_stack_size() > 1 {
        for dest in &order {
            if stack.is_empty() {
                break;
            }
            let Some(occupant) = slots.menu_get(*dest) else {
                continue;
            };
            if !ItemStack::same_item_same_components(&occupant, &stack) {
                continue;
            }
            let max = slots.slot_max_stack(*dest, &stack);
            let total = occupant.count() + stack.count();
            if total <= max {
                stack.set_count(0);
                slots.menu_set(*dest, Some(occupant.with_count(total)));
                changed = true;
            } else if occupant.count() < max {
                let moved = max - occupant.count();
                stack.shrink(moved);
                slots.menu_set(*dest, Some(occupant.with_count(max)));
                changed = true;
            }
        }
    }
    if !stack.is_empty() {
        for dest in &order {
            if stack.is_empty() {
                break;
            }
            if slots.menu_get(*dest).is_some() || !slots.slot_may_place(*dest) {
                continue;
            }
            let placed = stack.split(stack.count().min(slots.slot_max_stack(*dest, &stack)));
            slots.menu_set(*dest, Some(placed));
            changed = true;
        }
    }
    slots.menu_set(menu_slot, (!stack.is_empty()).then_some(stack));
    changed
}

/// SWAP with hotbar slots (button 0..8) or the offhand (button 40).
fn apply_swap<S: MenuSlots + ?Sized>(slots: &mut S, slot_num: i16, button: i32) {
    let Some(menu_slot) = valid_menu_slot(slots, slot_num) else {
        return;
    };
    let swap_slot = button as usize;
    let source = slots.swap_get(swap_slot);
    let target = slots.menu_get(menu_slot);
    match (source, target) {
        (None, None) => {}
        (None, Some(target)) => {
            // mayPickup is unconditional for these menus.
            slots.swap_set(swap_slot, Some(target));
            slots.menu_set(menu_slot, None);
        }
        (Some(source), None) => {
            if !slots.slot_may_place(menu_slot) {
                return;
            }
            let max = slots.slot_max_stack(menu_slot, &source);
            if source.count() > max {
                let head = source.with_count(max);
                let rest = source.with_count(source.count() - max);
                slots.swap_set(swap_slot, Some(rest));
                slots.menu_set(menu_slot, Some(head));
            } else {
                slots.swap_set(swap_slot, None);
                slots.menu_set(menu_slot, Some(source));
            }
        }
        (Some(source), Some(target)) => {
            if !slots.slot_may_place(menu_slot) {
                return;
            }
            let max = slots.slot_max_stack(menu_slot, &source);
            if source.count() > max {
                let head = source.with_count(max);
                let rest = source.with_count(source.count() - max);
                slots.swap_set(swap_slot, Some(rest));
                slots.menu_set(menu_slot, Some(head));
                // The displaced target stack goes back into the inventory;
                // overflow would drop (no entities yet, so it is discarded).
                let _ = slots.insert_into_inventory(target);
            } else {
                slots.swap_set(swap_slot, Some(target));
                slots.menu_set(menu_slot, Some(source));
            }
        }
    }
}

/// CLONE: creative-only copy of the clicked stack at its own max size.
fn apply_clone<S: MenuSlots + ?Sized>(
    session: &mut MenuSession,
    slots: &S,
    creative: bool,
    slot_num: i16,
) {
    if !creative || session.carried.is_some() {
        return;
    }
    let Some(menu_slot) = valid_menu_slot(slots, slot_num) else {
        return;
    };
    let Some(clicked) = slots.menu_get(menu_slot) else {
        return;
    };
    session.carried = Some(clicked.with_count(clicked.max_stack_size()));
}

/// THROW: button 0 throws one item, button 1 the whole stack (with the
/// ctrl-loop draining same-item follow-ups). `canDropItems` is true (no
/// spectator state yet).
fn apply_throw<S: MenuSlots + ?Sized>(
    session: &MenuSession,
    slots: &mut S,
    slot_num: i16,
    button: i8,
) {
    if session.carried.is_some() || slot_num < 0 {
        return;
    }
    let Some(menu_slot) = valid_menu_slot(slots, slot_num) else {
        return;
    };
    let Some(current) = slots.menu_get(menu_slot) else {
        return;
    };
    let amount = if button == 0 { 1 } else { current.count() };
    let thrown = take_from_slot(slots, menu_slot, amount);
    // No item entities yet: the thrown stack leaves the inventory.
    let _ = thrown;
    if button == 1 {
        while let Some(next) = slots.menu_get(menu_slot) {
            if !ItemStack::same_item_same_components(&next, &current) || next.is_empty() {
                break;
            }
            let _ = take_from_slot(slots, menu_slot, next.count());
        }
    }
}

/// PICKUP_ALL: the double-click gather. Two passes over the menu (forward
/// for button 0, backward for 1); the first pass skips full stacks. The
/// result slot is excluded (`canTakeItemForPickAll`).
fn apply_pickup_all<S: MenuSlots + ?Sized>(
    session: &mut MenuSession,
    slots: &mut S,
    slot_num: i16,
    button: i8,
) {
    // Vanilla gates on the clicked slot being empty or takeable;
    // mayPickup always holds for these menus, so only the range check
    // remains. Out-of-range clicks are a no-op here (vanilla would throw
    // on slots.get).
    if valid_menu_slot(slots, slot_num).is_none() {
        return;
    }
    let size = slots.menu_size();
    let Some(mut carried) = session.carried.clone() else {
        return;
    };
    let step: i32 = if button == 0 { 1 } else { -1 };
    let start: i32 = if button == 0 { 0 } else { size as i32 - 1 };
    for pass in 0..2 {
        let mut i = start;
        while (0..size as i32).contains(&i) && carried.count() < carried.max_stack_size() {
            let menu = i as usize;
            if slots.may_pick_all(menu) {
                if let Some(occupant) = slots.menu_get(menu) {
                    if can_quick_replace(Some(&occupant), &carried)
                        && !(pass == 0 && occupant.count() == occupant.max_stack_size())
                    {
                        let room = carried.max_stack_size() - carried.count();
                        if let Some(taken) = take_from_slot(slots, menu, occupant.count().min(room))
                        {
                            carried.grow(taken.count());
                        }
                    }
                }
            }
            i += step;
        }
    }
    session.carried = (!carried.is_empty()).then_some(carried);
}

/// QUICK_CRAFT (drag): the header/type machine from `doClick`. The button
/// byte packs header (bits 0-1: 0 start, 1 continue, 2 end) and type
/// (bits 2-3: 0 charitable, 1 greedy, 2 clone).
fn apply_quick_craft<S: MenuSlots + ?Sized>(
    session: &mut MenuSession,
    slots: &mut S,
    creative: bool,
    slot_num: i16,
    button: i8,
) {
    let header = (button & 3) as u8;
    let kind = ((button >> 2) & 3) as u8;
    let expected = session.quickcraft.status;
    session.quickcraft.status = header;
    // Only (continue->end) and same-header clicks extend a drag; anything
    // else resets it.
    if !((expected == 1 && header == 2) || expected == header) {
        session.quickcraft.reset();
        return;
    }
    if session.carried.is_none() {
        session.quickcraft.reset();
        return;
    }
    match header {
        0 => {
            // START: arm the drag with this click's type.
            session.quickcraft.kind = kind;
            if is_valid_quickcraft_type(kind, creative) {
                session.quickcraft.status = 1;
                session.quickcraft.slots.clear();
            } else {
                session.quickcraft.reset();
            }
        }
        1 => {
            // CONTINUE: add one slot to the drag set.
            let Some(menu_slot) = valid_menu_slot(slots, slot_num) else {
                return;
            };
            let Some(carried) = session.carried.clone() else {
                return;
            };
            let occupant = slots.menu_get(menu_slot);
            if !can_quick_replace(occupant.as_ref(), &carried)
                || !slots.slot_may_place(menu_slot)
                || (session.quickcraft.kind != 2
                    && carried.count() <= session.quickcraft.slots.len() as i32)
                || session.quickcraft.slots.contains(&menu_slot)
            {
                return;
            }
            session.quickcraft.slots.push(menu_slot);
        }
        2 => {
            // END: distribute (a single-slot drag degenerates to a PICKUP
            // with the drag type as the button).
            let drag_slots = session.quickcraft.slots.clone();
            let kind = session.quickcraft.kind;
            session.quickcraft.reset();
            if drag_slots.len() == 1 {
                apply_pickup_or_drop(session, slots, drag_slots[0] as i16, kind == 0);
                return;
            }
            if drag_slots.is_empty() {
                return;
            }
            let Some(source) = session.carried.clone() else {
                return;
            };
            let per = quickcraft_place_count(drag_slots.len() as i32, kind, &source);
            let mut remaining = source.count();
            for menu_slot in drag_slots.iter().copied() {
                let Some(carried) = session.carried.clone() else {
                    break;
                };
                let occupant = slots.menu_get(menu_slot);
                if !can_quick_replace(occupant.as_ref(), &carried)
                    || !slots.slot_may_place(menu_slot)
                    || (kind != 2 && carried.count() < drag_slots.len() as i32)
                {
                    continue;
                }
                let carry = occupant.as_ref().map_or(0, ItemStack::count);
                let max = slots.slot_max_stack(menu_slot, &source);
                let new_count = (per + carry).min(max);
                remaining -= new_count - carry;
                slots.menu_set(menu_slot, Some(source.with_count(new_count)));
            }
            session.carried = (remaining > 0).then(|| source.with_count(remaining));
        }
        _ => session.quickcraft.reset(),
    }
}

fn is_valid_quickcraft_type(kind: u8, creative: bool) -> bool {
    matches!(kind, 0 | 1) || (kind == 2 && creative)
}

/// `getQuickCraftPlaceCount`: charitable = floor(count / slots), greedy =
/// one per slot, clone = the full max stack.
fn quickcraft_place_count(slots: i32, kind: u8, source: &ItemStack) -> i32 {
    match kind {
        0 => source.count() / slots,
        1 => 1,
        2 => source.max_stack_size(),
        _ => source.count(),
    }
}

// ---------------------------------------------------------------------
// Clientbound packet bodies
// ---------------------------------------------------------------------

/// One decoded `container_set_content`: (containerId, stateId, slots,
/// carried).
pub type SetContent = (i32, i32, Vec<Option<ItemStack>>, Option<ItemStack>);

/// `container_set_content`: containerId, stateId, the slot list, carried.
pub fn encode_container_set_content(
    container_id: i32,
    state_id: i32,
    slots: &[Option<ItemStack>],
    carried: Option<&ItemStack>,
) -> Vec<u8> {
    let mut body = Vec::with_capacity(8 + slots.len() * 4);
    write_varint(&mut body, container_id);
    write_varint(&mut body, state_id);
    write_varint(&mut body, slots.len() as i32);
    for slot in slots {
        encode_item_stack(&mut body, slot.as_ref());
    }
    encode_item_stack(&mut body, carried);
    body
}

/// Decodes a container_set_content body (harness-side checks).
pub fn decode_container_set_content(body: &[u8]) -> Result<SetContent> {
    let mut r = Reader::new(body);
    let container_id = r.read_varint().context("container id")?;
    let state_id = r.read_varint().context("state id")?;
    let count = read_count(&mut r, 1024, "content slots")?;
    let mut slots = Vec::with_capacity(count);
    for _ in 0..count {
        slots.push(decode_item_stack(&mut r)?);
    }
    let carried = decode_item_stack(&mut r)?;
    Ok((container_id, state_id, slots, carried))
}

/// `container_set_slot`: containerId, stateId, slot i16, stack.
pub fn encode_container_set_slot(
    container_id: i32,
    state_id: i32,
    slot: i16,
    stack: Option<&ItemStack>,
) -> Vec<u8> {
    let mut body = Vec::with_capacity(12);
    write_varint(&mut body, container_id);
    write_varint(&mut body, state_id);
    body.extend_from_slice(&slot.to_be_bytes());
    encode_item_stack(&mut body, stack);
    body
}

/// `set_cursor_item`: just the optional carried stack (no containerId).
pub fn encode_set_cursor_item(stack: Option<&ItemStack>) -> Vec<u8> {
    let mut body = Vec::with_capacity(4);
    encode_item_stack(&mut body, stack);
    body
}

/// `set_held_slot`: the selected hotbar slot as a VarInt.
pub fn encode_set_held_slot(slot: u8) -> Vec<u8> {
    let mut body = Vec::with_capacity(2);
    write_varint(&mut body, slot as i32);
    body
}

/// `set_player_inventory`: a raw container slot (0..42) plus the stack.
pub fn encode_set_player_inventory(slot: usize, stack: Option<&ItemStack>) -> Vec<u8> {
    let mut body = Vec::with_capacity(6);
    write_varint(&mut body, slot as i32);
    encode_item_stack(&mut body, stack);
    body
}

/// Reads a big-endian i16 (fixed 2 bytes).
fn read_i16(r: &mut Reader) -> Result<i16> {
    Ok(r.read_u16().context("i16")? as i16)
}

/// Reads a big-endian i32 (fixed 4 bytes).
fn read_i32be(r: &mut Reader) -> Result<i32> {
    let bytes = r.read_bytes(4).context("i32")?;
    Ok(i32::from_be_bytes(bytes.try_into().expect("4 bytes")))
}

// ---------------------------------------------------------------------
// Game-thread hooks
// ---------------------------------------------------------------------

impl Game {
    /// `/give <target> <item> [count]` (harness driver): resolves the item,
    /// adds the stack, and broadcasts the player's inventory menu.
    /// NOTE(inventory): ungated by gamemode - the creative check arrives
    /// with the abilities system.
    pub(crate) fn give_item(&mut self, conn: ConnId, item: &str, count: i32) {
        let Some(item) = item_id(item) else {
            eprintln!("[game] give: unknown item {item}");
            return;
        };
        let Some(p) = self.players.get_mut(&conn) else {
            return;
        };
        let stack = ItemStack::new(item, count.clamp(1, 6400));
        if let Some(leftover) = p.inv.inventory.add(stack) {
            eprintln!(
                "[game] give: inventory full, discarding {}x{}",
                leftover.count(),
                leftover.item()
            );
        }
        self.broadcast_inventory(conn);
    }

    /// Serverbound set_carried_item: select a hotbar slot. No echo packet
    /// is sent to the selecting client.
    pub(crate) fn select_hotbar_slot(&mut self, conn: ConnId, slot: i16) {
        let Some(p) = self.players.get_mut(&conn) else {
            return;
        };
        if (0..9).contains(&slot) {
            p.inv.inventory.set_selected(slot as u8);
        }
    }

    /// Serverbound container_click against the always-open inventory menu.
    ///
    /// NOTE(inventory): the client's HashedStack predictions are decoded
    /// for shape validation but not applied; the response is one
    /// authoritative container_set_content of the affected container after
    /// applying the click server-side (always a correct server response  -
    /// full CRC32C hash prediction is future work and is not faked here).
    pub(crate) fn container_clicked(&mut self, conn: ConnId, click: &ContainerClick) {
        // Vanilla ignores clicks against a menu the player does not have
        // open; container menus route through containers.rs.
        if click.container_id != 0 {
            // --- containers hooks (containers.rs) ---
            self.container_menu_clicked(conn, click);
            return;
        }
        let Some(p) = self.players.get_mut(&conn) else {
            return;
        };
        let creative = p.inv.creative;
        apply_click(&mut p.inv.session, &mut p.inv.inventory, creative, click);
        self.broadcast_inventory(conn);
    }

    /// Full resync of the player's inventory menu: one
    /// container_set_content with every menu slot plus the carried stack.
    fn broadcast_inventory(&mut self, conn: ConnId) {
        let body = {
            let Some(p) = self.players.get_mut(&conn) else {
                return;
            };
            let state_id = p.inv.session.next_state_id();
            let mut slots = Vec::with_capacity(INVENTORY_MENU_SIZE);
            for menu_slot in 0..INVENTORY_MENU_SIZE {
                slots.push(menu_slot_get(&p.inv.inventory, menu_slot));
            }
            let carried = p.inv.session.carried.clone();
            encode_container_set_content(0, state_id, &slots, carried.as_ref())
        };
        self.send(conn, PACKET_CONTAINER_SET_CONTENT, &body);
        // The carried stack rides inside set_content (the client applies
        // it via initializeContents); set_cursor_item exists for the
        // targeted cursor-only syncs vanilla sends on click prediction.
    }

    /// The per-tick menu broadcast: each slot changed outside a click
    /// syncs as its own set_slot, and only while the inventory menu is
    /// the open menu (a container menu open at broadcast time masks the
    /// change; its open snapshot already carries it).
    pub(crate) fn broadcast_pending_inventory(&mut self) {
        let conns: Vec<ConnId> = self.players.keys().copied().collect();
        for conn in conns {
            let frames = {
                let Some(p) = self.players.get_mut(&conn) else {
                    continue;
                };
                if p.menu.is_some() {
                    p.inv.pending_sync.clear();
                    continue;
                }
                let slots = std::mem::take(&mut p.inv.pending_sync);
                if slots.is_empty() {
                    continue;
                }
                slots
                    .into_iter()
                    .filter_map(|slot| {
                        let state_id = p.inv.session.next_state_id();
                        let menu = crate::inventory::container_to_menu(slot)?;
                        Some(crate::inventory::encode_container_set_slot(
                            0,
                            state_id,
                            menu as i16,
                            p.inv.inventory.get(slot).as_ref(),
                        ))
                    })
                    .collect::<Vec<_>>()
            };
            for body in frames {
                if !body.is_empty() {
                    self.send(conn, PACKET_CONTAINER_SET_SLOT, &body);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
impl Game {
    /// The selected hotbar slot (test accessor).
    pub(crate) fn selected_slot_for_test(&self, conn: ConnId) -> u8 {
        self.players
            .get(&conn)
            .map_or(0, |p| p.inv.inventory.selected())
    }

    /// Flips the infinite-materials flag (`hasInfiniteMaterials`).
    #[cfg(test)]
    pub(crate) fn player_inv_state_for_test(
        &self,
        conn: ConnId,
    ) -> Option<&crate::inventory::PlayerInvState> {
        self.players.get(&conn).map(|p| &p.inv)
    }

    pub(crate) fn set_creative_for_test(&mut self, conn: ConnId, creative: bool) {
        if let Some(p) = self.players.get_mut(&conn) {
            p.inv.creative = creative;
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::game::{Game, Inbound, Outbound};

    // -- codec golden bytes ------------------------------------------------

    #[test]
    fn empty_slot_is_a_single_zero() {
        let mut buf = Vec::new();
        encode_item_stack(&mut buf, None);
        assert_eq!(buf, vec![0x00]);
        // A zero count encodes identically (empty convention).
        let mut buf = Vec::new();
        encode_item_stack(&mut buf, Some(&ItemStack::new(1, 0)));
        assert_eq!(buf, vec![0x00]);
        assert_eq!(decode_item_stack(&mut Reader::new(&[0x00])).unwrap(), None);
    }

    #[test]
    fn plain_stack_golden() {
        // 64x stone (item id 1): count, item, empty patch (0, 0).
        let mut buf = Vec::new();
        encode_item_stack(&mut buf, Some(&ItemStack::new(1, 64)));
        assert_eq!(buf, vec![0x40, 0x01, 0x00, 0x00]);
        let stack = decode_item_stack(&mut Reader::new(&buf))
            .unwrap()
            .expect("non-empty");
        assert_eq!((stack.count(), stack.item()), (64, 1));
        assert_eq!(stack.max_stack_size(), 64);
    }

    #[test]
    fn patched_stack_golden_and_roundtrip() {
        // 1x stone with max_stack_size=16: count 1, item 1, patch with one
        // added (type 1, VarInt 16) and no removals.
        let patch = ComponentPatch {
            added: vec![(component::MAX_STACK_SIZE, ComponentValue::VarInt(16))],
            removed: vec![],
        };
        let stack = ItemStack::new(1, 1).with_patch(patch);
        let mut buf = Vec::new();
        encode_item_stack(&mut buf, Some(&stack));
        assert_eq!(buf, vec![0x01, 0x01, 0x01, 0x01, 0x10, 0x00]);
        let back = decode_item_stack(&mut Reader::new(&buf))
            .unwrap()
            .expect("non-empty");
        assert_eq!(back, stack);
        assert_eq!(back.max_stack_size(), 16);
    }

    #[test]
    fn unit_component_roundtrip() {
        // 1x diamond_sword (1050, VarInt 9a 08) with unbreakable (type 4,
        // Unit: zero payload bytes) added and rarity (type 12) removed:
        // count, item, 1 added {type}, 1 removed {type}.
        let patch = ComponentPatch {
            added: vec![(component::UNBREAKABLE, ComponentValue::Unit)],
            removed: vec![component::RARITY],
        };
        let stack = ItemStack::new(1050, 1).with_patch(patch);
        let mut buf = Vec::new();
        encode_item_stack(&mut buf, Some(&stack));
        assert_eq!(buf, vec![0x01, 0x9a, 0x08, 0x01, 0x04, 0x01, 0x0c]);
        let back = decode_item_stack(&mut Reader::new(&buf))
            .unwrap()
            .expect("non-empty");
        assert_eq!(back, stack);
    }

    #[test]
    fn unknown_component_decode_rejected() {
        // Patch with an added component of unknown type 99: count 1, item
        // 1, then type 99.
        let bytes = vec![0x01, 0x01, 0x01, 0x63];
        assert!(decode_item_stack(&mut Reader::new(&bytes)).is_err());
    }

    /// The join capture's inventory packet, verbatim: containerId 0,
    /// stateId 1, 46 empty slots, empty carried. This is the golden shape
    /// of the whole broadcast path.
    #[test]
    fn join_set_content_golden() {
        let mut captured = vec![0x00, 0x01, 0x2e];
        captured.extend(std::iter::repeat_n(0x00, 47));
        let (container_id, state_id, slots, carried) =
            decode_container_set_content(&captured).unwrap();
        assert_eq!(container_id, 0);
        assert_eq!(state_id, 1);
        assert_eq!(slots.len(), 46);
        assert!(slots.iter().all(Option::is_none));
        assert_eq!(carried, None);
        // Our encoder produces byte-identical output for the empty state.
        let empty = vec![None; 46];
        assert_eq!(encode_container_set_content(0, 1, &empty, None), captured);
    }

    #[test]
    fn hashed_stack_golden_and_roundtrip() {
        // Empty hashed stack: just the false boolean.
        let mut buf = Vec::new();
        HashedStack::default().encode(&mut buf);
        assert_eq!(buf, vec![0x00]);
        // Present: flag, item, count, one added (type 1, hash 0xde_ad_be_ef
        // big-endian), one removed (type 4).
        let hashed = HashedStack {
            present: true,
            item: 1,
            count: 2,
            added: vec![(1, -559_038_737)], // 0xdeadbeef as i32
            removed: vec![4],
        };
        let mut buf = Vec::new();
        hashed.encode(&mut buf);
        assert_eq!(
            buf,
            vec![0x01, 0x01, 0x02, 0x01, 0x01, 0xde, 0xad, 0xbe, 0xef, 0x01, 0x04]
        );
        assert_eq!(HashedStack::decode(&mut Reader::new(&buf)).unwrap(), hashed);
    }

    #[test]
    fn container_click_parse() {
        // containerId 0, stateId 3, slotNum 36 (0x0024), button 0, input 0
        // (pickup), no changed slots, empty carried hash.
        let bytes = vec![0x00, 0x03, 0x00, 0x24, 0x00, 0x00, 0x00, 0x00];
        let click = parse_container_click(&bytes).unwrap();
        assert_eq!(click.slot_num, 36);
        assert_eq!(click.kind, ClickKind::Pickup);
        assert_eq!(click.state_id, 3);
        assert!(click.changed_slots.is_empty());
        assert!(!click.carried.present);
        // A changed-slot entry: slot 36 with a present hashed stack.
        let mut bytes = vec![0x00, 0x03, 0x00, 0x24, 0x00, 0x01, 0x01, 0x00, 0x24];
        bytes.extend_from_slice(&[0x01, 0x01, 0x05, 0x00, 0x00]);
        bytes.push(0x00);
        let click = parse_container_click(&bytes).unwrap();
        assert_eq!(click.changed_slots.len(), 1);
        assert_eq!(click.changed_slots[0].0, 36);
        assert!(click.changed_slots[0].1.present);
    }

    #[test]
    fn set_carried_item_parse() {
        assert_eq!(parse_set_carried_item(&[0x00, 0x03]).unwrap(), 3);
        assert!(parse_set_carried_item(&[0x00, 0x03, 0x00]).is_err());
    }

    #[test]
    fn clientbound_writer_goldens() {
        // set_slot: containerId 0, stateId 2, slot 36, 5x stone.
        assert_eq!(
            encode_container_set_slot(0, 2, 36, Some(&ItemStack::new(1, 5))),
            vec![0x00, 0x02, 0x00, 0x24, 0x05, 0x01, 0x00, 0x00]
        );
        // set_cursor_item: 7x stone.
        assert_eq!(
            encode_set_cursor_item(Some(&ItemStack::new(1, 7))),
            vec![0x07, 0x01, 0x00, 0x00]
        );
        // set_held_slot: slot 0 - byte-identical to the capture's `6b 00`.
        assert_eq!(encode_set_held_slot(0), vec![0x00]);
        // set_player_inventory: raw container slot 40, empty stack.
        assert_eq!(encode_set_player_inventory(40, None), vec![0x28, 0x00]);
    }

    // -- static helpers ----------------------------------------------------

    #[test]
    fn menu_slot_mapping() {
        assert_eq!(menu_to_container(0), None);
        assert_eq!(menu_to_container(4), None);
        assert_eq!(menu_to_container(5), Some(SLOT_HEAD));
        assert_eq!(menu_to_container(8), Some(SLOT_FEET));
        assert_eq!(menu_to_container(9), Some(9));
        assert_eq!(menu_to_container(35), Some(35));
        assert_eq!(menu_to_container(36), Some(0));
        assert_eq!(menu_to_container(44), Some(8));
        assert_eq!(menu_to_container(45), Some(SLOT_OFFHAND));
        for container in 0..=SLOT_OFFHAND {
            let menu = container_to_menu(container).expect("mapped");
            assert_eq!(menu_to_container(menu), Some(container));
        }
        assert_eq!(container_to_menu(SLOT_BODY_ARMOR), None);
        assert_eq!(container_to_menu(SLOT_SADDLE), None);
    }

    #[test]
    fn inventory_add_fills_selected_then_empties() {
        let mut inv = PlayerInventory::default();
        inv.set_selected(3);
        // Empty inventory: no partial stacks, so the first free slot (0)
        // wins - the free-slot scan runs the 36-item list in order.
        let leftover = inv.add(ItemStack::new(1, 5));
        assert!(leftover.is_none());
        assert_eq!(inv.get(0).map(|s| s.count()), Some(5));
        // Merging prefers the selected hotbar slot.
        inv.set(3, Some(ItemStack::new(1, 7)));
        let leftover = inv.add(ItemStack::new(1, 10));
        assert!(leftover.is_none());
        assert_eq!(inv.get(3).map(|s| s.count()), Some(17));
        // Overflow continues through the merge order (selected, offhand,
        // then 0..35): slot 3 caps at 64, the rest joins slot 0's stack.
        let leftover = inv.add(ItemStack::new(1, 60));
        assert!(leftover.is_none());
        assert_eq!(inv.get(3).map(|s| s.count()), Some(64));
        assert_eq!(inv.get(0).map(|s| s.count()), Some(18));
    }

    // -- game-thread integration -------------------------------------------

    /// A game with one viewer registered for outbound frames (no chunks or
    /// registry needed - inventory state is chunk-independent).
    fn harness() -> (Game, std::sync::mpsc::Receiver<Outbound>) {
        let (_tx, rx) = std::sync::mpsc::channel::<Inbound>();
        let mut g = Game::new(rx, None, None);
        let (tx_out, rx_out) = std::sync::mpsc::channel::<Outbound>();
        g.join_viewer_for_test(0, &[], tx_out);
        (g, rx_out)
    }

    fn give(g: &mut Game, item: &str, count: i32) {
        g.handle(Inbound::Give {
            conn: 0,
            item: item.to_string(),
            count,
        });
    }

    fn click(g: &mut Game, click: ContainerClick) {
        g.handle(Inbound::ContainerClick { conn: 0, click });
    }

    /// The next set_content frame, decoded to (containerId, stateId,
    /// slots, carried) - container 0, stateId 1 on the first broadcast.
    fn next_content(
        rx: &std::sync::mpsc::Receiver<Outbound>,
    ) -> (i32, i32, Vec<Option<ItemStack>>, Option<ItemStack>) {
        while let Ok(frame) = rx.try_recv() {
            if let Outbound::Frame { id, body } = frame {
                if id == PACKET_CONTAINER_SET_CONTENT {
                    return decode_container_set_content(&body).expect("valid set_content");
                }
            }
        }
        panic!("no set_content frame queued");
    }

    fn click_raw(
        container_id: i32,
        state_id: i32,
        slot: i16,
        button: i8,
        kind: ClickKind,
    ) -> ContainerClick {
        ContainerClick {
            container_id,
            state_id,
            slot_num: slot,
            button_num: button,
            kind,
            changed_slots: Vec::new(),
            carried: HashedStack::default(),
        }
    }

    /// Drains every queued set_content frame and returns the last (each
    /// click broadcasts one frame; drag clicks are several in a row).
    fn last_content(
        rx: &std::sync::mpsc::Receiver<Outbound>,
    ) -> (i32, i32, Vec<Option<ItemStack>>, Option<ItemStack>) {
        let mut last = None;
        while let Ok(frame) = rx.try_recv() {
            if let Outbound::Frame { id, body } = frame {
                if id == PACKET_CONTAINER_SET_CONTENT {
                    last = Some(decode_container_set_content(&body).expect("valid set_content"));
                }
            }
        }
        last.expect("no set_content frame queued")
    }

    #[test]
    fn gamemode_gates_creative_slot() {
        let (mut g, rx) = harness();
        g.handle(Inbound::GameMode {
            conn: 0,
            creative: false,
        });
        while rx.try_recv().is_ok() {}
        let survival_set = parse_set_creative_slot(&[0, 36, 1, 1, 1, 0, 0]).unwrap();
        g.handle(Inbound::CreativeSlot {
            conn: 0,
            set: survival_set,
        });
        let got = g
            .player_inv_state_for_test(0)
            .expect("player")
            .inventory
            .get(0);
        assert!(got.is_none(), "survival ignores creative pushes");

        g.handle(Inbound::GameMode {
            conn: 0,
            creative: true,
        });
        let creative_set = parse_set_creative_slot(&[0, 36, 1, 1, 1, 0, 0]).unwrap();
        g.handle(Inbound::CreativeSlot {
            conn: 0,
            set: creative_set,
        });
        let got = g
            .player_inv_state_for_test(0)
            .expect("player")
            .inventory
            .get(0)
            .expect("creative push lands");
        assert_eq!((got.count(), got.item()), (1, 1));
    }

    #[test]
    fn give_broadcasts_set_content() {
        let (mut g, rx) = harness();
        give(&mut g, "minecraft:stone", 5);
        let (container_id, state_id, slots, carried) = next_content(&rx);
        assert_eq!((container_id, state_id), (0, 1));
        // The stack lands in hotbar container slot 0 = menu slot 36.
        assert_eq!(slots[36].as_ref().map(ItemStack::count), Some(5));
        assert_eq!(
            slots[36].as_ref().map(ItemStack::item),
            Some(item_id("minecraft:stone").unwrap())
        );
        assert!(slots
            .iter()
            .enumerate()
            .all(|(i, s)| i == 36 || s.is_none()));
        assert_eq!(carried, None);
    }

    #[test]
    fn give_resolves_default_namespace_and_rejects_unknown() {
        let (mut g, rx) = harness();
        give(&mut g, "dirt", 1);
        let (_, _, slots, _) = next_content(&rx);
        assert_eq!(
            slots[36].as_ref().map(ItemStack::item),
            Some(item_id("minecraft:dirt").unwrap())
        );
        // Drain the command's feedback reply before the next assertion.
        while matches!(rx.try_recv(), Ok(Outbound::Frame { id: 0x7c, .. })) {}
        give(&mut g, "minecraft:not_an_item", 1);
        // The command still draws its feedback reply (the reference
        // answers failed commands too); nothing else may follow.
        match rx.try_recv() {
            Ok(Outbound::Frame { id: 0x7c, .. }) => {}
            _ => panic!("expected command feedback"),
        }
        assert!(rx.try_recv().is_err(), "no broadcast for unknown item");
    }

    #[test]
    fn pickup_click_lifts_stack_to_carried() {
        let (mut g, rx) = harness();
        give(&mut g, "stone", 5);
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 36, 0, ClickKind::Pickup));
        let (_, _, slots, carried) = next_content(&rx);
        assert!(slots[36].is_none(), "slot emptied by the pickup");
        let carried = carried.expect("carried the stack");
        assert_eq!((carried.count(), carried.item()), (5, 1));
        // Put it back on a different slot.
        click(&mut g, click_raw(0, 2, 37, 0, ClickKind::Pickup));
        let (_, _, slots, carried) = next_content(&rx);
        assert_eq!(slots[37].as_ref().map(ItemStack::count), Some(5));
        assert_eq!(carried, None);
    }

    #[test]
    fn pickup_secondary_takes_half() {
        let (mut g, rx) = harness();
        give(&mut g, "stone", 5);
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 36, 1, ClickKind::Pickup));
        let (_, _, slots, carried) = next_content(&rx);
        assert_eq!(slots[36].as_ref().map(ItemStack::count), Some(2));
        assert_eq!(carried.expect("half").count(), 3); // ceil(5/2)
    }

    #[test]
    fn pickup_swaps_different_items() {
        let (mut g, rx) = harness();
        give(&mut g, "stone", 5);
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 36, 0, ClickKind::Pickup)); // lift stone
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 37, 0, ClickKind::Pickup)); // place at c1
        let _ = next_content(&rx);
        // Dirt goes to the first empty container slot (0 = menu 36).
        give(&mut g, "dirt", 2);
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 36, 0, ClickKind::Pickup)); // dirt to cursor
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 37, 0, ClickKind::Pickup)); // swap onto stone
        let (_, _, slots, carried) = next_content(&rx);
        assert_eq!(
            slots[37].as_ref().map(ItemStack::item),
            item_id("minecraft:dirt")
        );
        assert_eq!(
            carried.expect("stone swapped out").item(),
            item_id("minecraft:stone").unwrap()
        );
        assert!(slots[36].is_none());
    }

    #[test]
    fn drop_outside_clears_carried() {
        let (mut g, rx) = harness();
        give(&mut g, "stone", 5);
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 36, 0, ClickKind::Pickup));
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 2, -999, 0, ClickKind::Pickup));
        let (_, _, slots, carried) = next_content(&rx);
        assert_eq!(carried, None, "primary outside drop discards all");
        assert!(slots.iter().all(Option::is_none));
    }

    #[test]
    fn swap_with_offhand_button_40() {
        let (mut g, rx) = harness();
        give(&mut g, "stone", 5);
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 36, 40, ClickKind::Swap));
        let (_, _, slots, _) = next_content(&rx);
        // Menu slot 36 is hotbar container 0; swap partner is container 40
        // (offhand), presented as menu slot 45.
        assert!(slots[36].is_none());
        assert_eq!(slots[45].as_ref().map(ItemStack::count), Some(5));
    }

    #[test]
    fn quick_move_hotbar_to_main() {
        let (mut g, rx) = harness();
        give(&mut g, "stone", 5);
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 36, 0, ClickKind::QuickMove));
        let (_, _, slots, _) = next_content(&rx);
        assert!(slots[36].is_none(), "left the hotbar");
        assert_eq!(
            slots[9].as_ref().map(ItemStack::count),
            Some(5),
            "landed in main 9"
        );
    }

    #[test]
    fn quick_move_merges_before_filling() {
        let (mut g, rx) = harness();
        // A partial stack at menu 10 (container 10) first.
        give(&mut g, "stone", 30);
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 36, 0, ClickKind::Pickup)); // lift stack
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 10, 0, ClickKind::Pickup)); // place at main 10
        let _ = next_content(&rx);
        // A second stack lands in hotbar 0 (menu 36), the first empty.
        give(&mut g, "stone", 40);
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 36, 0, ClickKind::QuickMove));
        let (_, _, slots, _) = next_content(&rx);
        // Merged into the partial stack first (menu 10 -> 64), the spill
        // (6) fills the first empty main slot (menu 9).
        assert_eq!(slots[10].as_ref().map(ItemStack::count), Some(64));
        assert_eq!(slots[9].as_ref().map(ItemStack::count), Some(6));
        assert!(slots[36].is_none());
    }

    #[test]
    fn throw_takes_one_or_all() {
        let (mut g, rx) = harness();
        give(&mut g, "stone", 5);
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 36, 0, ClickKind::Throw));
        let (_, _, slots, _) = next_content(&rx);
        assert_eq!(
            slots[36].as_ref().map(ItemStack::count),
            Some(4),
            "threw one"
        );
        click(&mut g, click_raw(0, 2, 36, 1, ClickKind::Throw));
        let (_, _, slots, _) = next_content(&rx);
        assert!(slots[36].is_none(), "threw the rest");
    }

    #[test]
    fn clone_requires_creative() {
        let (mut g, rx) = harness();
        give(&mut g, "stone", 5);
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 36, 0, ClickKind::Clone));
        let (_, _, _, carried) = next_content(&rx);
        assert_eq!(carried, None, "no clone without infinite materials");
        g.set_creative_for_test(0, true);
        click(&mut g, click_raw(0, 2, 36, 0, ClickKind::Clone));
        let (_, _, slots, carried) = next_content(&rx);
        assert_eq!(carried.expect("cloned").count(), 64, "clone at max stack");
        assert_eq!(
            slots[36].as_ref().map(ItemStack::count),
            Some(5),
            "source kept"
        );
    }

    #[test]
    fn drag_distributes_charitably() {
        let (mut g, rx) = harness();
        give(&mut g, "stone", 64);
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 36, 0, ClickKind::Pickup)); // carry all 64
        let _ = next_content(&rx);
        // The client's drag choreography: START where the button went down,
        // one CONTINUE per hovered slot INCLUDING that starting slot (the
        // first mouseDragged re-adds it), then END from -999. Drag over menu
        // slots 9/10/11: charitable = floor(64/3) = 21 per slot, 1 stays.
        click(&mut g, click_raw(0, 2, 9, 0, ClickKind::QuickCraft)); // start
        click(&mut g, click_raw(0, 2, 9, 1, ClickKind::QuickCraft)); // re-add start
        click(&mut g, click_raw(0, 2, 10, 1, ClickKind::QuickCraft));
        click(&mut g, click_raw(0, 2, 11, 1, ClickKind::QuickCraft));
        click(&mut g, click_raw(0, 2, -999, 2, ClickKind::QuickCraft)); // end
        let (_, _, slots, carried) = last_content(&rx);
        assert_eq!(slots[9].as_ref().map(ItemStack::count), Some(21));
        assert_eq!(slots[10].as_ref().map(ItemStack::count), Some(21));
        assert_eq!(slots[11].as_ref().map(ItemStack::count), Some(21));
        assert_eq!(carried.expect("remainder kept").count(), 1);
    }

    #[test]
    fn drag_single_slot_degenerates_to_pickup() {
        let (mut g, rx) = harness();
        give(&mut g, "stone", 64);
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 36, 0, ClickKind::Pickup));
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 2, 9, 0, ClickKind::QuickCraft)); // start
        click(&mut g, click_raw(0, 2, 9, 1, ClickKind::QuickCraft)); // re-add start
        click(&mut g, click_raw(0, 2, -999, 2, ClickKind::QuickCraft)); // end
        let (_, _, slots, carried) = last_content(&rx);
        // Charitable drag type 0 as the button: primary pickup places all.
        assert_eq!(slots[9].as_ref().map(ItemStack::count), Some(64));
        assert_eq!(carried, None);
    }

    #[test]
    fn pickup_all_gathers_same_item() {
        let (mut g, rx) = harness();
        give(&mut g, "stone", 64); // hotbar 0 (menu 36), full
        let _ = next_content(&rx);
        give(&mut g, "stone", 10); // cannot merge: lands in hotbar 1 (menu 37)
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 37, 0, ClickKind::Pickup)); // lift the 10
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 2, 37, 0, ClickKind::PickupAll));
        let (_, _, _, carried) = next_content(&rx);
        // Gathers up to 64: the full stack at menu 36 merges in (pass 1
        // skips full stacks only on pass 0).
        assert_eq!(carried.expect("gathered").count(), 64);
    }

    #[test]
    fn set_carried_item_tracks_selection_without_echo() {
        let (mut g, rx) = harness();
        g.handle(Inbound::SetCarriedItem { conn: 0, slot: 3 });
        assert_eq!(g.selected_slot_for_test(0), 3);
        assert!(rx.try_recv().is_err(), "no echo to the selecting client");
        g.handle(Inbound::SetCarriedItem { conn: 0, slot: 9 });
        assert_eq!(g.selected_slot_for_test(0), 3, "out-of-range ignored");
    }

    #[test]
    fn clicks_on_foreign_containers_are_ignored() {
        let (mut g, rx) = harness();
        give(&mut g, "stone", 5);
        let _ = next_content(&rx);
        click(&mut g, click_raw(7, 1, 0, 0, ClickKind::Pickup));
        // Drain the give command's pending feedback, then expect silence.
        while let Ok(frame) = rx.try_recv() {
            assert!(
                matches!(frame, Outbound::Frame { id: 0x7c, .. }),
                "no response for a closed menu"
            );
        }
    }

    #[test]
    fn drop_outside_secondary_leaves_one() {
        let (mut g, rx) = harness();
        give(&mut g, "stone", 5);
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 1, 36, 0, ClickKind::Pickup));
        let _ = next_content(&rx);
        click(&mut g, click_raw(0, 2, -999, 1, ClickKind::Pickup));
        let (_, _, slots, carried) = next_content(&rx);
        let carried = carried.expect("secondary drop leaves the rest");
        assert_eq!(carried.count(), 4, "one item dropped outside");
        assert!(slots.iter().all(Option::is_none));
    }
}
