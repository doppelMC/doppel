//! Block breaking (`player_action`): the serverbound parse, the per-block
//! hardness table, and the dig state machine (start / redirect / stop /
//! abort, delayed destroys, the destruction-stage overlay) with its
//! per-player state. The handlers live in the `impl Game` block below,
//! invoked from game.rs's dispatch and tick schedule.

use anyhow::{bail, Context, Result};
use doppel_protocol::Reader;

use crate::game::{ConnId, Game, AIR_STATE, WORLD_MAX_Y};

// --- breaking hooks ---

/// Serverbound `player_action`, play state: registration order 41 in the
/// pinned 26.3 template (26.2's 0x28 + the insertions before it).
pub const SERVERBOUND_PLAYER_ACTION: i32 = 0x29;

/// Serverbound `punch` (the arm swing), play state: registration order
/// 46. Empty body; the broadcast back is client-cosmetic animation.
pub const SERVERBOUND_PUNCH: i32 = 0x2e;

/// Clientbound `block_destruction`: registration order 5, directly ahead
/// of the pinned block_entity_data (6) / block_event (7) / block_update
/// (8) trio.
pub const CLIENTBOUND_BLOCK_DESTRUCTION: i32 = 0x05;

/// player_action action ordinals (the reference enum order).
pub const ACTION_START_DESTROY: i32 = 0;
pub const ACTION_CHANGE_DESTROY_DIRECTION: i32 = 1;
pub const ACTION_ABORT_DESTROY: i32 = 2;
pub const ACTION_STOP_DESTROY: i32 = 3;
pub const ACTION_DROP_ALL: i32 = 4;
pub const ACTION_DROP_ITEM: i32 = 5;
pub const ACTION_RELEASE_USE: i32 = 6;
pub const ACTION_SWAP_OFFHAND: i32 = 7;
pub const ACTION_STAB: i32 = 8;

/// One decoded player_action.
pub struct PlayerAction {
    pub action: i32,
    pub x: i32,
    pub y: i32,
    pub z: i32,
    /// Direction 3D data value (0=down .. 5=east).
    pub direction: u8,
    pub sequence: i32,
}

/// Parses a serverbound player_action body (after the packet id):
/// action VarInt, pos i64 (packed BlockPos), direction VarInt,
/// sequence VarInt. The drop/swap arms carry a pos too (ignored there);
/// every arm shares the one body layout.
pub fn parse_player_action(body: &[u8]) -> Result<PlayerAction> {
    let mut r = Reader::new(body);
    let action = r.read_varint().context("action")?;
    if !(0..=8).contains(&action) {
        bail!("player_action action {action}");
    }
    let packed = r.read_i64().context("block pos")?;
    let direction = r.read_varint().context("direction")?;
    if !(0..=5).contains(&direction) {
        bail!("player_action direction {direction}");
    }
    let sequence = r.read_varint().context("sequence")?;
    if r.remaining() != 0 {
        bail!("{} trailing bytes in player_action", r.remaining());
    }
    let (x, y, z) = crate::placement::unpack_block_pos(packed);
    Ok(PlayerAction {
        action,
        x,
        y,
        z,
        direction: direction as u8,
        sequence,
    })
}

/// The per-tick destroy progress for a bare hand (the only holder this
/// build models): speed 1.0 / destroyTime / (30 when the block drops
/// without a tool, else 100). destroyTime 0 divides to infinity
/// (insta-break); a negative time returns 0.0 (the reference's
/// unbreakable short-circuit).
pub fn per_tick_progress(destroy_time: f32, requires_tool: bool) -> f32 {
    if destroy_time < 0.0 {
        return 0.0;
    }
    let modifier = if requires_tool { 100.0f32 } else { 30.0 };
    1.0f32 / destroy_time / modifier
}

/// The wire-progress stage for a progress fraction: (int)(p * 10).
pub fn destroy_stage(progress: f32) -> i32 {
    (progress * 10.0f32) as i32
}

/// (destroyTime, requiresCorrectToolForDrops) for the families the
/// circuits and the survival loop touch. Unlisted blocks read 0.0/false:
/// the reference's float-field default, which insta-breaks.
///
/// NOTE(breaking): the full per-block table is future registry work.
pub fn hardness(name: &str) -> (f32, bool) {
    match name {
        "minecraft:torch"
        | "minecraft:wall_torch"
        | "minecraft:redstone_torch"
        | "minecraft:redstone_wall_torch"
        | "minecraft:redstone_wire"
        | "minecraft:repeater"
        | "minecraft:comparator" => (0.0, false),
        "minecraft:glass" => (0.3, false),
        "minecraft:dirt" => (0.5, false),
        "minecraft:lever" => (0.5, false),
        "minecraft:grass_block" => (0.6, false),
        "minecraft:piston" | "minecraft:sticky_piston" => (1.5, false),
        "minecraft:stone"
        | "minecraft:granite"
        | "minecraft:polished_granite"
        | "minecraft:diorite"
        | "minecraft:polished_diorite"
        | "minecraft:andesite"
        | "minecraft:polished_andesite" => (1.5, true),
        "minecraft:oak_planks" | "minecraft:oak_log" | "minecraft:shulker_box" => (2.0, false),
        "minecraft:cobblestone" => (2.0, true),
        "minecraft:chest"
        | "minecraft:trapped_chest"
        | "minecraft:crafting_table"
        | "minecraft:barrel" => (2.5, false),
        "minecraft:deepslate" | "minecraft:hopper" | "minecraft:observer" => (3.0, true),
        "minecraft:cobbled_deepslate" => (3.5, false),
        "minecraft:furnace" | "minecraft:dispenser" | "minecraft:dropper" => (3.5, true),
        "minecraft:redstone_block" => (5.0, true),
        "minecraft:ender_chest" => (22.5, false),
        "minecraft:obsidian" => (50.0, true),
        // destroyTime -1: unbreakable (progress pins at zero).
        "minecraft:bedrock" | "minecraft:barrier" | "minecraft:moving_piston" => (-1.0, false),
        _ => (0.0, false),
    }
}

// ---------------------------------------------------------------------
// Per-player dig state
// ---------------------------------------------------------------------

/// The dig state machine grouped on Player. Vanilla keeps this in its
/// destroying-block state plus a per-player game-mode tick counter.
pub(crate) struct PlayerDigState {
    /// An active dig (vanilla's destroying-block state).
    dig: Option<DigState>,
    /// Vanilla's delayed destroy: a STOP below the finish threshold
    /// keeps grinding with the dig's start tick as the spent-tick count.
    delayed_destroy: Option<DelayedDestroy>,
    /// The last destruction stage sent for this player (dedup guard;
    /// -1 means no overlay).
    last_stage: i32,
    /// Ticks since this player joined: the counter dig progress runs on
    /// (vanilla's per-player game-mode ticks, not world time).
    game_ticks: u64,
}

impl Default for PlayerDigState {
    fn default() -> Self {
        PlayerDigState {
            dig: None,
            delayed_destroy: None,
            last_stage: -1,
            game_ticks: 0,
        }
    }
}

/// One active dig. `start_tick` is the digger's own tick count when the
/// dig began (vanilla's destroyProgressStart).
#[derive(Clone, Copy)]
struct DigState {
    pos: (i32, i32, i32),
    start_tick: u64,
    direction: u8,
}

/// A destroy that STOPped below the finish threshold: it completes on a
/// later tick with the ORIGINAL dig's start tick as the multiplier.
#[derive(Clone, Copy)]
struct DelayedDestroy {
    pos: (i32, i32, i32),
    tick_start: u64,
}

impl Game {
    /// One serverbound player_action: dig lifecycle, drops, swaps.
    pub(crate) fn player_action(&mut self, conn: ConnId, act: crate::dig::PlayerAction) {
        let crate::dig::PlayerAction {
            action,
            x,
            y,
            z,
            direction,
            sequence,
        } = act;
        // The sequence would echo in a block_changed_ack; this build
        // sends none (the NOTE(placement) wire gap).
        let _ = sequence;
        match action {
            crate::dig::ACTION_START_DESTROY
            | crate::dig::ACTION_CHANGE_DESTROY_DIRECTION
            | crate::dig::ACTION_ABORT_DESTROY
            | crate::dig::ACTION_STOP_DESTROY => {
                self.handle_block_break_action(conn, action, (x, y, z), direction)
            }
            crate::dig::ACTION_DROP_ALL => self.drop_held(conn, true),
            crate::dig::ACTION_DROP_ITEM => self.drop_held(conn, false),
            // STAB (piercing weapons), the offhand swap, and item release
            // have no models yet; parse-accept matches the wire contract.
            _ => {}
        }
    }

    /// The dig state machine (start / redirect / stop / abort), with the
    /// interaction-range gate every arm shares.
    fn handle_block_break_action(
        &mut self,
        conn: ConnId,
        action: i32,
        pos: (i32, i32, i32),
        direction: u8,
    ) {
        if !self.within_reach(conn, pos) {
            // An abort is the one arm that still runs out of range, and
            // only against the dig it interrupts.
            if action == crate::dig::ACTION_ABORT_DESTROY
                && self.players.get(&conn).is_some_and(|p| p.dig.dig.is_some())
            {
                self.abort_destroy(conn, pos);
            }
            return;
        }
        if pos.1 > WORLD_MAX_Y {
            // The reference echoes the block back to the actor here; no
            // per-actor block-update path exists yet.
            return;
        }
        match action {
            crate::dig::ACTION_START_DESTROY => self.start_destroy(conn, pos, direction),
            crate::dig::ACTION_CHANGE_DESTROY_DIRECTION => {
                if let Some(p) = self.players.get_mut(&conn) {
                    if let Some(dig) = p.dig.dig.as_mut() {
                        dig.direction = direction;
                    }
                }
            }
            crate::dig::ACTION_STOP_DESTROY => self.stop_destroy(conn, pos),
            crate::dig::ACTION_ABORT_DESTROY => self.abort_destroy(conn, pos),
            _ => {}
        }
    }

    /// Interaction reach: distance from the eye (feet + 1.62) to the
    /// target's unit cube, inside block range + 1.0 of slack. Creative
    /// reaches 5.0, survival 4.5.
    fn within_reach(&self, conn: ConnId, pos: (i32, i32, i32)) -> bool {
        let Some(p) = self.players.get(&conn) else {
            return false;
        };
        let axis = |v: f64, lo: f64| (lo - v).max(0.0).max(v - (lo + 1.0));
        let (ex, ey, ez) = (p.x, p.y + 1.62, p.z);
        let dx = axis(ex, pos.0 as f64);
        let dy = axis(ey, pos.1 as f64);
        let dz = axis(ez, pos.2 as f64);
        let range = if p.inv.creative { 5.0 } else { 4.5 } + 1.0;
        dx * dx + dy * dy + dz * dz < range * range
    }

    fn start_destroy(&mut self, conn: ConnId, pos: (i32, i32, i32), direction: u8) {
        let creative = self.players.get(&conn).is_some_and(|p| p.inv.creative);
        let block = self.get_block(pos.0, pos.1, pos.2);
        let name = block.as_ref().map(|(n, _)| n.clone());
        // Unloaded reads count as air: the reference always sees a block.
        let is_air = !matches!(name.as_deref(), Some(n) if n != "minecraft:air");
        if creative {
            self.break_block(conn, pos);
            return;
        }
        // One tick of progress: bare hand, the block's own hardness.
        let (dt, tool) = name
            .as_deref()
            .filter(|_| !is_air)
            .map(crate::dig::hardness)
            .unwrap_or((0.0, false));
        let progress = if is_air {
            1.0f32
        } else {
            crate::dig::per_tick_progress(dt, tool)
        };
        if !is_air && progress >= 1.0 {
            // Insta-break (destroyTime 0 divides to infinity).
            self.break_block(conn, pos);
            return;
        }
        {
            let Some(p) = self.players.get_mut(&conn) else {
                return;
            };
            // NOTE(breaking): interrupting a different dig echoes the old
            // target's block to the actor in the reference; skipped.
            let start_tick = p.dig.game_ticks;
            p.dig.dig = Some(DigState {
                pos,
                start_tick,
                direction,
            });
        }
        let stage = crate::dig::destroy_stage(progress);
        self.broadcast_destruction(conn, pos, stage);
        if let Some(p) = self.players.get_mut(&conn) {
            p.dig.last_stage = stage;
        }
    }

    fn stop_destroy(&mut self, conn: ConnId, pos: (i32, i32, i32)) {
        let Some((start_tick, now)) = self.players.get(&conn).and_then(|p| {
            p.dig
                .dig
                .filter(|d| d.pos == pos)
                .map(|d| (d.start_tick, p.dig.game_ticks))
        }) else {
            return;
        };
        let Some((name, _)) = self.get_block(pos.0, pos.1, pos.2) else {
            return;
        };
        if name == "minecraft:air" {
            return;
        }
        let (dt, tool) = crate::dig::hardness(&name);
        let progress = crate::dig::per_tick_progress(dt, tool) * (now - start_tick + 1) as f32;
        if progress >= 0.7 {
            if let Some(p) = self.players.get_mut(&conn) {
                p.dig.dig = None;
            }
            self.broadcast_destruction(conn, pos, -1);
            self.break_block(conn, pos);
            return;
        }
        // Below threshold the dig converts to a delayed destroy that
        // keeps the original start tick as its spent-tick count.
        if let Some(p) = self.players.get_mut(&conn) {
            if p.dig.delayed_destroy.is_none() {
                p.dig.dig = None;
                p.dig.delayed_destroy = Some(DelayedDestroy {
                    pos,
                    tick_start: start_tick,
                });
            }
        }
    }

    fn abort_destroy(&mut self, conn: ConnId, pos: (i32, i32, i32)) {
        let old = self
            .players
            .get_mut(&conn)
            .and_then(|p| p.dig.dig.take())
            .map(|d| d.pos);
        if let Some(old) = old {
            if old != pos {
                self.broadcast_destruction(conn, old, -1);
            }
        }
        self.broadcast_destruction(conn, pos, -1);
    }

    /// Removes the block (air + neighbor notifications + broadcast) and
    /// spawns its drop; the surrounding removal choreography (pairing) is
    /// the set_block path's own.
    fn break_block(&mut self, conn: ConnId, pos: (i32, i32, i32)) {
        // --- survival hooks (entities.rs) ---
        // Creative breaks keep no drops; tool-gated blocks drop nothing
        // bare-handed (that gate lives in spawn_break_drop).
        let creative = self.players.get(&conn).is_some_and(|p| p.inv.creative);
        let name = self
            .get_block(pos.0, pos.1, pos.2)
            .map(|(n, _)| n)
            .filter(|n| n != "minecraft:air");
        self.set_block(pos.0, pos.1, pos.2, AIR_STATE, true);
        if !creative {
            if let Some(name) = name {
                self.spawn_break_drop(pos, &name);
            }
        }
    }

    /// Spends the held stack: whole stack for drop-all, one item for the
    /// single drop. The discarded half spawns a thrown item entity; the
    /// slot change queues for the per-tick menu broadcast like any other
    /// non-click change.
    fn drop_held(&mut self, conn: ConnId, all: bool) {
        let dropped = {
            let Some(p) = self.players.get_mut(&conn) else {
                return;
            };
            let slot = p.inv.inventory.selected() as usize;
            let held = p.inv.inventory.get(slot);
            if held.is_none() {
                return;
            }
            let (dropped, remaining) = if all {
                (held, None)
            } else {
                let mut stack = held.expect("held");
                let one = stack.split(1);
                let remaining = (!stack.is_empty()).then_some(stack);
                (Some(one), remaining)
            };
            p.inv.inventory.set(slot, remaining);
            p.inv.pending_sync.insert(slot);
            dropped
        };
        // --- survival hooks (entities.rs) ---
        if let Some(stack) = dropped {
            self.spawn_thrown_drop(conn, stack);
        }
    }

    /// Per-tick dig advance: delayed destroys finish, active digs deepen
    /// their overlay. The two are exclusive per player (a pending delayed
    /// destroy starves the active dig). Breaks land in this tick's flush.
    pub(crate) fn advance_digs(&mut self) {
        enum Step {
            ClearDelayed,
            DelayedStage((i32, i32, i32), i32),
            DelayedBreak((i32, i32, i32), i32, bool),
            ClearDig((i32, i32, i32)),
            DigStage((i32, i32, i32), i32),
        }
        let conns: Vec<ConnId> = self.players.keys().copied().collect();
        for conn in conns {
            // The dig clock is per player and counts every tick, dig or
            // not (the reference increments its counter before branching).
            if let Some(p) = self.players.get_mut(&conn) {
                p.dig.game_ticks += 1;
            }
            let step = {
                let Some(p) = self.players.get(&conn) else {
                    continue;
                };
                if let Some(d) = p.dig.delayed_destroy {
                    match self.get_block(d.pos.0, d.pos.1, d.pos.2) {
                        Some((n, _)) if n != "minecraft:air" => {
                            let (dt, tool) = crate::dig::hardness(&n);
                            // The delayed path passes the START tick, not
                            // the ticks since the STOP: progress is frozen
                            // unless the dig began late in the session.
                            let progress =
                                crate::dig::per_tick_progress(dt, tool) * (d.tick_start + 1) as f32;
                            let stage = crate::dig::destroy_stage(progress);
                            if progress >= 1.0 {
                                Step::DelayedBreak(d.pos, stage, stage != p.dig.last_stage)
                            } else if stage != p.dig.last_stage {
                                Step::DelayedStage(d.pos, stage)
                            } else {
                                continue;
                            }
                        }
                        _ => Step::ClearDelayed,
                    }
                } else if let Some(dig) = p.dig.dig {
                    match self.get_block(dig.pos.0, dig.pos.1, dig.pos.2) {
                        Some((n, _)) if n != "minecraft:air" => {
                            let (dt, tool) = crate::dig::hardness(&n);
                            let spent = (p.dig.game_ticks - dig.start_tick) as f32;
                            let progress = crate::dig::per_tick_progress(dt, tool) * (spent + 1.0);
                            let stage = crate::dig::destroy_stage(progress);
                            if stage != p.dig.last_stage {
                                Step::DigStage(dig.pos, stage)
                            } else {
                                continue;
                            }
                        }
                        _ => Step::ClearDig(dig.pos),
                    }
                } else {
                    continue;
                }
            };
            match step {
                Step::ClearDelayed => {
                    if let Some(p) = self.players.get_mut(&conn) {
                        p.dig.delayed_destroy = None;
                    }
                }
                Step::DelayedStage(pos, stage) => {
                    self.broadcast_destruction(conn, pos, stage);
                    if let Some(p) = self.players.get_mut(&conn) {
                        p.dig.last_stage = stage;
                    }
                }
                Step::DelayedBreak(pos, stage, changed) => {
                    if changed {
                        self.broadcast_destruction(conn, pos, stage);
                    }
                    if let Some(p) = self.players.get_mut(&conn) {
                        p.dig.last_stage = stage;
                        p.dig.delayed_destroy = None;
                    }
                    self.break_block(conn, pos);
                }
                Step::ClearDig(pos) => {
                    self.broadcast_destruction(conn, pos, -1);
                    if let Some(p) = self.players.get_mut(&conn) {
                        p.dig.last_stage = -1;
                        p.dig.dig = None;
                    }
                }
                Step::DigStage(pos, stage) => {
                    self.broadcast_destruction(conn, pos, stage);
                    if let Some(p) = self.players.get_mut(&conn) {
                        p.dig.last_stage = stage;
                    }
                }
            }
        }
    }

    /// The destruction overlay broadcast: every player within 32 blocks
    /// of the block's origin except the digger, whose own client renders
    /// the overlay locally.
    fn broadcast_destruction(&mut self, digger: ConnId, pos: (i32, i32, i32), stage: i32) {
        let entity_id = self.players.get(&digger).map(|p| p.entity_id).unwrap_or(0);
        let packed = (((pos.0 as i64) & 0x3ff_ffff) << 38)
            | (((pos.2 as i64) & 0x3ff_ffff) << 12)
            | ((pos.1 as i64) & 0xfff);
        let mut body = Vec::with_capacity(14);
        doppel_protocol::write_varint(&mut body, entity_id);
        body.extend_from_slice(&packed.to_be_bytes());
        // writeByte semantics: the low 8 bits of the stage int.
        body.push(stage as u8);
        let targets: Vec<ConnId> = self
            .players
            .iter()
            .filter(|(&conn, p)| {
                if conn == digger {
                    return false;
                }
                let (dx, dy, dz) = (pos.0 as f64 - p.x, pos.1 as f64 - p.y, pos.2 as f64 - p.z);
                dx * dx + dy * dy + dz * dz < 1024.0
            })
            .map(|(&conn, _)| conn)
            .collect();
        for conn in targets {
            self.send(conn, crate::dig::CLIENTBOUND_BLOCK_DESTRUCTION, &body);
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Inbound, Outbound, DIR_UP};
    use crate::inventory::{ItemStack, PACKET_CONTAINER_SET_SLOT};
    use crate::WireChunk;

    /// A game with stone-floored chunks (floor y=99) and one viewer, the
    /// piston-test layout.
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

    fn give(g: &mut Game, item: &str, count: i32) {
        g.handle(Inbound::Give {
            conn: 0,
            item: item.to_string(),
            count,
        });
    }

    fn use_on(g: &mut Game, x: i32, y: i32, z: i32, face: u8, hand: u8) {
        g.handle(Inbound::UseItemOn {
            conn: 0,
            x,
            y,
            z,
            face,
            cursor_x: 0.5,
            cursor_y: 1.0,
            cursor_z: 0.5,
            hand,
            sequence: 1,
        });
    }

    fn at(g: &Game, x: i32, y: i32, z: i32) -> String {
        g.block_label_for_test(x, y, z)
    }

    /// The next queued container_set_slot frame, decoded to
    /// (containerId, stateId, menu slot, stack).
    fn next_set_slot(
        rx: &std::sync::mpsc::Receiver<Outbound>,
    ) -> Option<(i32, i32, i16, Option<ItemStack>)> {
        while let Ok(frame) = rx.try_recv() {
            if let Outbound::Frame { id, body } = frame {
                if id != PACKET_CONTAINER_SET_SLOT {
                    continue;
                }
                let mut r = doppel_protocol::Reader::new(&body);
                let container_id = r.read_varint().ok()?;
                let state_id = r.read_varint().ok()?;
                let slot = r.read_u16().ok()? as i16;
                let stack = crate::inventory::decode_item_stack(&mut r).ok()?;
                return Some((container_id, state_id, slot, stack));
            }
        }
        None
    }

    fn packed_pos(x: i32, y: i32, z: i32) -> i64 {
        (((x as i64) & 0x3ff_ffff) << 38) | (((z as i64) & 0x3ff_ffff) << 12) | (y as i64 & 0xfff)
    }

    // -- breaking ------------------------------------------------------

    fn action_bytes(action: i32, x: i32, y: i32, z: i32, dir: i32, sequence: i32) -> Vec<u8> {
        let mut body = Vec::new();
        doppel_protocol::write_varint(&mut body, action);
        body.extend(&packed_pos(x, y, z).to_be_bytes());
        doppel_protocol::write_varint(&mut body, dir);
        doppel_protocol::write_varint(&mut body, sequence);
        body
    }

    #[test]
    fn parse_player_action_golden_and_negative() {
        let a = parse_player_action(&action_bytes(0, 10, 100, 7, 1, 42)).unwrap();
        assert_eq!((a.action, a.direction, a.sequence), (0, 1, 42));
        assert_eq!((a.x, a.y, a.z), (10, 100, 7));
        let a = parse_player_action(&action_bytes(8, -5, -70, -33, 5, 1)).unwrap();
        assert_eq!((a.x, a.y, a.z), (-5, -70, -33));
        assert_eq!(a.action, 8);
        assert_eq!(a.direction, 5);
    }

    #[test]
    fn parse_player_action_rejects_bad_fields() {
        assert!(parse_player_action(&action_bytes(9, 0, 0, 0, 1, 1)).is_err());
        assert!(parse_player_action(&action_bytes(0, 0, 0, 0, 6, 1)).is_err());
        let mut trailing = action_bytes(0, 0, 0, 0, 1, 1);
        trailing.push(0);
        assert!(parse_player_action(&trailing).is_err());
        assert!(parse_player_action(&[0x00]).is_err());
    }

    /// Stone bare-handed: speed 1.0 / 1.5 destroyTime / 100 (wrong tool)
    /// = one tick of progress in 150; dirt carries no tool requirement,
    /// so its divisor is 30.
    #[test]
    fn destroy_progress_math() {
        let stone = per_tick_progress(1.5, true);
        assert!(stone > 0.00666 && stone < 0.00667, "{stone}");
        assert!(stone * 149.0 < 1.0);
        assert!(stone * 150.0 >= 1.0);
        let dirt = per_tick_progress(0.5, false);
        assert!(dirt > 0.0666 && dirt < 0.0667, "{dirt}");
        assert!(dirt * 14.0 < 1.0 && dirt * 15.0 >= 1.0);
        // destroyTime 0 divides to infinity: the insta-break family.
        assert!(per_tick_progress(0.0, false) >= 1.0);
        // Negative destroyTime is the unbreakable short-circuit.
        assert_eq!(per_tick_progress(-1.0, false), 0.0);
        // Stage boundaries over the stone dig (ticks spent 0..=149).
        assert_eq!(destroy_stage(stone * 1.0), 0);
        assert_eq!(destroy_stage(stone * 15.0), 1);
        assert_eq!(destroy_stage(stone * 149.0), 9);
        assert_eq!(destroy_stage(stone * 150.0), 10);
        // The hardness table spot-checks: requires-tool stone family,
        // tool-free chests, unbreakable-looking obsidian at 50.
        assert_eq!(hardness("minecraft:stone"), (1.5, true));
        assert_eq!(hardness("minecraft:chest"), (2.5, false));
        assert_eq!(hardness("minecraft:obsidian"), (50.0, true));
        assert_eq!(hardness("minecraft:redstone_wire"), (0.0, false));
        // Unlisted blocks read the float-field default: insta-break.
        assert_eq!(hardness("minecraft:whatever"), (0.0, false));
    }

    /// A second player near the dig site receives the overlay frames.
    fn dig_harness() -> (Game, std::sync::mpsc::Receiver<Outbound>) {
        let (mut g, _rx0) = harness();
        // Put the digger above the target and bring a witness in range.
        g.handle(Inbound::Tp {
            conn: 0,
            x: 5.0,
            y: 101.0,
            z: 5.0,
        });
        let (tx1, rx1) = std::sync::mpsc::channel::<Outbound>();
        g.join_viewer_for_test(1, &[(-1, 0), (0, 0), (1, 0), (2, 0)], tx1);
        g.handle(Inbound::Tp {
            conn: 1,
            x: 5.0,
            y: 101.0,
            z: 6.0,
        });
        (g, rx1)
    }

    fn act(g: &mut Game, action: i32, x: i32, y: i32, z: i32) {
        g.handle(Inbound::PlayerAction {
            conn: 0,
            act: PlayerAction {
                action,
                x,
                y,
                z,
                direction: DIR_UP,
                sequence: 1,
            },
        });
    }

    /// The witness's block_destruction frames as (entity, pos, stage).
    fn dig_frames(rx: &std::sync::mpsc::Receiver<Outbound>) -> Vec<(i32, (i32, i32, i32), i32)> {
        let mut out = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            let Outbound::Frame { id, body } = frame else {
                continue;
            };
            if id != CLIENTBOUND_BLOCK_DESTRUCTION {
                continue;
            }
            let mut o = 0usize;
            let mut entity = 0i32;
            let mut sh = 0u32;
            while o < body.len() {
                let b = body[o];
                o += 1;
                entity |= i32::from(b & 0x7f) << sh;
                sh += 7;
                if b & 0x80 == 0 {
                    break;
                }
            }
            if body.len() < o + 9 {
                continue;
            }
            let packed = i64::from_be_bytes(body[o..o + 8].try_into().unwrap());
            let x = (packed >> 38) as i32;
            let z = ((packed >> 12) & 0x3ff_ffff) as i32;
            let y = (packed & 0xfff) as i32;
            out.push((entity, (x, y, z), body[o + 8] as i8 as i32));
        }
        out
    }

    #[test]
    fn dig_broadcasts_stages_and_holds() {
        let (mut g, rx1) = dig_harness();
        // Bare-hand stone floor: stage 0 first, deepening one step per
        // 15 ticks. A held dig never breaks on its own: the client ends
        // digs, not the server.
        act(&mut g, ACTION_START_DESTROY, 5, 99, 5);
        let frames = dig_frames(&rx1);
        assert_eq!(frames, vec![(1, (5, 99, 5), 0)], "{frames:?}");
        for _ in 0..35 {
            g.tick_once_for_test();
        }
        let frames = dig_frames(&rx1);
        let stages: Vec<i32> = frames.iter().map(|f| f.2).collect();
        assert_eq!(stages, vec![1, 2], "{frames:?}");
        assert_eq!(at(&g, 5, 99, 5), "minecraft:stone[]");
        // Hold past 0.7 (105 ticks of progress), then STOP: the break
        // lands immediately with a -1 clear first.
        for _ in 0..75 {
            g.tick_once_for_test();
        }
        act(&mut g, ACTION_STOP_DESTROY, 5, 99, 5);
        assert_eq!(at(&g, 5, 99, 5), "minecraft:air[]");
        let frames = dig_frames(&rx1);
        assert_eq!(frames.last(), Some(&(1, (5, 99, 5), -1)), "{frames:?}");
    }

    #[test]
    fn stop_below_threshold_breaks_next_tick() {
        let (mut g, rx1) = dig_harness();
        // The digger has been in the world 200 ticks when the dig starts;
        // the delayed pass multiplies progress by that per-player counter.
        for _ in 0..200 {
            g.tick_once_for_test();
        }
        // START and STOP in the same tick: below 0.7 progress the dig
        // converts to a delayed destroy that finishes on the next tick.
        act(&mut g, ACTION_START_DESTROY, 5, 99, 5);
        act(&mut g, ACTION_STOP_DESTROY, 5, 99, 5);
        assert_eq!(at(&g, 5, 99, 5), "minecraft:stone[]");
        g.tick_once_for_test();
        assert_eq!(at(&g, 5, 99, 5), "minecraft:air[]");
        let frames = dig_frames(&rx1);
        assert_eq!(frames.len(), 2, "{frames:?}");
        assert_eq!(frames[0], (1, (5, 99, 5), 0));
        // The delayed pass multiplies by the dig's start tick, so the
        // closing stage wraps past the byte's 0..10 overlay range.
        assert!(frames[1].2 != 0 && frames[1].2 != -1, "{frames:?}");
    }

    #[test]
    fn stop_below_threshold_arms_frozen_delayed_destroy() {
        let (mut g, rx1) = dig_harness();
        // Sixty ticks in the world when the dig starts: the delayed pass
        // multiplies one tick of stone progress by the start tick.
        for _ in 0..60 {
            g.tick_once_for_test();
        }
        act(&mut g, ACTION_START_DESTROY, 5, 99, 5);
        act(&mut g, ACTION_STOP_DESTROY, 5, 99, 5);
        g.tick_once_for_test();
        // Start stage 0, then the frozen delayed stage (61/150 -> 4); the
        // progress never grows again, so nothing more broadcasts.
        let stages: Vec<i32> = dig_frames(&rx1).into_iter().map(|f| f.2).collect();
        assert_eq!(stages, vec![0, 4], "{stages:?}");
        assert_eq!(at(&g, 5, 99, 5), "minecraft:stone[]");
        for _ in 0..40 {
            g.tick_once_for_test();
        }
        assert!(dig_frames(&rx1).is_empty());
        assert_eq!(at(&g, 5, 99, 5), "minecraft:stone[]");
    }

    #[test]
    fn armed_delayed_destroy_starves_held_dig() {
        let (mut g, rx1) = dig_harness();
        // Release a fresh dig below the threshold: the pending delayed
        // destroy owns the per-tick pass, so a later dig never deepens.
        act(&mut g, ACTION_START_DESTROY, 5, 99, 5);
        act(&mut g, ACTION_STOP_DESTROY, 5, 99, 5);
        let stages: Vec<i32> = dig_frames(&rx1).into_iter().map(|f| f.2).collect();
        assert_eq!(stages, vec![0]);
        act(&mut g, ACTION_START_DESTROY, 5, 99, 5);
        let stages: Vec<i32> = dig_frames(&rx1).into_iter().map(|f| f.2).collect();
        assert_eq!(stages, vec![0]);
        for _ in 0..200 {
            g.tick_once_for_test();
        }
        assert!(dig_frames(&rx1).is_empty());
        assert_eq!(at(&g, 5, 99, 5), "minecraft:stone[]");
    }

    #[test]
    fn abort_clears_the_overlay() {
        let (mut g, rx1) = dig_harness();
        act(&mut g, ACTION_START_DESTROY, 5, 99, 5);
        act(&mut g, ACTION_ABORT_DESTROY, 5, 99, 5);
        let stages: Vec<i32> = dig_frames(&rx1).into_iter().map(|f| f.2).collect();
        assert_eq!(stages, vec![0, -1]);
        for _ in 0..40 {
            g.tick_once_for_test();
        }
        // Nothing further broadcasts and the block stands.
        assert!(dig_frames(&rx1).is_empty());
        assert_eq!(at(&g, 5, 99, 5), "minecraft:stone[]");
    }

    #[test]
    fn creative_breaks_instantly_without_overlay() {
        let (mut g, rx1) = dig_harness();
        g.set_creative_for_test(0, true);
        act(&mut g, ACTION_START_DESTROY, 5, 99, 5);
        assert_eq!(at(&g, 5, 99, 5), "minecraft:air[]");
        assert!(dig_frames(&rx1).is_empty(), "no overlay for insta-breaks");
    }

    #[test]
    fn insta_break_family_breaks_without_creative() {
        let (mut g, rx1) = dig_harness();
        g.handle(Inbound::Setblock {
            conn: 0,
            x: 6,
            y: 100,
            z: 6,
            name: "minecraft:torch".to_string(),
        });
        assert_eq!(at(&g, 6, 100, 6), "minecraft:torch[]");
        act(&mut g, ACTION_START_DESTROY, 6, 100, 6);
        assert_eq!(at(&g, 6, 100, 6), "minecraft:air[]");
        assert!(dig_frames(&rx1).is_empty());
    }

    #[test]
    fn unbreakable_never_breaks() {
        let (mut g, rx1) = dig_harness();
        g.handle(Inbound::Setblock {
            conn: 0,
            x: 7,
            y: 100,
            z: 7,
            name: "minecraft:bedrock".to_string(),
        });
        act(&mut g, ACTION_START_DESTROY, 7, 100, 7);
        for _ in 0..20 {
            g.tick_once_for_test();
        }
        act(&mut g, ACTION_STOP_DESTROY, 7, 100, 7);
        for _ in 0..20 {
            g.tick_once_for_test();
        }
        assert_eq!(at(&g, 7, 100, 7), "minecraft:bedrock[]");
        // Only the stage-0 start ever went out.
        let stages: Vec<i32> = dig_frames(&rx1).into_iter().map(|f| f.2).collect();
        assert_eq!(stages, vec![0]);
    }

    /// A finished dirt dig removes its support: the redstone torch on
    /// top pops through the same neighbor-update path setblock drives.
    #[test]
    fn break_pops_attached_torch() {
        let (mut g, rx1) = dig_harness();
        g.handle(Inbound::Setblock {
            conn: 0,
            x: 5,
            y: 100,
            z: 5,
            name: "minecraft:dirt".to_string(),
        });
        give(&mut g, "minecraft:redstone_torch", 1);
        use_on(&mut g, 5, 100, 5, DIR_UP, 0);
        assert_eq!(at(&g, 5, 101, 5), "minecraft:redstone_torch[lit=true]");
        // Dirt bare-handed: 1.0/0.5/30 -> 0.7 at 11 ticks spent.
        act(&mut g, ACTION_START_DESTROY, 5, 100, 5);
        for _ in 0..10 {
            g.tick_once_for_test();
        }
        act(&mut g, ACTION_STOP_DESTROY, 5, 100, 5);
        assert_eq!(at(&g, 5, 100, 5), "minecraft:air[]");
        // The support loss schedules the torch's pop for the next tick.
        g.tick_once_for_test();
        assert_eq!(at(&g, 5, 101, 5), "minecraft:air[]");
        let _ = dig_frames(&rx1);
    }
    #[test]
    fn drop_all_then_drop_item() {
        let (mut g, rx) = harness();
        // A drop's slot change rides the next tick's menu broadcast, not
        // the drop itself.
        let held_count = |g: &Game| {
            let inv = g.player_inv_state_for_test(0).unwrap();
            let slot = inv.inventory.selected() as usize;
            inv.inventory.get(slot).map(|s| s.count())
        };
        give(&mut g, "minecraft:stone", 5);
        assert_eq!(held_count(&g), Some(5));
        g.handle(Inbound::PlayerAction {
            conn: 0,
            act: PlayerAction {
                action: ACTION_DROP_ALL,
                x: 0,
                y: 0,
                z: 0,
                direction: DIR_UP,
                sequence: 1,
            },
        });
        assert_eq!(held_count(&g), None, "drop-all empties the slot");
        assert!(
            next_set_slot(&rx).is_none(),
            "the sync waits for the next tick"
        );
        g.tick_once_for_test();
        let (_, _, slot, stack) = next_set_slot(&rx).unwrap();
        assert_eq!(slot, 36);
        assert_eq!(stack, None, "the broadcast shows the emptied slot");
        // Dropping from an empty hand changes nothing.
        g.handle(Inbound::PlayerAction {
            conn: 0,
            act: PlayerAction {
                action: ACTION_DROP_ITEM,
                x: 0,
                y: 0,
                z: 0,
                direction: DIR_UP,
                sequence: 2,
            },
        });
        assert_eq!(held_count(&g), None);
        g.tick_once_for_test();
        assert!(next_set_slot(&rx).is_none());
        // The single drop spends one item.
        give(&mut g, "minecraft:stone", 3);
        g.handle(Inbound::PlayerAction {
            conn: 0,
            act: PlayerAction {
                action: ACTION_DROP_ITEM,
                x: 0,
                y: 0,
                z: 0,
                direction: DIR_UP,
                sequence: 3,
            },
        });
        assert_eq!(held_count(&g), Some(2));
        g.tick_once_for_test();
        let (_, _, _, stack) = next_set_slot(&rx).unwrap();
        assert_eq!(stack.map(|s| s.count()), Some(2));
    }
    #[test]
    fn out_of_reach_dig_is_ignored() {
        let (mut g, rx1) = dig_harness();
        // Stand the digger away from the target (survival reach 4.5+1).
        g.handle(Inbound::Tp {
            conn: 0,
            x: 30.0,
            y: 101.0,
            z: 5.0,
        });
        act(&mut g, ACTION_START_DESTROY, 5, 99, 5);
        for _ in 0..5 {
            g.tick_once_for_test();
        }
        act(&mut g, ACTION_STOP_DESTROY, 5, 99, 5);
        for _ in 0..5 {
            g.tick_once_for_test();
        }
        assert_eq!(at(&g, 5, 99, 5), "minecraft:stone[]");
        assert!(dig_frames(&rx1).is_empty());
    }

    #[test]
    fn punch_is_accepted_without_effects() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Punch { conn: 0 });
        while let Ok(frame) = rx.try_recv() {
            let Outbound::Frame { id, .. } = frame else {
                continue;
            };
            panic!("punch broadcast 0x{id:02x}");
        }
    }
}
