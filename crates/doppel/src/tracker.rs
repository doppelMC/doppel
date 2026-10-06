//! The per-player entity tracker: which drops each connection sees, and
//! the movement sync each seen connection receives every tick. Pairing is
//! horizontal distance against the view distance plus chunk-stream
//! membership; the flush runs at a fixed phase between the block
//! broadcast point and the entity pass, so a drop spawned this tick pairs
//! immediately but first syncs next tick.

use std::collections::{BTreeMap, BTreeSet};

use super::entities::{
    encode_add_entity, encode_item_stack_data, encode_move_pos, encode_position_sync,
    encode_remove_entities, encode_set_motion, ENTITY_TYPE_ITEM, PACKET_ADD_ENTITY,
    PACKET_ENTITY_POSITION_SYNC, PACKET_MOVE_ENTITY_POS, PACKET_REMOVE_ENTITIES,
    PACKET_SET_ENTITY_DATA, PACKET_SET_ENTITY_MOTION,
};
use super::{ConnId, Game, VIEW_RADIUS};

/// The item family's pairing reach in blocks (6 chunks).
const ITEM_REACH: i32 = 6 * 16;
/// The movement sync cadence: the gate opens at least this often.
const SYNC_INTERVAL: i32 = 20;
/// A full position packet goes out at least this often, moving or not.
const FULL_SYNC_INTERVAL: i32 = 60;
/// After this many gated ticks without a position packet, one goes out.
const TELEPORT_DELAY_CAP: i32 = 400;
/// The squared-displacement threshold that counts as a position change.
const POSITION_CHANGED_EPS: f64 = 7.62939453125e-6;
/// The squared velocity difference that triggers a motion packet.
const MOTION_EPS: f64 = 1.0e-7;
/// Movement deltas ride the wire as 1/4096-block units.
const DELTA_SCALE: f64 = 4096.0;

/// One tracked entity's sync state (the counters the flush carries).
struct TrackedState {
    /// The position the last position packet implied.
    base: (f64, f64, f64),
    /// The velocity of the last motion packet.
    last_movement: (f64, f64, f64),
    tick_count: i32,
    teleport_delay: i32,
    was_on_ground: bool,
}

/// Tracker state the game thread owns.
#[derive(Default)]
pub(crate) struct EntityTrackers {
    states: BTreeMap<i32, TrackedState>,
    /// Entity id -> connections currently seeing it.
    seen: BTreeMap<i32, BTreeSet<ConnId>>,
    /// Entity id -> its section at the last flush (re-pairing trigger).
    last_section: BTreeMap<i32, (i32, i32, i32)>,
}

/// A wire unit of one movement component: floor(v * 4096 + 0.5).
fn encode_component(v: f64) -> i64 {
    (v * DELTA_SCALE + 0.5).floor() as i64
}

/// The quantization error a component would carry on the wire.
fn precision_loss(v: f64) -> f64 {
    encode_component(v) as f64 / DELTA_SCALE - v
}

/// The pairing test: horizontal distance inside the smaller of the
/// family reach and the view distance, and the player streams the
/// entity's chunk.
fn player_sees(g: &Game, conn: ConnId, x: f64, z: f64) -> bool {
    let Some(p) = g.players.get(&conn) else {
        return false;
    };
    let range = ITEM_REACH.min(VIEW_RADIUS * 16) as f64;
    let dx = p.x - x;
    let dz = p.z - z;
    dx * dx + dz * dz <= range * range
        && p.sent
            .contains(&(x.div_euclid(16.0) as i32, z.div_euclid(16.0) as i32))
}

impl Game {
    // --- tracker hooks (tracker.rs) ---

    /// Pairs one entity with one connection: the spawn packet, then the
    /// stack's entity data.
    fn pair_entity(&mut self, conn: ConnId, id: i32) {
        let Some(item) = self.survival.items.iter().find(|i| i.id == id) else {
            return;
        };
        let add = encode_add_entity(
            item.id,
            &item.uuid,
            ENTITY_TYPE_ITEM,
            item.x,
            item.y,
            item.z,
            (item.vx, item.vy, item.vz),
            item.yaw,
            0.0,
            item.yaw,
            0,
        );
        let data = encode_item_stack_data(item.id, Some(&item.stack));
        self.tracking.seen.entry(id).or_default().insert(conn);
        self.send(conn, PACKET_ADD_ENTITY, &add);
        self.send(conn, PACKET_SET_ENTITY_DATA, &data);
    }

    /// Unpairs one entity from one connection (one remove packet).
    fn unpair_entity(&mut self, conn: ConnId, id: i32) {
        self.send(conn, PACKET_REMOVE_ENTITIES, &encode_remove_entities(&[id]));
    }

    /// A new entity: register its sync state and pair with every player
    /// already in range.
    pub(crate) fn track_entity_spawned(&mut self, id: i32) {
        let Some(item) = self.survival.items.iter().find(|i| i.id == id) else {
            return;
        };
        let state = TrackedState {
            base: (item.x, item.y, item.z),
            last_movement: (item.vx, item.vy, item.vz),
            tick_count: 0,
            teleport_delay: 0,
            was_on_ground: item.on_ground,
        };
        self.tracking.states.insert(id, state);
        self.tracking.last_section.insert(
            id,
            (
                item.x.div_euclid(16.0) as i32,
                item.y.div_euclid(16.0) as i32,
                item.z.div_euclid(16.0) as i32,
            ),
        );
        let conns: Vec<ConnId> = self.players.keys().copied().collect();
        for conn in conns {
            let Some(item) = self.survival.items.iter().find(|i| i.id == id) else {
                continue;
            };
            if player_sees(self, conn, item.x, item.z) {
                self.pair_entity(conn, id);
            }
        }
    }

    /// A departing entity: one remove packet per seeing connection, then
    /// the state drops.
    pub(crate) fn track_entity_removed(&mut self, id: i32) {
        if self.tracking.states.remove(&id).is_none() {
            return;
        }
        self.tracking.last_section.remove(&id);
        let Some(seen) = self.tracking.seen.remove(&id) else {
            return;
        };
        for conn in seen {
            self.unpair_entity(conn, id);
        }
    }

    /// Sends one packet to every connection seeing the entity.
    pub(crate) fn entity_broadcast(&mut self, id: i32, packet: i32, body: &[u8]) {
        let Some(seen) = self.tracking.seen.get(&id) else {
            return;
        };
        let conns: Vec<ConnId> = seen.iter().copied().collect();
        for conn in conns {
            self.send(conn, packet, body);
        }
    }

    /// A player moved or joined: re-check every live entity's pairing
    /// against that player.
    pub(crate) fn track_player_view(&mut self, conn: ConnId) {
        let positions: Vec<(i32, f64, f64)> = self
            .survival
            .items
            .iter()
            .map(|i| (i.id, i.x, i.z))
            .collect();
        for (id, x, z) in positions {
            let paired = self
                .tracking
                .seen
                .get(&id)
                .is_some_and(|s| s.contains(&conn));
            let visible = player_sees(self, conn, x, z);
            match (paired, visible) {
                (false, true) => self.pair_entity(conn, id),
                (true, false) => {
                    if let Some(seen) = self.tracking.seen.get_mut(&id) {
                        seen.remove(&conn);
                    }
                    self.unpair_entity(conn, id);
                }
                _ => {}
            }
        }
    }

    /// A player left: forget the pairing (no packets, the connection is
    /// closing).
    pub(crate) fn track_player_left(&mut self, conn: ConnId) {
        for seen in self.tracking.seen.values_mut() {
            seen.remove(&conn);
        }
    }

    /// A player respawned: the client wiped its world, so every pairing
    /// is forgotten; the next view pass re-pairs what is in range.
    pub(crate) fn track_player_respawned(&mut self, conn: ConnId) {
        for seen in self.tracking.seen.values_mut() {
            seen.remove(&conn);
        }
    }

    /// The fixed-phase flush: re-pair entities that crossed a section,
    /// then sync movement per entity. Entities spawned this tick wait for
    /// the next one (the entity pass runs after this phase); entities
    /// outside every player's simulation range hold their counters.
    pub(crate) fn track_entities(&mut self) {
        let tick = self.tick;
        struct Snap {
            id: i32,
            x: f64,
            y: f64,
            z: f64,
            vx: f64,
            vy: f64,
            vz: f64,
            yaw: f32,
            on_ground: bool,
            vertical_collision: bool,
            horizontal_collision: bool,
            needs_sync: bool,
            stack_dirty: bool,
        }
        let snaps: Vec<Snap> = self
            .survival
            .items
            .iter()
            .filter(|i| i.born_tick != tick)
            .map(|i| Snap {
                id: i.id,
                x: i.x,
                y: i.y,
                z: i.z,
                vx: i.vx,
                vy: i.vy,
                vz: i.vz,
                yaw: i.yaw,
                on_ground: i.on_ground,
                vertical_collision: i.vertical_collision,
                horizontal_collision: i.horizontal_collision,
                needs_sync: i.needs_sync,
                stack_dirty: i.stack_dirty,
            })
            .collect();
        // Player positions and streamed chunks for the pairing and range
        // decisions.
        let players: Vec<(ConnId, f64, f64, i32, i32)> = self
            .players
            .iter()
            .map(|(&c, p)| {
                (
                    c,
                    p.x,
                    p.z,
                    p.x.div_euclid(16.0) as i32,
                    p.z.div_euclid(16.0) as i32,
                )
            })
            .collect();
        let mut states = std::mem::take(&mut self.tracking.states);
        let mut last_section = std::mem::take(&mut self.tracking.last_section);
        let mut clear_sync: Vec<i32> = Vec::new();
        let mut clear_dirty: Vec<i32> = Vec::new();
        for s in &snaps {
            let Some(state) = states.get_mut(&s.id) else {
                continue;
            };
            // A section crossing re-pairs against every player.
            let section = (
                s.x.div_euclid(16.0) as i32,
                s.y.div_euclid(16.0) as i32,
                s.z.div_euclid(16.0) as i32,
            );
            let section_changed = last_section.get(&s.id).copied() != Some(section);
            if section_changed {
                let paired: Vec<ConnId> = self
                    .tracking
                    .seen
                    .get(&s.id)
                    .map(|seen| seen.iter().copied().collect())
                    .unwrap_or_default();
                for (conn, _, _, _, _) in &players {
                    let was = paired.contains(conn);
                    let sees = player_sees(self, *conn, s.x, s.z);
                    match (was, sees) {
                        (false, true) => self.pair_entity(*conn, s.id),
                        (true, false) => {
                            if let Some(seen) = self.tracking.seen.get_mut(&s.id) {
                                seen.remove(conn);
                            }
                            self.unpair_entity(*conn, s.id);
                        }
                        _ => {}
                    }
                }
                last_section.insert(s.id, section);
            }
            // Outside every player's simulation range and quiet: the
            // counters hold.
            let in_ticking_range = players.iter().any(|(_, _, _, pcx, pcz)| {
                (s.x.div_euclid(16.0) as i32 - pcx).abs() <= VIEW_RADIUS
                    && (s.z.div_euclid(16.0) as i32 - pcz).abs() <= VIEW_RADIUS
            });
            if !section_changed && !s.needs_sync && !s.stack_dirty && !in_ticking_range {
                continue;
            }
            let gate = s.needs_sync || state.tick_count % SYNC_INTERVAL == 0 || s.stack_dirty;
            if gate {
                state.teleport_delay += 1;
                let (bx, by, bz) = state.base;
                let (dx, dy, dz) = (s.x - bx, s.y - by, s.z - bz);
                let position_changed = dx * dx + dy * dy + dz * dz >= POSITION_CHANGED_EPS;
                let should_send_position =
                    position_changed || state.tick_count % FULL_SYNC_INTERVAL == 0;
                // The move packet choice: a ground flip or a very stale
                // delay forces the full packet; else a short delta when
                // the cadence asks and the delta fits.
                enum Move {
                    Sync,
                    Pos(i16, i16, i16),
                    None,
                }
                let full =
                    state.teleport_delay > TELEPORT_DELAY_CAP || state.was_on_ground != s.on_ground;
                let kind = if full {
                    state.was_on_ground = s.on_ground;
                    state.teleport_delay = 0;
                    Move::Sync
                } else if should_send_position {
                    let xa = encode_component(s.x) - encode_component(bx);
                    let ya = encode_component(s.y) - encode_component(by);
                    let za = encode_component(s.z) - encode_component(bz);
                    let wire_short = -32768..=32767;
                    let too_big = !wire_short.contains(&xa)
                        || !wire_short.contains(&ya)
                        || !wire_short.contains(&za);
                    // Collision turns strict, wire loss on a crossed axis
                    // forces the full packet (the vertical case pairs each
                    // axis with its own loss, the horizontal one crosses).
                    let full_precision = (s.vertical_collision
                        && (xa != 0 && precision_loss(s.x) != 0.0
                            || za != 0 && precision_loss(s.z) != 0.0))
                        || (s.horizontal_collision
                            && (xa != 0 && precision_loss(s.z) != 0.0
                                || za != 0 && precision_loss(s.x) != 0.0));
                    if too_big || full_precision {
                        Move::Sync
                    } else {
                        Move::Pos(xa as i16, ya as i16, za as i16)
                    }
                } else {
                    Move::None
                };
                // The motion packet rides ahead of the move packet.
                let (mvx, mvy, mvz) = (
                    s.vx - state.last_movement.0,
                    s.vy - state.last_movement.1,
                    s.vz - state.last_movement.2,
                );
                let diff = mvx * mvx + mvy * mvy + mvz * mvz;
                let still = s.vx == 0.0 && s.vy == 0.0 && s.vz == 0.0;
                if diff > MOTION_EPS || (diff > 0.0 && still) {
                    state.last_movement = (s.vx, s.vy, s.vz);
                    let body = encode_set_motion(s.id, s.vx, s.vy, s.vz);
                    self.entity_broadcast(s.id, PACKET_SET_ENTITY_MOTION, &body);
                }
                match kind {
                    Move::Sync => {
                        let body =
                            encode_position_sync(s.id, s.x, s.y, s.z, s.yaw, 0.0, s.on_ground);
                        state.base = (s.x, s.y, s.z);
                        self.entity_broadcast(s.id, PACKET_ENTITY_POSITION_SYNC, &body);
                    }
                    Move::Pos(xa, ya, za) => {
                        let body = encode_move_pos(s.id, xa, ya, za, s.on_ground);
                        state.base = (s.x, s.y, s.z);
                        self.entity_broadcast(s.id, PACKET_MOVE_ENTITY_POS, &body);
                    }
                    Move::None => {}
                }
                if s.stack_dirty {
                    let Some(item) = self.survival.items.iter().find(|i| i.id == s.id) else {
                        continue;
                    };
                    let body = encode_item_stack_data(item.id, Some(&item.stack));
                    clear_dirty.push(s.id);
                    self.entity_broadcast(s.id, PACKET_SET_ENTITY_DATA, &body);
                }
                clear_sync.push(s.id);
            }
            state.tick_count += 1;
        }
        self.tracking.states = states;
        self.tracking.last_section = last_section;
        for id in clear_sync {
            if let Some(item) = self.survival.items.iter_mut().find(|i| i.id == id) {
                item.needs_sync = false;
            }
        }
        for id in clear_dirty {
            if let Some(item) = self.survival.items.iter_mut().find(|i| i.id == id) {
                item.stack_dirty = false;
            }
        }
    }
}
