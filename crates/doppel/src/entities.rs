//! Item entities (drops) and random ticks: the survival loop's ground
//! half. Break and hand drops spawn here, fall, merge, and vacuum into
//! player inventories; the per-chunk random tick pass grows and kills
//! grass. The wire shapes come from the 26.3 registration order (the
//! packet constants); the behavior constants track the reference's item
//! entity.

use std::collections::BTreeSet;

use doppel_protocol::write_varint;
use doppel_world::chunk_codec::Container;

use crate::game::{get_section_cell, ConnId, Game};
use crate::inventory::{encode_item_stack, item_id, ItemStack};

// ---------------------------------------------------------------------
// Wire packet ids (clientbound play state)
// ---------------------------------------------------------------------

/// `add_entity`: registration order 2, with the bundle delimiter holding
/// id 0 (template position minus one, the same numbering as the pinned
/// 0x05/0x12/0x7c anchors).
pub const PACKET_ADD_ENTITY: i32 = 0x01;
/// `entity_position_sync`: registration order 36.
pub const PACKET_ENTITY_POSITION_SYNC: i32 = 0x23;
/// `move_entity_pos` / `move_entity_pos_rot`: orders 55/56. Drops send
/// the pos form (they never rotate after spawn); the pos+rot id stays
/// pinned for the shared shape.
pub const PACKET_MOVE_ENTITY_POS: i32 = 0x36;
#[allow(dead_code)]
pub const PACKET_MOVE_ENTITY_POS_ROT: i32 = 0x37;
/// `remove_entities`: registration order 79.
pub const PACKET_REMOVE_ENTITIES: i32 = 0x4e;
/// `set_entity_data`: registration order 102.
pub const PACKET_SET_ENTITY_DATA: i32 = 0x65;
/// `set_entity_motion`: registration order 104.
pub const PACKET_SET_ENTITY_MOTION: i32 = 0x67;
/// `take_item_entity`: registration order 128.
pub const PACKET_TAKE_ITEM_ENTITY: i32 = 0x7f;
/// `teleport_entity`: registration order 129. The reference's item
/// full-sync path rides entity_position_sync instead; the id stays pinned
/// against the registration order.
#[allow(dead_code)]
pub const PACKET_TELEPORT_ENTITY: i32 = 0x80;

/// `minecraft:item` in the entity-type registry (registration order 73,
/// 0-based).
pub const ENTITY_TYPE_ITEM: i32 = 72;

/// The entity-data serializer id for item stacks.
const ITEM_STACK_SERIALIZER: i32 = 7;
/// The entity-data accessor an item entity's stack binds to (the base
/// entity owns accessors 0..8).
const DATA_ITEM: u8 = 8;

// ---------------------------------------------------------------------
// Behavior constants
// ---------------------------------------------------------------------

/// Gravity per tick.
const GRAVITY: f64 = 0.04;
/// Drag applied to every axis after a move.
const AIR_DRAG: f64 = 0.98;
/// Ground friction of the block families the flat world offers.
const FRICTION_DEFAULT: f64 = 0.6;
const FRICTION_ICE: f64 = 0.98;
const FRICTION_SLIME: f64 = 0.8;
/// Water drag on the horizontal axes while submerged.
const WATER_DRAG: f64 = 0.99;
/// Lava drag.
const LAVA_DRAG: f64 = 0.95;
/// Buoyancy added while the vertical motion stays under this speed.
const BUOYANCY: f64 = 5.0e-4;
const BUOYANCY_MAX_VY: f64 = 0.06;
/// Pickup delay for block drops.
const PICKUP_DELAY_BREAK: i32 = 10;
/// Pickup delay for player-thrown drops.
const PICKUP_DELAY_THROW: i32 = 40;
/// Lifetime before a drop despawns, in ticks.
const LIFETIME: i32 = 6000;
/// Hitbox half width and height (0.25 x 0.25).
const HALF_WIDTH: f64 = 0.125;
const HEIGHT: f64 = 0.25;
/// The pickup test: the player's box inflated by (1.0, 0.5, 1.0) against
/// the drop's box. 1.425 = inflate 1.0 + player half width 0.3 + drop
/// half width 0.125.
const PICKUP_INFLATE_XZ: f64 = 1.425;
const PICKUP_DOWN: f64 = 0.5;
const PICKUP_UP: f64 = 1.8 + 0.5;
/// The default stack cap merges honor.
const MAX_MERGE: i32 = 64;
/// Random ticks per randomly-ticking section per tick (the
/// randomTickSpeed gamerule's default).
const DEFAULT_TICK_SPEED: usize = 3;
/// Spread attempts a healthy grass block gets per random tick.
const SPREAD_ATTEMPTS: usize = 4;

// ---------------------------------------------------------------------
// Wire encoders
// ---------------------------------------------------------------------

/// The packed movement vector (`LpVec3`): a zero byte for a zero vector,
/// else 6 bytes of marker bits plus chessboard-quantized components, with
/// a VarInt continuation when the scale exceeds its 2 marker bits.
pub fn encode_lp_movement(buf: &mut Vec<u8>, x: f64, y: f64, z: f64) {
    let sanitize = |v: f64| {
        if v.is_nan() {
            0.0
        } else {
            v.clamp(-1.7179869184e10, 1.7179869184e10)
        }
    };
    let (x, y, z) = (sanitize(x), sanitize(y), sanitize(z));
    let chessboard = x.abs().max(y.abs().max(z.abs()));
    if chessboard < 3.051944088384301e-5 {
        buf.push(0);
        return;
    }
    let scale = chessboard.ceil() as i64;
    let partial = (scale & 3) != scale;
    let markers = if partial { scale & 3 | 4 } else { scale };
    let pack = |v: f64| (((v / scale as f64) * 0.5 + 0.5) * 32766.0).round() as i64;
    let buffer = markers | (pack(x) << 3) | (pack(y) << 18) | (pack(z) << 33);
    buf.push(buffer as u8);
    buf.push((buffer >> 8) as u8);
    buf.extend_from_slice(&((buffer >> 16) as u32).to_be_bytes());
    if partial {
        write_varint(buf, (scale >> 2) as i32);
    }
}

/// Degrees packed to the wire byte: `deg * 256 / 360`, truncated and
/// wrapped through the low 8 bits (180 degrees lands on 128, not 127).
fn pack_degrees(deg: f32) -> u8 {
    ((deg * 256.0 / 360.0) as i64 & 0xff) as u8
}

/// `add_entity`: id, uuid, type, position, movement, rotations, data.
#[allow(clippy::too_many_arguments)]
pub fn encode_add_entity(
    id: i32,
    uuid: &[u8; 16],
    entity_type: i32,
    x: f64,
    y: f64,
    z: f64,
    movement: (f64, f64, f64),
    yaw: f32,
    pitch: f32,
    head_yaw: f32,
    data: i32,
) -> Vec<u8> {
    let mut body = Vec::with_capacity(50);
    write_varint(&mut body, id);
    body.extend_from_slice(uuid);
    write_varint(&mut body, entity_type);
    body.extend_from_slice(&x.to_be_bytes());
    body.extend_from_slice(&y.to_be_bytes());
    body.extend_from_slice(&z.to_be_bytes());
    encode_lp_movement(&mut body, movement.0, movement.1, movement.2);
    body.push(pack_degrees(pitch));
    body.push(pack_degrees(yaw));
    body.push(pack_degrees(head_yaw));
    write_varint(&mut body, data);
    body
}

/// `set_entity_data` for one item stack: the accessor/serializer header,
/// the stack payload, the list terminator.
pub fn encode_item_stack_data(entity_id: i32, stack: Option<&ItemStack>) -> Vec<u8> {
    let mut body = Vec::with_capacity(12);
    write_varint(&mut body, entity_id);
    body.push(DATA_ITEM);
    write_varint(&mut body, ITEM_STACK_SERIALIZER);
    encode_item_stack(&mut body, stack);
    body.push(0xff);
    body
}

/// `remove_entities`: one count plus the ids.
pub fn encode_remove_entities(ids: &[i32]) -> Vec<u8> {
    let mut body = Vec::with_capacity(4 + ids.len());
    write_varint(&mut body, ids.len() as i32);
    for id in ids {
        write_varint(&mut body, *id);
    }
    body
}

/// `take_item_entity`: the drop, the collector, the amount.
pub fn encode_take_item(item_id: i32, player_id: i32, amount: i32) -> Vec<u8> {
    let mut body = Vec::with_capacity(9);
    write_varint(&mut body, item_id);
    write_varint(&mut body, player_id);
    write_varint(&mut body, amount);
    body
}

/// `move_entity_pos`: id, the properties varint (on-ground bit, step
/// count 0), then the linear delta in 1/4096-block units.
pub fn encode_move_pos(entity_id: i32, xa: i16, ya: i16, za: i16, on_ground: bool) -> Vec<u8> {
    let mut body = Vec::with_capacity(10);
    write_varint(&mut body, entity_id);
    write_varint(&mut body, i32::from(on_ground));
    body.extend_from_slice(&xa.to_be_bytes());
    body.extend_from_slice(&ya.to_be_bytes());
    body.extend_from_slice(&za.to_be_bytes());
    body
}

/// `set_entity_motion`: id plus the packed movement vector.
pub fn encode_set_motion(entity_id: i32, vx: f64, vy: f64, vz: f64) -> Vec<u8> {
    let mut body = Vec::with_capacity(9);
    write_varint(&mut body, entity_id);
    encode_lp_movement(&mut body, vx, vy, vz);
    body
}

/// `entity_position_sync` with a linear path: id, path kind 0, position,
/// rotations, on-ground flag.
pub fn encode_position_sync(
    entity_id: i32,
    x: f64,
    y: f64,
    z: f64,
    yaw: f32,
    pitch: f32,
    on_ground: bool,
) -> Vec<u8> {
    let mut body = Vec::with_capacity(40);
    write_varint(&mut body, entity_id);
    write_varint(&mut body, 0);
    body.extend_from_slice(&x.to_be_bytes());
    body.extend_from_slice(&y.to_be_bytes());
    body.extend_from_slice(&z.to_be_bytes());
    body.extend_from_slice(&yaw.to_be_bytes());
    body.extend_from_slice(&pitch.to_be_bytes());
    body.push(u8::from(on_ground));
    body
}

// ---------------------------------------------------------------------
// Drop state
// ---------------------------------------------------------------------

/// One drop on the ground.
pub struct ItemEntity {
    pub(crate) id: i32,
    pub(crate) uuid: [u8; 16],
    /// Feet position (the hitbox's bottom center).
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) z: f64,
    pub(crate) vx: f64,
    pub(crate) vy: f64,
    pub(crate) vz: f64,
    /// The random facing the spawn draws (the item sprite's rotation).
    pub(crate) yaw: f32,
    pub(crate) stack: ItemStack,
    /// Ticks lived; despawns at 6000.
    pub(crate) age: i32,
    /// Ticks before a player can vacuum it.
    pub(crate) pickup_delay: i32,
    /// The thrower, for drop attribution.
    #[allow(dead_code)]
    owner: Option<ConnId>,
    pub(crate) on_ground: bool,
    /// Ticks of this entity's own pass (the entity tick counter, one
    /// behind at spawn: the first pass sees 1).
    pub(crate) tick_count: i32,
    /// The game tick the drop spawned in; a pass over the same tick
    /// skips it (entities added mid-tick do not tick that tick).
    pub(crate) born_tick: u64,
    /// Whether the last move clipped vertically / horizontally.
    pub(crate) vertical_collision: bool,
    pub(crate) horizontal_collision: bool,
    /// Whether the next tracker flush must consider this entity.
    pub(crate) needs_sync: bool,
    /// Whether the stack changed since its last entity-data sync.
    pub(crate) stack_dirty: bool,
}

impl ItemEntity {
    /// The merge gate: alive, under the lifetime cap, stack below the max.
    fn mergable(&self) -> bool {
        self.age < LIFETIME && self.stack.count() < MAX_MERGE
    }

    fn new(id: i32, uuid: [u8; 16], x: f64, y: f64, z: f64, stack: ItemStack) -> ItemEntity {
        ItemEntity {
            id,
            uuid,
            x,
            y,
            z,
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
            yaw: 0.0,
            stack,
            age: 0,
            pickup_delay: PICKUP_DELAY_BREAK,
            owner: None,
            on_ground: false,
            tick_count: 0,
            born_tick: 0,
            vertical_collision: false,
            horizontal_collision: false,
            needs_sync: false,
            stack_dirty: false,
        }
    }
}

/// Survival state the game thread owns: live drops plus the random-tick
/// bookkeeping.
pub(crate) struct SurvivalState {
    pub(crate) items: Vec<ItemEntity>,
    /// Seed for the drop uuid splitmix stream.
    uuid_seed: u64,
    /// Seed for the spread-target picks.
    spread_seed: u64,
    /// Picks per randomly ticking section per tick (randomTickSpeed).
    tick_speed: usize,
    /// The per-level random tick LCG (`randValue`).
    rand_value: i32,
    /// Sections holding a randomly ticking block; `unknown` holds the
    /// ones already scanned.
    ticking: BTreeSet<(i32, i32, i32)>,
    unknown: BTreeSet<(i32, i32, i32)>,
}

impl Default for SurvivalState {
    fn default() -> Self {
        SurvivalState {
            items: Vec::new(),
            uuid_seed: 0,
            spread_seed: 0,
            tick_speed: DEFAULT_TICK_SPEED,
            rand_value: 0,
            ticking: BTreeSet::new(),
            unknown: BTreeSet::new(),
        }
    }
}

impl SurvivalState {
    fn splitmix(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// A random f64 in [0, 1).
    fn next_unit(&mut self) -> f64 {
        (Self::splitmix(&mut self.spread_seed) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A random f32 in [0, 1) with the 24-bit mantissa of a float draw.
    fn next_f32(&mut self) -> f32 {
        (Self::splitmix(&mut self.spread_seed) >> 40) as f32 / (1u32 << 24) as f32
    }

    /// The float draw against an externally held seed (the physics pass
    /// borrows the game immutably).
    fn draw_f32(seed: &mut u64) -> f32 {
        (Self::splitmix(seed) >> 40) as f32 / (1u32 << 24) as f32
    }

    /// A random f64 in [lo, hi).
    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_unit()
    }

    fn next_uuid(&mut self) -> [u8; 16] {
        let a = Self::splitmix(&mut self.uuid_seed);
        let b = Self::splitmix(&mut self.uuid_seed);
        let mut uuid = [0u8; 16];
        uuid[..8].copy_from_slice(&a.to_be_bytes());
        uuid[8..].copy_from_slice(&b.to_be_bytes());
        // Version 4, variant 1.
        uuid[6] = (uuid[6] & 0x0f) | 0x40;
        uuid[8] = (uuid[8] & 0x3f) | 0x80;
        uuid
    }
}

// ---------------------------------------------------------------------
// Block predicates
// ---------------------------------------------------------------------

/// Blocks a falling drop passes through (no collision shape).
fn passable(name: &str) -> bool {
    matches!(
        name,
        "minecraft:air"
            | "minecraft:water"
            | "minecraft:lava"
            | "minecraft:torch"
            | "minecraft:wall_torch"
            | "minecraft:redstone_torch"
            | "minecraft:redstone_wall_torch"
            | "minecraft:redstone_wire"
            | "minecraft:lever"
            | "minecraft:short_grass"
            | "minecraft:tall_grass"
            | "minecraft:short_dry_grass"
            | "minecraft:tall_dry_grass"
            | "minecraft:snow"
    ) || name.contains("sapling")
        || name.contains("flower")
        || name.contains("propagule")
}

/// Blocks that fully dampen light: the grass-killer test. Glass and the
/// attachable decoration pass light; the solid families block it.
/// Fluids pass (the light engine is future work).
fn light_impermeable(name: &str) -> bool {
    !passable(name) && name != "minecraft:glass"
}

/// A solid cell for drop collision. Unloaded reads count as solid.
fn block_solid(g: &Game, x: i32, y: i32, z: i32) -> bool {
    match g.get_block(x, y, z) {
        None => true,
        Some((n, _)) => !passable(&n),
    }
}

/// The ground friction of the block a drop rests on.
fn block_friction(name: &str) -> f64 {
    match name {
        "minecraft:ice" | "minecraft:packed_ice" | "minecraft:blue_ice" => FRICTION_ICE,
        "minecraft:slime_block" => FRICTION_SLIME,
        _ => FRICTION_DEFAULT,
    }
}

// ---------------------------------------------------------------------
// Drop table
// ---------------------------------------------------------------------

/// The item a broken block drops, as a registry id. None for the no-loot
/// families; unlisted blocks drop themselves when the curated item table
/// knows the name.
pub fn block_drop_item(name: &str) -> Option<i32> {
    match name {
        "minecraft:air" | "minecraft:glass" | "minecraft:bedrock" | "minecraft:barrier" => None,
        "minecraft:stone" => item_id("minecraft:cobblestone"),
        "minecraft:grass_block" => item_id("minecraft:dirt"),
        "minecraft:wall_torch" => item_id("minecraft:torch"),
        "minecraft:redstone_wall_torch" => item_id("minecraft:redstone_torch"),
        "minecraft:redstone_wire" => item_id("minecraft:redstone"),
        other => item_id(other),
    }
}

// ---------------------------------------------------------------------
// Physics core (borrow-separated from the Game pass)
// ---------------------------------------------------------------------

/// The movable state of one drop for the physics step.
#[derive(Clone, Copy)]
struct Motion {
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    vz: f64,
    on_ground: bool,
    vertical_collision: bool,
    horizontal_collision: bool,
    needs_sync: bool,
}

/// An axis-aligned box: min/max corners.
#[derive(Clone, Copy)]
struct Box3 {
    min: [f64; 3],
    max: [f64; 3],
}

impl Box3 {
    fn of(x: f64, y: f64, z: f64) -> Box3 {
        Box3 {
            min: [x - HALF_WIDTH, y, z - HALF_WIDTH],
            max: [x + HALF_WIDTH, y + HEIGHT, z + HALF_WIDTH],
        }
    }

    fn shift(&mut self, axis: usize, d: f64) {
        self.min[axis] += d;
        self.max[axis] += d;
    }

    /// Whether any solid cell overlaps the box shrunk by epsilon.
    fn intersects_solid(&self, g: &Game) -> bool {
        let cells = |axis: usize| {
            let lo = (self.min[axis] + 1.0e-7).floor() as i32;
            let hi = (self.max[axis] - 1.0e-7).floor() as i32;
            lo..=hi
        };
        for x in cells(0) {
            for y in cells(1) {
                for z in cells(2) {
                    if block_solid(g, x, y, z) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// The per-axis clip against full-cube cells: the movement distance after
/// the first blocking cell along the axis (scanning from the box face),
/// clamped to the requested distance. Mirrors the sweep's epsilon pair
/// (1e-7 shrink on the cross axes, 1e-7 slack on the blocking face).
fn clip_axis(g: &Game, b: &Box3, axis: usize, distance: f64) -> f64 {
    if distance.abs() < 1.0e-7 {
        return 0.0;
    }
    let (b1, b2) = ((axis + 1) % 3, (axis + 2) % 3);
    let cross = |lo: f64, hi: f64| ((lo + 1.0e-7).floor() as i32)..=(hi - 1.0e-7).floor() as i32;
    let c1 = cross(b.min[b1], b.max[b1]);
    let c2 = cross(b.min[b2], b.max[b2]);
    let solid = |a: i32, c1: i32, c2: i32| -> bool {
        let (x, y, z) = match axis {
            0 => (a, c1, c2),
            1 => (c1, a, c2),
            _ => (c1, c2, a),
        };
        block_solid(g, x, y, z)
    };
    let mut distance = distance;
    if distance > 0.0 {
        let face_max = b.max[axis];
        let mut a = (face_max - 1.0e-7).floor() as i32 + 1;
        loop {
            let new_distance = a as f64 - face_max;
            if new_distance > distance {
                return distance;
            }
            let hit = c1.clone().any(|c1| c2.clone().any(|c2| solid(a, c1, c2)));
            if hit {
                if new_distance >= -1.0e-7 {
                    distance = distance.min(new_distance);
                }
                return distance;
            }
            a += 1;
        }
    }
    let face_min = b.min[axis];
    let mut a = (face_min + 1.0e-7).floor() as i32 - 1;
    loop {
        let new_distance = (a + 1) as f64 - face_min;
        if new_distance < distance {
            return distance;
        }
        let hit = c1.clone().any(|c1| c2.clone().any(|c2| solid(a, c1, c2)));
        if hit {
            if new_distance <= 1.0e-7 {
                distance = distance.max(new_distance);
            }
            return distance;
        }
        a -= 1;
    }
}

/// The axis-ordered sweep: vertical first, then the dominant horizontal
/// axis. Returns the clipped delta.
fn collide_move(g: &Game, m: &Motion) -> (f64, f64, f64) {
    let mut b = Box3::of(m.x, m.y, m.z);
    let dy = clip_axis(g, &b, 1, m.vy);
    b.shift(1, dy);
    // |x| < |z| resolves z first (ties resolve x first).
    if m.vx.abs() < m.vz.abs() {
        let dz = clip_axis(g, &b, 2, m.vz);
        b.shift(2, dz);
        let dx = clip_axis(g, &b, 0, m.vx);
        (dx, dy, dz)
    } else {
        let dx = clip_axis(g, &b, 0, m.vx);
        b.shift(0, dx);
        let dz = clip_axis(g, &b, 2, m.vz);
        (dx, dy, dz)
    }
}

/// Whether the drop's box overlaps a fluid cell (source depth stands in
/// for the fluid height).
fn submerged(g: &Game, m: &Motion, fluid: &str) -> bool {
    let b = Box3::of(m.x, m.y, m.z);
    let cells = |axis: usize| b.min[axis].floor() as i32..=b.max[axis].floor() as i32;
    for x in cells(0) {
        for y in cells(1) {
            for z in cells(2) {
                if let Some((n, _)) = g.get_block(x, y, z) {
                    if n == fluid {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// A drop stuck inside a solid cell gets a nudge toward the nearest open
/// neighbor (north, south, west, east, up) with a random speed.
fn move_towards_closest_space(g: &Game, m: &mut Motion, next_f32: &mut dyn FnMut() -> f32) {
    let y_mid = m.y + HEIGHT / 2.0;
    let (bx, by, bz) = (m.x.floor() as i32, y_mid.floor() as i32, m.z.floor() as i32);
    let frac = (m.x - bx as f64, y_mid - by as f64, m.z - bz as f64);
    // (dx, dy, dz, axis, positive direction)
    const PROBES: [(i32, i32, i32, usize, bool); 5] = [
        (0, 0, -1, 2, false), // north
        (0, 0, 1, 2, true),   // south
        (-1, 0, 0, 0, false), // west
        (1, 0, 0, 0, true),   // east
        (0, 1, 0, 1, true),   // up
    ];
    let mut best_axis = 1usize;
    let mut best_positive = true;
    let mut best = f64::MAX;
    for (dx, dy, dz, axis, positive) in PROBES {
        if block_solid(g, bx + dx, by + dy, bz + dz) {
            continue;
        }
        let d = match axis {
            0 => frac.0,
            1 => frac.1,
            _ => frac.2,
        };
        let oriented = if positive { 1.0 - d } else { d };
        if oriented < best {
            best = oriented;
            best_axis = axis;
            best_positive = positive;
        }
    }
    let speed = next_f32() * 0.2 + 0.1;
    m.vx *= 0.75;
    m.vy *= 0.75;
    m.vz *= 0.75;
    let step = if best_positive {
        speed as f64
    } else {
        -(speed as f64)
    };
    match best_axis {
        0 => m.vx = step,
        1 => m.vy = step,
        _ => m.vz = step,
    }
}

/// One tick of the drop's own integration: fluid or gravity, the
/// stuck-in-solid nudge, the rest-gated move with collision zeroing,
/// and the post-move drags.
fn step_motion(
    g: &Game,
    m: &mut Motion,
    tick_count: i32,
    entity_id: i32,
    next_f32: &mut dyn FnMut() -> f32,
) {
    let old = (m.vx, m.vy, m.vz);
    if submerged(g, m, "minecraft:water") {
        m.vx *= WATER_DRAG;
        m.vz *= WATER_DRAG;
        if m.vy < BUOYANCY_MAX_VY {
            m.vy += BUOYANCY;
        }
    } else if submerged(g, m, "minecraft:lava") {
        m.vx *= LAVA_DRAG;
        m.vz *= LAVA_DRAG;
        if m.vy < BUOYANCY_MAX_VY {
            m.vy += BUOYANCY;
        }
    } else {
        m.vy -= GRAVITY;
    }
    if Box3::of(m.x, m.y, m.z).intersects_solid(g) {
        move_towards_closest_space(g, m, next_f32);
    }
    // The rest gate: a grounded, nearly still drop moves only every
    // fourth tick. Skipped ticks skip the drags too; gravity accrues.
    let horizontal = m.vx * m.vx + m.vz * m.vz;
    if !m.on_ground || horizontal > 1.0e-5 || (tick_count + entity_id).rem_euclid(4) == 0 {
        let (dx, dy, dz) = collide_move(g, m);
        m.x += dx;
        m.y += dy;
        m.z += dz;
        let x_collision = (dx - m.vx).abs() >= 1.0e-7;
        let z_collision = (dz - m.vz).abs() >= 1.0e-7;
        m.horizontal_collision = x_collision || z_collision;
        m.vertical_collision = dy != m.vy;
        m.on_ground = m.vertical_collision && m.vy < 0.0;
        // Restitution is zero for drops: collided components stop dead.
        if m.horizontal_collision || (m.vertical_collision && m.vy != 0.0) {
            if x_collision {
                m.vx = 0.0;
            }
            if z_collision {
                m.vz = 0.0;
            }
            if m.vertical_collision {
                m.vy = 0.0;
            }
        }
        let ground_friction = if m.on_ground {
            let below_y = (m.y - 0.999999).floor() as i32;
            let friction = g
                .get_block(m.x.floor() as i32, below_y, m.z.floor() as i32)
                .map(|(n, _)| block_friction(&n))
                .unwrap_or(FRICTION_DEFAULT);
            AIR_DRAG * friction
        } else {
            AIR_DRAG
        };
        m.vx *= ground_friction;
        m.vz *= ground_friction;
        m.vy *= AIR_DRAG;
    }
    let (dvx, dvy, dvz) = (m.vx - old.0, m.vy - old.1, m.vz - old.2);
    if dvx * dvx + dvy * dvy + dvz * dvz > 0.01 {
        m.needs_sync = true;
    }
}

// ---------------------------------------------------------------------
// Game hooks
// ---------------------------------------------------------------------

impl Game {
    // --- survival hooks (entities.rs) ---

    /// Spawns a dropped item from a broken block: the centered pop - the
    /// level random draws the in-cell position (x, y, z), the entity
    /// random draws the bob phase, the facing, and the pop velocity - on
    /// the 10-tick pickup delay. Bare-hand breaks of tool-required blocks
    /// drop nothing.
    pub(crate) fn spawn_break_drop(&mut self, pos: (i32, i32, i32), name: &str) {
        let (_, requires_tool) = crate::dig::hardness(name);
        if requires_tool {
            return;
        }
        let Some(item) = block_drop_item(name) else {
            return;
        };
        let x = pos.0 as f64 + 0.5 + self.survival.next_range(-0.25, 0.25);
        let y = pos.1 as f64 + 0.5 + self.survival.next_range(-0.25, 0.25) - HEIGHT / 2.0;
        let z = pos.2 as f64 + 0.5 + self.survival.next_range(-0.25, 0.25);
        // The constructor draws: the bob phase (client cosmetic), the
        // facing, then the pop velocity's two draws.
        let _bob = self.survival.next_f32();
        let yaw = self.survival.next_f32() * 360.0;
        let vx = self.survival.next_unit() * 0.2 - 0.1;
        let vz = self.survival.next_unit() * 0.2 - 0.1;
        self.spawn_item(x, y, z, (vx, 0.2, vz), yaw, ItemStack::new(item, 1), None);
    }

    /// Spawns a player-thrown drop: at eye height minus 0.3, on the
    /// 40-tick delay, flying along the look direction. The scatter
    /// arithmetic runs in f32 (the reference's float pipeline), the
    /// look products widen to f64 exactly where the reference widens.
    pub(crate) fn spawn_thrown_drop(&mut self, conn: ConnId, stack: ItemStack) {
        let Some(p) = self.players.get(&conn) else {
            return;
        };
        let (yaw, pitch) = (p.yaw, p.pitch);
        let (x, y, z) = (p.x, p.y + 1.62 - 0.3, p.z);
        let deg = std::f32::consts::PI / 180.0;
        let sin_x = (pitch * deg).sin();
        let cos_x = (pitch * deg).cos();
        let sin_y = (yaw * deg).sin();
        let cos_y = (yaw * deg).cos();
        // The constructor draws first (bob, facing, a pop velocity the
        // throw replaces), then the throw's scatter draws.
        let _bob = self.survival.next_f32();
        let facing = self.survival.next_f32() * 360.0;
        let _pop_x = self.survival.next_unit();
        let _pop_z = self.survival.next_unit();
        let dir = self.survival.next_f32() * std::f32::consts::TAU;
        let pow = 0.02f32 * self.survival.next_f32();
        let (f1, f2) = (self.survival.next_f32(), self.survival.next_f32());
        let vx = (-sin_y * cos_x * 0.3f32) as f64 + (dir as f64).cos() * pow as f64;
        let vy = (-sin_x * 0.3f32 + 0.1f32 + (f1 - f2) * 0.1f32) as f64;
        let vz = (cos_y * cos_x * 0.3f32) as f64 + (dir as f64).sin() * pow as f64;
        self.spawn_item(x, y, z, (vx, vy, vz), facing, stack, Some(conn));
    }

    /// The shared spawn: allocate the id and uuid, then pair the entity
    /// with every in-range player.
    fn spawn_item(
        &mut self,
        x: f64,
        y: f64,
        z: f64,
        velocity: (f64, f64, f64),
        yaw: f32,
        stack: ItemStack,
        owner: Option<ConnId>,
    ) {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        let delay = if owner.is_some() {
            PICKUP_DELAY_THROW
        } else {
            PICKUP_DELAY_BREAK
        };
        let uuid = self.survival.next_uuid();
        let mut item = ItemEntity::new(id, uuid, x, y, z, stack);
        item.vx = velocity.0;
        item.vy = velocity.1;
        item.vz = velocity.2;
        item.yaw = yaw;
        item.pickup_delay = delay;
        item.owner = owner;
        item.born_tick = self.tick;
        // The stack accessor spawns dirty: the pairing carries its
        // snapshot, and the first sync pass re-sends it (the flush the
        // reference's data-dirty flag performs after every spawn).
        item.stack_dirty = true;
        self.survival.items.push(item);
        self.track_entity_spawned(id);
    }

    /// The entity pass, after the block-event phase: the pickup vacuum
    /// first (players tick ahead of items), then each drop's physics,
    /// merge, and lifetime.
    pub(crate) fn tick_entities(&mut self) {
        self.entity_pickup();
        self.entity_item_pass();
    }

    /// One pass over the drops: physics, the merge cadence, aging, and
    /// the age-out discard, interleaved per entity (the reference merges
    /// inside each entity's own tick). Entities spawned during this same
    /// game tick wait for the next one.
    fn entity_item_pass(&mut self) {
        let tick = self.tick;
        let mut gone: Vec<i32> = Vec::new();
        let mut dead: Vec<usize> = Vec::new();
        let mut i = 0usize;
        while i < self.survival.items.len() {
            if dead.contains(&i) {
                i += 1;
                continue;
            }
            let fresh = self.survival.items[i].born_tick == tick;
            // An empty stack discards without ticking.
            if !fresh && self.survival.items[i].stack.is_empty() {
                gone.push(self.survival.items[i].id);
                dead.push(i);
                i += 1;
                continue;
            }
            if !fresh {
                let (id, mut motion, xo, yo, zo, tick_count) = {
                    let item = &mut self.survival.items[i];
                    item.tick_count += 1;
                    if item.pickup_delay > 0 {
                        item.pickup_delay -= 1;
                    }
                    (
                        item.id,
                        Motion {
                            x: item.x,
                            y: item.y,
                            z: item.z,
                            vx: item.vx,
                            vy: item.vy,
                            vz: item.vz,
                            on_ground: item.on_ground,
                            vertical_collision: item.vertical_collision,
                            horizontal_collision: item.horizontal_collision,
                            needs_sync: item.needs_sync,
                        },
                        item.x,
                        item.y,
                        item.z,
                        item.tick_count,
                    )
                };
                let mut rng_seed = self.survival.spread_seed;
                let mut draw = || SurvivalState::draw_f32(&mut rng_seed);
                step_motion(self, &mut motion, tick_count, id, &mut draw);
                self.survival.spread_seed = rng_seed;
                {
                    let item = &mut self.survival.items[i];
                    item.x = motion.x;
                    item.y = motion.y;
                    item.z = motion.z;
                    item.vx = motion.vx;
                    item.vy = motion.vy;
                    item.vz = motion.vz;
                    item.on_ground = motion.on_ground;
                    item.vertical_collision = motion.vertical_collision;
                    item.horizontal_collision = motion.horizontal_collision;
                    item.needs_sync |= motion.needs_sync;
                }
                // The merge cadence: every 2 ticks while crossing cells,
                // else every 40.
                let moved = (
                    motion.x.floor() as i32,
                    motion.y.floor() as i32,
                    motion.z.floor() as i32,
                ) != (xo.floor() as i32, yo.floor() as i32, zo.floor() as i32);
                let rate = if moved { 2 } else { 40 };
                if tick_count % rate == 0 && self.survival.items[i].mergable() {
                    let absorbed_here = self.merge_from(i, &mut dead);
                    if absorbed_here {
                        gone.push(id);
                        dead.push(i);
                        // The absorbed entity finishes its tick: age only.
                        self.survival.items[i].age += 1;
                        i += 1;
                        continue;
                    }
                }
                self.survival.items[i].age += 1;
                if self.survival.items[i].age >= LIFETIME {
                    gone.push(id);
                    dead.push(i);
                }
            }
            i += 1;
        }
        if !dead.is_empty() {
            let mut kept: Vec<ItemEntity> = Vec::new();
            let drained = std::mem::take(&mut self.survival.items);
            for (idx, item) in drained.into_iter().enumerate() {
                if !dead.contains(&idx) {
                    kept.push(item);
                }
            }
            self.survival.items = kept;
        }
        for id in gone {
            self.track_entity_removed(id);
        }
    }

    /// The merge attempt of item `i` against the live others: the 0.5
    /// box inflation, the larger stack absorbing (ties go to the other).
    /// Returns true when `i` itself was absorbed.
    fn merge_from(&mut self, i: usize, dead: &mut Vec<usize>) -> bool {
        let (x, y, z) = {
            let item = &self.survival.items[i];
            (item.x, item.y, item.z)
        };
        let mut j = 0usize;
        while j < self.survival.items.len() {
            if j == i || dead.contains(&j) {
                j += 1;
                continue;
            }
            let near = {
                let other = &self.survival.items[j];
                (other.x - x).abs() < 0.75
                    && (other.y - y).abs() < 0.25
                    && (other.z - z).abs() < 0.75
                    && other.mergable()
                    && ItemStack::same_item_same_components(
                        &other.stack,
                        &self.survival.items[i].stack,
                    )
                    && other.stack.count() + self.survival.items[i].stack.count() <= MAX_MERGE
            };
            if !near {
                j += 1;
                continue;
            }
            // The strictly smaller stack folds; ties fold this one.
            let (victim, keeper) =
                if self.survival.items[i].stack.count() > self.survival.items[j].stack.count() {
                    (j, i)
                } else {
                    (i, j)
                };
            let (vcount, vage, vdelay) = {
                let v = &self.survival.items[victim];
                (v.stack.count(), v.age, v.pickup_delay)
            };
            {
                let keeper = &mut self.survival.items[keeper];
                keeper.stack.set_count(keeper.stack.count() + vcount);
                keeper.age = keeper.age.min(vage);
                keeper.pickup_delay = keeper.pickup_delay.max(vdelay);
                keeper.stack_dirty = true;
            }
            dead.push(victim);
            if victim == i {
                return true;
            }
            j += 1;
        }
        false
    }

    /// The vacuum, from the player pass: overlapping players absorb drops
    /// whose pickup delay expired. Full takes discard the entity and
    /// broadcast the take animation (the reference sends it with the
    /// pre-pickup count); a full inventory leaves the drop alone.
    fn entity_pickup(&mut self) {
        let tick = self.tick;
        let mut taken: Vec<usize> = Vec::new();
        let mut i = 0usize;
        while i < self.survival.items.len() {
            if taken.contains(&i) || self.survival.items[i].born_tick == tick {
                i += 1;
                continue;
            }
            let (x, y, z, ready, item_entity_id, original_count, stack) = {
                let item = &self.survival.items[i];
                (
                    item.x,
                    item.y,
                    item.z,
                    item.pickup_delay == 0,
                    item.id,
                    item.stack.count(),
                    item.stack.clone(),
                )
            };
            if !ready {
                i += 1;
                continue;
            }
            let conns: Vec<(ConnId, i32)> = self
                .players
                .iter()
                .filter(|(_, p)| {
                    (p.x - x).abs() < PICKUP_INFLATE_XZ
                        && (p.z - z).abs() < PICKUP_INFLATE_XZ
                        && y < p.y + PICKUP_UP
                        && y + HEIGHT > p.y - PICKUP_DOWN
                })
                .map(|(&conn, p)| (conn, p.entity_id))
                .collect();
            for (conn, player_id) in conns {
                let before: Vec<Option<ItemStack>> = {
                    let Some(p) = self.players.get(&conn) else {
                        continue;
                    };
                    (0..46).map(|s| p.inv.inventory.get(s)).collect()
                };
                let leftover = {
                    let Some(p) = self.players.get_mut(&conn) else {
                        continue;
                    };
                    p.inv.inventory.add(stack.clone())
                };
                let moved_in = original_count - leftover.as_ref().map(|s| s.count()).unwrap_or(0);
                if moved_in <= 0 {
                    continue;
                }
                let take = encode_take_item(item_entity_id, player_id, original_count);
                self.entity_broadcast(item_entity_id, PACKET_TAKE_ITEM_ENTITY, &take);
                // The changed slots ride this tick's menu broadcast.
                if let Some(p) = self.players.get_mut(&conn) {
                    for (slot, was) in before.iter().enumerate() {
                        if p.inv.inventory.get(slot) != *was {
                            p.inv.pending_sync.insert(slot);
                        }
                    }
                }
                match leftover {
                    None => {
                        self.track_entity_removed(item_entity_id);
                        taken.push(i);
                        break;
                    }
                    Some(rest) => {
                        let item = &mut self.survival.items[i];
                        item.stack = rest;
                        item.stack_dirty = true;
                    }
                }
            }
            i += 1;
        }
        if !taken.is_empty() {
            let mut kept: Vec<ItemEntity> = Vec::new();
            let drained = std::mem::take(&mut self.survival.items);
            for (idx, item) in drained.into_iter().enumerate() {
                if !taken.contains(&idx) {
                    kept.push(item);
                }
            }
            self.survival.items = kept;
        }
    }

    // --- survival hooks (entities.rs): random ticks ---

    /// The per-chunk random tick pass: 3 picks per randomly ticking
    /// section, grass decay under light-blocking cover, grass spread onto
    /// bare dirt.
    pub(crate) fn random_ticks(&mut self) {
        let mut decay: Vec<(i32, i32, i32)> = Vec::new();
        let mut spread: Vec<(i32, i32, i32)> = Vec::new();
        let chunks: Vec<(i32, i32)> = self
            .viewers
            .iter()
            .filter(|(_, viewers)| !viewers.is_empty())
            .map(|(&c, _)| c)
            .collect();
        for (cx, cz) in chunks {
            for sy in 0..24i32 {
                if !self.section_randomly_ticking(cx, cz, sy) {
                    continue;
                }
                let base_y = (sy - 4) * 16;
                for _ in 0..self.survival.tick_speed {
                    // The per-level LCG; chunk iteration order is this
                    // build's own (the reference shares the stream across
                    // its tick-chunk order).
                    self.survival.rand_value = self
                        .survival
                        .rand_value
                        .wrapping_mul(3)
                        .wrapping_add(1013904223);
                    let val = self.survival.rand_value >> 2;
                    let x = cx * 16 + (val & 0xf);
                    let y = base_y + ((val >> 16) & 0xf);
                    let z = cz * 16 + ((val >> 8) & 0xf);
                    let Some((name, _)) = self.get_block(x, y, z) else {
                        continue;
                    };
                    if name != "minecraft:grass_block" {
                        continue;
                    }
                    // Decay: full light dampening directly above.
                    let capped = self
                        .get_block(x, y + 1, z)
                        .is_some_and(|(n, _)| light_impermeable(&n));
                    if capped {
                        decay.push((x, y, z));
                        continue;
                    }
                    // Spread: a bright-enough source seeds four nearby
                    // cells; sky exposure stands in for brightness.
                    if !self.sky_exposed(x, y + 1, z) {
                        continue;
                    }
                    for _ in 0..SPREAD_ATTEMPTS {
                        let dx = (self.survival.next_unit() * 3.0) as i32 - 1;
                        let dy = (self.survival.next_unit() * 5.0) as i32 - 3;
                        let dz = (self.survival.next_unit() * 3.0) as i32 - 1;
                        let (tx, ty, tz) = (x + dx, y + dy, z + dz);
                        if !self
                            .get_block(tx, ty, tz)
                            .is_some_and(|(n, _)| n == "minecraft:dirt")
                        {
                            continue;
                        }
                        let clear = self
                            .get_block(tx, ty + 1, tz)
                            .is_some_and(|(n, _)| !light_impermeable(&n));
                        if clear {
                            spread.push((tx, ty, tz));
                        }
                    }
                }
            }
        }
        for (x, y, z) in decay {
            if let Some(state) = self.resolve_state("minecraft:dirt") {
                self.set_block(x, y, z, state, true);
            }
        }
        for (x, y, z) in spread {
            // The default grass state: the registry lists snowy=true first,
            // but a spread plant is snowy=false.
            if let Some(state) = self.resolve_state("minecraft:grass_block[snowy=false]") {
                self.set_block(x, y, z, state, true);
            }
        }
    }

    /// Whether a section holds a randomly ticking block (grass, this
    /// build's only one). Cached; a write into the section re-arms the
    /// scan.
    fn section_randomly_ticking(&mut self, cx: i32, cz: i32, sy: i32) -> bool {
        let key = (cx, cz, sy);
        if self.survival.ticking.contains(&key) {
            return true;
        }
        if self.survival.unknown.contains(&key) {
            return false;
        }
        let mut found = false;
        if let Some(chunk) = self.chunks.get(&(cx, cz)) {
            if let Some(section) = chunk.wire.sections.get(sy as usize) {
                if section.non_empty > 0 {
                    let is_grass = |state: u32| {
                        self.registry.as_ref().is_some_and(|r| {
                            r.state_of(state)
                                .is_some_and(|(n, _)| n == "minecraft:grass_block")
                        })
                    };
                    found = match &section.block_states {
                        Container::Single(state) => is_grass(*state),
                        // A palette entry marks the section ticking even
                        // if that state no longer sits in storage: picks
                        // on non-grass cells no-op, so the miss side is
                        // only cost.
                        Container::Palette { entries, .. } => entries.iter().any(|&s| is_grass(s)),
                        Container::Global { .. } => (0..4096usize).any(|idx| {
                            get_section_cell(&chunk.wire, sy as usize, idx).is_some_and(is_grass)
                        }),
                    };
                }
            }
        }
        if found {
            self.survival.ticking.insert(key);
        }
        self.survival.unknown.insert(key);
        found
    }

    /// Invalidation hook: a write into a section re-arms its tick scan.
    pub(crate) fn invalidate_section_ticks(&mut self, cx: i32, cz: i32, sy: i32) {
        self.survival.ticking.remove(&(cx, cz, sy));
        self.survival.unknown.remove(&(cx, cz, sy));
    }

    /// `gamerule random_tick_speed N`: picks per randomly ticking section.
    pub(crate) fn set_tick_speed(&mut self, speed: usize) {
        self.survival.tick_speed = speed;
    }

    /// Sky exposure standing in for the brightness check: no
    /// light-blocking block in the column above. Exact under an open sky;
    /// the light engine is future work.
    fn sky_exposed(&self, x: i32, y: i32, z: i32) -> bool {
        let cx = x.div_euclid(16);
        let cz = z.div_euclid(16);
        let Some(chunk) = self.chunks.get(&(cx, cz)) else {
            return true;
        };
        let lx = (x - cx * 16) as usize;
        let lz = (z - cz * 16) as usize;
        let opaque = |state: u32| {
            self.registry
                .as_ref()
                .and_then(|r| r.state_of(state))
                .is_some_and(|(n, _)| light_impermeable(n))
        };
        for (si, section) in chunk.wire.sections.iter().enumerate() {
            let base = (si as i32 - 4) * 16;
            if base + 16 <= y {
                continue;
            }
            match &section.block_states {
                Container::Single(0) => {}
                Container::Single(state) => {
                    if opaque(*state) {
                        return false;
                    }
                }
                Container::Palette { .. } | Container::Global { .. } => {
                    for ly in 0..16i32 {
                        let wy = base + ly;
                        if wy < y {
                            continue;
                        }
                        let idx = ((ly << 8) | ((lz as i32) << 4) | lx as i32) as usize;
                        if let Some(state) = get_section_cell(&chunk.wire, si, idx) {
                            if opaque(state) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Game, Inbound, Outbound};
    use crate::WireChunk;
    use std::sync::mpsc;

    // -- encoders ------------------------------------------------------

    #[test]
    fn lp_movement_zero_small_and_continuation() {
        let mut buf = Vec::new();
        encode_lp_movement(&mut buf, 0.0, 0.0, 0.0);
        assert_eq!(buf, vec![0x00]);
        let mut buf = Vec::new();
        encode_lp_movement(&mut buf, 0.0, 0.2, 0.0);
        assert_eq!(buf.len(), 6, "{buf:02x?}");
        assert_eq!(buf[0] & 0x07, 1, "scale 1 fits the marker bits");
        let mut buf = Vec::new();
        encode_lp_movement(&mut buf, 30.0, 0.0, 0.0);
        assert!(buf.len() > 6, "continuation varint present: {buf:02x?}");
        assert_eq!(buf[0] & 0x04, 0x04, "continuation flag");
        assert_eq!(buf[0] & 0x03, 30 & 3, "low scale bits");
    }

    #[test]
    fn add_entity_golden_shape() {
        let uuid = [7u8; 16];
        let body = encode_add_entity(
            5,
            &uuid,
            ENTITY_TYPE_ITEM,
            1.5,
            64.25,
            -3.0,
            (0.0, 0.0, 0.0),
            0.0,
            0.0,
            0.0,
            0,
        );
        let mut expect = Vec::new();
        write_varint(&mut expect, 5);
        expect.extend_from_slice(&uuid);
        write_varint(&mut expect, 72);
        expect.extend_from_slice(&1.5f64.to_be_bytes());
        expect.extend_from_slice(&64.25f64.to_be_bytes());
        expect.extend_from_slice(&(-3.0f64).to_be_bytes());
        expect.push(0);
        expect.extend_from_slice(&[0, 0, 0]);
        write_varint(&mut expect, 0);
        assert_eq!(body, expect);
    }

    #[test]
    fn item_stack_data_golden() {
        // Dirt (item 55) x1: accessor 8, serializer 7, stack (count, item,
        // empty patch as two zero varints), terminator.
        let body = encode_item_stack_data(9, Some(&ItemStack::new(55, 1)));
        assert_eq!(body, vec![0x09, 0x08, 0x07, 0x01, 0x37, 0x00, 0x00, 0xff]);
        // An empty stack encodes as count 0.
        let body = encode_item_stack_data(9, None);
        assert_eq!(body, vec![0x09, 0x08, 0x07, 0x00, 0xff]);
    }

    #[test]
    fn remove_and_take_golden() {
        assert_eq!(encode_remove_entities(&[3, 70]), vec![0x02, 0x03, 0x46]);
        assert_eq!(encode_take_item(70, 1, 2), vec![0x46, 0x01, 0x02]);
    }

    #[test]
    fn position_sync_golden() {
        let body = encode_position_sync(7, 1.0, 60.0, 2.0, 0.0, 0.0, true);
        let mut expect = Vec::new();
        write_varint(&mut expect, 7);
        write_varint(&mut expect, 0);
        expect.extend_from_slice(&1.0f64.to_be_bytes());
        expect.extend_from_slice(&60.0f64.to_be_bytes());
        expect.extend_from_slice(&2.0f64.to_be_bytes());
        expect.extend_from_slice(&0.0f32.to_be_bytes());
        expect.extend_from_slice(&0.0f32.to_be_bytes());
        expect.push(1);
        assert_eq!(body, expect);
    }

    #[test]
    fn move_pos_and_motion_goldens() {
        // id, properties (on-ground bit), then 3 big-endian shorts.
        assert_eq!(
            encode_move_pos(2, -1, 0, 4096, true),
            vec![0x02, 0x01, 0xff, 0xff, 0x00, 0x00, 0x10, 0x00]
        );
        assert_eq!(
            encode_move_pos(2, 0, 0, 0, false),
            vec![0x02, 0x00, 0, 0, 0, 0, 0, 0]
        );
        // The packed movement's zero vector is one zero byte.
        assert_eq!(encode_set_motion(2, 0.0, 0.0, 0.0), vec![0x02, 0x00]);
    }

    // -- tables --------------------------------------------------------

    #[test]
    fn drop_table_families() {
        let cobble = item_id("minecraft:cobblestone").unwrap();
        let dirt = item_id("minecraft:dirt").unwrap();
        let torch = item_id("minecraft:torch").unwrap();
        assert_eq!(block_drop_item("minecraft:stone"), Some(cobble));
        assert_eq!(block_drop_item("minecraft:grass_block"), Some(dirt));
        assert_eq!(block_drop_item("minecraft:dirt"), Some(dirt));
        assert_eq!(block_drop_item("minecraft:wall_torch"), Some(torch));
        assert_eq!(block_drop_item("minecraft:torch"), Some(torch));
        assert_eq!(block_drop_item("minecraft:glass"), None);
        assert_eq!(block_drop_item("minecraft:air"), None);
        // Tool-required stone drops nothing bare-handed (the gate lives
        // in spawn_break_drop).
        assert_eq!(
            crate::dig::hardness("minecraft:stone"),
            (1.5, true),
            "stone stays in the tool-gated family"
        );
    }

    #[test]
    fn light_and_collision_predicates() {
        assert!(light_impermeable("minecraft:stone"));
        assert!(!light_impermeable("minecraft:glass"));
        assert!(!light_impermeable("minecraft:air"));
        assert!(!light_impermeable("minecraft:torch"));
        assert!(passable("minecraft:air"));
        assert!(passable("minecraft:torch"));
        assert!(!passable("minecraft:stone"));
    }

    // -- game harness --------------------------------------------------

    /// A grass-floored chunk (surface y=99) with one player.
    fn harness() -> (Game, mpsc::Receiver<Outbound>) {
        let (_tx, rx) = mpsc::channel::<Inbound>();
        let mut g = Game::new(rx, None, None);
        assert!(g.registry_for_test(), "pins/blocks.json not found");
        let grass = g.resolve_state("minecraft:grass_block").unwrap();
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
                        entries: vec![0, grass],
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
        g.seed_chunk_for_test(0, 0, wire(0));
        let (tx_out, rx_out) = mpsc::channel::<Outbound>();
        g.join_viewer_for_test(0, &[(0, 0)], tx_out);
        (g, rx_out)
    }

    /// Every frame currently queued, as (packet id, body).
    fn drain(rx: &mpsc::Receiver<Outbound>) -> Vec<(i32, Vec<u8>)> {
        let mut out = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            if let Outbound::Frame { id, body } = frame {
                out.push((id, body));
            }
        }
        out
    }

    fn of(frames: &[(i32, Vec<u8>)], want: i32) -> Vec<&[u8]> {
        frames
            .iter()
            .filter(|(id, _)| *id == want)
            .map(|(_, body)| body.as_slice())
            .collect()
    }

    #[test]
    fn break_spawns_drop_that_lands_and_vacuums() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 5.5,
            y: 100.0,
            z: 5.5,
        });
        g.handle(Inbound::Setblock {
            conn: 0,
            x: 5,
            y: 100,
            z: 5,
            name: "minecraft:torch".to_string(),
        });
        assert_eq!(g.block_label_for_test(5, 100, 5), "minecraft:torch[]");
        // Insta-break: the drop spawns with the break.
        g.handle(Inbound::PlayerAction {
            conn: 0,
            act: crate::dig::PlayerAction {
                action: crate::dig::ACTION_START_DESTROY,
                x: 5,
                y: 100,
                z: 5,
                direction: 0,
                sequence: 1,
            },
        });
        assert_eq!(g.block_label_for_test(5, 100, 5), "minecraft:air[]");
        assert_eq!(g.survival.items.len(), 1, "the torch drop spawned");
        // The pairing: add_entity with the item type + entity data. The
        // player holds entity id 1; the drop takes 2.
        let frames = drain(&rx);
        let adds = of(&frames, PACKET_ADD_ENTITY);
        assert_eq!(adds.len(), 1, "one add_entity");
        assert_eq!(adds[0][0], 0x02, "drop entity id 2");
        // id(1 byte) + uuid(16) then the type varint.
        assert_eq!(adds[0][17], 72, "entity type minecraft:item");
        let datas = of(&frames, PACKET_SET_ENTITY_DATA);
        assert_eq!(datas.len(), 1, "the stack's entity data");
        assert_eq!(datas[0][0], 0x02);
        assert_eq!(datas[0][1], DATA_ITEM);
        // Fall + the 10-tick pickup delay, then the vacuum.
        for _ in 0..20 {
            g.tick_once_for_test();
        }
        assert!(g.survival.items.is_empty(), "the drop was picked up");
        let frames = drain(&rx);
        let takes = of(&frames, PACKET_TAKE_ITEM_ENTITY);
        assert_eq!(takes.len(), 1, "the take animation");
        assert_eq!(takes[0], &[0x02, 0x01, 0x01], "item 2, player 1, count 1");
        assert!(!of(&frames, PACKET_REMOVE_ENTITIES).is_empty());
        // The torch landed in the inventory (add: selected hotbar first).
        let inv = g.player_inv_state_for_test(0).unwrap();
        let torch = item_id("minecraft:torch").unwrap();
        let found = (0..46).any(|s| {
            inv.inventory
                .get(s)
                .is_some_and(|st| st.item() == torch && st.count() == 1)
        });
        assert!(found, "torch in the inventory");
    }

    #[test]
    fn stone_break_drops_nothing_bare_handed() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 5.5,
            y: 100.0,
            z: 5.5,
        });
        g.handle(Inbound::Setblock {
            conn: 0,
            x: 5,
            y: 100,
            z: 5,
            name: "minecraft:stone".to_string(),
        });
        g.handle(Inbound::PlayerAction {
            conn: 0,
            act: crate::dig::PlayerAction {
                action: crate::dig::ACTION_START_DESTROY,
                x: 5,
                y: 100,
                z: 5,
                direction: 0,
                sequence: 1,
            },
        });
        // Stone is not insta-break: no drop until the dig finishes, and a
        // bare hand earns none anyway.
        for _ in 0..5 {
            g.tick_once_for_test();
        }
        g.handle(Inbound::PlayerAction {
            conn: 0,
            act: crate::dig::PlayerAction {
                action: crate::dig::ACTION_STOP_DESTROY,
                x: 5,
                y: 100,
                z: 5,
                direction: 0,
                sequence: 2,
            },
        });
        g.tick_once_for_test();
        assert!(
            g.survival.items.is_empty(),
            "no drop for a toolless stone break"
        );
        assert!(of(&drain(&rx), PACKET_ADD_ENTITY).is_empty());
    }

    #[test]
    fn thrown_drop_uses_the_look_direction() {
        let (mut g, _rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 5.5,
            y: 100.0,
            z: 5.5,
        });
        g.handle(Inbound::Rotated {
            conn: 0,
            yaw: 0.0,
            pitch: 0.0,
        });
        g.handle(Inbound::Give {
            conn: 0,
            item: "minecraft:dirt".to_string(),
            count: 3,
        });
        g.handle(Inbound::PlayerAction {
            conn: 0,
            act: crate::dig::PlayerAction {
                action: crate::dig::ACTION_DROP_ITEM,
                x: 0,
                y: 0,
                z: 0,
                direction: 0,
                sequence: 1,
            },
        });
        assert_eq!(g.survival.items.len(), 1, "the thrown drop spawned");
        let item = &g.survival.items[0];
        assert_eq!(item.owner, Some(0), "the thrower is attributed");
        assert_eq!(item.stack.count(), 1);
        // Spawn: eye height minus 0.3, level with the block top; facing
        // south (yaw 0): +z flight.
        assert!((item.x - 5.5).abs() < 1e-9);
        assert!((item.y - (100.0 + 1.62 - 0.3)).abs() < 1e-9);
        assert!(item.vz > 0.15, "flown south: {:?}", item.vz);
        assert!(item.vx.abs() < 0.05);
        // The 40-tick throw delay holds the vacuum: the drop flies clear.
        for _ in 0..10 {
            g.tick_once_for_test();
        }
        assert_eq!(g.survival.items.len(), 1, "still on the ground");
    }

    #[test]
    fn same_stack_drops_merge_on_the_rest_cadence() {
        let (mut g, _rx) = harness();
        let torch = item_id("minecraft:torch").unwrap();
        // Two drops spawned apart horizontally, both above the floor.
        g.spawn_item(
            5.0,
            100.3,
            5.0,
            (0.0, 0.0, 0.0),
            0.0,
            ItemStack::new(torch, 1),
            None,
        );
        g.spawn_item(
            5.5,
            100.3,
            5.5,
            (0.0, 0.0, 0.0),
            0.0,
            ItemStack::new(torch, 1),
            None,
        );
        // Land (a few ticks), then the 40-tick rest cadence merges them.
        for _ in 0..45 {
            g.tick_once_for_test();
        }
        assert_eq!(g.survival.items.len(), 1, "merged into one entity");
        assert_eq!(g.survival.items[0].stack.count(), 2);
    }

    #[test]
    fn drops_despawn_at_lifetime() {
        let (mut g, _rx) = harness();
        let torch = item_id("minecraft:torch").unwrap();
        g.spawn_item(
            5.0,
            100.3,
            5.0,
            (0.0, 0.0, 0.0),
            0.0,
            ItemStack::new(torch, 1),
            None,
        );
        for _ in 0..LIFETIME {
            g.tick_once_for_test();
        }
        assert!(g.survival.items.is_empty(), "aged out at 6000");
    }

    #[test]
    fn landing_stops_the_drop_without_bouncing() {
        let (mut g, _rx) = harness();
        let torch = item_id("minecraft:torch").unwrap();
        g.spawn_item(
            5.0,
            101.0,
            5.0,
            (0.0, 0.0, 0.0),
            0.0,
            ItemStack::new(torch, 1),
            None,
        );
        // Restitution is zero: after first ground contact the drop never
        // rises again (gravity accrues between the gated moves; the moves
        // zero the collided component).
        let mut landed = false;
        for _ in 0..16 {
            g.tick_once_for_test();
            let item = &g.survival.items[0];
            if item.on_ground {
                assert!(
                    (item.y - 100.0).abs() < 1e-9,
                    "rests at the floor top, not {}",
                    item.y
                );
                landed = true;
            } else {
                assert!(!landed, "the drop rose after landing");
            }
        }
        assert!(landed, "the drop landed");
        assert!(g.survival.items[0].vy <= 0.0, "no upward motion");
    }

    #[test]
    fn resting_drop_sends_one_frame_per_full_sync_cadence() {
        let (mut g, rx) = harness();
        let torch = item_id("minecraft:torch").unwrap();
        g.spawn_item(
            5.0,
            101.0,
            5.0,
            (0.0, 0.0, 0.0),
            0.0,
            ItemStack::new(torch, 1),
            None,
        );
        // Fall, land, settle past the landing sync.
        for _ in 0..12 {
            g.tick_once_for_test();
        }
        drain(&rx);
        // A resting drop sends no position frames except the
        // full-position cadence: one per 60 tracker ticks. (Motion frames
        // keep flowing: gravity accrues between the gated moves.)
        for _ in 0..60 {
            g.tick_once_for_test();
        }
        let frames = drain(&rx);
        let movement = of(&frames, PACKET_MOVE_ENTITY_POS).len()
            + of(&frames, PACKET_ENTITY_POSITION_SYNC).len();
        assert_eq!(movement, 1, "only the cadence frame in 60 ticks");
    }

    #[test]
    fn tracker_pairs_unpairs_and_repairs_the_same_id() {
        let (mut g, rx) = harness();
        let torch = item_id("minecraft:torch").unwrap();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 5.5,
            y: 100.0,
            z: 5.5,
        });
        g.spawn_item(
            5.0,
            100.5,
            5.0,
            (0.0, 0.0, 0.0),
            0.0,
            ItemStack::new(torch, 1),
            None,
        );
        assert_eq!(of(&drain(&rx), PACKET_ADD_ENTITY).len(), 1, "paired");
        // Out past the 64-block pairing range.
        g.handle(Inbound::Moved {
            conn: 0,
            x: 70.5,
            y: 100.0,
            z: 5.5,
            yaw: None,
            pitch: None,
        });
        let frames = drain(&rx);
        let removes = of(&frames, PACKET_REMOVE_ENTITIES);
        assert_eq!(removes.len(), 1, "unpaired");
        // Back in range (the chunk grant stands in for the streaming the
        // world-less harness cannot do).
        g.grant_chunks_for_test(0, &[(0, 0)]);
        g.handle(Inbound::Moved {
            conn: 0,
            x: 5.5,
            y: 100.0,
            z: 5.5,
            yaw: None,
            pitch: None,
        });
        let frames = drain(&rx);
        let adds = of(&frames, PACKET_ADD_ENTITY);
        assert_eq!(adds.len(), 1, "re-paired on return");
        assert_eq!(adds[0][0], removes[0][1], "the same entity id");
    }

    #[test]
    fn teleport_back_does_not_repair_until_the_next_move() {
        let (mut g, rx) = harness();
        let torch = item_id("minecraft:torch").unwrap();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 5.5,
            y: 100.0,
            z: 5.5,
        });
        g.spawn_item(
            5.0,
            100.5,
            5.0,
            (0.0, 0.0, 0.0),
            0.0,
            ItemStack::new(torch, 1),
            None,
        );
        assert_eq!(of(&drain(&rx), PACKET_ADD_ENTITY).len(), 1, "paired");
        // Away, past the view ring (the harness keeps failed chunk loads
        // out of `sent`, so the drop's chunk leaves the set here): the
        // distance drops the pairing and the away view drops the chunk.
        g.handle(Inbound::Tp {
            conn: 0,
            x: 5.5,
            y: 100.0,
            z: -100.5,
        });
        let away = drain(&rx);
        let removes = of(&away, PACKET_REMOVE_ENTITIES);
        assert_eq!(removes.len(), 1, "unpaired at the away teleport");
        // Straight back: the pairing pass reads the away view, so the
        // drop stays unpaired (the reference swaps its tracking view
        // only after the pairing pass).
        g.handle(Inbound::Tp {
            conn: 0,
            x: 5.5,
            y: 100.0,
            z: 5.5,
        });
        let back = drain(&rx);
        assert!(
            of(&back, PACKET_ADD_ENTITY).is_empty(),
            "no re-pair at the return teleport"
        );
        // The next move re-checks against the restored view and pairs
        // (the chunk grant stands in for the streaming the world-less
        // harness cannot do).
        g.grant_chunks_for_test(0, &[(0, 0)]);
        g.handle(Inbound::Moved {
            conn: 0,
            x: 5.5,
            y: 100.0,
            z: 5.5,
            yaw: None,
            pitch: None,
        });
        let moved = drain(&rx);
        let adds = of(&moved, PACKET_ADD_ENTITY);
        assert_eq!(adds.len(), 1, "re-paired at the move");
        assert_eq!(adds[0][0], removes[0][1], "the same entity id");
    }

    // -- random ticks --------------------------------------------------

    #[test]
    fn capped_grass_decays_to_dirt() {
        let (mut g, _rx) = harness();
        for x in 4..8 {
            for z in 4..8 {
                g.handle(Inbound::Setblock {
                    conn: 0,
                    x,
                    y: 100,
                    z,
                    name: "minecraft:stone".to_string(),
                });
            }
        }
        for _ in 0..2000 {
            g.random_ticks();
        }
        let dirt = (4..8)
            .flat_map(|x| (4..8).map(move |z| (x, z)))
            .filter(|&(x, z)| {
                g.block_label_for_test(x, 99, z)
                    .starts_with("minecraft:dirt")
            })
            .count();
        assert!(dirt >= 12, "{dirt}/16 capped cells decayed");
    }

    #[test]
    fn grass_spreads_onto_bare_dirt() {
        let (mut g, _rx) = harness();
        for x in 4..7 {
            for z in 4..7 {
                g.handle(Inbound::Setblock {
                    conn: 0,
                    x,
                    y: 99,
                    z,
                    name: "minecraft:dirt".to_string(),
                });
            }
        }
        for _ in 0..2000 {
            g.random_ticks();
        }
        let grown = (4..7)
            .flat_map(|x| (4..7).map(move |z| (x, z)))
            .filter(|&(x, z)| {
                g.block_label_for_test(x, 99, z)
                    .starts_with("minecraft:grass_block")
            })
            .count();
        assert!(grown >= 1, "{grown}/9 dirt cells grew grass");
    }

    #[test]
    fn grass_under_a_torch_survives() {
        let (mut g, _rx) = harness();
        g.handle(Inbound::Setblock {
            conn: 0,
            x: 5,
            y: 100,
            z: 5,
            name: "minecraft:torch".to_string(),
        });
        for _ in 0..500 {
            g.random_ticks();
        }
        assert!(
            g.block_label_for_test(5, 99, 5)
                .starts_with("minecraft:grass_block"),
            "a torch passes light"
        );
    }

    #[test]
    fn gamerule_randomtickspeed_accelerates_decay() {
        let (mut g, _rx) = harness();
        for x in 4..8 {
            for z in 4..8 {
                g.handle(Inbound::Setblock {
                    conn: 0,
                    x,
                    y: 100,
                    z,
                    name: "minecraft:stone".to_string(),
                });
            }
        }
        g.set_tick_speed(400);
        for _ in 0..50 {
            g.random_ticks();
        }
        let dirt = (4..8)
            .flat_map(|x| (4..8).map(move |z| (x, z)))
            .filter(|&(x, z)| {
                g.block_label_for_test(x, 99, z)
                    .starts_with("minecraft:dirt")
            })
            .count();
        assert!(
            dirt >= 12,
            "{dirt}/16 capped cells decayed in 50 fast ticks"
        );
    }
}
