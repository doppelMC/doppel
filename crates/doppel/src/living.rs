//! Living entities: the mob base (health, hurt timing, death, ground
//! movement, the per-tick tracker sync) and the flag-preempting goal
//! selector. Wire shapes follow the 26.3 registration order; behavior
//! constants carry the ground mob's values. Per-mob behavior lives in
//! per-mob modules behind `MobKind`; there is no mob enum.

use crate::game::entities::{
    block_solid, encode_add_entity, encode_position_sync, encode_remove_entities,
    PACKET_ADD_ENTITY, PACKET_ENTITY_POSITION_SYNC, PACKET_MOVE_ENTITY_POS,
    PACKET_MOVE_ENTITY_POS_ROT, PACKET_REMOVE_ENTITIES, PACKET_SET_ENTITY_DATA,
};
use crate::game::{ConnId, Game};
use doppel_protocol::write_varint;

// ---------------------------------------------------------------------
// Wire packet ids (clientbound play state)
// ---------------------------------------------------------------------

/// `damage_event`: registration order 26. TODO wire-verify at the gate.
pub const PACKET_DAMAGE_EVENT: i32 = 0x19;
/// `entity_event`: registration order 35. TODO wire-verify at the gate.
pub const PACKET_ENTITY_EVENT: i32 = 0x22;
/// `hurt_animation`: registration order 44. TODO wire-verify at the gate.
pub const PACKET_HURT_ANIMATION: i32 = 0x2b;
/// `move_entity_rot`: registration order 58. TODO wire-verify at the gate.
pub const PACKET_MOVE_ENTITY_ROT: i32 = 0x39;
/// `rotate_head`: registration order 86. TODO wire-verify at the gate.
pub const PACKET_ROTATE_HEAD: i32 = 0x55;
/// `set_equipment`: registration order 105. Bare mobs send none (the
/// packet only carries non-empty slots).
pub const PACKET_SET_EQUIPMENT: i32 = 0x68;
/// Equipment slot ordinals (declaration order): main hand.
pub const EQUIP_MAIN_HAND: u8 = 0;
/// `update_attributes`: registration order 135. TODO wire-verify at the
/// gate.
pub const PACKET_UPDATE_ATTRIBUTES: i32 = 0x86;

/// `minecraft:zombie` in the entity-type registry (registration order
/// 155, 0-based). TODO wire-verify at the gate.
pub const ENTITY_TYPE_ZOMBIE: i32 = 154;
/// `minecraft:skeleton` (registration order 119, 0-based).
/// TODO wire-verify at the gate.
pub const ENTITY_TYPE_SKELETON: i32 = 118;
/// `minecraft:creeper` (registration order 33, 0-based).
/// TODO wire-verify at the gate.
pub const ENTITY_TYPE_CREEPER: i32 = 32;
/// `minecraft:spider` (registration order 128, 0-based).
/// TODO wire-verify at the gate.
pub const ENTITY_TYPE_SPIDER: i32 = 127;

// ---------------------------------------------------------------------
// Entity data and attributes
// ---------------------------------------------------------------------

/// Entity-data serializer ids (registration order): BYTE.
pub const SER_BYTE: i32 = 0;
/// Entity-data serializer ids (registration order): INT.
pub const SER_INT: i32 = 1;
/// Entity-data serializer ids (registration order): FLOAT.
pub const SER_FLOAT: i32 = 3;
/// The base entity flags accessor; bit 0x01 = on fire.
pub const DATA_ENTITY_FLAGS: u8 = 0;
/// The health accessor on the living hierarchy (serializer FLOAT).
pub const DATA_LIVING_HEALTH: u8 = 9;
/// The mob flags accessor; bit 0x04 = aggressive.
pub const DATA_MOB_FLAGS: u8 = 15;
/// The wall-crawler's flag byte accessor (bit 0x01 = climbing).
pub const DATA_CLIMBING_FLAGS: u8 = 16;
/// The creeper's swell-direction accessor (INT: -1 shrinking, 1
/// swelling).
pub const DATA_SWELL_DIR: u8 = 16;

/// `movement_speed` in the attribute registry (alphabetical
/// registration); wire-verified: the reference's zombie snapshot carries
/// it alone (default-valued attributes are omitted).
pub const ATTR_MOVEMENT_SPEED: i32 = 26;
/// `max_health` in the attribute registry (alphabetical registration).
/// TODO wire-verify at the gate.
pub const ATTR_MAX_HEALTH: i32 = 23;
/// The max-health attribute's registry default; the pairing omits it.
const DEFAULT_MAX_HEALTH: f32 = 20.0;
/// A player's full health bar; mob melee damage beyond it means the
/// player is down and no longer a target.
const PLAYER_HEALTH: f32 = 20.0;
/// `mob_attack` in the damage-type registry (alphabetical order 28).
/// TODO wire-verify at the gate.
pub const DAMAGE_TYPE_MOB_ATTACK: i32 = 28;

/// The entity-data entries a bare mob pairs with: only health sits off
/// its default.
pub const PAIRING_DATA_LAYOUT: &[(u8, i32)] = &[(DATA_LIVING_HEALTH, SER_FLOAT)];

/// entity_event payload: the death animation.
pub const EVENT_DEATH: u8 = 3;
/// entity_event payload: the death animation completes.
pub const EVENT_DEATH_FINISH: u8 = 60;

// ---------------------------------------------------------------------
// Behavior constants
// ---------------------------------------------------------------------

/// Gravity per tick at the default attribute.
const GRAVITY: f64 = 0.08;
/// Horizontal friction per tick on default ground (slipperiness 0.6).
const H_FRICTION: f64 = 0.6 * 0.91;
/// Vertical drag per tick.
const V_DRAG: f64 = 0.98;
/// Components below this clamp to zero.
const V_MIN: f64 = 0.003;
/// Jump velocity at the default attribute.
const JUMP_POWER: f64 = 0.42;
/// Ticks between jump control firings.
const JUMP_DELAY: i32 = 10;
/// Mob tracking range: clientTrackingRange 8 chunks.
pub const MOB_TRACK_RANGE: f64 = 128.0;
/// Movement sync cadence: the tracker updateInterval for mob types.
const SYNC_INTERVAL: i64 = 3;
/// Ticks without a full sync before the next movement sends as one.
const TELEPORT_DELAY_MAX: i64 = 400;
/// Movement deltas are 1/4096-block shorts.
const DELTA_SCALE: f64 = 4096.0;
/// Damage cooldown after a full hit.
const HURT_COOLDOWN: i32 = 20;
/// Ticks of the hurt flash.
const HURT_TIME: i32 = 10;
/// Fire ticks per ignition (8 seconds).
pub const FIRE_IGNITE_TICKS: i32 = 320;
/// Instant-despawn distance from the nearest player.
const DESPAWN_DISTANCE_SQ: f64 = 128.0 * 128.0;
/// Distance under which the no-action clock resets.
const NO_DESPAWN_DISTANCE_SQ: f64 = 32.0 * 32.0;
/// No-action ticks before the random despawn roll starts.
const NO_ACTION_LIMIT: i32 = 600;
/// One-in-N random despawn roll per tick past the limit.
const DESPAWN_ROLL: u64 = 800;
/// Ticks of the death animation before removal.
const DEATH_TICKS: i32 = 20;
/// Body-yaw turn cap per tick (the move control's 90 degrees).
const TURN_RATE: f32 = 90.0;
/// Player eye height above the feet.
pub const PLAYER_EYE: f64 = 1.62;
/// The climb rise per tick while pressed against a wall.
const CLIMB_RISE: f64 = 0.2;

// ---------------------------------------------------------------------
// Wire encoders
// ---------------------------------------------------------------------

/// `set_entity_data` for a float entry (health).
pub fn encode_float_data(entity_id: i32, accessor: u8, value: f32) -> Vec<u8> {
    let mut body = Vec::with_capacity(10);
    write_varint(&mut body, entity_id);
    body.push(accessor);
    write_varint(&mut body, SER_FLOAT);
    body.extend_from_slice(&value.to_be_bytes());
    body.push(0xff);
    body
}

/// Entity-data serializer ids (registration order): BOOLEAN.
pub const SER_BOOLEAN: i32 = 10;

/// `set_entity_data` for an int entry (the swell direction). Int
/// values ride the entity-data channel as varints.
pub fn encode_int_data(entity_id: i32, accessor: u8, value: i32) -> Vec<u8> {
    let mut body = Vec::with_capacity(10);
    write_varint(&mut body, entity_id);
    body.push(accessor);
    write_varint(&mut body, SER_INT);
    write_varint(&mut body, value);
    body.push(0xff);
    body
}

/// `set_entity_data` for a boolean entry.
pub fn encode_boolean_data(entity_id: i32, accessor: u8, value: bool) -> Vec<u8> {
    let mut body = Vec::with_capacity(8);
    write_varint(&mut body, entity_id);
    body.push(accessor);
    write_varint(&mut body, SER_BOOLEAN);
    body.push(u8::from(value));
    body.push(0xff);
    body
}

/// `set_entity_data` for a byte entry (entity flags, mob flags).
pub fn encode_byte_data(entity_id: i32, accessor: u8, value: u8) -> Vec<u8> {
    let mut body = Vec::with_capacity(8);
    write_varint(&mut body, entity_id);
    body.push(accessor);
    write_varint(&mut body, SER_BYTE);
    body.push(value);
    body.push(0xff);
    body
}

/// `update_attributes`: id, count, then per attribute (id, f64 value,
/// zero modifiers).
pub fn encode_update_attributes(entity_id: i32, attrs: &[(i32, f64)]) -> Vec<u8> {
    let mut body = Vec::with_capacity(8 + attrs.len() * 12);
    write_varint(&mut body, entity_id);
    write_varint(&mut body, attrs.len() as i32);
    for (attr, value) in attrs {
        write_varint(&mut body, *attr);
        body.extend_from_slice(&value.to_be_bytes());
        write_varint(&mut body, 0);
    }
    body
}

/// `move_entity_pos`: id, three 1/4096-block shorts, on-ground.
pub fn encode_move_pos(entity_id: i32, dx: i64, dy: i64, dz: i64, on_ground: bool) -> Vec<u8> {
    let mut body = Vec::with_capacity(9);
    write_varint(&mut body, entity_id);
    body.extend_from_slice(&(dx as i16).to_be_bytes());
    body.extend_from_slice(&(dy as i16).to_be_bytes());
    body.extend_from_slice(&(dz as i16).to_be_bytes());
    body.push(u8::from(on_ground));
    body
}

/// `move_entity_pos_rot`: the delta shorts plus the packed rotations.
/// The yaw/pitch byte order carries a TODO until the gate pins it.
pub fn encode_move_pos_rot(
    entity_id: i32,
    dx: i64,
    dy: i64,
    dz: i64,
    yaw: u8,
    pitch: u8,
    on_ground: bool,
) -> Vec<u8> {
    let mut body = Vec::with_capacity(11);
    write_varint(&mut body, entity_id);
    body.extend_from_slice(&(dx as i16).to_be_bytes());
    body.extend_from_slice(&(dy as i16).to_be_bytes());
    body.extend_from_slice(&(dz as i16).to_be_bytes());
    body.push(yaw);
    body.push(pitch);
    body.push(u8::from(on_ground));
    body
}

/// `move_entity_rot`: id, packed yaw, packed pitch, on-ground.
pub fn encode_move_rot(entity_id: i32, yaw: u8, pitch: u8, on_ground: bool) -> Vec<u8> {
    let mut body = Vec::with_capacity(6);
    write_varint(&mut body, entity_id);
    body.push(yaw);
    body.push(pitch);
    body.push(u8::from(on_ground));
    body
}

/// `rotate_head`: id plus the packed head yaw.
pub fn encode_rotate_head(entity_id: i32, head_yaw: u8) -> Vec<u8> {
    let mut body = Vec::with_capacity(6);
    write_varint(&mut body, entity_id);
    body.push(head_yaw);
    body
}

/// `entity_event`: id plus the event byte.
pub fn encode_entity_event(entity_id: i32, event: u8) -> Vec<u8> {
    let mut body = Vec::with_capacity(6);
    write_varint(&mut body, entity_id);
    body.push(event);
    body
}

/// `hurt_animation`: the hurt entity id plus the source-direction yaw.
pub fn encode_hurt_animation(entity_id: i32, yaw: f32) -> Vec<u8> {
    let mut body = Vec::with_capacity(8);
    write_varint(&mut body, entity_id);
    body.extend_from_slice(&yaw.to_be_bytes());
    body
}

/// `damage_event`: the hurt entity, the damage-type id, the causing
/// entity, the direct cause.
pub fn encode_damage_event(entity_id: i32, damage_type: i32, cause: i32, direct: i32) -> Vec<u8> {
    let mut body = Vec::with_capacity(11);
    write_varint(&mut body, entity_id);
    write_varint(&mut body, damage_type);
    // Optional entity ids travel as id + 1, so -1 (absent) encodes 0.
    write_varint(&mut body, cause + 1);
    write_varint(&mut body, direct + 1);
    // No source position.
    body.push(0);
    body
}

/// `set_equipment`: id, then (slot byte, stack) pairs; the 0x80 bit on
/// the slot byte marks a following entry.
pub fn encode_equipment(entity_id: i32, slots: &[(u8, &crate::inventory::ItemStack)]) -> Vec<u8> {
    let mut body = Vec::with_capacity(8 + slots.len() * 6);
    write_varint(&mut body, entity_id);
    for (i, (slot, stack)) in slots.iter().enumerate() {
        let more = if i + 1 < slots.len() { 0x80 } else { 0x00 };
        body.push(slot | more);
        crate::inventory::encode_item_stack(&mut body, Some(stack));
    }
    body
}

/// Degrees packed to the wire byte: `deg * 256 / 360`.
pub fn pack_degrees(deg: f32) -> u8 {
    (deg * 256.0 / 360.0) as i8 as u8
}

/// Whether the sight line between two eye points is clear.
pub(crate) fn visible(world: &Game, from: (f64, f64, f64), to: (f64, f64, f64)) -> bool {
    let (dx, dy, dz) = (to.0 - from.0, to.1 - from.1, to.2 - from.2);
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    let steps = (dist * 2.0).ceil() as i32;
    for s in 1..steps {
        let t = s as f64 / steps as f64;
        let (x, y, z) = (from.0 + dx * t, from.1 + dy * t, from.2 + dz * t);
        if block_solid(world, x.floor() as i32, y.floor() as i32, z.floor() as i32) {
            return false;
        }
    }
    true
}

/// An eye position `eye` blocks above the feet.
pub(crate) fn eye_at(eye: f64, pos: (f64, f64, f64)) -> (f64, f64, f64) {
    (pos.0, pos.1 + eye, pos.2)
}

/// The look angles from an eye point toward a target point.
pub(crate) fn look_angles(from: (f64, f64, f64), to: (f64, f64, f64)) -> (f32, f32) {
    let (dx, dy, dz) = (to.0 - from.0, to.1 - from.1, to.2 - from.2);
    let horiz = (dx * dx + dz * dz).sqrt();
    (
        (-dx).atan2(dz).to_degrees() as f32,
        -dy.atan2(horiz).to_degrees() as f32,
    )
}

// ---------------------------------------------------------------------
// Goal system
// ---------------------------------------------------------------------

/// The mutually exclusive resources goals lock.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GoalFlags(u8);

impl GoalFlags {
    pub const MOVE: GoalFlags = GoalFlags(1);
    pub const LOOK: GoalFlags = GoalFlags(2);
    #[allow(dead_code)]
    pub const JUMP: GoalFlags = GoalFlags(4);
    pub const TARGET: GoalFlags = GoalFlags(8);

    pub fn union(self, other: GoalFlags) -> GoalFlags {
        GoalFlags(self.0 | other.0)
    }

    fn is_disjoint(self, other: GoalFlags) -> bool {
        self.0 & other.0 == 0
    }
}

/// What a goal sees: the world, the mob's own state, and the mob's RNG.
/// Goals never own the mob.
pub struct GoalCtx<'a> {
    pub world: &'a Game,
    pub body: &'a mut MobBody,
    pub rand: &'a mut u64,
}

impl GoalCtx<'_> {
    /// One splitmix draw from the mob's stream.
    pub fn draw(&mut self) -> u64 {
        *self.rand = self.rand.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *self.rand;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// A uniform draw in [0, n).
    pub fn below(&mut self, n: u64) -> u64 {
        self.draw() % n
    }

    /// The nearest living player within `range` of the mob, if any.
    pub fn nearest_player(&self, range: f64) -> Option<(ConnId, (f64, f64, f64))> {
        let mut best: Option<(f64, ConnId, (f64, f64, f64))> = None;
        for (&conn, p) in &self.world.players {
            if self.downed(conn) {
                continue;
            }
            let (dx, dy, dz) = (p.x - self.body.x, p.y - self.body.y, p.z - self.body.z);
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 > range * range {
                continue;
            }
            if best.as_ref().is_none_or(|(b, ..)| d2 < *b) {
                best = Some((d2, conn, (p.x, p.y, p.z)));
            }
        }
        best.map(|(_, conn, pos)| (conn, pos))
    }

    /// A player's position by connection; a downed player has none.
    pub fn player_pos(&self, conn: ConnId) -> Option<(f64, f64, f64)> {
        if self.downed(conn) {
            return None;
        }
        self.world.players.get(&conn).map(|p| (p.x, p.y, p.z))
    }

    /// Whether a player has absorbed a full health bar of mob damage.
    fn downed(&self, conn: ConnId) -> bool {
        self.world
            .mobs
            .player_damage
            .get(&conn)
            .is_some_and(|d| *d >= PLAYER_HEALTH)
    }
}

/// One prioritized behavior. `can_use` gates a start, `can_continue` a
/// keep; flags stay locked until stop or preemption.
pub trait Goal: Send {
    fn flags(&self) -> GoalFlags;
    fn can_use(&mut self, ctx: &mut GoalCtx) -> bool;
    fn can_continue_to_use(&mut self, ctx: &mut GoalCtx) -> bool;
    fn start(&mut self, _ctx: &mut GoalCtx) {}
    fn stop(&mut self, _ctx: &mut GoalCtx) {}
    fn tick(&mut self, _ctx: &mut GoalCtx) {}
    /// Whether the goal also ticks on the off cadence.
    fn requires_every_tick(&self) -> bool {
        false
    }
}

struct GoalEntry {
    priority: i32,
    goal: Box<dyn Goal>,
    running: bool,
}

/// The flag-preempting selector: registration order breaks ties, a
/// strictly smaller priority number takes flags from a running goal,
/// equal priorities never preempt.
pub struct GoalSelector {
    entries: Vec<GoalEntry>,
}

impl GoalSelector {
    pub fn new() -> GoalSelector {
        GoalSelector {
            entries: Vec::new(),
        }
    }

    pub fn add(&mut self, priority: i32, goal: Box<dyn Goal>) {
        self.entries.push(GoalEntry {
            priority,
            goal,
            running: false,
        });
    }

    /// One selector pass. `full` adds the start phase; running goals
    /// tick (only every-tick goals on the off cadence).
    pub fn tick(&mut self, full: bool, ctx: &mut GoalCtx) {
        // Stop phase: release the flags of finished goals.
        for e in &mut self.entries {
            if e.running && !e.goal.can_continue_to_use(ctx) {
                e.goal.stop(ctx);
                e.running = false;
            }
        }
        if !full {
            for e in &mut self.entries {
                if e.running && e.goal.requires_every_tick() {
                    e.goal.tick(ctx);
                }
            }
            return;
        }
        // Start phase, registration order: a goal starts when every
        // flag it needs is free or held by a less important goal.
        let order: Vec<usize> = (0..self.entries.len()).collect();
        for i in order {
            if self.entries[i].running {
                continue;
            }
            let flags = self.entries[i].goal.flags();
            let mut blocked = false;
            let mut displaced: Vec<usize> = Vec::new();
            for j in 0..self.entries.len() {
                if j == i || !self.entries[j].running {
                    continue;
                }
                if self.entries[j].goal.flags().is_disjoint(flags) {
                    continue;
                }
                if self.entries[j].priority <= self.entries[i].priority {
                    blocked = true;
                    break;
                }
                displaced.push(j);
            }
            if blocked || !self.entries[i].goal.can_use(ctx) {
                continue;
            }
            for j in displaced {
                self.entries[j].goal.stop(ctx);
                self.entries[j].running = false;
            }
            self.entries[i].goal.start(ctx);
            self.entries[i].running = true;
        }
        // Tick phase.
        for e in &mut self.entries {
            if e.running {
                e.goal.tick(ctx);
            }
        }
    }

    #[cfg(test)]
    pub fn running_priorities(&self) -> Vec<i32> {
        self.entries
            .iter()
            .filter(|e| e.running)
            .map(|e| e.priority)
            .collect()
    }
}

impl Default for GoalSelector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------
// Shared goals
// ---------------------------------------------------------------------

/// The local brightness ratio: the darkened sky light over 15, zero
/// under cover (block light stays unmodeled).
pub(crate) fn brightness(world: &Game, body: &MobBody) -> f64 {
    let eye = (
        body.x.floor() as i32,
        (body.y + 1.0) as i32,
        body.z.floor() as i32,
    );
    if world.sky_exposed(eye.0, eye.1, eye.2) {
        (15 - crate::spawning::sky_darken(world.spawning.day_time)).max(0) as f64 / 15.0
    } else {
        0.0
    }
}

/// The idle stroll: a random nearby column on a roll.
pub(crate) struct IdleStrollGoal {
    range: i64,
    chance: u64,
    give_up: i32,
    speed: f64,
}

impl IdleStrollGoal {
    pub(crate) fn new(range: i64, chance: u64, give_up: i32, speed: f64) -> IdleStrollGoal {
        IdleStrollGoal {
            range,
            chance,
            give_up,
            speed,
        }
    }
}

impl Goal for IdleStrollGoal {
    fn flags(&self) -> GoalFlags {
        GoalFlags::MOVE
    }

    fn can_use(&mut self, ctx: &mut GoalCtx) -> bool {
        if ctx.body.no_action_time >= 100 {
            return false;
        }
        if ctx.below(self.chance) != 0 {
            return false;
        }
        let span = (self.range * 2 + 1) as u64;
        let x = ctx.body.x + (ctx.below(span) as f64 - self.range as f64);
        let z = ctx.body.z + (ctx.below(span) as f64 - self.range as f64);
        ctx.body.nav.move_to(x, z, self.speed);
        true
    }

    fn can_continue_to_use(&mut self, ctx: &mut GoalCtx) -> bool {
        ctx.body.nav.in_progress() && ctx.body.nav.tick_age(self.give_up)
    }
}

/// Look at the nearest visible player inside the range.
pub(crate) struct WatchPlayerGoal {
    range: f64,
    chance: u64,
    remaining: i32,
    duration: i32,
    conn: Option<ConnId>,
}

impl WatchPlayerGoal {
    pub(crate) fn new(range: f64, chance: u64) -> WatchPlayerGoal {
        WatchPlayerGoal {
            range,
            chance,
            remaining: 0,
            duration: 0,
            conn: None,
        }
    }
}

impl Goal for WatchPlayerGoal {
    fn flags(&self) -> GoalFlags {
        GoalFlags::LOOK
    }

    fn can_use(&mut self, ctx: &mut GoalCtx) -> bool {
        if ctx.below(self.chance) != 0 {
            return false;
        }
        let Some((conn, pos)) = ctx.nearest_player(self.range) else {
            return false;
        };
        if !visible(ctx.world, eye_of(ctx.body), eye_at(PLAYER_EYE, pos)) {
            return false;
        }
        self.conn = Some(conn);
        self.duration = 20 + ctx.below(20) as i32;
        true
    }

    fn can_continue_to_use(&mut self, ctx: &mut GoalCtx) -> bool {
        self.remaining > 0
            && self.conn.is_some_and(|c| {
                ctx.player_pos(c).is_some_and(|p| {
                    let (dx, dy, dz) = (p.0 - ctx.body.x, p.1 - ctx.body.y, p.2 - ctx.body.z);
                    dx * dx + dy * dy + dz * dz < (self.range + 1.0) * (self.range + 1.0)
                })
            })
    }

    fn start(&mut self, _ctx: &mut GoalCtx) {
        self.remaining = self.duration;
    }

    fn stop(&mut self, ctx: &mut GoalCtx) {
        ctx.body.look = None;
    }

    fn tick(&mut self, ctx: &mut GoalCtx) {
        self.remaining -= 1;
        if let Some(conn) = self.conn {
            if let Some(pos) = ctx.player_pos(conn) {
                ctx.body.look = Some(look_angles(eye_of(ctx.body), eye_at(PLAYER_EYE, pos)));
            }
        }
    }
}

/// A random horizontal glance.
pub(crate) struct GlanceGoal {
    chance: u64,
    remaining: i32,
    duration: i32,
    want: (f32, f32),
}

impl GlanceGoal {
    pub(crate) fn new(chance: u64) -> GlanceGoal {
        GlanceGoal {
            chance,
            remaining: 0,
            duration: 0,
            want: (0.0, 0.0),
        }
    }
}

impl Goal for GlanceGoal {
    fn flags(&self) -> GoalFlags {
        GoalFlags::MOVE.union(GoalFlags::LOOK)
    }

    fn can_use(&mut self, ctx: &mut GoalCtx) -> bool {
        if ctx.below(self.chance) != 0 {
            return false;
        }
        self.want = (ctx.below(360) as f32, 0.0);
        self.duration = 20 + ctx.below(20) as i32;
        true
    }

    fn can_continue_to_use(&mut self, _ctx: &mut GoalCtx) -> bool {
        self.remaining > 0
    }

    fn start(&mut self, _ctx: &mut GoalCtx) {
        self.remaining = self.duration;
    }

    fn stop(&mut self, ctx: &mut GoalCtx) {
        ctx.body.look = None;
    }

    fn tick(&mut self, ctx: &mut GoalCtx) {
        self.remaining -= 1;
        ctx.body.look = Some(self.want);
    }
}

/// Target the nearest visible player inside the follow range; drop it
/// when unseen or out of range. The light floor, when set, keeps the
/// goal from starting in bright places.
pub(crate) struct NearestPlayerTargetGoal {
    scan_every: i32,
    unseen_limit: i32,
    follow_range: f64,
    hostile_below: Option<f64>,
    scan_in: i32,
    unseen: i32,
}

impl NearestPlayerTargetGoal {
    pub(crate) fn new(
        scan_every: i32,
        unseen_limit: i32,
        follow_range: f64,
        hostile_below: Option<f64>,
    ) -> NearestPlayerTargetGoal {
        NearestPlayerTargetGoal {
            scan_every,
            unseen_limit,
            follow_range,
            hostile_below,
            scan_in: 0,
            unseen: 0,
        }
    }
}

impl Goal for NearestPlayerTargetGoal {
    fn flags(&self) -> GoalFlags {
        GoalFlags::TARGET
    }

    fn can_use(&mut self, ctx: &mut GoalCtx) -> bool {
        if self.scan_in > 0 {
            self.scan_in -= 1;
            return false;
        }
        self.scan_in = self.scan_every;
        if self
            .hostile_below
            .is_some_and(|limit| brightness(ctx.world, ctx.body) >= limit)
        {
            return false;
        }
        let Some((conn, pos)) = ctx.nearest_player(self.follow_range) else {
            return false;
        };
        if !visible(ctx.world, eye_of(ctx.body), eye_at(PLAYER_EYE, pos)) {
            return false;
        }
        self.unseen = 0;
        ctx.body.target = Some(conn);
        true
    }

    fn can_continue_to_use(&mut self, ctx: &mut GoalCtx) -> bool {
        let Some(conn) = ctx.body.target else {
            return false;
        };
        let Some(pos) = ctx.player_pos(conn) else {
            return false;
        };
        let (dx, dy, dz) = (pos.0 - ctx.body.x, pos.1 - ctx.body.y, pos.2 - ctx.body.z);
        if dx * dx + dy * dy + dz * dz > self.follow_range * self.follow_range {
            return false;
        }
        if visible(ctx.world, eye_of(ctx.body), eye_at(PLAYER_EYE, pos)) {
            self.unseen = 0;
        } else {
            self.unseen += 1;
        }
        self.unseen <= self.unseen_limit
    }

    fn stop(&mut self, ctx: &mut GoalCtx) {
        ctx.body.target = None;
    }
}

/// The melee approach: chase the target, look at it, hit inside the
/// reach once the cooldown spends. Approaches without hitting when
/// `hits` is false.
pub(crate) struct ChaseHitGoal {
    reach: f64,
    follow_range: f64,
    hits: bool,
    check_in: i32,
    cooldown: i32,
    target: Option<ConnId>,
    last_path: (f64, f64),
}

impl ChaseHitGoal {
    pub(crate) fn new(reach: f64, follow_range: f64, hits: bool) -> ChaseHitGoal {
        ChaseHitGoal {
            reach,
            follow_range,
            hits,
            check_in: 0,
            cooldown: 0,
            target: None,
            last_path: (0.0, 0.0),
        }
    }
}

/// Attack cooldown, in ticks (the 20-tick interval halved).
const MELEE_COOLDOWN: i32 = 10;

impl Goal for ChaseHitGoal {
    fn flags(&self) -> GoalFlags {
        GoalFlags::MOVE.union(GoalFlags::LOOK)
    }

    fn can_use(&mut self, ctx: &mut GoalCtx) -> bool {
        if self.check_in > 0 {
            self.check_in -= 1;
            return false;
        }
        self.check_in = MELEE_COOLDOWN;
        let Some(conn) = ctx.body.target else {
            return false;
        };
        self.target = ctx.player_pos(conn).map(|_| conn);
        self.target.is_some()
    }

    fn can_continue_to_use(&mut self, ctx: &mut GoalCtx) -> bool {
        let Some(conn) = self.target else {
            return false;
        };
        if ctx.body.target != Some(conn) {
            return false;
        }
        match ctx.player_pos(conn) {
            Some((x, y, z)) => {
                let (dx, dy, dz) = (x - ctx.body.x, y - ctx.body.y, z - ctx.body.z);
                dx * dx + dy * dy + dz * dz <= self.follow_range * self.follow_range
            }
            None => false,
        }
    }

    fn start(&mut self, ctx: &mut GoalCtx) {
        self.cooldown = MELEE_COOLDOWN;
        ctx.body.melee_active = true;
        if let Some((x, _, z)) = ctx.body.target.and_then(|conn| ctx.player_pos(conn)) {
            self.last_path = (x, z);
            ctx.body.nav.move_to(x, z, 1.0);
        }
    }

    fn stop(&mut self, ctx: &mut GoalCtx) {
        ctx.body.melee_active = false;
        ctx.body.look = None;
        ctx.body.nav.stop();
        self.target = None;
    }

    fn tick(&mut self, ctx: &mut GoalCtx) {
        self.cooldown -= 1;
        let Some(conn) = self.target.or(ctx.body.target) else {
            return;
        };
        let Some((px, py, pz)) = ctx.player_pos(conn) else {
            return;
        };
        ctx.body.look = Some(look_angles(
            eye_of(ctx.body),
            eye_at(PLAYER_EYE, (px, py, pz)),
        ));
        // Re-path when the target moved a block or on the 5% roll.
        let moved = (px - self.last_path.0) * (px - self.last_path.0)
            + (pz - self.last_path.1) * (pz - self.last_path.1);
        if moved >= 1.0 || ctx.below(20) == 0 {
            self.last_path = (px, pz);
            ctx.body.nav.retarget(px, pz, 1.0);
        }
        if !self.hits || self.cooldown > 0 {
            return;
        }
        let (dy, horiz) = (
            py + PLAYER_EYE - eye_of(ctx.body).1,
            ((px - ctx.body.x) * (px - ctx.body.x) + (pz - ctx.body.z) * (pz - ctx.body.z)).sqrt(),
        );
        if horiz >= self.reach || dy.abs() > 2.5 {
            return;
        }
        if !visible(
            ctx.world,
            eye_of(ctx.body),
            eye_at(PLAYER_EYE, (px, py, pz)),
        ) {
            return;
        }
        ctx.body.pending_hit = Some(conn);
        self.cooldown = MELEE_COOLDOWN;
    }

    fn requires_every_tick(&self) -> bool {
        true
    }
}

/// The mob eye position at the kind's eye height.
pub(crate) fn eye_of(body: &MobBody) -> (f64, f64, f64) {
    (body.x, body.y + body.eye, body.z)
}

// ---------------------------------------------------------------------
// Mob state
// ---------------------------------------------------------------------

/// The kind-specific parameters every mob carries.
pub trait MobKind: Send {
    /// The entity-type registry id.
    fn type_id(&self) -> i32;
    /// Hitbox half width.
    fn half_width(&self) -> f64;
    /// Hitbox height.
    fn height(&self) -> f64;
    /// Eye height above the feet.
    fn eye(&self) -> f64;
    /// The movement-speed attribute.
    fn base_speed(&self) -> f64;
    /// The attack-damage attribute.
    fn attack_damage(&self) -> f32;
    /// The follow-range attribute.
    fn follow_range(&self) -> f64;
    /// Whether the navigator climbs walls.
    fn can_climb(&self) -> bool {
        false
    }
    /// The max-health attribute.
    fn max_health(&self) -> f32 {
        20.0
    }
    /// The equipment a fresh spawn carries: (slot ordinal, stack).
    fn equipment(&self) -> Option<(u8, crate::inventory::ItemStack)> {
        None
    }
    /// Registers the behavior and target goals at their priorities.
    fn register_goals(&self, goals: &mut GoalSelector, targets: &mut GoalSelector);
    /// The per-tick kind hook (daylight burning and kin).
    fn kind_tick(&mut self, ctx: &mut GoalCtx);
}

/// Everything the goals and the tracker read or write, minus the
/// selectors themselves.
pub struct MobBody {
    pub id: i32,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    /// Eye height above the feet (the kind's).
    pub eye: f64,
    /// Body yaw, degrees.
    pub yaw: f32,
    /// Head yaw, degrees.
    pub head_yaw: f32,
    pub pitch: f32,
    pub on_ground: bool,
    pub health: f32,
    pub max_health: f32,
    /// Hurt-flash ticks remaining.
    pub hurt_time: i32,
    /// Invulnerability cooldown after a full hit.
    pub damage_cooldown: i32,
    /// The damage the cooldown remembers.
    pub last_hurt: f32,
    /// Ticks since death started; 0 while alive.
    pub death_time: i32,
    /// Ticks without a hit or a nearby player.
    pub no_action_time: i32,
    /// Fire ticks remaining.
    pub fire_ticks: i32,
    /// Cached column sky exposure, refreshed while `ttl` counts down.
    pub exposed: bool,
    pub exposure_ttl: i32,
    /// The jump control cooldown.
    pub jump_cooldown: i32,
    /// The current navigation.
    pub nav: crate::pathing::Nav,
    /// The wanted look yaw/pitch, when a goal steers the head.
    pub look: Option<(f32, f32)>,
    /// The targeted player connection.
    pub target: Option<ConnId>,
    /// Whether the melee goal holds the move lock.
    pub melee_active: bool,
    /// A melee hit awaiting the wire send.
    pub pending_hit: Option<ConnId>,
    /// A bow shot awaiting the spawn: the target and the draw power.
    pub pending_shot: Option<(ConnId, f64)>,
    /// A detonation awaiting the blast: the radius.
    pub pending_blast: Option<f64>,
    /// Whether the last move clipped horizontally.
    pub horiz_collided: bool,
    /// The climbing state (the wall-crawler's metadata bit).
    pub climbing: bool,
    /// The swell direction: -1 shrinking, 1 swelling (the creeper).
    pub swell_dir: i32,
    /// The fuse counter (the creeper).
    pub fuse: i32,
    /// An instant removal without the death animation.
    pub discard: bool,
}

impl MobBody {
    fn new(id: i32, x: f64, y: f64, z: f64, max_health: f32) -> MobBody {
        MobBody {
            id,
            x,
            y,
            z,
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
            eye: 1.74,
            yaw: 0.0,
            head_yaw: 0.0,
            pitch: 0.0,
            on_ground: true,
            health: max_health,
            max_health,
            hurt_time: 0,
            damage_cooldown: 0,
            last_hurt: 0.0,
            death_time: 0,
            no_action_time: 0,
            fire_ticks: 0,
            exposed: false,
            exposure_ttl: 0,
            jump_cooldown: 0,
            nav: crate::pathing::Nav::new(),
            look: None,
            target: None,
            melee_active: false,
            pending_hit: None,
            pending_shot: None,
            pending_blast: None,
            horiz_collided: false,
            climbing: false,
            swell_dir: -1,
            fuse: 0,
            discard: false,
        }
    }
}

/// One mob: the body, the kind, and the two selectors.
pub struct Mob {
    pub uuid: [u8; 16],
    pub body: MobBody,
    pub kind: Box<dyn MobKind>,
    goals: GoalSelector,
    targets: GoalSelector,
    /// The mob's RNG stream seed.
    rand: u64,
    /// Ticks since spawn.
    tick: i64,
    /// Quantized last-synced position (1/4096 blocks).
    sent: (i64, i64, i64),
    /// Packed last-synced rotation bytes: yaw, pitch, head.
    sent_rot: (u8, u8, u8),
    sent_ground: bool,
    /// Health bits last sent.
    sent_health: u32,
    /// Ticks since the last full position sync.
    teleport_delay: i64,
    /// The sync cadence counter.
    sync_phase: i64,
    /// The last sent entity/mob flag bytes.
    sent_flags: (u8, u8),
    /// The last sent climbing state.
    sent_climbing: bool,
    /// The last sent swell direction.
    sent_swell: i32,
}

impl Mob {
    /// Builds a mob from its kind at a spawn position.
    pub fn new(
        id: i32,
        uuid: [u8; 16],
        x: f64,
        y: f64,
        z: f64,
        kind: Box<dyn MobKind>,
        rand_seed: u64,
    ) -> Mob {
        let mut goals = GoalSelector::new();
        let mut targets = GoalSelector::new();
        kind.register_goals(&mut goals, &mut targets);
        let max_health = kind.max_health();
        let (climb, eye) = (kind.can_climb(), kind.eye());
        let mut body = MobBody::new(id, x, y, z, max_health);
        body.eye = eye;
        body.nav.set_climb(climb);
        Mob {
            uuid,
            body,
            kind,
            goals,
            targets,
            rand: rand_seed,
            tick: 0,
            sent: (
                (x * DELTA_SCALE).round() as i64,
                (y * DELTA_SCALE).round() as i64,
                (z * DELTA_SCALE).round() as i64,
            ),
            sent_rot: (0, 0, 0),
            sent_ground: true,
            sent_health: max_health.to_bits(),
            teleport_delay: 0,
            sync_phase: 0,
            sent_flags: (0, 0),
            sent_climbing: false,
            sent_swell: -1,
        }
    }

    /// One uniform draw in [0, n) from the mob's stream.
    fn draw(&mut self, n: u64) -> u64 {
        self.rand = self.rand.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.rand;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        (z ^ (z >> 31)) % n
    }

    /// The spawn pairing: add_entity, the health datum, the attribute
    /// snapshot (the reference omits default-valued attributes), then
    /// the equipment a kind carries.
    pub fn pairing_frames(&self) -> Vec<(i32, Vec<u8>)> {
        let mut attrs = vec![(ATTR_MOVEMENT_SPEED, self.kind.base_speed())];
        if self.kind.max_health() != DEFAULT_MAX_HEALTH {
            attrs.push((ATTR_MAX_HEALTH, self.kind.max_health() as f64));
        }
        let mut frames = vec![
            (
                PACKET_ADD_ENTITY,
                encode_add_entity(
                    self.body.id,
                    &self.uuid,
                    self.kind.type_id(),
                    self.body.x,
                    self.body.y,
                    self.body.z,
                    (self.body.vx, self.body.vy, self.body.vz),
                    self.body.yaw,
                    self.body.pitch,
                    self.body.head_yaw,
                    0,
                ),
            ),
            (
                PACKET_SET_ENTITY_DATA,
                encode_float_data(self.body.id, DATA_LIVING_HEALTH, self.body.health),
            ),
            (
                PACKET_UPDATE_ATTRIBUTES,
                encode_update_attributes(self.body.id, &attrs),
            ),
        ];
        if let Some((slot, stack)) = self.kind.equipment() {
            frames.push((
                PACKET_SET_EQUIPMENT,
                encode_equipment(self.body.id, &[(slot, &stack)]),
            ));
        }
        frames
    }

    /// Damage plus knockback under the partial-hit rule; returns true
    /// when this hit starts death.
    pub fn hurt(&mut self, damage: f32, kx: f64, kz: f64) -> bool {
        if self.body.death_time > 0 {
            return false;
        }
        self.body.no_action_time = 0;
        if self.body.damage_cooldown > HURT_COOLDOWN - HURT_TIME {
            let extra = damage - self.body.last_hurt;
            if extra <= 0.0 {
                return false;
            }
            self.body.health -= extra;
        } else {
            self.body.last_hurt = damage;
            self.body.damage_cooldown = HURT_COOLDOWN;
            self.body.hurt_time = HURT_TIME;
            self.body.health -= damage;
            // Knockback: 0.4 away from the hit direction.
            let power = 0.4f64;
            let len = (kx * kx + kz * kz).sqrt();
            if len > 1.0e-4 {
                let (dx, dz) = (kx / len, kz / len);
                let rise = if self.body.on_ground {
                    (self.body.vy / 2.0 + power).min(0.4)
                } else {
                    self.body.vy
                };
                self.body.vx = self.body.vx / 2.0 - dx * power;
                self.body.vz = self.body.vz / 2.0 - dz * power;
                self.body.vy = rise;
            }
        }
        if self.body.health <= 0.0 {
            self.body.health = 0.0;
            self.body.death_time = 1;
            true
        } else {
            false
        }
    }
}

/// One queued mob broadcast, range-filtered at drain time.
pub struct OutFrame {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub id: i32,
    pub body: Vec<u8>,
}

// ---------------------------------------------------------------------
// Physics
// ---------------------------------------------------------------------

/// One ground movement step: input accel along the body yaw, axis-
/// separated collision with a 1-block step-up and jump, gravity,
/// friction, and the small-vector clamp.
fn step_mob(
    world: &Game,
    body: &mut MobBody,
    half: f64,
    height: f64,
    base_speed: f64,
    forward: f64,
) {
    body.horiz_collided = false;
    if forward > 0.0 {
        let speed = forward * base_speed;
        let accel = speed * base_speed;
        let yaw = body.yaw.to_radians() as f64;
        body.vx += -yaw.sin() * accel;
        body.vz += yaw.cos() * accel;
    }
    // Horizontal move with step-up.
    for axis in 0..2 {
        let v = if axis == 0 { body.vx } else { body.vz };
        if v == 0.0 {
            continue;
        }
        let (nx, nz) = if axis == 0 {
            (body.x + v, body.z)
        } else {
            (body.x, body.z + v)
        };
        let feet_y = body.y.floor() as i32;
        let mid = body.y + height / 2.0;
        let solid = |cx: f64, cy: i32, cz: f64| {
            block_solid(world, cx.floor() as i32, cy, cz.floor() as i32)
        };
        let corner_clear = |cx: f64, cy: i32, cz: f64| {
            !solid(cx - half, cy, cz - half)
                && !solid(cx + half, cy, cz - half)
                && !solid(cx - half, cy, cz + half)
                && !solid(cx + half, cy, cz + half)
        };
        if corner_clear(nx, mid.floor() as i32, nz) {
            body.x = nx;
            body.z = nz;
            continue;
        }
        // Step-up: a 1-block rise with headroom clears a single block.
        if body.on_ground && corner_clear(nx, feet_y + 1, nz) && corner_clear(nx, feet_y + 2, nz) {
            body.x = nx;
            body.z = nz;
            body.y = (feet_y + 1) as f64;
            continue;
        }
        body.horiz_collided = true;
        if body.on_ground && body.jump_cooldown == 0 {
            body.vy = body.vy.max(JUMP_POWER);
            body.jump_cooldown = JUMP_DELAY;
        }
        if axis == 0 {
            body.vx = 0.0;
        } else {
            body.vz = 0.0;
        }
    }
    // Vertical move.
    let ny = body.y + body.vy;
    if body.vy < 0.0 {
        let lands = |cx: f64, cz: f64| {
            block_solid(
                world,
                cx.floor() as i32,
                ny.floor() as i32,
                cz.floor() as i32,
            )
        };
        if lands(body.x - half, body.z - half)
            || lands(body.x + half, body.z - half)
            || lands(body.x - half, body.z + half)
            || lands(body.x + half, body.z + half)
        {
            body.y = ny.floor() + 1.0;
            body.vy = 0.0;
            body.on_ground = true;
        } else {
            body.y = ny;
            body.on_ground = false;
        }
    } else {
        let top = (body.y + height + body.vy).floor() as i32;
        if block_solid(world, body.x.floor() as i32, top, body.z.floor() as i32) {
            body.vy = 0.0;
        } else {
            body.y = ny;
            body.on_ground = false;
        }
    }
    // Gravity after the move, then drag and the clamp.
    body.vy -= GRAVITY;
    body.vx *= H_FRICTION;
    body.vz *= H_FRICTION;
    body.vy *= V_DRAG;
    if body.vx.abs() < V_MIN {
        body.vx = 0.0;
    }
    if body.vz.abs() < V_MIN {
        body.vz = 0.0;
    }
    if body.on_ground && body.vy.abs() < V_MIN {
        body.vy = 0.0;
    }
}

/// Chases `want` from `now` by at most `step` degrees on the circle.
fn rotate_towards(now: f32, want: f32, step: f32) -> f32 {
    let mut diff = (want - now + 180.0).rem_euclid(360.0) - 180.0;
    if diff > step {
        diff = step;
    }
    if diff < -step {
        diff = -step;
    }
    now + diff
}

/// The yaw that faces `dx`, `dz` (the standard packed-degree sense).
fn facing_yaw(dx: f64, dz: f64) -> f32 {
    (-dx).atan2(dz).to_degrees() as f32
}

// ---------------------------------------------------------------------
// Game hooks
// ---------------------------------------------------------------------

/// Mob state owned by the game thread.
pub(crate) struct MobState {
    pub mobs: Vec<Mob>,
    /// Seed for the mob RNG splitmix stream.
    pub seed: u64,
    /// Mob melee damage absorbed per player; at a full health bar the
    /// player is down and mobs stop targeting it.
    pub player_damage: std::collections::BTreeMap<ConnId, f32>,
}

impl Default for MobState {
    fn default() -> Self {
        MobState {
            mobs: Vec::new(),
            seed: 0x5eed_0000,
            player_damage: Default::default(),
        }
    }
}

impl MobState {
    /// One splitmix draw.
    pub fn next(&mut self) -> u64 {
        self.seed = self.seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

impl Game {
    /// Sends one frame to every player within `range` of the position.
    pub(crate) fn send_within(&mut self, x: f64, y: f64, z: f64, range: f64, id: i32, body: &[u8]) {
        let targets: Vec<ConnId> = self
            .players
            .iter()
            .filter(|(_, p)| {
                let (dx, dy, dz) = (p.x - x, p.y - y, p.z - z);
                dx * dx + dy * dy + dz * dz < range * range
            })
            .map(|(&conn, _)| conn)
            .collect();
        for conn in targets {
            self.send(conn, id, body);
        }
    }

    /// Spawns a mob: allocate the id and uuid, broadcast the pairing.
    pub(crate) fn spawn_mob(&mut self, x: f64, y: f64, z: f64, kind: Box<dyn MobKind>) {
        let id = self.alloc_entity_id();
        let uuid = self.next_uuid();
        let seed = self.mobs.next();
        let mob = Mob::new(id, uuid, x, y, z, kind, seed);
        let frames = mob.pairing_frames();
        self.mobs.mobs.push(mob);
        for (pid, body) in frames {
            self.send_within(x, y, z, MOB_TRACK_RANGE, pid, &body);
        }
    }

    /// The per-tick mob pass: death, despawn checks, AI, physics, sync.
    pub(crate) fn tick_mobs(&mut self) {
        let mut frames: Vec<OutFrame> = Vec::new();
        let mut removed: Vec<usize> = Vec::new();
        let mut mobs = std::mem::take(&mut self.mobs.mobs);
        for (i, mob) in mobs.iter_mut().enumerate() {
            if mob.body.discard {
                frames.push(OutFrame {
                    x: mob.body.x,
                    y: mob.body.y,
                    z: mob.body.z,
                    id: PACKET_REMOVE_ENTITIES,
                    body: encode_remove_entities(&[mob.body.id]),
                });
                removed.push(i);
                continue;
            }
            if mob.body.death_time > 0 {
                if mob.body.death_time == 1 {
                    frames.push(OutFrame {
                        x: mob.body.x,
                        y: mob.body.y,
                        z: mob.body.z,
                        id: PACKET_ENTITY_EVENT,
                        body: encode_entity_event(mob.body.id, EVENT_DEATH),
                    });
                }
                mob.body.death_time += 1;
                if mob.body.death_time >= DEATH_TICKS {
                    frames.push(OutFrame {
                        x: mob.body.x,
                        y: mob.body.y,
                        z: mob.body.z,
                        id: PACKET_ENTITY_EVENT,
                        body: encode_entity_event(mob.body.id, EVENT_DEATH_FINISH),
                    });
                    frames.push(OutFrame {
                        x: mob.body.x,
                        y: mob.body.y,
                        z: mob.body.z,
                        id: PACKET_REMOVE_ENTITIES,
                        body: encode_remove_entities(&[mob.body.id]),
                    });
                    removed.push(i);
                }
                continue;
            }
            if Self::despawn_check(self, mob, &mut frames) {
                removed.push(i);
                continue;
            }
            self.mob_ai(mob, &mut frames);
        }
        let mut blasts: Vec<(i32, f64, f64, f64, f64)> = Vec::new();
        for mob in mobs.iter_mut() {
            if let Some(radius) = mob.body.pending_blast.take() {
                blasts.push((mob.body.id, mob.body.x, mob.body.y, mob.body.z, radius));
            }
        }
        if removed.is_empty() {
            self.mobs.mobs = mobs;
        } else {
            let mut kept: Vec<Mob> = Vec::with_capacity(mobs.len() - removed.len());
            for (i, mob) in mobs.into_iter().enumerate() {
                if !removed.contains(&i) {
                    kept.push(mob);
                }
            }
            self.mobs.mobs = kept;
        }
        // The detonations the fuses reached: they run with the mob
        // list whole, so the blast sees every mob, and the spent
        // creeper leaves without a corpse.
        for (id, x, y, z, radius) in blasts {
            self.explode_at(x, y, z, radius, id);
            if let Some(mob) = self.mobs.mobs.iter_mut().find(|m| m.body.id == id) {
                mob.body.discard = true;
            }
        }
        for frame in &frames {
            self.send_within(
                frame.x,
                frame.y,
                frame.z,
                MOB_TRACK_RANGE,
                frame.id,
                &frame.body,
            );
        }
    }

    /// True when the mob leaves the world this tick: peaceful
    /// difficulty, no players, beyond the instant-despawn distance, or
    /// the random roll past the no-action limit.
    fn despawn_check(g: &Game, mob: &mut Mob, frames: &mut Vec<OutFrame>) -> bool {
        if g.spawning.peaceful {
            frames.push(OutFrame {
                x: mob.body.x,
                y: mob.body.y,
                z: mob.body.z,
                id: PACKET_REMOVE_ENTITIES,
                body: encode_remove_entities(&[mob.body.id]),
            });
            return true;
        }
        let mut nearest = f64::INFINITY;
        for p in g.players.values() {
            let d2 = (p.x - mob.body.x) * (p.x - mob.body.x)
                + (p.y - mob.body.y) * (p.y - mob.body.y)
                + (p.z - mob.body.z) * (p.z - mob.body.z);
            nearest = nearest.min(d2);
        }
        if nearest == f64::INFINITY || nearest > DESPAWN_DISTANCE_SQ {
            frames.push(OutFrame {
                x: mob.body.x,
                y: mob.body.y,
                z: mob.body.z,
                id: PACKET_REMOVE_ENTITIES,
                body: encode_remove_entities(&[mob.body.id]),
            });
            return true;
        }
        if nearest <= NO_DESPAWN_DISTANCE_SQ {
            mob.body.no_action_time = 0;
        }
        if mob.body.no_action_time > NO_ACTION_LIMIT
            && nearest > NO_DESPAWN_DISTANCE_SQ
            && mob.draw(DESPAWN_ROLL) == 0
        {
            frames.push(OutFrame {
                x: mob.body.x,
                y: mob.body.y,
                z: mob.body.z,
                id: PACKET_REMOVE_ENTITIES,
                body: encode_remove_entities(&[mob.body.id]),
            });
            return true;
        }
        false
    }

    /// AI, movement, and the tracker sync for one living mob.
    fn mob_ai(&mut self, mob: &mut Mob, frames: &mut Vec<OutFrame>) {
        mob.tick += 1;
        mob.body.no_action_time += 1;
        if mob.body.hurt_time > 0 {
            mob.body.hurt_time -= 1;
        }
        if mob.body.damage_cooldown > 0 {
            mob.body.damage_cooldown -= 1;
        }
        if mob.body.jump_cooldown > 0 {
            mob.body.jump_cooldown -= 1;
        }
        // Fire: ticks down, damages every 20 ticks.
        let burning = mob.body.fire_ticks > 0;
        if burning {
            mob.body.fire_ticks -= 1;
            if mob.body.fire_ticks % 20 == 0 {
                mob.hurt(1.0, 0.0, 0.0);
            }
        }
        // Cached sky exposure for the brightness checks.
        if mob.body.exposure_ttl <= 0 {
            let eye = (
                mob.body.x.floor() as i32,
                (mob.body.y + 1.0) as i32,
                mob.body.z.floor() as i32,
            );
            mob.body.exposed = self.sky_exposed(eye.0, eye.1, eye.2);
            mob.body.exposure_ttl = 20;
        }
        mob.body.exposure_ttl -= 1;
        // Selectors: the full pass every second tick offset by the
        // entity id; the target selector runs before the behaviors.
        let full = (mob.tick + mob.body.id as i64) % 2 == 0 || mob.tick <= 1;
        let mut goals = std::mem::take(&mut mob.goals);
        let mut targets = std::mem::take(&mut mob.targets);
        let mut rand = mob.rand;
        {
            let body = &mut mob.body;
            let kind = &mut *mob.kind;
            let mut ctx = GoalCtx {
                world: self,
                body,
                rand: &mut rand,
            };
            targets.tick(full, &mut ctx);
            goals.tick(full, &mut ctx);
            kind.kind_tick(&mut ctx);
        }
        mob.rand = rand;
        mob.goals = goals;
        mob.targets = targets;
        // A bow shot the goal queued: the skeleton module aims and
        // spawns the arrow.
        if let Some((conn, power)) = mob.body.pending_shot.take() {
            let (x, y, z, eye, zid) = (
                mob.body.x,
                mob.body.y,
                mob.body.z,
                mob.body.eye,
                mob.body.id,
            );
            let target = self.players.get(&conn).map(|p| (p.x, p.y, p.z));
            if let Some(target) = target {
                let mut seed = mob.rand;
                crate::skeleton::fire_shot(self, x, y, z, eye, zid, target, power, &mut seed);
                mob.rand = seed;
            }
        }
        // A melee hit the goal queued: damage_event to the target plus
        // the shared hurt_animation, and the damage counts toward the
        // target's health bar.
        if let Some(conn) = mob.body.pending_hit.take() {
            let (zx, zy, zz, zid) = (mob.body.x, mob.body.y, mob.body.z, mob.body.id);
            let damage = mob.kind.attack_damage();
            if let Some(p) = self.players.get(&conn) {
                let (px, pz, pid) = (p.x, p.z, p.entity_id);
                let yaw = facing_yaw(zx - px, zz - pz);
                frames.push(OutFrame {
                    x: zx,
                    y: zy,
                    z: zz,
                    id: PACKET_HURT_ANIMATION,
                    body: encode_hurt_animation(pid, yaw),
                });
                frames.push(OutFrame {
                    x: zx,
                    y: zy,
                    z: zz,
                    id: PACKET_DAMAGE_EVENT,
                    body: encode_damage_event(pid, DAMAGE_TYPE_MOB_ATTACK, zid, zid),
                });
            }
            *self.mobs.player_damage.entry(conn).or_insert(0.0) += damage;
        }
        // Look control: the wanted head yaw, else the body yaw, clamped
        // within 75 degrees of the body.
        let (want_yaw, want_pitch) = mob.body.look.unwrap_or((mob.body.yaw, 0.0));
        mob.body.head_yaw = rotate_towards(mob.body.head_yaw, want_yaw, 30.0);
        mob.body.pitch = rotate_towards(mob.body.pitch, want_pitch, 30.0);
        let over = (mob.body.head_yaw - mob.body.yaw)
            .rem_euclid(360.0)
            .min((mob.body.yaw - mob.body.head_yaw).rem_euclid(360.0));
        if over > 75.0 {
            mob.body.head_yaw = rotate_towards(mob.body.head_yaw, mob.body.yaw, over - 75.0);
        }
        // Move control: run the navigation housekeeping, face the
        // wanted waypoint, walk forward.
        let (bx, by, bz, bground) = (mob.body.x, mob.body.y, mob.body.z, mob.body.on_ground);
        mob.body
            .nav
            .nav_tick(bx, by, bz, bground, &|x, y, z| block_solid(self, x, y, z));
        let mut forward = 0.0f64;
        if let Some((tx, tz, modifier)) = mob.body.nav.wanted() {
            mob.body.yaw = rotate_towards(
                mob.body.yaw,
                facing_yaw(tx - mob.body.x, tz - mob.body.z),
                TURN_RATE,
            );
            forward = modifier;
            if mob.body.nav.arrived(mob.body.x, mob.body.z) {
                mob.body.nav.stop();
            }
        }
        let half = mob.kind.half_width();
        let height = mob.kind.height();
        let base_speed = mob.kind.base_speed();
        step_mob(self, &mut mob.body, half, height, base_speed, forward);
        // The wall-crawl rule: pressed into a wall, the body rises.
        if mob.kind.can_climb() {
            mob.body.climbing = mob.body.horiz_collided && mob.body.nav.in_progress();
            if mob.body.climbing {
                mob.body.vy = mob.body.vy.max(CLIMB_RISE);
            }
        }
        mob_sync(mob, frames);
    }
}

/// The tracker-side per-tick sync: interval-3 movement (delta packets;
/// full sync on ground flips, out-of-range deltas, and the 400-tick
/// resync), rotate_head on packed-byte change, and dirty entity data.
fn mob_sync(mob: &mut Mob, frames: &mut Vec<OutFrame>) {
    let qx = (mob.body.x * DELTA_SCALE).round() as i64;
    let qy = (mob.body.y * DELTA_SCALE).round() as i64;
    let qz = (mob.body.z * DELTA_SCALE).round() as i64;
    let rot = (
        pack_degrees(mob.body.yaw),
        pack_degrees(mob.body.pitch),
        pack_degrees(mob.body.head_yaw),
    );
    mob.teleport_delay += 1;
    mob.sync_phase += 1;
    let moved = (qx, qy, qz) != mob.sent;
    let rot_changed = (rot.0, rot.1) != (mob.sent_rot.0, mob.sent_rot.1);
    let head_changed = rot.2 != mob.sent_rot.2;
    let ground_flip = mob.body.on_ground != mob.sent_ground;
    let over_delay = mob.teleport_delay > TELEPORT_DELAY_MAX;
    let out_of_range = |d: i64| d < i16::MIN as i64 || d > i16::MAX as i64;
    let (dx, dy, dz) = (qx - mob.sent.0, qy - mob.sent.1, qz - mob.sent.2);
    let delta_ok = !out_of_range(dx) && !out_of_range(dy) && !out_of_range(dz);
    let on_cadence = mob.sync_phase % SYNC_INTERVAL == 0;
    if ground_flip || over_delay || !delta_ok {
        frames.push(OutFrame {
            x: mob.body.x,
            y: mob.body.y,
            z: mob.body.z,
            id: PACKET_ENTITY_POSITION_SYNC,
            body: encode_position_sync(
                mob.body.id,
                mob.body.x,
                mob.body.y,
                mob.body.z,
                mob.body.yaw,
                mob.body.pitch,
                mob.body.on_ground,
            ),
        });
        mob.sent = (qx, qy, qz);
        mob.sent_rot = rot;
        mob.sent_ground = mob.body.on_ground;
        mob.sent_health = u32::MAX; // the full sync re-pairs the data
        mob.teleport_delay = 0;
    } else if on_cadence && (moved || rot_changed) {
        if moved && rot_changed {
            frames.push(OutFrame {
                x: mob.body.x,
                y: mob.body.y,
                z: mob.body.z,
                id: PACKET_MOVE_ENTITY_POS_ROT,
                body: encode_move_pos_rot(
                    mob.body.id,
                    dx,
                    dy,
                    dz,
                    rot.0,
                    rot.1,
                    mob.body.on_ground,
                ),
            });
            mob.sent_rot = (rot.0, rot.1, mob.sent_rot.2);
        } else if moved {
            frames.push(OutFrame {
                x: mob.body.x,
                y: mob.body.y,
                z: mob.body.z,
                id: PACKET_MOVE_ENTITY_POS,
                body: encode_move_pos(mob.body.id, dx, dy, dz, mob.body.on_ground),
            });
        } else {
            frames.push(OutFrame {
                x: mob.body.x,
                y: mob.body.y,
                z: mob.body.z,
                id: PACKET_MOVE_ENTITY_ROT,
                body: encode_move_rot(mob.body.id, rot.0, rot.1, mob.body.on_ground),
            });
            mob.sent_rot = (rot.0, rot.1, mob.sent_rot.2);
        }
        mob.sent = (qx, qy, qz);
        mob.sent_ground = mob.body.on_ground;
    }
    // Head rotation follows its own packed-byte change.
    if head_changed && (on_cadence || rot_changed || moved) {
        frames.push(OutFrame {
            x: mob.body.x,
            y: mob.body.y,
            z: mob.body.z,
            id: PACKET_ROTATE_HEAD,
            body: encode_rotate_head(mob.body.id, rot.2),
        });
        mob.sent_rot.2 = rot.2;
    }
    // Entity data: the fire bit, the aggressive bit, and health when
    // they sit off what was last sent.
    let flags = if mob.body.fire_ticks > 0 { 0x01 } else { 0x00 };
    let mob_flags = if mob.body.melee_active { 0x04 } else { 0x00 };
    if flags != mob.sent_flags.0
        || mob_flags != mob.sent_flags.1
        || mob.body.health.to_bits() != mob.sent_health
    {
        if flags != mob.sent_flags.0 {
            frames.push(OutFrame {
                x: mob.body.x,
                y: mob.body.y,
                z: mob.body.z,
                id: PACKET_SET_ENTITY_DATA,
                body: encode_byte_data(mob.body.id, DATA_ENTITY_FLAGS, flags),
            });
            mob.sent_flags.0 = flags;
        }
        if mob_flags != mob.sent_flags.1 {
            frames.push(OutFrame {
                x: mob.body.x,
                y: mob.body.y,
                z: mob.body.z,
                id: PACKET_SET_ENTITY_DATA,
                body: encode_byte_data(mob.body.id, DATA_MOB_FLAGS, mob_flags),
            });
            mob.sent_flags.1 = mob_flags;
        }
        if mob.body.health.to_bits() != mob.sent_health {
            frames.push(OutFrame {
                x: mob.body.x,
                y: mob.body.y,
                z: mob.body.z,
                id: PACKET_SET_ENTITY_DATA,
                body: encode_float_data(mob.body.id, DATA_LIVING_HEALTH, mob.body.health),
            });
            mob.sent_health = mob.body.health.to_bits();
        }
        // The creeper's swell direction rides its own accessor.
        if mob.body.swell_dir != mob.sent_swell {
            frames.push(OutFrame {
                x: mob.body.x,
                y: mob.body.y,
                z: mob.body.z,
                id: PACKET_SET_ENTITY_DATA,
                body: encode_int_data(mob.body.id, DATA_SWELL_DIR, mob.body.swell_dir),
            });
            mob.sent_swell = mob.body.swell_dir;
        }
        // The wall-crawler's climbing bit rides its own accessor.
        if mob.kind.can_climb() && mob.body.climbing != mob.sent_climbing {
            let value = if mob.body.climbing { 0x01 } else { 0x00 };
            frames.push(OutFrame {
                x: mob.body.x,
                y: mob.body.y,
                z: mob.body.z,
                id: PACKET_SET_ENTITY_DATA,
                body: encode_byte_data(mob.body.id, DATA_CLIMBING_FLAGS, value),
            });
            mob.sent_climbing = mob.body.climbing;
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Game, Inbound, Outbound};
    use crate::zombie::Zombie;
    use doppel_world::WireChunk;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::mpsc;
    use std::sync::Arc;

    /// A bare game: no chunks, no players (selector tests need none).
    fn bare_game() -> Game {
        let (_tx, rx) = mpsc::channel::<Inbound>();
        Game::new(rx, None, None)
    }

    #[derive(Default)]
    struct Tally {
        started: AtomicU32,
        stopped: AtomicU32,
        ticked: AtomicU32,
    }

    /// A scripted goal the test flips from outside.
    struct Probe {
        flags: GoalFlags,
        usable: Arc<AtomicBool>,
        cont: Arc<AtomicBool>,
        tally: Arc<Tally>,
        every: bool,
    }

    impl Goal for Probe {
        fn flags(&self) -> GoalFlags {
            self.flags
        }
        fn can_use(&mut self, _ctx: &mut GoalCtx) -> bool {
            self.usable.load(Ordering::Relaxed)
        }
        fn can_continue_to_use(&mut self, _ctx: &mut GoalCtx) -> bool {
            self.cont.load(Ordering::Relaxed)
        }
        fn start(&mut self, _ctx: &mut GoalCtx) {
            self.tally.started.fetch_add(1, Ordering::Relaxed);
        }
        fn stop(&mut self, _ctx: &mut GoalCtx) {
            self.tally.stopped.fetch_add(1, Ordering::Relaxed);
        }
        fn tick(&mut self, _ctx: &mut GoalCtx) {
            self.tally.ticked.fetch_add(1, Ordering::Relaxed);
        }
        fn requires_every_tick(&self) -> bool {
            self.every
        }
    }

    /// The goal plus the switches: usable, cont, tally.
    fn probe(
        flags: GoalFlags,
        usable: bool,
        cont: bool,
        every: bool,
    ) -> (Probe, Arc<AtomicBool>, Arc<AtomicBool>, Arc<Tally>) {
        let usable = Arc::new(AtomicBool::from(usable));
        let cont = Arc::new(AtomicBool::from(cont));
        let tally = Arc::<Tally>::default();
        (
            Probe {
                flags,
                usable: usable.clone(),
                cont: cont.clone(),
                tally: tally.clone(),
                every,
            },
            usable,
            cont,
            tally,
        )
    }

    fn body() -> MobBody {
        MobBody::new(1, 0.0, 0.0, 0.0, 20.0)
    }

    fn run(g: &Game, b: &mut MobBody, rand: &mut u64, sel: &mut GoalSelector, full: bool) {
        let mut ctx = GoalCtx {
            world: g,
            body: b,
            rand,
        };
        sel.tick(full, &mut ctx);
    }

    #[test]
    fn smaller_priority_number_preempts() {
        let g = bare_game();
        let mut b = body();
        let mut rand = 1u64;
        let mut sel = GoalSelector::new();
        let (stroll, stroll_usable, _sc, stroll_tally) = probe(GoalFlags::MOVE, false, true, false);
        let (melee, melee_usable, _mc, melee_tally) = probe(GoalFlags::MOVE, false, true, false);
        sel.add(7, Box::new(stroll));
        sel.add(3, Box::new(melee));
        run(&g, &mut b, &mut rand, &mut sel, true);
        assert!(sel.running_priorities().is_empty());
        stroll_usable.store(true, Ordering::Relaxed);
        run(&g, &mut b, &mut rand, &mut sel, true);
        assert_eq!(sel.running_priorities(), vec![7]);
        melee_usable.store(true, Ordering::Relaxed);
        run(&g, &mut b, &mut rand, &mut sel, true);
        assert_eq!(sel.running_priorities(), vec![3]);
        assert_eq!(stroll_tally.stopped.load(Ordering::Relaxed), 1);
        assert_eq!(melee_tally.started.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn equal_priority_never_preempts() {
        let g = bare_game();
        let mut b = body();
        let mut rand = 1u64;
        let mut sel = GoalSelector::new();
        let (first, first_usable, _fc, first_tally) = probe(GoalFlags::MOVE, false, true, false);
        let (second, second_usable, _sc, second_tally) = probe(GoalFlags::MOVE, false, true, false);
        sel.add(8, Box::new(first));
        sel.add(8, Box::new(second));
        first_usable.store(true, Ordering::Relaxed);
        run(&g, &mut b, &mut rand, &mut sel, true);
        second_usable.store(true, Ordering::Relaxed);
        run(&g, &mut b, &mut rand, &mut sel, true);
        assert_eq!(sel.running_priorities(), vec![8]);
        assert_eq!(first_tally.stopped.load(Ordering::Relaxed), 0);
        assert_eq!(second_tally.started.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn stop_releases_flags_for_lower_priority() {
        let g = bare_game();
        let mut b = body();
        let mut rand = 1u64;
        let mut sel = GoalSelector::new();
        let (holder, holder_usable, holder_cont, holder_tally) =
            probe(GoalFlags::MOVE, true, true, false);
        let (waiter, _wu, _wc, waiter_tally) = probe(GoalFlags::MOVE, true, true, false);
        sel.add(3, Box::new(holder));
        sel.add(7, Box::new(waiter));
        run(&g, &mut b, &mut rand, &mut sel, true);
        assert_eq!(sel.running_priorities(), vec![3]);
        assert_eq!(waiter_tally.started.load(Ordering::Relaxed), 0);
        holder_cont.store(false, Ordering::Relaxed);
        // Also unusable now, or it restarts the same pass.
        holder_usable.store(false, Ordering::Relaxed);
        run(&g, &mut b, &mut rand, &mut sel, true);
        assert_eq!(sel.running_priorities(), vec![7]);
        assert_eq!(holder_tally.stopped.load(Ordering::Relaxed), 1);
        assert_eq!(waiter_tally.started.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn off_tick_ticks_only_every_tick_goals() {
        let g = bare_game();
        let mut b = body();
        let mut rand = 1u64;
        let mut sel = GoalSelector::new();
        let (plain, plain_usable, _pc, plain_tally) = probe(GoalFlags::MOVE, false, true, false);
        let (steady, steady_usable, _sc, steady_tally) = probe(GoalFlags::LOOK, false, true, true);
        sel.add(3, Box::new(plain));
        sel.add(8, Box::new(steady));
        plain_usable.store(true, Ordering::Relaxed);
        steady_usable.store(true, Ordering::Relaxed);
        run(&g, &mut b, &mut rand, &mut sel, true);
        let plain_before = plain_tally.ticked.load(Ordering::Relaxed);
        let steady_before = steady_tally.ticked.load(Ordering::Relaxed);
        run(&g, &mut b, &mut rand, &mut sel, false);
        assert_eq!(
            plain_tally.ticked.load(Ordering::Relaxed),
            plain_before,
            "the plain goal sleeps off cadence"
        );
        assert_eq!(
            steady_tally.ticked.load(Ordering::Relaxed),
            steady_before + 1,
            "the every-tick goal runs off cadence"
        );
    }

    #[test]
    fn disjoint_flags_run_concurrently() {
        let g = bare_game();
        let mut b = body();
        let mut rand = 1u64;
        let mut sel = GoalSelector::new();
        let (walker, walker_usable, _wc, _wt) = probe(GoalFlags::MOVE, false, true, false);
        let (watcher, watcher_usable, _lc, _lt) = probe(GoalFlags::LOOK, false, true, false);
        sel.add(7, Box::new(walker));
        sel.add(8, Box::new(watcher));
        walker_usable.store(true, Ordering::Relaxed);
        watcher_usable.store(true, Ordering::Relaxed);
        run(&g, &mut b, &mut rand, &mut sel, true);
        assert_eq!(sel.running_priorities(), vec![7, 8]);
    }

    // -- physics -------------------------------------------------------

    /// A grass-floored chunk (surface y=99) with an optional stone
    /// shelf one block up covering x >= 6 at y=100, plus one player.
    fn harness(wall: bool) -> (Game, mpsc::Receiver<Outbound>) {
        let (_tx, rx) = mpsc::channel::<Inbound>();
        let mut g = Game::new(rx, None, None);
        assert!(g.registry_for_test(), "pins/blocks.json not found");
        let grass = g.resolve_state("minecraft:grass_block").unwrap();
        let stone = g.resolve_state("minecraft:stone").unwrap();
        let mut w = WireChunk {
            x: 0,
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
                        let v: u64 = if (i >> 8) == 3 {
                            1
                        } else if wall && (i >> 8) == 4 && (i & 0xf) >= 6 {
                            2
                        } else {
                            0
                        };
                        *slot |= v << (j * 4);
                    }
                }
                doppel_world::chunk_codec::Container::Palette {
                    bits: 4,
                    entries: vec![0, grass, stone],
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
        g.seed_chunk_for_test(0, 0, w);
        let (tx_out, rx_out) = mpsc::channel::<Outbound>();
        g.join_viewer_for_test(0, &[(0, 0)], tx_out);
        (g, rx_out)
    }

    fn drain(rx: &mpsc::Receiver<Outbound>) -> Vec<(i32, Vec<u8>)> {
        let mut out = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            if let Outbound::Frame { id, body } = frame {
                out.push((id, body));
            }
        }
        out
    }

    #[test]
    fn step_up_climbs_a_one_block_rise() {
        let (g, _rx) = harness(true);
        let mut b = MobBody::new(5, 5.5, 100.0, 5.5, 20.0);
        // Face +x, straight at the shelf edge.
        b.yaw = -90.0;
        for _ in 0..40 {
            step_mob(&g, &mut b, 0.3, 1.95, 0.23, 1.0);
        }
        assert!(b.x > 6.9, "onto the shelf, x = {}", b.x);
        assert!((b.y - 101.0).abs() < 0.01, "on the shelf top, y = {}", b.y);
    }

    #[test]
    fn gravity_lands_a_falling_body() {
        let (g, _rx) = harness(false);
        let mut b = MobBody::new(5, 5.5, 110.0, 8.5, 20.0);
        for _ in 0..40 {
            step_mob(&g, &mut b, 0.3, 1.95, 0.23, 0.0);
        }
        assert!(b.on_ground);
        assert!((b.y - 100.0).abs() < 0.01, "feet on the grass, y = {}", b.y);
    }

    // -- pairing -------------------------------------------------------

    #[test]
    fn spawn_mob_pairs_and_tracks() {
        let (mut g, rx) = harness(false);
        g.handle(Inbound::Tp {
            conn: 0,
            x: 0.5,
            y: 100.0,
            z: 0.5,
        });
        g.spawn_mob(5.5, 100.0, 5.5, Box::new(Zombie::new()));
        g.flush_connections();
        assert_eq!(g.mobs.mobs.len(), 1);
        let (got_id, got_health) = {
            let mob = &g.mobs.mobs[0];
            assert_eq!(mob.kind.type_id(), ENTITY_TYPE_ZOMBIE);
            assert_eq!(
                (mob.body.vx, mob.body.vy, mob.body.vz),
                (0.0, 0.0, 0.0),
                "spawn velocity is zero"
            );
            (mob.body.id, mob.body.health)
        };
        let frames = drain(&rx);
        let ids: Vec<i32> = frames.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&PACKET_ADD_ENTITY), "{ids:?}");
        assert!(ids.contains(&PACKET_SET_ENTITY_DATA), "{ids:?}");
        assert!(ids.contains(&PACKET_UPDATE_ATTRIBUTES), "{ids:?}");
        let data = frames
            .iter()
            .find(|(id, _)| *id == PACKET_SET_ENTITY_DATA)
            .map(|(_, b)| b.clone())
            .unwrap();
        assert_eq!(
            data,
            encode_float_data(got_id, DATA_LIVING_HEALTH, got_health)
        );
    }

    #[test]
    fn hurt_partial_rule_and_death_start() {
        let mut m = Mob::new(2, [0u8; 16], 0.0, 0.0, 0.0, Box::new(Zombie::new()), 7);
        // A full hit: cooldown 20, flash 10, health 17.
        assert!(!m.hurt(3.0, 1.0, 0.0));
        assert_eq!(m.body.health, 17.0);
        assert_eq!(m.body.damage_cooldown, HURT_COOLDOWN);
        assert_eq!(m.body.hurt_time, HURT_TIME);
        // Inside the flash window a stronger hit lands only its excess.
        m.body.damage_cooldown = 15;
        assert!(!m.hurt(4.0, 1.0, 0.0));
        assert_eq!(m.body.health, 16.0, "only the excess over lastHurt");
        // A weaker hit inside the window lands nothing.
        m.body.damage_cooldown = 15;
        assert!(!m.hurt(3.0, 1.0, 0.0));
        assert_eq!(m.body.health, 16.0);
        // Beyond the window a hit is full again; enough to kill.
        m.body.damage_cooldown = 0;
        assert!(m.hurt(20.0, 0.0, 0.0), "the killing hit starts death");
        assert_eq!(m.body.death_time, 1);
        assert_eq!(m.body.health, 0.0);
        assert!(!m.hurt(5.0, 0.0, 0.0), "the dead take no damage");
    }

    #[test]
    fn death_removes_after_twenty_ticks() {
        let (mut g, rx) = harness(false);
        g.handle(Inbound::Tp {
            conn: 0,
            x: 0.5,
            y: 100.0,
            z: 0.5,
        });
        g.spawn_mob(5.5, 100.0, 5.5, Box::new(Zombie::new()));
        let id = g.mobs.mobs[0].body.id;
        assert!(g.mobs.mobs[0].hurt(20.0, 0.0, 0.0));
        for _ in 0..21 {
            g.tick_once_for_test();
        }
        assert!(g.mobs.mobs.is_empty(), "the corpse leaves the world");
        let frames = drain(&rx);
        let events: Vec<(i32, Vec<u8>)> = frames
            .iter()
            .filter(|(pid, _)| *pid == PACKET_ENTITY_EVENT)
            .cloned()
            .collect();
        assert!(
            events
                .iter()
                .any(|(_, b)| b == &encode_entity_event(id, EVENT_DEATH)),
            "the death animation fires"
        );
        assert!(
            events
                .iter()
                .any(|(_, b)| b == &encode_entity_event(id, EVENT_DEATH_FINISH)),
            "the finish event fires"
        );
        assert!(
            frames.iter().any(|(pid, b)| {
                *pid == PACKET_REMOVE_ENTITIES && b == &encode_remove_entities(&[id])
            }),
            "the removal lands on the wire"
        );
    }
}
