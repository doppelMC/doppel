//! The differential mobs gate: natural monster spawning plus the living
//! entity lifecycle. Both servers boot the flat world at the default
//! difficulty, turn the clock to midnight, and let the monster spawn
//! cycle run while a single opped bot observes. When the first zombie
//! appears the bot plants itself 26 blocks away (inside the follow
//! range, past the spawn exclusion) so the chase engages, and the
//! capture watches the approach. The checks are structural, not exact:
//! the zombie's add/pairing packet shapes, the spawn geometry rules
//! (block centers, integer feet, the 24..128 player window), and the
//! chase closing the distance.

use anyhow::{Context, Result};
use doppel_protocol::load_pin;
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::parity_break::{capture_clean_blobs, default_doppel_bin, wait_for_port};
use crate::{bot, capture, vanilla};

const VANILLA_PORT: u16 = 25566;
const DOPPEL_PORT: u16 = 25565;

/// Port overrides for local runs that share the machine with another
/// gate's servers; CI uses the defaults. An overridden port implies a
/// private vanilla run directory, whose boot-time wipe would otherwise
/// hit the shared one.
fn vanilla_port() -> u16 {
    match std::env::var("MOBS_VANILLA_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(port) => {
            vanilla::default_run_dir("mobs");
            port
        }
        None => VANILLA_PORT,
    }
}

/// The doppel-side override twin.
fn doppel_port() -> u16 {
    std::env::var("MOBS_DOPPEL_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DOPPEL_PORT)
}

/// minecraft:zombie in the entity-type registry (registration order 155,
/// 0-based). The gate asserts both servers' zombie adds carry it.
const ZOMBIE_TYPE: i32 = 154;
/// The wave-2 mob types (the same registration-order counting).
const SKELETON_TYPE: i32 = 118;
const CREEPER_TYPE: i32 = 32;
const SPIDER_TYPE: i32 = 127;
const ARROW_TYPE: i32 = 6;
/// Where the chase probe plants the bot: past the 24-block spawn
/// exclusion, inside the zombie's follow range (35).
const CHASE_STAND: f64 = 26.0;
/// The movement packets' delta unit: 1/4096-block shorts.
const DELTA_SCALE: f64 = 4096.0;
/// How far the chase must close for the trend check.
const CHASE_CLOSE: f64 = 2.0;

// Clientbound play packet ids (registration order minus one; the set is
// anchored by the survival gate's verified 0x01/0x23/0x36/0x37/0x4e/0x65).
const P_ADD_ENTITY: i32 = 0x01;
const P_ANIMATE: i32 = 0x02;
const P_SWING_ANIMATION: i32 = 0x7b;
const PACKET_REMOVE_ENTITIES_LOCAL: i32 = 0x4e;
const P_BLOCK_UPDATE: i32 = 0x08;
const P_SECTION_BLOCKS_UPDATE: i32 = 0x56;
const P_DAMAGE_EVENT: i32 = 0x19;
const P_EXPLODE: i32 = 0x24;
const P_SET_EQUIPMENT: i32 = 0x68;
const P_PLAYER_POSITION: i32 = 0x49;
const P_ENTITY_POSITION_SYNC: i32 = 0x23;
const P_MOVE_ENTITY_POS: i32 = 0x36;
const P_MOVE_ENTITY_POS_ROT: i32 = 0x37;
const P_MOVE_ENTITY_ROT: i32 = 0x39;
const P_SET_ENTITY_DATA: i32 = 0x65;
const P_ROTATE_HEAD: i32 = 0x55;
const P_UPDATE_ATTRIBUTES: i32 = 0x86;
const P_ENTITY_EVENT: i32 = 0x22;
/// The creeper's swell-direction accessor.
const DATA_SWELL_ACCESSOR: u8 = 16;

fn rd_varint(raw: &[u8], o: &mut usize) -> Option<i32> {
    let mut v: i32 = 0;
    let mut sh = 0u32;
    while *o < raw.len() {
        let b = raw[*o];
        *o += 1;
        v |= i32::from(b & 0x7f) << sh;
        sh += 7;
        if b & 0x80 == 0 {
            return Some(v);
        }
    }
    None
}

fn rd_f64(raw: &[u8], o: &mut usize) -> Option<f64> {
    if *o + 8 > raw.len() {
        return None;
    }
    let b: [u8; 8] = raw[*o..*o + 8].try_into().ok()?;
    *o += 8;
    Some(f64::from_be_bytes(b))
}

fn rd_i64(raw: &[u8], o: &mut usize) -> Option<i64> {
    if *o + 8 > raw.len() {
        return None;
    }
    let b: [u8; 8] = raw[*o..*o + 8].try_into().ok()?;
    *o += 8;
    Some(i64::from_be_bytes(b))
}

fn rd_f32(raw: &[u8], o: &mut usize) -> Option<f32> {
    if *o + 4 > raw.len() {
        return None;
    }
    let b: [u8; 4] = raw[*o..*o + 4].try_into().ok()?;
    *o += 4;
    Some(f32::from_be_bytes(b))
}

fn rd_i16(raw: &[u8], o: &mut usize) -> Option<i16> {
    if *o + 2 > raw.len() {
        return None;
    }
    let b: [u8; 2] = raw[*o..*o + 2].try_into().ok()?;
    *o += 2;
    Some(i16::from_be_bytes(b))
}

/// An update_attributes payload after the entity id: count, then (attr
/// id, f64 value, modifier count, modifiers) triples. A truncated list
/// yields what decoded, like the rest of the tolerant readers.
fn rd_attrs(raw: &[u8], o: &mut usize) -> Option<Vec<(i32, f64)>> {
    let count = rd_varint(raw, o)?;
    let mut attrs = Vec::new();
    for _ in 0..count.max(0) {
        let (Some(attr), Some(value), Some(mods)) =
            (rd_varint(raw, o), rd_f64(raw, o), rd_varint(raw, o))
        else {
            return Some(attrs);
        };
        attrs.push((attr, value));
        for _ in 0..mods.max(0) {
            // uuid + amount + operation
            *o += 16;
            if rd_f64(raw, o).is_none() || rd_varint(raw, o).is_none() {
                return Some(attrs);
            }
        }
    }
    Some(attrs)
}

fn rd_varlong(raw: &[u8], o: &mut usize) -> Option<i64> {
    let mut v: i64 = 0;
    let mut sh = 0u32;
    while *o < raw.len() {
        let b = raw[*o];
        *o += 1;
        v |= i64::from(b & 0x7f) << sh;
        sh += 7;
        if b & 0x80 == 0 {
            return Some(v);
        }
    }
    None
}

/// Degrees to the packed wire byte: deg * 256 / 360.
fn pack_byte(deg: f32) -> u8 {
    (deg * 256.0 / 360.0) as i8 as u8
}

/// A packed wire yaw byte back to degrees on the -180..180 circle.
fn unpack_yaw_deg(b: u8) -> f64 {
    (b as i8) as f64 * 360.0 / 256.0
}

/// Unpacks a wire block position: x 26 bits at 38, z 26 at 12, y 12.
fn unpack_block_pos(packed: i64) -> (i32, i32, i32) {
    let x = (packed >> 38) as i32;
    let z = ((packed << 26) >> 38) as i32;
    let y = ((packed << 52) >> 52) as i32;
    (x, y, z)
}

/// The bearing (yaw degrees) from `from` to `to` on the XZ plane, in the
/// packed wire sense: 0 faces +Z, -90 faces +X.
fn bearing(from: (f64, f64), to: (f64, f64)) -> f64 {
    let (dx, dz) = (to.0 - from.0, to.1 - from.1);
    -dx.atan2(dz).to_degrees()
}

/// The absolute angular distance on the yaw circle.
fn yaw_dist(a: f64, b: f64) -> f64 {
    ((a - b + 180.0).rem_euclid(360.0) - 180.0).abs()
}

/// Sign-extends the low `bits` of `v`.
fn sext(v: i64, bits: u32) -> i64 {
    let shift = 64 - bits;
    (v << shift) >> shift
}

/// Unpacks a wire section position: x 22 bits at 42, z 22 at 20, y 20
/// at 0.
fn unpack_section_pos(packed: i64) -> (i32, i32, i32) {
    (
        sext(packed >> 42, 22) as i32,
        sext(packed, 20) as i32,
        sext((packed >> 20) & 0x3F_FFFF, 22) as i32,
    )
}

/// Feeds the crater set from block_update / section_blocks_update frames:
/// every position whose new state reads as registry id 0 (air).
fn decode_air_updates(
    raw: &[u8],
    id: i32,
    crater: &mut std::collections::BTreeSet<(i32, i32, i32)>,
) {
    let mut o = 0usize;
    if id == P_BLOCK_UPDATE {
        let Some(packed) = rd_i64(raw, &mut o) else {
            return;
        };
        let state = rd_varint(raw, &mut o).unwrap_or(-1);
        if state == 0 {
            crater.insert(unpack_block_pos(packed));
        }
        return;
    }
    // section_blocks_update: packed section pos, count, then varlongs of
    // (state << 12 | local pos, y low nibble, z mid, x high).
    let Some(packed) = rd_i64(raw, &mut o) else {
        return;
    };
    let (sx, sy, sz) = unpack_section_pos(packed);
    let Some(count) = rd_varint(raw, &mut o) else {
        return;
    };
    for _ in 0..count.max(0) {
        let Some(change) = rd_varlong(raw, &mut o) else {
            return;
        };
        let state = (change >> 12) as i32;
        if state != 0 {
            continue;
        }
        let rel = (change & 0xfff) as i32;
        let (lx, lz, ly) = ((rel >> 8) & 0xf, (rel >> 4) & 0xf, rel & 0xf);
        crater.insert((sx * 16 + lx, sy * 16 + ly, sz * 16 + lz));
    }
}

/// One observed detonation: the packet index, the center, the radius,
/// the destroyed-block count, and the receiving player's own
/// knockback when the frame carries one.
type BlastFrame = (usize, f64, f64, f64, f32, i32, Option<(f64, f64, f64)>);

/// One stream's reduced observation of the session.
#[derive(Default)]
struct Obs {
    /// Player teleports from player_position frames: (packet index, x,
    /// y, z) in arrival order.
    bot_pos: Vec<(usize, f64, f64, f64)>,
    /// Every add_entity: (packet index, entity id, type, x, y, z, the
    /// byte after the positions).
    adds: Vec<(usize, i32, i32, f64, f64, f64, u8)>,
    /// set_entity_data entries: (packet index, entity id, accessor,
    /// serializer, value when the serializer is FLOAT).
    data: Vec<(usize, i32, u8, i32, Option<f32>)>,
    /// update_attributes payloads: (entity id, (attr id, value) pairs).
    attrs: Vec<(i32, Vec<(i32, f64)>)>,
    /// Absolute position syncs: (packet index, entity id, x, y, z).
    syncs: Vec<(usize, i32, f64, f64, f64)>,
    /// Delta moves: (packet index, entity id, dx, dy, dz, kind).
    /// Kind: true when the packet carried rotation bytes.
    deltas: Vec<(usize, i32, f64, f64, f64, bool)>,
    /// Body-yaw observations: (packet index, entity id, packed yaw byte)
    /// from move_entity_pos_rot, move_entity_rot, and position syncs.
    body_yaws: Vec<(usize, i32, u8)>,
    /// Damage events: (packet index, target entity id, damage type id).
    damage: Vec<(usize, i32, i32)>,
    /// rotate_head frames: (packet index, entity id, packed head yaw).
    head_rots: Vec<(usize, i32, u8)>,
    /// swing_animation frames: (packet index, entity id, hand, anim type,
    /// duration).
    swings: Vec<(usize, i32, i32, i32, i32)>,
    /// animate frames: (packet index, entity id, action byte).
    animates: Vec<(usize, i32, u8)>,
    /// entity_event frames: (packet index, entity id, event byte).
    entity_events: Vec<(usize, i32, u8)>,
    /// set_equipment leading entries: (entity id, slot byte).
    equipment: Vec<(i32, u8)>,
    /// explode packets: (packet index, center x, y, z, radius,
    /// destroyed count, the bot's own knockback when the frame
    /// carries one).
    explodes: Vec<BlastFrame>,
    /// Block positions the update stream turned to air: the crater set.
    crater: std::collections::BTreeSet<(i32, i32, i32)>,
    /// block_update plus section_blocks_update frame count.
    block_updates: usize,
    /// remove_entities payloads: (packet index, ids).
    removes: Vec<(usize, Vec<i32>)>,
    /// The chase probe's note, resolved after the loop: (packet index,
    /// entity id, bot x, y, z).
    chase: Option<(usize, i32, f64, f64, f64)>,
}

fn analyze(pkts: &[bot::CapturedPacket]) -> Obs {
    let mut obs = Obs::default();
    let mut chase_note: Option<(usize, f64, f64, f64)> = None;
    // The doppel transcript opens with the replayed join burst, whose
    // tracked entities would masquerade as doppel's own. The first
    // command reply marks live play on both servers: the replay carries
    // none, and no monster spawns before the clock turns.
    let play_start = pkts
        .iter()
        .position(|p| p.id == 0x7c)
        .map(|i| i + 1)
        .unwrap_or(0);
    for (i, p) in pkts.iter().enumerate() {
        if let Some(note) = &p.note {
            if let Some(rest) = note.strip_prefix("chase: tp @s ") {
                let vals: Vec<f64> = rest
                    .split_whitespace()
                    .filter_map(|v| v.parse().ok())
                    .collect();
                if vals.len() == 3 {
                    chase_note = Some((i, vals[0], vals[1], vals[2]));
                }
            }
        }
        if p.id < 0 {
            continue;
        }
        // Pre-play frames (the replay burst) only contribute the bot's
        // position; entity traffic counts from live play onward.
        if i < play_start && p.id != P_PLAYER_POSITION {
            continue;
        }
        let raw = hex::decode(&p.head_hex).unwrap_or_default();
        decode_packet(p.id, i, &raw, &mut obs);
    }
    if let Some((i, x, y, z)) = chase_note {
        let id = obs
            .adds
            .iter()
            .rev()
            .find(|(ai, ..)| *ai == i)
            .map(|(_, id, ..)| *id);
        if let Some(id) = id {
            obs.chase = Some((i, id, x, y, z));
        }
    }
    obs
}

/// Decodes the position and movement frames into the observation
/// set: the bot's teleports, adds, syncs, and the delta stream.
fn decode_movement_packet(id: i32, i: usize, raw: &[u8], obs: &mut Obs) {
    let mut o = 0usize;
    match id {
        // player_position: teleport id, position, ...
        P_PLAYER_POSITION => {
            if rd_varint(raw, &mut o).is_some() {
                if let (Some(x), Some(y), Some(z)) = (
                    rd_f64(raw, &mut o),
                    rd_f64(raw, &mut o),
                    rd_f64(raw, &mut o),
                ) {
                    obs.bot_pos.push((i, x, y, z));
                }
            }
        }
        // add_entity: id, uuid(16), type, pos, movement, rotations...
        P_ADD_ENTITY => {
            if let Some(id) = rd_varint(raw, &mut o) {
                o += 16;
                if let Some(typ) = rd_varint(raw, &mut o) {
                    if let (Some(x), Some(y), Some(z)) = (
                        rd_f64(raw, &mut o),
                        rd_f64(raw, &mut o),
                        rd_f64(raw, &mut o),
                    ) {
                        let tail = raw.get(o).copied().unwrap_or(0xff);
                        obs.adds.push((i, id, typ, x, y, z, tail));
                    }
                }
            }
        }
        _ => {}
    }
}

/// Decodes the state and event frames: entity data, attributes,
/// damage, swings, equipment, blasts, block updates, removals.
fn decode_state_packet(id: i32, i: usize, raw: &[u8], obs: &mut Obs) {
    let mut o = 0usize;
    match id {
        // set_entity_data: id, then entries until the 0xff
        // terminator. INT values ride as varints; FLOAT as be4.
        // Entries batch (the creeper's swell follows its flags),
        // so every entry decodes, not just the leading one.
        P_SET_ENTITY_DATA => {
            if let Some(id) = rd_varint(raw, &mut o) {
                while let Some(&accessor) = raw.get(o) {
                    if accessor == 0xff {
                        break;
                    }
                    o += 1;
                    let Some(ser) = rd_varint(raw, &mut o) else {
                        break;
                    };
                    let value = match ser {
                        3 => rd_f32(raw, &mut o),
                        1 => rd_varint(raw, &mut o).map(|v| v as f32),
                        // BYTE and BOOLEAN both ride single bytes.
                        0 | 8 => raw.get(o).map(|v| {
                            o += 1;
                            f32::from(*v)
                        }),
                        _ => None,
                    };
                    obs.data.push((i, id, accessor, ser, value));
                }
            }
        }
        // update_attributes: id, then the attribute list.
        P_UPDATE_ATTRIBUTES => {
            if let Some(id) = rd_varint(raw, &mut o) {
                if let Some(attrs) = rd_attrs(raw, &mut o) {
                    obs.attrs.push((id, attrs));
                }
            }
        }
        // entity_position_sync: id, path kind, pos, yaw, pitch, ...
        P_ENTITY_POSITION_SYNC => {
            if let (Some(id), Some(kind)) = (rd_varint(raw, &mut o), rd_varint(raw, &mut o)) {
                if kind == 0 {
                    if let (Some(x), Some(y), Some(z)) = (
                        rd_f64(raw, &mut o),
                        rd_f64(raw, &mut o),
                        rd_f64(raw, &mut o),
                    ) {
                        obs.syncs.push((i, id, x, y, z));
                        if let Some(yaw) = rd_f32(raw, &mut o) {
                            obs.body_yaws.push((i, id, pack_byte(yaw)));
                        }
                    }
                }
            }
        }
        P_MOVE_ENTITY_POS | P_MOVE_ENTITY_POS_ROT => {
            if let Some(id) = rd_varint(raw, &mut o) {
                // The properties varint: on-ground in bit 0, the step
                // count in the bits above. A zero step count carries
                // one linear delta; a positive count carries that many
                // (ticks varint, three shorts) sub-steps, and the
                // movement is their sum.
                if let Some(props) = rd_varint(raw, &mut o) {
                    let steps = (props >> 1) as usize;
                    let mut dx = 0i32;
                    let mut dy = 0i32;
                    let mut dz = 0i32;
                    let mut ok = true;
                    for _ in 0..steps.max(1) {
                        if steps > 0 && rd_varint(raw, &mut o).is_none() {
                            ok = false;
                            break;
                        }
                        match (
                            rd_i16(raw, &mut o),
                            rd_i16(raw, &mut o),
                            rd_i16(raw, &mut o),
                        ) {
                            (Some(a), Some(b), Some(c)) => {
                                dx += a as i32;
                                dy += b as i32;
                                dz += c as i32;
                            }
                            _ => {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if ok {
                        obs.deltas.push((
                            i,
                            id,
                            dx as f64 / DELTA_SCALE,
                            dy as f64 / DELTA_SCALE,
                            dz as f64 / DELTA_SCALE,
                            id == P_MOVE_ENTITY_POS_ROT,
                        ));
                        // The rotation bytes follow the delta body in
                        // both the linear and sub-stepped forms.
                        if id == P_MOVE_ENTITY_POS_ROT {
                            if let Some(&yaw) = raw.get(o) {
                                obs.body_yaws.push((i, id, yaw));
                            }
                        }
                    }
                }
            }
        }
        P_MOVE_ENTITY_ROT => {
            if let Some(id) = rd_varint(raw, &mut o) {
                // id, on-ground bool, packed yaw, packed pitch.
                if let (Some(_ground), Some(yaw)) = (raw.get(o), raw.get(o + 1)) {
                    obs.body_yaws.push((i, id, *yaw));
                    obs.deltas.push((i, id, 0.0, 0.0, 0.0, true));
                }
            }
        }
        // damage_event: target id, type id, cause, direct cause.
        P_DAMAGE_EVENT => {
            if let Some(target) = rd_varint(raw, &mut o) {
                let ty = rd_varint(raw, &mut o).unwrap_or(-1);
                obs.damage.push((i, target, ty));
            }
        }
        P_ROTATE_HEAD => {
            if let Some(id) = rd_varint(raw, &mut o) {
                if let Some(&yaw) = raw.get(o) {
                    obs.head_rots.push((i, id, yaw));
                }
            }
        }
        // swing_animation: id, hand, animation type, duration.
        P_SWING_ANIMATION => {
            if let (Some(id), Some(hand), Some(anim), Some(dur)) = (
                rd_varint(raw, &mut o),
                rd_varint(raw, &mut o),
                rd_varint(raw, &mut o),
                rd_varint(raw, &mut o),
            ) {
                obs.swings.push((i, id, hand, anim, dur));
            }
        }
        // animate: id, action byte.
        P_ANIMATE => {
            if let Some(id) = rd_varint(raw, &mut o) {
                if let Some(&action) = raw.get(o) {
                    obs.animates.push((i, id, action));
                }
            }
        }
        // entity_event: id (fixed i32), event byte.
        P_ENTITY_EVENT => {
            if raw.len() >= 5 {
                let id = i32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
                obs.entity_events.push((i, id, raw[4]));
            }
        }
        // set_equipment: id, then the first slot byte.
        P_SET_EQUIPMENT => {
            if let Some(id) = rd_varint(raw, &mut o) {
                if let Some(&slot) = raw.get(o) {
                    obs.equipment.push((id, slot));
                }
            }
        }
        P_EXPLODE => {
            decode_explode_packet(i, raw, obs);
        }
        P_BLOCK_UPDATE | P_SECTION_BLOCKS_UPDATE => {
            obs.block_updates += 1;
            decode_air_updates(raw, id, &mut obs.crater);
        }
        // remove_entities: count then ids.
        PACKET_REMOVE_ENTITIES_LOCAL => {
            let mut ids = Vec::new();
            if let Some(count) = rd_varint(raw, &mut o) {
                for _ in 0..count.max(0) {
                    match rd_varint(raw, &mut o) {
                        Some(id) => ids.push(id),
                        None => break,
                    }
                }
            }
            obs.removes.push((i, ids));
        }
        _ => {}
    }
}

/// Decodes one packet into the observation set, dispatched by
/// family so each decoder stays under the length limits.
fn decode_packet(id: i32, i: usize, raw: &[u8], obs: &mut Obs) {
    match id {
        P_PLAYER_POSITION
        | P_ADD_ENTITY
        | P_ENTITY_POSITION_SYNC
        | P_MOVE_ENTITY_POS
        | P_MOVE_ENTITY_POS_ROT
        | P_MOVE_ENTITY_ROT => {
            decode_movement_packet(id, i, raw, obs);
        }
        _ => {
            decode_state_packet(id, i, raw, obs);
        }
    }
}

/// Decodes the explode frame: center doubles, radius, the
/// fixed-width destroyed count, and the optional own-knockback
/// vector.
fn decode_explode_packet(i: usize, raw: &[u8], obs: &mut Obs) {
    let mut o = 0usize;
    if let (Some(x), Some(y), Some(z)) = (
        rd_f64(raw, &mut o),
        rd_f64(raw, &mut o),
        rd_f64(raw, &mut o),
    ) {
        let radius = rd_f32(raw, &mut o).unwrap_or(f32::NAN);
        let count = if raw.len() >= o + 4 {
            i32::from_be_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]])
        } else {
            -1
        };
        o += 4;
        let knock = raw.get(o).is_some_and(|v| *v == 1).then(|| {
            o += 1;
            let mut k = [0f64; 3];
            for slot in k.iter_mut() {
                *slot = rd_f64(raw, &mut o).unwrap_or(f64::NAN);
            }
            k
        });
        obs.explodes
            .push((i, x, y, z, radius, count, knock.map(|k| (k[0], k[1], k[2]))));
    }
}

/// The chased entity's tracked positions at every movement frame after
/// `from_index`, folded from the add position through the delta and
/// sync stream.
fn movement_track(obs: &Obs, id: i32, from_index: usize) -> Vec<(f64, f64, f64)> {
    let Some(&(_, _, _, x, y, z, _)) = obs.adds.iter().find(|(_, aid, ..)| *aid == id) else {
        return Vec::new();
    };
    let mut events: Vec<(usize, bool, f64, f64, f64)> = Vec::new();
    for (i, eid, sx, sy, sz) in &obs.syncs {
        if *eid == id && *i > from_index {
            events.push((*i, true, *sx, *sy, *sz));
        }
    }
    for (i, eid, dx, dy, dz, _) in &obs.deltas {
        if *eid == id && *i > from_index {
            events.push((*i, false, *dx, *dy, *dz));
        }
    }
    events.sort_by_key(|(i, ..)| *i);
    let (mut px, mut py, mut pz) = (x, y, z);
    let mut out = Vec::new();
    for (_, absolute, a, b, c) in events {
        if absolute {
            (px, py, pz) = (a, b, c);
        } else {
            px += a;
            py += b;
            pz += c;
        }
        out.push((px, py, pz));
    }
    out
}

/// The bot's position at a packet index: the latest teleport before it.
fn bot_pos_at(obs: &Obs, index: usize) -> Option<(f64, f64, f64)> {
    obs.bot_pos
        .iter()
        .rfind(|(i, ..)| *i < index)
        .map(|(_, x, y, z)| (*x, *y, *z))
}

/// One observation session against the server on `port`.
fn run_session(port: u16, protocol: i32) -> Result<Vec<bot::CapturedPacket>> {
    let commands: Vec<String> = vec![
        "gamerule spawn_mobs true".into(),
        "time set midnight".into(),
    ];
    let login = capture::login_start_c("Doppel");
    bot::login_capture_chase(
        "127.0.0.1",
        port,
        protocol,
        &login,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(60)),
            max_packets: Some(20000),
            dump_dir: None,
            commands: &commands,
            walk_chunks: None,
            raw_packets: &[],
        },
        bot::ChaseFirst {
            entity_type: ZOMBIE_TYPE,
            stand: CHASE_STAND,
        },
    )
}

/// The scripted summon session: the three wave-2 mobs near a moving
/// opped bot. Skeleton first (the bow engages at eight blocks), the
/// spider second (an open chase), the creeper last (the blast closes
/// the session).
fn run_session2(port: u16, protocol: i32) -> Result<Vec<bot::CapturedPacket>> {
    // One stance, three summons, one long step: the bot never moves
    // mid-scenario (teleports race the tracker and the pairing), and
    // the frozen clock lets a single step carry every leg - the
    // skeleton's draw, the spider's chase, the creeper's fuse.
    let commands: Vec<String> = vec![
        "gamerule spawn_mobs false".into(),
        "tick freeze".into(),
        "tp @s 100.5 -60 100.5".into(),
        "time set midnight".into(),
        "summon minecraft:skeleton 105.5 -60 100.5".into(),
        "summon minecraft:spider 100.5 -60 104.5".into(),
        "summon minecraft:creeper 96.5 -60 96.5".into(),
        "tick step 700".into(),
        "tick unfreeze".into(),
    ];
    let login = capture::login_start_c("Doppel");
    // MOBS_DUMP=<dir> writes every scenario-two packet body for local
    // diagnosis; unset in CI.
    let dump = std::env::var("MOBS_DUMP")
        .ok()
        .map(std::path::PathBuf::from);
    bot::login_capture(
        "127.0.0.1",
        port,
        protocol,
        &login,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(60)),
            max_packets: Some(20000),
            dump_dir: dump.as_deref(),
            commands: &commands,
            walk_chunks: None,
            raw_packets: &[],
        },
    )
}

/// The follow-and-attack session: a zombie summoned eight blocks from
/// the bot, the clock frozen while the bot strafes sideways between
/// short step barriers so the target bearing swings, and enough frozen
/// ticks for the chase to close and land several melee hits. The swing,
/// head-rotation, body-yaw, and damage rhythm of the attack is what the
/// visual-fidelity assertions read.
fn run_session3(port: u16, protocol: i32) -> Result<Vec<bot::CapturedPacket>> {
    let commands: Vec<String> = vec![
        "gamerule spawn_mobs false".into(),
        "tp @s 100.5 -60.0 100.5".into(),
        // Midnight keeps the undead from burning mid-scenario.
        "time set midnight".into(),
        "summon minecraft:zombie 108.5 -60.0 100.5".into(),
        "tick freeze".into(),
        "tick step 80".into(),
        "tp @s 100.5 -60.0 102.5".into(),
        "tick step 50".into(),
        "tp @s 100.5 -60.0 98.5".into(),
        "tick step 50".into(),
        "tp @s 100.5 -60.0 100.5".into(),
        "tick step 100".into(),
        "tick unfreeze".into(),
    ];
    let login = capture::login_start_c("Doppel");
    let dump = std::env::var("MOBS_DUMP3")
        .ok()
        .map(std::path::PathBuf::from);
    bot::login_capture(
        "127.0.0.1",
        port,
        protocol,
        &login,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(20)),
            max_packets: Some(6000),
            dump_dir: dump.as_deref(),
            commands: &commands,
            walk_chunks: None,
            raw_packets: &[],
        },
    )
}

/// One session against a freshly booted vanilla server: the session
/// thread runs until it idles out, and the server stops at the cap so
/// a session whose bot died (keep-alives keep its reads fed) still
/// returns its transcript.
fn vanilla_session_capped(
    pin: &doppel_protocol::Pin,
    jar: &std::path::Path,
    port: u16,
    protocol: i32,
    run: fn(u16, i32) -> Result<Vec<bot::CapturedPacket>>,
) -> Result<Vec<bot::CapturedPacket>> {
    let server = vanilla::boot(pin, jar, port)?;
    std::thread::sleep(Duration::from_secs(2));
    let handle = std::thread::spawn(move || run(port, protocol));
    let cap = std::time::Instant::now() + Duration::from_secs(160);
    while !handle.is_finished() && std::time::Instant::now() < cap {
        std::thread::sleep(Duration::from_millis(500));
    }
    drop(server);
    handle
        .join()
        .map_err(|_| anyhow::anyhow!("vanilla session thread panicked"))?
}

/// The doppel twin: one child per session, stopped at the cap.
fn doppel_session_capped(
    port: u16,
    protocol: i32,
    spawn_child: impl Fn() -> Result<std::process::Child>,
    run: fn(u16, i32) -> Result<Vec<bot::CapturedPacket>>,
) -> Result<Vec<bot::CapturedPacket>> {
    let mut child = spawn_child()?;
    wait_for_port(port, Duration::from_secs(30))?;
    std::thread::sleep(Duration::from_secs(2));
    let handle = std::thread::spawn(move || run(port, protocol));
    let cap = std::time::Instant::now() + Duration::from_secs(160);
    while !handle.is_finished() && std::time::Instant::now() < cap {
        std::thread::sleep(Duration::from_millis(500));
    }
    let _ = child.kill();
    let _ = child.wait();
    handle
        .join()
        .map_err(|_| anyhow::anyhow!("doppel session thread panicked"))?
}

/// The differential mobs test.
pub fn parity_mobs() -> Result<bool> {
    let pin = load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let root = doppel_protocol::find_repo_root()?;
    let blobs_dir = root.join("target").join("vanilla").join("blobs-mobs");
    let pristine_world = root
        .join("target")
        .join("vanilla")
        .join("pristine-world-mobs");
    for dir in [&blobs_dir, &pristine_world] {
        if dir.exists() {
            std::fs::remove_dir_all(dir)?;
        }
    }
    let protocol = pin.protocol.unwrap_or(0);

    capture_clean_blobs(&pin, &jar, &blobs_dir, &pristine_world, vanilla_port())?;

    // Vanilla reference sessions, one boot per scenario: a session
    // whose bot dies never idles out (keep-alives keep the read loop
    // fed), so each server is let go when its session's wall clock
    // ends, exactly how the single-session gate always ran.
    let vport = vanilla_port();
    let vworker = std::thread::spawn(move || -> Result<_> {
        let one = vanilla_session_capped(&pin, &jar, vport, protocol, run_session)?;
        let two = vanilla_session_capped(&pin, &jar, vport, protocol, run_session2)?;
        let three = vanilla_session_capped(&pin, &jar, vport, protocol, run_session3)?;
        Ok((one, two, three))
    });
    let vdeadline = std::time::Instant::now() + Duration::from_secs(560);
    while !vworker.is_finished() && std::time::Instant::now() < vdeadline {
        std::thread::sleep(Duration::from_millis(500));
    }
    let (v_pkts, v2_pkts, v3_pkts) = vworker
        .join()
        .map_err(|_| anyhow::anyhow!("vanilla session thread panicked"))??;

    // MOBS_SKIP_DOPPEL=1 runs the reference sessions alone (capture
    // passes and fact gathering); CI never sets it.
    let (d_pkts, d2_pkts, d3_pkts) = if std::env::var_os("MOBS_SKIP_DOPPEL").is_some() {
        (Vec::new(), Vec::new(), Vec::new())
    } else {
        // The doppel side runs the same three boots.
        let bin = default_doppel_bin()?;
        let pin_path = doppel_protocol::pin_path()?;
        let spawn_doppel = move || -> Result<_> {
            let child = Command::new(&bin)
                .env("DOPPEL_ADDR", "127.0.0.1")
                .env("DOPPEL_PORT", doppel_port().to_string())
                .env("DOPPEL_PIN", &pin_path)
                .env("DOPPEL_BLOBS", &blobs_dir)
                .env("DOPPEL_WORLD", &pristine_world)
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .spawn()
                .with_context(|| format!("spawning {}", bin.display()))?;
            Ok(child)
        };
        let dport = doppel_port();
        let worker = std::thread::spawn(move || -> Result<_> {
            let one = doppel_session_capped(dport, protocol, &spawn_doppel, run_session)?;
            let two = doppel_session_capped(dport, protocol, &spawn_doppel, run_session2)?;
            let three = doppel_session_capped(dport, protocol, &spawn_doppel, run_session3)?;
            Ok((one, two, three))
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(560);
        while !worker.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(500));
        }
        worker
            .join()
            .map_err(|_| anyhow::anyhow!("doppel session thread panicked"))??
    };

    let v = analyze(&v_pkts);
    let d = analyze(&d_pkts);
    let v2 = analyze(&v2_pkts);
    let d2 = analyze(&d2_pkts);
    let v3 = analyze(&v3_pkts);
    let d3 = analyze(&d3_pkts);
    report(&[("vanilla", &v_pkts, &v), ("doppel", &d_pkts, &d)]);
    report(&[("vanilla-s2", &v2_pkts, &v2), ("doppel-s2", &d2_pkts, &d2)]);
    report(&[("vanilla-s3", &v3_pkts, &v3), ("doppel-s3", &d3_pkts, &d3)]);
    report_attack_facts("vanilla-s3", &v3);
    if !d3_pkts.is_empty() {
        report_attack_facts("doppel-s3", &d3);
    }
    report_blast_facts("vanilla-s2", &v2);
    if !d2_pkts.is_empty() {
        report_blast_facts("doppel-s2", &d2);
    }

    // Capture-only mode ends after the reference reports.
    if std::env::var_os("MOBS_SKIP_DOPPEL").is_some() {
        println!("PASS: mobs parity (reference capture only)");
        return Ok(true);
    }

    let mut failures = Vec::new();

    // Zombies appear on both sides, typed 154 in the entity registry.
    for (who, s) in [("vanilla", &v), ("doppel", &d)] {
        let zombies: Vec<_> = s
            .adds
            .iter()
            .filter(|(_, _, t, ..)| *t == ZOMBIE_TYPE)
            .collect();
        if zombies.is_empty() {
            let total = s.adds.len();
            failures.push(format!(
                "{who}: {total} adds, none typed {ZOMBIE_TYPE} (zombie)"
            ));
        }
    }

    // Spawn geometry: block centers, integer feet, 24..128 from the bot.
    for (who, s) in [("vanilla", &v), ("doppel", &d)] {
        check_spawn_geometry(who, s, &mut failures);
    }

    // The chase probe fired on both sides; the chased zombie pairs with
    // the health datum, the movement-speed attribute, and zero spawn
    // motion.
    for (who, s) in [("vanilla", &v), ("doppel", &d)] {
        check_pairing(who, s, &mut failures);
    }

    // The chase closes on doppel: last distance well under the first
    // across >= 3 movement samples. The reference's own chase carries
    // run-to-run variance (its zombie sometimes strolls off first), so
    // vanilla supplies the packet-shape pins above, not this trend.
    for (who, s) in [("doppel", &d)] {
        let Some((ci, cid, bx, by, bz)) = s.chase else {
            continue;
        };
        let track = movement_track(s, cid, ci);
        let dist = |p: &(f64, f64, f64)| {
            let (dx, dy, dz) = (p.0 - bx, p.1 - by, p.2 - bz);
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        if track.len() < 3 {
            failures.push(format!(
                "{who}: chase produced {} movement samples, want >= 3",
                track.len()
            ));
        } else {
            let first = dist(&track[0]);
            // The reach of the chase is the closest approach: once the
            // player is down the zombie strolls off, so the closing is
            // measured as the minimum, not the endpoint.
            let closest = track.iter().map(dist).fold(f64::INFINITY, f64::min);
            if closest >= first - CHASE_CLOSE {
                failures.push(format!(
                    "{who}: chase did not close (start {first:.1}, closest {closest:.1}, want a drop of {CHASE_CLOSE})"
                ));
            }
        }
    }

    // The melee lands: at least one damage event after the chase.
    for (who, s) in [("vanilla", &v), ("doppel", &d)] {
        let Some((ci, ..)) = s.chase else {
            continue;
        };
        let hits = s.damage.iter().filter(|(i, _, _)| *i > ci).count();
        if hits == 0 {
            failures.push(format!("{who}: no damage events after the chase"));
        }
    }

    // Scenario two: the scripted summons on both servers.
    for (who, s) in [("vanilla", &v2), ("doppel", &d2)] {
        check_scenario_two(who, s, &mut failures);
    }

    // Scenario three: the follow-and-attack visual assertions, run
    // identically against both servers.
    for (who, s) in [("vanilla", &v3), ("doppel", &d3)] {
        check_follow_and_attack(who, s, &mut failures);
    }

    // The detonation comparison, by value, across the two servers.
    compare_blasts(&v2, &d2, &mut failures);

    if failures.is_empty() {
        println!("PASS: mobs parity");
        Ok(true)
    } else {
        println!("FAIL: {} mobs difference(s):", failures.len());
        for f in failures.iter().take(16) {
            println!("  {f}");
        }
        Ok(false)
    }
}

/// Scenario two's structural checks: the three summons pair with their
/// types and metadata, the skeleton fires an arrow, the creeper swells
/// and detonates, and the spider closes on the bot.
fn check_scenario_two(who: &str, s: &Obs, failures: &mut Vec<String>) {
    // The three summons appear with their registry types.
    for (ty, name) in [
        (SKELETON_TYPE, "skeleton"),
        (CREEPER_TYPE, "creeper"),
        (SPIDER_TYPE, "spider"),
    ] {
        if !s.adds.iter().any(|(_, _, t, ..)| *t == ty) {
            failures.push(format!("{who}: no {name} add (type {ty})"));
        }
    }
    // The spider pairs movement speed alone (the reference's mob
    // snapshots carry no max health, even the spider's off-default
    // 16.0): attr 26 = 0.3, and no 23 entry.
    let spider_id = s
        .adds
        .iter()
        .find(|(_, _, t, ..)| *t == SPIDER_TYPE)
        .map(|(_, id, ..)| *id);
    if let Some(sid) = spider_id {
        let carries = |attr: i32, want: f64| {
            s.attrs.iter().any(|(id, list)| {
                *id == sid
                    && list
                        .iter()
                        .any(|(a, v)| *a == attr && (v - want).abs() < 0.01)
            })
        };
        let has_max_health = s
            .attrs
            .iter()
            .any(|(id, list)| *id == sid && list.iter().any(|(a, _)| *a == 23));
        if !carries(26, 0.3) {
            failures.push(format!(
                "{who}: spider {sid} pairs no movement-speed attribute (26, 0.3)"
            ));
        }
        if has_max_health {
            failures.push(format!(
                "{who}: spider {sid} pairs a max-health attribute (23); the reference omits it"
            ));
        }
    }
    // The skeleton's draw raises the using-item bit on the living
    // flags datum (the aim pose).
    let skel_for_aim = s
        .adds
        .iter()
        .find(|(_, _, t, ..)| *t == SKELETON_TYPE)
        .map(|(_, id, ..)| *id);
    if let Some(sk) = skel_for_aim {
        let drew = s.data.iter().any(|(_, id, acc, ser, val)| {
            *id == sk && *acc == 8 && *ser == 0 && val.is_some_and(|v| v as u8 & 0x01 == 0x01)
        });
        if !drew {
            failures.push(format!(
                "{who}: skeleton {sk} never raises the using-item datum (acc 8)"
            ));
        }
    }
    // The skeleton pairs with a main-hand equipment entry.
    let skel_id = s
        .adds
        .iter()
        .find(|(_, _, t, ..)| *t == SKELETON_TYPE)
        .map(|(_, id, ..)| *id);
    if let Some(id) = skel_id {
        if !s
            .equipment
            .iter()
            .any(|(eid, slot)| *eid == id && *slot == 0)
        {
            failures.push(format!(
                "{who}: skeleton {id} pairs with no main-hand equipment"
            ));
        }
    } else {
        failures.push(format!("{who}: no skeleton to equip"));
    }
    // The arrow: an add typed 6, then movement or a landed hit.
    match s.adds.iter().find(|(_, _, t, ..)| *t == ARROW_TYPE) {
        Some((ai, aid, ..)) => {
            let moved = s.syncs.iter().any(|(i, id, ..)| i > ai && *id == *aid)
                || s.deltas.iter().any(|(i, id, ..)| i > ai && *id == *aid);
            let landed = s.damage.iter().any(|(i, _, _)| i > ai);
            if !moved && !landed {
                failures.push(format!("{who}: arrow {aid} neither moves nor lands"));
            }
        }
        None => failures.push(format!("{who}: no arrow add after the skeleton engages")),
    }
    // The creeper: the swell datum flips positive, the explosion lands
    // near its last position, and it leaves without a corpse.
    let creeper = s
        .adds
        .iter()
        .find(|(_, _, t, ..)| *t == CREEPER_TYPE)
        .map(|(ai, id, _, x, y, z, _)| (*ai, *id, (x, y, z)));
    match creeper {
        Some((_, cid, _)) => {
            let swelled = s.data.iter().any(|(_, id, acc, ser, val)| {
                *id == cid && *acc == 16 && *ser == 1 && val.is_some_and(|v| v > 0.0)
            });
            if !swelled {
                failures.push(format!(
                    "{who}: creeper {cid} never swells (accessor 16 INT)"
                ));
            }
            // The swell timeline: the fuse is 30 ticks, so the datum's
            // positive flip must precede the blast frame, and the
            // detonation ends the story (no deflation flip after).
            let flip = s
                .data
                .iter()
                .find(|(_, id, acc, ser, val)| {
                    *id == cid && *acc == 16 && *ser == 1 && val.is_some_and(|v| v > 0.0)
                })
                .map(|(i, ..)| *i);
            let deflated_after = s.data.iter().any(|(i, id, acc, ser, val)| {
                let after_blast = s.explodes.first().is_some_and(|(bi, ..)| *i > *bi);
                after_blast && *id == cid && *acc == 16 && *ser == 1 && val.is_some_and(|v| v < 0.0)
            });
            if deflated_after {
                failures.push(format!("{who}: creeper {cid} deflates after the blast"));
            }
            if let (Some(blast_i), Some(flip_i)) = (s.explodes.first().map(|(i, ..)| *i), flip) {
                if flip_i >= blast_i {
                    failures.push(format!(
                        "{who}: swell flip at frame {flip_i} not before the blast at {blast_i}"
                    ));
                }
            }
            let removed = s.removes.iter().any(|(_, ids)| ids.contains(&cid));
            if !removed {
                failures.push(format!("{who}: creeper {cid} never removed"));
            }
            let near_blast = s.explodes.iter().any(|(_, x, y, z, ..)| {
                let track = movement_track(s, cid, 0);
                track.last().is_some_and(|p| {
                    let (dx, dy, dz) = (p.0 - x, p.1 - y, p.2 - z);
                    (dx * dx + dy * dy + dz * dz).sqrt() < 8.0
                })
            });
            if s.explodes.is_empty() {
                failures.push(format!("{who}: no explosion packet"));
            } else if !near_blast {
                failures.push(format!(
                    "{who}: the explosion center sits away from the creeper"
                ));
            }
            if s.block_updates == 0 {
                failures.push(format!("{who}: no block updates after the blast"));
            }
        }
        None => failures.push(format!("{who}: no creeper to swell")),
    }
    // The spider closes on the bot's stance during its leg.
    let spider = s
        .adds
        .iter()
        .find(|(_, _, t, ..)| *t == SPIDER_TYPE)
        .map(|(ai, id, ..)| (*ai, *id));
    match spider {
        Some((ai, sid)) => {
            let track = movement_track(s, sid, ai);
            if track.len() < 3 {
                failures.push(format!(
                    "{who}: spider produced {} movement samples, want >= 3",
                    track.len()
                ));
            } else {
                // The bot stands at one stance the whole scenario;
                // its last teleport bounds the anchor.
                let anchor = bot_pos_at(s, ai).unwrap_or((100.5, -60.0, 100.5));
                let dist = |p: &(f64, f64, f64)| {
                    let (dx, dz) = (p.0 - anchor.0, p.2 - anchor.2);
                    (dx * dx + dz * dz).sqrt()
                };
                let closest = track.iter().map(dist).fold(f64::INFINITY, f64::min);
                if closest > 5.0 {
                    failures.push(format!(
                        "{who}: spider chase closes only to {closest:.1}, want < 5"
                    ));
                }
            }
        }
        None => failures.push(format!("{who}: no spider to chase")),
    }
}

/// The follow-and-attack assertions, derived from the vanilla capture
/// (docs/mobs-research.md section 22): a swing frame rides every
/// landed melee hit; the settled head tracks the target's bearing; the
/// walking body yaw faces the approach.
fn check_follow_and_attack(who: &str, s: &Obs, failures: &mut Vec<String>) {
    let Some(&(_, zid, ..)) = s.adds.iter().find(|(_, _, t, ..)| *t == ZOMBIE_TYPE) else {
        failures.push(format!("{who}: no zombie add in the follow-and-attack leg"));
        return;
    };
    let swings: Vec<usize> = s
        .swings
        .iter()
        .filter(|(_, sid, ..)| *sid == zid)
        .map(|(i, ..)| *i)
        .collect();
    // (a) A swing per landed hit: every mob_attack damage frame on the
    // bot carries a zombie swing a few frames earlier. The reference
    // sends the swing 1-2 frames ahead; the window leaves room for a
    // metadata frame between them.
    let hits: Vec<usize> = s
        .damage
        .iter()
        .filter(|(_, target, ty)| *ty == 28 && *target != zid)
        .map(|(i, ..)| *i)
        .collect();
    if hits.len() < 3 {
        failures.push(format!(
            "{who}: {} melee hits landed, want >= 3",
            hits.len()
        ));
    }
    for hi in &hits {
        if !swings.iter().any(|si| si <= hi && hi - si <= 8) {
            failures.push(format!(
                "{who}: melee hit at frame {hi} carries no swing within 8 frames"
            ));
        }
    }
    // The bare-hand swing shape: main hand, WHACK, 6 ticks.
    for (si, sid, hand, anim, dur) in &s.swings {
        if *sid == zid && (*hand, *anim, *dur) != (0, 1, 6) {
            failures.push(format!(
                "{who}: swing at frame {si} shape hand{hand}/anim{anim}/dur{dur}, want 0/1/6"
            ));
        }
    }
    // The fight window: the assertions below read the fight, not the
    // aftermath. Once the bot dies the zombie strolls off and facing
    // it means nothing; the window ends at the last landed hit.
    let (win_lo, win_hi) = match (hits.first(), hits.last()) {
        (Some(lo), Some(hi)) => (*lo, *hi),
        _ => (0, usize::MAX),
    };
    // (b) Head tracking. The reference swings regardless of head
    // position and its body drags the head while walking (observed
    // up to ~77 deg, and in-fight tracking fractions from 29% to 59%
    // across its own runs), so the honest claims are: the head
    // CONVERGED on the target before the fight began, and it re-locks
    // (or never needs to move) inside the fight.
    let head_err_at = |i: usize, yaw: u8| -> Option<f64> {
        let zpos = track_at(s, zid, i)?;
        let bpos = bot_pos_at(s, i)?;
        Some(yaw_dist(
            unpack_yaw_deg(yaw),
            bearing((zpos.0, zpos.2), (bpos.0, bpos.2)),
        ))
    };
    // The approach converged: the last head frame before the first
    // hit sits locked on the target.
    let pre = s
        .head_rots
        .iter()
        .filter(|(_, id, _)| *id == zid)
        .filter(|(fi, _, _)| *fi < win_lo)
        .filter_map(|(fi, _, yaw)| head_err_at(*fi, *yaw))
        .next_back();
    match pre {
        Some(err) if err <= HEAD_SETTLED_DEG => {}
        Some(err) => failures.push(format!(
            "{who}: the head entered the fight {err:.0} deg off the target bearing, want <= {HEAD_SETTLED_DEG}"
        )),
        None => failures.push(format!(
            "{who}: no pre-fight head frame to verify the approach convergence"
        )),
    }
    let mut frames_err: Vec<f64> = Vec::new();
    for (i, _, yaw) in s
        .head_rots
        .iter()
        .filter(|(_, id, _)| *id == zid)
        .filter(|(fi, _, _)| *fi >= win_lo && *fi <= win_hi)
    {
        if let Some(err) = head_err_at(*i, *yaw) {
            frames_err.push(err);
        }
    }
    if !frames_err.is_empty() {
        let locked = frames_err
            .iter()
            .filter(|e| **e <= HEAD_SETTLED_DEG)
            .count();
        if locked == 0 {
            failures.push(format!(
                "{who}: {} head frames in the fight, none within {} deg of the target",
                frames_err.len(),
                HEAD_SETTLED_DEG
            ));
        }
    }
    // (c) Body yaw while moving: at least half the walking frames from
    // the summon to the fight's end face the approach direction within
    // BODY_TRACK_DEG (the walking happens on the approach; the fight
    // itself is mostly standing).
    let mut facing = 0usize;
    let mut moving = 0usize;
    for (i, id, yaw) in s
        .body_yaws
        .iter()
        .filter(|(_, id, _)| *id == zid)
        .filter(|(fi, _, _)| *fi <= win_hi)
    {
        let Some(delta) = s.deltas.iter().find(|(di, did, ..)| di == i && *did == *id) else {
            continue;
        };
        if delta.2 * delta.2 + delta.4 * delta.4 < 1.0e-8 {
            continue;
        }
        let Some(zpos) = track_at(s, zid, *i) else {
            continue;
        };
        let Some(bpos) = bot_pos_at(s, *i) else {
            continue;
        };
        moving += 1;
        let err = yaw_dist(
            unpack_yaw_deg(*yaw),
            bearing((zpos.0, zpos.2), (bpos.0, bpos.2)),
        );
        if err <= BODY_TRACK_DEG {
            facing += 1;
        }
    }
    if moving < 6 {
        failures.push(format!("{who}: {} moving yaw frames, want >= 6", moving));
    } else if facing * 2 < moving {
        failures.push(format!(
            "{who}: {facing}/{moving} moving frames face the approach within {} deg",
            BODY_TRACK_DEG
        ));
    }
}

/// The detonation comparison: the explode packet's center, radius, and
/// destroyed-block count, the victim's knockback vector, and the crater
/// set, compared by value across the two servers. Tolerances come from
/// observed vanilla-versus-vanilla variance across two reference runs
/// (the ray grid's per-ray power rolls differ run to run).
fn compare_blasts(v: &Obs, d: &Obs, failures: &mut Vec<String>) {
    let (Some(vb), Some(db)) = (v.explodes.first(), d.explodes.first()) else {
        if v.explodes.is_empty() || d.explodes.is_empty() {
            failures.push(format!(
                "blast: vanilla has {} explodes, doppel {} (want one each)",
                v.explodes.len(),
                d.explodes.len()
            ));
        }
        return;
    };
    let center_off = (
        (vb.1 - db.1).abs(),
        (vb.2 - db.2).abs(),
        (vb.3 - db.3).abs(),
    );
    if center_off.0 > BLAST_CENTER_TOL
        || center_off.1 > BLAST_CENTER_TOL
        || center_off.2 > BLAST_CENTER_TOL
    {
        failures.push(format!(
            "blast: centers differ vanilla ({:.2},{:.2},{:.2}) vs doppel ({:.2},{:.2},{:.2})",
            vb.1, vb.2, vb.3, db.1, db.2, db.3
        ));
    }
    if (vb.4 - db.4).abs() > 0.01 {
        failures.push(format!(
            "blast: radius vanilla {:.3} vs doppel {:.3}",
            vb.4, db.4
        ));
    }
    let counts = (vb.5 as f64, db.5 as f64);
    let rel = (counts.0 - counts.1).abs() / counts.0.max(counts.1).max(1.0);
    if rel > BLAST_COUNT_REL {
        failures.push(format!(
            "blast: destroyed count vanilla {} vs doppel {} (relative diff {rel:.2})",
            vb.5, db.5
        ));
    }
    // The knockback: a per-side physics check, because the two walks
    // stop at different points inside the swell window and any
    // cross-comparison of absolute vectors reads the walk, not the
    // blast. Each side's vector must equal the unit ray from its own
    // blast center to the victim's eye, scaled by (1 - feet distance
    // / doubled radius) with full exposure.
    for (who, s, b) in [("vanilla", v, vb), ("doppel", d, db)] {
        let Some(k) = &b.6 else {
            failures.push(format!("blast: {who} carries no own-knockback"));
            continue;
        };
        let Some(bot) = bot_pos_at(s, b.0) else {
            failures.push(format!("blast: {who} has no bot stance"));
            continue;
        };
        let eye = (bot.0, bot.1 + PLAYER_EYE_HEIGHT, bot.2);
        let ray = (eye.0 - b.1, eye.1 - b.2, eye.2 - b.3);
        let rl = (ray.0 * ray.0 + ray.1 * ray.1 + ray.2 * ray.2).sqrt();
        let unit = (ray.0 / rl, ray.1 / rl, ray.2 / rl);
        let feet = ((bot.0 - b.1).powi(2) + (bot.1 - b.2).powi(2) + (bot.2 - b.3).powi(2)).sqrt();
        let want_mag = (1.0 - feet / (b.4 as f64 * 2.0)).max(0.0);
        let mag = (k.0 * k.0 + k.1 * k.1 + k.2 * k.2).sqrt();
        if (mag - want_mag).abs() > BLAST_KNOCK_MAG_TOL {
            failures.push(format!(
                "blast: {who} knockback magnitude {mag:.3}, physics says {want_mag:.3}"
            ));
        }
        let dot = k.0 * unit.0 + k.1 * unit.1 + k.2 * unit.2;
        let cos = dot / mag.max(1.0e-9);
        let angle = cos.clamp(-1.0, 1.0).acos().to_degrees();
        if angle > BLAST_KNOCK_ANGLE_DEG {
            failures.push(format!(
                "blast: {who} knockback direction {angle:.1} deg off its own center-to-eye ray"
            ));
        }
    }
    // The crater sets: cells the update stream turned to air, each
    // translated against its own blast center so the comparison reads
    // the ray physics rather than the walk's stopping point.
    let rel = |set: &std::collections::BTreeSet<(i32, i32, i32)>,
               c: (f64, f64, f64)|
     -> std::collections::BTreeSet<(i32, i32, i32)> {
        let (bx, by, bz) = (c.0.floor() as i32, c.1.floor() as i32, c.2.floor() as i32);
        set.iter().map(|p| (p.0 - bx, p.1 - by, p.2 - bz)).collect()
    };
    let v_rel = rel(&v.crater, (vb.1, vb.2, vb.3));
    let d_rel = rel(&d.crater, (db.1, db.2, db.3));
    let overlap = v_rel.intersection(&d_rel).count();
    let union = v_rel.union(&d_rel).count();
    if union == 0 || (overlap as f64) / (union as f64) < BLAST_CRATER_JACCARD {
        failures.push(format!(
            "blast: crater (center-relative) overlap {overlap} of union {union} (vanilla {}, doppel {})",
            v.crater.len(),
            d.crater.len()
        ));
    }
}

/// Blast comparison tolerances, from vanilla-versus-vanilla variance
/// across three reference captures of the detonation scenario: the
/// packet's destroyed count (air included) read 321/332/338 (5.1%
/// max spread), the crater sets 38/46/46 cells with a strict-subset
/// overlap (Jaccard 0.826), the center and knockback identical
/// (knockback components within 0.006, magnitude ~0.56). The bounds
/// sit below the observed variance with margin, not at it; the
/// knockback and crater compare center-relative because the two
/// walks stop at different points inside the swell window.
const BLAST_CENTER_TOL: f64 = 2.0;
const BLAST_COUNT_REL: f64 = 0.15;
const BLAST_KNOCK_MAG_TOL: f64 = 0.05;
const BLAST_KNOCK_ANGLE_DEG: f64 = 5.0;
const BLAST_CRATER_JACCARD: f64 = 0.6;
/// The victim's eye height over the feet, for the knockback ray.
const PLAYER_EYE_HEIGHT: f64 = 1.62;

/// The turn-in exemption window after a bot strafe, in frames; the
/// reference's head turns 30 deg/tick, so a 180-degree flip needs 6
/// ticks, about two interval-3 syncs plus their frame neighbors.
const TURN_WINDOW: usize = 24;
/// The settled head's allowed miss of the target bearing: the wire
/// byte granularity alone is 1.4 deg, and finishing turns read up to
/// ~21 deg on the reference mid-fight before locking to 0.
const HEAD_SETTLED_DEG: f64 = 20.0;
/// The walking body's allowed miss of the approach bearing; the
/// reference turns at 90 deg/tick and lags most during turn-in.
const BODY_TRACK_DEG: f64 = 45.0;

/// The detonation digest: the explode packet values and the crater set
/// the block-update stream laid down.
fn report_blast_facts(who: &str, s: &Obs) {
    // Swing and damage traffic by entity, for the summoned-mob leg.
    let by_entity = |label: &str, picks: &[i32]| {
        for want in picks {
            let (ty, name) = match *want {
                SKELETON_TYPE => (SKELETON_TYPE, "skeleton"),
                CREEPER_TYPE => (CREEPER_TYPE, "creeper"),
                SPIDER_TYPE => (SPIDER_TYPE, "spider"),
                ZOMBIE_TYPE => (ZOMBIE_TYPE, "zombie"),
                _ => (*want, "?"),
            };
            let Some(id) = s
                .adds
                .iter()
                .find(|(_, _, t, ..)| *t == ty)
                .map(|(_, id, ..)| *id)
            else {
                continue;
            };
            let swings = s.swings.iter().filter(|(_, sid, ..)| *sid == id).count();
            let hits_taken = s.damage.iter().filter(|(_, t, _)| *t == id).count();
            println!("[oracle] {who} {label} {name} id {id}: {swings} swings, {hits_taken} damage frames as target");
        }
    };
    by_entity("s2", &[SKELETON_TYPE, CREEPER_TYPE, SPIDER_TYPE]);
    // Damage traffic by type, the arrow/melee/explosion ids.
    let mut by_type: std::collections::BTreeMap<i32, usize> = Default::default();
    for (_, _, ty, ..) in &s.damage {
        *by_type.entry(*ty).or_default() += 1;
    }
    let by_type = by_type
        .into_iter()
        .map(|(ty, n)| format!("type{ty} x{n}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("[oracle] {who} damage by type: {by_type}");
    // The creeper's swell datum timeline relative to the blast.
    if let Some((_, cid, ..)) = s.adds.iter().find(|(_, _, t, ..)| *t == CREEPER_TYPE) {
        for (i, id, acc, ser, val) in s.data.iter() {
            if *id == *cid && *acc == DATA_SWELL_ACCESSOR {
                println!(
                    "[oracle] {who} creeper swell datum @ {i}: ser {ser} = {}",
                    val.map(|v| format!("{v:.0}")).unwrap_or_else(|| "?".into())
                );
            }
        }
    }
    for (i, x, y, z, radius, count, knock) in &s.explodes {
        println!(
            "[oracle] {who} blast @ {i}: center ({x:.2},{y:.2},{z:.2}) radius {radius:.3} count {count} knockback {:?} crater {} blocks",
            knock.map(|k| format!("({:.3},{:.3},{:.3})", k.0, k.1, k.2)),
            s.crater.len()
        );
        let ys: std::collections::BTreeSet<i32> = s.crater.iter().map(|p| p.1).collect();
        let span = s
            .crater
            .iter()
            .fold((i32::MAX, i32::MAX, i32::MIN, i32::MIN), |a, p| {
                (a.0.min(p.0), a.1.min(p.2), a.2.max(p.0), a.3.max(p.2))
            });
        println!(
            "[oracle] {who} crater: {} cells, y layers {ys:?}, x {}..{} z {}..{}",
            s.crater.len(),
            span.0,
            span.2,
            span.1,
            span.3
        );
        // The full cell list, for offline set comparisons.
        let cells = s
            .crater
            .iter()
            .map(|(x, y, z)| format!("{x},{y},{z}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("[oracle] {who} crater cells: {cells}");
    }
}

/// The follow-and-attack evidence digest: per-hit swing pairing, the
/// mob-flags datum flip, the head-rotation cadence and tracking error,
/// and the body-yaw alignment while moving.
fn report_attack_facts(who: &str, s: &Obs) {
    let Some(&(_, zid, ..)) = s.adds.iter().find(|(_, _, t, ..)| *t == ZOMBIE_TYPE) else {
        println!("[oracle] {who}: no zombie add; nothing to report");
        return;
    };
    let swings: Vec<_> = s.swings.iter().filter(|(_, id, ..)| *id == zid).collect();
    let animates: Vec<_> = s.animates.iter().filter(|(_, id, _)| *id == zid).collect();
    let damages: Vec<_> = s.damage.iter().collect();
    println!(
        "[oracle] {who} attack facts: zombie {zid}, {} swing frames, {} animate frames, {} damage events, {} rotate_head, {} body-yaw frames",
        swings.len(),
        animates.len(),
        damages.len(),
        s.head_rots.iter().filter(|(_, id, _)| *id == zid).count(),
        s.body_yaws.iter().filter(|(_, id, _)| *id == zid).count(),
    );
    // Swing payload shapes.
    let mut shapes: std::collections::BTreeMap<(i32, i32, i32), usize> =
        std::collections::BTreeMap::new();
    for (_, _, hand, anim, dur) in &swings {
        *shapes.entry((*hand, *anim, *dur)).or_default() += 1;
    }
    let shapes = shapes
        .into_iter()
        .map(|((hand, anim, dur), n)| format!("hand{hand}/anim{anim}/dur{dur} x{n}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("[oracle] {who} swing shapes: {shapes}");
    // Per-hit pairing: the nearest zombie swing before each damage event.
    for (di, dtarget, dty) in damages.iter().take(12) {
        let nearest = swings
            .iter()
            .filter(|(si, ..)| si <= di)
            .map(|(si, ..)| di - si)
            .next_back();
        match nearest {
            Some(gap) => println!(
                "[oracle] {who} hit @ {di}: target {dtarget} type {dty}, swing {gap} frame(s) earlier"
            ),
            None => println!(
                "[oracle] {who} hit @ {di}: target {dtarget} type {dty}, NO preceding swing"
            ),
        }
    }
    // The metadata timeline around the attack: mob flags and health data.
    for (i, _, acc, ser, val) in s.data.iter().filter(|(_, id, ..)| *id == zid).take(16) {
        println!(
            "[oracle] {who} zombie datum @ {i}: acc {acc} ser {ser} = {}",
            val.map(|v| format!("{v:.2}")).unwrap_or_else(|| "?".into())
        );
    }
    // Head tracking: the error between the rotate_head yaw and the
    // bearing to the bot at each frame, using the zombie's tracked
    // position and the bot's latest teleport.
    let strafes: Vec<usize> = s.bot_pos.iter().map(|(i, ..)| *i).skip(1).collect();
    let strafe_list = strafes
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    println!("[oracle] {who} bot strafes at packets: {strafe_list}");
    let mut max_err = 0.0f64;
    let mut over30 = 0usize;
    let mut frames = 0usize;
    for (fi, _, yaw) in s.head_rots.iter().filter(|(_, id, _)| *id == zid) {
        let i = *fi;
        let Some(zpos) = track_at(s, zid, i) else {
            continue;
        };
        let Some(bpos) = bot_pos_at(s, i) else {
            continue;
        };
        let b = bearing((zpos.0, zpos.2), (bpos.0, bpos.2));
        let err = yaw_dist(unpack_yaw_deg(*yaw), b);
        // A frame sits in the turn window when a strafe landed within
        // the last 24 packets (the head turns at 30 deg/tick; a 180
        // flip needs 6 ticks, about two interval-3 syncs and their
        // surrounding frames).
        let in_window = strafes.iter().any(|si| *si <= i && i < si + TURN_WINDOW);
        frames += 1;
        if err > 30.0 && !in_window {
            over30 += 1;
        }
        println!(
            "[oracle] {who} head @ {i}: yaw {:.0} bearing {:.0} err {err:.0}{}",
            unpack_yaw_deg(*yaw),
            b,
            if in_window { " (turn window)" } else { "" }
        );
        max_err = max_err.max(err);
    }
    println!(
        "[oracle] {who} head-vs-bearing: {frames} frames, max err {max_err:.1} deg, over-30 {over30}",
    );
    // Body yaw versus the movement direction while the zombie walks:
    // only frames with a real delta and a yaw byte.
    let mut aligned = 0usize;
    let mut moving = 0usize;
    let mut max_lag = 0.0f64;
    for (i, id, yaw) in s.body_yaws.iter().filter(|(_, id, _)| *id == zid) {
        let Some(delta) = s.deltas.iter().find(|(di, did, ..)| di == i && *did == *id) else {
            continue;
        };
        let (dx, dz) = (delta.2, delta.4);
        if dx * dx + dz * dz < 1.0e-8 {
            continue;
        }
        moving += 1;
        let err = yaw_dist(unpack_yaw_deg(*yaw), bearing((0.0, 0.0), (dx, dz)));
        if err <= 30.0 {
            aligned += 1;
        }
        max_lag = max_lag.max(err);
    }
    println!(
        "[oracle] {who} body-vs-move: {moving} moving frames, {aligned} within 30 deg, max lag {max_lag:.1} deg"
    );
}

/// The zombie's tracked position at a packet index: the add position
/// folded through every movement event (syncs set, deltas add) at or
/// before the index, in arrival order.
fn track_at(s: &Obs, id: i32, index: usize) -> Option<(f64, f64, f64)> {
    let (_, _, _, ax, ay, az, _) = *s.adds.iter().find(|(_, eid, ..)| *eid == id)?;
    let mut events: Vec<(usize, bool, f64, f64, f64)> = Vec::new();
    for (i, eid, x, y, z) in &s.syncs {
        if *eid == id && *i <= index {
            events.push((*i, true, *x, *y, *z));
        }
    }
    for (i, eid, dx, dy, dz, _) in &s.deltas {
        if *eid == id && *i <= index {
            events.push((*i, false, *dx, *dy, *dz));
        }
    }
    events.sort_by_key(|(i, ..)| *i);
    let (mut px, mut py, mut pz) = (ax, ay, az);
    for (_, absolute, a, b, c) in events {
        if absolute {
            (px, py, pz) = (a, b, c);
        } else {
            px += a;
            py += b;
            pz += c;
        }
    }
    Some((px, py, pz))
}

/// The per-side session digest: the frame histogram, the add types, the
/// resolved chase with its pairing packets, and the damage counts.
fn report(sides: &[(&str, &[bot::CapturedPacket], &Obs)]) {
    let histogram = |pkts: &[bot::CapturedPacket]| {
        let mut hist: std::collections::BTreeMap<i32, usize> = Default::default();
        for p in pkts.iter().filter(|p| p.id >= 0) {
            *hist.entry(p.id).or_default() += 1;
        }
        hist.into_iter()
            .map(|(id, n)| format!("{id:#04x}:{n}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    for &(who, pkts, s) in sides {
        println!(
            "[oracle] {who}: {} frames, ids: {}",
            pkts.len(),
            histogram(pkts)
        );
        let mut types: std::collections::BTreeMap<i32, usize> = Default::default();
        for (_, _, typ, ..) in &s.adds {
            *types.entry(*typ).or_default() += 1;
        }
        let types = types
            .into_iter()
            .map(|(t, n)| format!("{t}:{n}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("[oracle] {who} add types: {types}");
        if let Some((ci, cid, x, y, z)) = s.chase {
            println!(
                "[oracle] {who} chase: zombie id={cid} at {ci}, bot at ({x:.1},{y:.1},{z:.1})"
            );
            let pairs = s
                .attrs
                .iter()
                .filter(|(id, _)| *id == cid)
                .map(|(_, attrs)| {
                    attrs
                        .iter()
                        .map(|(a, v)| format!("{a}={v:.3}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect::<Vec<_>>()
                .join(" | ");
            println!("[oracle] {who} chased zombie attrs: {pairs}");
            let data = s
                .data
                .iter()
                .filter(|(_, id, ..)| *id == cid)
                .map(|(_, _, acc, ser, val)| match val {
                    Some(v) => format!("acc{acc}/ser{ser}={v:.2}"),
                    None => format!("acc{acc}/ser{ser}"),
                })
                .collect::<Vec<_>>()
                .join(" ");
            println!("[oracle] {who} chased zombie data: {data}");
            let track = movement_track(s, cid, ci);
            let dist = |p: (f64, f64, f64)| {
                let (dx, dy, dz) = (p.0 - x, p.1 - y, p.2 - z);
                (dx * dx + dy * dy + dz * dz).sqrt()
            };
            if let (Some(first), Some(last)) = (track.first(), track.last()) {
                println!(
                    "[oracle] {who} chase track: {} samples, {:.1} -> {:.1}",
                    track.len(),
                    dist(*first),
                    dist(*last)
                );
            }
        } else {
            println!("[oracle] {who} chase: never fired");
        }
        println!(
            "[oracle] {who} damage frames after chase: {}, rotate_head: {}",
            s.damage
                .iter()
                .filter(|(i, _, _)| s.chase.is_some_and(|(ci, ..)| *i > ci))
                .count(),
            s.head_rots.len()
        );
    }
}

/// The center check's movement slack: when the tracker's add lands a
/// tick after the spawn, the zombie has already taken one stroll step
/// (observed 0.27 past center), while a mis-centered spawn (the missing
/// +0.5 sits a half-block out) still fails.
const CENTER_SLACK: f64 = 0.4;

/// Spawn geometry: block centers, integer feet, 24..128 from the bot.
fn check_spawn_geometry(who: &str, s: &Obs, failures: &mut Vec<String>) {
    for (ai, _, _, x, y, z, _) in s.adds.iter().filter(|(_, _, t, ..)| *t == ZOMBIE_TYPE) {
        if (x - x.floor() - 0.5).abs() > CENTER_SLACK || (z - z.floor() - 0.5).abs() > CENTER_SLACK
        {
            failures.push(format!(
                "{who}: zombie spawns off the block center ({x:.3},{z:.3})"
            ));
        }
        if y.fract() != 0.0 {
            failures.push(format!("{who}: zombie spawns at non-integer y {y:.3}"));
        }
        let Some((bx, by, bz)) = bot_pos_at(s, *ai) else {
            failures.push(format!(
                "{who}: no bot teleport before the zombie at ({x:.1},{y:.1},{z:.1})"
            ));
            continue;
        };
        let (dx, dy, dz) = (x - bx, y - by, z - bz);
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        if !(23.9..=128.1).contains(&dist) {
            failures.push(format!(
                "{who}: zombie spawns {dist:.1} from the bot, want 24..128"
            ));
        }
    }
}

/// The chased zombie's pairing packets: zero spawn motion, the health
/// datum, and the movement-speed attribute.
fn check_pairing(who: &str, s: &Obs, failures: &mut Vec<String>) {
    let Some((ci, cid, ..)) = s.chase else {
        failures.push(format!("{who}: the chase probe never fired"));
        return;
    };
    let tail = s
        .adds
        .iter()
        .find(|(ai, ..)| *ai == ci)
        .map(|(_, _, _, _, _, _, t)| *t);
    match tail {
        Some(0x00) => {}
        other => failures.push(format!(
            "{who}: chased zombie's movement byte {other:?}, want 00 (zero velocity)"
        )),
    }
    let health = s.data.iter().any(|(_, id, acc, ser, val)| {
        *id == cid && *acc == 9 && *ser == 3 && val.is_some_and(|v| (v - 20.0).abs() < 0.01)
    });
    if !health {
        failures.push(format!(
            "{who}: no health datum (accessor 9, FLOAT 20.0) for zombie {cid}"
        ));
    }
    // The reference's zombie snapshot carries movement speed alone;
    // default-valued attributes (max health included) are omitted.
    let speed = s.attrs.iter().any(|(id, attrs)| {
        *id == cid
            && attrs
                .iter()
                .any(|(a, v)| *a == 26 && (v - 0.23).abs() < 0.01)
    });
    if !speed {
        failures.push(format!(
            "{who}: no movement-speed attribute (id 26, 0.23) for zombie {cid}"
        ));
    }
}
