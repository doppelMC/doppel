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
/// gate's servers; CI uses the defaults.
fn vanilla_port() -> u16 {
    std::env::var("MOBS_VANILLA_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(VANILLA_PORT)
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
const P_SET_ENTITY_DATA: i32 = 0x65;
const P_ROTATE_HEAD: i32 = 0x55;
const P_UPDATE_ATTRIBUTES: i32 = 0x86;

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

/// One stream's reduced observation of the session.
#[derive(Default)]
struct Obs {
    /// Player teleports from player_position frames: (packet index, x,
    /// y, z) in arrival order.
    bot_pos: Vec<(usize, f64, f64, f64)>,
    /// Every add_entity: (packet index, entity id, type, x, y, z, the
    /// byte after the positions).
    adds: Vec<(usize, i32, i32, f64, f64, f64, u8)>,
    /// set_entity_data leading entries: (entity id, accessor,
    /// serializer, value when the serializer is FLOAT).
    data: Vec<(i32, u8, i32, Option<f32>)>,
    /// update_attributes payloads: (entity id, (attr id, value) pairs).
    attrs: Vec<(i32, Vec<(i32, f64)>)>,
    /// Absolute position syncs: (packet index, entity id, x, y, z).
    syncs: Vec<(usize, i32, f64, f64, f64)>,
    /// Delta moves: (packet index, entity id, dx, dy, dz).
    deltas: Vec<(usize, i32, f64, f64, f64)>,
    /// Damage events: (packet index, target entity id).
    damage: Vec<(usize, i32)>,
    /// rotate_head frames: packet indices.
    head_rots: Vec<usize>,
    /// set_equipment leading entries: (entity id, slot byte).
    equipment: Vec<(i32, u8)>,
    /// explode packets: (packet index, center x, y, z).
    explodes: Vec<(usize, f64, f64, f64)>,
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
        let mut o = 0usize;
        match p.id {
            // player_position: teleport id, position, ...
            P_PLAYER_POSITION => {
                if rd_varint(&raw, &mut o).is_some() {
                    if let (Some(x), Some(y), Some(z)) = (
                        rd_f64(&raw, &mut o),
                        rd_f64(&raw, &mut o),
                        rd_f64(&raw, &mut o),
                    ) {
                        obs.bot_pos.push((i, x, y, z));
                    }
                }
            }
            // add_entity: id, uuid(16), type, pos, movement, rotations...
            P_ADD_ENTITY => {
                if let Some(id) = rd_varint(&raw, &mut o) {
                    o += 16;
                    if let Some(typ) = rd_varint(&raw, &mut o) {
                        if let (Some(x), Some(y), Some(z)) = (
                            rd_f64(&raw, &mut o),
                            rd_f64(&raw, &mut o),
                            rd_f64(&raw, &mut o),
                        ) {
                            let tail = raw.get(o).copied().unwrap_or(0xff);
                            obs.adds.push((i, id, typ, x, y, z, tail));
                        }
                    }
                }
            }
            // set_entity_data: id, then entries; only the leading entry
            // decodes (it leads the list on both servers). INT entries
            // widen to f32 for the value slot.
            P_SET_ENTITY_DATA => {
                if let Some(id) = rd_varint(&raw, &mut o) {
                    if let Some(&accessor) = raw.get(o) {
                        o += 1;
                        if let Some(ser) = rd_varint(&raw, &mut o) {
                            let value = match ser {
                                3 => rd_f32(&raw, &mut o),
                                1 => raw
                                    .get(o..o + 4)
                                    .and_then(|b| <[u8; 4]>::try_from(b).ok())
                                    .map(i32::from_be_bytes)
                                    .map(|v| {
                                        o += 4;
                                        v as f32
                                    }),
                                _ => None,
                            };
                            obs.data.push((id, accessor, ser, value));
                        }
                    }
                }
            }
            // update_attributes: id, then the attribute list.
            P_UPDATE_ATTRIBUTES => {
                if let Some(id) = rd_varint(&raw, &mut o) {
                    if let Some(attrs) = rd_attrs(&raw, &mut o) {
                        obs.attrs.push((id, attrs));
                    }
                }
            }
            // entity_position_sync: id, path kind, pos, ...
            P_ENTITY_POSITION_SYNC => {
                if let (Some(id), Some(kind)) = (rd_varint(&raw, &mut o), rd_varint(&raw, &mut o)) {
                    if kind == 0 {
                        if let (Some(x), Some(y), Some(z)) = (
                            rd_f64(&raw, &mut o),
                            rd_f64(&raw, &mut o),
                            rd_f64(&raw, &mut o),
                        ) {
                            obs.syncs.push((i, id, x, y, z));
                        }
                    }
                }
            }
            P_MOVE_ENTITY_POS | P_MOVE_ENTITY_POS_ROT => {
                if let Some(id) = rd_varint(&raw, &mut o) {
                    if let (Some(dx), Some(dy), Some(dz)) = (
                        rd_i16(&raw, &mut o),
                        rd_i16(&raw, &mut o),
                        rd_i16(&raw, &mut o),
                    ) {
                        obs.deltas.push((
                            i,
                            id,
                            dx as f64 / DELTA_SCALE,
                            dy as f64 / DELTA_SCALE,
                            dz as f64 / DELTA_SCALE,
                        ));
                    }
                }
            }
            // damage_event: target id, type id, cause, direct cause.
            P_DAMAGE_EVENT => {
                if let Some(target) = rd_varint(&raw, &mut o) {
                    obs.damage.push((i, target));
                }
            }
            P_ROTATE_HEAD => {
                obs.head_rots.push(i);
            }
            // set_equipment: id, then the first slot byte.
            P_SET_EQUIPMENT => {
                if let Some(id) = rd_varint(&raw, &mut o) {
                    if let Some(&slot) = raw.get(o) {
                        obs.equipment.push((id, slot));
                    }
                }
            }
            // explode: center doubles lead the body.
            P_EXPLODE => {
                if let (Some(x), Some(y), Some(z)) = (
                    rd_f64(&raw, &mut o),
                    rd_f64(&raw, &mut o),
                    rd_f64(&raw, &mut o),
                ) {
                    obs.explodes.push((i, x, y, z));
                }
            }
            P_BLOCK_UPDATE | P_SECTION_BLOCKS_UPDATE => {
                obs.block_updates += 1;
            }
            // remove_entities: count then ids.
            PACKET_REMOVE_ENTITIES_LOCAL => {
                let mut ids = Vec::new();
                if let Some(count) = rd_varint(&raw, &mut o) {
                    for _ in 0..count.max(0) {
                        match rd_varint(&raw, &mut o) {
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
    for (i, eid, dx, dy, dz) in &obs.deltas {
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
    let commands: Vec<String> = vec![
        "gamerule spawn_mobs false".into(),
        "tp @s 100.5 -60 100.5".into(),
        "time set midnight".into(),
        "summon minecraft:skeleton 108.5 -60 100.5".into(),
        "tick step 180".into(),
        "tp @s 160.5 -60 100.5".into(),
        "tick step 20".into(),
        "summon minecraft:spider 160.5 -60 108.5".into(),
        "tick step 90".into(),
        "tp @s 100.5 -60 160.5".into(),
        "tick step 20".into(),
        "summon minecraft:creeper 104.5 -60 160.5".into(),
        "tick step 140".into(),
    ];
    let login = capture::login_start_c("Doppel");
    bot::login_capture(
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
    )
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

    capture_clean_blobs(&pin, &jar, &blobs_dir, &pristine_world)?;

    // Vanilla reference sessions. The default difficulty boots easy;
    // scenario one opens spawning at midnight, scenario two runs the
    // scripted summons.
    let vport = vanilla_port();
    let server = vanilla::boot(&pin, &jar, vport)?;
    let vworker = std::thread::spawn(move || -> Result<_> {
        let one = run_session(vport, protocol)?;
        let two = run_session2(vport, protocol)?;
        Ok((one, two))
    });
    let vdeadline = std::time::Instant::now() + Duration::from_secs(280);
    while !vworker.is_finished() && std::time::Instant::now() < vdeadline {
        std::thread::sleep(Duration::from_millis(500));
    }
    drop(server);
    let (v_pkts, v2_pkts) = vworker
        .join()
        .map_err(|_| anyhow::anyhow!("vanilla session thread panicked"))??;

    let bin = default_doppel_bin()?;
    let pin_path = doppel_protocol::pin_path()?;
    let mut child = Command::new(&bin)
        .env("DOPPEL_ADDR", "127.0.0.1")
        .env("DOPPEL_PORT", doppel_port().to_string())
        .env("DOPPEL_PIN", &pin_path)
        .env("DOPPEL_BLOBS", &blobs_dir)
        .env("DOPPEL_WORLD", &pristine_world)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawning {}", bin.display()))?;
    let dport = doppel_port();
    wait_for_port(dport, Duration::from_secs(30))?;
    std::thread::sleep(Duration::from_secs(2));
    let worker = std::thread::spawn(move || -> Result<_> {
        let one = run_session(dport, protocol)?;
        let two = run_session2(dport, protocol)?;
        Ok((one, two))
    });
    let deadline = std::time::Instant::now() + Duration::from_secs(280);
    while !worker.is_finished() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(500));
    }
    let _ = child.kill();
    let _ = child.wait();
    let (d_pkts, d2_pkts) = worker
        .join()
        .map_err(|_| anyhow::anyhow!("doppel session thread panicked"))??;

    let v = analyze(&v_pkts);
    let d = analyze(&d_pkts);
    let v2 = analyze(&v2_pkts);
    let d2 = analyze(&d2_pkts);
    report(&[("vanilla", &v_pkts, &v), ("doppel", &d_pkts, &d)]);
    report(&[("vanilla-s2", &v2_pkts, &v2), ("doppel-s2", &d2_pkts, &d2)]);

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
        let hits = s.damage.iter().filter(|(i, _)| *i > ci).count();
        if hits == 0 {
            failures.push(format!("{who}: no damage events after the chase"));
        }
    }

    // Scenario two: the scripted summons on both servers.
    for (who, s) in [("vanilla", &v2), ("doppel", &d2)] {
        check_scenario_two(who, s, &mut failures);
    }

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
            let landed = s.damage.iter().any(|(i, _)| i > ai);
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
            let swelled = s.data.iter().any(|(id, acc, ser, val)| {
                *id == cid && *acc == 16 && *ser == 1 && val.is_some_and(|v| v > 0.0)
            });
            if !swelled {
                failures.push(format!(
                    "{who}: creeper {cid} never swells (accessor 16 INT)"
                ));
            }
            let removed = s.removes.iter().any(|(_, ids)| ids.contains(&cid));
            if !removed {
                failures.push(format!("{who}: creeper {cid} never removed"));
            }
            let near_blast = s.explodes.iter().any(|(_, x, y, z)| {
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
                let dist = |p: &(f64, f64, f64)| {
                    let (dx, dz) = (p.0 - 160.5, p.2 - 100.5);
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
                .filter(|(id, ..)| *id == cid)
                .map(|(_, acc, ser, val)| match val {
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
                .filter(|(i, _)| s.chase.is_some_and(|(ci, ..)| *i > ci))
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
    let health = s.data.iter().any(|(id, acc, ser, val)| {
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
