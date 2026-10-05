//! The differential survival gate: the survival loop's full round trip.
//! A scripted session covers grass and insta-breaks of two torches, then
//! a third connection pairs, unpairs, and re-pairs the drops: paced
//! teleport commands stage it beside the drops, walk it 79 blocks out
//! (past this server's 64-block item reach), and bring it back over a
//! chunk line, where this server re-pairs at the next pairing pass.
//! One final teleport walks the walker onto the first drop, so the
//! return's pairing and the pickup vacuum fire in a fixed order on both
//! servers. The second drop rests through the whole window (the fall,
//! the landing sync, the resting cadence); the random tick pass runs
//! amplified (gamerule random_tick_speed) so decay and spread land
//! inside the capture window. Both sides must agree on each drop's
//! spawn offset formula, the ordered movement-frame sequence against a
//! shadow of the reference tracker, the pairing lifecycle, the take
//! animation, removal, the inventory sync, and the grass transitions.

use anyhow::{Context, Result};
use doppel_protocol::load_pin;
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::parity_break::{
    capture_clean_blobs, decode_update_writes, default_doppel_bin, wait_for_port,
};
use crate::{bot, capture, vanilla};

const VANILLA_PORT: u16 = 25566;
const DOPPEL_PORT: u16 = 25565;

/// Local-run overrides for machines where another server holds a default
/// port; CI uses the defaults.
fn vanilla_port() -> u16 {
    std::env::var("SURVIVAL_VANILLA_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(VANILLA_PORT)
}

/// The doppel-side override twin.
fn doppel_port() -> u16 {
    std::env::var("SURVIVAL_DOPPEL_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DOPPEL_PORT)
}

// The flat world's surface: grass at y=-61, plants at y=-60 (the spawn
// teleport pins the standing height at -60.0).
const GRASS_Y: i32 = -61;
const PLANT_Y: i32 = -60;
/// The dug torch the walk returns onto (an insta-break block).
const TORCH: (i32, i32, i32) = (2, PLANT_Y, 8);
/// The second dug torch: its drop never meets a player, so it falls,
/// lands, and rests through the whole capture window. Four blocks from
/// the first torch along z: a slide carries a drop at most ~1.3 from
/// its cell center, so the two settle zones stay at least 1.4 apart -
/// outside the 0.75 merge box on every draw.
const TORCH_B: (i32, i32, i32) = (2, PLANT_Y, 12);
/// The stone caps (PLANT_Y cells) whose covered grass decays.
const CAP_X: std::ops::RangeInclusive<i32> = 2..=5;
const CAP_Z: std::ops::RangeInclusive<i32> = 2..=5;
/// The bare-dirt patch (GRASS_Y cells) the surrounding grass reclaims.
const PATCH_X: std::ops::RangeInclusive<i32> = 8..=10;
const PATCH_Z: std::ops::RangeInclusive<i32> = 2..=4;
/// The random tick rate the window runs at (the default 3 is too slow
/// for a bounded capture).
const TICK_SPEED: usize = 300;
/// The shadow's tick budget: far past any fall-and-slide prefix (the
/// settle sync lands within the first ~40 ticks) and deep into the
/// rest that follows, yet short of the tracker's teleport-delay full
/// sync (the 401st gate opening, around pass 1600) - a sync the
/// observed windows never reach and the prefix compare must not see.
const SIM_TICKS: i32 = 600;

// The reference tracker's constants (mirrored by the shadow below).
const TRACK_SYNC_INTERVAL: i32 = 20;
const TRACK_FULL_INTERVAL: i32 = 60;
const TRACK_TELEPORT_CAP: i32 = 400;
const TRACK_POSITION_EPS: f64 = 7.62939453125e-6;
const TRACK_MOTION_EPS: f64 = 1.0e-7;
const TRACK_DELTA_SCALE: f64 = 4096.0;
/// One LpVec3 component's wire resolution (scale 1): 2/32766.
const LP_STEP: f64 = 2.0 / 32766.0;
/// The walker-tail compare budget past a drop's re-add: the resting
/// cadence's period fits several times over, and the capture lengths
/// past it follow the servers' tick rates.
const WALKER_TAIL_MAX: usize = 30;

/// player_action body: action VarInt, packed pos, direction, sequence.
fn build_player_action(action: i32, pos: (i32, i32, i32), sequence: i32) -> Vec<u8> {
    let mut b = Vec::new();
    doppel_protocol::write_varint(&mut b, action);
    b.extend_from_slice(&bot::pack_block_pos(pos.0, pos.1, pos.2).to_be_bytes());
    doppel_protocol::write_varint(&mut b, 1); // direction: up
    doppel_protocol::write_varint(&mut b, sequence);
    b
}

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
    let v = i16::from_be_bytes([raw[*o], raw[*o + 1]]);
    *o += 2;
    Some(v)
}

/// The packed movement vector (LpVec3): a zero byte for a zero vector,
/// else six little-endian bytes of markers plus chessboard-quantized
/// components, with a VarInt scale continuation when the scale exceeds
/// its two marker bits.
fn rd_lp_vec3(raw: &[u8], o: &mut usize) -> Option<(f64, f64, f64)> {
    let b0 = *raw.get(*o)?;
    *o += 1;
    if b0 == 0 {
        return Some((0.0, 0.0, 0.0));
    }
    if *o + 5 > raw.len() {
        return None;
    }
    // Bytes 2..5 carry bits 16..47 big-endian (the encoder writes the
    // marker pair low, then the high half as one big-endian u32).
    let mut buffer: u64 = b0 as u64 | ((*raw.get(*o)? as u64) << 8);
    for (i, shift) in [16u32, 24, 32, 40].into_iter().enumerate() {
        buffer |= (*raw.get(*o + 4 - i)? as u64) << shift;
    }
    *o += 5;
    let partial = b0 & 0x04 != 0;
    let scale = if partial {
        let low = i64::from(b0 & 0x03);
        let high = i64::from(rd_varint(raw, o)?);
        (high << 2) | low
    } else {
        i64::from(b0 & 0x03)
    };
    let unpack = |shift: u32| ((buffer >> shift) & 0x7fff) as f64;
    let decode = |stored: f64| (stored / 32766.0 * 2.0 - 1.0) * scale as f64;
    Some((decode(unpack(3)), decode(unpack(18)), decode(unpack(33))))
}

/// One entity frame, decoded in stream order.
#[derive(Clone, Copy, PartialEq)]
enum EFrame {
    /// add_entity: the id and the entity-type id (late pairings key on
    /// the pair, not the position: a re-pair add carries the settled
    /// position, far from the spawn cell).
    Add(i32, i32),
    Data(i32),
    Motion(i32, [f64; 3]),
    /// move_entity_pos: the on-ground bit and the 1/4096-block deltas.
    Pos(i32, bool, [i32; 3]),
    /// move_entity_pos_rot: same body plus two rotation bytes.
    PosRot(i32, bool, [i32; 3]),
    Sync(i32, f64, f64, f64, bool),
    Take(i32, i32, i32),
    Remove(i32),
}

impl EFrame {
    fn entity(&self) -> i32 {
        match self {
            EFrame::Add(e, _)
            | EFrame::Data(e)
            | EFrame::Motion(e, _)
            | EFrame::Pos(e, _, _)
            | EFrame::PosRot(e, _, _)
            | EFrame::Sync(e, _, _, _, _)
            | EFrame::Take(e, _, _)
            | EFrame::Remove(e) => *e,
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            EFrame::Add(..) => "add",
            EFrame::Data(_) => "data",
            EFrame::Motion(_, _) => "motion",
            EFrame::Pos(_, _, _) => "pos",
            EFrame::PosRot(_, _, _) => "posrot",
            EFrame::Sync(_, _, _, _, _) => "sync",
            EFrame::Take(_, _, _) => "take",
            EFrame::Remove(_) => "remove",
        }
    }
}

/// One stream's reduced observation of the session. `marker` splits the
/// scripted volley (command feedback) from the interaction traffic; the
/// entity and inventory decodes count post-volley frames only, the block
/// writes cover the whole stream.
#[derive(Default)]
struct Obs {
    /// Every post-volley add_entity: (entity id, type, x, y, z, movement).
    adds: Vec<(i32, i32, f64, f64, f64, [f64; 3])>,
    /// set_entity_data item payloads: (entity id, count, item id).
    stacks: Vec<(i32, i32, i32)>,
    /// linear entity_position_sync frames: (entity id, x, y, z, ground).
    syncs: Vec<(i32, f64, f64, f64, bool)>,
    /// take_item_entity frames: (item entity, player, amount).
    takes: Vec<(i32, i32, i32)>,
    /// remove_entities frames: (count, first id).
    removes: Vec<(i32, i32)>,
    /// container set_slot frames carrying a stack: (menu slot, item,
    /// count).
    slots: Vec<(i32, i32, i32)>,
    /// block writes: (pos, state), in order.
    writes: Vec<((i32, i32, i32), u32)>,
    /// every entity frame, in stream order.
    eframes: Vec<EFrame>,
}

impl Obs {
    /// The item-entity adds inside one block cell, in order (the spawn
    /// pairing first, any re-pairing after).
    fn adds_in(&self, cell: (i32, i32, i32)) -> Vec<(i32, i32, f64, f64, f64, [f64; 3])> {
        self.adds
            .iter()
            .filter(|(id, typ, x, y, z, _)| {
                *typ == 72
                    && (x - cell.0 as f64 - 0.5).abs() <= 0.35
                    && (z - cell.2 as f64 - 0.5).abs() <= 0.35
                    && (PLANT_Y as f64 - 0.1..=PLANT_Y as f64 + 0.8).contains(y)
                    && *id >= 0
            })
            .copied()
            .collect()
    }

    /// The movement, take, and removal frames after the entity's last
    /// add (the re-pairing boundary): the frames its sync broadcast sent.
    fn tail_frames(&self, id: i32) -> Vec<EFrame> {
        let last_add = self
            .eframes
            .iter()
            .rposition(|f| matches!(f, EFrame::Add(e, _) if *e == id))
            .unwrap_or(usize::MAX);
        self.eframes
            .iter()
            .skip(last_add + 1)
            .filter(|f| f.entity() == id && !matches!(f, EFrame::Add(..) | EFrame::Data(_)))
            .copied()
            .collect()
    }

    /// The stream positions of one item-typed entity's add frames (the
    /// walker's late pairings: their adds carry the settled position,
    /// outside the spawn cell, so the cell window cannot find them).
    fn item_add_frames(&self, id: i32) -> Vec<usize> {
        self.eframes
            .iter()
            .enumerate()
            .filter(|(_, f)| matches!(f, EFrame::Add(e, t) if *e == id && *t == 72))
            .map(|(i, _)| i)
            .collect()
    }
}

fn analyze(pkts: &[bot::CapturedPacket], marker: Option<usize>) -> Obs {
    let mut obs = Obs::default();
    let refs: Vec<_> = pkts.iter().filter(|p| p.id >= 0).collect();
    obs.writes = decode_update_writes(&refs)
        .into_iter()
        .map(|(p, s, _)| (p, s))
        .collect();
    let after = |i: usize| marker.is_none_or(|m| i > m);
    for (i, p) in pkts.iter().enumerate() {
        if p.id < 0 || !after(i) {
            continue;
        }
        let raw = hex::decode(&p.head_hex).unwrap_or_default();
        let mut o = 0usize;
        match p.id {
            // add_entity: id, uuid(16), type, pos, movement, rotations...
            0x01 => {
                let Some(id) = rd_varint(&raw, &mut o) else {
                    continue;
                };
                o += 16; // uuid
                let Some(typ) = rd_varint(&raw, &mut o) else {
                    continue;
                };
                if let (Some(x), Some(y), Some(z)) = (
                    rd_f64(&raw, &mut o),
                    rd_f64(&raw, &mut o),
                    rd_f64(&raw, &mut o),
                ) {
                    let movement = rd_lp_vec3(&raw, &mut o)
                        .map(|(x, y, z)| [x, y, z])
                        .unwrap_or([0.0; 3]);
                    obs.adds.push((id, typ, x, y, z, movement));
                    obs.eframes.push(EFrame::Add(id, typ));
                }
            }
            // set_entity_data: id, then packed entries; only the item
            // accessor decodes (it leads the list on both servers).
            0x65 => {
                let Some(id) = rd_varint(&raw, &mut o) else {
                    continue;
                };
                if raw.get(o) != Some(&8) {
                    continue;
                }
                o += 1;
                if rd_varint(&raw, &mut o) != Some(7) {
                    continue;
                }
                let (Some(count), Some(item)) = (rd_varint(&raw, &mut o), rd_varint(&raw, &mut o))
                else {
                    continue;
                };
                obs.stacks.push((id, count, item));
                obs.eframes.push(EFrame::Data(id));
            }
            // entity_position_sync: id, path kind, pos, rotations, flag.
            0x23 => {
                let (Some(id), Some(kind)) = (rd_varint(&raw, &mut o), rd_varint(&raw, &mut o))
                else {
                    continue;
                };
                if kind != 0 {
                    continue;
                }
                if let (Some(x), Some(y), Some(z)) = (
                    rd_f64(&raw, &mut o),
                    rd_f64(&raw, &mut o),
                    rd_f64(&raw, &mut o),
                ) {
                    let _ = (rd_f32(&raw, &mut o), rd_f32(&raw, &mut o));
                    let ground = raw.get(o).is_some_and(|b| *b != 0);
                    obs.syncs.push((id, x, y, z, ground));
                    obs.eframes.push(EFrame::Sync(id, x, y, z, ground));
                }
            }
            // move_entity_pos: id, properties (on-ground bit), 3 shorts.
            0x36 => {
                let (Some(id), Some(props)) = (rd_varint(&raw, &mut o), rd_varint(&raw, &mut o))
                else {
                    continue;
                };
                let ground = props & 1 != 0;
                if let (Some(xa), Some(ya), Some(za)) = (
                    rd_i16(&raw, &mut o),
                    rd_i16(&raw, &mut o),
                    rd_i16(&raw, &mut o),
                ) {
                    obs.eframes
                        .push(EFrame::Pos(id, ground, [xa as i32, ya as i32, za as i32]));
                }
            }
            // move_entity_pos_rot: the pos body plus two angle bytes.
            0x37 => {
                let (Some(id), Some(props)) = (rd_varint(&raw, &mut o), rd_varint(&raw, &mut o))
                else {
                    continue;
                };
                let ground = props & 1 != 0;
                if let (Some(xa), Some(ya), Some(za)) = (
                    rd_i16(&raw, &mut o),
                    rd_i16(&raw, &mut o),
                    rd_i16(&raw, &mut o),
                ) {
                    obs.eframes.push(EFrame::PosRot(
                        id,
                        ground,
                        [xa as i32, ya as i32, za as i32],
                    ));
                }
            }
            // set_entity_motion: id plus the packed movement vector.
            0x67 => {
                let Some(id) = rd_varint(&raw, &mut o) else {
                    continue;
                };
                if let Some((x, y, z)) = rd_lp_vec3(&raw, &mut o) {
                    obs.eframes.push(EFrame::Motion(id, [x, y, z]));
                }
            }
            // take_item_entity: item, player, amount.
            0x7f => {
                if let (Some(item), Some(player), Some(amount)) = (
                    rd_varint(&raw, &mut o),
                    rd_varint(&raw, &mut o),
                    rd_varint(&raw, &mut o),
                ) {
                    obs.takes.push((item, player, amount));
                    obs.eframes.push(EFrame::Take(item, player, amount));
                }
            }
            // remove_entities: count, then ids.
            0x4e => {
                if let Some(count) = rd_varint(&raw, &mut o) {
                    let first = rd_varint(&raw, &mut o).unwrap_or(-1);
                    obs.removes.push((count, first));
                    obs.eframes.push(EFrame::Remove(first));
                }
            }
            // container set_slot: container, state, slot i16, stack.
            0x14 => {
                if rd_varint(&raw, &mut o).is_none() || rd_varint(&raw, &mut o).is_none() {
                    continue;
                }
                if o + 2 > raw.len() {
                    continue;
                }
                let slot = i16::from_be_bytes([raw[o], raw[o + 1]]);
                o += 2;
                let Some(count) = rd_varint(&raw, &mut o) else {
                    continue;
                };
                if count <= 0 {
                    continue; // empty-slot syncs carry no stack
                }
                let Some(item) = rd_varint(&raw, &mut o) else {
                    continue;
                };
                obs.slots.push((slot as i32, item, count));
            }
            _ => {}
        }
    }
    obs
}

/// The scripted session triple against one server: a fixed witness, a
/// digger that stages beside the torches and insta-breaks both in one
/// burst, and a walker that pairs, unpairs, and re-pairs the drops with
/// paced teleports (commands ride the tick loop; a raw move burst would
/// trip the reference's move-speed checks). The walker's final teleport
/// lands on the first drop, so the vacuum fires after the re-pairing.
fn run_sessions(
    port: u16,
    protocol: i32,
) -> Result<(
    Vec<bot::CapturedPacket>,
    Vec<bot::CapturedPacket>,
    Vec<bot::CapturedPacket>,
)> {
    let mut commands = Vec::new();
    // 26.3 gates spawning on gamerules (the server.properties flags are
    // gone); silence every spawner so nothing wanders into the cells.
    for rule in [
        "spawn_mobs",
        "spawn_monsters",
        "spawn_patrols",
        "spawn_phantoms",
        "spawn_wandering_traders",
    ] {
        commands.push(format!("gamerule {rule} false"));
    }
    // Stage the digger before any cell is written: the world spawn moves
    // between runs and could otherwise leave a player standing inside the
    // setup cells for the whole volley. The post stands 1.7 blocks east
    // of both settle zones' near edge - the vacuum needs both horizontal
    // axes inside 1.425, so the drops are safe - while sitting 3.6
    // blocks from each torch (inside the 4.5-block dig reach), so the
    // drops live until the walker collects one. The digger walks the
    // commandless witness to its fixed post the same way (the witness
    // holds no op rights; only the named teleport can move it).
    commands.push(format!("tp @s 5.5 {} 10.5", PLANT_Y));
    commands.push(format!("tp Doppelist2 2.5 {} -3.5", PLANT_Y));
    for x in CAP_X.clone() {
        for z in CAP_Z.clone() {
            commands.push(format!("setblock {x} {PLANT_Y} {z} minecraft:stone"));
        }
    }
    for x in PATCH_X.clone() {
        for z in PATCH_Z.clone() {
            commands.push(format!("setblock {x} {PLANT_Y} {z} minecraft:air"));
        }
    }
    for x in PATCH_X.clone() {
        for z in PATCH_Z.clone() {
            commands.push(format!("setblock {x} {GRASS_Y} {z} minecraft:dirt"));
        }
    }
    commands.push(format!(
        "setblock {} {} {} minecraft:torch",
        TORCH.0, TORCH.1, TORCH.2
    ));
    commands.push(format!(
        "setblock {} {} {} minecraft:torch",
        TORCH_B.0, TORCH_B.1, TORCH_B.2
    ));
    commands.push(format!("gamerule random_tick_speed {TICK_SPEED}"));
    let raw: Vec<(i32, Vec<u8>)> = vec![
        (0x2c, Vec::new()), // player_loaded: the reference gates input on it
        // Insta-break both torches: the drops spawn with the digs.
        (0x29, build_player_action(0, TORCH, 10)),
        (0x29, build_player_action(0, TORCH_B, 11)),
    ];
    // The witness observes from a fixed post: it joins first and runs no
    // commands at all (no op rights, no replies to pace on); the digger's
    // named teleport walks it to the post before any cell is written. The
    // post is 12 blocks from the torches (outside pickup range, inside
    // the view distance), so it holds the drops' pairing through the
    // whole window.
    let witness_commands: Vec<String> = Vec::new();
    // The walker's legs, each padded with no-op gamerule re-sets so every
    // teleport lands on its own tick: stage east of the cap (outside its
    // stone cells and both settle zones' pickup boxes), walk 79 blocks
    // out (past this server's 64-block item reach), back to the staging
    // spot, then onto the first drop's settle zone. This server applies
    // a teleport's position at the client's ack and re-pairs at the next
    // pairing pass over the settled view: the chunk-boundary hop after
    // the padding is that pass, and the second hop doubles it. All land
    // before the vacuum's take.
    let mut walker_commands = vec![format!("tp @s 6.5 {} 5.5", PLANT_Y)];
    for _ in 0..2 {
        walker_commands.push(format!("gamerule random_tick_speed {TICK_SPEED}"));
    }
    walker_commands.push(format!("tp @s 6.5 {} -70.5", PLANT_Y));
    for _ in 0..10 {
        walker_commands.push(format!("gamerule random_tick_speed {TICK_SPEED}"));
    }
    walker_commands.push(format!("tp @s 6.5 {} 5.5", PLANT_Y));
    for _ in 0..10 {
        walker_commands.push(format!("gamerule random_tick_speed {TICK_SPEED}"));
    }
    walker_commands.push(format!("tp @s 16.5 {} 4.5", PLANT_Y));
    for _ in 0..2 {
        walker_commands.push(format!("gamerule random_tick_speed {TICK_SPEED}"));
    }
    walker_commands.push(format!("tp @s 6.5 {} 5.5", PLANT_Y));
    for _ in 0..2 {
        walker_commands.push(format!("gamerule random_tick_speed {TICK_SPEED}"));
    }
    // The closing teleport stands inside drop A's settle zone on both
    // axes (each within 1.3 of the landing, under the 1.425 box) and at
    // least 2.7 blocks from drop B's zone (outside it).
    walker_commands.push(format!("tp @s 2.5 {} 8.5", PLANT_Y));

    let dump_root = std::env::var_os("DOPPEL_SURVIVAL_DUMP").map(|v| {
        let p = std::path::PathBuf::from(v).join(format!("p{port}"));
        let _ = std::fs::create_dir_all(&p);
        p
    });
    let sub = |name: &str| dump_root.as_deref().map(|p| p.join(name));

    let witness_login = capture::login_start_c("Doppelist2");
    let wit_dump = sub("wit");
    let witness = std::thread::spawn(move || {
        bot::login_capture(
            "127.0.0.1",
            port,
            protocol,
            &witness_login,
            &bot::CaptureOpts {
                idle_timeout: Some(Duration::from_secs(30)),
                max_packets: Some(24000),
                dump_dir: wit_dump.as_deref(),
                commands: &witness_commands,
                raw_packets: &[],
                walk_chunks: None,
            },
        )
    });
    std::thread::sleep(Duration::from_secs(5));
    let login = capture::login_start_c("Doppel");
    let dig_dump = sub("dig");
    let digger = std::thread::spawn(move || {
        bot::login_capture(
            "127.0.0.1",
            port,
            protocol,
            &login,
            &bot::CaptureOpts {
                idle_timeout: Some(Duration::from_secs(25)),
                max_packets: Some(12000),
                dump_dir: dig_dump.as_deref(),
                commands: &commands,
                raw_packets: &raw,
                walk_chunks: None,
            },
        )
    });
    // The walker joins late enough that the digger's volley (and its raw
    // dig burst) has already spawned the drops: the legs pair, unpair,
    // and re-pair against LIVING drops, and the final teleport's vacuum
    // lands on a rested drop.
    std::thread::sleep(Duration::from_secs(10));
    let walker_login = capture::login_start_c("Doppelist");
    let walk_dump = sub("walk");
    let walker = std::thread::spawn(move || {
        bot::login_capture(
            "127.0.0.1",
            port,
            protocol,
            &walker_login,
            &bot::CaptureOpts {
                idle_timeout: Some(Duration::from_secs(25)),
                max_packets: Some(8000),
                dump_dir: walk_dump.as_deref(),
                commands: &walker_commands,
                raw_packets: &[],
                walk_chunks: None,
            },
        )
    });
    let digger = digger
        .join()
        .map_err(|_| anyhow::anyhow!("digger thread panicked"))?
        .context("capturing digger session")?;
    let walker = walker
        .join()
        .map_err(|_| anyhow::anyhow!("walker thread panicked"))?
        .context("capturing walker session")?;
    let witness = witness
        .join()
        .map_err(|_| anyhow::anyhow!("witness thread panicked"))?
        .context("capturing witness session")?;
    Ok((digger, walker, witness))
}

// ---------------------------------------------------------------------
// The shadow: the reference item physics and tracker gates, replayed
// from an observed spawn.
// ---------------------------------------------------------------------

/// One wire unit of a movement component: floor(v * 4096 + 0.5).
fn track_encode(v: f64) -> i64 {
    (v * TRACK_DELTA_SCALE + 0.5).floor() as i64
}

/// The quantization error a component would carry on the wire.
fn track_loss(v: f64) -> f64 {
    track_encode(v) as f64 / TRACK_DELTA_SCALE - v
}

/// The shadow's copy of one drop: the physics state plus the tracker
/// counters the flush carries.
#[derive(Clone, Copy)]
struct SimDrop {
    id: i32,
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
    /// The entity's own tick counter (the rest gate phases on it).
    tick: i32,
    /// The tracker's pass counter (the sync cadence phases on it).
    pass: i32,
    base: (f64, f64, f64),
    last_movement: (f64, f64, f64),
    teleport_delay: i32,
    was_on_ground: bool,
}

/// One predicted frame.
#[derive(Clone, Copy, PartialEq)]
enum SimFrame {
    Motion([f64; 3]),
    Pos([i64; 3], bool),
    Sync((f64, f64, f64), bool),
}

impl SimFrame {
    fn kind(&self) -> &'static str {
        match self {
            SimFrame::Motion(_) => "motion",
            SimFrame::Pos(_, _) => "pos",
            SimFrame::Sync(_, _) => "sync",
        }
    }
}

/// One physics tick of the reference item entity over a flat floor: the
/// gravity step, the rest-gated move with zeroed collision components,
/// and the post-move drags.
fn sim_tick(s: &mut SimDrop, floor: f64) {
    let old = (s.vx, s.vy, s.vz);
    s.tick += 1;
    s.vy -= 0.04;
    let horizontal = s.vx * s.vx + s.vz * s.vz;
    if !s.on_ground || horizontal > 1.0e-5 || (s.tick + s.id).rem_euclid(4) == 0 {
        // The floor plane is the only collider in the scenario's field.
        let dy = if s.y + s.vy < floor {
            floor - s.y
        } else {
            s.vy
        };
        s.y += dy;
        s.x += s.vx;
        s.z += s.vz;
        s.vertical_collision = dy != s.vy;
        s.horizontal_collision = false;
        s.on_ground = s.vertical_collision && s.vy < 0.0;
        if s.vertical_collision && s.vy != 0.0 {
            s.vy = 0.0;
        }
        let friction = if s.on_ground { 0.98 * 0.6 } else { 0.98 };
        s.vx *= friction;
        s.vz *= friction;
        s.vy *= 0.98;
    }
    let (dvx, dvy, dvz) = (s.vx - old.0, s.vy - old.1, s.vz - old.2);
    if dvx * dvx + dvy * dvy + dvz * dvz > 0.01 {
        s.needs_sync = true;
    }
}

/// One tracker pass: the gate, the motion packet ahead of the move
/// packet, the short-delta-versus-full-sync choice, and the base update.
fn sim_pass(s: &mut SimDrop, out: &mut Vec<SimFrame>) {
    let gate = s.needs_sync || s.pass % TRACK_SYNC_INTERVAL == 0;
    if gate {
        s.teleport_delay += 1;
        let (bx, by, bz) = s.base;
        let (dx, dy, dz) = (s.x - bx, s.y - by, s.z - bz);
        let position_changed = dx * dx + dy * dy + dz * dz >= TRACK_POSITION_EPS;
        let should_send_position = position_changed || s.pass % TRACK_FULL_INTERVAL == 0;
        let (xa, ya, za) = (
            track_encode(s.x) - track_encode(bx),
            track_encode(s.y) - track_encode(by),
            track_encode(s.z) - track_encode(bz),
        );
        let full = s.teleport_delay > TRACK_TELEPORT_CAP || s.was_on_ground != s.on_ground;
        let kind = if full {
            s.was_on_ground = s.on_ground;
            s.teleport_delay = 0;
            2
        } else if should_send_position {
            let wire_short = -32768..=32767;
            let too_big =
                !wire_short.contains(&xa) || !wire_short.contains(&ya) || !wire_short.contains(&za);
            let full_precision = (s.vertical_collision
                && (xa != 0 && track_loss(s.x) != 0.0 || za != 0 && track_loss(s.z) != 0.0))
                || (s.horizontal_collision
                    && (xa != 0 && track_loss(s.z) != 0.0 || za != 0 && track_loss(s.x) != 0.0));
            if too_big || full_precision {
                2
            } else {
                1
            }
        } else {
            0
        };
        // The motion packet rides ahead of the move packet.
        let (mx, my, mz) = (
            s.vx - s.last_movement.0,
            s.vy - s.last_movement.1,
            s.vz - s.last_movement.2,
        );
        let diff = mx * mx + my * my + mz * mz;
        let still = s.vx == 0.0 && s.vy == 0.0 && s.vz == 0.0;
        if diff > TRACK_MOTION_EPS || (diff > 0.0 && still) {
            s.last_movement = (s.vx, s.vy, s.vz);
            out.push(SimFrame::Motion([s.vx, s.vy, s.vz]));
        }
        match kind {
            2 => {
                out.push(SimFrame::Sync((s.x, s.y, s.z), s.on_ground));
                s.base = (s.x, s.y, s.z);
            }
            1 => {
                out.push(SimFrame::Pos([xa, ya, za], s.on_ground));
                s.base = (s.x, s.y, s.z);
            }
            _ => {}
        }
        s.needs_sync = false;
    }
    s.pass += 1;
}

/// The shadow replay: per game tick, one tracker pass (the flush runs
/// ahead of the entity pass), then the physics. The drop rests through
/// the whole budget; the observed streams cut the tail early (their
/// capture windows) or end it with the take and removal frames.
fn simulate(spawn: &(i32, i32, f64, f64, f64, [f64; 3])) -> Vec<SimFrame> {
    let (id, _, x, y, z, movement) = *spawn;
    let mut s = SimDrop {
        id,
        x,
        y,
        z,
        vx: movement[0],
        vy: movement[1],
        vz: movement[2],
        on_ground: false,
        vertical_collision: false,
        horizontal_collision: false,
        needs_sync: false,
        tick: 0,
        pass: 0,
        base: (x, y, z),
        last_movement: (movement[0], movement[1], movement[2]),
        teleport_delay: 0,
        was_on_ground: false,
    };
    // The floor: the grass top under the plant cell.
    let floor = (PLANT_Y - 1) as f64 + 1.0;
    let mut out = Vec::new();
    for _ in 1..=SIM_TICKS {
        sim_pass(&mut s, &mut out);
        sim_tick(&mut s, floor);
    }
    out
}

/// Two observed tails of the same drop, one per server: identical frame
/// kinds and values inside the wire tolerances, rest syncs excepted (a
/// resting drop's absolute position belongs to its server's spawn draw,
/// so those compare ground bits only; a resting cadence slot upgrades
/// to a full sync only where that draw sits off the 4096 grid, so a
/// zero-delta position frame and a sync name the same slot). The
/// cadences phase on the entity id, so a cadence divergence shows up
/// here as a slot-count mismatch. Both tails run from the re-pair at
/// the closing legs (this server's at the hop's pairing pass, the
/// reference's when its away-view work completes), and each capture's
/// end jitters a couple of ticks, so the tails may differ in length by
/// up to 4 frames; the shared prefix carries the comparison.
fn compare_obs_pair(name: &str, a: &[EFrame], b: &[EFrame], failures: &mut Vec<String>) {
    let head = |f: &[EFrame]| -> String {
        f.iter()
            .take(12)
            .map(|x| x.kind())
            .collect::<Vec<_>>()
            .join(",")
    };
    if a.is_empty() && b.is_empty() {
        failures.push(format!("drop {name}: the walker saw no movement frames"));
        return;
    }
    let shared = a.len().min(b.len());
    if shared < 2 {
        failures.push(format!(
            "drop {name} walker tails share only {shared} frames: vanilla {} [{}] vs doppel {} [{}]",
            a.len(),
            head(a),
            b.len(),
            head(b)
        ));
        return;
    }
    if a.len().abs_diff(b.len()) > 4 {
        failures.push(format!(
            "drop {name} walker tails drift {} frames apart: vanilla {} vs doppel {}",
            a.len().abs_diff(b.len()),
            a.len(),
            b.len()
        ));
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        // A resting cadence slot's kind rides the server's own lattice
        // phase: the periodic send upgrades to a full sync only where
        // the draw's absolute position sits off the 4096 grid.
        let slot_swap = matches!(
            (x, y),
            (EFrame::Sync(..), EFrame::Pos(_, _, [0, 0, 0]))
                | (EFrame::Pos(_, _, [0, 0, 0]), EFrame::Sync(..))
        );
        if x.kind() != y.kind() && !slot_swap {
            failures.push(format!(
                "drop {name} walker frame {i}: vanilla {} vs doppel {} (vanilla [{}], doppel [{}])",
                x.kind(),
                y.kind(),
                head(a),
                head(b)
            ));
            return;
        }
        let bad = match (x, y) {
            (EFrame::Motion(_, ov), EFrame::Motion(_, dv)) => ov
                .iter()
                .zip(dv.iter())
                .any(|(p, q)| (p - q).abs() > 1.0e-4),
            (EFrame::Pos(_, og, od), EFrame::Pos(_, dg, dd))
            | (EFrame::PosRot(_, og, od), EFrame::Pos(_, dg, dd)) => {
                *og != *dg || od.iter().zip(dd.iter()).any(|(p, q)| (p - q).abs() > 8)
            }
            // A resting drop's full sync carries its absolute position,
            // which each server drew at the spawn; the ground bit is the
            // shared signal (the cadence slot count and the short deltas
            // carry the rest).
            (EFrame::Sync(_, _, _, _, og), EFrame::Sync(_, _, _, _, dg)) => *og != *dg,
            (EFrame::Pos(_, og, [0, 0, 0]), EFrame::Sync(_, _, _, _, dg))
            | (EFrame::Sync(_, _, _, _, og), EFrame::Pos(_, dg, [0, 0, 0])) => *og != *dg,
            (EFrame::Take(oi, op, oa), EFrame::Take(di, dp, da)) => (oi, op, oa) != (di, dp, da),
            _ => false,
        };
        if bad {
            failures.push(format!(
                "drop {name} walker frame {i} ({}) disagrees across servers",
                x.kind()
            ));
            return;
        }
    }
}

/// The last written state at a position.
fn final_at(obs: &Obs, pos: (i32, i32, i32)) -> Option<u32> {
    obs.writes
        .iter()
        .rev()
        .find(|(p, _)| *p == pos)
        .map(|(_, s)| *s)
}

/// The cap cells that ended on the decay product: (count, state). Cells
/// with no write at all do not count.
fn decayed(obs: &Obs) -> (usize, Option<u32>) {
    let somes: Vec<u32> = (CAP_X.clone())
        .flat_map(|x| (CAP_Z.clone()).map(move |z| (x, z)))
        .filter_map(|(x, z)| final_at(obs, (x, GRASS_Y, z)))
        .collect();
    let state = somes.first().copied();
    let count = match state {
        Some(s) => somes.iter().filter(|&&x| x == s).count(),
        None => 0,
    };
    (count, state)
}

/// The patch cells whose final state left the setblock'd dirt behind:
/// (count, state). Cells that never changed off the setblock do not count.
fn grown(obs: &Obs) -> (usize, Option<u32>) {
    let first_at =
        |pos: (i32, i32, i32)| obs.writes.iter().find(|(p, _)| *p == pos).map(|(_, s)| *s);
    let somes: Vec<u32> = (PATCH_X.clone())
        .flat_map(|x| (PATCH_Z.clone()).map(move |z| (x, z)))
        .filter_map(|(x, z)| {
            let pos = (x, GRASS_Y, z);
            match final_at(obs, pos) {
                Some(s) if Some(s) != first_at(pos) => Some(s),
                _ => None,
            }
        })
        .collect();
    let state = somes.first().copied();
    let count = match state {
        Some(s) => somes.iter().filter(|&&x| x == s).count(),
        None => 0,
    };
    (count, state)
}

fn sorted_slots(s: &Obs) -> Vec<(i32, i32, i32)> {
    let mut slots = s.slots.clone();
    slots.sort_unstable();
    slots
}

/// One frame pair's values agree inside the wire tolerances (the LpVec3
/// decode for positions, a few wire units for short deltas, exact ground
/// bits).
fn tail_pair_ok(o: &EFrame, s: &SimFrame) -> bool {
    match (o, s) {
        (EFrame::Motion(_, ov), SimFrame::Motion(sv)) => ov
            .iter()
            .zip(sv.iter())
            .all(|(a, b)| (a - b).abs() <= 1.0e-4),
        (EFrame::Pos(_, og, od), SimFrame::Pos(sd, sg))
        | (EFrame::PosRot(_, og, od), SimFrame::Pos(sd, sg)) => {
            *og == *sg
                && od
                    .iter()
                    .zip(sd.iter())
                    .all(|(a, b)| (*a as i64 - *b).abs() <= 8)
        }
        (EFrame::Sync(_, ox, oy, oz, og), SimFrame::Sync(sp, sg)) => {
            *og == *sg
                && (ox - sp.0).abs() <= 5.0e-3
                && (oy - sp.1).abs() <= 5.0e-3
                && (oz - sp.2).abs() <= 5.0e-3
        }
        _ => false,
    }
}

/// The settle mark of an observed tail: the last sync before the first
/// zero-delta position frame. A resting drop never moves, so its
/// cadence frames carry all-zero deltas; while falling or sliding every
/// frame moves at least one wire unit. The opening position frame also
/// carries zero deltas (it fires at the spawn pass, before the first
/// physics tick), so the rest signature is the first zero-delta frame
/// past that one. Falls back to the window's last sync when no rest
/// frame landed inside it.
fn settle_mark(frames: &[EFrame]) -> Option<usize> {
    let zero = |f: &EFrame| {
        matches!(
            f,
            EFrame::Pos(_, _, [0, 0, 0]) | EFrame::PosRot(_, _, [0, 0, 0])
        )
    };
    let first_rest = frames
        .get(1..)
        .and_then(|rest| rest.iter().position(zero).map(|i| i + 1));
    let scope = first_rest.map_or(frames, |z| &frames[..z]);
    scope.iter().rposition(|f| matches!(f, EFrame::Sync(..)))
}

/// The settle mark of a shadow replay: the same rule on the shadow's
/// frames. The replay runs deep into the rest, so the mark always
/// lands on the settle sync, never on a later one.
fn settle_mark_sim(frames: &[SimFrame]) -> Option<usize> {
    let zero = |f: &SimFrame| matches!(f, SimFrame::Pos([0, 0, 0], _));
    let first_rest = frames
        .get(1..)
        .and_then(|rest| rest.iter().position(zero).map(|i| i + 1));
    let scope = first_rest.map_or(frames, |z| &frames[..z]);
    scope.iter().rposition(|f| matches!(f, SimFrame::Sync(..)))
}

/// How many frames must be dropped from either kind run to make them
/// equal (total deletions; the shared subsequence is the longest
/// common one).
fn kinds_delete_distance(a: &[&str], b: &[&str]) -> usize {
    let mut prev = vec![0usize; b.len() + 1];
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            cur[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1] + 1
            } else {
                prev[j].max(cur[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut cur);
        cur.iter_mut().for_each(|x| *x = 0);
    }
    a.len() + b.len() - 2 * prev[b.len()]
}

/// The observed tail against its shadow, cut at the settle mark: the
/// last sync before the rest cadence's first zero-delta position frame
/// (the landing and slide syncs end there). The fall-and-slide prefix
/// must match frame for frame - kinds, values, ground bits - save for
/// bounded elasticities: either side may carry one extra motion
/// anywhere (the spawn's pop reaches the wire through the LpVec3
/// grid), one run of consecutive motions at the prefix tail may drop
/// from either side, and the prefixes may differ in length past a
/// shared front of at least eight frames - where the decayed speed
/// crosses the rest gate's threshold inside the gate's 4-tick phase
/// decides how many of the last openings fire before the settle, and
/// whether a precision sync lands inside the terminal slide, so the
/// last few slide frames belong to the draw, not the model: one side
/// may settle a few openings before the other, leaving the longer
/// prefix an overhang of plain motions up to its own mark. The
/// shared front carries the fall, the landing sync, and the slide's
/// start, and the terminal overhang may not pass eight frames. Past the mark
/// the replay's fidelity ends: the resting cadence interleaves the
/// gravity accrual with the rest gate at a phase the shadow does not
/// model (the two drops rest differently - one on motion pairs, one on
/// bare position frames - by their id-phased rest gates, the ids
/// pinned equal across servers, at spacings the shadow's periodic rest
/// cannot place), so the resting signal is carried cross-server below,
/// where the two servers agree on the same cadence directly.
fn compare_tail(
    who: &str,
    name: &str,
    obs: &[EFrame],
    sim: &[SimFrame],
    failures: &mut Vec<String>,
) {
    if obs.is_empty() {
        failures.push(format!("{who}: drop {name} sent no movement frames"));
        return;
    }
    let head = |f: &[EFrame]| -> String {
        f.iter()
            .take(24)
            .map(|x| x.kind())
            .collect::<Vec<_>>()
            .join(",")
    };
    let head_sim = |f: &[SimFrame]| -> String {
        f.iter()
            .take(24)
            .map(|x| x.kind())
            .collect::<Vec<_>>()
            .join(",")
    };
    let (Some(obs_mark), Some(sim_mark)) = (settle_mark(obs), settle_mark_sim(sim)) else {
        failures.push(format!(
            "{who}: drop {name} tail never settles (no sync inside the window)"
        ));
        return;
    };
    let pair_ok = |o: &[EFrame], s: &[SimFrame]| -> bool {
        o.len() == s.len()
            && o.iter()
                .zip(s.iter())
                .all(|(a, b)| a.kind() == b.kind() && tail_pair_ok(a, b))
    };
    let drop_a_motion = |f: &[EFrame], at: usize| -> Vec<EFrame> {
        let mut v = f.to_vec();
        v.remove(at);
        v
    };
    let (op, sp) = (&obs[..=obs_mark], &sim[..=sim_mark]);
    let run_len = |f: &[EFrame]| -> usize {
        let mut n = 0;
        while n < 6 && n + 1 < f.len() && matches!(f[f.len() - 2 - n], EFrame::Motion(..)) {
            n += 1;
        }
        n
    };
    let run_len_sim = |f: &[SimFrame]| -> usize {
        let mut n = 0;
        while n < 6 && n + 1 < f.len() && matches!(f[f.len() - 2 - n], SimFrame::Motion(_)) {
            n += 1;
        }
        n
    };
    // The front must carry the physics it is trusted to pin: frames off
    // the ground (the fall), the first grounded frame (the landing),
    // and at least two frames of horizontal travel past it (the slide's
    // decay).
    let front = |f: &[EFrame]| -> bool {
        let land = f.iter().position(|x| match x {
            EFrame::Pos(_, g, _) | EFrame::PosRot(_, g, _) | EFrame::Sync(_, _, _, _, g) => *g,
            _ => false,
        });
        match land {
            Some(i) if i > 0 => {
                f[i + 1..]
                    .iter()
                    .filter(|x| match x {
                        EFrame::Motion(_, v) => v[0] != 0.0 || v[2] != 0.0,
                        EFrame::Pos(_, _, d) | EFrame::PosRot(_, _, d) => d[0] != 0 || d[2] != 0,
                        _ => false,
                    })
                    .count()
                    >= 2
            }
            _ => false,
        }
    };
    if !front(op) {
        failures.push(format!(
            "{who}: drop {name} fall/slide prefix lacks landing evidence [{}]",
            head(obs)
        ));
        return;
    }
    // The terminal precision sync lands at whichever gate opening the
    // draw's lattice loss trips, so one side's sync can sit a few
    // openings past the other's with plain slide motions between. The
    // value-matched front up to that divergence carries the compare;
    // each side's band from there to its own mark may hold motions and
    // nothing else, bounded.
    let band_ok_obs = |f: &[EFrame], from: usize, mark: usize| -> bool {
        mark - from <= 8
            && f.get(from + 1..mark)
                .is_none_or(|seg| seg.iter().all(|x| matches!(x, EFrame::Motion(..))))
    };
    let band_ok_sim = |f: &[SimFrame], from: usize, mark: usize| -> bool {
        mark - from <= 8
            && f.get(from + 1..mark)
                .is_none_or(|seg| seg.iter().all(|x| matches!(x, SimFrame::Motion(_))))
    };
    let first_mismatch = (0..op.len().min(sp.len()))
        .find(|&i| !(op[i].kind() == sp[i].kind() && tail_pair_ok(&op[i], &sp[i])));
    let shared = op.len().min(sp.len());
    let prefix_ok = pair_ok(op, sp)
        || (shared >= 8
            && op.len().abs_diff(sp.len()) <= 6
            && pair_ok(&op[..shared], &sp[..shared]))
        || (0..op.len())
            .any(|i| matches!(op[i], EFrame::Motion(..)) && pair_ok(&drop_a_motion(op, i), sp))
        || (0..sp.len()).any(|i| {
            matches!(sim[i], SimFrame::Motion(_)) && {
                let mut cut = sim.to_vec();
                cut.remove(i);
                pair_ok(op, &cut[..sp.len() - 1])
            }
        })
        || (1..=run_len(op)).any(|k| {
            let mut v = op.to_vec();
            v.drain(v.len() - 1 - k..v.len() - 1);
            pair_ok(&v, sp)
        })
        || (1..=run_len_sim(sp)).any(|k| {
            let mut v = sp.to_vec();
            v.drain(v.len() - 1 - k..v.len() - 1);
            pair_ok(op, &v)
        })
        || first_mismatch.is_some_and(|d| {
            obs_mark > d
                && sim_mark > d
                && matches!(
                    (&op[d], &sp[d]),
                    (EFrame::Sync(..), SimFrame::Motion(_))
                        | (EFrame::Motion(..), SimFrame::Sync(..))
                )
                && band_ok_obs(op, d, obs_mark)
                && band_ok_sim(sp, d, sim_mark)
        })
        || (first_mismatch.is_none() && {
            // One side settles first: past the fully matched front the
            // longer prefix runs plain motions to its own mark.
            let short = op.len().min(sp.len());
            let overhang = op.len().max(sp.len()) - short;
            overhang > 0
                && overhang <= 8
                && if op.len() > sp.len() {
                    op.get(short..obs_mark)
                        .is_none_or(|seg| seg.iter().all(|x| matches!(x, EFrame::Motion(..))))
                } else {
                    sp.get(short..sim_mark)
                        .is_none_or(|seg| seg.iter().all(|x| matches!(x, SimFrame::Motion(_))))
                }
        });
    if !prefix_ok {
        failures.push(format!(
            "{who}: drop {name} fall/slide prefix disagrees with the shadow (observed mark {obs_mark} [{}], shadow mark {sim_mark} [{}])",
            head(obs),
            head_sim(sim)
        ));
    }
}

/// The differential survival test.
pub fn parity_survival() -> Result<bool> {
    let pin = load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let root = doppel_protocol::find_repo_root()?;
    let blobs_dir = root.join("target").join("vanilla").join("blobs-survival");
    let pristine_world = root
        .join("target")
        .join("vanilla")
        .join("pristine-world-survival");
    for dir in [&blobs_dir, &pristine_world] {
        if dir.exists() {
            std::fs::remove_dir_all(dir)?;
        }
    }
    let protocol = pin.protocol.unwrap_or(0);

    capture_clean_blobs(&pin, &jar, &blobs_dir, &pristine_world)?;

    // Vanilla reference sessions; the deadline ends the idle-fed
    // captures, and with them the random tick window. The peaceful boot
    // keeps monsters out of the scenario window entirely. The budget
    // covers a slow host: the walker's away leg needs the reference's
    // view work to run to completion, and its passes scale with wall
    // time, not the session's own pacing.
    let vport = vanilla_port();
    let server = vanilla::boot_peaceful(&pin, &jar, vport)?;
    let vworker = std::thread::spawn(move || run_sessions(vport, protocol));
    let vdeadline = std::time::Instant::now() + Duration::from_secs(150);
    while !vworker.is_finished() && std::time::Instant::now() < vdeadline {
        std::thread::sleep(Duration::from_millis(500));
    }
    drop(server);
    let (v_digger, v_walker, v_witness) = vworker
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
    let worker = std::thread::spawn(move || run_sessions(dport, protocol));
    let deadline = std::time::Instant::now() + Duration::from_secs(150);
    while !worker.is_finished() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(500));
    }
    let _ = child.kill();
    let _ = child.wait();
    let (d_digger, d_walker, d_witness) = worker
        .join()
        .map_err(|_| anyhow::anyhow!("doppel session thread panicked"))??;

    // Every stream is observation from its first frame: the entity and
    // inventory decodes key on the drop cells and ids (join traffic never
    // matches), and the grass writes cover the whole stream anyway.
    let v = analyze(&v_digger, None);
    let d = analyze(&d_digger, None);
    let v_walk = analyze(&v_walker, None);
    let d_walk = analyze(&d_walker, None);
    let v_all = analyze(&v_digger, None);
    let d_all = analyze(&d_digger, None);
    let v_wit = analyze(&v_witness, None);
    let d_wit = analyze(&d_witness, None);

    let histogram = |pkts: &[bot::CapturedPacket]| {
        let mut hist: std::collections::BTreeMap<i32, usize> = Default::default();
        for p in pkts.iter().filter(|p| p.id >= 0) {
            *hist.entry(p.id).or_default() += 1;
        }
        let mut rows: Vec<_> = hist.into_iter().collect();
        rows.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        rows.iter()
            .map(|(id, n)| format!("{id:#04x}:{n}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let end_note = |pkts: &[bot::CapturedPacket]| -> String {
        pkts.last()
            .filter(|p| p.id < 0)
            .and_then(|p| p.note.clone())
            .unwrap_or_default()
    };
    for (who, digger, walker, s, walk, all, wit) in [
        ("vanilla", &v_digger, &v_walker, &v, &v_walk, &v_all, &v_wit),
        ("doppel", &d_digger, &d_walker, &d, &d_walk, &d_all, &d_wit),
    ] {
        println!(
            "[oracle] {who}: digger {} frames, ids: {}",
            digger.len(),
            histogram(digger)
        );
        println!(
            "[oracle] {who}: walker {} frames, ids: {}",
            walker.len(),
            histogram(walker)
        );
        for (name, note) in [("digger", end_note(digger)), ("walker", end_note(walker))] {
            if !note.is_empty() {
                println!("[oracle] {who} {name} {note}");
            }
        }
        for (name, cell) in [("A", TORCH), ("B", TORCH_B)] {
            for (id, typ, x, y, z, movement) in s.adds_in(cell) {
                println!(
                    "[oracle] {who} drop {name} add: id={id} type={typ} at ({x:.3},{y:.3},{z:.3}) movement=({:.4},{:.4},{:.4})",
                    movement[0], movement[1], movement[2]
                );
            }
            for (id, typ, x, y, z, movement) in wit.adds_in(cell) {
                let _ = (id, typ, movement);
                println!(
                    "[oracle] {who} witness drop {name} add: id={id} at ({x:.3},{y:.3},{z:.3})"
                );
            }
            if let Some(id) = s.adds_in(cell).first().map(|a| a.0) {
                for (wid, typ, x, y, z, movement) in
                    walk.adds.iter().filter(|a| a.0 == id && a.1 == 72)
                {
                    println!(
                        "[oracle] {who} walker drop {name} add: id={wid} type={typ} at ({x:.3},{y:.3},{z:.3}) movement=({:.4},{:.4},{:.4})",
                        movement[0], movement[1], movement[2]
                    );
                }
            }
        }
        let mut types: std::collections::BTreeMap<i32, usize> = Default::default();
        for (_, typ, ..) in &s.adds {
            *types.entry(*typ).or_default() += 1;
        }
        let types = types
            .into_iter()
            .map(|(t, n)| format!("{t}:{n}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("[oracle] {who} post-volley add types: {types}");
        for (id, count, item) in &s.stacks {
            println!("[oracle] {who} drop stack: id={id} count={count} item={item}");
        }
        for (id, x, y, z, ground) in &s.syncs {
            println!("[oracle] {who} sync: id={id} at ({x:.3},{y:.3},{z:.3}) ground={ground}");
        }
        for (item, player, amount) in &s.takes {
            println!("[oracle] {who} take: item={item} player={player} amount={amount}");
        }
        for (count, first) in &s.removes {
            println!("[oracle] {who} remove: count={count} first={first}");
        }
        for (item, player, amount) in &walk.takes {
            println!("[oracle] {who} walker take: item={item} player={player} amount={amount}");
        }
        for (count, first) in &walk.removes {
            println!("[oracle] {who} walker remove: count={count} first={first}");
        }
        for (slot, item, count) in &walk.slots {
            println!("[oracle] {who} walker set_slot: slot={slot} item={item} count={count}");
        }
        for (name, id) in [
            ("A", s.adds_in(TORCH).first().map(|a| a.0)),
            ("B", s.adds_in(TORCH_B).first().map(|a| a.0)),
        ] {
            if let Some(id) = id {
                let tail = s.tail_frames(id);
                let head: Vec<&str> = tail.iter().take(16).map(|f| f.kind()).collect();
                println!(
                    "[oracle] {who} drop {name} movement frames: {} [{}]",
                    tail.len(),
                    head.join(",")
                );
                let tail = walk.tail_frames(id);
                let head: Vec<&str> = tail.iter().take(16).map(|f| f.kind()).collect();
                println!(
                    "[oracle] {who} drop {name} walker movement frames: {} [{}]",
                    tail.len(),
                    head.join(",")
                );
            }
        }
        let (cap_n, cap_s) = decayed(all);
        let (grow_n, grow_s) = grown(all);
        println!(
            "[oracle] {who} grass: decayed {cap_n}/16 -> {cap_s:?}, grown {grow_n}/9 -> {grow_s:?}"
        );
        for (item, player, amount) in &wit.takes {
            println!("[oracle] {who} witness take: item={item} player={player} amount={amount}");
        }
        for (slot, item, count) in &wit.slots {
            println!("[oracle] {who} witness set_slot: slot={slot} item={item} count={count}");
        }
        let (wcap, wstate) = decayed(wit);
        let (wgrow, _) = grown(wit);
        println!("[oracle] {who} witness grass: decayed {wcap}/16, grown {wgrow}/9 ({wstate:?})");
    }

    let mut failures = Vec::new();

    // One item entity spawns per torch cell on each side, typed as the
    // item entity (72 in the entity-type registry). The digger and the
    // witness hold each spawn pairing once (their adds carry the spawn
    // position, inside the cell window); the walker's pairings are
    // id-keyed below (its adds carry the settled position).
    for (who, s, wit) in [("vanilla", &v, &v_wit), ("doppel", &d, &d_wit)] {
        for (name, cell) in [("A", TORCH), ("B", TORCH_B)] {
            if s.adds_in(cell).len() != 1 {
                failures.push(format!(
                    "{who}: {} add_entity frames in drop {name}'s cell, want 1 (the spawn pairing)",
                    s.adds_in(cell).len()
                ));
            }
            if wit.adds_in(cell).len() != 1 {
                failures.push(format!(
                    "{who}: witness saw {} item add_entity frames for drop {name}, want 1",
                    wit.adds_in(cell).len()
                ));
            }
        }
    }

    // The spawn offset formula: the level draws each axis inside the
    // cell (x/z within 0.25 of center, y within 0.25 of mid-cell minus
    // the half-height), the pop draws the horizontal velocity inside
    // 0.1, and the vertical pop is fixed.
    for (who, s) in [("vanilla", &v), ("doppel", &d)] {
        for (name, cell) in [("A", TORCH), ("B", TORCH_B)] {
            for (id, typ, x, y, z, movement) in s.adds_in(cell) {
                let _ = id;
                if typ != 72 {
                    failures.push(format!(
                        "{who}: drop {name} entity type {typ}, want 72 (minecraft:item)"
                    ));
                }
                let (lx, lz) = (
                    (x - cell.0 as f64 - 0.5).abs(),
                    (z - cell.2 as f64 - 0.5).abs(),
                );
                if lx > 0.25 + 1.0e-9 || lz > 0.25 + 1.0e-9 {
                    failures.push(format!(
                        "{who}: drop {name} spawns off the cell ({lx:.4},{lz:.4} > 0.25)"
                    ));
                }
                let (lo, hi) = (
                    PLANT_Y as f64 + 0.5 - 0.25 - 0.125,
                    PLANT_Y as f64 + 0.5 + 0.25 - 0.125,
                );
                if !(lo - 1.0e-9..=hi + 1.0e-9).contains(&y) {
                    failures.push(format!(
                        "{who}: drop {name} spawns at y={y:.4}, want inside [{lo:.3},{hi:.3}]"
                    ));
                }
                if movement[1] < 0.2 - 5.0e-5 || movement[1] > 0.2 + 5.0e-5 {
                    failures.push(format!(
                        "{who}: drop {name} pop vy={:.5}, want 0.2",
                        movement[1]
                    ));
                }
                for (axis, v) in [("x", movement[0]), ("z", movement[2])] {
                    if v.abs() > 0.1 + LP_STEP / 2.0 {
                        failures.push(format!(
                            "{who}: drop {name} pop v{axis}={v:.5} leaves the 0.1 draw"
                        ));
                    }
                }
            }
        }
    }

    // The two servers draw their own randoms: the spawn positions agree
    // only inside the formula's span (0.5 per axis).
    for (name, cell) in [("A", TORCH), ("B", TORCH_B)] {
        let vs = v.adds_in(cell);
        let ds = d.adds_in(cell);
        if let (Some(vv), Some(dd)) = (vs.first(), ds.first()) {
            for (axis, a, b) in [("x", vv.2, dd.2), ("y", vv.3, dd.3), ("z", vv.4, dd.4)] {
                if (a - b).abs() > 0.5 + 1.0e-9 {
                    failures.push(format!(
                        "drop {name} spawn {axis} drifts across servers: {a:.4} vs {b:.4}"
                    ));
                }
            }
            if vv.0 != dd.0 {
                failures.push(format!(
                    "drop {name} entity id: vanilla {}, doppel {} (the cadences phase on it)",
                    vv.0, dd.0
                ));
            }
        }
    }
    for (name, cell) in [("A", TORCH), ("B", TORCH_B)] {
        let vw = v_wit.adds_in(cell).first().map(|a| a.0);
        let dw = d_wit.adds_in(cell).first().map(|a| a.0);
        if vw.is_some() && dw.is_some() && vw != dw {
            failures.push(format!(
                "drop {name} witness entity id: vanilla {vw:?} vs doppel {dw:?}"
            ));
        }
    }

    // The pairing's entity data carries the same one-item stack per drop.
    let stack_of = |s: &Obs, id: i32| s.stacks.iter().copied().find(|(eid, _, _)| *eid == id);
    for (name, cell) in [("A", TORCH), ("B", TORCH_B)] {
        let v_stack = v
            .adds_in(cell)
            .first()
            .and_then(|(id, ..)| stack_of(&v, *id));
        let d_stack = d
            .adds_in(cell)
            .first()
            .and_then(|(id, ..)| stack_of(&d, *id));
        match (v_stack, d_stack) {
            (Some((_, vc, vi)), Some((_, dc, di))) => {
                if vc != 1 || dc != 1 {
                    failures.push(format!(
                        "drop {name} stack counts: vanilla {vc}, doppel {dc}, want 1"
                    ));
                }
                if vi != di {
                    failures.push(format!(
                        "drop {name} stack items differ: vanilla {vi} vs doppel {di}"
                    ));
                }
            }
            (a, b) => failures.push(format!(
                "drop {name} stack entity data missing: vanilla {a:?} vs doppel {b:?}"
            )),
        }
    }

    // The walker's lifecycle, keyed by the ids the digger's spawn adds
    // pinned (the walker's own adds carry each drop's settled position,
    // outside the spawn cell): both drops pair at the join, unpair at
    // the away leg, and re-pair inside the closing legs - this server
    // at the hop's pairing pass (deterministic), the reference when its
    // away-view chunk work completes, a wall-clock-bound batch that a
    // slow host can leave unfinished at the window's end. So this
    // server must always run the full cycle; the reference either runs
    // it (checked in full) or stays paired throughout (checked as the
    // degenerate shape: one add, drop A's discard after its take, drop B
    // untouched). Drop A's take and discard land after its re-add.
    for (who, dig, walk) in [("vanilla", &v, &v_walk), ("doppel", &d, &d_walk)] {
        for (name, cell, taken) in [("A", TORCH, true), ("B", TORCH_B, false)] {
            let Some(id) = dig.adds_in(cell).first().map(|a| a.0) else {
                continue;
            };
            let adds = walk.item_add_frames(id);
            let removes: Vec<usize> = walk
                .eframes
                .iter()
                .enumerate()
                .filter(|(_, f)| matches!(f, EFrame::Remove(e) if *e == id))
                .map(|(i, _)| i)
                .collect();
            let takes: Vec<usize> = walk
                .eframes
                .iter()
                .enumerate()
                .filter(|(_, f)| matches!(f, EFrame::Take(e, _, _) if *e == id))
                .map(|(i, _)| i)
                .collect();
            if who == "vanilla" && adds.len() == 1 {
                // The reference never unpaired: the away work stayed
                // mid-flight, so the join pairing holds to the end.
                if taken {
                    if removes.len() != 1
                        || removes[0] < adds[0]
                        || takes.first().is_none_or(|&t| t > removes[0])
                    {
                        failures.push(format!(
                            "vanilla: drop {name} stayed paired but its removal frames are {removes:?} against the add {adds:?} and takes {takes:?}"
                        ));
                    }
                } else if !removes.is_empty() {
                    failures.push(format!(
                        "vanilla: drop {name} stayed paired but removal frames {removes:?} appeared"
                    ));
                }
                continue;
            }
            if adds.len() != 2 {
                failures.push(format!(
                    "{who}: drop {name} walker add frames {}, want 2 (join pair + re-pair)",
                    adds.len()
                ));
                continue;
            }
            if taken {
                if removes.len() != 2 {
                    failures.push(format!(
                        "{who}: drop {name} walker removal frames {}, want 2 (unpair + discard)",
                        removes.len()
                    ));
                } else if !(adds[0] < removes[0] && removes[0] < adds[1] && adds[1] < removes[1]) {
                    failures.push(format!(
                        "{who}: drop {name} walker re-pair ordering: adds {adds:?} removes {removes:?}"
                    ));
                }
                if removes.len() == 2
                    && (takes.len() != 1 || takes[0] < adds[1] || takes[0] > removes[1])
                {
                    failures.push(format!(
                        "{who}: drop {name} take frames at {takes:?}, want one between the re-pair and the discard"
                    ));
                }
            } else if removes.len() != 1 || removes[0] < adds[0] || removes[0] > adds[1] {
                failures.push(format!(
                    "{who}: drop {name} walker removal frames {removes:?} against adds {adds:?}, want the unpair alone"
                ));
            }
            for (count, rid) in &walk.removes {
                if *rid == id && *count != 1 {
                    failures.push(format!(
                        "{who}: drop {name} removal carried {} entities, want it alone",
                        count
                    ));
                }
            }
        }
    }

    // The take: one on the walker (the first drop, full stack - the
    // closing teleport re-paired it before the vacuum, so its stream
    // carries the frame), the same triple on both servers, and the same
    // broadcast on the standing streams.
    let v_takes = v_walk.takes.clone();
    let d_takes = d_walk.takes.clone();
    let a_id = |s: &Obs| s.adds_in(TORCH).first().map(|a| a.0);
    if v_takes != d_takes {
        failures.push(format!(
            "take frames differ: vanilla {v_takes:?} vs doppel {d_takes:?}"
        ));
    } else if v_takes.len() != 1 || v_takes[0].2 != 1 {
        failures.push(format!(
            "walker takes {v_takes:?}, want one full-stack take"
        ));
    } else if let Some(id) = a_id(&v) {
        if v_takes[0].0 != id {
            failures.push(format!(
                "the taken entity {} is not drop A ({id})",
                v_takes[0].0
            ));
        }
    }
    for (who, s) in [("vanilla", &v), ("doppel", &d)] {
        let amounts: Vec<i32> = s.takes.iter().map(|(_, _, a)| *a).collect();
        if amounts != vec![1] {
            failures.push(format!("{who}: digger take amounts {amounts:?}, want [1]"));
        }
    }

    // The movement frames: each standing stream's tail after the spawn
    // add must match the shadow replayed from that stream's own observed
    // spawn. Drop A's story ends with the walker's take and the discard
    // (stripped before the prefix compare); drop B rests (the stream may
    // cut the tail early).
    for (who, dig, wit) in [("vanilla", &v, &v_wit), ("doppel", &d, &d_wit)] {
        for (name, cell, taken) in [("A", TORCH, true), ("B", TORCH_B, false)] {
            for (stream, s) in [("digger", dig), ("witness", wit)] {
                let Some(spawn) = s.adds_in(cell).first().copied() else {
                    continue;
                };
                let sim = simulate(&spawn);
                if sim.len() < 3 {
                    failures.push(format!(
                        "{who}: drop {name} shadow produced {} frames",
                        sim.len()
                    ));
                    continue;
                }
                let mut tail = s.tail_frames(spawn.0);
                if taken {
                    match (tail.pop(), tail.pop()) {
                        (Some(EFrame::Remove(_)), Some(EFrame::Take(e, _, _))) if e == spawn.0 => {}
                        _ => failures.push(format!(
                            "{who}: drop {name}/{stream} tail does not end with its take and removal"
                        )),
                    }
                }
                compare_tail(who, &format!("{name}/{stream}"), &tail, &sim, &mut failures);
                if !taken && tail.len() < 15 {
                    failures.push(format!(
                        "{who}: drop {name}/{stream} sent {} resting movement frames, want >= 15",
                        tail.len()
                    ));
                }
            }
        }
    }

    // The walker's own tails: the re-paired cadence (and drop A's take)
    // agree across servers frame for frame, keyed by the digger-pinned
    // ids. Drop A's tail is trimmed to its take: the frames between the
    // re-add and the take are cadence noise around the kill tick.
    let trim_at_take = |t: &[EFrame], id: i32| -> Vec<EFrame> {
        match t
            .iter()
            .position(|f| matches!(f, EFrame::Take(e, _, _) if *e == id))
        {
            Some(i) => t[i..].to_vec(),
            None => t.to_vec(),
        }
    };
    for (name, cell, taken) in [("A", TORCH, true), ("B", TORCH_B, false)] {
        let (Some(iv), Some(idd)) = (
            v.adds_in(cell).first().map(|a| a.0),
            d.adds_in(cell).first().map(|a| a.0),
        ) else {
            failures.push(format!(
                "drop {name}: spawn add missing, cannot compare the walker tail"
            ));
            continue;
        };
        // The tails anchor at each stream's last add; with the reference
        // still paired its anchor is the join, not the re-pair, so the
        // value compare runs only when both servers re-paired.
        if v_walk.item_add_frames(iv).len() != 2 || d_walk.item_add_frames(idd).len() != 2 {
            continue;
        }
        let mut vt = v_walk.tail_frames(iv);
        let mut dt = d_walk.tail_frames(idd);
        if taken {
            vt = trim_at_take(&vt, iv);
            dt = trim_at_take(&dt, idd);
        }
        // The resting cadence repeats at the id-phased period, so the
        // first WALKER_TAIL_MAX frames past the re-add carry the whole
        // shape; past them the two capture lengths drift with the
        // servers' tick rates, not with parity.
        vt.truncate(WALKER_TAIL_MAX);
        dt.truncate(WALKER_TAIL_MAX);
        compare_obs_pair(name, &vt, &dt, &mut failures);
    }

    // Cross-server on the standing witnesses, split at each tail's own
    // settle mark. The two servers draw their own spawns (0.25 per axis,
    // the pop 0.1), so the slide lengths legitimately differ: one side's
    // draw can open one more rest gate before the threshold and trip one
    // more precision sync on the 4096-grid, so the prefixes must agree
    // in kinds inside a deletion budget of three frames (the spread the
    // six observed tails show). The resting cadences - the same periodic
    // pattern at a draw-dependent lattice phase, cut at the next sync -
    // must agree in kinds at some rotation within one period. Values
    // stay per-server (their draws differ); the walker tails compare
    // values exactly (one shared entity, re-paired past its draw).
    for (name, cell, taken) in [("A", TORCH, true), ("B", TORCH_B, false)] {
        // Drop A's tail ends in the take broadcast and the discard; the
        // two captures cut at different motion counts around them, so
        // the compare stops at the take.
        let tail = |s: &Obs| -> Vec<EFrame> {
            let mut t: Vec<EFrame> = s
                .adds_in(cell)
                .first()
                .map(|a| s.tail_frames(a.0))
                .unwrap_or_default();
            if taken {
                if let Some(i) = t.iter().position(|f| matches!(f, EFrame::Take(..))) {
                    t.truncate(i);
                }
            }
            t
        };
        let (vt, dt) = (tail(&v_wit), tail(&d_wit));
        let kinds = |t: &[EFrame]| -> Vec<&str> { t.iter().map(|f| f.kind()).collect::<Vec<_>>() };
        let (vk, dk) = (kinds(&vt), kinds(&dt));
        let head = |k: &[&str]| k.iter().take(12).copied().collect::<Vec<_>>().join(",");
        let (Some(vm), Some(dm)) = (settle_mark(&vt), settle_mark(&dt)) else {
            failures.push(format!(
                "drop {name} witness tail never settles: vanilla [{}] vs doppel [{}]",
                head(&vk),
                head(&dk)
            ));
            continue;
        };
        // The prefixes may also differ only past a shared front of at
        // least eight frames, within six frames of each other: the
        // terminal slide's length - where the decayed speed crosses the
        // rest gate's threshold inside the gate's 4-tick phase, and
        // whether a precision sync lands inside it - belongs to each
        // server's own draw. The same draw moves each side's terminal
        // precision sync a few gate openings, so marks a bounded band
        // apart with motions between them also pass, cut at the earlier
        // mark. Phase tolerance (owner-approved 2026-10-05): an exact
        // index-for-index tail order is runner-phase coupled - the same
        // tree failed CI twice on terminal kind order while passing
        // locally, and the reference's own capture has disagreed with
        // its shadow model on other days - so the settled prefix also
        // passes when the kind MIX matches over the compared prefix and
        // the deterministic establishment front agrees: same kinds, same
        // counts, same opening order, timing-free tail.
        let (vp, dp) = (&vk[..=vm], &dk[..=dm]);
        let shared = vp.len().min(dp.len());
        let early = vm.min(dm);
        let band_ok = |k: &[&str], m: usize| -> bool {
            m - early <= 7
                && k.get(early + 1..m)
                    .is_none_or(|seg| seg.iter().all(|x| *x == "motion"))
        };
        let kind_counts = |k: &[&str]| -> std::collections::BTreeMap<String, usize> {
            let mut m: std::collections::BTreeMap<String, usize> = Default::default();
            for x in k {
                *m.entry((*x).to_string()).or_default() += 1;
            }
            m
        };
        if kinds_delete_distance(vp, dp) > 3
            && !(shared >= 8 && vp.len().abs_diff(dp.len()) <= 6 && vp[..shared] == dp[..shared])
            && !(vm != dm && vk[..early] == dk[..early] && band_ok(&vk, vm) && band_ok(&dk, dm))
            && !(kind_counts(vp) == kind_counts(dp) && vk[..early] == dk[..early])
        {
            failures.push(format!(
                "drop {name} witness fall/slide kinds differ: vanilla [{}] vs doppel [{}]",
                head(&vk),
                head(&dk)
            ));
            continue;
        }
        let rest_kinds = |t: &[EFrame], m: usize| -> Vec<&str> {
            let r = &t[m + 1..];
            let end = r
                .iter()
                .position(|f| matches!(f, EFrame::Sync(..)))
                .unwrap_or(r.len());
            r[..end].iter().map(|f| f.kind()).collect()
        };
        let (vr, dr) = (rest_kinds(&vt, vm), rest_kinds(&dt, dm));
        let mut r_ok = false;
        for shift in 0..8 {
            let check = |a: &[&str], b: &[&str]| -> bool {
                !a.is_empty()
                    && a.len() >= 12
                    && a.len() <= b.len()
                    && a.iter().zip(b.iter()).all(|(x, y)| x == y)
            };
            if (vr.len() > shift && check(&vr[shift..], &dr))
                || (dr.len() > shift && check(&dr[shift..], &vr))
            {
                r_ok = true;
                break;
            }
        }
        if !r_ok {
            failures.push(format!(
                "drop {name} witness resting kinds disagree at every rotation: vanilla [{}] vs doppel [{}]",
                vr.iter().take(12).copied().collect::<Vec<_>>().join(","),
                dr.iter().take(12).copied().collect::<Vec<_>>().join(",")
            ));
        }
    }

    // The stack lands in the walker's inventory: identical set_slot
    // traffic carrying the drop's stack.
    let v_slots = sorted_slots(&v_walk);
    let d_slots = sorted_slots(&d_walk);
    if v_slots != d_slots {
        failures.push(format!(
            "walker set_slot traffic differs: vanilla {v_slots:?} vs doppel {d_slots:?}"
        ));
    } else if v_slots.is_empty() {
        failures.push("no set_slot frames carry the picked-up stack".into());
    } else if let Some((_, _, vi)) =
        stack_of(&v, v.adds_in(TORCH).first().map(|a| a.0).unwrap_or(-1))
    {
        for (slot, item, count) in &v_slots {
            if *item != vi || *count != 1 {
                failures.push(format!(
                    "set_slot ({slot},{item},{count}) does not carry the drop's stack"
                ));
            }
        }
    }

    // Grass: the caps decay to one shared dirt state, the patch regrows
    // one shared grass state, and the two servers agree on both ids.
    for (who, all) in [("vanilla", &v_all), ("doppel", &d_all)] {
        let (n, _) = decayed(all);
        if n < 12 {
            failures.push(format!("{who}: {n}/16 capped cells decayed, want >= 12"));
        }
        let (n, _) = grown(all);
        if n < 1 {
            failures.push(format!("{who}: 0/9 patch cells regrew, want >= 1"));
        }
        match final_at(all, TORCH) {
            Some(0) => {}
            s => failures.push(format!("{who}: torch cell final state {s:?}, want air (0)")),
        }
        match final_at(all, TORCH_B) {
            Some(0) => {}
            s => failures.push(format!(
                "{who}: torch B cell final state {s:?}, want air (0)"
            )),
        }
    }
    let (_, v_dirt) = decayed(&v_all);
    let (_, d_dirt) = decayed(&d_all);
    if v_dirt.is_some() && v_dirt != d_dirt {
        failures.push(format!(
            "decay states differ: vanilla {v_dirt:?} vs doppel {d_dirt:?}"
        ));
    }
    let (_, v_grass) = grown(&v_all);
    let (_, d_grass) = grown(&d_all);
    if v_grass.is_some() && v_grass != d_grass {
        failures.push(format!(
            "regrown states differ: vanilla {v_grass:?} vs doppel {d_grass:?}"
        ));
    }
    if v_dirt.is_some() && v_grass.is_some() && v_dirt == v_grass {
        failures.push(format!(
            "decay and regrowth landed on the same state {v_dirt:?}"
        ));
    }

    // The standing witness sees the same broadcast lifecycle.
    for (who, wit) in [("vanilla", &v_wit), ("doppel", &d_wit)] {
        let amounts: Vec<i32> = wit.takes.iter().map(|(_, _, a)| *a).collect();
        if amounts != vec![1] {
            failures.push(format!("{who}: witness take amounts {amounts:?}, want [1]"));
        }
        let (n, _) = decayed(wit);
        if n < 12 {
            failures.push(format!("{who}: witness saw {n}/16 capped cells decay"));
        }
        let (n, _) = grown(wit);
        if n < 1 {
            failures.push(format!("{who}: witness saw {n}/9 patch cells regrow"));
        }
    }

    if failures.is_empty() {
        println!("PASS: survival parity");
        Ok(true)
    } else {
        println!("FAIL: {} survival difference(s):", failures.len());
        for f in failures.iter().take(16) {
            println!("  {f}");
        }
        Ok(false)
    }
}
