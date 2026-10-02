//! The differential survival gate: the survival loop's full round trip.
//! A scripted session covers grass, digs an insta-break torch, and walks
//! onto the drop; the random tick pass runs amplified
//! (gamerule random_tick_speed) so decay and spread land inside the
//! capture window. Both sides must
//! agree on the drop's pairing, landing, take animation, removal, the
//! inventory sync, and the grass transitions.

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

// The flat world's surface: grass at y=-61, plants at y=-60 (the spawn
// teleport pins the standing height at -60.0).
const GRASS_Y: i32 = -61;
const PLANT_Y: i32 = -60;
/// The cell the dug torch occupies (an insta-break block).
const TORCH: (i32, i32, i32) = (2, PLANT_Y, 8);
/// The stone caps (PLANT_Y cells) whose covered grass decays.
const CAP_X: std::ops::RangeInclusive<i32> = 2..=5;
const CAP_Z: std::ops::RangeInclusive<i32> = 2..=5;
/// The bare-dirt patch (GRASS_Y) the surrounding grass reclaims.
const PATCH_X: std::ops::RangeInclusive<i32> = 8..=10;
const PATCH_Z: std::ops::RangeInclusive<i32> = 2..=4;
/// The random tick rate the window runs at (the default 3 is too slow
/// for a bounded capture).
const TICK_SPEED: usize = 300;

/// player_action body: action VarInt, packed pos, direction, sequence.
fn build_player_action(action: i32, pos: (i32, i32, i32), sequence: i32) -> Vec<u8> {
    let mut b = Vec::new();
    doppel_protocol::write_varint(&mut b, action);
    b.extend_from_slice(&bot::pack_block_pos(pos.0, pos.1, pos.2).to_be_bytes());
    doppel_protocol::write_varint(&mut b, 1); // direction: up
    doppel_protocol::write_varint(&mut b, sequence);
    b
}

/// move_player_pos body: x y z f64 + flags u8 (all absolute).
fn build_move_pos(x: f64, y: f64, z: f64) -> Vec<u8> {
    let mut b = Vec::with_capacity(25);
    b.extend_from_slice(&x.to_be_bytes());
    b.extend_from_slice(&y.to_be_bytes());
    b.extend_from_slice(&z.to_be_bytes());
    b.push(0);
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

/// One stream's reduced observation of the session. `marker` splits the
/// scripted volley (command feedback) from the interaction traffic; the
/// entity and inventory decodes count post-volley frames only, the block
/// writes cover the whole stream.
#[derive(Default)]
struct Obs {
    /// Every post-volley add_entity: (entity id, type, x, y, z).
    adds: Vec<(i32, i32, f64, f64, f64)>,
    /// add_entity frames inside the torch cell: (id, type, x, y, z).
    item_adds: Vec<(i32, i32, f64, f64, f64)>,
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
                    obs.adds.push((id, typ, x, y, z));
                    let in_cell = (x - TORCH.0 as f64 - 0.5).abs() <= 0.35
                        && (z - TORCH.2 as f64 - 0.5).abs() <= 0.35
                        && (PLANT_Y as f64 - 0.1..=PLANT_Y as f64 + 0.8).contains(&y);
                    if in_cell {
                        obs.item_adds.push((id, typ, x, y, z));
                    }
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
                }
            }
            // remove_entities: count, then ids.
            0x4e => {
                if let Some(count) = rd_varint(&raw, &mut o) {
                    let first = rd_varint(&raw, &mut o).unwrap_or(-1);
                    obs.removes.push((count, first));
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

/// The scripted session pair against one server.
fn run_sessions(
    port: u16,
    protocol: i32,
) -> Result<(Vec<bot::CapturedPacket>, Vec<bot::CapturedPacket>)> {
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
    // setup cells for the whole volley.
    commands.push(format!("tp @s 1.5 {} 8.5", PLANT_Y));
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
    commands.push(format!("gamerule random_tick_speed {TICK_SPEED}"));
    let raw: Vec<(i32, Vec<u8>)> = vec![
        (0x2c, Vec::new()), // player_loaded: the reference gates input on it
        // Insta-break the torch: the drop spawns with the dig.
        (0x29, build_player_action(0, TORCH, 10)),
        // Step onto the drop's cell; the pickup delay expires mid-idle.
        (
            0x1e,
            build_move_pos(TORCH.0 as f64 + 0.5, PLANT_Y as f64, TORCH.2 as f64 + 0.5),
        ),
    ];
    // The witness observes from a fixed post: it joins first, so wherever
    // the variable world spawn lands is where it stands - and a spawn on
    // the drop's cell wins the pickup race ahead of the digger. The post
    // is 12 blocks from the torch (outside pickup range, inside the view
    // distance).
    let witness_commands = vec![format!("tp @s 2.5 {} -3.5", PLANT_Y)];
    let witness_login = capture::login_start_c("Doppelist");
    let witness = std::thread::spawn(move || {
        bot::login_capture(
            "127.0.0.1",
            port,
            protocol,
            &witness_login,
            &bot::CaptureOpts {
                idle_timeout: Some(Duration::from_secs(30)),
                max_packets: Some(24000),
                dump_dir: None,
                commands: &witness_commands,
                walk_chunks: None,
                raw_packets: &[],
            },
        )
    });
    std::thread::sleep(Duration::from_secs(8));
    let login = capture::login_start_c("Doppel");
    let digger = bot::login_capture(
        "127.0.0.1",
        port,
        protocol,
        &login,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(25)),
            max_packets: Some(12000),
            dump_dir: None,
            commands: &commands,
            walk_chunks: None,
            raw_packets: &raw,
        },
    )
    .context("capturing digger session")?;
    let witness = witness
        .join()
        .map_err(|_| anyhow::anyhow!("witness thread panicked"))?
        .context("capturing witness session")?;
    Ok((digger, witness))
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
    // captures, and with them the random tick window.
    let server = vanilla::boot(&pin, &jar, VANILLA_PORT)?;
    let vworker = std::thread::spawn(move || run_sessions(VANILLA_PORT, protocol));
    let vdeadline = std::time::Instant::now() + Duration::from_secs(40);
    while !vworker.is_finished() && std::time::Instant::now() < vdeadline {
        std::thread::sleep(Duration::from_millis(500));
    }
    drop(server);
    let (v_digger, v_witness) = vworker
        .join()
        .map_err(|_| anyhow::anyhow!("vanilla session thread panicked"))??;

    let bin = default_doppel_bin()?;
    let pin_path = doppel_protocol::pin_path()?;
    let mut child = Command::new(&bin)
        .env("DOPPEL_ADDR", "127.0.0.1")
        .env("DOPPEL_PORT", DOPPEL_PORT.to_string())
        .env("DOPPEL_PIN", &pin_path)
        .env("DOPPEL_BLOBS", &blobs_dir)
        .env("DOPPEL_WORLD", &pristine_world)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawning {}", bin.display()))?;
    wait_for_port(DOPPEL_PORT, Duration::from_secs(30))?;
    std::thread::sleep(Duration::from_secs(2));
    let worker = std::thread::spawn(move || run_sessions(DOPPEL_PORT, protocol));
    let deadline = std::time::Instant::now() + Duration::from_secs(40);
    while !worker.is_finished() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(500));
    }
    let _ = child.kill();
    let _ = child.wait();
    let (d_digger, d_witness) = worker
        .join()
        .map_err(|_| anyhow::anyhow!("doppel session thread panicked"))??;

    let marker_of = |pkts: &[bot::CapturedPacket]| pkts.iter().rposition(|p| p.id == 0x7c);
    // Post-volley traffic carries the drop lifecycle and the inventory
    // sync; the full stream carries the grass writes. The witness never
    // sends commands, so its whole stream is observation.
    let v = analyze(&v_digger, marker_of(&v_digger));
    let d = analyze(&d_digger, marker_of(&d_digger));
    let v_all = analyze(&v_digger, Some(usize::MAX));
    let d_all = analyze(&d_digger, Some(usize::MAX));
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
    for (who, digger, s, all, wit) in [
        ("vanilla", &v_digger, &v, &v_all, &v_wit),
        ("doppel", &d_digger, &d, &d_all, &d_wit),
    ] {
        println!(
            "[oracle] {who}: digger {} frames, ids: {}",
            digger.len(),
            histogram(digger)
        );
        for (id, typ, x, y, z) in &s.item_adds {
            println!("[oracle] {who} drop add: id={id} type={typ} at ({x:.3},{y:.3},{z:.3})");
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
        for (slot, item, count) in &s.slots {
            println!("[oracle] {who} set_slot: slot={slot} item={item} count={count}");
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

    // One item entity spawns on each side, inside the torch's cell, typed
    // as the item entity (72 in the entity-type registry).
    for (who, s) in [("vanilla", &v), ("doppel", &d)] {
        if s.item_adds.len() != 1 {
            failures.push(format!(
                "{who}: {} add_entity frames in the torch cell, want exactly 1",
                s.item_adds.len()
            ));
        }
    }
    if v.item_adds.len() == 1 && d.item_adds.len() == 1 {
        let check_pos = |who: &str, x: f64, y: f64, z: f64| -> Vec<String> {
            let mut bad = Vec::new();
            let lx = (x - TORCH.0 as f64 - 0.5).abs();
            let lz = (z - TORCH.2 as f64 - 0.5).abs();
            if lx > 0.3 || lz > 0.3 {
                bad.push(format!("{who}: drop spawns off the cell ({lx:.3},{lz:.3})"));
            }
            if !(PLANT_Y as f64 - 0.1..=PLANT_Y as f64 + 0.7).contains(&y) {
                bad.push(format!(
                    "{who}: drop spawns at y={y:.3}, want inside the torch cell"
                ));
            }
            bad
        };
        let (vx, vy, vz) = (v.item_adds[0].2, v.item_adds[0].3, v.item_adds[0].4);
        let (dx, dy, dz) = (d.item_adds[0].2, d.item_adds[0].3, d.item_adds[0].4);
        failures.extend(check_pos("vanilla", vx, vy, vz));
        failures.extend(check_pos("doppel", dx, dy, dz));
        if (vx - dx).abs() > 0.6 || (vy - dy).abs() > 0.6 || (vz - dz).abs() > 0.6 {
            failures.push(format!(
                "drop spawn drifts across servers: vanilla ({vx:.3},{vy:.3},{vz:.3}) vs doppel ({dx:.3},{dy:.3},{dz:.3})"
            ));
        }
        let (vt, dt) = (v.item_adds[0].1, d.item_adds[0].1);
        if vt != 72 || dt != 72 {
            failures.push(format!(
                "drop entity type: vanilla {vt}, doppel {dt}, want 72 (minecraft:item)"
            ));
        }
    }

    // The pairing's entity data carries the same one-item stack.
    let stack_of = |s: &Obs, id: i32| s.stacks.iter().copied().find(|(eid, _, _)| *eid == id);
    let v_stack = v.item_adds.first().and_then(|(id, ..)| stack_of(&v, *id));
    let d_stack = d.item_adds.first().and_then(|(id, ..)| stack_of(&d, *id));
    match (v_stack, d_stack) {
        (Some((_, vc, vi)), Some((_, dc, di))) => {
            if vc != 1 || dc != 1 {
                failures.push(format!(
                    "drop stack counts: vanilla {vc}, doppel {dc}, want 1"
                ));
            }
            if vi != di {
                failures.push(format!(
                    "drop stack items differ: vanilla {vi} vs doppel {di}"
                ));
            }
        }
        (a, b) => failures.push(format!(
            "drop stack entity data missing: vanilla {a:?} vs doppel {b:?}"
        )),
    }

    // The drop lands and the player vacuums it: any position syncs stay
    // in the torch cell (the 10-tick pickup delay can expire mid-fall,
    // so a grounded sync is not guaranteed), one take with the full
    // count, one removal of that entity.
    for (who, s) in [("vanilla", &v), ("doppel", &d)] {
        let Some((id, ..)) = s.item_adds.first() else {
            continue;
        };
        let mine: Vec<_> = s.syncs.iter().filter(|(eid, ..)| eid == id).collect();
        if let Some(&(_, x, y, z, _)) = mine.last() {
            if (x - TORCH.0 as f64 - 0.5).abs() > 0.4
                || (z - TORCH.2 as f64 - 0.5).abs() > 0.4
                || !(GRASS_Y as f64 - 0.2..=PLANT_Y as f64 + 0.9).contains(y)
            {
                failures.push(format!(
                    "{who}: drop's last sync ({x:.3},{y:.3},{z:.3}) left the torch cell"
                ));
            }
        }
        let takes: Vec<_> = s.takes.iter().filter(|(item, ..)| item == id).collect();
        if takes.len() != 1 || takes[0].2 != 1 {
            failures.push(format!(
                "{who}: take frames {takes:?}, want exactly one with amount 1"
            ));
        }
        let gone: Vec<_> = s.removes.iter().filter(|(_, rid)| rid == id).collect();
        if gone.len() != 1 || gone[0].0 != 1 {
            failures.push(format!(
                "{who}: removal frames for the drop {gone:?}, want one carrying it alone"
            ));
        }
    }

    // The stack lands in the inventory: identical set_slot traffic.
    let v_slots = sorted_slots(&v);
    let d_slots = sorted_slots(&d);
    if v_slots != d_slots {
        failures.push(format!(
            "set_slot traffic differs: vanilla {v_slots:?} vs doppel {d_slots:?}"
        ));
    } else if v_slots.is_empty() {
        failures.push("no set_slot frames carry the picked-up stack".into());
    } else if let Some((_, _, vi)) = v_stack {
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
        if wit.item_adds.len() != 1 {
            failures.push(format!(
                "{who}: witness saw {} item add_entity frames, want 1",
                wit.item_adds.len()
            ));
        }
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
