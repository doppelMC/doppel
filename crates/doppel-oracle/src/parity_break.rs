//! The differential breaking gate: one scripted survival dig session -
//! place a block, right-click a chest, release an obsidian dig below the
//! finish threshold, abort and re-hold a stone dig, drop the stack,
//! swing - run against vanilla and doppel. Two transcripts per side:
//! the digger's own (menus, inventory syncs) and a second client
//! standing at spawn (destruction overlays, chest lid, world writes).
//!
//! The release dig targets obsidian on purpose: a fresh player's delayed
//! destroy multiplies one tick of progress by the dig's per-player start
//! tick, so a hard block keeps that product below every stage and break
//! threshold for the whole session. The dig never completes, which also
//! pins the starvation quirk: while a delayed destroy is pending, the
//! per-tick pass never deepens a held dig.

use anyhow::{Context, Result};
use doppel_protocol::load_pin;
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::{bot, capture, vanilla};

const VANILLA_PORT: u16 = 25566;
const DOPPEL_PORT: u16 = 25565;

/// The setblock'd stone floor block the held dig targets.
const ANCHOR: (i32, i32, i32) = (1, -60, 1);
/// The block right-clicked onto the anchor; only placed, never dug.
const PLACED: (i32, i32, i32) = (1, -59, 1);
/// The setblock'd chest the second right-click opens.
const CHEST: (i32, i32, i32) = (4, -60, 4);
/// The setblock'd obsidian the dig-and-release targets.
const OBSI: (i32, i32, i32) = (2, -60, 2);

/// player_action body: action VarInt, pos i64 (packed BlockPos),
/// direction VarInt, sequence VarInt. Direction "up" everywhere; the
/// drop arms ignore it.
fn build_player_action(action: i32, x: i32, y: i32, z: i32, sequence: i32) -> Vec<u8> {
    let mut b = Vec::new();
    doppel_protocol::write_varint(&mut b, action);
    b.extend_from_slice(&bot::pack_block_pos(x, y, z).to_be_bytes());
    doppel_protocol::write_varint(&mut b, 1); // direction: up
    doppel_protocol::write_varint(&mut b, sequence);
    b
}

/// Recursively copies a directory (a tiny std-only `cp -r`).
pub(crate) fn copy_dir(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir(&from, &to)?;
        } else {
            std::fs::copy(&from, &to).with_context(|| format!("copying {}", from.display()))?;
        }
    }
    Ok(())
}

pub(crate) fn default_doppel_bin() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("DOPPEL_BIN") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Ok(path);
        }
        anyhow::bail!("DOPPEL_BIN={} does not exist", path.display());
    }
    let exe = std::env::current_exe().context("locating current exe")?;
    let dir = exe
        .parent()
        .with_context(|| format!("no parent of {}", exe.display()))?;
    for name in ["doppel", "doppel.exe"] {
        let candidate = dir.join(name);
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    anyhow::bail!(
        "could not find the doppel binary next to doppel-oracle: \
         run `cargo build` first or set DOPPEL_BIN"
    )
}

pub(crate) fn wait_for_port(port: u16, timeout: Duration) -> Result<()> {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return Ok(());
        }
        if std::time::Instant::now() > deadline {
            anyhow::bail!("port {port} never opened");
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// Boots vanilla WITHOUT running any commands, captures the clean join
/// transcript as replay blobs, and snapshots the pristine flat world
/// (doppel boots against both).
pub(crate) fn capture_clean_blobs(
    pin: &doppel_protocol::Pin,
    jar: &std::path::Path,
    blobs_dir: &std::path::Path,
    pristine_world: &std::path::Path,
) -> Result<()> {
    let server = vanilla::boot(pin, jar, VANILLA_PORT)?;
    std::thread::sleep(Duration::from_secs(2));
    if pristine_world.exists() {
        std::fs::remove_dir_all(pristine_world)?;
    }
    let world = crate::vanilla::vanilla_dir()?.join("run").join("world");
    anyhow::ensure!(world.is_dir(), "vanilla world dir missing after boot");
    // session.lock stays OS-locked by the live server on Windows and is
    // meaningless to a reader; skip it.
    std::fs::create_dir_all(pristine_world)?;
    for entry in std::fs::read_dir(&world)? {
        let entry = entry?;
        if entry.file_name() == "session.lock" {
            continue;
        }
        let from = entry.path();
        let to = pristine_world.join(entry.file_name());
        if from.is_dir() {
            copy_dir(&from, &to)?;
        } else {
            std::fs::copy(&from, &to).with_context(|| format!("copying {}", from.display()))?;
        }
    }
    let login = capture::login_start_c("Doppel");
    let protocol = pin.protocol.unwrap_or(0);
    let v = bot::login_capture(
        "127.0.0.1",
        VANILLA_PORT,
        protocol,
        &login,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(5)),
            max_packets: Some(300),
            dump_dir: Some(blobs_dir),
            commands: &[],
            walk_chunks: None,
            raw_packets: &[],
        },
    )
    .context("capturing clean vanilla join")?;
    drop(server);
    let entries = capture::write_manifest(&v, blobs_dir)?;
    anyhow::ensure!(entries > 0, "no blobs captured from vanilla");
    println!(
        "[oracle] clean join blobs ({} packets) + pristine world at {}",
        v.iter().filter(|p| p.id >= 0).count(),
        pristine_world.display()
    );
    Ok(())
}

/// Decodes a captured update stream (0x08 + 0x56 packets) into ordered
/// (pos, state, t_ms) writes.
pub(crate) fn decode_update_writes(
    pkts: &[&bot::CapturedPacket],
) -> Vec<((i32, i32, i32), u32, u64)> {
    let mut out = Vec::new();
    for p in pkts {
        let raw = hex::decode(&p.head_hex).unwrap_or_default();
        if p.id == 0x08 && raw.len() >= 9 {
            let packed = i64::from_be_bytes(raw[0..8].try_into().unwrap());
            let x = ((packed >> 38) & 0x3ff_ffff) << 38 >> 38;
            let z = ((packed >> 12) & 0x3ff_ffff) << 38 >> 38;
            let y = (((packed & 0xfff) as i32) << 20) >> 20;
            let mut st = 0u32;
            let mut sh = 0u32;
            let mut o = 8usize;
            while o < raw.len() {
                let b = raw[o];
                o += 1;
                st |= u32::from(b & 0x7f) << sh;
                sh += 7;
                if b & 0x80 == 0 {
                    break;
                }
            }
            out.push(((x as i32, y, z as i32), st, p.t_ms as u64));
        } else if p.id == 0x56 && raw.len() >= 8 {
            let sec = i64::from_be_bytes(raw[0..8].try_into().unwrap());
            let sx = (sec >> 42) & 0x3f_ffff;
            let sz = (sec >> 20) & 0x3f_ffff;
            // 20-bit section Y sign-extends against i64's width.
            let sy = ((sec & 0xf_ffff) << 44) >> 44;
            let sx = (sx << 10) >> 10;
            let sz = (sz << 10) >> 10;
            let mut o = 8usize;
            let rd = |raw: &[u8], o: &mut usize| -> u64 {
                let mut v: u64 = 0;
                let mut sh = 0u32;
                while *o < raw.len() {
                    let b = raw[*o];
                    *o += 1;
                    v |= u64::from(b & 0x7f) << sh;
                    sh += 7;
                    if b & 0x80 == 0 {
                        break;
                    }
                }
                v
            };
            let count = rd(&raw, &mut o);
            for _ in 0..count {
                let e = rd(&raw, &mut o);
                let local = (e & 0xfff) as i32;
                let st = (e >> 12) as u32;
                let lx = (local >> 8) & 0xf;
                let lz = (local >> 4) & 0xf;
                let ly = local & 0xf;
                out.push((
                    (
                        (sx * 16 + lx as i64) as i32,
                        (sy * 16 + ly as i64) as i32,
                        (sz * 16 + lz as i64) as i32,
                    ),
                    st,
                    p.t_ms as u64,
                ));
            }
        }
    }
    out
}

/// Decodes block_destruction (0x05) frames into (pos, stage) pairs; the
/// digger's entity id is masked (each server allocates ids its own way).
fn decode_overlays(pkts: &[&bot::CapturedPacket]) -> Vec<((i32, i32, i32), i32)> {
    let mut out = Vec::new();
    for p in pkts {
        if p.id != 0x05 {
            continue;
        }
        let raw = hex::decode(&p.head_hex).unwrap_or_default();
        let mut o = 0usize;
        while let Some(&b) = raw.get(o) {
            o += 1;
            if b & 0x80 == 0 {
                break;
            }
        }
        if raw.len() < o + 9 {
            continue;
        }
        let packed = i64::from_be_bytes(raw[o..o + 8].try_into().unwrap());
        let x = ((packed >> 38) & 0x3ff_ffff) << 38 >> 38;
        let z = ((packed >> 12) & 0x3ff_ffff) << 38 >> 38;
        let y = (((packed & 0xfff) as i32) << 20) >> 20;
        out.push(((x as i32, y, z as i32), raw[o + 8] as i8 as i32));
    }
    out
}

/// Frames the order vote labels: one char per cross-type family. Types
/// outside the table (keep-alives, teleports) carry tick-independent
/// timing and stay unlabeled.
fn order_label(id: i32) -> Option<char> {
    Some(match id {
        0x3c => 'A',        // open_screen
        0x12 => 'C',        // container set_content
        0x14 => 'S',        // container set_slot
        0x08 | 0x56 => 'U', // block_update / section_blocks_update
        0x07 => 'B',        // block_event
        0x05 => 'O',        // block_destruction
        0x73 => 'T',        // set_time
        0x01 => 'E',        // add_entity
        0x65 => 'D',        // set_entity_data
        0x23 => 'M',        // entity_position_sync
        0x7f => 'P',        // set_equipped_item (pickup)
        0x4f => 'X',        // remove_entities
        _ => return None,
    })
}

/// The stream's cross-type order votes: `(label pair) -> [a-first, b-first]`
/// counts, pair labels sorted. Frames split into bursts at >15ms gaps -
/// one server tick's frames land inside a single burst on both servers -
/// and each pair of labels present in a burst votes on which label first
/// appeared in it. Packet identity differs per server; the relative order
/// of same-tick types is the comparable.
fn order_votes(
    pkts: &[bot::CapturedPacket],
    from: usize,
) -> std::collections::BTreeMap<(char, char), [usize; 2]> {
    let mut votes: std::collections::BTreeMap<(char, char), [usize; 2]> = Default::default();
    let mut burst: Vec<(char, usize)> = Vec::new();
    let mut last_ms = None;
    let flush = |burst: &mut Vec<(char, usize)>,
                 votes: &mut std::collections::BTreeMap<(char, char), [usize; 2]>| {
        let first: std::collections::BTreeMap<char, usize> = burst.iter().copied().collect();
        let labels: Vec<char> = first.keys().copied().collect();
        for (i, &a) in labels.iter().enumerate() {
            for &b in labels.iter().skip(i + 1) {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                let flipped = first[&lo] > first[&hi];
                let slot = votes.entry((lo, hi)).or_insert([0, 0]);
                slot[flipped as usize] += 1;
            }
        }
        burst.clear();
    };
    for p in pkts.iter().skip(from) {
        if p.id < 0 {
            continue;
        }
        let Some(label) = order_label(p.id) else {
            continue;
        };
        if last_ms.is_some_and(|t| p.t_ms.saturating_sub(t) > 15) {
            flush(&mut burst, &mut votes);
        }
        last_ms = Some(p.t_ms);
        if !burst.iter().any(|(l, _)| *l == label) {
            burst.push((label, burst.len()));
        }
    }
    flush(&mut burst, &mut votes);
    votes
}

/// One side's reduced observation: what the digger saw about itself and
/// what the standing witness saw about the world.
struct Side {
    /// The witness's 0x08/0x56 writes.
    updates: Vec<((i32, i32, i32), u32, u64)>,
    /// The digger's own 0x08/0x56 writes.
    digger_updates: Vec<((i32, i32, i32), u32, u64)>,
    /// The witness's 0x05 overlays, in order.
    overlays: Vec<((i32, i32, i32), i32)>,
    /// The witness's chest-lid block_event body (pos, event, param, id).
    lid: Option<Vec<u8>>,
    /// The digger's own 0x05 count (its client renders the overlay
    /// locally; the server never echoes it back).
    digger_overlays: usize,
    /// The digger's open_screen bodies after the command volley.
    screens: Vec<Vec<u8>>,
    /// The digger's container set_content frames after the volley.
    contents: usize,
    /// The digger's container set_slot frames after the volley.
    slots: usize,
}

fn analyze(digger: &[bot::CapturedPacket], witness: &[bot::CapturedPacket]) -> Side {
    let wrefs: Vec<_> = witness.iter().filter(|p| p.id >= 0).collect();
    let drefs: Vec<_> = digger.iter().filter(|p| p.id >= 0).collect();
    let updates = decode_update_writes(&wrefs);
    let digger_updates = decode_update_writes(&drefs);
    let overlays = decode_overlays(&wrefs);
    let lid = wrefs
        .iter()
        .filter(|p| p.id == 0x07)
        .map(|p| hex::decode(&p.head_hex).unwrap_or_default())
        .find(|raw| {
            raw.len() >= 9 && {
                let packed = i64::from_be_bytes(raw[0..8].try_into().unwrap());
                let x = ((packed >> 38) & 0x3ff_ffff) << 38 >> 38;
                let z = ((packed >> 12) & 0x3ff_ffff) << 38 >> 38;
                let y = (((packed & 0xfff) as i32) << 20) >> 20;
                (x as i32, y, z as i32) == CHEST
            }
        });
    // The volley marker: the last command's feedback precedes the raw
    // interaction burst on both servers. Frame ORDER draws the line, not
    // timestamps: a command's own packets can land in the same
    // millisecond as its feedback.
    let marker = digger.iter().rposition(|p| p.id == 0x7c);
    let after = |idx: usize| marker.is_some_and(|m| idx > m);
    let screens = digger
        .iter()
        .enumerate()
        .filter(|(i, p)| p.id == 0x3c && after(*i))
        .map(|(_, p)| hex::decode(&p.head_hex).unwrap_or_default())
        .collect();
    let contents = digger
        .iter()
        .enumerate()
        .filter(|(i, p)| p.id == 0x12 && after(*i))
        .count();
    let slots = digger
        .iter()
        .enumerate()
        .filter(|(i, p)| p.id == 0x14 && after(*i))
        .count();
    Side {
        updates,
        digger_updates,
        overlays,
        lid,
        digger_overlays: digger.iter().filter(|p| p.id >= 0 && p.id == 0x05).count(),
        screens,
        contents,
        slots,
    }
}

/// The scripted session pair against one server: the witness joins and
/// stands at spawn; the digger joins `lead` later, runs the paced
/// command volley, fires the raw interaction burst, and both idle out.
fn run_sessions(
    port: u16,
    protocol: i32,
) -> Result<(Vec<bot::CapturedPacket>, Vec<bot::CapturedPacket>)> {
    let commands: Vec<String> = [
        // Clear the tall grass standing over the dig and chest cells so
        // no neighbor-pop noise rides into the write streams.
        format!(
            "setblock {} {} {} minecraft:air",
            PLACED.0, PLACED.1, PLACED.2
        ),
        format!(
            "setblock {} {} {} minecraft:air",
            CHEST.0,
            CHEST.1 + 1,
            CHEST.2
        ),
        format!(
            "setblock {} {} {} minecraft:obsidian",
            OBSI.0, OBSI.1, OBSI.2
        ),
        format!(
            "setblock {} {} {} minecraft:stone",
            ANCHOR.0, ANCHOR.1, ANCHOR.2
        ),
        format!(
            "setblock {} {} {} minecraft:chest[facing=north,type=single,waterlogged=false]",
            CHEST.0, CHEST.1, CHEST.2
        ),
        "tp @s 1 -59 3".to_string(),
        "give @s minecraft:stone 64".to_string(),
    ]
    .to_vec();
    let raw: Vec<(i32, Vec<u8>)> = vec![
        (0x2c, Vec::new()), // player_loaded
        (
            0x42,
            bot::build_use_item_on_top(ANCHOR.0, ANCHOR.1, ANCHOR.2, 1),
        ),
        (
            0x42,
            bot::build_use_item_on_top(CHEST.0, CHEST.1, CHEST.2, 2),
        ),
        // Dig the obsidian and release below the finish threshold: the
        // dig converts to a delayed destroy whose frozen progress never
        // reaches a stage change or the break on a young player.
        (0x29, build_player_action(0, OBSI.0, OBSI.1, OBSI.2, 10)),
        (0x29, build_player_action(3, OBSI.0, OBSI.1, OBSI.2, 11)),
        // Dig the anchor, abort, and dig it again: the overlay stream
        // must clear on abort and resume from zero, then stay flat: the
        // pending delayed destroy starves the held dig's deepening.
        (
            0x29,
            build_player_action(0, ANCHOR.0, ANCHOR.1, ANCHOR.2, 12),
        ),
        (
            0x29,
            build_player_action(2, ANCHOR.0, ANCHOR.1, ANCHOR.2, 13),
        ),
        (
            0x29,
            build_player_action(0, ANCHOR.0, ANCHOR.1, ANCHOR.2, 14),
        ),
        (0x29, build_player_action(4, 0, 0, 0, 15)),
        (0x29, build_player_action(5, 0, 0, 0, 16)),
        (0x2e, Vec::new()),
    ];
    let witness_login = capture::login_start_c("Doppelist");
    let witness = std::thread::spawn(move || {
        bot::login_capture(
            "127.0.0.1",
            port,
            protocol,
            &witness_login,
            &bot::CaptureOpts {
                idle_timeout: Some(Duration::from_secs(20)),
                max_packets: Some(24000),
                dump_dir: None,
                commands: &[],
                walk_chunks: None,
                raw_packets: &[],
            },
        )
    });
    // The witness settles first; dig timing runs on each player's own
    // tick counter from join, so the stagger keeps nothing else moving.
    std::thread::sleep(Duration::from_secs(8));
    let login = capture::login_start_c("Doppel");
    let digger = bot::login_capture(
        "127.0.0.1",
        port,
        protocol,
        &login,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(8)),
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

/// The differential breaking test.
pub fn parity_break() -> Result<bool> {
    let pin = load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let root = doppel_protocol::find_repo_root()?;
    let blobs_dir = root.join("target").join("vanilla").join("blobs-break");
    let pristine_world = root
        .join("target")
        .join("vanilla")
        .join("pristine-world-break");
    for dir in [&blobs_dir, &pristine_world] {
        if dir.exists() {
            std::fs::remove_dir_all(dir)?;
        }
    }
    let protocol = pin.protocol.unwrap_or(0);

    // 1. Clean blobs + pristine world.
    capture_clean_blobs(&pin, &jar, &blobs_dir, &pristine_world)?;

    // 2. Vanilla reference sessions. Keep-alive traffic can outlive the
    // idle timers, so the phase carries its own wall-clock bound: closing
    // the server ends any still-blocked capture from outside.
    let server = vanilla::boot(&pin, &jar, VANILLA_PORT)?;
    let vworker = std::thread::spawn(move || run_sessions(VANILLA_PORT, protocol));
    let vdeadline = std::time::Instant::now() + Duration::from_secs(90);
    while !vworker.is_finished() && std::time::Instant::now() < vdeadline {
        std::thread::sleep(Duration::from_millis(500));
    }
    drop(server);
    let (v_digger, v_witness) = vworker
        .join()
        .map_err(|_| anyhow::anyhow!("vanilla session thread panicked"))??;

    // 3. Doppel sessions. The warm-up sleep puts doppel's tick counter
    // past the delayed-destroy threshold, matching a booted server.
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
    std::thread::sleep(Duration::from_secs(10));
    // Both servers keep an idle-based capture alive with steady traffic
    // (time pushes, entity tracking at spawn), so this phase carries the
    // same wall-clock bound as the vanilla one.
    let worker = std::thread::spawn(move || run_sessions(DOPPEL_PORT, protocol));
    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    while !worker.is_finished() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(500));
    }
    let _ = child.kill();
    let _ = child.wait();
    let (d_digger, d_witness) = worker
        .join()
        .map_err(|_| anyhow::anyhow!("doppel session thread panicked"))??;

    let v = analyze(&v_digger, &v_witness);
    let d = analyze(&d_digger, &d_witness);
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
    let show_writes = |w: &[((i32, i32, i32), u32, u64)]| {
        w.iter()
            .map(|(p, s, t)| format!("{p:?}={s}@{t}ms"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let marker_of = |digger: &[bot::CapturedPacket]| digger.iter().rposition(|p| p.id == 0x7c);
    for (who, digger, witness, side) in [
        ("vanilla", &v_digger, &v_witness, &v),
        ("doppel", &d_digger, &d_witness, &d),
    ] {
        println!(
            "[oracle] {who}: digger {} frames, witness {} frames",
            digger.len(),
            witness.len()
        );
        println!("[oracle] {who} digger ids: {}", histogram(digger));
        println!("[oracle] {who} witness ids: {}", histogram(witness));
        println!(
            "[oracle] {who} witness writes: {}",
            show_writes(&side.updates)
        );
        println!(
            "[oracle] {who} digger writes: {}",
            show_writes(&side.digger_updates)
        );
        let marker = marker_of(digger);
        let tail: Vec<String> = digger
            .iter()
            .enumerate()
            .filter(|(i, p)| {
                marker.is_some_and(|m| *i > m)
                    && matches!(p.id, 0x05 | 0x07 | 0x08 | 0x12 | 0x14 | 0x3c | 0x56)
            })
            .map(|(_, p)| format!("{:#04x}@{}ms", p.id, p.t_ms))
            .collect();
        println!("[oracle] {who} digger post-volley tail: {}", tail.join(" "));
        println!(
            "[oracle] {who} overlays: {:?}",
            side.overlays
                .iter()
                .map(|(p, s)| format!("{p:?}:{s}"))
                .collect::<Vec<_>>()
        );
    }

    let final_at = |side: &Side, target: (i32, i32, i32)| {
        side.updates
            .iter()
            .rev()
            .find(|(p, _, _)| *p == target)
            .map(|(_, s, _)| *s)
    };
    let overlays_at = |side: &Side, target: (i32, i32, i32)| {
        side.overlays
            .iter()
            .filter(|(p, _)| *p == target)
            .map(|(_, s)| *s)
            .collect::<Vec<_>>()
    };

    let mut failures = Vec::new();

    // Final block states: nothing breaks this session (the released
    // obsidian dig freezes below every threshold), so the four scenario
    // cells land on their command- and interaction-set states.
    for (who, side) in [("vanilla", &v), ("doppel", &d)] {
        match final_at(side, ANCHOR) {
            Some(1) => {}
            s => failures.push(format!("{who}: anchor final state {s:?}, want stone (1)")),
        }
        match final_at(side, PLACED) {
            Some(1) => {}
            s => failures.push(format!("{who}: placed final state {s:?}, want stone (1)")),
        }
        match final_at(side, OBSI) {
            Some(s) if s != 0 => {}
            s => failures.push(format!(
                "{who}: obsidian final state {s:?}, want a nonzero obsidian state"
            )),
        }
        match final_at(side, CHEST) {
            Some(_) => {}
            None => failures.push(format!("{who}: chest never written")),
        }
    }
    // The cleared cell above the chest (worldgen grass cover varies per
    // boot, so only the two servers' agreement is pinned there).
    let above_chest = (CHEST.0, CHEST.1 + 1, CHEST.2);
    for target in [ANCHOR, PLACED, OBSI, CHEST, above_chest] {
        if final_at(&v, target) != final_at(&d, target) {
            failures.push(format!(
                "final states differ at {target:?}: vanilla {:?} vs doppel {:?}",
                final_at(&v, target),
                final_at(&d, target)
            ));
        }
    }

    // Overlays: the digger never receives its own; the witness sees the
    // obsidian dig open (and nothing more: the frozen delayed destroy
    // never crosses a stage), and the anchor dig clear on abort, resume,
    // and then stay flat while the delayed destroy starves it.
    for (who, side) in [("vanilla", &v), ("doppel", &d)] {
        if side.digger_overlays != 0 {
            failures.push(format!(
                "{who}: digger received {} of its own overlays",
                side.digger_overlays
            ));
        }
        let stray: Vec<_> = side
            .overlays
            .iter()
            .filter(|(p, _)| *p != ANCHOR && *p != OBSI)
            .collect();
        if !stray.is_empty() {
            failures.push(format!("{who}: overlays at unexpected positions {stray:?}"));
        }
        let obsi = overlays_at(side, OBSI);
        if obsi != vec![0] {
            failures.push(format!(
                "{who}: release-dig overlay stream {obsi:?}, want [0]"
            ));
        }
        let anchor = overlays_at(side, ANCHOR);
        if anchor != vec![0, -1, 0] {
            failures.push(format!(
                "{who}: held-dig overlay stream {anchor:?}, want [0, -1, 0]"
            ));
        }
        let placed = overlays_at(side, PLACED);
        if !placed.is_empty() {
            failures.push(format!(
                "{who}: overlays on the never-dug placed block {placed:?}"
            ));
        }
    }

    // The chest interaction: one menu open with identical bytes, a lid
    // event with identical bytes, and matching inventory syncs.
    if v.screens.len() != 1 {
        failures.push(format!("vanilla: {} open_screens", v.screens.len()));
    }
    if d.screens.len() != 1 {
        failures.push(format!("doppel: {} open_screens", d.screens.len()));
    }
    if v.screens.len() == 1 && d.screens.len() == 1 && v.screens[0] != d.screens[0] {
        failures.push(format!(
            "open_screen bodies differ: vanilla {:02x?} vs doppel {:02x?}",
            v.screens[0], d.screens[0]
        ));
    }
    match (&v.lid, &d.lid) {
        (Some(a), Some(b)) => {
            if a != b {
                failures.push(format!(
                    "chest lid block_events differ: vanilla {a:02x?} vs doppel {b:02x?}"
                ));
            }
        }
        (None, None) => failures.push("neither server fired the chest lid".into()),
        (who, _) => failures.push(format!("chest lid missing on one side ({who:?})")),
    }
    if v.contents != d.contents {
        failures.push(format!(
            "container set_content frames after volley: vanilla {} vs doppel {}",
            v.contents, d.contents
        ));
    }
    // Neither the placement's spent stack (masked by the open chest
    // menu) nor the drops sync a slot on either server.
    if v.slots != d.slots {
        failures.push(format!(
            "container set_slot frames after volley: vanilla {} vs doppel {}",
            v.slots, d.slots
        ));
    }

    // Cross-type same-tick order: a pair the reference orders unanimously
    // in at least two bursts is a settled invariant; doppel fails only
    // when its votes MAJORITY-oppose it. Single-burst evidence stays
    // unjudged, and split doppel votes abstain: wire captures show both
    // servers emit entity spawn before entity data (verified frame by
    // frame), so the lone opposite vote traces to a burst boundary
    // splitting one entity's data packet from another entity's spawn -
    // a measurement artifact, not an ordering change. A genuine reorder
    // draws lopsided opposite votes, never a tie. The positive-pair
    // count keeps an empty capture from passing.
    let mut compared_pairs = 0usize;
    for (stream, vpk, dpk) in [
        ("digger", &v_digger, &d_digger),
        ("witness", &v_witness, &d_witness),
    ] {
        let vstart = match marker_of(vpk) {
            Some(m) => m + 1,
            None => 0,
        };
        let dstart = match marker_of(dpk) {
            Some(m) => m + 1,
            None => 0,
        };
        let vvotes = order_votes(vpk, vstart);
        let dvotes = order_votes(dpk, dstart);
        for ((lo, hi), [lo_first, hi_first]) in &vvotes {
            let van_votes = lo_first + hi_first;
            let van_unanimous = van_votes >= 2 && (*lo_first == 0) != (*hi_first == 0);
            if !van_unanimous {
                continue;
            }
            let want_first = if *lo_first > 0 { *lo } else { *hi };
            let [dlo, dhi] = dvotes.get(&(*lo, *hi)).copied().unwrap_or([0, 0]);
            let d_want = if *lo_first > 0 { dlo } else { dhi };
            let d_other = if *lo_first > 0 { dhi } else { dlo };
            if d_other > d_want {
                failures.push(format!(
                    "{stream}: same-tick order {want_first} first voted opposite                      ({lo}?{hi}: vanilla {lo_first}+{hi_first}, doppel {dlo}+{dhi})"
                ));
                // Name the offending bursts: which packets, what order,
                // what times - a flush-merge artifact and a real reorder
                // read differently here (different entity ids across a
                // seam vs the same id inside one flush).
                let mut shown = 0;
                let mut burst: Vec<(char, i32, u128)> = Vec::new();
                let mut last_ms: Option<u128> = None;
                for p in dpk.iter().skip(dstart) {
                    if p.id < 0 {
                        continue;
                    }
                    let Some(l) = order_label(p.id) else {
                        continue;
                    };
                    if last_ms.is_some_and(|t| p.t_ms.saturating_sub(t) > 15) {
                        let firsts: std::collections::BTreeMap<char, u128> =
                            burst.iter().map(|(l, _, t)| (*l, *t)).collect();
                        if let (Some(at), Some(bt)) = (firsts.get(lo), firsts.get(hi)) {
                            if at < bt {
                                println!(
                                    "  {stream} opposite burst: {}",
                                    burst
                                        .iter()
                                        .map(|(l, id, t)| format!("{l}=0x{id:02x}@{t}ms"))
                                        .collect::<Vec<_>>()
                                        .join(" ")
                                );
                                shown += 1;
                            }
                        }
                        if shown >= 3 {
                            break;
                        }
                        burst.clear();
                    }
                    burst.push((l, p.id, p.t_ms));
                    last_ms = Some(p.t_ms);
                }
            }
            if d_want > 0 {
                compared_pairs += 1;
            }
        }
    }
    if compared_pairs == 0 {
        failures.push("no cross-type same-tick order pairs compared".into());
    }
    println!("[oracle] cross-type same-tick order: {compared_pairs} positively compared pair(s)");

    if failures.is_empty() {
        println!("PASS: breaking parity");
        Ok(true)
    } else {
        println!("FAIL: {} breaking difference(s):", failures.len());
        for f in failures.iter().take(12) {
            println!("  {f}");
        }
        Ok(false)
    }
}
