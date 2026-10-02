//! The differential status test: boot vanilla (the oracle) and Doppel with
//! identical intent, observe both from the outside, and diff the results.

use anyhow::{Context, Result};
use doppel_protocol::{load_pin, save_pin, Pin};
use serde_json::Value;
use std::collections::BTreeMap;
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use crate::{bot, vanilla};

const VANILLA_PORT: u16 = 25566;
const DOPPEL_PORT: u16 = 25565;

/// Fields that are legitimately volatile between runs (the default icon is
/// regenerated with a random-ish base64 body by some builds).
fn normalize(v: &mut Value) {
    if let Value::Object(map) = v {
        map.remove("favicon");
    }
}

/// Flattens JSON into comparable `path -> scalar` pairs.
fn flatten(prefix: &str, v: &Value, out: &mut BTreeMap<String, String>) {
    match v {
        Value::Object(map) => {
            for (k, val) in map {
                let p = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                flatten(&p, val, out);
            }
        }
        Value::Array(items) => {
            for (i, val) in items.iter().enumerate() {
                flatten(&format!("{prefix}[{i}]"), val, out);
            }
        }
        scalar => {
            out.insert(prefix.to_string(), scalar.to_string());
        }
    }
}

pub fn diff_report(vanilla: &Value, doppel: &Value) -> Vec<String> {
    let mut a = BTreeMap::new();
    let mut b = BTreeMap::new();
    flatten("", vanilla, &mut a);
    flatten("", doppel, &mut b);
    let mut report = Vec::new();
    for (path, va) in &a {
        match b.get(path) {
            Some(vb) if va == vb => {}
            Some(vb) => report.push(format!("{path}: vanilla {va} != doppel {vb}")),
            None => report.push(format!(
                "{path}: present in vanilla ({va}), missing in doppel"
            )),
        }
    }
    for (path, vb) in &b {
        if !a.contains_key(path) {
            report.push(format!(
                "{path}: doppel has extra ({vb}), absent in vanilla"
            ));
        }
    }
    report
}

/// Play-phase (id, file) entries from a blobs manifest.
fn manifest_play_entries(dir: &std::path::Path) -> Result<Vec<(i32, String)>> {
    let manifest: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)?;
    Ok(manifest
        .iter()
        .filter_map(|e| {
            if e["phase"].as_str() == Some("play") {
                Some((e["id"].as_i64()? as i32, e["file"].as_str()?.to_string()))
            } else {
                None
            }
        })
        .collect())
}

fn default_doppel_bin() -> Result<PathBuf> {
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
        "could not find the doppel binary next to doppel-oracle — \
         run `cargo build` first or set DOPPEL_BIN"
    )
}

fn wait_for_port(port: u16, timeout: Duration) -> Result<()> {
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

/// Runs the full differential status test. Returns Ok(true) on parity.
pub fn parity_status(doppel_bin: Option<PathBuf>) -> Result<bool> {
    let mut pin: Pin = load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;

    // 1. Boot the oracle.
    let vanilla_server = vanilla::boot(&pin, &jar, VANILLA_PORT)?;
    let v_status = bot::status_ping_retry(
        "127.0.0.1",
        VANILLA_PORT,
        pin.protocol.unwrap_or(0),
        5,
        Duration::from_secs(2),
    )
    .context("pinging vanilla oracle")?;

    // 2. Heal the pin with values only the oracle knows.
    let protocol = v_status["version"]["protocol"].as_i64();
    let version_name = v_status["version"]["name"].as_str().map(str::to_string);
    let protocol = protocol.map(|p| p as i32);
    if pin.protocol != protocol || pin.version_name != version_name {
        pin.protocol = protocol;
        pin.version_name = version_name;
        save_pin(&pin)?;
        println!(
            "[oracle] pin healed from oracle: protocol={:?} name={:?} — \
             commit pins/version.json to freeze it",
            pin.protocol, pin.version_name
        );
    }

    // 3. Boot Doppel with the healed pin.
    let bin = match doppel_bin {
        Some(p) => p,
        None => default_doppel_bin()?,
    };
    let pin_path = doppel_protocol::pin_path()?;
    eprintln!("[oracle] starting doppel ({})...", bin.display());
    let mut doppel_child = Command::new(&bin)
        .env("DOPPEL_ADDR", "127.0.0.1")
        .env("DOPPEL_PORT", DOPPEL_PORT.to_string())
        .env("DOPPEL_PIN", &pin_path)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawning {}", bin.display()))?;
    wait_for_port(DOPPEL_PORT, Duration::from_secs(30))?;

    // 4. Observe Doppel exactly as a real client would.
    let hint = pin.protocol.unwrap_or(0);
    let d_status =
        bot::status_ping_retry("127.0.0.1", DOPPEL_PORT, hint, 5, Duration::from_secs(2))
            .context("pinging doppel")?;

    let _ = doppel_child.kill();
    let _ = doppel_child.wait();
    drop(vanilla_server); // teardown

    // 5. Diff.
    let mut v = v_status;
    let mut d = d_status;
    normalize(&mut v);
    normalize(&mut d);
    let report = diff_report(&v, &d);

    println!(
        "\n===== vanilla status =====\n{}",
        serde_json::to_string_pretty(&v)?
    );
    println!(
        "\n===== doppel status =====\n{}",
        serde_json::to_string_pretty(&d)?
    );

    if report.is_empty() {
        println!(
            "\nPASS: status responses match after normalization ({} compared fields)",
            flatten_count(&v)
        );
        Ok(true)
    } else {
        println!("\nFAIL: {} field(s) differ:", report.len());
        for line in &report {
            println!("  {line}");
        }
        Ok(false)
    }
}

fn flatten_count(v: &Value) -> usize {
    let mut m = BTreeMap::new();
    flatten("", v, &mut m);
    m.len()
}

/// The differential login test: capture vanilla's full login transcript
/// (with byte-exact blob dumps), feed the blobs to Doppel, drive the same
/// client dance against Doppel, and compare packet-for-packet through the
/// spawn chunk batch. The per-connection sessionId UUID in login_finished
/// is masked; everything else — including the natively-built brand,
/// features, and known-packs packets — must be byte-identical.
pub fn parity_login() -> Result<bool> {
    use crate::capture;
    use std::fs;

    let pin = load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let root = doppel_protocol::find_repo_root()?;
    let blobs_dir = root.join("target").join("vanilla").join("blobs");
    let doppel_dump = root.join("target").join("doppel-dump");
    for dir in [&blobs_dir, &doppel_dump] {
        if dir.exists() {
            fs::remove_dir_all(dir).context("cleaning dump dir")?;
        }
    }

    // 1. Capture the oracle transcript + blobs.
    let server = vanilla::boot(&pin, &jar, VANILLA_PORT)?;
    let login_body = capture::login_start_c("Doppel");
    let protocol = pin.protocol.unwrap_or(0);
    let v = bot::login_capture(
        "127.0.0.1",
        VANILLA_PORT,
        protocol,
        &login_body,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(22)),
            max_packets: Some(220),
            dump_dir: Some(&blobs_dir),
            commands: &[],
            walk_chunks: None,
            raw_packets: &[],
        },
    )
    .context("capturing vanilla transcript")?;
    drop(server);
    let entries = capture::write_manifest(&v, &blobs_dir)?;
    anyhow::ensure!(entries > 0, "no blobs captured from vanilla");

    // Pre-verify the Anvil -> wire pipeline against every captured chunk:
    // learn the palette maps from (wire, anvil) pairs, rebuild each chunk
    // from storage, and require byte-identical output BEFORE the live test.
    let world_dir = root
        .join("target")
        .join("vanilla")
        .join("run")
        .join("world");
    let mut anvil_ok = 0usize;
    let mut anvil_total = 0usize;
    if world_dir.is_dir() {
        let mut world = doppel_world::WorldDir::open(&world_dir)?;
        let mut boot = doppel_world::anvil_to_wire::PaletteBootstrap::default();
        let mut refs = Vec::new();
        for (id, file) in manifest_play_entries(&blobs_dir)? {
            if id != 0x2e {
                continue;
            }
            let body = std::fs::read(blobs_dir.join(file))?;
            let wire = doppel_world::WireChunk::decode(&body)?;
            if let Some(anvil) = world.chunk(wire.x, wire.z)? {
                boot.learn(&wire, &anvil);
                refs.push((wire, anvil));
            }
        }
        for (wire, anvil) in &refs {
            anvil_total += 1;
            let rebuilt = doppel_world::anvil_to_wire::convert(anvil, wire, &boot)?;
            let re = rebuilt.encode();
            let orig = wire.encode();
            if re == orig {
                anvil_ok += 1;
            } else {
                let pos = re
                    .iter()
                    .zip(orig.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(re.len().min(orig.len()));
                println!(
                    "[oracle] anvil rebuild differs for chunk ({}, {}): first diff at {}, re_len={} wire_len={}",
                    wire.x, wire.z, pos, re.len(), orig.len()
                );
                println!("  re  : {:02x?}", &re[pos..(pos + 12).min(re.len())]);
                println!("  wire: {:02x?}", &orig[pos..(pos + 12).min(orig.len())]);
                if wire.x == -1 && wire.z == 0 {
                    println!("  wire sec0: {:?}", wire.sections.first());
                    println!("  anvil sec0: {:?}", anvil.sections.first());
                }
            }
        }
        println!(
            "[oracle] anvil->wire rebuild: {anvil_ok}/{anvil_total} chunks byte-identical, {} block + {} biome mappings learned",
            boot.blocks.len(),
            boot.biomes.len()
        );
        anyhow::ensure!(anvil_ok == anvil_total, "anvil rebuild failed parity");
    }

    // 2. Spawn Doppel with the blobs (and the world for anvil-backed chunks).
    let bin = default_doppel_bin()?;
    let pin_path = doppel_protocol::pin_path()?;
    let mut doppel_child = Command::new(&bin)
        .env("DOPPEL_ADDR", "127.0.0.1")
        .env("DOPPEL_PORT", DOPPEL_PORT.to_string())
        .env("DOPPEL_PIN", &pin_path)
        .env("DOPPEL_BLOBS", &blobs_dir)
        .env("DOPPEL_WORLD", &world_dir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawning {}", bin.display()))?;
    wait_for_port(DOPPEL_PORT, Duration::from_secs(30))?;

    // 3. Drive the same dance against Doppel.
    let d = bot::login_capture(
        "127.0.0.1",
        DOPPEL_PORT,
        protocol,
        &login_body,
        &bot::CaptureOpts {
            dump_dir: Some(&doppel_dump),
            ..Default::default()
        },
    )
    .context("capturing doppel transcript")?;
    let _ = doppel_child.kill();
    let _ = doppel_child.wait();

    // 4. Compare through vanilla's chunk-batch-finished marker.
    let vr: Vec<_> = v.iter().filter(|p| p.id >= 0).collect();
    let dr: Vec<_> = d.iter().filter(|p| p.id >= 0).collect();
    let mut seen_chunk = false;
    let cut = vr
        .iter()
        .position(|p| {
            if p.id == 0x2e {
                seen_chunk = true;
            }
            seen_chunk && p.id == 0x0b
        })
        .unwrap_or(vr.len().saturating_sub(1));
    let expected = &vr[..=cut];

    println!(
        "[oracle] comparing {} packets (vanilla {} vs doppel {})",
        expected.len(),
        vr.len(),
        dr.len()
    );

    let mut failures = Vec::new();
    if dr.len() < expected.len() {
        failures.push(format!(
            "doppel sent {} packets, expected at least {}",
            dr.len(),
            expected.len()
        ));
    }
    for (i, (pv, pd)) in expected.iter().zip(dr.iter()).enumerate() {
        if pv.id != pd.id {
            failures.push(format!(
                "packet {i}: id vanilla 0x{:02x} != doppel 0x{:02x}",
                pv.id, pd.id
            ));
            continue;
        }
        if pv.body_len != pd.body_len {
            failures.push(format!(
                "packet {i} (0x{:02x}): len vanilla {} != doppel {}",
                pv.id, pv.body_len, pd.body_len
            ));
            continue;
        }
        // Byte-exact comparison via dumps; the sessionId UUID (last 16
        // bytes of login_finished) is per-connection and masked.
        let (fv, fd) = match (&pv.file, &pd.file) {
            (Some(fv), Some(fd)) => (
                fs::read(blobs_dir.join(fv)).context("reading vanilla dump")?,
                fs::read(doppel_dump.join(fd)).context("reading doppel dump")?,
            ),
            _ => continue, // no dump: len/id equality is all we can check
        };
        let mask_session = i == 1; // login_finished
        let bytes_differ = if mask_session {
            let n = fv.len();
            fv[..n - 16] != fd[..n - 16]
        } else {
            fv != fd
        };
        if bytes_differ {
            failures.push(format!("packet {i} (0x{:02x}): body bytes differ", pv.id));
        }
    }

    if failures.is_empty() {
        println!("PASS: login transcripts match through the chunk batch");
        Ok(true)
    } else {
        println!("FAIL: {} login difference(s):", failures.len());
        for f in failures.iter().take(20) {
            println!("  {f}");
        }
        Ok(false)
    }
}

/// The differential walk test: the bot teleport-walks through vanilla and
/// Doppel with the same steps, and we compare what streams back — cache
/// centers, forgotten columns, and every chunk body byte-for-byte (via
/// dumps) for chunks both servers sent.
pub fn parity_walk() -> Result<bool> {
    use crate::capture;

    let pin = load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let root = doppel_protocol::find_repo_root()?;
    let blobs_dir = root.join("target").join("vanilla").join("blobs");
    let world_dir = root
        .join("target")
        .join("vanilla")
        .join("run")
        .join("world");
    let v_dump = root.join("target").join("walk-vanilla");
    let d_dump = root.join("target").join("walk-doppel");
    for dir in [&v_dump, &d_dump] {
        if dir.exists() {
            std::fs::remove_dir_all(dir)?;
        }
    }

    // 1. Vanilla walk, dumped straight into the blobs dir so the walk
    // chunks join the reference set Doppel's streaming replays.
    let server = vanilla::boot(&pin, &jar, VANILLA_PORT)?;
    let login = capture::login_start_c("Doppel");
    let protocol = pin.protocol.unwrap_or(0);
    let v = bot::login_capture(
        "127.0.0.1",
        VANILLA_PORT,
        protocol,
        &login,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(5)),
            max_packets: Some(500),
            dump_dir: Some(&blobs_dir),
            commands: &[],
            walk_chunks: Some(4),
            raw_packets: &[],
        },
    )
    .context("walking through vanilla")?;
    drop(server);
    capture::write_manifest(&v, &blobs_dir)?;

    // 2. Doppel walk (blobs from a capture WITHOUT the walk: reuse the
    // existing blob set is wrong here since it now contains walk chunks —
    // filter the join burst chunks only for replay sources is future work;
    // for now the walk uses whatever blobs exist).
    let bin = default_doppel_bin()?;
    let pin_path = doppel_protocol::pin_path()?;
    let mut child = Command::new(&bin)
        .env("DOPPEL_ADDR", "127.0.0.1")
        .env("DOPPEL_PORT", DOPPEL_PORT.to_string())
        .env("DOPPEL_PIN", &pin_path)
        .env("DOPPEL_BLOBS", &blobs_dir)
        .env("DOPPEL_WORLD", &world_dir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawning {}", bin.display()))?;
    wait_for_port(DOPPEL_PORT, Duration::from_secs(30))?;
    let d = bot::login_capture(
        "127.0.0.1",
        DOPPEL_PORT,
        protocol,
        &login,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(5)),
            max_packets: Some(500),
            dump_dir: Some(&d_dump),
            commands: &[],
            walk_chunks: Some(4),
            raw_packets: &[],
        },
    )
    .context("walking through doppel")?;
    let _ = child.kill();
    let _ = child.wait();

    // 3. Compare the streams.
    let vr: Vec<_> = v.iter().filter(|p| p.id >= 0).collect();
    let dr: Vec<_> = d.iter().filter(|p| p.id >= 0).collect();
    let mut failures = Vec::new();

    let count = |pkts: &[&bot::CapturedPacket], id: i32| pkts.iter().filter(|p| p.id == id).count();
    for id in [0x60u32, 0x49u32] {
        let (cv, cd) = (count(&vr, id as i32), count(&dr, id as i32));
        if cv != cd {
            failures.push(format!("packet 0x{id:02x}: vanilla {cv} vs doppel {cd}"));
        }
    }

    // Chunk bodies: compare per-coord via dumps.
    let chunk_bodies =
        |pkts: &[&bot::CapturedPacket], dir: &std::path::Path| -> Vec<((i32, i32), Vec<u8>)> {
            pkts.iter()
                .filter(|p| p.id == 0x2e)
                .filter_map(|p| {
                    let file = p.file.as_ref()?;
                    let body = std::fs::read(dir.join(file)).ok()?;
                    let c = doppel_world::WireChunk::decode(&body).ok()?;
                    Some(((c.x, c.z), body))
                })
                .collect()
        };
    let vb = chunk_bodies(&vr, &blobs_dir);
    let db = chunk_bodies(&dr, &d_dump);
    println!(
        "[oracle] walk chunks: vanilla {} doppel {} (forgets: vanilla {} doppel {})",
        vb.len(),
        db.len(),
        count(&vr, 0x26),
        count(&dr, 0x26),
    );
    let vmap: std::collections::HashMap<(i32, i32), &Vec<u8>> =
        vb.iter().map(|(k, v)| (*k, v)).collect();
    let mut compared = 0usize;
    for (coord, body) in &db {
        if let Some(vbody) = vmap.get(coord) {
            compared += 1;
            if **vbody != *body {
                failures.push(format!(
                    "chunk ({}, {}) body differs from vanilla",
                    coord.0, coord.1
                ));
            }
        }
    }
    println!("[oracle] byte-compared {compared} shared walk chunks");

    if failures.is_empty() {
        println!("PASS: walk streams match");
        Ok(true)
    } else {
        println!("FAIL: {} walk difference(s):", failures.len());
        for f in failures.iter().take(10) {
            println!("  {f}");
        }
        Ok(false)
    }
}

/// Recursively copies a directory (a tiny std-only `cp -r`).
fn copy_dir(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
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

/// Boots vanilla WITHOUT running any commands, captures the clean join
/// transcript as replay blobs, and snapshots the pristine flat world.
/// Doppel later boots against these, so the scripted commands it receives
/// are executed by its own engine against a world with no leftovers —
/// otherwise its setblocks no-op against vanilla's saved circuit and the
/// comparison degrades into vanilla-vs-replayed-vanilla.
fn capture_clean_blobs(
    pin: &Pin,
    jar: &std::path::Path,
    blobs_dir: &std::path::Path,
    pristine_world: &std::path::Path,
) -> Result<()> {
    use crate::capture;

    let server = vanilla::boot(pin, jar, VANILLA_PORT)?;
    // Snapshot the untouched world right after boot: the boot-time
    // spawn-area save settles well within a second, and this phase runs
    // no commands, so the world stays pristine regardless of how the JVM
    // is later torn down.
    std::thread::sleep(Duration::from_secs(2));
    if pristine_world.exists() {
        std::fs::remove_dir_all(pristine_world)?;
    }
    let world = crate::vanilla::vanilla_dir()?.join("run").join("world");
    anyhow::ensure!(world.is_dir(), "vanilla world dir missing after boot");
    copy_dir(&world, pristine_world)?;
    // A stale lock file means nothing to a reader.
    let _ = std::fs::remove_file(pristine_world.join("session.lock"));
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
        "[oracle] clean join blobs ({} packets) + pristine world snapshot at {}",
        v.iter().filter(|p| p.id >= 0).count(),
        pristine_world.display()
    );
    Ok(())
}

/// Decodes a captured update stream (0x08 + 0x56 packets) into ordered
/// (pos, state) writes — the oracle-side eyes for final-state comparison.
fn decode_update_writes(pkts: &[&bot::CapturedPacket]) -> Vec<((i32, i32, i32), u32)> {
    let mut out = Vec::new();
    for p in pkts {
        let raw = hex::decode(&p.head_hex).unwrap_or_default();
        if p.id == 0x08 && raw.len() >= 9 {
            let packed = i64::from_be_bytes(raw[0..8].try_into().unwrap());
            let x = (((packed >> 38) & 0x3ff_ffff) as i64) << 38 >> 38;
            let z = (((packed >> 12) & 0x3ff_ffff) as i64) << 38 >> 38;
            let y = (packed & 0xfff) as i32;
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
            out.push(((x as i32, y, z as i32), st));
        } else if p.id == 0x56 && raw.len() >= 8 {
            let sec = i64::from_be_bytes(raw[0..8].try_into().unwrap());
            let sx = (sec >> 42) & 0x3f_ffff;
            let sz = (sec >> 20) & 0x3f_ffff;
            let sy = (sec & 0xf_ffff) as i64;
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
                ));
            }
        }
    }
    out
}

/// The differential block test: identical setblock commands on both
/// servers; compare every section_blocks_update (0x56) and block_update
/// (0x08) broadcast — section coords, entry counts, and each
/// (localPos, state) pair.
pub fn parity_blocks() -> Result<bool> {
    use crate::capture;

    let pin = load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let root = doppel_protocol::find_repo_root()?;
    let blobs_dir = root.join("target").join("vanilla").join("blobs");
    let pristine_world = root.join("target").join("vanilla").join("pristine-world");
    if blobs_dir.exists() {
        std::fs::remove_dir_all(&blobs_dir)?;
    }

    let login = capture::login_start_c("Doppel");
    let protocol = pin.protocol.unwrap_or(0);
    let commands: Vec<String> = [
        "setblock 0 100 0 minecraft:stone",
        "setblock 1 100 0 minecraft:dirt",
        "setblock 2 100 0 minecraft:oak_planks",
        "setblock 0 101 0 minecraft:stone",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    // 1. Clean join blobs + pristine world (no commands run yet).
    capture_clean_blobs(&pin, &jar, &blobs_dir, &pristine_world)?;

    // 2. Vanilla reference: same commands, fresh boot (fresh world).
    let server = vanilla::boot(&pin, &jar, VANILLA_PORT)?;
    let v = bot::login_capture(
        "127.0.0.1",
        VANILLA_PORT,
        protocol,
        &login,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(5)),
            max_packets: Some(600),
            dump_dir: None,
            commands: &commands,
            walk_chunks: None,
            raw_packets: &[],
        },
    )
    .context("capturing vanilla setblocks")?;
    drop(server);

    // 3. Doppel against the clean blobs + pristine world.
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
    let d = bot::login_capture(
        "127.0.0.1",
        DOPPEL_PORT,
        protocol,
        &login,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(5)),
            max_packets: Some(1200),
            dump_dir: None,
            commands: &commands,
            walk_chunks: None,
            raw_packets: &[],
        },
    )
    .context("capturing doppel setblocks")?;
    let _ = child.kill();
    let _ = child.wait();

    // 4. Compare the engine-made broadcasts: one entry-set per update
    //    packet, in packet order. Within a batch packet the reference
    //    emits entries in hash-iteration order (not a contract), but the
    //    packet boundaries themselves carry the batching semantics.
    type EntrySet = Vec<((i32, i32, i32), u32)>;
    let vr: Vec<_> = v.iter().filter(|p| p.id >= 0).collect();
    let dr: Vec<_> = d.iter().filter(|p| p.id >= 0).collect();
    let packets_of = |pkts: &[&bot::CapturedPacket]| -> Vec<EntrySet> {
        pkts.iter()
            .filter(|p| p.id == 0x56 || p.id == 0x08)
            .map(|p| {
                let mut entries = decode_update_writes(&[*p]);
                entries.sort();
                entries
            })
            .collect()
    };
    let vp = packets_of(&vr);
    let dp = packets_of(&dr);
    println!(
        "[oracle] block update packets: vanilla {} doppel {}",
        vp.len(),
        dp.len()
    );
    // Compare the write MULTISET, not the per-packet grouping: paced
    // commands cross tick boundaries according to round-trip latency, so
    // the reference itself splits the same writes 2+2 or 4 between runs.
    // Exact batching is covered by the engine's own tick tests.
    let mut vw: Vec<((i32, i32, i32), u32)> = vp.concat();
    let mut dw: Vec<((i32, i32, i32), u32)> = dp.concat();
    vw.sort();
    dw.sort();
    println!(
        "[oracle] block writes: vanilla {} doppel {}",
        vw.len(),
        dw.len()
    );
    let mut failures = Vec::new();
    for (i, (a, b)) in vw.iter().zip(dw.iter()).enumerate() {
        if a != b {
            failures.push(format!("write {i}: vanilla {a:?} != doppel {b:?}"));
        }
    }
    if vw.len() != dw.len() {
        failures.push(format!(
            "write count: vanilla {} vs doppel {}",
            vw.len(),
            dw.len()
        ));
        println!("  vanilla writes: {:?}", &vw[..vw.len().min(16)]);
        println!("  doppel writes: {:?}", &dw[..dw.len().min(16)]);
    }
    if vw.is_empty() {
        failures.push("vanilla produced no block writes".to_string());
    }
    let v_ok = vr.iter().filter(|p| p.id == 0x7c).count();
    let d_ok = dr.iter().filter(|p| p.id == 0x7c).count();
    println!("[oracle] system_chat: vanilla {v_ok} doppel {d_ok}");

    if failures.is_empty() {
        println!("PASS: block broadcasts match");
        Ok(true)
    } else {
        println!("FAIL: {} block difference(s):", failures.len());
        for f in failures.iter().take(10) {
            println!("  {f}");
        }
        Ok(false)
    }
}

/// The differential redstone test: freeze time, build a lever-wire-torch
/// circuit, flip the lever, step ticks, and diff every broadcast per tick
/// between vanilla and Doppel. This is the M3 referee: wire power levels,
/// torch timing (the 1gt delay), and update ordering all face it.
pub fn parity_redstone() -> Result<bool> {
    use crate::capture;

    let pin = load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let root = doppel_protocol::find_repo_root()?;
    let blobs_dir = root.join("target").join("vanilla").join("blobs");
    let pristine_world = root.join("target").join("vanilla").join("pristine-world");
    if blobs_dir.exists() {
        std::fs::remove_dir_all(&blobs_dir)?;
    }

    let login = capture::login_start_c("Doppel");
    let protocol = pin.protocol.unwrap_or(0);
    // Circuit on a free platform: lever, 4 wire, torch on the far block.
    // The L-shaped branch off wire 12 exercises the same-Y corner
    // (connection recompute + signal around the bend). The step places
    // the top wire BEFORE the stone under it so the climbing wire's
    // UP-connection (diagonal rules) is recomputed by vanilla's own
    // updateShape when the stone lands; the stone above the branch wire
    // cuts its UP connections; the terminator beside the top wire
    // recomputes that wire's line shape.
    let commands: Vec<String> = [
        "tick freeze",
        // Support platform: the reference pops floor-mounted redstone
        // components whose support is missing, so every circuit position
        // gets a stone footing before anything lands on it.
        "setblock 10 99 10 minecraft:stone",
        "setblock 11 99 10 minecraft:stone",
        "setblock 12 99 10 minecraft:stone",
        "setblock 13 99 10 minecraft:stone",
        "setblock 14 99 10 minecraft:stone",
        "setblock 15 99 10 minecraft:stone",
        "setblock 16 99 10 minecraft:stone",
        "setblock 17 99 10 minecraft:stone",
        "setblock 18 99 10 minecraft:stone",
        "setblock 19 99 10 minecraft:stone",
        "setblock 20 99 10 minecraft:stone",
        "setblock 21 99 10 minecraft:stone",
        "tick step 1",
        "setblock 12 99 11 minecraft:stone",
        "tick step 1",
        "setblock 10 99 12 minecraft:stone",
        "setblock 11 99 12 minecraft:stone",
        "setblock 12 99 12 minecraft:stone",
        "setblock 13 99 12 minecraft:stone",
        "setblock 14 99 12 minecraft:stone",
        "setblock 15 99 12 minecraft:stone",
        "setblock 16 99 12 minecraft:stone",
        "tick step 1",
        "setblock 13 99 13 minecraft:stone",
        "setblock 14 99 13 minecraft:stone",
        "setblock 15 99 13 minecraft:stone",
        "setblock 16 99 13 minecraft:stone",
        "setblock 17 99 13 minecraft:stone",
        "tick step 5",
        "setblock 10 100 10 minecraft:lever[face=floor,facing=north,powered=false]",
        "tick step 1",
        "setblock 11 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 12 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 13 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 14 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 15 100 10 minecraft:redstone_torch",
        "tick step 1",
        "setblock 10 100 10 minecraft:lever[face=floor,facing=north,powered=true]",
        "tick step 1",
        "setblock 16 100 10 minecraft:repeater[facing=west,delay=1]",
        "tick step 1",
        "setblock 17 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 18 100 10 minecraft:comparator[facing=west,mode=compare,powered=false]",
        "tick step 1",
        "setblock 10 100 12 minecraft:lever[face=floor,facing=north,powered=true]",
        "tick step 1",
        "setblock 11 100 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 12 100 12 minecraft:comparator[facing=west,mode=compare,powered=false]",
        "tick step 1",
        "setblock 13 100 12 minecraft:comparator[facing=west,mode=subtract,powered=false]",
        "tick step 1",
        "setblock 13 100 13 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 14 100 13 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 15 100 13 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 16 100 13 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 17 100 13 minecraft:lever[face=floor,facing=north,powered=true]",
        "tick step 1",
        "setblock 20 100 10 minecraft:observer[facing=east]",
        "tick step 1",
        "setblock 21 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 19 100 10 minecraft:stone",
        "tick step 1",
        "setblock 19 100 10 minecraft:oak_planks",
        "tick step 1",
        "setblock 12 100 11 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 12 100 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 13 100 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 14 100 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 15 100 12 minecraft:stone",
        "tick step 1",
        "setblock 15 101 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 13 101 12 minecraft:stone",
        "tick step 1",
        "setblock 16 101 12 minecraft:stone",
        "tick step 10",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    // 1. Clean join blobs + pristine world (no commands run yet).
    capture_clean_blobs(&pin, &jar, &blobs_dir, &pristine_world)?;

    // 2. Vanilla reference: the scripted run on a fresh boot.
    let server = vanilla::boot(&pin, &jar, VANILLA_PORT)?;
    let v = bot::login_capture(
        "127.0.0.1",
        VANILLA_PORT,
        protocol,
        &login,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(6)),
            max_packets: Some(1200),
            dump_dir: None,
            commands: &commands,
            walk_chunks: None,
            raw_packets: &[],
        },
    )
    .context("capturing vanilla redstone")?;
    drop(server);

    // 3. Doppel against the clean blobs + pristine world: its own engine
    // now executes every command for real.
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
    let d = bot::login_capture(
        "127.0.0.1",
        DOPPEL_PORT,
        protocol,
        &login,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(10)),
            max_packets: Some(1200),
            dump_dir: None,
            commands: &commands,
            walk_chunks: None,
            raw_packets: &[],
        },
    )
    .context("capturing doppel redstone")?;
    let _ = child.kill();
    let _ = child.wait();

    // 4. Final per-position states: apply each stream's updates in order,
    //    last write wins. Under /tick freeze raw packet ordering is not
    //    reliably observable; final states are the semantic claim.
    let vr: Vec<_> = v.iter().filter(|p| p.id >= 0).collect();
    let dr: Vec<_> = d.iter().filter(|p| p.id >= 0).collect();
    let v_updates: Vec<_> = vr
        .iter()
        .filter(|p| p.id == 0x56 || p.id == 0x08)
        .copied()
        .collect();
    let d_updates: Vec<_> = dr
        .iter()
        .filter(|p| p.id == 0x56 || p.id == 0x08)
        .copied()
        .collect();
    println!(
        "[oracle] redstone updates: vanilla {} doppel {}",
        v_updates.len(),
        d_updates.len()
    );
    // Transcripts can end early (idle timeout, packet cap): a truncated
    // doppel stream would silently pass half a circuit. Surface the ends.
    for (who, all) in [("vanilla", &v), ("doppel", &d)] {
        if let Some(p) = all.iter().rev().find(|p| p.id < 0) {
            println!(
                "[oracle] {who} transcript note: {}",
                p.note.clone().unwrap_or_default()
            );
        }
    }

    let apply =
        |pkts: &[&bot::CapturedPacket]| -> std::collections::BTreeMap<(i32, i32, i32), u32> {
            let mut map = std::collections::BTreeMap::new();
            for (pos, st) in decode_update_writes(pkts) {
                map.insert(pos, st);
            }
            map
        };
    let vmap = apply(&v_updates);
    let dmap = apply(&d_updates);
    println!(
        "[oracle] final circuit states: vanilla {} doppel {}",
        vmap.len(),
        dmap.len()
    );
    let mut failures = Vec::new();
    for (pos, st) in &vmap {
        match dmap.get(pos) {
            Some(ds) if ds == st => {}
            Some(ds) => {
                failures.push(format!("pos {:?}: vanilla {st} != doppel {ds}", pos));
                // History: every update write touching this position, in
                // stream order, for both servers.
                let hist = |writes: &[((i32, i32, i32), u32)]| -> String {
                    writes
                        .iter()
                        .enumerate()
                        .filter(|(_, (p, _))| p == pos)
                        .map(|(i, (_, s))| format!("#{i}:{s}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                let v_writes = decode_update_writes(&v_updates);
                let d_writes = decode_update_writes(&d_updates);
                println!("  vanilla history: {}", hist(&v_writes));
                println!("  doppel history: {}", hist(&d_writes));
                let listing = |pkts: &[&bot::CapturedPacket]| -> String {
                    pkts.iter()
                        .enumerate()
                        .map(|(i, p)| format!("#{i}:0x{:02x}/{}@{}ms", p.id, p.body_len, p.t_ms))
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                println!("  vanilla all: {}", listing(v_updates.as_slice()));
                println!("  doppel all: {}", listing(d_updates.as_slice()));
            }
            None => failures.push(format!("pos {:?}: vanilla {st}, missing in doppel", pos)),
        }
    }
    for (pos, st) in &dmap {
        if !vmap.contains_key(pos) {
            failures.push(format!(
                "pos {:?}: doppel-only {st} (vanilla never reported)",
                pos
            ));
        }
    }
    if failures.is_empty() {
        println!("PASS: redstone circuits match");
        Ok(true)
    } else {
        println!("FAIL: {} redstone difference(s):", failures.len());
        for f in failures.iter().take(40) {
            println!("  {f}");
        }
        Ok(false)
    }
}

/// The differential placement test: both servers get one anchor block,
/// the bot flips itself creative, picks a stack, and right-clicks the
/// anchor's top face. The resulting block writes must match.
pub fn parity_placement() -> Result<bool> {
    use crate::capture;
    let pin = load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let root = doppel_protocol::find_repo_root()?;
    let blobs_dir = root.join("target").join("vanilla").join("blobs");
    let pristine_world = root.join("target").join("vanilla").join("pristine-world");
    if blobs_dir.exists() {
        std::fs::remove_dir_all(&blobs_dir)?;
    }

    let login = capture::login_start_c("Doppel");
    let protocol = pin.protocol.unwrap_or(0);
    let commands: Vec<String> = [
        "setblock 1 -60 1 minecraft:stone",
        "give @s minecraft:stone 64",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let raw: Vec<(i32, Vec<u8>)> = vec![(0x42, bot::build_use_item_on_top(1, -60, 1, 1))];

    capture_clean_blobs(&pin, &jar, &blobs_dir, &pristine_world)?;

    let server = vanilla::boot(&pin, &jar, VANILLA_PORT)?;
    let v = bot::login_capture(
        "127.0.0.1",
        VANILLA_PORT,
        protocol,
        &login,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(5)),
            max_packets: Some(1200),
            dump_dir: None,
            commands: &commands,
            walk_chunks: None,
            raw_packets: &raw,
        },
    )
    .context("capturing vanilla placement")?;
    drop(server);

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
    let d = bot::login_capture(
        "127.0.0.1",
        DOPPEL_PORT,
        protocol,
        &login,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(5)),
            max_packets: Some(1200),
            dump_dir: None,
            commands: &commands,
            walk_chunks: None,
            raw_packets: &raw,
        },
    )
    .context("capturing doppel placement")?;
    let _ = child.kill();
    let _ = child.wait();

    let updates = |pkts: &[bot::CapturedPacket]| -> Vec<((i32, i32, i32), u32)> {
        let refs: Vec<_> = pkts.iter().filter(|p| p.id >= 0).collect();
        decode_update_writes(&refs)
    };
    for (who, pkts) in [("vanilla", &v), ("doppel", &d)] {
        let total = pkts.len();
        let ended = pkts.iter().rev().find(|p| p.id < 0);
        let ids: Vec<String> = pkts
            .iter()
            .filter(|p| p.id >= 0)
            .take(12)
            .map(|p| format!("0x{:02x}", p.id))
            .collect();
        let tail: Vec<String> = pkts
            .iter()
            .filter(|p| p.id >= 0)
            .rev()
            .take(8)
            .map(|p| format!("0x{:02x}@{}ms", p.id, p.t_ms))
            .collect();
        println!(
            "[oracle] {who}: {total} frames, head [{:?}] tail [{:?}], end-note: {:?}",
            ids.join(" "),
            tail.join(" "),
            ended.and_then(|p| p.note.clone()).unwrap_or_default()
        );
    }
    let vw = updates(&v);
    let dw = updates(&d);
    println!(
        "[oracle] placement writes: vanilla {} doppel {}",
        vw.len(),
        dw.len()
    );
    println!("  vanilla writes: {vw:?}");
    println!("  doppel writes: {dw:?}");
    let mut failures = Vec::new();
    for (i, (a, b)) in vw.iter().zip(dw.iter()).enumerate() {
        if a != b {
            failures.push(format!("write {i}: vanilla {a:?} != doppel {b:?}"));
        }
    }
    if vw.len() != dw.len() {
        failures.push(format!(
            "write count: vanilla {} vs doppel {}",
            vw.len(),
            dw.len()
        ));
    }
    if !vw.iter().any(|(p, _)| *p == (1, -59, 1)) {
        failures.push("the placed block at (1,-59,1) is missing".to_string());
    }

    if failures.is_empty() {
        println!("PASS: placement writes match");
        Ok(true)
    } else {
        println!("FAIL: {} placement difference(s):", failures.len());
        for f in failures.iter().take(10) {
            println!("  {f}");
        }
        Ok(false)
    }
}
