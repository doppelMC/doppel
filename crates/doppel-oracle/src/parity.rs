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
        },
    )
    .context("capturing vanilla transcript")?;
    drop(server);
    let entries = capture::write_manifest(&v, &blobs_dir)?;
    anyhow::ensure!(entries > 0, "no blobs captured from vanilla");

    // 2. Spawn Doppel with the blobs.
    let bin = default_doppel_bin()?;
    let pin_path = doppel_protocol::pin_path()?;
    let mut doppel_child = Command::new(&bin)
        .env("DOPPEL_ADDR", "127.0.0.1")
        .env("DOPPEL_PORT", DOPPEL_PORT.to_string())
        .env("DOPPEL_PIN", &pin_path)
        .env("DOPPEL_BLOBS", &blobs_dir)
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
