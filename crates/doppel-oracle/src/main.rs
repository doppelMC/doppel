//! doppel-oracle: the differential test harness.

mod bot;
mod capture;
mod parity;
mod parity_break;
mod parity_survival;
mod parity_worldgen;
mod registry;
mod scenario;
mod vanilla;

use anyhow::{Context, Result};
use doppel_protocol::{save_pin, Pin};

const MANIFEST_URL: &str = "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json";

fn usage() -> ! {
    eprintln!(
        "usage:
  doppel-oracle pin                   write pins/version.json for the latest release
  doppel-oracle status <host> <port>  ping any server and print its status JSON
  doppel-oracle parity-status         differential test: vanilla oracle vs doppel
                                       (set DOPPEL_BIN=<path> to override the binary)
  doppel-oracle parity-worldgen       boot vanilla at a pinned seed with normal
                                       terrain, capture the spawn chunks, and
                                       structurally diff them against the
                                       seeded terrain generator
  doppel-oracle parity-survival     differential test: drops, pickup, grass
                                       decay/spread (break + random ticks)
  doppel-oracle capture-vanilla-login [out.jsonl] [blobs-dir]
                                      record vanilla's login transcript; with a
                                      blobs dir, dump byte-exact packet bodies
                                      plus manifest.json for replay"
    );
    std::process::exit(2);
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[oracle] error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("pin") => cmd_pin()?,
        Some("status") => {
            let host = args.get(1).unwrap_or_else(|| usage()).clone();
            let port: u16 = args
                .get(2)
                .unwrap_or_else(|| usage())
                .parse()
                .unwrap_or_else(|_| usage());
            cmd_status(&host, port)?
        }
        Some("parity-status") => {
            if !parity::parity_status(None)? {
                std::process::exit(1);
            }
        }
        Some("parity-placement") => {
            if !parity::parity_placement()? {
                std::process::exit(1);
            }
        }
        Some("parity-break") => {
            if !parity_break::parity_break()? {
                std::process::exit(1);
            }
        }
        Some("parity-survival") => {
            if !parity_survival::parity_survival()? {
                std::process::exit(1);
            }
        }
        Some("parity-worldgen") => {
            if !parity_worldgen::run()? {
                std::process::exit(1);
            }
        }
        Some("parity-redstone") => {
            if !parity::parity_redstone()? {
                std::process::exit(1);
            }
        }
        Some("parity-blocks") => {
            if !parity::parity_blocks()? {
                std::process::exit(1);
            }
        }
        Some("parity-walk") => {
            if !parity::parity_walk()? {
                std::process::exit(1);
            }
        }
        Some("parity-login") => {
            if !parity::parity_login()? {
                std::process::exit(1);
            }
        }
        Some("capture-vanilla-login") => {
            let out = args
                .get(1)
                .cloned()
                .unwrap_or_else(|| "captures/vanilla-login.jsonl".into());
            let blobs = args.get(2).cloned();
            capture::run(
                std::path::Path::new(&out),
                blobs.as_deref().map(std::path::Path::new),
            )?
        }
        Some("registry") => registry::run()?,
        Some("scenario") => {
            let out = args
                .get(1)
                .cloned()
                .unwrap_or_else(|| "captures/scenario.jsonl".into());
            scenario::run(std::path::Path::new(&out))?
        }
        _ => usage(),
    };
    Ok(())
}

/// Resolves the latest Mojang release and (re)writes the version pin.
fn cmd_pin() -> Result<()> {
    let manifest: serde_json::Value = fetch_json(MANIFEST_URL)?;
    let latest = manifest["latest"]["release"]
        .as_str()
        .context("manifest has no latest.release")?
        .to_string();
    let version_url = manifest["versions"]
        .as_array()
        .context("manifest has no versions")?
        .iter()
        .find(|v| v["id"].as_str() == Some(latest.as_str()))
        .with_context(|| format!("release {latest} not found in manifest"))?["url"]
        .as_str()
        .context("version entry has no url")?
        .to_string();

    let vjson = fetch_json(&version_url)?;
    let server = &vjson["downloads"]["server"];
    let mut pin = Pin {
        id: vjson["id"].as_str().context("no id")?.to_string(),
        release_time: vjson["releaseTime"]
            .as_str()
            .context("no releaseTime")?
            .to_string(),
        server_jar_url: server["url"].as_str().context("no server url")?.to_string(),
        server_jar_sha1: server["sha1"]
            .as_str()
            .context("no server sha1")?
            .to_string(),
        java_major: vjson["javaVersion"]["majorVersion"].as_u64().unwrap_or(21) as u32,
        // The protocol number is intentionally discovered from the running
        // oracle by `parity-status`, not trusted from metadata.
        protocol: None,
        version_name: None,
    };
    // Repinning the SAME version must keep the oracle-healed values —
    // wiping them would un-teach the capture bot the protocol number.
    if let Ok(prev) = doppel_protocol::load_pin() {
        if prev.id == pin.id {
            pin.protocol = prev.protocol;
            pin.version_name = prev.version_name;
        }
    }
    save_pin(&pin)?;
    println!(
        "[oracle] pinned {} (java {}+, jar sha1 {})",
        pin.id,
        pin.java_major,
        &pin.server_jar_sha1[..8]
    );
    Ok(())
}

fn cmd_status(host: &str, port: u16) -> Result<()> {
    let status = bot::status_ping(host, port, 0, std::time::Duration::from_secs(10))?;
    println!("{}", serde_json::to_string_pretty(&status)?);
    Ok(())
}

fn fetch_json(url: &str) -> Result<serde_json::Value> {
    let resp = ureq::get(url)
        .timeout(std::time::Duration::from_secs(30))
        .call()
        .with_context(|| format!("fetching {url}"))?;
    let text = resp
        .into_string()
        .with_context(|| format!("reading body of {url}"))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {url}"))
}
