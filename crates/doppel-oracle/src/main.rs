//! doppel-oracle: the differential test harness.

mod bot;
mod parity;
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
                                       (set DOPPEL_BIN=<path> to override the binary)"
    );
    std::process::exit(2);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("pin") => cmd_pin(),
        Some("status") => {
            let host = args.get(1).unwrap_or_else(|| usage()).clone();
            let port: u16 = args
                .get(2)
                .unwrap_or_else(|| usage())
                .parse()
                .unwrap_or_else(|_| usage());
            cmd_status(&host, port)
        }
        Some("parity-status") => parity::parity_status(None).map(|_| ()),
        _ => usage(),
    };
    if let Err(e) = result {
        eprintln!("[oracle] error: {e:#}");
        std::process::exit(1);
    }
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
    let pin = Pin {
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
