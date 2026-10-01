//! Boots vanilla and records a login transcript for protocol discovery.
//! The transcript is the spec: whatever packets vanilla 26.3 actually sends
//! during login/configuration is what Doppel will learn to send.

use anyhow::{Context, Result};
use std::path::Path;
use std::time::Duration;

use crate::{bot, vanilla};

const VANILLA_PORT: u16 = 25567;

/// The offline-mode profile UUID vanilla derives from a player name:
/// UUIDv3 (MD5) of "OfflinePlayer:<name>".
fn offline_uuid(username: &str) -> [u8; 16] {
    use md5::Digest;
    let mut hash = md5::Md5::digest(format!("OfflinePlayer:{username}").as_bytes());
    hash[6] = (hash[6] & 0x0f) | 0x30; // version 3
    hash[8] = (hash[8] & 0x3f) | 0x80; // RFC 4122 variant
    let mut out = [0u8; 16];
    out.copy_from_slice(&hash);
    out
}

/// Candidate field layouts for serverbound `minecraft:hello` (Login Start).
/// Variant C is the confirmed 26.3 layout (String name + bare UUID); the
/// others remain for regression documentation.
fn login_start_variants(username: &str) -> Vec<(String, Vec<u8>)> {
    let uuid16 = offline_uuid(username);
    let mut v = Vec::new();

    let mut c = Vec::new();
    doppel_protocol::write_string(&mut c, username);
    c.extend_from_slice(&uuid16); // bare UUID, no flag — confirmed layout
    v.push(("C:name+uuid".into(), c));

    let mut a = Vec::new();
    doppel_protocol::write_string(&mut a, username);
    a.push(0x00);
    v.push(("A:name+opt_uuid(false)".into(), a));

    v
}

pub fn run(out_path: &Path) -> Result<()> {
    let pin = doppel_protocol::load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let server = vanilla::boot(&pin, &jar, VANILLA_PORT)?;

    let variants = login_start_variants("Doppel");
    let mut lines: Vec<String> = Vec::new();
    for (label, body) in &variants {
        let packets = bot::login_capture(
            "127.0.0.1",
            VANILLA_PORT,
            pin.protocol.unwrap_or(0),
            body,
            Duration::from_secs(5),
            120,
        )
        .with_context(|| format!("capturing variant {label}"))?;
        println!("[oracle] variant {label}: {} packets", packets.len());
        for p in &packets {
            let mut obj = serde_json::to_value(p)?;
            obj["variant"] = serde_json::json!(label);
            lines.push(obj.to_string());
        }
    }
    drop(server); // teardown

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = lines.join("\n") + "\n";
    std::fs::write(out_path, &text).with_context(|| format!("writing {}", out_path.display()))?;
    println!(
        "[oracle] captured {} packets across {} variants -> {}",
        lines.len(),
        variants.len(),
        out_path.display()
    );
    Ok(())
}
