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
/// Variant C is the confirmed 26.3 layout (String name + bare UUID).
fn login_start_c(username: &str) -> Vec<u8> {
    let mut c = Vec::new();
    doppel_protocol::write_string(&mut c, username);
    c.extend_from_slice(&offline_uuid(username));
    c
}

pub fn run(out_path: &Path) -> Result<()> {
    let pin = doppel_protocol::load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let server = vanilla::boot(&pin, &jar, VANILLA_PORT)?;

    // Probe the serverbound configuration state: same Client Information
    // body sent under each candidate packet id on a fresh connection.
    // A valid id either continues the handshake or errors with the packet's
    // name; an invalid id closes the connection silently.
    let login_body = login_start_c("Doppel");
    let info = bot::client_information_body();
    let mut variants: Vec<(String, Vec<u8>, (i32, Vec<u8>))> = (0x00..=0x0a)
        .map(|id| {
            (
                format!("cfg0x{id:02x}"),
                login_body.clone(),
                (id, info.clone()),
            )
        })
        .collect();
    // Control: no config probe at all (documented silent wait).
    variants.push(("no-probe".into(), login_body.clone(), (0x63, info.clone())));

    let mut lines: Vec<String> = Vec::new();
    for (label, login, probe) in &variants {
        let packets = bot::login_capture(
            "127.0.0.1",
            VANILLA_PORT,
            pin.protocol.unwrap_or(0),
            login,
            probe.clone(),
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
