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
pub fn login_start_c(username: &str) -> Vec<u8> {
    let mut c = Vec::new();
    doppel_protocol::write_string(&mut c, username);
    c.extend_from_slice(&offline_uuid(username));
    c
}

/// Write manifest.json mapping dumped packets to phase + id, so Doppel can
/// replay them in choreographic order. Phase flips after the server's
/// finish_configuration.
pub fn write_manifest(packets: &[bot::CapturedPacket], dir: &std::path::Path) -> Result<usize> {
    let mut play = false;
    let mut manifest: Vec<serde_json::Value> = Vec::new();
    for p in packets {
        // The finish_configuration packet itself stays in the config phase —
        // Doppel sends it natively. The phase flips for everything AFTER it.
        if let Some(file) = &p.file {
            manifest.push(serde_json::json!({
                "file": file,
                "id": p.id,
                "phase": if play { "play" } else { "config" },
            }));
        }
        if p.note
            .as_deref()
            .is_some_and(|n| n.contains("finish configuration"))
        {
            play = true;
        }
    }
    let path = dir.join("manifest.json");
    std::fs::write(&path, serde_json::to_string_pretty(&manifest)? + "\n")?;
    Ok(manifest.len())
}

pub fn run(out_path: &Path, blobs_dir: Option<&Path>) -> Result<()> {
    let pin = doppel_protocol::load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let server = vanilla::boot(&pin, &jar, VANILLA_PORT)?;

    let login_body = login_start_c("Doppel");
    let mut lines: Vec<String> = Vec::new();
    let packets = bot::login_capture(
        "127.0.0.1",
        VANILLA_PORT,
        pin.protocol.unwrap_or(0),
        &login_body,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(22)),
            max_packets: Some(220),
            dump_dir: blobs_dir,
            commands: &[],
            walk_chunks: None,
        },
    )
    .context("capturing full login transcript")?;
    println!("[oracle] full transcript: {} packets", packets.len());
    for p in &packets {
        let mut obj = serde_json::to_value(p)?;
        obj["variant"] = serde_json::json!("full");
        lines.push(obj.to_string());
    }
    drop(server); // teardown

    if let Some(dir) = blobs_dir {
        let n = write_manifest(&packets, dir)?;
        println!("[oracle] {n} blob entries -> {}", dir.display());
    }

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let text = lines.join("\n") + "\n";
    std::fs::write(out_path, &text).with_context(|| format!("writing {}", out_path.display()))?;
    println!(
        "[oracle] captured {} packets (login through play) -> {}",
        lines.len(),
        out_path.display()
    );
    Ok(())
}
