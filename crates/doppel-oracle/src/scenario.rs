//! Scenario harness seed: boots vanilla with an opped bot, runs a scripted
//! command sequence after the join, and captures the responses. This grows
//! into the tick-stepped differential harness (/tick freeze + step, block
//! state queries) that M2/M3 parity is measured with.

use anyhow::{Context, Result};
use std::path::Path;
use std::time::Duration;

use crate::{bot, vanilla};

const VANILLA_PORT: u16 = 25567;

/// The first scenario: freeze the tick, place a block far above the world,
/// read its state back, step ticks, read again. The transcript shows which
/// clientbound packets carry command output — the harness's query channel.
pub fn run(out_path: &Path) -> Result<()> {
    let pin = doppel_protocol::load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let server = vanilla::boot(&pin, &jar, VANILLA_PORT)?;

    let commands = vec![
        "/help".to_string(),
        "/tick freeze".to_string(),
        "/setblock 0 100 0 minecraft:stone".to_string(),
        "/data get block 0 100 0".to_string(),
        "/tick step 5".to_string(),
        "/data get block 0 100 0".to_string(),
    ];
    let login = crate::capture::login_start_c("Doppel");
    let packets = bot::login_capture(
        "127.0.0.1",
        VANILLA_PORT,
        pin.protocol.unwrap_or(0),
        &login,
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(12)),
            max_packets: Some(600),
            commands: &commands,
            walk_chunks: Some(8),
            ..Default::default()
        },
    )
    .context("running scenario")?;
    drop(server); // teardown

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut text = String::new();
    for p in &packets {
        text.push_str(&serde_json::to_string(p)?);
        text.push('\n');
    }
    std::fs::write(out_path, &text).with_context(|| format!("writing {}", out_path.display()))?;

    println!(
        "[oracle] scenario: {} packets -> {}",
        packets.len(),
        out_path.display()
    );
    for p in &packets {
        if p.id < 0 {
            println!("  END: {}", p.note.as_deref().unwrap_or(""));
        } else if p.note.is_some() || p.body_len < 300 {
            let raw = hex::decode(&p.head_hex).unwrap_or_default();
            let preview = raw
                .iter()
                .map(|&b| {
                    if (0x20..0x7f).contains(&b) {
                        b as char
                    } else {
                        '.'
                    }
                })
                .collect::<String>();
            println!(
                "  t={} id=0x{:02x} len={} {} {}",
                p.t_ms,
                p.id,
                p.body_len,
                p.note.as_deref().unwrap_or(""),
                &preview[..preview.len().min(90)]
            );
        }
    }
    Ok(())
}
