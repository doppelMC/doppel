//! Boots vanilla and records a login transcript for protocol discovery.
//! The transcript is the spec: whatever packets vanilla 26.3 actually sends
//! during login/configuration is what Doppel will learn to send.

use anyhow::{Context, Result};
use std::path::Path;
use std::time::Duration;

use crate::{bot, vanilla};

const VANILLA_PORT: u16 = 25567;

pub fn run(out_path: &Path) -> Result<()> {
    let pin = doppel_protocol::load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let server = vanilla::boot(&pin, &jar, VANILLA_PORT)?;

    let packets = bot::login_capture(
        "127.0.0.1",
        VANILLA_PORT,
        pin.protocol.unwrap_or(0),
        "Doppel",
        Duration::from_secs(10),
        300,
    )
    .context("capturing login transcript")?;
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
        "[oracle] captured {} packets -> {}",
        packets.len(),
        out_path.display()
    );
    for p in &packets {
        println!(
            "  id=0x{:02x} len={} {}",
            p.id,
            p.body_len,
            p.note.as_deref().unwrap_or("")
        );
    }
    Ok(())
}
