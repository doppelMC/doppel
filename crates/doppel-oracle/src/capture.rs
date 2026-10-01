//! Boots vanilla and records a login transcript for protocol discovery.
//! The transcript is the spec: whatever packets vanilla 26.3 actually sends
//! during login/configuration is what Doppel will learn to send.

use anyhow::{Context, Result};
use std::path::Path;
use std::time::Duration;

use crate::{bot, vanilla};

const VANILLA_PORT: u16 = 25567;

/// Candidate field layouts for serverbound `minecraft:hello` (Login Start).
/// We do not yet know 26.3's exact encoding, so the capture probes each
/// variant on a fresh connection and records which one vanilla accepts.
fn login_start_variants(username: &str) -> Vec<(String, Vec<u8>)> {
    let uuid16 = [0u8; 16];
    let mut v = Vec::new();

    let mut a = Vec::new();
    doppel_protocol::write_string(&mut a, username);
    a.push(0x00); // optional UUID absent
    v.push(("A:name+opt_uuid(false)".into(), a));

    let mut b = Vec::new();
    doppel_protocol::write_string(&mut b, username);
    b.push(0x01); // optional UUID present
    b.extend_from_slice(&uuid16);
    v.push(("B:name+opt_uuid(true)+uuid".into(), b));

    let mut c = Vec::new();
    doppel_protocol::write_string(&mut c, username);
    c.extend_from_slice(&uuid16); // bare UUID, no flag
    v.push(("C:name+uuid".into(), c));

    let mut d = Vec::new();
    doppel_protocol::write_string(&mut d, username);
    v.push(("D:name_only".into(), d));

    let mut e = Vec::new();
    doppel_protocol::write_string(&mut e, username);
    e.push(0x00);
    doppel_protocol::write_varint(&mut e, 0); // maybe a trailing intent varint
    v.push(("E:name+opt_uuid(false)+varint0".into(), e));

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
