//! Replay blobs captured from the vanilla oracle: byte-exact packet bodies
//! for the data-heavy configuration and play packets (registry data, update
//! tags, the join packet, spawn chunks). Trivial packets are built natively
//! in `lib.rs`; these carry the content that vanilla generates from its jar
//! at runtime, so Doppel loads them from a capture instead of embedding
//! anything in the repository.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct ManifestEntry {
    file: String,
    id: i32,
    phase: String,
}

pub struct Blobs {
    /// Configuration-state registry_data bodies, in vanilla's send order.
    pub registries: Vec<Vec<u8>>,
    /// Configuration-state update_tags body.
    pub update_tags: Option<Vec<u8>>,
    /// Play-state join packet (id + body).
    pub join: Option<(i32, Vec<u8>)>,
    /// Play-state chunk batch: start marker, chunks, finished marker.
    pub batch_start: Option<(i32, Vec<u8>)>,
    pub chunks: Vec<(i32, Vec<u8>)>,
    pub batch_finished: Option<(i32, Vec<u8>)>,
}

pub fn load(dir: &Path) -> Result<Blobs> {
    let manifest_path = dir.join("manifest.json");
    let manifest: Vec<ManifestEntry> = serde_json::from_str(
        &std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("reading {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parsing {}", manifest_path.display()))?;

    let mut blobs = Blobs {
        registries: Vec::new(),
        update_tags: None,
        join: None,
        batch_start: None,
        chunks: Vec::new(),
        batch_finished: None,
    };
    for entry in &manifest {
        let body = std::fs::read(dir.join(&entry.file))
            .with_context(|| format!("reading blob {}", entry.file))?;
        match (entry.phase.as_str(), entry.id) {
            ("config", 0x07) => blobs.registries.push(body),
            ("config", 0x0e) => blobs.update_tags = Some(body),
            ("play", 0x32) => blobs.join = Some((entry.id, body)),
            ("play", 0x0c) => blobs.batch_start = Some((entry.id, body)),
            ("play", 0x2e) => blobs.chunks.push((entry.id, body)),
            ("play", 0x0b) => blobs.batch_finished = Some((entry.id, body)),
            // Everything else (time updates, entity traffic after the join
            // burst) is live content, not replayed.
            _ => {}
        }
    }
    Ok(blobs)
}

/// The offline-mode profile UUID vanilla derives from a player name:
/// UUIDv3 (MD5) of "OfflinePlayer:<name>".
pub fn offline_uuid(username: &str) -> [u8; 16] {
    use md5::Digest;
    let mut hash = md5::Md5::digest(format!("OfflinePlayer:{username}").as_bytes());
    hash[6] = (hash[6] & 0x0f) | 0x30; // version 3
    hash[8] = (hash[8] & 0x3f) | 0x80; // RFC 4122 variant
    let mut out = [0u8; 16];
    out.copy_from_slice(&hash);
    out
}

/// A per-connection session UUID for login_finished. Vanilla generates a
/// random one; ours is time-and-counter derived, v4-shaped.
pub fn session_uuid() -> [u8; 16] {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&nanos.to_be_bytes());
    out[8..].copy_from_slice(&count.to_be_bytes());
    out[6] = (out[6] & 0x0f) | 0x40; // version 4
    out[8] = (out[8] & 0x3f) | 0x80; // RFC 4122 variant
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_uuid_matches_vanilla_ground_truth() {
        // Observed on the wire: vanilla echoed this UUID back for the
        // username "Doppel" in login_finished.
        let expected = hex("97e9cb14470c3c15a9762b16dcd2e827");
        assert_eq!(offline_uuid("Doppel"), expected);
    }

    fn hex(s: &str) -> [u8; 16] {
        let mut out = [0u8; 16];
        for (i, b) in out.iter_mut().enumerate() {
            *b = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap();
        }
        out
    }
}
