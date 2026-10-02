//! Worldgen config pinning: unpacks the vanilla server jar (a bundler
//! wrapper around a nested game jar) and copies the world generation
//! data pack under pins/worldgen/, mirroring the block registry pins.

use anyhow::{bail, Context, Result};
use doppel_protocol::{find_repo_root, load_pin};
use std::io::Read;

/// The worldgen directories worth pinning. The rest of the pack
/// (structures, features, biomes, carvers) stays out until a consumer
/// needs it.
const PINNED_DIRS: [&str; 6] = [
    "noise",
    "noise_settings",
    "density_function",
    "material_rule",
    "material_condition",
    "multi_noise_biome_source_parameter_list",
];

const PACK_PREFIX: &str = "data/minecraft/worldgen/";

pub fn run() -> Result<()> {
    let pin = load_pin()?;
    let jar = crate::vanilla::ensure_jar(&pin)?;
    let outer = std::fs::read(&jar).with_context(|| format!("reading {}", jar.display()))?;
    let entries = read_directory(&outer)?;
    let inner_entry = entries
        .iter()
        .find(|e| e.name.starts_with("META-INF/versions/") && e.name.ends_with(".jar"))
        .context("bundler jar carries no nested game jar")?;
    let inner_name = inner_entry.name.clone();
    println!("[oracle] nested game jar: {inner_name}");
    let inner = read_entry(&outer, inner_entry)?;
    let inner_entries = read_directory(&inner)?;

    let out_root = find_repo_root()?.join("pins").join("worldgen");
    if out_root.exists() {
        std::fs::remove_dir_all(&out_root).context("cleaning stale worldgen pins")?;
    }
    let mut copied = 0usize;
    for entry in inner_entries.iter().filter(|e| wanted(&e.name)) {
        let rel = entry
            .name
            .strip_prefix(PACK_PREFIX)
            .expect("wanted names carry the pack prefix");
        let dest = out_root.join(rel);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let body = read_entry(&inner, entry)?;
        std::fs::write(&dest, &body).with_context(|| format!("writing {}", dest.display()))?;
        copied += 1;
    }
    if copied == 0 {
        bail!("no worldgen configs found in the nested jar");
    }
    println!("[oracle] pinned {copied} worldgen configs under pins/worldgen/");
    Ok(())
}

fn wanted(name: &str) -> bool {
    let Some(rel) = name.strip_prefix(PACK_PREFIX) else {
        return false;
    };
    if !rel.ends_with(".json") {
        return false;
    }
    PINNED_DIRS
        .iter()
        .any(|dir| rel.starts_with(&format!("{dir}/")))
}

/// One central-directory record.
struct ZipEntry {
    name: String,
    method: u16,
    compressed_size: usize,
    uncompressed_size: usize,
    header_offset: usize,
}

fn u16le(data: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(data[at..at + 2].try_into().unwrap())
}

fn u32le(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(data[at..at + 4].try_into().unwrap())
}

fn read_directory(data: &[u8]) -> Result<Vec<ZipEntry>> {
    // The end-of-central-directory record sits in the final 64 KiB; scan
    // backwards for its signature.
    const EOCD_SIG: [u8; 4] = [0x50, 0x4b, 0x05, 0x06];
    const CD_SIG: [u8; 4] = [0x50, 0x4b, 0x01, 0x02];
    let floor = data.len().saturating_sub(65_536 + 22);
    let mut eocd = None;
    let mut at = data.len() - 22;
    loop {
        if data[at..at + 4] == EOCD_SIG {
            eocd = Some(at);
            break;
        }
        if at == floor {
            break;
        }
        at -= 1;
    }
    let eocd = eocd.context("zip has no end-of-central-directory record")?;
    let count = u16le(data, eocd + 10) as usize;
    let mut cursor = u32le(data, eocd + 16) as usize;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        if cursor + 46 > data.len() || data[cursor..cursor + 4] != CD_SIG {
            bail!("corrupt central directory at byte {cursor}");
        }
        let name_len = u16le(data, cursor + 28) as usize;
        let extra_len = u16le(data, cursor + 30) as usize;
        let comment_len = u16le(data, cursor + 32) as usize;
        let name_start = cursor + 46;
        entries.push(ZipEntry {
            name: String::from_utf8_lossy(&data[name_start..name_start + name_len]).into_owned(),
            method: u16le(data, cursor + 10),
            compressed_size: u32le(data, cursor + 20) as usize,
            uncompressed_size: u32le(data, cursor + 24) as usize,
            header_offset: u32le(data, cursor + 42) as usize,
        });
        cursor = name_start + name_len + extra_len + comment_len;
    }
    Ok(entries)
}

fn read_entry(data: &[u8], entry: &ZipEntry) -> Result<Vec<u8>> {
    const LOCAL_SIG: [u8; 4] = [0x50, 0x4b, 0x03, 0x04];
    let at = entry.header_offset;
    if at + 30 > data.len() || data[at..at + 4] != LOCAL_SIG {
        bail!("corrupt local header for {}", entry.name);
    }
    let name_len = u16le(data, at + 26) as usize;
    let extra_len = u16le(data, at + 28) as usize;
    let body = at + 30 + name_len + extra_len;
    let raw = &data[body..body + entry.compressed_size];
    let out = match entry.method {
        0 => raw.to_vec(),
        8 => {
            let mut decoder = flate2::read::DeflateDecoder::new(raw);
            let mut out = Vec::with_capacity(entry.uncompressed_size);
            decoder
                .read_to_end(&mut out)
                .with_context(|| format!("inflating {}", entry.name))?;
            out
        }
        other => bail!("{} uses unsupported zip method {other}", entry.name),
    };
    if out.len() != entry.uncompressed_size {
        bail!(
            "{} inflated to {} bytes, expected {}",
            entry.name,
            out.len(),
            entry.uncompressed_size
        );
    }
    Ok(out)
}
