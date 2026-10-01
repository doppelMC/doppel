//! Anvil world loading. Region files (.mca) hold 32x32 chunks each: an 8 KiB
//! header (1024 location entries of sector offset/count, then timestamps),
//! followed by 4 KiB sectors of length-prefixed, zlib-compressed NBT.
//!
//! This crate is storage only — network chunk serialization lives in the
//! server crate once the M2 wire format is pinned against the oracle.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::Path;

pub mod chunk_codec;

pub use chunk_codec::WireChunk;

/// Minecraft protocol VarInt writer (shared with the codec module).
pub fn write_varint(buf: &mut Vec<u8>, value: i32) {
    let mut v = value as u32;
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if v == 0 {
            break;
        }
    }
}

/// A parsed Anvil chunk section.
#[derive(Debug, Deserialize, Serialize)]
pub struct Section {
    #[serde(rename = "Y")]
    pub y: i8,
    #[serde(rename = "block_states", default)]
    pub block_states: Option<BlockStates>,
}

/// Paletted block state storage for one section.
#[derive(Debug, Deserialize, Serialize)]
pub struct BlockStates {
    pub palette: Vec<PaletteEntry>,
    #[serde(default)]
    pub data: Option<Vec<i64>>,
}

/// One palette entry: a block type, plus optional state properties.
#[derive(Debug, Deserialize, Serialize)]
pub struct PaletteEntry {
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Properties", default)]
    pub properties: Option<fastnbt::Value>,
}

/// A parsed Anvil chunk (fields kept to what M2 needs so far).
#[derive(Debug, Deserialize, Serialize)]
pub struct Chunk {
    #[serde(rename = "DataVersion")]
    pub data_version: i32,
    #[serde(rename = "xPos")]
    pub x: i32,
    #[serde(rename = "zPos")]
    pub z: i32,
    #[serde(rename = "Status")]
    pub status: String,
    #[serde(rename = "sections", default)]
    pub sections: Vec<Section>,
}

/// An open region file.
pub struct Region {
    data: Vec<u8>,
}

const SECTOR: usize = 4096;

impl Region {
    pub fn open(path: &Path) -> Result<Region> {
        let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        if data.len() < 2 * SECTOR {
            bail!("region file too small: {} bytes", data.len());
        }
        Ok(Region { data })
    }

    /// Raw (still compressed) chunk bytes at in-region coordinates (0..32).
    fn locate(&self, x: usize, z: usize) -> Option<(usize, usize)> {
        let idx = (x & 31) + ((z & 31) << 5);
        let entry = u32::from_be_bytes([
            self.data[idx * 4],
            self.data[idx * 4 + 1],
            self.data[idx * 4 + 2],
            self.data[idx * 4 + 3],
        ]);
        let offset = (entry >> 8) as usize;
        let count = (entry & 0xff) as usize;
        if offset == 0 || count == 0 {
            return None; // not generated
        }
        Some((offset * SECTOR, count * SECTOR))
    }

    /// Decompressed NBT bytes for the chunk at in-region coordinates.
    pub fn chunk_nbt(&self, x: usize, z: usize) -> Result<Option<Vec<u8>>> {
        let (start, _) = match self.locate(x, z) {
            Some(v) => v,
            None => return Ok(None), // chunk not generated
        };
        if start + 5 > self.data.len() {
            bail!("chunk header out of bounds");
        }
        let len = u32::from_be_bytes([
            self.data[start],
            self.data[start + 1],
            self.data[start + 2],
            self.data[start + 3],
        ]) as usize;
        let compression = self.data[start + 4];
        let body = &self.data[start + 5..(start + 4 + len).min(self.data.len())];
        match compression {
            1 => {
                let mut out = Vec::new();
                flate2::read::GzDecoder::new(body)
                    .read_to_end(&mut out)
                    .context("decompressing chunk (gzip)")?;
                Ok(Some(out))
            }
            2 => {
                let mut out = Vec::new();
                flate2::read::ZlibDecoder::new(body)
                    .read_to_end(&mut out)
                    .context("decompressing chunk (zlib)")?;
                Ok(Some(out))
            }
            3 => Ok(Some(body.to_vec())),
            other => bail!("unknown chunk compression type {other}"),
        }
    }

    /// Parsed chunk at in-region coordinates.
    pub fn chunk(&self, x: usize, z: usize) -> Result<Option<Chunk>> {
        match self.chunk_nbt(x, z)? {
            None => Ok(None),
            Some(nbt) => {
                let chunk: Chunk = fastnbt::from_bytes(&nbt)
                    .with_context(|| format!("parsing chunk ({x},{z}) NBT"))?;
                Ok(Some(chunk))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Builds a one-chunk region file in memory and reads it back.
    #[test]
    fn roundtrip_minimal_region() {
        // Minimal chunk NBT: root compound with the required fields.
        let chunk = Chunk {
            data_version: 4189,
            x: 3,
            z: 7,
            status: "minecraft:full".into(),
            sections: vec![Section {
                y: -1,
                block_states: Some(BlockStates {
                    palette: vec![PaletteEntry {
                        name: "minecraft:air".into(),
                        properties: None,
                    }],
                    data: None,
                }),
            }],
        };
        let nbt = fastnbt::to_bytes(&chunk).expect("serialize chunk nbt");
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&nbt).unwrap();
        let zipped = enc.finish().unwrap();

        // Sector: 4-byte length (incl. compression byte) + type 2 + zlib NBT.
        let mut sector = Vec::new();
        sector.extend_from_slice(&((zipped.len() + 1) as u32).to_be_bytes());
        sector.push(2);
        sector.extend_from_slice(&zipped);
        sector.resize(SECTOR, 0);

        let mut file = vec![0u8; 2 * SECTOR];
        // Location for chunk (3,7): idx = 3 + 7*32 = 227 -> offset sector 2, 1 sector.
        let idx = 3 + 7 * 32;
        let entry: u32 = (2 << 8) | 1;
        file[idx * 4..idx * 4 + 4].copy_from_slice(&entry.to_be_bytes());
        file.extend_from_slice(&sector);

        let dir = std::env::temp_dir().join("doppel-world-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("r.0.0.mca");
        std::fs::write(&path, &file).unwrap();

        let region = Region::open(&path).unwrap();
        let parsed = region.chunk(3, 7).unwrap().expect("chunk present");
        assert_eq!(parsed.x, 3);
        assert_eq!(parsed.z, 7);
        assert_eq!(parsed.status, "minecraft:full");
        assert_eq!(parsed.sections.len(), 1);
        assert_eq!(
            parsed.sections[0].block_states.as_ref().unwrap().palette[0].name,
            "minecraft:air"
        );
        // Missing chunk reads as None, not an error.
        assert!(region.chunk(0, 0).unwrap().is_none());
    }
}
