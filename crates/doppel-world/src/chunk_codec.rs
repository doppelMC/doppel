//! Network chunk codec: byte-exact decode/encode of `level_chunk_with_light`
//! bodies per the 26.x wire layout.
//!
//! Layout: x i32, z i32 | heightmaps (VarInt count, per map VarInt type +
//! VarInt longCount + BE longs) | VarInt-length byte array of concatenated
//! sections (i16 nonEmpty, i16 fluid, block-state container, biome container)
//! | block entities (VarInt count, per entry i8 packedXZ + i16 y + VarInt
//! type + nullable NBT) | light (4 BitSets + 2 arrays of length-prefixed
//! byte arrays).
//!
//! Containers keep their raw storage longs: decode -> encode is therefore
//! byte-faithful by construction, which the parity harness proves live by
//! replaying chunks THROUGH this codec.

use anyhow::{bail, Context, Result};

/// A paletted container exactly as it appears on the wire.
#[derive(Debug, Clone, PartialEq)]
pub enum Container {
    /// bits == 0: single value, no storage longs.
    Single(u32),
    /// bits 1..=8 (block states) or 1..=3 (biomes): palette list + storage.
    Palette {
        bits: u8,
        entries: Vec<u32>,
        longs: Vec<u64>,
    },
    /// bits >= 9: direct global ids, no palette list.
    Global { bits: u8, longs: Vec<u64> },
}

/// Per-container-kind palette rules: blocks use indirect palettes at 4..=8
/// bits then direct at >=9; biomes use 1..=3 then direct at >=4 (64 entries).
#[derive(Clone, Copy)]
pub enum ContainerKind {
    Blocks,
    Biomes,
}

impl ContainerKind {
    fn max_indirect_bits(self) -> u8 {
        match self {
            ContainerKind::Blocks => 8,
            ContainerKind::Biomes => 3,
        }
    }

    /// Hard sanity cap on the bits byte — hostile values above this would
    /// otherwise make 64/bits zero and panic (div-by-zero on remote input).
    fn max_bits(self) -> u8 {
        match self {
            ContainerKind::Blocks => 16,
            ContainerKind::Biomes => 8,
        }
    }
}

impl Container {
    fn decode(r: &mut Reader, entry_count: usize, kind: ContainerKind) -> Result<Container> {
        let bits = r.read_u8().context("container bits")?;
        if bits > kind.max_bits() {
            bail!("container bits {bits} out of range");
        }
        match bits {
            0 => Ok(Container::Single(
                r.read_varint().context("single value")? as u32
            )),
            1..=8 if bits <= kind.max_indirect_bits() => {
                let size = r.read_varint().context("palette size")? as usize;
                if size > 65536 {
                    bail!("palette size {size} out of range");
                }
                let mut entries = Vec::with_capacity(size);
                for _ in 0..size {
                    entries.push(r.read_varint().context("palette entry")? as u32);
                }
                let values_per_long = 64 / bits as usize;
                let longs = expected_longs(entry_count, values_per_long);
                let mut storage = Vec::with_capacity(longs);
                for _ in 0..longs {
                    storage.push(r.read_u64().context("storage long")?);
                }
                Ok(Container::Palette {
                    bits,
                    entries,
                    longs: storage,
                })
            }
            _ => {
                let values_per_long = 64 / bits as usize;
                let longs = expected_longs(entry_count, values_per_long);
                let mut storage = Vec::with_capacity(longs);
                for _ in 0..longs {
                    storage.push(r.read_u64().context("storage long")?);
                }
                Ok(Container::Global {
                    bits,
                    longs: storage,
                })
            }
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Container::Single(v) => {
                out.push(0);
                crate::write_varint(out, *v as i32);
            }
            Container::Palette {
                bits,
                entries,
                longs,
            } => {
                out.push(*bits);
                crate::write_varint(out, entries.len() as i32);
                for e in entries {
                    crate::write_varint(out, *e as i32);
                }
                for l in longs {
                    out.extend_from_slice(&l.to_be_bytes());
                }
            }
            Container::Global { bits, longs } => {
                out.push(*bits);
                for l in longs {
                    out.extend_from_slice(&l.to_be_bytes());
                }
            }
        }
    }
}

fn expected_longs(entry_count: usize, values_per_long: usize) -> usize {
    entry_count.div_ceil(values_per_long)
}

/// One chunk section from the concatenated data array.
#[derive(Debug, Clone, PartialEq)]
pub struct WireSection {
    pub non_empty: i16,
    pub fluid: i16,
    pub block_states: Container,
    pub biomes: Container,
}

/// One block entity entry (NBT kept as raw bytes for byte-faithful relay).
#[derive(Debug, Clone, PartialEq)]
pub struct WireBlockEntity {
    pub packed_xz: u8,
    pub y: i16,
    pub ty: u32,
    pub tag: Option<Vec<u8>>,
}

/// The light payload of a chunk packet. Masks use `ByteBufCodecs.BIT_SET`:
/// VarInt byte-length + `BitSet.toByteArray()` (little-endian bit order) —
/// NOT the long-count encoding used elsewhere.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WireLight {
    pub sky_mask: Vec<u8>,
    pub block_mask: Vec<u8>,
    pub empty_sky_mask: Vec<u8>,
    pub empty_block_mask: Vec<u8>,
    pub sky_updates: Vec<Vec<u8>>,
    pub block_updates: Vec<Vec<u8>>,
}

/// A decoded chunk packet body.
#[derive(Debug, Clone, PartialEq)]
pub struct WireChunk {
    pub x: i32,
    pub z: i32,
    pub heightmaps: Vec<(u32, Vec<u64>)>,
    pub sections: Vec<WireSection>,
    pub block_entities: Vec<WireBlockEntity>,
    pub light: WireLight,
}

impl WireChunk {
    pub fn decode(body: &[u8]) -> Result<WireChunk> {
        let mut r = Reader::new(body);
        let x = r.read_i32().context("chunk x")?;
        let z = r.read_i32().context("chunk z")?;

        let map_count = r.read_varint().context("heightmap count")? as usize;
        let mut heightmaps = Vec::with_capacity(map_count);
        for _ in 0..map_count {
            let ty = r.read_varint().context("heightmap type")? as u32;
            let longs = r.read_varint().context("heightmap longs")? as usize;
            let mut values = Vec::with_capacity(longs);
            for _ in 0..longs {
                values.push(r.read_u64().context("heightmap long")?);
            }
            heightmaps.push((ty, values));
        }

        let data_len = r.read_varint().context("data array length")? as usize;
        if data_len > 4 * 1024 * 1024 {
            bail!("chunk data array too large: {data_len}");
        }
        let data = r.read_bytes(data_len).context("data array")?;
        let mut sr = Reader::new(&data);
        let mut sections = Vec::new();
        while sr.remaining() > 0 {
            let non_empty = sr.read_i16().context("nonEmpty count")?;
            let fluid = sr.read_i16().context("fluid count")?;
            let block_states =
                Container::decode(&mut sr, 4096, ContainerKind::Blocks).context("block states")?;
            let biomes = Container::decode(&mut sr, 64, ContainerKind::Biomes).context("biomes")?;
            sections.push(WireSection {
                non_empty,
                fluid,
                block_states,
                biomes,
            });
        }

        let be_count = r.read_varint().context("block entity count")? as usize;
        let mut block_entities = Vec::with_capacity(be_count);
        for _ in 0..be_count {
            let packed_xz = r.read_u8().context("packed xz")?;
            let y = r.read_i16().context("block entity y")?;
            let ty = r.read_varint().context("block entity type")? as u32;
            let tag = read_nullable_nbt(&mut r)?;
            block_entities.push(WireBlockEntity {
                packed_xz,
                y,
                ty,
                tag,
            });
        }

        let sky_mask = read_bitset(&mut r).context("sky mask")?;
        let block_mask = read_bitset(&mut r).context("block mask")?;
        let empty_sky_mask = read_bitset(&mut r).context("empty sky mask")?;
        let empty_block_mask = read_bitset(&mut r).context("empty block mask")?;
        let sky_updates = read_light_arrays(&mut r).context("sky updates")?;
        let block_updates = read_light_arrays(&mut r).context("block updates")?;

        Ok(WireChunk {
            x,
            z,
            heightmaps,
            sections,
            block_entities,
            light: WireLight {
                sky_mask,
                block_mask,
                empty_sky_mask,
                empty_block_mask,
                sky_updates,
                block_updates,
            },
        })
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4096);
        out.extend_from_slice(&self.x.to_be_bytes());
        out.extend_from_slice(&self.z.to_be_bytes());

        crate::write_varint(&mut out, self.heightmaps.len() as i32);
        for (ty, longs) in &self.heightmaps {
            crate::write_varint(&mut out, *ty as i32);
            crate::write_varint(&mut out, longs.len() as i32);
            for l in longs {
                out.extend_from_slice(&l.to_be_bytes());
            }
        }

        let mut data = Vec::with_capacity(2048);
        for s in &self.sections {
            data.extend_from_slice(&s.non_empty.to_be_bytes());
            data.extend_from_slice(&s.fluid.to_be_bytes());
            s.block_states.encode(&mut data);
            s.biomes.encode(&mut data);
        }
        crate::write_varint(&mut out, data.len() as i32);
        out.extend_from_slice(&data);

        crate::write_varint(&mut out, self.block_entities.len() as i32);
        for be in &self.block_entities {
            out.push(be.packed_xz);
            out.extend_from_slice(&be.y.to_be_bytes());
            crate::write_varint(&mut out, be.ty as i32);
            write_nullable_nbt(&mut out, &be.tag);
        }

        write_bitset(&mut out, &self.light.sky_mask);
        write_bitset(&mut out, &self.light.block_mask);
        write_bitset(&mut out, &self.light.empty_sky_mask);
        write_bitset(&mut out, &self.light.empty_block_mask);
        write_light_arrays(&mut out, &self.light.sky_updates);
        write_light_arrays(&mut out, &self.light.block_updates);
        out
    }
}

// --- packet primitives (chunk-local; the shared protocol crate's Reader is
// --- oracle-internal, so this module carries its own minimal set) ---

pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn read_u8(&mut self) -> Result<u8> {
        let b = *self
            .buf
            .get(self.pos)
            .with_context(|| format!("truncated at byte {}", self.pos))?;
        self.pos += 1;
        Ok(b)
    }

    pub fn read_varint(&mut self) -> Result<i32> {
        let mut value: u32 = 0;
        for i in 0..5 {
            let b = self.read_u8()?;
            value |= u32::from(b & 0x7f) << (7 * i);
            if b & 0x80 == 0 {
                return Ok(value as i32);
            }
        }
        bail!("varint too long")
    }

    pub fn read_i16(&mut self) -> Result<i16> {
        let hi = self.read_u8()? as i16;
        let lo = self.read_u8()? as i16;
        Ok((hi << 8) | (lo & 0xff))
    }

    pub fn read_i32(&mut self) -> Result<i32> {
        let mut bytes = [0u8; 4];
        for b in bytes.iter_mut() {
            *b = self.read_u8()?;
        }
        Ok(i32::from_be_bytes(bytes))
    }

    pub fn read_u64(&mut self) -> Result<u64> {
        let mut bytes = [0u8; 8];
        for b in bytes.iter_mut() {
            *b = self.read_u8()?;
        }
        Ok(u64::from_be_bytes(bytes))
    }

    pub fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        let start = self.pos;
        self.pos += n;
        let bytes = self
            .buf
            .get(start..self.pos)
            .with_context(|| format!("truncated reading {n} bytes"))?;
        Ok(bytes.to_vec())
    }
}

fn read_bitset(r: &mut Reader) -> Result<Vec<u8>> {
    let len = r.read_varint().context("bitset length")? as usize;
    r.read_bytes(len).context("bitset bytes")
}

fn write_bitset(out: &mut Vec<u8>, bytes: &[u8]) {
    crate::write_varint(out, bytes.len() as i32);
    out.extend_from_slice(bytes);
}

fn read_light_arrays(r: &mut Reader) -> Result<Vec<Vec<u8>>> {
    let count = r.read_varint().context("light array count")? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let len = r.read_varint().context("light array length")? as usize;
        out.push(r.read_bytes(len).context("light array")?);
    }
    Ok(out)
}

fn write_light_arrays(out: &mut Vec<u8>, arrays: &[Vec<u8>]) {
    crate::write_varint(out, arrays.len() as i32);
    for a in arrays {
        crate::write_varint(out, a.len() as i32);
        out.extend_from_slice(a);
    }
}

/// Reads a nullable network NBT compound, returning its raw bytes
/// (including the presence bool, excluding nothing) for byte-faithful relay.
fn read_nullable_nbt(r: &mut Reader) -> Result<Option<Vec<u8>>> {
    let present = r.read_u8().context("nbt presence")?;
    if present == 0 {
        return Ok(None);
    }
    let start = r.pos - 1; // include the presence byte in the passthrough
    let root_tag = r.read_u8().context("nbt root tag")?;
    if root_tag == 0 {
        return Ok(Some(r.buf[start..r.pos].to_vec()));
    }
    skip_nbt_payload(r, root_tag).context("walking nbt")?;
    Ok(Some(r.buf[start..r.pos].to_vec()))
}

fn write_nullable_nbt(out: &mut Vec<u8>, tag: &Option<Vec<u8>>) {
    match tag {
        None => out.push(0),
        Some(bytes) => out.extend_from_slice(bytes),
    }
}

/// Walks one NBT payload (after its type byte), advancing past it.
fn skip_nbt_payload(r: &mut Reader, tag: u8) -> Result<()> {
    match tag {
        1 => {
            r.read_u8()?;
        }
        2 => {
            r.read_i16()?;
        }
        3 => {
            r.read_i32()?;
        }
        4 => {
            r.read_u64()?;
        }
        5 => {
            r.read_bytes(4)?;
        }
        6 => {
            r.read_bytes(8)?;
        }
        7 => {
            let len = r.read_i32()? as usize;
            r.read_bytes(len)?;
        }
        8 => {
            let len = u16::from_be_bytes([r.read_u8()?, r.read_u8()?]) as usize;
            r.read_bytes(len)?;
        }
        9 => {
            let elem = r.read_u8()?;
            let len = r.read_i32()? as usize;
            for _ in 0..len {
                skip_nbt_payload(r, elem)?;
            }
        }
        10 => loop {
            let t = r.read_u8().context("compound entry tag")?;
            if t == 0 {
                break;
            }
            let name_len = u16::from_be_bytes([r.read_u8()?, r.read_u8()?]) as usize;
            r.read_bytes(name_len)?;
            skip_nbt_payload(r, t)?;
        },
        11 => {
            let len = r.read_i32()? as usize;
            r.read_bytes(len * 4)?;
        }
        12 => {
            let len = r.read_i32()? as usize;
            r.read_bytes(len * 8)?;
        }
        other => bail!("unknown nbt tag {other}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_chunk() -> WireChunk {
        WireChunk {
            x: 12,
            z: -3,
            heightmaps: vec![(1, vec![0x1234; 37]), (4, vec![0xabcd; 37])],
            sections: vec![
                WireSection {
                    non_empty: 4096,
                    fluid: 0,
                    block_states: Container::Single(1),
                    biomes: Container::Single(9),
                },
                WireSection {
                    non_empty: 1024,
                    fluid: 3,
                    block_states: Container::Palette {
                        bits: 4,
                        entries: vec![88, 10, 9, 0],
                        longs: vec![0x0123_4567_89ab_cdef; 256],
                    },
                    biomes: Container::Palette {
                        bits: 2,
                        entries: vec![9, 10],
                        longs: vec![u64::MAX; 2],
                    },
                },
            ],
            block_entities: vec![WireBlockEntity {
                packed_xz: 0x12,
                y: -60,
                ty: 8,
                tag: None,
            }],
            light: WireLight {
                sky_mask: vec![0x06],
                block_mask: vec![],
                empty_sky_mask: vec![0x01],
                empty_block_mask: vec![0x07],
                sky_updates: vec![vec![0x0f; 2048]],
                block_updates: vec![],
            },
        }
    }

    #[test]
    fn chunk_roundtrip_synthetic() {
        let chunk = build_test_chunk();
        let encoded = chunk.encode();
        let decoded = WireChunk::decode(&encoded).expect("decode");
        assert_eq!(decoded, chunk);
        assert_eq!(decoded.encode(), encoded);
    }

    /// Validates the codec against REAL vanilla-captured chunk blobs when
    /// the capture directory is present (local dev machines; CI proves the
    /// same property live via parity-through-codec).
    #[test]
    fn chunk_roundtrip_captured_blob() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scratch");
        let manifest_path = root.join("captures/blobs/manifest.json");
        let Ok(manifest) = std::fs::read_to_string(&manifest_path) else {
            eprintln!("skipping: no captured blobs at {}", manifest_path.display());
            return;
        };
        let entries: Vec<(String, i32)> = serde_json::from_str::<Vec<serde_json::Value>>(&manifest)
            .expect("manifest")
            .iter()
            .map(|e| {
                (
                    e["file"].as_str().unwrap().to_string(),
                    e["id"].as_i64().unwrap() as i32,
                )
            })
            .collect();
        let mut checked = 0;
        for (file, id) in entries {
            if id != 0x2e {
                continue;
            }
            let body = std::fs::read(root.join("captures/blobs").join(&file)).expect("blob file");
            let decoded = WireChunk::decode(&body).expect("decode captured chunk");
            let re = decoded.encode();
            if re != body {
                let pos = re
                    .iter()
                    .zip(body.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(re.len().min(body.len()));
                let hex = |b: &[u8]| -> String {
                    b.iter()
                        .map(|x| format!("{x:02x}"))
                        .collect::<Vec<_>>()
                        .join("")
                };
                let lo = pos.saturating_sub(8);
                eprintln!(
                    "lens re={} orig={} first diff at {}",
                    re.len(),
                    body.len(),
                    pos
                );
                eprintln!("re  : {}", hex(&re[lo..(pos + 24).min(re.len())]));
                eprintln!("orig: {}", hex(&body[lo..(pos + 24).min(body.len())]));
                panic!("re-encode differs for {file}");
            }
            checked += 1;
        }
        assert!(checked > 0, "no chunk blobs found in manifest");
    }
}
