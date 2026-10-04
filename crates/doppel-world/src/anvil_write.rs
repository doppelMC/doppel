//! Anvil chunk writing: the inverse of the read path. Wire chunks become
//! chunk NBT, and a region rewriter keeps every other payload while
//! replacing the chunks handed to it. Disk palettes differ from wire
//! palettes: block storage never uses fewer than 4 bits, biome storage
//! starts at 1, and single-entry palettes drop the storage array.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::anvil_to_wire::{pack, unpack};
use crate::chunk_codec::{Container, WireChunk, WireLight};
use crate::registry::BlockRegistry;
use crate::{Biomes, BlockStates, Chunk, PaletteEntry, Section};

/// The chunk DataVersion written for the pinned build.
pub const DATA_VERSION: i32 = 4189;

const SECTOR: usize = 4096;
const LIGHT_SECTIONS: usize = 26;
const LIGHT_BYTES: usize = 2048;

/// Resolves wire ids for the write path: block states through the
/// registry, biome ids through the learned name map inverted.
pub struct WritePalette<'a> {
    pub registry: &'a BlockRegistry,
    pub biome_names: &'a HashMap<u32, String>,
}

/// Converts a wire chunk to chunk NBT. All 24 sections are written so a
/// reload reproduces every section, including empty ones and their biomes.
pub fn wire_to_anvil(wire: &WireChunk, palette: &WritePalette) -> Result<Chunk> {
    if wire.sections.len() != 24 {
        bail!(
            "wire chunk carries {} sections, expected 24",
            wire.sections.len()
        );
    }
    let sky = layer_payloads(&wire.light, true);
    let block = layer_payloads(&wire.light, false);
    let mut sections = Vec::with_capacity(24);
    for (i, ws) in wire.sections.iter().enumerate() {
        let bit = i + 1; // light section index = section y + 5, y = i - 4
        sections.push(Section {
            y: i as i8 - 4,
            block_states: Some(write_blocks(&ws.block_states, palette)?),
            biomes: Some(write_biomes(&ws.biomes, palette)?),
            block_light: block[bit].clone().map(fastnbt::ByteArray::new),
            sky_light: sky[bit].clone().map(fastnbt::ByteArray::new),
        });
    }
    Ok(Chunk {
        data_version: DATA_VERSION,
        x: wire.x,
        z: wire.z,
        status: "minecraft:full".into(),
        sections,
        heightmaps: Some(write_heightmaps(wire)),
    })
}

/// One palette entry for a state id: the bare block name when the state is
/// the block's default, otherwise the {Name, Properties} form.
fn palette_entry(registry: &BlockRegistry, id: u32) -> Result<PaletteEntry> {
    let Some((name, props)) = registry.state_of(id) else {
        bail!("state {id} is not in the registry");
    };
    if registry.state_id(name, "") == Some(id) {
        return Ok(PaletteEntry::Name(name.to_string()));
    }
    let mut properties = HashMap::new();
    for pair in props.split(',').filter(|p| !p.is_empty()) {
        if let Some((k, v)) = pair.split_once('=') {
            properties.insert(k.to_string(), fastnbt::Value::String(v.to_string()));
        }
    }
    Ok(PaletteEntry::Full {
        name: name.to_string(),
        properties: Some(fastnbt::Value::Compound(properties)),
    })
}

/// Disk block bits: at least 4, then enough for the palette size.
fn disk_block_bits(len: usize) -> usize {
    let mut bits = 4;
    while (1 << bits) < len {
        bits += 1;
    }
    bits
}

/// Disk biome bits: at least 1, then enough for the palette size.
fn disk_biome_bits(len: usize) -> usize {
    let mut bits = 1;
    while (1 << bits) < len {
        bits += 1;
    }
    bits
}

fn write_blocks(container: &Container, palette: &WritePalette) -> Result<BlockStates> {
    let (palette_entries, cells) = match container {
        Container::Single(v) => (vec![*v], Vec::new()),
        Container::Palette {
            entries,
            longs,
            bits,
        } => {
            let cells = unpack(longs, *bits as usize, 4096);
            (entries.clone(), cells)
        }
        Container::Global { bits, longs } => {
            // Direct storage carries global ids: rebuild a local palette
            // from the states actually present.
            let cells = unpack(longs, *bits as usize, 4096);
            let mut entries: Vec<u32> = Vec::new();
            let mut index_of: HashMap<u32, u16> = HashMap::new();
            let mut local = Vec::with_capacity(4096);
            for &cell in &cells {
                let id = cell as u32;
                let idx = match index_of.get(&id) {
                    Some(&i) => i,
                    None => {
                        let i = entries.len() as u16;
                        entries.push(id);
                        index_of.insert(id, i);
                        i
                    }
                };
                local.push(idx);
            }
            (entries, local)
        }
    };
    let palette_out = palette_entries
        .iter()
        .map(|&id| palette_entry(palette.registry, id))
        .collect::<Result<Vec<_>>>()?;
    if palette_out.len() == 1 {
        return Ok(BlockStates {
            palette: palette_out,
            data: None,
        });
    }
    let bits = disk_block_bits(palette_out.len());
    Ok(BlockStates {
        palette: palette_out,
        data: Some(fastnbt::LongArray::new(
            pack(&cells, bits).into_iter().map(|v| v as i64).collect(),
        )),
    })
}

fn write_biomes(container: &Container, palette: &WritePalette) -> Result<Biomes> {
    let name_of = |id: u32| -> Result<String> {
        palette
            .biome_names
            .get(&id)
            .cloned()
            .with_context(|| format!("no biome name learned for id {id}"))
    };
    let (entries, cells) = match container {
        Container::Single(v) => (vec![*v], Vec::new()),
        Container::Palette {
            entries,
            longs,
            bits,
        } => (entries.clone(), unpack(longs, *bits as usize, 64)),
        Container::Global { bits, longs } => {
            let cells = unpack(longs, *bits as usize, 64);
            let mut entries: Vec<u32> = Vec::new();
            let mut index_of: HashMap<u32, u16> = HashMap::new();
            let mut local = Vec::with_capacity(64);
            for &cell in &cells {
                let id = cell as u32;
                let idx = match index_of.get(&id) {
                    Some(&i) => i,
                    None => {
                        let i = entries.len() as u16;
                        entries.push(id);
                        index_of.insert(id, i);
                        i
                    }
                };
                local.push(idx);
            }
            (entries, local)
        }
    };
    let names = entries
        .iter()
        .map(|&id| name_of(id))
        .collect::<Result<Vec<_>>>()?;
    if names.len() == 1 {
        return Ok(Biomes {
            palette: names,
            data: None,
        });
    }
    let bits = disk_biome_bits(names.len());
    Ok(Biomes {
        palette: names,
        data: Some(fastnbt::LongArray::new(
            pack(&cells, bits).into_iter().map(|v| v as i64).collect(),
        )),
    })
}

/// Heightmaps in storage form: the wire's long values under the three
/// client-facing map names; unknown map ids have no storage name.
fn write_heightmaps(wire: &WireChunk) -> fastnbt::Value {
    let mut map = HashMap::new();
    for (ty, longs) in &wire.heightmaps {
        let name = match ty {
            4 => "MOTION_BLOCKING",
            5 => "MOTION_BLOCKING_NO_LEAVES",
            1 => "WORLD_SURFACE",
            _ => continue,
        };
        map.insert(
            name.to_string(),
            fastnbt::Value::LongArray(fastnbt::LongArray::new(
                longs.iter().map(|&v| v as i64).collect(),
            )),
        );
    }
    fastnbt::Value::Compound(map)
}

/// Light layer payloads by light-section index: a set mask bit carries the
/// matching payload, an empty mask bit becomes an all-zero layer, absent
/// layers stay absent.
fn layer_payloads(light: &WireLight, sky: bool) -> Vec<Option<Vec<i8>>> {
    let (mask, empty_mask, updates) = if sky {
        (&light.sky_mask, &light.empty_sky_mask, &light.sky_updates)
    } else {
        (
            &light.block_mask,
            &light.empty_block_mask,
            &light.block_updates,
        )
    };
    let is_set = |bytes: &[u8], bit: usize| {
        bytes
            .get(bit / 8)
            .is_some_and(|b| b & (1 << (bit % 8)) != 0)
    };
    let mut by_bit = vec![None; LIGHT_SECTIONS];
    let mut payloads = updates.iter();
    for (bit, slot) in by_bit.iter_mut().enumerate() {
        if is_set(mask, bit) {
            *slot = payloads
                .next()
                .map(|p| p.iter().map(|&b| b as i8).collect());
        } else if is_set(empty_mask, bit) {
            *slot = Some(vec![0i8; LIGHT_BYTES]);
        }
    }
    by_bit
}

/// The region directory for a world root: the modern overworld layout when
/// it exists, the legacy one when it exists, otherwise the modern path
/// (created on demand by the writer).
pub fn region_dir(root: &Path) -> PathBuf {
    let modern = root.join("dimensions/minecraft/overworld/region");
    if modern.is_dir() {
        return modern;
    }
    let legacy = root.join("region");
    if legacy.is_dir() {
        return legacy;
    }
    modern
}

/// Serializes chunk NBT to zlib bytes ready for a region sector. The
/// fast level keeps a save sweep inside its per-tick budget.
pub fn chunk_to_zlib(chunk: &Chunk) -> Result<Vec<u8>> {
    let nbt = fastnbt::to_bytes(chunk).context("serializing chunk nbt")?;
    let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
    enc.write_all(&nbt).context("compressing chunk nbt")?;
    enc.finish().context("finishing chunk nbt compression")
}

/// Rewrites region files: reads the existing payload table, replaces the
/// chunks handed over, keeps the rest byte-for-byte, and re-lays out all
/// sectors before swapping a tmp file into place.
pub struct RegionWriter {
    path: PathBuf,
}

impl RegionWriter {
    pub fn open(root: &Path, rx: i32, rz: i32) -> RegionWriter {
        RegionWriter {
            path: region_dir(root).join(format!("r.{rx}.{rz}.mca")),
        }
    }

    /// Writes the chunks `(x, z, nbt)` (in-region coordinates 0..32) into
    /// this region, preserving every other stored chunk.
    pub fn write(&self, updates: &[(usize, usize, Chunk)]) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }
        let mut payloads: Vec<Option<(u8, Vec<u8>)>> = vec![None; 1024];
        let mut old_timestamps = [0u32; 1024];
        let existing = std::fs::read(&self.path).unwrap_or_default();
        // A truncated file carries nothing worth keeping.
        if existing.len() >= 2 * SECTOR {
            let data = existing;
            for idx in 0..1024 {
                let entry = u32::from_be_bytes([
                    data[idx * 4],
                    data[idx * 4 + 1],
                    data[idx * 4 + 2],
                    data[idx * 4 + 3],
                ]);
                old_timestamps[idx] = u32::from_be_bytes([
                    data[SECTOR + idx * 4],
                    data[SECTOR + idx * 4 + 1],
                    data[SECTOR + idx * 4 + 2],
                    data[SECTOR + idx * 4 + 3],
                ]);
                let (offset, count) = ((entry >> 8) as usize, (entry & 0xff) as usize);
                if offset == 0 || count == 0 || offset * SECTOR + 5 > data.len() {
                    continue;
                }
                let start = offset * SECTOR;
                let len = u32::from_be_bytes([
                    data[start],
                    data[start + 1],
                    data[start + 2],
                    data[start + 3],
                ]) as usize;
                let end = (start + 4 + len).min(data.len());
                if start + 5 > end {
                    continue;
                }
                payloads[idx] = Some((data[start + 4], data[start + 5..end].to_vec()));
            }
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0);
        for (x, z, chunk) in updates {
            if *x > 31 || *z > 31 {
                bail!("chunk coordinate ({x},{z}) outside its region");
            }
            payloads[x + z * 32] = Some((2, chunk_to_zlib(chunk)?));
            old_timestamps[x + z * 32] = now;
        }

        let mut locations = [0u32; 1024];
        let mut body: Vec<u8> = Vec::new();
        let mut sector = 2u32;
        for (idx, payload) in payloads.iter().enumerate() {
            let Some((compression, bytes)) = payload else {
                continue;
            };
            let total = 5 + bytes.len();
            let count = total.div_ceil(SECTOR) as u32;
            if count >= 256 {
                bail!("chunk {idx} needs {count} sectors, above the table limit");
            }
            locations[idx] = (sector << 8) | count;
            body.extend_from_slice(&((1 + bytes.len()) as u32).to_be_bytes());
            body.push(*compression);
            body.extend_from_slice(bytes);
            let padding = count as usize * SECTOR - total;
            body.extend(std::iter::repeat_n(0u8, padding));
            sector += count;
        }

        let mut file: Vec<u8> = Vec::with_capacity(2 * SECTOR + body.len());
        for entry in locations {
            file.extend_from_slice(&entry.to_be_bytes());
        }
        for stamp in old_timestamps {
            file.extend_from_slice(&stamp.to_be_bytes());
        }
        file.extend_from_slice(&body);

        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let tmp = self.path.with_extension("mca.tmp");
        std::fs::write(&tmp, &file).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &self.path)
            .with_context(|| format!("swapping {} into place", self.path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anvil_to_wire::{convert_uncaptured, pack, PaletteBootstrap};
    use crate::chunk_codec::{Container, WireChunk, WireSection};
    use crate::{Biomes, PaletteEntry};

    fn registry() -> BlockRegistry {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/blocks.json");
        BlockRegistry::load(&path).expect("block registry pins")
    }

    /// Distinct block names in pin order, for palette-size fixtures.
    fn block_names(count: usize) -> Vec<String> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/blocks.json");
        let raw = std::fs::read_to_string(&path).unwrap();
        let entries: Vec<serde_json::Value> = serde_json::from_str(&raw).unwrap();
        let mut names: Vec<String> = Vec::new();
        for e in &entries {
            let name = e["name"].as_str().unwrap().to_string();
            if !names.contains(&name) {
                names.push(name);
            }
            if names.len() == count {
                break;
            }
        }
        names
    }

    fn boot(reg: &BlockRegistry, names: &[String]) -> PaletteBootstrap {
        let mut boot = PaletteBootstrap::default();
        for name in names {
            let id = reg.state_id(name, "").expect("default state");
            boot.blocks.insert(name.clone(), id);
        }
        for (name, id) in [
            ("minecraft:plains", 41u32),
            ("minecraft:desert", 2),
            ("minecraft:forest", 7),
        ] {
            boot.biomes.insert(name.into(), id);
        }
        boot
    }

    fn zlib(nbt: &[u8]) -> Vec<u8> {
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(nbt).unwrap();
        enc.finish().unwrap()
    }

    /// Lays out a region image by hand (idx -> zlib NBT), independent of
    /// the writer under test.
    fn layout(pairs: &[(usize, Vec<u8>)]) -> Vec<u8> {
        let mut file = vec![0u8; 2 * SECTOR];
        let mut sector = 2usize;
        for &(idx, ref body) in pairs {
            let total = 5 + body.len();
            let count = total.div_ceil(SECTOR);
            let entry = ((sector as u32) << 8) | count as u32;
            file[idx * 4..idx * 4 + 4].copy_from_slice(&entry.to_be_bytes());
            let start = sector * SECTOR;
            file.resize(start + count * SECTOR, 0);
            file[start..start + 4].copy_from_slice(&((1 + body.len()) as u32).to_be_bytes());
            file[start + 4] = 2;
            file[start + 5..start + 5 + body.len()].copy_from_slice(body);
            sector += count;
        }
        file
    }

    fn write_region(tag: &str, image: &[u8]) -> std::path::PathBuf {
        let root =
            std::env::temp_dir().join(format!("doppel-anvilwrite-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let region = root.join("region");
        std::fs::create_dir_all(&region).unwrap();
        std::fs::write(region.join("r.0.0.mca"), image).unwrap();
        root
    }

    #[allow(clippy::too_many_arguments)]
    fn states_section(
        y: i8,
        names: &[&str],
        cells: &[u16],
        biome_names: &[&str],
        biome_cells: &[u16],
        sky: Option<Vec<i8>>,
        block_light: Option<Vec<i8>>,
    ) -> Section {
        let palette: Vec<PaletteEntry> = names
            .iter()
            .map(|&n| PaletteEntry::Name(n.into()))
            .collect();
        let data = (palette.len() > 1).then(|| {
            let mut bits = 4;
            while (1 << bits) < palette.len() {
                bits += 1;
            }
            fastnbt::LongArray::new(pack(cells, bits).into_iter().map(|v| v as i64).collect())
        });
        let biomes = Biomes {
            palette: biome_names.iter().map(|s| s.to_string()).collect(),
            data: (biome_names.len() > 1).then(|| {
                let mut bits = 1;
                while (1 << bits) < biome_names.len() {
                    bits += 1;
                }
                fastnbt::LongArray::new(
                    pack(biome_cells, bits)
                        .into_iter()
                        .map(|v| v as i64)
                        .collect(),
                )
            }),
        };
        Section {
            y,
            block_states: Some(BlockStates { palette, data }),
            biomes: Some(biomes),
            block_light: block_light.map(fastnbt::ByteArray::new),
            sky_light: sky.map(fastnbt::ByteArray::new),
        }
    }

    /// Builds zlib chunk NBT with the given sections and heightmaps.
    fn chunk_image(
        x: i32,
        z: i32,
        sections: Vec<Section>,
        heightmaps: Option<fastnbt::Value>,
    ) -> Vec<u8> {
        zlib(
            &fastnbt::to_bytes(&Chunk {
                data_version: DATA_VERSION,
                x,
                z,
                status: "minecraft:full".into(),
                sections,
                heightmaps,
            })
            .unwrap(),
        )
    }

    fn heightmap_value() -> fastnbt::Value {
        let longs: Vec<i64> = pack(&[7u16; 256], 9)
            .into_iter()
            .map(|v| v as i64)
            .collect();
        fastnbt::Value::Compound(HashMap::from([(
            "MOTION_BLOCKING".to_string(),
            fastnbt::Value::LongArray(fastnbt::LongArray::new(longs)),
        )]))
    }

    fn invert(map: &HashMap<String, u32>) -> HashMap<u32, String> {
        map.iter().map(|(k, v)| (*v, k.clone())).collect()
    }

    fn write_palette_of<'a>(
        reg: &'a BlockRegistry,
        biome_names: &'a HashMap<u32, String>,
    ) -> WritePalette<'a> {
        WritePalette {
            registry: reg,
            biome_names,
        }
    }

    /// read -> wire A, write -> read -> wire B: byte-identical.
    fn assert_roundtrip(tag: &str, image: &[u8], x: i32, z: i32, boot: &PaletteBootstrap) {
        let root_a = write_region(&format!("{tag}-a"), image);
        let region = crate::Region::open(&root_a.join("region/r.0.0.mca")).unwrap();
        let anvil = region
            .chunk(x.rem_euclid(32) as usize, z.rem_euclid(32) as usize)
            .unwrap()
            .expect("chunk present");
        let a = convert_uncaptured(&anvil, boot).expect("wire conversion");
        let reg = registry();
        let biome_names = invert(&boot.biomes);
        let out =
            wire_to_anvil(&a, &write_palette_of(&reg, &biome_names)).expect("anvil conversion");
        let root = write_region(&format!("{tag}-b"), &[]);
        RegionWriter::open(&root, 0, 0)
            .write(&[(x.rem_euclid(32) as usize, z.rem_euclid(32) as usize, out)])
            .unwrap();
        let reread = crate::Region::open(&root.join("region/r.0.0.mca")).unwrap();
        let anvil_b = reread
            .chunk(x.rem_euclid(32) as usize, z.rem_euclid(32) as usize)
            .unwrap()
            .expect("chunk still present");
        let b = convert_uncaptured(&anvil_b, boot).expect("wire reconversion");
        assert_eq!(a.encode(), b.encode(), "wire bytes differ for {tag}");
        assert_eq!(a, b, "wire chunks differ for {tag}");
    }

    #[test]
    fn roundtrip_multi_palette_chunk_with_light_and_heightmaps() {
        let reg = registry();
        let names = block_names(6);
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut boot = boot(&reg, &names);
        boot.blocks.insert("minecraft:air".into(), 0);
        boot.blocks.insert(
            "minecraft:stone".into(),
            reg.state_id("minecraft:stone", "").unwrap(),
        );
        let cells: Vec<u16> = (0..4096).map(|i| (i % 6) as u16).collect();
        let biome_cells: Vec<u16> = (0..64).map(|i| (i % 3) as u16).collect();
        let sections = vec![
            states_section(
                -4,
                &name_refs,
                &cells,
                &["minecraft:plains", "minecraft:desert", "minecraft:forest"],
                &biome_cells,
                Some(vec![15i8; 2048]),
                Some(vec![0i8; 2048]),
            ),
            states_section(
                -3,
                &["minecraft:stone"],
                &[],
                &["minecraft:plains"],
                &[],
                Some(vec![0i8; 2048]),
                Some(vec![3i8; 2048]),
            ),
        ];
        let image = layout(&[(
            5 + 9 * 32,
            chunk_image(5, 9, sections, Some(heightmap_value())),
        )]);
        assert_roundtrip("multi", &image, 5, 9, &boot);
    }

    #[test]
    fn roundtrip_sparse_chunk_fills_missing_sections() {
        let reg = registry();
        let mut boot = boot(&reg, &[]);
        boot.blocks.insert("minecraft:air".into(), 0);
        boot.blocks.insert("minecraft:stone".into(), 1);
        let sections = vec![states_section(
            2,
            &["minecraft:stone"],
            &[],
            &["minecraft:plains"],
            &[],
            None,
            None,
        )];
        let image = layout(&[(0, chunk_image(0, 0, sections, None))]);
        assert_roundtrip("sparse", &image, 0, 0, &boot);
    }

    #[test]
    fn roundtrip_palette_bit_boundaries() {
        let reg = registry();
        for size in [2usize, 16, 17, 40] {
            let names = block_names(size + 1);
            let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
            let mut boot = boot(&reg, &names);
            boot.blocks.insert("minecraft:air".into(), 0);
            let cells: Vec<u16> = (0..4096).map(|i| (i % (size + 1)) as u16).collect();
            let sections = vec![states_section(
                0,
                &name_refs,
                &cells,
                &["minecraft:plains", "minecraft:desert"],
                &[1; 64],
                None,
                None,
            )];
            let image = layout(&[(
                3 + 3 * 32,
                chunk_image(3, 3, sections, Some(heightmap_value())),
            )]);
            assert_roundtrip(&format!("bits-{size}"), &image, 3, 3, &boot);
        }
    }

    #[test]
    fn roundtrip_mixed_biomes() {
        let reg = registry();
        let mut boot = boot(&reg, &[]);
        boot.blocks.insert("minecraft:air".into(), 0);
        let cases: Vec<Vec<u16>> = vec![vec![0u16; 64], (0..64).map(|i| (i % 2) as u16).collect()];
        for (i, cells) in cases.into_iter().enumerate() {
            let sections = vec![states_section(
                -1,
                &["minecraft:air"],
                &[],
                &["minecraft:plains", "minecraft:desert"],
                &cells,
                None,
                None,
            )];
            let image = layout(&[(7, chunk_image(7, 0, sections, None))]);
            assert_roundtrip(&format!("biomes-{i}"), &image, 7, 0, &boot);
        }
    }

    #[test]
    fn direct_container_expands_to_local_palette() {
        let reg = registry();
        let names = block_names(3);
        let mut boot = boot(&reg, &names);
        boot.blocks.insert("minecraft:air".into(), 0);
        let ids: Vec<u32> = names.iter().map(|n| reg.state_id(n, "").unwrap()).collect();
        // 15-bit direct storage: every cell carries a global id.
        let cells: Vec<u16> = (0..4096).map(|i| ids[i % 3] as u16).collect();
        let longs = pack(&cells, 15);
        let wire = WireChunk {
            x: 4,
            z: 30,
            heightmaps: Vec::new(),
            sections: (0..24)
                .map(|i| WireSection {
                    non_empty: if i == 0 { 4096 } else { 0 },
                    fluid: 0,
                    block_states: if i == 0 {
                        Container::Global {
                            bits: 15,
                            longs: longs.clone(),
                        }
                    } else {
                        Container::Single(0)
                    },
                    biomes: Container::Single(41),
                })
                .collect(),
            block_entities: Vec::new(),
            light: crate::chunk_codec::WireLight::default(),
        };
        let biome_names = invert(&boot.biomes);
        let out = wire_to_anvil(&wire, &write_palette_of(&reg, &biome_names)).unwrap();
        let root = write_region("direct", &[]);
        RegionWriter::open(&root, 0, 0)
            .write(&[(4, 30, out)])
            .unwrap();
        let region = crate::Region::open(&root.join("region/r.0.0.mca")).unwrap();
        let anvil = region.chunk(4, 30).unwrap().expect("chunk stored");
        let back = convert_uncaptured(&anvil, &boot).unwrap();
        // Cell content, not container shape, is what survives storage.
        let rebuilt = match &back.sections[0].block_states {
            Container::Palette { entries, longs, .. } => {
                let stored = unpack(longs, 4, 4096);
                stored
                    .iter()
                    .map(|&i| entries[i as usize])
                    .collect::<Vec<_>>()
            }
            other => panic!("expected a rebuilt palette, got {other:?}"),
        };
        let original: Vec<u32> = cells.iter().map(|&c| c as u32).collect();
        assert_eq!(rebuilt, original);
    }

    #[test]
    fn region_writer_keeps_neighbors_and_rewrites_edges() {
        let reg = registry();
        let mut boot = boot(&reg, &[]);
        boot.blocks.insert("minecraft:air".into(), 0);
        boot.blocks.insert("minecraft:stone".into(), 1);
        let air_section = states_section(
            0,
            &["minecraft:air"],
            &[],
            &["minecraft:plains"],
            &[],
            None,
            None,
        );
        let stone_section = |x, z| Chunk {
            data_version: DATA_VERSION,
            x,
            z,
            status: "minecraft:full".into(),
            sections: vec![states_section(
                0,
                &["minecraft:stone"],
                &[],
                &["minecraft:plains"],
                &[],
                None,
                None,
            )],
            heightmaps: None,
        };
        // Chunk (5,9) lives at index 5 + 9*32 = 293; index 0 is (0,0) and
        // index 1023 is (31,31).
        let image = layout(&[(293, chunk_image(5, 9, vec![air_section], None))]);
        let root = write_region("edges", &image);
        let writer = RegionWriter::open(&root, 0, 0);
        writer
            .write(&[(0, 0, stone_section(0, 0)), (31, 31, stone_section(31, 31))])
            .unwrap();
        let region = crate::Region::open(&root.join("region/r.0.0.mca")).unwrap();
        // The untouched neighbor still parses and keeps its position.
        let kept = region.chunk(5, 9).unwrap().expect("neighbor kept");
        assert_eq!((kept.x, kept.z), (5, 9));
        assert_eq!(
            kept.sections[0]
                .block_states
                .as_ref()
                .unwrap()
                .palette
                .len(),
            1
        );
        // The two edge chunks land where the header points.
        assert_eq!(region.chunk(0, 0).unwrap().unwrap().x, 0);
        assert_eq!(region.chunk(31, 31).unwrap().unwrap().x, 31);
        // Header entries are populated for all three.
        let data = std::fs::read(root.join("region/r.0.0.mca")).unwrap();
        for idx in [0usize, 293, 1023] {
            let entry = u32::from_be_bytes([
                data[idx * 4],
                data[idx * 4 + 1],
                data[idx * 4 + 2],
                data[idx * 4 + 3],
            ]);
            assert_ne!(entry, 0, "index {idx} missing from the header");
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
