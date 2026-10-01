//! Anvil chunk -> wire chunk conversion. Block/biome NAME -> global-id
//! mappings are bootstrapped by pairing a vanilla network
//! capture (blob WireChunk) with the same world's Anvil data: when a
//! section's storage longs are identical on both sides, palette entry i of
//! the Anvil palette IS palette entry i of the wire palette, so the names
//! and ids zip together. The conversion then rebuilds chunks purely from
//! Anvil storage, and the parity harness proves the result byte-identical
//! to what vanilla sent.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::chunk_codec::{Container, ContainerKind, WireChunk, WireLight, WireSection};
use crate::{Chunk, Region, Section};

/// Unpacks SimpleBitStorage: values packed LSB-first within each long,
/// `64/bits` values per long, no spans across longs.
pub fn unpack(longs: &[u64], bits: usize, count: usize) -> Vec<u16> {
    let mask: u64 = if bits >= 64 {
        u64::MAX
    } else {
        (1 << bits) - 1
    };
    let per_long = 64 / bits.max(1);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let long = longs.get(i / per_long).copied().unwrap_or(0);
        let shift = (i % per_long) * bits;
        out.push(((long >> shift) & mask) as u16);
    }
    out
}

/// Wire palette bits per vanilla rules, per container kind: blocks start at
/// 4 bits and stay indirect up to 8; biomes start at 1 and stay indirect up
/// to 3. Larger palettes would require DIRECT conversion (global ids in the
/// longs, repacked) — not implemented yet, so this bails instead of
/// silently corrupting the passthrough longs.
fn palette_bits(len: usize, kind: ContainerKind) -> Result<u8> {
    let (min, max) = match kind {
        ContainerKind::Blocks => (4usize, 8usize),
        ContainerKind::Biomes => (1, 3),
    };
    let mut bits = min;
    while (1usize << bits) < len {
        bits += 1;
    }
    if bits > max {
        bail!("{len} palette entries need direct-mode conversion (not yet implemented)");
    }
    Ok(bits as u8)
}

/// Learned name -> global-id maps (blocks and biomes).
#[derive(Default)]
pub struct PaletteBootstrap {
    pub blocks: HashMap<String, u32>,
    pub biomes: HashMap<String, u32>,
}

impl PaletteBootstrap {
    /// Records a mapping, refusing conflicting re-learns (a name that
    /// already maps to a different id means the pairing was unsound).
    fn record(blocks: &mut HashMap<String, u32>, name: &str, id: u32, learned: &mut usize) -> bool {
        match blocks.get(name) {
            Some(existing) if *existing != id => {
                eprintln!("[world] palette conflict for {name}: {existing} vs {id}");
                false
            }
            None => {
                blocks.insert(name.to_string(), id);
                *learned += 1;
                true
            }
            _ => true,
        }
    }

    /// Learns mappings from one (wire, anvil) chunk pair. Sections are
    /// paired by Y — Anvil section lists may be unordered, carry extra
    /// light-only sections, or omit empty ones, so positional zip would
    /// misalign. Returns how many new mappings were learned.
    #[allow(clippy::collapsible_if)]
    pub fn learn(&mut self, wire: &WireChunk, anvil: &Chunk) -> usize {
        let by_y: HashMap<i8, &Section> = anvil.sections.iter().map(|s| (s.y, s)).collect();
        let mut learned = 0;
        for (i, ws) in wire.sections.iter().enumerate() {
            let y = i as i8 - 4; // wire sections are bottom-to-top, y=-4..=19
            let Some(as_) = by_y.get(&y).copied() else {
                continue;
            };
            if let Container::Palette { entries, longs, .. } = &ws.block_states {
                if let Some(anvil_bs) = &as_.block_states {
                    let wire_longs: Vec<u64> = longs.to_vec();
                    if anvil_bs.palette.len() == entries.len() {
                        if let Some(data) = &anvil_bs.data {
                            let anvil_longs: Vec<u64> = data.iter().map(|&v| v as u64).collect();
                            if wire_longs == anvil_longs {
                                for (entry, id) in anvil_bs.palette.iter().zip(entries.iter()) {
                                    if entry.properties.is_none() {
                                        Self::record(
                                            &mut self.blocks,
                                            &entry.name,
                                            *id,
                                            &mut learned,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    // Biomes likewise.
                    if let Container::Palette {
                        entries: be,
                        longs: bl,
                        ..
                    } = &ws.biomes
                    {
                        if let Some(ab) = &as_.biomes {
                            let abl = ab.data.as_deref().unwrap_or_default();
                            let same = ab.palette.len() == be.len()
                                && !bl.is_empty()
                                && bl.iter().map(|&v| v as i64).eq(abl.iter().copied());
                            if same {
                                for (name, id) in ab.palette.iter().zip(be.iter()) {
                                    Self::record(&mut self.biomes, name, *id, &mut learned);
                                }
                            }
                        }
                    }
                }
            }
            // Single-value sections learn their one mapping directly.
            if let (Container::Single(id), Some(ab)) = (&ws.block_states, &as_.block_states) {
                if ab.palette.len() == 1 && ab.palette[0].properties.is_none() {
                    Self::record(&mut self.blocks, &ab.palette[0].name, *id, &mut learned);
                }
            }
            if let (Container::Single(id), Some(ab)) = (&ws.biomes, &as_.biomes) {
                if ab.palette.len() == 1 {
                    Self::record(&mut self.biomes, &ab.palette[0], *id, &mut learned);
                }
            }
        }
        learned
    }
}

/// Rebuilds a wire chunk from Anvil storage. `reference` supplies the light
/// payload and the heightmap order vanilla used (both byte-copied; light
/// comes from the light engine and heightmap order from vanilla's enum
/// iteration — neither is derivable from storage alone yet).
pub fn convert(anvil: &Chunk, reference: &WireChunk, boot: &PaletteBootstrap) -> Result<WireChunk> {
    let air = boot.blocks.get("minecraft:air").copied().unwrap_or(0);
    let mut sections = Vec::with_capacity(24);
    let by_y: HashMap<i8, &Section> = anvil.sections.iter().map(|s| (s.y, s)).collect();
    for y in -4..=19 {
        let Some(sec) = by_y.get(&y).copied() else {
            sections.push(empty_section(air, reference));
            continue;
        };
        sections.push(convert_section(sec, air, boot)?);
    }

    Ok(WireChunk {
        x: anvil.x,
        z: anvil.z,
        heightmaps: reference.heightmaps.clone(),
        sections,
        block_entities: reference.block_entities.clone(),
        light: clone_light(&reference.light),
    })
}

fn empty_section(air: u32, reference: &WireChunk) -> WireSection {
    // Preserve the all-air section's biome from the reference (usually the
    // world's default biome as a single value).
    let biome = reference
        .sections
        .first()
        .and_then(|s| match &s.biomes {
            Container::Single(v) => Some(Container::Single(*v)),
            _ => None,
        })
        .unwrap_or(Container::Single(0));
    WireSection {
        non_empty: 0,
        fluid: 0,
        block_states: Container::Single(air),
        biomes: biome,
    }
}

fn convert_section(sec: &Section, air: u32, boot: &PaletteBootstrap) -> Result<WireSection> {
    let (block_states, non_empty, fluid) = match &sec.block_states {
        None => (Container::Single(air), 0, 0),
        Some(bs) => {
            let ids: Vec<u32> = bs
                .palette
                .iter()
                .map(|p| {
                    if p.properties.is_some() {
                        bail!("state-specific block mapping not yet supported: {}", p.name);
                    }
                    boot.blocks
                        .get(&p.name)
                        .copied()
                        .with_context(|| format!("no global id learned for {}", p.name))
                })
                .collect::<Result<_>>()?;
            let bits = palette_bits(ids.len(), ContainerKind::Blocks)?;
            match (&bs.data, ids.len()) {
                (None, 1) => {
                    let id = ids[0];
                    let non_empty = if id == air { 0 } else { 4096 };
                    (Container::Single(id), non_empty, 0)
                }
                (Some(data), n) if n > 1 => {
                    let longs: Vec<u64> = data.iter().map(|&v| v as u64).collect();
                    let unpacked = unpack(&longs, bits as usize, 4096);
                    let mut non_empty = 0i16;
                    let mut fluid = 0i16;
                    for &idx in &unpacked {
                        let id = ids[idx as usize];
                        if id != air {
                            non_empty += 1;
                        }
                        let name = sec
                            .block_states
                            .as_ref()
                            .and_then(|b| b.palette.get(idx as usize).map(|p| p.name.clone()))
                            .unwrap_or_default();
                        if name.contains("water") || name.contains("lava") {
                            fluid += 1;
                        }
                    }
                    (
                        Container::Palette {
                            bits,
                            entries: ids,
                            longs,
                        },
                        non_empty,
                        fluid,
                    )
                }
                _ => bail!("section with {} palette entries but no data", ids.len()),
            }
        }
    };
    let biomes = match &sec.biomes {
        Some(b) if b.palette.len() == 1 => Container::Single(
            boot.biomes
                .get(&b.palette[0])
                .copied()
                .with_context(|| format!("no biome id learned for {}", b.palette[0]))?,
        ),
        Some(b) => {
            let ids: Vec<u32> = b
                .palette
                .iter()
                .map(|n| {
                    boot.biomes
                        .get(n)
                        .copied()
                        .with_context(|| format!("no biome id learned for {n}"))
                })
                .collect::<Result<_>>()?;
            let bits = palette_bits(ids.len(), ContainerKind::Biomes)?;
            match (&b.data, ids.len()) {
                (None, 1) => Container::Single(ids[0]),
                (Some(data), _) => Container::Palette {
                    bits,
                    entries: ids,
                    longs: data.iter().map(|&v| v as u64).collect(),
                },
                _ => bail!("biome section with no data"),
            }
        }
        None => Container::Single(0),
    };
    Ok(WireSection {
        non_empty,
        fluid,
        block_states,
        biomes,
    })
}

fn clone_light(l: &WireLight) -> WireLight {
    l.clone()
}

/// A lazily-opened set of region files under a world's region directory.
pub struct WorldDir {
    root: PathBuf,
    regions: HashMap<(i32, i32), Region>,
}

impl WorldDir {
    pub fn open(path: &Path) -> Result<WorldDir> {
        // Modern versions keep overworld regions under
        // dimensions/minecraft/overworld; older layouts use the root.
        let candidates = [
            path.join("dimensions/minecraft/overworld/region"),
            path.join("region"),
        ];
        let root = candidates
            .into_iter()
            .find(|p| p.is_dir())
            .with_context(|| format!("no region dir under {}", path.display()))?;
        Ok(WorldDir {
            root,
            regions: HashMap::new(),
        })
    }

    pub fn chunk(&mut self, x: i32, z: i32) -> Result<Option<Chunk>> {
        let rx = x.div_euclid(32);
        let rz = z.div_euclid(32);
        if !self.regions.contains_key(&(rx, rz)) {
            let path = self.root.join(format!("r.{rx}.{rz}.mca"));
            let region = match Region::open(&path) {
                Ok(r) => r,
                Err(_) => {
                    // Missing region = no chunks there.
                    return Ok(None);
                }
            };
            self.regions.insert((rx, rz), region);
        }
        let region = self.regions.get(&(rx, rz)).expect("just inserted");
        region.chunk(x.rem_euclid(32) as usize, z.rem_euclid(32) as usize)
    }
}
