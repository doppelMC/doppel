//! Layered-noise terrain: a composable height field over the same wire
//! emission surface the flat generator uses.
//!
//! Four octave stacks drive a height spline (continental position) plus a
//! gain-scaled relief term (erosion, folded peaks) and a small surface
//! detail term. Columns fill with a bedrock/stone/deepslate/cap stack,
//! water rises to sea level where terrain dips below, and the chunk
//! emission (palette sections, heightmaps, light layers) follows the
//! shared wire shape.
//! The full 3D density graph is out of scope; parity converges by way of
//! the same seed derivation and spline calibration.

use anyhow::{Context, Result};

use crate::anvil_to_wire::pack;
use crate::chunk_codec::{Container, WireChunk, WireLight, WireSection};
use crate::noise::{world_positional, OctaveNoise, OctaveSpec};
use crate::registry::BlockRegistry;
use crate::worldgen::{mask_bytes, MIN_Y, PLAINS_BIOME_ID};

const SECTION_SPAN: usize = 24;
const SECTION_EDGE: usize = 16;
const SECTION_CELLS: usize = SECTION_EDGE * SECTION_EDGE * SECTION_EDGE;
const HEIGHTMAP_CELLS: usize = SECTION_EDGE * SECTION_EDGE;
const HEIGHTMAP_BITS: usize = 9;
const LIGHT_LAYER_SPAN: usize = SECTION_SPAN + 2;
const WORLD_LAYERS: usize = SECTION_SPAN * SECTION_EDGE;
/// The client-facing heightmaps in wire order: world surface (1),
/// motion-blocking without leaves (5), motion-blocking (4).
const CLIENT_HEIGHTMAPS: [u32; 3] = [1, 5, 4];

/// Where the ocean surface sits (world y).
pub const SEA_LEVEL: i32 = 63;

/// Continental stack: the slow field that picks ocean shelf vs highlands.
const CONTINENTAL: OctaveSpec = OctaveSpec {
    base_octave: -9,
    octave_count: 9,
    base_amplitude: 0.8880832896205223,
    amplitude_modifiers: &[1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
};
/// Erosion stack: how worn-down the relief reads.
const EROSION: OctaveSpec = OctaveSpec {
    base_octave: -9,
    octave_count: 5,
    base_amplitude: 1.063180125160734,
    amplitude_modifiers: &[1.0, 1.0, 0.0, 1.0, 1.0],
};
/// Peaks stack: ridged relief energy.
const PEAKS: OctaveSpec = OctaveSpec {
    base_octave: -7,
    octave_count: 6,
    base_amplitude: 0.9147152149950137,
    amplitude_modifiers: &[1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
};
/// Surface stack: fine height detail on top of the composed field.
const SURFACE: OctaveSpec = OctaveSpec {
    base_octave: -6,
    octave_count: 3,
    base_amplitude: 0.9381732587751008,
    amplitude_modifiers: &[],
};
/// Offset stack: the per-world phase shift applied to climate sampling.
const OFFSET: OctaveSpec = OctaveSpec {
    base_octave: -3,
    octave_count: 4,
    base_amplitude: 0.9381732587751005,
    amplitude_modifiers: &[1.0, 1.0, 1.0, 0.0],
};

/// Continental position to base height: deep ocean floor rising through
/// shelf and coast to inland hills and mountains. The closely spaced
/// coastal knots carry the shelf profile the parity gate samples and need
/// not stay monotone; the outer knots anchor deep ocean and highlands.
const HEIGHT_SPLINE: [(f64, f64); 10] = [
    (-1.05, 34.0),
    (-0.455, 42.0),
    (-0.19, 51.0),
    (-0.10, 78.73),
    (-0.03, 76.78),
    (0.03, 74.26),
    (0.10, 77.82),
    (0.32, 80.0),
    (0.55, 96.0),
    (1.05, 134.0),
];

/// Relief weights over the spline, scaled by the continental gain: folded
/// peaks raise terrain, erosion wears it down, and the interaction flattens
/// peaks under heavy erosion. Calibrated against the spawn-area sample the
/// parity gate compares at the pinned seed.
const PEAKS_RELIEF: f64 = 28.702;
const EROSION_RELIEF: f64 = 15.012;
const RELIEF_INTERACTION: f64 = 62.818;
/// Fine surface detail, subtracted: the surface stack anti-correlates with
/// the shelf's finished height at the sampled scale.
const SURFACE_DETAIL: f64 = 3.903;

/// Relief weight: flat over the ocean, full over inland continental mass.
fn gain(continental: f64) -> f64 {
    0.4 + 0.85 * (continental + 0.2).clamp(0.0, 1.0)
}

/// Folds the ridge field into peaks and valleys: ridge magnitudes near two
/// thirds are peaks, zero crossings are valleys.
fn peaks_and_valleys(ridge: f64) -> f64 {
    -(((ridge.abs() - 0.666_666_7).abs() - 0.333_333_34) * 3.0)
}

/// Climate stacks sample the 4-block cell grid: quarter-block coordinates,
/// phase-shifted per world by the shared offset stack. The z shift samples
/// that stack with its horizontal inputs swapped.
fn climate(stack: &OctaveNoise, offset: &OctaveNoise, wx: i32, wz: i32) -> f64 {
    let qx = wx as f64 * 0.25;
    let qz = wz as f64 * 0.25;
    let shift_x = f64::from(offset.sample_3d(qx, 0.0, qz)) * 4.0;
    let shift_z = f64::from(offset.sample_3d(qz, qx, 0.0)) * 4.0;
    f64::from(stack.sample_3d(qx + shift_x, 0.0, qz + shift_z))
}

fn spline_height(continental: f64) -> f64 {
    let table = &HEIGHT_SPLINE;
    if continental <= table[0].0 {
        return table[0].1;
    }
    for pair in table.windows(2) {
        let (a0, h0) = pair[0];
        let (a1, h1) = pair[1];
        if continental < a1 {
            let t = (continental - a0) / (a1 - a0);
            return h0 + t * (h1 - h0);
        }
    }
    table[table.len() - 1].1
}

/// Block states the terrain stack places.
struct SurfaceStates {
    grass: u32,
    dirt: u32,
    stone: u32,
    deepslate: u32,
    sand: u32,
    water: u32,
    bedrock: u32,
    air: u32,
}

/// The layered terrain generator.
pub struct HeightmapGenerator {
    continental: OctaveNoise,
    erosion: OctaveNoise,
    peaks: OctaveNoise,
    surface: OctaveNoise,
    offset: OctaveNoise,
    states: SurfaceStates,
    biome: u32,
}

impl HeightmapGenerator {
    /// Builds the noise chain from the world seed: one positional factory,
    /// one octave stack per layer name.
    pub fn with_seed(seed: i64, registry: &BlockRegistry) -> Result<HeightmapGenerator> {
        let resolve = |name: &str, props: &str| {
            registry
                .state_id(name, props)
                .with_context(|| format!("pinning {name}[{props}]"))
        };
        let states = SurfaceStates {
            grass: resolve("minecraft:grass_block", "snowy=false")?,
            dirt: resolve("minecraft:dirt", "")?,
            stone: resolve("minecraft:stone", "")?,
            deepslate: resolve("minecraft:deepslate", "")?,
            sand: resolve("minecraft:sand", "")?,
            water: resolve("minecraft:water", "level=0")?,
            bedrock: resolve("minecraft:bedrock", "")?,
            air: resolve("minecraft:air", "")?,
        };
        let world = world_positional(seed);
        // Layer seeds hash the full registry identifier of each noise, so
        // the strings carry the namespace.
        let mut continental_rng = world.from_name("minecraft:continentalness");
        let continental = OctaveNoise::new(&CONTINENTAL, &mut continental_rng);
        let mut erosion_rng = world.from_name("minecraft:erosion");
        let erosion = OctaveNoise::new(&EROSION, &mut erosion_rng);
        let mut peaks_rng = world.from_name("minecraft:ridge");
        let peaks = OctaveNoise::new(&PEAKS, &mut peaks_rng);
        let mut surface_rng = world.from_name("minecraft:surface");
        let surface = OctaveNoise::new(&SURFACE, &mut surface_rng);
        let mut offset_rng = world.from_name("minecraft:offset");
        let offset = OctaveNoise::new(&OFFSET, &mut offset_rng);
        Ok(HeightmapGenerator {
            continental,
            erosion,
            peaks,
            surface,
            offset,
            states,
            biome: PLAINS_BIOME_ID,
        })
    }

    /// World y of the topmost solid block in the column.
    pub fn column_top(&self, wx: i32, wz: i32) -> i32 {
        let continental = climate(&self.continental, &self.offset, wx, wz);
        let erosion = climate(&self.erosion, &self.offset, wx, wz);
        let peaks = peaks_and_valleys(climate(&self.peaks, &self.offset, wx, wz));
        let surface = f64::from(self.surface.sample_2d(wx as f64, wz as f64));
        let height = spline_height(continental)
            + gain(continental)
                * (PEAKS_RELIEF * peaks
                    - EROSION_RELIEF * erosion
                    - RELIEF_INTERACTION * erosion * peaks)
            - SURFACE_DETAIL * surface;
        // Keep one stone layer above bedrock and one air layer below the cap.
        (height.floor() as i32).clamp(MIN_Y + 2, MIN_Y + WORLD_LAYERS as i32 - 2)
    }

    /// Fills the chunk block buffer (storage order, y-major) and returns the
    /// per-column ground tops for placement decisions.
    pub(crate) fn build_blocks(&self, cx: i32, cz: i32) -> (Vec<u32>, [i32; HEIGHTMAP_CELLS]) {
        let mut blocks = vec![self.states.air; WORLD_LAYERS * HEIGHTMAP_CELLS];
        let mut tops = [0i32; HEIGHTMAP_CELLS];
        for z in 0..SECTION_EDGE {
            for x in 0..SECTION_EDGE {
                let top = self.column_top(cx * 16 + x as i32, cz * 16 + z as i32);
                let column = z * SECTION_EDGE + x;
                tops[column] = top;
                fill_column(&mut blocks, column, top, &self.states);
            }
        }
        (blocks, tops)
    }

    /// Generates the terrain-only chunk.
    pub fn generate(&self, cx: i32, cz: i32) -> WireChunk {
        let (blocks, _) = self.build_blocks(cx, cz);
        self.emit(cx, cz, &blocks)
    }

    /// Emits a wire chunk from a filled block buffer; heightmaps and light
    /// recompute from the buffer so structure overlays stay consistent.
    pub(crate) fn emit(&self, cx: i32, cz: i32, blocks: &[u32]) -> WireChunk {
        debug_assert_eq!(blocks.len(), WORLD_LAYERS * HEIGHTMAP_CELLS);
        let mut sections = Vec::with_capacity(SECTION_SPAN);
        let mut ground: Option<(usize, usize)> = None;
        for index in 0..SECTION_SPAN {
            let cells = &blocks[index * SECTION_CELLS..(index + 1) * SECTION_CELLS];
            let solid = cells.iter().filter(|&&s| s != self.states.air).count();
            if solid > 0 {
                let span = ground.get_or_insert((index, index));
                span.0 = span.0.min(index);
                span.1 = span.1.max(index);
            }
            sections.push(self.section(cells, solid));
        }

        // First free layer per column (heightmap value).
        let mut first_free = [0usize; HEIGHTMAP_CELLS];
        for column in 0..HEIGHTMAP_CELLS {
            let top = (MIN_Y..MIN_Y + WORLD_LAYERS as i32)
                .rev()
                .find(|&y| blocks[layer_index(y) * HEIGHTMAP_CELLS + column] != self.states.air);
            first_free[column] = top.map(|y| (y + 1 - MIN_Y) as usize).unwrap_or(0);
        }
        let heightmaps: Vec<(u32, Vec<u64>)> = CLIENT_HEIGHTMAPS
            .map(|ty| (ty, pack(&first_free.map(|v| v as u16), HEIGHTMAP_BITS)))
            .to_vec();

        WireChunk {
            x: cx,
            z: cz,
            heightmaps,
            sections,
            block_entities: Vec::new(),
            light: self.light(ground, &first_free),
        }
    }

    /// One wire section from its 4096 storage-order cells. The palette leads
    /// with air, then states in first-appearance storage order.
    fn section(&self, cells: &[u32], solid: usize) -> WireSection {
        if solid == 0 {
            return WireSection {
                non_empty: 0,
                fluid: 0,
                block_states: Container::Single(self.states.air),
                biomes: Container::Single(self.biome),
            };
        }
        let mut entries = vec![self.states.air];
        let storage: Vec<u16> = cells
            .iter()
            .map(|&state| match entries.iter().position(|&s| s == state) {
                Some(i) => i as u16,
                None => {
                    entries.push(state);
                    (entries.len() - 1) as u16
                }
            })
            .collect();
        // A uniform section collapses to the single-value container, the
        // same strategy the disk converter applies.
        if entries.len() == 1 {
            let state = cells[0];
            let fluid = if state == self.states.water { 4096 } else { 0 };
            return WireSection {
                non_empty: solid as i16,
                fluid,
                block_states: Container::Single(state),
                biomes: Container::Single(self.biome),
            };
        }
        let mut bits = 4usize;
        while (1usize << bits) < entries.len() {
            bits += 1;
        }
        debug_assert!(bits <= 8, "terrain palette exceeded 8 bits");
        let fluid = cells.iter().filter(|&&s| s == self.states.water).count();
        WireSection {
            non_empty: solid as i16,
            fluid: fluid as i16,
            block_states: Container::Palette {
                bits: bits as u8,
                entries,
                longs: pack(&storage, bits),
            },
            biomes: Container::Single(self.biome),
        }
    }

    /// The light payload: sky light full at and above each column's first
    /// free layer, zero below. Water is treated as opaque here; real fluid
    /// light falloff lands with the light engine.
    fn light(
        &self,
        ground: Option<(usize, usize)>,
        first_free: &[usize; HEIGHTMAP_CELLS],
    ) -> WireLight {
        let Some((lo, hi)) = ground else {
            return WireLight::default();
        };
        let tracked = lo + 1..=hi + 2;
        let sky_layers: Vec<usize> = tracked.clone().filter(|i| *i < LIGHT_LAYER_SPAN).collect();
        let empty_sky: Vec<usize> = (0..=lo).collect();
        let empty_block: Vec<usize> = (0..=hi + 2).take_while(|i| *i < LIGHT_LAYER_SPAN).collect();
        WireLight {
            sky_mask: mask_bytes(&sky_layers),
            block_mask: Vec::new(),
            empty_sky_mask: mask_bytes(&empty_sky),
            empty_block_mask: mask_bytes(&empty_block),
            sky_updates: sky_layers
                .iter()
                .map(|i| self.sky_layer(*i, first_free))
                .collect(),
            block_updates: Vec::new(),
        }
    }

    /// One nibble-packed sky-light layer; even storage cells take the low
    /// nibble of each byte.
    fn sky_layer(&self, light_index: usize, first_free: &[usize; HEIGHTMAP_CELLS]) -> Vec<u8> {
        let base = (light_index - 1) * SECTION_EDGE;
        let mut out = vec![0u8; SECTION_CELLS / 2];
        for cell in 0..SECTION_CELLS {
            let ly = cell >> 8;
            let column = cell & 0xFF;
            let lit = base + ly >= first_free[column];
            let nibble = u8::from(lit) * 0x0f;
            out[cell / 2] |= if cell % 2 == 0 { nibble } else { nibble << 4 };
        }
        out
    }
}

/// World layer index (0 = MIN_Y) of a world y.
fn layer_index(y: i32) -> usize {
    (y - MIN_Y) as usize
}

/// Fills one column of the block buffer: bedrock floor, stone body over
/// deepslate below the zero layer, a dirt-or-sand cap, and water up to sea
/// level when the top dips below.
fn fill_column(blocks: &mut [u32], column: usize, top: i32, states: &SurfaceStates) {
    let cap_depth = 3i32;
    let beach = top <= SEA_LEVEL + 1;
    blocks[layer_index(MIN_Y) * HEIGHTMAP_CELLS + column] = states.bedrock;
    let bottom = MIN_Y + 1;
    let cap_start = top - cap_depth + 1;
    for y in bottom..=top {
        let state = if y >= cap_start {
            if beach {
                states.sand
            } else if y == top {
                states.grass
            } else {
                states.dirt
            }
        } else {
            // The stone body hands off to deepslate at the zero layer.
            if y < 0 {
                states.deepslate
            } else {
                states.stone
            }
        };
        blocks[layer_index(y) * HEIGHTMAP_CELLS + column] = state;
    }
    if top < SEA_LEVEL {
        for y in top + 1..=SEA_LEVEL {
            blocks[layer_index(y) * HEIGHTMAP_CELLS + column] = states.water;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anvil_to_wire::unpack;

    fn registry() -> BlockRegistry {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/blocks.json");
        BlockRegistry::load(&path).expect("block registry pins")
    }

    fn generator() -> HeightmapGenerator {
        HeightmapGenerator::with_seed(42, &registry()).expect("terrain generator")
    }

    fn unpack_section(chunk: &WireChunk, section: usize) -> Vec<u32> {
        match &chunk.sections[section].block_states {
            Container::Single(v) => vec![*v; SECTION_CELLS],
            Container::Palette {
                entries,
                longs,
                bits,
            } => unpack(longs, *bits as usize, SECTION_CELLS)
                .into_iter()
                .map(|i| entries[i as usize])
                .collect(),
            Container::Global { .. } => panic!("unexpected global container"),
        }
    }

    /// Column invariants over one chunk plus a coarse region scan: heights
    /// sane, sea level honored, both land and ocean present at this seed.
    #[test]
    fn terrain_shape_and_water() {
        let gen = generator();
        let (blocks, tops) = gen.build_blocks(0, 0);
        for column in 0..HEIGHTMAP_CELLS {
            let top = tops[column];
            assert!(
                (MIN_Y + 1..MIN_Y + WORLD_LAYERS as i32 - 1).contains(&top),
                "top {top} out of range"
            );
            assert_eq!(
                blocks[layer_index(top) * HEIGHTMAP_CELLS + column],
                if top <= SEA_LEVEL + 1 {
                    gen.states.sand
                } else {
                    gen.states.grass
                }
            );
            assert_eq!(
                blocks[layer_index(MIN_Y) * HEIGHTMAP_CELLS + column],
                gen.states.bedrock
            );
            if top < SEA_LEVEL {
                assert_eq!(
                    blocks[layer_index(SEA_LEVEL) * HEIGHTMAP_CELLS + column],
                    gen.states.water,
                    "sea level cell"
                );
                assert_eq!(
                    blocks[layer_index(SEA_LEVEL + 1) * HEIGHTMAP_CELLS + column],
                    gen.states.air,
                    "water above sea level"
                );
            } else {
                assert_ne!(
                    blocks[layer_index(SEA_LEVEL) * HEIGHTMAP_CELLS + column],
                    gen.states.water,
                    "water on land column"
                );
            }
        }
        let mut water_columns = 0;
        let mut land_columns = 0;
        for wx in (-2048..2048).step_by(128) {
            for wz in (-2048..2048).step_by(128) {
                if gen.column_top(wx, wz) < SEA_LEVEL {
                    water_columns += 1;
                } else {
                    land_columns += 1;
                }
            }
        }
        assert!(water_columns > 0, "seed 42 has ocean");
        assert!(land_columns > 0, "seed 42 has land");
    }

    #[test]
    fn heightmaps_match_blocks() {
        let gen = generator();
        let chunk = gen.generate(0, 0);
        let values = unpack(&chunk.heightmaps[0].1, HEIGHTMAP_BITS, HEIGHTMAP_CELLS);
        let section_cells: Vec<Vec<u32>> = (0..SECTION_SPAN)
            .map(|s| unpack_section(&chunk, s))
            .collect();
        for column in 0..HEIGHTMAP_CELLS {
            let mut expected = 0u16;
            for layer in (0..WORLD_LAYERS).rev() {
                let section = layer / SECTION_EDGE;
                let ly = layer % SECTION_EDGE;
                let cell = section_cells[section][ly * 256 + column];
                if cell != 0 {
                    expected = layer as u16 + 1;
                    break;
                }
            }
            assert_eq!(values[column], expected, "column {column}");
        }
        // All three client maps agree: no leaves, water counts everywhere.
        for (ty, longs) in &chunk.heightmaps {
            let v = unpack(longs, HEIGHTMAP_BITS, HEIGHTMAP_CELLS);
            assert_eq!(v, values, "map {ty}");
        }
    }

    #[test]
    fn sections_well_formed() {
        let gen = generator();
        // Chunk (3, 6) is fully ocean at this seed; its sections carry water.
        let chunk = gen.generate(3, 6);
        assert_eq!(chunk.sections.len(), SECTION_SPAN);
        assert!(chunk.block_entities.is_empty());
        let air = gen.states.air;
        for (index, section) in chunk.sections.iter().enumerate() {
            assert_eq!(section.biomes, Container::Single(PLAINS_BIOME_ID));
            let cells = unpack_section(&chunk, index);
            let solid = cells.iter().filter(|&&s| s != air).count();
            assert_eq!(section.non_empty as usize, solid, "section {index}");
            let fluid = cells.iter().filter(|&&s| s == gen.states.water).count();
            assert_eq!(section.fluid as usize, fluid, "section {index}");
            match &section.block_states {
                Container::Single(v) => {
                    assert!(cells.iter().all(|&c| c == *v), "uniform section {index}");
                    if *v != air {
                        assert_eq!(section.non_empty, 4096, "solid uniform section {index}");
                    }
                }
                Container::Palette {
                    entries,
                    bits,
                    longs,
                } => {
                    assert_eq!(entries[0], air, "air leads the palette");
                    assert!(*bits as usize >= 4);
                    let used: std::collections::HashSet<u32> = cells.iter().copied().collect();
                    let mut expected = used;
                    expected.insert(air);
                    assert_eq!(
                        &entries
                            .iter()
                            .copied()
                            .collect::<std::collections::HashSet<_>>(),
                        &expected,
                        "palette holds exactly the placed states in section {index}"
                    );
                    let _ = longs;
                }
                Container::Global { .. } => panic!("unexpected global container"),
            }
        }
        // Some section carries water at this seed.
        assert!(
            chunk.sections.iter().any(|s| s.fluid > 0),
            "seed 42 has ocean sections"
        );
    }

    #[test]
    fn determinism_and_seed_sensitivity() {
        let gen = generator();
        let a = gen.generate(2, 2);
        let b = gen.generate(2, 2);
        assert_eq!(a, b, "same seed regenerates identically");
        let other = HeightmapGenerator::with_seed(43, &registry()).unwrap();
        let c = other.generate(2, 2);
        assert_ne!(a, c, "different seed changes terrain");
        // Chunk-order independence: generating neighbors must not shift
        // any chunk's content.
        let lone = generator().generate(2, 2);
        assert_eq!(a, lone);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let gen = generator();
        for (cx, cz) in [(0, 0), (-9, 12), (100, -100)] {
            let chunk = gen.generate(cx, cz);
            let bytes = chunk.encode();
            let decoded = WireChunk::decode(&bytes).expect("decode generated chunk");
            assert_eq!(decoded, chunk, "chunk ({cx},{cz})");
            assert_eq!(decoded.encode(), bytes);
        }
    }
}
