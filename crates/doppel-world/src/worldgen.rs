//! Flat ("superflat") world generation: builds whole chunks from a layer
//! stack without touching disk.
//!
//! The classic stack is one bedrock layer at the bottom of the world,
//! two dirt layers, and a grass layer — everything above is air, the biome
//! is plains everywhere, and no structures or block entities exist.

use std::collections::HashMap;

use anyhow::{bail, Context, Result};

use crate::anvil_to_wire::pack;
use crate::chunk_codec::{Container, WireChunk, WireLight, WireSection};
use crate::registry::BlockRegistry;

/// Overworld floor (26.3): sections -4..=19, 384 layers total.
pub const MIN_Y: i32 = -64;
const SECTION_SPAN: usize = 24;
/// Light layers reach one section past each end of the world (-5..=20).
const LIGHT_LAYER_SPAN: usize = SECTION_SPAN + 2;
const SECTION_EDGE: usize = 16;
const SECTION_CELLS: usize = SECTION_EDGE * SECTION_EDGE * SECTION_EDGE;
const HEIGHTMAP_CELLS: usize = SECTION_EDGE * SECTION_EDGE;
/// 384 possible heights plus zero pack into 9 bits; values never straddle
/// a long, so 7 per long and ceil(256/7) = 37 longs per map.
const HEIGHTMAP_BITS: usize = 9;

/// `minecraft:plains` in the biome registry the join replay carries. The
/// biome registry itself is replayed from captured blobs rather than pinned,
/// so the id is quoted from the capture until that changes.
pub const PLAINS_BIOME_ID: u32 = 41;

/// The client-facing heightmaps vanilla sends, in wire order:
/// world surface (1), motion-blocking without leaves (5), motion-blocking
/// (4). Flat columns are uniform, so for solid-topped stacks all three
/// carry the same surface value.
const CLIENT_HEIGHTMAPS: [u32; 3] = [1, 5, 4];

/// One run of the layer stack: `depth` consecutive world layers of `state`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layer {
    pub state: u32,
    pub depth: u16,
}

/// A flat world specification, expanded per world layer at construction.
#[derive(Clone, Debug)]
pub struct FlatGenerator {
    /// Block state for each world layer, index 0 = `MIN_Y`. Always padded
    /// to the full world height with air.
    layers: Vec<u32>,
    /// Whether each layer's state is a fluid (counts toward the fluid tick
    /// counter on the wire).
    fluid: Vec<bool>,
    /// Height in layers: one past the topmost non-air layer (0 for void).
    surface: usize,
    biome: u32,
    air: u32,
}

impl FlatGenerator {
    /// Builds a generator from a bottom-up layer stack.
    pub fn new(
        stack: &[Layer],
        biome: u32,
        air: u32,
        registry: &BlockRegistry,
    ) -> Result<FlatGenerator> {
        let world_depth = SECTION_SPAN * SECTION_EDGE;
        let mut layers = Vec::with_capacity(world_depth);
        let mut fluid = Vec::with_capacity(world_depth);
        for run in stack {
            let name = registry
                .state_of(run.state)
                .map(|(n, _)| n.to_string())
                .with_context(|| format!("layer state {} not in registry", run.state))?;
            let is_fluid = name.contains("water") || name.contains("lava");
            for _ in 0..run.depth as usize {
                layers.push(run.state);
                fluid.push(is_fluid);
            }
        }
        if layers.len() > world_depth {
            bail!(
                "layer stack is {} deep, world holds {world_depth}",
                layers.len()
            );
        }
        let surface = layers
            .iter()
            .rposition(|&s| s != air)
            .map(|top| top + 1)
            .unwrap_or(0);
        layers.resize(world_depth, air);
        fluid.resize(world_depth, false);
        Ok(FlatGenerator {
            layers,
            fluid,
            surface,
            biome,
            air,
        })
    }

    /// The classic superflat stack: bedrock, two dirt, grass on top, plains
    /// everywhere. Vanilla places each block's DEFAULT state, which for
    /// grass means `snowy=false` (requested explicitly because the pin file
    /// lists that block's states in the opposite order).
    pub fn classic(registry: &BlockRegistry) -> Result<FlatGenerator> {
        let resolve = |name: &str, props: &str| {
            registry
                .state_id(name, props)
                .with_context(|| format!("pinning {name}[{props}]"))
        };
        let stack = [
            Layer {
                state: resolve("minecraft:bedrock", "")?,
                depth: 1,
            },
            Layer {
                state: resolve("minecraft:dirt", "")?,
                depth: 2,
            },
            Layer {
                state: resolve("minecraft:grass_block", "snowy=false")?,
                depth: 1,
            },
        ];
        let air = resolve("minecraft:air", "")?;
        FlatGenerator::new(&stack, PLAINS_BIOME_ID, air, registry)
    }

    /// Flat worlds ignore the seed; this exists so callers can swap the
    /// generator behind one constructor shape.
    pub fn with_seed(_seed: i64, registry: &BlockRegistry) -> Result<FlatGenerator> {
        FlatGenerator::classic(registry)
    }

    /// Where a player spawns: one layer above the stack top (vanilla's
    /// spawn-height rule for flat worlds).
    pub fn spawn_y(&self) -> i32 {
        MIN_Y + self.surface as i32
    }

    /// Generates the chunk at (cx, cz). Flat terrain is position-independent,
    /// so the coordinates only label the result.
    pub fn generate(&self, cx: i32, cz: i32) -> WireChunk {
        let mut sections = Vec::with_capacity(SECTION_SPAN);
        let mut ground: Option<(usize, usize)> = None;
        for index in 0..SECTION_SPAN {
            let layers = &self.layers[index * SECTION_EDGE..(index + 1) * SECTION_EDGE];
            let fluids = &self.fluid[index * SECTION_EDGE..(index + 1) * SECTION_EDGE];
            let solid = layers.iter().filter(|&&s| s != self.air).count();
            if solid > 0 {
                let span = ground.get_or_insert((index, index));
                span.0 = span.0.min(index);
                span.1 = span.1.max(index);
            }
            sections.push(self.section(layers, fluids, solid));
        }

        // Heightmaps: every column is identical, so each map is one value
        // repeated. The stored value is (first free y) - MIN_Y, which is
        // exactly the layer count up to the surface.
        let heightmaps: Vec<(u32, Vec<u64>)> = CLIENT_HEIGHTMAPS
            .map(|ty| {
                (
                    ty,
                    pack(&[self.surface as u16; HEIGHTMAP_CELLS], HEIGHTMAP_BITS),
                )
            })
            .to_vec();

        WireChunk {
            x: cx,
            z: cz,
            heightmaps,
            sections,
            block_entities: Vec::new(),
            light: self.light(ground),
        }
    }

    /// One wire section from 16 consecutive world layers. Uniform layers
    /// mean the palette is small and every storage plane is a single value.
    fn section(&self, layers: &[u32], fluids: &[bool], solid_layers: usize) -> WireSection {
        if solid_layers == 0 {
            return WireSection {
                non_empty: 0,
                fluid: 0,
                block_states: Container::Single(self.air),
                biomes: Container::Single(self.biome),
            };
        }

        // Palette: air first, then each further state in first-appearance
        // order scanning the section bottom-up — the order vanilla's
        // in-memory container ends up in when terrain is built bottom-up.
        let mut entries = vec![self.air];
        let mut index_of: HashMap<u32, u16> = HashMap::from([(self.air, 0)]);
        let mut layer_index = [0u16; SECTION_EDGE];
        for (y, &state) in layers.iter().enumerate() {
            match index_of.get(&state) {
                Some(&i) => layer_index[y] = i,
                None => {
                    let next = entries.len() as u16;
                    entries.push(state);
                    index_of.insert(state, next);
                    layer_index[y] = next;
                }
            }
        }

        // Storage cells run y-major: (y << 8) | (z << 4) | x.
        let cells: Vec<u16> = layer_index
            .iter()
            .flat_map(|&index| std::iter::repeat_n(index, SECTION_EDGE * SECTION_EDGE))
            .collect();

        // Uniform layers cap the palette at 16 states, so 4 bits always
        // suffice (the growth loop is defensive only).
        let mut bits = 4usize;
        while (1usize << bits) < entries.len() {
            bits += 1;
        }
        debug_assert!(bits <= 8);

        let fluid_layers = fluids
            .iter()
            .zip(layers.iter().map(|&s| s != self.air))
            .filter(|(fluid, solid)| **fluid && *solid)
            .count();
        let cells_per_layer = SECTION_EDGE * SECTION_EDGE;

        WireSection {
            non_empty: (solid_layers * cells_per_layer) as i16,
            fluid: (fluid_layers * cells_per_layer) as i16,
            block_states: Container::Palette {
                bits: bits as u8,
                entries,
                longs: pack(&cells, bits),
            },
            biomes: Container::Single(self.biome),
        }
    }

    /// The light payload: sky light is full above the surface and zero
    /// below it; block light is zero everywhere (no sources in flat
    /// terrain). Vanilla's serializer marks all-zero layers in the empty
    /// masks and leaves layers above the tracked range absent — the client
    /// treats sky light above the stored range as full.
    fn light(&self, ground: Option<(usize, usize)>) -> WireLight {
        let Some((lo, hi)) = ground else {
            return WireLight::default();
        };
        // Light layer i belongs to world section i - 1.
        let tracked = lo + 1..=hi + 2;
        let sky_layers: Vec<usize> = tracked.clone().filter(|i| *i < LIGHT_LAYER_SPAN).collect();
        let empty_sky: Vec<usize> = (0..=lo).collect();
        let empty_block: Vec<usize> = (0..=hi + 2).take_while(|i| *i < LIGHT_LAYER_SPAN).collect();
        WireLight {
            sky_mask: mask_bytes(&sky_layers),
            block_mask: Vec::new(),
            empty_sky_mask: mask_bytes(&empty_sky),
            empty_block_mask: mask_bytes(&empty_block),
            sky_updates: sky_layers.iter().map(|i| self.sky_layer(i)).collect(),
            block_updates: Vec::new(),
        }
    }

    /// One nibble-packed sky-light layer. Flat light is uniform per world
    /// layer, so each byte (two x-neighbours) is fully dark or fully lit.
    fn sky_layer(&self, light_index: &usize) -> Vec<u8> {
        let base = (light_index - 1) * SECTION_EDGE;
        (0..SECTION_CELLS / 2)
            .map(|byte| {
                let layer = base + (2 * byte) / (SECTION_EDGE * SECTION_EDGE);
                let lit = layer >= self.surface;
                if lit {
                    0xff
                } else {
                    0x00
                }
            })
            .collect()
    }
}

/// Java `BitSet.toByteArray`: bit k lives in byte k/8, LSB-first, trailing
/// zero bytes trimmed (an empty set serializes to no bytes at all).
pub(crate) fn mask_bytes(bits: &[usize]) -> Vec<u8> {
    let Some(&top) = bits.last() else {
        return Vec::new();
    };
    let mut out = vec![0u8; top / 8 + 1];
    for &b in bits {
        out[b / 8] |= 1 << (b % 8);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anvil_to_wire::unpack;

    fn registry() -> BlockRegistry {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/blocks.json");
        BlockRegistry::load(&path).expect("block registry pins")
    }

    fn classic() -> FlatGenerator {
        FlatGenerator::classic(&registry()).expect("classic flat generator")
    }

    /// The observed heightmap longs for the classic stack: the surface
    /// value 4 repeated at 9-bit stride, seven values per long (captured
    /// from vanilla; the final long holds the leftover four values).
    const CLASSIC_HEIGHTMAP_LONG: u64 = 0x0100_8040_2010_0804;
    const CLASSIC_HEIGHTMAP_TAIL: u64 = 0x0000_0000_2010_0804;

    #[test]
    fn classic_layer_positions() {
        let gen = classic();
        let chunk = gen.generate(3, -7);
        assert_eq!(chunk.sections.len(), SECTION_SPAN);

        // Only the bottom section carries blocks: 4 layers x 256 cells.
        let ground = &chunk.sections[0];
        assert_eq!(ground.non_empty, 1024);
        assert_eq!(ground.fluid, 0);
        let Container::Palette {
            bits,
            entries,
            longs,
        } = &ground.block_states
        else {
            panic!("ground section uses a palette");
        };
        assert_eq!(*bits, 4);
        // [air, bedrock, dirt, grass]: air leads, then bottom-up placement
        // order (bedrock 88, dirt 10, grass default state 9).
        let air = registry().state_id("minecraft:air", "").unwrap();
        let bedrock = registry().state_id("minecraft:bedrock", "").unwrap();
        let dirt = registry().state_id("minecraft:dirt", "").unwrap();
        let grass = registry()
            .state_id("minecraft:grass_block", "snowy=false")
            .unwrap();
        assert_eq!(entries, &vec![air, bedrock, dirt, grass]);

        // Storage is y-major: layer planes bottom-up.
        let cells = unpack(longs, *bits as usize, SECTION_CELLS);
        for (y, expect) in [(0, 1), (1, 2), (2, 2), (3, 3), (4, 0), (15, 0)] {
            assert_eq!(
                cells[y * 256..(y + 1) * 256],
                vec![expect; 256],
                "layer plane {y}"
            );
        }

        // Every other section is empty air with the world biome.
        for section in &chunk.sections[1..] {
            assert_eq!(section.non_empty, 0);
            assert_eq!(section.block_states, Container::Single(air));
            assert_eq!(section.biomes, Container::Single(PLAINS_BIOME_ID));
        }
        assert_eq!(chunk.sections[0].biomes, Container::Single(PLAINS_BIOME_ID));
        assert!(chunk.block_entities.is_empty());
        assert_eq!(gen.spawn_y(), -60);
    }

    #[test]
    fn classic_heightmaps() {
        let chunk = classic().generate(0, 0);
        let order: Vec<u32> = chunk.heightmaps.iter().map(|(ty, _)| *ty).collect();
        assert_eq!(order, vec![1, 5, 4]);
        for (ty, longs) in &chunk.heightmaps {
            assert_eq!(longs.len(), 37, "map {ty} long count");
            assert_eq!(
                longs[..36],
                [CLASSIC_HEIGHTMAP_LONG; 36],
                "map {ty} full longs"
            );
            assert_eq!(longs[36], CLASSIC_HEIGHTMAP_TAIL, "map {ty} tail long");
        }
        // The stored value is the surface height above MIN_Y: grass at -61
        // means first free y -60, stored as 4.
        let values = unpack(&chunk.heightmaps[0].1, HEIGHTMAP_BITS, HEIGHTMAP_CELLS);
        assert!(values.iter().all(|&v| v == 4));
    }

    #[test]
    fn classic_light() {
        let chunk = classic().generate(0, 0);
        let light = &chunk.light;
        // Light layers 1 (ground section -4) and 2 (one above) carry sky
        // arrays; everything below is empty; block light is empty-marked
        // through the tracked range only.
        assert_eq!(light.sky_mask, vec![0x06]);
        assert_eq!(light.block_mask, Vec::<u8>::new());
        assert_eq!(light.empty_sky_mask, vec![0x01]);
        assert_eq!(light.empty_block_mask, vec![0x07]);
        assert_eq!(light.block_updates.len(), 0);

        assert_eq!(light.sky_updates.len(), 2);
        assert_eq!(light.sky_updates[0].len(), 2048);
        // Ground section: dark below y=-60 (bytes 0..512), full above.
        assert_eq!(&light.sky_updates[0][..512], &vec![0x00; 512]);
        assert_eq!(&light.sky_updates[0][512..], &vec![0xff; 1536]);
        // The section above the surface is entirely lit.
        assert_eq!(light.sky_updates[1], vec![0xff; 2048]);
    }

    #[test]
    fn deeper_stack_reaches_higher_light_layers() {
        let reg = registry();
        let stone = reg.state_id("minecraft:stone", "").unwrap();
        let air = reg.state_id("minecraft:air", "").unwrap();
        // 35 stone layers: ground spans sections 0..=2 (the top three
        // layers sit in section 2), surface at layer 35.
        let stack = vec![Layer {
            state: stone,
            depth: 35,
        }];
        let gen = FlatGenerator::new(&stack, PLAINS_BIOME_ID, air, &reg).unwrap();
        let chunk = gen.generate(0, 0);
        assert_eq!(chunk.sections[0].non_empty, 4096);
        assert_eq!(chunk.sections[1].non_empty, 4096);
        assert_eq!(chunk.sections[2].non_empty, 768);
        assert_eq!(chunk.sections[3].non_empty, 0);
        // Sky arrays for light layers 1..=4 (ground sections plus one),
        // empty sky {0}, empty block 0..=4.
        assert_eq!(chunk.light.sky_mask, vec![0x1e]);
        assert_eq!(chunk.light.empty_sky_mask, vec![0x01]);
        assert_eq!(chunk.light.empty_block_mask, vec![0x1f]);
        assert_eq!(chunk.light.sky_updates.len(), 4);
        // Sky layer 3 (section 2) is dark for its first three layers.
        let dark = chunk.light.sky_updates[2]
            .iter()
            .filter(|&&b| b == 0)
            .count();
        assert_eq!(dark, 3 * 256 / 2);
        let heightmaps = unpack(&chunk.heightmaps[0].1, HEIGHTMAP_BITS, HEIGHTMAP_CELLS);
        assert!(heightmaps.iter().all(|&v| v == 35));
    }

    #[test]
    fn encode_decode_roundtrip() {
        let chunk = classic().generate(-12, 40);
        let bytes = chunk.encode();
        let decoded = WireChunk::decode(&bytes).expect("decode generated chunk");
        assert_eq!(decoded, chunk);
        assert_eq!(decoded.encode(), bytes);
    }

    /// The strongest local oracle: generated chunks must be byte-identical
    /// to vanilla's captured flat-world chunk packets (the capture ran with
    /// the classic preset and generate-structures=false). Skips when the
    /// capture directory is not checked out.
    #[test]
    fn byte_parity_with_captured_flat_chunks() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scratch");
        let blobs = ["x/captures/blobs", "captures/blobs"]
            .iter()
            .map(|rel| root.join(rel))
            .find(|p| p.join("manifest.json").is_file());
        let Some(blobs) = blobs else {
            eprintln!("skipping: no captured flat-world blobs under scratch/");
            return;
        };
        let manifest: Vec<serde_json::Value> = serde_json::from_str(
            &std::fs::read_to_string(blobs.join("manifest.json")).expect("manifest"),
        )
        .expect("manifest json");
        let gen = classic();
        let mut compared = 0;
        for entry in &manifest {
            if entry["id"].as_i64() != Some(0x2e) {
                continue;
            }
            let file = entry["file"].as_str().expect("file name");
            let body = std::fs::read(blobs.join(file)).expect("blob body");
            let reference = WireChunk::decode(&body).expect("decode captured chunk");
            let generated = gen.generate(reference.x, reference.z);
            let bytes = generated.encode();
            assert_eq!(
                bytes, body,
                "generated bytes differ from captured chunk {file} ({},{})",
                reference.x, reference.z
            );
            compared += 1;
        }
        assert!(compared > 0, "capture had no chunk packets");
        eprintln!("byte parity held for {compared} captured flat chunks");
    }
}
