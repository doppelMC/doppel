//! Structure placement: seeded region grids pick feature chunks, piece
//! lists fill block volumes, and chunk borders clip whatever spills.
//!
//! One structure exists so far, a sandstone-rim pool (the well class):
//! a sunk water basin in a sandstone floor with a raised rim. Piece
//! randomness keys off the feature chunk with its own salt, so placement
//! and piece layout are pure functions of the world seed.

use anyhow::{Context, Result};

use crate::chunk_codec::WireChunk;
use crate::noise::Lcg48;
use crate::registry::BlockRegistry;
use crate::terrain::{HeightmapGenerator, SEA_LEVEL};
use crate::worldgen::MIN_Y;

const SECTION_EDGE: i32 = 16;
const WORLD_LAYERS: usize = 384;
const HEIGHTMAP_CELLS: usize = 256;

/// Cell-keyed seeding: each axis mixes with its own large odd constant.
const CELL_X_MIX: i64 = 341873128712;
const CELL_Z_MIX: i64 = 132897987541;
/// The piece stream's salt, distinct from the placement salt.
const PIECE_SALT: i64 = 14357618;

/// A grid of spacing-wide cells; each cell picks one feature chunk inside
/// it at a seeded offset. Separation keeps neighbors apart.
#[derive(Clone, Copy)]
pub struct RegionGrid {
    spacing: i32,
    separation: i32,
    salt: i64,
}

impl RegionGrid {
    pub const fn new(spacing: i32, separation: i32, salt: i64) -> RegionGrid {
        RegionGrid {
            spacing,
            separation,
            salt,
        }
    }

    /// The feature chunk chosen inside cell (cell_x, cell_z).
    pub fn feature_chunk(&self, world_seed: i64, cell_x: i32, cell_z: i32) -> (i32, i32) {
        let seed = (cell_x as i64)
            .wrapping_mul(CELL_X_MIX)
            .wrapping_add((cell_z as i64).wrapping_mul(CELL_Z_MIX))
            .wrapping_add(world_seed)
            .wrapping_add(self.salt);
        let mut rng = Lcg48::new(seed);
        let limit = self.spacing - self.separation;
        (
            cell_x * self.spacing + rng.next_int(limit),
            cell_z * self.spacing + rng.next_int(limit),
        )
    }

    /// Whether (chunk_x, chunk_z) is the feature chunk of its cell.
    pub fn is_feature_chunk(&self, world_seed: i64, chunk_x: i32, chunk_z: i32) -> bool {
        let (fx, fz) = self.feature_chunk(
            world_seed,
            chunk_x.div_euclid(self.spacing),
            chunk_z.div_euclid(self.spacing),
        );
        (fx, fz) == (chunk_x, chunk_z)
    }
}

/// The well grid parameters: one candidate per 32x32 chunks, at least 8
/// apart.
pub fn well_grid() -> RegionGrid {
    RegionGrid::new(32, 8, 14357617)
}

/// Block states the well places.
pub struct WellBlocks {
    sandstone: u32,
    water: u32,
}

impl WellBlocks {
    pub fn from_registry(registry: &BlockRegistry) -> Result<WellBlocks> {
        let resolve = |name: &str, props: &str| {
            registry
                .state_id(name, props)
                .with_context(|| format!("pinning {name}[{props}]"))
        };
        Ok(WellBlocks {
            sandstone: resolve("minecraft:sandstone", "")?,
            water: resolve("minecraft:water", "level=0")?,
        })
    }
}

/// A filled axis-aligned box of one state, in world coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockVolume {
    pub min: (i32, i32, i32),
    /// Extents along x, y, z.
    pub size: (u16, u16, u16),
    pub state: u32,
}

/// The well piece list around `center`: sandstone floor with a water
/// basin sunk into it and a rim rising `rim_layers` above the floor.
/// Volumes apply in order, so the water (listed after the floor) carves
/// the basin.
pub fn well_volumes(
    blocks: &WellBlocks,
    center: (i32, i32),
    base: i32,
    rim_layers: i32,
) -> Vec<BlockVolume> {
    let (x, z) = center;
    let mut volumes = vec![
        BlockVolume {
            min: (x - 2, base, z - 2),
            size: (5, 1, 5),
            state: blocks.sandstone,
        },
        BlockVolume {
            min: (x - 1, base, z - 1),
            size: (3, 1, 3),
            state: blocks.water,
        },
    ];
    for layer in 0..rim_layers {
        let y = base + 1 + layer;
        volumes.extend([
            BlockVolume {
                min: (x - 2, y, z - 2),
                size: (5, 1, 1),
                state: blocks.sandstone,
            },
            BlockVolume {
                min: (x - 2, y, z + 2),
                size: (5, 1, 1),
                state: blocks.sandstone,
            },
            BlockVolume {
                min: (x - 2, y, z - 1),
                size: (1, 1, 3),
                state: blocks.sandstone,
            },
            BlockVolume {
                min: (x + 2, y, z - 1),
                size: (1, 1, 3),
                state: blocks.sandstone,
            },
        ]);
    }
    volumes
}

/// Writes one volume into a chunk's block buffer, clipping cells outside
/// the chunk and outside the world height.
pub(crate) fn write_volume(blocks: &mut [u32], cx: i32, cz: i32, volume: &BlockVolume) {
    for dy in 0..volume.size.1 as i32 {
        for dz in 0..volume.size.2 as i32 {
            for dx in 0..volume.size.0 as i32 {
                let y = volume.min.1 + dy;
                if !(MIN_Y..MIN_Y + WORLD_LAYERS as i32).contains(&y) {
                    continue;
                }
                let wx = volume.min.0 + dx;
                let wz = volume.min.2 + dz;
                let (lx, lz) = (wx - cx * SECTION_EDGE, wz - cz * SECTION_EDGE);
                if !(0..SECTION_EDGE).contains(&lx) || !(0..SECTION_EDGE).contains(&lz) {
                    continue;
                }
                let cell =
                    ((y - MIN_Y) as usize) * HEIGHTMAP_CELLS + (lz as usize) * 16 + lx as usize;
                blocks[cell] = volume.state;
            }
        }
    }
}

/// The well pieces that reach one chunk: every well feature chunk within
/// one chunk of it. A well only places on ground above sea level; ocean
/// feature chunks stay empty.
pub fn well_volumes_near(
    terrain: &HeightmapGenerator,
    well: &WellBlocks,
    world_seed: i64,
    cx: i32,
    cz: i32,
) -> Vec<BlockVolume> {
    const SPACING: i32 = 32;
    let grid = well_grid();
    let mut volumes = Vec::new();
    // Cells whose feature chunk can reach this chunk's neighborhood.
    let gx_range = [(cx - 1).div_euclid(SPACING), (cx + 1).div_euclid(SPACING)];
    let gz_range = [(cz - 1).div_euclid(SPACING), (cz + 1).div_euclid(SPACING)];
    for gx in gx_range[0]..=gx_range[1] {
        for gz in gz_range[0]..=gz_range[1] {
            let (fx, fz) = grid.feature_chunk(world_seed, gx, gz);
            if (fx - cx).abs() > 1 || (fz - cz).abs() > 1 {
                continue;
            }
            let seed = (fx as i64)
                .wrapping_mul(CELL_X_MIX)
                .wrapping_add((fz as i64).wrapping_mul(CELL_Z_MIX))
                .wrapping_add(world_seed)
                .wrapping_add(PIECE_SALT);
            let mut rng = Lcg48::new(seed);
            let ox = rng.next_int(SECTION_EDGE);
            let oz = rng.next_int(SECTION_EDGE);
            let rim_layers = 1 + rng.next_int(2);
            let center = (fx * SECTION_EDGE + ox, fz * SECTION_EDGE + oz);
            let base = terrain.column_top(center.0, center.1);
            if base > SEA_LEVEL {
                volumes.extend(well_volumes(well, center, base, rim_layers));
            }
        }
    }
    volumes
}

/// Generates one chunk: terrain fill, then the well pieces that reach it.
pub fn generate_chunk(
    terrain: &HeightmapGenerator,
    well: &WellBlocks,
    world_seed: i64,
    cx: i32,
    cz: i32,
) -> Result<WireChunk> {
    let (mut blocks, _) = terrain.build_blocks(cx, cz);
    for volume in well_volumes_near(terrain, well, world_seed, cx, cz) {
        write_volume(&mut blocks, cx, cz, &volume);
    }
    let biomes = terrain.section_biomes(cx, cz);
    terrain.emit_with(cx, cz, &blocks, biomes.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anvil_to_wire::unpack;
    use crate::chunk_codec::Container;

    fn registry() -> BlockRegistry {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/blocks.json");
        BlockRegistry::load(&path).expect("block registry pins")
    }

    #[test]
    fn region_grid_spot_values() {
        let grid = well_grid();
        // Seed 42, spacing 32, separation 8, salt 14357617.
        assert_eq!(grid.feature_chunk(42, 0, 0), (1, 7));
        assert_eq!(grid.feature_chunk(42, 1, 0), (51, 15));
        assert_eq!(grid.feature_chunk(42, -1, 2), (-25, 84));
        assert_eq!(grid.feature_chunk(42, 0, -1), (0, -17));
        assert_eq!(grid.feature_chunk(42, 3, -4), (96, -109));
    }

    #[test]
    fn feature_chunks_mark_exactly_their_own_cells() {
        let grid = well_grid();
        let mut marked = std::collections::HashSet::new();
        for gx in -3..=3 {
            for gz in -3..=3 {
                let feature = grid.feature_chunk(42, gx, gz);
                assert!(marked.insert(feature), "feature {feature:?} chosen twice");
                assert!(
                    grid.is_feature_chunk(42, feature.0, feature.1),
                    "feature {feature:?} must self-mark"
                );
            }
        }
        // Chunks in a scan window mark exactly the features that landed
        // inside it (cell features can fall outside the window).
        let mut features = 0;
        for cx in -100..100 {
            for cz in -100..100 {
                if grid.is_feature_chunk(42, cx, cz) {
                    features += 1;
                    assert!(marked.contains(&(cx, cz)));
                }
            }
        }
        let expected = marked
            .iter()
            .filter(|&&(x, z)| (-100..100).contains(&x) && (-100..100).contains(&z))
            .count();
        assert_eq!(features, expected);
    }

    #[test]
    fn well_pieces_shape() {
        let reg = registry();
        let well = WellBlocks::from_registry(&reg).unwrap();
        let volumes = well_volumes(&well, (100, -200), 70, 2);
        // Floor first, then the water basin, then four rim strips per layer.
        assert_eq!(volumes.len(), 2 + 4 * 2);
        assert_eq!(volumes[0].min, (98, 70, -202));
        assert_eq!(volumes[0].size, (5, 1, 5));
        assert_eq!(volumes[0].state, well.sandstone);
        assert_eq!(volumes[1].min, (99, 70, -201));
        assert_eq!(volumes[1].size, (3, 1, 3));
        assert_eq!(volumes[1].state, well.water);
        for (i, volume) in volumes.iter().enumerate() {
            let max = (
                volume.min.0 + volume.size.0 as i32,
                volume.min.1 + volume.size.1 as i32,
                volume.min.2 + volume.size.2 as i32,
            );
            // Everything stays within two blocks of the center column.
            assert!(
                (98..=103).contains(&volume.min.0) && max.0 <= 103,
                "volume {i}"
            );
            assert!(
                (70..=73).contains(&volume.min.1) && max.1 <= 73,
                "volume {i}"
            );
            assert!(
                (-203..=-197).contains(&volume.min.2) && max.2 <= -197,
                "volume {i}"
            );
        }
    }

    #[test]
    fn write_volume_clips_at_chunk_and_world_borders() {
        let reg = registry();
        let well = WellBlocks::from_registry(&reg).unwrap();
        let air = reg.state_id("minecraft:air", "").unwrap();
        // A box straddling the corner between chunks (0,0), (1,0), (0,1),
        // (1,1) and poking below the floor and above the ceiling.
        let volume = BlockVolume {
            min: (12, MIN_Y - 1, 12),
            size: (8, WORLD_LAYERS as u16 + 4, 8),
            state: well.sandstone,
        };
        let mut blocks = vec![air; WORLD_LAYERS * HEIGHTMAP_CELLS];
        write_volume(&mut blocks, 0, 0, &volume);
        let solid = blocks.iter().filter(|&&b| b != air).count();
        // Chunk-local x/z clipped to 4x4 columns; y clipped to the world.
        assert_eq!(solid, 4 * 4 * WORLD_LAYERS);
        // The straddling cells belong to the neighbor's buffer, not ours.
        let mut neighbor = vec![air; WORLD_LAYERS * HEIGHTMAP_CELLS];
        write_volume(&mut neighbor, 1, 1, &volume);
        assert_eq!(
            neighbor.iter().filter(|&&b| b != air).count(),
            4 * 4 * WORLD_LAYERS
        );
    }

    fn unpack_section(chunk: &WireChunk, section: usize) -> Vec<u32> {
        match &chunk.sections[section].block_states {
            Container::Single(v) => vec![*v; 4096],
            Container::Palette {
                entries,
                longs,
                bits,
            } => unpack(longs, *bits as usize, 4096)
                .into_iter()
                .map(|i| entries[i as usize])
                .collect(),
            Container::Global { .. } => panic!("unexpected global container"),
        }
    }

    fn block_at(chunk: &WireChunk, x: i32, y: i32, z: i32) -> u32 {
        let section = (y.div_euclid(16) + 4) as usize;
        let cells = unpack_section(chunk, section);
        let ly = y.rem_euclid(16) as usize;
        let lz = z.rem_euclid(16) as usize;
        let lx = x.rem_euclid(16) as usize;
        cells[ly * 256 + lz * 16 + lx]
    }

    /// Feature chunk (-347, 388) at seed 42 lands its well on high ground
    /// with the center column at x=-5537 (local 15), so the rim crosses
    /// into chunk (-346, 388).
    #[test]
    fn well_places_and_clips_across_the_chunk_border() {
        let reg = registry();
        let terrain = HeightmapGenerator::with_seed(42, &reg).unwrap();
        let well = WellBlocks::from_registry(&reg).unwrap();
        let a = generate_chunk(&terrain, &well, 42, -347, 388).unwrap();
        let b = generate_chunk(&terrain, &well, 42, -346, 388).unwrap();

        let sandstone = well.sandstone;
        let water = well.water;
        // The basin center at (-5537, 6213), base = column top there; this
        // piece draws one rim layer and the floor crosses x=-5536 into
        // chunk -346.
        let base = terrain.column_top(-5537, 6213);
        assert!(base > SEA_LEVEL, "test target is on land");
        assert_eq!(block_at(&a, -5537, base, 6213), water, "basin center");
        assert_eq!(block_at(&a, -5538, base, 6213), water, "basin interior");
        assert_eq!(block_at(&a, -5539, base, 6213), sandstone, "floor ring");
        assert_eq!(block_at(&a, -5539, base + 1, 6213), sandstone, "rim layer");
        assert_ne!(
            block_at(&a, -5539, base + 2, 6213),
            sandstone,
            "single rim layer"
        );
        assert_eq!(
            block_at(&b, -5535, base, 6213),
            sandstone,
            "floor spills over"
        );
        assert_eq!(block_at(&b, -5535, base + 1, 6213), sandstone, "east rim");
        // The far side of the neighbor is untouched terrain: the clipped
        // volume wrote nothing beyond the well edge.
        assert_ne!(block_at(&b, -5534, base + 1, 6213), sandstone);
        // The basin interior opens to the sky above the water.
        assert_eq!(block_at(&a, -5537, base + 1, 6213), 0);
    }

    #[test]
    fn chunk_generation_is_order_independent() {
        let reg = registry();
        let terrain = HeightmapGenerator::with_seed(42, &reg).unwrap();
        let well = WellBlocks::from_registry(&reg).unwrap();
        let area: Vec<(i32, i32)> = (50..=53)
            .flat_map(|cx| (14..=17).map(move |cz| (cx, cz)))
            .collect();
        let forward: Vec<WireChunk> = area
            .iter()
            .map(|&(cx, cz)| generate_chunk(&terrain, &well, 42, cx, cz).unwrap())
            .collect();
        let backward: Vec<WireChunk> = area
            .iter()
            .rev()
            .map(|&(cx, cz)| generate_chunk(&terrain, &well, 42, cx, cz).unwrap())
            .collect();
        for (a, b) in forward.iter().zip(backward.iter().rev()) {
            assert_eq!(a, b);
        }
        // A lone chunk matches its in-context twin byte for byte.
        let lone = generate_chunk(&terrain, &well, 42, 51, 15).unwrap();
        let in_context = &forward[area.iter().position(|&c| c == (51, 15)).unwrap()];
        assert_eq!(&lone.encode(), &in_context.encode());
    }
}
