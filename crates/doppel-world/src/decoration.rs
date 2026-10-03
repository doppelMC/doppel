//! Feature decoration: the per-chunk feature step that runs after
//! terrain and structures.
//!
//! A 3x3 chunk region holds the block buffers features write into. Each
//! chunk decorates once: the decoration seed derives from the world seed
//! and the chunk origin, every feature reseeds from its step and index
//! before placing, and a placement stack of modifiers (count, rarity,
//! square spread, heightmap snap, biome gate) narrows the chunk origin
//! down to the positions the feature itself then fills.
//!
//! Feature order comes from the parameter-list biomes: a topological
//! walk over each biome's feature list keeps every biome's ordering
//! intact while interleaving features across biomes, exactly the order
//! the reference sorter builds.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::biome::BiomeTable;
use crate::chunk_codec::WireChunk;
use crate::features;
use crate::noise::Xoroshiro;
use crate::registry::BlockRegistry;
use crate::structures::{well_volumes_near, write_volume, WellBlocks};
use crate::terrain::{HeightmapGenerator, SectionBiomes};
use crate::worldgen::MIN_Y;

/// Chunk sections over the world height (-64..320).
const SECTION_SPAN: usize = 24;
/// Blocks per chunk edge.
const EDGE: i32 = 16;
/// Columns per chunk.
const COLUMNS: usize = 256;
/// World layers (384).
const LAYERS: usize = SECTION_SPAN * 16;

// ---------------------------------------------------------------------------
// The decoration random.
// ---------------------------------------------------------------------------

/// The decoration random: the rotate-xor stream reseeded per stage, with
/// every draw sliced from the high bits of one stream step.
pub struct DecorRng {
    rng: Xoroshiro,
    /// Stream steps consumed so far; reseed does not clear it.
    pub(crate) words: u64,
}

impl Default for DecorRng {
    fn default() -> DecorRng {
        DecorRng::new()
    }
}

impl DecorRng {
    pub fn new() -> DecorRng {
        DecorRng {
            rng: Xoroshiro::new(-7046029254386353131, 0),
            words: 0,
        }
    }

    fn set_seed(&mut self, seed: i64) {
        self.rng.set_seed_wide(seed);
    }

    /// One draw slice: the high `bits` bits of one stream step.
    fn next(&mut self, bits: u32) -> i32 {
        self.words += 1;
        (self.rng.next_long() as u64 >> (64 - bits)) as u32 as i32
    }

    pub fn next_long(&mut self) -> i64 {
        let upper = self.next(32) as i64;
        let lower = self.next(32) as i64;
        (upper << 32).wrapping_add(lower)
    }

    /// The modulo draw with rejection for bias; power-of-two bounds take
    /// the scaled product path.
    pub fn next_int(&mut self, bound: i32) -> i32 {
        debug_assert!(bound > 0);
        if bound & bound.wrapping_sub(1) == 0 {
            return ((bound as i64 * self.next(31) as i64) >> 31) as i32;
        }
        loop {
            let sample = self.next(31);
            let value = sample % bound;
            if sample.wrapping_sub(value).wrapping_add(bound - 1) >= 0 {
                return value;
            }
        }
    }

    pub fn next_f32(&mut self) -> f32 {
        self.next(24) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// The boolean draw: the high bit of one stream step.
    pub fn next_bool(&mut self) -> bool {
        self.next(1) != 0
    }

    /// The per-chunk decoration seed: two odd draws scale the chunk
    /// origin, the world seed folds in, and the stream reseeds from the
    /// result. Input coordinates are block coordinates.
    pub fn decoration_seed(&mut self, world_seed: i64, block_x: i32, block_z: i32) -> i64 {
        self.set_seed(world_seed);
        let x_scale = self.next_long() | 1;
        let z_scale = self.next_long() | 1;
        let seed = ((block_x as i64)
            .wrapping_mul(x_scale)
            .wrapping_add((block_z as i64).wrapping_mul(z_scale)))
            ^ world_seed;
        self.set_seed(seed);
        seed
    }

    /// The per-feature reseed: the feature's index within its step and
    /// the step itself salt the decoration seed.
    pub fn set_feature_seed(&mut self, decoration_seed: i64, index: i32, step: i32) {
        self.set_seed(
            decoration_seed
                .wrapping_add(index as i64)
                .wrapping_add((10_000i64).wrapping_mul(step as i64)),
        );
    }
}

// ---------------------------------------------------------------------------
// Integer draws for count and offset modifiers.
// ---------------------------------------------------------------------------

/// An integer draw: constant draws nothing; the others consume the
/// stream in fixed shapes.
#[derive(Clone)]
pub(crate) enum IntDraw {
    Constant(i32),
    Uniform {
        min: i32,
        max: i32,
    },
    Trapezoid {
        min: i32,
        max: i32,
        plateau: i32,
    },
    Biased {
        min: i32,
        max: i32,
    },
    Clamped {
        min: i32,
        max: i32,
        source: Box<IntDraw>,
    },
    Weighted {
        total: i32,
        entries: Vec<(i32, IntDraw)>,
    },
    /// Recognized but never sampled: the modifier carrying one is
    /// unsupported and its feature skips.
    Unsupported,
}

impl IntDraw {
    pub(crate) fn parse(v: &Value) -> Result<IntDraw> {
        // A bare integer is the constant draw (the reference codec accepts
        // the literal form everywhere a provider fits).
        if let Some(n) = v.as_i64() {
            return Ok(IntDraw::Constant(n as i32));
        }
        let kind = v.get("type").and_then(Value::as_str).context("draw type")?;
        let int = |v: &Value, key: &str| -> Result<i32> {
            v.get(key)
                .and_then(Value::as_i64)
                .with_context(|| format!("draw field {key}"))
                .map(|n| n as i32)
        };
        match kind {
            "minecraft:constant" => Ok(IntDraw::Constant(int(v, "value")?)),
            "minecraft:uniform" => Ok(IntDraw::Uniform {
                min: int(v, "min_inclusive")?,
                max: int(v, "max_inclusive")?,
            }),
            "minecraft:trapezoid" => Ok(IntDraw::Trapezoid {
                min: int(v, "min")?,
                max: int(v, "max")?,
                plateau: int(v, "plateau")?,
            }),
            "minecraft:biased_to_bottom" => Ok(IntDraw::Biased {
                min: int(v, "min_inclusive")?,
                max: int(v, "max_inclusive")?,
            }),
            "minecraft:clamped" => Ok(IntDraw::Clamped {
                min: int(v, "min_inclusive")?,
                max: int(v, "max_inclusive")?,
                source: Box::new(IntDraw::parse(v.get("source").context("clamped source")?)?),
            }),
            "minecraft:weighted_list" => {
                let mut entries = Vec::new();
                for entry in v
                    .get("distribution")
                    .and_then(Value::as_array)
                    .context("weighted distribution")?
                {
                    entries.push((
                        entry
                            .get("weight")
                            .and_then(Value::as_i64)
                            .context("weight")? as i32,
                        IntDraw::parse(entry.get("data").context("weighted data")?)?,
                    ));
                }
                let total: i32 = entries.iter().map(|(w, _)| w).sum();
                if total <= 0 {
                    bail!("weighted draw needs positive weight");
                }
                Ok(IntDraw::Weighted { total, entries })
            }
            "minecraft:clamped_normal" => Ok(IntDraw::Unsupported),
            other => bail!("unsupported integer draw {other}"),
        }
    }

    pub(crate) fn sample(&self, rng: &mut DecorRng) -> i32 {
        match self {
            IntDraw::Constant(value) => *value,
            IntDraw::Uniform { min, max } => rng.next_int(max - min + 1) + min,
            IntDraw::Trapezoid { min, max, plateau } => {
                if *plateau == 0 && *max == -*min {
                    return rng.next_int(max + 1) - rng.next_int(max + 1);
                }
                let range = max - min;
                if plateau == &range {
                    return rng.next_int(range + 1) + min;
                }
                let plateau_start = (range - plateau) / 2;
                let plateau_end = range - plateau_start;
                min + rng.next_int(plateau_end + 1) + rng.next_int(plateau_start + 1)
            }
            IntDraw::Biased { min, max } => {
                let inner = rng.next_int(max - min + 1);
                min + rng.next_int(inner + 1)
            }
            IntDraw::Clamped { min, max, source } => source.sample(rng).clamp(*min, *max),
            IntDraw::Weighted { total, entries } => {
                let mut pick = rng.next_int(*total);
                for (weight, draw) in entries {
                    if pick < *weight {
                        return draw.sample(rng);
                    }
                    pick -= weight;
                }
                entries[entries.len() - 1].1.sample(rng)
            }
            IntDraw::Unsupported => {
                debug_assert!(false, "unsupported draw sampled");
                0
            }
        }
    }

    /// Whether the draw never samples (its modifier is unsupported).
    pub(crate) fn is_unsupported(&self) -> bool {
        matches!(self, IntDraw::Unsupported)
    }
}

// ---------------------------------------------------------------------------
// Placement modifiers.
// ---------------------------------------------------------------------------

/// The heightmap kinds the placement stack snaps to. The wire carries
/// worldgen and final variants per surface: during decoration the final
/// maps track every feature write while the worldgen maps stay frozen
/// at the terrain snapshot.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum HeightKind {
    WorldSurface,
    WorldSurfaceWg,
    OceanFloor,
    OceanFloorWg,
    MotionBlocking,
    MotionBlockingNoLeaves,
}

impl HeightKind {
    fn parse(name: &str) -> Result<HeightKind> {
        match name {
            "WORLD_SURFACE" => Ok(HeightKind::WorldSurface),
            "WORLD_SURFACE_WG" => Ok(HeightKind::WorldSurfaceWg),
            "OCEAN_FLOOR" => Ok(HeightKind::OceanFloor),
            "OCEAN_FLOOR_WG" => Ok(HeightKind::OceanFloorWg),
            "MOTION_BLOCKING" => Ok(HeightKind::MotionBlocking),
            "MOTION_BLOCKING_NO_LEAVES" => Ok(HeightKind::MotionBlockingNoLeaves),
            other => bail!("unsupported heightmap {other}"),
        }
    }

    /// Whether feature writes leave this kind untouched (only the worldgen
    /// maps freeze; the final maps live through decoration).
    fn frozen(self) -> bool {
        matches!(self, HeightKind::WorldSurfaceWg | HeightKind::OceanFloorWg)
    }
}

/// A vertical anchor: absolute world y or offsets from the build floor
/// and ceiling.
#[derive(Clone, Copy)]
enum Anchor {
    Absolute(i32),
    AboveBottom(i32),
    BelowTop(i32),
}

impl Anchor {
    fn parse(v: &Value) -> Result<Anchor> {
        let Some(fields) = v.as_object() else {
            bail!("anchor is not an object");
        };
        if fields.len() != 1 {
            bail!("anchor needs exactly one field, found {}", fields.len());
        }
        if let Some(n) = v.get("absolute").and_then(Value::as_i64) {
            return Ok(Anchor::Absolute(n as i32));
        }
        if let Some(n) = v.get("above_bottom").and_then(Value::as_i64) {
            return Ok(Anchor::AboveBottom(n as i32));
        }
        if let Some(n) = v.get("below_top").and_then(Value::as_i64) {
            return Ok(Anchor::BelowTop(n as i32));
        }
        bail!("unsupported anchor")
    }

    fn resolve(&self, min_y: i32, height: i32) -> i32 {
        match *self {
            Anchor::Absolute(y) => y,
            Anchor::AboveBottom(off) => min_y + off,
            Anchor::BelowTop(off) => min_y + height - off,
        }
    }
}

/// A block predicate for placement filters and feature checks.
#[derive(Clone)]
pub(crate) enum Predicate {
    Blocks {
        names: Vec<String>,
        offset: [i32; 3],
    },
    Tag {
        tag: String,
        offset: [i32; 3],
    },
    Fluids {
        names: Vec<String>,
        offset: [i32; 3],
    },
    /// The named state's block must survive at the position.
    Survive {
        name: String,
    },
    Replaceable,
    AllOf(Vec<Predicate>),
    AnyOf(Vec<Predicate>),
    Not(Box<Predicate>),
    True,
    /// Recognized but never true: the filter leg carrying one refuses.
    Unsupported,
}

fn predicate_offset(v: &Value) -> [i32; 3] {
    let mut offset = [0i32; 3];
    if let Some(list) = v.get("offset").and_then(Value::as_array) {
        for (slot, n) in list.iter().take(3).enumerate() {
            offset[slot] = n.as_i64().unwrap_or(0) as i32;
        }
    }
    offset
}

fn name_list(v: &Value, key: &str) -> Vec<String> {
    match v.get(key) {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(list)) => list
            .iter()
            .filter_map(Value::as_str)
            .map(String::from)
            .collect(),
        _ => Vec::new(),
    }
}

impl Predicate {
    pub(crate) fn parse(v: &Value) -> Result<Predicate> {
        let kind = v.get("type").and_then(Value::as_str).context("predicate")?;
        match kind {
            "minecraft:matching_blocks" => Ok(Predicate::Blocks {
                names: name_list(v, "blocks"),
                offset: predicate_offset(v),
            }),
            "minecraft:matching_block_tag" => {
                let tag = v
                    .get("tag")
                    .and_then(Value::as_str)
                    .context("predicate tag")?;
                let tag = tag.strip_prefix("minecraft:").unwrap_or(tag);
                Ok(Predicate::Tag {
                    tag: tag.to_string(),
                    offset: predicate_offset(v),
                })
            }
            "minecraft:matching_fluids" => Ok(Predicate::Fluids {
                names: name_list(v, "fluids"),
                offset: predicate_offset(v),
            }),
            "minecraft:would_survive" => {
                let state = v
                    .get("state")
                    .and_then(Value::as_str)
                    .context("survive state")?;
                let (name, _) = BlockRegistry::split_state(state);
                Ok(Predicate::Survive {
                    name: name.to_string(),
                })
            }
            "minecraft:replaceable" => Ok(Predicate::Replaceable),
            "minecraft:all_of" | "minecraft:any_of" => {
                let inner: Vec<Predicate> = v
                    .get("predicates")
                    .and_then(Value::as_array)
                    .context("predicate list")?
                    .iter()
                    .map(Predicate::parse)
                    .collect::<Result<_>>()?;
                if kind == "minecraft:all_of" {
                    Ok(Predicate::AllOf(inner))
                } else {
                    Ok(Predicate::AnyOf(inner))
                }
            }
            "minecraft:not" => Ok(Predicate::Not(Box::new(Predicate::parse(
                v.get("predicate").context("not predicate")?,
            )?))),
            "minecraft:true" => Ok(Predicate::True),
            "minecraft:has_sturdy_face" | "minecraft:solid" | "minecraft:volume_match" => {
                Ok(Predicate::Unsupported)
            }
            other => bail!("unsupported predicate {other}"),
        }
    }
}

/// One modifier in a placed feature's placement stack.
#[derive(Clone)]
enum Modifier {
    Count(IntDraw),
    Rarity(i32),
    InSquare,
    Heightmap(HeightKind),
    Biome,
    HeightRange {
        min: Anchor,
        max: Anchor,
    },
    Offset(IntDraw, IntDraw, IntDraw),
    WaterDepth(i32),
    SurfaceRelative {
        kind: HeightKind,
        min: i32,
        max: i32,
    },
    Filter(Predicate),
    /// Recognized but not evaluated: features carrying one are skipped.
    Unsupported,
}

impl Modifier {
    fn parse(v: &Value) -> Result<Modifier> {
        let kind = v.get("type").and_then(Value::as_str).context("modifier")?;
        match kind {
            "minecraft:count" => {
                let draw = IntDraw::parse(v.get("count").context("count")?)?;
                if draw.is_unsupported() {
                    return Ok(Modifier::Unsupported);
                }
                Ok(Modifier::Count(draw))
            }
            "minecraft:rarity_filter" => Ok(Modifier::Rarity(
                v.get("chance")
                    .and_then(Value::as_i64)
                    .context("rarity chance")? as i32,
            )),
            "minecraft:in_square" => Ok(Modifier::InSquare),
            "minecraft:heightmap" => Ok(Modifier::Heightmap(HeightKind::parse(
                v.get("heightmap")
                    .and_then(Value::as_str)
                    .context("heightmap")?,
            )?)),
            "minecraft:biome" => Ok(Modifier::Biome),
            "minecraft:height_range" => {
                let height = v.get("height").context("height range")?;
                let shape = height.get("type").and_then(Value::as_str);
                if shape != Some("minecraft:uniform") {
                    return Ok(Modifier::Unsupported);
                }
                Ok(Modifier::HeightRange {
                    min: Anchor::parse(height.get("min_inclusive").context("range min")?)?,
                    max: Anchor::parse(height.get("max_inclusive").context("range max")?)?,
                })
            }
            "minecraft:offset" => {
                let draws = [
                    IntDraw::parse(v.get("x").context("offset x")?)?,
                    IntDraw::parse(v.get("y").context("offset y")?)?,
                    IntDraw::parse(v.get("z").context("offset z")?)?,
                ];
                if draws.iter().any(IntDraw::is_unsupported) {
                    return Ok(Modifier::Unsupported);
                }
                let [x, y, z] = draws;
                Ok(Modifier::Offset(x, y, z))
            }
            "minecraft:surface_water_depth_filter" => Ok(Modifier::WaterDepth(
                v.get("max_water_depth")
                    .and_then(Value::as_i64)
                    .context("water depth")? as i32,
            )),
            "minecraft:surface_relative_threshold_filter" => Ok(Modifier::SurfaceRelative {
                kind: HeightKind::parse(
                    v.get("heightmap")
                        .and_then(Value::as_str)
                        .context("heightmap")?,
                )?,
                min: v
                    .get("min_inclusive")
                    .and_then(Value::as_i64)
                    .unwrap_or(i32::MIN as i64) as i32,
                max: v
                    .get("max_inclusive")
                    .and_then(Value::as_i64)
                    .unwrap_or(i32::MAX as i64) as i32,
            }),
            "minecraft:block_predicate_filter" => Ok(Modifier::Filter(Predicate::parse(
                v.get("predicate").context("filter predicate")?,
            )?)),
            "minecraft:environment_scan"
            | "minecraft:count_on_every_layer"
            | "minecraft:noise_based_count"
            | "minecraft:noise_threshold_count"
            | "minecraft:random_chance"
            | "minecraft:randomly_selected"
            | "minecraft:cuboid"
            | "minecraft:fixed_placement" => Ok(Modifier::Unsupported),
            other => bail!("unsupported placement modifier {other}"),
        }
    }
}

/// A placed feature: the feature config plus its placement stack.
#[derive(Clone)]
pub struct PlacedFeatureCfg {
    pub feature: Value,
    placement: Vec<Modifier>,
}

impl PlacedFeatureCfg {
    fn load(pins: &Path, key: &str) -> Result<PlacedFeatureCfg> {
        let path = pins.join("placed_feature").join(format!("{key}.json"));
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let v: Value =
            serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
        PlacedFeatureCfg::from_value(&v)
    }

    /// Parses a placed feature from its json object (the pinned files and
    /// the inline references share the shape).
    pub(crate) fn from_value(v: &Value) -> Result<PlacedFeatureCfg> {
        let placement = v
            .get("placement")
            .and_then(Value::as_array)
            .map(|stack| {
                stack
                    .iter()
                    .map(Modifier::parse)
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        Ok(PlacedFeatureCfg {
            feature: v.get("feature").cloned().unwrap_or(Value::Null),
            placement,
        })
    }

    /// Whether the placement stack carries a modifier this engine skips.
    pub fn has_unsupported(&self) -> bool {
        self.placement
            .iter()
            .any(|m| matches!(m, Modifier::Unsupported))
    }
}

// ---------------------------------------------------------------------------
// Feature order: the topological walk over biome feature lists.
// ---------------------------------------------------------------------------

/// The decoration plan: per-step feature order and each biome's feature
/// sets for gating.
pub struct FeaturePlan {
    /// Per step, the feature keys in placement order (the key indexes
    /// `names`).
    pub steps: Vec<Vec<usize>>,
    /// Feature key -> placed feature id (namespace stripped for pins).
    pub names: Vec<String>,
    /// Biome wire id -> per step, the feature keys the biome lists.
    biome_keys: HashMap<u32, Vec<Vec<usize>>>,
    /// Feature key -> its step and position within that step's order.
    positions: HashMap<usize, (u32, usize)>,
}

impl FeaturePlan {
    /// Builds the plan over the parameter-list biomes in generation
    /// order: every biome's feature list contributes its ordering, a
    /// depth-first walk over the joined constraints produces one order
    /// that satisfies them all, and features land in their step buckets.
    pub fn build(pins: &Path, table: &BiomeTable) -> Result<FeaturePlan> {
        let order: Vec<String> = serde_json::from_str(
            &std::fs::read_to_string(pins.join("biome_registry_order.json"))
                .context("registry order file")?,
        )
        .context("registry order json")?;
        let key_of = |biome: u32| -> String {
            let name = order
                .get(biome as usize)
                .cloned()
                .unwrap_or_else(|| biome.to_string());
            name.strip_prefix("minecraft:").unwrap_or(&name).to_string()
        };

        // Global keys in first-encounter order, then the edge graph.
        let mut names: Vec<String> = Vec::new();
        let mut keys: HashMap<String, usize> = HashMap::new();
        let mut biome_lists: Vec<(u32, Vec<Vec<usize>>)> = Vec::new();
        let mut max_step = 0usize;
        for biome in table.biome_order() {
            let key = key_of(biome);
            let path = pins.join("biome").join(format!("{key}.json"));
            let raw = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            let v: Value = serde_json::from_str(&raw)
                .with_context(|| format!("parsing {}", path.display()))?;
            let steps = v
                .get("features")
                .and_then(Value::as_array)
                .context("biome features")?;
            max_step = max_step.max(steps.len());
            let mut per_step: Vec<Vec<usize>> = Vec::new();
            for list in steps.iter() {
                let mut step_keys = Vec::new();
                for entry in list.as_array().context("feature step list")? {
                    let id = entry.as_str().context("placed feature id")?;
                    let key = id.strip_prefix("minecraft:").unwrap_or(id);
                    let next = *keys.entry(key.to_string()).or_insert_with(|| {
                        names.push(key.to_string());
                        names.len() - 1
                    });
                    step_keys.push(next);
                }
                per_step.push(step_keys);
            }
            biome_lists.push((biome, per_step));
        }
        let lists: Vec<Vec<Vec<usize>>> =
            biome_lists.iter().map(|(_, lists)| lists.clone()).collect();
        let steps = order_features(&lists, max_step)?;
        let mut positions: HashMap<usize, (u32, usize)> = HashMap::new();
        for (step, list) in steps.iter().enumerate() {
            for (i, &key) in list.iter().enumerate() {
                positions.insert(key, (step as u32, i));
            }
        }

        let mut biome_keys: HashMap<u32, Vec<Vec<usize>>> = HashMap::new();
        for (biome, per_step) in biome_lists {
            biome_keys.insert(biome, per_step);
        }
        Ok(FeaturePlan {
            steps,
            names,
            biome_keys,
            positions,
        })
    }

    /// The biome's features for one step, as indices into that step's
    /// order, ascending (the driver reseeds in this order).
    pub fn step_indices(&self, biome: u32, step: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = self
            .biome_keys
            .get(&biome)
            .and_then(|lists| lists.get(step))
            .map(|keys| {
                keys.iter()
                    .filter_map(|&k| self.positions.get(&k).filter(|(s, _)| *s as usize == step))
                    .map(|&(_, i)| i)
                    .collect()
            })
            .unwrap_or_default();
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    /// Whether the biome lists the feature (of any step) at all: the
    /// biome gate every position passes through.
    pub fn biome_has_feature(&self, biome: u32, key: usize) -> bool {
        self.biome_keys
            .get(&biome)
            .is_some_and(|lists| lists.iter().any(|list| list.contains(&key)))
    }
}

/// Joins the biome feature lists into per-step orders: every feature of
/// every biome becomes a graph node (list tails carry empty successor
/// sets but still anchor their place in the start iteration), edges run
/// from each feature to its successor, a depth-first walk over the
/// joined graph (visiting in step, then first-encounter, order) emits
/// post-order, and reversing gives an order every biome's list agrees
/// with.
fn order_features(lists: &[Vec<Vec<usize>>], max_step: usize) -> Result<Vec<Vec<usize>>> {
    let mut edges: BTreeMap<(u32, usize), BTreeSet<(u32, usize)>> = BTreeMap::new();
    for list in lists {
        let flat: Vec<(u32, usize)> = list
            .iter()
            .enumerate()
            .flat_map(|(step, keys)| keys.iter().map(move |&k| (step as u32, k)))
            .collect();
        for (i, node) in flat.iter().enumerate() {
            let successors = edges.entry(*node).or_default();
            if let Some(&next) = flat.get(i + 1) {
                successors.insert(next);
            }
        }
    }
    let mut discovered: BTreeSet<(u32, usize)> = BTreeSet::new();
    let mut visiting: BTreeSet<(u32, usize)> = BTreeSet::new();
    let mut ordered: Vec<(u32, usize)> = Vec::new();
    let starts: Vec<(u32, usize)> = edges.keys().copied().collect();
    for start in starts {
        if !depth_first(&edges, &mut discovered, &mut visiting, &mut ordered, start)? {
            bail!("feature order cycle at {start:?}");
        }
    }
    ordered.reverse();
    let mut steps: Vec<Vec<usize>> = vec![Vec::new(); max_step];
    for &(step, key) in &ordered {
        steps[step as usize].push(key);
    }
    Ok(steps)
}

/// One depth-first pass; false means the walk hit a cycle.
fn depth_first(
    edges: &BTreeMap<(u32, usize), BTreeSet<(u32, usize)>>,
    discovered: &mut BTreeSet<(u32, usize)>,
    visiting: &mut BTreeSet<(u32, usize)>,
    ordered: &mut Vec<(u32, usize)>,
    current: (u32, usize),
) -> Result<bool> {
    if discovered.contains(&current) {
        return Ok(true);
    }
    if visiting.contains(&current) {
        return Ok(false);
    }
    visiting.insert(current);
    if let Some(successors) = edges.get(&current) {
        for next in successors {
            if !depth_first(edges, discovered, visiting, ordered, *next)? {
                return Ok(false);
            }
        }
    }
    visiting.remove(&current);
    discovered.insert(current);
    ordered.push(current);
    Ok(true)
}

// ---------------------------------------------------------------------------
// Block state traits (heightmap kinds).
// ---------------------------------------------------------------------------

/// Heightmap membership per state, memoized: air, leaves, fluid, the
/// two motion tags.
#[derive(Default)]
struct StateTraits {
    memo: HashMap<u32, StateFlags>,
}

#[derive(Clone, Copy, Default)]
struct StateFlags {
    air: bool,
    leaves: bool,
    fluid: bool,
    motion: bool,
    motion_no_leaves: bool,
}

impl StateTraits {
    fn flags(
        &mut self,
        registry: &BlockRegistry,
        tags: &mut TagResolver,
        state: u32,
    ) -> StateFlags {
        if let Some(f) = self.memo.get(&state) {
            return *f;
        }
        let name = registry
            .state_of(state)
            .map(|(name, _)| name.to_string())
            .unwrap_or_default();
        let f = StateFlags {
            air: tags.contains("air", &name),
            leaves: tags.contains("leaves", &name),
            fluid: name == "minecraft:water" || name == "minecraft:lava",
            motion: false,
            motion_no_leaves: tags.contains("blocks_motion_no_leaves", &name),
        };
        let f = StateFlags {
            motion: f.motion_no_leaves || f.leaves,
            ..f
        };
        self.memo.insert(state, f);
        f
    }

    fn counts(kind: HeightKind, f: StateFlags) -> bool {
        match kind {
            HeightKind::WorldSurface | HeightKind::WorldSurfaceWg => !f.air,
            HeightKind::OceanFloor | HeightKind::OceanFloorWg => f.motion,
            HeightKind::MotionBlocking => f.motion || f.fluid,
            HeightKind::MotionBlockingNoLeaves => f.motion_no_leaves || f.fluid,
        }
    }
}

/// Block tag pins resolved on demand; unlisted nested tags resolve empty.
struct TagResolver {
    dir: std::path::PathBuf,
    cache: HashMap<String, Vec<String>>,
}

impl TagResolver {
    fn new(pins: &Path) -> TagResolver {
        TagResolver {
            dir: pins.join("tags").join("block"),
            cache: HashMap::new(),
        }
    }

    fn entries(&mut self, tag: &str) -> Vec<String> {
        if let Some(v) = self.cache.get(tag) {
            return v.clone();
        }
        let path = self.dir.join(format!("{tag}.json"));
        let values: Vec<String> = std::fs::read_to_string(&path)
            .ok()
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
            .and_then(|v| {
                v.get("values").and_then(Value::as_array).map(|a| {
                    a.iter()
                        .filter_map(|e| e.as_str().map(String::from))
                        .collect()
                })
            })
            .unwrap_or_default();
        self.cache.insert(tag.to_string(), values.clone());
        values
    }

    fn contains(&mut self, tag: &str, name: &str) -> bool {
        for entry in self.entries(tag) {
            if let Some(nested) = entry.strip_prefix('#') {
                let nested = nested.strip_prefix("minecraft:").unwrap_or(nested);
                if self.contains(nested, name) {
                    return true;
                }
            } else if entry == name {
                return true;
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// The decoration region.
// ---------------------------------------------------------------------------

/// One chunk inside the region: block buffer, climate biomes, cached
/// heights per kind, and the decorate-once flag.
struct ChunkState {
    blocks: Vec<u32>,
    biomes: SectionBiomes,
    heights: HashMap<HeightKind, Vec<i32>>,
    decorated: bool,
}

/// The 3x3 decoration region and the driver. Chunks enter the region on
/// demand (terrain fill) and decorate at most once; feature writes land
/// in whatever region chunk owns the block.
pub struct Decorator<'a> {
    terrain: &'a HeightmapGenerator,
    world_seed: i64,
    plan: FeaturePlan,
    pins: std::path::PathBuf,
    registry: &'a BlockRegistry,
    well: Option<WellBlocks>,
    features: HashMap<usize, Option<PlacedFeatureCfg>>,
    named: HashMap<String, Option<PlacedFeatureCfg>>,
    feature_configs: HashMap<String, Option<Value>>,
    chunks: HashMap<(i32, i32), ChunkState>,
    traits: StateTraits,
    tags: TagResolver,
    /// Probe of the placements that reached a feature, for tests.
    #[cfg(test)]
    pub(crate) visits: Option<Vec<(String, i32, i32, i32)>>,
    /// Probe of the stream word count at each placement, for tests.
    #[cfg(test)]
    pub(crate) try_words: Option<Vec<u64>>,
    /// Probe of the ground-scatter verdicts, for tests.
    #[cfg(test)]
    pub(crate) scatter_log: Option<Vec<String>>,
    /// One plan feature decoration skips entirely, for tests.
    #[cfg(test)]
    pub(crate) skip: Option<usize>,
    /// Seed override for one plan feature: (key, index, step) the driver
    /// reseeds from instead of the plan position, for tests.
    #[cfg(test)]
    pub(crate) seed_override: Option<(usize, i32, i32)>,
    /// One step the driver runs, for tests.
    #[cfg(test)]
    pub(crate) only_step: Option<usize>,
}

impl<'a> Decorator<'a> {
    /// Builds the decorator over the terrain engine: the feature plan
    /// loads from the pinned biome lists.
    pub fn new(
        terrain: &'a HeightmapGenerator,
        registry: &'a BlockRegistry,
        world_seed: i64,
    ) -> Result<Decorator<'a>> {
        let pins = crate::density::locate_pins()?;
        let table = BiomeTable::load(&pins)?;
        let plan = FeaturePlan::build(&pins, &table)?;
        let mut dec = Decorator {
            terrain,
            world_seed,
            plan,
            tags: TagResolver::new(&pins),
            pins,
            registry,
            well: WellBlocks::from_registry(registry).ok(),
            features: HashMap::new(),
            named: HashMap::new(),
            feature_configs: HashMap::new(),
            chunks: HashMap::new(),
            traits: StateTraits::default(),
            #[cfg(test)]
            visits: None,
            #[cfg(test)]
            try_words: None,
            #[cfg(test)]
            scatter_log: None,
            #[cfg(test)]
            skip: None,
            #[cfg(test)]
            seed_override: None,
            #[cfg(test)]
            only_step: None,
        };
        dec.validate_pins()?;
        Ok(dec)
    }

    /// Loads every plan feature and validates the configs they reach: a
    /// missing pin, an unrecognized kind, or a malformed selector fails
    /// here instead of skipping silently at placement time.
    fn validate_pins(&mut self) -> Result<()> {
        for key in 0..self.plan.names.len() {
            if self.load_feature(key).is_none() {
                bail!("placed feature {} failed to load", self.plan.names[key]);
            }
        }
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for key in 0..self.plan.names.len() {
            let cfg = self
                .load_feature(key)
                .expect("loaded in the pass above")
                .clone();
            let name = self.plan.names[key].clone();
            self.validate_feature_value(&cfg.feature, &mut seen)
                .with_context(|| format!("validating placed feature {name}"))?;
        }
        Ok(())
    }

    /// Whether a pin file exists under the given directory.
    fn pin_exists(&self, dir: &str, key: &str) -> bool {
        self.pins.join(dir).join(format!("{key}.json")).exists()
    }

    /// Validates one feature config: a pinned id resolves, the kind is
    /// placed or explicitly unplaced, selectors carry well formed entries,
    /// and every nested reference recurses. References the pin snapshot
    /// flattened out of their subdirectory (ids carrying a slash) accept as
    /// unplaced; a missing flat-named pin fails the load.
    fn validate_feature_value(&mut self, v: &Value, seen: &mut BTreeSet<String>) -> Result<()> {
        match v {
            Value::String(id) => {
                let key = id.strip_prefix("minecraft:").unwrap_or(id);
                if !seen.insert(format!("feature:{key}")) {
                    return Ok(());
                }
                if !self.pin_exists("feature", key) {
                    if key.contains('/') {
                        return Ok(());
                    }
                    bail!("missing feature config {key}");
                }
                let cfg = self
                    .load_feature_config(key)
                    .with_context(|| format!("feature config {key} failed to parse"))?;
                self.validate_feature_value(&cfg, seen)
            }
            Value::Object(_) => {
                let kind = v
                    .get("type")
                    .and_then(Value::as_str)
                    .context("feature type")?;
                if features::UNPLACED_FEATURE_KINDS.contains(&kind) {
                    return Ok(());
                }
                match kind {
                    "minecraft:random_selector" => {
                        let list = v
                            .get("features")
                            .and_then(Value::as_array)
                            .context("selector list")?;
                        if list.is_empty() {
                            bail!("random selector without entries");
                        }
                        for entry in list {
                            if !entry.get("chance").is_some_and(Value::is_number) {
                                bail!("random selector entry without a numeric chance");
                            }
                            self.validate_placed_ref(
                                entry.get("feature").context("selector feature")?,
                                seen,
                            )?;
                        }
                        self.validate_placed_ref(
                            v.get("default").context("selector default")?,
                            seen,
                        )
                    }
                    "minecraft:simple_random_selector" => {
                        let list = v
                            .get("features")
                            .and_then(Value::as_array)
                            .context("selector list")?;
                        if list.is_empty() {
                            bail!("simple selector without entries");
                        }
                        for entry in list {
                            self.validate_placed_ref(entry, seen)?;
                        }
                        Ok(())
                    }
                    "minecraft:weighted_random_selector" => {
                        let list = v
                            .get("features")
                            .and_then(Value::as_array)
                            .context("selector list")?;
                        if list.is_empty() {
                            bail!("weighted selector without entries");
                        }
                        let mut total = 0i64;
                        for entry in list {
                            let weight = entry
                                .get("weight")
                                .and_then(Value::as_i64)
                                .context("weighted entry weight")?;
                            if weight <= 0 {
                                bail!("weighted entry needs a positive weight");
                            }
                            total += weight;
                            self.validate_placed_ref(
                                entry.get("data").context("weighted data")?,
                                seen,
                            )?;
                        }
                        if total > i32::MAX as i64 {
                            bail!("weighted selector total weight overflows a draw");
                        }
                        Ok(())
                    }
                    "minecraft:tree" => {
                        let unplaced_shape = |v: &Value, key: &str| {
                            v.get(key)
                                .and_then(|s| s.get("type"))
                                .and_then(Value::as_str)
                                .is_some_and(|t| features::UNPLACED_TREE_SHAPES.contains(&t))
                        };
                        let mut unplaced = unplaced_shape(v, "trunk_placer")
                            || unplaced_shape(v, "foliage_placer");
                        for deco in v
                            .get("decorators")
                            .and_then(Value::as_array)
                            .into_iter()
                            .flatten()
                        {
                            if deco
                                .get("type")
                                .and_then(Value::as_str)
                                .is_some_and(|t| features::UNPLACED_TREE_SHAPES.contains(&t))
                            {
                                unplaced = true;
                            }
                        }
                        if unplaced {
                            return Ok(());
                        }
                        if features::parse_tree(self, v).is_some() {
                            Ok(())
                        } else {
                            bail!("tree config rejected by the parser")
                        }
                    }
                    "minecraft:fallen_tree" => {
                        let unplaced = ["log_decorators", "stump_decorators"].iter().any(|key| {
                            v.get(key)
                                .and_then(Value::as_array)
                                .into_iter()
                                .flatten()
                                .any(|deco| {
                                    deco.get("type").and_then(Value::as_str).is_some_and(|t| {
                                        features::UNPLACED_TREE_SHAPES.contains(&t)
                                    })
                                })
                        });
                        if unplaced {
                            return Ok(());
                        }
                        if features::parse_fallen(self, v).is_some() {
                            Ok(())
                        } else {
                            bail!("fallen tree config rejected by the parser")
                        }
                    }
                    "minecraft:huge_brown_mushroom" | "minecraft:huge_red_mushroom" => {
                        let red = kind.ends_with("red_mushroom");
                        if features::parse_mushroom(self, v, red).is_some() {
                            Ok(())
                        } else {
                            bail!("mushroom config rejected by the parser")
                        }
                    }
                    "minecraft:simple_block" => {
                        if v.get("to_place").is_some() {
                            Ok(())
                        } else {
                            bail!("simple block without a provider")
                        }
                    }
                    "minecraft:block_column" => {
                        let direction = v
                            .get("direction")
                            .and_then(Value::as_str)
                            .context("column direction")?;
                        if direction != "up" && direction != "down" {
                            bail!("unsupported column direction {direction}");
                        }
                        let layers = v
                            .get("layers")
                            .and_then(Value::as_array)
                            .context("column layers")?;
                        if layers.is_empty() {
                            bail!("column without layers");
                        }
                        if v.get("allowed_placement").is_none() {
                            bail!("column without an allowed placement predicate");
                        }
                        Ok(())
                    }
                    "minecraft:multiface_growth" => {
                        v.get("block")
                            .and_then(Value::as_str)
                            .context("multiface block")?;
                        let surfaces = v.get("can_be_placed_on");
                        let empty = surfaces
                            .and_then(Value::as_array)
                            .is_some_and(|l| l.is_empty());
                        if surfaces.is_none() || empty {
                            bail!("multiface growth without surfaces");
                        }
                        Ok(())
                    }
                    other => bail!("unsupported feature kind {other}"),
                }
            }
            _ => bail!("unsupported feature reference"),
        }
    }

    /// Validates a placed feature reference: a pinned id loads with a
    /// placement stack that parses, an inline object parses directly, and
    /// the wrapped feature config recurses. Slash-path ids the pin snapshot
    /// flattened away accept as unplaced.
    fn validate_placed_ref(&mut self, v: &Value, seen: &mut BTreeSet<String>) -> Result<()> {
        match v {
            Value::String(id) => {
                let key = id.strip_prefix("minecraft:").unwrap_or(id);
                if !seen.insert(format!("placed:{key}")) {
                    return Ok(());
                }
                if !self.pin_exists("placed_feature", key) {
                    if key.contains('/') {
                        return Ok(());
                    }
                    bail!("missing placed feature {key}");
                }
                let cfg = self
                    .load_named(key)
                    .with_context(|| format!("placed feature {key} failed to parse"))?;
                self.validate_feature_value(&cfg.feature, seen)
            }
            Value::Object(_) => {
                let cfg = PlacedFeatureCfg::from_value(v)?;
                self.validate_feature_value(&cfg.feature, seen)
            }
            _ => bail!("unsupported placed feature reference"),
        }
    }

    /// The number of placement steps in the plan.
    pub fn step_count(&self) -> usize {
        self.plan.steps.len()
    }

    /// The name of a feature key.
    pub fn feature_name(&self, key: usize) -> &str {
        &self.plan.names[key]
    }

    /// The plan key of a placed feature id, if the plan carries it.
    pub fn key_of(&self, name: &str) -> Option<usize> {
        self.plan.names.iter().position(|n| n == name)
    }

    /// Whether the biome's generation settings list the feature (any
    /// step): the gate the biome modifier enforces.
    pub fn biome_has_feature(&self, biome: u32, key: usize) -> bool {
        self.plan.biome_has_feature(biome, key)
    }

    /// Ensures the chunk exists in the region (terrain, structure pieces,
    /// biomes): structures precede every feature that reads the chunk.
    pub(crate) fn ensure_chunk(&mut self, cx: i32, cz: i32) {
        if self.chunks.contains_key(&(cx, cz)) {
            return;
        }
        let (mut blocks, _) = self.terrain.build_blocks(cx, cz);
        // The worldgen heightmaps freeze at the post-noise snapshot:
        // structure pieces and feature writes never move them.
        let mut frozen = HashMap::new();
        for kind in [HeightKind::WorldSurfaceWg, HeightKind::OceanFloorWg] {
            let mut heights = Vec::with_capacity(COLUMNS);
            for col in 0..COLUMNS {
                heights.push(scan_height_blocks(
                    &blocks,
                    kind,
                    col,
                    self.registry,
                    &mut self.traits,
                    &mut self.tags,
                ));
            }
            frozen.insert(kind, heights);
        }
        if let Some(well) = &self.well {
            for volume in well_volumes_near(self.terrain, well, self.world_seed, cx, cz) {
                write_volume(&mut blocks, cx, cz, &volume);
            }
        }
        let biomes = self
            .terrain
            .section_biomes(cx, cz)
            .unwrap_or([[0u32; 64]; SECTION_SPAN]);
        let state = ChunkState {
            blocks,
            biomes,
            heights: frozen,
            decorated: false,
        };
        self.chunks.insert((cx, cz), state);
    }

    /// Decorates the chunk exactly once: possible biomes come from the
    /// 3x3 neighborhood's stored cells, and each candidate feature
    /// reseeds from its step and index before its placement runs.
    pub fn decorate(&mut self, cx: i32, cz: i32) {
        for dx in -1..=1 {
            for dz in -1..=1 {
                self.ensure_chunk(cx + dx, cz + dz);
            }
        }
        if self.chunks.get(&(cx, cz)).is_some_and(|c| c.decorated) {
            return;
        }
        self.chunks.get_mut(&(cx, cz)).unwrap().decorated = true;

        let mut possible: BTreeSet<u32> = BTreeSet::new();
        for dx in -1..=1 {
            for dz in -1..=1 {
                if let Some(chunk) = self.chunks.get(&(cx + dx, cz + dz)) {
                    for section in &chunk.biomes {
                        possible.extend(section.iter().copied());
                    }
                }
            }
        }

        let mut rng = DecorRng::new();
        let decoration_seed = rng.decoration_seed(self.world_seed, cx * EDGE, cz * EDGE);
        for step in 0..self.plan.steps.len() {
            #[cfg(test)]
            if self.only_step.is_some_and(|only| only != step) {
                continue;
            }
            let mut indices: BTreeSet<usize> = BTreeSet::new();
            for biome in &possible {
                indices.extend(self.plan.step_indices(*biome, step));
            }
            for index in indices {
                let key = self.plan.steps[step][index];
                #[cfg(test)]
                let (seed_index, seed_step) = match self.seed_override {
                    Some((k, i, s)) if k == key => (i, s),
                    _ => (index as i32, step as i32),
                };
                #[cfg(not(test))]
                let (seed_index, seed_step) = (index as i32, step as i32);
                rng.set_feature_seed(decoration_seed, seed_index, seed_step);
                self.place_placed(key, &mut rng, cx * EDGE, cz * EDGE);
            }
        }
    }

    /// Runs one placed feature of the plan from the chunk origin.
    fn place_placed(&mut self, key: usize, rng: &mut DecorRng, ox: i32, oz: i32) {
        #[cfg(test)]
        if self.skip == Some(key) {
            return;
        }
        let Some(cfg) = self.load_feature(key).cloned() else {
            return;
        };
        let name = self.plan.names[key].clone();
        self.eval_placement(&name, Some(key), &cfg, rng, ox, MIN_Y, oz);
    }

    /// The placement worklist: modifier entries resolve to a position
    /// list, deeper modifiers run the resolved positions depth-first, and
    /// the last modifier's outputs run the feature itself.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn eval_placement(
        &mut self,
        name: &str,
        key: Option<usize>,
        cfg: &PlacedFeatureCfg,
        rng: &mut DecorRng,
        ox: i32,
        oy: i32,
        oz: i32,
    ) {
        if cfg.has_unsupported() {
            return;
        }
        // An empty placement stack places at the position it was given.
        if cfg.placement.is_empty() {
            self.place_final(name, &cfg.feature, rng, ox, oy, oz);
            return;
        }
        let span = LAYERS as i32;
        // Worklist entries: (x, y, z, modifier index).
        let mut stack: Vec<(i32, i32, i32, usize)> = vec![(ox, oy, oz, 0)];
        let mut out: Vec<(i32, i32, i32)> = Vec::new();
        while let Some((x, y, z, index)) = stack.pop() {
            out.clear();
            match &cfg.placement[index] {
                Modifier::Count(draw) => {
                    let n = draw.sample(rng);
                    for _ in 0..n {
                        out.push((x, y, z));
                    }
                }
                Modifier::Rarity(chance) => {
                    if rng.next_f32() < 1.0f32 / (*chance as f32) {
                        out.push((x, y, z));
                    }
                }
                Modifier::InSquare => {
                    let sx = rng.next_int(16) + x;
                    let sz = rng.next_int(16) + z;
                    out.push((sx, y, sz));
                }
                Modifier::Heightmap(kind) => {
                    let h = self.height(*kind, x, z);
                    if h > MIN_Y {
                        out.push((x, h, z));
                    }
                }
                Modifier::Biome => {
                    let listed =
                        key.is_some_and(|k| self.plan.biome_has_feature(self.biome_at(x, y, z), k));
                    if listed {
                        out.push((x, y, z));
                    }
                }
                Modifier::HeightRange { min, max } => {
                    let lo = min.resolve(MIN_Y, span);
                    let hi = max.resolve(MIN_Y, span);
                    // The drawn height replaces the y axis outright.
                    out.push((x, rng.next_int(hi - lo + 1) + lo, z));
                }
                Modifier::Offset(dx, dy, dz) => {
                    out.push((x + dx.sample(rng), y + dy.sample(rng), z + dz.sample(rng)));
                }
                Modifier::WaterDepth(max) => {
                    let floor = self.height(HeightKind::OceanFloor, x, z);
                    let surface = self.height(HeightKind::WorldSurface, x, z);
                    if surface - floor <= *max {
                        out.push((x, y, z));
                    }
                }
                Modifier::SurfaceRelative { kind, min, max } => {
                    let surface = self.height(*kind, x, z);
                    if y >= surface + *min && y <= surface + *max {
                        out.push((x, y, z));
                    }
                }
                Modifier::Filter(p) => {
                    if features::test_predicate(self, p, x, y, z) {
                        out.push((x, y, z));
                    }
                }
                Modifier::Unsupported => return,
            }
            let next = index + 1;
            if next < cfg.placement.len() {
                for pos in out.iter().rev() {
                    stack.push((pos.0, pos.1, pos.2, next));
                }
            } else {
                for &(px, py, pz) in &out {
                    self.place_final(name, &cfg.feature, rng, px, py, pz);
                }
            }
        }
    }

    /// Runs the configured feature at one resolved position.
    fn place_final(
        &mut self,
        name: &str,
        feature: &Value,
        rng: &mut DecorRng,
        x: i32,
        y: i32,
        z: i32,
    ) {
        #[cfg(test)]
        if let Some(log) = self.visits.as_mut() {
            if !name.is_empty() {
                log.push((name.to_string(), x, y, z));
            }
        }
        #[cfg(not(test))]
        let _ = name;
        #[cfg(test)]
        if let Some(words) = self.try_words.as_mut() {
            words.push(rng.words);
        }
        #[cfg(test)]
        if let Some(log) = self.scatter_log.as_mut() {
            log.push(format!("try {name} ({x},{y},{z}) start w{}", rng.words));
        }
        features::run_feature(self, feature, rng, x, y, z);
        #[cfg(test)]
        if let Some(log) = self.scatter_log.as_mut() {
            log.push(format!("try {name} end w{}", rng.words));
        }
    }

    /// Loads (and caches) a placed feature config; None marks a pin that
    /// failed to load or parse.
    fn load_feature(&mut self, key: usize) -> Option<&PlacedFeatureCfg> {
        if !self.features.contains_key(&key) {
            let cfg = PlacedFeatureCfg::load(&self.pins, &self.plan.names[key]).ok();
            self.features.insert(key, cfg);
        }
        self.features.get(&key).and_then(|c| c.as_ref())
    }

    /// Loads a placed feature config by id (the inner references inside
    /// feature configs).
    pub(crate) fn load_named(&mut self, id: &str) -> Option<PlacedFeatureCfg> {
        if !self.named.contains_key(id) {
            let cfg = PlacedFeatureCfg::load(&self.pins, id).ok();
            self.named.insert(id.to_string(), cfg);
        }
        self.named.get(id).cloned().flatten()
    }

    /// Loads (and caches) a raw feature config by id: the configured
    /// feature a placed feature's feature field names.
    pub(crate) fn load_feature_config(&mut self, id: &str) -> Option<Value> {
        if !self.feature_configs.contains_key(id) {
            let cfg = std::fs::read_to_string(self.pins.join("feature").join(format!("{id}.json")))
                .ok()
                .and_then(|raw| serde_json::from_str::<Value>(&raw).ok());
            self.feature_configs.insert(id.to_string(), cfg);
        }
        self.feature_configs.get(id).cloned().flatten()
    }

    /// The climate biome at a block position, from the stored section
    /// cells.
    pub fn biome_at(&self, x: i32, y: i32, z: i32) -> u32 {
        let cx = x.div_euclid(EDGE);
        let cz = z.div_euclid(EDGE);
        let Some(chunk) = self.chunks.get(&(cx, cz)) else {
            return 0;
        };
        let lx = x.rem_euclid(EDGE) / 4;
        let lz = z.rem_euclid(EDGE) / 4;
        let section = ((y - MIN_Y) / 16).clamp(0, SECTION_SPAN as i32 - 1) as usize;
        let ly = (y - MIN_Y).rem_euclid(16) / 4;
        chunk.biomes[section][(ly * 16 + lz * 4 + lx) as usize]
    }

    /// The cached height of a kind at a column; the first query scans
    /// the chunk's columns, later writes keep the cache current.
    pub(crate) fn height(&mut self, kind: HeightKind, x: i32, z: i32) -> i32 {
        let cx = x.div_euclid(EDGE);
        let cz = z.div_euclid(EDGE);
        let column = (z.rem_euclid(EDGE) * EDGE + x.rem_euclid(EDGE)) as usize;
        if !self
            .chunks
            .get(&(cx, cz))
            .is_some_and(|c| c.heights.contains_key(&kind))
        {
            let Some(blocks) = self.chunks.get(&(cx, cz)).map(|c| c.blocks.clone()) else {
                return MIN_Y;
            };
            let mut heights = Vec::with_capacity(COLUMNS);
            for col in 0..COLUMNS {
                heights.push(scan_height_blocks(
                    &blocks,
                    kind,
                    col,
                    self.registry,
                    &mut self.traits,
                    &mut self.tags,
                ));
            }
            if let Some(chunk) = self.chunks.get_mut(&(cx, cz)) {
                chunk.heights.insert(kind, heights);
            }
        }
        self.chunks
            .get(&(cx, cz))
            .and_then(|c| c.heights.get(&kind))
            .map_or(MIN_Y, |h| h[column])
    }

    /// Writes a block into the owning region chunk and refreshes its
    /// cached heights.
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, state: u32) {
        let cx = x.div_euclid(EDGE);
        let cz = z.div_euclid(EDGE);
        let column = (z.rem_euclid(EDGE) * EDGE + x.rem_euclid(EDGE)) as usize;
        let layer = (y - MIN_Y) as usize;
        if layer >= LAYERS {
            return;
        }
        if !self.chunks.contains_key(&(cx, cz)) {
            return;
        }
        let flags = self.traits.flags(self.registry, &mut self.tags, state);
        let chunk = self.chunks.get_mut(&(cx, cz)).expect("checked above");
        chunk.blocks[layer * COLUMNS + column] = state;
        for (kind, heights) in chunk.heights.iter_mut() {
            // The worldgen maps stay frozen at the terrain snapshot; the
            // final maps track every write.
            if kind.frozen() {
                continue;
            }
            let counts = StateTraits::counts(*kind, flags);
            if counts && y + 1 > heights[column] {
                heights[column] = y + 1;
            } else if !counts && y + 1 == heights[column] {
                let blocks = chunk.blocks.clone();
                heights[column] = scan_height_blocks(
                    &blocks,
                    *kind,
                    column,
                    self.registry,
                    &mut self.traits,
                    &mut self.tags,
                );
            }
        }
    }

    /// Reads a block from the region.
    pub fn block(&self, x: i32, y: i32, z: i32) -> u32 {
        let cx = x.div_euclid(EDGE);
        let cz = z.div_euclid(EDGE);
        let column = (z.rem_euclid(EDGE) * EDGE + x.rem_euclid(EDGE)) as usize;
        let layer = (y - MIN_Y) as usize;
        self.chunks
            .get(&(cx, cz))
            .map_or(0, |c| c.blocks[layer * COLUMNS + column])
    }

    /// The block registry the region resolves states against.
    pub(crate) fn registry(&self) -> &BlockRegistry {
        self.registry
    }

    /// The block name of a state id (empty when unknown).
    pub(crate) fn block_name(&self, state: u32) -> &str {
        self.registry.state_of(state).map_or("", |(name, _)| name)
    }

    /// The state id of a block name plus property text.
    pub(crate) fn state_id_of(&self, name: &str, props: &str) -> Option<u32> {
        self.registry.state_id(name, props)
    }

    /// Whether a block tag (pinned) lists the block name.
    pub(crate) fn tag_contains(&mut self, tag: &str, name: &str) -> bool {
        self.tags.contains(tag, name)
    }

    /// Emits the chunk as a wire chunk from its current buffer.
    pub fn emit(&mut self, cx: i32, cz: i32) -> Result<WireChunk> {
        self.ensure_chunk(cx, cz);
        let chunk = self.chunks.get(&(cx, cz)).expect("ensured above");
        let blocks = chunk.blocks.clone();
        let biomes = chunk.biomes;
        self.terrain.emit_with(cx, cz, &blocks, Some(&biomes))
    }
}

fn scan_height_blocks(
    blocks: &[u32],
    kind: HeightKind,
    column: usize,
    registry: &BlockRegistry,
    traits: &mut StateTraits,
    tags: &mut TagResolver,
) -> i32 {
    for layer in (0..LAYERS).rev() {
        let state = blocks[layer * COLUMNS + column];
        let flags = traits.flags(registry, tags, state);
        if StateTraits::counts(kind, flags) {
            return MIN_Y + layer as i32 + 1;
        }
    }
    MIN_Y
}

/// The client-facing heightmaps the wire carries (world surface,
/// motion-blocking without leaves, motion-blocking) over a finished block
/// buffer: the first free layer per column under each map's predicate,
/// as (wire type, values) pairs.
pub(crate) fn wire_heightmaps(
    registry: &BlockRegistry,
    blocks: &[u32],
) -> Result<[(u32, Vec<u16>); 3]> {
    let pins = crate::density::locate_pins()?;
    let mut traits = StateTraits::default();
    let mut tags = TagResolver::new(&pins);
    let mut surface = Vec::with_capacity(COLUMNS);
    let mut blocking = Vec::with_capacity(COLUMNS);
    let mut no_leaves = Vec::with_capacity(COLUMNS);
    for col in 0..COLUMNS {
        surface.push(
            (scan_height_blocks(
                blocks,
                HeightKind::WorldSurface,
                col,
                registry,
                &mut traits,
                &mut tags,
            ) - MIN_Y) as u16,
        );
        blocking.push(
            (scan_height_blocks(
                blocks,
                HeightKind::MotionBlocking,
                col,
                registry,
                &mut traits,
                &mut tags,
            ) - MIN_Y) as u16,
        );
        no_leaves.push(
            (scan_height_blocks(
                blocks,
                HeightKind::MotionBlockingNoLeaves,
                col,
                registry,
                &mut traits,
                &mut tags,
            ) - MIN_Y) as u16,
        );
    }
    Ok([(1, surface), (5, no_leaves), (4, blocking)])
}

#[cfg(test)]
mod tests {
    // Spot values are exact bit patterns; trimming their digits would risk
    // parsing a neighboring float.
    #![allow(clippy::excessive_precision)]

    use super::*;

    fn pins() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/worldgen")
    }

    fn registry() -> BlockRegistry {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/blocks.json");
        BlockRegistry::load(&path).expect("block registry pins")
    }

    fn draw(text: &str) -> IntDraw {
        let v: Value = serde_json::from_str(text).unwrap();
        IntDraw::parse(&v).unwrap()
    }

    fn seeded() -> DecorRng {
        let mut rng = DecorRng::new();
        rng.decoration_seed(42, 16, 32);
        rng
    }

    /// The chunk target a DIAG_CHUNK env override names.
    fn diag_target() -> (i32, i32) {
        match std::env::var("DIAG_CHUNK") {
            Ok(text) => {
                let parts: Vec<i32> = text.split(',').filter_map(|n| n.parse().ok()).collect();
                (parts[0], parts[1])
            }
            Err(_) => (-1, -3),
        }
    }

    /// The dump's chunk packets whose coordinates sit within one chunk of
    /// the target, in capture order. The dump holds every packet body;
    /// non-chunk bodies decode as garbage, so the leading chunk
    /// coordinates match before a decode.
    fn scan_dump(target: (i32, i32)) -> Vec<(u32, (i32, i32), WireChunk)> {
        let dump = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/vanilla/worldgen-capture");
        let mut hits: Vec<(u32, (i32, i32), WireChunk)> = Vec::new();
        for entry in std::fs::read_dir(&dump).unwrap() {
            let path = entry.unwrap().path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            let Some(index) = name
                .strip_prefix("p")
                .and_then(|n| n.strip_suffix(".bin"))
                .and_then(|n| n.parse::<u32>().ok())
            else {
                continue;
            };
            let Ok(body) = std::fs::read(&path) else {
                continue;
            };
            let coords_match = body.len() >= 8
                && body[0..4]
                    .try_into()
                    .map(|b: [u8; 4]| i32::from_be_bytes(b))
                    .map(|x| (x - target.0).abs() <= 1)
                    .unwrap_or(false)
                && body[4..8]
                    .try_into()
                    .map(|b: [u8; 4]| i32::from_be_bytes(b))
                    .map(|z| (z - target.1).abs() <= 1)
                    .unwrap_or(false);
            if !coords_match {
                continue;
            }
            let Ok(chunk) = WireChunk::decode(&body) else {
                continue;
            };
            if chunk.sections.len() != SECTION_SPAN || chunk.heightmaps.len() > 8 {
                continue;
            }
            hits.push((index, (chunk.x, chunk.z), chunk));
        }
        hits.sort_by_key(|(index, _, _)| *index);
        hits
    }

    /// The flat state cells of a wire chunk, in storage order.
    fn wire_cells(chunk: &WireChunk) -> Vec<u32> {
        chunk
            .sections
            .iter()
            .flat_map(|section| match &section.block_states {
                crate::chunk_codec::Container::Single(v) => vec![*v; 4096],
                crate::chunk_codec::Container::Palette {
                    entries,
                    longs,
                    bits,
                } => crate::anvil_to_wire::unpack(longs, *bits as usize, 4096)
                    .into_iter()
                    .map(|i| entries.get(i as usize).copied().unwrap_or(0))
                    .collect(),
                crate::chunk_codec::Container::Global { longs, bits } => {
                    crate::anvil_to_wire::unpack(longs, *bits as usize, 4096)
                        .into_iter()
                        .map(u32::from)
                        .collect()
                }
            })
            .collect()
    }

    /// Prints the surface litter cells of the first vegetation tree's
    /// scatter box against the capture: which columns hold litter in each
    /// run and which state, so the first divergent placement stands out.
    #[test]
    #[ignore = "diagnostic: needs a live capture dump"]
    fn probe_scatter_cells() {
        let target = (-1i32, -3i32);
        let reg = registry();
        let terrain = HeightmapGenerator::with_seed(42, &reg).unwrap();
        let hits = scan_dump(target);
        let mut by_pos: Vec<((i32, i32), Vec<u32>)> = Vec::new();
        for (_, pos, cells) in &hits {
            by_pos.push((*pos, wire_cells(cells)));
        }
        let mut order: Vec<(i32, i32)> = hits.iter().map(|(_, pos, _)| *pos).collect();
        order.dedup();
        if order.len() != 9 {
            order = (target.0 - 1..=target.0 + 1)
                .flat_map(|x| (target.1 - 1..=target.1 + 1).map(move |z| (x, z)))
                .collect();
            by_pos.clear();
        }
        let mut dec = Decorator::new(&terrain, &reg, 42).unwrap();
        for &(cx, cz) in &order {
            dec.decorate(cx, cz);
        }
        let describe = |state: u32| -> String {
            reg.state_of(state)
                .map_or("unknown".into(), |(n, p)| format!("{n}[{p}]"))
        };
        let read = |cells: &[((i32, i32), Vec<u32>)], wx: i32, wz: i32, y: i32| -> u32 {
            let cx = wx.div_euclid(16);
            let cz = wz.div_euclid(16);
            for (pos, data) in cells {
                if pos.0 == cx && pos.1 == cz {
                    let lx = (wx - cx * 16) as usize;
                    let lz = (wz - cz * 16) as usize;
                    let layer = (y - MIN_Y) as usize;
                    return data[layer * COLUMNS + lz * 16 + lx];
                }
            }
            0
        };
        let mut ours: Vec<((i32, i32), Vec<u32>)> = Vec::new();
        for &(cx, cz) in &order {
            let chunk = dec.emit(cx, cz).unwrap();
            ours.push(((cx, cz), wire_cells(&chunk)));
        }
        // Tree one stands at local (7,11); its scatter box spans local
        // x 3..12 and z 7..16, y range base-2..base+2 with base 63.
        for lz in 6..=17i32 {
            let mut row = String::new();
            for lx in 2..=13i32 {
                let wx = target.0 * 16 + lx;
                let wz = target.1 * 16 + lz;
                let a = read(&ours, wx, wz, 63);
                let b = read(&by_pos, wx, wz, 63);
                let mark = if a == b { '.' } else { '!' };
                let short = |s: &str| -> String {
                    s.trim_start_matches("minecraft:")
                        .replace(",facing", " f")
                        .replace(",segment_amount=", "s")
                };
                row.push_str(&format!(
                    "{mark}{}:{:>18}|{:>18} ",
                    if lz == 16 { '>' } else { ' ' },
                    short(&describe(a)),
                    short(&describe(b))
                ));
            }
            println!("z{lz:2} {row}");
        }
    }

    /// Compares the terrain-only surface of every dump column against the
    /// capture: the highest non-air block of the undecorated region versus
    /// the capture's ground layer, skipping columns the capture's features
    /// occupy, so terrain height drift shows up per column.
    #[test]
    #[ignore = "diagnostic: needs a live capture dump"]
    fn probe_terrain_surface() {
        let target = (-1i32, -3i32);
        let reg = registry();
        let terrain = HeightmapGenerator::with_seed(42, &reg).unwrap();
        let hits = scan_dump(target);
        let mut order: Vec<(i32, i32)> = hits.iter().map(|(_, pos, _)| *pos).collect();
        order.dedup();
        if order.len() != 9 {
            order = (target.0 - 1..=target.0 + 1)
                .flat_map(|x| (target.1 - 1..=target.1 + 1).map(move |z| (x, z)))
                .collect();
        }
        let mut dec = Decorator::new(&terrain, &reg, 42).unwrap();
        let mut ours: Vec<((i32, i32), Vec<u32>)> = Vec::new();
        for &(cx, cz) in &order {
            let chunk = dec.emit(cx, cz).unwrap();
            ours.push(((cx, cz), wire_cells(&chunk)));
        }
        let name_of = |cells: &[((i32, i32), Vec<u32>)], wx: i32, wz: i32, y: i32| -> String {
            let cx = wx.div_euclid(16);
            let cz = wz.div_euclid(16);
            for (pos, data) in cells {
                if pos.0 == cx && pos.1 == cz {
                    let lx = (wx - cx * 16) as usize;
                    let lz = (wz - cz * 16) as usize;
                    let layer = (y - MIN_Y) as usize;
                    let state = data[layer * COLUMNS + lz * 16 + lx];
                    return reg
                        .state_of(state)
                        .map_or("unknown".into(), |(n, _)| n.to_string());
                }
            }
            "missing".into()
        };
        let top_non_air = |cells: &[((i32, i32), Vec<u32>)], wx: i32, wz: i32| -> i32 {
            let cx = wx.div_euclid(16);
            let cz = wz.div_euclid(16);
            for (pos, data) in cells {
                if pos.0 == cx && pos.1 == cz {
                    let lx = (wx - cx * 16) as usize;
                    let lz = (wz - cz * 16) as usize;
                    for layer in (0..LAYERS).rev() {
                        let state = data[layer * COLUMNS + lz * 16 + lx];
                        let name = reg.state_of(state).map_or("", |(n, _)| n);
                        if name != "minecraft:air" {
                            return MIN_Y + layer as i32;
                        }
                    }
                }
            }
            MIN_Y
        };
        // Ground blocks in the capture that reveal the terrain layer even
        // under placed features; feature-cover columns are skipped.
        let ground = [
            "minecraft:grass_block",
            "minecraft:dirt",
            "minecraft:coarse_dirt",
            "minecraft:podzol",
            "minecraft:stone",
            "minecraft:gravel",
            "minecraft:clay",
            "minecraft:water",
            "minecraft:sand",
        ];
        let feature_cover = [
            "minecraft:dark_oak_log",
            "minecraft:oak_log",
            "minecraft:birch_log",
            "minecraft:dark_oak_leaves",
            "minecraft:oak_leaves",
            "minecraft:birch_leaves",
        ];
        let mut diff = 0usize;
        let mut same = 0usize;
        let mut hist: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
        for (_, pos, cells) in &hits {
            let theirs = wire_cells(cells);
            let _ = pos;
            for lz in 0..16usize {
                for lx in 0..16usize {
                    let wx = pos.0 * 16 + lx as i32;
                    let wz = pos.1 * 16 + lz as i32;
                    // Find the capture's ground layer: the highest cell that
                    // is a ground block or a ground plant, then the block
                    // under it.
                    let mut van_top = None;
                    for layer in (0..LAYERS).rev() {
                        let state = theirs[layer * COLUMNS + lz * 16 + lx];
                        let name = reg.state_of(state).map_or("", |(n, _)| n);
                        if feature_cover.contains(&name) {
                            continue;
                        }
                        if name == "minecraft:air" {
                            continue;
                        }
                        van_top = Some((MIN_Y + layer as i32, name.to_string()));
                        break;
                    }
                    let Some((vy, vname)) = van_top else {
                        continue;
                    };
                    if !ground.contains(&vname.as_str()) {
                        // Litter, grass, mushrooms: the ground is below.
                        continue;
                    }
                    let oy = top_non_air(&ours, wx, wz);
                    let oname = name_of(&ours, wx, wz, oy);
                    if !ground.contains(&oname.as_str()) {
                        continue;
                    }
                    let d = vy - oy;
                    *hist.entry(d).or_default() += 1;
                    if d == 0 {
                        same += 1;
                    } else {
                        diff += 1;
                        // The four cells under a trunk base: soil rings read
                        // as +1 ground but are feature placements.
                        let below = if vy - 1 > MIN_Y {
                            let state = theirs[(vy - 1 - MIN_Y) as usize * COLUMNS + lz * 16 + lx];
                            reg.state_of(state).map_or("", |(n, _)| n)
                        } else {
                            ""
                        };
                        println!(
                            "  ({wx},{wz}) d={d} vanilla {vname}@{vy} over {below} ours {oname}@{oy}"
                        );
                    }
                }
            }
        }
        println!("terrain ground columns: same {same}, diff {diff}");
        println!("delta histogram {hist:?}");
    }

    /// Prints the ground-scatter verdicts of the first tree's two litter
    /// passes: every try position, the verdict, and the heightmap read,
    /// then matches the placed cells against the capture's final states.
    #[test]
    #[ignore = "diagnostic: needs a live capture dump"]
    fn probe_scatter_trace() {
        let target = (-1i32, -3i32);
        let reg = registry();
        let terrain = HeightmapGenerator::with_seed(42, &reg).unwrap();
        let hits = scan_dump(target);
        let vanilla = hits
            .iter()
            .find(|(_, pos, _)| *pos == target)
            .map(|(_, _, chunk)| wire_cells(chunk))
            .expect("target chunk in the capture dump");
        let read_vanilla = |wx: i32, y: i32, wz: i32| -> String {
            let lx = (wx - target.0 * 16) as usize;
            let lz = (wz - target.1 * 16) as usize;
            if lx > 15 || lz > 15 || y <= MIN_Y {
                return "outside".into();
            }
            let state = vanilla[(y - MIN_Y) as usize * COLUMNS + lz * 16 + lx];
            reg.state_of(state)
                .map_or("unknown".into(), |(n, p)| format!("{n}[{p}]"))
        };
        let mut dec = Decorator::new(&terrain, &reg, 42).unwrap();
        dec.scatter_log = Some(Vec::new());
        dec.decorate(target.0, target.1);
        let log = dec.scatter_log.take().unwrap_or_default();
        println!("{} scatter entries in the target chunk", log.len());
        // Vanilla trunk bases: columns whose y-63..64 cells hold dark oak
        // logs, clustered as 2x2 groups.
        {
            let log_id = |y: i32, x: i32, z: i32| {
                read_vanilla(x, y, z).starts_with("minecraft:dark_oak_log")
            };
            let mut bases: Vec<(i32, i32)> = Vec::new();
            for lz in 0..15i32 {
                for lx in 0..15i32 {
                    let (wx, wz) = (target.0 * 16 + lx, target.1 * 16 + lz);
                    for y in [63i32, 64, 65, 66] {
                        if log_id(y, wx, wz)
                            && log_id(y, wx + 1, wz)
                            && log_id(y, wx, wz + 1)
                            && log_id(y, wx + 1, wz + 1)
                        {
                            let local = (lx, lz);
                            if !bases.contains(&local) {
                                bases.push(local);
                            }
                            break;
                        }
                    }
                }
            }
            println!("vanilla trunk 2x2 bases (local): {bases:?}");
        }
        for (i, entry) in log.iter().enumerate() {
            if entry.contains("dark_forest")
                || entry.starts_with("tree ")
                || entry.starts_with("pass")
            {
                println!("mark {i:4} {entry}");
            }
        }
        // The first tree's two scatter passes, verdict by verdict.
        if let Some(first) = log.iter().position(|e| e.starts_with("pass")) {
            for (i, entry) in log.iter().skip(first).take(250).enumerate() {
                println!("v{i:3} {entry}");
            }
        }
        // Pass markers split the log; each pass logs one verdict per try.
        let mut passes: Vec<(usize, usize, [i32; 6], i64)> = Vec::new();
        for (i, entry) in log.iter().enumerate() {
            if entry.starts_with("pass") {
                let nums: Vec<i32> = entry
                    .split_whitespace()
                    .filter_map(|t| t.parse::<i32>().ok())
                    .collect();
                let words = entry
                    .split_whitespace()
                    .find_map(|t| t.strip_prefix('w'))
                    .and_then(|t| t.parse::<i64>().ok())
                    .unwrap_or(0);
                // tries, x0, x1, y0, y1, z0, z1
                passes.push((
                    i,
                    nums[0] as usize,
                    [nums[1], nums[2], nums[3], nums[4], nums[5], nums[6]],
                    words,
                ));
            }
            if entry.starts_with("tree (") {
                println!("tree marker {i}: {entry}");
            }
        }
        for (idx, (start, tries, box_, words)) in passes.iter().enumerate() {
            let end = passes.get(idx + 1).map_or(log.len(), |(s, _, _, _)| *s);
            let placed = log[*start..end]
                .iter()
                .filter(|e| e.contains(" place "))
                .count();
            let last_words = passes.get(idx + 1).map_or(*words, |(_, _, _, w)| *w);
            println!(
                "pass {idx} log {start} tries {tries} entries {} placed {placed} words {words}..{last_words} box x {}..{} y {}..{} z {}..{}",
                end - start - 1,
                box_[0],
                box_[1],
                box_[2],
                box_[3],
                box_[4],
                box_[5]
            );
        }
        // Tree one stands at world (-9,-37): the first pass whose box holds it.
        let origin = (-9i32, -37i32);
        let holds = |box_: &[i32; 6]| {
            box_[0] <= origin.0 && origin.0 <= box_[1] && box_[4] <= origin.1 && origin.1 <= box_[5]
        };
        let tree_passes: Vec<(usize, usize)> = passes
            .iter()
            .enumerate()
            .filter(|(_, (_, _, box_, _))| holds(box_))
            .map(|(idx, (start, _, _, _))| {
                let end = passes.get(idx + 1).map_or(log.len(), |(s, _, _, _)| *s);
                (*start, end)
            })
            .collect();
        println!("tree-one passes: {:?}", tree_passes);
        let mut ours: Vec<(i32, i32, i32)> = Vec::new();
        let mut ours_order: Vec<usize> = Vec::new();
        for (start, end) in tree_passes.iter().take(2) {
            let mut v = 0usize;
            for entry in &log[*start..*end] {
                if let Some(rest) = entry.strip_prefix('(') {
                    let Some((pos, _)) = rest.split_once(')') else {
                        continue;
                    };
                    let placed = entry.contains(" place ");
                    if !placed {
                        v += 1;
                        continue;
                    }
                    let nums: Vec<i32> = pos.split(',').filter_map(|n| n.parse().ok()).collect();
                    // The verdict position is the ground cell; litter lands above.
                    ours.push((nums[0], nums[1] + 1, nums[2]));
                    ours_order.push(v);
                }
                v += 1;
            }
        }
        // Match status in try order: e = exact state, c = cell only,
        // m = vanilla cell empty.
        {
            let mut row = String::new();
            for (k, &(x, y, z)) in ours.iter().enumerate() {
                let van = read_vanilla(x, y, z);
                let ours_state = dec
                    .registry()
                    .state_of(dec.block(x, y, z))
                    .map_or(String::new(), |(n, p)| format!("{n}[{p}]"));
                let mark = if ours_state == van {
                    'e'
                } else if van.starts_with("minecraft:leaf_litter") {
                    'c'
                } else {
                    'm'
                };
                row.push_str(&format!("v{}{mark} ", ours_order[k]));
            }
            println!("placement agreement in try order: {row}");
        }
        println!("tree-one placements: {}", ours.len());
        let mut extra = 0;
        for &(x, y, z) in &ours {
            let there = read_vanilla(x, y, z);
            if !there.starts_with("minecraft:leaf_litter") {
                extra += 1;
                // The full column both sides: a vanilla-only motion block
                // above the cell would flip its heightmap verdict.
                let column = |read: &dyn Fn(i32, i32, i32) -> String| -> String {
                    (y - 1..=y + 15)
                        .map(|yy| {
                            let name = read(x, yy, z);
                            let name = name.split('[').next().unwrap_or("");
                            match name.strip_prefix("minecraft:") {
                                Some("air") => String::new(),
                                Some(n) => format!("{yy}:{n} "),
                                None => String::new(),
                            }
                        })
                        .collect()
                };
                let ours_col = {
                    let dec_ref = &dec;
                    let read = move |xx: i32, yy: i32, zz: i32| -> String {
                        dec_ref
                            .registry()
                            .state_of(dec_ref.block(xx, yy, zz))
                            .map_or(String::new(), |(n, _)| n.to_string())
                    };
                    column(&read)
                };
                println!(
                    "  ours ({x},{y},{z}) vanilla {there}\n    vanilla col {van}\n    ours    col {ours_col}",
                    van = {
                        let read = |xx: i32, yy: i32, zz: i32| read_vanilla(xx, yy, zz);
                        column(&read)
                    }
                );
            }
        }
        // Exact placement agreement: same cell and same state (the
        // segment_amount pins the weighted draw) means the same try on the
        // same stream; the patch feature's litter would share cells by
        // coincidence but not states.
        {
            let mut exact = 0usize;
            let mut cell_only = 0usize;
            for &(x, y, z) in &ours {
                let van = read_vanilla(x, y, z);
                let ours_state = dec
                    .registry()
                    .state_of(dec.block(x, y, z))
                    .map_or(String::new(), |(n, p)| format!("{n}[{p}]"));
                if ours_state == van {
                    exact += 1;
                } else if van.starts_with("minecraft:leaf_litter") {
                    cell_only += 1;
                    println!("  state split ({x},{y},{z}) ours {ours_state} vanilla {van}");
                }
            }
            println!(
                "exact-state placements {exact}/{}, same-cell-different-state {cell_only}",
                ours.len()
            );
        }
        println!("placements without vanilla litter: {extra}");
        // Vanilla litter in the first pass box that we did not place.
        if let Some((_, _, box_, _)) = passes.iter().find(|(_, _, b, _)| holds(b)) {
            let mut missed = 0;
            for y in box_[2]..=box_[3] {
                for z in box_[4]..=box_[5] {
                    for x in box_[0]..=box_[1] {
                        if read_vanilla(x, y, z).starts_with("minecraft:leaf_litter")
                            && !ours.contains(&(x, y, z))
                        {
                            missed += 1;
                            if missed <= 30 {
                                println!(
                                    "  vanilla-only ({x},{y},{z}) ours {}",
                                    dec.block_name(dec.block(x, y, z))
                                );
                            }
                        }
                    }
                }
            }
            println!("vanilla-only litter cells in the box: {missed}");
        }
        // Tree zero's footprint, cell by cell: every log or leaf column in
        // the tree's neighborhood, ours against the capture.
        {
            let base = (-9i32, -37i32);
            let short = |state: u32, reg: &BlockRegistry| -> char {
                reg.state_of(state)
                    .map_or('.', |(n, _)| match n.trim_start_matches("minecraft:") {
                        "dark_oak_log" => 'L',
                        "oak_log" | "birch_log" => 'l',
                        "dark_oak_leaves" => 'D',
                        "oak_leaves" => 'o',
                        "birch_leaves" => 'b',
                        "leaf_litter" => '.',
                        "air" => '.',
                        _ => '?',
                    })
            };
            for z in base.1 - 6..=base.1 + 7 {
                let mut ours_row = String::new();
                let mut van_row = String::new();
                for x in base.0 - 6..=base.0 + 6 {
                    let mut col_o = String::new();
                    let mut col_v = String::new();
                    for y in 62..=78 {
                        col_o.push(short(dec.block(x, y, z), &reg));
                        col_v.push(short(
                            {
                                let lx = (x - target.0 * 16) as usize;
                                let lz = (z - target.1 * 16) as usize;
                                if lx > 15 || lz > 15 {
                                    0
                                } else {
                                    vanilla[(y - MIN_Y) as usize * COLUMNS + lz * 16 + lx]
                                }
                            },
                            &reg,
                        ));
                    }
                    ours_row.push_str(&format!("{col_o} "));
                    van_row.push_str(&format!("{col_v} "));
                }
                println!("z{z:4} ours {ours_row}| van {van_row}");
            }
        }
        // Full state names for the floating-canopy columns, both sides.
        {
            let cells = [
                (-8, -34),
                (-7, -33),
                (-4, -41),
                (-3, -40),
                (-9, -40),
                (-15, -36),
                (-16, -37),
                (-11, -39),
            ];
            for (x, z) in cells {
                let mut ours_col = String::new();
                let mut van_col = String::new();
                for y in 62..=76 {
                    let name = |state: u32| {
                        reg.state_of(state)
                            .map_or(String::new(), |(n, p)| format!("{n}[{p}]"))
                    };
                    let o = name(dec.block(x, y, z));
                    if !o.starts_with("minecraft:air") && !o.is_empty() {
                        ours_col.push_str(&format!("{y}:{o} "));
                    }
                    let v = read_vanilla(x, y, z);
                    if !v.starts_with("minecraft:air") && v != "outside" {
                        van_col.push_str(&format!("{y}:{v} "));
                    }
                }
                println!("col ({x},{z})\n  ours {ours_col}\n  van  {van_col}");
            }
        }
    }

    /// Replays the first tree's two ground-scatter passes straight from
    /// the feature stream: each try draws x, y, z (one word each) and a
    /// placed try draws one weighted state word, so a start offset plus a
    /// heightmap policy fixes the whole placement sequence. Every
    /// candidate offset is scored against the capture's litter cells and
    /// states; the offset and policy that reproduce them name where the
    /// reference run's pass began and which heightmap it read.
    #[test]
    #[ignore = "diagnostic: needs a live capture dump"]
    fn probe_scatter_simulate() {
        let target = (-1i32, -3i32);
        let reg = registry();
        let hits = scan_dump(target);
        let vanilla = hits
            .iter()
            .find(|(_, pos, _)| *pos == target)
            .map(|(_, _, chunk)| wire_cells(chunk))
            .expect("target chunk in the capture dump");
        let read_vanilla = |wx: i32, y: i32, wz: i32| -> String {
            let lx = (wx - target.0 * 16) as usize;
            let lz = (wz - target.1 * 16) as usize;
            if lx > 15 || lz > 15 || y <= MIN_Y {
                return "outside".into();
            }
            let state = vanilla[(y - MIN_Y) as usize * COLUMNS + lz * 16 + lx];
            reg.state_of(state)
                .map_or("unknown".into(), |(n, p)| format!("{n}[{p}]"))
        };

        // The two decorators' weighted tables in entry order.
        let pin: Value = serde_json::from_str(
            &std::fs::read_to_string(pins().join("feature").join("dark_oak_leaf_litter.json"))
                .unwrap(),
        )
        .unwrap();
        let terrain = HeightmapGenerator::with_seed(42, &reg).unwrap();
        let dec = Decorator::new(&terrain, &reg, 42).unwrap();
        let table = |deco: usize| -> Vec<String> {
            pin.get("decorators").and_then(Value::as_array).unwrap()[deco]
                .get("block_state_provider")
                .and_then(|p| p.get("entries"))
                .and_then(Value::as_array)
                .unwrap()
                .iter()
                .map(|e| {
                    let data = e.get("data").unwrap();
                    let (name, props) = match data {
                        Value::String(id) => (id.clone(), String::new()),
                        Value::Object(_) => {
                            let id = data.get("id").and_then(Value::as_str).unwrap().to_string();
                            let mut pairs: Vec<String> = data
                                .get("properties")
                                .and_then(Value::as_object)
                                .map(|m| {
                                    m.iter()
                                        .map(|(k, v)| format!("{k}={}", v.as_str().unwrap_or("")))
                                        .collect()
                                })
                                .unwrap_or_default();
                            pairs.sort();
                            (id, pairs.join(","))
                        }
                        _ => (String::new(), String::new()),
                    };
                    let state = dec.state_id_of(&name, &props).unwrap();
                    let (n, p) = dec.registry().state_of(state).unwrap();
                    format!("{n}[{p}]")
                })
                .collect()
        };
        let tables = [table(0), table(1)];
        println!(
            "tables: {} and {} entries",
            tables[0].len(),
            tables[1].len()
        );

        // Log tops per column from the capture: the live heightmap policy
        // reads motion blocks (logs) above the ground layer.
        let mut log_top: HashMap<(i32, i32), i32> = HashMap::new();
        for z in -44..=-28i32 {
            for x in -18..=0i32 {
                for y in 63..=90i32 {
                    let name = read_vanilla(x, y, z);
                    let name = name.split('[').next().unwrap_or("");
                    if name.ends_with("_log") {
                        log_top.insert((x, z), y);
                    }
                }
            }
        }
        println!("log columns in the neighborhood: {}", log_top.len());

        // The raw feature stream: one word per draw at these bounds.
        let mut probe = DecorRng::new();
        let deco = probe.decoration_seed(42, target.0 * EDGE, target.1 * EDGE);
        probe.set_feature_seed(deco, 20, 9);
        let raw: Vec<i32> = (0..1600).map(|_| probe.next_int(2147483647)).collect();
        let ni = |w: i32, bound: i32| -> i32 {
            let v = raw[w as usize];
            if bound & bound.wrapping_sub(1) == 0 {
                ((bound as i64 * v as i64) >> 31) as i32
            } else {
                v % bound
            }
        };

        // The trunk 2x2 blocks the above-check at y 63.
        let trunk: Vec<(i32, i32)> = (-9..=-8)
            .flat_map(|x| (-37..=-36).map(move |z| (x, z)))
            .collect();

        // Ground level per box column: the capture's top ground block,
        // the capture's litter layer above it, and our terrain fill.
        let ground = [
            "grass_block",
            "dirt",
            "stone",
            "coarse_dirt",
            "podzol",
            "water",
        ];
        let van_surface = |x: i32, z: i32| -> i32 {
            for y in (58..=66).rev() {
                let name = read_vanilla(x, y, z);
                let name = name
                    .split('[')
                    .next()
                    .unwrap_or("")
                    .trim_start_matches("minecraft:");
                if ground.contains(&name) {
                    return y;
                }
            }
            0
        };
        {
            let mut dec2 = Decorator::new(&terrain, &reg, 42).unwrap();
            dec2.ensure_chunk(target.0, target.1);
            let mut raised = 0;
            for z in -41..=-32i32 {
                for x in -13..=-4i32 {
                    let mut ours_y = 0;
                    for y in (58..=66).rev() {
                        let name = dec2
                            .registry()
                            .state_of(dec2.block(x, y, z))
                            .map_or(String::new(), |(n, _)| n.to_string());
                        let name = name.trim_start_matches("minecraft:");
                        if ground.contains(&name) {
                            ours_y = y;
                            break;
                        }
                    }
                    let van_y = van_surface(x, z);
                    let litter_y = (van_y + 1..=van_y + 2)
                        .find(|&y| read_vanilla(x, y, z).starts_with("minecraft:leaf_litter"))
                        .unwrap_or(0);
                    if (x, z) == (-6, -38) {
                        let look = |props: &str| {
                            dec2.registry()
                                .state_id("minecraft:grass_block", props)
                                .map(|id| {
                                    format!(
                                        "{id}={}",
                                        dec2.registry()
                                            .state_of(id)
                                            .map_or("?".to_string(), |(n, p)| {
                                                format!("{n}[{p}]")
                                            })
                                    )
                                })
                                .unwrap_or_else(|| "none".into())
                        };
                        println!(
                            "  grass probe ({x},{z}): ours 62 {} van 62 {} van 63 {} | state_id false {} default {} true {}",
                            dec2.registry()
                                .state_of(dec2.block(x, 62, z))
                                .map_or("?".into(), |(n, p)| format!("{n}[{p}]")),
                            read_vanilla(x, 62, z),
                            read_vanilla(x, 63, z),
                            look("snowy=false"),
                            look(""),
                            look("snowy=true"),
                        );
                    }
                    if van_y != 62 || ours_y != van_y {
                        raised += 1;
                        let col = (59..=65)
                            .map(|y| {
                                format!(
                                    "{y}:{}",
                                    read_vanilla(x, y, z)
                                        .trim_start_matches("minecraft:")
                                        .replace("[]", "")
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        println!("  col ({x},{z}) van surface {van_y} ours {ours_y} litter {litter_y}: {col}");
                    }
                }
            }
            println!("columns off the flat 62 surface: {raised}");
        }

        // Per-column ground level from the capture drives the checks.
        let mut surface: HashMap<(i32, i32), i32> = HashMap::new();
        for z in -41..=-32i32 {
            for x in -13..=-4i32 {
                let y = van_surface(x, z);
                if y > 0 {
                    surface.insert((x, z), y);
                }
            }
        }

        // One pass: the end word plus per-placement agreement.
        let sim = |start: i32,
                   tries: i32,
                   bx0: i32,
                   bx1: i32,
                   bz0: i32,
                   bz1: i32,
                   tbl: &[String],
                   policy: usize,
                   litter: &mut std::collections::HashSet<(i32, i32)>,
                   detail: &mut Vec<String>|
         -> (i32, usize, usize, usize) {
            let mut w = start;
            let (mut exact, mut cellonly, mut miss) = (0usize, 0usize, 0usize);
            for i in 0..tries {
                let x = bx0 + ni(w, bx1 - bx0 + 1);
                let y = 60 + ni(w + 1, 5);
                let z = bz0 + ni(w + 2, bz1 - bz0 + 1);
                w += 3;
                let s = surface.get(&(x, z)).copied().unwrap_or(62);
                let above_ok = y == s && !trunk.contains(&(x, z)) && !litter.contains(&(x, z));
                let h_ok = policy == 0 || log_top.get(&(x, z)).copied().unwrap_or(0) <= s;
                if above_ok && h_ok {
                    let idx = ni(w, tbl.len() as i32) as usize;
                    w += 1;
                    let state = tbl[idx.min(tbl.len() - 1)].clone();
                    let van = read_vanilla(x, s + 1, z);
                    let mark = if van == state {
                        exact += 1;
                        "e"
                    } else if van.starts_with("minecraft:leaf_litter") {
                        cellonly += 1;
                        "c"
                    } else {
                        miss += 1;
                        "m"
                    };
                    detail.push(format!(
                        "  t{i} ({x},{},{z}) {state} van {van} {mark}",
                        s + 1
                    ));
                    litter.insert((x, z));
                }
            }
            (w, exact, cellonly, miss)
        };

        let mut best: Vec<(usize, i32, usize)> = Vec::new();
        for policy in 0..2usize {
            for s in 0..140i32 {
                let mut litter = std::collections::HashSet::new();
                let mut detail = Vec::new();
                let (w0, e0, c0, m0) = sim(
                    s,
                    96,
                    -13,
                    -4,
                    -41,
                    -32,
                    &tables[0],
                    policy,
                    &mut litter,
                    &mut detail,
                );
                let (_w1, e1, c1, m1) = sim(
                    w0,
                    150,
                    -11,
                    -6,
                    -39,
                    -34,
                    &tables[1],
                    policy,
                    &mut litter,
                    &mut detail,
                );
                let score = e0 + e1;
                if score >= 8 {
                    println!(
                        "policy {policy} start {s}: exact {e0}+{e1}={score} cellonly {c0}+{c1} miss {m0}+{m1}"
                    );
                    for row in detail.iter() {
                        println!("{row}");
                    }
                }
                best.push((score, s, policy));
            }
        }
        best.sort_by_key(|b| std::cmp::Reverse(b.0));
        println!(
            "top (score, start, policy): {:?}",
            &best[..8.min(best.len())]
        );
    }

    /// Minimal-distance BFS over the captured 3x3 region versus the
    /// captured distance property: every leaf cell's vanilla distance
    /// against the true shortest path to any log. Non-minimal captures
    /// name the pop-order overwrite the bucket walk produces, so the
    /// match rate decides whether the engine models a clean BFS.
    #[test]
    #[ignore = "diagnostic: needs a live capture dump"]
    fn probe_leaf_distance() {
        let target = (-1i32, -3i32);
        let reg = registry();
        let hits = scan_dump(target);
        let (x0, z0) = ((target.0 - 1) * 16, (target.1 - 1) * 16);
        // The region block volume, non-air cells only.
        let mut names: HashMap<(i32, i32, i32), String> = HashMap::new();
        let mut props: HashMap<(i32, i32, i32), String> = HashMap::new();
        for (_, pos, chunk) in &hits {
            let cells = wire_cells(chunk);
            for lz in 0..16i32 {
                for lx in 0..16i32 {
                    for y in 40..=120i32 {
                        let state =
                            cells[((y - MIN_Y) as usize) * COLUMNS + ((lz * 16 + lx) as usize)];
                        let Some((n, p)) = reg.state_of(state) else {
                            continue;
                        };
                        if n == "minecraft:air" {
                            continue;
                        }
                        let at = (pos.0 * 16 + lx, y, pos.1 * 16 + lz);
                        names.insert(at, n.to_string());
                        props.insert(at, p.to_string());
                    }
                }
            }
        }
        // Multi-source BFS from every log cell; expansion runs through
        // leaf cells only.
        let is_log = |n: &str| n.ends_with("_log");
        let is_leaf = |n: &str| n.ends_with("_leaves");
        let mut dist: HashMap<(i32, i32, i32), i32> = HashMap::new();
        let mut queue: Vec<(i32, i32, i32)> = Vec::new();
        for (&at, n) in &names {
            if is_log(n) {
                dist.insert(at, 0);
                queue.push(at);
            }
        }
        let mut head = 0usize;
        while head < queue.len() {
            let at @ (x, y, z) = queue[head];
            head += 1;
            let d = *dist.get(&at).unwrap();
            if d >= 7 {
                continue;
            }
            for (dx, dy, dz) in [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ] {
                let next = (x + dx, y + dy, z + dz);
                if dist.contains_key(&next) {
                    continue;
                }
                if names.get(&next).is_some_and(|n| is_leaf(n)) {
                    dist.insert(next, d + 1);
                    queue.push(next);
                }
            }
        }
        // Score every leaf cell two blocks inside the region edge.
        let mut same = 0usize;
        let mut off: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
        for (&at @ (x, _y, z), n) in &names {
            if !is_leaf(n) {
                continue;
            }
            if x < x0 + 2 || x >= x0 + 46 || z < z0 + 2 || z >= z0 + 46 {
                continue;
            }
            let captured = BlockRegistry::prop_int(&props[&at], "distance").unwrap_or(7);
            let minimal = dist.get(&at).copied().unwrap_or(7).min(7);
            if captured == minimal {
                same += 1;
            } else {
                *off.entry(captured - minimal).or_default() += 1;
            }
        }
        let total = same + off.values().sum::<usize>();
        println!("minimal BFS: {same}/{total} leaf cells match, captured-minus-minimal {off:?}");

        // The bucket walk the reference runs: buckets of pending cells by
        // distance, logs at zero, one blind pop per step (a cell visiting
        // two buckets has its distance written twice, the later bucket
        // overwriting), expansion into unwritten distance-carrying
        // neighbors at min(current, bucket + 1), and pops reading the
        // pending set in spread-hash table order with insertion-order
        // ties. Cell order decides which cells take the overwrite.
        let pos_hash = |p: &(i32, i32, i32)| -> i32 {
            (p.1.wrapping_add(p.2.wrapping_mul(31)))
                .wrapping_mul(31)
                .wrapping_add(p.0)
        };
        let spread = |h: i32| -> u32 { (h as u32) ^ ((h as u32) >> 16) };
        let cap_for = |max_size: usize| -> u32 {
            let mut cap = 16u32;
            while max_size > (cap as usize * 3) / 4 {
                cap *= 2;
            }
            cap
        };
        // (position, insertion sequence, capacity reached at max size)
        type WalkCell = ((i32, i32, i32), usize);
        let mut buckets: Vec<Vec<WalkCell>> = vec![Vec::new(); 7];
        let mut caps = [16u32; 7];
        let mut written: HashMap<(i32, i32, i32), i32> = HashMap::new();
        let mut shape: std::collections::HashSet<(i32, i32, i32)> =
            std::collections::HashSet::new();
        let mut seq = 0usize;
        let mut seeds: Vec<(i32, i32, i32)> = names
            .iter()
            .filter(|(_, n)| is_log(n))
            .map(|(&at, _)| at)
            .collect();
        seeds.sort_unstable_by_key(|p| p.1);
        for at in seeds {
            buckets[0].push((at, seq));
            seq += 1;
        }
        caps[0] = cap_for(buckets[0].len());
        let mut smallest = 0usize;
        let mut pops = 0usize;
        let mut overwrites = 0usize;
        loop {
            while smallest < 7 && buckets[smallest].is_empty() {
                smallest += 1;
            }
            if smallest >= 7 {
                break;
            }
            // Pop the first cell in table order: the low spread bits name
            // the table slot, the insertion sequence orders within it.
            let cap = caps[smallest];
            let mask = cap - 1;
            let pick = buckets[smallest]
                .iter()
                .enumerate()
                .min_by_key(|&(k, &(at, s))| (spread(pos_hash(&at)) & mask, s, k))
                .map(|(k, _)| k)
                .expect("bucket nonempty above");
            let (at @ (px, py, pz), _) = buckets[smallest][pick];
            buckets[smallest].swap_remove(pick);
            pops += 1;
            if shape.contains(&at) {
                overwrites += 1;
            }
            if smallest != 0 {
                written.insert(at, smallest as i32);
            }
            shape.insert(at);
            for (dx, dy, dz) in [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ] {
                let next = (px + dx, py + dy, pz + dz);
                if shape.contains(&next) {
                    continue;
                }
                let Some(n) = names.get(&next) else {
                    continue;
                };
                // Logs read as distance zero; leaves carry their property.
                let current_distance = if is_log(n) {
                    0
                } else {
                    match props
                        .get(&next)
                        .and_then(|p| BlockRegistry::prop_int(p, "distance"))
                    {
                        Some(v) => v,
                        None => continue,
                    }
                };
                let new_distance = current_distance.min(smallest as i32 + 1);
                if new_distance < 7 {
                    let bucket = &mut buckets[new_distance as usize];
                    if bucket.iter().any(|&(b, _)| b == next) {
                        continue;
                    }
                    bucket.push((next, seq));
                    seq += 1;
                    // The table only grows; removals never shrink it.
                    while bucket.len() > (caps[new_distance as usize] as usize * 3) / 4 {
                        caps[new_distance as usize] *= 2;
                    }
                    smallest = smallest.min(new_distance as usize);
                }
            }
        }
        let mut same = 0usize;
        let mut off: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
        let mut worst: Vec<((i32, i32, i32), i32, i32)> = Vec::new();
        for (&at @ (x, _y, z), n) in &names {
            if !is_leaf(n) {
                continue;
            }
            if x < x0 + 2 || x >= x0 + 46 || z < z0 + 2 || z >= z0 + 46 {
                continue;
            }
            let captured = BlockRegistry::prop_int(&props[&at], "distance").unwrap_or(7);
            let walked = written.get(&at).copied().unwrap_or(7);
            if captured == walked {
                same += 1;
            } else {
                *off.entry(captured - walked).or_default() += 1;
                if worst.len() < 20 {
                    worst.push((at, captured, walked));
                }
            }
        }
        let total = same + off.values().sum::<usize>();
        println!(
            "bucket walk: {same}/{total} leaf cells match ({pops} pops, {overwrites} overwrite pops), captured-minus-walked {off:?}"
        );
        for ((x, y, z), c, w) in &worst {
            println!("  ({x},{y},{z}) captured {c} walked {w}");
        }
    }

    #[test]
    #[ignore = "diagnostic: needs a live capture dump"]
    fn diagnose_chunk_divergence() {
        let target = diag_target();
        let reg = registry();
        let terrain = HeightmapGenerator::with_seed(42, &reg).unwrap();
        let hits = scan_dump(target);
        let vanilla = hits
            .iter()
            .find(|(_, pos, _)| *pos == target)
            .map(|(_, _, chunk)| chunk)
            .expect("target chunk in the capture dump")
            .clone();

        // Several decoration orders over the 3x3 neighborhood: capture
        // order, x-major, z-major.
        let mut capture: Vec<(i32, i32)> = hits.iter().map(|(_, pos, _)| *pos).collect();
        capture.dedup();
        let mut xmajor: Vec<(i32, i32)> = (target.0 - 1..=target.0 + 1)
            .flat_map(|x| (target.1 - 1..=target.1 + 1).map(move |z| (x, z)))
            .collect();
        let mut zmajor: Vec<(i32, i32)> = (target.1 - 1..=target.1 + 1)
            .flat_map(|z| (target.0 - 1..=target.0 + 1).map(move |x| (x, z)))
            .collect();
        let mut orders: Vec<(&str, Vec<(i32, i32)>)> = Vec::new();
        if capture.len() == 9 {
            orders.push(("capture", capture));
        }
        orders.push(("x-major", xmajor.clone()));
        orders.push(("z-major", zmajor.clone()));
        xmajor.reverse();
        orders.push(("x-major-rev", xmajor));
        zmajor.reverse();
        orders.push(("z-major-rev", zmajor));
        let filter = std::env::var("DIAG_ORDERS").unwrap_or_default();
        if !filter.is_empty() {
            orders.retain(|(label, _)| filter.split(',').any(|f| f == *label));
        }
        for (label, order) in &orders {
            let mut dec = Decorator::new(&terrain, &reg, 42).unwrap();
            dec.visits = Some(Vec::new());
            for &(cx, cz) in order {
                dec.decorate(cx, cz);
            }
            if *label == "capture" {
                // The vegetation try sequence: every position the placement
                // stack resolved for this chunk, in draw order.
                let tries: Vec<String> = dec
                    .visits
                    .as_ref()
                    .unwrap()
                    .iter()
                    .filter(|(n, _, _, _)| n == "dark_forest_vegetation")
                    .filter(|(_, x, _, z)| {
                        *x >= target.0 * 16
                            && *x < target.0 * 16 + 16
                            && *z >= target.1 * 16
                            && *z < target.1 * 16 + 16
                    })
                    .map(|(_, x, y, z)| {
                        format!("({x},{y},{z}){}", {
                            // A log column at the try marks a placed trunk.
                            let name = dec.block_name(dec.block(*x, *y, *z));
                            if name.ends_with("_log") {
                                "*"
                            } else {
                                ""
                            }
                        })
                    })
                    .collect();
                println!("vegetation tries: {}", tries.join(" "));
                // The joined step-9 order the driver seeds from.
                let step9: Vec<String> = dec.plan.steps[9]
                    .iter()
                    .map(|&k| dec.plan.names[k].clone())
                    .collect();
                println!(
                    "step-9 order ({} entries): {}",
                    step9.len(),
                    step9
                        .iter()
                        .enumerate()
                        .map(|(i, n)| format!("{i}:{n}"))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
            dec.visits = None;
            let mine = dec.emit(target.0, target.1).unwrap();
            let mut delta: i64 = 0;
            let mut names: HashMap<String, i64> = HashMap::new();
            let mut leaf_cells: Vec<(usize, u32, u32)> = Vec::new();
            for (i, (a, b)) in wire_cells(&mine)
                .iter()
                .zip(wire_cells(&vanilla).iter())
                .enumerate()
            {
                let name = |state: u32| -> String {
                    reg.state_of(state)
                        .map_or("unknown".to_string(), |(n, _)| n.to_string())
                };
                if a != b {
                    let (na, nb) = (name(*a), name(*b));
                    let leaf =
                        |n: &str| i64::from(n.ends_with("_leaves") || n == "minecraft:leaf_litter");
                    delta += leaf(&na) + leaf(&nb);
                    *names.entry(format!("{na} != {nb}")).or_default() += 1;
                    if leaf(&na) > 0 || leaf(&nb) > 0 {
                        leaf_cells.push((i, *a, *b));
                    }
                }
            }
            println!("order {label}: leaf-ish mismatched cells {delta}");
            if *label == "capture" {
                let mut rows: Vec<(i64, String)> = names.into_iter().map(|(k, v)| (v, k)).collect();
                rows.sort_by_key(|(count, _)| std::cmp::Reverse(*count));
                println!("mismatched-cell histogram:");
                for (count, key) in rows.iter().take(15) {
                    println!("  {count:8} {key}");
                }
            }
            if *label == "capture" {
                // Litter agreement: same cell, same state means the same
                // weighted draw; a cliff in agreement marks the stream flip.
                let ours_cells = wire_cells(&mine);
                let theirs_cells = wire_cells(&vanilla);
                let litter = |state: u32| {
                    reg.state_of(state)
                        .is_some_and(|(n, _)| n == "minecraft:leaf_litter")
                };
                let mut both = 0u64;
                let mut ours_only = 0u64;
                let mut theirs_only = 0u64;
                for (a, b) in ours_cells.iter().zip(theirs_cells.iter()) {
                    let (la, lb) = (litter(*a), litter(*b));
                    if la && lb {
                        both += u64::from(a == b);
                    }
                    if la && !lb {
                        ours_only += 1;
                    }
                    if lb && !la {
                        theirs_only += 1;
                    }
                }
                println!(
                    "litter cells: both={both} exact-match, ours-only={ours_only}, vanilla-only={theirs_only}"
                );
                // Litter segment counts name the decorator that placed a
                // cell: the tree's first scatter stops at three segments,
                // the second reaches four, the patch feature draws its own
                // mix.
                let mut segs: HashMap<String, (u64, u64)> = HashMap::new();
                for (a, b) in ours_cells.iter().zip(theirs_cells.iter()) {
                    let (la, lb) = (litter(*a), litter(*b));
                    if !(la || lb) {
                        continue;
                    }
                    let seg = |state: u32| {
                        reg.state_of(state).map_or("?".into(), |(_, p)| {
                            p.split(',')
                                .find(|pair| pair.starts_with("segment_amount"))
                                .unwrap_or("none")
                                .to_string()
                        })
                    };
                    let entry = segs.entry(seg(*a)).or_default();
                    if la {
                        entry.0 += 1;
                    }
                    if lb {
                        let entry = segs.entry(seg(*b)).or_default();
                        entry.1 += 1;
                    }
                }
                let mut seg_rows: Vec<(String, u64, u64)> =
                    segs.into_iter().map(|(k, v)| (k, v.0, v.1)).collect();
                seg_rows.sort();
                println!("litter by segment (ours, vanilla):");
                for (seg, a, b) in seg_rows {
                    println!("  {seg}: {a}, {b}");
                }
                // Trunk bases (a ground-level log column) name the trees;
                // litter per tree splits shape divergence from scatter
                // divergence.
                let is_log = |state: u32| {
                    reg.state_of(state)
                        .is_some_and(|(n, _)| n.ends_with("_log"))
                };
                let count_bases = |cells: &[u32]| -> u64 {
                    let ground = 63 - MIN_Y;
                    let section = (ground / 16) as usize;
                    let ly = (ground % 16) as usize;
                    let mut cols = std::collections::HashSet::new();
                    for lz in 0..16usize {
                        for lx in 0..16usize {
                            let i = section * 4096 + ly * 256 + lz * 16 + lx;
                            if is_log(cells[i]) {
                                cols.insert((lx, lz));
                            }
                        }
                    }
                    cols.len() as u64
                };
                println!(
                    "trunk-base columns (ours, vanilla): ({}, {})",
                    count_bases(&ours_cells),
                    count_bases(&theirs_cells)
                );
                // Feature-level agreement probe: glow lichen (step 9,
                // index 0) and the surface patches.
                for probe in ["glow_lichen", "short_grass", "brown_mushroom", "pumpkin"] {
                    let is_probe =
                        |state: u32| reg.state_of(state).is_some_and(|(n, _)| n == probe);
                    let a = ours_cells.iter().filter(|s| is_probe(**s)).count();
                    let b = theirs_cells.iter().filter(|s| is_probe(**s)).count();
                    println!("{probe} cells (ours, vanilla): ({a}, {b})");
                }
                // Per-column log y-ranges show each side's tree layout:
                // base, height, and shape of every trunk.
                let log_range = |cells: &[u32]| -> String {
                    let mut parts = Vec::new();
                    for lz in 0..16usize {
                        for lx in 0..16usize {
                            let mut lo = i32::MAX;
                            let mut hi = i32::MIN;
                            for layer in 0..LAYERS {
                                if is_log(cells[layer * COLUMNS + lz * 16 + lx]) {
                                    lo = lo.min(MIN_Y + layer as i32);
                                    hi = hi.max(MIN_Y + layer as i32);
                                }
                            }
                            if lo <= hi {
                                parts.push(format!("({lx},{lz}):{lo}-{hi}"));
                            }
                        }
                    }
                    parts.join(" ")
                };
                println!("our log columns: {}", log_range(&ours_cells));
                println!("vanilla log columns: {}", log_range(&theirs_cells));
            }
            if *label == "capture" {
                for (i, a, b) in leaf_cells.iter().take(40) {
                    let describe = |state: u32| -> String {
                        reg.state_of(state)
                            .map_or("unknown".to_string(), |(n, p)| format!("{n}[{p}]#{state}"))
                    };
                    let section = i / 4096;
                    let within = i % 4096;
                    let ly = within / 256;
                    let lz = (within % 256) / 16;
                    let lx = within % 16;
                    println!(
                        "   ({},{},{}) ours={} vanilla={}",
                        target.0 * 16 + lx as i32,
                        MIN_Y + (section as i32) * 16 + ly as i32,
                        target.1 * 16 + lz as i32,
                        describe(*a),
                        describe(*b)
                    );
                }
            }
        }
    }

    /// Sweeps the (step, index) seed inputs one feature's placement draws
    /// from, replaying it over the region the rest of the decoration
    /// produced: the exact-match count against the captured vanilla cells
    /// spikes at the inputs the reference run used, so a low flat sweep
    /// clears the seed and convicts the draw shape instead.
    #[test]
    #[ignore = "diagnostic: needs a live capture dump"]
    fn probe_feature_seed_index() {
        let target = diag_target();
        let feature = std::env::var("DIAG_FEATURE").unwrap_or_else(|_| "patch_leaf_litter".into());
        let feature = feature
            .strip_prefix("minecraft:")
            .unwrap_or(&feature)
            .to_string();
        let block = std::env::var("DIAG_BLOCK").unwrap_or_else(|_| "minecraft:leaf_litter".into());
        let reg = registry();
        let terrain = HeightmapGenerator::with_seed(42, &reg).unwrap();
        let hits = scan_dump(target);
        let vanilla = hits
            .iter()
            .find(|(_, pos, _)| *pos == target)
            .map(|(_, _, chunk)| chunk)
            .expect("target chunk in the capture dump")
            .clone();

        let mut dec = Decorator::new(&terrain, &reg, 42).unwrap();
        let key = dec.key_of(&feature).expect("plan carries the feature");
        dec.skip = Some(key);
        let mut order: Vec<(i32, i32)> = hits.iter().map(|(_, pos, _)| *pos).collect();
        order.dedup();
        if order.len() != 9 {
            order = (target.0 - 1..=target.0 + 1)
                .flat_map(|x| (target.1 - 1..=target.1 + 1).map(move |z| (x, z)))
                .collect();
        }
        for &(cx, cz) in &order {
            dec.decorate(cx, cz);
        }
        dec.skip = None;

        let mut probe_rng = DecorRng::new();
        let deco = probe_rng.decoration_seed(42, target.0 * EDGE, target.1 * EDGE);
        let theirs = wire_cells(&vanilla);
        let is_probe_block = |state: u32| reg.state_of(state).is_some_and(|(n, _)| n == block);
        // Everything but the probe feature: how much of the block each
        // side carries from the features that already ran.
        let base = wire_cells(&dec.emit(target.0, target.1).unwrap());
        let base_ours = base.iter().filter(|s| is_probe_block(**s)).count();
        let base_theirs = theirs.iter().filter(|s| is_probe_block(**s)).count();
        println!(
            "skipped baseline: ours={base_ours} vanilla={base_theirs} {block} cells (trees and other features)"
        );
        // For the vanilla cells our baseline lacks: what our region holds at
        // the cell and below it, plus the column height the heightmap
        // modifier would read. This pins which filter leg refuses.
        let mut shown = 0;
        for (i, (a, b)) in theirs.iter().zip(base.iter()).enumerate() {
            if !is_probe_block(*a) || is_probe_block(*b) {
                continue;
            }
            shown += 1;
            if shown > 30 {
                break;
            }
            let section = i / 4096;
            let within = i % 4096;
            let (wx, wy, wz) = (
                target.0 * 16 + (within % 16) as i32,
                MIN_Y + (section as i32) * 16 + (within / 256) as i32,
                target.1 * 16 + ((within % 256) / 16) as i32,
            );
            let at = reg
                .state_of(dec.block(wx, wy, wz))
                .map_or("?".into(), |(n, p)| format!("{n}[{p}]"));
            let below = reg
                .state_of(dec.block(wx, wy - 1, wz))
                .map_or("?".into(), |(n, p)| format!("{n}[{p}]"));
            let h = dec.height(HeightKind::WorldSurfaceWg, wx, wz);
            println!("  probe v({wx},{wy},{wz}): ours_at={at} ours_below={below} surf_h={h}");
        }
        let mut results: Vec<(u64, i32, i32)> = Vec::new();
        let widest = dec.plan.steps.iter().map(|s| s.len()).max().unwrap_or(0) as i32;
        // The vanilla trunk columns name the try positions the reference
        // run drew: a candidate seed whose stream passes through them is
        // the index the reference driver used. Score every candidate by
        // how many origin columns its opening draws hit; a placed tree
        // consumes draws between tries, so the naive pair stream runs
        // ahead of the try count and the window runs long.
        let origins =
            std::env::var("DIAG_ORIGINS").unwrap_or_else(|_| "6,3;1,4;2,9;7,11;13,15".into());
        let wants: Vec<(i32, i32)> = origins
            .split(';')
            .filter_map(|pair| {
                let nums: Vec<i32> = pair.split(',').filter_map(|n| n.parse().ok()).collect();
                (nums.len() == 2).then_some((nums[0], nums[1]))
            })
            .collect();
        let mut hits: Vec<(usize, i32, i32)> = Vec::new();
        for step in 0..=10i32 {
            for index in 0..=widest {
                let mut r = DecorRng::new();
                r.set_feature_seed(deco, index, step);
                // Every tree placed mid-sequence consumes its own draws, so
                // the square draws of later tries sit at arbitrary stream
                // offsets: an origin column matches any consecutive draw
                // pair, either alignment.
                let draws: Vec<i32> = (0..96).map(|_| r.next_int(16)).collect();
                let mut found = 0usize;
                for want in &wants {
                    if draws.windows(2).any(|w| w[0] == want.0 && w[1] == want.1) {
                        found += 1;
                    }
                }
                if found >= 4 {
                    println!("signature s{step}i{index}: {found} wants in 96 draws");
                    hits.push((found, step, index));
                }
            }
        }
        hits.sort_by_key(|h| std::cmp::Reverse(h.0));
        hits.truncate(12);
        if hits.is_empty() {
            println!(
                "no (step, index) seed carries four vanilla trunk columns: the divergence sits below the feature seed"
            );
        }
        // Each surviving candidate redecorates the whole region with the
        // seed override riding the feature's plan slot: later features
        // (grass, litter) raise the live surface the water-depth filter
        // reads, so replaying the feature last flips its verdicts.
        for (_, step, index) in &hits {
            let mut cand = Decorator::new(&terrain, &reg, 42).unwrap();
            cand.seed_override = Some((key, *index, *step));
            for &(cx, cz) in &order {
                cand.decorate(cx, cz);
            }
            let ours = wire_cells(&cand.emit(target.0, target.1).unwrap());
            let ours_total = ours.iter().filter(|s| is_probe_block(**s)).count();
            let mut exact = 0u64;
            let mut ours_only = 0u64;
            let mut theirs_only = 0u64;
            for (a, b) in ours.iter().zip(theirs.iter()) {
                let (pa, pb) = (is_probe_block(*a), is_probe_block(*b));
                if pa && pb {
                    exact += u64::from(a == b);
                } else if pa {
                    ours_only += 1;
                } else if pb {
                    theirs_only += 1;
                }
            }
            println!(
                "step {step} index {index}: exact={exact} ours-only={ours_only} vanilla-only={theirs_only} ours-total={ours_total}"
            );
            results.push((exact, *step, *index));
        }
        results.sort_by_key(|a| std::cmp::Reverse(a.0));
        println!(
            "best (exact, step, index): {:?}",
            &results[..5.min(results.len())]
        );

        // The best candidate's own try sequence and cells, against the
        // skipped baseline and the vanilla cells it still lacks.
        if results.is_empty() {
            return;
        }
        let (step, index) = (results[0].1, results[0].2);
        let mut cand = Decorator::new(&terrain, &reg, 42).unwrap();
        cand.seed_override = Some((key, index, step));
        cand.visits = Some(Vec::new());
        for &(cx, cz) in &order {
            cand.decorate(cx, cz);
        }
        let ours = wire_cells(&cand.emit(target.0, target.1).unwrap());
        let describe = |state: u32| -> String {
            reg.state_of(state)
                .map_or("unknown".into(), |(n, p)| format!("{n}[{p}]"))
        };
        let tries: Vec<String> = cand
            .visits
            .take()
            .unwrap_or_default()
            .iter()
            .filter(|(n, _, _, _)| n == &feature)
            .filter(|(_, x, _, z)| {
                *x >= target.0 * 16
                    && *x < target.0 * 16 + 16
                    && *z >= target.1 * 16
                    && *z < target.1 * 16 + 16
            })
            .map(|(_, x, y, z)| {
                let marker = if cand.block_name(cand.block(*x, *y, *z)).ends_with("_log") {
                    "*"
                } else {
                    ""
                };
                format!("({x},{y},{z}){marker}")
            })
            .collect();
        println!("best candidate ({step},{index}) tries: {}", tries.join(" "));
        println!("best candidate ({step},{index}) cells over the skipped baseline:");
        for (i, (a, b)) in ours.iter().zip(base.iter()).enumerate() {
            if a != b {
                let section = i / 4096;
                let within = i % 4096;
                println!(
                    "  +({},{},{}) {}",
                    target.0 * 16 + (within % 16) as i32,
                    MIN_Y + (section as i32) * 16 + (within / 256) as i32,
                    target.1 * 16 + ((within % 256) / 16) as i32,
                    describe(*a)
                );
            }
        }
        println!("vanilla cells the baseline lacks:");
        let mut shown = 0;
        for (i, (a, b)) in theirs.iter().zip(base.iter()).enumerate() {
            if is_probe_block(*a) && !is_probe_block(*b) {
                shown += 1;
                if shown > 60 {
                    break;
                }
                let section = i / 4096;
                let within = i % 4096;
                println!(
                    "  v({},{},{}) {}",
                    target.0 * 16 + (within % 16) as i32,
                    MIN_Y + (section as i32) * 16 + (within / 256) as i32,
                    target.1 * 16 + ((within % 256) / 16) as i32,
                    describe(*a)
                );
            }
        }
    }

    /// Tree alignment: vanilla trunk columns versus the try origins the
    /// driver drew for the vegetation feature. A vanilla trunk column that
    /// appears in our try list convicts a filter; one that never shows up
    /// convicts the seed stream.
    #[test]
    #[ignore = "diagnostic: needs a live capture dump"]
    fn probe_tree_alignment() {
        let target = diag_target();
        let reg = registry();
        let terrain = HeightmapGenerator::with_seed(42, &reg).unwrap();
        let hits = scan_dump(target);
        let vanilla = hits
            .iter()
            .find(|(_, pos, _)| *pos == target)
            .map(|(_, _, chunk)| chunk)
            .expect("target chunk in the capture dump")
            .clone();
        let mut order: Vec<(i32, i32)> = hits.iter().map(|(_, pos, _)| *pos).collect();
        order.dedup();
        if order.len() != 9 {
            order = (target.0 - 1..=target.0 + 1)
                .flat_map(|x| (target.1 - 1..=target.1 + 1).map(move |z| (x, z)))
                .collect();
        }
        let mut dec = Decorator::new(&terrain, &reg, 42).unwrap();
        dec.visits = Some(Vec::new());
        for &(cx, cz) in &order {
            dec.decorate(cx, cz);
        }
        let feature = "dark_forest_vegetation";
        let tries: Vec<(i32, i32)> = dec
            .visits
            .take()
            .unwrap_or_default()
            .iter()
            .filter(|(n, _, _, _)| n == feature)
            .map(|(_, x, _, z)| (*x, *z))
            .collect();

        // Vanilla trunk columns: any log cell in the target chunk.
        let cells = wire_cells(&vanilla);
        let mut vanilla_cols: Vec<(i32, i32)> = Vec::new();
        for lz in 0..16usize {
            for lx in 0..16usize {
                for layer in 0..LAYERS {
                    let state = cells[layer * COLUMNS + lz * 16 + lx];
                    if reg
                        .state_of(state)
                        .is_some_and(|(n, _)| n.ends_with("_log"))
                    {
                        vanilla_cols.push((target.0 * 16 + lx as i32, target.1 * 16 + lz as i32));
                        break;
                    }
                }
            }
        }
        println!(
            "vanilla trunk columns ({}): {:?}",
            vanilla_cols.len(),
            vanilla_cols
        );
        println!("our {feature} try origins ({}): {:?}", tries.len(), tries);
        let try_hit = vanilla_cols.iter().filter(|c| tries.contains(c)).count();
        println!(
            "vanilla columns present in our try list: {try_hit}/{}",
            vanilla_cols.len()
        );
        // Systematic shift probe: does any constant offset map our tries
        // onto the vanilla columns?
        for dz in -2..=2i32 {
            let mut row = Vec::new();
            for dx in -2..=2i32 {
                let matched = tries
                    .iter()
                    .filter(|t| vanilla_cols.contains(&(t.0 + dx, t.1 + dz)))
                    .count();
                row.push(matched);
            }
            println!("shift row dz={dz}: {row:?}");
        }
        // Vanilla tree origins: the lowest log layer of each contiguous
        // cluster (the 2x2 trunk base stands exactly at the in_square
        // origin; lean and branches only add columns above).
        let mut col_y: std::collections::HashMap<(i32, i32), i32> =
            std::collections::HashMap::new();
        for lz in 0..16usize {
            for lx in 0..16usize {
                for layer in 0..LAYERS {
                    let state = cells[layer * COLUMNS + lz * 16 + lx];
                    if reg
                        .state_of(state)
                        .is_some_and(|(n, _)| n.ends_with("_log"))
                    {
                        col_y.insert(
                            (target.0 * 16 + lx as i32, target.1 * 16 + lz as i32),
                            MIN_Y + layer as i32,
                        );
                        break;
                    }
                }
            }
        }
        let mut origins: Vec<(i32, i32)> = Vec::new();
        for (&(x, z), &y) in &col_y {
            let neighbor_lower = [(x - 1, z), (x, z - 1), (x - 1, z - 1)]
                .iter()
                .any(|&(nx, nz)| col_y.get(&(nx, nz)).is_some_and(|&ny| ny <= y));
            let is_base = !neighbor_lower
                && col_y.get(&(x + 1, z)).is_some_and(|&ny| ny == y)
                && col_y.get(&(x, z + 1)).is_some_and(|&ny| ny == y)
                && col_y.get(&(x + 1, z + 1)).is_some_and(|&ny| ny == y);
            if is_base {
                origins.push((x, z));
            }
        }
        origins.sort();
        origins.dedup();
        println!("vanilla tree origins ({}): {origins:?}", origins.len());
        // Sweep (step, index) seeds: the true derivation passes through the
        // vanilla origins as consecutive in_square draws, interleaved with
        // whatever the surviving tries consume between them.
        let mut probe_rng = DecorRng::new();
        let deco = probe_rng.decoration_seed(42, target.0 * EDGE, target.1 * EDGE);
        let widest = dec.plan.steps.iter().map(|s| s.len()).max().unwrap_or(0) as i32;
        let mut scored: Vec<(usize, i32, i32)> = Vec::new();
        for step in 0..=10i32 {
            for index in 0..=widest {
                let mut r = DecorRng::new();
                r.set_feature_seed(deco, index, step);
                let draws: Vec<i32> = (0..256).map(|_| r.next_int(16)).collect();
                let found = origins
                    .iter()
                    .filter(|o| {
                        draws
                            .windows(2)
                            .any(|w| w[0] == o.0.rem_euclid(16) && w[1] == o.1.rem_euclid(16))
                    })
                    .count();
                if found * 2 > origins.len() {
                    scored.push((found, step, index));
                }
            }
        }
        scored.sort_by_key(|s| std::cmp::Reverse(s.0));
        println!("seed candidates (hits, step, index): {scored:?}");
        // Control: our own derivation must appear in the same sweep.
        let our_index = dec
            .plan
            .positions
            .get(&dec.key_of(feature).unwrap())
            .map(|(s, i)| (*s as i32, *i as i32))
            .unwrap_or((-1, -1));
        println!(
            "our plan slot for {feature}: (step {}, index {})",
            our_index.0, our_index.1
        );
        let mut r = DecorRng::new();
        r.set_feature_seed(deco, our_index.1, our_index.0);
        let draws: Vec<i32> = (0..256).map(|_| r.next_int(16)).collect();
        let our_hits = tries
            .iter()
            .filter(|t| {
                draws
                    .windows(2)
                    .any(|w| w[0] == t.0.rem_euclid(16) && w[1] == t.1.rem_euclid(16))
            })
            .count();
        println!(
            "control: our own try origins found in our own raw stream: {our_hits}/{}",
            tries
                .iter()
                .filter(|t| {
                    t.0 >= target.0 * 16
                        && t.0 < target.0 * 16 + 16
                        && t.1 >= target.1 * 16
                        && t.1 < target.1 * 16 + 16
                })
                .count()
        );
        // Which filter leg refused each unmatched vanilla column.
        for col in vanilla_cols.iter().take(24) {
            if tries.contains(col) {
                continue;
            }
            let floor = dec.height(HeightKind::OceanFloor, col.0, col.1);
            let surface = dec.height(HeightKind::WorldSurface, col.0, col.1);
            let biome = dec.biome_at(col.0, floor, col.1);
            let name = dec
                .plan
                .steps
                .iter()
                .enumerate()
                .find(|(_, s)| !s.is_empty())
                .map(|_| biome.to_string())
                .unwrap_or_default();
            println!(
                "  miss ({},{}): floor={floor} surface={surface} biome={biome} {name}",
                col.0, col.1
            );
        }
    }

    /// Prints the spot values the chain test pins, after draw semantics
    /// changes shift every stream.
    #[test]
    #[ignore = "prints regenerated spot values"]
    fn print_decoration_spot_values() {
        let mut rng = DecorRng::new();
        println!("deco(42,16,32) = {}", rng.decoration_seed(42, 16, 32));
        let mut rng = DecorRng::new();
        println!("deco(42,16,0) = {}", rng.decoration_seed(42, 16, 0));
        let mut rng = DecorRng::new();
        println!("deco(42,0,16) = {}", rng.decoration_seed(42, 0, 16));

        let deco = DecorRng::new().decoration_seed(42, 16, 32);
        let mut rng = DecorRng::new();
        rng.set_feature_seed(deco, 3, 9);
        let draws: Vec<i32> = (0..4).map(|_| rng.next_int(16)).collect();
        println!("feature(3,9) first 4 x16: {draws:?}");
        println!("feature(3,9) then f32: {}", rng.next_f32());
        let mut rng = DecorRng::new();
        rng.set_feature_seed(deco, 4, 9);
        println!(
            "feature(4,9) first 2 x16: {:?}",
            (0..2).map(|_| rng.next_int(16)).collect::<Vec<_>>()
        );
        let mut rng = DecorRng::new();
        rng.set_feature_seed(deco, 3, 10);
        println!(
            "feature(3,10) first 2 x16: {:?}",
            (0..2).map(|_| rng.next_int(16)).collect::<Vec<_>>()
        );
    }

    /// Scores every (step, index) feature seed by its first in_square pair:
    /// the opening two draws of the seeded stream are the first try's local
    /// (x, z), so for the seed the reference run used that pair lands on a
    /// vanilla trunk column in most chunks that carry trees. The whole
    /// capture dump votes, one decoration-free stream per candidate per
    /// chunk, and both decoration-seed input conventions run side by side.
    #[test]
    #[ignore = "diagnostic: needs a live capture dump"]
    fn probe_first_draw_alignment() {
        let reg = registry();
        let dump = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/vanilla/worldgen-capture");
        type ChunkPairs = ((i32, i32), Vec<(i32, i32)>);
        let mut chunks: Vec<ChunkPairs> = Vec::new();
        for entry in std::fs::read_dir(&dump).unwrap() {
            let path = entry.unwrap().path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !name.starts_with('p') || !name.ends_with(".bin") {
                continue;
            }
            let Ok(body) = std::fs::read(&path) else {
                continue;
            };
            // The dump holds every packet body; non-chunk bodies decode as
            // garbage, so the leading chunk coordinates gate the decode and
            // a panic guard skips anything that still slips through.
            let coords_ok = body.len() >= 8
                && body[0..4]
                    .try_into()
                    .map(|b: [u8; 4]| i32::from_be_bytes(b).abs() <= 48)
                    .unwrap_or(false)
                && body[4..8]
                    .try_into()
                    .map(|b: [u8; 4]| i32::from_be_bytes(b).abs() <= 48)
                    .unwrap_or(false);
            if !coords_ok {
                continue;
            }
            let cells = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let chunk = WireChunk::decode(&body).ok()?;
                if chunk.sections.len() != SECTION_SPAN || chunk.heightmaps.len() > 8 {
                    return None;
                }
                Some(((chunk.x, chunk.z), wire_cells(&chunk)))
            }))
            .ok()
            .flatten();
            let Some((pos, cells)) = cells else {
                continue;
            };
            if chunks.iter().any(|(p, _)| *p == pos) {
                continue;
            }
            let mut pairs: Vec<(i32, i32)> = Vec::new();
            for lz in 0..16usize {
                for lx in 0..16usize {
                    let trunk = (0..LAYERS).any(|layer| {
                        reg.state_of(cells[layer * COLUMNS + lz * 16 + lx])
                            .is_some_and(|(n, _)| n.ends_with("_log"))
                    });
                    if trunk {
                        pairs.push((lx as i32, lz as i32));
                    }
                }
            }
            chunks.push((pos, pairs));
        }
        chunks.sort_by_key(|(pos, _)| *pos);
        chunks.dedup_by_key(|(pos, _)| *pos);
        let forest: Vec<&ChunkPairs> = chunks
            .iter()
            .filter(|(_, pairs)| !pairs.is_empty())
            .collect();
        println!(
            "dump: {} chunks, {} with trunk columns",
            chunks.len(),
            forest.len()
        );

        let widest = {
            let table = BiomeTable::load(&pins()).unwrap();
            let plan = FeaturePlan::build(&pins(), &table).unwrap();
            plan.steps.iter().map(|s| s.len()).max().unwrap_or(0) as i32
        };
        let variants = [("block", EDGE), ("chunk", 1i32)];
        let mut ranked: Vec<(usize, &'static str, i32, i32)> = Vec::new();
        for (label, scale) in variants {
            for step in 0..=10i32 {
                for index in 0..=widest {
                    let mut hits = 0usize;
                    for ((cx, cz), pairs) in &forest {
                        let mut probe = DecorRng::new();
                        let deco = probe.decoration_seed(42, cx * scale, cz * scale);
                        probe.set_feature_seed(deco, index, step);
                        let pair = (probe.next_int(16), probe.next_int(16));
                        if pairs.contains(&pair) {
                            hits += 1;
                        }
                    }
                    ranked.push((hits, label, step, index));
                }
            }
        }
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(b.1)));
        println!("top (chunk votes, seed inputs, step, index):");
        for row in ranked.iter().take(12) {
            println!("  {row:?}");
        }
        let chance = forest.iter().map(|(_, pairs)| pairs.len()).sum::<usize>() as f64
            / 256.0
            / forest.len() as f64;
        println!(
            "chance level {chance:.2} over {} voting chunks",
            forest.len()
        );
    }

    /// Locates every recorded try origin of the target chunk inside the
    /// seeded word stream (each nextInt(16) consumes one word), so the
    /// words spent per try body are visible and the word offset of the
    /// reference run's own try sequence can be read off the same stream.
    #[test]
    #[ignore = "diagnostic: needs a live capture dump"]
    fn probe_word_offsets() {
        let target = (-1i32, -3i32);
        let reg = registry();
        let terrain = HeightmapGenerator::with_seed(42, &reg).unwrap();
        let hits = scan_dump(target);
        let mut order: Vec<(i32, i32)> = hits.iter().map(|(_, pos, _)| *pos).collect();
        order.dedup();
        if order.len() != 9 {
            order = (target.0 - 1..=target.0 + 1)
                .flat_map(|x| (target.1 - 1..=target.1 + 1).map(move |z| (x, z)))
                .collect();
        }
        let mut dec = Decorator::new(&terrain, &reg, 42).unwrap();
        dec.visits = Some(Vec::new());
        dec.try_words = Some(Vec::new());
        for &(cx, cz) in &order {
            dec.decorate(cx, cz);
        }
        let feature = "dark_forest_vegetation";
        let visits = dec.visits.take().unwrap_or_default();
        let words = dec.try_words.take().unwrap_or_default();
        let mut rows: Vec<(i32, i32, u64)> = Vec::new();
        for (visit, word) in visits.iter().zip(words.iter()) {
            let (n, x, _, z) = visit;
            if n != feature {
                continue;
            }
            let in_chunk = *x >= target.0 * 16
                && *x < target.0 * 16 + 16
                && *z >= target.1 * 16
                && *z < target.1 * 16 + 16;
            if in_chunk {
                rows.push((*x, *z, *word));
            }
        }
        let base = rows.first().map(|r| r.2).unwrap_or(0) as i64;
        println!("{} tries in the target chunk", rows.len());
        let mut prev: Option<i64> = None;
        for (k, (x, z, word)) in rows.iter().enumerate() {
            let word = *word as i64;
            let body = prev.map(|p| word - p);
            println!(
                "try {k:2} ({},{}) pair words {}..{} (body {body:?})",
                x - target.0 * 16,
                z - target.1 * 16,
                word - base - 2,
                word - base - 1
            );
            prev = Some(word);
        }
        // The reference run drew its own tries from the same stream; the
        // vanilla trunk bases mark where its try pairs landed.
        let mut r = DecorRng::new();
        let deco = r.decoration_seed(42, target.0 * EDGE, target.1 * EDGE);
        r.set_feature_seed(deco, 20, 9);
        let stream: Vec<i32> = (0..8192).map(|_| r.next_int(16)).collect();
        let origins = [(2i32, 9i32), (6, 3), (0, 1), (14, 6)];
        for pair in origins {
            let mut hits = Vec::new();
            for i in 0..stream.len() - 1 {
                if stream[i] == pair.0 && stream[i + 1] == pair.1 {
                    hits.push(i);
                }
            }
            println!("origin {pair:?} pair words {hits:?}");
        }
    }

    /// Sweeps the feature index (one step) by replaying the vegetation
    /// feature over the target chunk per candidate seed and scoring the
    /// try origins against the captured trunk columns: the index the
    /// reference run seeded from lands whole trunk bases on its try list,
    /// every other index stays at chance.
    #[test]
    #[ignore = "diagnostic: needs a live capture dump"]
    fn probe_seed_sweep() {
        let target = diag_target();
        let reg = registry();
        let terrain = HeightmapGenerator::with_seed(42, &reg).unwrap();
        let hits = scan_dump(target);
        let vanilla = hits
            .iter()
            .find(|(_, pos, _)| *pos == target)
            .map(|(_, _, chunk)| chunk)
            .expect("target chunk in the capture dump")
            .clone();
        let cells = wire_cells(&vanilla);
        let mut cols: Vec<(i32, i32)> = Vec::new();
        for lz in 0..16usize {
            for lx in 0..16usize {
                for layer in 0..LAYERS {
                    let state = cells[layer * COLUMNS + lz * 16 + lx];
                    if reg
                        .state_of(state)
                        .is_some_and(|(n, _)| n.ends_with("_log"))
                    {
                        cols.push((target.0 * 16 + lx as i32, target.1 * 16 + lz as i32));
                        break;
                    }
                }
            }
        }

        let feature = "dark_forest_vegetation";
        let step = std::env::var("DIAG_STEP")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(9);
        let widest = {
            let probe = Decorator::new(&terrain, &reg, 42).unwrap();
            probe.plan.steps.iter().map(|s| s.len()).max().unwrap_or(0)
        };
        let key = {
            let probe = Decorator::new(&terrain, &reg, 42).unwrap();
            probe.key_of(feature).expect("plan key")
        };
        let mut scored: Vec<(usize, i32, usize)> = Vec::new();
        for index in 0..=widest as i32 {
            let mut dec = Decorator::new(&terrain, &reg, 42).unwrap();
            dec.seed_override = Some((key, index, step as i32));
            dec.only_step = Some(step);
            dec.visits = Some(Vec::new());
            dec.decorate(target.0, target.1);
            let tries: Vec<(i32, i32)> = dec
                .visits
                .take()
                .unwrap_or_default()
                .iter()
                .filter(|(n, ..)| n == feature)
                .map(|(_, x, _, z)| (*x, *z))
                .collect();
            let hit = tries.iter().filter(|t| cols.contains(t)).count();
            scored.push((hit, index, tries.len()));
        }
        let mut ranked = scored.clone();
        ranked.sort_by_key(|(hit, _, _)| std::cmp::Reverse(*hit));
        println!("candidate (trunk-column hits, index, tries):");
        for row in ranked.iter().take(10) {
            println!("  {row:?}");
        }
        let own = Decorator::new(&terrain, &reg, 42)
            .unwrap()
            .plan
            .positions
            .get(&key)
            .map(|(s, i)| (*s as i32, *i as i32));
        println!("our plan slot: {own:?}");
        // The top candidate's tries against the captured columns.
        let best = ranked[0].1;
        let mut dec = Decorator::new(&terrain, &reg, 42).unwrap();
        dec.seed_override = Some((key, best, step as i32));
        dec.only_step = Some(step);
        dec.visits = Some(Vec::new());
        dec.decorate(target.0, target.1);
        let tries: Vec<(i32, i32)> = dec
            .visits
            .take()
            .unwrap_or_default()
            .iter()
            .filter(|(n, ..)| n == feature)
            .map(|(_, x, _, z)| (*x, *z))
            .collect();
        let hits_of: Vec<String> = tries
            .iter()
            .map(|t| {
                if cols.contains(t) {
                    format!("{t:?}*")
                } else {
                    format!("{t:?}")
                }
            })
            .collect();
        println!("best index {best} tries: {hits_of:?}");
        println!("vanilla trunk columns: {cols:?}");
    }

    /// The decoration seed folds the world seed with two odd scaling draws
    /// of the chunk origin; the per-feature reseed mixes the step and the
    /// index within the step. Spot values come from the reference chain:
    /// the origin folds to the world seed itself, and different chunks,
    /// indexes, or steps draw different streams.
    #[test]
    fn decoration_seed_chain_spot_values() {
        let mut rng = DecorRng::new();
        assert_eq!(rng.decoration_seed(42, 0, 0), 42);
        let mut rng = DecorRng::new();
        assert_eq!(rng.decoration_seed(42, 16, 32), -2_907_997_360_337_813_702);
        let mut rng = DecorRng::new();
        assert_eq!(rng.decoration_seed(42, 16, 0), -1_348_197_766_006_825_830);
        let mut rng = DecorRng::new();
        assert_eq!(rng.decoration_seed(42, 0, 16), 8_443_472_239_689_281_818);

        let mut rng = seeded();
        rng.set_feature_seed(-2_907_997_360_337_813_702, 3, 9);
        assert_eq!(
            (0..4).map(|_| rng.next_int(16)).collect::<Vec<_>>(),
            [8, 13, 5, 11]
        );
        assert_eq!(rng.next_f32(), 0.6658985f32);

        let mut rng = seeded();
        rng.set_feature_seed(-2_907_997_360_337_813_702, 4, 9);
        assert_eq!(
            (0..2).map(|_| rng.next_int(16)).collect::<Vec<_>>(),
            [14, 3],
            "the step index moves the stream"
        );

        let mut rng = seeded();
        rng.set_feature_seed(-2_907_997_360_337_813_702, 3, 10);
        assert_eq!(
            (0..2).map(|_| rng.next_int(16)).collect::<Vec<_>>(),
            [14, 13],
            "the step moves the stream"
        );
    }

    /// Every draw shape consumes the stream exactly the way the reference
    /// provider does: constants draw nothing, a pinned uniform still draws,
    /// the symmetric trapezoid draws twice, and weighted picks follow the
    /// weights.
    #[test]
    fn integer_draw_shapes() {
        // A constant leaves the stream untouched.
        let constant = draw(r#"{"type":"minecraft:constant","value":3}"#);
        let mut a = seeded();
        let first = a.next_int(100);
        let second = a.next_int(100);
        let mut b = seeded();
        assert_eq!(b.next_int(100), first);
        assert_eq!(constant.sample(&mut b), 3);
        assert_eq!(b.next_int(100), second, "constant drew nothing");

        // A uniform pinned to one value still consumes its draw.
        let pinned = draw(r#"{"type":"minecraft:uniform","min_inclusive":5,"max_inclusive":5}"#);
        let mut a = seeded();
        let value = pinned.sample(&mut a);
        let mut b = seeded();
        let _ = b.next_int(1);
        assert_eq!(value, 5);
        assert_eq!(a.next_int(100), b.next_int(100), "pinned uniform drew once");

        // The symmetric trapezoid (zero plateau around zero) draws two
        // bounded ints; values stay inside the triangle.
        let symmetric = draw(r#"{"type":"minecraft:trapezoid","min":-4,"max":4,"plateau":0}"#);
        let mut a = seeded();
        let mut b = seeded();
        for _ in 0..64 {
            let v = symmetric.sample(&mut a);
            assert!((-8..=8).contains(&v), "trapezoid value {v} out of range");
            let _ = b.next_int(5);
            let _ = b.next_int(5);
        }
        assert_eq!(
            a.next_int(100),
            b.next_int(100),
            "symmetric trapezoid drew twice"
        );

        // A clamped draw wraps its source.
        let clamped = draw(
            r#"{"type":"minecraft:clamped","min_inclusive":0,"max_inclusive":3,
                "source":{"type":"minecraft:uniform","min_inclusive":-5,"max_inclusive":5}}"#,
        );
        let mut rng = seeded();
        for _ in 0..32 {
            let v = clamped.sample(&mut rng);
            assert!((0..=3).contains(&v), "clamped value {v} out of range");
        }

        // A weighted list picks entries by weight.
        let weighted = draw(
            r#"{"type":"minecraft:weighted_list","distribution":[
                {"data":10,"weight":1},{"data":20,"weight":3}]}"#,
        );
        let mut rng = seeded();
        let mut tens = 0;
        let mut twenties = 0;
        for _ in 0..256 {
            match weighted.sample(&mut rng) {
                10 => tens += 1,
                20 => twenties += 1,
                other => panic!("weighted value {other}"),
            }
        }
        assert!(
            tens > 0 && twenties > tens * 2,
            "weights 1:3 got {tens}:{twenties}"
        );

        // The bare literal parses as the constant draw.
        let bare: Value = serde_json::from_str("16").unwrap();
        assert!(matches!(
            IntDraw::parse(&bare).unwrap(),
            IntDraw::Constant(16)
        ));
    }

    /// The feature order is the reference topological walk, not a stable
    /// sort by first encounter: with biome A listing f1 then f3 across
    /// steps and biome B listing f2 then f3, step 0 comes out [f2, f1]
    /// even though f1 was encountered first, because B's edge f2->f3 and
    /// A's edge f1->f3 both drain before the step fills. A feature listed
    /// under two steps lands in both. A feature that only ever ends a
    /// biome list still anchors the start iteration: it emits at its own
    /// (step, first-encounter) slot, not wherever a predecessor's walk
    /// reaches it. A contradictory pair of lists reports the cycle
    /// instead.
    #[test]
    fn feature_order_matches_the_reference_walk() {
        let a = vec![vec![0usize], vec![1]];
        let b = vec![vec![2], vec![1]];
        let steps = order_features(&[a, b], 2).unwrap();
        assert_eq!(steps, vec![vec![2, 0], vec![1]]);

        let a = vec![vec![0usize], vec![1]];
        let b = vec![vec![2, 1]];
        let steps = order_features(&[a, b], 2).unwrap();
        assert_eq!(steps, vec![vec![2, 1, 0], vec![1]]);

        // f0 ends every list it appears in; f1 sorts between it and f2.
        // The start loop emits f0 at its own slot, after f2 and before
        // f1's predecessor drains, so step 0 reads [f2, f1, f0].
        let solo0 = vec![vec![0usize]];
        let solo1 = vec![vec![1]];
        let tail = vec![vec![2, 0]];
        let steps = order_features(&[solo0, solo1, tail], 1).unwrap();
        assert_eq!(steps, vec![vec![2, 1, 0]]);

        let cycle_a = vec![vec![0usize, 1]];
        let cycle_b = vec![vec![1, 0]];
        assert!(order_features(&[cycle_a, cycle_b], 1).is_err());
    }

    /// The pinned plan: eleven steps, no feature twice in a step, every
    /// biome's step list ascending, and the dark forest's vegetal step
    /// carries its tree feature.
    #[test]
    fn pinned_plan_shapes() {
        let table = BiomeTable::load(&pins()).expect("biome table");
        let plan = FeaturePlan::build(&pins(), &table).expect("feature plan");
        assert_eq!(plan.steps.len(), 11, "decoration steps");

        let mut seen = std::collections::HashSet::new();
        for (step, keys) in plan.steps.iter().enumerate() {
            let mut positions = std::collections::HashSet::new();
            for &key in keys {
                assert!(positions.insert(key), "feature twice in step {step}");
                assert!(seen.insert((step, key)), "repeat ({step},{key})");
            }
        }
        for biome in table.biome_order() {
            for step in 0..plan.steps.len() {
                let indices = plan.step_indices(biome, step);
                for pair in indices.windows(2) {
                    assert!(
                        pair[0] < pair[1],
                        "biome {biome} step {step} order broke: {indices:?}"
                    );
                }
            }
        }

        let order: Vec<String> = serde_json::from_str(
            &std::fs::read_to_string(pins().join("biome_registry_order.json")).unwrap(),
        )
        .unwrap();
        let dark = order
            .iter()
            .position(|name| name == "minecraft:dark_forest")
            .unwrap() as u32;
        let vegetal = plan
            .names
            .iter()
            .position(|n| n == "dark_forest_vegetation");
        let Some(vegetal) = vegetal else {
            panic!("plan lacks dark_forest_vegetation");
        };
        assert!(plan.biome_has_feature(dark, vegetal));
        let step9: Vec<usize> = plan
            .step_indices(dark, 9)
            .iter()
            .map(|&i| plan.steps[9][i])
            .collect();
        assert!(step9.contains(&vegetal), "step 9 keys {step9:?}");
    }

    /// The driver over the pins at the spawn seed: features fire only in
    /// their biomes, positions stay inside the chunk column band the
    /// modifiers can produce, the trees write logs and leaves, and the
    /// visit list is a pure function of the chunk.
    #[test]
    fn decorates_dark_forest_chunk() {
        let reg = registry();
        let terrain = HeightmapGenerator::with_seed(42, &reg).expect("density generator");
        assert_eq!(terrain.biome_at(0, 64, 0), 9, "spawn chunk is dark forest");
        let mut dec = Decorator::new(&terrain, &reg, 42).expect("decorator");
        let vegetal = dec.key_of("dark_forest_vegetation").expect("plan key");

        // The river crossing the spawn window drowns the spawn chunk's
        // own trees (its water depth filter rejects every position), so
        // the tree assertions ride on the first dry forest chunk around
        // it.
        let mut tree_chunk = None;
        'search: for cx in -2..=2i32 {
            for cz in -2..=2i32 {
                let (x, z) = (cx * 16 + 8, cz * 16 + 8);
                dec.ensure_chunk(cx, cz);
                let floor = dec.height(HeightKind::OceanFloor, x, z);
                let surface = dec.height(HeightKind::WorldSurface, x, z);
                let biome = dec.biome_at(x, floor, z);
                if surface == floor && dec.biome_has_feature(biome, vegetal) {
                    tree_chunk = Some((cx, cz));
                    break 'search;
                }
            }
        }
        let tree_chunk = tree_chunk.expect("a dry forest chunk in the neighborhood");

        dec.visits = Some(Vec::new());
        dec.decorate(0, 0);
        let spawn_visits = dec.visits.take().expect("probe armed");
        dec.visits = Some(Vec::new());
        dec.decorate(tree_chunk.0, tree_chunk.1);
        let tree_visits = dec.visits.take().expect("probe armed");

        let check = |dec: &Decorator, visits: &[(String, i32, i32, i32)], cx: i32, cz: i32| {
            assert!(!visits.is_empty(), "supported features placed");
            let (bx, bz) = (cx * 16, cz * 16);
            for (name, x, y, z) in visits {
                // The square pick stays in the chunk; an offset modifier
                // may then walk a position into the neighboring band.
                assert!(
                    (bx - 16..bx + 32).contains(x) && (bz - 16..bz + 32).contains(z),
                    "{name} landed at ({x},{z})"
                );
                assert!(
                    *y >= MIN_Y && *y < MIN_Y + LAYERS as i32,
                    "{name} height {y}"
                );
                // The biome gate runs mid-stack, so an offset can carry a
                // visit into a biome that never listed the feature; only
                // a stack without offsets gates where the visit lands.
                if name == "glow_lichen" {
                    let key = dec.key_of("glow_lichen").expect("glow lichen key");
                    assert!(
                        dec.biome_has_feature(dec.biome_at(*x, *y, *z), key),
                        "glow_lichen placed outside its biome"
                    );
                }
            }
        };
        check(&dec, &spawn_visits, 0, 0);
        check(&dec, &tree_visits, tree_chunk.0, tree_chunk.1);

        let count = |name: &str| spawn_visits.iter().filter(|(n, ..)| n == name).count();
        assert!(count("glow_lichen") > 0, "glow lichen survived all filters");
        assert!(count("glow_lichen") <= 157, "count bounds the visits");
        assert!(
            count("forest_flowers") <= 1,
            "rarity 7 gate then clamped count 0..1"
        );
        assert_eq!(
            count("flower_default"),
            0,
            "predicate filter skips wholesale"
        );
        assert_eq!(
            count("seagrass_normal"),
            0,
            "predicate filter skips wholesale"
        );
        assert_eq!(count("kelp_cold"), 0, "noise count skips wholesale");
        assert_eq!(
            count("patch_tall_grass_2"),
            0,
            "noise threshold skips wholesale"
        );

        let (bx, bz) = (tree_chunk.0 * 16, tree_chunk.1 * 16);
        let mut logs = 0;
        let mut leaves = 0;
        let mut near_trunk = 0;
        for y in MIN_Y..MIN_Y + LAYERS as i32 {
            for x in bx..bx + 16 {
                for z in bz..bz + 16 {
                    let state = dec.block(x, y, z);
                    let name = dec.block_name(state);
                    if name.ends_with("_log") {
                        logs += 1;
                    }
                    if name.ends_with("_leaves") {
                        leaves += 1;
                        let props = dec.registry().state_of(state).map_or("", |(_, p)| p);
                        if BlockRegistry::prop_int(props, "distance").is_some_and(|v| v < 7) {
                            near_trunk += 1;
                        }
                    }
                }
            }
        }
        assert!(logs > 0, "trees wrote logs");
        assert!(leaves > 0, "trees wrote leaves");
        assert!(
            near_trunk * 4 > leaves * 3 / 2,
            "the leaf walk brought most leaves near a trunk: {near_trunk} of {leaves}"
        );

        // The visit lists are a pure function of the chunk set: a fresh
        // region over the same terrain reproduces them exactly.
        let mut again = Decorator::new(&terrain, &reg, 42).expect("decorator");
        again.visits = Some(Vec::new());
        again.decorate(0, 0);
        let replay_spawn = again.visits.take().expect("probe armed");
        again.visits = Some(Vec::new());
        again.decorate(tree_chunk.0, tree_chunk.1);
        let replay_tree = again.visits.take().expect("probe armed");
        assert_eq!(spawn_visits, replay_spawn, "decoration is deterministic");
        assert_eq!(tree_visits, replay_tree, "decoration is deterministic");
    }
}
