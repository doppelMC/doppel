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
use crate::noise::Xoroshiro;
use crate::registry::BlockRegistry;
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

/// The decoration random: the rotate-xor stream reseeded per stage.
pub struct DecorRng {
    rng: Xoroshiro,
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
        }
    }

    fn set_seed(&mut self, seed: i64) {
        self.rng.set_seed_wide(seed);
    }

    pub fn next_long(&mut self) -> i64 {
        self.rng.next_long()
    }

    /// The wide-multiply draw with rejection for bias.
    pub fn next_int(&mut self, bound: i32) -> i32 {
        debug_assert!(bound > 0);
        self.rng.next_int(bound)
    }

    pub fn next_f32(&mut self) -> f32 {
        self.rng.next_f32()
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
enum IntDraw {
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
    Clamped {
        min: i32,
        max: i32,
        source: Box<IntDraw>,
    },
    Weighted {
        total: i32,
        entries: Vec<(i32, IntDraw)>,
    },
}

impl IntDraw {
    fn parse(v: &Value) -> Result<IntDraw> {
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
            other => bail!("unsupported integer draw {other}"),
        }
    }

    fn sample(&self, rng: &mut DecorRng) -> i32 {
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
        }
    }
}

// ---------------------------------------------------------------------------
// Placement modifiers.
// ---------------------------------------------------------------------------

/// The heightmap kinds the placement stack snaps to.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum HeightKind {
    WorldSurface,
    OceanFloor,
    MotionBlocking,
    MotionBlockingNoLeaves,
}

impl HeightKind {
    fn parse(name: &str) -> Result<HeightKind> {
        match name {
            "WORLD_SURFACE" | "WORLD_SURFACE_WG" => Ok(HeightKind::WorldSurface),
            "OCEAN_FLOOR" | "OCEAN_FLOOR_WG" => Ok(HeightKind::OceanFloor),
            "MOTION_BLOCKING" => Ok(HeightKind::MotionBlocking),
            "MOTION_BLOCKING_NO_LEAVES" => Ok(HeightKind::MotionBlockingNoLeaves),
            other => bail!("unsupported heightmap {other}"),
        }
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
    /// Recognized but not evaluated: features carrying one are skipped.
    Unsupported,
}

impl Modifier {
    fn parse(v: &Value) -> Result<Modifier> {
        let kind = v.get("type").and_then(Value::as_str).context("modifier")?;
        match kind {
            "minecraft:count" => Ok(Modifier::Count(IntDraw::parse(
                v.get("count").context("count")?,
            )?)),
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
            "minecraft:offset" => Ok(Modifier::Offset(
                IntDraw::parse(v.get("x").context("offset x")?)?,
                IntDraw::parse(v.get("y").context("offset y")?)?,
                IntDraw::parse(v.get("z").context("offset z")?)?,
            )),
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
            "minecraft:block_predicate_filter"
            | "minecraft:environment_scan"
            | "minecraft:count_on_every_layer"
            | "minecraft:noise_based_count"
            | "minecraft:noise_threshold_count"
            | "minecraft:random_chance"
            | "minecraft:randomly_selected" => Ok(Modifier::Unsupported),
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
        let placement = v
            .get("placement")
            .and_then(Value::as_array)
            .context("placement stack")?
            .iter()
            .map(Modifier::parse)
            .collect::<Result<Vec<_>>>()?;
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

/// Joins the biome feature lists into per-step orders: edges run from
/// each biome's feature to its successor, a depth-first walk over the
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
        for pair in flat.windows(2) {
            edges.entry(pair[0]).or_default().insert(pair[1]);
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
            HeightKind::WorldSurface => !f.air,
            HeightKind::OceanFloor => f.motion,
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
    features: HashMap<usize, Option<PlacedFeatureCfg>>,
    chunks: HashMap<(i32, i32), ChunkState>,
    traits: StateTraits,
    tags: TagResolver,
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
        Ok(Decorator {
            terrain,
            world_seed,
            plan,
            tags: TagResolver::new(&pins),
            pins,
            registry,
            features: HashMap::new(),
            chunks: HashMap::new(),
            traits: StateTraits::default(),
        })
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

    /// Ensures the chunk exists in the region (terrain + biomes).
    fn ensure_chunk(&mut self, cx: i32, cz: i32) {
        if self.chunks.contains_key(&(cx, cz)) {
            return;
        }
        let (blocks, _) = self.terrain.build_blocks(cx, cz);
        let biomes = self
            .terrain
            .section_biomes(cx, cz)
            .unwrap_or([[0u32; 64]; SECTION_SPAN]);
        self.chunks.insert(
            (cx, cz),
            ChunkState {
                blocks,
                biomes,
                heights: HashMap::new(),
                decorated: false,
            },
        );
    }

    /// Decorates the chunk exactly once: possible biomes come from the
    /// 3x3 neighborhood's stored cells, and each candidate feature
    /// reseeds from its step and index before its placement runs.
    /// `place` receives the final positions.
    pub fn decorate<F>(&mut self, cx: i32, cz: i32, place: &mut F)
    where
        F: FnMut(&mut Decorator<'a>, usize, &mut DecorRng, i32, i32, i32) -> bool,
    {
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
            let mut indices: BTreeSet<usize> = BTreeSet::new();
            for biome in &possible {
                indices.extend(self.plan.step_indices(*biome, step));
            }
            for index in indices {
                let key = self.plan.steps[step][index];
                rng.set_feature_seed(decoration_seed, index as i32, step as i32);
                self.place_placed(key, &mut rng, cx * EDGE, cz * EDGE, place);
            }
        }
    }

    /// Runs one placed feature's placement stack from the chunk origin.
    fn place_placed<F>(&mut self, key: usize, rng: &mut DecorRng, ox: i32, oz: i32, place: &mut F)
    where
        F: FnMut(&mut Decorator<'a>, usize, &mut DecorRng, i32, i32, i32) -> bool,
    {
        let Some(cfg) = self.load_feature(key).cloned() else {
            return;
        };
        if cfg.has_unsupported() || cfg.placement.is_empty() {
            return;
        }
        let span = LAYERS as i32;
        // Worklist entries: (x, y, z, modifier index).
        let mut stack: Vec<(i32, i32, i32, usize)> = vec![(ox, MIN_Y, oz, 0)];
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
                    if self.plan.biome_has_feature(self.biome_at(x, y, z), key) {
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
                Modifier::Unsupported => return,
            }
            let next = index + 1;
            if next < cfg.placement.len() {
                for pos in out.iter().rev() {
                    stack.push((pos.0, pos.1, pos.2, next));
                }
            } else {
                for &(px, py, pz) in &out {
                    place(self, key, rng, px, py, pz);
                }
            }
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
    fn height(&mut self, kind: HeightKind, x: i32, z: i32) -> i32 {
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

    /// Emits the chunk as a wire chunk from its current buffer.
    pub fn emit(&mut self, cx: i32, cz: i32) -> WireChunk {
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
        assert_eq!(rng.decoration_seed(42, 16, 32), -6_221_029_433_860_675_718);
        let mut rng = DecorRng::new();
        assert_eq!(rng.decoration_seed(42, 16, 0), -1_348_197_764_963_659_302);
        let mut rng = DecorRng::new();
        assert_eq!(rng.decoration_seed(42, 0, 16), 6_786_956_202_406_267_546);

        let mut rng = seeded();
        rng.set_feature_seed(-6_221_029_433_860_675_718, 3, 9);
        assert_eq!(
            (0..4).map(|_| rng.next_int(16)).collect::<Vec<_>>(),
            [2, 7, 11, 8]
        );
        assert_eq!(rng.next_f32(), 0.7069545388221741f32);

        let mut rng = seeded();
        rng.set_feature_seed(-6_221_029_433_860_675_718, 4, 9);
        assert_eq!(rng.next_int(16), 4, "the step index moves the stream");

        let mut rng = seeded();
        rng.set_feature_seed(-6_221_029_433_860_675_718, 3, 10);
        assert_eq!(rng.next_int(16), 11, "the step moves the stream");
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
    /// under two steps lands in both. A contradictory pair of lists
    /// reports the cycle instead.
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
    /// modifiers can produce, and the visit list is a pure function of the
    /// chunk.
    #[test]
    fn decorates_dark_forest_chunk() {
        let reg = registry();
        let terrain = HeightmapGenerator::with_seed(42, &reg).expect("density generator");
        assert_eq!(terrain.biome_at(0, 64, 0), 9, "spawn chunk is dark forest");
        let mut dec = Decorator::new(&terrain, &reg, 42).expect("decorator");

        let mut visits: Vec<(String, i32, i32, i32)> = Vec::new();
        {
            let mut sink =
                |d: &mut Decorator, key: usize, _rng: &mut DecorRng, x: i32, y: i32, z: i32| {
                    visits.push((d.feature_name(key).to_string(), x, y, z));
                    true
                };
            dec.decorate(0, 0, &mut sink);
        }
        assert!(!visits.is_empty(), "supported features placed");
        for (name, x, y, z) in &visits {
            assert!(
                (0..16).contains(x) && (0..16).contains(z),
                "{name} landed at ({x},{z})"
            );
            assert!(
                *y >= MIN_Y && *y < MIN_Y + LAYERS as i32,
                "{name} height {y}"
            );
            assert!(
                dec.biome_has_feature(dec.biome_at(*x, *y, *z), dec.key_of(name).unwrap()),
                "{name} placed outside its biome"
            );
        }
        let count = |name: &str| visits.iter().filter(|(n, ..)| n == name).count();
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

        // The visit list is a pure function of the chunk: a fresh region
        // over the same terrain reproduces it exactly.
        let mut again = Decorator::new(&terrain, &reg, 42).expect("decorator");
        let mut replay: Vec<(String, i32, i32, i32)> = Vec::new();
        {
            let mut sink =
                |d: &mut Decorator, key: usize, _rng: &mut DecorRng, x: i32, y: i32, z: i32| {
                    replay.push((d.feature_name(key).to_string(), x, y, z));
                    true
                };
            again.decorate(0, 0, &mut sink);
        }
        assert_eq!(visits, replay, "decoration is deterministic");
    }
}
