//! The feature interpreter: the configs a placement stack hands a
//! position to, run against the region's blocks.
//!
//! Every draw the reference features make is reproduced in shape and
//! order, because one feature's leftover stream moves every later
//! position of that same feature. Configs therefore resolve their state
//! providers up front: an unsupported shape skips the feature before the
//! stream moves, and everything the spawn set uses is supported.

use std::collections::HashMap;
use std::f64::consts::PI;

use serde_json::Value;

use crate::decoration::{
    DecorRng, Decorator, HeightKind, IntDraw, PlacedFeatureCfg, Predicate, LAYERS,
};
use crate::registry::BlockRegistry;
use crate::worldgen::MIN_Y;

/// The world build ceiling.
const WORLD_TOP: i32 = 320;

// ---------------------------------------------------------------------------
// Faces.
// ---------------------------------------------------------------------------

/// One face: a horizontal step plus its property name (vertical faces
/// carry no horizontal step).
#[derive(Clone, Copy, PartialEq, Eq)]
struct Face {
    dx: i32,
    dz: i32,
    prop: &'static str,
}

const UP: Face = Face {
    dx: 0,
    dz: 0,
    prop: "up",
};
const DOWN: Face = Face {
    dx: 0,
    dz: 0,
    prop: "down",
};
const NORTH: Face = Face {
    dx: 0,
    dz: -1,
    prop: "north",
};
const EAST: Face = Face {
    dx: 1,
    dz: 0,
    prop: "east",
};
const SOUTH: Face = Face {
    dx: 0,
    dz: 1,
    prop: "south",
};
const WEST: Face = Face {
    dx: -1,
    dz: 0,
    prop: "west",
};

/// Horizontal faces in the reference plane order (north, east, south,
/// west); every horizontal draw indexes this list.
const HORIZONTAL: [Face; 4] = [NORTH, EAST, SOUTH, WEST];

/// All six faces in the reference enum order (down, up, north, south,
/// west, east); shuffles draw over this list.
const ALL: [Face; 6] = [DOWN, UP, NORTH, SOUTH, WEST, EAST];

impl Face {
    fn parse(name: &str) -> Option<Face> {
        Some(match name {
            "up" => UP,
            "down" => DOWN,
            "north" => NORTH,
            "east" => EAST,
            "south" => SOUTH,
            "west" => WEST,
            _ => return None,
        })
    }

    fn step(&self, x: i32, y: i32, z: i32) -> (i32, i32, i32) {
        match *self {
            UP => (x, y + 1, z),
            DOWN => (x, y - 1, z),
            other => (x + other.dx, y, z + other.dz),
        }
    }

    fn opposite(&self) -> Face {
        match self.prop {
            "up" => DOWN,
            "down" => UP,
            "north" => SOUTH,
            "south" => NORTH,
            "east" => WEST,
            _ => EAST,
        }
    }

    /// The pillar axis name a log lying along this face takes.
    fn axis(&self) -> &'static str {
        match self.dx {
            0 => "z",
            _ => "x",
        }
    }

    /// The world axis group (y, z, x) the face steps along.
    fn axis_group(&self) -> u8 {
        match (self.dx, self.dz) {
            (0, 0) => 0,
            (0, _) => 1,
            _ => 2,
        }
    }
}

// ---------------------------------------------------------------------------
// State providers.
// ---------------------------------------------------------------------------

/// A block state source. Weighted providers draw once per sample; the
/// soil provider looks at the position block; the int overlay draws
/// its source first and then the property value.
#[derive(Clone)]
enum StateProvider {
    Fixed(u32),
    Weighted {
        total: i32,
        entries: Vec<(i32, StateProvider)>,
    },
    RandomizedInt {
        property: String,
        values: IntDraw,
        source: Box<StateProvider>,
    },
    /// Dirt unless the position block vetoes replacement.
    Soil,
    None,
}

/// Property text from a json properties object.
fn props_text(v: Option<&Value>) -> String {
    let Some(fields) = v.and_then(Value::as_object) else {
        return String::new();
    };
    let mut pairs: Vec<(&str, &str)> = fields
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str().unwrap_or("")))
        .collect();
    pairs.sort_unstable();
    pairs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// A plain state (bare name or id-plus-properties object) as a state id.
pub(crate) fn plain_state(d: &Decorator, v: &Value) -> Option<u32> {
    match v {
        Value::String(name) => d.state_id_of(name, ""),
        Value::Object(_) => {
            let id = v.get("id").and_then(Value::as_str)?;
            d.state_id_of(id, &props_text(v.get("properties")))
        }
        _ => None,
    }
}

/// A block's default state with one property forced: the shape the
/// reference composes when it derives states from defaults.
fn state_with_prop(d: &Decorator, name: &str, prop: &str, value: &str) -> Option<u32> {
    let base = d.state_id_of(name, "")?;
    let (_, props) = d.registry().state_of(base)?;
    if !props.split(',').any(|pair| pair.starts_with(prop)) {
        return Some(base);
    }
    d.state_id_of(name, &BlockRegistry::with_prop(props, prop, value))
}

impl StateProvider {
    fn parse(d: &mut Decorator, v: &Value) -> StateProvider {
        match v {
            Value::String(name) if name == "minecraft:soil_beneath_tree" => StateProvider::Soil,
            // A bare name resolves through the provider registry first,
            // then falls back to a plain block state.
            Value::String(name) => {
                let key = name.strip_prefix("minecraft:").unwrap_or(name);
                match d.load_state_provider(key) {
                    Some(inner) => StateProvider::parse(d, &inner),
                    None => plain_state(d, v).map_or(StateProvider::None, StateProvider::Fixed),
                }
            }
            Value::Object(_) => match v.get("type").and_then(Value::as_str) {
                Some("minecraft:weighted") => {
                    let mut entries = Vec::new();
                    for entry in v
                        .get("entries")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                    {
                        let Some(weight) = entry.get("weight").and_then(Value::as_i64) else {
                            return StateProvider::None;
                        };
                        let Some(data) = entry.get("data") else {
                            return StateProvider::None;
                        };
                        let provider = StateProvider::parse(d, data);
                        if matches!(provider, StateProvider::None) {
                            return StateProvider::None;
                        }
                        entries.push((weight as i32, provider));
                    }
                    let total: i64 = entries.iter().map(|(w, _)| *w as i64).sum();
                    if total <= 0 || total > i32::MAX as i64 {
                        return StateProvider::None;
                    }
                    StateProvider::Weighted {
                        total: total as i32,
                        entries,
                    }
                }
                Some("minecraft:randomized_int") => {
                    let Some(source) = v.get("source") else {
                        return StateProvider::None;
                    };
                    let source = StateProvider::parse(d, source);
                    if matches!(source, StateProvider::None) {
                        return StateProvider::None;
                    }
                    let Some(property) = v.get("property").and_then(Value::as_str) else {
                        return StateProvider::None;
                    };
                    let Ok(values) = IntDraw::parse(v.get("values").unwrap_or(&Value::Null)) else {
                        return StateProvider::None;
                    };
                    StateProvider::RandomizedInt {
                        property: property.to_string(),
                        values,
                        source: Box::new(source),
                    }
                }
                Some("minecraft:soil_beneath_tree") => StateProvider::Soil,
                Some("minecraft:simple_state_provider") => match v.get("state") {
                    Some(state) => {
                        plain_state(d, state).map_or(StateProvider::None, StateProvider::Fixed)
                    }
                    None => StateProvider::None,
                },
                _ => plain_state(d, v).map_or(StateProvider::None, StateProvider::Fixed),
            },
            _ => StateProvider::None,
        }
    }

    /// Draws the state for a position; None places nothing.
    fn sample(&self, d: &mut Decorator, rng: &mut DecorRng, x: i32, y: i32, z: i32) -> Option<u32> {
        match self {
            StateProvider::Fixed(state) => Some(*state),
            StateProvider::Weighted { total, entries } => {
                let mut pick = rng.next_int(*total);
                for (weight, provider) in entries {
                    if pick < *weight {
                        return provider.sample(d, rng, x, y, z);
                    }
                    pick -= weight;
                }
                entries.last()?.1.sample(d, rng, x, y, z)
            }
            // The source draws first; a state without the property keeps
            // the source's value and draws nothing.
            StateProvider::RandomizedInt {
                property,
                values,
                source,
            } => {
                let base = source.sample(d, rng, x, y, z)?;
                let (name, props) = d.registry().state_of(base)?;
                if !props.split(',').any(|pair| pair.starts_with(property)) {
                    return Some(base);
                }
                let value = values.sample(rng);
                let props = BlockRegistry::with_prop(props, property, &value.to_string());
                d.state_id_of(name, &props)
            }
            StateProvider::Soil => {
                let here = d.block_name(d.block(x, y, z)).to_string();
                if d.tag_contains("cannot_replace_below_tree_trunk", &here) {
                    None
                } else {
                    d.state_id_of("minecraft:dirt", "")
                }
            }
            StateProvider::None => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Feature dispatch.
// ---------------------------------------------------------------------------

/// Feature kinds the pins carry but the engine places nothing for yet.
/// Plan validation accepts exactly these and the dispatch arms; any other
/// kind fails the plan load instead of skipping silently.
pub(crate) const UNPLACED_FEATURE_KINDS: [&str; 33] = [
    "minecraft:overlay",
    "minecraft:block_pile",
    "minecraft:huge_fungus",
    "minecraft:sequence",
    "minecraft:bamboo",
    "minecraft:netherrack_replace_blobs",
    "minecraft:speleothem_cluster",
    "minecraft:coral_claw",
    "minecraft:coral_tree",
    "minecraft:speleothem",
    "minecraft:stepped_column_cluster",
    "minecraft:end_gateway",
    "minecraft:end_podium",
    "minecraft:fossil",
    "minecraft:iceberg",
    "minecraft:scattered_ore",
    "minecraft:geode",
    "minecraft:blue_ice",
    "minecraft:bonus_chest",
    "minecraft:chorus_plant",
    "minecraft:delta_feature",
    "minecraft:end_island",
    "minecraft:end_platform",
    "minecraft:end_spike",
    "minecraft:block_blob",
    "minecraft:freeze_top_layer",
    "minecraft:random_neighbor_spread",
    "minecraft:spike",
    "minecraft:lake",
    "minecraft:large_dripstone",
    "minecraft:monster_room",
    "minecraft:underwater_magma",
    "minecraft:void_start_platform",
];

/// Trunk placers, foliage placers, and tree decorators the tree parser
/// does not model: a tree config carrying one accepts as unplaced (the
/// runtime rejects it there), and plan validation accepts exactly these.
pub(crate) const UNPLACED_TREE_SHAPES: [&str; 22] = [
    "minecraft:cherry_trunk_placer",
    "minecraft:forking_trunk_placer",
    "minecraft:giant_trunk_placer",
    "minecraft:mega_jungle_trunk_placer",
    "minecraft:poplar_trunk_placer",
    "minecraft:upwards_branching_trunk_placer",
    "minecraft:acacia_foliage_placer",
    "minecraft:bush_foliage_placer",
    "minecraft:cherry_foliage_placer",
    "minecraft:jungle_foliage_placer",
    "minecraft:mega_pine_foliage_placer",
    "minecraft:pine_foliage_placer",
    "minecraft:poplar_foliage_placer",
    "minecraft:spruce_foliage_placer",
    "minecraft:alter_ground",
    "minecraft:attached_to_leaves",
    "minecraft:cocoa",
    "minecraft:creaking_heart",
    "minecraft:leave_vine",
    "minecraft:pale_moss",
    "minecraft:shelf_mushroom",
    "minecraft:trunk_vine",
];

/// Runs the feature config at a resolved position and reports whether
/// the feature placed. A bare string names a registered config; unplaced
/// kinds draw nothing and place nothing, and plan validation has already
/// rejected everything else.
pub(crate) fn run_feature(
    d: &mut Decorator,
    v: &Value,
    rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
) -> bool {
    let resolved;
    let v = match v {
        Value::String(id) => {
            let key = id.strip_prefix("minecraft:").unwrap_or(id);
            match d.load_feature_config(key) {
                Some(cfg) => {
                    resolved = cfg;
                    &resolved
                }
                None => return false,
            }
        }
        other => other,
    };
    let Some(kind) = v.get("type").and_then(Value::as_str) else {
        return false;
    };
    match kind {
        "minecraft:random_selector" => {
            // One float per entry in list order; the first pass wins and
            // the default feature runs when none does.
            let Some(list) = v.get("features").and_then(Value::as_array) else {
                debug_assert!(false, "selector without a features list");
                return false;
            };
            for entry in list {
                let Some(chance) = entry.get("chance").and_then(Value::as_f64) else {
                    debug_assert!(false, "selector entry without a chance");
                    return false;
                };
                if rng.next_f32() < chance as f32 {
                    return run_placed_value(d, entry.get("feature"), rng, x, y, z);
                }
            }
            run_placed_value(d, v.get("default"), rng, x, y, z)
        }
        "minecraft:simple_random_selector" => {
            let Some(list) = v.get("features").and_then(Value::as_array) else {
                debug_assert!(false, "selector without a features list");
                return false;
            };
            if list.is_empty() {
                debug_assert!(false, "selector without entries");
                return false;
            }
            let pick = rng.next_int(list.len() as i32) as usize;
            run_placed_value(d, list.get(pick), rng, x, y, z)
        }
        "minecraft:weighted_random_selector" => {
            let Some(raw) = v.get("features").and_then(Value::as_array) else {
                debug_assert!(false, "selector without a features list");
                return false;
            };
            let mut entries: Vec<(i32, &Value)> = Vec::with_capacity(raw.len());
            for entry in raw {
                let Some(weight) = entry.get("weight").and_then(Value::as_i64) else {
                    debug_assert!(false, "weighted entry without a weight");
                    return false;
                };
                let Some(data) = entry.get("data") else {
                    debug_assert!(false, "weighted entry without data");
                    return false;
                };
                entries.push((weight as i32, data));
            }
            let total: i64 = entries.iter().map(|(w, _)| *w as i64).sum();
            if total <= 0 || total > i32::MAX as i64 {
                debug_assert!(false, "weighted total out of range");
                return false;
            }
            let mut pick = rng.next_int(total as i32);
            let mut chosen = entries.len() - 1;
            for (i, (weight, _)) in entries.iter().enumerate() {
                if pick < *weight {
                    chosen = i;
                    break;
                }
                pick -= weight;
            }
            run_placed_value(d, Some(entries[chosen].1), rng, x, y, z)
        }
        "minecraft:simple_block" => run_simple_block(d, v, rng, x, y, z),
        "minecraft:tree" => run_tree(d, v, rng, x, y, z),
        "minecraft:fallen_tree" => {
            run_fallen_tree(d, v, rng, x, y, z);
            true
        }
        "minecraft:huge_brown_mushroom" | "minecraft:huge_red_mushroom" => {
            run_huge_mushroom(d, v, rng, x, y, z, kind.ends_with("red_mushroom"));
            true
        }
        "minecraft:block_column" => run_block_column(d, v, rng, x, y, z),
        "minecraft:multiface_growth" => {
            run_multiface(d, v, rng, x, y, z);
            true
        }
        "minecraft:ore" => {
            crate::ores::run_ore(d, v, rng, x, y, z);
            true
        }
        "minecraft:disk" => {
            crate::ores::run_disk(d, v, rng, x, y, z);
            true
        }
        "minecraft:spring_feature" => crate::lush::run_spring(d, v, rng, x, y, z),
        "minecraft:vegetation_patch" | "minecraft:waterlogged_vegetation_patch" => {
            crate::lush::run_patch(d, v, rng, x, y, z)
        }
        "minecraft:vines" => crate::lush::run_vines(d, v, rng, x, y, z),
        "minecraft:root_system" => crate::lush::run_root(d, v, rng, x, y, z),
        // One boolean draw picks the branch; the chosen placed feature
        // runs with the same stream.
        "minecraft:random_boolean_selector" => {
            let chosen = if rng.next_bool() {
                v.get("feature_true")
            } else {
                v.get("feature_false")
            };
            run_placed_value(d, chosen, rng, x, y, z)
        }
        other => {
            debug_assert!(
                UNPLACED_FEATURE_KINDS.contains(&other),
                "unvalidated feature kind {other}"
            );
            false
        }
    }
}

/// Runs a placed-feature reference (a registered id or an inline object)
/// and reports whether anything placed.
pub(crate) fn run_placed_value(
    d: &mut Decorator,
    v: Option<&Value>,
    rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
) -> bool {
    match v {
        Some(Value::String(id)) => {
            let id = id.strip_prefix("minecraft:").unwrap_or(id).to_string();
            let key = d.key_of(&id);
            match d.load_named(&id) {
                Some(cfg) => d.eval_placement(&id, key, &cfg, rng, x, y, z),
                None => false,
            }
        }
        Some(value @ Value::Object(_)) => match PlacedFeatureCfg::from_value(value) {
            Ok(cfg) => {
                let name = value
                    .get("feature")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                d.eval_placement(&name, None, &cfg, rng, x, y, z)
            }
            Err(_) => false,
        },
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Predicates and survival.
// ---------------------------------------------------------------------------

/// Evaluates a block predicate at a position (its offset applied).
pub(crate) fn test_predicate(d: &mut Decorator, p: &Predicate, x: i32, y: i32, z: i32) -> bool {
    match p {
        Predicate::Blocks { names, offset } => {
            let name = d
                .block_name(d.block(x + offset[0], y + offset[1], z + offset[2]))
                .to_string();
            names.iter().any(|n| n == &name)
        }
        Predicate::Tag { tag, offset } => {
            let name = d
                .block_name(d.block(x + offset[0], y + offset[1], z + offset[2]))
                .to_string();
            d.tag_contains(tag, &name)
        }
        Predicate::Fluids { names, offset } => {
            let name = d
                .block_name(d.block(x + offset[0], y + offset[1], z + offset[2]))
                .to_string();
            let water = name == "minecraft:water";
            let lava = name == "minecraft:lava";
            names.iter().any(|n| match n.as_str() {
                "minecraft:water" | "minecraft:flowing_water" => water,
                "minecraft:lava" | "minecraft:flowing_lava" => lava,
                _ => false,
            })
        }
        Predicate::Survive { name } => match d.state_id_of(name, "") {
            Some(state) => can_survive(d, state, x, y, z),
            None => false,
        },
        // Both vertical face directions collapse onto the motion tag.
        Predicate::SturdyFace { offset } => {
            let name = d
                .block_name(d.block(x + offset[0], y + offset[1], z + offset[2]))
                .to_string();
            d.tag_contains("blocks_motion_no_leaves", &name)
        }
        Predicate::InsideBounds { offset } => {
            let y = y + offset[1];
            (MIN_Y..=MIN_Y + LAYERS as i32 - 1).contains(&y)
        }
        Predicate::Replaceable => {
            let name = d.block_name(d.block(x, y, z)).to_string();
            d.tag_contains("replaceable", &name)
        }
        Predicate::Solid { offset } => {
            let name = d.block_name(d.block(x + offset[0], y + offset[1], z + offset[2]));
            !matches!(
                name,
                "minecraft:air"
                    | "minecraft:cave_air"
                    | "minecraft:void_air"
                    | "minecraft:water"
                    | "minecraft:lava"
            )
        }
        Predicate::AllOf(list) => list.iter().all(|p| test_predicate(d, p, x, y, z)),
        Predicate::AnyOf(list) => list.iter().any(|p| test_predicate(d, p, x, y, z)),
        Predicate::Not(inner) => !test_predicate(d, inner, x, y, z),
        Predicate::True => true,
        Predicate::Unsupported => false,
    }
}

/// Blocks that survive on the vegetation substrate.
const VEGETATION: [&str; 26] = [
    "minecraft:short_grass",
    "minecraft:fern",
    "minecraft:tall_grass",
    "minecraft:large_fern",
    "minecraft:bush",
    "minecraft:firefly_bush",
    "minecraft:poppy",
    "minecraft:dandelion",
    "minecraft:lily_of_the_valley",
    "minecraft:azure_bluet",
    "minecraft:allium",
    "minecraft:blue_orchid",
    "minecraft:red_tulip",
    "minecraft:orange_tulip",
    "minecraft:white_tulip",
    "minecraft:pink_tulip",
    "minecraft:oxeye_daisy",
    "minecraft:cornflower",
    "minecraft:torchflower",
    "minecraft:sunflower",
    "minecraft:lilac",
    "minecraft:rose_bush",
    "minecraft:peony",
    "minecraft:oak_sapling",
    "minecraft:birch_sapling",
    "minecraft:dark_oak_sapling",
];

/// Blocks that survive on the dry substrate.
const DRY_VEGETATION: [&str; 2] = ["minecraft:short_dry_grass", "minecraft:tall_dry_grass"];

/// The double-height plants simple_block writes in two halves.
const DOUBLE_PLANTS: [&str; 8] = [
    "minecraft:sunflower",
    "minecraft:lilac",
    "minecraft:rose_bush",
    "minecraft:peony",
    "minecraft:tall_grass",
    "minecraft:large_fern",
    "minecraft:tall_seagrass",
    "minecraft:small_dripleaf",
];

/// Whether a state's block survives at the position: the rules the
/// spawn set exercises. Sturdy-face checks approximate with the motion
/// tag (full opaque cubes), and light reads as worldgen-dark.
fn can_survive(d: &mut Decorator, state: u32, x: i32, y: i32, z: i32) -> bool {
    let name = d.block_name(state).to_string();
    let below = d.block_name(d.block(x, y - 1, z)).to_string();
    match name.as_str() {
        "minecraft:sugar_cane" => {
            if below == "minecraft:sugar_cane" {
                return true;
            }
            if !d.tag_contains("supports_sugar_cane", &below) {
                return false;
            }
            HORIZONTAL.iter().any(|face| {
                let side = d
                    .block_name(d.block(x + face.dx, y - 1, z + face.dz))
                    .to_string();
                side == "minecraft:water" || d.tag_contains("supports_sugar_cane_adjacently", &side)
            })
        }
        "minecraft:leaf_litter" => d.tag_contains("blocks_motion_no_leaves", &below),
        // The blossom hangs: a full center face above and dry air at the
        // cell itself.
        "minecraft:spore_blossom" => {
            let above = d.block_name(d.block(x, y + 1, z)).to_string();
            d.tag_contains("blocks_motion_no_leaves", &above)
                && d.block_name(d.block(x, y, z)) != "minecraft:water"
        }
        // Azaleas root on the azalea substrate; a carpet needs any block
        // below, and the small dripleaf stands on its own substrate or in
        // shallow water over vegetation ground.
        "minecraft:azalea" | "minecraft:flowering_azalea" => {
            d.tag_contains("supports_azalea", &below)
        }
        "minecraft:moss_carpet" => !crate::lush::is_air_name(&below),
        "minecraft:small_dripleaf" => {
            if d.tag_contains("supports_small_dripleaf", &below) {
                true
            } else {
                d.block_name(d.block(x, y, z)) == "minecraft:water"
                    && d.tag_contains("supports_vegetation", &below)
            }
        }
        "minecraft:brown_mushroom" | "minecraft:red_mushroom" => {
            d.tag_contains("overrides_mushroom_light_requirement", &below)
                || d.tag_contains("blocks_motion_no_leaves", &below)
        }
        "minecraft:seagrass" => d.tag_contains("blocks_motion_no_leaves", &below),
        "minecraft:tall_seagrass" => {
            let here = d.block_name(d.block(x, y, z)).to_string();
            d.tag_contains("supports_vegetation", &below) && here == "minecraft:water"
        }
        other => {
            if VEGETATION.contains(&other) {
                d.tag_contains("supports_vegetation", &below)
            } else if DRY_VEGETATION.contains(&other) {
                d.tag_contains("supports_dry_vegetation", &below)
            } else {
                true
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Simple block.
// ---------------------------------------------------------------------------

fn run_simple_block(
    d: &mut Decorator,
    v: &Value,
    rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
) -> bool {
    let Some(to_place) = v.get("to_place") else {
        return false;
    };
    let provider = StateProvider::parse(d, to_place);
    if matches!(provider, StateProvider::None) {
        return false;
    }
    let Some(state) = provider.sample(d, rng, x, y, z) else {
        return false;
    };
    if !can_survive(d, state, x, y, z) {
        return false;
    }
    let name = d.block_name(state).to_string();
    if DOUBLE_PLANTS.contains(&name.as_str()) {
        // The upper half needs air, or a replaceable block holding the
        // same fluid as the plant (tall seagrass carries water).
        let above = d.block_name(d.block(x, y + 1, z)).to_string();
        if above != "minecraft:air" {
            let plant_fluid = if name == "minecraft:tall_seagrass" {
                "water"
            } else {
                let (_, props) = d.registry().state_of(state).unwrap_or(("", ""));
                if props.contains("waterlogged=true") {
                    "water"
                } else {
                    ""
                }
            };
            let above_fluid = match above.as_str() {
                "minecraft:water" => "water",
                "minecraft:lava" => "lava",
                _ => "",
            };
            let replaceable = d.tag_contains("replaceable", &above);
            if !(replaceable && plant_fluid == above_fluid) {
                return false;
            }
        }
        let (_, props) = d.registry().state_of(state).unwrap_or(("", ""));
        let lower_props = if props.contains("waterlogged") {
            let wet = d.block_name(d.block(x, y, z)) == "minecraft:water";
            BlockRegistry::with_prop(props, "waterlogged", if wet { "true" } else { "false" })
        } else {
            props.to_string()
        };
        let upper_props = BlockRegistry::with_prop(&lower_props, "half", "upper");
        let lower_props = BlockRegistry::with_prop(&lower_props, "half", "lower");
        if let (Some(lower), Some(upper)) = (
            d.state_id_of(&name, &lower_props),
            d.state_id_of(&name, &upper_props),
        ) {
            d.set_block(x, y, z, lower);
            d.set_block(x, y + 1, z, upper);
        }
    } else {
        d.set_block(x, y, z, state);
    }
    true
}

// ---------------------------------------------------------------------------
// Trees.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum FoliageKind {
    Blob,
    Fancy,
    DarkOak,
    RandomSpread,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TrunkKind {
    Straight,
    DarkOak,
    Fancy,
    Bending,
}

#[derive(Clone)]
struct FoliageCfg {
    kind: FoliageKind,
    radius: IntDraw,
    offset: IntDraw,
    height: i32,
    /// The scattered canopy's height provider; the other kinds read the
    /// plain height field.
    spread_height: IntDraw,
    attempts: i32,
}

#[derive(Clone, Copy)]
enum SizeKind {
    Two {
        limit: i32,
        lower: i32,
        upper: i32,
    },
    Three {
        limit: i32,
        upper_limit: i32,
        lower: i32,
        middle: i32,
        upper: i32,
    },
}

#[derive(Clone)]
struct SizeCfg {
    kind: SizeKind,
    min_clipped: Option<i32>,
}

impl SizeCfg {
    fn size_at(&self, tree_height: i32, yo: i32) -> i32 {
        match self.kind {
            SizeKind::Two {
                limit,
                lower,
                upper,
            } => {
                if yo < limit {
                    lower
                } else {
                    upper
                }
            }
            SizeKind::Three {
                limit,
                upper_limit,
                lower,
                middle,
                upper,
            } => {
                if yo < limit {
                    lower
                } else if yo >= tree_height - upper_limit {
                    upper
                } else {
                    middle
                }
            }
        }
    }
}

#[derive(Clone)]
enum TreeDecoratorCfg {
    PlaceOnGround {
        tries: i32,
        radius: i32,
        height: i32,
        provider: StateProvider,
    },
    Beehive {
        probability: f32,
    },
}

pub(crate) struct TreeCfg {
    trunk: StateProvider,
    leaves: StateProvider,
    below_trunk: StateProvider,
    trunk_kind: TrunkKind,
    base: i32,
    rand_a: i32,
    rand_b: i32,
    bend_length: IntDraw,
    min_height_for_leaves: i32,
    foliage: FoliageCfg,
    size: SizeCfg,
    decorators: Vec<TreeDecoratorCfg>,
}

fn int_field(v: &Value, key: &str, default: i32) -> i32 {
    v.get(key)
        .and_then(Value::as_i64)
        .map_or(default, |n| n as i32)
}

fn draw_field(v: &Value, key: &str) -> Option<IntDraw> {
    IntDraw::parse(v.get(key)?).ok()
}

/// Parses a tree config; None marks a shape this engine does not run.
pub(crate) fn parse_tree(d: &mut Decorator, v: &Value) -> Option<TreeCfg> {
    let trunk_placer = v.get("trunk_placer")?;
    let trunk_kind = match trunk_placer.get("type").and_then(Value::as_str)? {
        "minecraft:straight_trunk_placer" => TrunkKind::Straight,
        "minecraft:dark_oak_trunk_placer" => TrunkKind::DarkOak,
        "minecraft:fancy_trunk_placer" => TrunkKind::Fancy,
        "minecraft:bending_trunk_placer" => TrunkKind::Bending,
        _ => return None,
    };
    let (bend_length, min_height_for_leaves) = match trunk_kind {
        TrunkKind::Bending => (
            draw_field(trunk_placer, "bend_length")?,
            int_field(trunk_placer, "min_height_for_leaves", 1),
        ),
        _ => (IntDraw::Constant(1), 0),
    };
    let foliage_json = v.get("foliage_placer")?;
    let kind = match foliage_json.get("type").and_then(Value::as_str)? {
        "minecraft:blob_foliage_placer" => FoliageKind::Blob,
        "minecraft:fancy_foliage_placer" => FoliageKind::Fancy,
        "minecraft:dark_oak_foliage_placer" => FoliageKind::DarkOak,
        "minecraft:random_spread_foliage_placer" => FoliageKind::RandomSpread,
        _ => return None,
    };
    let (spread_height, attempts) = match kind {
        FoliageKind::RandomSpread => (
            draw_field(foliage_json, "foliage_height")?,
            int_field(foliage_json, "leaf_placement_attempts", 0),
        ),
        _ => (IntDraw::Constant(0), 0),
    };
    let foliage = FoliageCfg {
        kind,
        radius: draw_field(foliage_json, "radius")?,
        offset: draw_field(foliage_json, "offset")?,
        height: int_field(foliage_json, "height", 0),
        spread_height,
        attempts,
    };
    let size_json = v.get("minimum_size")?;
    let min_clipped = size_json
        .get("min_clipped_height")
        .and_then(Value::as_i64)
        .map(|n| n as i32);
    let kind = match size_json.get("type").and_then(Value::as_str)? {
        "minecraft:two_layers_feature_size" => SizeKind::Two {
            limit: int_field(size_json, "limit", 1),
            lower: int_field(size_json, "lower_size", 0),
            upper: int_field(size_json, "upper_size", 1),
        },
        "minecraft:three_layers_feature_size" => SizeKind::Three {
            limit: int_field(size_json, "limit", 1),
            upper_limit: int_field(size_json, "upper_limit", 1),
            lower: int_field(size_json, "lower_size", 0),
            middle: int_field(size_json, "middle_size", 1),
            upper: int_field(size_json, "upper_size", 1),
        },
        _ => return None,
    };
    let mut decorators = Vec::new();
    for deco in v
        .get("decorators")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match deco.get("type").and_then(Value::as_str) {
            Some("minecraft:place_on_ground") => {
                let provider = StateProvider::parse(
                    d,
                    deco.get("block_state_provider").unwrap_or(&Value::Null),
                );
                if matches!(provider, StateProvider::None) {
                    return None;
                }
                decorators.push(TreeDecoratorCfg::PlaceOnGround {
                    tries: int_field(deco, "tries", 128),
                    radius: int_field(deco, "radius", 2),
                    height: int_field(deco, "height", 1),
                    provider,
                });
            }
            Some("minecraft:beehive") => {
                decorators.push(TreeDecoratorCfg::Beehive {
                    probability: deco
                        .get("probability")
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0) as f32,
                });
            }
            _ => return None,
        }
    }
    let trunk = StateProvider::parse(d, v.get("trunk_provider")?);
    let leaves = StateProvider::parse(d, v.get("foliage_provider")?);
    let below_trunk = StateProvider::parse(d, v.get("below_trunk_provider")?);
    if matches!(trunk, StateProvider::None)
        || matches!(leaves, StateProvider::None)
        || matches!(below_trunk, StateProvider::None)
    {
        return None;
    }
    Some(TreeCfg {
        trunk,
        leaves,
        below_trunk,
        trunk_kind,
        base: int_field(trunk_placer, "base_height", 0),
        rand_a: int_field(trunk_placer, "height_rand_a", 0),
        rand_b: int_field(trunk_placer, "height_rand_b", 0),
        bend_length,
        min_height_for_leaves,
        foliage,
        size: SizeCfg { kind, min_clipped },
        decorators,
    })
}

/// One foliage anchor: the attachment block and whether the trunk is a
/// double column.
struct Attachment {
    x: i32,
    y: i32,
    z: i32,
    double: bool,
}

fn valid_tree_pos(d: &mut Decorator, x: i32, y: i32, z: i32) -> bool {
    let state = d.block(x, y, z);
    if d.block_name(state) == "minecraft:air" {
        return true;
    }
    let name = d.block_name(state).to_string();
    d.tag_contains("replaceable_by_trees", &name)
}

fn is_free(d: &mut Decorator, x: i32, y: i32, z: i32) -> bool {
    if valid_tree_pos(d, x, y, z) {
        return true;
    }
    let name = d.block_name(d.block(x, y, z)).to_string();
    d.tag_contains("logs", &name)
}

fn is_air_or_leaves(d: &mut Decorator, x: i32, y: i32, z: i32) -> bool {
    let state = d.block(x, y, z);
    if d.block_name(state) == "minecraft:air" {
        return true;
    }
    let name = d.block_name(state).to_string();
    d.tag_contains("leaves", &name)
}

/// Places one log if the position accepts it; records the write.
fn place_log(
    d: &mut Decorator,
    cfg: &TreeCfg,
    rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
    logs: &mut Vec<(i32, i32, i32)>,
) -> bool {
    if !valid_tree_pos(d, x, y, z) {
        return false;
    }
    if let Some(state) = cfg.trunk.sample(d, rng, x, y, z) {
        d.set_block(x, y, z, state);
        logs.push((x, y, z));
    }
    true
}

/// Places the soil block under a trunk cell when the provider yields one.
fn place_below_trunk(
    d: &mut Decorator,
    cfg: &TreeCfg,
    rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
    logs: &mut Vec<(i32, i32, i32)>,
) {
    if let Some(state) = cfg.below_trunk.sample(d, rng, x, y, z) {
        d.set_block(x, y, z, state);
        logs.push((x, y, z));
    }
}

fn run_tree(d: &mut Decorator, v: &Value, rng: &mut DecorRng, x: i32, y: i32, z: i32) -> bool {
    let Some(cfg) = parse_tree(d, v) else {
        return false;
    };
    let tree_height = cfg.base + rng.next_int(cfg.rand_a + 1) + rng.next_int(cfg.rand_b + 1);
    let foliage_height = match cfg.foliage.kind {
        FoliageKind::Blob | FoliageKind::Fancy => cfg.foliage.height,
        FoliageKind::DarkOak => 4,
        FoliageKind::RandomSpread => cfg.foliage.spread_height.sample(rng),
    };
    let leaf_radius = cfg.foliage.radius.sample(rng);
    #[cfg(test)]
    let stage_height = rng.words;
    if y < MIN_Y + 1 || y + tree_height + 1 > WORLD_TOP + 1 {
        return false;
    }
    let clipped = max_free_tree_height(d, &cfg.size, tree_height, x, y, z);
    if clipped < tree_height && cfg.size.min_clipped.is_none_or(|m| clipped < m) {
        return false;
    }
    let mut logs: Vec<(i32, i32, i32)> = Vec::new();
    let mut leaves: Vec<(i32, i32, i32)> = Vec::new();
    let attachments = match cfg.trunk_kind {
        TrunkKind::Straight => straight_trunk(d, &cfg, rng, x, y, z, clipped, &mut logs),
        TrunkKind::DarkOak => dark_oak_trunk(d, &cfg, rng, x, y, z, clipped, &mut logs),
        TrunkKind::Fancy => fancy_trunk(d, &cfg, rng, x, y, z, clipped, &mut logs),
        TrunkKind::Bending => bending_trunk(d, &cfg, rng, x, y, z, clipped, &mut logs),
    };
    #[cfg(test)]
    let stage_trunk = rng.words;
    for att in &attachments {
        create_foliage(d, &cfg, rng, att, foliage_height, leaf_radius, &mut leaves);
    }
    #[cfg(test)]
    if let Some(log) = d.scatter_log.as_mut() {
        log.push(format!(
            "tree ({x},{y},{z}) h {tree_height} words height {stage_height} trunk {stage_trunk} foliage {} logs {} leaves {}",
            rng.words,
            logs.len(),
            leaves.len()
        ));
    }
    // The decorator context sorts the position sets by height.
    logs.sort_unstable_by_key(|p| p.1);
    leaves.sort_unstable_by_key(|p| p.1);
    let mut decorations: Vec<(i32, i32, i32)> = Vec::new();
    for deco in &cfg.decorators {
        match *deco {
            TreeDecoratorCfg::PlaceOnGround {
                tries,
                radius,
                height,
                ref provider,
            } => place_on_ground(
                d,
                rng,
                tries,
                radius,
                height,
                provider,
                &logs,
                &mut decorations,
            ),
            TreeDecoratorCfg::Beehive { probability } => {
                beehive(d, rng, probability, &logs, &leaves, &mut decorations)
            }
        }
    }
    update_leaf_distances(d, &logs, &leaves, &decorations);
    !logs.is_empty() || !leaves.is_empty()
}

/// The position hash the leaf walk orders pending cells by: the axes fold
/// into one word.
fn leaf_walk_hash(p: &(i32, i32, i32)) -> i32 {
    (p.1.wrapping_add(p.2.wrapping_mul(31)))
        .wrapping_mul(31)
        .wrapping_add(p.0)
}

/// Trunk-family blocks count as distance zero in the leaf walk.
fn trunk_like(name: &str) -> bool {
    name.ends_with("_log")
        || name.ends_with("_wood")
        || name.ends_with("_stem")
        || name.ends_with("_hyphae")
}

/// The leaf-distance walk: pending cells sit in buckets by distance, the
/// trunk logs at zero, each step pops the first cell of the smallest
/// non-empty bucket in spread-hash table order (insertion order within a
/// table slot), and a popped cell takes its bucket as its distance
/// property. A cell can enter two buckets; the later bucket's write wins.
/// Expansion offers unwritten in-box neighbors carrying a distance
/// property the minimum of their current value and the bucket plus one;
/// trunk-family blocks count as distance zero. The walk stays inside the
/// box the tree's own placements enclose and rewrites distance only, so
/// the heightmaps stay untouched.
fn update_leaf_distances(
    d: &mut Decorator,
    logs: &[(i32, i32, i32)],
    leaves: &[(i32, i32, i32)],
    decorations: &[(i32, i32, i32)],
) {
    if logs.is_empty() && leaves.is_empty() {
        return;
    }
    // The walk never leaves the box enclosing everything the tree placed.
    let (mut bx0, mut bx1, mut by0, mut by1, mut bz0, mut bz1) =
        (i32::MAX, i32::MIN, i32::MAX, i32::MIN, i32::MAX, i32::MIN);
    for &(x, y, z) in logs.iter().chain(leaves).chain(decorations) {
        bx0 = bx0.min(x);
        bx1 = bx1.max(x);
        by0 = by0.min(y);
        by1 = by1.max(y);
        bz0 = bz0.min(z);
        bz1 = bz1.max(z);
    }
    // A pending cell plus its insertion sequence.
    type WalkCell = ((i32, i32, i32), usize);
    let mut buckets: Vec<Vec<WalkCell>> = vec![Vec::new(); 7];
    // The table grows past load thresholds and never shrinks; a table slot
    // holding eight cells under capacity sixty-four grows the table too.
    let mut caps = [16u32; 7];
    let mut slots: Vec<HashMap<u32, usize>> = vec![HashMap::new(); 7];
    let slot_of = |caps: &u32, at: &(i32, i32, i32)| -> u32 {
        let h = leaf_walk_hash(at) as u32;
        (h ^ (h >> 16)) & (caps - 1)
    };
    let mut seq = 0usize;
    for &at in logs {
        buckets[0].push((at, seq));
        seq += 1;
    }
    while buckets[0].len() > (caps[0] as usize * 3) / 4 {
        caps[0] *= 2;
    }
    slots[0] = buckets[0].iter().fold(HashMap::new(), |mut m, &(at, _)| {
        *m.entry(slot_of(&caps[0], &at)).or_default() += 1;
        m
    });
    let mut seen: std::collections::HashSet<(i32, i32, i32)> = std::collections::HashSet::new();
    let mut smallest = 0usize;
    loop {
        while smallest < 7 && buckets[smallest].is_empty() {
            smallest += 1;
        }
        if smallest >= 7 {
            return;
        }
        let mask = caps[smallest] - 1;
        let pick = buckets[smallest]
            .iter()
            .enumerate()
            .min_by_key(|&(k, &(at, s))| {
                let h = leaf_walk_hash(&at) as u32;
                ((h ^ (h >> 16)) & mask, s, k)
            })
            .map(|(k, _)| k)
            .expect("bucket nonempty above");
        let (at @ (px, py, pz), _) = buckets[smallest][pick];
        buckets[smallest].swap_remove(pick);
        if smallest != 0 {
            let state = d.block(px, py, pz);
            let Some((name, props)) = d
                .registry()
                .state_of(state)
                .map(|(n, p)| (n.to_string(), p.to_string()))
            else {
                seen.insert(at);
                continue;
            };
            if BlockRegistry::prop_int(&props, "distance").is_some() {
                let walked = BlockRegistry::with_prop(&props, "distance", &smallest.to_string());
                if let Some(id) = d.state_id_of(&name, &walked) {
                    d.set_block(px, py, pz, id);
                }
            }
        }
        seen.insert(at);
        for (dx, dy, dz) in [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ] {
            let next = (px + dx, py + dy, pz + dz);
            if seen.contains(&next) {
                continue;
            }
            if next.0 < bx0
                || next.0 > bx1
                || next.1 < by0
                || next.1 > by1
                || next.2 < bz0
                || next.2 > bz1
            {
                continue;
            }
            let state = d.block(next.0, next.1, next.2);
            let name = d.block_name(state).to_string();
            let current = if trunk_like(&name) {
                0
            } else {
                match d
                    .registry()
                    .state_of(state)
                    .and_then(|(_, p)| BlockRegistry::prop_int(p, "distance"))
                {
                    Some(v) => v,
                    None => continue,
                }
            };
            let new_distance = current.min(smallest as i32 + 1);
            if new_distance < 7 {
                let bucket = &mut buckets[new_distance as usize];
                if !bucket.iter().any(|&(b, _)| b == next) {
                    bucket.push((next, seq));
                    seq += 1;
                    let idx = new_distance as usize;
                    let slot = slot_of(&caps[idx], &next);
                    let filled = {
                        *slots[idx].entry(slot).or_default() += 1;
                        slots[idx][&slot]
                    };
                    let mut grow = bucket.len() > (caps[idx] as usize * 3) / 4
                        || filled >= 8 && caps[idx] < 64;
                    while grow {
                        caps[idx] *= 2;
                        slots[idx] = bucket.iter().fold(HashMap::new(), |mut m, &(at, _)| {
                            *m.entry(slot_of(&caps[idx], &at)).or_default() += 1;
                            m
                        });
                        grow = bucket.len() > (caps[idx] as usize * 3) / 4;
                    }
                    smallest = smallest.min(idx);
                }
            }
        }
    }
}

/// The height the tree reaches before clipping against unreplaceable
/// blocks, per the minimum-size radius profile.
fn max_free_tree_height(
    d: &mut Decorator,
    size: &SizeCfg,
    max_tree_height: i32,
    x: i32,
    y: i32,
    z: i32,
) -> i32 {
    for yo in 0..=max_tree_height + 1 {
        let r = size.size_at(max_tree_height, yo);
        for dx in -r..=r {
            for dz in -r..=r {
                if !is_free(d, x + dx, y + yo, z + dz) {
                    return yo - 2;
                }
            }
        }
    }
    max_tree_height
}

/// The bending trunk: a vertical column that crooks one or two cells
/// sideways near the top, then runs one or two further logs out along
/// the same side, anchoring foliage from the leaf height upward.
#[allow(clippy::too_many_arguments)]
fn bending_trunk(
    d: &mut Decorator,
    cfg: &TreeCfg,
    rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
    tree_height: i32,
    logs: &mut Vec<(i32, i32, i32)>,
) -> Vec<Attachment> {
    let direction = HORIZONTAL[rng.next_int(4) as usize];
    let log_height = tree_height - 1;
    place_below_trunk(d, cfg, rng, x, y - 1, z, logs);
    let mut attachments = Vec::new();
    let (mut px, mut py, mut pz) = (x, y, z);
    for i in 0..=log_height {
        // The crook draw runs every level; once it fires the column
        // keeps drifting sideways.
        if i + 1 >= log_height + rng.next_int(2) {
            px += direction.dx;
            pz += direction.dz;
        }
        place_log(d, cfg, rng, px, py, pz, logs);
        if i >= cfg.min_height_for_leaves {
            attachments.push(Attachment {
                x: px,
                y: py,
                z: pz,
                double: false,
            });
        }
        py += 1;
    }
    let dir_length = cfg.bend_length.sample(rng);
    for _ in 0..=dir_length {
        place_log(d, cfg, rng, px, py, pz, logs);
        attachments.push(Attachment {
            x: px,
            y: py,
            z: pz,
            double: false,
        });
        px += direction.dx;
        pz += direction.dz;
    }
    attachments
}

#[allow(clippy::too_many_arguments)]
fn straight_trunk(
    d: &mut Decorator,
    cfg: &TreeCfg,
    rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
    tree_height: i32,
    logs: &mut Vec<(i32, i32, i32)>,
) -> Vec<Attachment> {
    place_below_trunk(d, cfg, rng, x, y - 1, z, logs);
    for dy in 0..tree_height {
        place_log(d, cfg, rng, x, y + dy, z, logs);
    }
    vec![Attachment {
        x,
        y: y + tree_height,
        z,
        double: false,
    }]
}

#[allow(clippy::too_many_arguments)]
fn dark_oak_trunk(
    d: &mut Decorator,
    cfg: &TreeCfg,
    rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
    tree_height: i32,
    logs: &mut Vec<(i32, i32, i32)>,
) -> Vec<Attachment> {
    let mut attachments = Vec::new();
    place_below_trunk(d, cfg, rng, x, y - 1, z, logs);
    place_below_trunk(d, cfg, rng, x + 1, y - 1, z, logs);
    place_below_trunk(d, cfg, rng, x, y - 1, z + 1, logs);
    place_below_trunk(d, cfg, rng, x + 1, y - 1, z + 1, logs);
    let lean = HORIZONTAL[rng.next_int(4) as usize];
    let lean_height = tree_height - rng.next_int(4);
    let mut lean_steps = 2 - rng.next_int(3);
    let (mut tx, mut tz) = (x, z);
    let ey = y + tree_height - 1;
    for dy in 0..tree_height {
        if dy >= lean_height && lean_steps > 0 {
            tx += lean.dx;
            tz += lean.dz;
            lean_steps -= 1;
        }
        if !is_air_or_leaves(d, tx, y + dy, tz) {
            continue;
        }
        place_log(d, cfg, rng, tx, y + dy, tz, logs);
        place_log(d, cfg, rng, tx + 1, y + dy, tz, logs);
        place_log(d, cfg, rng, tx, y + dy, tz + 1, logs);
        place_log(d, cfg, rng, tx + 1, y + dy, tz + 1, logs);
    }
    attachments.push(Attachment {
        x: tx,
        y: ey,
        z: tz,
        double: true,
    });
    for ox in -1..=2 {
        for oz in -1..=2 {
            if (0..=1).contains(&ox) && (0..=1).contains(&oz) {
                continue;
            }
            if rng.next_int(3) > 0 {
                continue;
            }
            let length = rng.next_int(3) + 2;
            for branch_y in 0..length {
                place_log(d, cfg, rng, x + ox, ey - branch_y - 1, z + oz, logs);
            }
            attachments.push(Attachment {
                x: x + ox,
                y: ey,
                z: z + oz,
                double: false,
            });
        }
    }
    attachments
}

/// The float floor that keeps negative halves away from zero.
fn floor_f32(v: f32) -> i32 {
    v.floor() as i32
}

fn floor_f64(v: f64) -> i32 {
    v.floor() as i32
}

/// The canopy profile: negative below the crown, zero at the poles.
fn tree_shape(height: i32, y: i32) -> f32 {
    if (y as f32) < (height as f32) * 0.3 {
        return -1.0;
    }
    let radius = (height as f32) / 2.0;
    let adjacent = radius - (y as f32);
    let inner = radius * radius - adjacent * adjacent;
    let mut distance = (inner as f64).sqrt() as f32;
    if adjacent == 0.0 {
        distance = radius;
    } else if adjacent.abs() >= radius {
        return 0.0;
    }
    distance * 0.5
}

/// Walks a straight limb between two points; place=false only checks.
fn make_limb(
    d: &mut Decorator,
    cfg: &TreeCfg,
    rng: &mut DecorRng,
    start: (i32, i32, i32),
    end: (i32, i32, i32),
    place: bool,
    logs: &mut Vec<(i32, i32, i32)>,
) -> bool {
    if !place && start == end {
        return true;
    }
    let delta = (end.0 - start.0, end.1 - start.1, end.2 - start.2);
    let steps = delta.0.abs().max(delta.1.abs()).max(delta.2.abs());
    let dx = delta.0 as f32 / steps as f32;
    let dy = delta.1 as f32 / steps as f32;
    let dz = delta.2 as f32 / steps as f32;
    for i in 0..=steps {
        let pos = (
            start.0 + floor_f32(0.5 + i as f32 * dx),
            start.1 + floor_f32(0.5 + i as f32 * dy),
            start.2 + floor_f32(0.5 + i as f32 * dz),
        );
        if place {
            // Pillar blocks record their axis from the limb direction.
            let xdiff = (pos.0 - start.0).abs();
            let zdiff = (pos.2 - start.2).abs();
            let max = xdiff.max(zdiff);
            let axis = if max > 0 {
                if xdiff == max {
                    "x"
                } else {
                    "z"
                }
            } else {
                "y"
            };
            if !valid_tree_pos(d, pos.0, pos.1, pos.2) {
                continue;
            }
            if let Some(state) = cfg.trunk.sample(d, rng, pos.0, pos.1, pos.2) {
                let (name, props) = d.registry().state_of(state).unwrap_or(("", ""));
                let state = if props.contains("axis") {
                    d.state_id_of(name, &BlockRegistry::with_prop(props, "axis", axis))
                        .unwrap_or(state)
                } else {
                    state
                };
                d.set_block(pos.0, pos.1, pos.2, state);
                logs.push(pos);
            }
        } else if !is_free(d, pos.0, pos.1, pos.2) {
            return false;
        }
    }
    true
}

fn trim_branches(height: i32, local_y: i32) -> bool {
    f64::from(local_y) >= f64::from(height) * 0.2
}

#[allow(clippy::too_many_arguments)]
fn fancy_trunk(
    d: &mut Decorator,
    cfg: &TreeCfg,
    rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
    tree_height: i32,
    logs: &mut Vec<(i32, i32, i32)>,
) -> Vec<Attachment> {
    let height = tree_height + 2;
    let trunk_height = floor_f64(f64::from(height) * 0.618);
    place_below_trunk(d, cfg, rng, x, y - 1, z, logs);
    let clusters_per_y = 1.min(floor_f64(1.382 + (f64::from(height) / 13.0).powi(2)));
    let trunk_top = y + trunk_height;
    let mut coords: Vec<((i32, i32, i32), i32)> = Vec::new();
    let top_rel = height - 5;
    coords.push(((x, y + top_rel, z), trunk_top));
    let mut scratch: Vec<(i32, i32, i32)> = Vec::new();
    for relative_y in (0..=top_rel).rev() {
        let shape = tree_shape(height, relative_y);
        if shape < 0.0 {
            continue;
        }
        for _ in 0..clusters_per_y {
            let radius = f64::from(shape) * (f64::from(rng.next_f32()) + 0.328);
            let angle = f64::from(rng.next_f32() * 2.0) * PI;
            let fx = radius * angle.sin() + 0.5;
            let fz = radius * angle.cos() + 0.5;
            let start = (x + floor_f64(fx), y + relative_y - 1, z + floor_f64(fz));
            if !make_limb(
                d,
                cfg,
                rng,
                start,
                (start.0, start.1 + 5, start.2),
                false,
                &mut scratch,
            ) {
                continue;
            }
            let dx = x - start.0;
            let dz = z - start.2;
            let branch_height = f64::from(start.1) - f64::from(dx * dx + dz * dz).sqrt() * 0.381;
            let branch_top = if branch_height > f64::from(trunk_top) {
                trunk_top
            } else {
                branch_height as i32
            };
            let base = (x, branch_top, z);
            if !make_limb(d, cfg, rng, base, start, false, &mut scratch) {
                continue;
            }
            coords.push((start, branch_top));
        }
    }
    make_limb(d, cfg, rng, (x, y, z), (x, y + trunk_height, z), true, logs);
    for &(pos, branch_base) in &coords {
        let base = (x, branch_base, z);
        if base == pos || !trim_branches(height, branch_base - y) {
            continue;
        }
        make_limb(d, cfg, rng, base, pos, true, logs);
    }
    coords
        .iter()
        .filter(|&&(_, branch_base)| trim_branches(height, branch_base - y))
        .map(|&(pos, _)| Attachment {
            x: pos.0,
            y: pos.1,
            z: pos.2,
            double: false,
        })
        .collect()
}

/// Whether a row cell drops out: corner dither for blobs, a circular
/// mask for fancy canopies, corner cuts for the dark oak.
fn should_skip(
    rng: &mut DecorRng,
    kind: FoliageKind,
    dx: i32,
    dz: i32,
    y: i32,
    radius: i32,
    double: bool,
) -> bool {
    if kind == FoliageKind::DarkOak && y == 0 && double {
        let open_x = dx != -radius && dx < radius;
        let open_z = dz != -radius && dz < radius;
        if !(open_x || open_z) {
            return true;
        }
    }
    let (mdx, mdz) = if double {
        (dx.abs().min((dx - 1).abs()), dz.abs().min((dz - 1).abs()))
    } else {
        (dx.abs(), dz.abs())
    };
    match kind {
        FoliageKind::Blob => mdx == radius && mdz == radius && (rng.next_int(2) == 0 || y == 0),
        FoliageKind::Fancy => {
            let fx = (mdx as f32) + 0.5;
            let fz = (mdz as f32) + 0.5;
            fx * fx + fz * fz > (radius * radius) as f32
        }
        // The scattered canopy places per attempt, never in rows.
        FoliageKind::RandomSpread => false,
        FoliageKind::DarkOak => {
            if y == -1 && !double {
                return mdx == radius && mdz == radius;
            }
            if y == 1 {
                return mdx + mdz > radius * 2 - 2;
            }
            false
        }
    }
}

/// Places one square ring of foliage around an anchor.
#[allow(clippy::too_many_arguments)]
fn place_leaves_row(
    d: &mut Decorator,
    cfg: &TreeCfg,
    rng: &mut DecorRng,
    ax: i32,
    ay: i32,
    az: i32,
    radius: i32,
    y: i32,
    double: bool,
    kind: FoliageKind,
    leaves: &mut Vec<(i32, i32, i32)>,
) {
    let ext = i32::from(double);
    for dx in -radius..=radius + ext {
        for dz in -radius..=radius + ext {
            if should_skip(rng, kind, dx, dz, y, radius, double) {
                continue;
            }
            let (lx, ly, lz) = (ax + dx, ay + y, az + dz);
            if !valid_tree_pos(d, lx, ly, lz) {
                continue;
            }
            if let Some(state) = cfg.leaves.sample(d, rng, lx, ly, lz) {
                let state = waterlogged_leaf(d, state, lx, ly, lz);
                d.set_block(lx, ly, lz, state);
                leaves.push((lx, ly, lz));
            }
        }
    }
}

/// Places one scattered leaf: a persistent leaf already at the cell
/// refuses the write, the cell must accept tree placement, and the
/// provider draws only then.
fn try_place_leaf(
    d: &mut Decorator,
    cfg: &TreeCfg,
    rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
    leaves: &mut Vec<(i32, i32, i32)>,
) {
    let persistent = d
        .registry()
        .state_of(d.block(x, y, z))
        .is_some_and(|(_, props)| props.split(',').any(|pair| pair == "persistent=true"));
    if persistent || !valid_tree_pos(d, x, y, z) {
        return;
    }
    if let Some(state) = cfg.leaves.sample(d, rng, x, y, z) {
        let state = waterlogged_leaf(d, state, x, y, z);
        d.set_block(x, y, z, state);
        leaves.push((x, y, z));
    }
}

/// A leaf state inside water logs itself.
fn waterlogged_leaf(d: &mut Decorator, state: u32, x: i32, y: i32, z: i32) -> u32 {
    let (name, props) = d.registry().state_of(state).unwrap_or(("", ""));
    if !props.contains("waterlogged") {
        return state;
    }
    if d.block_name(d.block(x, y, z)) != "minecraft:water" {
        return state;
    }
    d.state_id_of(
        name,
        &BlockRegistry::with_prop(props, "waterlogged", "true"),
    )
    .unwrap_or(state)
}

#[allow(clippy::too_many_arguments)]
fn create_foliage(
    d: &mut Decorator,
    cfg: &TreeCfg,
    rng: &mut DecorRng,
    att: &Attachment,
    foliage_height: i32,
    leaf_radius: i32,
    leaves: &mut Vec<(i32, i32, i32)>,
) {
    let offset = cfg.foliage.offset.sample(rng);
    let kind = cfg.foliage.kind;
    match kind {
        // The scattered canopy: each attempt draws its own offset, three
        // doubled draws around the anchor.
        FoliageKind::RandomSpread => {
            for _ in 0..cfg.foliage.attempts {
                let dx = rng.next_int(leaf_radius) - rng.next_int(leaf_radius);
                let dy = rng.next_int(foliage_height) - rng.next_int(foliage_height);
                let dz = rng.next_int(leaf_radius) - rng.next_int(leaf_radius);
                try_place_leaf(d, cfg, rng, att.x + dx, att.y + dy, att.z + dz, leaves);
            }
        }
        FoliageKind::Blob => {
            for yo in ((offset - foliage_height)..=offset).rev() {
                let r = (leaf_radius - 1 - yo / 2).max(0);
                place_leaves_row(
                    d, cfg, rng, att.x, att.y, att.z, r, yo, att.double, kind, leaves,
                );
            }
        }
        FoliageKind::Fancy => {
            for yo in ((offset - foliage_height)..=offset).rev() {
                let r = leaf_radius + i32::from(yo != offset && yo != offset - foliage_height);
                place_leaves_row(
                    d, cfg, rng, att.x, att.y, att.z, r, yo, att.double, kind, leaves,
                );
            }
        }
        FoliageKind::DarkOak => {
            let py = att.y + offset;
            let (ax, az) = (att.x, att.z);
            if att.double {
                place_leaves_row(
                    d,
                    cfg,
                    rng,
                    ax,
                    py,
                    az,
                    leaf_radius + 2,
                    -1,
                    true,
                    kind,
                    leaves,
                );
                place_leaves_row(
                    d,
                    cfg,
                    rng,
                    ax,
                    py,
                    az,
                    leaf_radius + 3,
                    0,
                    true,
                    kind,
                    leaves,
                );
                place_leaves_row(
                    d,
                    cfg,
                    rng,
                    ax,
                    py,
                    az,
                    leaf_radius + 2,
                    1,
                    true,
                    kind,
                    leaves,
                );
                if rng.next_bool() {
                    place_leaves_row(d, cfg, rng, ax, py, az, leaf_radius, 2, true, kind, leaves);
                }
            } else {
                place_leaves_row(
                    d,
                    cfg,
                    rng,
                    ax,
                    py,
                    az,
                    leaf_radius + 2,
                    -1,
                    false,
                    kind,
                    leaves,
                );
                place_leaves_row(
                    d,
                    cfg,
                    rng,
                    ax,
                    py,
                    az,
                    leaf_radius + 1,
                    0,
                    false,
                    kind,
                    leaves,
                );
            }
        }
    }
}

/// Scatters ground cover around the lowest trunk row.
#[allow(clippy::too_many_arguments)]
fn place_on_ground(
    d: &mut Decorator,
    rng: &mut DecorRng,
    tries: i32,
    radius: i32,
    height: i32,
    provider: &StateProvider,
    logs: &[(i32, i32, i32)],
    decorations: &mut Vec<(i32, i32, i32)>,
) {
    if logs.is_empty() {
        return;
    }
    let min_y = logs[0].1;
    let (mut x0, mut x1, mut z0, mut z1) = (i32::MAX, i32::MIN, i32::MAX, i32::MIN);
    for &(lx, ly, lz) in logs {
        if ly != min_y {
            continue;
        }
        x0 = x0.min(lx);
        x1 = x1.max(lx);
        z0 = z0.min(lz);
        z1 = z1.max(lz);
    }
    let (bx0, bx1) = (x0 - radius, x1 + radius);
    let (by0, by1) = (min_y - height, min_y + height);
    let (bz0, bz1) = (z0 - radius, z1 + radius);
    #[cfg(test)]
    if let Some(log) = d.scatter_log.as_mut() {
        log.push(format!(
            "pass {tries} {bx0} {bx1} {by0} {by1} {bz0} {bz1} w{}",
            rng.words
        ));
    }
    for _ in 0..tries {
        let px = bx0 + rng.next_int(bx1 - bx0 + 1);
        let py = by0 + rng.next_int(by1 - by0 + 1);
        let pz = bz0 + rng.next_int(bz1 - bz0 + 1);
        let above = d.block_name(d.block(px, py + 1, pz)).to_string();
        if above != "minecraft:air" && above != "minecraft:vine" {
            #[cfg(test)]
            if let Some(log) = d.scatter_log.as_mut() {
                log.push(format!("({px},{py},{pz}) above {above} skip"));
            }
            continue;
        }
        let ground = d.block_name(d.block(px, py, pz)).to_string();
        if !d.tag_contains("blocks_motion_no_leaves", &ground) {
            #[cfg(test)]
            if let Some(log) = d.scatter_log.as_mut() {
                log.push(format!("({px},{py},{pz}) above air ground {ground} skip"));
            }
            continue;
        }
        let h = d.height(HeightKind::MotionBlockingNoLeaves, px, pz);
        if h > py + 1 {
            #[cfg(test)]
            if let Some(log) = d.scatter_log.as_mut() {
                log.push(format!(
                    "({px},{py},{pz}) above air ground {ground} h {h} skip"
                ));
            }
            continue;
        }
        if let Some(state) = provider.sample(d, rng, px, py + 1, pz) {
            #[cfg(test)]
            let placed_name = d
                .registry()
                .state_of(state)
                .map_or(String::new(), |(n, _)| n.to_string());
            #[cfg(test)]
            if let Some(log) = d.scatter_log.as_mut() {
                log.push(format!(
                    "({px},{py},{pz}) above air ground {ground} h {h} place {placed_name}"
                ));
            }
            d.set_block(px, py + 1, pz, state);
            decorations.push((px, py + 1, pz));
        }
    }
}

/// The reference list shuffle: each index from the top draws its swap.
fn shuffle<T>(items: &mut [T], rng: &mut DecorRng) {
    for i in (2..=items.len()).rev() {
        let swap_to = rng.next_int(i as i32) as usize;
        items.swap(i - 1, swap_to);
    }
}

/// Rarely hangs a nest on the trunk; the resident draws stay even though
/// the residents themselves are not entities here.
fn beehive(
    d: &mut Decorator,
    rng: &mut DecorRng,
    probability: f32,
    logs: &[(i32, i32, i32)],
    leaves: &[(i32, i32, i32)],
    decorations: &mut Vec<(i32, i32, i32)>,
) {
    if logs.is_empty() {
        return;
    }
    if rng.next_f32() >= probability {
        return;
    }
    let hive_y = if !leaves.is_empty() {
        (leaves[0].1 - 1).max(logs[0].1 + 1)
    } else {
        (logs[0].1 + 1 + rng.next_int(3)).min(logs[logs.len() - 1].1)
    };
    let mut placements: Vec<((i32, i32, i32), Face)> = Vec::new();
    for &(lx, ly, lz) in logs {
        if ly != hive_y {
            continue;
        }
        for face in [EAST, SOUTH, WEST] {
            placements.push((face.step(lx, ly, lz), face));
        }
    }
    if placements.is_empty() {
        return;
    }
    shuffle(&mut placements, rng);
    for &((px, py, pz), face) in &placements {
        let here = d.block_name(d.block(px, py, pz)) == "minecraft:air";
        let (ox, oy, oz) = face.step(px, py, pz);
        let outward = d.block_name(d.block(ox, oy, oz)) == "minecraft:air";
        if here && outward {
            let facing = face.opposite().prop;
            if let Some(state) = state_with_prop(d, "minecraft:bee_nest", "facing", facing) {
                d.set_block(px, py, pz, state);
                decorations.push((px, py, pz));
            }
            let bees = 2 + rng.next_int(2);
            for _ in 0..bees {
                rng.next_int(599);
            }
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Fallen trees.
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum FallenDecorator {
    TrunkVine,
    AttachedToLogs {
        probability: f32,
        provider: StateProvider,
        directions: Vec<Face>,
    },
}

pub(crate) struct FallenCfg {
    trunk: StateProvider,
    log_length: IntDraw,
    stump_decorators: Vec<FallenDecorator>,
    log_decorators: Vec<FallenDecorator>,
}

pub(crate) fn parse_fallen(d: &mut Decorator, v: &Value) -> Option<FallenCfg> {
    let trunk = StateProvider::parse(d, v.get("trunk_provider")?);
    if matches!(trunk, StateProvider::None) {
        return None;
    }
    let log_length = draw_field(v, "log_length")?;
    let mut parse_list = |key: &str| -> Option<Vec<FallenDecorator>> {
        let mut out = Vec::new();
        for deco in v.get(key).and_then(Value::as_array).into_iter().flatten() {
            match deco.get("type").and_then(Value::as_str) {
                Some("minecraft:trunk_vine") => out.push(FallenDecorator::TrunkVine),
                Some("minecraft:attached_to_logs") => {
                    let provider =
                        StateProvider::parse(d, deco.get("block_provider").unwrap_or(&Value::Null));
                    if matches!(provider, StateProvider::None) {
                        return None;
                    }
                    let mut directions = Vec::new();
                    for name in deco
                        .get("directions")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                    {
                        directions.push(Face::parse(name.as_str()?)?);
                    }
                    if directions.is_empty() {
                        return None;
                    }
                    out.push(FallenDecorator::AttachedToLogs {
                        probability: deco
                            .get("probability")
                            .and_then(Value::as_f64)
                            .unwrap_or(0.0) as f32,
                        provider,
                        directions,
                    });
                }
                _ => return None,
            }
        }
        Some(out)
    };
    Some(FallenCfg {
        trunk,
        log_length,
        stump_decorators: parse_list("stump_decorators")?,
        log_decorators: parse_list("log_decorators")?,
    })
}

/// The log set a fallen-tree decorator sees, in placement order.
fn run_fallen_decorators(
    d: &mut Decorator,
    rng: &mut DecorRng,
    decorators: &[FallenDecorator],
    logs: &[(i32, i32, i32)],
) {
    for deco in decorators {
        match deco {
            FallenDecorator::TrunkVine => {
                for &(lx, ly, lz) in logs {
                    for (face, attach) in [
                        (WEST, "east"),
                        (EAST, "west"),
                        (NORTH, "south"),
                        (SOUTH, "north"),
                    ] {
                        if rng.next_int(3) > 0 {
                            let (vx, vy, vz) = face.step(lx, ly, lz);
                            if d.block_name(d.block(vx, vy, vz)) == "minecraft:air" {
                                if let Some(state) =
                                    state_with_prop(d, "minecraft:vine", attach, "true")
                                {
                                    d.set_block(vx, vy, vz, state);
                                }
                            }
                        }
                    }
                }
            }
            FallenDecorator::AttachedToLogs {
                probability,
                provider,
                directions,
            } => {
                let mut order: Vec<(i32, i32, i32)> = logs.to_vec();
                shuffle(&mut order, rng);
                for (lx, ly, lz) in order {
                    let face = directions[rng.next_int(directions.len() as i32) as usize];
                    let (px, py, pz) = face.step(lx, ly, lz);
                    if rng.next_f32() > *probability {
                        continue;
                    }
                    if d.block_name(d.block(px, py, pz)) != "minecraft:air" {
                        continue;
                    }
                    if let Some(state) = provider.sample(d, rng, px, py, pz) {
                        d.set_block(px, py, pz, state);
                    }
                }
            }
        }
    }
}

fn run_fallen_tree(d: &mut Decorator, v: &Value, rng: &mut DecorRng, x: i32, y: i32, z: i32) {
    let Some(cfg) = parse_fallen(d, v) else {
        return;
    };
    // The stump: one log, unconditionally.
    if let Some(state) = cfg.trunk.sample(d, rng, x, y, z) {
        d.set_block(x, y, z, state);
    }
    run_fallen_decorators(d, rng, &cfg.stump_decorators, &[(x, y, z)]);
    let direction = HORIZONTAL[rng.next_int(4) as usize];
    let log_length = cfg.log_length.sample(rng) - 2;
    // The log starts two or three cells out from the stump, then climbs
    // one and settles down onto the ground (up to six steps).
    let steps = 2 + rng.next_int(2);
    let (mut px, mut py, mut pz) = (x, y, z);
    for _ in 0..steps {
        let (nx, _, nz) = direction.step(px, py, pz);
        px = nx;
        pz = nz;
    }
    py += 1;
    // Find the ground: up one, then down until the cell can hold a log
    // over solid ground.
    py += 1;
    for _ in 0..6 {
        let sturdy = d.block_name(d.block(px, py - 1, pz)).to_string();
        if valid_tree_pos(d, px, py, pz) && d.tag_contains("blocks_motion_no_leaves", &sturdy) {
            break;
        }
        py -= 1;
    }
    // The whole log must fit over ground with at most two gap cells.
    let mut fits = true;
    let mut gap = 0;
    {
        let (mut cx, mut cz) = (px, pz);
        for _ in 0..log_length {
            if !valid_tree_pos(d, cx, py, cz) {
                fits = false;
                break;
            }
            let below = d.block_name(d.block(cx, py - 1, cz)).to_string();
            if !d.tag_contains("blocks_motion_no_leaves", &below) {
                gap += 1;
                if gap > 2 {
                    fits = false;
                    break;
                }
            } else {
                gap = 0;
            }
            cx += direction.dx;
            cz += direction.dz;
        }
    }
    if !fits {
        return;
    }
    let axis = direction.axis();
    let mut placed: Vec<(i32, i32, i32)> = Vec::new();
    let (mut cx, mut cz) = (px, pz);
    for _ in 0..log_length {
        if let Some(state) = cfg.trunk.sample(d, rng, cx, py, cz) {
            let (name, props) = d.registry().state_of(state).unwrap_or(("", ""));
            let state = if props.contains("axis") {
                d.state_id_of(name, &BlockRegistry::with_prop(props, "axis", axis))
                    .unwrap_or(state)
            } else {
                state
            };
            d.set_block(cx, py, cz, state);
        }
        placed.push((cx, py, cz));
        cx += direction.dx;
        cz += direction.dz;
    }
    run_fallen_decorators(d, rng, &cfg.log_decorators, &placed);
}

// ---------------------------------------------------------------------------
// Huge mushrooms.
// ---------------------------------------------------------------------------

pub(crate) struct MushroomCfg {
    cap: StateProvider,
    stem: StateProvider,
    radius: i32,
    can_place_on: Predicate,
    red: bool,
}

pub(crate) fn parse_mushroom(d: &mut Decorator, v: &Value, red: bool) -> Option<MushroomCfg> {
    let cap = StateProvider::parse(d, v.get("cap_provider")?);
    let stem = StateProvider::parse(d, v.get("stem_provider")?);
    if matches!(cap, StateProvider::None) || matches!(stem, StateProvider::None) {
        return None;
    }
    let can_place_on = Predicate::parse(v.get("can_place_on")?).ok()?;
    Some(MushroomCfg {
        cap,
        stem,
        radius: int_field(v, "foliage_radius", 2),
        can_place_on,
        red,
    })
}

/// Writes a mushroom block over air or replaceable ground cover.
fn place_mushroom_block(d: &mut Decorator, x: i32, y: i32, z: i32, state: u32) {
    let here = d.block_name(d.block(x, y, z)).to_string();
    if here == "minecraft:air" || d.tag_contains("replaceable_by_mushrooms", &here) {
        d.set_block(x, y, z, state);
    }
}

/// Applies one face property when the state carries it.
fn with_prop_if_present(d: &Decorator, state: u32, prop: &str, value: bool) -> u32 {
    let (name, props) = d.registry().state_of(state).unwrap_or(("", ""));
    if !props.contains(prop) {
        return state;
    }
    let text = if value { "true" } else { "false" };
    d.state_id_of(name, &BlockRegistry::with_prop(props, prop, text))
        .unwrap_or(state)
}

fn run_huge_mushroom(
    d: &mut Decorator,
    v: &Value,
    rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
    red: bool,
) {
    let Some(cfg) = parse_mushroom(d, v, red) else {
        return;
    };
    let tree_height = rng.next_int(3) + 4;
    let tree_height = if rng.next_int(12) == 0 {
        tree_height * 2
    } else {
        tree_height
    };
    if y < MIN_Y + 1 || y + tree_height + 1 > WORLD_TOP {
        return;
    }
    if !test_predicate(d, &cfg.can_place_on, x, y - 1, z) {
        return;
    }
    // The free-space box: the brown checks its cap radius above the short
    // trunk; the red checks only its column.
    for dy in 0..=tree_height {
        let r = if cfg.red || dy <= 3 { 0 } else { cfg.radius };
        for dx in -r..=r {
            for dz in -r..=r {
                if !is_air_or_leaves(d, x + dx, y + dy, z + dz) {
                    return;
                }
            }
        }
    }
    make_cap(d, &cfg, rng, x, y, z, tree_height);
    for dy in 0..tree_height {
        if let Some(state) = cfg.stem.sample(d, rng, x, y, z) {
            place_mushroom_block(d, x, y + dy, z, state);
        }
    }
}

fn make_cap(
    d: &mut Decorator,
    cfg: &MushroomCfg,
    rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
    tree_height: i32,
) {
    let r = cfg.radius;
    if cfg.red {
        for dy in tree_height - 3..=tree_height {
            let radius = if dy < tree_height { r } else { r - 1 };
            let center = r - 2;
            for dx in -radius..=radius {
                for dz in -radius..=radius {
                    let x_edge = dx == -radius || dx == radius;
                    let z_edge = dz == -radius || dz == radius;
                    if dy < tree_height && x_edge == z_edge {
                        continue;
                    }
                    let Some(state) = cfg.cap.sample(d, rng, x, y, z) else {
                        continue;
                    };
                    let state = with_prop_if_present(d, state, "up", dy >= tree_height - 1);
                    let state = with_prop_if_present(d, state, "west", dx < -center);
                    let state = with_prop_if_present(d, state, "east", dx > center);
                    let state = with_prop_if_present(d, state, "north", dz < -center);
                    let state = with_prop_if_present(d, state, "south", dz > center);
                    place_mushroom_block(d, x + dx, y + dy, z + dz, state);
                }
            }
        }
    } else {
        for dx in -r..=r {
            for dz in -r..=r {
                let min_x = dx == -r;
                let max_x = dx == r;
                let min_z = dz == -r;
                let max_z = dz == r;
                let x_edge = min_x || max_x;
                let z_edge = min_z || max_z;
                if x_edge && z_edge {
                    continue;
                }
                let west = min_x || (z_edge && dx == 1 - r);
                let east = max_x || (z_edge && dx == r - 1);
                let north = min_z || (x_edge && dz == 1 - r);
                let south = max_z || (x_edge && dz == r - 1);
                let Some(state) = cfg.cap.sample(d, rng, x, y, z) else {
                    continue;
                };
                let state = with_prop_if_present(d, state, "west", west);
                let state = with_prop_if_present(d, state, "east", east);
                let state = with_prop_if_present(d, state, "north", north);
                let state = with_prop_if_present(d, state, "south", south);
                place_mushroom_block(d, x + dx, y + tree_height, z + dz, state);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Block columns.
// ---------------------------------------------------------------------------

struct ColumnLayer {
    height: IntDraw,
    provider: StateProvider,
}

fn run_block_column(
    d: &mut Decorator,
    v: &Value,
    rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
) -> bool {
    let Some(layers_json) = v.get("layers").and_then(Value::as_array) else {
        return false;
    };
    let mut layers: Vec<ColumnLayer> = Vec::new();
    for layer in layers_json {
        let Some(height_json) = layer.get("height") else {
            return false;
        };
        let Ok(height) = IntDraw::parse(height_json) else {
            return false;
        };
        let provider = StateProvider::parse(d, layer.get("provider").unwrap_or(&Value::Null));
        if matches!(provider, StateProvider::None) {
            return false;
        }
        layers.push(ColumnLayer { height, provider });
    }
    if layers.is_empty() {
        return false;
    }
    let up = match v.get("direction").and_then(Value::as_str) {
        Some("up") => true,
        Some("down") => false,
        _ => return false,
    };
    let Some(allowed_json) = v.get("allowed_placement") else {
        return false;
    };
    let Ok(allowed) = Predicate::parse(allowed_json) else {
        return false;
    };
    let prioritize_tip = v
        .get("prioritize_tip")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut heights: Vec<i32> = layers.iter().map(|l| l.height.sample(rng)).collect();
    let total: i32 = heights.iter().sum();
    if total == 0 {
        return false;
    }
    // Walk the cells past the origin; the first refusal clips the column.
    for step in 0..total {
        let y_off = y + if up { step + 1 } else { -(step + 1) };
        if !test_predicate(d, &allowed, x, y_off, z) {
            truncate(&mut heights, total, step, prioritize_tip);
            break;
        }
    }
    let mut cy = y;
    for (layer, &count) in layers.iter().zip(&heights) {
        for _ in 0..count {
            if let Some(state) = layer.provider.sample(d, rng, x, cy, z) {
                d.set_block(x, cy, z, state);
            }
            cy += if up { 1 } else { -1 };
        }
    }
    true
}

/// Trims layer heights back to a new column height, from the base or
/// the tip depending on priority.
fn truncate(heights: &mut [i32], total: i32, new_height: i32, prioritize_tip: bool) {
    let mut amount = total - new_height;
    let direction = if prioritize_tip { 1 } else { -1 };
    let mut i = if prioritize_tip {
        0
    } else {
        heights.len() as i32 - 1
    };
    let end = if prioritize_tip {
        heights.len() as i32
    } else {
        -1
    };
    while i != end && amount > 0 {
        let remove = heights[i as usize].min(amount);
        heights[i as usize] -= remove;
        amount -= remove;
        i += direction;
    }
}

// ---------------------------------------------------------------------------
// Multiface growth (glow lichen).
// ---------------------------------------------------------------------------

struct MultifaceCfg {
    block: String,
    surfaces: Vec<String>,
    chance: f32,
}

/// Whether a props string carries a face flag as true.
fn has_face_props(props: &str, prop: &str) -> bool {
    props.split(',').any(|pair| {
        pair.split_once('=')
            .is_some_and(|(k, v)| k == prop && v == "true")
    })
}

/// Whether a state's block carries a face flag.
fn has_face(d: &Decorator, state: u32, prop: &str) -> bool {
    d.registry()
        .state_of(state)
        .is_some_and(|(_, props)| has_face_props(props, prop))
}

/// The neighbor a face points to offers a full face to attach against.
fn can_attach(d: &mut Decorator, x: i32, y: i32, z: i32, face: Face) -> bool {
    let (nx, ny, nz) = face.step(x, y, z);
    let name = d.block_name(d.block(nx, ny, nz)).to_string();
    d.tag_contains("blocks_motion_no_leaves", &name)
}

/// The state a patch takes at a position: an existing patch grows the
/// new face, bare ground takes the default (logged under water), and a
/// face the patch already carries refuses the write.
fn growth_state(d: &Decorator, cfg: &MultifaceCfg, existing: u32, face: Face) -> Option<u32> {
    let (name, props) = d.registry().state_of(existing)?;
    let base_props = if name == cfg.block {
        if has_face_props(props, face.prop) {
            return None;
        }
        props.to_string()
    } else {
        let default_state = d.state_id_of(&cfg.block, "")?;
        let (_, default_props) = d.registry().state_of(default_state)?;
        if name == "minecraft:water" {
            BlockRegistry::with_prop(default_props, "waterlogged", "true")
        } else {
            default_props.to_string()
        }
    };
    let props = BlockRegistry::with_prop(&base_props, face.prop, "true");
    d.state_id_of(&cfg.block, &props)
}

fn run_multiface(d: &mut Decorator, v: &Value, rng: &mut DecorRng, x: i32, y: i32, z: i32) {
    let Some(block) = v.get("block").and_then(Value::as_str) else {
        return;
    };
    let cfg = MultifaceCfg {
        block: block.to_string(),
        surfaces: match v.get("can_be_placed_on") {
            Some(Value::String(s)) => vec![s.clone()],
            Some(Value::Array(list)) => list
                .iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect(),
            _ => return,
        },
        chance: v
            .get("chance_of_spreading")
            .and_then(Value::as_f64)
            .unwrap_or(0.5) as f32,
    };
    if cfg.surfaces.is_empty() {
        return;
    }
    let ceiling = v
        .get("can_place_on_ceiling")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let floor = v
        .get("can_place_on_floor")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let wall = v
        .get("can_place_on_wall")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut valid: Vec<Face> = Vec::new();
    if ceiling {
        valid.push(UP);
    }
    if floor {
        valid.push(DOWN);
    }
    if wall {
        valid.extend_from_slice(&HORIZONTAL);
    }
    if valid.is_empty() {
        return;
    }
    let origin = d.block_name(d.block(x, y, z)).to_string();
    if origin != "minecraft:air" && origin != "minecraft:water" {
        return;
    }
    let mut search = valid.clone();
    shuffle(&mut search, rng);
    if place_growth(d, rng, &cfg, x, y, z, &search) {
        return;
    }
    for dir in &search {
        let mut placement: Vec<Face> = valid
            .iter()
            .copied()
            .filter(|f| *f != dir.opposite())
            .collect();
        shuffle(&mut placement, rng);
        // The search walk re-anchors at one step out every pass, so
        // however far the config's search range reaches, each pass
        // retries the same cell; a failed try draws nothing and changes
        // nothing, which one attempt reproduces.
        let (px, py, pz) = dir.step(x, y, z);
        let name = d.block_name(d.block(px, py, pz)).to_string();
        if name != "minecraft:air" && name != "minecraft:water" && name != cfg.block {
            continue;
        }
        if place_growth(d, rng, &cfg, px, py, pz, &placement) {
            return;
        }
    }
}

/// Tries one position against its face list: the first face with a
/// placeable surface neighbor writes the patch and rolls one spread; a
/// face the patch already carries ends the whole try.
fn place_growth(
    d: &mut Decorator,
    rng: &mut DecorRng,
    cfg: &MultifaceCfg,
    x: i32,
    y: i32,
    z: i32,
    dirs: &[Face],
) -> bool {
    for &dir in dirs {
        let (nx, ny, nz) = dir.step(x, y, z);
        let neighbor = d.block_name(d.block(nx, ny, nz)).to_string();
        if !cfg.surfaces.contains(&neighbor) {
            continue;
        }
        let existing = d.block(x, y, z);
        let Some(state) = growth_state(d, cfg, existing, dir) else {
            return false;
        };
        d.set_block(x, y, z, state);
        if rng.next_f32() < cfg.chance {
            spread(d, rng, cfg, x, y, z, dir);
        }
        return true;
    }
    false
}

/// Grows one extra cell off a fresh patch: one shuffle of all six faces
/// (five draws, kept whatever comes of them), then the first direction
/// whose target accepts a face wins.
fn spread(
    d: &mut Decorator,
    rng: &mut DecorRng,
    cfg: &MultifaceCfg,
    x: i32,
    y: i32,
    z: i32,
    from: Face,
) {
    let mut order = ALL;
    shuffle(&mut order, rng);
    for dir in order {
        if dir.axis_group() == from.axis_group() {
            continue;
        }
        if has_face(d, d.block(x, y, z), dir.prop) {
            continue;
        }
        // The three spread shapes, in the reference's order: the same
        // cell grows another face, the neighbor grows this face, or the
        // diagonal wraps around onto the far side.
        let (sx, sy, sz) = dir.step(x, y, z);
        let (wx, wy, wz) = from.step(x, y, z);
        for (cell, face) in [
            ((x, y, z), dir),
            ((sx, sy, sz), from),
            (dir.step(wx, wy, wz), dir.opposite()),
        ] {
            if try_spread_cell(d, cfg, cell, face) {
                return;
            }
        }
    }
}

/// Whether one spread target takes a face: the cell holds air, water,
/// or the patch block itself; the face is unclaimed; and the neighbor
/// along the face offers an attachment.
fn try_spread_cell(
    d: &mut Decorator,
    cfg: &MultifaceCfg,
    cell: (i32, i32, i32),
    face: Face,
) -> bool {
    let (cx, cy, cz) = cell;
    let existing = d.block(cx, cy, cz);
    let name = d.block_name(existing).to_string();
    if name != "minecraft:air" && name != "minecraft:water" && name != cfg.block {
        return false;
    }
    let Some(state) = growth_state(d, cfg, existing, face) else {
        return false;
    };
    if !can_attach(d, cx, cy, cz, face) {
        return false;
    }
    d.set_block(cx, cy, cz, state);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> BlockRegistry {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/blocks.json");
        BlockRegistry::load(&path).expect("block registry pins")
    }

    fn region() -> Decorator<'static> {
        let reg = Box::leak(Box::new(registry()));
        let terrain = Box::leak(Box::new(
            crate::terrain::HeightmapGenerator::with_seed(42, reg).expect("density generator"),
        ));
        let mut dec = Decorator::new(terrain, reg, 42).expect("decorator");
        dec.ensure_chunk(0, 0);
        dec
    }

    fn test_rng() -> DecorRng {
        let mut rng = DecorRng::new();
        rng.set_feature_seed(7, 0, 9);
        rng
    }

    fn feature_value(name: &str) -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../pins/worldgen/feature")
            .join(format!("{name}.json"));
        let raw = std::fs::read_to_string(&path).expect("feature pin");
        serde_json::from_str(&raw).expect("feature json")
    }

    fn name_at(d: &Decorator, x: i32, y: i32, z: i32) -> String {
        d.block_name(d.block(x, y, z)).to_string()
    }

    /// The first free cell above the terrain surface.
    fn surface(d: &mut Decorator, x: i32, z: i32) -> i32 {
        d.height(HeightKind::WorldSurface, x, z)
    }

    /// A tree writes the states it drew: the soil swap under the trunk,
    /// the trunk column, and a leaf shell above it.
    #[test]
    fn oak_tree_writes_trunk_and_canopy() {
        let mut d = region();
        let cfg = feature_value("oak");
        let base = surface(&mut d, 8, 8);
        run_feature(&mut d, &cfg, &mut test_rng(), 8, base, 8);
        assert_eq!(name_at(&d, 8, base, 8), "minecraft:oak_log");
        assert_eq!(
            name_at(&d, 8, base - 1, 8),
            "minecraft:dirt",
            "grass under the trunk swaps to dirt"
        );
        let mut logs = 0;
        let mut leaves = 0;
        for dy in 0..12 {
            for dx in -3..=3 {
                for dz in -3..=3 {
                    let name = name_at(&d, 8 + dx, base + dy, 8 + dz);
                    if name == "minecraft:oak_log" {
                        logs += 1;
                    }
                    if name == "minecraft:oak_leaves" {
                        leaves += 1;
                    }
                }
            }
        }
        assert!(logs >= 4, "trunk height at least the base height");
        assert!(leaves > 8, "canopy wrote leaves");
    }

    /// The dark oak trunk is a leaning 2x2 column with branch stubs and
    /// four soil swaps below.
    #[test]
    fn dark_oak_trunk_is_double() {
        let mut d = region();
        let cfg = feature_value("dark_oak");
        let base = surface(&mut d, 8, 8);
        run_feature(&mut d, &cfg, &mut test_rng(), 8, base, 8);
        assert_eq!(name_at(&d, 8, base - 1, 8), "minecraft:dirt");
        assert_eq!(name_at(&d, 9, base - 1, 9), "minecraft:dirt");
        assert_eq!(name_at(&d, 8, base, 8), "minecraft:dark_oak_log");
        assert_eq!(name_at(&d, 9, base, 8), "minecraft:dark_oak_log");
        assert_eq!(name_at(&d, 8, base, 9), "minecraft:dark_oak_log");
        assert_eq!(name_at(&d, 9, base, 9), "minecraft:dark_oak_log");
        let mut leaves = 0;
        for dy in 0..14 {
            for dx in -4..=4 {
                for dz in -4..=4 {
                    if name_at(&d, 8 + dx, base + dy, 8 + dz) == "minecraft:dark_oak_leaves" {
                        leaves += 1;
                    }
                }
            }
        }
        assert!(leaves > 20, "the dark crown wrote leaves");
    }

    /// Glow lichen hugs the first placeable face: a stone ceiling above
    /// the searched cell grows an up-facing patch.
    #[test]
    fn glow_lichen_attaches_to_stone() {
        let mut d = region();
        let stone = d.state_id_of("minecraft:stone", "").expect("stone");
        let top = surface(&mut d, 8, 2);
        d.set_block(8, top + 6, 2, stone);
        let cfg = feature_value("glow_lichen");
        run_feature(&mut d, &cfg, &mut test_rng(), 8, top + 5, 2);
        assert_eq!(name_at(&d, 8, top + 5, 2), "minecraft:glow_lichen");
        let (_, props) = d.registry().state_of(d.block(8, top + 5, 2)).unwrap();
        assert!(props.contains("up=true"), "props {props}");
    }

    /// A block column truncates against ground and only writes up to
    /// the free run.
    #[test]
    fn block_column_truncates_against_ground() {
        let mut d = region();
        let top = surface(&mut d, 10, 10);
        let dirt = d.state_id_of("minecraft:dirt", "").expect("dirt");
        d.set_block(10, top + 4, 10, dirt);
        let cfg: Value = serde_json::from_str(
            r#"{
                "type": "minecraft:block_column",
                "direction": "up",
                "allowed_placement": {"type": "minecraft:matching_block_tag", "tag": "minecraft:air"},
                "prioritize_tip": false,
                "layers": [
                    {"height": 5, "provider": {"id": "minecraft:oak_log", "properties": {"axis": "y"}}}
                ]
            }"#,
        )
        .unwrap();
        run_feature(&mut d, &cfg, &mut test_rng(), 10, top, 10);
        assert_eq!(name_at(&d, 10, top, 10), "minecraft:oak_log");
        assert_eq!(name_at(&d, 10, top + 1, 10), "minecraft:oak_log");
        assert_eq!(name_at(&d, 10, top + 2, 10), "minecraft:oak_log");
        assert_eq!(
            name_at(&d, 10, top + 3, 10),
            "minecraft:air",
            "the column stopped before the blocker"
        );
        assert_eq!(name_at(&d, 10, top + 4, 10), "minecraft:dirt");
    }

    /// A cave vine hangs from the ceiling: body cells from the provider
    /// registry reference, the berry-tipped head at the tip.
    #[test]
    fn cave_vine_hangs_from_ceiling() {
        let mut d = region();
        let top = surface(&mut d, 8, 8);
        let floor = top - 12;
        let air = d.state_id_of("minecraft:air", "").expect("air");
        for dy in 0..5 {
            d.set_block(8, floor + dy, 8, air);
        }
        let cfg = feature_value("cave_vine");
        run_feature(&mut d, &cfg, &mut test_rng(), 8, floor + 4, 8);
        let mut cells: Vec<(i32, String)> = Vec::new();
        for dy in 0..6 {
            let name = name_at(&d, 8, floor + dy, 8);
            if name.starts_with("minecraft:cave_vines") {
                cells.push((floor + dy, name));
            }
        }
        assert!(!cells.is_empty(), "the vine column placed");
        let head_y = cells[0].0;
        assert_eq!(
            name_at(&d, 8, head_y, 8),
            "minecraft:cave_vines",
            "the tip is a head"
        );
        let head = d.block(8, head_y, 8);
        let (_, props) = d.registry().state_of(head).unwrap();
        let age = BlockRegistry::prop_int(props, "age").expect("head age");
        assert!((23..=25).contains(&age), "head age {age}");
    }

    /// The weighted state provider draws once and lands inside its
    /// weights.
    #[test]
    fn weighted_provider_draws_once() {
        let mut d = region();
        let cfg = feature_value("leaf_litter");
        let provider = StateProvider::parse(&mut d, cfg.get("to_place").unwrap());
        let StateProvider::Weighted { total, ref entries } = provider else {
            panic!("leaf litter is a weighted provider");
        };
        assert_eq!(total, 12, "twelve equal segments");
        assert_eq!(entries.len(), 12);
        let mut a = test_rng();
        let _ = a.next_int(total);
        let probe = a.next_int(1 << 30);
        let mut b = test_rng();
        let _ = provider.sample(&mut d, &mut b, 0, 0, 0);
        assert_eq!(
            b.next_int(1 << 30),
            probe,
            "sampling consumed exactly one draw"
        );
    }
}
