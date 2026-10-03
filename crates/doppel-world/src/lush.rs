//! Lush-cave features: springs and the cave-surface ground patches with
//! the vegetation that grows on them.
//!
//! Every draw the reference features make is reproduced in shape and
//! order, because one feature's leftover stream moves every later
//! placement of that same feature.

use serde_json::Value;

use crate::decoration::{DecorRng, Decorator, IntDraw};

/// Whether a block name is one of the air family.
pub(crate) fn is_air_name(name: &str) -> bool {
    matches!(
        name,
        "minecraft:air" | "minecraft:cave_air" | "minecraft:void_air"
    )
}

/// Places a fluid pocket embedded in stone: the walls around the cell
/// must be the config's blocks with exactly the configured open sides,
/// and the draw stream stands still. The source fluid's legacy block is
/// the plain fluid block.
pub(crate) fn run_spring(
    d: &mut Decorator,
    v: &Value,
    _rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
) -> bool {
    let Some(fluid) = v
        .get("state")
        .and_then(|s| s.get("id"))
        .and_then(Value::as_str)
    else {
        return false;
    };
    let Some(state) = d.state_id_of(fluid, "") else {
        return false;
    };
    let requires_below = v
        .get("requires_block_below")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let rock_count = int_field(v, "rock_count", 4);
    let hole_count = int_field(v, "hole_count", 1);
    let Some(valid) = v.get("valid_blocks").and_then(Value::as_array) else {
        return false;
    };
    let valid: Vec<&str> = valid.iter().filter_map(Value::as_str).collect();
    if valid.is_empty() {
        return false;
    }
    let name_at = |d: &Decorator, x: i32, y: i32, z: i32| -> String {
        d.block_name(d.block(x, y, z)).to_string()
    };
    let is_valid = move |name: &str| valid.contains(&name);
    if !is_valid(&name_at(d, x, y + 1, z)) {
        return false;
    }
    if requires_below && !is_valid(&name_at(d, x, y - 1, z)) {
        return false;
    }
    let here = name_at(d, x, y, z);
    if !is_air_name(&here) && !is_valid(&here) {
        return false;
    }
    let mut rocks = 0;
    let mut holes = 0;
    for (nx, ny, nz) in [
        (x - 1, y, z),
        (x + 1, y, z),
        (x, y, z - 1),
        (x, y, z + 1),
        (x, y - 1, z),
    ] {
        let name = name_at(d, nx, ny, nz);
        if is_valid(&name) {
            rocks += 1;
        }
        if is_air_name(&name) {
            holes += 1;
        }
    }
    if rocks == rock_count && holes == hole_count {
        d.set_block(x, y, z, state);
        return true;
    }
    false
}

/// A config integer with its vanilla default.
fn int_field(v: &Value, key: &str, default: i32) -> i32 {
    v.get(key)
        .and_then(Value::as_i64)
        .map_or(default, |n| n as i32)
}

/// A config float with its vanilla default.
fn float_field(v: &Value, key: &str, default: f32) -> f32 {
    v.get(key)
        .and_then(Value::as_f64)
        .map_or(default, |n| n as f32)
}

/// The ground a patch may replace: a block tag or an explicit list.
enum ReplaceSet {
    Tag(String),
    Names(Vec<String>),
}

impl ReplaceSet {
    fn parse(v: &Value) -> Option<ReplaceSet> {
        match v {
            Value::String(raw) => {
                if let Some(tag) = raw.strip_prefix('#') {
                    let tag = tag.strip_prefix("minecraft:").unwrap_or(tag);
                    Some(ReplaceSet::Tag(tag.to_string()))
                } else {
                    Some(ReplaceSet::Names(vec![raw.clone()]))
                }
            }
            Value::Array(list) => {
                let names = list
                    .iter()
                    .filter_map(|entry| entry.as_str().map(String::from))
                    .collect::<Vec<_>>();
                if names.is_empty() {
                    None
                } else {
                    Some(ReplaceSet::Names(names))
                }
            }
            _ => None,
        }
    }

    fn test(&self, d: &mut Decorator, name: &str) -> bool {
        match self {
            ReplaceSet::Tag(tag) => d.tag_contains(tag, name),
            ReplaceSet::Names(names) => names.iter().any(|n| n == name),
        }
    }
}

/// One cave-surface ground patch; the waterlogged variant floods the
/// sheltered ground cells and grows the vegetation inside the water.
pub(crate) struct PatchCfg {
    replaceable: ReplaceSet,
    ground: u32,
    ground_name: String,
    vegetation: Value,
    ceiling: bool,
    depth: IntDraw,
    extra_bottom: f32,
    vertical_range: i32,
    vegetation_chance: f32,
    xz_radius: IntDraw,
    extra_edge: f32,
    waterlogged: bool,
}

/// Parses a patch config; None marks a shape this engine does not run
/// (the ground state provider must resolve to a plain state).
pub(crate) fn parse_patch(d: &Decorator, v: &Value, waterlogged: bool) -> Option<PatchCfg> {
    let replaceable = ReplaceSet::parse(v.get("replaceable")?)?;
    let ground = crate::features::plain_state(d, v.get("ground_state")?)?;
    let ground_name = d.block_name(ground).to_string();
    let surface = v.get("surface").and_then(Value::as_str)?;
    if surface != "floor" && surface != "ceiling" {
        return None;
    }
    let vegetation = v.get("vegetation_feature")?.clone();
    let depth = IntDraw::parse(v.get("depth")?).ok()?;
    let xz_radius = IntDraw::parse(v.get("xz_radius")?).ok()?;
    Some(PatchCfg {
        replaceable,
        ground,
        ground_name,
        vegetation,
        ceiling: surface == "ceiling",
        depth,
        extra_bottom: float_field(v, "extra_bottom_block_chance", 0.0),
        vertical_range: int_field(v, "vertical_range", 1),
        vegetation_chance: float_field(v, "vegetation_chance", 0.0),
        xz_radius,
        extra_edge: float_field(v, "extra_edge_column_chance", 0.0),
        waterlogged,
    })
}

/// The position hash the reference sets order by: the axes fold into
/// one word.
fn cell_hash(x: i32, y: i32, z: i32) -> u32 {
    (y.wrapping_add(z.wrapping_mul(31)))
        .wrapping_mul(31)
        .wrapping_add(x) as u32
}

/// The iteration order of the reference hash set over inserted cells:
/// table buckets at the final capacity (the table doubles past the load
/// threshold, and a bucket reaching eight cells forces a doubling under
/// capacity sixty-four), bucket order first, insertion order within a
/// bucket.
fn hash_order(cells: &[(i32, i32, i32)]) -> Vec<usize> {
    let mut cap: u32 = 16;
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for (i, &(x, y, z)) in cells.iter().enumerate() {
        let h = cell_hash(x, y, z);
        let bucket = (h ^ (h >> 16)) & (cap - 1);
        let filled = {
            let entry = counts.entry(bucket).or_default();
            *entry += 1;
            *entry
        };
        let size = i as u32 + 1;
        let mut grow = size > cap * 3 / 4 || (filled >= 8 && cap < 64);
        while grow {
            cap *= 2;
            let mut recount: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for &(cx, cy, cz) in cells.iter().take(i + 1) {
                let ch = cell_hash(cx, cy, cz);
                *recount.entry((ch ^ (ch >> 16)) & (cap - 1)).or_default() += 1;
            }
            counts = recount;
            grow = size > cap * 3 / 4;
        }
    }
    let mask = cap - 1;
    let mut order: Vec<usize> = (0..cells.len()).collect();
    order.sort_by_key(|&i| {
        let h = cell_hash(cells[i].0, cells[i].1, cells[i].2);
        ((h ^ (h >> 16)) & mask, i)
    });
    order
}

/// Whether a ground cell opens sideways or downward onto anything but a
/// sturdy face.
fn is_exposed(d: &mut Decorator, x: i32, y: i32, z: i32) -> bool {
    [
        (x, y, z - 1),
        (x + 1, y, z),
        (x, y, z + 1),
        (x - 1, y, z),
        (x, y - 1, z),
    ]
    .iter()
    .any(|&(nx, ny, nz)| {
        let name = d.block_name(d.block(nx, ny, nz)).to_string();
        !d.tag_contains("blocks_motion_no_leaves", &name)
    })
}

/// Places a disk of ground against the cave surface and grows the
/// configured vegetation over it: one radius pair up front, then per
/// column an edge float, a depth draw with its extra-bottom float, and
/// the vegetation floats over the surface set in hash order.
pub(crate) fn run_patch(
    d: &mut Decorator,
    v: &Value,
    rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
) -> bool {
    let waterlogged =
        v.get("type").and_then(Value::as_str) == Some("minecraft:waterlogged_vegetation_patch");
    let Some(cfg) = parse_patch(d, v, waterlogged) else {
        return false;
    };
    let x_radius = cfg.xz_radius.sample(rng) + 1;
    let z_radius = cfg.xz_radius.sample(rng) + 1;
    // The floor patch digs downward into its ground; the ceiling patch
    // digs upward.
    let inward = if cfg.ceiling { 1 } else { -1 };
    let outward = -inward;
    let air_at = |d: &Decorator, x: i32, y: i32, z: i32| -> bool {
        is_air_name(d.block_name(d.block(x, y, z)))
    };
    let mut surface: Vec<(i32, i32, i32)> = Vec::new();
    for dx in -x_radius..=x_radius {
        let x_edge = dx == -x_radius || dx == x_radius;
        for dz in -z_radius..=z_radius {
            let z_edge = dz == -z_radius || dz == z_radius;
            // Corners drop without a draw; edges thin out on a float.
            let is_corner = x_edge && z_edge;
            let edge_not_corner = (x_edge || z_edge) && !is_corner;
            if is_corner
                || (edge_not_corner && (cfg.extra_edge == 0.0 || rng.next_f32() > cfg.extra_edge))
            {
                continue;
            }
            let (px, pz) = (x + dx, z + dz);
            let mut py = y;
            let mut steps = 0;
            while air_at(d, px, py, pz) && steps < cfg.vertical_range {
                py += inward;
                steps += 1;
            }
            steps = 0;
            while !air_at(d, px, py, pz) && steps < cfg.vertical_range {
                py += outward;
                steps += 1;
            }
            if !air_at(d, px, py, pz) {
                continue;
            }
            let ground_name = d.block_name(d.block(px, py + inward, pz)).to_string();
            if !d.tag_contains("blocks_motion_no_leaves", &ground_name) {
                continue;
            }
            let depth = cfg.depth.sample(rng)
                + i32::from(cfg.extra_bottom > 0.0 && rng.next_f32() < cfg.extra_bottom);
            // The ground walk swaps blocks deeper until it runs out of
            // depth, hits the same block, or meets non-replaceable
            // ground (placed nothing at depth zero means no cell).
            let mut ground_placed = true;
            'ground: {
                let mut gy = py + inward;
                for i in 0..depth {
                    let existing = d.block_name(d.block(px, gy, pz)).to_string();
                    if existing == cfg.ground_name {
                        continue;
                    }
                    if !cfg.replaceable.test(d, &existing) {
                        ground_placed = i != 0;
                        break 'ground;
                    }
                    d.set_block(px, gy, pz, cfg.ground);
                    gy += inward;
                }
            }
            if !ground_placed {
                continue;
            }
            surface.push((px, py + inward, pz));
        }
    }
    // The waterlogged variant floods the cells the patch does not
    // expose, and grows its vegetation inside the flooded cells.
    if cfg.waterlogged {
        let mut water: Vec<(i32, i32, i32)> = Vec::new();
        for &i in &hash_order(&surface) {
            let at = surface[i];
            if !is_exposed(d, at.0, at.1, at.2) {
                water.push(at);
            }
        }
        if let Some(water_state) = d.state_id_of("minecraft:water", "") {
            for &(wx, wy, wz) in &water {
                d.set_block(wx, wy, wz, water_state);
            }
        }
        distribute_vegetation(d, &cfg, rng, &water);
        !water.is_empty()
    } else {
        distribute_vegetation(d, &cfg, rng, &surface);
        !surface.is_empty()
    }
}

/// Runs the vegetation feature over a surface set in hash order: one
/// float per cell (when the chance is positive), the feature placed at
/// the open cell beside the ground, or inside the flooded cell itself.
fn distribute_vegetation(
    d: &mut Decorator,
    cfg: &PatchCfg,
    rng: &mut DecorRng,
    cells: &[(i32, i32, i32)],
) {
    let outward = if cfg.ceiling { -1 } else { 1 };
    for &i in &hash_order(cells) {
        let (gx, gy, gz) = cells[i];
        if !(cfg.vegetation_chance > 0.0 && rng.next_f32() < cfg.vegetation_chance) {
            continue;
        }
        if cfg.waterlogged {
            let placed =
                crate::features::run_placed_value(d, Some(&cfg.vegetation), rng, gx, gy, gz);
            if placed {
                waterlog_cell(d, gx, gy, gz);
            }
        } else {
            crate::features::run_placed_value(d, Some(&cfg.vegetation), rng, gx, gy + outward, gz);
        }
    }
}

/// Sets the waterlogged property on a cell the flooded patch just grew
/// vegetation into.
fn waterlog_cell(d: &mut Decorator, x: i32, y: i32, z: i32) {
    let state = d.block(x, y, z);
    let Some((name, props)) = d
        .registry()
        .state_of(state)
        .map(|(n, p)| (n.to_string(), p.to_string()))
    else {
        return;
    };
    if !props.split(',').any(|pair| pair == "waterlogged=false") {
        return;
    }
    if let Some(id) = d.state_id_of(
        &name,
        &crate::registry::BlockRegistry::with_prop(&props, "waterlogged", "true"),
    ) {
        d.set_block(x, y, z, id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region() -> Decorator<'static> {
        let blocks =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/blocks.json");
        let reg = Box::leak(Box::new(
            crate::registry::BlockRegistry::load(&blocks).expect("block registry pins"),
        ));
        let terrain = Box::leak(Box::new(
            crate::terrain::HeightmapGenerator::with_seed(42, reg).expect("density generator"),
        ));
        let mut dec = Decorator::new(terrain, reg, 42).expect("decorator");
        dec.ensure_chunk(0, 0);
        dec
    }

    fn feature_value(name: &str) -> Value {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../pins/worldgen/feature")
            .join(format!("{name}.json"));
        let raw = std::fs::read_to_string(&path).expect("feature pin");
        serde_json::from_str(&raw).expect("feature json")
    }

    /// A moss patch carpets a carved cave floor: the ground under the
    /// open cells swaps to moss and vegetation grows over the surface.
    #[test]
    fn moss_patch_carves_and_grows() {
        let mut d = region();
        let top = d.height(crate::decoration::HeightKind::WorldSurface, 8, 8);
        assert!(top > MIN_TEST_Y + 12, "room to carve below the surface");
        let floor = top - 10;
        let air = d.state_id_of("minecraft:air", "").expect("air");
        for dx in -6..=6 {
            for dz in -6..=6 {
                for dy in 0..4 {
                    d.set_block(8 + dx, floor + dy, 8 + dz, air);
                }
            }
        }
        let cfg = feature_value("moss_patch");
        let mut rng = DecorRng::new();
        rng.set_feature_seed(7, 0, 9);
        let placed = run_patch(&mut d, &cfg, &mut rng, 8, floor + 2, 8);
        assert!(placed, "the patch found ground");
        let mut moss = 0;
        let mut plants = 0;
        for dx in -8..=8 {
            for dz in -8..=8 {
                for dy in -2..5 {
                    let name = d
                        .block_name(d.block(8 + dx, floor + dy, 8 + dz))
                        .to_string();
                    match name.as_str() {
                        "minecraft:moss_block" => moss += 1,
                        "minecraft:moss_carpet"
                        | "minecraft:short_grass"
                        | "minecraft:tall_grass"
                        | "minecraft:azalea"
                        | "minecraft:flowering_azalea" => plants += 1,
                        _ => {}
                    }
                }
            }
        }
        assert!(moss > 4, "moss ground written, found {moss}");
        assert!(plants > 0, "vegetation grew, found {plants}");
    }

    /// The waterlogged patch floods the cells the clay disk does not
    /// expose and grows its vegetation inside the water.
    #[test]
    fn clay_pool_floods_sheltered_cells() {
        let mut d = region();
        let top = d.height(crate::decoration::HeightKind::WorldSurface, 8, 8);
        let floor = top - 10;
        let air = d.state_id_of("minecraft:air", "").expect("air");
        // A wide, shallow basin: air above a stone rim, so the disk's
        // interior cells sit enclosed by ground on every side but up.
        for dx in -8..=8 {
            for dz in -8..=8 {
                for dy in 0..4 {
                    d.set_block(8 + dx, floor + dy, 8 + dz, air);
                }
            }
        }
        let cfg = feature_value("clay_pool_with_dripleaves");
        let mut rng = DecorRng::new();
        rng.set_feature_seed(7, 0, 9);
        let placed = run_patch(&mut d, &cfg, &mut rng, 8, floor + 2, 8);
        assert!(placed, "the pool found ground");
        let mut clay = 0;
        let mut water = 0;
        for dx in -8..=8 {
            for dz in -8..=8 {
                for dy in -3..5 {
                    let name = d
                        .block_name(d.block(8 + dx, floor + dy, 8 + dz))
                        .to_string();
                    match name.as_str() {
                        "minecraft:clay" => clay += 1,
                        "minecraft:water" => water += 1,
                        _ => {}
                    }
                }
            }
        }
        assert!(clay > 4, "clay ground written, found {clay}");
        assert!(water > 0, "sheltered cells flooded, found {water}");
    }

    /// The floor is far underground, inside the build range the test
    /// world guarantees.
    const MIN_TEST_Y: i32 = crate::worldgen::MIN_Y + 64;
}
