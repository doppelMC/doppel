//! Ore blobs and surface disks: the two replacement features that
//! overwrite terrain in place instead of adding to it.
//!
//! Every draw lands in the exact reference order because the two features
//! share one stream with their placement stack: a blob draws its heading
//! float, two vertical offsets, then one swell double per ball; a disk
//! draws only its radius. The air-exposure discard draws per target that
//! passes its rule, inside the per-cell walk.

use std::f32::consts::PI;
use std::sync::LazyLock;

use serde_json::Value;

use crate::decoration::{DecorRng, Decorator, HeightKind, IntDraw, Predicate};
use crate::features::{plain_state, test_predicate};
use crate::worldgen::MIN_Y;

/// The world build ceiling.
const WORLD_TOP: i32 = 320;

// ---------------------------------------------------------------------------
// The quarter-wave sine lookup the reference float math reads.
// ---------------------------------------------------------------------------

/// Table scale: radians to index units over 65536 steps of a full turn.
const SIN_SCALE: f64 = 10430.378350470453;

static SIN: LazyLock<Vec<f32>> = LazyLock::new(|| {
    (0..65536u32)
        .map(|i| ((i as f64 / SIN_SCALE).sin()) as f32)
        .collect()
});

/// The table sine: the double argument truncates to an index, the table
/// mask wraps it, and the entry is a float.
fn table_sin(a: f32) -> f32 {
    let index = ((a as f64 * SIN_SCALE) as i64 as u64 & 0xFFFF) as usize;
    SIN[index]
}

// ---------------------------------------------------------------------------
// Replacement rules.
// ---------------------------------------------------------------------------

/// A target rule: the state a rule matches against, with optional bounds
/// on the world position.
#[derive(Clone)]
pub(crate) enum RuleTest {
    True,
    Tag(String),
    Block(String),
    Height { min: i32, max: i32 },
    All(Vec<RuleTest>),
    Any(Vec<RuleTest>),
    Not(Box<RuleTest>),
}

impl RuleTest {
    /// Parses a rule; None names a shape the engine does not evaluate.
    pub(crate) fn parse(v: &Value) -> Option<RuleTest> {
        let kind = v.get("predicate_type").and_then(Value::as_str)?;
        let rules = |key: &str| -> Option<Vec<RuleTest>> {
            let list = v.get(key).and_then(Value::as_array)?;
            list.iter().map(RuleTest::parse).collect()
        };
        match kind {
            "minecraft:always_true" => Some(RuleTest::True),
            "minecraft:tag_match" => {
                let tag = v.get("tag").and_then(Value::as_str)?;
                let tag = tag.strip_prefix("minecraft:").unwrap_or(tag);
                Some(RuleTest::Tag(tag.to_string()))
            }
            "minecraft:block_match" => Some(RuleTest::Block(
                v.get("block").and_then(Value::as_str)?.to_string(),
            )),
            "minecraft:height_match" => Some(RuleTest::Height {
                min: v.get("min_inclusive").and_then(Value::as_i64)? as i32,
                max: v.get("max_inclusive").and_then(Value::as_i64)? as i32,
            }),
            "minecraft:all_of" => Some(RuleTest::All(rules("rules")?)),
            "minecraft:any_of" => Some(RuleTest::Any(rules("rules")?)),
            "minecraft:not" => Some(RuleTest::Not(Box::new(RuleTest::parse(v.get("rule")?)?))),
            _ => None,
        }
    }

    /// Whether the state at the position matches the rule.
    fn test(&self, d: &mut Decorator, name: &str, y: i32) -> bool {
        match self {
            RuleTest::True => true,
            RuleTest::Tag(tag) => d.tag_contains(tag, name),
            RuleTest::Block(block) => block == name,
            RuleTest::Height { min, max } => y >= *min && y <= *max,
            RuleTest::All(list) => list.iter().all(|r| r.test(d, name, y)),
            RuleTest::Any(list) => list.iter().any(|r| r.test(d, name, y)),
            RuleTest::Not(inner) => !inner.test(d, name, y),
        }
    }
}

// ---------------------------------------------------------------------------
// Ore blobs.
// ---------------------------------------------------------------------------

/// An ore config: ball count, the discard gate, and the replacement list.
pub(crate) struct OreCfg {
    size: i32,
    discard: f32,
    targets: Vec<(RuleTest, u32)>,
}

impl OreCfg {
    /// Parses the config; None skips the feature before any draw moves.
    pub(crate) fn parse(d: &Decorator, v: &Value) -> Option<OreCfg> {
        let size = v.get("size").and_then(Value::as_i64)? as i32;
        if !(0..=64).contains(&size) {
            return None;
        }
        let discard = v
            .get("discard_chance_on_air_exposure")
            .and_then(Value::as_f64)
            .unwrap_or(0.0) as f32;
        let mut targets = Vec::new();
        for entry in v.get("targets").and_then(Value::as_array)? {
            let rule = RuleTest::parse(entry.get("target")?)?;
            let state = plain_state(d, entry.get("state")?)?;
            targets.push((rule, state));
        }
        if targets.is_empty() {
            return None;
        }
        Some(OreCfg {
            size,
            discard,
            targets,
        })
    }
}

/// The blob carve: the heading and offsets fix the spine, one swell per
/// ball sizes it, overlapping balls drop out, and the surviving spheres
/// walk their cells top to bottom.
pub(crate) fn run_ore(d: &mut Decorator, v: &Value, rng: &mut DecorRng, ox: i32, oy: i32, oz: i32) {
    let Some(cfg) = OreCfg::parse(d, v) else {
        return;
    };
    let dir = rng.next_f32() * PI;
    let spread = cfg.size as f32 / 8.0;
    let reach = (((cfg.size as f32 / 16.0 * 2.0 + 1.0) / 2.0).ceil()) as i32;
    let sin = (dir as f64).sin();
    let cos = (dir as f64).cos();
    let x0 = ox as f64 + sin * spread as f64;
    let x1 = ox as f64 - sin * spread as f64;
    let z0 = oz as f64 + cos * spread as f64;
    let z1 = oz as f64 - cos * spread as f64;
    let y0 = (oy + rng.next_int(3) - 2) as f64;
    let y1 = (oy + rng.next_int(3) - 2) as f64;
    let pad = spread.ceil() as i32;
    let x_start = ox - pad - reach;
    let y_start = oy - 2 - reach;
    let z_start = oz - pad - reach;
    let span_xz = 2 * (pad + reach);
    let span_y = 2 * (2 + reach);
    // The carve runs only where the frozen ocean floor sits at or above
    // the blob floor somewhere in the box.
    let mut grounded = false;
    'probe: for px in x_start..=x_start + span_xz {
        for pz in z_start..=z_start + span_xz {
            if y_start <= d.height(HeightKind::OceanFloorWg, px, pz) {
                grounded = true;
                break 'probe;
            }
        }
    }
    if !grounded {
        return;
    }

    // One spine point per ball: the lerp position and a swelled radius.
    let mut balls: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(cfg.size as usize);
    for i in 0..cfg.size {
        let step = (i as f32 / cfg.size as f32) as f64;
        let swell = rng.next_double() * cfg.size as f64 / 16.0;
        let radius =
            ((table_sin(PI * (i as f32 / cfg.size as f32)) + 1.0) as f64 * swell + 1.0) / 2.0;
        balls.push((
            x0 + step * (x1 - x0),
            y0 + step * (y1 - y0),
            z0 + step * (z1 - z0),
            radius,
        ));
    }
    // A ball inside a larger neighbor drops out; a tie keeps the earlier
    // index by killing the later one.
    for first in 0..cfg.size as usize - 1 {
        if balls[first].3 <= 0.0 {
            continue;
        }
        for second in first + 1..cfg.size as usize {
            if balls[second].3 <= 0.0 {
                continue;
            }
            let (fx, fy, fz, fr) = balls[first];
            let (sx, sy, sz, sr) = balls[second];
            let dr = fr - sr;
            let gap = (fx - sx) * (fx - sx) + (fy - sy) * (fy - sy) + (fz - sz) * (fz - sz);
            if dr * dr > gap {
                if dr > 0.0 {
                    balls[second].3 = -1.0;
                } else {
                    balls[first].3 = -1.0;
                }
            }
        }
    }

    let mut tested = vec![false; (span_xz * span_y * span_xz) as usize];
    // The dedup index leaves the x axis unscaled by the y span, so an
    // index can run past the array; the reference bit set grows there
    // instead of refusing, and this side set carries those cells.
    let mut tested_over: std::collections::HashSet<i32> = std::collections::HashSet::new();
    for &(bx, by, bz, r) in &balls {
        if r < 0.0 {
            continue;
        }
        let x_min = ((bx - r).floor() as i32).max(x_start);
        let y_min = ((by - r).floor() as i32).max(y_start);
        let z_min = ((bz - r).floor() as i32).max(z_start);
        let x_max = ((bx + r).floor() as i32).max(x_min);
        let y_max = ((by + r).floor() as i32).max(y_min);
        let z_max = ((bz + r).floor() as i32).max(z_min);
        for x in x_min..=x_max {
            let xd = (x as f64 + 0.5 - bx) / r;
            let xs = xd * xd;
            if xs >= 1.0 {
                continue;
            }
            for y in y_min..=y_max {
                let yd = (y as f64 + 0.5 - by) / r;
                if xs + yd * yd >= 1.0 {
                    continue;
                }
                for z in z_min..=z_max {
                    let zd = (z as f64 + 0.5 - bz) / r;
                    if xs + yd * yd + zd * zd >= 1.0 {
                        continue;
                    }
                    if !(MIN_Y..WORLD_TOP).contains(&y) {
                        continue;
                    }
                    let bit =
                        (x - x_start) + (y - y_start) * span_xz + (z - z_start) * span_xz * span_y;
                    let inside = bit >= 0 && (bit as usize) < tested.len();
                    let seen = if inside {
                        tested[bit as usize]
                    } else {
                        tested_over.contains(&bit)
                    };
                    if seen {
                        continue;
                    }
                    if inside {
                        tested[bit as usize] = true;
                    } else {
                        tested_over.insert(bit);
                    }
                    let here = d.block_name(d.block(x, y, z)).to_string();
                    for (rule, state) in &cfg.targets {
                        if !rule.test(d, &here, y) {
                            continue;
                        }
                        if !skips_air_check(rng, cfg.discard) && touches_air(d, x, y, z) {
                            continue;
                        }
                        d.set_block(x, y, z, *state);
                        break;
                    }
                }
            }
        }
    }
}

/// Whether the exposure check skips: no discard chance always skips, a
/// full chance never does, and anything between draws.
fn skips_air_check(rng: &mut DecorRng, discard: f32) -> bool {
    if discard <= 0.0 {
        return true;
    }
    if discard >= 1.0 {
        return false;
    }
    rng.next_f32() >= discard
}

/// Whether any of the six neighbors is air, in the reference face order.
/// A missing section reads as air, the same as the reference bulk reader.
fn touches_air(d: &Decorator, x: i32, y: i32, z: i32) -> bool {
    const FACES: [(i32, i32, i32); 6] = [
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
        (-1, 0, 0),
        (1, 0, 0),
    ];
    FACES.iter().any(|&(dx, dy, dz)| {
        matches!(
            d.block_name(d.block(x + dx, y + dy, z + dz)),
            "minecraft:air" | "minecraft:cave_air" | "minecraft:void_air"
        )
    })
}

// ---------------------------------------------------------------------------
// Surface disks.
// ---------------------------------------------------------------------------

/// A disk state source: one state, or the first rule whose predicate
/// holds at the write position over a fallback.
enum DiskState {
    Fixed(u32),
    Rules {
        fallback: u32,
        rules: Vec<(Predicate, u32)>,
    },
}

impl DiskState {
    /// The state for a position; the rule walk draws nothing.
    fn sample(&self, d: &mut Decorator, x: i32, y: i32, z: i32) -> u32 {
        match self {
            DiskState::Fixed(state) => *state,
            DiskState::Rules { fallback, rules } => {
                for (predicate, state) in rules {
                    if test_predicate(d, predicate, x, y, z) {
                        return *state;
                    }
                }
                *fallback
            }
        }
    }
}

/// A disk config: the state source, the target predicate, the radius
/// draw, and the vertical half span.
pub(crate) struct DiskCfg {
    state: DiskState,
    target: Predicate,
    radius: IntDraw,
    half: i32,
}

impl DiskCfg {
    /// Parses the config; None skips the feature before any draw moves.
    pub(crate) fn parse(d: &Decorator, v: &Value) -> Option<DiskCfg> {
        let provider = v.get("state_provider")?;
        let state = match provider.get("type").and_then(Value::as_str) {
            Some("minecraft:rule_based") => {
                let fallback = plain_state(d, provider.get("fallback")?)?;
                let mut rules = Vec::new();
                for rule in provider.get("rules").and_then(Value::as_array)? {
                    let predicate = Predicate::parse(rule.get("if_true")?).ok()?;
                    let state = plain_state(d, rule.get("then")?)?;
                    rules.push((predicate, state));
                }
                DiskState::Rules { fallback, rules }
            }
            _ => DiskState::Fixed(plain_state(d, provider)?),
        };
        Some(DiskCfg {
            state,
            target: Predicate::parse(v.get("target")?).ok()?,
            radius: IntDraw::parse(v.get("radius")?).ok()?,
            half: v.get("half_height").and_then(Value::as_i64)? as i32,
        })
    }
}

/// The disk: one radius draw, then every column inside it walks from its
/// top down to its floor, replacing cells the target predicate names.
pub(crate) fn run_disk(
    d: &mut Decorator,
    v: &Value,
    rng: &mut DecorRng,
    ox: i32,
    oy: i32,
    oz: i32,
) {
    let Some(cfg) = DiskCfg::parse(d, v) else {
        return;
    };
    let radius = cfg.radius.sample(rng);
    let top = oy + cfg.half;
    let bottom = oy - cfg.half - 1;
    for z in oz - radius..=oz + radius {
        for x in ox - radius..=ox + radius {
            let dx = x - ox;
            let dz = z - oz;
            if dx * dx + dz * dz > radius * radius {
                continue;
            }
            for y in (bottom + 1..=top).rev() {
                if !test_predicate(d, &cfg.target, x, y, z) {
                    continue;
                }
                let state = cfg.state.sample(d, x, y, z);
                d.set_block(x, y, z, state);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoration::Decorator;
    use crate::registry::BlockRegistry;
    use crate::terrain::HeightmapGenerator;

    fn registry() -> BlockRegistry {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/blocks.json");
        BlockRegistry::load(&path).expect("block registry pins")
    }

    fn decorator() -> (HeightmapGenerator, BlockRegistry) {
        let reg = registry();
        let terrain = HeightmapGenerator::with_seed(42, &reg).expect("density generator");
        (terrain, reg)
    }

    /// One ore placement writes a ball of ore cells inside its box, all
    /// of them over target-rule blocks, and the stream moves by exactly
    /// the heading float, two offsets, and one swell per ball.
    #[test]
    fn ore_blob_writes_within_its_box() {
        let (terrain, reg) = decorator();
        let mut d = Decorator::new(&terrain, &reg, 42).expect("decorator");
        d.ensure_chunk(0, 0);
        let mut rng = DecorRng::new();
        let deco = rng.decoration_seed(42, 0, 0);
        rng.set_feature_seed(deco, 0, 6);

        let v: Value = serde_json::from_str(
            r#"{
                "type": "minecraft:ore",
                "discard_chance_on_air_exposure": 0.0,
                "size": 8,
                "targets": [
                    {"state": "minecraft:iron_ore", "target": {
                        "predicate_type": "minecraft:tag_match",
                        "tag": "minecraft:deepslate_ore_replaceables"}}
                ]
            }"#,
        )
        .unwrap();
        // A deep origin under solid stone: cells exist on every side.
        let (ox, oy, oz) = (8, -40, 8);
        let before = rng.words;
        let mut written = 0usize;
        let mut foreign = 0usize;
        let mut outside = 0usize;
        let mut touched = Vec::new();
        for x in -6..22 {
            for y in -60..-20 {
                for z in -6..22 {
                    let state = d.block(x, y, z);
                    if state != 0 {
                        touched.push((x, y, z, state));
                    }
                }
            }
        }
        let baseline: std::collections::HashMap<(i32, i32, i32), u32> = touched
            .into_iter()
            .map(|(x, y, z, s)| ((x, y, z), s))
            .collect();
        run_ore(&mut d, &v, &mut rng, ox, oy, oz);
        for x in -6..22 {
            for y in -60..-20 {
                for z in -6..22 {
                    let state = d.block(x, y, z);
                    if baseline.get(&(x, y, z)).copied().unwrap_or(0) != state {
                        let name = d.block_name(state);
                        if name == "minecraft:iron_ore" {
                            written += 1;
                            let reach =
                                8 / 8 + 2 + ((8.0f32 / 16.0 * 2.0 + 1.0) / 2.0).ceil() as i32;
                            if (x - ox).abs() > reach
                                || (z - oz).abs() > reach
                                || (y - oy).abs() > reach
                            {
                                outside += 1;
                            }
                        } else {
                            foreign += 1;
                        }
                    }
                }
            }
        }
        assert!(written > 0, "the blob placed ore");
        assert_eq!(foreign, 0, "only the ore state was written");
        assert_eq!(outside, 0, "every cell sits inside the blob box");
        // Three leading draws plus one double (two stream steps) per ball.
        assert_eq!(rng.words - before, 3 + 2 * 8);
    }

    /// The depth rule splits the variants: the stone-family tag above the
    /// split becomes the stone ore, deepslate below becomes the deepslate
    /// ore, and the height-specific tag follows the split line.
    #[test]
    fn depth_rules_split_stone_and_deepslate_states() {
        let v: Value = serde_json::from_str(
            r#"{
                "type": "minecraft:ore",
                "discard_chance_on_air_exposure": 0.0,
                "size": 4,
                "targets": [
                    {"state": "minecraft:iron_ore", "target": {
                        "predicate_type": "minecraft:any_of",
                        "rules": [
                            {"predicate_type": "minecraft:all_of", "rules": [
                                {"predicate_type": "minecraft:tag_match",
                                 "tag": "minecraft:height_specific_ore_replaceables"},
                                {"predicate_type": "minecraft:height_match",
                                 "min_inclusive": 0, "max_inclusive": 2031}]},
                            {"predicate_type": "minecraft:all_of", "rules": [
                                {"predicate_type": "minecraft:not", "rule": {
                                    "predicate_type": "minecraft:tag_match",
                                    "tag": "minecraft:height_specific_ore_replaceables"}},
                                {"predicate_type": "minecraft:tag_match",
                                 "tag": "minecraft:stone_ore_replaceables"}]}
                        ]}},
                    {"state": "minecraft:deepslate_iron_ore", "target": {
                        "predicate_type": "minecraft:any_of",
                        "rules": [
                            {"predicate_type": "minecraft:all_of", "rules": [
                                {"predicate_type": "minecraft:tag_match",
                                 "tag": "minecraft:height_specific_ore_replaceables"},
                                {"predicate_type": "minecraft:height_match",
                                 "min_inclusive": -2032, "max_inclusive": 8}]},
                            {"predicate_type": "minecraft:all_of", "rules": [
                                {"predicate_type": "minecraft:not", "rule": {
                                    "predicate_type": "minecraft:tag_match",
                                    "tag": "minecraft:height_specific_ore_replaceables"}},
                                {"predicate_type": "minecraft:tag_match",
                                 "tag": "minecraft:deepslate_ore_replaceables"}]}
                        ]}}
                ]
            }"#,
        )
        .unwrap();
        let cfg = OreCfg::parse(
            &Decorator::new(&decorator().0, &registry(), 42).unwrap(),
            &v,
        )
        .expect("config parses");
        assert_eq!(cfg.targets.len(), 2);

        // The rule verdicts over the tag families at the split heights:
        // stone family names map by height, tuff flips at the line, and
        // deepslate always takes the deepslate target.
        let (terrain, reg) = decorator();
        let mut d = Decorator::new(&terrain, &reg, 42).unwrap();
        let mut verdict = |name: &str, y: i32| -> usize {
            cfg.targets
                .iter()
                .position(|(rule, _)| rule.test(&mut d, name, y))
                .unwrap_or(usize::MAX)
        };
        assert_eq!(verdict("minecraft:stone", 40), 0, "stone high is iron");
        assert_eq!(
            verdict("minecraft:stone", -40),
            0,
            "stone family below zero stays the stone target"
        );
        assert_eq!(
            verdict("minecraft:deepslate", -40),
            1,
            "deepslate takes the deepslate target"
        );
        assert_eq!(
            verdict("minecraft:deepslate", 40),
            1,
            "deepslate keeps the deepslate target at any height"
        );
        assert_eq!(verdict("minecraft:tuff", 40), 0, "tuff above the line");
        assert_eq!(verdict("minecraft:tuff", 0), 0, "the line is inclusive");
        assert_eq!(
            verdict("minecraft:tuff", -1),
            1,
            "tuff below the line is deepslate-side"
        );
        assert_eq!(
            verdict("minecraft:dirt", 40),
            usize::MAX,
            "dirt matches neither target"
        );
    }

    /// The exposure gate: a full discard chance refuses every cell that
    /// touches air, while the zero chance ignores exposure entirely, and
    /// a fractional chance draws once per passing target.
    #[test]
    fn exposure_gate_discards_blobs_touching_air() {
        let (terrain, reg) = decorator();
        let mut d = Decorator::new(&terrain, &reg, 42).expect("decorator");
        d.ensure_chunk(0, 0);
        // A stone origin under partially open sky: some blob cells will
        // touch air while the buried ones do not.
        let mut origin = None;
        'search: for y in 0..80 {
            for x in 2..14 {
                for z in 2..14 {
                    let name = d.block_name(d.block(x, y, z));
                    if name == "minecraft:stone" {
                        origin = Some((x, y, z));
                        break 'search;
                    }
                }
            }
        }
        let (ox, oy, oz) = origin.expect("stone near the surface");
        let near_air = |d: &Decorator| -> bool {
            (0..3).any(|dx| {
                (0..3).any(|dz| {
                    let above = d.block_name(d.block(ox + dx, oy + 1, oz + dz));
                    above == "minecraft:air"
                })
            })
        };
        if !near_air(&d) {
            return;
        }

        let make = |discard: f32| -> Value {
            serde_json::from_str(&format!(
                r#"{{
                    "type": "minecraft:ore",
                    "discard_chance_on_air_exposure": {discard},
                    "size": 4,
                    "targets": [
                        {{"state": "minecraft:gold_ore", "target": {{
                            "predicate_type": "minecraft:block_match",
                            "block": "minecraft:stone"}}}}
                    ]
                }}"#
            ))
            .unwrap()
        };
        // The full-chance run checks exposure without drawing: buried
        // cells place, exposed ones refuse.
        let mut rng = DecorRng::new();
        let deco = rng.decoration_seed(42, 0, 0);
        rng.set_feature_seed(deco, 1, 6);
        let before = rng.words;
        run_ore(&mut d, &make(1.0), &mut rng, ox, oy, oz);
        let full_words = rng.words - before;
        let gold = |d: &Decorator| -> usize {
            (ox - 4..=ox + 4)
                .flat_map(|x| {
                    (oy - 4..=oy + 4).flat_map(move |y| (oz - 4..=oz + 4).map(move |z| (x, y, z)))
                })
                .filter(|&(x, y, z)| d.block_name(d.block(x, y, z)) == "minecraft:gold_ore")
                .count()
        };
        // Gold may appear when the blob avoids the air column entirely.
        let full_placed = gold(&d);
        assert!(full_words == 0 || full_placed > 0);

        // The zero-chance run places without any exposure draw.
        let mut rng = DecorRng::new();
        rng.set_feature_seed(deco, 2, 6);
        let before = rng.words;
        run_ore(&mut d, &make(0.0), &mut rng, ox, oy, oz);
        assert_eq!(rng.words - before, 3 + 2 * 4);
        assert!(
            gold(&d) > full_placed || full_placed > 0,
            "the ungated run placed at least as much gold"
        );
    }

    /// The disk replaces only its target blocks inside the radius circle,
    /// the rule provider picks grass under open sky and dirt elsewhere,
    /// and one radius draw is the whole stream cost.
    #[test]
    fn disk_replaces_targets_inside_the_radius() {
        let (terrain, reg) = decorator();
        let mut d = Decorator::new(&terrain, &reg, 42).expect("decorator");
        d.ensure_chunk(0, 0);
        let v: Value = serde_json::from_str(
            r#"{
                "type": "minecraft:disk",
                "half_height": 1,
                "radius": {"type": "minecraft:uniform", "min_inclusive": 3, "max_inclusive": 3},
                "state_provider": {"id": "minecraft:clay"},
                "target": {
                    "type": "minecraft:matching_blocks",
                    "blocks": ["minecraft:dirt", "minecraft:grass_block"]}
            }"#,
        )
        .unwrap();
        // A dirt column band at mid depth: plant a 7x7 dirt pad under
        // stone so the disk has bounded targets.
        let (ox, oy, oz) = (8, 30, 8);
        for x in ox - 4..=ox + 4 {
            for z in oz - 4..=oz + 4 {
                d.set_block(x, oy, z, d.state_id_of("minecraft:dirt", "").unwrap());
            }
        }
        let mut rng = DecorRng::new();
        let deco = rng.decoration_seed(42, 0, 0);
        rng.set_feature_seed(deco, 3, 6);
        let before = rng.words;
        run_disk(&mut d, &v, &mut rng, ox, oy, oz);
        assert_eq!(rng.words - before, 1, "the radius draw is the only draw");
        let mut clay = 0usize;
        let mut outside = 0usize;
        for x in ox - 4..=ox + 4 {
            for z in oz - 4..=oz + 4 {
                for y in oy - 2..=oy + 2 {
                    if d.block_name(d.block(x, y, z)) == "minecraft:clay" {
                        clay += 1;
                        let dx = x - ox;
                        let dz = z - oz;
                        if dx * dx + dz * dz > 9 {
                            outside += 1;
                        }
                    }
                }
            }
        }
        assert!(clay > 0, "the disk replaced dirt with clay");
        assert_eq!(outside, 0, "no write outside the radius circle");
    }

    /// The rule-based provider: the air-above cell takes grass, the water
    /// or solid cover takes the fallback.
    #[test]
    fn disk_rule_provider_follows_the_cover_above() {
        let (terrain, reg) = decorator();
        let mut d = Decorator::new(&terrain, &reg, 42).expect("decorator");
        d.ensure_chunk(0, 0);
        let v: Value = serde_json::from_str(
            r#"{
                "type": "minecraft:disk",
                "half_height": 0,
                "radius": {"type": "minecraft:uniform", "min_inclusive": 2, "max_inclusive": 2},
                "state_provider": {
                    "type": "minecraft:rule_based",
                    "fallback": {"id": "minecraft:dirt"},
                    "rules": [{
                        "if_true": {
                            "type": "minecraft:not",
                            "predicate": {"type": "minecraft:any_of", "predicates": [
                                {"type": "minecraft:solid", "offset": [0, 1, 0]},
                                {"type": "minecraft:matching_fluids",
                                 "fluids": "minecraft:water", "offset": [0, 1, 0]}]}},
                        "then": {"id": "minecraft:grass_block", "properties": {"snowy": "false"}}}]
                },
                "target": {
                    "type": "minecraft:matching_blocks",
                    "blocks": ["minecraft:dirt"]}
            }"#,
        )
        .unwrap();
        let (ox, oy, oz) = (8, 30, 8);
        let dirt = d.state_id_of("minecraft:dirt", "").unwrap();
        for x in ox - 3..=ox + 3 {
            for z in oz - 3..=oz + 3 {
                d.set_block(x, oy, z, dirt);
                d.set_block(x, oy + 1, z, 0);
            }
        }
        // One covered column and one wet column.
        d.set_block(
            ox - 1,
            oy + 1,
            oz,
            d.state_id_of("minecraft:stone", "").unwrap(),
        );
        d.set_block(
            ox + 1,
            oy + 1,
            oz,
            d.state_id_of("minecraft:water", "").unwrap(),
        );
        let mut rng = DecorRng::new();
        let deco = rng.decoration_seed(42, 0, 0);
        rng.set_feature_seed(deco, 4, 6);
        run_disk(&mut d, &v, &mut rng, ox, oy, oz);
        assert_eq!(
            d.block_name(d.block(ox, oy, oz)),
            "minecraft:grass_block",
            "open sky takes grass"
        );
        assert_eq!(
            d.block_name(d.block(ox - 1, oy, oz)),
            "minecraft:dirt",
            "solid cover takes the fallback"
        );
        assert_eq!(
            d.block_name(d.block(ox + 1, oy, oz)),
            "minecraft:dirt",
            "water cover takes the fallback"
        );
    }
}
