//! Config-driven density terrain: the pinned worldgen configs (noise
//! settings, density functions, material rules) interpreted as a
//! sampled 3D density field, an aquifer grid, and a per-column surface
//! pass, seeded to match the reference chain.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::biome::{BiomeTable, AXIS_COUNT};
use crate::noise::{world_positional, Perlin, Positional, Xoroshiro};
use crate::registry::BlockRegistry;

/// Blocks per chunk edge.
const EDGE: i32 = 16;
/// Columns per chunk.
const COLUMNS: usize = 256;
/// Below this depth the global fluid column turns to lava.
const GLOBAL_LAVA_LEVEL: i32 = -54;
/// Returned instead of a fluid when an aquifer cell has no reachable
/// surface level.
const WAY_BELOW: i32 = i32::MIN;

// ---------------------------------------------------------------------------
// Sampling helpers matching the reference float granularity.
// ---------------------------------------------------------------------------

fn floor_mod(a: i32, m: i32) -> i32 {
    ((a % m) + m) % m
}

fn lerp(alpha: f32, a: f32, b: f32) -> f32 {
    a + alpha * (b - a)
}

fn smoothstep(x: f32) -> f32 {
    x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
}

fn grad_dot(hash: i32, x: f32, y: f32, z: f32) -> f32 {
    const GRADIENTS: [[i32; 3]; 16] = [
        [1, 1, 0],
        [-1, 1, 0],
        [1, -1, 0],
        [-1, -1, 0],
        [1, 0, 1],
        [-1, 0, 1],
        [1, 0, -1],
        [-1, 0, -1],
        [0, 1, 1],
        [0, -1, 1],
        [0, 1, -1],
        [0, -1, -1],
        [1, 1, 0],
        [0, -1, 1],
        [-1, 1, 0],
        [0, -1, -1],
    ];
    let g = &GRADIENTS[(hash & 0xF) as usize];
    g[0] as f32 * x + g[1] as f32 * y + g[2] as f32 * z
}

/// Keeps far coordinates inside the band where float fractions still
/// resolve.
fn wrap(x: f64) -> f64 {
    const HALF_ROUND_OFF: f64 = 16777215.999999998;
    if (-HALF_ROUND_OFF..HALF_ROUND_OFF).contains(&x) {
        return x;
    }
    x - (x / 33554432.0 + 0.5).floor() * 33554432.0
}

fn map(value: f64, from_min: f64, from_max: f64, to_min: f64, to_max: f64) -> f64 {
    let t = ((value - from_min) / (from_max - from_min)).clamp(0.0, 1.0);
    to_min + t * (to_max - to_min)
}

fn unclamped_map(value: f64, from_min: f64, from_max: f64, to_min: f64, to_max: f64) -> f64 {
    let t = (value - from_min) / (from_max - from_min);
    to_min + t * (to_max - to_min)
}

fn quantize(value: f64, step: i32) -> i32 {
    (value / f64::from(step)).floor() as i32 * step
}

/// The float draw of the rotate-xor stream: the high 24 bits.
fn next_f32(rng: &mut Xoroshiro) -> f32 {
    ((rng.next_long() as u64) >> 40) as f32 * (1.0 / (1u32 << 24) as f32)
}

// ---------------------------------------------------------------------------
// Normalized octave stacks from pinned noise configs.
// ---------------------------------------------------------------------------

/// Pinned noise parameters: octaves double in frequency from the base.
struct NoiseParams {
    base_octave: i32,
    octave_count: usize,
    base_amplitude: f64,
    modifiers: Vec<f64>,
}

impl NoiseParams {
    fn parse(v: &Value) -> Result<NoiseParams> {
        let modifiers = v
            .get("amplitude_modifiers")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default();
        Ok(NoiseParams {
            base_octave: v
                .get("base_octave")
                .and_then(Value::as_i64)
                .context("noise base_octave")? as i32,
            octave_count: v.get("octave_count").and_then(Value::as_i64).unwrap_or(1) as usize,
            base_amplitude: v
                .get("base_amplitude")
                .and_then(Value::as_f64)
                .unwrap_or(1.0),
            modifiers,
        })
    }
}

struct StackLayer {
    noise: Perlin,
    frequency: f64,
    amplitude: f32,
}

/// A normalized octave stack: every octave contributes two lattice noise
/// layers scaled so the combined deviation reaches a third of the summed
/// amplitude.
pub struct StackNoise {
    layers: Vec<StackLayer>,
}

impl StackNoise {
    fn new(params: &NoiseParams, rng: &mut Xoroshiro) -> StackNoise {
        let first = rng.fork_positional();
        let second = rng.fork_positional();
        let mut frequency = 2f64.powi(params.base_octave);
        let mut amplitude = params.base_amplitude * 2f64.powi(params.octave_count as i32 - 1)
            / (2f64.powi(params.octave_count as i32) - 1.0);
        let mut octaves: Vec<(i32, f64, f64)> = Vec::new();
        for i in 0..params.octave_count {
            let modifier = params.modifiers.get(i).copied().unwrap_or(1.0);
            if modifier != 0.0 {
                octaves.push((
                    params.base_octave + i as i32,
                    frequency,
                    amplitude * modifier,
                ));
            }
            frequency *= 2.0;
            amplitude *= 0.5;
        }
        let target: f64 = octaves.iter().map(|(_, _, a)| a.abs()).sum();
        let variance: f64 = octaves
            .iter()
            .map(|(_, _, a)| {
                let deviation = 0.2702247831245211 * a.abs();
                deviation * deviation
            })
            .sum();
        let deviation = variance.sqrt();
        let normalization = if deviation == 0.0 {
            0.0
        } else {
            (target / 3.0) / (deviation * 2f64.sqrt())
        };
        let mut layers = Vec::with_capacity(octaves.len() * 2);
        for (index, frequency, amplitude) in octaves {
            let name = format!("octave_{index}");
            let value_factor = (normalization * amplitude) as f32;
            layers.push(StackLayer {
                noise: Perlin::new(&mut first.from_name(&name)),
                frequency,
                amplitude: value_factor,
            });
            layers.push(StackLayer {
                noise: Perlin::new(&mut second.from_name(&name)),
                frequency: frequency * 1.0181268882175227,
                amplitude: value_factor,
            });
        }
        StackNoise { layers }
    }

    fn get(&self, x: f64, y: f64, z: f64) -> f32 {
        let mut value = 0f32;
        for layer in &self.layers {
            value += layer.amplitude
                * layer.noise.sample(
                    x * layer.frequency,
                    y * layer.frequency,
                    z * layer.frequency,
                );
        }
        value
    }
}

// ---------------------------------------------------------------------------
// The smeared lattice and the legacy blended field.
// ---------------------------------------------------------------------------

/// A lattice noise whose y axis snaps to a coarse fudge grid, so sampled
/// slabs stretch vertically instead of grading.
struct SmearedLattice {
    ox: f64,
    oy: f64,
    oz: f64,
    perms: [u8; 256],
    fudge_y: f64,
}

impl SmearedLattice {
    fn new(rng: &mut Xoroshiro, fudge_y: f64) -> SmearedLattice {
        let ox = rng.next_f64() * 256.0;
        let oy = rng.next_f64() * 256.0;
        let oz = rng.next_f64() * 256.0;
        let mut perms = std::array::from_fn(|i| i as u8);
        for i in 0..256usize {
            let offset = rng.next_int((256 - i) as i32) as usize;
            perms.swap(i, i + offset);
        }
        SmearedLattice {
            ox,
            oy,
            oz,
            perms,
            fudge_y,
        }
    }

    fn permute(&self, x: i32) -> i32 {
        self.perms[(x & 0xFF) as usize] as i32
    }

    fn get(&self, x_in: f64, y_in: f64, z_in: f64) -> f32 {
        let x = wrap(x_in) + self.ox;
        let y = wrap(y_in) + self.oy;
        let z = wrap(z_in) + self.oz;
        let fx = x.floor() as i32;
        let fy = y.floor() as i32;
        let fz = z.floor() as i32;
        let rx = (x - fx as f64) as f32;
        let ry = y - fy as f64;
        let rz = (z - fz as f64) as f32;
        // The y fractions snap toward the fudge grid before the corner
        // lerp; the smoothstep weight keeps the unfudged fraction.
        let fudge_limit = if y_in >= 0.0 && y_in < ry { y_in } else { ry };
        let fudge = (fudge_limit / self.fudge_y + f64::from(1.0e-7f32)).floor() as i64 as f64
            * self.fudge_y;
        let fudged = (ry - fudge) as f32;
        self.sample_and_lerp(fx, fy, fz, rx, fudged, rz, ry as f32)
    }

    #[allow(clippy::too_many_arguments)]
    fn sample_and_lerp(
        &self,
        x: i32,
        y: i32,
        z: i32,
        rx: f32,
        ry: f32,
        rz: f32,
        original_ry: f32,
    ) -> f32 {
        let lerp2 = |a1: f32, a2: f32, x00: f32, x10: f32, x01: f32, x11: f32| {
            lerp(a2, lerp(a1, x00, x10), lerp(a1, x01, x11))
        };
        let x0 = self.permute(x);
        let x1 = self.permute(x + 1);
        let xy00 = self.permute(x0 + y);
        let xy01 = self.permute(x0 + y + 1);
        let xy10 = self.permute(x1 + y);
        let xy11 = self.permute(x1 + y + 1);
        let d000 = grad_dot(self.permute(xy00 + z), rx, ry, rz);
        let d100 = grad_dot(self.permute(xy10 + z), rx - 1.0, ry, rz);
        let d010 = grad_dot(self.permute(xy01 + z), rx, ry - 1.0, rz);
        let d110 = grad_dot(self.permute(xy11 + z), rx - 1.0, ry - 1.0, rz);
        let d001 = grad_dot(self.permute(xy00 + z + 1), rx, ry, rz - 1.0);
        let d101 = grad_dot(self.permute(xy10 + z + 1), rx - 1.0, ry, rz - 1.0);
        let d011 = grad_dot(self.permute(xy01 + z + 1), rx, ry - 1.0, rz - 1.0);
        let d111 = grad_dot(self.permute(xy11 + z + 1), rx - 1.0, ry - 1.0, rz - 1.0);
        lerp(
            smoothstep(rz),
            lerp2(
                smoothstep(rx),
                smoothstep(original_ry),
                d000,
                d100,
                d010,
                d110,
            ),
            lerp2(
                smoothstep(rx),
                smoothstep(original_ry),
                d001,
                d101,
                d011,
                d111,
            ),
        )
    }
}

struct FbmLayer {
    noise: SmearedLattice,
    frequency: f64,
    amplitude: f32,
}

/// A frequency ladder of smeared lattices.
struct FbmStack {
    layers: Vec<FbmLayer>,
}

impl FbmStack {
    fn get(&self, x: f64, y: f64, z: f64) -> f32 {
        let mut value = 0f32;
        for layer in &self.layers {
            value += layer.amplitude
                * layer.noise.get(
                    x * layer.frequency,
                    y * layer.frequency,
                    z * layer.frequency,
                );
        }
        value
    }
}

fn fbm(rng: &mut Xoroshiro, first_octave: i32, smear: f64, limit: f64) -> FbmStack {
    let octaves = (-first_octave + 1) as usize;
    let mut factor = 1.0f64;
    let mut value_factor = limit / (2f64.powi(octaves as i32) - 1.0);
    let mut layers = Vec::with_capacity(octaves);
    for _ in 0..octaves {
        layers.push(FbmLayer {
            noise: SmearedLattice::new(rng, smear * factor),
            frequency: factor,
            amplitude: value_factor as f32,
        });
        factor /= 2.0;
        value_factor *= 2.0;
    }
    FbmStack { layers }
}

/// The legacy blended field: two limit fields cross-faded by a main
/// field, all on the smeared lattice.
struct BlendedField {
    min: FbmStack,
    max: FbmStack,
    main: FbmStack,
    xz_factor: f64,
    y_factor: f64,
    xz_scale: f64,
    y_scale: f64,
}

impl BlendedField {
    fn new(
        rng: &mut Xoroshiro,
        xz_scale: f64,
        y_scale: f64,
        xz_factor: f64,
        y_factor: f64,
        smear: f64,
    ) -> BlendedField {
        let y_mult = 684.412 * y_scale;
        let limit_smear = y_mult * smear;
        let main_smear = limit_smear / y_factor;
        BlendedField {
            min: fbm(rng, -15, limit_smear, 0.9999847412109375),
            max: fbm(rng, -15, limit_smear, 0.9999847412109375),
            main: fbm(rng, -7, main_smear, 12.75),
            xz_factor,
            y_factor,
            xz_scale,
            y_scale,
        }
    }

    fn sample(&self, x: f64, y: f64, z: f64) -> f32 {
        let xz_mult = 684.412 * self.xz_scale;
        let y_mult = 684.412 * self.y_scale;
        let main = self.main.get(
            x * (xz_mult / self.xz_factor),
            y * (y_mult / self.y_factor),
            z * (xz_mult / self.xz_factor),
        );
        let alpha = (main + 0.5).clamp(0.0, 1.0);
        if alpha == 0.0 {
            return self.min.get(x * xz_mult, y * y_mult, z * xz_mult);
        }
        if alpha == 1.0 {
            return self.max.get(x * xz_mult, y * y_mult, z * xz_mult);
        }
        let lo = self.min.get(x * xz_mult, y * y_mult, z * xz_mult);
        let hi = self.max.get(x * xz_mult, y * y_mult, z * xz_mult);
        lerp(alpha, lo, hi)
    }
}

// ---------------------------------------------------------------------------
// Density function graph.
// ---------------------------------------------------------------------------

type NodeRef = usize;

#[derive(Clone)]
enum SplineValue {
    Fixed(f32),
    Nested(Box<SplineNode>),
}

#[derive(Clone)]
struct SplinePoint {
    location: f32,
    derivative: f32,
    value: SplineValue,
}

#[derive(Clone)]
struct SplineNode {
    coordinate: NodeRef,
    points: Vec<SplinePoint>,
}

enum UnOp {
    Abs,
    Square,
    Cube,
    Negate,
    Squeeze,
    HalfNegative,
    QuarterNegative,
}

enum BinOp {
    Add,
    Mul,
    Min,
    Max,
    Sub,
    Div,
}

enum Node {
    Constant(f32),
    Stack {
        stack: Arc<StackNoise>,
        xz_scale: f64,
        y_scale: f64,
        shift_x: Option<NodeRef>,
        shift_y: Option<NodeRef>,
        shift_z: Option<NodeRef>,
    },
    ShiftA(Arc<StackNoise>),
    ShiftB(Arc<StackNoise>),
    Gradient {
        axis: u8,
        from_coord: i32,
        to_coord: i32,
        from_value: f32,
        to_value: f32,
    },
    Unary {
        op: UnOp,
        input: NodeRef,
    },
    Binary {
        op: BinOp,
        left: NodeRef,
        right: NodeRef,
    },
    Clamp {
        input: NodeRef,
        min: f32,
        max: f32,
    },
    Lerp {
        alpha: NodeRef,
        first: NodeRef,
        second: NodeRef,
    },
    RangeChoice {
        input: NodeRef,
        min: f32,
        max: f32,
        in_range: NodeRef,
        out_of_range: NodeRef,
    },
    IntervalSelect {
        input: NodeRef,
        thresholds: Vec<f32>,
        functions: Vec<NodeRef>,
    },
    Spline(SplineNode),
    Interpolated {
        input: NodeRef,
        cell_xz: i32,
        cell_y: i32,
    },
    Cache {
        input: NodeRef,
    },
    FindTopSurface {
        density: NodeRef,
        upper: NodeRef,
        lower: i32,
        cell_height: i32,
    },
    Blended(BlendedField),
}

struct Graph {
    nodes: Vec<Node>,
    /// Whether each node's value varies with the sampled y.
    uses_y: Vec<bool>,
}

impl Graph {
    fn finish(&mut self) {
        let mut uses = vec![false; self.nodes.len()];
        for i in 0..self.nodes.len() {
            uses[i] = match &self.nodes[i] {
                Node::Constant(_) | Node::ShiftA(_) | Node::ShiftB(_) => false,
                Node::Stack {
                    y_scale,
                    shift_x,
                    shift_y,
                    shift_z,
                    ..
                } => {
                    *y_scale != 0.0
                        || shift_x.is_some_and(|n| uses[n])
                        || shift_y.is_some_and(|n| uses[n])
                        || shift_z.is_some_and(|n| uses[n])
                }
                Node::Gradient { axis, .. } => *axis == 1,
                Node::Unary { input, .. } | Node::Cache { input } => uses[*input],
                Node::Binary { left, right, .. } => uses[*left] || uses[*right],
                Node::Clamp { input, .. } => uses[*input],
                Node::Lerp {
                    alpha,
                    first,
                    second,
                } => uses[*alpha] || uses[*first] || uses[*second],
                Node::RangeChoice {
                    input,
                    in_range,
                    out_of_range,
                    ..
                } => uses[*input] || uses[*in_range] || uses[*out_of_range],
                Node::IntervalSelect {
                    input, functions, ..
                } => uses[*input] || functions.iter().any(|n| uses[*n]),
                Node::Spline(s) => spline_uses_y(s, &uses),
                Node::Interpolated { input, .. } => uses[*input],
                Node::FindTopSurface { .. } => false,
                Node::Blended(_) => true,
            };
        }
        self.uses_y = uses;
    }
}

fn spline_uses_y(s: &SplineNode, uses: &[bool]) -> bool {
    let mut any = uses[s.coordinate];
    for point in &s.points {
        any |= match &point.value {
            SplineValue::Fixed(_) => false,
            SplineValue::Nested(n) => spline_uses_y(n, uses),
        };
    }
    any
}

// ---------------------------------------------------------------------------
// Config loading.
// ---------------------------------------------------------------------------

fn field_f64(v: &Value, key: &str) -> Result<f64> {
    v.get(key)
        .and_then(Value::as_f64)
        .with_context(|| format!("field {key}"))
}

fn field_i64(v: &Value, key: &str) -> Result<i64> {
    v.get(key)
        .and_then(Value::as_i64)
        .with_context(|| format!("field {key}"))
}

struct Loader<'a> {
    pins: &'a Path,
    world: Positional,
    registry: &'a BlockRegistry,
    min_y: i32,
    height: i32,
    graph: Graph,
    stacks: HashMap<String, Arc<StackNoise>>,
    functions: HashMap<String, NodeRef>,
    /// Position-keyed random factories named by the surface rules.
    gradient_factories: HashMap<String, Positional>,
}

impl<'a> Loader<'a> {
    fn new(
        pins: &'a Path,
        seed: i64,
        registry: &'a BlockRegistry,
        min_y: i32,
        height: i32,
    ) -> Self {
        Loader {
            pins,
            world: world_positional(seed),
            registry,
            min_y,
            height,
            graph: Graph {
                nodes: Vec::new(),
                uses_y: Vec::new(),
            },
            stacks: HashMap::new(),
            functions: HashMap::new(),
            gradient_factories: HashMap::new(),
        }
    }

    fn push(&mut self, node: Node) -> NodeRef {
        self.graph.nodes.push(node);
        self.graph.nodes.len() - 1
    }

    fn read_json(&self, dir: &str, key: &str) -> Result<Value> {
        let path = self.pins.join(dir).join(format!("{key}.json"));
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
    }

    fn stack(&mut self, name: &str) -> Result<Arc<StackNoise>> {
        if let Some(stack) = self.stacks.get(name) {
            return Ok(Arc::clone(stack));
        }
        let key = name.strip_prefix("minecraft:").unwrap_or(name);
        let config = self.read_json("noise", key)?;
        let params = NoiseParams::parse(&config)?;
        let mut rng = self.world.from_name(name);
        let stack = Arc::new(StackNoise::new(&params, &mut rng));
        self.stacks.insert(name.to_string(), Arc::clone(&stack));
        Ok(stack)
    }

    fn function_id(&mut self, id: &str) -> Result<NodeRef> {
        if let Some(&node) = self.functions.get(id) {
            return Ok(node);
        }
        let key = id.strip_prefix("minecraft:").unwrap_or(id);
        let config = self.read_json("density_function", key)?;
        let node = self.function(&config)?;
        self.functions.insert(id.to_string(), node);
        Ok(node)
    }

    fn function(&mut self, v: &Value) -> Result<NodeRef> {
        match v {
            Value::Number(n) => {
                let value = n.as_f64().unwrap_or(0.0) as f32;
                Ok(self.push(Node::Constant(value)))
            }
            Value::String(s) => self.function_id(s),
            Value::Object(_) => self.op(v),
            _ => bail!("unsupported density function json"),
        }
    }

    fn op(&mut self, v: &Value) -> Result<NodeRef> {
        let kind = v.get("type").and_then(Value::as_str).context("op type")?;
        let node = match kind {
            "minecraft:noise" => {
                let name = v.get("noise").and_then(Value::as_str).context("noise id")?;
                let stack = self.stack(name)?;
                Node::Stack {
                    stack,
                    xz_scale: field_f64(v, "xz_scale")?,
                    y_scale: field_f64(v, "y_scale")?,
                    shift_x: self.opt_function(v, "shift_x")?,
                    shift_y: self.opt_function(v, "shift_y")?,
                    shift_z: self.opt_function(v, "shift_z")?,
                }
            }
            "minecraft:shift_a" | "minecraft:shift_b" => {
                let name = v.get("noise").and_then(Value::as_str).context("noise id")?;
                let stack = self.stack(name)?;
                if kind == "minecraft:shift_a" {
                    Node::ShiftA(stack)
                } else {
                    Node::ShiftB(stack)
                }
            }
            "minecraft:gradient" => Node::Gradient {
                axis: match v.get("axis").and_then(Value::as_str).context("axis")? {
                    "x" => 0,
                    "y" => 1,
                    _ => 2,
                },
                from_coord: field_i64(v, "from_coordinate")? as i32,
                to_coord: field_i64(v, "to_coordinate")? as i32,
                from_value: field_f64(v, "from_value")? as f32,
                to_value: field_f64(v, "to_value")? as f32,
            },
            "minecraft:abs"
            | "minecraft:square"
            | "minecraft:cube"
            | "minecraft:negate"
            | "minecraft:squeeze"
            | "minecraft:half_negative"
            | "minecraft:quarter_negative" => {
                let op = match kind {
                    "minecraft:abs" => UnOp::Abs,
                    "minecraft:square" => UnOp::Square,
                    "minecraft:cube" => UnOp::Cube,
                    "minecraft:negate" => UnOp::Negate,
                    "minecraft:squeeze" => UnOp::Squeeze,
                    "minecraft:half_negative" => UnOp::HalfNegative,
                    _ => UnOp::QuarterNegative,
                };
                Node::Unary {
                    op,
                    input: self.field_node(v, "input")?,
                }
            }
            "minecraft:add" | "minecraft:mul" | "minecraft:min" | "minecraft:max"
            | "minecraft:sub" | "minecraft:div" => {
                let op = match kind {
                    "minecraft:add" => BinOp::Add,
                    "minecraft:mul" => BinOp::Mul,
                    "minecraft:min" => BinOp::Min,
                    "minecraft:max" => BinOp::Max,
                    "minecraft:sub" => BinOp::Sub,
                    _ => BinOp::Div,
                };
                Node::Binary {
                    op,
                    left: self.field_node(v, "left")?,
                    right: self.field_node(v, "right")?,
                }
            }
            "minecraft:clamp" => Node::Clamp {
                input: self.field_node(v, "input")?,
                min: field_f64(v, "min")? as f32,
                max: field_f64(v, "max")? as f32,
            },
            "minecraft:lerp" => Node::Lerp {
                alpha: self.field_node(v, "alpha")?,
                first: self.field_node(v, "first")?,
                second: self.field_node(v, "second")?,
            },
            "minecraft:range_choice" => Node::RangeChoice {
                input: self.field_node(v, "input")?,
                min: field_f64(v, "min_inclusive")? as f32,
                max: field_f64(v, "max_exclusive")? as f32,
                in_range: self.field_node(v, "when_in_range")?,
                out_of_range: self.field_node(v, "when_out_of_range")?,
            },
            "minecraft:interval_select" => {
                let thresholds = v
                    .get("thresholds")
                    .and_then(Value::as_array)
                    .context("thresholds")?
                    .iter()
                    .map(|t| t.as_f64().unwrap_or(0.0) as f32)
                    .collect();
                let functions = v
                    .get("functions")
                    .and_then(Value::as_array)
                    .context("functions")?
                    .iter()
                    .map(|f| self.function(f))
                    .collect::<Result<Vec<_>>>()?;
                Node::IntervalSelect {
                    input: self.field_node(v, "input")?,
                    thresholds,
                    functions,
                }
            }
            "minecraft:spline" => {
                let spline = v.get("spline").context("spline body")?;
                Node::Spline(self.spline(spline)?)
            }
            "minecraft:interpolated" => Node::Interpolated {
                input: self.field_node(v, "input")?,
                cell_xz: field_i64(v, "cell_size_xz")? as i32,
                cell_y: field_i64(v, "cell_size_y")? as i32,
            },
            "minecraft:find_top_surface" => Node::FindTopSurface {
                density: self.field_node(v, "density")?,
                upper: self.field_node(v, "upper_bound")?,
                lower: field_i64(v, "lower_bound")? as i32,
                cell_height: field_i64(v, "cell_height")? as i32,
            },
            "minecraft:old_blended_noise" => {
                let mut rng = self.world.from_name("minecraft:terrain");
                Node::Blended(BlendedField::new(
                    &mut rng,
                    field_f64(v, "xz_scale")?,
                    field_f64(v, "y_scale")?,
                    field_f64(v, "xz_factor")?,
                    field_f64(v, "y_factor")?,
                    field_f64(v, "smear_scale_multiplier")?,
                ))
            }
            "minecraft:cache" => Node::Cache {
                input: self.field_node(v, "input")?,
            },
            "minecraft:blend_density" => return self.field_node(v, "input"),
            "minecraft:beardifier" => Node::Constant(0.0),
            "minecraft:blend_alpha" => Node::Constant(1.0),
            "minecraft:blend_offset" => Node::Constant(0.0),
            other => bail!("unsupported density op {other}"),
        };
        Ok(self.push(node))
    }

    fn field_node(&mut self, v: &Value, key: &str) -> Result<NodeRef> {
        let child = v.get(key).context(format!("missing {key}"))?;
        self.function(child)
    }

    fn opt_function(&mut self, v: &Value, key: &str) -> Result<Option<NodeRef>> {
        match v.get(key) {
            Some(child) => self.function(child).map(Some),
            None => Ok(None),
        }
    }

    fn spline(&mut self, v: &Value) -> Result<SplineNode> {
        let coordinate = self.field_node(v, "coordinate")?;
        let mut points = Vec::new();
        for point in v
            .get("points")
            .and_then(Value::as_array)
            .context("points")?
        {
            let value = match point.get("value") {
                Some(Value::Number(n)) => SplineValue::Fixed(n.as_f64().unwrap_or(0.0) as f32),
                Some(other) => SplineValue::Nested(Box::new(self.spline(other)?)),
                None => bail!("spline point without value"),
            };
            points.push(SplinePoint {
                location: field_f64(point, "location")? as f32,
                derivative: field_f64(point, "derivative")? as f32,
                value,
            });
        }
        Ok(SplineNode { coordinate, points })
    }
}

// ---------------------------------------------------------------------------
// Per-chunk evaluation.
// ---------------------------------------------------------------------------

/// Lazily filled corner grid of one interpolated node.
struct Grid {
    nx: usize,
    ny: usize,
    nz: usize,
    values: Vec<f32>,
}

impl Grid {
    fn new(cell_xz: i32, cell_y: i32, height: i32) -> Grid {
        let nx = (EDGE / cell_xz + 1) as usize;
        let ny = (height / cell_y + 1) as usize;
        Grid {
            nx,
            ny,
            nz: nx,
            values: vec![f32::NAN; nx * ny * nx],
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn index(
        &self,
        x: i32,
        y: i32,
        z: i32,
        start_x: i32,
        start_z: i32,
        min_y: i32,
        cell_xz: i32,
        cell_y: i32,
    ) -> Option<usize> {
        let rx = x - start_x;
        let ry = y - min_y;
        let rz = z - start_z;
        if rx < 0 || rz < 0 || ry < 0 {
            return None;
        }
        if rx % cell_xz != 0 || rz % cell_xz != 0 || ry % cell_y != 0 {
            return None;
        }
        let (ix, iy, iz) = (
            (rx / cell_xz) as usize,
            (ry / cell_y) as usize,
            (rz / cell_xz) as usize,
        );
        if ix >= self.nx || iy >= self.ny || iz >= self.nz {
            return None;
        }
        Some((iy * self.nz + iz) * self.nx + ix)
    }
}

/// Evaluation state for one chunk: corner grids and node memos.
struct ChunkCtx {
    start_x: i32,
    start_z: i32,
    min_y: i32,
    height: i32,
    grids: HashMap<NodeRef, Grid>,
    memo2: HashMap<(NodeRef, i32, i32), f32>,
    memo3: HashMap<(NodeRef, i32, i32, i32), f32>,
}

impl ChunkCtx {
    fn new(start_x: i32, start_z: i32, min_y: i32, height: i32) -> ChunkCtx {
        ChunkCtx {
            start_x,
            start_z,
            min_y,
            height,
            grids: HashMap::new(),
            memo2: HashMap::new(),
            memo3: HashMap::new(),
        }
    }
}

/// Corner value of an interpolated node, drawn from the chunk grid.
#[allow(clippy::too_many_arguments)]
fn corner(
    graph: &Graph,
    ctx: &mut ChunkCtx,
    node: NodeRef,
    input: NodeRef,
    cell_xz: i32,
    cell_y: i32,
    x: i32,
    y: i32,
    z: i32,
) -> f32 {
    if !ctx.grids.contains_key(&node) {
        ctx.grids
            .insert(node, Grid::new(cell_xz, cell_y, ctx.height));
    }
    let idx = {
        let grid = &ctx.grids[&node];
        grid.index(
            x,
            y,
            z,
            ctx.start_x,
            ctx.start_z,
            ctx.min_y,
            cell_xz,
            cell_y,
        )
        .map(|i| (i, grid.values[i]))
    };
    match idx {
        Some((_, v)) if !v.is_nan() => v,
        Some((i, _)) => {
            let v = sample_node(graph, ctx, input, x, y, z);
            if let Some(grid) = ctx.grids.get_mut(&node) {
                grid.values[i] = v;
            }
            v
        }
        None => sample_node(graph, ctx, input, x, y, z),
    }
}

fn sample_node(graph: &Graph, ctx: &mut ChunkCtx, node: NodeRef, x: i32, y: i32, z: i32) -> f32 {
    match &graph.nodes[node] {
        Node::Constant(value) => *value,
        Node::Stack {
            stack,
            xz_scale,
            y_scale,
            shift_x,
            shift_y,
            shift_z,
        } => {
            let sx = shift_x
                .map(|n| f64::from(sample_node(graph, ctx, n, x, y, z)))
                .unwrap_or(0.0);
            let sy = shift_y
                .map(|n| f64::from(sample_node(graph, ctx, n, x, y, z)))
                .unwrap_or(0.0);
            let sz = shift_z
                .map(|n| f64::from(sample_node(graph, ctx, n, x, y, z)))
                .unwrap_or(0.0);
            stack.get(
                f64::from(x) * xz_scale + sx,
                f64::from(y) * y_scale + sy,
                f64::from(z) * xz_scale + sz,
            )
        }
        Node::ShiftA(stack) => stack.get(f64::from(x) * 0.25, 0.0, f64::from(z) * 0.25) * 4.0,
        Node::ShiftB(stack) => stack.get(f64::from(z) * 0.25, f64::from(x) * 0.25, 0.0) * 4.0,
        Node::Gradient {
            axis,
            from_coord,
            to_coord,
            from_value,
            to_value,
        } => {
            let coordinate = match axis {
                0 => x,
                1 => y,
                _ => z,
            };
            let min = (*from_coord).min(*to_coord);
            let max = (*from_coord).max(*to_coord);
            let relative = coordinate.clamp(min, max) - from_coord;
            let factor = (to_value - from_value) / (*to_coord - *from_coord) as f32;
            from_value + relative as f32 * factor
        }
        Node::Unary { op, input } => {
            let v = sample_node(graph, ctx, *input, x, y, z);
            match op {
                UnOp::Abs => v.abs(),
                UnOp::Square => v * v,
                UnOp::Cube => v * v * v,
                UnOp::Negate => -v,
                UnOp::Squeeze => {
                    let c = v.clamp(-1.0, 1.0);
                    c / 2.0 - c * c * c / 24.0
                }
                UnOp::HalfNegative => {
                    if v > 0.0 {
                        v
                    } else {
                        v * 0.5
                    }
                }
                UnOp::QuarterNegative => {
                    if v > 0.0 {
                        v
                    } else {
                        v * 0.25
                    }
                }
            }
        }
        Node::Binary { op, left, right } => {
            let a = sample_node(graph, ctx, *left, x, y, z);
            let b = sample_node(graph, ctx, *right, x, y, z);
            match op {
                BinOp::Add => a + b,
                BinOp::Mul => a * b,
                BinOp::Min => a.min(b),
                BinOp::Max => a.max(b),
                BinOp::Sub => a - b,
                BinOp::Div => a / b,
            }
        }
        Node::Clamp { input, min, max } => {
            sample_node(graph, ctx, *input, x, y, z).clamp(*min, *max)
        }
        Node::Lerp {
            alpha,
            first,
            second,
        } => {
            let a = sample_node(graph, ctx, *alpha, x, y, z);
            if a == 0.0 {
                sample_node(graph, ctx, *first, x, y, z)
            } else if a == 1.0 {
                sample_node(graph, ctx, *second, x, y, z)
            } else {
                let lo = sample_node(graph, ctx, *first, x, y, z);
                let hi = sample_node(graph, ctx, *second, x, y, z);
                lerp(a, lo, hi)
            }
        }
        Node::RangeChoice {
            input,
            min,
            max,
            in_range,
            out_of_range,
        } => {
            let v = sample_node(graph, ctx, *input, x, y, z);
            if v >= *min && v < *max {
                sample_node(graph, ctx, *in_range, x, y, z)
            } else {
                sample_node(graph, ctx, *out_of_range, x, y, z)
            }
        }
        Node::IntervalSelect {
            input,
            thresholds,
            functions,
        } => {
            let v = sample_node(graph, ctx, *input, x, y, z);
            let mut index = 0;
            while index < thresholds.len() && v >= thresholds[index] {
                index += 1;
            }
            let pick = index.min(functions.len() - 1);
            sample_node(graph, ctx, functions[pick], x, y, z)
        }
        Node::Spline(s) => eval_spline(graph, ctx, s, x, y, z),
        Node::Interpolated {
            input,
            cell_xz,
            cell_y,
        } => {
            let ix = floor_mod(x, *cell_xz);
            let iy = floor_mod(y, *cell_y);
            let iz = floor_mod(z, *cell_xz);
            if ix == 0 && iy == 0 && iz == 0 {
                return sample_node(graph, ctx, *input, x, y, z);
            }
            let x0 = x - ix;
            let y0 = y - iy;
            let z0 = z - iz;
            let v000 = corner(graph, ctx, node, *input, *cell_xz, *cell_y, x0, y0, z0);
            let v100 = corner(
                graph,
                ctx,
                node,
                *input,
                *cell_xz,
                *cell_y,
                x0 + cell_xz,
                y0,
                z0,
            );
            let v010 = corner(
                graph,
                ctx,
                node,
                *input,
                *cell_xz,
                *cell_y,
                x0,
                y0 + cell_y,
                z0,
            );
            let v110 = corner(
                graph,
                ctx,
                node,
                *input,
                *cell_xz,
                *cell_y,
                x0 + cell_xz,
                y0 + cell_y,
                z0,
            );
            let v001 = corner(
                graph,
                ctx,
                node,
                *input,
                *cell_xz,
                *cell_y,
                x0,
                y0,
                z0 + cell_xz,
            );
            let v101 = corner(
                graph,
                ctx,
                node,
                *input,
                *cell_xz,
                *cell_y,
                x0 + cell_xz,
                y0,
                z0 + cell_xz,
            );
            let v011 = corner(
                graph,
                ctx,
                node,
                *input,
                *cell_xz,
                *cell_y,
                x0,
                y0 + cell_y,
                z0 + cell_xz,
            );
            let v111 = corner(
                graph,
                ctx,
                node,
                *input,
                *cell_xz,
                *cell_y,
                x0 + cell_xz,
                y0 + cell_y,
                z0 + cell_xz,
            );
            let lerp2 = |a1: f32, a2: f32, x00: f32, x10: f32, x01: f32, x11: f32| {
                lerp(a2, lerp(a1, x00, x10), lerp(a1, x01, x11))
            };
            lerp(
                iz as f32 / *cell_xz as f32,
                lerp2(
                    ix as f32 / *cell_xz as f32,
                    iy as f32 / *cell_y as f32,
                    v000,
                    v100,
                    v010,
                    v110,
                ),
                lerp2(
                    ix as f32 / *cell_xz as f32,
                    iy as f32 / *cell_y as f32,
                    v001,
                    v101,
                    v011,
                    v111,
                ),
            )
        }
        Node::Cache { input } => {
            if graph.uses_y[*input] {
                if let Some(v) = ctx.memo3.get(&(node, x, y, z)) {
                    return *v;
                }
                let v = sample_node(graph, ctx, *input, x, y, z);
                ctx.memo3.insert((node, x, y, z), v);
                v
            } else {
                if let Some(v) = ctx.memo2.get(&(node, x, z)) {
                    return *v;
                }
                let v = sample_node(graph, ctx, *input, x, y, z);
                ctx.memo2.insert((node, x, z), v);
                v
            }
        }
        Node::FindTopSurface {
            density,
            upper,
            lower,
            cell_height,
        } => {
            let upper_value = sample_node(graph, ctx, *upper, x, 0, z);
            let top = (upper_value / *cell_height as f32).floor() as i32 * cell_height;
            if top <= *lower {
                return *lower as f32;
            }
            let mut probe = top;
            while probe >= *lower {
                if sample_node(graph, ctx, *density, x, probe, z) > 0.0 {
                    return probe as f32;
                }
                probe -= cell_height;
            }
            *lower as f32
        }
        Node::Blended(field) => field.sample(f64::from(x), f64::from(y), f64::from(z)),
    }
}

fn eval_spline(graph: &Graph, ctx: &mut ChunkCtx, s: &SplineNode, x: i32, y: i32, z: i32) -> f32 {
    let input = sample_node(graph, ctx, s.coordinate, x, y, z);
    let len = s.points.len();
    let mut stop = 0;
    while stop < len && input >= s.points[stop].location {
        stop += 1;
    }
    let value_at = |graph: &Graph, ctx: &mut ChunkCtx, point: &SplinePoint| match &point.value {
        SplineValue::Fixed(v) => *v,
        SplineValue::Nested(n) => eval_spline(graph, ctx, n, x, y, z),
    };
    let extend = |graph: &Graph, ctx: &mut ChunkCtx, index: usize, input: f32| -> f32 {
        let point = &s.points[index];
        let value = value_at(graph, ctx, point);
        if point.derivative == 0.0 {
            value
        } else {
            value + point.derivative * (input - point.location)
        }
    };
    // stop is the first point past the input; the interval starts before.
    if stop == 0 {
        return extend(graph, ctx, 0, input);
    }
    if stop == len {
        return extend(graph, ctx, len - 1, input);
    }
    let first = &s.points[stop - 1];
    let second = &s.points[stop];
    let x1 = first.location;
    let x2 = second.location;
    let t = (input - x1) / (x2 - x1);
    let y1 = value_at(graph, ctx, first);
    let y2 = value_at(graph, ctx, second);
    let a = first.derivative * (x2 - x1) - (y2 - y1);
    let b = -second.derivative * (x2 - x1) + (y2 - y1);
    lerp(t, y1, y2) + t * (1.0 - t) * lerp(t, a, b)
}

// ---------------------------------------------------------------------------
// Material rules.
// ---------------------------------------------------------------------------

enum Cond {
    StoneDepth {
        ceiling: bool,
        offset: i32,
        add_surface: bool,
        secondary_range: i32,
    },
    Water {
        offset: i32,
        multiplier: i32,
        add_stone: bool,
    },
    YAbove {
        anchor_y: i32,
        multiplier: i32,
        add_stone: bool,
    },
    VerticalGradient {
        name: String,
        true_below: i32,
        false_above: i32,
    },
    NoiseThreshold {
        stack: Arc<StackNoise>,
        min: f64,
        max: f64,
        three_d: bool,
    },
    /// Resolved at parse: the single-biome world either matches or not.
    Biome(bool),
    AbovePreliminary,
    Not(Box<Cond>),
    Hole,
    Steep,
}

struct OreVein {
    density: NodeRef,
    richness: NodeRef,
    filler_gap: NodeRef,
    ore: u32,
    raw: u32,
    filler: u32,
    raw_chance: f32,
}

enum Rule {
    Sequence(Vec<Rule>),
    Condition(Cond, Box<Rule>),
    Block(u32),
    OreVein(Box<OreVein>),
    /// Terracotta band pattern; only reachable under biome gates the
    /// single-biome world never satisfies, so the stone below stays.
    Bands,
}

impl<'a> Loader<'a> {
    fn anchor_y(&self, v: &Value) -> Result<i32> {
        if let Some(n) = v.get("absolute").and_then(Value::as_i64) {
            return Ok(n as i32);
        }
        if let Some(n) = v.get("above_bottom").and_then(Value::as_i64) {
            return Ok(self.min_y + n as i32);
        }
        if let Some(n) = v.get("below_top").and_then(Value::as_i64) {
            return Ok(self.min_y + self.height - n as i32);
        }
        bail!("unsupported anchor")
    }

    fn rule_id(&mut self, id: &str) -> Result<Rule> {
        let key = id.strip_prefix("minecraft:").unwrap_or(id);
        let config = self.read_json("material_rule", key)?;
        self.rule(&config)
    }

    fn rule(&mut self, v: &Value) -> Result<Rule> {
        if let Value::String(id) = v {
            return self.rule_id(id);
        }
        let kind = v.get("type").and_then(Value::as_str).context("rule type")?;
        match kind {
            "minecraft:sequence" => {
                let list = v
                    .get("sequence")
                    .and_then(Value::as_array)
                    .context("sequence")?;
                let rules = list
                    .iter()
                    .map(|r| self.rule(r))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Rule::Sequence(rules))
            }
            "minecraft:condition" => {
                let cond = v.get("if_true").context("if_true")?;
                let cond = self.condition(cond)?;
                let then = self.rule(v.get("then_run").context("then_run")?)?;
                Ok(Rule::Condition(cond, Box::new(then)))
            }
            "minecraft:block" => {
                let name = v
                    .get("result_state")
                    .and_then(Value::as_str)
                    .context("result_state")?;
                let state = self
                    .registry
                    .state_id(name, "")
                    .with_context(|| format!("rule block {name}"))?;
                Ok(Rule::Block(state))
            }
            "minecraft:ore_vein" => {
                let density = self.field_node(v, "density")?;
                let richness = self.field_node(v, "richness")?;
                let filler_gap = self.field_node(v, "filler_gap")?;
                let resolve = |name: &str| {
                    self.registry
                        .state_id(name, "")
                        .with_context(|| format!("vein block {name}"))
                };
                Ok(Rule::OreVein(Box::new(OreVein {
                    density,
                    richness,
                    filler_gap,
                    ore: resolve(
                        v.get("ore_block")
                            .and_then(Value::as_str)
                            .context("ore_block")?,
                    )?,
                    raw: resolve(
                        v.get("raw_ore_block")
                            .and_then(Value::as_str)
                            .context("raw_ore_block")?,
                    )?,
                    filler: resolve(
                        v.get("filler_block")
                            .and_then(Value::as_str)
                            .context("filler_block")?,
                    )?,
                    raw_chance: field_f64(v, "raw_ore_chance")? as f32,
                })))
            }
            "minecraft:bandlands" => Ok(Rule::Bands),
            other => bail!("unsupported material rule {other}"),
        }
    }

    fn condition_id(&mut self, id: &str) -> Result<Cond> {
        let key = id.strip_prefix("minecraft:").unwrap_or(id);
        let config = self.read_json("material_condition", key)?;
        self.condition(&config)
    }

    fn condition(&mut self, v: &Value) -> Result<Cond> {
        if let Value::String(id) = v {
            return self.condition_id(id);
        }
        let kind = v
            .get("type")
            .and_then(Value::as_str)
            .context("condition type")?;
        match kind {
            "minecraft:stone_depth" => Ok(Cond::StoneDepth {
                ceiling: v.get("surface_type").and_then(Value::as_str) == Some("ceiling"),
                offset: field_i64(v, "offset")? as i32,
                add_surface: v.get("add_surface_depth").and_then(Value::as_bool) == Some(true),
                secondary_range: field_i64(v, "secondary_depth_range")? as i32,
            }),
            "minecraft:water" => Ok(Cond::Water {
                offset: field_i64(v, "offset")? as i32,
                multiplier: field_i64(v, "surface_depth_multiplier")? as i32,
                add_stone: v.get("add_stone_depth").and_then(Value::as_bool) == Some(true),
            }),
            "minecraft:y_above" => Ok(Cond::YAbove {
                anchor_y: self.anchor_y(v.get("anchor").context("anchor")?)?,
                multiplier: field_i64(v, "surface_depth_multiplier")? as i32,
                add_stone: v.get("add_stone_depth").and_then(Value::as_bool) == Some(true),
            }),
            "minecraft:vertical_gradient" => {
                let name = v
                    .get("random_name")
                    .and_then(Value::as_str)
                    .context("random_name")?
                    .to_string();
                if !self.gradient_factories.contains_key(&name) {
                    let factory = self.world.from_name(&name).fork_positional();
                    self.gradient_factories.insert(name.clone(), factory);
                }
                Ok(Cond::VerticalGradient {
                    name,
                    true_below: self.anchor_y(v.get("true_at_and_below").context("anchor")?)?,
                    false_above: self.anchor_y(v.get("false_at_and_above").context("anchor")?)?,
                })
            }
            "minecraft:noise_threshold" => {
                let name = v.get("noise").and_then(Value::as_str).context("noise")?;
                Ok(Cond::NoiseThreshold {
                    stack: self.stack(name)?,
                    min: field_f64(v, "min_threshold")?,
                    max: field_f64(v, "max_threshold")?,
                    three_d: v.get("is_3d").and_then(Value::as_bool) == Some(true),
                })
            }
            "minecraft:biome" => {
                let list: Vec<&str> = match v.get("biome_is") {
                    Some(Value::String(s)) => vec![s.as_str()],
                    Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).collect(),
                    _ => bail!("biome_is"),
                };
                Ok(Cond::Biome(list.contains(&"minecraft:plains")))
            }
            "minecraft:above_preliminary_surface" => Ok(Cond::AbovePreliminary),
            "minecraft:not" => {
                let inner = self.condition(v.get("invert").context("invert")?)?;
                Ok(Cond::Not(Box::new(inner)))
            }
            "minecraft:hole" => Ok(Cond::Hole),
            "minecraft:steep" => Ok(Cond::Steep),
            other => bail!("unsupported material condition {other}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Aquifer.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
struct FluidStatus {
    level: i32,
    lava: bool,
}

/// Per-chunk aquifer state: grid locations, resolved fluid statuses,
/// and the preliminary-surface memo.
struct AquiferState {
    locations: HashMap<(i32, i32, i32), (i32, i32, i32)>,
    statuses: HashMap<(i32, i32, i32), FluidStatus>,
    surface: HashMap<(i32, i32), i32>,
    skip_above: i32,
}

impl AquiferState {
    fn location(&mut self, pos: &Positional, gx: i32, gy: i32, gz: i32) -> (i32, i32, i32) {
        if let Some(loc) = self.locations.get(&(gx, gy, gz)) {
            return *loc;
        }
        let mut rng = pos.at(gx, gy, gz);
        let x = (gx << 4) + rng.next_int(10);
        let y = gy * 12 + rng.next_int(9);
        let z = (gz << 4) + rng.next_int(10);
        self.locations.insert((gx, gy, gz), (x, y, z));
        (x, y, z)
    }

    /// The preliminary surface level at quart-quantized coordinates.
    fn surface_level(
        &mut self,
        graph: &Graph,
        ctx: &mut ChunkCtx,
        preliminary: NodeRef,
        x: i32,
        z: i32,
    ) -> i32 {
        let qx = (x >> 2) << 2;
        let qz = (z >> 2) << 2;
        if let Some(level) = self.surface.get(&(qx, qz)) {
            return *level;
        }
        let sample = sample_node(graph, ctx, preliminary, qx, 0, qz);
        let level = sample.floor() as i32;
        self.surface.insert((qx, qz), level);
        level
    }

    /// Highest adjusted surface level over the chunk's grid span; sets
    /// the depth above which aquifer sampling is skipped.
    #[allow(clippy::too_many_arguments)]
    fn max_surface_level(
        &mut self,
        graph: &Graph,
        ctx: &mut ChunkCtx,
        preliminary: NodeRef,
        min_x: i32,
        max_x: i32,
        min_z: i32,
        max_z: i32,
    ) -> i32 {
        let mut highest = i32::MIN;
        let mut x = min_x;
        while x <= max_x {
            let mut z = min_z;
            while z <= max_z {
                let level = self.surface_level(graph, ctx, preliminary, x, z);
                highest = highest.max(level);
                z += 4;
            }
            x += 4;
        }
        highest
    }
}

// ---------------------------------------------------------------------------
// The terrain engine.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Climate sampling and biome selection.
// ---------------------------------------------------------------------------

/// The six climate router functions plus the parameter table that turns
/// their samples into biome wire ids.
pub struct ClimateSampler {
    temperature: NodeRef,
    humidity: NodeRef,
    continentalness: NodeRef,
    erosion: NodeRef,
    depth: NodeRef,
    weirdness: NodeRef,
    table: BiomeTable,
}

impl ClimateSampler {
    /// Binds the router's climate functions and loads the pinned
    /// parameter table. Router key names map onto the parameter axes:
    /// vegetation carries humidity, continents carry continentalness,
    /// ridges carry weirdness.
    fn from_router(loader: &mut Loader, router: &Value, pins: &Path) -> Result<ClimateSampler> {
        let axis = |loader: &mut Loader, key: &str| -> Result<NodeRef> {
            let id = router
                .get(key)
                .and_then(Value::as_str)
                .with_context(|| format!("noise router {key}"))?;
            loader.function_id(id)
        };
        Ok(ClimateSampler {
            temperature: axis(loader, "temperature")?,
            humidity: axis(loader, "vegetation")?,
            continentalness: axis(loader, "continents")?,
            erosion: axis(loader, "erosion")?,
            depth: axis(loader, "depth")?,
            weirdness: axis(loader, "ridges")?,
            table: BiomeTable::load(pins)?,
        })
    }

    /// Samples the climate functions at block coordinates and resolves
    /// the biome. Each axis narrows through the quantized search.
    fn biome(&self, graph: &Graph, ctx: &mut ChunkCtx, x: i32, y: i32, z: i32) -> u32 {
        let axes = self.axes(graph, ctx, x, y, z);
        self.table.find_axes(axes)
    }

    /// Raw climate samples at block coordinates.
    fn axes(&self, graph: &Graph, ctx: &mut ChunkCtx, x: i32, y: i32, z: i32) -> [f32; AXIS_COUNT] {
        let nodes = [
            self.temperature,
            self.humidity,
            self.continentalness,
            self.erosion,
            self.depth,
            self.weirdness,
        ];
        let mut axes = [0f32; AXIS_COUNT];
        for (slot, &node) in nodes.iter().enumerate() {
            axes[slot] = sample_node(graph, ctx, node, x, y, z);
        }
        axes
    }
}

/// The density-driven terrain generator.
pub struct NoiseTerrain {
    graph: Graph,
    world: Positional,
    aquifer_pos: Positional,
    final_density: NodeRef,
    preliminary: NodeRef,
    chunk_surface_level: NodeRef,
    aquifer_barrier: NodeRef,
    aquifer_floodedness: NodeRef,
    aquifer_spread: NodeRef,
    aquifer_lava: NodeRef,
    aquifer_exclusion: NodeRef,
    climate: ClimateSampler,
    rule: Rule,
    surface_stack: Arc<StackNoise>,
    secondary_stack: Arc<StackNoise>,
    gradient_factories: HashMap<String, Positional>,
    ore_pos: Positional,
    carvers: crate::caves::Carvers,
    stone: u32,
    water: u32,
    lava: u32,
    air: u32,
    grass: u32,
    mycelium: u32,
    dirt: u32,
    uncarvable: std::collections::HashSet<u32>,
    pub sea_level: i32,
    pub min_y: i32,
    pub height: i32,
}

impl NoiseTerrain {
    /// Builds the graph from the pinned configs at this seed.
    pub fn with_seed(seed: i64, registry: &BlockRegistry, pins: &Path) -> Result<NoiseTerrain> {
        let settings = read(pins.join("noise_settings").join("overworld.json"))?;
        let min_y = settings
            .pointer("/noise/min_y")
            .and_then(Value::as_i64)
            .context("min_y")? as i32;
        let height = settings
            .pointer("/noise/height")
            .and_then(Value::as_i64)
            .context("height")? as i32;
        let sea_level = settings
            .get("sea_level")
            .and_then(Value::as_i64)
            .context("sea_level")? as i32;
        let mut loader = Loader::new(pins, seed, registry, min_y, height);
        let router = settings.get("noise_router").context("noise_router")?;
        let final_density = loader.function_id(
            router
                .get("final_density")
                .and_then(Value::as_str)
                .context("final_density")?,
        )?;
        let chunk_surface_level = loader.function_id(
            router
                .get("chunk_surface_level")
                .and_then(Value::as_str)
                .context("chunk_surface_level")?,
        )?;
        let aquifers = settings.get("aquifers").context("aquifers")?;
        let aq = |loader: &mut Loader, key: &str| -> Result<NodeRef> {
            let v = aquifers.get(key).context(format!("aquifer {key}"))?;
            loader.function(v)
        };
        let aquifer_barrier = aq(&mut loader, "barrier")?;
        let aquifer_floodedness = aq(&mut loader, "fluid_level_floodedness")?;
        let aquifer_spread = aq(&mut loader, "fluid_level_spread")?;
        let aquifer_lava = aq(&mut loader, "lava")?;
        let aquifer_exclusion = aq(&mut loader, "exclusion")?;
        let preliminary = aq(&mut loader, "surface_level")?;
        let climate = ClimateSampler::from_router(&mut loader, router, pins)?;
        let rule = loader.rule_id(
            settings
                .get("material_rule")
                .and_then(Value::as_str)
                .context("material_rule")?,
        )?;
        let surface_stack = loader.stack("minecraft:surface")?;
        let secondary_stack = loader.stack("minecraft:surface_secondary")?;
        loader.graph.finish();

        let world = world_positional(seed);
        let resolve = |name: &str| {
            registry
                .state_id(name, "")
                .with_context(|| format!("terrain block {name}"))
        };
        let stone = resolve(
            settings
                .get("default_block")
                .and_then(Value::as_str)
                .context("default_block")?,
        )?;
        let water = resolve(
            settings
                .get("default_fluid")
                .and_then(Value::as_str)
                .context("default_fluid")?,
        )?;
        let carvers = crate::caves::Carvers::load(pins, seed, min_y, height)?;
        let mut uncarvable = std::collections::HashSet::new();
        let tag: Value = serde_json::from_str(
            &std::fs::read_to_string(pins.join("tags").join("block").join("uncarvable.json"))
                .context("uncarvable tag")?,
        )
        .context("parsing uncarvable tag")?;
        if let Some(values) = tag.get("values").and_then(Value::as_array) {
            for name in values.iter().filter_map(Value::as_str) {
                uncarvable.insert(resolve(name)?);
            }
        }
        Ok(NoiseTerrain {
            world,
            aquifer_pos: world_positional(seed)
                .from_name("minecraft:aquifer")
                .fork_positional(),
            gradient_factories: loader.gradient_factories,
            graph: loader.graph,
            final_density,
            preliminary,
            chunk_surface_level,
            aquifer_barrier,
            aquifer_floodedness,
            aquifer_spread,
            aquifer_lava,
            aquifer_exclusion,
            climate,
            rule,
            surface_stack,
            secondary_stack,
            ore_pos: world_positional(seed)
                .from_name("minecraft:ore")
                .fork_positional(),
            carvers,
            stone,
            water,
            lava: resolve("minecraft:lava")?,
            air: resolve("minecraft:air")?,
            grass: registry
                .state_id("minecraft:grass_block", "snowy=false")
                .context("terrain block minecraft:grass_block")?,
            mycelium: resolve("minecraft:mycelium")?,
            dirt: resolve("minecraft:dirt")?,
            uncarvable,
            sea_level,
            min_y,
            height,
        })
    }

    /// The block buffer for one chunk (layer-major, column = z * 16 + x).
    pub fn fill_chunk(&self, cx: i32, cz: i32) -> Vec<u32> {
        let start_x = cx * EDGE;
        let start_z = cz * EDGE;
        let mut ctx = ChunkCtx::new(start_x, start_z, self.min_y, self.height);
        let mut aquifer = self.aquifer_state(&mut ctx);
        let mut blocks = vec![self.air; self.height as usize * COLUMNS];
        for y in 0..self.height {
            let world_y = self.min_y + y;
            for z in 0..EDGE {
                for x in 0..EDGE {
                    let bx = start_x + x;
                    let bz = start_z + z;
                    let density = f64::from(sample_node(
                        &self.graph,
                        &mut ctx,
                        self.final_density,
                        bx,
                        world_y,
                        bz,
                    ));
                    let state = self.substance(&mut ctx, &mut aquifer, bx, world_y, bz, density);
                    let column = (z * EDGE + x) as usize;
                    blocks[y as usize * COLUMNS + column] = state;
                }
            }
        }
        self.surface_pass(&mut ctx, &mut blocks);
        self.carve_pass(&mut ctx, &mut aquifer, &mut blocks);
        blocks
    }

    /// The topmost density-supported stone layer of a column; -1 when
    /// the column holds no terrain.
    pub fn column_top(&self, wx: i32, wz: i32) -> i32 {
        let start_x = wx.div_euclid(EDGE) * EDGE;
        let start_z = wz.div_euclid(EDGE) * EDGE;
        let mut ctx = ChunkCtx::new(start_x, start_z, self.min_y, self.height);
        for y in (0..self.height).rev() {
            let world_y = self.min_y + y;
            let density = f64::from(sample_node(
                &self.graph,
                &mut ctx,
                self.final_density,
                wx,
                world_y,
                wz,
            ));
            if density > 0.0 {
                return world_y;
            }
        }
        -1
    }

    /// Biome wire ids for every section's 4x4x4 cell grid, storage order
    /// (x fastest, then z, then y). Each cell resolves at its minimum
    /// corner on the quart grid.
    pub fn section_biomes(&self, cx: i32, cz: i32) -> Vec<[u32; 64]> {
        let sections = (self.height / 16) as usize;
        let mut ctx = ChunkCtx::new(cx * EDGE, cz * EDGE, self.min_y, self.height);
        let mut out = Vec::with_capacity(sections);
        let mut cells = [0u32; 64];
        for section in 0..sections {
            let base_quart_y = self.min_y.div_euclid(4) + section as i32 * 4;
            for y4 in 0..4 {
                for z4 in 0..4 {
                    for x4 in 0..4 {
                        let x = cx * EDGE + x4 * 4;
                        let z = cz * EDGE + z4 * 4;
                        let y = (base_quart_y + y4) * 4;
                        let cell = (y4 * 16 + z4 * 4 + x4) as usize;
                        cells[cell] = self.climate.biome(&self.graph, &mut ctx, x, y, z);
                    }
                }
            }
            out.push(cells);
        }
        out
    }

    /// The biome holding one block position, sampled at the quart cell
    /// corner that owns it.
    pub fn biome_at(&self, wx: i32, wy: i32, wz: i32) -> u32 {
        let x = wx.div_euclid(4) * 4;
        let z = wz.div_euclid(4) * 4;
        let y = wy.div_euclid(4) * 4;
        let mut ctx = ChunkCtx::new(
            wx.div_euclid(EDGE) * EDGE,
            wz.div_euclid(EDGE) * EDGE,
            self.min_y,
            self.height,
        );
        self.climate.biome(&self.graph, &mut ctx, x, y, z)
    }

    /// Raw climate axis samples at one block position: temperature,
    /// humidity, continentalness, erosion, depth, weirdness.
    #[cfg(test)]
    pub(crate) fn climate_axes(&self, wx: i32, wy: i32, wz: i32) -> [f32; AXIS_COUNT] {
        let mut ctx = ChunkCtx::new(
            wx.div_euclid(EDGE) * EDGE,
            wz.div_euclid(EDGE) * EDGE,
            self.min_y,
            self.height,
        );
        self.climate.axes(&self.graph, &mut ctx, wx, wy, wz)
    }

    fn global_fluid(&self, y: i32) -> FluidStatus {
        if f64::from(y) < f64::from(GLOBAL_LAVA_LEVEL.min(self.sea_level)) {
            FluidStatus {
                level: GLOBAL_LAVA_LEVEL,
                lava: true,
            }
        } else {
            FluidStatus {
                level: self.sea_level,
                lava: false,
            }
        }
    }

    fn fluid_state(&self, status: FluidStatus, y: i32) -> u32 {
        if y < status.level {
            if status.lava {
                self.lava
            } else {
                self.water
            }
        } else {
            self.air
        }
    }

    fn aquifer_state(&self, ctx: &mut ChunkCtx) -> AquiferState {
        let start_x = ctx.start_x;
        let start_z = ctx.start_z;
        let min_grid_x = (start_x - 5) >> 4;
        let max_grid_x = ((start_x + EDGE - 1 - 5) >> 4) + 1;
        let min_grid_z = (start_z - 5) >> 4;
        let max_grid_z = ((start_z + EDGE - 1 - 5) >> 4) + 1;
        let mut state = AquiferState {
            locations: HashMap::new(),
            statuses: HashMap::new(),
            surface: HashMap::new(),
            skip_above: self.min_y - 1,
        };
        let max_surface = state.max_surface_level(
            &self.graph,
            ctx,
            self.preliminary,
            min_grid_x << 4,
            (max_grid_x << 4) + 9,
            min_grid_z << 4,
            (max_grid_z << 4) + 9,
        );
        let skip_grid_y = (max_surface + 8 + 12).div_euclid(12) + 1;
        state.skip_above = skip_grid_y * 12 + 11 - 1;
        state
    }

    /// Stone when the density and barriers agree on rock, else the fluid
    /// or air of the governing fluid status.
    fn substance(
        &self,
        ctx: &mut ChunkCtx,
        aquifer: &mut AquiferState,
        x: i32,
        y: i32,
        z: i32,
        density: f64,
    ) -> u32 {
        if density > 0.0 {
            return self.stone;
        }
        let global = self.global_fluid(y);
        if y > aquifer.skip_above {
            return self.fluid_state(global, y);
        }
        if y < global.level && global.lava {
            return self.lava;
        }
        let anchor_x = (x - 5) >> 4;
        let anchor_y = (y + 1).div_euclid(12);
        let anchor_z = (z - 5) >> 4;
        let mut distances = [i32::MAX; 4];
        let mut cells = [(0i32, 0i32, 0i32); 4];
        for x1 in 0..=1 {
            for y1 in -1..=1 {
                for z1 in 0..=1 {
                    let gx = anchor_x + x1;
                    let gy = anchor_y + y1;
                    let gz = anchor_z + z1;
                    let (lx, ly, lz) = aquifer.location(&self.aquifer_pos, gx, gy, gz);
                    let distance = (lx - x) * (lx - x) + (ly - y) * (ly - y) + (lz - z) * (lz - z);
                    if distances[0] >= distance {
                        distances[3] = distances[2];
                        distances[2] = distances[1];
                        distances[1] = distances[0];
                        distances[0] = distance;
                        cells[3] = cells[2];
                        cells[2] = cells[1];
                        cells[1] = cells[0];
                        cells[0] = (gx, gy, gz);
                    } else if distances[1] >= distance {
                        distances[3] = distances[2];
                        distances[2] = distances[1];
                        distances[1] = distance;
                        cells[3] = cells[2];
                        cells[2] = cells[1];
                        cells[1] = (gx, gy, gz);
                    } else if distances[2] >= distance {
                        distances[3] = distances[2];
                        distances[2] = distance;
                        cells[3] = cells[2];
                        cells[2] = (gx, gy, gz);
                    } else if distances[3] >= distance {
                        distances[3] = distance;
                        cells[3] = (gx, gy, gz);
                    }
                }
            }
        }
        let first = self.status(ctx, aquifer, cells[0]);
        let similarity12 = 1.0 - (distances[1] - distances[0]) as f64 / 25.0;
        if similarity12 <= 0.0 {
            return self.fluid_state(first, y);
        }
        // Water sitting over the global lava floor flows instead of
        // pressurizing.
        if self.fluid_state(first, y) == self.water
            && self.fluid_state(self.global_fluid(y - 1), y - 1) == self.lava
        {
            return self.water;
        }
        let mut barrier_noise = f64::NAN;
        let second = self.status(ctx, aquifer, cells[1]);
        let barrier12 =
            similarity12 * self.pressure(ctx, x, y, z, first, second, &mut barrier_noise);
        if density + barrier12 > 0.0 {
            return self.stone;
        }
        let third = self.status(ctx, aquifer, cells[2]);
        let similarity13 = 1.0 - (distances[2] - distances[0]) as f64 / 25.0;
        if similarity13 > 0.0 {
            let barrier13 = similarity12
                * similarity13
                * self.pressure(ctx, x, y, z, first, third, &mut barrier_noise);
            if density + barrier13 > 0.0 {
                return self.stone;
            }
        }
        let similarity23 = 1.0 - (distances[2] - distances[1]) as f64 / 25.0;
        if similarity23 > 0.0 {
            let barrier23 = similarity12
                * similarity23
                * self.pressure(ctx, x, y, z, second, third, &mut barrier_noise);
            if density + barrier23 > 0.0 {
                return self.stone;
            }
        }
        self.fluid_state(first, y)
    }

    /// The barrier pressure between two fluid cells at this position.
    #[allow(clippy::too_many_arguments)]
    fn pressure(
        &self,
        ctx: &mut ChunkCtx,
        x: i32,
        y: i32,
        z: i32,
        first: FluidStatus,
        second: FluidStatus,
        barrier_noise: &mut f64,
    ) -> f64 {
        let type_first = y < first.level;
        let type_second = y < second.level;
        // Fluid cells of differing kind (water against lava) hold a solid
        // barrier between them.
        if type_first && type_second && first.lava != second.lava {
            return 2.0;
        }
        // Level arithmetic wraps like the reference int math when a level
        // sits below the world floor.
        let fluid_diff = first.level.wrapping_sub(second.level).wrapping_abs();
        if fluid_diff == 0 {
            return 0.0;
        }
        let average = 0.5 * f64::from(first.level.wrapping_add(second.level));
        let above_average = f64::from(y) + 0.5 - average;
        let base = fluid_diff as f64 / 2.0;
        let towards_middle = base - above_average.abs();
        let gradient = if above_average > 0.0 {
            let center = towards_middle;
            if center > 0.0 {
                center / 1.5
            } else {
                center / 2.5
            }
        } else {
            let center = 3.0 + towards_middle;
            if center > 0.0 {
                center / 3.0
            } else {
                center / 10.0
            }
        };
        let noise = if !(-2.0..=2.0).contains(&gradient) {
            0.0
        } else if barrier_noise.is_nan() {
            let sample = f64::from(sample_node(&self.graph, ctx, self.aquifer_barrier, x, y, z));
            *barrier_noise = sample;
            sample
        } else {
            *barrier_noise
        };
        2.0 * (noise + gradient)
    }

    fn status(
        &self,
        ctx: &mut ChunkCtx,
        aquifer: &mut AquiferState,
        cell: (i32, i32, i32),
    ) -> FluidStatus {
        if let Some(status) = aquifer.statuses.get(&cell) {
            return *status;
        }
        let (lx, ly, lz) = aquifer.locations[&cell];
        let status = self.compute_status(ctx, aquifer, lx, ly, lz);
        aquifer.statuses.insert(cell, status);
        status
    }

    /// Resolves the fluid status of one aquifer cell from the 13-point
    /// preliminary surface ring.
    fn compute_status(
        &self,
        ctx: &mut ChunkCtx,
        aquifer: &mut AquiferState,
        x: i32,
        y: i32,
        z: i32,
    ) -> FluidStatus {
        const RING: [(i32, i32); 13] = [
            (0, 0),
            (-2, -1),
            (-1, -1),
            (0, -1),
            (1, -1),
            (-3, 0),
            (-2, 0),
            (-1, 0),
            (1, 0),
            (-2, 1),
            (-1, 1),
            (0, 1),
            (1, 1),
        ];
        let global = self.global_fluid(y);
        let mut lowest = i32::MAX;
        let top = y + 12;
        let bottom = y - 12;
        let mut under_center = false;
        for &(ox, oz) in &RING {
            let sample_x = x + ox * 16;
            let sample_z = z + oz * 16;
            let surface =
                aquifer.surface_level(&self.graph, ctx, self.preliminary, sample_x, sample_z);
            let adjusted = surface + 8;
            let start = ox == 0 && oz == 0;
            if start && bottom > adjusted {
                return global;
            }
            let pokes = top > adjusted;
            if pokes || start {
                let fluid_at_surface = self.global_fluid(adjusted);
                let fluid = fluid_at_surface.level > adjusted;
                if fluid {
                    if start {
                        under_center = true;
                    }
                    if pokes {
                        return fluid_at_surface;
                    }
                }
            }
            lowest = lowest.min(surface);
        }
        let level = self.cell_surface_level(ctx, x, y, z, global, lowest, under_center);
        let lava = self.cell_is_lava(ctx, x, y, z, global, level);
        FluidStatus { level, lava }
    }

    #[allow(clippy::too_many_arguments)]
    fn cell_surface_level(
        &self,
        ctx: &mut ChunkCtx,
        x: i32,
        y: i32,
        z: i32,
        global: FluidStatus,
        lowest: i32,
        under_center: bool,
    ) -> i32 {
        let (partial, full);
        if f64::from(sample_node(
            &self.graph,
            ctx,
            self.aquifer_exclusion,
            x,
            y,
            z,
        )) > 0.0
        {
            partial = -1.0;
            full = -1.0;
        } else {
            let distance_below = (lowest + 8 - y) as f64;
            let factor = if under_center {
                map(distance_below, 0.0, 64.0, 1.0, 0.0)
            } else {
                0.0
            };
            let noise = f64::from(sample_node(
                &self.graph,
                ctx,
                self.aquifer_floodedness,
                x,
                y,
                z,
            ))
            .clamp(-1.0, 1.0);
            full = noise - map(factor, 1.0, 0.0, -0.3, 0.8);
            partial = noise - map(factor, 1.0, 0.0, -0.8, 0.4);
        }
        if full > 0.0 {
            return global.level;
        }
        if partial > 0.0 {
            let cell_x = x.div_euclid(16);
            let cell_y = y.div_euclid(40);
            let cell_z = z.div_euclid(16);
            let spread = f64::from(
                sample_node(
                    &self.graph,
                    ctx,
                    self.aquifer_spread,
                    cell_x,
                    cell_y,
                    cell_z,
                ) * 10.0f32,
            );
            let target = cell_y * 40 + 20 + quantize(spread, 3);
            return lowest.min(target);
        }
        WAY_BELOW
    }

    fn cell_is_lava(
        &self,
        ctx: &mut ChunkCtx,
        x: i32,
        y: i32,
        z: i32,
        global: FluidStatus,
        level: i32,
    ) -> bool {
        if global.lava || level == WAY_BELOW || level > -10 {
            return global.lava;
        }
        let noise = f64::from(sample_node(
            &self.graph,
            ctx,
            self.aquifer_lava,
            x.div_euclid(64),
            y.div_euclid(40),
            z.div_euclid(64),
        ));
        noise.abs() > 0.3
    }

    // -----------------------------------------------------------------
    // Carve pass.
    // -----------------------------------------------------------------

    /// Sweeps the carver sources around the chunk, then rewrites each marked
    /// cell with the aquifer's fluid decision at density zero. Grass-topped
    /// columns regrow their surface block under the new opening.
    fn carve_pass(&self, ctx: &mut ChunkCtx, aquifer: &mut AquiferState, blocks: &mut [u32]) {
        let start_x = ctx.start_x;
        let start_z = ctx.start_z;
        let mut biome_of = |x: i32, z: i32| self.climate.biome(&self.graph, ctx, x, 0, z);
        let mask = self.carvers.build_mask(
            &mut biome_of,
            start_x.div_euclid(EDGE),
            start_z.div_euclid(EDGE),
        );
        if mask.is_empty() {
            return;
        }
        let mut heights = [0i32; COLUMNS];
        for column in 0..COLUMNS {
            let mut top = self.min_y;
            for y in (0..self.height).rev() {
                if blocks[y as usize * COLUMNS + column] != self.air {
                    top = self.min_y + y + 1;
                    break;
                }
            }
            heights[column] = top;
        }
        let mut job = SurfaceJob {
            terrain: self,
            depth: HashMap::new(),
            secondary: HashMap::new(),
            min_level: HashMap::new(),
        };
        mask.for_each_column(&mut |x: usize, z: usize, bottom: i32, top: i32| {
            let column = z * EDGE as usize + x;
            let world_x = start_x + x as i32;
            let world_z = start_z + z as i32;
            let mut has_grass = false;
            for world_y in (bottom..=top).rev() {
                let index = (world_y - self.min_y) as usize * COLUMNS + column;
                let state = blocks[index];
                if self.uncarvable.contains(&state) {
                    continue;
                }
                if state == self.grass || state == self.mycelium {
                    has_grass = true;
                }
                let carved = self.substance(ctx, aquifer, world_x, world_y, world_z, 0.0);
                blocks[index] = carved;
                self.update_height(blocks, &mut heights, column, world_y, carved != self.air);
                if !has_grass || world_y - 1 < self.min_y {
                    continue;
                }
                let below = index - COLUMNS;
                if blocks[below] != self.dirt {
                    continue;
                }
                let col_x_fwd = z * EDGE as usize + (x + 1).min(15);
                let col_x_back = z * EDGE as usize + x.saturating_sub(1);
                let col_z_fwd = (z + 1).min(15) * EDGE as usize + x;
                let col_z_back = z.saturating_sub(1) * EDGE as usize + x;
                let gradient_x = heights[col_x_fwd] - heights[col_x_back];
                let gradient_z = heights[col_z_fwd] - heights[col_z_back];
                let under_fluid = carved == self.water || carved == self.lava;
                let walk = Walk {
                    stone_above: 1,
                    stone_below: 1,
                    water_height: if under_fluid { world_y + 1 } else { i32::MIN },
                    gradient_x,
                    gradient_z,
                };
                if let Some(surface) = try_apply(
                    &self.rule,
                    &mut job,
                    ctx,
                    &walk,
                    world_x,
                    world_y - 1,
                    world_z,
                ) {
                    blocks[below] = surface;
                    self.update_height(
                        blocks,
                        &mut heights,
                        column,
                        world_y - 1,
                        surface != self.air,
                    );
                }
            }
        });
    }

    /// First-free height maintenance as the carve rewrites cells.
    fn update_height(
        &self,
        blocks: &[u32],
        heights: &mut [i32; COLUMNS],
        column: usize,
        y: i32,
        solid: bool,
    ) {
        let current = heights[column];
        if solid {
            if y >= current {
                heights[column] = y + 1;
            }
        } else if current == y + 1 {
            let mut scan = y;
            while scan >= self.min_y
                && blocks[(scan - self.min_y) as usize * COLUMNS + column] == self.air
            {
                scan -= 1;
            }
            heights[column] = scan + 1;
        }
    }

    // -----------------------------------------------------------------
    // Surface pass.
    // -----------------------------------------------------------------

    fn surface_pass(&self, ctx: &mut ChunkCtx, blocks: &mut [u32]) {
        let start_x = ctx.start_x;
        let start_z = ctx.start_z;
        let mut first_free = [0i32; COLUMNS];
        for column in 0..COLUMNS {
            let mut top = self.min_y;
            for y in (0..self.height).rev() {
                if blocks[y as usize * COLUMNS + column] != self.air {
                    top = self.min_y + y + 1;
                    break;
                }
            }
            first_free[column] = top;
        }
        let mut job = SurfaceJob {
            terrain: self,
            depth: HashMap::new(),
            secondary: HashMap::new(),
            min_level: HashMap::new(),
        };
        for z in 0..EDGE {
            for x in 0..EDGE {
                let column = (z * EDGE + x) as usize;
                let bx = start_x + x;
                let bz = start_z + z;
                let neighbor_x = (z * EDGE + (x + 1).min(EDGE - 1)) as usize;
                let neighbor_back_x = (z * EDGE + (x - 1).max(0)) as usize;
                let neighbor_z = ((z + 1).min(EDGE - 1) * EDGE + x) as usize;
                let neighbor_back_z = ((z - 1).max(0) * EDGE + x) as usize;
                let mut walk = Walk {
                    stone_above: 0,
                    stone_below: 0,
                    water_height: i32::MIN,
                    gradient_x: first_free[neighbor_x] - first_free[neighbor_back_x],
                    gradient_z: first_free[neighbor_z] - first_free[neighbor_back_z],
                };
                let mut next_ceiling = i32::MAX;
                let start = first_free[column];
                let mut y = start - 1;
                while y >= self.min_y {
                    let state = blocks[(y - self.min_y) as usize * COLUMNS + column];
                    if state == self.air {
                        walk.stone_above = 0;
                        walk.water_height = i32::MIN;
                        y -= 1;
                        continue;
                    }
                    if state == self.water || state == self.lava {
                        if walk.water_height == i32::MIN {
                            walk.water_height = y + 1;
                        }
                        y -= 1;
                        continue;
                    }
                    if next_ceiling >= y {
                        next_ceiling = WAY_BELOW;
                        let mut look = y - 1;
                        while look >= self.min_y - 1 {
                            let below = if look >= self.min_y {
                                blocks[(look - self.min_y) as usize * COLUMNS + column]
                            } else {
                                self.air
                            };
                            if below == self.air || below == self.water || below == self.lava {
                                next_ceiling = look + 1;
                                break;
                            }
                            look -= 1;
                        }
                    }
                    walk.stone_below = y.wrapping_sub(next_ceiling).wrapping_add(1);
                    walk.stone_above += 1;
                    let replaced = try_apply(&self.rule, &mut job, ctx, &walk, bx, y, bz);
                    if let Some(state) = replaced {
                        blocks[(y - self.min_y) as usize * COLUMNS + column] = state;
                    }
                    y -= 1;
                }
            }
        }
    }
}

fn read(path: PathBuf) -> Result<Value> {
    let raw =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))
}

/// Column walk state handed to surface conditions.
struct Walk {
    stone_above: i32,
    stone_below: i32,
    water_height: i32,
    gradient_x: i32,
    gradient_z: i32,
}

struct SurfaceJob<'a> {
    terrain: &'a NoiseTerrain,
    depth: HashMap<(i32, i32), i32>,
    secondary: HashMap<(i32, i32), f64>,
    min_level: HashMap<(i32, i32), i32>,
}

impl SurfaceJob<'_> {
    fn surface_depth(&mut self, x: i32, z: i32) -> i32 {
        if let Some(depth) = self.depth.get(&(x, z)) {
            return *depth;
        }
        let noise = f64::from(
            self.terrain
                .surface_stack
                .get(f64::from(x), 0.0, f64::from(z)),
        );
        let random = self.terrain.world.at(x, 0, z).next_f64();
        let depth = (noise * 2.75 + 3.0 + random * 0.25) as i32;
        self.depth.insert((x, z), depth);
        depth
    }

    fn surface_secondary(&mut self, x: i32, z: i32) -> f64 {
        if let Some(v) = self.secondary.get(&(x, z)) {
            return *v;
        }
        let v = f64::from(
            self.terrain
                .secondary_stack
                .get(f64::from(x), 0.0, f64::from(z)),
        );
        self.secondary.insert((x, z), v);
        v
    }

    fn min_surface_level(&mut self, ctx: &mut ChunkCtx, x: i32, z: i32) -> i32 {
        if let Some(level) = self.min_level.get(&(x, z)) {
            return *level;
        }
        let sample = sample_node(
            &self.terrain.graph,
            ctx,
            self.terrain.chunk_surface_level,
            x,
            0,
            z,
        );
        let level = sample.floor() as i32 + self.surface_depth(x, z) - 8;
        self.min_level.insert((x, z), level);
        level
    }
}

fn try_apply(
    rule: &Rule,
    job: &mut SurfaceJob,
    ctx: &mut ChunkCtx,
    walk: &Walk,
    x: i32,
    y: i32,
    z: i32,
) -> Option<u32> {
    match rule {
        Rule::Sequence(rules) => {
            for r in rules {
                if let Some(state) = try_apply(r, job, ctx, walk, x, y, z) {
                    return Some(state);
                }
            }
            None
        }
        Rule::Condition(cond, then) => {
            if cond_holds(cond, job, ctx, walk, x, y, z) {
                try_apply(then, job, ctx, walk, x, y, z)
            } else {
                None
            }
        }
        Rule::Block(state) => Some(*state),
        Rule::Bands => None,
        Rule::OreVein(vein) => {
            let density = sample_node(&job.terrain.graph, ctx, vein.density, x, y, z);
            if density <= 0.0 {
                return None;
            }
            let mut random = job.terrain.ore_pos.at(x, y, z);
            if next_f32(&mut random) > density {
                return None;
            }
            let richness = sample_node(&job.terrain.graph, ctx, vein.richness, x, y, z);
            if next_f32(&mut random) < richness
                && sample_node(&job.terrain.graph, ctx, vein.filler_gap, x, y, z) < 0.0
            {
                return Some(if next_f32(&mut random) < vein.raw_chance {
                    vein.raw
                } else {
                    vein.ore
                });
            }
            Some(vein.filler)
        }
    }
}

fn cond_holds(
    cond: &Cond,
    job: &mut SurfaceJob,
    ctx: &mut ChunkCtx,
    walk: &Walk,
    x: i32,
    y: i32,
    z: i32,
) -> bool {
    match cond {
        Cond::StoneDepth {
            ceiling,
            offset,
            add_surface,
            secondary_range,
        } => {
            let depth = if *ceiling {
                walk.stone_below
            } else {
                walk.stone_above
            };
            let surface = if *add_surface {
                job.surface_depth(x, z)
            } else {
                0
            };
            let secondary = if *secondary_range == 0 {
                0
            } else {
                unclamped_map(
                    job.surface_secondary(x, z),
                    -1.0,
                    1.0,
                    0.0,
                    f64::from(*secondary_range),
                ) as i32
            };
            depth <= 1 + offset + surface + secondary
        }
        Cond::Water {
            offset,
            multiplier,
            add_stone,
        } => {
            let depth = if *add_stone { walk.stone_above } else { 0 };
            walk.water_height == i32::MIN
                || y + depth >= walk.water_height + offset + job.surface_depth(x, z) * multiplier
        }
        Cond::YAbove {
            anchor_y,
            multiplier,
            add_stone,
        } => {
            let depth = if *add_stone { walk.stone_above } else { 0 };
            y + depth >= anchor_y + job.surface_depth(x, z) * multiplier
        }
        Cond::VerticalGradient {
            name,
            true_below,
            false_above,
        } => {
            if y <= *true_below {
                return true;
            }
            if y >= *false_above {
                return false;
            }
            let probability = unclamped_map(
                f64::from(y),
                f64::from(*true_below),
                f64::from(*false_above),
                1.0,
                0.0,
            );
            let mut random = job.terrain.gradient_factories[name].at(x, y, z);
            f64::from(next_f32(&mut random)) < probability
        }
        Cond::NoiseThreshold {
            stack,
            min,
            max,
            three_d,
        } => {
            let value = if *three_d {
                stack.get(f64::from(x), f64::from(y), f64::from(z))
            } else {
                stack.get(f64::from(x), 0.0, f64::from(z))
            };
            let value = f64::from(value);
            value >= *min && value <= *max
        }
        Cond::Biome(hit) => *hit,
        Cond::AbovePreliminary => y >= job.min_surface_level(ctx, x, z),
        Cond::Not(inner) => !cond_holds(inner, job, ctx, walk, x, y, z),
        Cond::Hole => job.surface_depth(x, z) <= 0,
        Cond::Steep => walk.gradient_x <= -4 || walk.gradient_z >= 4,
    }
}

// ---------------------------------------------------------------------------
// Pins location.
// ---------------------------------------------------------------------------

/// Walks up from the working directory and the executable to find the
/// pinned worldgen configs.
pub fn locate_pins() -> Result<PathBuf> {
    let mut roots = vec![std::env::current_dir().context("working directory")?];
    if let Ok(exe) = std::env::current_exe() {
        roots.push(exe.parent().map(Path::to_path_buf).context("exe parent")?);
    }
    for root in roots {
        for ancestor in root.ancestors() {
            let pins = ancestor.join("pins").join("worldgen");
            if pins.join("noise_settings").join("overworld.json").is_file() {
                return Ok(pins);
            }
        }
    }
    bail!("worldgen pins not found; run from the repository root")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)]
    use super::*;

    fn pins() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/worldgen")
    }

    fn registry() -> BlockRegistry {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/blocks.json");
        BlockRegistry::load(&path).expect("block registry pins")
    }

    fn loader() -> Loader<'static> {
        let world = world_positional(42);
        Loader {
            pins: Box::leak(pins().into_boxed_path()),
            world,
            registry: Box::leak(Box::new(registry())),
            min_y: -64,
            height: 384,
            graph: Graph {
                nodes: Vec::new(),
                uses_y: Vec::new(),
            },
            stacks: HashMap::new(),
            functions: HashMap::new(),
            gradient_factories: HashMap::new(),
        }
    }

    /// The duplicated stack construction must match the reference
    /// octave implementation sample for sample.
    #[test]
    fn stack_noise_matches_octave_noise() {
        const MODIFIERS: &[f64] = &[1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0];
        let spec = crate::noise::OctaveSpec {
            base_octave: -9,
            octave_count: 9,
            base_amplitude: 0.8880832896205223,
            amplitude_modifiers: MODIFIERS,
        };
        let mut l = loader();
        let stack = l.stack("minecraft:continentalness").unwrap();
        let world = world_positional(42);
        let reference = crate::noise::OctaveNoise::new(
            &spec,
            &mut world.from_name("minecraft:continentalness"),
        );
        for &(x, y, z) in &[
            (0.0f64, 0.0f64, 0.0f64),
            (100.0, -35.0, 12.0),
            (1234.0, 567.0, -89.0),
            (-4321.0, 64.0, 999.0),
        ] {
            assert_eq!(stack.get(x, y, z), reference.sample_3d(x, y, z));
        }
    }

    fn manual_graph(nodes: Vec<Node>) -> Graph {
        let mut graph = Graph {
            nodes,
            uses_y: Vec::new(),
        };
        graph.finish();
        graph
    }

    fn spline_value(points: Vec<SplinePoint>, input: f32) -> f32 {
        let graph = manual_graph(vec![
            Node::Constant(input),
            Node::Spline(SplineNode {
                coordinate: 0,
                points,
            }),
        ]);
        let mut ctx = ChunkCtx::new(0, 0, -64, 384);
        sample_node(&graph, &mut ctx, 1, 0, 0, 0)
    }

    fn flat_point(location: f32, value: f32, derivative: f32) -> SplinePoint {
        SplinePoint {
            location,
            derivative,
            value: SplineValue::Fixed(value),
        }
    }

    /// Spline: knot values exact, flat derivatives give straight lerp,
    /// and the ends hold when the end derivatives are flat.
    #[test]
    fn spline_knots_and_hermite() {
        let points = vec![flat_point(0.0, 10.0, 0.0), flat_point(4.0, 20.0, 0.0)];
        assert_eq!(spline_value(points.clone(), 0.0), 10.0);
        assert_eq!(spline_value(points.clone(), 4.0), 20.0);
        assert_eq!(spline_value(points.clone(), 2.0), 15.0);
        assert_eq!(spline_value(points.clone(), -5.0), 10.0, "holds below");
        assert_eq!(spline_value(points, 9.0), 20.0, "holds above");
    }

    /// Spline derivatives bend the segment between the endpoints.
    #[test]
    fn spline_derivative_bends() {
        // Segment 10 -> 20 over [0, 4] with d1 = 2, d2 = -2:
        // a = 2*4 - 10 = -2, b = 2*4 + 10 = 18; at t = 0.5 the
        // correction adds 0.25 * lerp(0.5, -2, 18) = 2.
        let points = vec![flat_point(0.0, 10.0, 2.0), flat_point(4.0, 20.0, -2.0)];
        assert_eq!(spline_value(points.clone(), 2.0), 17.0);
        assert_eq!(spline_value(points, 1.0), 13.0625);
    }

    /// Gradients clamp at their coordinate span.
    #[test]
    fn gradient_clamps() {
        let graph = manual_graph(vec![Node::Gradient {
            axis: 1,
            from_coord: -64,
            to_coord: 320,
            from_value: 1.5,
            to_value: -1.5,
        }]);
        let mut ctx = ChunkCtx::new(0, 0, -64, 384);
        assert_eq!(sample_node(&graph, &mut ctx, 0, 0, -64, 0), 1.5);
        assert_eq!(sample_node(&graph, &mut ctx, 0, 0, -100, 0), 1.5);
        assert_eq!(sample_node(&graph, &mut ctx, 0, 0, 320, 0), -1.5);
        assert_eq!(sample_node(&graph, &mut ctx, 0, 0, 400, 0), -1.5);
        // Midpoint: 1.5 + 192 * (-3.0 / 384.0)
        assert_eq!(sample_node(&graph, &mut ctx, 0, 0, 128, 0), 0.0);
    }

    /// Cache wraps its input transparently and memoizes per position.
    #[test]
    fn cache_matches_inner() {
        let graph = manual_graph(vec![
            Node::Gradient {
                axis: 0,
                from_coord: 0,
                to_coord: 64,
                from_value: 2.0,
                to_value: 0.0,
            },
            Node::Cache { input: 0 },
        ]);
        let mut ctx = ChunkCtx::new(0, 0, -64, 384);
        let a = sample_node(&graph, &mut ctx, 1, 3, 0, 7);
        let b = sample_node(&graph, &mut ctx, 1, 3, 99, 7);
        let direct = sample_node(&graph, &mut ctx, 0, 3, 5, 7);
        assert_eq!(a, b, "y-independent input memoizes by column");
        assert_eq!(a, direct);
    }

    /// Interpolation trilerps between cell corners.
    #[test]
    fn interpolated_trilerps() {
        let graph = manual_graph(vec![
            Node::Gradient {
                axis: 1,
                from_coord: 0,
                to_coord: 64,
                from_value: 0.0,
                to_value: 64.0,
            },
            Node::Interpolated {
                input: 0,
                cell_xz: 4,
                cell_y: 8,
            },
        ]);
        let mut ctx = ChunkCtx::new(0, 0, 0, 384);
        assert_eq!(sample_node(&graph, &mut ctx, 1, 0, 8, 0), 8.0);
        assert_eq!(sample_node(&graph, &mut ctx, 1, 0, 4, 0), 4.0);
        assert_eq!(sample_node(&graph, &mut ctx, 1, 0, 2, 0), 2.0);
    }

    /// The surface probe walks down the cell grid to the first solid.
    #[test]
    fn find_top_surface_probes() {
        let graph = manual_graph(vec![
            Node::Constant(0.5),
            Node::Constant(100.0),
            Node::Constant(-0.5),
            Node::FindTopSurface {
                density: 0,
                upper: 1,
                lower: -64,
                cell_height: 8,
            },
            Node::FindTopSurface {
                density: 2,
                upper: 1,
                lower: -64,
                cell_height: 8,
            },
        ]);
        let mut ctx = ChunkCtx::new(0, 0, -64, 384);
        assert_eq!(sample_node(&graph, &mut ctx, 3, 5, 3, 7), 96.0);
        assert_eq!(sample_node(&graph, &mut ctx, 4, 5, 3, 7), -64.0);
    }

    /// The blended field stays inside its limit-stack range and repeats.
    #[test]
    fn blended_field_bounded_and_deterministic() {
        let world = world_positional(42);
        let mut rng = world.from_name("minecraft:terrain");
        let field = BlendedField::new(&mut rng, 0.25, 0.125, 80.0, 160.0, 8.0);
        let mut again = world.from_name("minecraft:terrain");
        let twin = BlendedField::new(&mut again, 0.25, 0.125, 80.0, 160.0, 8.0);
        for i in 0..64 {
            let x = f64::from(i) * 97.0;
            let y = f64::from(i) * -13.0;
            let z = f64::from(i) * 31.0;
            let a = field.sample(x, y, z);
            assert_eq!(a, twin.sample(x, y, z), "deterministic at {i}");
            assert!(a.abs() <= 2.5, "value {a} out of range at {i}");
        }
    }

    /// Local-vanilla coast diagnostic: reads the spawn regions saved by the
    /// parity capture and reports where the engines split on land vs water.
    #[test]
    #[ignore = "needs the saved vanilla spawn world; run explicitly"]
    fn coast_diagnostic() {
        use crate::anvil_to_wire::unpack;
        use crate::Region;
        let world_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/vanilla/run/world/dimensions/minecraft/overworld/region");
        let regions = [
            world_dir.join("r.-1.-1.mca"),
            world_dir.join("r.-1.0.mca"),
            world_dir.join("r.0.-1.mca"),
            world_dir.join("r.0.0.mca"),
        ];
        // Chunk coords -> per-column top non-air, ground (top non-fluid),
        // surface biome, and per-section palette cells for block lookups.
        struct VanillaChunk {
            top: [i32; 256],
            ground: [i32; 256],
            biomes: Vec<String>,
            sections: Vec<(i32, Vec<u16>, Vec<String>)>,
        }
        let mut vanilla: HashMap<(i32, i32), VanillaChunk> = HashMap::new();
        let name_at = |c: &VanillaChunk, y: i32, column: usize| -> String {
            let sy = y.div_euclid(16);
            let cell = (y.rem_euclid(16) as usize) * 256 + column;
            for (section_y, cells, palette) in &c.sections {
                if *section_y == sy && cell < cells.len() {
                    return palette
                        .get(cells[cell] as usize)
                        .cloned()
                        .unwrap_or_default();
                }
            }
            String::new()
        };
        // 26.3 palette entries come in two shapes: the older bare-name
        // string and the newer compound with an empty or "id" key.
        let entry_name = |entry: &crate::PaletteEntry| -> String {
            match entry {
                crate::PaletteEntry::Other(fastnbt::Value::Compound(map)) => map
                    .get("id")
                    .or_else(|| map.get(""))
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                other => other.name().to_string(),
            }
        };
        for path in &regions {
            let Ok(region) = Region::open(path) else {
                continue;
            };
            for rx in 0..32usize {
                for rz in 0..32usize {
                    let Ok(Some(chunk)) = region.chunk(rx, rz) else {
                        continue;
                    };
                    let mut top = [-i32::MAX; 256];
                    let mut ground = [-i32::MAX; 256];
                    // (section y, palette indices, palette names)
                    let mut sections: Vec<(i32, Vec<u16>, Vec<String>)> = Vec::new();
                    for section in &chunk.sections {
                        let Some(states) = &section.block_states else {
                            continue;
                        };
                        let palette_bits =
                            (((states.palette.len().max(2) - 1).ilog2() + 1).max(4)) as usize;
                        let cells: Vec<u16> = match &states.data {
                            None => vec![0u16; 4096],
                            Some(data) => {
                                let longs: Vec<u64> = data.iter().map(|&v| v as u64).collect();
                                unpack(&longs, palette_bits, 4096).into_iter().collect()
                            }
                        };
                        let palette: Vec<String> = states.palette.iter().map(entry_name).collect();
                        sections.push((i32::from(section.y), cells, palette));
                    }
                    let is_air = |name: &str| {
                        matches!(
                            name,
                            "minecraft:air" | "minecraft:cave_air" | "minecraft:void_air"
                        )
                    };
                    let is_water = |name: &str| name == "minecraft:water";
                    for (sy, cells, palette) in &sections {
                        let masks: Vec<(bool, bool)> =
                            palette.iter().map(|n| (is_air(n), is_water(n))).collect();
                        for ly in (0..16usize).rev() {
                            let y = sy * 16 + ly as i32;
                            if !(-64..=319).contains(&y) {
                                continue;
                            }
                            for column in 0..256usize {
                                let cell = ly * 256 + column;
                                let (a, w) = masks[cells[cell] as usize];
                                if !a {
                                    top[column] = top[column].max(y);
                                    if !w {
                                        ground[column] = ground[column].max(y);
                                    }
                                }
                            }
                        }
                    }
                    // Surface biome name from the top biome layer of the
                    // highest section.
                    let mut names = vec![String::new(); 256];
                    let mut biome_layers: Vec<(i32, Vec<String>)> = chunk
                        .sections
                        .iter()
                        .filter_map(|s| {
                            let b = s.biomes.as_ref()?;
                            if b.data.is_none() {
                                return Some((
                                    i32::from(s.y),
                                    vec![b.palette.first().cloned().unwrap_or_default(); 64],
                                ));
                            }
                            let longs: Vec<u64> =
                                b.data.as_ref().unwrap().iter().map(|&v| v as u64).collect();
                            let bits = (((b.palette.len().max(2) - 1).ilog2() + 1).max(4)) as usize;
                            Some((
                                i32::from(s.y),
                                unpack(&longs, bits, 64)
                                    .into_iter()
                                    .map(|i| b.palette.get(i as usize).cloned().unwrap_or_default())
                                    .collect(),
                            ))
                        })
                        .collect();
                    biome_layers.sort_by_key(|l| -l.0);
                    if let Some((_, biomes)) = biome_layers.first() {
                        if biomes.len() == 64 {
                            for (column, name) in names.iter_mut().enumerate() {
                                let bx = (column % 16) / 4;
                                let bz = (column / 16) / 4;
                                *name = biomes[3 * 16 + bz * 4 + bx].clone();
                            }
                        }
                    }
                    vanilla.insert(
                        (chunk.x, chunk.z),
                        VanillaChunk {
                            top,
                            ground,
                            biomes: names,
                            sections,
                        },
                    );
                }
            }
        }
        assert!(!vanilla.is_empty(), "no saved vanilla chunks");

        let reg = registry();
        let pin_root = pins();
        let terrain = NoiseTerrain::with_seed(42, &reg, &pin_root).expect("engine build");
        let mut loader = Loader::new(&pin_root, 42, &reg, -64, 384);
        let continents = loader
            .function_id("minecraft:overworld/continents")
            .unwrap();
        let erosion = loader.function_id("minecraft:overworld/erosion").unwrap();
        let ridges = loader.function_id("minecraft:overworld/ridges").unwrap();
        let folded = loader
            .function_id("minecraft:overworld/ridges_folded")
            .unwrap();
        let offset = loader.function_id("minecraft:overworld/offset").unwrap();
        let cheese = loader
            .function_id("minecraft:overworld/sloped_cheese")
            .unwrap();
        let base3d = loader
            .function_id("minecraft:overworld/base_3d_noise")
            .unwrap();
        let entrances = loader
            .function_id("minecraft:overworld/caves/entrances")
            .unwrap();
        let depth_fn = loader.function_id("minecraft:overworld/depth").unwrap();
        let noodle_fn = loader
            .function_id("minecraft:overworld/caves/noodle")
            .unwrap();
        let final_fn = loader
            .function_id("minecraft:overworld/final_density")
            .unwrap();
        loader.graph.finish();
        let mut climate_ctx = ChunkCtx::new(0, 0, -64, 384);

        let mut failing_chunks: Vec<(i32, i32)> = Vec::new();
        let mut failing_columns = 0usize;
        let mut fail_vanilla_top = std::collections::BTreeMap::new();
        let mut fail_vanilla_ground = std::collections::BTreeMap::new();
        let mut fail_our_ground = std::collections::BTreeMap::new();
        let count_at = |map: &mut std::collections::BTreeMap<i32, usize>, y: i32| {
            *map.entry(y).or_default() += 1;
        };
        let mut ground_deltas: Vec<i32> = Vec::new();
        let mut ocean_floor_deltas: std::collections::BTreeMap<i32, usize> =
            std::collections::BTreeMap::new();
        let mut density_says_land = 0usize;
        // (x, z, vanilla top, vanilla ground, our ground, density top,
        // continents, erosion, ridges, ridges folded, offset)
        #[allow(clippy::type_complexity)]
        let mut samples: Vec<(i32, i32, i32, i32, i32, i32, f32, f32, f32, f32, f32)> = Vec::new();
        let mut biomes_of_fail: HashMap<String, usize> = HashMap::new();
        let mut cheese_floor_hist: std::collections::BTreeMap<i32, usize> =
            std::collections::BTreeMap::new();
        let mut block_at_y: std::collections::BTreeMap<(i32, String), usize> =
            std::collections::BTreeMap::new();
        let mut ladder_above_floor = 0usize;
        let mut noodle_negative_above_floor = 0usize;
        let mut total = 0usize;
        let mut land_vanilla = 0usize;
        let mut land_ours = 0usize;
        let mut agree = 0usize;
        let mut keys: Vec<(i32, i32)> = vanilla.keys().copied().collect();
        keys.sort_unstable();
        for &(cx, cz) in &keys {
            let vc = &vanilla[&(cx, cz)];
            let vtop = &vc.top;
            let vground = &vc.ground;
            let vbiome = &vc.biomes;
            let blocks = terrain.fill_chunk(cx, cz);
            let mut our_top = [-i32::MAX; 256];
            let mut our_ground = [-i32::MAX; 256];
            for y in 0..terrain.height {
                for column in 0..256 {
                    let state = blocks[y as usize * 256 + column];
                    if state != terrain.air {
                        our_top[column] = our_top[column].max(terrain.min_y + y);
                        if state != terrain.water {
                            our_ground[column] = our_ground[column].max(terrain.min_y + y);
                        }
                    }
                }
            }
            let mut chunk_fails = 0usize;
            for column in 0..256 {
                let land_v = vtop[column] >= 64;
                let land_o = our_top[column] >= 64;
                total += 1;
                land_vanilla += usize::from(land_v);
                land_ours += usize::from(land_o);
                agree += usize::from(land_v == land_o);
                if vground[column] > 40 && our_ground[column] > 40 {
                    let shared_ocean = vtop[column] <= 63 && our_top[column] <= 63;
                    if shared_ocean {
                        count_at(
                            &mut ocean_floor_deltas,
                            vground[column] - our_ground[column],
                        );
                    } else {
                        ground_deltas.push(vground[column] - our_ground[column]);
                    }
                }
                if !(land_v && !land_o) {
                    continue;
                }
                chunk_fails += 1;
                failing_columns += 1;
                count_at(&mut fail_vanilla_top, vtop[column]);
                count_at(&mut fail_vanilla_ground, vground[column]);
                count_at(&mut fail_our_ground, our_ground[column]);
                *biomes_of_fail.entry(vbiome[column].clone()).or_default() += 1;
                for y in [62, 63, 64] {
                    *block_at_y.entry((y, name_at(vc, y, column))).or_default() += 1;
                }
                {
                    let wx = cx * 16 + (column % 16) as i32;
                    let wz = cz * 16 + (column / 16) as i32;
                    let g = our_ground[column];
                    let mut ladder_floor = g;
                    for y in (g - 4..=g + 8).rev() {
                        let cheese_v =
                            sample_node(&loader.graph, &mut climate_ctx, cheese, wx, y, wz);
                        let ent5_v = 5.0
                            * sample_node(&loader.graph, &mut climate_ctx, entrances, wx, y, wz);
                        if cheese_v.min(ent5_v) > 0.0 {
                            ladder_floor = y;
                            break;
                        }
                    }
                    count_at(&mut cheese_floor_hist, ladder_floor);
                    if ladder_floor > g {
                        ladder_above_floor += 1;
                    }
                    if sample_node(&loader.graph, &mut climate_ctx, noodle_fn, wx, g + 1, wz) < 0.0
                    {
                        noodle_negative_above_floor += 1;
                    }
                }
                if chunk_fails == 130 {
                    let wx = cx * 16 + (column % 16) as i32;
                    let wz = cz * 16 + (column / 16) as i32;
                    let start_x = cx * 16;
                    let start_z = cz * 16;
                    let mut ctx = ChunkCtx::new(start_x, start_z, -64, 384);
                    let mut dtop = -64;
                    for y in (0..384).rev() {
                        let d = f64::from(sample_node(
                            &terrain.graph,
                            &mut ctx,
                            terrain.final_density,
                            wx,
                            y - 64,
                            wz,
                        ));
                        if d > 0.0 {
                            dtop = y - 64;
                            break;
                        }
                    }
                    if dtop >= 64 {
                        density_says_land += 1;
                    }
                    samples.push((
                        wx,
                        wz,
                        vtop[column],
                        vground[column],
                        our_ground[column],
                        dtop,
                        sample_node(&loader.graph, &mut climate_ctx, continents, wx, 0, wz),
                        sample_node(&loader.graph, &mut climate_ctx, erosion, wx, 0, wz),
                        sample_node(&loader.graph, &mut climate_ctx, ridges, wx, 0, wz),
                        sample_node(&loader.graph, &mut climate_ctx, folded, wx, 0, wz),
                        sample_node(&loader.graph, &mut climate_ctx, offset, wx, 0, wz),
                    ));
                }
            }
            if chunk_fails > 0 {
                failing_chunks.push((cx, cz));
            }
        }
        println!("[diag] chunks={} columns={total}", keys.len());
        println!(
            "[diag] vanilla land {land_vanilla} our land {land_ours} agree {agree} ({:.1}%)",
            100.0 * agree as f64 / total as f64
        );
        println!(
            "[diag] failing columns: {failing_columns} across {} chunks",
            failing_chunks.len()
        );
        let show = |label: &str, map: &std::collections::BTreeMap<i32, usize>| {
            print!("[diag] {label}:");
            for (y, n) in map {
                print!(" {y}:{n}");
            }
            println!();
        };
        show("vanilla top hist", &fail_vanilla_top);
        show("our ground hist", &fail_our_ground);
        show("vanilla ground hist", &fail_vanilla_ground);
        show("cheese-ladder floor hist", &cheese_floor_hist);
        println!(
            "[diag] ladder floor above ours: {ladder_above_floor}/{failing_columns}; noodle negative at our floor+1: {noodle_negative_above_floor}/{failing_columns}"
        );
        if !ground_deltas.is_empty() {
            let mut sorted = ground_deltas.clone();
            sorted.sort_unstable();
            let n = sorted.len();
            println!(
                "[diag] ground delta (vanilla - ours, near-surface columns): median {:+} p10 {:+} p90 {:+} (n={n})",
                sorted[n / 2],
                sorted[n / 10],
                sorted[(n * 9) / 10]
            );
        }
        show("shared-ocean floor delta hist", &ocean_floor_deltas);
        for y in [62, 63, 64] {
            let mut names: Vec<(usize, &String)> = block_at_y
                .iter()
                .filter(|((yy, _), _)| *yy == y)
                .map(|((_, n), c)| (*c, n))
                .collect();
            names.sort_unstable_by_key(|(c, _)| std::cmp::Reverse(*c));
            let summary: Vec<String> = names
                .iter()
                .take(5)
                .map(|(c, n)| format!("{n} x{c}"))
                .collect();
            println!(
                "[diag] failing-column blocks at y={y}: {}",
                summary.join(", ")
            );
        }
        println!(
            "[diag] sampled failing columns where density alone makes land: {density_says_land}/{}",
            samples.len()
        );
        let mut biomes: Vec<(usize, String)> =
            biomes_of_fail.into_iter().map(|(k, v)| (v, k)).collect();
        biomes.sort_unstable_by_key(|(n, _)| std::cmp::Reverse(*n));
        for (n, name) in biomes.iter().take(4) {
            println!("[diag] failing-column biome: {name} x{n}");
        }
        for s in samples.iter().take(14) {
            println!(
                "[diag] ({},{}) vtop={} vground={} ourground={} dtop={} C={:.4} E={:.4} W={:.4} PV={:.4} off={:.4}",
                s.0, s.1, s.2, s.3, s.4, s.5, s.6, s.7, s.8, s.9, s.10
            );
        }
        // Per-layer term breakdown at the first sample column.
        if let Some(s) = samples.first() {
            let (wx, wz) = (s.0, s.1);
            let start_x = wx.div_euclid(16) * 16;
            let start_z = wz.div_euclid(16) * 16;
            let mut ctx = ChunkCtx::new(start_x, start_z, -64, 384);
            println!("[diag] terms at ({wx},{wz}): y final cheese base3d ent ent5 depth");
            for y in 58..=72 {
                let g = &loader.graph;
                println!(
                    "[diag]   {y} {:>8.4} {:>8.4} {:>7.4} {:>7.4} {:>7.4} {:>7.4}",
                    sample_node(g, &mut ctx, final_fn, wx, y, wz),
                    sample_node(g, &mut ctx, cheese, wx, y, wz),
                    sample_node(g, &mut ctx, base3d, wx, y, wz),
                    sample_node(g, &mut ctx, entrances, wx, y, wz),
                    5.0 * sample_node(g, &mut ctx, entrances, wx, y, wz),
                    sample_node(g, &mut ctx, depth_fn, wx, y, wz),
                );
            }
        }
    }

    /// Full-graph smoke: the pinned configs produce a sane chunk and a
    /// repeatable one.
    #[test]
    fn engine_fill_smoke() {
        let reg = registry();
        let terrain = NoiseTerrain::with_seed(42, &reg, &pins()).expect("engine build");
        assert_eq!(terrain.sea_level, 63);
        assert_eq!(terrain.min_y, -64);
        assert_eq!(terrain.height, 384);
        let blocks = terrain.fill_chunk(0, 0);
        assert_eq!(blocks.len(), 384 * 256);
        let stone = blocks.iter().filter(|&&b| b == terrain.stone).count();
        assert!(stone > 4096, "chunk carries a stone body ({stone})");
        let top = terrain.column_top(0, 0);
        assert!(
            (terrain.min_y + 1..terrain.min_y + terrain.height - 1).contains(&top),
            "column top {top} in range"
        );
        let repeat = terrain.fill_chunk(0, 0);
        assert_eq!(blocks, repeat, "same seed refills identically");
    }

    /// Carve agreement against the captured vanilla chunks: air-like,
    /// water, and lava cells below y=50, full-column material deltas, and
    /// the y bands where moss and clay land on each side. Requires the
    /// worldgen capture a parity gate run leaves in target/vanilla.
    #[test]
    #[ignore = "needs the worldgen capture; run after a parity gate run"]
    fn carve_diagnostic() {
        use crate::anvil_to_wire::unpack;
        use crate::chunk_codec::{Container, WireChunk};
        let capture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/vanilla/worldgen-capture");
        let Ok(entries) = std::fs::read_dir(&capture) else {
            panic!("no worldgen capture under target/vanilla");
        };
        let reg = registry();
        let terrain = NoiseTerrain::with_seed(42, &reg, &pins()).expect("engine build");
        let name_of = |id: u32| -> &str { reg.state_of(id).map(|(name, _)| name).unwrap_or("?") };
        let class_of = |name: &str| -> u8 {
            match name {
                "minecraft:air" | "minecraft:cave_air" | "minecraft:void_air" => 1,
                "minecraft:water" => 2,
                "minecraft:lava" => 3,
                _ => 0,
            }
        };
        let cells_of = |chunk: &WireChunk| -> Vec<Vec<u32>> {
            chunk
                .sections
                .iter()
                .map(|section| match &section.block_states {
                    Container::Single(v) => vec![*v; 4096],
                    Container::Palette {
                        entries,
                        longs,
                        bits,
                    } => unpack(longs, *bits as usize, 4096)
                        .into_iter()
                        .map(|i| entries[i as usize])
                        .collect(),
                    Container::Global { longs, bits } => unpack(longs, *bits as usize, 4096)
                        .into_iter()
                        .map(|i| i as u32)
                        .collect(),
                })
                .collect()
        };
        let mut vanilla: HashMap<(i32, i32), WireChunk> = HashMap::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("bin") {
                continue;
            }
            let Ok(body) = std::fs::read(&path) else {
                continue;
            };
            if body.len() < 64 || body.len() > 2_000_000 {
                continue;
            }
            let x = i32::from_be_bytes([body[0], body[1], body[2], body[3]]);
            let z = i32::from_be_bytes([body[4], body[5], body[6], body[7]]);
            if x.abs() > 48 || z.abs() > 48 {
                continue;
            }
            let Ok(chunk) = WireChunk::decode(&body) else {
                continue;
            };
            if chunk.sections.len() != 24 {
                continue;
            }
            vanilla.entry((chunk.x, chunk.z)).or_insert(chunk);
        }
        assert!(!vanilla.is_empty(), "capture held no chunk packets");

        let mut class_totals = [(0u64, 0u64); 4];
        let mut air_agree = 0u64;
        let mut exact = 0u64;
        let mut below = 0u64;
        let mut probes = 0u64;
        let mut per_chunk: Vec<((i32, i32), u64, u64)> = Vec::new();
        let mut state_delta: HashMap<String, i64> = HashMap::new();
        let mut moss_ours: [u64; 24] = [0; 24];
        let mut moss_vanilla: [u64; 24] = [0; 24];
        let mut keys: Vec<(i32, i32)> = vanilla.keys().copied().collect();
        keys.sort_unstable();
        for (cx, cz) in keys {
            let theirs = cells_of(&vanilla[&(cx, cz)]);
            let ours = terrain.fill_chunk(cx, cz);
            let mut agree = 0u64;
            let mut total = 0u64;
            for y in terrain.min_y..50 {
                let sy = ((y - terrain.min_y) / 16) as usize;
                let ly = y.rem_euclid(16) as usize;
                let cells = &theirs[sy];
                for column in 0..256usize {
                    let mine = ours[sy * 4096 + ly * 256 + column];
                    let their = cells[ly * 256 + column];
                    let mc = class_of(name_of(mine));
                    let tc = class_of(name_of(their));
                    class_totals[mc as usize].0 += 1;
                    class_totals[tc as usize].1 += 1;
                    if (mc == 1) == (tc == 1) {
                        agree += 1;
                    }
                    if mc == tc {
                        exact += 1;
                    } else if probes < 12 {
                        probes += 1;
                        let wx = cx * 16 + (column % 16) as i32;
                        let wz = cz * 16 + (column / 16) as i32;
                        eprintln!(
                            "[carve] ({wx},{y},{wz}) chunk ({cx},{cz}): ours {} vanilla {}",
                            name_of(mine),
                            name_of(their)
                        );
                    }
                    total += 1;
                }
            }
            below += total;
            air_agree += agree;
            per_chunk.push(((cx, cz), agree, total));
            for (sy, cells) in theirs.iter().enumerate() {
                for cell in 0..4096usize {
                    let mine = name_of(ours[sy * 4096 + cell]);
                    let their = name_of(cells[cell]);
                    if mine != their {
                        *state_delta.entry(their.to_string()).or_insert(0) += 1;
                        *state_delta.entry(mine.to_string()).or_insert(0) -= 1;
                    }
                    let mossy = |n: &str| n == "minecraft:moss_block" || n == "minecraft:clay";
                    if mossy(mine) {
                        moss_ours[sy] += 1;
                    }
                    if mossy(their) {
                        moss_vanilla[sy] += 1;
                    }
                }
            }
        }
        let class_names = ["solid", "air", "water", "lava"];
        for (class, (mine, theirs)) in class_totals.iter().enumerate() {
            eprintln!(
                "[carve] {:>6}: ours {mine:>7} vanilla {theirs:>7}",
                class_names[class]
            );
        }
        eprintln!(
            "[carve] below y=50: air-mask agreement {:.3}% exact-class {:.3}% over {below} cells",
            100.0 * air_agree as f64 / below as f64,
            100.0 * exact as f64 / below as f64
        );
        let mut deltas: Vec<(i64, String)> = state_delta.into_iter().map(|(k, v)| (v, k)).collect();
        deltas.sort_by_key(|&(d, _)| -d.abs());
        for (delta, name) in deltas.iter().take(12) {
            eprintln!("[carve] delta {delta:+7} {name}");
        }
        for sy in 0..24 {
            if moss_ours[sy] > 0 || moss_vanilla[sy] > 0 {
                eprintln!(
                    "[carve] moss+clay y={:>4}: ours {:>6} vanilla {:>6}",
                    -64 + sy as i32 * 16,
                    moss_ours[sy],
                    moss_vanilla[sy]
                );
            }
        }
        per_chunk.sort_by_key(|&(_, a, _)| a);
        for &((cx, cz), a, t) in per_chunk.iter().take(8) {
            eprintln!("[carve] worst ({cx},{cz}): {a}/{t}");
        }
    }
}
