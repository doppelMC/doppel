//! Cave and canyon carving: pinned configs drive a sweep over nearby source
//! chunks whose tunnels and canyons mark ellipsoid cells into a per-chunk
//! mask; the mask is applied later against the aquifer's fluid decision.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::Path;
use std::sync::OnceLock;

use crate::noise::Lcg48;

const EDGE: i32 = 16;
/// Source chunks whose carvers can reach the target chunk.
const SOURCE_RADIUS: i32 = 8;
/// Carver reach in chunks; the step budget derives from it.
const RANGE: i32 = 4;
const MAX_DISTANCE: i32 = (RANGE * 2 - 1) * EDGE;
/// Layers at the top of the world that stay untouched.
const TOP_PROTECTION: i32 = 7;
const SIN_SCALE: f64 = 10430.378350470453;

// --- trigonometry ----------------------------------------------------------

static SIN: OnceLock<Vec<f32>> = OnceLock::new();

fn sin_table() -> &'static [f32] {
    SIN.get_or_init(|| {
        (0..65536)
            .map(|i| ((i as f64) / SIN_SCALE).sin() as f32)
            .collect()
    })
}

fn mth_sin(x: f64) -> f32 {
    sin_table()[(((x * SIN_SCALE) as i64) & 0xFFFF) as usize]
}

fn mth_cos(x: f64) -> f32 {
    sin_table()[(((x * SIN_SCALE + 16384.0) as i64) & 0xFFFF) as usize]
}

// --- config draws ------------------------------------------------------------

enum FloatDraw {
    Constant(f32),
    Uniform { min: f32, max: f32 },
    Trapezoid { min: f32, max: f32, plateau: f32 },
}

impl FloatDraw {
    fn parse(v: &Value) -> Result<FloatDraw> {
        if let Some(value) = v.as_f64() {
            return Ok(FloatDraw::Constant(value as f32));
        }
        let kind = str_field(v, "type")?;
        match kind {
            "minecraft:constant" => Ok(FloatDraw::Constant(float_field(v, "value")?)),
            "minecraft:uniform" => Ok(FloatDraw::Uniform {
                min: float_field(v, "min_inclusive")?,
                max: float_field(v, "max_exclusive")?,
            }),
            "minecraft:trapezoid" => Ok(FloatDraw::Trapezoid {
                min: float_field(v, "min")?,
                max: float_field(v, "max")?,
                plateau: float_field(v, "plateau")?,
            }),
            other => bail!("float draw type {other}"),
        }
    }

    fn sample(&self, rng: &mut Lcg48) -> f32 {
        match *self {
            FloatDraw::Constant(value) => value,
            FloatDraw::Uniform { min, max } => rng.next_f32() * (max - min) + min,
            FloatDraw::Trapezoid { min, max, plateau } => {
                let range = max - min;
                let plateau_start = (range - plateau) / 2.0;
                let plateau_end = range - plateau_start;
                min + rng.next_f32() * plateau_end + rng.next_f32() * plateau_start
            }
        }
    }
}

enum IntDraw {
    Constant(i32),
    VeryBiasedToBottom { min: i32, max: i32 },
    Uniform { min: i32, max: i32 },
}

impl IntDraw {
    fn parse(v: &Value) -> Result<IntDraw> {
        if let Some(value) = v.as_i64() {
            return Ok(IntDraw::Constant(value as i32));
        }
        let kind = str_field(v, "type")?;
        match kind {
            "minecraft:constant" => Ok(IntDraw::Constant(int_field(v, "value")?)),
            "minecraft:very_biased_to_bottom" => Ok(IntDraw::VeryBiasedToBottom {
                min: int_field(v, "min_inclusive")?,
                max: int_field(v, "max_inclusive")?,
            }),
            "minecraft:uniform" => Ok(IntDraw::Uniform {
                min: int_field(v, "min_inclusive")?,
                max: int_field(v, "max_inclusive")?,
            }),
            other => bail!("int draw type {other}"),
        }
    }

    fn sample(&self, rng: &mut Lcg48) -> i32 {
        match *self {
            IntDraw::Constant(value) => value,
            IntDraw::VeryBiasedToBottom { min, max } => {
                let span = max - min + 1;
                let inner = rng.next_int(span);
                let middle = rng.next_int(inner + 1);
                min + rng.next_int(middle + 1)
            }
            IntDraw::Uniform { min, max } => rng.next_int(max - min + 1) + min,
        }
    }
}

/// A uniform height range between two resolved anchors.
struct HeightRange {
    min: i32,
    max: i32,
}

impl HeightRange {
    fn parse(v: &Value, min_gen_y: i32, gen_depth: i32) -> Result<HeightRange> {
        let kind = str_field(v, "type")?;
        if kind != "minecraft:uniform" {
            bail!("height range type {kind}");
        }
        let anchor = |key: &str| -> Result<i32> {
            let a = v.get(key).context("height range anchor")?;
            if let Some(y) = a.get("absolute").and_then(Value::as_i64) {
                return Ok(y as i32);
            }
            if let Some(off) = a.get("above_bottom").and_then(Value::as_i64) {
                return Ok(min_gen_y + off as i32);
            }
            if let Some(off) = a.get("below_top").and_then(Value::as_i64) {
                return Ok(gen_depth - 1 + min_gen_y - off as i32);
            }
            bail!("height anchor {a}")
        };
        Ok(HeightRange {
            min: anchor("min_inclusive")?,
            max: anchor("max_inclusive")?,
        })
    }

    fn sample(&self, rng: &mut Lcg48) -> i32 {
        rng.next_int(self.max - self.min + 1) + self.min
    }
}

// --- configs -----------------------------------------------------------------

struct CaveCfg {
    probability: f32,
    y: HeightRange,
    count: IntDraw,
    thickness: FloatDraw,
    weird_bias: bool,
    room_v: FloatDraw,
    h_mult: FloatDraw,
    v_mult: FloatDraw,
    start_v: FloatDraw,
    floor_level: FloatDraw,
}

struct CanyonCfg {
    probability: f32,
    y: HeightRange,
    v_rotation: FloatDraw,
    distance_factor: FloatDraw,
    thickness: FloatDraw,
    width_smoothness: i32,
    h_radius_factor: FloatDraw,
    v_default: f32,
    v_center: f32,
    y_scale: FloatDraw,
}

enum Carver {
    Cave(CaveCfg),
    Canyon(CanyonCfg),
}

impl Carver {
    fn probability(&self) -> f32 {
        match self {
            Carver::Cave(cfg) => cfg.probability,
            Carver::Canyon(cfg) => cfg.probability,
        }
    }

    fn parse(v: &Value, min_gen_y: i32, gen_depth: i32) -> Result<Carver> {
        let kind = str_field(v, "type")?;
        match kind {
            "minecraft:cave" => Ok(Carver::Cave(CaveCfg {
                probability: float_field(v, "probability")?,
                y: HeightRange::parse(obj_field(v, "y")?, min_gen_y, gen_depth)?,
                count: IntDraw::parse(obj_field(v, "count")?)?,
                thickness: FloatDraw::parse(obj_field(v, "thickness")?)?,
                weird_bias: v
                    .get("weird_thickness_bias")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                room_v: FloatDraw::parse(obj_field(v, "room_vertical_radius_multiplier")?)?,
                h_mult: FloatDraw::parse(obj_field(v, "horizontal_radius_multiplier")?)?,
                v_mult: FloatDraw::parse(obj_field(v, "vertical_radius_multiplier")?)?,
                start_v: match v.get("start_vertical_radius_multiplier") {
                    Some(node) => FloatDraw::parse(node)?,
                    None => FloatDraw::Constant(1.0),
                },
                floor_level: FloatDraw::parse(obj_field(v, "floor_level")?)?,
            })),
            "minecraft:canyon" => {
                let shape = obj_field(v, "shape")?;
                Ok(Carver::Canyon(CanyonCfg {
                    probability: float_field(v, "probability")?,
                    y: HeightRange::parse(obj_field(v, "y")?, min_gen_y, gen_depth)?,
                    v_rotation: FloatDraw::parse(obj_field(v, "vertical_rotation")?)?,
                    distance_factor: FloatDraw::parse(obj_field(shape, "distance_factor")?)?,
                    thickness: FloatDraw::parse(obj_field(shape, "thickness")?)?,
                    width_smoothness: int_field(shape, "width_smoothness")?,
                    h_radius_factor: FloatDraw::parse(obj_field(
                        shape,
                        "horizontal_radius_factor",
                    )?)?,
                    v_default: float_field(shape, "vertical_radius_default_factor")?,
                    v_center: float_field(shape, "vertical_radius_center_factor")?,
                    y_scale: FloatDraw::parse(obj_field(shape, "y_scale")?)?,
                }))
            }
            other => bail!("carver type {other}"),
        }
    }
}

// --- mask --------------------------------------------------------------------

/// Marked carve cells for one chunk; bits run y-fastest, then z, then x.
pub(crate) struct CarveMask {
    min_y: i32,
    height: i32,
    bits: Vec<u64>,
}

impl CarveMask {
    fn new(min_y: i32, max_y: i32) -> CarveMask {
        let height = (max_y - min_y + 1) as usize;
        CarveMask {
            min_y,
            height: height as i32,
            bits: vec![0u64; (height * 256).div_ceil(64)],
        }
    }

    fn min_y(&self) -> i32 {
        self.min_y
    }

    fn max_y(&self) -> i32 {
        self.min_y + self.height - 1
    }

    fn carve(&mut self, x: i32, y: i32, z: i32) {
        let column = (x as usize) << 4 | z as usize;
        let index = (y - self.min_y) as usize + column * self.height as usize;
        self.bits[index / 64] |= 1u64 << (index % 64);
    }

    fn get(&self, index: usize) -> bool {
        self.bits[index / 64] & (1u64 << (index % 64)) != 0
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }

    /// Visits each column's contiguous y runs in bit order (x-major, z
    /// inner, runs ascending).
    pub(crate) fn for_each_column(&self, mut visit: impl FnMut(usize, usize, i32, i32)) {
        for column in 0..256usize {
            let base = column * self.height as usize;
            let mut y = 0;
            while y < self.height as usize {
                if !self.get(base + y) {
                    y += 1;
                    continue;
                }
                let bottom = y;
                while y < self.height as usize && self.get(base + y) {
                    y += 1;
                }
                visit(
                    column >> 4,
                    column & 0xF,
                    self.min_y + bottom as i32,
                    self.min_y + y as i32 - 1,
                );
            }
        }
    }
}

// --- carving geometry ----------------------------------------------------------

/// The chunk receiving carve marks.
struct Target {
    min_x: i32,
    min_z: i32,
    middle_x: i32,
    middle_z: i32,
}

/// Per-ellipsoid cell rejection rules: caves keep a floor band, canyons
/// widen with height.
enum Skip {
    Cave {
        floor_level: f64,
    },
    Canyon {
        width_factors: Vec<f32>,
        min_gen_y: i32,
    },
}

impl Skip {
    fn should_skip(&self, xd: f64, yd: f64, zd: f64, y: i32) -> bool {
        match self {
            Skip::Cave { floor_level } => yd <= *floor_level || xd * xd + yd * yd + zd * zd >= 1.0,
            Skip::Canyon {
                width_factors,
                min_gen_y,
            } => {
                let index = (y - *min_gen_y) as usize;
                (xd * xd + zd * zd) * f64::from(width_factors[index - 1]) + yd * yd / 6.0 >= 1.0
            }
        }
    }
}

fn can_reach(
    target: &Target,
    x: f64,
    z: f64,
    current_step: i32,
    total_steps: i32,
    thickness: f32,
) -> bool {
    let xd = x - f64::from(target.middle_x);
    let zd = z - f64::from(target.middle_z);
    let remaining = f64::from(total_steps - current_step);
    let rr = f64::from(thickness + 2.0 + 16.0);
    xd * xd + zd * zd - remaining * remaining <= rr * rr
}

#[allow(clippy::too_many_arguments)]
fn carve_ellipsoid(
    target: &Target,
    x: f64,
    y: f64,
    z: f64,
    horizontal_radius: f64,
    vertical_radius: f64,
    mask: &mut CarveMask,
    skip: &Skip,
) {
    let max_delta = 16.0 + horizontal_radius * 2.0;
    if (x - f64::from(target.middle_x)).abs() > max_delta
        || (z - f64::from(target.middle_z)).abs() > max_delta
    {
        return;
    }
    let min_x = (((x - horizontal_radius).floor() as i32) - target.min_x - 1).max(0);
    let max_x = (((x + horizontal_radius).floor() as i32) - target.min_x).min(EDGE - 1);
    let min_z = (((z - horizontal_radius).floor() as i32) - target.min_z - 1).max(0);
    let max_z = (((z + horizontal_radius).floor() as i32) - target.min_z).min(EDGE - 1);
    let min_y = (((y - vertical_radius).floor() as i32) - 1).max(mask.min_y());
    let max_y = (((y + vertical_radius).floor() as i32) + 1).min(mask.max_y());
    for xi in min_x..=max_x {
        let world_x = target.min_x + xi;
        let xd = (f64::from(world_x) + 0.5 - x) / horizontal_radius;
        for zi in min_z..=max_z {
            let world_z = target.min_z + zi;
            let zd = (f64::from(world_z) + 0.5 - z) / horizontal_radius;
            if xd * xd + zd * zd >= 1.0 {
                continue;
            }
            let mut world_y = max_y;
            while world_y > min_y {
                let yd = (f64::from(world_y) - 0.5 - y) / vertical_radius;
                if !skip.should_skip(xd, yd, zd, world_y) {
                    mask.carve(xi, world_y, zi);
                }
                world_y -= 1;
            }
        }
    }
}

// --- cave driver ----------------------------------------------------------------

impl CaveCfg {
    fn carve(
        &self,
        rng: &mut Lcg48,
        target: &Target,
        source_x: i32,
        source_z: i32,
        mask: &mut CarveMask,
    ) {
        let cave_count = self.count.sample(rng);
        for _ in 0..cave_count {
            let x = f64::from(source_x * EDGE + rng.next_int(16));
            let y = f64::from(self.y.sample(rng));
            let z = f64::from(source_z * EDGE + rng.next_int(16));
            let h_mult = f64::from(self.h_mult.sample(rng));
            let v_mult = f64::from(self.v_mult.sample(rng));
            let start_v = f64::from(self.start_v.sample(rng));
            let floor_level = f64::from(self.floor_level.sample(rng));
            let mut tunnels = 1;
            if rng.next_int(4) == 0 {
                let y_scale = f64::from(self.room_v.sample(rng));
                let thickness = 1.0f32 + rng.next_f32() * 6.0;
                create_room(target, x, y, z, thickness, y_scale, mask, floor_level);
                tunnels += rng.next_int(4);
            }
            for _ in 0..tunnels {
                let h_rotation = rng.next_f32() * (std::f32::consts::PI * 2.0);
                let v_rotation = (rng.next_f32() - 0.5) / 4.0;
                let thickness = self.draw_thickness(rng);
                let distance = MAX_DISTANCE - rng.next_int(MAX_DISTANCE / 4);
                create_tunnel(
                    target,
                    rng.next_long(),
                    x,
                    y,
                    z,
                    h_mult,
                    v_mult,
                    thickness,
                    h_rotation,
                    v_rotation,
                    0,
                    distance,
                    start_v,
                    mask,
                    floor_level,
                );
            }
        }
    }

    fn draw_thickness(&self, rng: &mut Lcg48) -> f32 {
        let mut thickness = self.thickness.sample(rng);
        if self.weird_bias && rng.next_int(10) == 0 {
            thickness *= rng.next_f32() * rng.next_f32() * 3.0 + 1.0;
        }
        thickness
    }
}

#[allow(clippy::too_many_arguments)]
fn create_room(
    target: &Target,
    x: f64,
    y: f64,
    z: f64,
    thickness: f32,
    y_scale: f64,
    mask: &mut CarveMask,
    floor_level: f64,
) {
    let horizontal = 1.5 + f64::from(mth_sin(f64::from(std::f32::consts::FRAC_PI_2)) * thickness);
    let vertical = horizontal * y_scale;
    let skip = Skip::Cave { floor_level };
    carve_ellipsoid(target, x + 1.0, y, z, horizontal, vertical, mask, &skip);
}

#[allow(clippy::too_many_arguments)]
fn create_tunnel(
    target: &Target,
    tunnel_seed: i64,
    x: f64,
    y: f64,
    z: f64,
    h_mult: f64,
    v_mult: f64,
    thickness: f32,
    horizontal_rotation: f32,
    vertical_rotation: f32,
    step: i32,
    dist: i32,
    y_scale: f64,
    mask: &mut CarveMask,
    floor_level: f64,
) {
    let mut rng = Lcg48::new(tunnel_seed);
    let split_point = rng.next_int(dist / 2) + dist / 4;
    let steep = rng.next_int(6) == 0;
    let mut y_rota = 0.0f32;
    let mut x_rota = 0.0f32;
    let mut x = x;
    let mut y = y;
    let mut z = z;
    let mut h_rotation = horizontal_rotation;
    let mut v_rotation = vertical_rotation;
    let skip = Skip::Cave { floor_level };
    for current_step in step..dist {
        let horizontal_radius = 1.5
            + f64::from(
                mth_sin(f64::from(
                    std::f32::consts::PI * current_step as f32 / dist as f32,
                )) * thickness,
            );
        let vertical_radius = horizontal_radius * y_scale;
        let cos_x = mth_cos(f64::from(v_rotation));
        x += f64::from(mth_cos(f64::from(h_rotation)) * cos_x);
        y += f64::from(mth_sin(f64::from(v_rotation)));
        z += f64::from(mth_sin(f64::from(h_rotation)) * cos_x);
        v_rotation *= if steep { 0.92 } else { 0.7 };
        v_rotation += x_rota * 0.1;
        h_rotation += y_rota * 0.1;
        x_rota *= 0.9;
        y_rota *= 0.75;
        x_rota += (rng.next_f32() - rng.next_f32()) * rng.next_f32() * 2.0;
        y_rota += (rng.next_f32() - rng.next_f32()) * rng.next_f32() * 4.0;
        if current_step == split_point && thickness > 1.0 {
            let seed_a = rng.next_long();
            let thickness_a = rng.next_f32() * 0.5 + 0.5;
            let seed_b = rng.next_long();
            let thickness_b = rng.next_f32() * 0.5 + 0.5;
            create_tunnel(
                target,
                seed_a,
                x,
                y,
                z,
                h_mult,
                v_mult,
                thickness_a,
                h_rotation - std::f32::consts::FRAC_PI_2,
                v_rotation / 3.0,
                current_step,
                dist,
                1.0,
                mask,
                floor_level,
            );
            create_tunnel(
                target,
                seed_b,
                x,
                y,
                z,
                h_mult,
                v_mult,
                thickness_b,
                h_rotation + std::f32::consts::FRAC_PI_2,
                v_rotation / 3.0,
                current_step,
                dist,
                1.0,
                mask,
                floor_level,
            );
            return;
        }
        if rng.next_int(4) == 0 {
            continue;
        }
        if !can_reach(target, x, z, current_step, dist, thickness) {
            return;
        }
        carve_ellipsoid(
            target,
            x,
            y,
            z,
            horizontal_radius * h_mult,
            vertical_radius * v_mult,
            mask,
            &skip,
        );
    }
}

// --- canyon driver ----------------------------------------------------------------

impl CanyonCfg {
    #[allow(clippy::too_many_arguments)]
    fn carve(
        &self,
        rng: &mut Lcg48,
        target: &Target,
        source_x: i32,
        source_z: i32,
        mask: &mut CarveMask,
        min_gen_y: i32,
        gen_depth: i32,
    ) {
        let x = f64::from(source_x * EDGE + rng.next_int(16));
        let y = f64::from(self.y.sample(rng));
        let z = f64::from(source_z * EDGE + rng.next_int(16));
        let h_rotation = rng.next_f32() * (std::f32::consts::PI * 2.0);
        let v_rotation = self.v_rotation.sample(rng);
        let y_scale = f64::from(self.y_scale.sample(rng));
        let thickness = self.thickness.sample(rng);
        let distance = (MAX_DISTANCE as f32 * self.distance_factor.sample(rng)) as i32;
        self.do_carve(
            target,
            rng.next_long(),
            x,
            y,
            z,
            thickness,
            h_rotation,
            v_rotation,
            0,
            distance,
            y_scale,
            mask,
            min_gen_y,
            gen_depth,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn do_carve(
        &self,
        target: &Target,
        tunnel_seed: i64,
        x: f64,
        y: f64,
        z: f64,
        thickness: f32,
        horizontal_rotation: f32,
        vertical_rotation: f32,
        step: i32,
        distance: i32,
        y_scale: f64,
        mask: &mut CarveMask,
        min_gen_y: i32,
        gen_depth: i32,
    ) {
        let mut rng = Lcg48::new(tunnel_seed);
        let width_factors = init_width_factors(gen_depth, self.width_smoothness, &mut rng);
        let mut y_rota = 0.0f32;
        let mut x_rota = 0.0f32;
        let mut x = x;
        let mut y = y;
        let mut z = z;
        let mut h_rotation = horizontal_rotation;
        let mut v_rotation = vertical_rotation;
        let skip = Skip::Canyon {
            width_factors,
            min_gen_y,
        };
        for current_step in step..distance {
            let h_radius = 1.5
                + f64::from(
                    mth_sin(f64::from(
                        current_step as f32 * std::f32::consts::PI / distance as f32,
                    )) * thickness,
                );
            let vertical_radius = h_radius * y_scale;
            let h_radius = h_radius * f64::from(self.h_radius_factor.sample(&mut rng));
            let vertical_radius = update_vertical_radius(
                &mut rng,
                vertical_radius,
                distance as f32,
                current_step as f32,
                self.v_default,
                self.v_center,
            );
            let xc = mth_cos(f64::from(v_rotation));
            let xs = mth_sin(f64::from(v_rotation));
            x += f64::from(mth_cos(f64::from(h_rotation)) * xc);
            y += f64::from(xs);
            z += f64::from(mth_sin(f64::from(h_rotation)) * xc);
            v_rotation *= 0.7;
            v_rotation += x_rota * 0.05;
            h_rotation += y_rota * 0.05;
            x_rota *= 0.8;
            y_rota *= 0.5;
            x_rota += (rng.next_f32() - rng.next_f32()) * rng.next_f32() * 2.0;
            y_rota += (rng.next_f32() - rng.next_f32()) * rng.next_f32() * 4.0;
            if rng.next_int(4) == 0 {
                continue;
            }
            if !can_reach(target, x, z, current_step, distance, thickness) {
                return;
            }
            carve_ellipsoid(target, x, y, z, h_radius, vertical_radius, mask, &skip);
        }
    }
}

fn init_width_factors(gen_depth: i32, smoothness: i32, rng: &mut Lcg48) -> Vec<f32> {
    let mut out = Vec::with_capacity(gen_depth as usize);
    let mut width = 1.0f32;
    for y_index in 0..gen_depth {
        if y_index == 0 || rng.next_int(smoothness) == 0 {
            width = 1.0 + rng.next_f32() * rng.next_f32();
        }
        out.push(width * width);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn update_vertical_radius(
    rng: &mut Lcg48,
    vertical_radius: f64,
    distance: f32,
    current_step: f32,
    v_default: f32,
    v_center: f32,
) -> f64 {
    let vertical_multiplier = 1.0 - (0.5 - current_step / distance).abs() * 2.0;
    let factor = v_default + v_center * vertical_multiplier;
    f64::from(factor) * vertical_radius * f64::from(rng.next_f32() * 0.25 + 0.75)
}

// --- source sweep -----------------------------------------------------------------

/// The carver set: pinned configs plus the per-biome carver lists that pick
/// them, seeded by the world. Source-chunk biomes memoize because adjacent
/// target chunks sweep overlapping neighborhoods.
pub(crate) struct Carvers {
    entries: Vec<Option<Carver>>,
    per_biome: Vec<Vec<Option<usize>>>,
    world_seed: i64,
    min_y: i32,
    height: i32,
    biome_memo: std::cell::RefCell<std::collections::HashMap<(i32, i32), u32>>,
}

fn large_feature_seed(seed: i64, chunk_x: i32, chunk_z: i32) -> Lcg48 {
    let mut rng = Lcg48::new(seed);
    let x_scale = rng.next_long();
    let z_scale = rng.next_long();
    let result =
        (chunk_x as i64).wrapping_mul(x_scale) ^ (chunk_z as i64).wrapping_mul(z_scale) ^ seed;
    Lcg48::new(result)
}

impl Carvers {
    /// Loads the pinned carver configs and binds each biome's carver list to
    /// them. List entries without a pinned config keep their position but
    /// stay inert.
    pub(crate) fn load(pins: &Path, world_seed: i64, min_y: i32, height: i32) -> Result<Carvers> {
        let order: Vec<String> = serde_json::from_str(
            &std::fs::read_to_string(pins.join("biome_registry_order.json"))
                .context("biome registry order")?,
        )
        .context("parsing biome registry order")?;
        let mut keys: Vec<String> = Vec::new();
        let mut per_biome = Vec::with_capacity(order.len());
        for name in &order {
            let key = name.strip_prefix("minecraft:").unwrap_or(name);
            let raw = std::fs::read_to_string(pins.join("biome").join(format!("{key}.json")))
                .with_context(|| format!("biome {name}"))?;
            let biome: Value =
                serde_json::from_str(&raw).with_context(|| format!("parsing biome {name}"))?;
            let list: Vec<String> = biome
                .get("carvers")
                .and_then(Value::as_array)
                .map(|entries| {
                    entries
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default();
            let mut slots = Vec::with_capacity(list.len());
            for carver_key in &list {
                let slot = keys.iter().position(|k| k == carver_key);
                if slot.is_none() {
                    keys.push(carver_key.clone());
                }
                slots.push(slot);
            }
            per_biome.push(slots);
        }
        let mut entries = Vec::with_capacity(keys.len());
        for key in &keys {
            let file = key.strip_prefix("minecraft:").unwrap_or(key);
            let raw = std::fs::read_to_string(pins.join("carver").join(format!("{file}.json")));
            let entry = match raw {
                Ok(raw) => {
                    let v: Value = serde_json::from_str(&raw)
                        .with_context(|| format!("parsing carver {key}"))?;
                    Some(Carver::parse(&v, min_y, height)?)
                }
                Err(_) => None,
            };
            entries.push(entry);
        }
        Ok(Carvers {
            entries,
            per_biome,
            world_seed,
            min_y,
            height,
            biome_memo: std::cell::RefCell::new(std::collections::HashMap::new()),
        })
    }

    fn biome_of_chunk(
        &self,
        biome_of: &mut dyn FnMut(i32, i32) -> u32,
        source_x: i32,
        source_z: i32,
    ) -> u32 {
        if let Some(biome) = self.biome_memo.borrow().get(&(source_x, source_z)) {
            return *biome;
        }
        let biome = biome_of(source_x * EDGE, source_z * EDGE);
        self.biome_memo
            .borrow_mut()
            .insert((source_x, source_z), biome);
        biome
    }

    /// Sweeps the source-chunk neighborhood and marks every carved cell of
    /// the target chunk. The biome resolver samples the source chunk's
    /// min-corner climate.
    pub(crate) fn build_mask(
        &self,
        biome_of: &mut dyn FnMut(i32, i32) -> u32,
        cx: i32,
        cz: i32,
    ) -> CarveMask {
        let mut mask = CarveMask::new(
            self.min_y + 1,
            self.min_y + self.height - 1 - TOP_PROTECTION,
        );
        let target = Target {
            min_x: cx * EDGE,
            min_z: cz * EDGE,
            middle_x: cx * EDGE + 8,
            middle_z: cz * EDGE + 8,
        };
        for dx in -SOURCE_RADIUS..=SOURCE_RADIUS {
            for dz in -SOURCE_RADIUS..=SOURCE_RADIUS {
                let source_x = cx + dx;
                let source_z = cz + dz;
                let biome = self.biome_of_chunk(biome_of, source_x, source_z);
                let Some(list) = self.per_biome.get(biome as usize) else {
                    continue;
                };
                for (index, slot) in list.iter().enumerate() {
                    let carver = slot
                        .and_then(|i| self.entries.get(i))
                        .and_then(|entry| entry.as_ref());
                    let Some(carver) = carver else {
                        continue;
                    };
                    let mut rng =
                        large_feature_seed(self.world_seed + index as i64, source_x, source_z);
                    if rng.next_f32() <= carver.probability() {
                        match carver {
                            Carver::Cave(cfg) => {
                                cfg.carve(&mut rng, &target, source_x, source_z, &mut mask)
                            }
                            Carver::Canyon(cfg) => cfg.carve(
                                &mut rng,
                                &target,
                                source_x,
                                source_z,
                                &mut mask,
                                self.min_y,
                                self.height,
                            ),
                        }
                    }
                }
            }
        }
        mask
    }
}

// --- json helpers -----------------------------------------------------------------

fn str_field<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v.get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("carver field {key}"))
}

fn float_field(v: &Value, key: &str) -> Result<f32> {
    v.get(key)
        .and_then(Value::as_f64)
        .map(|x| x as f32)
        .with_context(|| format!("carver field {key}"))
}

fn int_field(v: &Value, key: &str) -> Result<i32> {
    v.get(key)
        .and_then(Value::as_i64)
        .map(|x| x as i32)
        .with_context(|| format!("carver field {key}"))
}

fn obj_field<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
    v.get(key).with_context(|| format!("carver field {key}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pins() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/worldgen")
    }

    fn fnv1a(cells: &[(i32, i32, i32)]) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for &(x, y, z) in cells {
            for byte in x
                .to_le_bytes()
                .into_iter()
                .chain(y.to_le_bytes())
                .chain(z.to_le_bytes())
            {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        hash
    }

    /// The sweep reproduces an independent model of the reference carve
    /// chain cell for cell at seed 42 (all sources on the plains carver
    /// list): count, y extent, and digest per target chunk.
    #[test]
    fn mask_matches_reference_model() {
        let carvers = Carvers::load(&pins(), 42, -64, 384).expect("carver pins");
        let order: Vec<String> = serde_json::from_str(
            &std::fs::read_to_string(pins().join("biome_registry_order.json")).unwrap(),
        )
        .unwrap();
        let plains = order
            .iter()
            .position(|name| name == "minecraft:plains")
            .unwrap() as u32;
        let expected = [
            (0i32, 0i32, 5377usize, 19i32, 129i32, 0x74a58b19dff4f31bu64),
            (3, -2, 3309, -11, 185, 0x9e2ce90c0f841981u64),
            (-5, 7, 722, -41, 85, 0xc0bb134073782e8eu64),
        ];
        for (cx, cz, count, y_lo, y_hi, digest) in expected {
            let mask = carvers.build_mask(&mut |_, _| plains, cx, cz);
            let mut cells = Vec::new();
            mask.for_each_column(&mut |x: usize, z: usize, bottom: i32, top: i32| {
                for y in bottom..=top {
                    cells.push((x as i32, y, z as i32));
                }
            });
            let lo = cells.iter().map(|c| c.1).min().unwrap();
            let hi = cells.iter().map(|c| c.1).max().unwrap();
            assert_eq!(cells.len(), count, "cell count at ({cx},{cz})");
            assert_eq!((lo, hi), (y_lo, y_hi), "y extent at ({cx},{cz})");
            assert_eq!(fnv1a(&cells), digest, "digest at ({cx},{cz})");
        }
    }

    /// Marked cells sit inside the mask bounds and the column visitor
    /// reproduces exactly the marked set.
    #[test]
    fn mask_column_visitor_covers_marked_cells() {
        let mut mask = CarveMask::new(-63, 312);
        for &(x, y, z) in &[(0, -63, 0), (15, 312, 15), (3, 40, 9), (3, 44, 9)] {
            mask.carve(x, y, z);
        }
        let mut visited = std::collections::HashSet::new();
        mask.for_each_column(&mut |x: usize, z: usize, bottom: i32, top: i32| {
            assert!(bottom <= top);
            for y in bottom..=top {
                assert!(visited.insert((x as i32, y, z as i32)));
            }
        });
        assert_eq!(visited.len(), 4);
        assert!(visited.contains(&(3, 44, 9)));
        assert!(!visited.contains(&(3, 41, 9)), "gaps split into runs");
    }

    /// The trig table hits exact values at the cardinal points.
    #[test]
    fn trig_table_cardinals() {
        assert_eq!(mth_sin(0.0), 0.0);
        assert_eq!(mth_cos(0.0), 1.0);
        assert!((mth_sin(f64::from(std::f32::consts::FRAC_PI_2)) - 1.0).abs() < 1e-6);
        assert!((mth_cos(f64::from(std::f32::consts::PI)) + 1.0).abs() < 1e-6);
    }
}
