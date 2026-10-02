//! Multi-noise biome selection: the pinned overworld climate parameter
//! table and the quantized distance search that turns a climate sample
//! into a biome wire id.
//!
//! The search narrows every axis to integer 10-thousandths before any
//! comparison, and the table keeps generation order so the first row with
//! strictly smaller fitness wins ties, mirroring the reference walk over
//! the parameter list.

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::Value;

/// Climate axes in parameter order: temperature, humidity (vegetation),
/// continentalness, erosion, depth, weirdness.
pub const AXIS_COUNT: usize = 6;

const AXIS_FIELDS: [&str; AXIS_COUNT] = [
    "temperature",
    "humidity",
    "continentalness",
    "erosion",
    "depth",
    "weirdness",
];

/// One climate sample, every axis scaled by 10 000 and truncated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetPoint {
    pub axes: [i64; AXIS_COUNT],
}

/// Scales and truncates one axis the way the reference quantizer does:
/// single-precision multiply, then integer truncation toward zero.
pub fn quantize_axis(value: f32) -> i64 {
    (value * 10000.0f32) as i64
}

/// A closed parameter span on one axis, in quantized units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub min: i64,
    pub max: i64,
}

impl Span {
    /// How far the target sits outside the span; zero inside it.
    pub fn distance(&self, target: i64) -> i64 {
        let above = target - self.max;
        if above > 0 {
            above
        } else {
            (self.min - target).max(0)
        }
    }
}

/// One table row: six axis spans, the offset penalty, and the biome.
#[derive(Clone)]
struct Row {
    spans: [Span; AXIS_COUNT],
    offset: i64,
    biome: u32,
}

/// The overworld parameter list resolved to biome wire ids. Rows stay in
/// generation order because the search keeps the first best match.
pub struct BiomeTable {
    rows: Vec<Row>,
}

impl BiomeTable {
    /// Reads the pinned overworld parameter list and maps each biome name
    /// to its wire id through the pinned registry order.
    pub fn load(pins: &Path) -> Result<BiomeTable> {
        let list_path = pins
            .join("multi_noise_biome_source_parameter_list")
            .join("overworld.json");
        let list: Value = serde_json::from_str(&fs::read_to_string(&list_path)?)
            .with_context(|| format!("parameter list {}", list_path.display()))?;
        let order_path = pins.join("biome_registry_order.json");
        let order: Vec<String> = serde_json::from_str(&fs::read_to_string(&order_path)?)
            .with_context(|| format!("biome registry order {}", order_path.display()))?;
        let id_of = |name: &str| -> Result<u32> {
            order
                .iter()
                .position(|candidate| candidate == name)
                .map(|i| i as u32)
                .with_context(|| format!("biome {name} missing from registry order"))
        };

        let entries = list
            .get("biomes")
            .and_then(Value::as_array)
            .context("parameter list biomes array")?;
        if entries.is_empty() {
            bail!("parameter list has no rows");
        }
        let mut rows = Vec::with_capacity(entries.len());
        for entry in entries {
            let name = entry
                .get("biome")
                .and_then(Value::as_str)
                .context("row biome name")?;
            let params = entry.get("parameters").context("row parameters")?;
            let mut spans = [Span { min: 0, max: 0 }; AXIS_COUNT];
            for (slot, key) in AXIS_FIELDS.iter().enumerate() {
                let pair = params
                    .get(*key)
                    .and_then(Value::as_array)
                    .with_context(|| format!("axis {key}"))?;
                if pair.len() != 2 {
                    bail!("axis {key} needs a min/max pair");
                }
                let narrow = |v: &Value| -> Result<i64> {
                    let number = v.as_f64().with_context(|| format!("axis {key} bound"))?;
                    Ok(quantize_axis(number as f32))
                };
                let min = narrow(&pair[0])?;
                let max = narrow(&pair[1])?;
                if min > max {
                    bail!("axis {key} has min {min} above max {max}");
                }
                spans[slot] = Span { min, max };
            }
            let offset = entry
                .pointer("/parameters/offset")
                .and_then(Value::as_f64)
                .context("row offset")?;
            rows.push(Row {
                spans,
                offset: quantize_axis(offset as f32),
                biome: id_of(name)?,
            });
        }
        Ok(BiomeTable { rows })
    }

    /// The first row's biome; every search returns at least this.
    pub fn fallback(&self) -> u32 {
        self.rows[0].biome
    }

    /// The number of table rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the table holds no rows (never true for a loaded table).
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The biome whose parameter point fits the target best; earlier rows
    /// win ties.
    pub fn find(&self, target: &TargetPoint) -> u32 {
        let mut best = self.rows[0].biome;
        let mut best_fitness = i64::MAX;
        for row in &self.rows {
            let mut fitness = row.offset * row.offset;
            for slot in 0..AXIS_COUNT {
                let distance = row.spans[slot].distance(target.axes[slot]);
                fitness += distance * distance;
            }
            if fitness < best_fitness {
                best_fitness = fitness;
                best = row.biome;
            }
        }
        best
    }

    /// Quantizes raw axis samples and resolves the biome.
    pub fn find_axes(&self, axes: [f32; AXIS_COUNT]) -> u32 {
        let mut target = [0i64; AXIS_COUNT];
        for (slot, value) in axes.iter().enumerate() {
            target[slot] = quantize_axis(*value);
        }
        self.find(&TargetPoint { axes: target })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pins() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/worldgen")
    }

    fn table() -> BiomeTable {
        BiomeTable::load(&pins()).expect("parameter table")
    }

    fn span_of(min: f32, max: f32) -> Span {
        Span {
            min: quantize_axis(min),
            max: quantize_axis(max),
        }
    }

    /// Quantization truncates toward zero in single precision, the way the
    /// reference cast does; hand-checked values include the negative case
    /// that a floor would get wrong.
    #[test]
    fn quantization_truncates_toward_zero() {
        assert_eq!(quantize_axis(0.1), 1000);
        assert_eq!(quantize_axis(-0.15), -1500);
        assert_eq!(quantize_axis(1.2), 12000);
        assert_eq!(quantize_axis(-1.2), -12000);
        // -0.5 truncates to zero, not -1.
        assert_eq!(quantize_axis(-0.00005), 0);
        assert_eq!(quantize_axis(0.0), 0);
    }

    /// Span distance is zero inside the span, linear outside it, from both
    /// sides, in quantized units.
    #[test]
    fn span_distance_shape() {
        let span = span_of(-1.0, 1.0);
        assert_eq!(span.distance(0), 0);
        assert_eq!(span.distance(10000), 0);
        assert_eq!(span.distance(15000), 5000);
        assert_eq!(span.distance(-15000), 5000);
        let point = span_of(0.5, 0.5);
        assert_eq!(point.distance(5000), 0);
        assert_eq!(point.distance(0), 5000);
        assert_eq!(point.distance(12000), 7000);
    }

    /// The search walks the table in order and keeps the first strictly
    /// better row, so equal fitness leaves the earlier biome in place.
    #[test]
    fn search_keeps_first_best() {
        // A row that misses the target on every axis lists first; the row
        // that contains the target wins despite listing later.
        let wide = Row {
            spans: [span_of(0.5, 2.0); AXIS_COUNT],
            offset: 0,
            biome: 10,
        };
        let mut exact_spans = [span_of(-2.0, 2.0); AXIS_COUNT];
        exact_spans[0] = span_of(0.0, 0.0);
        let exact = Row {
            spans: exact_spans,
            offset: 0,
            biome: 20,
        };
        let table = BiomeTable {
            rows: vec![wide.clone(), exact.clone()],
        };
        let target = TargetPoint {
            axes: [0, 0, 0, 0, 0, 0],
        };
        assert_eq!(table.find(&target), 20);

        // Two rows with identical spans: the first stays.
        let twin = Row {
            spans: exact_spans,
            offset: 0,
            biome: 30,
        };
        let table = BiomeTable {
            rows: vec![wide, exact.clone(), twin],
        };
        assert_eq!(table.find(&target), 20);

        // A nonzero offset adds its square to fitness; the offset-free row
        // wins even though both spans sit at the target.
        let offset_row = Row {
            spans: exact_spans,
            offset: quantize_axis(0.05),
            biome: 40,
        };
        let table = BiomeTable {
            rows: vec![offset_row, exact],
        };
        assert_eq!(table.find(&target), 20);
    }

    /// Fitness follows the reference arithmetic: the sum of squared axis
    /// distances plus the squared offset, all in quantized units. Row B
    /// carries three small penalties (temperature exact, vegetation 0.2 off,
    /// depth 0.15 off, offset 0.1) totalling 7 250 000, beating row A whose
    /// single temperature miss of 0.5 costs 25 000 000 even though it lists
    /// first.
    #[test]
    fn fitness_matches_hand_computation() {
        let point = span_of(0.0, 0.0);
        let mut a_spans = [point; AXIS_COUNT];
        a_spans[0] = span_of(0.5, 0.5);
        let a = Row {
            spans: a_spans,
            offset: 0,
            biome: 10,
        };
        let mut b_spans = [point; AXIS_COUNT];
        b_spans[0] = span_of(-2.0, 2.0);
        b_spans[1] = span_of(0.2, 0.2);
        b_spans[4] = span_of(0.15, 0.15);
        let b = Row {
            spans: b_spans,
            offset: quantize_axis(0.1),
            biome: 11,
        };
        let search = BiomeTable { rows: vec![a, b] };
        let target = TargetPoint {
            axes: [0, 0, 0, 0, 0, 0],
        };
        assert_eq!(search.find(&target), 11);
        // The pinned row count and distinct biomes: the transcription
        // carries 7594 rows over 56 biomes.
        let loaded = table();
        assert_eq!(loaded.len(), 7594);
        let distinct: std::collections::HashSet<u32> =
            loaded.rows.iter().map(|row| row.biome).collect();
        assert_eq!(distinct.len(), 56);
    }

    /// The generation-order head of the pinned table is the mushroom
    /// fields pair, at its registry wire id.
    #[test]
    fn pinned_head_row_is_mushroom_fields() {
        let loaded = table();
        let order: Vec<String> = serde_json::from_str(
            &fs::read_to_string(pins().join("biome_registry_order.json")).unwrap(),
        )
        .unwrap();
        let expected = order
            .iter()
            .position(|name| name == "minecraft:mushroom_fields")
            .unwrap() as u32;
        assert_eq!(loaded.fallback(), expected);
    }
}
