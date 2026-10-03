//! Worldgen parity gate: boots vanilla with a pinned seed and normal
//! terrain, captures the spawn-area chunk packets from a bot join, and
//! compares them structurally against the seeded terrain generator at the
//! same seed.
//!
//! The comparison is deliberately not byte-exact (that needs the full
//! density graph): it checks terrain shape agreement (heightmap deltas),
//! material agreement (block histograms), cell agreement, and reports the
//! biome spread. Thresholds are convergence targets that tighten as the
//! generator approaches vanilla.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use doppel_protocol::load_pin;
use doppel_world::anvil_to_wire::unpack;
use doppel_world::chunk_codec::{Container, WireChunk};
use doppel_world::decoration::Decorator;
use doppel_world::registry::BlockRegistry;
use doppel_world::terrain::HeightmapGenerator;

use crate::{bot, capture, vanilla};

const PORT: u16 = 25571;
/// The world seed both sides generate from.
const SEED: i64 = 42;
/// Capture floor: below this the join burst failed, not the generator.
const MIN_CHUNKS: usize = 20;
/// Heightmap convergence targets (world layers). The median column delta
/// is zero; the tail sits at canopy edges where the tree stream desyncs.
const MAX_MEDIAN_DELTA: f64 = 2.0;
const MAX_P95_DELTA: f64 = 10.0;
/// Shape alignment: Pearson correlation of the height fields.
const MIN_CORRELATION: f64 = 0.75;
/// Material agreement: histogram overlap over block names.
const MIN_OVERLAP: f64 = 0.98;
/// Cell-level agreement (air-dominated, so a low bar that catches gross
/// breakage like wrong world height or offset sections). The carve and
/// aquifer fields match; the remaining gap is lush-cave ground cover.
const MIN_CELL_AGREEMENT: f64 = 0.97;
/// Coastline agreement: the density surface matches vanilla block for
/// block; the remaining disagreement columns carry vanilla ground cover
/// (leaf litter, grass, tree canopies) the tree stream desync moves.
const MIN_LANDMASK: f64 = 0.94;

pub fn run() -> Result<bool> {
    let pin = load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let server = vanilla::boot_seeded(&pin, &jar, PORT, SEED)?;

    let dump_dir = vanilla::vanilla_dir()?.join("worldgen-capture");
    if dump_dir.exists() {
        std::fs::remove_dir_all(&dump_dir).context("cleaning worldgen capture dir")?;
    }
    std::fs::create_dir_all(&dump_dir)?;

    let packets = bot::login_capture(
        "127.0.0.1",
        PORT,
        pin.protocol.unwrap_or(0),
        &capture::login_start_c("Doppel"),
        &bot::CaptureOpts {
            // The session reads until it goes idle. Vanilla keep-alives
            // every 15s would keep a longer timeout alive indefinitely, and
            // the answered keep-alives reset the reader, so the timeout must
            // fit between those beats to end the capture after the burst.
            idle_timeout: Some(Duration::from_secs(12)),
            // Normal terrain streams entity traffic alongside the chunk
            // burst; a cap sized for the entity-free flat capture truncates
            // the join before all spawn chunks arrive.
            max_packets: Some(6000),
            dump_dir: Some(&dump_dir),
            commands: &[],
            walk_chunks: None,
            raw_packets: &[],
        },
    )
    .context("capturing vanilla join burst")?;
    drop(server);

    let mut chunks: Vec<WireChunk> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for p in &packets {
        if p.id != 0x2e {
            continue;
        }
        let Some(file) = &p.file else {
            continue;
        };
        let body = std::fs::read(dump_dir.join(file)).context("reading dumped chunk body")?;
        let chunk = WireChunk::decode(&body).context("decoding captured chunk")?;
        if seen.insert((chunk.x, chunk.z)) {
            chunks.push(chunk);
        }
    }
    println!("[worldgen] captured {} spawn-area chunks", chunks.len());
    let chunk_packets = packets.iter().filter(|p| p.id == 0x2e).count();
    let end_note = packets
        .iter()
        .find(|p| p.id == -1)
        .and_then(|p| p.note.as_deref());
    println!(
        "[worldgen] capture: {} packets total, {chunk_packets} chunk packets, ended by {}",
        packets.len(),
        end_note.unwrap_or("reaching the packet cap")
    );
    if chunks.len() < MIN_CHUNKS {
        bail!(
            "only {} chunks captured (need {MIN_CHUNKS}): join burst incomplete",
            chunks.len()
        );
    }

    let root = doppel_protocol::find_repo_root()?;
    let registry = BlockRegistry::load(&root.join("pins").join("blocks.json"))
        .context("loading block registry pins")?;
    let terrain = HeightmapGenerator::with_seed(SEED, &registry)?;

    compare(&chunks, &terrain, &registry, &dump_dir)
}

fn compare(
    captured: &[WireChunk],
    terrain: &HeightmapGenerator,
    registry: &BlockRegistry,
    dump_dir: &Path,
) -> Result<bool> {
    // Every captured chunk decorates before any of them is emitted, so the
    // cross-chunk writes features make land before the comparison snapshot.
    let mut decorator =
        Decorator::new(terrain, registry, SEED).context("building the feature decorator")?;
    for v in captured {
        decorator.decorate(v.x, v.z);
    }

    let mut heights = HeightCompare::default();
    let mut ours_hist: HashMap<String, u64> = HashMap::new();
    let mut vanilla_hist: HashMap<String, u64> = HashMap::new();
    let mut biomes: HashMap<u32, u64> = HashMap::new();
    let mut ours_biomes: HashMap<u32, u64> = HashMap::new();
    let mut cells_total = 0u64;
    let mut cells_equal = 0u64;
    // Section-count disagreement would silently truncate the cell zip, so a
    // mismatch fails the gate instead of comparing a prefix.
    let mut structural_fail = false;
    // Per-chunk convergence probes: heightmap-1 outlier columns and the
    // block-name deltas that mark a mis-placed feature.
    #[derive(Default)]
    struct ChunkProbe {
        pos: (i32, i32),
        outliers: u64,
        mismatched: u64,
        leaf: u64,
        log: u64,
        total: u64,
    }
    let mut per_chunk: Vec<ChunkProbe> = Vec::new();
    let leaf_like = |name: &str| -> u64 {
        u64::from(name.ends_with("_leaves") || name == "minecraft:leaf_litter")
    };
    let log_like = |name: &str| -> u64 { u64::from(name.ends_with("_log")) };

    for v in captured {
        let mine = decorator.emit(v.x, v.z).context("emitting our chunk")?;
        heights.add_chunk(v, &mine);
        let ours_cells = chunk_cells(&mine, registry);
        let vanilla_cells = chunk_cells(v, registry);
        if mine.sections.len() != v.sections.len() {
            println!(
                "[worldgen] FAIL chunk ({},{}): {} sections ours vs {} vanilla",
                v.x,
                v.z,
                mine.sections.len(),
                v.sections.len()
            );
            structural_fail = true;
        }
        let mut outlier_columns = 0u64;
        let mut mismatched = 0u64;
        let mut leaf_gap = 0i64;
        let mut log_gap = 0i64;
        if let (Some((_, a)), Some((_, b))) = (
            mine.heightmaps.iter().find(|(ty, _)| *ty == 1),
            v.heightmaps.iter().find(|(ty, _)| *ty == 1),
        ) {
            let ours = unpack(a, 9, 256);
            let theirs = unpack(b, 9, 256);
            for i in 0..256 {
                if (ours[i] as i32 - theirs[i] as i32).abs() > 4 {
                    outlier_columns += 1;
                }
            }
        }
        for (name, count) in &ours_cells.hist {
            *ours_hist.entry(name.clone()).or_default() += *count;
            leaf_gap -= (*count as i64) * (leaf_like(name) as i64);
            log_gap -= (*count as i64) * (log_like(name) as i64);
        }
        for (name, count) in &vanilla_cells.hist {
            *vanilla_hist.entry(name.clone()).or_default() += *count;
            leaf_gap += (*count as i64) * (leaf_like(name) as i64);
            log_gap += (*count as i64) * (log_like(name) as i64);
        }
        for (a, b) in ours_cells.cells.iter().zip(vanilla_cells.cells.iter()) {
            cells_total += 1;
            mismatched += u64::from(a != b);
            cells_equal += u64::from(a == b);
        }
        per_chunk.push(ChunkProbe {
            pos: (v.x, v.z),
            outliers: outlier_columns,
            mismatched,
            leaf: leaf_gap.unsigned_abs(),
            log: log_gap.unsigned_abs(),
            total: leaf_gap.unsigned_abs() + log_gap.unsigned_abs(),
        });
        for (id, count) in vanilla_cells.biomes {
            *biomes.entry(id).or_default() += count;
        }
        for (id, count) in ours_cells.biomes {
            *ours_biomes.entry(id).or_default() += count;
        }
    }
    per_chunk.sort_by_key(|c| std::cmp::Reverse(c.total));
    println!("[worldgen] worst chunks by leaf+log delta (chunk, hm outliers, cell mismatches, leaf delta, log delta):");
    for c in per_chunk.iter().take(6) {
        println!(
            "[worldgen]   {:?}: {}, {}, {}, {}",
            c.pos, c.outliers, c.mismatched, c.leaf, c.log
        );
    }

    println!("[worldgen] chunks compared: {}", captured.len());

    // Heightmap deltas per wire map type.
    let mut worst_median = 0f64;
    let mut worst_p95 = 0f64;
    for (ty, d) in heights.map_types.iter().zip(heights.deltas.iter()) {
        let Some(ty) = *ty else {
            continue;
        };
        if d.is_empty() {
            // A shared type with no compared columns would leave the
            // zero-initialized worsts passing vacuously.
            println!("[worldgen] FAIL heightmap {ty} compared no columns");
            return Ok(false);
        }
        let mut sorted = d.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let n = sorted.len();
        let median = sorted[n / 2].abs();
        let p95 = sorted[(n * 95) / 100].abs();
        let max = sorted[n - 1].abs();
        let mean: f64 = sorted.iter().sum::<f64>() / n as f64;
        println!(
            "[worldgen] heightmap {ty}: median|d|={median:.1} p95|d|={p95:.1} max|d|={max:.1} mean_signed={mean:+.1} (layers, n={n})"
        );
        worst_median = worst_median.max(median);
        worst_p95 = worst_p95.max(p95);
    }
    let mut worst_corr = 1f64;
    if heights.corr_type.is_some() {
        if heights.pairs_a.is_empty() {
            println!("[worldgen] FAIL correlation pair capture is empty");
            return Ok(false);
        }
        worst_corr = pearson(&heights.pairs_a, &heights.pairs_b);
        println!("[worldgen] height correlation: {worst_corr:.3}");
    }

    // Land-mask split: coastline agreement and per-class height bias.
    let masks = heights.masks;
    let landmask = if masks.total == 0 {
        0.0
    } else {
        masks.agree as f64 / masks.total as f64
    };
    let disagree = masks.total - masks.agree;
    println!("[worldgen] land mask agreement: {landmask:.3} ({disagree} columns disagree)");
    let gap = |d: &[f64], label: &str| {
        if d.is_empty() {
            return;
        }
        let mut sorted = d.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let n = sorted.len();
        println!(
            "[worldgen] {label}: n={n} median gap {:+.1} p25 {:+.1} p75 {:+.1}",
            sorted[n / 2],
            sorted[n / 4],
            sorted[(n * 3) / 4]
        );
    };
    gap(
        &masks.vanilla_land_gap,
        "vanilla-land columns we call water",
    );
    gap(&masks.ours_land_gap, "our-land columns vanilla calls water");
    let split = |d: &[f64], label: &str| {
        if d.is_empty() {
            println!("[worldgen] {label}: no shared columns");
            return;
        }
        let mut sorted = d.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let n = sorted.len();
        let median = sorted[n / 2];
        let p95 = sorted[(n * 95) / 100].abs();
        println!("[worldgen] {label}: median signed {median:+.1} p95|d|={p95:.1} (n={n})");
    };
    split(&masks.ocean_deltas, "shared-ocean delta");
    split(&masks.land_deltas, "shared-land delta");

    // Block histogram overlap.
    let mut overlap_lo = 0u64;
    let mut overlap_hi = 0u64;
    let mut names: std::collections::BTreeSet<&str> =
        ours_hist.keys().map(String::as_str).collect();
    names.extend(vanilla_hist.keys().map(String::as_str));
    let mut diffs: Vec<(i64, &str)> = Vec::new();
    for name in names {
        let a = ours_hist.get(name).copied().unwrap_or(0);
        let b = vanilla_hist.get(name).copied().unwrap_or(0);
        overlap_lo += a.min(b);
        overlap_hi += a.max(b);
        if a != b {
            diffs.push((b as i64 - a as i64, name));
        }
    }
    // A zero denominator divides to NaN, which every threshold comparison
    // treats as passing: empty histogram mass reads as zero overlap.
    let overlap = if overlap_hi == 0 {
        println!("[worldgen] FAIL block histograms carry no mass to compare");
        0.0
    } else {
        overlap_lo as f64 / overlap_hi as f64
    };
    println!("[worldgen] block histogram overlap: {overlap:.3}");
    diffs.sort_by_key(|(d, _)| d.abs());
    println!("[worldgen] largest material deltas (vanilla-ours):");
    for (d, name) in diffs.iter().rev().take(6) {
        println!("[worldgen]   {name}: {d:+}");
    }

    // Cell agreement and biome spread. Zero compared cells must fail rather
    // than divide to NaN, which every threshold comparison would treat as
    // passing.
    let agreement = if cells_total == 0 {
        0.0
    } else {
        cells_equal as f64 / cells_total as f64
    };
    println!("[worldgen] exact cell agreement: {agreement:.3} ({cells_equal}/{cells_total})");
    let mut biome_list: Vec<(u64, u32)> = biomes.into_iter().map(|(k, v)| (v, k)).collect();
    biome_list.sort_unstable_by_key(|(count, _)| std::cmp::Reverse(*count));
    let shown: Vec<String> = biome_list
        .iter()
        .take(8)
        .map(|(count, id)| format!("{id}x{count}"))
        .collect();
    println!(
        "[worldgen] vanilla biomes (id x cells): {}",
        shown.join(" ")
    );
    let mut ours_list: Vec<(u64, u32)> = ours_biomes.into_iter().map(|(k, v)| (v, k)).collect();
    ours_list.sort_unstable_by_key(|(count, _)| std::cmp::Reverse(*count));
    let ours_shown: Vec<String> = ours_list
        .iter()
        .take(8)
        .map(|(count, id)| format!("{id}x{count}"))
        .collect();
    println!(
        "[worldgen] our biomes (id x cells): {}",
        ours_shown.join(" ")
    );
    let vanilla_ids: std::collections::HashSet<u32> =
        biome_list.iter().map(|(_, id)| *id).collect();
    let shared = ours_list
        .iter()
        .filter(|(_, id)| vanilla_ids.contains(id))
        .count();
    println!(
        "[worldgen] shared biome ids: {shared}/{} ours, {}/{} vanilla",
        ours_list.len(),
        shared,
        vanilla_ids.len()
    );

    // Persist the raw pairs for offline calibration.
    let report = dump_dir.join("summary.txt");
    std::fs::write(
        &report,
        format!(
            "chunks={}\noverlap={overlap:.4}\nagreement={agreement:.4}\ncorr={worst_corr:.4}\nmedian={worst_median:.2}\np95={worst_p95:.2}\nlandmask={landmask:.4}\n",
            captured.len()
        ),
    )
    .ok();

    let mut ok = true;
    if heights.map_types.iter().all(|t| t.is_none()) {
        // Without a shared heightmap type the delta and correlation metrics
        // above never ran; their zero-initialized worsts would pass vacuously.
        println!("[worldgen] FAIL no shared heightmap type between vanilla and generated chunks");
        ok = false;
    }
    if structural_fail {
        ok = false;
    }
    if masks.total == 0 {
        println!("[worldgen] FAIL land mask compared no columns");
        ok = false;
    }
    if cells_total == 0 {
        println!("[worldgen] FAIL cell comparison covered no cells");
        ok = false;
    }
    if worst_median > MAX_MEDIAN_DELTA {
        println!("[worldgen] FAIL median height delta {worst_median:.1} > {MAX_MEDIAN_DELTA}");
        ok = false;
    }
    if worst_p95 > MAX_P95_DELTA {
        println!("[worldgen] FAIL p95 height delta {worst_p95:.1} > {MAX_P95_DELTA}");
        ok = false;
    }
    if worst_corr < MIN_CORRELATION {
        println!("[worldgen] FAIL height correlation {worst_corr:.3} < {MIN_CORRELATION}");
        ok = false;
    }
    if overlap < MIN_OVERLAP {
        println!("[worldgen] FAIL histogram overlap {overlap:.3} < {MIN_OVERLAP}");
        ok = false;
    }
    if agreement < MIN_CELL_AGREEMENT {
        println!("[worldgen] FAIL cell agreement {agreement:.3} < {MIN_CELL_AGREEMENT}");
        ok = false;
    }
    if landmask < MIN_LANDMASK {
        println!("[worldgen] FAIL land mask agreement {landmask:.3} < {MIN_LANDMASK}");
        ok = false;
    }
    if ok {
        println!("[worldgen] gate green: all convergence targets met");
    }
    Ok(ok)
}

/// Height comparison accumulators: per-map-type deltas for every type both
/// sides carry, plus the raw correlation pairs and the land-mask split fed
/// by the first common type.
#[derive(Default)]
struct HeightCompare {
    deltas: Vec<Vec<f64>>,
    map_types: Vec<Option<u32>>,
    corr_type: Option<u32>,
    pairs_a: Vec<f64>,
    pairs_b: Vec<f64>,
    masks: MaskStats,
}

impl HeightCompare {
    fn add_chunk(&mut self, vanilla: &WireChunk, mine: &WireChunk) {
        if self.deltas.is_empty() {
            self.deltas = vec![Vec::new(); mine.heightmaps.len()];
            self.map_types = vec![None; mine.heightmaps.len()];
        }
        for (slot, ours) in mine.heightmaps.iter().enumerate() {
            let Some((_, vlongs)) = vanilla.heightmaps.iter().find(|(ty, _)| *ty == ours.0) else {
                continue;
            };
            self.map_types[slot] = Some(ours.0);
            let a = unpack(&ours.1, 9, 256);
            let b = unpack(vlongs, 9, 256);
            let take_pairs = match self.corr_type {
                None => {
                    self.corr_type = Some(ours.0);
                    true
                }
                Some(t) => t == ours.0,
            };
            for i in 0..256 {
                let delta = b[i] as f64 - a[i] as f64;
                self.deltas[slot].push(delta);
                if take_pairs {
                    self.pairs_a.push(b[i] as f64);
                    self.pairs_b.push(a[i] as f64);
                    // Stored values are (first free y) - MIN_Y; the ocean
                    // rests at exactly sea level, so land is > 128.
                    let land_a = a[i] > 128;
                    let land_b = b[i] > 128;
                    self.masks.total += 1;
                    if land_a == land_b {
                        self.masks.agree += 1;
                        if land_a {
                            self.masks.land_deltas.push(delta);
                        } else {
                            self.masks.ocean_deltas.push(delta);
                        }
                    } else if land_b {
                        // How far the column sits from flipping to land on
                        // our side (positive = still water-side of the line).
                        self.masks.vanilla_land_only += 1;
                        self.masks.vanilla_land_gap.push(delta);
                    } else {
                        self.masks.ours_land_only += 1;
                        self.masks.ours_land_gap.push(-delta);
                    }
                }
            }
        }
    }
}

/// Coastline agreement plus height deltas split by shared surface class.
/// The mask isolates the noise field from the height formula: a matching
/// field with a mis-calibrated spline keeps the mask high while the split
/// deltas expose the per-class bias.
#[derive(Default)]
struct MaskStats {
    agree: u64,
    total: u64,
    ocean_deltas: Vec<f64>,
    land_deltas: Vec<f64>,
    vanilla_land_only: u64,
    ours_land_only: u64,
    vanilla_land_gap: Vec<f64>,
    ours_land_gap: Vec<f64>,
}

/// Per-chunk aggregates for the comparison: block histogram by name, biome
/// histogram by id, and the flat cell list by global state id.
struct ChunkStats {
    hist: HashMap<String, u64>,
    biomes: HashMap<u32, u64>,
    cells: Vec<u32>,
}

fn chunk_cells(chunk: &WireChunk, registry: &BlockRegistry) -> ChunkStats {
    let mut hist: HashMap<String, u64> = HashMap::new();
    let mut biomes: HashMap<u32, u64> = HashMap::new();
    let mut cells: Vec<u32> = Vec::with_capacity(chunk.sections.len() * 4096);
    for section in &chunk.sections {
        let states = section_states(section);
        for &state in &states {
            let name = registry
                .state_of(state)
                .map(|(n, _)| n.to_string())
                .unwrap_or_else(|| format!("state {state}"));
            *hist.entry(name).or_default() += 1;
        }
        match &section.biomes {
            Container::Single(id) => *biomes.entry(*id).or_default() += 64,
            Container::Palette {
                entries,
                longs,
                bits,
            } => {
                for idx in unpack(longs, *bits as usize, 64) {
                    let id = entries.get(idx as usize).copied().unwrap_or(0);
                    *biomes.entry(id).or_default() += 1;
                }
            }
            Container::Global { .. } => {}
        }
        cells.extend(states);
    }
    ChunkStats {
        hist,
        biomes,
        cells,
    }
}

/// The 4096 global state ids of one section, in storage order.
fn section_states(section: &doppel_world::chunk_codec::WireSection) -> Vec<u32> {
    match &section.block_states {
        Container::Single(v) => vec![*v; 4096],
        Container::Palette {
            entries,
            longs,
            bits,
        } => unpack(longs, *bits as usize, 4096)
            .into_iter()
            .map(|i| entries.get(i as usize).copied().unwrap_or(0))
            .collect(),
        Container::Global { longs, bits } => unpack(longs, *bits as usize, 4096)
            .into_iter()
            .map(u32::from)
            .collect(),
    }
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len()) as f64;
    if n == 0.0 {
        return 0.0;
    }
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..n as usize {
        let (da, db) = (a[i] - ma, b[i] - mb);
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va == 0.0 || vb == 0.0 {
        return 0.0;
    }
    cov / (va.sqrt() * vb.sqrt())
}
