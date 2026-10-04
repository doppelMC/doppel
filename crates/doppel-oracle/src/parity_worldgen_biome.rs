//! Biome-locate worldgen parity gate: the worldgen gate's structural
//! comparison run over chunks captured far from spawn. The capture bot
//! locates each target biome, teleports there, and captures the new
//! chunk stream, so oracle coverage reaches biomes the spawn window
//! never shows (cherry groves, mangrove swamps).
//!
//! The comparison runs the decorated pipeline: the biome legs carry
//! cherry groves and mangrove swamps, so the material floors read the
//! trees, roots, and petals the decorator places. Height thresholds
//! sit at spawn-parity levels (the spawn leg prints its own scores as
//! calibration), and the printed per-block deltas are the feature
//! to-do list for the next biome wave. Height correlation is printed
//! but not gated: a flat biome carries only small-scale shape, where
//! the two generators sit within a block of each other in absolute
//! terms while Pearson noise dominates the ratio.

use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use doppel_protocol::load_pin;
use doppel_world::anvil_to_wire::unpack;
use doppel_world::chunk_codec::{Container, WireChunk};
use doppel_world::registry::BlockRegistry;
#[cfg(test)]
use doppel_world::structures::{generate_chunk, WellBlocks};
use doppel_world::terrain::HeightmapGenerator;

use crate::{bot, capture, vanilla};

const PORT: u16 = 25573;
/// The world seed both sides generate from, same as the worldgen gate.
const SEED: i64 = 42;
/// Biomes the gate hops to, in leg order. The swamp goes first: its
/// chunk stream stalls late in a long session, so it runs on the
/// freshest server while the grove, which streams reliably, takes the
/// later hop.
const BIOMES: &[&str] = &["minecraft:mangrove_swamp", "minecraft:cherry_grove"];
/// Capture floors: below these the leg failed, not the generator.
const MIN_SPAWN_CHUNKS: usize = 20;
const MIN_BIOME_CHUNKS: usize = 20;
/// The hop must land: every biome-leg chunk center within this many
/// blocks of the located position.
const MAX_LANDING_DRIFT: i32 = 128;

/// Heightmap convergence targets. The decorated legs measure median 0
/// with p95 8 at spawn and p95 at most 6 in the biome legs, so the
/// thresholds sit far above the measurements while a terrain-shape
/// regression still crosses them.
const MAX_MEDIAN_DELTA: f64 = 12.0;
const MAX_P95_DELTA: f64 = 28.0;
/// Coastline agreement floor. The decorated legs measure 0.954 at
/// spawn, 0.925 in the mangrove swamp and 0.999 in the cherry grove;
/// the floor sits under the swamp leg with room for the capture window
/// to shift.
const MIN_LANDMASK: f64 = 0.9;
/// Material floors over the decorated chunks. The legs measure
/// histogram overlap 0.993-0.995 and cell agreement 0.975-0.985; the
/// floors hold the decorator to the shapes it writes while absorbing
/// leg-window edges (features rooted in chunks the hop did not
/// capture).
const MIN_OVERLAP: f64 = 0.97;
const MIN_CELL_AGREEMENT: f64 = 0.95;

pub fn run() -> Result<bool> {
    let pin = load_pin()?;
    let jar = vanilla::ensure_jar(&pin)?;
    let server = vanilla::boot_seeded(&pin, &jar, PORT, SEED)?;

    let dump_dir = vanilla::vanilla_dir()?.join("biome-capture");
    if dump_dir.exists() {
        std::fs::remove_dir_all(&dump_dir).context("cleaning biome capture dir")?;
    }
    std::fs::create_dir_all(&dump_dir)?;

    // The hop teleports land high in the air, and the server's anti-flight
    // check disconnects a floating survival player 80 ticks after the
    // teleport ack (the ack is what makes chunk tracking follow the hop).
    // Spectators are exempt from the check, and spectator mode changes
    // nothing about generated chunk content.
    let commands = vec!["gamemode spectator".to_string()];
    let session = bot::login_capture_legs(
        "127.0.0.1",
        PORT,
        pin.protocol.unwrap_or(0),
        &capture::login_start_c("Doppel"),
        &bot::CaptureOpts {
            idle_timeout: Some(Duration::from_secs(12)),
            // Entity chatter floods ~400 packets/s while the hops run, so
            // the hop choreography (every phase wall-clock bounded) ends
            // the session; this cap is only a flood backstop.
            max_packets: Some(100_000),
            dump_dir: Some(&dump_dir),
            commands: &commands,
            walk_chunks: None,
            raw_packets: &[],
        },
        BIOMES,
    )
    .context("capturing legged join burst")?;
    drop(server);

    let end = session.packets.iter().find(|p| p.id == -1);
    let end_note = end.and_then(|p| p.note.as_deref());
    let end_t = end
        .or_else(|| session.packets.last())
        .map(|p| p.t_ms)
        .unwrap_or(0);
    println!(
        "[biome] capture: {} packets in {end_t}ms, ended by {}",
        session.packets.len(),
        end_note.unwrap_or("reaching the packet cap")
    );
    // Hop choreography with timestamps: locate latency and stream settle
    // beats are what a flaky CI run questions first.
    for p in &session.packets {
        let Some(note) = &p.note else { continue };
        let hop_note = [
            "locating",
            "locate replied",
            "hop chunk batch",
            "chunk quiet closed",
            "keep-alive armed",
            "teleport (",
        ]
        .iter()
        .any(|m| note.contains(m));
        if hop_note {
            println!("[biome] t={:>6}ms {note}", p.t_ms);
        }
    }
    let mut ids: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
    for p in &session.packets {
        *ids.entry(p.id).or_default() += 1;
    }
    let id_line: Vec<String> = ids.iter().map(|(id, n)| format!("{id:#04x}x{n}")).collect();
    println!("[biome] packet ids: {}", id_line.join(" "));

    let mut leg_chunks: Vec<(String, Vec<WireChunk>)> = Vec::new();
    for (i, leg) in session.legs.iter().enumerate() {
        let end = session
            .legs
            .get(i + 1)
            .map(|next| next.start)
            .unwrap_or(session.packets.len());
        let mut chunks: Vec<WireChunk> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for p in &session.packets[leg.start.min(session.packets.len())..end] {
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
        let leg_t = session
            .packets
            .get(leg.start.min(session.packets.len().saturating_sub(1)))
            .map(|p| p.t_ms)
            .unwrap_or(0);
        match leg.pos {
            Some((x, _, z)) => println!(
                "[biome] leg {i} {}: {} chunks at ({x}, {z}) starting t={leg_t}ms [{}]",
                leg.label,
                chunks.len(),
                leg.reply
            ),
            None => println!(
                "[biome] leg {i} {}: {} chunks starting t={leg_t}ms [{}]",
                leg.label,
                chunks.len(),
                leg.reply
            ),
        }
        leg_chunks.push((leg.label.clone(), chunks));
    }

    if leg_chunks.len() != 1 + BIOMES.len() {
        bail!(
            "expected {} legs, got {}: a locate hop failed",
            1 + BIOMES.len(),
            leg_chunks.len()
        );
    }
    let (_, spawn_chunks) = &leg_chunks[0];
    if spawn_chunks.len() < MIN_SPAWN_CHUNKS {
        bail!(
            "only {} spawn chunks captured (need {MIN_SPAWN_CHUNKS}): join burst incomplete",
            spawn_chunks.len()
        );
    }
    for (leg, biome) in session.legs.iter().skip(1).zip(BIOMES) {
        if leg.pos.is_none() {
            bail!("locate {biome} failed: {}", leg.reply);
        }
        // The structural NBT parse and the flattened reply text must
        // agree, or the transcript is not what it claims to be.
        let pos = leg.pos.expect("checked above");
        if bot::parse_locate_text(&leg.reply) != Some(pos) {
            bail!(
                "locate {biome} reply text disagrees with parsed position: [{}]",
                leg.reply
            );
        }
    }
    for (label, chunks) in leg_chunks.iter().skip(1) {
        if chunks.len() < MIN_BIOME_CHUNKS {
            bail!(
                "only {label} chunks captured ({} < {MIN_BIOME_CHUNKS}): hop stream incomplete",
                chunks.len()
            );
        }
    }
    // The hop must land where the locate said: chunk centers near the
    // reply position, or the comparison reads the wrong biome.
    for (leg, (label, chunks)) in session.legs.iter().skip(1).zip(leg_chunks.iter().skip(1)) {
        let (x, _, z) = leg.pos.expect("checked above");
        let drift = chunks
            .iter()
            .map(|c| (c.x * 16 + 8 - x).abs().max((c.z * 16 + 8 - z).abs()))
            .max()
            .unwrap_or(i32::MAX);
        if drift > MAX_LANDING_DRIFT {
            bail!("{label} chunks drift {drift} blocks from the located position");
        }
    }

    let root = doppel_protocol::find_repo_root()?;
    let registry = BlockRegistry::load(&root.join("pins").join("blocks.json"))
        .context("loading block registry pins")?;
    let terrain = HeightmapGenerator::with_seed(SEED, &registry)?;

    let spawn = compare_leg("spawn", spawn_chunks, &terrain, &registry, &dump_dir)?;
    println!(
        "[biome] calibration: spawn leg scores corr {:.3} overlap {:.3} cells {:.3} landmask {:.3} median {:.1} p95 {:.1}",
        spawn.correlation,
        spawn.overlap,
        spawn.agreement,
        spawn.landmask,
        spawn.worst_median,
        spawn.worst_p95
    );

    let mut ok = true;
    for (label, chunks) in leg_chunks.iter().skip(1) {
        let m = compare_leg(label, chunks, &terrain, &registry, &dump_dir)?;
        ok &= gate_leg(label, &m);
    }
    if ok {
        println!("[biome] gate green: decorated legs hold the shape and material floors");
    }
    Ok(ok)
}

/// Metrics of one compared leg.
struct LegMetrics {
    worst_median: f64,
    worst_p95: f64,
    correlation: f64,
    overlap: f64,
    agreement: f64,
    landmask: f64,
    has_heightmaps: bool,
}

fn gate_leg(label: &str, m: &LegMetrics) -> bool {
    let mut ok = true;
    if !m.has_heightmaps {
        println!("[biome] FAIL {label}: no shared heightmap type");
        ok = false;
    }
    if m.worst_median > MAX_MEDIAN_DELTA {
        println!(
            "[biome] FAIL {label}: median height delta {:.1} > {MAX_MEDIAN_DELTA}",
            m.worst_median
        );
        ok = false;
    }
    if m.worst_p95 > MAX_P95_DELTA {
        println!(
            "[biome] FAIL {label}: p95 height delta {:.1} > {MAX_P95_DELTA}",
            m.worst_p95
        );
        ok = false;
    }
    if m.landmask < MIN_LANDMASK {
        println!(
            "[biome] FAIL {label}: land mask agreement {:.3} < {MIN_LANDMASK}",
            m.landmask
        );
        ok = false;
    }
    if m.overlap < MIN_OVERLAP {
        println!(
            "[biome] FAIL {label}: histogram overlap {:.3} < {MIN_OVERLAP}",
            m.overlap
        );
        ok = false;
    }
    if m.agreement < MIN_CELL_AGREEMENT {
        println!(
            "[biome] FAIL {label}: cell agreement {:.3} < {MIN_CELL_AGREEMENT}",
            m.agreement
        );
        ok = false;
    }
    ok
}

/// The leg comparison over the decorated pipeline: every chunk in the
/// leg decorates before any of them emits, so features crossing chunk
/// borders land whole.
fn compare_leg(
    label: &str,
    captured: &[WireChunk],
    terrain: &HeightmapGenerator,
    registry: &BlockRegistry,
    dump_dir: &Path,
) -> Result<LegMetrics> {
    let mut decorator =
        doppel_world::decoration::Decorator::new(terrain, registry, SEED).context("decorator")?;
    for v in captured {
        decorator.decorate(v.x, v.z);
    }
    let mut ours = Vec::with_capacity(captured.len());
    for v in captured {
        ours.push(
            decorator
                .emit(v.x, v.z)
                .with_context(|| format!("emitting chunk ({}, {})", v.x, v.z))?,
        );
    }
    compare_prepared(label, captured, &ours, registry, dump_dir)
}

/// The terrain-only variant: the same comparison over undecorated
/// chunks, for separating a shape regression from a feature one.
#[cfg(test)]
fn compare_leg_terrain(
    label: &str,
    captured: &[WireChunk],
    terrain: &HeightmapGenerator,
    well: &WellBlocks,
    registry: &BlockRegistry,
    dump_dir: &Path,
) -> Result<LegMetrics> {
    let mut ours = Vec::with_capacity(captured.len());
    for v in captured {
        ours.push(
            generate_chunk(terrain, well, SEED, v.x, v.z)
                .with_context(|| format!("generating chunk ({}, {})", v.x, v.z))?,
        );
    }
    compare_prepared(label, captured, &ours, registry, dump_dir)
}

/// The metric core: every captured chunk paired with our emitted chunk
/// for the same position.
fn compare_prepared(
    label: &str,
    captured: &[WireChunk],
    ours: &[WireChunk],
    registry: &BlockRegistry,
    dump_dir: &Path,
) -> Result<LegMetrics> {
    let mut heights = HeightCompare::default();
    let mut ours_hist: HashMap<String, u64> = HashMap::new();
    let mut vanilla_hist: HashMap<String, u64> = HashMap::new();
    let mut biomes: HashMap<u32, u64> = HashMap::new();
    let mut cells_total = 0u64;
    let mut cells_equal = 0u64;

    for (v, mine) in captured.iter().zip(ours.iter()) {
        heights.add_chunk(v, mine);
        let ours_cells = chunk_cells(mine, registry);
        let vanilla_cells = chunk_cells(v, registry);
        for (name, count) in ours_cells.hist {
            *ours_hist.entry(name).or_default() += count;
        }
        for (name, count) in vanilla_cells.hist {
            *vanilla_hist.entry(name).or_default() += count;
        }
        for (id, count) in vanilla_cells.biomes {
            *biomes.entry(id).or_default() += count;
        }
        for (a, b) in ours_cells.cells.iter().zip(vanilla_cells.cells.iter()) {
            cells_total += 1;
            cells_equal += u64::from(a == b);
        }
    }
    println!("[biome] {label}: {} chunks compared", captured.len());

    let mut worst_median = 0f64;
    let mut worst_p95 = 0f64;
    for (ty, d) in heights.map_types.iter().zip(heights.deltas.iter()) {
        let Some(ty) = *ty else {
            continue;
        };
        let mut sorted = d.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let n = sorted.len();
        let median = sorted[n / 2].abs();
        let p95 = sorted[(n * 95) / 100].abs();
        let max = sorted[n - 1].abs();
        let mean: f64 = sorted.iter().sum::<f64>() / n as f64;
        println!(
            "[biome] {label} heightmap {ty}: median|d|={median:.1} p95|d|={p95:.1} max|d|={max:.1} mean_signed={mean:+.1} (layers, n={n})"
        );
        worst_median = worst_median.max(median);
        worst_p95 = worst_p95.max(p95);
    }
    let mut correlation = 1f64;
    if heights.corr_type.is_some() {
        correlation = pearson(&heights.pairs_a, &heights.pairs_b);
        println!("[biome] {label} height correlation: {correlation:.3}");
    }
    let masks = heights.masks;
    let landmask = if masks.total == 0 {
        0.0
    } else {
        masks.agree as f64 / masks.total as f64
    };
    println!(
        "[biome] {label} land mask agreement: {landmask:.3} ({} columns disagree)",
        masks.total - masks.agree
    );
    let split = |d: &[f64], kind: &str| {
        if d.is_empty() {
            return;
        }
        let mut sorted = d.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let n = sorted.len();
        println!(
            "[biome] {label} shared-{kind} delta: median signed {:+.1} p95|d|={:.1} (n={n})",
            sorted[n / 2],
            sorted[(n * 95) / 100].abs()
        );
    };
    split(&masks.ocean_deltas, "ocean");
    split(&masks.land_deltas, "land");

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
    let overlap = overlap_lo as f64 / overlap_hi as f64;
    println!("[biome] {label} block histogram overlap: {overlap:.3}");
    // The feature to-do list: what vanilla places here that we do not
    // (positive) and what we place instead (negative).
    diffs.sort_by_key(|(d, _)| d.abs());
    println!("[biome] {label} largest material deltas (vanilla-ours):");
    let abs = |name: &str| {
        let ours = ours_hist.get(name).copied().unwrap_or(0);
        let vanilla = vanilla_hist.get(name).copied().unwrap_or(0);
        (ours, vanilla)
    };
    for (d, name) in diffs.iter().rev().take(25) {
        let (ours, vanilla) = abs(name);
        println!("[biome] {label}   {name}: {d:+} (vanilla {vanilla}, ours {ours})");
    }

    let agreement = if cells_total == 0 {
        0.0
    } else {
        cells_equal as f64 / cells_total as f64
    };
    println!("[biome] {label} exact cell agreement: {agreement:.3} ({cells_equal}/{cells_total})");
    let mut biome_list: Vec<(u64, u32)> = biomes.into_iter().map(|(k, v)| (v, k)).collect();
    biome_list.sort_unstable_by_key(|(count, _)| std::cmp::Reverse(*count));
    let shown: Vec<String> = biome_list
        .iter()
        .take(8)
        .map(|(count, id)| format!("{id}x{count}"))
        .collect();
    println!(
        "[biome] {label} vanilla biomes (id x cells): {}",
        shown.join(" ")
    );

    let slug = label.replace(':', "_");
    std::fs::write(
        dump_dir.join(format!("summary-{slug}.txt")),
        format!(
            "chunks={}\noverlap={overlap:.4}\nagreement={agreement:.4}\ncorr={correlation:.4}\nmedian={worst_median:.2}\np95={worst_p95:.2}\nlandmask={landmask:.4}\n",
            captured.len()
        ),
    )
    .ok();

    Ok(LegMetrics {
        worst_median,
        worst_p95,
        correlation,
        overlap,
        agreement,
        landmask,
        has_heightmaps: heights.map_types.iter().any(|t| t.is_some()),
    })
}

/// Height comparison accumulators, as in the worldgen gate.
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
                    // Stored values are (first free y) - MIN_Y; sea level
                    // rests at exactly 128, so land is > 128.
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
                        self.masks.vanilla_land_only += 1;
                    } else {
                        self.masks.ours_land_only += 1;
                    }
                }
            }
        }
    }
}

/// Coastline agreement plus height deltas split by surface class.
#[derive(Default)]
struct MaskStats {
    agree: u64,
    total: u64,
    ocean_deltas: Vec<f64>,
    land_deltas: Vec<f64>,
    vanilla_land_only: u64,
    ours_land_only: u64,
}

/// Per-chunk aggregates: block histogram by name, biome histogram by
/// id, and the flat cell list by global state id.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The leg windows in chunk coordinates: the three capture hops land
    /// far apart, so membership by position is unambiguous.
    const LEG_WINDOWS: &[(&str, i32, i32, i32, i32)] = &[
        ("spawn", -16, 8, -12, 12),
        ("mangrove_swamp", -90, -70, -70, -45),
        ("cherry_grove", -135, -110, -50, -25),
    ];

    /// Replays a capture dump through compare_leg without booting the
    /// vanilla server: the three legs fall out of the chunk coordinates,
    /// and DOPPEL_BIOME_PROBE=decorate runs the decorated pipeline
    /// instead of terrain-only.
    #[test]
    #[ignore = "diagnostic: needs a live capture dump"]
    fn offline_leg_probe() {
        let dump_dir = crate::vanilla::vanilla_dir()
            .expect("vanilla dir")
            .join("biome-capture");
        let entries = std::fs::read_dir(&dump_dir).expect("biome capture dump");
        let mut chunks: Vec<WireChunk> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("bin") {
                continue;
            }
            let body = std::fs::read(&path).expect("chunk body");
            let Ok(chunk) = WireChunk::decode(&body) else {
                continue;
            };
            if seen.insert((chunk.x, chunk.z)) {
                chunks.push(chunk);
            }
        }
        assert!(!chunks.is_empty(), "capture dump held no chunks");

        let root = doppel_protocol::find_repo_root().expect("repo root");
        let registry =
            BlockRegistry::load(&root.join("pins").join("blocks.json")).expect("registry");
        let terrain = HeightmapGenerator::with_seed(SEED, &registry).expect("terrain");
        let well = WellBlocks::from_registry(&registry).expect("well blocks");
        let decorate = std::env::var("DOPPEL_BIOME_PROBE").as_deref() == Ok("decorate");

        for (label, x0, x1, z0, z1) in LEG_WINDOWS {
            let mut leg: Vec<&WireChunk> = chunks
                .iter()
                .filter(|c| (*x0..=*x1).contains(&c.x) && (*z0..=*z1).contains(&c.z))
                .collect();
            leg.sort_by_key(|c| (c.x, c.z));
            if leg.is_empty() {
                println!("[probe] {label}: no chunks in window");
                continue;
            }
            let owned: Vec<WireChunk> = leg.into_iter().cloned().collect();
            if decorate {
                compare_leg(label, &owned, &terrain, &registry, &dump_dir).expect("decorated leg");
            } else {
                compare_leg_terrain(label, &owned, &terrain, &well, &registry, &dump_dir)
                    .expect("terrain leg");
            }
        }
    }
}
