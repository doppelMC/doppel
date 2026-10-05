//! World persistence policy: dirty-chunk tracking, the autosave sweep,
//! and the stop flush. The disk formats live in doppel-world; this module
//! decides when and what to write.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use doppel_protocol::Reader;
use doppel_world::anvil_write::{wire_to_anvil, RegionWriter, WritePalette};
use doppel_world::level::LevelMeta;
use doppel_world::playerdata::{PlayerData, SavedSlot};

use crate::blobs;
use crate::game::Game;
use crate::inventory::{decode_item_stack, encode_item_stack, item_name, ItemStack, TOTAL_SLOTS};

/// One prepared chunk write: in-region coordinates plus its NBT.
type RegionUpdate = (usize, usize, doppel_world::Chunk);

/// Ticks between autosave sweeps.
pub const AUTOSAVE_INTERVAL_TICKS: u64 = 6000;
/// Chunks written by one sweep call.
pub const SAVE_BUDGET_PER_TICK: usize = 4;
/// Wall-clock budget for one sweep call, inside one tick's 50ms.
pub const SWEEP_DEADLINE: Duration = Duration::from_millis(40);

/// Chunks awaiting a region write, plus the stop-flush marker.
#[derive(Default)]
pub(crate) struct DirtyChunks {
    pub(crate) chunks: BTreeSet<(i32, i32)>,
    stop_flushed: bool,
}

/// Save state owned by the game thread.
#[derive(Default)]
pub(crate) struct Persistence {
    pub(crate) dirty: DirtyChunks,
    /// Ticks counted since the last autosave arm.
    since_arm: u64,
    /// True while a drain is in progress between autosave cadences.
    draining: bool,
    /// Spawn point carried from level.dat.
    spawn: (i32, i32, i32),
    /// Chunks whose conversion failed once; silences repeat warnings.
    failed: BTreeSet<(i32, i32)>,
}

/// Converts a live stack to its saved form: vanilla id/count plus the
/// stack's codec bytes, which carry the component patch exactly.
pub(crate) fn stack_to_saved(stack: &ItemStack) -> SavedSlot {
    let mut extra = Vec::new();
    encode_item_stack(&mut extra, Some(stack));
    SavedSlot {
        slot: 0,
        id: item_name(stack.item())
            .map(str::to_string)
            .unwrap_or_else(|| "minecraft:unknown_item".into()),
        count: stack.count(),
        extra: Some(extra),
    }
}

/// Converts a saved slot back to a live stack: the codec bytes when
/// present, otherwise the vanilla id/count pair.
pub(crate) fn saved_to_stack(saved: &SavedSlot) -> Option<ItemStack> {
    if let Some(extra) = &saved.extra {
        if let Ok(Some(stack)) = decode_item_stack(&mut Reader::new(extra)) {
            return Some(stack);
        }
    }
    let id = crate::inventory::item_id(&saved.id)?;
    Some(ItemStack::new(id, saved.count.max(1)))
}

impl Game {
    /// Applies the attached store's level meta: spawn point, clocks, and
    /// game rules.
    pub(crate) fn boot_from_level(&mut self) {
        let Some(world) = self.world.clone() else {
            return;
        };
        let level = {
            let w = world.lock().unwrap_or_else(|e| e.into_inner());
            w.level.clone()
        };
        self.day_time = level.day_time;
        self.spawning.day_time = level.day_time.rem_euclid(24000) as u64;
        if level.game_time > 0 {
            self.spawning.total_ticks = level.game_time as u64;
            self.spawning.time_running = true;
        }
        self.persistence.spawn = level.spawn;
        for (rule, value) in &level.game_rules {
            match rule.as_str() {
                "random_tick_speed" => {
                    if let Ok(speed) = value.parse::<usize>() {
                        self.set_tick_speed(speed);
                    }
                }
                "spawn_mobs" => {
                    if let Ok(enabled) = value.parse::<bool>() {
                        self.spawning.spawn_mobs = enabled;
                    }
                }
                _ => {}
            }
        }
    }

    /// Saved player state for a name, when the store has a file for it.
    pub(crate) fn load_saved_player(&self, name: &str) -> Option<PlayerData> {
        let world = self.world.as_ref()?;
        let w = world.lock().unwrap_or_else(|e| e.into_inner());
        doppel_world::playerdata::load(&w.root, &blobs::offline_uuid(name))
            .ok()
            .flatten()
    }

    /// Saves one player's state (the disconnect path).
    pub(crate) fn save_player_data(&mut self, conn: crate::game::ConnId) {
        let Some((uuid, data)) = self.player_snapshot(conn) else {
            return;
        };
        let Some(root) = self.world_root() else {
            return;
        };
        if let Err(e) = doppel_world::playerdata::save(&root, &uuid, &data) {
            eprintln!("[persistence] playerdata save failed: {e:#}");
        }
    }

    fn player_snapshot(&self, conn: crate::game::ConnId) -> Option<([u8; 16], PlayerData)> {
        let p = self.players.get(&conn)?;
        Some((
            blobs::offline_uuid(&p.name),
            player_data(p.x, p.y, p.z, p.yaw, p.pitch, &p.inv),
        ))
    }

    /// Marks a chunk column for the next sweep. Tracking only exists with
    /// a world store attached.
    pub(crate) fn mark_chunk_dirty(&mut self, cx: i32, cz: i32) {
        if self.world.is_some() {
            self.persistence.dirty.chunks.insert((cx, cz));
        }
    }

    /// The save phase of the tick: arm at the autosave cadence, then keep
    /// sweeping every tick until the dirty set drains.
    pub(crate) fn persistence_tick(&mut self) {
        if self.world.is_none() {
            return;
        }
        self.persistence.since_arm += 1;
        if self.persistence.since_arm >= AUTOSAVE_INTERVAL_TICKS {
            self.persistence.since_arm = 0;
            self.persistence.draining = true;
        }
        if !self.persistence.draining || self.persistence.dirty.chunks.is_empty() {
            return;
        }
        self.save_sweep(SAVE_BUDGET_PER_TICK, SWEEP_DEADLINE);
        if self.persistence.dirty.chunks.is_empty() {
            self.persistence.draining = false;
        }
    }

    /// Writes at most `budget` dirty chunks, stopping early past
    /// `deadline` (the first region pass always runs). Chunks not written
    /// stay dirty. Returns how many chunks left the set.
    pub(crate) fn save_sweep(&mut self, budget: usize, deadline: Duration) -> usize {
        let deadline_at = Instant::now() + deadline;
        let batch: Vec<(i32, i32)> = self
            .persistence
            .dirty
            .chunks
            .iter()
            .take(budget)
            .copied()
            .collect();
        self.write_batch(&batch, Some(deadline_at))
    }

    /// Writes every dirty chunk, then level.dat and the playerdata of all
    /// online players. The stop flush.
    pub(crate) fn flush_all(&mut self) {
        if self.world.is_none() {
            return;
        }
        let batch: Vec<(i32, i32)> = self.persistence.dirty.chunks.iter().copied().collect();
        self.write_batch(&batch, None);
        self.persistence.dirty.stop_flushed = true;
        self.save_level();
        let conns: Vec<crate::game::ConnId> = self.players.keys().copied().collect();
        let Some(root) = self.world_root() else {
            return;
        };
        for conn in conns {
            if let Some((uuid, data)) = self.player_snapshot(conn) {
                if let Err(e) = doppel_world::playerdata::save(&root, &uuid, &data) {
                    eprintln!("[persistence] playerdata save failed: {e:#}");
                }
            }
        }
    }

    /// Converts and writes one batch, grouped into a single region pass
    /// per region file. Conversion failures drop their chunk from the
    /// set (retrying a deterministic failure would wedge the drain);
    /// write failures keep it.
    fn write_batch(&mut self, batch: &[(i32, i32)], deadline: Option<Instant>) -> usize {
        let Some(root) = self.world_root() else {
            return 0;
        };
        // Prepare phase: everything borrowed from self ends here.
        let mut by_region: BTreeMap<(i32, i32), Vec<RegionUpdate>>;
        let mut dropped: Vec<(i32, i32)>;
        {
            let Some(registry) = self.registry.as_ref() else {
                return 0;
            };
            let biome_names: HashMap<u32, String> = self
                .world
                .as_ref()
                .map(|w| {
                    let w = w.lock().unwrap_or_else(|e| e.into_inner());
                    w.boot.biomes.iter().map(|(k, v)| (*v, k.clone())).collect()
                })
                .unwrap_or_default();
            let palette = WritePalette {
                registry,
                biome_names: &biome_names,
            };
            by_region = BTreeMap::new();
            dropped = Vec::new();
            for &(cx, cz) in batch {
                let Some(cached) = self.chunks.get(&(cx, cz)) else {
                    dropped.push((cx, cz));
                    continue;
                };
                match wire_to_anvil(&cached.wire, &palette) {
                    Ok(chunk) => by_region
                        .entry((cx.div_euclid(32), cz.div_euclid(32)))
                        .or_default()
                        .push((
                            cx.rem_euclid(32) as usize,
                            cz.rem_euclid(32) as usize,
                            chunk,
                        )),
                    Err(e) => {
                        if self.persistence.failed.insert((cx, cz)) {
                            eprintln!("[persistence] chunk ({cx},{cz}) not convertible: {e:#}");
                        }
                        dropped.push((cx, cz));
                    }
                }
            }
        }
        for at in dropped {
            self.persistence.dirty.chunks.remove(&at);
        }
        let mut written = 0usize;
        let mut newly_saved: Vec<(i32, i32)> = Vec::new();
        for ((rx, rz), group) in by_region {
            if written > 0 && deadline.is_some_and(|d| Instant::now() > d) {
                break;
            }
            let updated: Vec<(i32, i32)> = group
                .iter()
                .map(|(x, z, _)| (rx * 32 + *x as i32, rz * 32 + *z as i32))
                .collect();
            match RegionWriter::open(&root, rx, rz).write(&group) {
                Ok(()) => {
                    for at in &updated {
                        self.persistence.dirty.chunks.remove(at);
                    }
                    written += group.len();
                    newly_saved.extend(updated);
                }
                Err(e) => eprintln!("[persistence] region ({rx},{rz}) write failed: {e:#}"),
            }
        }
        if !newly_saved.is_empty() {
            if let Some(world) = &self.world {
                let mut w = world.lock().unwrap_or_else(|e| e.into_inner());
                let before = w.saved.len();
                w.saved.extend(newly_saved);
                if w.saved.len() != before {
                    if let Err(e) = crate::write_saved_set(&w.root, &w.saved) {
                        eprintln!("[persistence] saved-set write failed: {e:#}");
                    }
                }
            }
        }
        written
    }

    /// The world root backing the store, when one is attached.
    fn world_root(&self) -> Option<PathBuf> {
        let world = self.world.as_ref()?;
        let w = world.lock().unwrap_or_else(|e| e.into_inner());
        Some(w.root.clone())
    }

    /// Writes level.dat from the live clocks, rules, and spawn. An
    /// unreadable foreign file stays untouched.
    fn save_level(&mut self) {
        let Some(world) = self.world.as_ref() else {
            return;
        };
        let (root, readonly) = {
            let w = world.lock().unwrap_or_else(|e| e.into_inner());
            (w.root.clone(), w.level_readonly)
        };
        if readonly {
            return;
        }
        let mut rules = BTreeMap::new();
        rules.insert(
            "random_tick_speed".to_string(),
            self.tick_speed().to_string(),
        );
        rules.insert(
            "spawn_mobs".to_string(),
            self.spawning.spawn_mobs.to_string(),
        );
        let meta = LevelMeta {
            spawn: self.persistence.spawn,
            day_time: self.day_time,
            game_time: self.spawning.total_ticks as i64,
            game_rules: rules,
            data_version: doppel_world::anvil_write::DATA_VERSION,
        };
        if let Err(e) = doppel_world::level::save(&root, &meta) {
            eprintln!("[persistence] level.dat save failed: {e:#}");
        }
    }
}

/// Captures a player's persisted state.
fn player_data(
    x: f64,
    y: f64,
    z: f64,
    yaw: f32,
    pitch: f32,
    inv: &crate::inventory::PlayerInvState,
) -> PlayerData {
    let mut inventory = Vec::new();
    for slot in 0..TOTAL_SLOTS {
        if let Some(stack) = inv.inventory.get(slot) {
            let mut saved = stack_to_saved(&stack);
            saved.slot = slot as i8;
            inventory.push(saved);
        }
    }
    PlayerData {
        pos: [x, y, z],
        yaw,
        pitch,
        game_mode: u8::from(inv.creative),
        inventory,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{ConnId, Game, Inbound, Outbound};
    use crate::WorldState;
    use doppel_world::anvil_to_wire::PaletteBootstrap;
    use doppel_world::registry::BlockRegistry;
    use doppel_world::worldgen::FlatGenerator;
    use doppel_world::WorldDir;
    use std::sync::{Arc, Mutex};

    /// A unique world root with a region directory, cleaned from any
    /// earlier run.
    fn world_root(tag: &str) -> std::path::PathBuf {
        let root =
            std::env::temp_dir().join(format!("doppel-persist-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("region")).unwrap();
        root
    }

    /// The palette a capture pairing would have learned for the flat
    /// world's blocks.
    fn seeded_boot(registry: &BlockRegistry) -> PaletteBootstrap {
        let mut boot = PaletteBootstrap::default();
        for name in [
            "minecraft:air",
            "minecraft:stone",
            "minecraft:bedrock",
            "minecraft:dirt",
            "minecraft:grass_block",
        ] {
            let id = registry.state_id(name, "").expect("default state");
            boot.blocks.insert(name.into(), id);
        }
        boot.biomes.insert(
            "minecraft:plains".into(),
            doppel_world::worldgen::PLAINS_BIOME_ID,
        );
        boot
    }

    fn game_over(root: &std::path::Path) -> (Game, std::sync::mpsc::Sender<Inbound>) {
        let pins = doppel_protocol::find_repo_root()
            .expect("repo root")
            .join("pins/blocks.json");
        let registry = BlockRegistry::load(&pins).expect("registry pins");
        game_over_with_boot(root, seeded_boot(&registry))
    }

    fn game_over_with_boot(
        root: &std::path::Path,
        boot: PaletteBootstrap,
    ) -> (Game, std::sync::mpsc::Sender<Inbound>) {
        let world = WorldState {
            dir: WorldDir::open(root).expect("region dir"),
            boot,
            saved: crate::load_saved_set(root),
            root: root.to_path_buf(),
            level: Default::default(),
            level_readonly: false,
        };
        let (tx, rx) = std::sync::mpsc::channel::<Inbound>();
        let game = Game::new(rx, Some(Arc::new(Mutex::new(world))), None);
        (game, tx)
    }

    #[cfg(test)]
    impl Game {
        fn player_pose_for_test(&self, conn: ConnId) -> Option<([f64; 3], f32, f32)> {
            self.players
                .get(&conn)
                .map(|p| ([p.x, p.y, p.z], p.yaw, p.pitch))
        }
    }

    /// Copies a directory tree (the test's world fixture).
    fn copy_tree(from: &std::path::Path, to: &std::path::Path) {
        std::fs::create_dir_all(to).unwrap();
        for entry in std::fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let src = entry.path();
            let dst = to.join(entry.file_name());
            if src.is_dir() {
                copy_tree(&src, &dst);
            } else if entry.file_name() != "session.lock" {
                std::fs::copy(&src, &dst).unwrap();
            }
        }
    }

    /// A copy of the local vanilla run world, when one exists.
    fn vanilla_world_copy(tag: &str) -> Option<std::path::PathBuf> {
        let root = doppel_protocol::find_repo_root()
            .expect("repo root")
            .join("target/vanilla/run/world");
        if !root.is_dir() {
            return None;
        }
        let copy =
            std::env::temp_dir().join(format!("doppel-persist-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&copy);
        copy_tree(&root, &copy);
        Some(copy)
    }

    /// The captured reference blob set, when one exists.
    fn reference_blobs() -> Option<Arc<crate::blobs::Blobs>> {
        let dir = doppel_protocol::find_repo_root()
            .expect("repo root")
            .join("target/vanilla/blobs");
        if !dir.join("manifest.json").is_file() {
            return None;
        }
        Some(Arc::new(crate::blobs::load(&dir).expect("blob set")))
    }

    /// A game over the given root, with the reference blobs attached when
    /// they exist.
    fn game_over_with_reference(
        root: &std::path::Path,
        blobs: Option<Arc<crate::blobs::Blobs>>,
    ) -> (Game, std::sync::mpsc::Sender<Inbound>) {
        let pins = doppel_protocol::find_repo_root()
            .expect("repo root")
            .join("pins/blocks.json");
        let registry = BlockRegistry::load(&pins).expect("registry pins");
        let world = WorldState {
            dir: WorldDir::open(root).expect("region dir"),
            boot: seeded_boot(&registry),
            saved: crate::load_saved_set(root),
            root: root.to_path_buf(),
            level: Default::default(),
            level_readonly: false,
        };
        let (tx, rx) = std::sync::mpsc::channel::<Inbound>();
        let game = Game::new(rx, Some(Arc::new(Mutex::new(world))), blobs);
        (game, tx)
    }

    /// Parses every stored chunk in every region file of a world root;
    /// returns (files, chunks, failure strings).
    fn scan_regions(root: &std::path::Path) -> (usize, usize, Vec<String>) {
        let region = doppel_world::anvil_write::region_dir(root);
        let (mut files, mut chunks, mut failures) = (0, 0, Vec::new());
        for entry in std::fs::read_dir(&region).expect("region dir") {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("mca") {
                continue;
            }
            files += 1;
            let region = doppel_world::Region::open(&path).expect("open region");
            for z in 0..32usize {
                for x in 0..32usize {
                    match region.chunk(x, z) {
                        Ok(Some(_)) => chunks += 1,
                        Ok(None) => {}
                        Err(e) => failures.push(format!("{} ({x},{z}): {e:#}", path.display())),
                    }
                }
            }
        }
        (files, chunks, failures)
    }

    /// (r) Chunks the flat path generates inside an existing vanilla
    /// region file, plus an edited captured chunk, save and reload with
    /// every stored chunk still parsing. Runs with the reference blobs
    /// attached when they exist, so captured chunks take the learn path
    /// the live server uses.
    #[test]
    fn generated_chunks_reload_in_vanilla_region() {
        let Some(root) = vanilla_world_copy("vanilla-region") else {
            return;
        };
        let blobs = reference_blobs();
        let (mut g, _tx) = game_over_with_reference(&root, blobs.clone());
        // (20,20) sits inside the existing r.0.0 region but has no stored
        // chunk; loading it runs the flat fallback.
        assert!(g.ensure_chunk_loaded(20, 20), "flat chunk generates");
        g.handle(Inbound::Setblock {
            conn: 0,
            x: 325,
            y: -60,
            z: 325,
            name: "minecraft:stone".into(),
        });
        // A walking line of generated chunks across region boundaries,
        // including a negative-coordinate region.
        for &(cx, cz) in &[(1i32, 0i32), (33, 0), (65, 1), (-33, 5)] {
            assert!(g.ensure_chunk_loaded(cx, cz), "flat chunk ({cx},{cz})");
            g.handle(Inbound::Setblock {
                conn: 0,
                x: cx * 16 + 2,
                y: -60,
                z: cz * 16 + 2,
                name: "minecraft:stone".into(),
            });
        }
        // (0,0) is a stored vanilla chunk; editing it rewrites a slot that
        // already carries a payload, through the reference conversion when
        // blobs are attached.
        g.handle(Inbound::Setblock {
            conn: 0,
            x: 3,
            y: -60,
            z: 5,
            name: "minecraft:stone".into(),
        });
        g.flush_connections();
        assert!(g.persistence.dirty.chunks.contains(&(20, 20)));
        g.flush_all();
        assert!(g.persistence.dirty.chunks.is_empty(), "everything flushes");
        drop(g);

        let (files, chunks, failures) = scan_regions(&root);
        eprintln!(
            "[vanilla-region] blobs={}, {files} files, {chunks} chunks, {} failures",
            blobs.is_some(),
            failures.len()
        );
        for f in failures.iter().take(10) {
            eprintln!("[vanilla-region] {f}");
        }
        assert!(
            failures.is_empty(),
            "{} of {chunks} chunks fail to parse",
            failures.len()
        );

        let (mut g2, _tx2) = game_over_with_reference(&root, blobs);
        assert!(g2.ensure_chunk_loaded(20, 20), "generated chunk reloads");
        assert_eq!(
            g2.get_block(325, -60, 325),
            Some(("minecraft:stone".into(), "".into())),
            "generated-chunk edit survives"
        );
        for &(cx, cz) in &[(1i32, 0i32), (33, 0), (65, 1), (-33, 5)] {
            assert!(g2.ensure_chunk_loaded(cx, cz), "chunk ({cx},{cz}) reloads");
            assert_eq!(
                g2.get_block(cx * 16 + 2, -60, cz * 16 + 2),
                Some(("minecraft:stone".into(), "".into())),
                "edit in ({cx},{cz}) survives"
            );
        }
        // Complete storage is authoritative over the capture: edits inside
        // the blob area survive reload too.
        assert!(g2.ensure_chunk_loaded(0, 0), "captured chunk reloads");
        assert_eq!(
            g2.get_block(3, -60, 5),
            Some(("minecraft:stone".into(), "".into())),
            "captured-chunk edit survives"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// (a) Blocks placed and removed through the engine survive a save
    /// and a fresh boot over the same directory.
    #[test]
    fn world_save_reload_roundtrip() {
        let root = world_root("reload");
        let (mut g, _tx) = game_over(&root);
        let flat = FlatGenerator::classic(&g.registry_snapshot_for_test()).unwrap();
        g.seed_chunk_for_test(0, 0, flat.generate(0, 0));
        g.handle(Inbound::Setblock {
            conn: 0,
            x: 3,
            y: -60,
            z: 5,
            name: "minecraft:stone".into(),
        });
        g.handle(Inbound::Setblock {
            conn: 0,
            x: 2,
            y: -61,
            z: 7,
            name: "minecraft:air".into(),
        });
        g.flush_connections();
        assert!(g.persistence.dirty.chunks.contains(&(0, 0)));
        g.flush_all();
        assert!(g.persistence.dirty.chunks.is_empty());
        assert!(g.persistence.dirty.stop_flushed);
        let level = doppel_world::level::load(&root)
            .unwrap()
            .expect("level written");
        assert_eq!(level.spawn, (0, -60, 0));
        drop(g);

        let (mut g2, _tx2) = game_over(&root);
        assert!(g2.ensure_chunk_loaded(0, 0));
        assert_eq!(
            g2.get_block(3, -60, 5),
            Some(("minecraft:stone".into(), "".into())),
            "placed block"
        );
        assert_eq!(
            g2.get_block(2, -61, 7),
            Some(("minecraft:air".into(), "".into())),
            "removed block"
        );
        // Untouched coordinates of the flat stack read identically.
        assert_eq!(
            g2.get_block(9, -64, 9),
            Some(("minecraft:bedrock".into(), "".into()))
        );
        assert_eq!(
            g2.get_block(9, -61, 9),
            Some(("minecraft:grass_block".into(), "snowy=false".into()))
        );
        assert_eq!(
            g2.get_block(7, -60, 11),
            Some(("minecraft:air".into(), "".into()))
        );
        assert_eq!(
            g2.get_block(2, -63, 7),
            Some(("minecraft:dirt".into(), "".into()))
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// (b) Position, rotation, game mode, and inventory (including a
    /// component patch) restore on rejoin, and the restored position
    /// reaches the client as a position sync.
    #[test]
    fn player_rejoin_restores_state() {
        let root = world_root("rejoin");
        let (mut g, _tx) = game_over(&root);
        let (out_tx, out_rx) = std::sync::mpsc::channel::<Outbound>();
        g.handle(Inbound::Joined {
            conn: 7,
            name: "bot".into(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            sent: Vec::new(),
            tx: out_tx,
        });
        g.handle(Inbound::Moved {
            conn: 7,
            x: 12.5,
            y: -60.0,
            z: 7.25,
            yaw: Some(-90.0),
            pitch: Some(12.5),
        });
        g.handle(Inbound::GameMode {
            conn: 7,
            mode: crate::game::GameMode::Creative,
        });
        // Menu slot 36 (hotbar 0) gets 1x stone with max_stack_size 16.
        let set = crate::inventory::parse_set_creative_slot(&[
            0x00, 0x24, 0x01, 0x01, 0x01, 0x01, 0x01, 0x10, 0x00,
        ])
        .unwrap();
        g.handle(Inbound::CreativeSlot { conn: 7, set });
        g.handle(Inbound::Give {
            conn: 7,
            item: "minecraft:dirt".into(),
            count: 5,
        });
        g.flush_connections();
        while out_rx.try_recv().is_ok() {}
        g.handle(Inbound::Left { conn: 7 });

        let (mut g2, _tx2) = game_over(&root);
        let (out2_tx, out2_rx) = std::sync::mpsc::channel::<Outbound>();
        g2.handle(Inbound::Joined {
            conn: 3,
            name: "bot".into(),
            x: 0.0,
            y: 0.0,
            z: 0.0,
            sent: Vec::new(),
            tx: out2_tx,
        });
        g2.flush_connections();
        while out2_rx.try_recv().is_ok() {}
        assert_eq!(
            g2.player_pose_for_test(3),
            Some(([12.5, -60.0, 7.25], -90.0, 12.5))
        );
        let inv = g2.player_inv_state_for_test(3).expect("player");
        assert!(inv.creative, "game mode restores");
        let patched = inv.inventory.get(0).expect("patched stack restored");
        assert_eq!((patched.count(), patched.item()), (1, 1));
        assert_eq!(
            patched.patch().added,
            vec![(
                crate::inventory::component::MAX_STACK_SIZE,
                crate::inventory::ComponentValue::VarInt(16)
            )]
        );
        let plain = inv.inventory.get(1).expect("plain stack restored");
        assert_eq!(
            (plain.count(), plain.item()),
            (5, crate::inventory::item_id("minecraft:dirt").unwrap())
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A world store with no capture-learned names still saves and
    /// reloads: the registry seeds the default states.
    #[test]
    fn standalone_world_saves_without_learned_boot() {
        let root = world_root("standalone");
        let (mut g, _tx) = game_over_with_boot(&root, PaletteBootstrap::default());
        let flat = FlatGenerator::classic(&g.registry_snapshot_for_test()).unwrap();
        g.seed_chunk_for_test(0, 0, flat.generate(0, 0));
        g.handle(Inbound::Setblock {
            conn: 0,
            x: 4,
            y: -60,
            z: 4,
            name: "minecraft:stone".into(),
        });
        g.flush_connections();
        g.flush_all();
        let (mut g2, _tx2) = game_over_with_boot(&root, PaletteBootstrap::default());
        assert!(g2.ensure_chunk_loaded(0, 0), "saved chunk reloads");
        assert_eq!(
            g2.get_block(4, -60, 4),
            Some(("minecraft:stone".into(), "".into()))
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// (c) A sweep writes at most the budgeted number of chunks and
    /// leaves the rest dirty; repeated sweeps drain the set.
    #[test]
    fn budget_limits_sweep_and_drains() {
        let root = world_root("budget");
        let (mut g, _tx) = game_over(&root);
        let flat = FlatGenerator::classic(&g.registry_snapshot_for_test()).unwrap();
        let mut seeded = 0;
        'seed: for cz in 0..32 {
            for cx in 0..32 {
                if seeded == 1000 {
                    break 'seed;
                }
                g.seed_chunk_for_test(cx, cz, flat.generate(cx, cz));
                g.mark_chunk_dirty(cx, cz);
                seeded += 1;
            }
        }
        assert_eq!(seeded, 1000);
        assert_eq!(g.persistence.dirty.chunks.len(), 1000);

        let written = g.save_sweep(7, SWEEP_DEADLINE);
        assert_eq!(written, 7);
        assert_eq!(g.persistence.dirty.chunks.len(), 993);

        let mut sweeps = 0;
        while !g.persistence.dirty.chunks.is_empty() {
            let written = g.save_sweep(64, Duration::from_secs(5));
            assert!(written > 0, "sweep made no progress");
            sweeps += 1;
            assert!(sweeps < 100, "drain did not finish");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A sweep respects its deadline: region passes after the deadline
    /// stay dirty for the next tick, and a minimal sweep does not stall.
    #[test]
    fn sweep_respects_deadline() {
        let root = world_root("deadline");
        let (mut g, _tx) = game_over(&root);
        let flat = FlatGenerator::classic(&g.registry_snapshot_for_test()).unwrap();
        // Two chunks in two different region files, one batch.
        g.seed_chunk_for_test(0, 0, flat.generate(0, 0));
        g.seed_chunk_for_test(32, 0, flat.generate(32, 0));
        g.mark_chunk_dirty(0, 0);
        g.mark_chunk_dirty(32, 0);
        // A zero deadline is already past: the first region pass runs,
        // the second is cut.
        let written = g.save_sweep(2, Duration::ZERO);
        assert_eq!(written, 1, "only the first region pass runs");
        assert_eq!(g.persistence.dirty.chunks.len(), 1);
        assert_eq!(
            g.persistence.dirty.chunks.iter().next().copied(),
            Some((32, 0)),
            "the unwritten region stays dirty"
        );
        // A normal-budget sweep of one chunk completes without stalling.
        let start = Instant::now();
        let written = g.save_sweep(1, SWEEP_DEADLINE);
        let elapsed = start.elapsed();
        assert_eq!(written, 1);
        assert!(
            g.persistence.dirty.chunks.is_empty(),
            "the leftover chunk drains"
        );
        assert!(
            elapsed <= Duration::from_millis(500),
            "sweep took {elapsed:?}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The autosave cadence arms the drain and the stop flush marks the
    /// dirty set.
    #[test]
    fn autosave_cadence_arms_sweep() {
        let root = world_root("cadence");
        let (mut g, _tx) = game_over(&root);
        let flat = FlatGenerator::classic(&g.registry_snapshot_for_test()).unwrap();
        g.seed_chunk_for_test(0, 0, flat.generate(0, 0));
        g.handle(Inbound::Setblock {
            conn: 0,
            x: 1,
            y: -60,
            z: 1,
            name: "minecraft:stone".into(),
        });
        g.flush_connections();
        // Ticks below the cadence do not sweep.
        for _ in 0..10 {
            g.persistence_tick();
        }
        assert!(g.persistence.dirty.chunks.contains(&(0, 0)));
        // The cadence arms the drain and the budgeted phase empties it.
        for _ in 0..AUTOSAVE_INTERVAL_TICKS {
            g.persistence_tick();
        }
        assert!(
            g.persistence.dirty.chunks.is_empty(),
            "autosave drains the dirty set"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
