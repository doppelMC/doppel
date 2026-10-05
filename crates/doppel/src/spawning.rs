//! Natural spawning for the monster category, plus the day-time model
//! the darkness and burn checks read. The cycle picks one random
//! column per spawnable chunk per tick, walks up to 3 groups of 4
//! positions with the +-6 jitter, and validates each position against
//! the player distances, the ground shape, and the darkness test.

use crate::creeper::Creeper;
use crate::game::entities::block_solid;
use crate::game::Game;
use crate::living::MobKind;
use crate::skeleton::Skeleton;
use crate::spider::Spider;
use crate::zombie::Zombie;

/// Monsters per 289 spawnable chunks (the full square at radius 8).
const CATEGORY_CAP: u64 = 70;
/// The cap divisor: 17^2.
const CAP_DIVISOR: u64 = 289;
/// Minimum 3D distance from every player and the world spawn, squared.
const MIN_DISTANCE_SQ: f64 = 24.0 * 24.0;
/// Despawn distance, squared: spawn positions stay inside it.
const DESPAWN_DISTANCE_SQ: f64 = 128.0 * 128.0;
/// World bottom.
const MIN_Y: i32 = -64;
/// Pack groups per chunk attempt.
const GROUPS: i32 = 3;
/// Pack size for the zombie entry (uniform 4..4).
const PACK_SIZE: i32 = 4;
/// Cluster cap per chunk attempt.
const CLUSTER_CAP: i32 = 4;
/// Jitter width per pack member.
const JITTER: u64 = 6;
/// Total spawn-list weight: spider 100, zombie 90, skeleton 100,
/// creeper 100 (the implemented slice of the plains monster list).
const LIST_WEIGHT: u64 = 390;
/// The spider's weight band and the zombie's.
const SPIDER_WEIGHT: u64 = 100;
const ZOMBIE_WEIGHT: u64 = 90;
/// The skeleton's weight band.
const SKELETON_WEIGHT: u64 = 100;

/// Spawner and world-time state owned by the game thread.
pub(crate) struct SpawnState {
    /// Day time in ticks; stays 0 until a `time set` freezes a value.
    pub day_time: u64,
    /// Whether day time advances each tick.
    pub time_running: bool,
    /// Total ticks counted since time started running.
    pub total_ticks: u64,
    /// The `spawn_mobs` gamerule.
    pub spawn_mobs: bool,
    /// Whether the difficulty is peaceful.
    pub peaceful: bool,
    /// Seed for the spawner splitmix stream.
    seed: u64,
}

impl Default for SpawnState {
    fn default() -> Self {
        SpawnState {
            day_time: 0,
            time_running: false,
            total_ticks: 0,
            spawn_mobs: true,
            peaceful: false,
            seed: 0x5eed_0057,
        }
    }
}

impl SpawnState {
    /// One uniform draw in [0, n).
    pub fn draw(&mut self, n: u64) -> u64 {
        self.seed = self.seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        (z ^ (z >> 31)) % n
    }
}

// ---------------------------------------------------------------------
// Day-time model
// ---------------------------------------------------------------------

/// The sky-light timeline on the 0..15 scale: full through the day,
/// falling to 4 across the dusk ramp, holding overnight, rising across
/// the dawn ramp that wraps to tick 133.
pub fn sky_light_level(t: u64) -> f64 {
    let t = t % 24000;
    if (133..=11867).contains(&t) {
        15.0
    } else if (11867..13670).contains(&t) {
        15.0 - (t - 11867) as f64 / (13670 - 11867) as f64 * 11.0
    } else if (13670..=22330).contains(&t) {
        4.0
    } else {
        let tt = if t > 22330 { t } else { t + 24000 };
        (4.0 + (tt - 22330) as f64 / 1803.0 * 11.0).min(15.0)
    }
}

/// Sky darkening: 0 by day, 11 across the night window.
pub fn sky_darken(t: u64) -> i32 {
    15 - sky_light_level(t).round() as i32
}

/// The undead burn window: dawn (23460) through dusk (12542), wrapping
/// through midnight's edge at 0.
pub fn monsters_burn(t: u64) -> bool {
    let t = t % 24000;
    !(12542..23460).contains(&t)
}

// ---------------------------------------------------------------------
// The spawn cycle
// ---------------------------------------------------------------------

impl Game {
    /// The named-kind summon: the mobs gate's deterministic driver.
    pub(crate) fn spawn_named(&mut self, kind: &str, x: f64, y: f64, z: f64) {
        let mob: Option<Box<dyn MobKind>> = match kind {
            "minecraft:zombie" => Some(Box::new(Zombie::new())),
            "minecraft:skeleton" => Some(Box::new(Skeleton::new())),
            "minecraft:creeper" => Some(Box::new(Creeper::new())),
            "minecraft:spider" => Some(Box::new(Spider::new())),
            _ => None,
        };
        if let Some(mob) = mob {
            self.spawn_mob(x, y, z, mob);
        }
    }

    /// The monster-category natural spawn pass, per spawnable chunk:
    /// cap gate, random start column, conductor abort, then the pack
    /// loop with the jitter and the position checks.
    pub(crate) fn natural_spawns(&mut self) {
        if !self.spawning.spawn_mobs || self.spawning.peaceful {
            return;
        }
        let chunks: Vec<(i32, i32)> = self.viewed_chunks();
        if chunks.is_empty() {
            return;
        }
        // The spawn pass reads real column data; the join replay streams
        // packets without populating the chunk cache, so pull the viewed
        // chunks in before scanning.
        for &(cx, cz) in &chunks {
            self.ensure_chunk_loaded(cx, cz);
        }
        let cap = category_cap(chunks.len() as u64);
        if self.mobs.mobs.len() as u64 >= cap {
            return;
        }
        for &(cx, cz) in &chunks {
            self.spawn_chunk(cx, cz, cap);
        }
    }

    /// One chunk's pack attempt.
    fn spawn_chunk(&mut self, cx: i32, cz: i32, cap: u64) {
        // The start column: one random cell in the chunk, one random y
        // between the world bottom and the surface + 1.
        let x = cx * 16 + self.spawning.draw(16) as i32;
        let z = cz * 16 + self.spawning.draw(16) as i32;
        let Some(surface) = self.column_surface(x, z) else {
            return;
        };
        let span = (surface + 1 - MIN_Y + 1) as u64;
        let y = MIN_Y + self.spawning.draw(span) as i32;
        // A solid start cell (the reference's conductor abort) ends the
        // whole chunk attempt: on a flat surface this is the y filter.
        if block_solid(self, x, y, z) {
            return;
        }
        let day = self.spawning.day_time;
        let mut px = x;
        let mut pz = z;
        let mut cluster = 0;
        let mut pick = 0u64;
        for _ in 0..GROUPS {
            // The group-size roll, spent before the members.
            let _size = self.spawning.draw(PACK_SIZE as u64);
            let mut picked = false;
            let mut member = 0;
            while member < PACK_SIZE && cluster < CLUSTER_CAP {
                px += self.spawning.draw(JITTER) as i32 - self.spawning.draw(JITTER) as i32;
                pz += self.spawning.draw(JITTER) as i32 - self.spawning.draw(JITTER) as i32;
                member += 1;
                if !self.spawn_distance_ok(px, y, pz) {
                    continue;
                }
                if !picked {
                    // The weighted-entry pick and its count resample,
                    // spent at the first distance-valid position.
                    picked = true;
                    pick = self.spawning.draw(LIST_WEIGHT);
                    let _count = self.spawning.draw(1);
                }
                if self.mobs.mobs.len() as u64 >= cap {
                    return;
                }
                let kind = make_kind(pick);
                if !self.spawn_ground_ok(px, y, pz, kind.half_width())
                    || !self.spawn_dark_ok(px, y, pz, day)
                {
                    continue;
                }
                self.spawn_mob(px as f64 + 0.5, y as f64, pz as f64 + 0.5, kind);
                cluster += 1;
            }
        }
    }

    /// The distance rules: beyond 24 from every player and the world
    /// spawn, inside 128 of some player, inside a spawnable chunk.
    fn spawn_distance_ok(&self, x: i32, y: i32, z: i32) -> bool {
        if !self.chunk_viewed(x.div_euclid(16), z.div_euclid(16)) {
            return false;
        }
        let spawn_d2 = (x as f64 - 0.5) * (x as f64 - 0.5) + (z as f64 - 0.5) * (z as f64 - 0.5);
        if spawn_d2 <= MIN_DISTANCE_SQ {
            return false;
        }
        let mut any_near = false;
        for p in self.players.values() {
            let (dx, dy, dz) = (p.x - x as f64, p.y - y as f64, p.z - z as f64);
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 <= MIN_DISTANCE_SQ {
                return false;
            }
            if d2 <= DESPAWN_DISTANCE_SQ {
                any_near = true;
            }
        }
        any_near
    }

    /// The ground rules: solid below, the cell and the one above
    /// clear, and the body's corner columns clear for wide types.
    fn spawn_ground_ok(&self, x: i32, y: i32, z: i32, half: f64) -> bool {
        if !(block_solid(self, x, y - 1, z)
            && !block_solid(self, x, y, z)
            && !block_solid(self, x, y + 1, z))
        {
            return false;
        }
        let cx = x as f64 + 0.5;
        let cz = z as f64 + 0.5;
        for (dx, dz) in [(-half, -half), (half, -half), (-half, half), (half, half)] {
            let (qx, qz) = ((cx + dx) as i32, (cz + dz) as i32);
            if block_solid(self, qx, y, qz) || block_solid(self, qx, y + 1, qz) {
                return false;
            }
        }
        true
    }

    /// The darkness test: the raw sky draw, the block-light ceiling,
    /// then the darkened brightness against the 0..7 sample.
    fn spawn_dark_ok(&mut self, x: i32, y: i32, z: i32, day: u64) -> bool {
        let exposed = self.sky_exposed(x, y, z);
        let raw_sky: i32 = if exposed { 15 } else { 0 };
        // Check 1 reads the undarkened sky light.
        if raw_sky > self.spawning.draw(32) as i32 {
            return false;
        }
        // Check 2: no block-light sources exist on this world yet.
        // Check 3: the darkened brightness against the sample.
        let brightness = (raw_sky - sky_darken(day)).max(0);
        brightness <= self.spawning.draw(8) as i32
    }
}

/// The weighted-entry pick over the implemented bands, in list order.
fn make_kind(pick: u64) -> Box<dyn MobKind> {
    if pick < SPIDER_WEIGHT {
        Box::new(Spider::new())
    } else if pick < SPIDER_WEIGHT + ZOMBIE_WEIGHT {
        Box::new(Zombie::new())
    } else if pick < SPIDER_WEIGHT + ZOMBIE_WEIGHT + SKELETON_WEIGHT {
        Box::new(Skeleton::new())
    } else {
        Box::new(Creeper::new())
    }
}

/// The global category cap: 70 x spawnable chunks / 289.
pub fn category_cap(spawnable_chunks: u64) -> u64 {
    CATEGORY_CAP * spawnable_chunks / CAP_DIVISOR
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_math() {
        assert_eq!(category_cap(81), 19);
        assert_eq!(category_cap(1), 0);
        assert_eq!(category_cap(9), 2);
        assert_eq!(category_cap(25), 6);
        assert_eq!(category_cap(0), 0);
    }

    #[test]
    fn sky_darken_keyframes() {
        // The dawn ramp wraps: tick 0 sits one short of full daylight.
        assert_eq!(sky_darken(0), 1);
        assert_eq!(sky_darken(6000), 0);
        assert_eq!(sky_darken(133), 0);
        // Dusk ramp: tick 13000 sits partway down.
        assert_eq!(sky_darken(13000), 7);
        // Night holds 11 across the window.
        assert_eq!(sky_darken(13670), 11);
        assert_eq!(sky_darken(18000), 11);
        assert_eq!(sky_darken(22330), 11);
        // The dawn ramp climbs back to 0 before 23460+.
        assert!(sky_darken(23460) < 11);
        assert_eq!(sky_darken(24000), sky_darken(0));
    }

    #[test]
    fn burn_window() {
        // Burn runs dawn to dusk, wrapping through 0.
        assert!(!monsters_burn(18000), "midnight does not burn");
        assert!(monsters_burn(6000), "noon burns");
        assert!(monsters_burn(0), "the wrap edge burns");
        assert!(!monsters_burn(12542), "dusk ends the window");
        assert!(monsters_burn(23460), "dawn opens the window");
        assert!(monsters_burn(24000 + 100));
    }

    #[test]
    fn darkness_pass_rate_under_open_midnight_sky() {
        // Raw sky 15 at midnight: check 1 passes 17/32 draws, check 3
        // passes 4/8; the combined rate sits near 0.27.
        let mut s = SpawnState::default();
        let mut pass = 0;
        for _ in 0..4000 {
            if 15 > s.draw(32) as i32 && (15 - sky_darken(18000)) <= s.draw(8) as i32 {
                pass += 1;
            }
        }
        let rate = pass as f64 / 4000.0;
        assert!(
            (0.20..0.34).contains(&rate),
            "midnight open-sky pass rate {rate}"
        );
        // Noon never passes: brightness 15 exceeds the sample ceiling.
        let mut s = SpawnState::default();
        for _ in 0..200 {
            assert!(!(15 > s.draw(32) as i32 && (15 - sky_darken(6000)) <= s.draw(8) as i32));
        }
    }

    // -- integration ---------------------------------------------------

    use crate::game::entities::PACKET_ADD_ENTITY;
    use crate::game::{Game, Inbound, Outbound};
    use crate::living::ENTITY_TYPE_ZOMBIE;
    use doppel_world::WireChunk;
    use std::sync::mpsc;

    /// A 5x5 grass-floored world (surface y=99) with one player at the
    /// center block.
    fn world() -> (Game, mpsc::Receiver<Outbound>) {
        let (_tx, rx) = mpsc::channel::<Inbound>();
        let mut g = Game::new(rx, None, None);
        assert!(g.registry_for_test(), "pins/blocks.json not found");
        let grass = g.resolve_state("minecraft:grass_block").unwrap();
        let mut chunks = Vec::new();
        for cx in -2..=2 {
            for cz in -2..=2 {
                chunks.push((cx, cz));
            }
        }
        for (cx, cz) in &chunks {
            let mut w = WireChunk {
                x: *cx,
                z: *cz,
                heightmaps: Vec::new(),
                sections: Vec::new(),
                block_entities: Vec::new(),
                light: Default::default(),
            };
            for sy in 0..24 {
                let block_states = if sy == 10 {
                    let mut longs = vec![0u64; 256];
                    for (l, slot) in longs.iter_mut().enumerate() {
                        for j in 0..16 {
                            let i = l * 16 + j;
                            let v: u64 = if (i >> 8) == 3 { 1 } else { 0 };
                            *slot |= v << (j * 4);
                        }
                    }
                    doppel_world::chunk_codec::Container::Palette {
                        bits: 4,
                        entries: vec![0, grass],
                        longs,
                    }
                } else {
                    doppel_world::chunk_codec::Container::Single(0)
                };
                w.sections.push(doppel_world::chunk_codec::WireSection {
                    non_empty: if sy == 10 { 256 } else { 0 },
                    fluid: 0,
                    block_states,
                    biomes: doppel_world::chunk_codec::Container::Single(0),
                });
            }
            g.seed_chunk_for_test(*cx, *cz, w);
        }
        let (tx_out, rx_out) = mpsc::channel::<Outbound>();
        g.join_viewer_for_test(0, &chunks, tx_out);
        g.handle(Inbound::Tp {
            conn: 0,
            x: 0.5,
            y: 100.0,
            z: 0.5,
        });
        (g, rx_out)
    }

    /// The (x, y, z, type) of every add_entity frame, varint-skipping
    /// the id and reading the doubles after the uuid and type.
    fn spawn_positions(frames: &[(i32, Vec<u8>)]) -> Vec<(f64, f64, f64, i32)> {
        let mut out = Vec::new();
        for (id, body) in frames {
            if *id != PACKET_ADD_ENTITY {
                continue;
            }
            let mut i = 0usize;
            let skip_varint = |i: &mut usize| {
                while body[*i] & 0x80 != 0 {
                    *i += 1;
                }
                *i += 1;
            };
            skip_varint(&mut i);
            i += 16;
            let mut ty = 0i32;
            let mut shift = 0;
            loop {
                let b = body[i];
                i += 1;
                ty |= ((b & 0x7f) as i32) << shift;
                shift += 7;
                if b & 0x80 == 0 {
                    break;
                }
            }
            let rd = |i: usize| f64::from_be_bytes(body[i..i + 8].try_into().unwrap());
            out.push((rd(i), rd(i + 8), rd(i + 16), ty));
        }
        out
    }

    fn drain(rx: &mpsc::Receiver<Outbound>) -> Vec<(i32, Vec<u8>)> {
        let mut out = Vec::new();
        while let Ok(frame) = rx.try_recv() {
            if let Outbound::Frame { id, body } = frame {
                out.push((id, body));
            }
        }
        out
    }

    #[test]
    fn weighted_pick_covers_all_four_kinds() {
        use crate::living::{ENTITY_TYPE_CREEPER, ENTITY_TYPE_SKELETON, ENTITY_TYPE_SPIDER};
        let ty = |pick: u64| make_kind(pick).type_id();
        assert_eq!(ty(0), ENTITY_TYPE_SPIDER);
        assert_eq!(ty(99), ENTITY_TYPE_SPIDER);
        assert_eq!(ty(100), ENTITY_TYPE_ZOMBIE);
        assert_eq!(ty(189), ENTITY_TYPE_ZOMBIE);
        assert_eq!(ty(190), ENTITY_TYPE_SKELETON);
        assert_eq!(ty(289), ENTITY_TYPE_SKELETON);
        assert_eq!(ty(290), ENTITY_TYPE_CREEPER);
        assert_eq!(ty(389), ENTITY_TYPE_CREEPER);
    }

    /// The wave-2 summon scenario, in process: the same command shape
    /// the mobs gate drives, with the gate's structural assertions.
    #[test]
    fn summon_scenario_matches_the_gate_checks() {
        use crate::explosion::PACKET_EXPLODE;
        use crate::game::entities::{
            PACKET_ADD_ENTITY, PACKET_REMOVE_ENTITIES, PACKET_SET_ENTITY_MOTION,
        };
        use crate::living::{
            ENTITY_TYPE_CREEPER, ENTITY_TYPE_SKELETON, ENTITY_TYPE_SPIDER, PACKET_SET_EQUIPMENT,
        };
        use crate::projectile::ENTITY_TYPE_ARROW;
        let (mut g, rx) = world();
        g.spawning.day_time = 18000;
        g.spawning.spawn_mobs = false;
        // The scripted volley: skeleton, retreat, spider, retreat,
        // creeper, blast.
        g.handle(Inbound::Tp {
            conn: 0,
            x: 0.5,
            y: 100.0,
            z: 0.5,
        });
        g.handle(Inbound::Summon {
            conn: 0,
            kind: "minecraft:skeleton".into(),
            x: 8.5,
            y: 100.0,
            z: 0.5,
        });
        for _ in 0..180 {
            g.tick_once_for_test();
        }
        g.handle(Inbound::Tp {
            conn: 0,
            x: 44.5,
            y: 100.0,
            z: 0.5,
        });
        for _ in 0..20 {
            g.tick_once_for_test();
        }
        g.handle(Inbound::Summon {
            conn: 0,
            kind: "minecraft:spider".into(),
            x: 44.5,
            y: 100.0,
            z: 8.5,
        });
        for _ in 0..90 {
            g.tick_once_for_test();
        }
        g.handle(Inbound::Tp {
            conn: 0,
            x: 0.5,
            y: 100.0,
            z: 36.5,
        });
        for _ in 0..20 {
            g.tick_once_for_test();
        }
        g.handle(Inbound::Summon {
            conn: 0,
            kind: "minecraft:creeper".into(),
            x: 4.5,
            y: 100.0,
            z: 36.5,
        });
        for _ in 0..140 {
            g.tick_once_for_test();
        }
        g.flush_connections();
        let frames = drain(&rx);
        // The add types.
        let mut types = std::collections::BTreeSet::new();
        let mut equipment_ids = Vec::new();
        for (id, body) in &frames {
            if *id != PACKET_ADD_ENTITY {
                continue;
            }
            let (x, y, z, ty) = spawn_decode(body);
            types.insert(ty);
            let _ = (x, y, z);
            if ty == ENTITY_TYPE_SKELETON {
                equipment_ids.push(entity_id_of(body));
            }
        }
        for want in [
            ENTITY_TYPE_SKELETON,
            ENTITY_TYPE_SPIDER,
            ENTITY_TYPE_CREEPER,
            ENTITY_TYPE_ARROW,
        ] {
            assert!(types.contains(&want), "missing add type {want}: {types:?}");
        }
        // The skeleton's main-hand equipment.
        assert!(
            frames.iter().any(|(id, b)| {
                *id == PACKET_SET_EQUIPMENT
                    && equipment_ids.contains(&entity_id_of(b))
                    && b[entity_header_len(b)] == 0
            }),
            "the skeleton pairs with the main-hand bow"
        );
        // The arrow flies: motion or position packets reference it.
        let arrow_id = frames
            .iter()
            .find_map(|(id, b)| {
                if *id == PACKET_ADD_ENTITY && spawn_decode(b).3 == ENTITY_TYPE_ARROW {
                    Some(entity_id_of(b))
                } else {
                    None
                }
            })
            .expect("the arrow add");
        let arrow_moves = frames.iter().any(|(id, b)| {
            (*id == PACKET_SET_ENTITY_MOTION
                || *id == crate::game::entities::PACKET_MOVE_ENTITY_POS
                || *id == crate::game::entities::PACKET_ENTITY_POSITION_SYNC)
                && entity_id_of(b) == arrow_id
        });
        assert!(arrow_moves, "the arrow sends movement");
        // The creeper swells, detonates, and leaves no corpse.
        let creeper_id = frames
            .iter()
            .find_map(|(id, b)| {
                if *id == PACKET_ADD_ENTITY && spawn_decode(b).3 == ENTITY_TYPE_CREEPER {
                    Some(entity_id_of(b))
                } else {
                    None
                }
            })
            .expect("the creeper add");
        let swelled = frames.iter().any(|(id, b)| {
            *id == crate::game::entities::PACKET_SET_ENTITY_DATA
                && entity_id_of(b) == creeper_id
                && b[entity_header_len(b)] == 16
        });
        assert!(swelled, "the swell datum goes out");
        assert!(
            frames.iter().any(|(id, _)| *id == PACKET_EXPLODE),
            "the explosion packet goes out"
        );
        assert!(
            frames.iter().any(|(id, _)| *id == PACKET_REMOVE_ENTITIES),
            "the creeper is removed"
        );
        let updates = frames
            .iter()
            .filter(|(id, _)| *id == 0x08 || *id == 0x56)
            .count();
        assert!(updates > 0, "the blast clears blocks");
    }

    /// The entity id leading a packet body.
    fn entity_id_of(body: &[u8]) -> i32 {
        let mut i = 0usize;
        let mut v = 0i32;
        let mut shift = 0;
        loop {
            let b = body[i];
            i += 1;
            v |= ((b & 0x7f) as i32) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                return v;
            }
        }
    }

    /// The byte length of the leading entity-id varint.
    fn entity_header_len(body: &[u8]) -> usize {
        let mut i = 0usize;
        while body[i] & 0x80 != 0 {
            i += 1;
        }
        i + 1
    }

    /// The (x, y, z, type) of an add_entity body.
    fn spawn_decode(body: &[u8]) -> (f64, f64, f64, i32) {
        let mut i = entity_header_len(body) + 16;
        let mut ty = 0i32;
        let mut shift = 0;
        loop {
            let b = body[i];
            i += 1;
            ty |= ((b & 0x7f) as i32) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                break;
            }
        }
        let rd = |i: usize| f64::from_be_bytes(body[i..i + 8].try_into().unwrap());
        (rd(i), rd(i + 8), rd(i + 16), ty)
    }

    /// Reads the streamed-world chunk under the scenario-two summons,
    /// when a local vanilla capture exists. The flat fallback and the
    /// uncaptured-anvil path must both read a solid floor.
    #[test]
    fn streamed_chunks_hold_the_floor() {
        let root = doppel_protocol::find_repo_root().expect("repo root");
        let pristine = root
            .join("target")
            .join("vanilla")
            .join("pristine-world-mobs");
        let blobs_dir = root.join("target").join("vanilla").join("blobs-mobs");
        if !pristine.is_dir() || !blobs_dir.is_dir() {
            return;
        }
        let (_tx, rx) = mpsc::channel::<Inbound>();
        let mut world = crate::WorldState {
            dir: doppel_world::WorldDir::open(&pristine).expect("world dir"),
            boot: Default::default(),
            root: pristine.clone(),
            level: Default::default(),
            level_readonly: false,
        };
        let blobs = crate::blobs::load(&blobs_dir).expect("blobs");
        // The join replay's learning, replayed here: every captured
        // chunk that exists on disk teaches the palette map.
        let mut learned = 0usize;
        for (id, body) in blobs.play.iter() {
            if *id != 0x2e {
                continue;
            }
            if let Ok(chunk) = doppel_world::WireChunk::decode(body) {
                if let Ok(Some(anvil)) = world.dir.chunk(chunk.x, chunk.z) {
                    world.boot.learn(&chunk, &anvil);
                    learned += 1;
                }
            }
        }
        eprintln!("[probe] learned {learned} reference chunks");
        let mut g = Game::new(
            rx,
            Some(std::sync::Arc::new(std::sync::Mutex::new(world))),
            Some(std::sync::Arc::new(blobs)),
        );
        assert!(g.registry_for_test());
        for &(x, z) in &[(100i32, 100i32), (160, 100), (100, 160), (96, 96)] {
            let cx = x.div_euclid(16);
            let cz = z.div_euclid(16);
            assert!(g.ensure_chunk_loaded(cx, cz), "chunk ({cx},{cz}) loads");
            let label = g.block_label_for_test(x, -61, z);
            eprintln!("[probe] ({x},{z}) floor label: {label}");
            assert!(
                label.starts_with("minecraft:grass") || label.starts_with("minecraft:dirt"),
                "floor at ({x},-61,{z}) reads {label}"
            );
            assert!(
                crate::game::entities::block_solid(&g, x, -61, z),
                "floor at ({x},-61,{z}) reads solid"
            );
        }
    }

    #[test]
    fn midnight_spawns_all_four_kinds() {
        use crate::living::{ENTITY_TYPE_CREEPER, ENTITY_TYPE_SKELETON, ENTITY_TYPE_SPIDER};
        let (mut g, rx) = world();
        g.spawning.day_time = 18000;
        g.spawning.time_running = true;
        for _ in 0..2000 {
            g.tick_once_for_test();
        }
        let frames = drain(&rx);
        let types: std::collections::BTreeSet<i32> = spawn_positions(&frames)
            .into_iter()
            .map(|(_, _, _, t)| t)
            .collect();
        for want in [
            ENTITY_TYPE_ZOMBIE,
            ENTITY_TYPE_SKELETON,
            ENTITY_TYPE_CREEPER,
            ENTITY_TYPE_SPIDER,
        ] {
            assert!(types.contains(&want), "missing type {want} in {types:?}");
        }
    }

    #[test]
    fn midnight_spawns_zombies_within_the_rules() {
        let (mut g, rx) = world();
        g.spawning.day_time = 18000;
        g.spawning.time_running = true;
        for _ in 0..600 {
            g.tick_once_for_test();
        }
        let frames = drain(&rx);
        let spawns = spawn_positions(&frames);
        assert!(!spawns.is_empty(), "monsters appear at midnight");
        assert!(!g.mobs.mobs.is_empty(), "the mob list holds the survivors");
        assert!(
            g.mobs.mobs.len() as u64 <= category_cap(25),
            "the cluster stays under the category cap"
        );
        for (x, y, z, _ty) in &spawns {
            assert!(
                (x - x.floor() - 0.5).abs() < 1.0e-9,
                "x at the block center: {x}"
            );
            assert!(
                (z - z.floor() - 0.5).abs() < 1.0e-9,
                "z at the block center: {z}"
            );
            assert!(y.fract() == 0.0, "integer feet: {y}");
            assert_eq!(*y, 100.0, "one cell above the surface");
            let d2 = (x - 0.5) * (x - 0.5) + (z - 0.5) * (z - 0.5);
            assert!(d2 > 24.0 * 24.0, "beyond 24 from the player: {d2}");
            assert!(d2 < 128.0 * 128.0, "inside 128 of the player: {d2}");
        }
    }

    #[test]
    fn noon_spawns_nothing() {
        let (mut g, rx) = world();
        g.spawning.day_time = 6000;
        g.spawning.time_running = true;
        for _ in 0..400 {
            g.tick_once_for_test();
        }
        assert!(spawn_positions(&drain(&rx)).is_empty());
        assert!(g.mobs.mobs.is_empty());
    }

    #[test]
    fn peaceful_spawns_nothing() {
        let (mut g, rx) = world();
        g.spawning.day_time = 18000;
        g.spawning.time_running = true;
        g.spawning.peaceful = true;
        for _ in 0..200 {
            g.tick_once_for_test();
        }
        assert!(spawn_positions(&drain(&rx)).is_empty());
        assert!(g.mobs.mobs.is_empty());
    }

    #[test]
    fn disabled_rule_spawns_nothing() {
        let (mut g, rx) = world();
        g.spawning.day_time = 18000;
        g.spawning.time_running = true;
        g.spawning.spawn_mobs = false;
        for _ in 0..200 {
            g.tick_once_for_test();
        }
        assert!(spawn_positions(&drain(&rx)).is_empty());
        assert!(g.mobs.mobs.is_empty());
    }
}
