//! The spider: a fast wall-crawling hostile that leaps at close
//! targets, climbs while pressed into walls, and turns hostile only in
//! the dark.

use crate::living::{
    ChaseHitGoal, GlanceGoal, Goal, GoalCtx, GoalFlags, GoalSelector, IdleStrollGoal, MobKind,
    NearestPlayerTargetGoal, WatchPlayerGoal,
};

/// Follow range (the attribute default).
const FOLLOW_RANGE: f64 = 32.0;
/// Movement speed (the attribute).
const SPEED: f64 = 0.3;
/// Attack damage (the attribute default).
const ATTACK_DAMAGE: f32 = 2.0;
/// Hitbox half width.
const HALF_WIDTH: f64 = 0.7;
/// Hitbox height.
const HEIGHT: f64 = 0.9;
/// Eye height.
const EYE: f64 = 0.65;
/// Max health (the attribute override).
const MAX_HEALTH: f32 = 16.0;
/// Melee reach: both half widths plus the 0.83 inflation.
const REACH: f64 = HALF_WIDTH + 0.3 + 0.83;
/// The leap window: squared 2..4 blocks.
const LEAP_MIN_SQ: f64 = 4.0;
const LEAP_MAX_SQ: f64 = 16.0;
/// The leap rise and push, with the carried momentum.
const LEAP_VY: f64 = 0.4;
const LEAP_PUSH: f64 = 0.4;
const LEAP_CARRY: f64 = 0.2;
/// Hostile below this brightness.
const HOSTILE_BELOW: f64 = 0.5;
/// Stroll target radius, horizontal.
const STROLL_RANGE: i64 = 10;
/// Stroll chance per full check (1/60).
const STROLL_CHANCE: u64 = 60;
/// Look-goal chance per full check (2%).
const LOOK_CHANCE: u64 = 50;
/// Look-at-player range.
const LOOK_RANGE: f64 = 8.0;
/// Stroll give-up, in ticks.
const STROLL_GIVE_UP: i32 = 200;
/// Ticks between target scans, in full-cadence passes.
const TARGET_SCAN_EVERY: i32 = 5;
/// Unseen-tick budget before a target drops.
const UNSEEN_LIMIT: i32 = 30;

// ---------------------------------------------------------------------
// Goals
// ---------------------------------------------------------------------

/// The leap: leave the ground toward the target from 2 to 4 blocks on
/// a one-in-three roll.
struct PounceGoal;

impl Goal for PounceGoal {
    fn flags(&self) -> GoalFlags {
        GoalFlags::JUMP.union(GoalFlags::MOVE)
    }

    fn can_use(&mut self, ctx: &mut GoalCtx) -> bool {
        let Some(conn) = ctx.body.target else {
            return false;
        };
        let Some((x, _, z)) = ctx.player_pos(conn) else {
            return false;
        };
        let d2 = (x - ctx.body.x) * (x - ctx.body.x) + (z - ctx.body.z) * (z - ctx.body.z);
        if !(LEAP_MIN_SQ..=LEAP_MAX_SQ).contains(&d2) {
            return false;
        }
        ctx.body.on_ground && ctx.below(3) == 0
    }

    fn can_continue_to_use(&mut self, ctx: &mut GoalCtx) -> bool {
        !ctx.body.on_ground
    }

    fn start(&mut self, ctx: &mut GoalCtx) {
        let Some(conn) = ctx.body.target else {
            return;
        };
        let Some((x, _, z)) = ctx.player_pos(conn) else {
            return;
        };
        let (ddx, ddz) = (x - ctx.body.x, z - ctx.body.z);
        let dl = (ddx * ddx + ddz * ddz).sqrt();
        if dl > 1.0e-4 {
            ctx.body.vx = ddx / dl * LEAP_PUSH + ctx.body.vx * LEAP_CARRY;
            ctx.body.vz = ddz / dl * LEAP_PUSH + ctx.body.vz * LEAP_CARRY;
            ctx.body.vy = LEAP_VY;
        }
    }
}

// ---------------------------------------------------------------------
// The kind
// ---------------------------------------------------------------------

/// The spider kind: a fast climbing melee mob hostile in darkness.
pub struct Spider;

impl Spider {
    pub fn new() -> Spider {
        Spider
    }
}

impl Default for Spider {
    fn default() -> Self {
        Self::new()
    }
}

impl MobKind for Spider {
    fn type_id(&self) -> i32 {
        crate::living::ENTITY_TYPE_SPIDER
    }

    fn half_width(&self) -> f64 {
        HALF_WIDTH
    }

    fn height(&self) -> f64 {
        HEIGHT
    }

    fn eye(&self) -> f64 {
        EYE
    }

    fn base_speed(&self) -> f64 {
        SPEED
    }

    fn attack_damage(&self) -> f32 {
        ATTACK_DAMAGE
    }

    fn display_name(&self) -> &'static str {
        "Spider"
    }

    fn follow_range(&self) -> f64 {
        FOLLOW_RANGE
    }

    fn max_health(&self) -> f32 {
        MAX_HEALTH
    }

    fn can_climb(&self) -> bool {
        true
    }

    fn register_goals(&self, goals: &mut GoalSelector, targets: &mut GoalSelector) {
        goals.add(3, Box::new(PounceGoal));
        goals.add(4, Box::new(ChaseHitGoal::new(REACH, FOLLOW_RANGE, true)));
        goals.add(
            5,
            Box::new(IdleStrollGoal::new(
                STROLL_RANGE,
                STROLL_CHANCE,
                STROLL_GIVE_UP,
                0.8,
            )),
        );
        goals.add(8, Box::new(WatchPlayerGoal::new(LOOK_RANGE, LOOK_CHANCE)));
        goals.add(8, Box::new(GlanceGoal::new(LOOK_CHANCE)));
        targets.add(
            2,
            Box::new(NearestPlayerTargetGoal::new(
                TARGET_SCAN_EVERY,
                UNSEEN_LIMIT,
                FOLLOW_RANGE,
                Some(HOSTILE_BELOW),
            )),
        );
    }

    fn kind_tick(&mut self, _ctx: &mut GoalCtx) {}
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::entities::{block_solid, PACKET_SET_ENTITY_DATA};
    use crate::game::{Game, Inbound, Outbound};
    use crate::living::{DATA_CLIMBING_FLAGS, ENTITY_TYPE_SPIDER};
    use doppel_world::WireChunk;
    use std::sync::mpsc;

    /// A grass-floored chunk (surface y=99) with one player the tests
    /// teleport around.
    fn harness() -> (Game, mpsc::Receiver<Outbound>) {
        let (_tx, rx) = mpsc::channel::<Inbound>();
        let mut g = Game::new(rx, None, None);
        assert!(g.registry_for_test(), "pins/blocks.json not found");
        let grass = g.resolve_state("minecraft:grass_block").unwrap();
        let mut w = WireChunk {
            x: 0,
            z: 0,
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
        g.seed_chunk_for_test(0, 0, w);
        let (tx_out, rx_out) = mpsc::channel::<Outbound>();
        g.join_viewer_for_test(0, &[(0, 0)], tx_out);
        (g, rx_out)
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
    fn hostile_only_in_the_dark() {
        let (mut g, _rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 6.5,
            y: 100.0,
            z: 5.5,
        });
        // Daylight: no target forms.
        g.spawning.day_time = 6000;
        g.spawn_mob(5.5, 100.0, 5.5, Box::new(Spider::new()));
        for _ in 0..80 {
            g.tick_once_for_test();
        }
        assert!(
            g.mobs.mobs[0].body.target.is_none(),
            "the bright spider stays neutral"
        );
        // Midnight: the target forms.
        g.spawning.day_time = 18000;
        for _ in 0..80 {
            g.tick_once_for_test();
            if g.mobs.mobs[0].body.target.is_some() {
                break;
            }
        }
        assert!(
            g.mobs.mobs[0].body.target.is_some(),
            "the dark spider turns hostile"
        );
    }

    #[test]
    fn the_climbing_path_is_chosen() {
        let (mut g, _rx) = harness();
        // A 3-tall wall sealing x=8 across the corridor.
        for z in 0..=15i32 {
            for y in 100..=102i32 {
                g.handle(Inbound::Setblock {
                    conn: 0,
                    x: 8,
                    y,
                    z,
                    name: "minecraft:stone".to_string(),
                });
            }
        }
        g.spawn_mob(5.5, 100.0, 5.5, Box::new(Spider::new()));
        let mut nav = crate::pathing::Nav::new();
        nav.set_climb(true);
        nav.move_to(12.5, 5.5, 1.0);
        nav.nav_tick(5.5, 100.0, 5.5, true, &|x, y, z| block_solid(&g, x, y, z));
        let route = nav.route();
        assert!(
            route.iter().any(|wp| wp.wall.is_some()),
            "the route climbs the wall: {route:?}"
        );
        assert_eq!(route.last().map(|wp| (wp.x, wp.z)), Some((12, 5)));
        // A non-climbing navigator finds no way over the seal.
        let mut plain = crate::pathing::Nav::new();
        plain.move_to(12.5, 5.5, 1.0);
        plain.nav_tick(5.5, 100.0, 5.5, true, &|x, y, z| block_solid(&g, x, y, z));
        assert!(
            !plain.in_progress(),
            "the plain walker gives up at the seal"
        );
    }

    #[test]
    fn the_climb_bit_sets_against_a_wall() {
        let (mut g, rx) = harness();
        for z in 0..=15i32 {
            for y in 100..=102i32 {
                g.handle(Inbound::Setblock {
                    conn: 0,
                    x: 8,
                    y,
                    z,
                    name: "minecraft:stone".to_string(),
                });
            }
        }
        g.handle(Inbound::Tp {
            conn: 0,
            x: 13.5,
            y: 100.0,
            z: 5.5,
        });
        g.spawning.day_time = 18000;
        g.spawn_mob(5.5, 100.0, 5.5, Box::new(Spider::new()));
        let mut climbed = false;
        for _ in 0..120 {
            g.tick_once_for_test();
            if g.mobs.mobs[0].body.climbing {
                climbed = true;
                break;
            }
        }
        assert!(climbed, "the pressed spider raises the climb bit");
        g.flush_connections();
        let frames = drain(&rx);
        assert!(
            frames.iter().any(|(id, b)| {
                *id == PACKET_SET_ENTITY_DATA && b.len() >= 4 && b[1] == DATA_CLIMBING_FLAGS
            }),
            "the climbing datum goes out"
        );
    }

    #[test]
    fn spiders_pair_with_their_attributes() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 5.5,
            y: 100.0,
            z: 5.5,
        });
        g.spawn_mob(6.5, 100.0, 5.5, Box::new(Spider::new()));
        g.flush_connections();
        assert_eq!(g.mobs.mobs[0].kind.type_id(), ENTITY_TYPE_SPIDER);
        let frames = drain(&rx);
        // The spider's snapshot carries movement speed alone: even the
        // off-default max health (16) stays off the reference's
        // pairing (id, count, (attr, f64, no modifiers) pairs).
        let attrs = frames
            .iter()
            .find(|(id, _)| *id == crate::living::PACKET_UPDATE_ATTRIBUTES)
            .map(|(_, b)| b.clone())
            .expect("the attribute packet pairs");
        let mut i = 0usize;
        let rd_varint = |body: &[u8], i: &mut usize| {
            let mut v = 0i32;
            let mut shift = 0;
            loop {
                let b = body[*i];
                *i += 1;
                v |= ((b & 0x7f) as i32) << shift;
                shift += 7;
                if b & 0x80 == 0 {
                    return v;
                }
            }
        };
        rd_varint(&attrs, &mut i);
        let count = rd_varint(&attrs, &mut i);
        assert_eq!(count, 1, "movement speed alone pairs");
        let mut seen = Vec::new();
        for _ in 0..count {
            let attr = rd_varint(&attrs, &mut i);
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&attrs[i..i + 8]);
            i += 8;
            let value = f64::from_be_bytes(bytes);
            let mods = rd_varint(&attrs, &mut i);
            assert_eq!(mods, 0, "no modifiers");
            seen.push((attr, value));
        }
        assert!(
            seen.contains(&(crate::living::ATTR_MOVEMENT_SPEED, 0.3)),
            "movement_speed 0.3: {seen:?}"
        );
    }
}
