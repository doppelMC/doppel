//! The zombie: an undead melee mob that targets the nearest visible
//! player, strolls and looks around while idle, and burns in daylight.

use crate::living::{
    brightness, ChaseHitGoal, GlanceGoal, GoalCtx, GoalSelector, IdleStrollGoal, MobKind,
    NearestPlayerTargetGoal, WatchPlayerGoal, FIRE_IGNITE_TICKS,
};
use crate::spawning::monsters_burn;

/// Follow range (the attribute at default).
const FOLLOW_RANGE: f64 = 35.0;
/// Movement speed (the attribute at default).
const SPEED: f64 = 0.23;
/// Attack damage (the attribute at default).
const ATTACK_DAMAGE: f32 = 3.0;
/// Hitbox half width.
const HALF_WIDTH: f64 = 0.3;
/// Hitbox height.
const HEIGHT: f64 = 1.95;
/// Eye height.
const EYE: f64 = 1.74;
/// Melee reach: both half widths plus the 0.83 inflation.
const REACH: f64 = HALF_WIDTH + 0.3 + 0.83;
/// Ticks between target scans, in full-cadence passes.
const TARGET_SCAN_EVERY: i32 = 5;
/// Unseen-tick budget before a target drops.
const UNSEEN_LIMIT: i32 = 30;
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

// ---------------------------------------------------------------------
// The kind
// ---------------------------------------------------------------------

/// The zombie kind: parameters, goals, and the daylight burn.
pub struct Zombie;

impl Zombie {
    pub fn new() -> Zombie {
        Zombie
    }
}

impl Default for Zombie {
    fn default() -> Self {
        Self::new()
    }
}

impl MobKind for Zombie {
    fn type_id(&self) -> i32 {
        crate::living::ENTITY_TYPE_ZOMBIE
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
        "Zombie"
    }

    fn follow_range(&self) -> f64 {
        FOLLOW_RANGE
    }

    fn register_goals(&self, goals: &mut GoalSelector, targets: &mut GoalSelector) {
        goals.add(3, Box::new(ChaseHitGoal::new(REACH, FOLLOW_RANGE, true)));
        goals.add(
            7,
            Box::new(IdleStrollGoal::new(
                STROLL_RANGE,
                STROLL_CHANCE,
                STROLL_GIVE_UP,
                1.0,
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
                None,
            )),
        );
    }

    fn kind_tick(&mut self, ctx: &mut GoalCtx) {
        let day = ctx.world.spawning.day_time;
        // Bright locations hasten the despawn clock.
        let bright = if ctx.body.exposed {
            brightness(ctx.world, ctx.body)
        } else {
            0.0
        };
        if bright > 0.5 {
            ctx.body.no_action_time += 2;
        }
        // Daylight ignition: the burn window, sky-exposed eyes, the
        // ratio above the threshold, then the roll.
        if ctx.body.fire_ticks > 0 || !monsters_burn(day) || !ctx.body.exposed || bright <= 0.5 {
            return;
        }
        let roll = (ctx.draw() % 1000) as f64 / 1000.0 * 30.0;
        if roll < (bright - 0.4) * 2.0 {
            ctx.body.fire_ticks = FIRE_IGNITE_TICKS;
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::entities::{block_solid, PACKET_SET_ENTITY_DATA};
    use crate::game::{Inbound, Outbound};
    use crate::living::PACKET_DAMAGE_EVENT;
    use doppel_world::WireChunk;
    use std::sync::mpsc;

    /// A grass-floored chunk (surface y=99) with one player the tests
    /// teleport around.
    fn harness() -> (crate::game::Game, mpsc::Receiver<Outbound>) {
        let (_tx, rx) = mpsc::channel::<Inbound>();
        let mut g = crate::game::Game::new(rx, None, None);
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
    fn melee_hits_on_the_cooldown_cadence() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 5.5,
            y: 100.0,
            z: 5.5,
        });
        // Inside the reach, line of sight clear.
        g.spawn_mob(6.5, 100.0, 5.5, Box::new(Zombie::new()));
        for _ in 0..12 {
            g.tick_once_for_test();
        }
        let first = drain(&rx);
        let hits = first
            .iter()
            .filter(|(id, _)| *id == PACKET_DAMAGE_EVENT)
            .count();
        assert!(hits >= 1, "the first hit lands inside 12 ticks");
        // The cooldown spaces the hits: no burst.
        for _ in 0..6 {
            g.tick_once_for_test();
        }
        let mid = drain(&rx);
        let mid_hits = mid
            .iter()
            .filter(|(id, _)| *id == PACKET_DAMAGE_EVENT)
            .count();
        assert!(mid_hits <= 1, "{mid_hits} hits inside the cooldown window");
        for _ in 0..12 {
            g.tick_once_for_test();
        }
        let late = drain(&rx);
        assert!(
            late.iter().any(|(id, _)| *id == PACKET_DAMAGE_EVENT),
            "the attack repeats after the cooldown"
        );
    }

    #[test]
    fn distant_player_draws_no_attack() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 40.5,
            y: 100.0,
            z: 5.5,
        });
        g.spawn_mob(0.5, 100.0, 5.5, Box::new(Zombie::new()));
        for _ in 0..60 {
            g.tick_once_for_test();
        }
        let frames = drain(&rx);
        assert!(
            !frames.iter().any(|(id, _)| *id == PACKET_DAMAGE_EVENT),
            "beyond the follow range no target forms"
        );
    }

    #[test]
    fn chase_closes_the_distance() {
        let (mut g, _rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 0.5,
            y: 100.0,
            z: 0.5,
        });
        g.spawn_mob(12.5, 100.0, 8.5, Box::new(Zombie::new()));
        let dist = |g: &crate::game::Game| {
            let m = &g.mobs.mobs[0];
            let (dx, dz) = (m.body.x - 0.5, m.body.z - 0.5);
            (dx * dx + dz * dz).sqrt()
        };
        let start = dist(&g);
        for _ in 0..60 {
            g.tick_once_for_test();
        }
        let end = dist(&g);
        assert!(end < start - 1.0, "chase moved closer: {start} -> {end}");
    }

    #[test]
    fn navigation_routes_around_a_wall() {
        let (mut g, _rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 0.5,
            y: 100.0,
            z: 1.5,
        });
        // A 2-tall wall at x=6 across z=0..=3; the gap opens north.
        for z in 0..=3i32 {
            for y in 100..=101i32 {
                g.handle(Inbound::Setblock {
                    conn: 0,
                    x: 6,
                    y,
                    z,
                    name: "minecraft:stone".to_string(),
                });
            }
        }
        g.spawn_mob(12.5, 100.0, 1.5, Box::new(Zombie::new()));
        let mut nav = crate::pathing::Nav::new();
        nav.move_to(0.5, 1.5, 1.0);
        nav.nav_tick(12.5, 100.0, 1.5, true, &|x, y, z| block_solid(&g, x, y, z));
        let route = nav.route();
        assert!(
            !route.iter().any(|wp| wp.x == 6 && wp.z <= 3),
            "the waypoints avoid the wall: {route:?}"
        );
        assert_eq!(route.last().map(|wp| (wp.x, wp.z)), Some((0, 1)));
    }

    #[test]
    fn chase_closes_through_a_gap() {
        let (mut g, _rx) = harness();
        // A 2-tall wall at x=6 across z=0..=5; the sight line and the
        // route pass through the z=6 gap.
        for z in 0..=5i32 {
            for y in 100..=101i32 {
                g.handle(Inbound::Setblock {
                    conn: 0,
                    x: 6,
                    y,
                    z,
                    name: "minecraft:stone".to_string(),
                });
            }
        }
        g.handle(Inbound::Tp {
            conn: 0,
            x: 0.5,
            y: 100.0,
            z: 6.5,
        });
        g.spawn_mob(12.5, 100.0, 6.5, Box::new(Zombie::new()));
        let dist = |g: &crate::game::Game| {
            let m = &g.mobs.mobs[0];
            let (dx, dz) = (m.body.x - 0.5, m.body.z - 6.5);
            (dx * dx + dz * dz).sqrt()
        };
        let start = dist(&g);
        for _ in 0..200 {
            g.tick_once_for_test();
        }
        let end = dist(&g);
        assert!(
            end < start - 8.0,
            "the chase crossed the gap: {start} -> {end}"
        );
    }

    #[test]
    fn noon_ignites_exposed_zombies() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 0.5,
            y: 100.0,
            z: 0.5,
        });
        g.spawning.day_time = 6000;
        g.spawn_mob(5.5, 100.0, 5.5, Box::new(Zombie::new()));
        let mut lit = false;
        for _ in 0..400 {
            g.tick_once_for_test();
            if g.mobs.mobs[0].body.fire_ticks > 0 {
                lit = true;
                break;
            }
        }
        assert!(lit, "full daylight under an open sky ignites");
        let frames = drain(&rx);
        assert!(
            frames.iter().any(|(id, b)| *id == PACKET_SET_ENTITY_DATA
                && b.len() >= 4
                && b[1] == 0
                && b[3] & 0x01 == 0x01),
            "the flame reaches the entity flags"
        );
    }

    #[test]
    fn midnight_never_ignites() {
        let (mut g, _rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 0.5,
            y: 100.0,
            z: 0.5,
        });
        g.spawning.day_time = 18000;
        g.spawn_mob(5.5, 100.0, 5.5, Box::new(Zombie::new()));
        for _ in 0..300 {
            g.tick_once_for_test();
            assert_eq!(
                g.mobs.mobs[0].body.fire_ticks, 0,
                "the burn window is closed at midnight"
            );
        }
    }
}
