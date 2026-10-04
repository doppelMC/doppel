//! The skeleton: an undead ranged mob that chases into its bow radius,
//! holds distance while strafing, draws and fires on a fixed cadence,
//! and burns in daylight.

use crate::game::{ConnId, Game};
use crate::inventory::{item_id, ItemStack};
use crate::living::{
    brightness, eye_at, eye_of, look_angles, visible, Goal, GoalCtx, GoalFlags, GoalSelector,
    LookAtPlayerGoal, MobKind, NearestPlayerTargetGoal, RandomLookGoal, RandomStrollGoal,
    EQUIP_MAIN_HAND, FIRE_IGNITE_TICKS, PLAYER_EYE,
};
use crate::projectile::{mob_base_damage, shot_velocity};
use crate::spawning::monsters_burn;

/// Follow range (the attribute default).
const FOLLOW_RANGE: f64 = 32.0;
/// Movement speed (the attribute).
const SPEED: f64 = 0.25;
/// Attack damage (the attribute; arrows carry their own).
const ATTACK_DAMAGE: f32 = 2.0;
/// Hitbox half width.
const HALF_WIDTH: f64 = 0.3;
/// Hitbox height.
const HEIGHT: f64 = 1.99;
/// Eye height.
const EYE: f64 = 1.74;
/// The bow radius, squared.
const ATTACK_RADIUS_SQ: f64 = 225.0;
/// Attack interval on the non-hard difficulties, in ticks.
const ATTACK_INTERVAL: i32 = 40;
/// Draw ticks to a full-power shot.
const DRAW_TICKS: i32 = 20;
/// Seen ticks before the skeleton holds position.
const SEE_TIME_HOLD: i32 = 20;
/// The unseen floor that keeps the draw alive.
const UNSEEN_FLOOR: i32 = -60;
/// Strafe flip cadence, in strafe ticks, and the flip chance (3/10).
const STRAFE_FLIP_TICKS: i32 = 20;
const STRAFE_FLIP_CHANCE: u64 = 10;
const STRAFE_FLIP_HITS: u64 = 3;
/// Backwards inside a quarter of the radius, forwards beyond three
/// quarters.
const STRAFE_BACK_SQ: f64 = ATTACK_RADIUS_SQ * 0.25;
const STRAFE_FWD_SQ: f64 = ATTACK_RADIUS_SQ * 0.75;
/// The strafe walk offset, in blocks.
const STRAFE_OFFSET: f64 = 2.0;
/// Launch speed.
const LAUNCH_SPEED: f64 = 1.6;
/// Aim uncertainty (the non-hard value).
const UNCERTAINTY: f64 = 10.0;
/// The vertical aim lead per block of distance.
const AIM_LEAD: f64 = 0.2;
/// The aim height above the target's feet.
const AIM_HEIGHT: f64 = 1.0 / 3.0;
/// The spawn origin drop below the eye.
const SPAWN_DROP: f64 = 0.1;
/// The shot draw power at a full 20-tick draw.
const FULL_POWER: f64 = 1.0;
/// The difficulty id feeding the damage triangle (easy).
const DIFFICULTY_ID: f64 = 1.0;
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

/// The ranged attack: chase into the radius, hold and strafe at mid
/// range, draw and release on the cadence.
struct RangedBowAttackGoal {
    attack_time: i32,
    see_time: i32,
    draw: Option<i32>,
    strafe_time: i32,
    clockwise: bool,
    backwards: bool,
    last_path: (f64, f64),
    target: Option<ConnId>,
}

impl RangedBowAttackGoal {
    fn new() -> RangedBowAttackGoal {
        RangedBowAttackGoal {
            attack_time: 0,
            see_time: 0,
            draw: None,
            strafe_time: -1,
            clockwise: false,
            backwards: false,
            last_path: (0.0, 0.0),
            target: None,
        }
    }
}

impl Goal for RangedBowAttackGoal {
    fn flags(&self) -> GoalFlags {
        GoalFlags::MOVE.union(GoalFlags::LOOK)
    }

    fn can_use(&mut self, ctx: &mut GoalCtx) -> bool {
        let Some(conn) = ctx.body.target else {
            return false;
        };
        self.target = ctx.player_pos(conn).map(|_| conn);
        self.target.is_some()
    }

    fn can_continue_to_use(&mut self, ctx: &mut GoalCtx) -> bool {
        self.target.is_some_and(|conn| {
            ctx.body.target == Some(conn) || ctx.body.nav.in_progress()
        })
    }

    fn start(&mut self, ctx: &mut GoalCtx) {
        ctx.body.melee_active = true;
    }

    fn stop(&mut self, ctx: &mut GoalCtx) {
        ctx.body.melee_active = false;
        ctx.body.look = None;
        ctx.body.nav.stop();
        self.see_time = 0;
        self.attack_time = 0;
        self.draw = None;
        self.strafe_time = -1;
        self.target = None;
    }

    fn tick(&mut self, ctx: &mut GoalCtx) {
        let Some(conn) = self.target.or(ctx.body.target) else {
            return;
        };
        let Some((px, py, pz)) = ctx.player_pos(conn) else {
            return;
        };
        let d2 = (px - ctx.body.x) * (px - ctx.body.x) + (pz - ctx.body.z) * (pz - ctx.body.z);
        let seen = visible(
            ctx.world,
            eye_of(ctx.body),
            eye_at(PLAYER_EYE, (px, py, pz)),
        );
        if seen != (self.see_time > 0) {
            self.see_time = 0;
        }
        self.see_time = if seen {
            self.see_time + 1
        } else {
            self.see_time - 1
        };
        ctx.body.look = Some(look_angles(
            eye_of(ctx.body),
            eye_at(PLAYER_EYE, (px, py, pz)),
        ));
        // Chase into the radius or while the target stays unseen;
        // otherwise hold position and strafe.
        if d2 > ATTACK_RADIUS_SQ || self.see_time < SEE_TIME_HOLD {
            let moved = (px - self.last_path.0) * (px - self.last_path.0)
                + (pz - self.last_path.1) * (pz - self.last_path.1);
            if moved >= 1.0 || ctx.below(20) == 0 {
                self.last_path = (px, pz);
                ctx.body.nav.retarget(px, pz, 1.0);
            }
            self.strafe_time = -1;
        } else {
            ctx.body.nav.stop();
            self.strafe_time += 1;
        }
        if self.strafe_time >= STRAFE_FLIP_TICKS {
            if ctx.below(STRAFE_FLIP_CHANCE) < STRAFE_FLIP_HITS {
                self.clockwise = !self.clockwise;
            }
            if ctx.below(STRAFE_FLIP_CHANCE) < STRAFE_FLIP_HITS {
                self.backwards = !self.backwards;
            }
            self.strafe_time = 0;
        }
        if self.strafe_time > -1 {
            if d2 > STRAFE_FWD_SQ {
                self.backwards = false;
            } else if d2 < STRAFE_BACK_SQ {
                self.backwards = true;
            }
            // A short offset walk: back or forward plus the side.
            let (ddx, ddz) = (px - ctx.body.x, pz - ctx.body.z);
            let dl = (ddx * ddx + ddz * ddz).sqrt().max(1.0e-4);
            let (fx, fz) = (ddx / dl, ddz / dl);
            let (rx, rz) = (-fz, fx);
            let fdir = if self.backwards { -1.0 } else { 1.0 };
            let sdir = if self.clockwise { 1.0 } else { -1.0 };
            let tx = ctx.body.x + (fx * fdir + rx * sdir) * STRAFE_OFFSET;
            let tz = ctx.body.z + (fz * fdir + rz * sdir) * STRAFE_OFFSET;
            ctx.body.nav.retarget(tx, tz, 1.0);
        }
        // The draw: start when the cooldown is spent and the target
        // seen; release at full draw.
        if let Some(t) = self.draw {
            if !seen && self.see_time < UNSEEN_FLOOR {
                self.draw = None;
            } else if seen && t + 1 >= DRAW_TICKS {
                ctx.body.pending_shot = Some((conn, FULL_POWER));
                self.attack_time = ATTACK_INTERVAL;
                self.draw = None;
            } else {
                self.draw = Some(t + 1);
            }
        } else if self.attack_time > 0 {
            self.attack_time -= 1;
            if self.attack_time == 0 && self.see_time >= UNSEEN_FLOOR {
                self.draw = Some(0);
            }
        } else if self.see_time >= UNSEEN_FLOOR {
            self.draw = Some(0);
        }
    }

    fn requires_every_tick(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------
// The shot
// ---------------------------------------------------------------------

/// The aim-and-fire step: origin at the eye minus 0.1, the one-third
/// aim height plus the distance lead, jittered and scaled.
pub(crate) fn fire_shot(
    g: &mut Game,
    x: f64,
    y: f64,
    z: f64,
    eye: f64,
    shooter: i32,
    target: (f64, f64, f64),
    power: f64,
    seed: &mut u64,
) {
    let (ax, ay, az) = (x, y + eye - SPAWN_DROP, z);
    let dist = ((target.0 - ax) * (target.0 - ax) + (target.2 - az) * (target.2 - az)).sqrt();
    let dx = target.0 - ax;
    let dy = target.1 + AIM_HEIGHT - ay + dist * AIM_LEAD;
    let dz = target.2 - az;
    let vel = shot_velocity(seed, dx, dy, dz, LAUNCH_SPEED, UNCERTAINTY);
    let base = mob_base_damage(seed, power, DIFFICULTY_ID);
    g.spawn_arrow(ax, ay, az, vel, shooter, base);
}

// ---------------------------------------------------------------------
// The kind
// ---------------------------------------------------------------------

/// The skeleton kind: parameters, the bow goal set, and the daylight
/// burn.
pub struct Skeleton;

impl Skeleton {
    pub fn new() -> Skeleton {
        Skeleton
    }
}

impl Default for Skeleton {
    fn default() -> Self {
        Self::new()
    }
}

impl MobKind for Skeleton {
    fn type_id(&self) -> i32 {
        crate::living::ENTITY_TYPE_SKELETON
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

    fn follow_range(&self) -> f64 {
        FOLLOW_RANGE
    }

    fn equipment(&self) -> Option<(u8, ItemStack)> {
        let bow = item_id("minecraft:bow").unwrap_or(1008);
        Some((EQUIP_MAIN_HAND, ItemStack::new(bow, 1)))
    }

    fn register_goals(&self, goals: &mut GoalSelector, targets: &mut GoalSelector) {
        goals.add(4, Box::new(RangedBowAttackGoal::new()));
        goals.add(
            5,
            Box::new(RandomStrollGoal::new(
                STROLL_RANGE,
                STROLL_CHANCE,
                STROLL_GIVE_UP,
                1.0,
            )),
        );
        goals.add(8, Box::new(LookAtPlayerGoal::new(LOOK_RANGE, LOOK_CHANCE)));
        goals.add(8, Box::new(RandomLookGoal::new(LOOK_CHANCE)));
        targets.add(2, Box::new(NearestPlayerTargetGoal::new(
            TARGET_SCAN_EVERY,
            UNSEEN_LIMIT,
            FOLLOW_RANGE,
            None,
        )));
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
    use crate::game::entities::PACKET_ADD_ENTITY;
    use crate::game::{Game, Inbound, Outbound};
    use crate::living::PACKET_SET_EQUIPMENT;
    use crate::projectile::ENTITY_TYPE_ARROW;
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

    fn player_dist(g: &Game) -> f64 {
        let m = &g.mobs.mobs[0];
        let p = g.players.get(&0).unwrap();
        let (dx, dz) = (m.body.x - p.x, m.body.z - p.z);
        (dx * dx + dz * dz).sqrt()
    }

    #[test]
    fn skeleton_pairs_with_a_bow() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 5.5,
            y: 100.0,
            z: 5.5,
        });
        g.spawn_mob(6.5, 100.0, 5.5, Box::new(Skeleton::new()));
        g.flush_connections();
        let frames = drain(&rx);
        let equip = frames
            .iter()
            .find(|(id, _)| *id == PACKET_SET_EQUIPMENT)
            .map(|(_, b)| b.clone())
            .expect("the equipment packet pairs");
        // id, slot byte 0 (main hand, no continuation), then the bow
        // stack: count 1, the item varint, two empty patches.
        let bow = item_id("minecraft:bow").unwrap();
        assert_eq!(equip[1], EQUIP_MAIN_HAND, "the main hand slot");
        assert_eq!(equip[2], 0x01, "stack count 1");
        let mut item = 0i32;
        let mut shift = 0;
        let mut i = 3usize;
        loop {
            let b = equip[i];
            i += 1;
            item |= ((b & 0x7f) as i32) << shift;
            shift += 7;
            if b & 0x80 == 0 {
                break;
            }
        }
        assert_eq!(item, bow, "the bow item id");
        assert_eq!(equip[i], 0x00, "no added components");
        assert_eq!(equip[i + 1], 0x00, "no removed components");
    }

    #[test]
    fn skeleton_fires_at_mid_range() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 15.5,
            y: 100.0,
            z: 5.5,
        });
        g.spawn_mob(5.5, 100.0, 5.5, Box::new(Skeleton::new()));
        let mut hit = false;
        for _ in 0..200 {
            g.tick_once_for_test();
            hit |= g.projectiles.arrows.len() > 0;
            if hit {
                break;
            }
        }
        assert!(hit, "the draw releases an arrow");
        let frames = drain(&rx);
        let arrow_add = frames.iter().any(|(id, b)| {
            if *id != PACKET_ADD_ENTITY || b.len() < 20 {
                return false;
            }
            let mut i = 0usize;
            while b[i] & 0x80 != 0 {
                i += 1;
            }
            i += 1 + 16;
            let mut ty = 0i32;
            let mut shift = 0;
            loop {
                let byte = b[i];
                i += 1;
                ty |= ((byte & 0x7f) as i32) << shift;
                shift += 7;
                if byte & 0x80 == 0 {
                    break;
                }
            }
            ty == ENTITY_TYPE_ARROW
        });
        assert!(arrow_add, "the arrow add packet carries the type");
    }

    #[test]
    fn skeleton_retreats_when_close() {
        let (mut g, _rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 8.5,
            y: 100.0,
            z: 5.5,
        });
        g.spawn_mob(5.5, 100.0, 5.5, Box::new(Skeleton::new()));
        // Let the target form and the strafe engage, then measure the
        // drift away from a 3-block player.
        for _ in 0..60 {
            g.tick_once_for_test();
        }
        let start = player_dist(&g);
        for _ in 0..60 {
            g.tick_once_for_test();
        }
        let end = player_dist(&g);
        assert!(
            end > start - 0.5,
            "the close target pushes the skeleton back: {start} -> {end}"
        );
    }

    #[test]
    fn skeleton_chases_when_far() {
        let (mut g, _rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 1.5,
            y: 100.0,
            z: 1.5,
        });
        g.spawn_mob(14.5, 100.0, 14.5, Box::new(Skeleton::new()));
        let start = player_dist(&g);
        for _ in 0..120 {
            g.tick_once_for_test();
        }
        let end = player_dist(&g);
        assert!(
            end < start - 5.0,
            "beyond the radius the skeleton closes: {start} -> {end}"
        );
    }
}
