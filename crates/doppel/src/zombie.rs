//! The zombie: an undead melee mob that targets the nearest visible
//! player, strolls and looks around while idle, and burns in daylight.

use crate::game::entities::block_solid;
use crate::game::ConnId;
use crate::game::Game;
use crate::living::{Goal, GoalCtx, GoalFlags, GoalSelector, MobKind, FIRE_IGNITE_TICKS};
use crate::spawning::{monsters_burn, sky_darken};

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
/// Player eye height.
const PLAYER_EYE: f64 = 1.62;
/// Melee reach: both half widths plus the 0.83 inflation.
const REACH: f64 = HALF_WIDTH + 0.3 + 0.83;
/// Attack cooldown, in ticks (the 20-tick interval halved).
const ATTACK_COOLDOWN: i32 = 10;
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

/// Whether the sight line between two eye points is clear.
fn visible(world: &Game, from: (f64, f64, f64), to: (f64, f64, f64)) -> bool {
    let (dx, dy, dz) = (to.0 - from.0, to.1 - from.1, to.2 - from.2);
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
    let steps = (dist * 2.0).ceil() as i32;
    for s in 1..steps {
        let t = s as f64 / steps as f64;
        let (x, y, z) = (from.0 + dx * t, from.1 + dy * t, from.2 + dz * t);
        if block_solid(world, x.floor() as i32, y.floor() as i32, z.floor() as i32) {
            return false;
        }
    }
    true
}

/// The mob eye position.
fn eye_of(body: &crate::living::MobBody) -> (f64, f64, f64) {
    (body.x, body.y + EYE, body.z)
}

/// The player eye position above the feet.
fn eye_at(pos: (f64, f64, f64)) -> (f64, f64, f64) {
    (pos.0, pos.1 + PLAYER_EYE, pos.2)
}

/// The look angles from the mob's eyes toward a point.
fn look_angles(from: (f64, f64, f64), to: (f64, f64, f64)) -> (f32, f32) {
    let (dx, dy, dz) = (to.0 - from.0, to.1 - from.1, to.2 - from.2);
    let horiz = (dx * dx + dz * dz).sqrt();
    (
        (-dx).atan2(dz).to_degrees() as f32,
        -dy.atan2(horiz).to_degrees() as f32,
    )
}

// ---------------------------------------------------------------------
// Goals
// ---------------------------------------------------------------------

/// The melee attack: chase the target, look at it, hit inside the
/// reach once the attack cooldown spends.
struct MeleeAttackGoal {
    check_in: i32,
    cooldown: i32,
    target: Option<ConnId>,
    last_path: (f64, f64),
}

impl MeleeAttackGoal {
    fn new() -> MeleeAttackGoal {
        MeleeAttackGoal {
            check_in: 0,
            cooldown: 0,
            target: None,
            last_path: (0.0, 0.0),
        }
    }
}

impl Goal for MeleeAttackGoal {
    fn flags(&self) -> GoalFlags {
        GoalFlags::MOVE.union(GoalFlags::LOOK)
    }

    fn can_use(&mut self, ctx: &mut GoalCtx) -> bool {
        // The scan gate: one evaluation per 20 game ticks.
        if self.check_in > 0 {
            self.check_in -= 1;
            return false;
        }
        self.check_in = ATTACK_COOLDOWN;
        let Some(conn) = ctx.body.target else {
            return false;
        };
        self.target = ctx.player_pos(conn).map(|_| conn);
        self.target.is_some()
    }

    fn can_continue_to_use(&mut self, ctx: &mut GoalCtx) -> bool {
        let Some(conn) = self.target else {
            return false;
        };
        if ctx.body.target != Some(conn) {
            return false;
        }
        match ctx.player_pos(conn) {
            Some((x, y, z)) => {
                let (dx, dy, dz) = (x - ctx.body.x, y - ctx.body.y, z - ctx.body.z);
                dx * dx + dy * dy + dz * dz <= FOLLOW_RANGE * FOLLOW_RANGE
            }
            None => false,
        }
    }

    fn start(&mut self, ctx: &mut GoalCtx) {
        self.cooldown = ATTACK_COOLDOWN;
        ctx.body.melee_active = true;
        if let Some((x, _, z)) = ctx.body.target.and_then(|conn| ctx.player_pos(conn)) {
            self.last_path = (x, z);
            ctx.body.nav.move_to(x, z, 1.0);
        }
    }

    fn stop(&mut self, ctx: &mut GoalCtx) {
        ctx.body.melee_active = false;
        ctx.body.look = None;
        ctx.body.nav.stop();
        self.target = None;
    }

    fn tick(&mut self, ctx: &mut GoalCtx) {
        self.cooldown -= 1;
        let Some(conn) = self.target.or(ctx.body.target) else {
            return;
        };
        let Some((px, py, pz)) = ctx.player_pos(conn) else {
            return;
        };
        // Look at the target's eyes.
        ctx.body.look = Some(look_angles(eye_of(ctx.body), eye_at((px, py, pz))));
        // Re-path when the target moved a block or on the 5% roll.
        let moved = (px - self.last_path.0) * (px - self.last_path.0)
            + (pz - self.last_path.1) * (pz - self.last_path.1);
        if moved >= 1.0 || ctx.below(20) == 0 {
            self.last_path = (px, pz);
            ctx.body.nav.retarget(px, pz, 1.0);
        }
        // The hit: cooldown spent, inside reach, sight line clear.
        if self.cooldown > 0 {
            return;
        }
        let (dy, horiz) = (
            py + PLAYER_EYE - ctx.body.y - EYE,
            ((px - ctx.body.x) * (px - ctx.body.x) + (pz - ctx.body.z) * (pz - ctx.body.z)).sqrt(),
        );
        if horiz >= REACH || dy.abs() > 2.5 {
            return;
        }
        if !visible(ctx.world, eye_of(ctx.body), eye_at((px, py, pz))) {
            return;
        }
        ctx.body.pending_hit = Some(conn);
        self.cooldown = ATTACK_COOLDOWN;
    }

    fn requires_every_tick(&self) -> bool {
        true
    }
}

/// The idle stroll: a random nearby column on a 1/60 roll.
struct RandomStrollGoal;

impl Goal for RandomStrollGoal {
    fn flags(&self) -> GoalFlags {
        GoalFlags::MOVE
    }

    fn can_use(&mut self, ctx: &mut GoalCtx) -> bool {
        if ctx.body.no_action_time >= 100 {
            return false;
        }
        if ctx.below(STROLL_CHANCE) != 0 {
            return false;
        }
        let span = (STROLL_RANGE * 2 + 1) as u64;
        let x = ctx.body.x + (ctx.below(span) as f64 - STROLL_RANGE as f64);
        let z = ctx.body.z + (ctx.below(span) as f64 - STROLL_RANGE as f64);
        ctx.body.nav.move_to(x, z, 1.0);
        true
    }

    fn can_continue_to_use(&mut self, ctx: &mut GoalCtx) -> bool {
        ctx.body.nav.in_progress() && ctx.body.nav.tick_age(STROLL_GIVE_UP)
    }
}

/// Look at the nearest player inside 8 blocks.
struct LookAtPlayerGoal {
    remaining: i32,
    duration: i32,
    conn: Option<ConnId>,
}

impl Goal for LookAtPlayerGoal {
    fn flags(&self) -> GoalFlags {
        GoalFlags::LOOK
    }

    fn can_use(&mut self, ctx: &mut GoalCtx) -> bool {
        if ctx.below(LOOK_CHANCE) != 0 {
            return false;
        }
        let Some((conn, pos)) = ctx.nearest_player(LOOK_RANGE) else {
            return false;
        };
        if !visible(ctx.world, eye_of(ctx.body), eye_at(pos)) {
            return false;
        }
        self.conn = Some(conn);
        self.duration = 20 + ctx.below(20) as i32;
        true
    }

    fn can_continue_to_use(&mut self, ctx: &mut GoalCtx) -> bool {
        self.remaining > 0
            && self.conn.is_some_and(|c| {
                ctx.player_pos(c).is_some_and(|p| {
                    let (dx, dy, dz) = (p.0 - ctx.body.x, p.1 - ctx.body.y, p.2 - ctx.body.z);
                    dx * dx + dy * dy + dz * dz < (LOOK_RANGE + 1.0) * (LOOK_RANGE + 1.0)
                })
            })
    }

    fn start(&mut self, _ctx: &mut GoalCtx) {
        self.remaining = self.duration;
    }

    fn stop(&mut self, ctx: &mut GoalCtx) {
        ctx.body.look = None;
    }

    fn tick(&mut self, ctx: &mut GoalCtx) {
        self.remaining -= 1;
        if let Some(conn) = self.conn {
            if let Some(pos) = ctx.player_pos(conn) {
                ctx.body.look = Some(look_angles(eye_of(ctx.body), eye_at(pos)));
            }
        }
    }
}

/// A random horizontal glance.
struct RandomLookGoal {
    remaining: i32,
    duration: i32,
    want: (f32, f32),
}

impl Goal for RandomLookGoal {
    fn flags(&self) -> GoalFlags {
        GoalFlags::MOVE.union(GoalFlags::LOOK)
    }

    fn can_use(&mut self, ctx: &mut GoalCtx) -> bool {
        if ctx.below(LOOK_CHANCE) != 0 {
            return false;
        }
        self.want = (ctx.below(360) as f32, 0.0);
        self.duration = 20 + ctx.below(20) as i32;
        true
    }

    fn can_continue_to_use(&mut self, _ctx: &mut GoalCtx) -> bool {
        self.remaining > 0
    }

    fn start(&mut self, _ctx: &mut GoalCtx) {
        self.remaining = self.duration;
    }

    fn stop(&mut self, ctx: &mut GoalCtx) {
        ctx.body.look = None;
    }

    fn tick(&mut self, ctx: &mut GoalCtx) {
        self.remaining -= 1;
        ctx.body.look = Some(self.want);
    }
}

/// Target the nearest player inside the follow range with a clear
/// sight line; drop it when unseen or out of range.
struct NearestPlayerTargetGoal {
    scan_in: i32,
    unseen: i32,
}

impl NearestPlayerTargetGoal {
    fn new() -> NearestPlayerTargetGoal {
        NearestPlayerTargetGoal {
            scan_in: 0,
            unseen: 0,
        }
    }
}

impl Goal for NearestPlayerTargetGoal {
    fn flags(&self) -> GoalFlags {
        GoalFlags::TARGET
    }

    fn can_use(&mut self, ctx: &mut GoalCtx) -> bool {
        if self.scan_in > 0 {
            self.scan_in -= 1;
            return false;
        }
        self.scan_in = TARGET_SCAN_EVERY;
        let Some((conn, pos)) = ctx.nearest_player(FOLLOW_RANGE) else {
            return false;
        };
        if !visible(ctx.world, eye_of(ctx.body), eye_at(pos)) {
            return false;
        }
        self.unseen = 0;
        ctx.body.target = Some(conn);
        true
    }

    fn can_continue_to_use(&mut self, ctx: &mut GoalCtx) -> bool {
        let Some(conn) = ctx.body.target else {
            return false;
        };
        let Some(pos) = ctx.player_pos(conn) else {
            return false;
        };
        let (dx, dy, dz) = (pos.0 - ctx.body.x, pos.1 - ctx.body.y, pos.2 - ctx.body.z);
        if dx * dx + dy * dy + dz * dz > FOLLOW_RANGE * FOLLOW_RANGE {
            return false;
        }
        if visible(ctx.world, eye_of(ctx.body), eye_at(pos)) {
            self.unseen = 0;
        } else {
            self.unseen += 1;
        }
        self.unseen <= UNSEEN_LIMIT
    }

    fn stop(&mut self, ctx: &mut GoalCtx) {
        ctx.body.target = None;
    }
}

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

    fn base_speed(&self) -> f64 {
        SPEED
    }

    fn attack_damage(&self) -> f32 {
        ATTACK_DAMAGE
    }

    fn follow_range(&self) -> f64 {
        FOLLOW_RANGE
    }

    fn register_goals(&self, goals: &mut GoalSelector, targets: &mut GoalSelector) {
        goals.add(3, Box::new(MeleeAttackGoal::new()));
        goals.add(7, Box::new(RandomStrollGoal));
        goals.add(
            8,
            Box::new(LookAtPlayerGoal {
                remaining: 0,
                duration: 0,
                conn: None,
            }),
        );
        goals.add(
            8,
            Box::new(RandomLookGoal {
                remaining: 0,
                duration: 0,
                want: (0.0, 0.0),
            }),
        );
        targets.add(2, Box::new(NearestPlayerTargetGoal::new()));
    }

    fn kind_tick(&mut self, ctx: &mut GoalCtx) {
        let day = ctx.world.spawning.day_time;
        // Bright locations hasten the despawn clock.
        let bright = if ctx.body.exposed {
            (15 - sky_darken(day)).max(0) as f64 / 15.0
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
    use crate::game::entities::PACKET_SET_ENTITY_DATA;
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
