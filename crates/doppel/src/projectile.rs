//! Arrows: mob-fired projectiles with per-tick gravity and drag, block
//! sticking with a bounded ground lifetime, and player hits scaled by
//! flight speed. Add, move and remove packets ride the entity encoders
//! at the arrow family's short tracking range.

use crate::game::entities::{
    block_solid, encode_add_entity, encode_move_pos, encode_position_sync, encode_remove_entities,
    encode_set_motion, PACKET_ADD_ENTITY, PACKET_ENTITY_POSITION_SYNC, PACKET_MOVE_ENTITY_POS,
    PACKET_REMOVE_ENTITIES, PACKET_SET_ENTITY_DATA, PACKET_SET_ENTITY_MOTION,
};
use crate::game::{ConnId, Game};
use crate::living::{
    encode_boolean_data, encode_damage_event, encode_hurt_animation, PACKET_DAMAGE_EVENT,
    PACKET_HURT_ANIMATION,
};

/// `minecraft:arrow` in the entity-type registry (registration order 7,
/// 0-based).
pub const ENTITY_TYPE_ARROW: i32 = 6;
/// The arrow damage type in the alphabetical damage-type registry.
/// TODO wire-verify at the gate.
pub const DAMAGE_TYPE_ARROW: i32 = 0;
/// Tracking range: clientTrackingRange 4 chunks.
const ARROW_TRACK_RANGE: f64 = 64.0;
/// Gravity per tick.
const GRAVITY: f64 = 0.05;
/// Air drag per tick.
const DRAG: f64 = 0.99;
/// Stuck lifetime before despawn, in ticks.
const LIFETIME: i32 = 1200;
/// Collision sampling density: samples per block of travel.
const SAMPLES_PER_BLOCK: f64 = 4.0;
/// The pullback off a hit face.
const PULLBACK: f64 = 0.05;
/// The in-ground metadata accessor (BOOLEAN serializer).
const DATA_IN_GROUND: u8 = 10;
/// Movement deltas are 1/4096-block shorts.
const DELTA_SCALE: f64 = 4096.0;
/// The movement sync cadence, in ticks.
const SYNC_INTERVAL: i64 = 20;
/// The squared-velocity delta that re-sends the motion vector.
const MOTION_EPS: f64 = 1.0e-6;
/// Player half width, for the hit box.
const PLAYER_HALF: f64 = 0.3;
/// Player height.
const PLAYER_HEIGHT: f64 = 1.8;
/// Arrow half width, for the hit box.
const ARROW_HALF: f64 = 0.25;

/// One flying or stuck arrow.
pub(crate) struct ArrowEntity {
    pub(crate) id: i32,
    #[allow(dead_code)]
    pub(crate) uuid: [u8; 16],
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) z: f64,
    pub(crate) vx: f64,
    pub(crate) vy: f64,
    pub(crate) vz: f64,
    /// The shooting entity's id (the add packet's data field).
    pub(crate) owner: i32,
    /// Damage scale: ceil(speed * base) on a hit.
    pub(crate) base_damage: f64,
    pub(crate) in_ground: bool,
    /// Ticks spent stuck.
    pub(crate) life: i32,
    /// Total ticks lived.
    pub(crate) age: i32,
    /// The game tick the arrow spawned in; the pass skips it.
    pub(crate) born_tick: u64,
    /// Sync state: quantized last-sent position and motion.
    sent: (i64, i64, i64),
    sent_motion: (f64, f64, f64),
    sync_phase: i64,
}

/// Projectile state the game thread owns.
#[derive(Default)]
pub(crate) struct ProjectileState {
    pub(crate) arrows: Vec<ArrowEntity>,
}

/// The yaw that faces a horizontal vector.
fn facing_yaw(dx: f64, dz: f64) -> f32 {
    (-dx).atan2(dz).to_degrees() as f32
}

/// One uniform draw in [0, 1) from a splitmix stream.
fn unit(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// A symmetric triangular draw in [-s, s).
fn triangle(seed: &mut u64, s: f64) -> f64 {
    (unit(seed) + unit(seed) - 1.0) * s
}

/// The aim-and-fire velocity: the normalized lead vector, jittered per
/// component and scaled by the launch speed.
pub(crate) fn shot_velocity(
    seed: &mut u64,
    dx: f64,
    dy: f64,
    dz: f64,
    speed: f64,
    uncertainty: f64,
) -> (f64, f64, f64) {
    let len = (dx * dx + dy * dy + dz * dz).sqrt();
    let jitter = 0.0172275 * uncertainty;
    let jx = triangle(seed, jitter);
    let jy = triangle(seed, jitter);
    let jz = triangle(seed, jitter);
    (
        (dx / len + jx) * speed,
        (dy / len + jy) * speed,
        (dz / len + jz) * speed,
    )
}

/// The mob arrow's base damage: power * 2 plus the difficulty triangle.
pub(crate) fn mob_base_damage(seed: &mut u64, power: f64, difficulty_id: f64) -> f64 {
    power * 2.0 + triangle(seed, 0.57425) + difficulty_id * 0.11
}

impl Game {
    /// Spawns an arrow in flight and pairs it with every player in
    /// range: the add packet carries the velocity and the owner id.
    pub(crate) fn spawn_arrow(
        &mut self,
        x: f64,
        y: f64,
        z: f64,
        velocity: (f64, f64, f64),
        owner: i32,
        base_damage: f64,
    ) {
        let id = self.alloc_entity_id();
        let uuid = self.next_uuid();
        let yaw = facing_yaw(velocity.0, velocity.2);
        let pitch = velocity
            .1
            .atan2((velocity.0 * velocity.0 + velocity.2 * velocity.2).sqrt())
            .to_degrees() as f32;
        let body = encode_add_entity(
            id,
            &uuid,
            ENTITY_TYPE_ARROW,
            x,
            y,
            z,
            velocity,
            yaw,
            pitch,
            yaw,
            owner,
        );
        self.send_within(x, y, z, ARROW_TRACK_RANGE, PACKET_ADD_ENTITY, &body);
        self.projectiles.arrows.push(ArrowEntity {
            id,
            uuid,
            x,
            y,
            z,
            vx: velocity.0,
            vy: velocity.1,
            vz: velocity.2,
            owner,
            base_damage,
            in_ground: false,
            life: 0,
            age: 0,
            born_tick: self.tick,
            sent: (
                (x * DELTA_SCALE).round() as i64,
                (y * DELTA_SCALE).round() as i64,
                (z * DELTA_SCALE).round() as i64,
            ),
            sent_motion: velocity,
            sync_phase: 0,
        });
    }

    /// The arrow pass: flight, hits, sticking, and the ground
    /// lifetime. Arrows spawned this tick wait for the next one.
    pub(crate) fn tick_projectiles(&mut self) {
        let tick = self.tick;
        let arrows = std::mem::take(&mut self.projectiles.arrows);
        let mut kept: Vec<ArrowEntity> = Vec::with_capacity(arrows.len());
        for mut a in arrows {
            if a.born_tick == tick {
                kept.push(a);
                continue;
            }
            let mut remove = false;
            let mut hit_player = false;
            let mut flight = (0.0f64, 0.0f32);
            let mut frames: Vec<(i32, Vec<u8>)> = Vec::new();
            a.age += 1;
            if a.age > LIFETIME + 20 {
                remove = true;
            } else if a.in_ground {
                a.life += 1;
                if a.life >= LIFETIME {
                    remove = true;
                }
            } else {
                // Fly: sample the segment; the first solid cell or
                // player box stops the arrow there.
                let len = (a.vx * a.vx + a.vy * a.vy + a.vz * a.vz).sqrt();
                let steps = (len * SAMPLES_PER_BLOCK).ceil().max(1.0) as i32;
                flight = (len, facing_yaw(a.vx, a.vz));
                let mut stop: Option<(f64, f64, f64)> = None;
                'seg: for s in 1..=steps {
                    let t = s as f64 / steps as f64;
                    let (px, py, pz) = (a.x + a.vx * t, a.y + a.vy * t, a.z + a.vz * t);
                    if block_solid(
                        self,
                        px.floor() as i32,
                        py.floor() as i32,
                        pz.floor() as i32,
                    ) {
                        stop = Some((
                            px - PULLBACK * a.vx.signum(),
                            py - PULLBACK * a.vy.signum(),
                            pz - PULLBACK * a.vz.signum(),
                        ));
                        break 'seg;
                    }
                    for p in self.players.values() {
                        if (px - p.x).abs() < PLAYER_HALF + ARROW_HALF
                            && (pz - p.z).abs() < PLAYER_HALF + ARROW_HALF
                            && py > p.y - ARROW_HALF
                            && py < p.y + PLAYER_HEIGHT + ARROW_HALF
                        {
                            hit_player = true;
                            stop = Some((px, py, pz));
                            break 'seg;
                        }
                    }
                }
                if let Some((nx, ny, nz)) = stop {
                    (a.x, a.y, a.z) = (nx, ny, nz);
                    if hit_player {
                        remove = true;
                    } else {
                        a.in_ground = true;
                        frames.push((
                            PACKET_SET_ENTITY_DATA,
                            encode_boolean_data(a.id, DATA_IN_GROUND, true),
                        ));
                    }
                    (a.vx, a.vy, a.vz) = (0.0, 0.0, 0.0);
                } else {
                    a.x += a.vx;
                    a.y += a.vy;
                    a.z += a.vz;
                    // Drag then gravity, after the move.
                    a.vx *= DRAG;
                    a.vy = a.vy * DRAG - GRAVITY;
                    a.vz *= DRAG;
                }
            }
            // The motion vector on any change.
            let dm = (a.vx - a.sent_motion.0).powi(2)
                + (a.vy - a.sent_motion.1).powi(2)
                + (a.vz - a.sent_motion.2).powi(2);
            if dm > MOTION_EPS {
                a.sent_motion = (a.vx, a.vy, a.vz);
                frames.push((
                    PACKET_SET_ENTITY_MOTION,
                    encode_set_motion(a.id, a.vx, a.vy, a.vz),
                ));
            }
            // The position: a delta packet per move, a full packet on
            // the cadence, a hit, or an out-of-range delta.
            a.sync_phase += 1;
            let qx = (a.x * DELTA_SCALE).round() as i64;
            let qy = (a.y * DELTA_SCALE).round() as i64;
            let qz = (a.z * DELTA_SCALE).round() as i64;
            let (dx, dy, dz) = (qx - a.sent.0, qy - a.sent.1, qz - a.sent.2);
            let fits = i16::MIN as i64..=i16::MAX as i64;
            let delta_fits = fits.contains(&dx) && fits.contains(&dy) && fits.contains(&dz);
            let yaw = facing_yaw(a.vx, a.vz);
            let pitch = a.vy.atan2((a.vx * a.vx + a.vz * a.vz).sqrt()).to_degrees() as f32;
            if (dx, dy, dz) == (0, 0, 0) {
                // No move this tick.
            } else if !delta_fits || a.sync_phase % SYNC_INTERVAL == 0 || a.in_ground {
                frames.push((
                    PACKET_ENTITY_POSITION_SYNC,
                    encode_position_sync(a.id, a.x, a.y, a.z, yaw, pitch, false),
                ));
                a.sent = (qx, qy, qz);
            } else {
                frames.push((
                    PACKET_MOVE_ENTITY_POS,
                    encode_move_pos(a.id, dx as i16, dy as i16, dz as i16, false),
                ));
                a.sent = (qx, qy, qz);
            }
            let (ax, ay, az) = (a.x, a.y, a.z);
            for (pid, body) in frames {
                self.send_within(ax, ay, az, ARROW_TRACK_RANGE, pid, &body);
            }
            if hit_player {
                let damage = (flight.0 * a.base_damage).ceil().max(1.0) as f32;
                self.arrow_hit_player(a.owner, damage, (ax, ay, az), flight.1);
            }
            if !remove {
                kept.push(a);
            } else {
                let body = encode_remove_entities(&[a.id]);
                self.send_within(ax, ay, az, ARROW_TRACK_RANGE, PACKET_REMOVE_ENTITIES, &body);
            }
        }
        self.projectiles.arrows = kept;
    }

    /// Damage and animation for a struck player: the damage event, the
    /// shared hurt animation along the flight direction, and the
    /// absorbed-damage counter.
    fn arrow_hit_player(&mut self, owner: i32, damage: f32, at: (f64, f64, f64), yaw: f32) {
        let struck: Vec<(ConnId, i32)> = self
            .players
            .iter()
            .filter(|(_, p)| {
                (at.0 - p.x).abs() < PLAYER_HALF + ARROW_HALF
                    && (at.2 - p.z).abs() < PLAYER_HALF + ARROW_HALF
                    && at.1 > p.y - ARROW_HALF
                    && at.1 < p.y + PLAYER_HEIGHT + ARROW_HALF
            })
            .map(|(&c, p)| (c, p.entity_id))
            .collect();
        for (conn, pid) in struck {
            self.send_within(
                at.0,
                at.1,
                at.2,
                ARROW_TRACK_RANGE,
                PACKET_HURT_ANIMATION,
                &encode_hurt_animation(pid, yaw),
            );
            self.send_within(
                at.0,
                at.1,
                at.2,
                ARROW_TRACK_RANGE,
                PACKET_DAMAGE_EVENT,
                &encode_damage_event(pid, DAMAGE_TYPE_ARROW, owner, owner),
            );
            *self.mobs.player_damage.entry(conn).or_insert(0.0) += damage;
        }
    }
}

/// The in-ground metadata entry for tests.
#[cfg(test)]
pub(crate) fn in_ground_data(entity_id: i32) -> Vec<u8> {
    crate::living::encode_boolean_data(entity_id, DATA_IN_GROUND, true)
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Game, Inbound, Outbound};
    use crate::zombie::Zombie;
    use doppel_world::WireChunk;
    use std::sync::mpsc;

    /// A grass-floored chunk (surface y=99) with one player.
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
    fn arrow_adds_with_owner_and_velocity() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 5.5,
            y: 100.0,
            z: 5.5,
        });
        g.spawn_mob(6.5, 100.0, 5.5, Box::new(Zombie::new()));
        let owner = g.mobs.mobs[0].body.id;
        g.spawn_arrow(6.5, 101.6, 5.5, (0.0, 0.0, 1.6), owner, 2.11);
        g.flush_connections();
        let frames = drain(&rx);
        // The zombie paired first; the arrow's add is the one whose
        // bytes match the encoder with the owner in the data field.
        let (aid, auuid) = {
            let a = &g.projectiles.arrows[0];
            (a.id, a.uuid)
        };
        let expect = encode_add_entity(
            aid,
            &auuid,
            ENTITY_TYPE_ARROW,
            6.5,
            101.6,
            5.5,
            (0.0, 0.0, 1.6),
            0.0,
            0.0,
            0.0,
            owner,
        );
        assert!(
            frames
                .iter()
                .any(|(id, b)| *id == PACKET_ADD_ENTITY && b == &expect),
            "the arrow add carries the type, velocity, and owner"
        );
    }

    #[test]
    fn arrow_sticks_and_despawns() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 8.5,
            y: 100.0,
            z: 5.5,
        });
        g.spawn_arrow(5.5, 103.0, 5.5, (0.0, -0.5, 0.0), 1, 2.11);
        // The flat-shot arrow drops into the floor and sticks.
        let mut stuck = false;
        for _ in 0..40 {
            g.tick_once_for_test();
            if g.projectiles.arrows[0].in_ground {
                stuck = true;
                break;
            }
        }
        assert!(stuck, "the arrow sticks in the ground");
        g.flush_connections();
        let frames = drain(&rx);
        assert!(
            frames.iter().any(|(id, b)| *id == PACKET_SET_ENTITY_DATA
                && b == &in_ground_data(g.projectiles.arrows[0].id)),
            "the in-ground datum goes out"
        );
        // The stuck arrow despawns at the lifetime.
        for _ in 0..LIFETIME {
            g.tick_once_for_test();
        }
        assert!(g.projectiles.arrows.is_empty(), "the stuck arrow ages out");
        assert!(
            drain(&rx)
                .iter()
                .any(|(id, _)| *id == PACKET_REMOVE_ENTITIES),
            "the removal lands on the wire"
        );
    }

    #[test]
    fn arrow_hits_the_player() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 8.5,
            y: 100.0,
            z: 5.5,
        });
        g.spawn_arrow(5.5, 101.6, 5.5, (1.6, 0.0, 0.0), 7, 2.11);
        for _ in 0..10 {
            g.tick_once_for_test();
        }
        assert!(g.projectiles.arrows.is_empty(), "the arrow is spent");
        let frames = drain(&rx);
        assert!(
            frames.iter().any(|(id, _)| *id == PACKET_DAMAGE_EVENT),
            "the hit sends the damage event"
        );
        assert!(
            frames.iter().any(|(id, _)| *id == PACKET_HURT_ANIMATION),
            "the hit sends the hurt animation"
        );
        assert!(
            (g.mobs.player_damage[&0] - 4.0).abs() < 0.5,
            "speed-scaled damage"
        );
    }

    #[test]
    fn shot_velocity_leads_and_jitters() {
        let mut seed = 42u64;
        // A dead-level shot keeps its direction.
        let (x, y, z) = shot_velocity(&mut seed, 10.0, 0.0, 0.0, 1.6, 10.0);
        assert!(x > 1.4 && y.abs() < 0.3 && z.abs() < 0.3, "({x},{y},{z})");
        // The magnitude stays near the launch speed.
        let m = (x * x + y * y + z * z).sqrt();
        assert!((m - 1.6).abs() < 0.3, "speed {m}");
    }
}
