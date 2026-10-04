//! Mob explosions: a bounded ray grid against block resistance, entity
//! damage and knockback by exposure and distance, air writes through
//! the block-update path, and the explode packet.

use crate::game::ConnId;
use crate::game::Game;
use crate::living::{
    encode_damage_event, encode_hurt_animation, PACKET_DAMAGE_EVENT, PACKET_HURT_ANIMATION,
};
use doppel_protocol::write_varint;

/// `explode`: registration order 36. Verified against 20+ pinned ids
/// by the same registration-chain count.
pub const PACKET_EXPLODE: i32 = 0x24;
/// The entity-attributed explosion damage type (alphabetical registry).
/// TODO wire-verify at the gate.
pub const DAMAGE_TYPE_EXPLOSION: i32 = 35;
/// The ray grid edge: 16 cells per axis, the surface casts.
const GRID: i32 = 16;
/// The ray step length, in blocks.
const STEP: f64 = 0.3;
/// Power decay per step.
const DECAY: f64 = 0.22500001;
/// The resistance charge shape: (resistance + 0.3) * 0.3.
const RESIST_ADD: f64 = 0.3;
const RESIST_SCALE: f64 = 0.3;
/// The approximated resistance of every non-air block (the flat
/// world's floor value; no resistance pin exists yet).
const RESISTANCE: f64 = 0.6;
/// The damage shape: (p*p + p)/2 * 7 * (radius * 2) + 1.
const DAMAGE_GAIN: f64 = 7.0;
/// Broadcast reach of the explode packet, in blocks.
const BROADCAST: f64 = 64.0;
/// The large-explosion particle type id (registration order).
const PARTICLE_EXPLOSION_EMITTER: i32 = 29;
/// The block-particle list entries: (particle id, scaling, speed,
/// weight).
const BLOCK_PARTICLES: [(i32, f32, f32, i32); 2] = [(69, 0.5, 1.0, 0), (72, 1.0, 1.0, 1)];
/// The explosion sound's registry id + 1 (reference-holder encoding).
const SOUND_EXPLODE: i32 = 672;
/// The ray power roll bounds.
const POWER_LO: f64 = 0.7;
const POWER_SPAN: f64 = 0.6;

/// The surface-ray directions of the grid, normalized: 16^3 - 14^3 =
/// 1352 rays.
pub fn ray_directions() -> Vec<(f64, f64, f64)> {
    let mut out = Vec::with_capacity(1352);
    for xx in 0..GRID {
        for yy in 0..GRID {
            for zz in 0..GRID {
                if xx != 0 && xx != GRID - 1 && yy != 0 && yy != GRID - 1 && zz != 0 && zz != GRID - 1
                {
                    continue;
                }
                let (mut dx, mut dy, mut dz) = (
                    xx as f64 / (GRID - 1) as f64 * 2.0 - 1.0,
                    yy as f64 / (GRID - 1) as f64 * 2.0 - 1.0,
                    zz as f64 / (GRID - 1) as f64 * 2.0 - 1.0,
                );
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                dx /= d;
                dy /= d;
                dz /= d;
                out.push((dx, dy, dz));
            }
        }
    }
    out
}

/// The explosion packet body: center, radius, destroyed count, the
/// receiving player's own knockback when hit, the particle, the sound,
/// the block-particle list, and the play-sound flag.
fn encode_explode(
    x: f64,
    y: f64,
    z: f64,
    radius: f32,
    count: usize,
    knockback: Option<(f64, f64, f64)>,
) -> Vec<u8> {
    let mut body = Vec::with_capacity(64);
    body.extend_from_slice(&x.to_be_bytes());
    body.extend_from_slice(&y.to_be_bytes());
    body.extend_from_slice(&z.to_be_bytes());
    body.extend_from_slice(&radius.to_be_bytes());
    write_varint(&mut body, count as i32);
    match knockback {
        Some((kx, ky, kz)) => {
            body.push(1);
            body.extend_from_slice(&kx.to_be_bytes());
            body.extend_from_slice(&ky.to_be_bytes());
            body.extend_from_slice(&kz.to_be_bytes());
        }
        None => body.push(0),
    }
    write_varint(&mut body, PARTICLE_EXPLOSION_EMITTER);
    write_varint(&mut body, SOUND_EXPLODE);
    write_varint(&mut body, BLOCK_PARTICLES.len() as i32);
    for (particle, scaling, speed, weight) in BLOCK_PARTICLES {
        write_varint(&mut body, particle);
        body.extend_from_slice(&scaling.to_be_bytes());
        body.extend_from_slice(&speed.to_be_bytes());
        write_varint(&mut body, weight);
    }
    body.push(1);
    body
}

/// The seen-percent exposure: a 2-per-block sample grid over the
/// entity box, counting samples whose clip to the center hits no
/// collider.
fn exposure(g: &Game, center: (f64, f64, f64), bmin: (f64, f64, f64), bmax: (f64, f64, f64)) -> f64 {
    let span = |lo: f64, hi: f64| 1.0 / ((hi - lo) * 2.0 + 1.0);
    let (xs, ys, zs) = (span(bmin.0, bmax.0), span(bmin.1, bmax.1), span(bmin.2, bmax.2));
    let x_off = (1.0 - (1.0 / xs).floor() * xs) / 2.0;
    let z_off = (1.0 - (1.0 / zs).floor() * zs) / 2.0;
    let clip_clear = |from: (f64, f64, f64)| -> bool {
        let (dx, dy, dz) = (
            center.0 - from.0,
            center.1 - from.1,
            center.2 - from.2,
        );
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        let steps = (dist * 2.0).ceil() as i32;
        for s in 1..steps {
            let t = s as f64 / steps as f64;
            let cell = (
                (from.0 + dx * t).floor() as i32,
                (from.1 + dy * t).floor() as i32,
                (from.2 + dz * t).floor() as i32,
            );
            if g.get_block(cell.0, cell.1, cell.2)
                .is_some_and(|(name, _)| name != "minecraft:air")
            {
                return false;
            }
        }
        true
    };
    let (mut hits, mut count) = (0u32, 0u32);
    let mut xx = 0.0f64;
    while xx <= 1.0 {
        let mut yy = 0.0f64;
        while yy <= 1.0 {
            let mut zz = 0.0f64;
            while zz <= 1.0 {
                let p = (
                    bmin.0 + (bmax.0 - bmin.0) * xx + x_off,
                    bmin.1 + (bmax.1 - bmin.1) * yy,
                    bmin.2 + (bmax.2 - bmin.2) * zz + z_off,
                );
                if clip_clear(p) {
                    hits += 1;
                }
                count += 1;
                zz += zs;
            }
            yy += ys;
        }
        xx += xs;
    }
    if count == 0 {
        0.0
    } else {
        hits as f64 / count as f64
    }
}

impl Game {
    /// One explosion at a position: the ray grid carves the destroyed
    /// set, entities inside the doubled radius take exposure-scaled
    /// damage and knockback, the blocks become air, and the packet
    /// reaches every player within 64 blocks.
    pub(crate) fn explode_at(&mut self, x: f64, y: f64, z: f64, radius: f64, source_id: i32) {
        // 1. The ray grid.
        let mut destroyed: std::collections::BTreeSet<(i32, i32, i32)> = Default::default();
        for (dx, dy, dz) in ray_directions() {
            let roll = (self.mobs.next() % 1000) as f64 / 1000.0;
            let mut power = radius * (POWER_LO + roll * POWER_SPAN);
            let (mut px, mut py, mut pz) = (x, y, z);
            let steps = (radius * 8.0) as i32 + 32;
            for _ in 0..steps {
                if power <= 0.0 {
                    break;
                }
                let cell = (
                    px.floor() as i32,
                    py.floor() as i32,
                    pz.floor() as i32,
                );
                let solid = self
                    .get_block(cell.0, cell.1, cell.2)
                    .is_some_and(|(name, _)| name != "minecraft:air");
                if solid {
                    power -= (RESISTANCE + RESIST_ADD) * RESIST_SCALE;
                    if power > 0.0 {
                        destroyed.insert(cell);
                    }
                }
                px += dx * STEP;
                py += dy * STEP;
                pz += dz * STEP;
                power -= DECAY;
            }
        }
        let count = destroyed.len();
        // 2. Entity damage and knockback.
        let double_radius = radius * 2.0;
        let mut struck: Vec<(ConnId, i32, f32, (f64, f64, f64))> = Vec::new();
        for (&conn, p) in self.players.iter() {
            let (dist, exposure) = {
                let (ddx, ddy, ddz) = (p.x - x, p.y + 0.9 - y, p.z - z);
                let dist = (ddx * ddx + ddy * ddy + ddz * ddz).sqrt() / double_radius;
                let ex = exposure(
                    self,
                    (x, y, z),
                    (p.x - 0.3, p.y, p.z - 0.3),
                    (p.x + 0.3, p.y + 1.8, p.z + 0.3),
                );
                (dist, ex)
            };
            if dist > 1.0 {
                continue;
            }
            let p_pow = (1.0 - dist) * exposure;
            let damage = ((p_pow * p_pow + p_pow) / 2.0 * DAMAGE_GAIN * double_radius + 1.0) as f32;
            let (ddx, ddz) = (p.x - x, p.z - z);
            let dl = (ddx * ddx + ddz * ddz).sqrt().max(1.0e-4);
            let knock = (1.0 - dist) * exposure;
            struck.push((
                conn,
                p.entity_id,
                damage,
                (ddx / dl * knock, 0.0, ddz / dl * knock),
            ));
        }
        for (conn, pid, damage, _) in &struck {
            self.send_within(
                x,
                y,
                z,
                BROADCAST,
                PACKET_HURT_ANIMATION,
                &encode_hurt_animation(*pid, 0.0),
            );
            self.send_within(
                x,
                y,
                z,
                BROADCAST,
                PACKET_DAMAGE_EVENT,
                &encode_damage_event(*pid, DAMAGE_TYPE_EXPLOSION, source_id, source_id),
            );
            *self.mobs.player_damage.entry(*conn).or_insert(0.0) += *damage;
        }
        // Mobs take the same shape (the source is excluded).
        let mob_hits: Vec<(usize, f32, (f64, f64, f64))> = self
            .mobs
            .mobs
            .iter()
            .enumerate()
            .filter(|(_, m)| m.body.id != source_id)
            .filter_map(|(i, m)| {
                let (ddx, ddy, ddz) = (m.body.x - x, m.body.y + m.body.eye - y, m.body.z - z);
                let dist = (ddx * ddx + ddy * ddy + ddz * ddz).sqrt() / double_radius;
                if dist > 1.0 {
                    return None;
                }
                let ex = exposure(
                    self,
                    (x, y, z),
                    (m.body.x - 0.3, m.body.y, m.body.z - 0.3),
                    (m.body.x + 0.3, m.body.y + 1.9, m.body.z + 0.3),
                );
                let p_pow = (1.0 - dist) * ex;
                let damage =
                    ((p_pow * p_pow + p_pow) / 2.0 * DAMAGE_GAIN * double_radius + 1.0) as f32;
                let (hx, hz) = (m.body.x - x, m.body.z - z);
                let hl = (hx * hx + hz * hz).sqrt().max(1.0e-4);
                let knock = (1.0 - dist) * ex;
                Some((i, damage, (hx / hl * knock, 0.0, hz / hl * knock)))
            })
            .collect();
        for (i, damage, knock) in mob_hits {
            let m = &mut self.mobs.mobs[i];
            let (kx, kz) = (knock.0, knock.2);
            m.hurt(damage, -kx, -kz);
            m.body.vx += kx;
            m.body.vz += kz;
        }
        // 3. The blocks become air through the update path.
        for (bx, by, bz) in &destroyed {
            self.set_block(*bx, *by, *bz, 0, true);
        }
        // 4. The packet, per player, with the struck player's own
        // knockback.
        let knocks: std::collections::BTreeMap<ConnId, (f64, f64, f64)> = struck
            .iter()
            .map(|(conn, _, _, k)| (*conn, *k))
            .collect();
        for conn in self.players.keys().copied().collect::<Vec<_>>() {
            let Some(p) = self.players.get(&conn) else {
                continue;
            };
            let near = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y) + (p.z - z) * (p.z - z)
                < BROADCAST * BROADCAST;
            if !near {
                continue;
            }
            let body = encode_explode(x, y, z, radius as f32, count, knocks.get(&conn).copied());
            self.send(conn, PACKET_EXPLODE, &body);
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::entities::PACKET_REMOVE_ENTITIES;
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
    fn ray_grid_counts_the_surface() {
        assert_eq!(ray_directions().len(), 1352);
    }

    #[test]
    fn close_blocks_die_far_blocks_survive() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 5.5,
            y: 100.0,
            z: 5.5,
        });
        g.explode_at(5.5, 100.5, 5.5, 3.0, 99);
        g.flush_connections();
        assert_eq!(
            g.block_label_for_test(5, 99, 5),
            "minecraft:air[]",
            "the floor block under the center dies"
        );
        assert!(
            g.block_label_for_test(12, 99, 5).starts_with("minecraft:grass"),
            "the floor survives beyond the radius"
        );
        let frames = drain(&rx);
        assert!(
            frames.iter().any(|(id, _)| *id == PACKET_EXPLODE),
            "the explode packet goes out"
        );
        assert!(
            frames
                .iter()
                .any(|(id, _)| *id == PACKET_DAMAGE_EVENT || *id == PACKET_REMOVE_ENTITIES),
            "the struck player sees the damage event"
        );
        assert!(
            g.mobs.player_damage.get(&0).copied().unwrap_or(0.0) > 0.0,
            "the player absorbs explosion damage"
        );
    }

    #[test]
    fn knockback_pushes_away_from_the_center() {
        let (mut g, _rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 5.5,
            y: 100.0,
            z: 5.5,
        });
        g.spawn_mob(7.5, 100.0, 5.5, Box::new(Zombie::new()));
        let before = g.mobs.mobs[0].body.x;
        g.explode_at(3.5, 100.5, 5.5, 3.0, 99);
        let mob = &g.mobs.mobs[0];
        assert!(
            mob.body.vx > 0.0,
            "the push points away from the center: {:?}",
            mob.body.vx
        );
        assert!(mob.body.health < 20.0, "the zombie takes damage");
        assert!(mob.body.x >= before, "no backward teleport");
    }
}
