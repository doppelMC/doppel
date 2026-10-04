//! The creeper: a hostile approach mob whose swell goal stops the walk,
//! holds the fuse while the target stays close and visible, and
//! detonates at the fuse end.

use crate::game::ConnId;
use crate::living::{
    eye_at, eye_of, visible, ChaseHitGoal, GlanceGoal, Goal, GoalCtx, GoalFlags, GoalSelector,
    IdleStrollGoal, MobKind, NearestPlayerTargetGoal, WatchPlayerGoal, PLAYER_EYE,
};

/// Follow range (the attribute default).
const FOLLOW_RANGE: f64 = 32.0;
/// Movement speed (the attribute).
const SPEED: f64 = 0.25;
/// Attack damage (the attribute; the melee approach never lands it).
const ATTACK_DAMAGE: f32 = 2.0;
/// Hitbox half width.
const HALF_WIDTH: f64 = 0.3;
/// Hitbox height.
const HEIGHT: f64 = 1.7;
/// Eye height (the 0.85 default of the height).
const EYE: f64 = 1.445;
/// Melee reach: both half widths plus the 0.83 inflation.
const REACH: f64 = HALF_WIDTH + 0.3 + 0.83;
/// The swell start: squared 3-block distance.
const SWELL_SQ: f64 = 9.0;
/// The swell cancel: squared 7-block distance.
const CANCEL_SQ: f64 = 49.0;
/// The fuse length, in ticks.
const FUSE: i32 = 30;
/// The explosion radius.
const RADIUS: f64 = 3.0;
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

/// The swell: stop the navigation and drive the fuse direction while
/// the target stays close, visible, and inside the cancel range.
struct FuseGoal {
    target: Option<ConnId>,
}

impl Goal for FuseGoal {
    fn flags(&self) -> GoalFlags {
        GoalFlags::MOVE
    }

    fn can_use(&mut self, ctx: &mut GoalCtx) -> bool {
        if ctx.body.swell_dir > 0 {
            return true;
        }
        let Some(conn) = ctx.body.target else {
            return false;
        };
        match ctx.player_pos(conn) {
            Some((x, _, z)) => {
                let d2 = (x - ctx.body.x) * (x - ctx.body.x) + (z - ctx.body.z) * (z - ctx.body.z);
                d2 < SWELL_SQ
            }
            None => false,
        }
    }

    fn can_continue_to_use(&mut self, ctx: &mut GoalCtx) -> bool {
        self.can_use(ctx)
    }

    fn start(&mut self, ctx: &mut GoalCtx) {
        ctx.body.nav.stop();
        self.target = ctx.body.target;
    }

    fn stop(&mut self, _ctx: &mut GoalCtx) {
        self.target = None;
    }

    fn tick(&mut self, ctx: &mut GoalCtx) {
        let Some(conn) = self.target else {
            ctx.body.swell_dir = -1;
            return;
        };
        let Some((x, y, z)) = ctx.player_pos(conn) else {
            ctx.body.swell_dir = -1;
            return;
        };
        let d2 = (x - ctx.body.x) * (x - ctx.body.x) + (z - ctx.body.z) * (z - ctx.body.z);
        if d2 > CANCEL_SQ {
            ctx.body.swell_dir = -1;
            return;
        }
        if !visible(ctx.world, eye_of(ctx.body), eye_at(PLAYER_EYE, (x, y, z))) {
            ctx.body.swell_dir = -1;
            return;
        }
        ctx.body.swell_dir = 1;
    }

    fn requires_every_tick(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------
// The kind
// ---------------------------------------------------------------------

/// The creeper kind: parameters, the swell-and-approach goal set, and
/// the fuse detonation.
pub struct Creeper;

impl Creeper {
    pub fn new() -> Creeper {
        Creeper
    }
}

impl Default for Creeper {
    fn default() -> Self {
        Self::new()
    }
}

impl MobKind for Creeper {
    fn type_id(&self) -> i32 {
        crate::living::ENTITY_TYPE_CREEPER
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

    fn register_goals(&self, goals: &mut GoalSelector, targets: &mut GoalSelector) {
        goals.add(2, Box::new(FuseGoal { target: None }));
        goals.add(4, Box::new(ChaseHitGoal::new(REACH, FOLLOW_RANGE, false)));
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
                None,
            )),
        );
    }

    fn kind_tick(&mut self, ctx: &mut GoalCtx) {
        ctx.body.fuse += ctx.body.swell_dir;
        if ctx.body.fuse < 0 {
            ctx.body.fuse = 0;
        }
        if ctx.body.fuse >= FUSE {
            ctx.body.fuse = FUSE;
            ctx.body.pending_blast = Some(RADIUS);
        }
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::explosion::PACKET_EXPLODE;
    use crate::game::entities::PACKET_SET_ENTITY_DATA;
    use crate::game::{Game, Inbound, Outbound};
    use crate::living::DATA_SWELL_DIR;
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
    fn swell_toggles_at_the_thresholds() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 7.5,
            y: 100.0,
            z: 5.5,
        });
        g.spawn_mob(5.5, 100.0, 5.5, Box::new(Creeper::new()));
        // Two blocks out: the swell engages.
        let mut swelled = false;
        for _ in 0..60 {
            g.tick_once_for_test();
            if g.mobs.mobs[0].body.swell_dir == 1 {
                swelled = true;
                break;
            }
        }
        assert!(swelled, "the close target swells the creeper");
        g.flush_connections();
        let frames = drain(&rx);
        assert!(
            frames.iter().any(|(id, b)| {
                *id == PACKET_SET_ENTITY_DATA && b.len() >= 4 && b[1] == DATA_SWELL_DIR
            }),
            "the swell datum goes out"
        );
        // Ten blocks out: past the cancel range, the swell deflates.
        g.handle(Inbound::Tp {
            conn: 0,
            x: 15.5,
            y: 100.0,
            z: 5.5,
        });
        let mut deflated = false;
        for _ in 0..60 {
            g.tick_once_for_test();
            if g.mobs.mobs[0].body.swell_dir == -1 {
                deflated = true;
                break;
            }
        }
        assert!(deflated, "the far target deflates the swell");
    }

    #[test]
    fn the_fuse_ends_in_an_explosion() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Tp {
            conn: 0,
            x: 6.5,
            y: 100.0,
            z: 5.5,
        });
        g.spawn_mob(5.5, 100.0, 5.5, Box::new(Creeper::new()));
        let mut exploded = false;
        for _ in 0..140 {
            g.tick_once_for_test();
            if g.mobs.mobs.is_empty() {
                exploded = true;
                break;
            }
        }
        assert!(exploded, "the creeper detonates within the fuse window");
        g.flush_connections();
        let frames = drain(&rx);
        assert!(
            frames.iter().any(|(id, _)| *id == PACKET_EXPLODE),
            "the explosion packet goes out"
        );
        assert!(
            frames
                .iter()
                .any(|(id, _)| *id == crate::game::entities::PACKET_REMOVE_ENTITIES),
            "the creeper leaves without a corpse"
        );
        assert!(
            g.mobs.player_damage.get(&0).copied().unwrap_or(0.0) > 0.0,
            "the player absorbs the blast"
        );
        assert_eq!(
            g.block_label_for_test(5, 99, 5),
            "minecraft:air[]",
            "the floor under the blast is gone"
        );
    }
}
