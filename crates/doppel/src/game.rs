//! The game thread: single owner of world state, players, and streaming
//! decisions. Connection actors. Connection threads are IO actors that
//! forward inbound events and drain outbound frames; nothing here touches
//! a socket. This is the skeleton the block-modification tick loop hangs
//! from.

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use crate::blobs::Blobs;
use crate::wire;
use crate::WireChunk;

pub type ConnId = u64;

/// Connection ids are self-assigned by the IO actors; the game thread only
/// learns of them via `Inbound::Joined` (no shared lock anywhere).
pub static CONN_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Allocates a fresh connection id.
pub fn next_conn_id() -> ConnId {
    CONN_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Events from connection actors to the game thread.
pub enum Inbound {
    /// A player finished the join burst; `sent` lists chunks already
    /// delivered by the replay.
    Joined {
        conn: ConnId,
        name: String,
        x: f64,
        y: f64,
        z: f64,
        sent: Vec<(i32, i32)>,
        /// The connection actor's outbound channel; the game thread is the
        /// sole writer once registration completes.
        tx: Sender<Outbound>,
    },
    Moved {
        conn: ConnId,
        x: f64,
        y: f64,
        z: f64,
        /// View rotation when the move packet carries it (the pos+rot
        /// form); None for the position-only form.
        yaw: Option<f32>,
        pitch: Option<f32>,
    },
    /// `tp @s x y z` (the walk-parity bot's vehicle).
    Tp {
        conn: ConnId,
        x: f64,
        y: f64,
        z: f64,
    },
    /// `tp <name> x y z`: the issuer teleports the named player (the
    /// survival gate stages its commandless witness this way).
    TpNamed {
        conn: ConnId,
        name: String,
        x: f64,
        y: f64,
        z: f64,
    },
    KeepAliveAnswer {
        conn: ConnId,
        id: i64,
    },
    /// `setblock x y z <name>` (state id resolved from the learned table).
    Setblock {
        conn: ConnId,
        x: i32,
        y: i32,
        z: i32,
        name: String,
    },
    /// `tick step N`: run N game ticks now, before any later-queued
    /// commands — vanilla's stepped ticks are the sequencing barrier that
    /// separates a circuit's setup from its power flips.
    TickStep {
        conn: ConnId,
        steps: u32,
    },
    /// `tick freeze` / `tick unfreeze`: while frozen the wall-clock tick
    /// loop suspends; `tick step N` still runs its ticks, exactly the
    /// reference's frozen stepping semantics.
    TickFreeze {
        conn: ConnId,
        frozen: bool,
    },
    // --- inventory (implementation in inventory.rs) ---
    /// `set_carried_item`: the client's selected hotbar slot.
    SetCarriedItem {
        conn: ConnId,
        slot: i16,
    },
    /// `container_click` against an open menu (only the player's own
    /// inventory menu exists so far).
    ContainerClick {
        conn: ConnId,
        click: crate::inventory::ContainerClick,
    },
    /// `gamemode <mode>`: flips the commanding player's creative flag.
    GameMode {
        conn: ConnId,
        creative: bool,
    },
    /// set_creative_mode_slot: a creative client pushing its picked stack.
    CreativeSlot {
        conn: ConnId,
        set: crate::inventory::CreativeSlotSet,
    },
    /// `give @s <item> [count]` — the harness driver for inventory tests.
    Give {
        conn: ConnId,
        item: String,
        count: i32,
    },
    // --- survival hooks (entities.rs) ---
    /// `gamerule random_tick_speed N`: the random tick rate.
    GameRule {
        conn: ConnId,
        tick_speed: usize,
    },
    /// A spawner gamerule other than `spawn_mobs` (nothing to do yet;
    /// the other categories have no entities).
    /// `time set <ticks>`: stores the day time and pushes set_time.
    TimeSet {
        conn: ConnId,
        value: i64,
    },
    /// A spawner gamerule (`spawn_mobs` and kin): a no-op here, this
    /// build has no mob spawning.
    GameRuleNoop {
        conn: ConnId,
    },
    // --- mob hooks (living.rs / spawning.rs) ---
    /// `gamerule spawn_mobs <bool>`: the natural-spawn gate.
    SpawnMobs {
        conn: ConnId,
        enabled: bool,
    },
    /// `difficulty <word>`: peaceful removes monsters.
    SetDifficulty {
        conn: ConnId,
        peaceful: bool,
    },
    // --- placement hooks (placement.rs) ---
    /// `move_player_rot`: view rotation without movement.
    Rotated {
        conn: ConnId,
        yaw: f32,
        pitch: f32,
    },
    /// `use_item_on`: right-click a block face with the held item.
    UseItemOn {
        conn: ConnId,
        x: i32,
        y: i32,
        z: i32,
        /// Clicked face, Direction 3D id (0=down .. 5=east).
        face: u8,
        cursor_x: f32,
        cursor_y: f32,
        cursor_z: f32,
        /// 0 = main hand, 1 = offhand.
        hand: u8,
        sequence: i32,
    },
    // --- breaking hooks (dig.rs) ---
    /// `player_action`: dig lifecycle, drops, and the offhand swap.
    PlayerAction {
        conn: ConnId,
        act: crate::dig::PlayerAction,
    },
    /// `punch`: the arm swing. The reference resets its last-action clock
    /// and swings; the broadcast back is client-cosmetic animation.
    Punch {
        conn: ConnId,
    },
    // --- containers hooks (containers.rs) ---
    /// `opencontainer x y z` (harness driver): open the container menu
    /// at a block position.
    OpenContainer {
        conn: ConnId,
        x: i32,
        y: i32,
        z: i32,
    },
    /// `container_close`: the client closed one of its menus.
    ContainerClose {
        conn: ConnId,
        container_id: i32,
    },
    Left {
        conn: ConnId,
    },
}

/// Frames the connection actor's writer thread puts on the wire.
pub enum Outbound {
    Frame { id: i32, body: Vec<u8> },
    Disconnect,
}

/// Per-connection player state. `pub(crate)` + the `inv` field exist for
/// the inventory module's `impl Game` hooks (inventory.rs).
pub(crate) struct Player {
    pub(crate) name: String,
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) z: f64,
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
    center: Option<(i32, i32)>,
    sent: std::collections::HashSet<(i32, i32)>,
    teleport_id: i32,
    pending_keep_alive: Option<(i64, Instant)>,
    /// When the last challenge was answered; spaces the next one an
    /// interval out instead of firing every tick.
    keep_alive_idle: Instant,
    // --- inventory hook (inventory.rs) ---
    pub(crate) inv: crate::inventory::PlayerInvState,
    // --- containers hooks (containers.rs) ---
    pub(crate) menu: containers::PlayerMenuState,
    // --- breaking hooks (dig.rs) ---
    /// The wire identity destruction overlays key on. Vanilla assigns
    /// incremental entity ids; this build has no entity model, so a
    /// server-side counter fills the slot.
    pub(crate) entity_id: i32,
    /// The dig state machine: the active dig, the delayed destroy, the
    /// last destruction stage sent (dedup guard), and the ticks-since-
    /// join counter dig progress runs on (vanilla's per-player
    /// game-mode ticks, not world time).
    pub(crate) dig: crate::dig::PlayerDigState,
}

/// One cached, versioned chunk. `wire` is the sendable form; block
/// modification will mutate sections and bump `version` (invalidating the
/// encoded-frame cache).
pub struct CachedChunk {
    pub wire: WireChunk,
    pub version: u64,
}

pub struct Game {
    pub(crate) chunks: std::collections::BTreeMap<(i32, i32), CachedChunk>,
    // pub(crate) for the inventory module's Game hooks (inventory.rs).
    pub(crate) players: std::collections::BTreeMap<ConnId, Player>,
    /// Inverse index: chunk column -> connections tracking it.
    viewers: std::collections::BTreeMap<(i32, i32), Vec<ConnId>>,
    inbound: Receiver<Inbound>,
    outbounds: HashMap<ConnId, Sender<Outbound>>,
    /// Frames buffered since the last connection flush (once per tick).
    pending_frames: Vec<(ConnId, i32, Vec<u8>)>,
    /// True inside a stepped-tick batch: per-tick flushes defer to one
    /// flush after the batch (the reference flushes once per server
    /// cycle, and a step batch is one cycle).
    flush_suspended: bool,
    pub(crate) world: Option<std::sync::Arc<std::sync::Mutex<crate::WorldState>>>,
    blobs: Option<std::sync::Arc<Blobs>>,
    /// Per-tick dirty sections: (chunkX, chunkZ, sectionY) -> ordered
    /// (localPos, state) changes awaiting the tick-end broadcast.
    dirty: std::collections::BTreeMap<(i32, i32, i32), Vec<(u64, u32)>>,
    /// Block-state registry (name+props -> id), loaded from pins/blocks.json.
    pub(crate) registry: Option<doppel_world::registry::BlockRegistry>,
    /// Dedup guard for dropped-world-write warnings: (cx, cz, reason).
    warned_writes: std::collections::BTreeSet<(i32, i32, String)>,
    /// `/tick freeze`: wall-clock ticks suspend; steps still run.
    frozen: bool,
    /// Flat-world generator: the fallback for chunks the Anvil store does
    /// not have, built from the same registry pins.
    flat: Option<doppel_world::worldgen::FlatGenerator>,
    /// Monotonic game tick.
    tick: u64,
    /// Day time in ticks, set by `time set` and carried by set_time.
    pub(crate) day_time: i64,
    /// Scheduled actions: fire at tick T with a behavior tag.
    scheduled: Vec<(u64, (i32, i32, i32), TickAction)>,
    /// Unified pending-transition guard: (pos, kind). Replaces the
    /// per-family sets and the single torch slot (review finding #3):
    /// stale entries are skipped at fire time when the block no longer
    /// matches — never eagerly purged.
    pending: std::collections::BTreeSet<((i32, i32, i32), PendingKind)>,
    /// Torch transitions: (pos, target_state, due_tick).
    torch_queue: Vec<((i32, i32, i32), u32, u64)>,
    /// Piston block events (vanilla `ServerLevel.blockEvents`): fired in
    /// the block-event phase of the tick they were queued in. Deduplicated
    /// on (pos, event, dir) like vanilla's ordered set.
    block_events: Vec<BlockEvent>,
    /// Live `moving_piston` carriers:
    /// progress 0 -> 2 in halves (0.0, 0.5, 1.0), landing on the tick the
    /// entry sees progress >= 1.0.
    moving: Vec<MovingPiston>,
    /// Comparator stored output (vanilla ComparatorBlockEntity OutputSignal).
    comparator_outputs: std::collections::HashMap<(i32, i32, i32), i32>,
    // --- containers hooks (containers.rs) ---
    containers: containers::ContainersState,
    /// Per-join entity ids for destruction overlays (see Player).
    next_entity_id: i32,
    // --- survival hooks (entities.rs) ---
    /// Live item entities and the random-tick bookkeeping.
    pub(crate) survival: entities::SurvivalState,
    // --- tracker hooks (tracker.rs) ---
    /// Per-entity sync state and per-player pairing.
    tracking: tracker::EntityTrackers, // --- mob hooks (living.rs / spawning.rs) ---
    /// Live mobs and their per-tick sync state.
    pub(crate) mobs: crate::living::MobState,
    /// World time and the natural-spawn state.
    pub(crate) spawning: crate::spawning::SpawnState,
    // --- persistence hooks (persistence.rs) ---
    /// Dirty-chunk tracking and the autosave cadence.
    pub(crate) persistence: crate::persistence::Persistence,
}

/// One queued piston block event. `event` is vanilla's TRIGGER_* id:
/// 0 extend, 1 contract, 2 drop (retract without pulling).
#[derive(Clone, Copy, PartialEq)]
struct BlockEvent {
    pos: (i32, i32, i32),
    /// Whether the queueing block was sticky (staleness re-check).
    sticky: bool,
    event: u8,
    /// Direction 3D id (0..5), vanilla's `b1` payload.
    dir: u8,
}

/// A `moving_piston` carrier: the block entity payload that remembers
/// which state lands where when the 3-tick animation completes.
#[derive(Clone, Copy)]
struct MovingPiston {
    pos: (i32, i32, i32),
    /// The state that lands at `pos` on completion.
    moved_state: u32,
    /// Piston facing (NOT movement direction).
    dir: u8,
    extending: bool,
    /// True for the piston's own pieces (head/arm, retracting base);
    /// false for carried ordinary blocks.
    is_source: bool,
    /// 0 = 0.0, 1 = 0.5, 2 = 1.0 — advances one half per BE tick.
    progress: u8,
}

/// Which family a pending guard belongs to.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PendingKind {
    Torch,
    Observer,
    Repeater,
    Comparator,
}

/// What a scheduled entry does when its tick arrives.
#[derive(Clone, Copy)]
pub enum TickAction {
    /// Recompute redstone behavior at the position (neighbor notified).
    NeighborUpdate,
    /// A block changed at pos + dir: wires recompute that one side (the
    /// reference's directional updateShape); others recompute behavior.
    ShapeUpdate {
        dx: i32,
        dy: i32,
        dz: i32,
    },
    /// Observer pulse edge: toggle powered, maybe chain the falling edge.
    ObserverToggle,
    /// Repeater output edge: apply the scheduled input change.
    RepeaterToggle,
    ComparatorToggle,
}

/// Face offsets for the six neighbors.
const NEIGHBORS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

#[path = "comparator.rs"]
mod comparator;

// --- containers hooks (containers.rs) ---
#[path = "containers.rs"]
pub(crate) mod containers;

// --- survival hooks (entities.rs) ---
#[path = "entities.rs"]
pub(crate) mod entities;

// --- tracker hooks (tracker.rs) ---
#[path = "tracker.rs"]
pub(crate) mod tracker;

pub(crate) const VIEW_RADIUS: i32 = 4;
const KEEP_ALIVE_INTERVAL: Duration = Duration::from_secs(15);
/// player_position (0x49): the absolute position sync.
pub(crate) const PACKET_PLAYER_POSITION: i32 = 0x49;

/// The player_position (0x49) body: teleport id, position, zero deltas,
/// rotation, absolute flags.
fn position_sync_body(teleport_id: i32, x: f64, y: f64, z: f64, yaw: f32, pitch: f32) -> Vec<u8> {
    let mut body = Vec::with_capacity(42);
    doppel_protocol::write_varint(&mut body, teleport_id);
    body.extend_from_slice(&x.to_be_bytes());
    body.extend_from_slice(&y.to_be_bytes());
    body.extend_from_slice(&z.to_be_bytes());
    body.extend_from_slice(&0.0f64.to_be_bytes());
    body.extend_from_slice(&0.0f64.to_be_bytes());
    body.extend_from_slice(&0.0f64.to_be_bytes());
    body.extend_from_slice(&yaw.to_be_bytes());
    body.extend_from_slice(&pitch.to_be_bytes());
    body.extend_from_slice(&0i32.to_be_bytes());
    body
}

impl Game {
    pub fn new(
        inbound: Receiver<Inbound>,
        world: Option<std::sync::Arc<std::sync::Mutex<crate::WorldState>>>,
        blobs: Option<std::sync::Arc<Blobs>>,
    ) -> Game {
        let mut game = Game {
            chunks: std::collections::BTreeMap::new(),
            players: std::collections::BTreeMap::new(),
            viewers: std::collections::BTreeMap::new(),
            inbound,
            outbounds: HashMap::new(),
            pending_frames: Vec::new(),
            flush_suspended: false,
            world,
            blobs,
            dirty: std::collections::BTreeMap::new(),
            warned_writes: std::collections::BTreeSet::new(),
            frozen: false,
            registry: doppel_protocol::find_repo_root().ok().and_then(|r| {
                let p = r.join("pins").join("blocks.json");
                p.exists()
                    .then(|| {
                        doppel_world::registry::BlockRegistry::load(&p)
                            .inspect_err(|e| eprintln!("[game] registry: {e:#}"))
                            .ok()
                    })
                    .flatten()
            }),
            tick: 0,
            day_time: 0,
            scheduled: Vec::new(),
            pending: std::collections::BTreeSet::new(),
            torch_queue: Vec::new(),
            flat: None,
            block_events: Vec::new(),
            moving: Vec::new(),
            comparator_outputs: std::collections::HashMap::new(),
            containers: Default::default(),
            next_entity_id: 1,
            survival: Default::default(),
            tracking: Default::default(),
            mobs: Default::default(),
            spawning: Default::default(),
            persistence: Default::default(),
        };
        // The flat fallback needs the registry pins; build it once here.
        game.flat = game.registry.as_ref().and_then(|r| {
            doppel_world::worldgen::FlatGenerator::classic(r)
                .inspect_err(|e| eprintln!("[game] flat generator: {e:#}"))
                .ok()
        });
        game.boot_from_level();
        // --- persistence hooks (persistence.rs) ---
        // Default-state names resolve without a capture; learned entries
        // keep precedence over the seeds.
        if let (Some(world), Some(registry)) = (&game.world, &game.registry) {
            let mut w = world.lock().unwrap_or_else(|e| e.into_inner());
            w.boot.seed_defaults(registry);
        }
        game
    }

    /// The event loop. The 1s recv timeout doubles as the coarse keep-alive
    /// tick; the real 20 TPS loop replaces this when simulation arrives.
    pub fn run(&mut self) {
        const TICK: Duration = Duration::from_millis(50);
        // Ticks fire on wall-clock deadlines, not per event batch: a tick
        // per command batch would flush each edit separately instead of
        // batching everything that landed inside one 50ms window.
        let mut next_tick = std::time::Instant::now() + TICK;
        loop {
            let wait = next_tick.saturating_duration_since(std::time::Instant::now());
            match self.inbound.recv_timeout(wait) {
                Ok(event) => {
                    self.handle(event);
                    // Drain everything else already queued: events arriving
                    // together group into one tick, matching vanilla's
                    // tick-end broadcast batching.
                    while let Ok(event) = self.inbound.try_recv() {
                        self.handle(event);
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    // --- persistence hooks (persistence.rs) ---
                    self.flush_all();
                    return;
                }
            }
            if std::time::Instant::now() >= next_tick {
                // The flush happens inside game_tick at the broadcast
                // point; edits from the phases below it deliberately stay
                // pending until the next tick's flush. A frozen clock only
                // advances through explicit `tick step`s.
                if !self.frozen {
                    self.game_tick();
                }
                // One wire flush per cycle, always: the reference
                // suspends connection flushing for the whole server tick
                // (world work, stepped batches, and command handling all
                // included) and flushes once after it. Nothing flushes
                // off-cycle.
                self.flush_connections();
                self.tick_keep_alives();
                next_tick += TICK;
            }
        }
    }

    /// One game tick (50ms), phases in the reference loop's order. The
    /// order is load-bearing: a write lands in this tick's broadcast
    /// point (flush_dirty) or defers to the next depending on which
    /// phase makes it.
    ///
    /// 1. tick counter advance
    /// 2. broadcast_time: set_time (0x73) every 20 ticks; a server
    ///    level sync that precedes the world tick
    /// 3. fire_torches: queued torch transitions (1gt input delay);
    ///    with 4 this is the scheduled-tick phase
    /// 4. run_scheduled_ticks: due redstone actions (neighbor and
    ///    shape updates, observer/repeater/comparator toggles)
    /// 5. random_ticks (entities.rs): the chunk tick's grass pass,
    ///    inside the chunk phase, before the broadcast point
    /// 6. flush_dirty: the broadcast point (0x08/0x56); edits after
    ///    this defer to the next tick
    /// 7. track_entities (tracker.rs): the entity sync flush (short
    ///    deltas, motion, full packets on cadence); rides with the
    ///    chunk-phase tracking sync, its motion packets ahead of the
    ///    move packets of the entity phase below
    /// 8. run_block_events: piston world mutations, in the same tick
    ///    they were queued
    /// 9. tick_entities (entities.rs): the pickup vacuum, then item
    ///    physics, merges, pickups, despawns
    /// 10. advance_digs (dig.rs): the per-player dig clock, delayed
    ///     destroys, stage overlays; rides the player entity tick, so
    ///     its breaks land in the NEXT tick's flush
    /// 11. broadcast_pending_inventory (inventory.rs): non-click slot
    ///     syncs, masked by an open container menu; rides the player
    ///     entity tick after the dig clock
    /// 12. tick_containers (containers.rs): chest lids, menu range
    ///     checks, hoppers; the block-entity phase
    /// 13. tick_moving_pistons: moving pistons advance +0.5 and land;
    ///     same block-entity phase (block entities interleave by
    ///     insertion order there, not by kind)
    /// 14. flush_block_entities (containers.rs): block entity data
    ///     packets for changed block entities.
    /// 15. flush_connections: the once-per-tick wire drain (the
    ///     reference flushes every connection after the world tick).
    ///
    /// One game tick, phases in order. The order is load-bearing: a
    /// write lands in this tick's broadcast point (flush_dirty) or
    /// defers to the next depending on which phase makes it.
    ///
    /// 1. tick counter advance (plus the day clock when it runs)
    /// 2. tick_entities (entities.rs): item physics, merges, pickups,
    ///    then mobs (living.rs)
    /// 3. advance_digs (dig.rs): the per-player dig clock, delayed
    ///    destroys, stage overlays
    /// 4. broadcast_pending_inventory (inventory.rs): non-click slot
    ///    syncs, masked by an open container menu
    /// 5. broadcast_time: set_time (0x73) every 20 ticks
    /// 6. fire_torches: queued torch transitions (1gt input delay)
    /// 7. run_scheduled_ticks: due redstone actions (neighbor and
    ///    shape updates, observer/repeater/comparator toggles)
    /// 8. natural_spawns (spawning.rs): the monster spawn pass
    /// 9. random_ticks (entities.rs): the chunk tick's grass pass
    /// 10. flush_dirty: the broadcast point (0x08/0x56); edits after
    ///     this defer to the next tick
    /// 11. run_block_events: piston world mutations (same tick as
    ///     queued)
    /// 12. tick_containers (containers.rs): chest lids, menu range
    ///     checks, hoppers
    /// 13. tick_moving_pistons: moving pistons advance and land
    /// 14. flush_block_entities (containers.rs): block entity data
    ///     syncs
    /// 15. persistence_tick (persistence.rs): the save phase. The
    ///     autosave interval (6000 ticks) arms a sweep of at most
    ///     SAVE_BUDGET_PER_TICK chunks per tick under a wall-clock
    ///     deadline; leftover chunks drain on later ticks.
    ///
    /// Reference loop slots with no implementation yet, in order: the
    /// world border tick; weather advance; the sleep wake check; game
    /// time advance with scheduled function events; fluid scheduled
    /// ticks (after the block ones); raids; natural mob spawning,
    /// thunder, and ice/snow precipitation (inside the chunk phase
    /// before random ticks); the entity tracking sync point (its own
    /// broadcast inside the chunk phase, before block events; entity
    /// frames here ride the entity phase instead); entity storage
    /// management; and the per-connection flush after the world tick
    /// (keep-alives sit there; run() sends them right after game_tick).
    fn game_tick(&mut self) {
        self.tick += 1;
        self.broadcast_time();
        // Scheduled-tick phase: torch transitions, then the rest of the
        // due redstone actions.
        self.fire_torches();
        self.run_scheduled_ticks();
        // Random ticks (the chunk tick's grass pass) land their writes
        // in this tick's flush.
        self.random_ticks();
        // The broadcast point sits after the chunk phase and before block
        // events: world edits made by the phases below broadcast one tick
        // later, batched with that tick's scheduled-phase edits.
        self.flush_dirty();
        // The entity sync flush rides inside the same chunk-source tick.
        self.track_entities();
        // Block events: piston world mutations happen here, in the same
        // tick they were queued.
        self.run_block_events();
        // The entity pass: pickup vacuum, drop physics/merges/lifetimes,
        // despawn, then the mob pass (living.rs).
        self.tick_entities();
        // The natural-spawn cycle (spawning.rs) after the entity pass.
        self.natural_spawns();
        // Dig progress advances with the tick counter (the player entity
        // tick's game mode pass); its breaks land in the next tick's
        // flush, like every post-broadcast write.
        self.advance_digs();
        // The per-tick menu broadcast: non-click inventory changes sync
        // here unless a container menu masks them.
        self.broadcast_pending_inventory();
        // Block-entity phase: chest lids, menu range checks, hoppers.
        self.tick_containers();
        // Block-entity phase: moving pistons advance +0.5 and land.
        self.tick_moving_pistons();
        // Changed block entities sync their data packet.
        self.flush_block_entities();
        // Connection flush: one wire drain per tick, after all world
        // work (the reference's post-tick flushQueue).
        self.flush_connections();
        // --- persistence hooks (persistence.rs) ---
        // The save phase: the autosave cadence arms a bounded sweep that
        // drains the dirty set across ticks.
        self.persistence_tick();
    }

    /// set_time (0x73): gameTime i64 + day counter, every 20 ticks.
    /// A world that never ran `time set` stays at the static 0/0 pair.
    fn broadcast_time(&mut self) {
        if self.tick.is_multiple_of(20) {
            let day = self.day_time;
            let game = if self.spawning.time_running {
                self.spawning.total_ticks
            } else {
                0
            };
            let mut body = Vec::with_capacity(18);
            body.extend_from_slice(&game.to_be_bytes());
            body.extend_from_slice(&day.to_be_bytes());
            body.push(0);
            let conns: Vec<ConnId> = self.players.keys().copied().collect();
            for c in conns {
                self.send(c, 0x73, &body);
            }
        }
    }

    /// Sends set_time (0x73) with the stored day time.
    fn send_set_time(&mut self) {
        let mut body = Vec::with_capacity(18);
        body.extend_from_slice(&0i64.to_be_bytes());
        body.extend_from_slice(&self.day_time.to_be_bytes());
        body.push(0);
        let conns: Vec<ConnId> = self.players.keys().copied().collect();
        for c in conns {
            self.send(c, 0x73, &body);
        }
    }

    /// The chunk columns at least one player tracks.
    pub(crate) fn viewed_chunks(&self) -> Vec<(i32, i32)> {
        self.viewers
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(&c, _)| c)
            .collect()
    }

    /// Whether at least one player tracks the chunk column.
    pub(crate) fn chunk_viewed(&self, cx: i32, cz: i32) -> bool {
        self.viewers.get(&(cx, cz)).is_some_and(|v| !v.is_empty())
    }

    /// Loads a chunk into the cache on demand; true when it is present.
    /// The join replay streams packets without populating the cache, so
    /// passes that read column data pull their chunks in first.
    pub(crate) fn ensure_chunk_loaded(&mut self, cx: i32, cz: i32) -> bool {
        if self.chunks.contains_key(&(cx, cz)) {
            return true;
        }
        let (Some(world), blobs) = (self.world.clone(), self.blobs.clone()) else {
            return false;
        };
        self.load_chunk(&world, blobs.as_ref(), cx, cz).is_ok()
    }

    /// The next entity id from the shared counter.
    pub(crate) fn alloc_entity_id(&mut self) -> i32 {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        id
    }

    /// A fresh entity uuid from the survival stream.
    pub(crate) fn next_uuid(&mut self) -> [u8; 16] {
        self.survival.next_uuid()
    }

    /// Torch transitions fire 1gt after their input change; stale
    /// entries (block replaced) are skipped at fire time.
    fn fire_torches(&mut self) {
        let due_torches: Vec<((i32, i32, i32), u32)> = self
            .torch_queue
            .iter()
            .filter(|(_, _, t)| *t <= self.tick)
            .map(|(pos, state, _)| (*pos, *state))
            .collect();
        self.torch_queue.retain(|(_, _, t)| *t > self.tick);
        for ((x, y, z), state) in due_torches {
            self.pending.remove(&((x, y, z), PendingKind::Torch));
            if let Some((n, _)) = self.get_block(x, y, z) {
                if n.contains("redstone_torch") {
                    self.set_block(x, y, z, state, true);
                }
            }
        }
    }

    /// The scheduled-tick phase: fire the redstone actions due this
    /// tick.
    fn run_scheduled_ticks(&mut self) {
        let due: Vec<((i32, i32, i32), TickAction)> = self
            .scheduled
            .iter()
            .filter(|(t, _, _)| *t <= self.tick)
            .map(|(_, pos, action)| (*pos, *action))
            .collect();
        self.scheduled.retain(|(t, _, _)| *t > self.tick);
        for ((x, y, z), action) in due {
            match action {
                TickAction::NeighborUpdate => self.update_block(x, y, z),
                TickAction::ShapeUpdate { dx, dy, dz } => {
                    self.update_block_from(x, y, z, dx, dy, dz)
                }
                TickAction::ObserverToggle => self.observer_toggle(x, y, z),
                TickAction::RepeaterToggle => self.repeater_toggle(x, y, z),
                TickAction::ComparatorToggle => self.comparator_toggle(x, y, z),
            }
        }
    }

    pub(crate) fn handle(&mut self, event: Inbound) {
        match event {
            Inbound::Joined {
                conn,
                name,
                x,
                y,
                z,
                sent,
                tx,
            } => self.handle_joined(conn, name, x, y, z, sent, tx),
            Inbound::Moved {
                conn,
                x,
                y,
                z,
                yaw,
                pitch,
            } => {
                if let Some(p) = self.players.get_mut(&conn) {
                    p.x = x;
                    p.y = y;
                    p.z = z;
                    if let Some(yaw) = yaw {
                        p.yaw = yaw;
                    }
                    if let Some(pitch) = pitch {
                        p.pitch = pitch;
                    }
                }
                // --- tracker hooks (tracker.rs) ---
                // A move re-checks every entity's pairing range before
                // the view swap (a section crossing pairs on the next
                // move, as the reference's move handler orders it).
                self.track_player_view(conn);
                self.stream_if_moved(conn);
            }
            Inbound::Tp { conn, x, y, z } => {
                self.teleport_conn(conn, x, y, z);
                self.send_command_feedback(conn);
            }
            Inbound::TpNamed {
                conn,
                name,
                x,
                y,
                z,
            } => {
                let target = self
                    .players
                    .iter()
                    .find(|(_, p)| p.name == name)
                    .map(|(&c, _)| c);
                if let Some(target) = target {
                    self.teleport_conn(target, x, y, z);
                }
                self.send_command_feedback(conn);
            }
            Inbound::KeepAliveAnswer { conn, id } => {
                if let Some(p) = self.players.get_mut(&conn) {
                    if let Some((challenge, _)) = p.pending_keep_alive {
                        if challenge == id {
                            p.pending_keep_alive = None;
                            p.keep_alive_idle = Instant::now();
                        }
                    }
                }
            }
            Inbound::Setblock {
                conn,
                x,
                y,
                z,
                name,
            } => {
                self.setblock(conn, x, y, z, name);
                self.send_command_feedback(conn);
            }
            // --- inventory hooks (inventory.rs) ---
            Inbound::SetCarriedItem { conn, slot } => self.select_hotbar_slot(conn, slot),
            Inbound::ContainerClick { conn, click } => self.container_clicked(conn, &click),
            Inbound::GameMode { conn, creative } => {
                if let Some(p) = self.players.get_mut(&conn) {
                    p.inv.creative = creative;
                }
                self.send_command_feedback(conn);
            }
            Inbound::CreativeSlot { conn, set } => {
                self.creative_slot(conn, set);
            }
            Inbound::Give { conn, item, count } => {
                self.give_item(conn, &item, count);
                self.send_command_feedback(conn);
            }
            // --- survival hooks (entities.rs) ---
            Inbound::GameRule { conn, tick_speed } => {
                self.set_tick_speed(tick_speed);
                self.send_command_feedback(conn);
            }
            Inbound::TimeSet { conn, value } => {
                self.day_time = value;
                // The spawn cycle's darkness/burn timelines read the
                // spawning clock, so the command drives both.
                self.spawning.day_time = value.rem_euclid(24000) as u64;
                self.spawning.time_running = true;
                self.send_set_time();
                self.send_command_feedback(conn);
            }
            Inbound::GameRuleNoop { conn } => {
                self.send_command_feedback(conn);
            }
            // --- mob hooks (living.rs / spawning.rs) ---
            Inbound::SpawnMobs { .. } | Inbound::SetDifficulty { .. } => {
                self.apply_mob_command(event);
            }
            // --- placement hooks (placement.rs) ---
            Inbound::Rotated { conn, yaw, pitch } => {
                if let Some(p) = self.players.get_mut(&conn) {
                    p.yaw = yaw;
                    p.pitch = pitch;
                }
            }
            Inbound::UseItemOn {
                conn,
                x,
                y,
                z,
                face,
                hand,
                ..
            } => self.place_from_hand(conn, x, y, z, face, hand),
            // --- breaking hooks (dig.rs) ---
            Inbound::PlayerAction { conn, act } => self.player_action(conn, act),
            // The swing has no server-visible effect yet: the reference
            // only resets an idle clock that this build never reads.
            Inbound::Punch { conn: _ } => {}
            // --- containers hooks (containers.rs) ---
            Inbound::OpenContainer { conn, x, y, z } => {
                self.open_container(conn, x, y, z);
                self.send_command_feedback(conn);
            }
            Inbound::ContainerClose { conn, container_id } => {
                self.client_closed_container(conn, container_id)
            }
            Inbound::Left { conn } => self.handle_left(conn),
            Inbound::TickStep { conn, steps } => {
                // Run the stepped ticks inline: commands queued behind this
                // event in the same channel batch land on later ticks,
                // matching vanilla's `tick step` barrier. The whole step
                // batch shares ONE connection flush: stepped ticks execute
                // inside a single server cycle, and the reference flushes
                // once after it.
                self.flush_suspended = true;
                for _ in 0..steps {
                    self.game_tick();
                }
                self.flush_suspended = false;
                self.send_command_feedback(conn);
            }
            Inbound::TickFreeze { conn, frozen } => {
                self.frozen = frozen;
                self.send_command_feedback(conn);
            }
        }
    }

    /// The join event: register the connection, restore saved player
    /// state when the store has it, and pair the newcomer with every
    /// tracked entity.
    #[allow(clippy::too_many_arguments)]
    fn handle_joined(
        &mut self,
        conn: ConnId,
        name: String,
        x: f64,
        y: f64,
        z: f64,
        sent: Vec<(i32, i32)>,
        tx: Sender<Outbound>,
    ) {
        self.outbounds.insert(conn, tx);
        // Register the viewer index for chunks the join burst
        // already delivered - without this, block broadcasts skip
        // players who never triggered movement streaming.
        for chunk in &sent {
            self.viewers.entry(*chunk).or_default().push(conn);
        }
        // --- persistence hooks (persistence.rs) ---
        // Saved player state restores before any join state goes
        // out: position, rotation, game mode, inventory.
        let saved = self.load_saved_player(&name);
        let (x, y, z, yaw, pitch) = match &saved {
            Some(data) => (data.pos[0], data.pos[1], data.pos[2], data.yaw, data.pitch),
            None => (x, y, z, 0.0, 0.0),
        };
        let entity_id = self.next_entity_id;
        self.next_entity_id += 1;
        self.players.insert(
            conn,
            Player {
                name,
                x,
                y,
                z,
                yaw,
                pitch,
                center: None,
                sent: sent.into_iter().collect(),
                teleport_id: 1,
                pending_keep_alive: None,
                keep_alive_idle: Instant::now(),
                inv: Default::default(),
                menu: Default::default(),
                entity_id,
                dig: Default::default(),
            },
        );
        if let Some(data) = saved {
            let mut sync = None;
            if let Some(p) = self.players.get_mut(&conn) {
                p.inv.creative = data.game_mode == 1;
                for slot in data.inventory {
                    if let Some(stack) = crate::persistence::saved_to_stack(&slot) {
                        p.inv.inventory.set(slot.slot as usize, Some(stack));
                    }
                }
                // The client stands where the join burst placed it; a
                // restored position needs its own sync to take hold
                // before the first move packet overwrites it.
                let teleport_id = p.teleport_id;
                p.teleport_id += 1;
                sync = Some(position_sync_body(
                    teleport_id,
                    p.x,
                    p.y,
                    p.z,
                    p.yaw,
                    p.pitch,
                ));
            }
            if let Some(body) = sync {
                self.send(conn, PACKET_PLAYER_POSITION, &body);
            }
        }
        // --- tracker hooks (tracker.rs) ---
        // The newcomer pairs with every entity already in range.
        self.track_player_view(conn);
    }

    /// The disconnect event: drop the menu, untrack, save the player's
    /// state, and forget the viewer index.
    fn handle_left(&mut self, conn: ConnId) {
        // --- containers hooks (containers.rs) ---
        // The carried stack of an open menu drops with the player.
        self.close_menu(conn, false, true);
        // --- tracker hooks (tracker.rs) ---
        self.track_player_left(conn);
        // --- persistence hooks (persistence.rs) ---
        self.save_player_data(conn);
        let Some(p) = self.players.remove(&conn) else {
            return;
        };
        self.outbounds.remove(&conn);
        for chunk in p.sent {
            if let Some(v) = self.viewers.get_mut(&chunk) {
                v.retain(|c| *c != conn);
            }
        }
    }

    /// The mob-hook commands: apply the state, then acknowledge; the
    /// three share one reply path and differ only in their writes.
    fn apply_mob_command(&mut self, event: Inbound) {
        match event {
            Inbound::SpawnMobs { conn, enabled } => {
                self.spawning.spawn_mobs = enabled;
                self.send_command_feedback(conn);
            }

            Inbound::SetDifficulty { conn, peaceful } => {
                self.spawning.peaceful = peaceful;
                self.send_command_feedback(conn);
            }
            _ => {}
        }
    }

    /// Known block-state ids (palette-learned + setblock-probed). The
    /// registry extraction replaces this table eventually.
    fn state_id(name: &str) -> Option<u32> {
        Some(match name {
            "minecraft:air" => 0,
            "minecraft:stone" => 1,
            "minecraft:dirt" => 10,
            "minecraft:grass_block" => 9,
            "minecraft:bedrock" => 88,
            "minecraft:oak_planks" => 15,
            _ => return None,
        })
    }

    /// The first world write: mutate the cached chunk's section, bump its
    /// version, and broadcast one section_blocks_update (0x56) to every
    /// viewer - exactly vanilla's batching (one packet per section per tick).
    fn setblock(&mut self, _conn: ConnId, x: i32, y: i32, z: i32, name: String) {
        let Some(state) = self.resolve_state(&name).or_else(|| Self::state_id(&name)) else {
            eprintln!("[game] setblock: unknown block {name}");
            return;
        };
        self.set_block(x, y, z, state, true);
    }

    // --- placement hooks (placement.rs) ---

    // --- placement hooks (placement.rs) ---
    // --- breaking hooks (dig.rs) ---

    /// Tick end: one broadcast per dirty section. Vanilla batches per
    /// section per tick in a position-keyed set (dedup, last state wins)
    /// and sends a plain block_update (0x08) when exactly one position
    /// changed (`ChunkHolder.broadcastChanges`).
    fn flush_dirty(&mut self) {
        if self.dirty.is_empty() {
            return;
        }
        let dirty = std::mem::take(&mut self.dirty);
        for ((cx, cz, sy), mut changes) in dirty {
            changes.sort_unstable_by_key(|(local, _)| *local);
            // Dedup keeping the LAST state per position (vanilla reads the
            // section at broadcast time): walk from the end, then re-reverse.
            let mut last_wins: Vec<(u64, u32)> = Vec::with_capacity(changes.len());
            for change in changes.iter().rev() {
                let prev_differs = !matches!(last_wins.last(), Some((l, _)) if *l == change.0);
                if last_wins.is_empty() || prev_differs {
                    last_wins.push(*change);
                }
            }
            last_wins.reverse();
            let changes = last_wins;
            let viewers: Vec<ConnId> = self.viewers.get(&(cx, cz)).cloned().unwrap_or_default();
            if viewers.is_empty() {
                continue;
            }
            if changes.len() == 1 {
                // block_update (0x08): packed BlockPos + VarInt state.
                let (local, state) = changes[0];
                let x = cx * 16 + ((local >> 8) & 0xf) as i32;
                let z = cz * 16 + ((local >> 4) & 0xf) as i32;
                let y = sy * 16 + (local & 0xf) as i32;
                let packed = (((x as i64) & 0x3ff_ffff) << 38)
                    | (((z as i64) & 0x3ff_ffff) << 12)
                    | ((y as i64) & 0xfff);
                let mut body = Vec::with_capacity(12);
                body.extend_from_slice(&packed.to_be_bytes());
                doppel_protocol::write_varint(&mut body, state as i32);
                for v in viewers {
                    self.send(v, 0x08, &body);
                }
                continue;
            }
            let sec_pos = (((cx as i64) & 0x3f_ffff) << 42)
                | (((cz as i64) & 0x3f_ffff) << 20)
                | ((sy as i64) & 0xf_ffff);
            let mut body = Vec::with_capacity(8 + changes.len() * 3);
            body.extend_from_slice(&sec_pos.to_be_bytes());
            doppel_protocol::write_varint(&mut body, changes.len() as i32);
            for (local, state) in &changes {
                write_u64_varlong(&mut body, ((*state as u64) << 12) | *local);
            }
            for v in viewers {
                self.send(v, 0x56, &body);
            }
        }
    }

    // ------------------------------------------------------------------
    // World block access + redstone
    // ------------------------------------------------------------------

    /// Reads the block state at world coords via the registry.
    pub(crate) fn get_block(&self, x: i32, y: i32, z: i32) -> Option<(String, String)> {
        let state = self.get_state_id(x, y, z)?;
        let reg = self.registry.as_ref()?;
        let (name, props) = reg.state_of(state)?;
        Some((name.to_string(), props.to_string()))
    }

    /// Raw state id at world coords (no registry lookup).
    ///
    /// Section storage follows vanilla's YZX layout (`y<<8 | z<<4 | x`);
    /// the section-update wire localPos is XZY (`x<<8 | z<<4 | y`) — two
    /// different conventions, both pinned against vanilla bytes.
    fn get_state_id(&self, x: i32, y: i32, z: i32) -> Option<u32> {
        let cx = x.div_euclid(16);
        let cz = z.div_euclid(16);
        let sec_index = (y.div_euclid(16) + 4) as usize;
        let idx = local_yzx(x, y, z);
        let chunk = self.chunks.get(&(cx, cz))?;
        get_section_cell(&chunk.wire, sec_index, idx)
    }

    /// Command feedback: an empty text component plus overlay=false.
    /// The reference answers every scripted command; the differential
    /// harness paces itself on these replies. Body parity (the exact
    /// component) is future work against a captured reference.
    fn send_command_feedback(&mut self, conn: ConnId) {
        let body = [0x00u8, 0x00];
        self.send(conn, 0x7c, &body);
    }

    /// The teleport body shared by `tp @s` and the named form: the position
    /// update, the 0x49 sync, the pairing re-check against the pre-teleport
    /// view, then the chunk stream catch-up. The reference's move handler
    /// re-checks pairing before it swaps the tracking view, so a teleport
    /// out and back never re-pairs on its own: the drops stay unpaired
    /// until the player's next move.
    fn teleport_conn(&mut self, conn: ConnId, x: f64, y: f64, z: f64) {
        let Some(p) = self.players.get_mut(&conn) else {
            return;
        };
        p.x = x;
        p.y = y;
        p.z = z;
        let sync = position_sync_body(p.teleport_id, x, y, z, p.yaw, p.pitch);
        p.teleport_id += 1;
        self.send(conn, PACKET_PLAYER_POSITION, &sync);
        // --- tracker hooks (tracker.rs) ---
        // The pairing pass reads the old view: entities in the fresh
        // surroundings wait for the player's next move, like the
        // reference's move handler.
        self.track_player_view(conn);
        self.stream_if_moved(conn);
    }

    /// Logs a dropped world write, once per (chunk, reason).
    fn warn_dropped_write(&mut self, cx: i32, cz: i32, reason: &str) {
        if self.warned_writes.insert((cx, cz, reason.to_string())) {
            eprintln!("[game] setblock dropped at chunk ({cx},{cz}): {reason}");
        }
    }

    /// Writes a block state, marking dirty + scheduling neighbor updates.
    /// A write equal to the current state is a full no-op (vanilla's
    /// `LevelChunk.setBlockState` returns null without broadcasting).
    pub(crate) fn set_block(&mut self, x: i32, y: i32, z: i32, state: u32, notify: bool) {
        let cx = x.div_euclid(16);
        let cz = z.div_euclid(16);
        let sec_index = (y.div_euclid(16) + 4) as usize;
        let idx = local_yzx(x, y, z);
        if self.get_state_id(x, y, z) == Some(state) {
            return;
        }
        if !self.chunks.contains_key(&(cx, cz)) {
            let (Some(world), blobs) = (self.world.clone(), self.blobs.clone()) else {
                self.warn_dropped_write(cx, cz, "no world or blobs configured");
                return;
            };
            if let Err(e) = self.load_chunk(&world, blobs.as_ref(), cx, cz) {
                self.warn_dropped_write(cx, cz, &format!("chunk load failed: {e:#}"));
                return;
            }
        }
        let chunk = self.chunks.get_mut(&(cx, cz)).expect("loaded");
        if !set_section_cell(&mut chunk.wire, sec_index, idx, state) {
            self.warn_dropped_write(
                cx,
                cz,
                &format!("section write rejected (section {sec_index} missing or palette >8 bits)"),
            );
            return;
        }
        chunk.version += 1;
        // --- persistence hooks (persistence.rs) ---
        self.mark_chunk_dirty(cx, cz);
        // --- survival hooks (entities.rs) ---
        // A write re-arms the section's random-tick scan.
        self.invalidate_section_ticks(cx, cz, y.div_euclid(16) + 4);
        self.dirty
            .entry((cx, cz, y.div_euclid(16)))
            .or_default()
            .push((local_xzy(x, y, z), state));
        if notify {
            for (dx, dy, dz) in NEIGHBORS {
                self.scheduled.push((
                    self.tick,
                    (x + dx, y + dy, z + dz),
                    TickAction::ShapeUpdate {
                        dx: -dx,
                        dy: -dy,
                        dz: -dz,
                    },
                ));
            }
            self.scheduled
                .push((self.tick, (x, y, z), TickAction::NeighborUpdate));
        }
        // --- containers hooks (containers.rs) ---
        // A successful write re-syncs the block-entity map with the block.
        self.sync_block_entity(x, y, z);
    }

    /// The state id for a spec like "name[k=v]".
    pub(crate) fn resolve_state(&self, spec: &str) -> Option<u32> {
        let reg = self.registry.as_ref()?;
        let (name, props) = doppel_world::registry::BlockRegistry::split_state(spec);
        reg.state_id(name, props)
    }

    /// A scheduled block update: recompute redstone behavior at this pos.
    /// True when the block can bear a floor-mounted component. Mirrors
    /// the reference's support rule for the redstone families: a full
    /// solid face below. The non-support set covers everything the
    /// circuits place that the reference treats as non-supporting.
    fn is_support(&self, x: i32, y: i32, z: i32) -> bool {
        match self.get_block(x, y, z) {
            None => false,
            Some((name, _)) => !matches!(
                name.as_str(),
                "minecraft:air"
                    | "minecraft:redstone_wire"
                    | "minecraft:redstone_torch"
                    | "minecraft:redstone_wall_torch"
                    | "minecraft:lever"
                    | "minecraft:repeater"
                    | "minecraft:comparator"
            ),
        }
    }

    /// The reference drops floor-mounted redstone components whose
    /// support is gone (updateOrDestroy): wire, repeater, comparator,
    /// standing torch, floor lever.
    fn check_survival(&mut self, x: i32, y: i32, z: i32, name: &str, props: &str) -> bool {
        let needs_floor_support = match name {
            "minecraft:redstone_wire" | "minecraft:repeater" | "minecraft:comparator" => true,
            "minecraft:redstone_torch" => true,
            "minecraft:lever" => props.contains("face=floor"),
            _ => false,
        };
        if needs_floor_support && !self.is_support(x, y - 1, z) {
            if std::env::var_os("POP_TRACE").is_some() {
                let below = self
                    .get_block(x, y - 1, z)
                    .map(|(n, p)| format!("{n}[{p}]"))
                    .unwrap_or_else(|| "none".into());
                eprintln!(
                    "[pop] tick {} {name} at ({x},{y},{z}): below={below}",
                    self.tick
                );
            }
            self.set_block(x, y, z, 0, true);
            return false;
        }
        true
    }

    fn update_block(&mut self, x: i32, y: i32, z: i32) {
        let Some((name, props)) = self.get_block(x, y, z) else {
            return;
        };
        if !self.check_survival(x, y, z, &name, &props) {
            return;
        }
        match name.as_str() {
            "minecraft:redstone_wire" => self.update_wire(x, y, z, &props),
            "minecraft:redstone_torch" | "minecraft:redstone_wall_torch" => {
                self.update_torch(x, y, z, &name, &props)
            }
            "minecraft:repeater" => self.update_repeater(x, y, z, &props),
            "minecraft:comparator" => self.update_comparator(x, y, z, &props),
            // --- containers hooks (containers.rs) ---
            "minecraft:hopper" => self.update_hopper(x, y, z, &props),
            "minecraft:piston" | "minecraft:sticky_piston" => self.update_piston(x, y, z),
            // The head forwards neighbor updates to its base
            // (`PistonHeadBlock.neighborChanged`).
            "minecraft:piston_head" => {
                let dir = dir_id(prop_dir(&props));
                let (bx, by, bz) = offset((x, y, z), dir_step(dir), -1);
                self.update_piston(bx, by, bz);
            }
            _ => {}
        }
    }

    /// Recompute a wire's power; on change, update + notify neighbors.
    fn update_wire(&mut self, x: i32, y: i32, z: i32, props: &str) {
        wire::update_wire_cascade(self, x, y, z, props);
    }

    /// A block changed at (x + dx, y + dy, z + dz): the reference's
    /// directional updateShape. Wires recompute the one side facing the
    /// change (a connectivity flip triggers the full re-derive; a visual
    /// up/side flip just swaps the side); everything else falls through
    /// to the undirected recompute.
    fn update_block_from(&mut self, x: i32, y: i32, z: i32, dx: i32, dy: i32, dz: i32) {
        let Some((name, props)) = self.get_block(x, y, z) else {
            return;
        };
        if name == "minecraft:redstone_wire" {
            wire::update_wire_side(self, (x, y, z), &props, dx, dy, dz);
            if let Some((n, p2)) = self.get_block(x, y, z) {
                if n == "minecraft:redstone_wire" {
                    wire::update_wire_power_only(self, x, y, z, &p2);
                }
            }
            return;
        }
        // Observers react only to changes at their watched face; their
        // own state writes must not re-trigger the pulse.
        if name == "minecraft:observer" {
            let facing = prop_dir(&props);
            if (dx, dy, dz) == dir_step(dir_id(facing)) {
                self.update_observer(x, y, z, &props);
            }
            return;
        }
        self.update_block(x, y, z);
    }

    /// Torch: lit unless its supporting block carries power. Simple model:
    /// check the neighbors of the block below the torch. 1gt delay via
    /// pending_torch applied at next tick.
    fn update_torch(&mut self, x: i32, y: i32, z: i32, name: &str, props: &str) {
        let (ax, ay, az) = (x, y - 1, z);
        let input_power = wire::signal_toward_consumer_above(self, ax, ay, az);
        let should_be_lit = input_power == 0;
        let lit = !props.contains("lit=false");
        if lit == should_be_lit {
            return;
        }
        let new_props = doppel_world::registry::BlockRegistry::with_prop(
            props,
            "lit",
            if should_be_lit { "true" } else { "false" },
        );
        let spec = format!("{name}[{new_props}]");
        let Some(new_state) = self.resolve_state(&spec) else {
            return;
        };
        // 1gt delay via the scheduled queue with the target state stashed
        // on the torch queue (single active transition per position).
        if !self.pending.contains(&((x, y, z), PendingKind::Torch)) {
            self.pending.insert(((x, y, z), PendingKind::Torch));
            self.torch_queue.push(((x, y, z), new_state, self.tick + 1));
        }
    }

    /// Observer trigger (vanilla updateShape + startSignal): when an
    /// unpowered observer's watched position changes, schedule its pulse
    /// 2gt out — unless one is already pending (hasScheduledTick guard).
    fn update_observer(&mut self, x: i32, y: i32, z: i32, props: &str) {
        if props.contains("powered=true") {
            return;
        }
        if self.pending.contains(&((x, y, z), PendingKind::Observer)) {
            return;
        }
        self.pending.insert(((x, y, z), PendingKind::Observer));
        self.scheduled
            .push((self.tick + 2, (x, y, z), TickAction::ObserverToggle));
    }

    /// Observer pulse edge (vanilla tick): rising sets powered and chains
    /// the falling edge 2gt later; falling clears it. Both notify the
    /// block behind (opposite the facing).
    fn observer_toggle(&mut self, x: i32, y: i32, z: i32) {
        self.pending.remove(&((x, y, z), PendingKind::Observer));
        let Some((name, props)) = self.get_block(x, y, z) else {
            return;
        };
        if name != "minecraft:observer" {
            return;
        }
        let powered = props.contains("powered=true");
        if std::env::var_os("OBS_TRACE").is_some() {
            eprintln!(
                "[obs] tick {} toggle at ({x},{y},{z}): {}",
                self.tick,
                if powered { "off" } else { "on" }
            );
        }
        let new_props = doppel_world::registry::BlockRegistry::with_prop(
            &props,
            "powered",
            if powered { "false" } else { "true" },
        );
        let spec = format!("{name}[{new_props}]");
        let Some(new_state) = self.resolve_state(&spec) else {
            return;
        };
        self.set_block(x, y, z, new_state, true);
        if !powered {
            // Chain the falling edge.
            self.pending.insert(((x, y, z), PendingKind::Observer));
            self.scheduled
                .push((self.tick + 2, (x, y, z), TickAction::ObserverToggle));
        }
    }

    /// Repeater input: power at the block on its facing side (vanilla
    /// getInputSignal reads pos.relative(FACING)).
    fn repeater_input(&self, x: i32, y: i32, z: i32, facing: &str) -> i32 {
        let (dx, dz) = match facing {
            "north" => (0, -1),
            "south" => (0, 1),
            "west" => (-1, 0),
            "east" => (1, 0),
            _ => (0, 0),
        };
        if (dx, dz) == (0, 0) {
            return 0;
        }
        // Direct source at the input position.
        if let Some((n, p)) = self.get_block(x + dx, y, z + dz) {
            match n.as_str() {
                "minecraft:redstone_wire" => {
                    return doppel_world::registry::BlockRegistry::prop_int(&p, "power")
                        .unwrap_or(0);
                }
                "minecraft:lever" if p.contains("powered=true") => return 15,
                "minecraft:repeater" if p.contains("powered=true") => return 15,
                "minecraft:comparator" if p.contains("powered=true") => return 15,
                "minecraft:redstone_torch" | "minecraft:redstone_wall_torch"
                    if !p.contains("lit=false") =>
                {
                    return 15
                }
                "minecraft:redstone_block" => return 15,
                "minecraft:observer" if p.contains("powered=true") => return 15,
                _ => {}
            }
        }
        0
    }

    /// Repeater locked: powered from a perpendicular side (vanilla isLocked).
    fn repeater_locked(&self, x: i32, y: i32, z: i32, facing: &str) -> bool {
        let sides: [(i32, i32); 2] = match facing {
            "north" | "south" => [(1, 0), (-1, 0)],
            _ => [(0, 1), (0, -1)],
        };
        for (dx, dz) in sides {
            if let Some((n, p)) = self.get_block(x + dx, y, z + dz) {
                let powered = match n.as_str() {
                    "minecraft:redstone_wire" => {
                        doppel_world::registry::BlockRegistry::prop_int(&p, "power").unwrap_or(0)
                            > 0
                    }
                    "minecraft:lever" => p.contains("powered=true"),
                    "minecraft:repeater" => p.contains("powered=true"),
                    _ => false,
                };
                if powered {
                    return true;
                }
            }
        }
        false
    }

    /// checkTickOnNeighbor: schedule the output change at delay*2 ticks
    /// when the desired state differs (vanilla's willTickThisTick guard).
    fn update_repeater(&mut self, x: i32, y: i32, z: i32, props: &str) {
        if props.contains("locked=true") {
            return;
        }
        let facing = prop_dir(props);
        let should_on = self.repeater_input(x, y, z, facing) > 0;
        let on = props.contains("powered=true");
        if on != should_on && !self.pending.contains(&((x, y, z), PendingKind::Repeater)) {
            let delay = doppel_world::registry::BlockRegistry::prop_int(props, "delay")
                .unwrap_or(1)
                .clamp(1, 4) as u64;
            self.pending.insert(((x, y, z), PendingKind::Repeater));
            self.scheduled
                .push((self.tick + delay * 2, (x, y, z), TickAction::RepeaterToggle));
        }
    }

    /// Repeater tick edge (vanilla DiodeBlock.tick): turn off, or turn on
    /// and chain a turn-off for short pulses.
    fn repeater_toggle(&mut self, x: i32, y: i32, z: i32) {
        self.pending.remove(&((x, y, z), PendingKind::Repeater));
        let Some((name, props)) = self.get_block(x, y, z) else {
            return;
        };
        if name != "minecraft:repeater" {
            return;
        }
        if self.repeater_locked(x, y, z, prop_dir(&props)) {
            return;
        }
        let should_on = self.repeater_input(x, y, z, prop_dir(&props)) > 0;
        let on = props.contains("powered=true");
        let new_powered = if on && !should_on {
            false
        } else if !on {
            true
        } else {
            return;
        };
        let new_props = doppel_world::registry::BlockRegistry::with_prop(
            &props,
            "powered",
            if new_powered { "true" } else { "false" },
        );
        let spec = format!("{name}[{new_props}]");
        let Some(new_state) = self.resolve_state(&spec) else {
            return;
        };
        self.set_block(x, y, z, new_state, true);
        if new_powered && !should_on {
            // Short pulse: chain the turn-off at the same delay.
            let delay = doppel_world::registry::BlockRegistry::prop_int(&props, "delay")
                .unwrap_or(1)
                .clamp(1, 4) as u64;
            self.pending.insert(((x, y, z), PendingKind::Repeater));
            self.scheduled
                .push((self.tick + delay * 2, (x, y, z), TickAction::RepeaterToggle));
        }
    }

    // ------------------------------------------------------------------
    // Pistons (block events, QC power scan, push structures, moving
    // ------------------------------------------------------------------

    /// Queue an extend/retract block
    /// event for the SAME tick's block-event phase.
    fn update_piston(&mut self, x: i32, y: i32, z: i32) {
        let Some((name, props)) = self.get_block(x, y, z) else {
            return;
        };
        if name != "minecraft:piston" && name != "minecraft:sticky_piston" {
            return;
        }
        let sticky = name == "minecraft:sticky_piston";
        let dir = dir_id(prop_dir(&props));
        let extended = props.contains("extended=true");
        let extend = self.piston_powered(x, y, z, dir);
        if extend && !extended {
            // Dry-run the push: the event is queued only if it would
            // succeed.
            if self.resolve_structure((x, y, z), dir, true).is_some() {
                self.queue_block_event(x, y, z, sticky, 0, dir);
            }
        } else if !extend && extended {
            let mut event = 1u8; // TRIGGER_CONTRACT
                                 // TRIGGER_DROP when the block two ahead is this piston's own
                                 // head carrier still extending and fresh.
            let (tx, ty, tz) = offset((x, y, z), dir_step(dir), 2);
            if self
                .moving
                .iter()
                .any(|m| m.pos == (tx, ty, tz) && m.extending && m.dir == dir && m.progress < 2)
            {
                event = 2;
            }
            self.queue_block_event(x, y, z, sticky, event, dir);
        }
    }

    /// The 5-neighbor scan excluding
    /// the facing side, plus quasi-connectivity — the same scan at the
    /// position above the piston.
    fn piston_powered(&self, x: i32, y: i32, z: i32, facing: u8) -> bool {
        for d in 0u8..6 {
            if d == facing {
                continue;
            }
            let (sx, sy, sz) = offset((x, y, z), dir_step(d), 1);
            if self.signal_toward(x, y, z, d, sx, sy, sz) {
                return true;
            }
        }
        // Step 2 of the vanilla scan (own position, DOWN) reads strong
        // power conducted through the piston — pistons are never
        // conductors, so it is always false in 26.3.
        let above = (x, y + 1, z);
        for d in 0u8..6 {
            if d == DIR_DOWN {
                continue;
            }
            let (sx, sy, sz) = offset(above, dir_step(d), 1);
            if self.signal_toward(above.0, above.1, above.2, d, sx, sy, sz) {
                return true;
            }
        }
        false
    }

    /// Whether the source block at (sx,sy,sz) powers the block at
    /// (tx,ty,tz) — `SignalGetter.hasSignal(target, d)` with d the
    /// direction from target to source.
    #[allow(clippy::too_many_arguments)]
    fn signal_toward(&self, tx: i32, ty: i32, tz: i32, d: u8, sx: i32, sy: i32, sz: i32) -> bool {
        let Some((name, props)) = self.get_block(sx, sy, sz) else {
            return false;
        };
        match name.as_str() {
            "minecraft:lever" => props.contains("powered=true"),
            "minecraft:redstone_torch" | "minecraft:redstone_wall_torch" => {
                !props.contains("lit=false")
            }
            "minecraft:redstone_block" => true,
            "minecraft:redstone_wire" => {
                if doppel_world::registry::BlockRegistry::prop_int(&props, "power").unwrap_or(0)
                    == 0
                {
                    return false;
                }
                match d {
                    DIR_UP => true,
                    DIR_DOWN => false,
                    _ => {
                        // Horizontal: the wire's side toward the target
                        // must be connected.
                        let side = dir_name(dir_opposite(d));
                        let v = prop_value(&props, side);
                        v == "side" || v == "up"
                    }
                }
            }
            // Diodes emit only from their facing side.
            "minecraft:repeater" | "minecraft:observer" | "minecraft:comparator" => {
                props.contains("powered=true") && dir_id(prop_dir(&props)) == dir_opposite(d)
            }
            _ => {
                let _ = (tx, ty, tz);
                false
            }
        }
    }

    /// `ServerLevel.blockEvent`: insertion-ordered, duplicates dropped.
    fn queue_block_event(&mut self, x: i32, y: i32, z: i32, sticky: bool, event: u8, dir: u8) {
        let e = BlockEvent {
            pos: (x, y, z),
            sticky,
            event,
            dir,
        };
        if !self.block_events.contains(&e) {
            self.block_events.push(e);
        }
    }

    /// `ServerLevel.runBlockEvents`: drains FIFO; events queued while
    /// draining (cascades) run in the same tick, bounded defensively.
    fn run_block_events(&mut self) {
        let mut guard = 0usize;
        while !self.block_events.is_empty() {
            let batch = std::mem::take(&mut self.block_events);
            for e in batch {
                guard += 1;
                if guard > 10_000 {
                    return;
                }
                self.fire_block_event(e);
            }
        }
    }

    /// The block-event handler incl. the server-side staleness
    /// re-validation.
    fn fire_block_event(&mut self, e: BlockEvent) {
        let Some((name, props)) = self.get_block(e.pos.0, e.pos.1, e.pos.2) else {
            return;
        };
        let sticky = name == "minecraft:sticky_piston";
        if name != "minecraft:piston" && !sticky {
            return; // block replaced while queued
        }
        if e.sticky != sticky {
            return; // piston kind changed
        }
        let dir = dir_id(prop_dir(&props));
        let extend = self.piston_powered(e.pos.0, e.pos.1, e.pos.2, dir);
        if extend && (e.event == 1 || e.event == 2) {
            // Retract event, but still powered: snap extended=true
            // (flag 2: clients only, no neighbor updates).
            let new_props =
                doppel_world::registry::BlockRegistry::with_prop(&props, "extended", "true");
            let spec = format!("{name}[{new_props}]");
            if let Some(state) = self.resolve_state(&spec) {
                self.set_block(e.pos.0, e.pos.1, e.pos.2, state, false);
            }
            return;
        }
        if !extend && e.event == 0 {
            return; // extend event, no longer powered
        }
        match e.event {
            0 => {
                self.piston_extend(e.pos, dir, sticky);
            }
            1 | 2 => {
                self.piston_retract(e.pos, dir, sticky, e.event == 1);
            }
            _ => {}
        }
    }

    /// The block move for extend, and the pull half of
    /// retract. Returns false when the structure fails to resolve.
    fn piston_move_blocks(
        &mut self,
        pos: (i32, i32, i32),
        dir: u8,
        sticky: bool,
        extending: bool,
    ) -> bool {
        if !extending {
            // clear a landed head at the arm before resolving the
            // pull, so the pulled line can slide into the arm cell.
            let arm = offset(pos, dir_step(dir), 1);
            if let Some((n, _)) = self.get_block(arm.0, arm.1, arm.2) {
                if n == "minecraft:piston_head" {
                    self.set_block(arm.0, arm.1, arm.2, AIR_STATE, false);
                }
            }
        }
        let Some((to_push, to_destroy)) = self.resolve_structure(pos, dir, extending) else {
            return false;
        };
        // Destroy phase: farthest first, no neighbor updates yet.
        for (p, _) in to_destroy.iter().rev() {
            self.set_block(p.0, p.1, p.2, AIR_STATE, false);
        }
        // Move phase: farthest first; carrier at p + push_dir.
        let push_dir = if extending { dir } else { dir_opposite(dir) };
        // Carried blocks ride a default-typed carrier; only the
        // piston's own pieces carry the sticky type.
        let carrier = self.carrier_state(dir);
        let mut targets = std::collections::BTreeSet::new();
        for (p, state) in to_push.iter().rev() {
            let np = offset(*p, dir_step(push_dir), 1);
            targets.insert(np);
            self.set_block(np.0, np.1, np.2, carrier, false);
            self.moving.push(MovingPiston {
                pos: np,
                moved_state: *state,
                dir,
                extending,
                is_source: false,
                progress: 0,
            });
        }
        if extending {
            // The piston's own arm: a sticky-typed carrier that
            // carries the head.
            let arm = offset(pos, dir_step(dir), 1);
            targets.insert(arm);
            self.set_block(
                arm.0,
                arm.1,
                arm.2,
                self.moving_piston_state(dir, sticky),
                false,
            );
            let head = self.piston_head_state(dir, sticky);
            self.moving.push(MovingPiston {
                pos: arm,
                moved_state: head,
                dir,
                extending: true,
                is_source: true,
                progress: 0,
            });
        }
        // Vacate phase: originals not overwritten by carriers.
        for (p, _) in &to_push {
            if !targets.contains(p) {
                self.set_block(p.0, p.1, p.2, AIR_STATE, false);
            }
        }
        // Neighbor-update pass: the destroyed, the vacated and the
        // moved positions all re-check their neighbors.
        let mut notify: Vec<(i32, i32, i32)> = Vec::new();
        for (p, _) in to_destroy.iter().rev() {
            notify.push(*p);
        }
        for (p, _) in to_push.iter().rev() {
            notify.push(*p);
        }
        if extending {
            notify.push(offset(pos, dir_step(dir), 1));
        }
        for (nx, ny, nz) in notify {
            self.schedule_neighbor_update(nx, ny, nz);
        }
        true
    }

    /// The extend path: moveBlocks + the base switching to extended.
    fn piston_extend(&mut self, pos: (i32, i32, i32), dir: u8, sticky: bool) -> bool {
        if !self.piston_move_blocks(pos, dir, sticky, true) {
            return false;
        }
        let Some((name, props)) = self.get_block(pos.0, pos.1, pos.2) else {
            return true;
        };
        let new_props =
            doppel_world::registry::BlockRegistry::with_prop(&props, "extended", "true");
        let spec = format!("{name}[{new_props}]");
        if let Some(state) = self.resolve_state(&spec) {
            // Flag 67: clients + neighbors + moved-by-piston.
            self.set_block(pos.0, pos.1, pos.2, state, true);
        }
        true
    }

    /// The retract path.
    fn piston_retract(&mut self, pos: (i32, i32, i32), dir: u8, sticky: bool, pull: bool) {
        let arm = offset(pos, dir_step(dir), 1);
        // Settle a lingering arm carrier (finalTick: source pieces go to
        // air).
        self.finalize_moving_piston_at(arm);
        // The retracting base becomes a carrier of its own retracted
        // state (flag 276: no neighbor updates).
        let carrier = self.moving_piston_state(dir, sticky);
        let base_state = self.piston_base_state(dir, sticky);
        self.set_block(pos.0, pos.1, pos.2, carrier, false);
        self.moving.push(MovingPiston {
            pos,
            moved_state: base_state,
            dir,
            extending: false,
            is_source: true,
            progress: 0,
        });
        // Sticky pull.
        if sticky {
            let two = offset(pos, dir_step(dir), 2);
            // Rule 1: the block two ahead is this piston's own carrier
            // still extending — settle it, pull nothing this tick.
            if self
                .moving
                .iter()
                .any(|m| m.pos == two && m.extending && m.dir == dir && m.is_source)
            {
                self.finalize_moving_piston_at(two);
                return;
            }
            // Rule 2: pull only on TRIGGER_CONTRACT, when the target is
            // pullable (PUSH_PULL reaction, or a piston).
            if pull {
                if let Some((n, p)) = self.get_block(two.0, two.1, two.2) {
                    let pushable = self.is_pushable(&n, &p, two, dir_opposite(dir), false, dir);
                    let pull_reaction = n == "minecraft:piston"
                        || n == "minecraft:sticky_piston"
                        || push_reaction(&n) == PushReaction::PushPull;
                    if n != "minecraft:air" && pushable && pull_reaction {
                        self.piston_move_blocks(pos, dir, sticky, false);
                        return;
                    }
                }
            }
        }
        // Rule 3: just remove the head (no drops).
        if let Some((n, _)) = self.get_block(arm.0, arm.1, arm.2) {
            if n == "minecraft:piston_head" {
                self.set_block(arm.0, arm.1, arm.2, AIR_STATE, false);
            }
        }
        self.schedule_neighbor_update(arm.0, arm.1, arm.2);
        self.schedule_neighbor_update(pos.0, pos.1, pos.2);
    }

    /// Schedules a neighbor update on the six neighbours of a position
    /// (`level.updateNeighborsAt`).
    fn schedule_neighbor_update(&mut self, x: i32, y: i32, z: i32) {
        for (dx, dy, dz) in NEIGHBORS {
            self.scheduled.push((
                self.tick,
                (x + dx, y + dy, z + dz),
                TickAction::NeighborUpdate,
            ));
        }
    }

    /// The push-list resolver: straight-line push list +
    /// head-on destroys. Slime/honey backward chains are honored; side
    /// branching (`addBranchingBlocks`) is NOT YET.
    #[allow(clippy::type_complexity)]
    fn resolve_structure(
        &self,
        piston_pos: (i32, i32, i32),
        dir: u8,
        extending: bool,
    ) -> Option<(Vec<((i32, i32, i32), u32)>, Vec<((i32, i32, i32), u32)>)> {
        let push_dir = if extending { dir } else { dir_opposite(dir) };
        let start = if extending {
            offset(piston_pos, dir_step(dir), 1)
        } else {
            offset(piston_pos, dir_step(dir), 2)
        };
        let mut to_push: Vec<(i32, i32, i32)> = Vec::new();
        let mut to_destroy: Vec<(i32, i32, i32)> = Vec::new();
        let (name, props) = self.get_block(start.0, start.1, start.2)?;
        if !self.is_pushable(&name, &props, start, push_dir, false, dir) {
            if extending && push_reaction(&name) == PushReaction::Popped {
                to_destroy.push(start); // head-on destructible block
                let states = |v: &Vec<(i32, i32, i32)>| {
                    v.iter()
                        .map(|p| (*p, self.get_state_id(p.0, p.1, p.2).unwrap_or(AIR_STATE)))
                        .collect()
                };
                return Some((states(&to_push), states(&to_destroy)));
            }
            return None;
        }
        self.add_block_line(start, piston_pos, push_dir, &mut to_push, &mut to_destroy)?;
        // Branching (slime side-attachments) is NOT YET implemented.
        let states = |v: &Vec<(i32, i32, i32)>| {
            v.iter()
                .map(|p| (*p, self.get_state_id(p.0, p.1, p.2).unwrap_or(AIR_STATE)))
                .collect()
        };
        Some((states(&to_push), states(&to_destroy)))
    }

    /// The block-line walk, minus the collision
    /// reorder (single-line structures only). None = resolution failure.
    fn add_block_line(
        &self,
        start: (i32, i32, i32),
        piston_pos: (i32, i32, i32),
        push_dir: u8,
        to_push: &mut Vec<(i32, i32, i32)>,
        to_destroy: &mut Vec<(i32, i32, i32)>,
    ) -> Option<()> {
        let Some((name, props)) = self.get_block(start.0, start.1, start.2) else {
            return Some(());
        };
        if name == "minecraft:air" {
            return Some(());
        }
        if !self.is_pushable(&name, &props, start, push_dir, false, push_dir) {
            return Some(()); // branch not part of the structure
        }
        if start == piston_pos || to_push.contains(&start) {
            return Some(());
        }
        let mut block_count = 1usize;
        if block_count + to_push.len() > MAX_PUSH_DEPTH {
            return None;
        }
        // Backward sticky chain: while the start block sticks, walk
        // against the push direction.
        let mut cursor_name = name.clone();
        while is_sticky(&cursor_name) {
            let pos = offset(start, dir_step(dir_opposite(push_dir)), block_count as i32);
            let Some((n, p)) = self.get_block(pos.0, pos.1, pos.2) else {
                break;
            };
            if n == "minecraft:air"
                || !can_stick_to_each_other(&cursor_name, &n)
                || !self.is_pushable(&n, &p, pos, push_dir, false, dir_opposite(push_dir))
                || pos == piston_pos
            {
                break;
            }
            block_count += 1;
            if block_count + to_push.len() > MAX_PUSH_DEPTH {
                return None;
            }
            cursor_name = n;
        }
        // Add the chain, farthest first (nearest-to-piston-first order).
        for i in (0..block_count).rev() {
            to_push.push(offset(start, dir_step(dir_opposite(push_dir)), i as i32));
        }
        // Forward walk until air / unpushable / destructible.
        let mut i = 1i32;
        loop {
            let pos = offset(start, dir_step(push_dir), i);
            if to_push.contains(&pos) {
                return Some(()); // fold into an earlier line (reorder NOT YET)
            }
            let Some((n, p)) = self.get_block(pos.0, pos.1, pos.2) else {
                return Some(());
            };
            if n == "minecraft:air" {
                return Some(());
            }
            if !self.is_pushable(&n, &p, pos, push_dir, true, push_dir) || pos == piston_pos {
                return None;
            }
            if push_reaction(&n) == PushReaction::Popped {
                to_destroy.push(pos); // destroyed instead of pushed
                return Some(());
            }
            if to_push.len() >= MAX_PUSH_DEPTH {
                return None;
            }
            to_push.push(pos);
            i += 1;
        }
    }

    /// The pushability check.
    fn is_pushable(
        &self,
        name: &str,
        props: &str,
        pos: (i32, i32, i32),
        push_dir: u8,
        allow_destroyable: bool,
        connection_dir: u8,
    ) -> bool {
        let y = pos.1;
        if !(WORLD_MIN_Y..=WORLD_MAX_Y).contains(&y) {
            return false;
        }
        if name == "minecraft:air" {
            return true;
        }
        if push_dir == DIR_DOWN && y == WORLD_MIN_Y {
            return false;
        }
        if push_dir == DIR_UP && y == WORLD_MAX_Y {
            return false;
        }
        if name == "minecraft:piston" || name == "minecraft:sticky_piston" {
            // Extended pistons block the push; retracted ones are pushable
            // despite their IMMOVEABLE registration.
            return !props.contains("extended=true");
        }
        if unbreakable(name) {
            return false; // destroy speed -1 (bedrock, moving_piston, ...)
        }
        match push_reaction(name) {
            PushReaction::Immoveable => false,
            PushReaction::Popped => allow_destroyable,
            PushReaction::Push => push_dir == connection_dir,
            PushReaction::PushPull => true,
            PushReaction::IgnoredEntity => true,
        }
    }

    /// The carrier for ordinary pushed/pulled blocks (always normal).
    fn carrier_state(&self, dir: u8) -> u32 {
        let spec = format!(
            "minecraft:moving_piston[facing={},type=normal]",
            dir_name(dir)
        );
        self.resolve_state(&spec).unwrap_or(AIR_STATE)
    }

    fn moving_piston_state(&self, dir: u8, sticky: bool) -> u32 {
        let spec = format!(
            "minecraft:moving_piston[facing={},type={}]",
            dir_name(dir),
            if sticky { "sticky" } else { "normal" }
        );
        self.resolve_state(&spec).unwrap_or(AIR_STATE)
    }

    fn piston_head_state(&self, dir: u8, sticky: bool) -> u32 {
        let spec = format!(
            "minecraft:piston_head[facing={},short=false,type={}]",
            dir_name(dir),
            if sticky { "sticky" } else { "normal" }
        );
        self.resolve_state(&spec).unwrap_or(AIR_STATE)
    }

    fn piston_base_state(&self, dir: u8, sticky: bool) -> u32 {
        let spec = format!(
            "minecraft:{}[extended=false,facing={}]",
            if sticky { "sticky_piston" } else { "piston" },
            dir_name(dir)
        );
        self.resolve_state(&spec).unwrap_or(AIR_STATE)
    }

    /// The moving-piston carrier tick: progress +0.5 per tick; the
    /// tick that sees progress >= 1.0 lands the carried state.
    fn tick_moving_pistons(&mut self) {
        let mut i = 0;
        while i < self.moving.len() {
            if self.moving[i].progress >= 2 {
                let mp = self.moving.remove(i);
                // BE validity guard: only land if the block is still a
                // moving_piston.
                if let Some((n, _)) = self.get_block(mp.pos.0, mp.pos.1, mp.pos.2) {
                    if n == "minecraft:moving_piston" {
                        self.set_block(mp.pos.0, mp.pos.1, mp.pos.2, mp.moved_state, true);
                    }
                }
            } else {
                self.moving[i].progress += 1;
                i += 1;
            }
        }
    }

    /// Force-settle now. Source
    /// pieces settle to air (the retracting base re-lands via its own
    // carrier at the same position).
    fn finalize_moving_piston_at(&mut self, pos: (i32, i32, i32)) {
        let Some(i) = self.moving.iter().position(|m| m.pos == pos) else {
            return;
        };
        let mp = self.moving.remove(i);
        if let Some((n, _)) = self.get_block(pos.0, pos.1, pos.2) {
            if n == "minecraft:moving_piston" {
                let state = if mp.is_source {
                    AIR_STATE
                } else {
                    mp.moved_state
                };
                self.set_block(pos.0, pos.1, pos.2, state, true);
            }
        }
    }

    fn tick_keep_alives(&mut self) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let conns: Vec<ConnId> = self.players.keys().copied().collect();
        for conn in conns {
            let Some(p) = self.players.get_mut(&conn) else {
                continue;
            };
            match p.pending_keep_alive {
                None if p.keep_alive_idle.elapsed() >= KEEP_ALIVE_INTERVAL => {
                    let body = now_ms.to_be_bytes().to_vec();
                    p.pending_keep_alive = Some((now_ms, Instant::now()));
                    self.send(conn, 0x2d, &body);
                }
                None => {}
                Some((_, sent)) if sent.elapsed() > KEEP_ALIVE_INTERVAL => {
                    eprintln!("[game] {}: keep-alive timeout", p.name);
                    self.send(conn, i32::MAX, &[]); // sentinel; Disconnect below
                    let _ = self
                        .outbounds
                        .get(&conn)
                        .map(|tx| tx.send(Outbound::Disconnect));
                    self.handle(Inbound::Left { conn });
                }
                Some(_) => {}
            }
        }
    }

    // pub(crate) for the inventory module's broadcast helpers.
    pub(crate) fn send(&mut self, conn: ConnId, id: i32, body: &[u8]) {
        // The reference suspends per-connection flushing for the duration
        // of the world tick and flushes once after it; buffering here and
        // draining at the tick end reproduces that pacing (frames leave
        // in identical order, once per tick).
        self.pending_frames.push((conn, id, body.to_vec()));
    }

    /// The once-per-tick connection flush: everything buffered this tick
    /// hits the writer channels in order.
    pub(crate) fn flush_connections(&mut self) {
        if self.flush_suspended || self.pending_frames.is_empty() {
            return;
        }
        let frames = std::mem::take(&mut self.pending_frames);
        for (conn, id, body) in frames {
            if let Some(tx) = self.outbounds.get(&conn) {
                let _ = tx.send(Outbound::Frame { id, body });
            }
        }
    }

    /// Vanilla's streaming choreography, verbatim from the capture: cache
    /// center, forgets, then one batch of entering chunks. All decisions
    /// happen up front (single &mut borrow); frames and loads are applied
    /// after the borrow ends.
    fn stream_if_moved(&mut self, conn: ConnId) {
        let Some(p) = self.players.get_mut(&conn) else {
            return;
        };
        let cx = p.x.floor().div_euclid(16.0) as i32;
        let cz = p.z.floor().div_euclid(16.0) as i32;
        if p.center == Some((cx, cz)) {
            return;
        }
        p.center = Some((cx, cz));

        let mut sends: Vec<(i32, Vec<u8>)> = Vec::new();
        let mut center = Vec::new();
        doppel_protocol::write_varint(&mut center, cx);
        doppel_protocol::write_varint(&mut center, cz);
        sends.push((0x60, center));

        let desired: std::collections::HashSet<(i32, i32)> = (-VIEW_RADIUS..=VIEW_RADIUS)
            .flat_map(move |dx| (-VIEW_RADIUS..=VIEW_RADIUS).map(move |dz| (cx + dx, cz + dz)))
            .collect();

        let leaving: Vec<(i32, i32)> = p.sent.difference(&desired).copied().collect();
        for (x, z) in leaving {
            let packed = ((x as i64 & 0x3ff_ffff) << 38) | ((z as i64 & 0x3ff_ffff) << 12);
            sends.push((0x26, packed.to_be_bytes().to_vec()));
            p.sent.remove(&(x, z));
        }

        let mut entering: Vec<(i32, i32)> = desired.difference(&p.sent).copied().collect();
        entering.sort_by_key(|(x, z)| (x - cx).abs() + (z - cz).abs());
        if entering.is_empty() {
            for (id, body) in sends {
                self.send(conn, id, &body);
            }
            return;
        }
        sends.push((0x0c, Vec::new()));

        // Load chunks (may lock the world), then emit everything.
        let (Some(world), blobs) = (self.world.clone(), self.blobs.clone()) else {
            for (id, body) in sends {
                self.send(conn, id, &body);
            }
            return;
        };
        let mut loaded: Vec<((i32, i32), Vec<u8>)> = Vec::new();
        for (x, z) in &entering {
            match self.load_chunk(&world, blobs.as_ref(), *x, *z) {
                Ok(chunk) => loaded.push(((*x, *z), chunk.wire.encode())),
                Err(e) => eprintln!("[game] chunk ({x},{z}) skipped: {e:#}"),
            }
        }
        for ((x, z), body) in &loaded {
            sends.push((0x2e, body.clone()));
            if let Some(p) = self.players.get_mut(&conn) {
                p.sent.insert((*x, *z));
            }
            self.viewers.entry((*x, *z)).or_default().push(conn);
        }
        let mut finished = Vec::new();
        doppel_protocol::write_varint(&mut finished, loaded.len() as i32);
        sends.push((0x0b, finished));

        for (id, body) in sends {
            self.send(conn, id, &body);
        }
    }

    /// Loads a chunk into the cache (validated against the capture
    /// when vanilla sent one; reference-free conversion otherwise).
    fn load_chunk(
        &mut self,
        world: &std::sync::Arc<std::sync::Mutex<crate::WorldState>>,
        blobs: Option<&std::sync::Arc<Blobs>>,
        cx: i32,
        cz: i32,
    ) -> anyhow::Result<&CachedChunk> {
        if let std::collections::btree_map::Entry::Vacant(slot) = self.chunks.entry((cx, cz)) {
            let mut w = world.lock().unwrap_or_else(|e| e.into_inner());
            let Some(anvil) = w.dir.chunk(cx, cz)? else {
                // No stored chunk: fall back to flat generation so
                // streaming extends past whatever the world has saved.
                let Some(flat) = &self.flat else {
                    anyhow::bail!("chunk not generated and flat fallback unavailable");
                };
                let wire = flat.generate(cx, cz);
                drop(w);
                slot.insert(CachedChunk { wire, version: 0 });
                return Ok(self.chunks.get(&(cx, cz)).expect("present: inserted above"));
            };
            let reference = blobs.and_then(|blobs| {
                blobs.play.iter().find_map(|(id, body)| {
                    if *id != 0x2e {
                        return None;
                    }
                    WireChunk::decode(body)
                        .ok()
                        .filter(|c| c.x == cx && c.z == cz)
                })
            });
            let wire = match reference {
                Some(reference) => {
                    w.boot.learn(&reference, &anvil);
                    doppel_world::anvil_to_wire::convert(&anvil, &reference, &w.boot)?
                }
                None => doppel_world::anvil_to_wire::convert_uncaptured(&anvil, &w.boot)?,
            };
            drop(w);
            slot.insert(CachedChunk { wire, version: 0 });
        }
        Ok(self.chunks.get(&(cx, cz)).expect("present: occupied above"))
    }
}

/// Writes a protocol VarLong (7 bits per byte, continuation bit 0x80).
fn write_u64_varlong(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Sets one cell (local pos, (x<<8)|(z<<4)|y) in a section's block-state
/// container, repacking storage when the palette grows. Returns false for
/// direct/global containers (unsupported mutation, rare above ground).
pub fn set_section_cell(chunk: &mut WireChunk, section: usize, idx: usize, state: u32) -> bool {
    use doppel_world::chunk_codec::Container;
    let Some(sec) = chunk.sections.get_mut(section) else {
        return false;
    };
    match &mut sec.block_states {
        Container::Single(v) => {
            if *v == state {
                return true;
            }
            let old = *v;
            let bits = 4usize;
            let per_long = 64 / bits;
            let mut longs = vec![0u64; 4096usize / per_long];
            for i in 0..4096 {
                let cell = if i == idx { 1u64 } else { 0 };
                longs[i / per_long] |= cell << ((i % per_long) * bits);
            }
            if state != 0 {
                sec.non_empty += 1;
            }
            sec.block_states = Container::Palette {
                bits: bits as u8,
                entries: vec![old, state],
                longs,
            };
            true
        }
        Container::Palette {
            entries,
            longs,
            bits,
        } => {
            let per_long = 64 / *bits as usize;
            let mask = (1u64 << *bits as usize) - 1;
            let mut cells = vec![0u32; 4096];
            for (i, cell) in cells.iter_mut().enumerate() {
                let long = longs.get(i / per_long).copied().unwrap_or(0);
                let shift = (i % per_long) * *bits as usize;
                *cell = entries
                    .get(((long >> shift) & mask) as usize)
                    .copied()
                    .unwrap_or(0);
            }
            let old = cells[idx];
            if old == state {
                return true;
            }
            let was_air = old == 0;
            let now_air = state == 0;
            sec.non_empty = match (was_air, now_air) {
                (true, false) => sec.non_empty.saturating_add(1),
                (false, true) => sec.non_empty.saturating_sub(1),
                _ => sec.non_empty,
            };
            cells[idx] = state;
            let mut new_entries: Vec<u32> = Vec::new();
            let mut index_of = std::collections::HashMap::new();
            for c in &cells {
                if !index_of.contains_key(c) {
                    index_of.insert(*c, new_entries.len() as u16);
                    new_entries.push(*c);
                }
            }
            let mut new_bits = 4usize;
            while (1usize << new_bits) < new_entries.len() {
                new_bits += 1;
            }
            if new_bits > 8 {
                return false;
            }
            let per = 64 / new_bits;
            let mut new_longs = vec![0u64; 4096usize.div_ceil(per)];
            for (i, c) in cells.iter().enumerate() {
                let v = u64::from(index_of[c]);
                new_longs[i / per] |= v << ((i % per) * new_bits);
            }
            *entries = new_entries;
            *bits = new_bits as u8;
            *longs = new_longs;
            true
        }
        Container::Global { .. } => false,
    }
}

/// Reads one cell from a section's block-state container.
pub fn get_section_cell(chunk: &WireChunk, section: usize, idx: usize) -> Option<u32> {
    use doppel_world::chunk_codec::Container;
    let sec = chunk.sections.get(section)?;
    match &sec.block_states {
        Container::Single(v) => Some(*v),
        Container::Palette {
            entries,
            longs,
            bits,
        } => {
            let per_long = 64 / *bits as usize;
            let mask = (1u64 << *bits as usize) - 1;
            let long = longs.get(idx / per_long).copied().unwrap_or(0);
            let shift = (idx % per_long) * *bits as usize;
            entries.get(((long >> shift) & mask) as usize).copied()
        }
        Container::Global { longs, bits } => {
            let per_long = 64 / *bits as usize;
            let mask = (1u64 << *bits as usize) - 1;
            let long = longs.get(idx / per_long).copied().unwrap_or(0);
            let shift = (idx % per_long) * *bits as usize;
            Some(((long >> shift) & mask) as u32)
        }
    }
}

/// Test-support accessors for the out-of-file piston tests: the lifecycle
/// tests drive the simulation directly through the same Inbound events
/// the network path uses.
#[cfg(test)]
impl Game {
    pub(crate) fn seed_chunk_for_test(&mut self, cx: i32, cz: i32, wire: WireChunk) {
        self.chunks
            .insert((cx, cz), CachedChunk { wire, version: 0 });
    }

    pub(crate) fn join_viewer_for_test(
        &mut self,
        conn: ConnId,
        chunks: &[(i32, i32)],
        tx: Sender<Outbound>,
    ) {
        self.outbounds.insert(conn, tx);
        let entity_id = self.next_entity_id;
        self.next_entity_id += 1;
        self.players.insert(
            conn,
            Player {
                name: "bot".into(),
                x: 0.0,
                y: 0.0,
                z: 0.0,
                yaw: 0.0,
                pitch: 0.0,
                center: Some((0, 0)),
                sent: chunks.iter().copied().collect(),
                teleport_id: 1,
                pending_keep_alive: None,
                keep_alive_idle: Instant::now(),
                inv: Default::default(),
                menu: Default::default(),
                entity_id,
                dig: Default::default(),
            },
        );
        for c in chunks {
            self.viewers.entry(*c).or_default().push(conn);
        }
    }

    /// One game tick including the internal flush, plus draining any
    /// leftover dirty sections (kept for call-shape parity with run()).
    pub(crate) fn tick_once_for_test(&mut self) {
        self.game_tick();
    }

    /// The internal tick counter (stepped `tick step` batches advance it
    /// past the harness's own tick count).
    pub(crate) fn tick_counter_for_test(&self) -> u64 {
        self.tick
    }

    /// Puts chunks in the player's streamed set without the streaming
    /// machinery (the tracker's re-pairing tests; the harness runs
    /// without a world store, so streaming cannot re-enter on its own).
    pub(crate) fn grant_chunks_for_test(&mut self, conn: ConnId, chunks: &[(i32, i32)]) {
        if let Some(p) = self.players.get_mut(&conn) {
            for c in chunks {
                p.sent.insert(*c);
            }
        }
    }

    pub(crate) fn state_label_for_test(&self, state: u32) -> String {
        match self.registry.as_ref().and_then(|r| r.state_of(state)) {
            Some((n, p)) => format!("{n}[{p}]"),
            None => format!("state {state}"),
        }
    }

    pub(crate) fn block_label_for_test(&self, x: i32, y: i32, z: i32) -> String {
        match self.get_block(x, y, z) {
            Some((n, p)) => format!("{n}[{p}]"),
            None => "none[]".to_string(),
        }
    }

    pub(crate) fn registry_for_test(&self) -> bool {
        self.registry.is_some()
    }

    #[cfg(test)]
    pub(crate) fn registry_snapshot_for_test(&self) -> doppel_world::registry::BlockRegistry {
        self.registry.clone().expect("registry loaded")
    }
}

/// Section-storage cell index, vanilla's YZX layout (`y<<8 | z<<4 | x`):
/// the order chunk sections pack their paletted longs on the wire and in
/// Anvil.
fn local_yzx(x: i32, y: i32, z: i32) -> usize {
    (((y.rem_euclid(16) as u64) << 8) | ((z.rem_euclid(16) as u64) << 4) | x.rem_euclid(16) as u64)
        as usize
}

/// Section-update wire localPos, XZY layout (`x<<8 | z<<4 | y`): the
/// packed position inside section_blocks_update (0x56) entries.
fn local_xzy(x: i32, y: i32, z: i32) -> u64 {
    ((x.rem_euclid(16) as u64) << 8) | ((z.rem_euclid(16) as u64) << 4) | y.rem_euclid(16) as u64
}

/// Reads the facing= direction from a props string (default north).
fn prop_dir(props: &str) -> &str {
    for pair in props.split(',') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == "facing" {
                return v;
            }
        }
    }
    "north"
}

/// Reads one property value from a props string.
fn prop_value<'a>(props: &'a str, name: &str) -> &'a str {
    for pair in props.split(',') {
        if let Some((k, v)) = pair.split_once('=') {
            if k == name {
                return v;
            }
        }
    }
    ""
}

// --- Direction model (vanilla `Direction` 3D data values) ---

pub const DIR_DOWN: u8 = 0;
pub const DIR_UP: u8 = 1;
pub const DIR_NORTH: u8 = 2;
pub const DIR_SOUTH: u8 = 3;
pub const DIR_WEST: u8 = 4;
pub const DIR_EAST: u8 = 5;

/// The unit step of a direction id.
fn dir_step(id: u8) -> (i32, i32, i32) {
    match id {
        DIR_DOWN => (0, -1, 0),
        DIR_UP => (0, 1, 0),
        DIR_NORTH => (0, 0, -1),
        DIR_SOUTH => (0, 0, 1),
        DIR_WEST => (-1, 0, 0),
        _ => (1, 0, 0),
    }
}

fn dir_opposite(id: u8) -> u8 {
    match id {
        DIR_DOWN => DIR_UP,
        DIR_UP => DIR_DOWN,
        DIR_NORTH => DIR_SOUTH,
        DIR_SOUTH => DIR_NORTH,
        DIR_WEST => DIR_EAST,
        _ => DIR_WEST,
    }
}

fn dir_name(id: u8) -> &'static str {
    match id {
        DIR_DOWN => "down",
        DIR_UP => "up",
        DIR_NORTH => "north",
        DIR_SOUTH => "south",
        DIR_WEST => "west",
        _ => "east",
    }
}

fn dir_id(name: &str) -> u8 {
    match name {
        "down" => DIR_DOWN,
        "up" => DIR_UP,
        "south" => DIR_SOUTH,
        "west" => DIR_WEST,
        "east" => DIR_EAST,
        _ => DIR_NORTH,
    }
}

/// `pos + step * n`.
fn offset(pos: (i32, i32, i32), step: (i32, i32, i32), n: i32) -> (i32, i32, i32) {
    (pos.0 + step.0 * n, pos.1 + step.1 * n, pos.2 + step.2 * n)
}

// --- Push reactions (vanilla `PushReaction`, default PUSH_PULL) ---

#[derive(Clone, Copy, PartialEq, Eq)]
enum PushReaction {
    PushPull,
    Push,
    Popped,
    Immoveable,
    /// `IGNORE_ENTITY`: pushes through like PUSH_PULL (no entity model
    /// yet, kept for table completeness).
    #[allow(dead_code)]
    IgnoredEntity,
}

/// The global state id of air.
pub(crate) const AIR_STATE: u32 = 0;
/// The maximum push depth (12).
const MAX_PUSH_DEPTH: usize = 12;
/// 24 sections of 16 (-64..319), matching the wire chunk layout.
pub(crate) const WORLD_MIN_Y: i32 = -64;
pub(crate) const WORLD_MAX_Y: i32 = 319;

/// Blocks with destroy speed -1 or IMMOVEABLE registration (the set the
/// circuits can encounter; the full table is NOT YET).
fn push_reaction(name: &str) -> PushReaction {
    if is_immoveable(name) {
        return PushReaction::Immoveable;
    }
    if name.ends_with("glazed_terracotta") {
        return PushReaction::Push;
    }
    if matches!(
        name,
        "minecraft:torch"
            | "minecraft:wall_torch"
            | "minecraft:soul_torch"
            | "minecraft:soul_wall_torch"
            | "minecraft:redstone_torch"
            | "minecraft:redstone_wall_torch"
            | "minecraft:lever"
            | "minecraft:redstone_wire"
            | "minecraft:repeater"
            | "minecraft:comparator"
            | "minecraft:rail"
            | "minecraft:powered_rail"
            | "minecraft:detector_rail"
            | "minecraft:activator_rail"
            | "minecraft:tripwire"
    ) {
        return PushReaction::Popped;
    }
    PushReaction::PushPull
}

fn is_immoveable(name: &str) -> bool {
    matches!(
        name,
        "minecraft:obsidian"
            | "minecraft:crying_obsidian"
            | "minecraft:bedrock"
            | "minecraft:piston_head"
            | "minecraft:moving_piston"
            | "minecraft:barrier"
            | "minecraft:command_block"
            | "minecraft:chain_command_block"
            | "minecraft:repeating_command_block"
            | "minecraft:end_portal_frame"
            | "minecraft:enchanting_table"
    )
}

/// `getDestroySpeed == -1.0` registrations.
fn unbreakable(name: &str) -> bool {
    matches!(
        name,
        "minecraft:bedrock" | "minecraft:barrier" | "minecraft:moving_piston"
    )
}

fn is_sticky(name: &str) -> bool {
    name == "minecraft:slime_block" || name == "minecraft:honey_block"
}

fn can_stick_to_each_other(a: &str, b: &str) -> bool {
    // Honey does not bond to slime.
    if (a == "minecraft:honey_block" && b == "minecraft:slime_block")
        || (a == "minecraft:slime_block" && b == "minecraft:honey_block")
    {
        return false;
    }
    is_sticky(a) || is_sticky(b)
}

impl wire::BlockView for Game {
    fn block_at(&self, x: i32, y: i32, z: i32) -> Option<(String, String)> {
        self.get_block(x, y, z)
    }
}

impl wire::WireHost for Game {
    fn set_wire_state(&mut self, x: i32, y: i32, z: i32, state: u32) {
        if std::env::var("WIRE_TRACE").is_ok() {
            eprintln!("[trace] set_wire ({x},{y},{z}) state={state}");
        }
        self.set_block(x, y, z, state, false);
    }
    fn resolve_wire_state(&self, conn: &wire::Connections, power: i32) -> Option<u32> {
        let spec = format!("minecraft:redstone_wire[{}]", conn.props_string(power));
        let reg = self.registry.as_ref()?;
        let (name, props) = doppel_world::registry::BlockRegistry::split_state(&spec);
        reg.state_id(name, props)
    }
    fn dispatch_neighbor_changed(&mut self, x: i32, y: i32, z: i32) {
        self.update_block(x, y, z);
    }
}
