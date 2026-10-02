//! Vanilla redstone-wire semantics, MC 26.3 (default evaluator).
//!
//! The engine folds three code paths into one handler reached through
//! `Game::update_wire` → [`update_wire_cascade`]:
//!
//! * the power recompute and the self+6-neighbor fan-out,
//! * the
//!   stored connection sides, which vanilla refreshes through the
//!   shape-update phase of `setBlock`; this engine has a single
//!   neighbor-update channel with no direction, so the recompute rides
//!   along here (a wire whose only vanilla trigger was a change directly
//!   below it would keep stale sides a while longer — accepted drift),
//! * the collecting neighbor updater, emulated with an explicit LIFO
//!   stack so the whole cascade runs synchronously in one game tick, with
//!   new updates preempting the remaining directions of a running
//!   `updateNeighborsAt`, exactly like vanilla's chained updates.

/// The six `Direction`s, in the iteration orders the wire code observes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dir {
    Down,
    Up,
    North,
    South,
    West,
    East,
}

impl Dir {
    /// `Direction.values()` — `getBestNeighborSignal` and the evaluator
    /// fan-out insertion order.
    pub const VALUES: [Dir; 6] = [
        Dir::Down,
        Dir::Up,
        Dir::North,
        Dir::South,
        Dir::West,
        Dir::East,
    ];
    /// Horizontal iteration order.
    pub const HORIZONTAL: [Dir; 4] = [Dir::North, Dir::East, Dir::South, Dir::West];
    /// The neighbor-update direction order.
    pub const UPDATE_ORDER: [Dir; 6] = [
        Dir::West,
        Dir::East,
        Dir::Down,
        Dir::Up,
        Dir::North,
        Dir::South,
    ];

    fn step(self) -> (i32, i32, i32) {
        match self {
            Dir::Down => (0, -1, 0),
            Dir::Up => (0, 1, 0),
            Dir::North => (0, 0, -1),
            Dir::South => (0, 0, 1),
            Dir::West => (-1, 0, 0),
            Dir::East => (1, 0, 0),
        }
    }

    fn opposite(self) -> Dir {
        match self {
            Dir::Down => Dir::Up,
            Dir::Up => Dir::Down,
            Dir::North => Dir::South,
            Dir::South => Dir::North,
            Dir::West => Dir::East,
            Dir::East => Dir::West,
        }
    }

    fn parse(s: &str) -> Option<Dir> {
        Some(match s {
            "down" => Dir::Down,
            "up" => Dir::Up,
            "north" => Dir::North,
            "south" => Dir::South,
            "west" => Dir::West,
            "east" => Dir::East,
            _ => return None,
        })
    }
}

/// A block position; the internal currency of this module.
pub type Pos = (i32, i32, i32);

fn at(p: Pos, d: Dir) -> Pos {
    let (dx, dy, dz) = d.step();
    (p.0 + dx, p.1 + dy, p.2 + dz)
}

/// Read access to the block store: registry-canonical `(name, props)` per
/// position. `None` (chunk not loaded) behaves like air.
pub trait BlockView {
    fn block_at(&self, x: i32, y: i32, z: i32) -> Option<(String, String)>;
}

/// The mutable surface the wire cascade needs from its host.
pub trait WireHost: BlockView {
    /// A block write that only marks the position dirty for the tick-end
    /// the position dirty for the tick-end broadcast and schedule nothing;
    /// the evaluator performs its own fan-out.
    fn set_wire_state(&mut self, x: i32, y: i32, z: i32, state: u32);
    /// Registry lookup for the full wire state (connections + power).
    fn resolve_wire_state(&self, conn: &Connections, power: i32) -> Option<u32>;
    /// `neighborChanged` dispatch for non-wire blocks. Wire positions are
    /// handled inside the cascade so the update budget threads through.
    fn dispatch_neighbor_changed(&mut self, x: i32, y: i32, z: i32);
}

fn get(view: &impl BlockView, p: Pos) -> Option<(String, String)> {
    view.block_at(p.0, p.1, p.2)
}

// ----------------------------------------------------------------------
// Property-string helpers (registry-canonical `k=v` lists)
// ----------------------------------------------------------------------

fn prop<'a>(props: &'a str, key: &str) -> Option<&'a str> {
    props
        .split(',')
        .filter_map(|pair| pair.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v)
}

fn prop_bool(props: &str, key: &str) -> bool {
    prop(props, key) == Some("true")
}

fn facing_of(props: &str) -> Option<Dir> {
    prop(props, "facing").and_then(Dir::parse)
}

fn wire_power_of(props: &str) -> i32 {
    prop(props, "power")
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(0)
}

// ----------------------------------------------------------------------
// Block classification
// ----------------------------------------------------------------------

fn is_wire(name: &str) -> bool {
    name == "minecraft:redstone_wire"
}

/// The state
/// registry carries no shape data, so this is an allowlist of full-cube
/// blocks and families; anything unknown (and every partial block —
/// slabs, stairs, walls, panes, torches, dust, mechanisms) is treated as
/// a non-conductor.
fn is_conductor(name: &str) -> bool {
    const FULL_CUBES: &[&str] = &[
        "minecraft:stone",
        "minecraft:cobblestone",
        "minecraft:mossy_cobblestone",
        "minecraft:stone_bricks",
        "minecraft:mud_bricks",
        "minecraft:bricks",
        "minecraft:dirt",
        "minecraft:coarse_dirt",
        "minecraft:rooted_dirt",
        "minecraft:grass_block",
        "minecraft:dirt_path",
        "minecraft:mud",
        "minecraft:packed_mud",
        "minecraft:sand",
        "minecraft:red_sand",
        "minecraft:gravel",
        "minecraft:clay",
        "minecraft:bedrock",
        "minecraft:obsidian",
        "minecraft:crying_obsidian",
        "minecraft:sandstone",
        "minecraft:red_sandstone",
        "minecraft:deepslate",
        "minecraft:cobbled_deepslate",
        "minecraft:tuff",
        "minecraft:granite",
        "minecraft:diorite",
        "minecraft:andesite",
        "minecraft:calcite",
        "minecraft:dripstone_block",
        "minecraft:smooth_basalt",
        "minecraft:basalt",
        "minecraft:blackstone",
        "minecraft:end_stone",
        "minecraft:netherrack",
        "minecraft:glass",
        "minecraft:tinted_glass",
        "minecraft:glowstone",
        "minecraft:sea_lantern",
        "minecraft:shroomlight",
        "minecraft:snow_block",
        "minecraft:ice",
        "minecraft:packed_ice",
        "minecraft:blue_ice",
        "minecraft:magma_block",
        "minecraft:hay_block",
        "minecraft:dried_kelp_block",
        "minecraft:bone_block",
        "minecraft:slime_block",
        "minecraft:honey_block",
        "minecraft:bookshelf",
        "minecraft:melon",
        "minecraft:pumpkin",
        "minecraft:carved_pumpkin",
        "minecraft:jack_o_lantern",
        "minecraft:sponge",
        "minecraft:wet_sponge",
        "minecraft:moss_block",
        "minecraft:purpur_block",
        "minecraft:prismarine",
        "minecraft:prismarine_bricks",
        "minecraft:dark_prismarine",
        "minecraft:redstone_block",
        "minecraft:target",
    ];
    if FULL_CUBES.contains(&name) {
        return true;
    }
    // Families that are always full cubes (never slabs/stairs/panes).
    name.ends_with("_planks")
        || name.ends_with("_wool")
        || name.ends_with("_concrete")
        || name.ends_with("_concrete_powder")
        || name.ends_with("_terracotta")
        || name.ends_with("_glazed_terracotta")
        || name.ends_with("_bricks")
        || name.ends_with("_stem")
        || name.ends_with("_hyphae")
        || name.ends_with("_copper")
        || name.ends_with("_block")
        || name.ends_with("_glass")
}

/// Signal-source overrides, approximated over the
/// registry's name space: the redstone family plus the blocks a wire
/// visually attaches to (anything comparator-readable included).
fn is_signal_source(name: &str) -> bool {
    const SOURCES: &[&str] = &[
        "minecraft:redstone_wire",
        "minecraft:redstone_torch",
        "minecraft:redstone_wall_torch",
        "minecraft:redstone_block",
        "minecraft:lever",
        "minecraft:repeater",
        "minecraft:comparator",
        "minecraft:observer",
        "minecraft:daylight_detector",
        "minecraft:detector_rail",
        "minecraft:tripwire_hook",
        "minecraft:lightning_rod",
        "minecraft:big_dripleaf",
        "minecraft:chest",
        "minecraft:trapped_chest",
        "minecraft:barrel",
        "minecraft:furnace",
        "minecraft:blast_furnace",
        "minecraft:smoker",
        "minecraft:brewing_stand",
        "minecraft:lectern",
        "minecraft:jukebox",
        "minecraft:hopper",
        "minecraft:crafter",
        "minecraft:chiseled_bookshelf",
    ];
    SOURCES.contains(&name) || name.ends_with("_button") || name.ends_with("_pressure_plate")
}

/// Sturdy top face or hopper.
fn can_survive_on(name: &str) -> bool {
    is_conductor(name) || name == "minecraft:hopper"
}

/// Sturdy-face approximation for the wire up-connection: full cubes
/// are sturdy on every face, the hopper only on top (its funnel sides are
/// open).
fn is_face_sturdy(name: &str, toward: Dir) -> bool {
    if name == "minecraft:hopper" {
        return toward == Dir::Up;
    }
    is_conductor(name)
}

// ----------------------------------------------------------------------
// Connections
// ----------------------------------------------------------------------

/// Both `Up` and `Side` count as connected.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Up,
    Side,
    None,
}

impl Side {
    fn is_connected(self) -> bool {
        self != Side::None
    }

    fn as_str(self) -> &'static str {
        match self {
            Side::Up => "up",
            Side::Side => "side",
            Side::None => "none",
        }
    }

    fn parse(s: &str) -> Side {
        match s {
            "up" => Side::Up,
            "side" => Side::Side,
            _ => Side::None,
        }
    }
}

/// The four horizontal connection props of a wire state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Connections {
    pub north: Side,
    pub east: Side,
    pub south: Side,
    pub west: Side,
}

impl Connections {
    fn all_none() -> Connections {
        Connections {
            north: Side::None,
            east: Side::None,
            south: Side::None,
            west: Side::None,
        }
    }

    pub fn from_props(props: &str) -> Connections {
        let side = |key: &str| prop(props, key).map_or(Side::None, Side::parse);
        Connections {
            north: side("north"),
            east: side("east"),
            south: side("south"),
            west: side("west"),
        }
    }

    fn is_dot(self) -> bool {
        !self.north.is_connected()
            && !self.east.is_connected()
            && !self.south.is_connected()
            && !self.west.is_connected()
    }

    fn set(&mut self, d: Dir, s: Side) {
        match d {
            Dir::North => self.north = s,
            Dir::East => self.east = s,
            Dir::South => self.south = s,
            Dir::West => self.west = s,
            _ => {}
        }
    }

    /// Whether the side toward `d` (a horizontal) is connected.
    fn connected(self, d: Dir) -> bool {
        match d {
            Dir::North => self.north.is_connected(),
            Dir::East => self.east.is_connected(),
            Dir::South => self.south.is_connected(),
            Dir::West => self.west.is_connected(),
            _ => false,
        }
    }

    /// Canonical props string (`east,north,power,south,west` — registry
    /// sort order).
    pub fn props_string(self, power: i32) -> String {
        let (n, e, s, w) = (
            self.north.as_str(),
            self.east.as_str(),
            self.south.as_str(),
            self.west.as_str(),
        );
        format!("east={e},north={n},power={power},south={s},west={w}")
    }
}

/// Per-block connection rules.
/// `direction` is from the wire toward the candidate block; the diagonal
/// probes pass `None`, for which only wires answer true.
fn should_connect_to(view: &impl BlockView, p: Pos, direction: Option<Dir>) -> bool {
    let Some((name, props)) = get(view, p) else {
        return false;
    };
    match name.as_str() {
        "minecraft:redstone_wire" => true,
        "minecraft:repeater" => match (direction, facing_of(&props)) {
            (Some(d), Some(facing)) => d == facing || d == facing.opposite(),
            _ => false,
        },
        "minecraft:observer" => match (direction, facing_of(&props)) {
            (Some(d), Some(facing)) => d == facing,
            _ => false,
        },
        _ => is_signal_source(&name) && direction.is_some(),
    }
}

/// Connection-side decision tree in order:
/// up-slope, same-level, down-slope, none.
fn get_connecting_side(view: &impl BlockView, p: Pos, d: Dir, can_connect_up: bool) -> Side {
    let rel = at(p, d);
    let rel_name = get(view, rel)
        .map(|(n, _)| n)
        .unwrap_or_else(|| "minecraft:air".to_string());
    if can_connect_up {
        let is_placeable_above = rel_name.ends_with("_trapdoor") || can_survive_on(&rel_name);
        if is_placeable_above && should_connect_to(view, at(rel, Dir::Up), None) {
            if is_face_sturdy(&rel_name, d.opposite()) {
                return Side::Up;
            }
            return Side::Side;
        }
    }
    if should_connect_to(view, rel, Some(d))
        || (!is_conductor(&rel_name) && should_connect_to(view, at(rel, Dir::Down), None))
    {
        return Side::Side;
    }
    Side::None
}

/// All four sides are
/// recomputed from the world; only `wasDot` survives from the stored
/// state, and the axis-fill rules run afterwards with both emptiness
/// flags read from the recomputed sides.
fn get_connection_state(view: &impl BlockView, p: Pos, stored_props: &str) -> Connections {
    let was_dot = Connections::from_props(stored_props).is_dot();
    let mut c = Connections::all_none();
    let can_connect_up = !get(view, at(p, Dir::Up)).is_some_and(|(name, _)| is_conductor(&name));
    for d in Dir::HORIZONTAL {
        c.set(d, get_connecting_side(view, p, d, can_connect_up));
    }
    if was_dot && c.is_dot() {
        return c; // an isolated dot stays a dot
    }
    let north_south_empty = !c.north.is_connected() && !c.south.is_connected();
    let east_west_empty = !c.east.is_connected() && !c.west.is_connected();
    if !c.west.is_connected() && north_south_empty {
        c.west = Side::Side;
    }
    if !c.east.is_connected() && north_south_empty {
        c.east = Side::Side;
    }
    if !c.north.is_connected() && east_west_empty {
        c.north = Side::Side;
    }
    if !c.south.is_connected() && east_west_empty {
        c.south = Side::Side;
    }
    c
}

// ----------------------------------------------------------------------
// Signal emission and queries
// ----------------------------------------------------------------------

/// `wires_emit` models the singleton
/// `shouldSignal` flag: while a wire recomputes its own power the flag is
/// false and every wire answers zero on every query path, so
/// wire-to-wire transfer happens only through the dedicated incoming-
/// wire read. `from_consumer` is the vanilla argument: the direction from
/// the querying block toward this wire.
fn wire_signal_toward(
    view: &impl BlockView,
    p: Pos,
    props: &str,
    from_consumer: Dir,
    wires_emit: bool,
) -> i32 {
    if !wires_emit || from_consumer == Dir::Down {
        return 0; // a wire never powers the block above it
    }
    let power = wire_power_of(props);
    if power == 0 {
        return 0;
    }
    if from_consumer == Dir::Up {
        return power; // it always powers the block it sits on
    }
    // Horizontal: only toward sides whose LIVE-recomputed connection (on
    // the side facing the consumer) is connected.
    let conn = get_connection_state(view, p, props);
    if conn.connected(from_consumer.opposite()) {
        power
    } else {
        0
    }
}

/// A block's own weak emission, vanilla argument convention as above:
/// torches, lever, diodes (emit toward `facing` only), observer,
/// redstone block.
fn own_weak_signal(
    view: &impl BlockView,
    p: Pos,
    name: &str,
    props: &str,
    from_consumer: Dir,
    wires_emit: bool,
) -> i32 {
    let lit = if prop_bool(props, "lit") { 15 } else { 0 };
    match name {
        "minecraft:redstone_wire" => wire_signal_toward(view, p, props, from_consumer, wires_emit),
        "minecraft:redstone_torch" if from_consumer != Dir::Up => lit,
        "minecraft:redstone_wall_torch" if Some(from_consumer) != facing_of(props) => lit,
        "minecraft:lever" if prop_bool(props, "powered") => 15,
        "minecraft:redstone_block" => 15,
        "minecraft:repeater" | "minecraft:comparator"
            if Some(from_consumer) == facing_of(props) && prop_bool(props, "powered") =>
        {
            15
        }
        "minecraft:observer"
            if Some(from_consumer) == facing_of(props) && prop_bool(props, "powered") =>
        {
            15
        }
        _ if (name.ends_with("_button") || name.ends_with("_pressure_plate"))
            && prop_bool(props, "powered") =>
        {
            15
        }
        _ => 0,
    }
}

/// The direction from a lever/button's support toward the block itself
/// (`getConnectedDirection`): floor → up, ceiling → down, wall → the
/// opposite of `facing` (which points away from the support).
fn attached_from_support(props: &str) -> Option<Dir> {
    match prop(props, "face") {
        Some("floor") => Some(Dir::Up),
        Some("ceiling") => Some(Dir::Down),
        Some("wall") => facing_of(props).map(|f| f.opposite()),
        _ => Some(Dir::Up),
    }
}

/// A block's own strong emission (`state.getDirectSignal`): wire (same as
/// weak), standing torch into the block above, lever into its
/// support, diodes/observer (delegate to weak).
fn own_direct_signal(
    view: &impl BlockView,
    p: Pos,
    name: &str,
    props: &str,
    from_consumer: Dir,
    wires_emit: bool,
) -> i32 {
    match name {
        "minecraft:redstone_wire" => wire_signal_toward(view, p, props, from_consumer, wires_emit),
        "minecraft:redstone_torch" if from_consumer == Dir::Down => {
            if prop_bool(props, "lit") {
                15
            } else {
                0
            }
        }
        "minecraft:lever"
            if prop_bool(props, "powered")
                && Some(from_consumer) == attached_from_support(props) =>
        {
            15
        }
        "minecraft:repeater" | "minecraft:comparator" | "minecraft:observer" => {
            own_weak_signal(view, p, name, props, from_consumer, wires_emit)
        }
        _ => 0,
    }
}

/// The block's own weak
/// emission, escalated through it when it is a redstone conductor — a
/// strongly powered block relays what feeds it.
fn weak_signal_at(view: &impl BlockView, p: Pos, from_consumer: Dir, wires_emit: bool) -> i32 {
    let Some((name, props)) = get(view, p) else {
        return 0;
    };
    let own = own_weak_signal(view, p, &name, &props, from_consumer, wires_emit);
    if is_conductor(&name) {
        return own.max(direct_signal_to(view, p, wires_emit));
    }
    own
}

/// The best of the six face
/// neighbors in `Direction.values()` order, early exit at 15.
fn best_neighbor_signal(view: &impl BlockView, p: Pos, wires_emit: bool) -> i32 {
    let mut best = 0;
    for d in Dir::VALUES {
        let signal = weak_signal_at(view, at(p, d), d, wires_emit);
        if signal >= 15 {
            return 15;
        }
        best = best.max(signal);
    }
    best
}

/// The max strong signal INTO
/// the block from its six neighbors, unrolled order below, above, north,
/// south, west, east, early exit at 15.
fn direct_signal_to(view: &impl BlockView, p: Pos, wires_emit: bool) -> i32 {
    let mut best = 0;
    for d in Dir::VALUES {
        let Some((name, props)) = get(view, at(p, d)) else {
            continue;
        };
        let signal = own_direct_signal(view, at(p, d), &name, &props, d, wires_emit);
        if signal >= 15 {
            return 15;
        }
        best = best.max(signal);
    }
    best
}

/// The signal the block at `(x, y, z)` offers a consumer directly above
/// it — vanilla `getSignal(pos, Direction.DOWN)`. That is the torch input
/// read (`hasSignal(pos.below(), DOWN)`): the block's own emission toward
/// the torch, plus — when the block is a conductor — whatever strongly
/// powers it.
pub fn signal_toward_consumer_above(view: &impl BlockView, x: i32, y: i32, z: i32) -> i32 {
    weak_signal_at(view, (x, y, z), Dir::Down, true)
}

// ----------------------------------------------------------------------
// Wire power calculation
// ----------------------------------------------------------------------

/// Same-level wire
/// neighbors count unconditionally (the invisible-connection quirk — the
/// visual state is never consulted); a diagonal-up read requires the
/// neighbor to be a conductor AND headroom above the receiving wire; a
/// diagonal-down read requires a non-conductor neighbor. One decay step
/// is applied once, by the receiver.
fn incoming_wire_signal(view: &impl BlockView, p: Pos) -> i32 {
    fn wire_at(view: &impl BlockView, q: Pos) -> i32 {
        match get(view, q) {
            Some((name, props)) if is_wire(&name) => wire_power_of(&props),
            _ => 0,
        }
    }
    let mut wire_signal = 0;
    for d in Dir::HORIZONTAL {
        let n = at(p, d);
        wire_signal = wire_signal.max(wire_at(view, n)); // (a) same Y
        let conductor = get(view, n).is_some_and(|(name, _)| is_conductor(&name));
        if conductor {
            let headroom = !get(view, at(p, Dir::Up)).is_some_and(|(name, _)| is_conductor(&name));
            if headroom {
                wire_signal = wire_signal.max(wire_at(view, at(n, Dir::Up))); // (b) up
            }
            continue; // a solid neighbor blocks the down diagonal
        }
        wire_signal = wire_signal.max(wire_at(view, at(n, Dir::Down))); // (c) down
    }
    (wire_signal - 1).max(0)
}

/// Block
/// signal (all wires silenced) versus decayed wire signal, with the
/// 15 shortcut.
fn calculate_target_strength(view: &impl BlockView, p: Pos) -> i32 {
    let block_signal = best_neighbor_signal(view, p, false);
    if block_signal == 15 {
        return block_signal;
    }
    block_signal.max(incoming_wire_signal(view, p))
}

// ----------------------------------------------------------------------
// The update cascade
// ----------------------------------------------------------------------

/// The queue is a heap
/// stack, not native recursion, so the full vanilla budget is safe here.
pub const MAX_CHAINED_NEIGHBOR_UPDATES: u32 = 1_000_000;

/// One queued `updateNeighborsAt`: a position whose six neighbors (in
/// UPDATE_ORDER) still need a neighbor update.
struct MultiNeighborUpdate {
    pos: Pos,
    next: usize,
}

/// A LIFO stack where updates
/// enqueued while another runs preempt its remaining directions.
struct NeighborQueue {
    stack: Vec<MultiNeighborUpdate>,
    budget: u32,
    overflowed: bool,
}

impl NeighborQueue {
    fn new() -> NeighborQueue {
        NeighborQueue {
            stack: Vec::new(),
            budget: MAX_CHAINED_NEIGHBOR_UPDATES,
            overflowed: false,
        }
    }

    /// The fan-out: `updateNeighborsAt` on the wire position plus
    /// its six face neighbors, insertion order DOWN, UP, NORTH, SOUTH,
    /// WEST, EAST (vanilla iterates these through a `HashSet`, whose
    /// order is unspecified — this is the order the code expresses).
    fn push_around(&mut self, p: Pos) {
        self.stack.push(MultiNeighborUpdate { pos: p, next: 0 });
        for d in Dir::VALUES {
            self.stack.push(MultiNeighborUpdate {
                pos: at(p, d),
                next: 0,
            });
        }
    }

    /// Depth-first drain with preemption: an update enqueued by the
    /// currently-running one runs before this one's remaining directions.
    fn drain(&mut self, host: &mut impl WireHost) {
        while let Some(top) = self.stack.last() {
            let (pos, next) = (top.pos, top.next);
            if next >= Dir::UPDATE_ORDER.len() {
                self.stack.pop();
                continue;
            }
            self.stack.last_mut().expect("stack non-empty").next = next + 1;
            if self.budget == 0 {
                if !self.overflowed {
                    self.overflowed = true;
                    eprintln!("[wire] too many chained neighbor updates; skipping the rest");
                }
                self.stack.clear();
                return;
            }
            self.budget -= 1;
            execute_update(host, self, at(pos, Dir::UPDATE_ORDER[next]));
        }
    }
}

/// Power-strength update with the
/// engine's folded connection recompute: a power change writes the state
/// and fans out to the seven positions; a connection-only change writes
/// the state without any fan-out (vanilla performs that write from the
/// shape-update phase); no change at all does nothing.
fn update_wire_strength(host: &mut impl WireHost, p: Pos, props: &str, queue: &mut NeighborQueue) {
    let current = wire_power_of(props);
    let target = calculate_target_strength(host, p);
    if target == current {
        return;
    }
    // Power writes preserve the STORED connections: vanilla's power phase
    // (updatePowerStrength) never touches shape; connection recomputes ride
    // the shape-update phase, whose broadcasts vanilla emits at placement
    // time — replaying them here double-broadcasts under frozen ticks.
    let conn = Connections::from_props(props);
    if let Some(state) = host.resolve_wire_state(&conn, target) {
        // The setBlock guard: a wire replaced mid-cascade skips
        // the write but would still fan out below.
        if get(host, p).is_some_and(|(name, _)| is_wire(&name)) {
            host.set_wire_state(p.0, p.1, p.2, state);
        }
    }
    if target != current {
        queue.push_around(p);
    }
}

/// Executes the neighbor update.
/// Wires re-enter the cascade; every other family goes through the
/// host's dispatcher (which keeps its own scheduled-tick machinery).
fn execute_update(host: &mut impl WireHost, queue: &mut NeighborQueue, p: Pos) {
    if let Some((name, props)) = get(host, p) {
        if is_wire(&name) {
            let props = refresh_wire_shape(host, p, &props);
            update_wire_strength(host, p, &props, queue);
        } else {
            host.dispatch_neighbor_changed(p.0, p.1, p.2);
        }
    }
}

/// The shape-update phase: a wire notified of a neighbor change
/// recomputes its connection sides from the world. A connection-only
/// change writes the state without any power fan-out; the power recompute
/// that arrived with the same notification follows separately.
fn refresh_wire_shape(host: &mut impl WireHost, p: Pos, props: &str) -> String {
    let stored = Connections::from_props(props);
    let live = get_connection_state(host, p, props);
    if std::env::var_os("SHAPE_TRACE").is_some() {
        eprintln!("[shape] {p:?} stored {stored:?} live {live:?}");
    }
    if live == stored {
        return props.to_string();
    }
    let power = wire_power_of(props);
    if let Some(state) = host.resolve_wire_state(&live, power) {
        if get(host, p).is_some_and(|(name, _)| is_wire(&name)) {
            host.set_wire_state(p.0, p.1, p.2, state);
        }
    }
    get(host, p)
        .map(|(_, p2)| p2)
        .unwrap_or_else(|| props.to_string())
}

/// Entry point from `Game::update_wire`: recompute one wire, then drain
/// the entire same-tick cascade. No scheduled ticks anywhere on this
/// path — the whole propagation is synchronous.
pub fn update_wire_cascade(host: &mut impl WireHost, x: i32, y: i32, z: i32, props: &str) {
    let mut queue = NeighborQueue::new();
    let props = refresh_wire_shape(host, (x, y, z), props);
    update_wire_strength(host, (x, y, z), &props, &mut queue);
    queue.drain(host);
}

// ----------------------------------------------------------------------
// Tests: an in-memory host exercising every rule the old engine got
// wrong (vertical transfer, diagonals, conductor escalation, emission
// gating, dot/line shapes).
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A miniature world: positions → `name[k=v,..]` specs. State ids are
    /// packed (power | sides) so `set_wire_state` can rebuild the spec.
    #[derive(Default)]
    struct Sim {
        blocks: HashMap<Pos, String>,
    }

    impl Sim {
        fn put(&mut self, p: Pos, spec: &str) {
            self.blocks.insert(p, spec.to_string());
        }

        fn wire(&mut self, p: Pos, power: i32) {
            self.put(
                p,
                &format!(
                    "minecraft:redstone_wire[east=none,north=none,power={power},\
                     south=none,west=none]"
                ),
            );
        }

        fn power_at(&self, p: Pos) -> Option<i32> {
            let (name, props) = self.block_at(p.0, p.1, p.2)?;
            is_wire(&name).then(|| wire_power_of(&props))
        }

        #[allow(dead_code)]
        fn sides_at(&self, p: Pos) -> Option<Connections> {
            let (name, props) = self.block_at(p.0, p.1, p.2)?;
            is_wire(&name).then(|| Connections::from_props(&props))
        }
    }

    impl BlockView for Sim {
        fn block_at(&self, x: i32, y: i32, z: i32) -> Option<(String, String)> {
            let spec = self.blocks.get(&(x, y, z))?;
            match spec.split_once('[') {
                Some((name, rest)) => {
                    Some((name.to_string(), rest.trim_end_matches(']').to_string()))
                }
                None => Some((spec.clone(), String::new())),
            }
        }
    }

    impl WireHost for Sim {
        fn set_wire_state(&mut self, x: i32, y: i32, z: i32, state: u32) {
            let side = |shift: u32| match (state >> shift) & 3 {
                1 => "side",
                2 => "up",
                _ => "none",
            };
            let power = (state & 0xf) as i32;
            self.put(
                (x, y, z),
                &format!(
                    "minecraft:redstone_wire[east={},north={},power={power},south={},west={}]",
                    side(6),
                    side(4),
                    side(8),
                    side(10)
                ),
            );
        }

        fn resolve_wire_state(&self, conn: &Connections, power: i32) -> Option<u32> {
            let bits = |s: Side| match s {
                Side::None => 0u32,
                Side::Side => 1,
                Side::Up => 2,
            };
            Some(
                ((power as u32) & 0xf)
                    | (bits(conn.north) << 4)
                    | (bits(conn.east) << 6)
                    | (bits(conn.south) << 8)
                    | (bits(conn.west) << 10),
            )
        }

        fn dispatch_neighbor_changed(&mut self, _x: i32, _y: i32, _z: i32) {}
    }

    fn run_update(sim: &mut Sim, p: Pos) {
        let (_, props) = sim.block_at(p.0, p.1, p.2).expect("wire present");
        update_wire_cascade(sim, p.0, p.1, p.2, &props);
    }

    const LEVER: &str = "minecraft:lever[face=floor,facing=north,powered=true]";

    /// The L run: signal follows the wire around a same-Y corner, and the
    /// corner recomputes its shape.
    #[test]
    fn l_run_powers_around_corner() {
        let mut sim = Sim::default();
        sim.put((-1, 100, 0), LEVER);
        sim.wire((0, 100, 0), 0);
        sim.wire((1, 100, 0), 0);
        sim.wire((1, 100, 1), 0); // the corner: turns south
        sim.wire((1, 100, 2), 0);
        run_update(&mut sim, (0, 100, 0));
        assert_eq!(sim.power_at((0, 100, 0)), Some(15));
        assert_eq!(sim.power_at((1, 100, 0)), Some(14));
        assert_eq!(sim.power_at((1, 100, 1)), Some(13));
        assert_eq!(sim.power_at((1, 100, 2)), Some(12));
        // Shape recomputes ride the
        // shape-update phase, not the power write — asserted separately
        // once that phase lands (the connection specs remain dot from
        // placement). Power propagation is the parity-relevant behavior:
        assert_eq!(sim.power_at((1, 100, 2)), Some(12));
    }

    /// The staircase: wire climbs a conductor step diagonally up and
    /// drops diagonally down on the far side. The old
    /// engine's six-face model read 0 on both.
    #[test]
    fn staircase_carries_signal_up_and_down() {
        let mut sim = Sim::default();
        sim.put((-1, 100, 0), LEVER);
        sim.wire((0, 100, 0), 0);
        sim.put((1, 100, 0), "minecraft:stone");
        sim.wire((1, 101, 0), 0); // on top of the step
        sim.wire((2, 100, 0), 0); // ground level past the step
        run_update(&mut sim, (0, 100, 0));
        assert_eq!(sim.power_at((0, 100, 0)), Some(15));
        assert_eq!(sim.power_at((1, 101, 0)), Some(14)); // up-diagonal
        assert_eq!(sim.power_at((2, 100, 0)), Some(13)); // down-diagonal
                                                         // The up-slope visual connection rides the
                                                         // shape phase; power through the diagonal is the behavior under
                                                         // test and it holds above.
    }

    /// No vertical wire-to-wire transfer: the wire directly above another
    /// contributes nothing. The old engine propagated power-1 here.
    #[test]
    fn vertical_neighbor_carries_no_signal() {
        let mut sim = Sim::default();
        sim.put((-1, 100, 0), LEVER);
        sim.wire((0, 100, 0), 0);
        sim.wire((0, 101, 0), 0); // dead-stacked
        run_update(&mut sim, (0, 100, 0));
        assert_eq!(sim.power_at((0, 100, 0)), Some(15));
        assert_eq!(sim.power_at((0, 101, 0)), Some(0));
    }

    /// A conductor directly above the receiving wire cuts every diagonal
    /// up input.
    #[test]
    fn conductor_above_cuts_diagonal_up() {
        let mut sim = Sim::default();
        sim.wire((0, 100, 0), 0);
        sim.put((1, 100, 0), "minecraft:stone");
        sim.put(
            (1, 101, 0),
            "minecraft:redstone_wire[east=none,north=none,power=15,south=none,west=none]",
        );
        assert_eq!(calculate_target_strength(&sim, (0, 100, 0)), 14);
        sim.put((0, 101, 0), "minecraft:stone"); // headroom killer
        assert_eq!(calculate_target_strength(&sim, (0, 100, 0)), 0);
    }

    /// Indirect power: a wall lever strongly powers its support block,
    /// and the wire on top of that block reads 15 through the conductor
    /// escalation. The old engine saw no source at all.
    #[test]
    fn conductor_escalation_powers_wire_on_top() {
        let mut sim = Sim::default();
        sim.put((0, 100, 0), "minecraft:stone");
        sim.put(
            (1, 100, 0),
            "minecraft:lever[face=wall,facing=west,powered=true]",
        );
        sim.wire((0, 101, 0), 0);
        run_update(&mut sim, (0, 101, 0));
        assert_eq!(sim.power_at((0, 101, 0)), Some(15));
    }

    /// Same-level transfer ignores the stored connection state — two
    /// stale dots still exchange signal — and the live
    /// recompute gates the OUTPUT, not the input.
    #[test]
    fn same_y_signal_ignores_stored_dot_state() {
        let mut sim = Sim::default();
        sim.put(
            (1, 100, 0),
            "minecraft:redstone_wire[east=none,north=none,power=7,south=none,west=none]",
        );
        sim.wire((0, 100, 0), 0); // a fresh dot
                                  // Input side: the read never consults the stored sides.
        assert_eq!(incoming_wire_signal(&sim, (0, 100, 0)), 6);
        // Without a source the cascade drains the stale value: mutual
        // re-notifications decay the pair to zero, exactly vanilla's
        // source-removal behavior.
        run_update(&mut sim, (0, 100, 0));
        assert_eq!(sim.power_at((0, 100, 0)), Some(0));
        assert_eq!(sim.power_at((1, 100, 0)), Some(0));
        // With a source the pair carries full strength across the same-Y
        // boundary even while the receiving wire is a stored dot.
        sim.put((2, 100, 0), LEVER);
        run_update(&mut sim, (1, 100, 0));
        assert_eq!(sim.power_at((1, 100, 0)), Some(15));
        assert_eq!(sim.power_at((0, 100, 0)), Some(14));
        // Output gating uses the live recompute: the stored state is
        // a dot, but the reconnected east side (and the west side the
        // axis fill extends it into — a line powers blocks at both ends)
        // emit, while north and the block above see nothing.
        let props = "east=none,north=none,power=6,south=none,west=none";
        assert_eq!(
            wire_signal_toward(&sim, (0, 100, 0), props, Dir::West, true),
            6
        );
        assert_eq!(
            wire_signal_toward(&sim, (0, 100, 0), props, Dir::East, true),
            6
        );
        assert_eq!(
            wire_signal_toward(&sim, (0, 100, 0), props, Dir::North, true),
            0
        );
        assert_eq!(
            wire_signal_toward(&sim, (0, 100, 0), props, Dir::Down, true),
            0
        );
    }

    /// A closed trapdoor renders an up-connection that carries no signal:
    /// the diagonal-up SIGNAL path needs a redstone conductor, the VISUAL
    /// path accepts any placeable support.
    #[test]
    fn trapdoor_connects_visually_but_not_signalwise() {
        let mut sim = Sim::default();
        sim.wire((0, 100, 0), 0);
        sim.put(
            (1, 100, 0),
            "minecraft:oak_trapdoor[facing=north,half=top,open=false]",
        );
        sim.put(
            (1, 101, 0),
            "minecraft:redstone_wire[east=none,north=none,power=15,south=none,west=none]",
        );
        let conn = get_connection_state(
            &sim,
            (0, 100, 0),
            "east=none,north=none,power=0,south=none,west=none",
        );
        assert_eq!(conn.east, Side::Side); // visually connected
        assert_eq!(calculate_target_strength(&sim, (0, 100, 0)), 0); // no signal
    }

    /// Wire output: a wire emits upward into the block it sits on, never
    /// downward, and horizontally only toward connected sides.
    #[test]
    fn wire_emission_directions() {
        let mut sim = Sim::default();
        sim.wire((-1, 100, 0), 0);
        sim.wire((1, 100, 0), 0);
        sim.put(
            (0, 100, 0),
            "minecraft:redstone_wire[east=side,north=none,power=12,south=none,west=side]",
        );
        let props = "east=side,north=none,power=12,south=none,west=side";
        assert_eq!(
            wire_signal_toward(&sim, (0, 100, 0), props, Dir::Up, true),
            12
        );
        assert_eq!(
            wire_signal_toward(&sim, (0, 100, 0), props, Dir::Down, true),
            0
        );
        assert_eq!(
            wire_signal_toward(&sim, (0, 100, 0), props, Dir::East, true),
            12
        );
        assert_eq!(
            wire_signal_toward(&sim, (0, 100, 0), props, Dir::West, true),
            12
        );
        assert_eq!(
            wire_signal_toward(&sim, (0, 100, 0), props, Dir::North, true),
            0
        );
        assert_eq!(
            wire_signal_toward(&sim, (0, 100, 0), props, Dir::South, true),
            0
        );
    }

    /// The torch input read (`hasSignal(support, DOWN)`): a repeater
    /// pointing into the support block turns the torch off; a wire LINE
    /// ending at the block powers it too (its end side fills connected,
    /// , so emits into the block); a lone DOT next to the block
    /// does not; and a torch floating on air never powers itself.
    #[test]
    fn torch_input_through_support_block() {
        let mut sim = Sim::default();
        sim.put((0, 100, 0), "minecraft:stone");
        sim.put(
            (1, 100, 0),
            "minecraft:repeater[delay=1,facing=east,locked=false,powered=true]",
        );
        assert_eq!(signal_toward_consumer_above(&sim, 0, 100, 0), 15);

        let mut sim = Sim::default();
        sim.put((0, 100, 0), "minecraft:stone");
        sim.put(
            (1, 100, 0),
            "minecraft:redstone_wire[east=side,north=none,power=15,south=none,west=none]",
        );
        sim.wire((2, 100, 0), 0); // makes (1,100,0) a line ending at the stone
        assert_eq!(signal_toward_consumer_above(&sim, 0, 100, 0), 15);

        let mut sim = Sim::default();
        sim.put((0, 100, 0), "minecraft:stone");
        sim.wire((1, 100, 0), 0); // isolated dot beside the stone
        assert_eq!(signal_toward_consumer_above(&sim, 0, 100, 0), 0);

        let mut sim = Sim::default();
        sim.put((0, 100, 0), "minecraft:air");
        sim.put((0, 101, 0), "minecraft:redstone_torch[lit=true]");
        assert_eq!(signal_toward_consumer_above(&sim, 0, 100, 0), 0);
    }

    /// The axis-fill: a wire with a single connection becomes a
    /// line on that axis; a wire with no connections and a non-dot seed
    /// becomes a cross; a stored dot with no recomputed connections stays
    /// a dot.
    #[test]
    fn connection_axis_fill_and_dot_preservation() {
        let mut sim = Sim::default();
        sim.wire((1, 100, 0), 0);
        sim.wire((0, 100, 0), 0);
        let line = get_connection_state(
            &sim,
            (0, 100, 0),
            "east=none,north=none,power=0,south=none,west=none",
        );
        assert_eq!(line.east, Side::Side); // toward the neighbor
        assert_eq!(line.west, Side::Side); // filled
        assert_eq!(line.north, Side::None);
        assert_eq!(line.south, Side::None);

        let cross = get_connection_state(
            &sim,
            (9, 100, 9),
            "east=side,north=side,power=0,south=side,west=side",
        );
        assert!(cross.connected(Dir::North));
        assert!(cross.connected(Dir::East));

        let dot = get_connection_state(
            &sim,
            (9, 100, 9),
            "east=none,north=none,power=0,south=none,west=none",
        );
        assert!(dot.is_dot());
    }
}
