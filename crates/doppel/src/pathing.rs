//! Ground navigation: a wanted position plus a speed modifier, backed
//! by a bounded block-grid search. The search runs from the game tick
//! where the position is known; goals keep their give-up clocks; the
//! move control consumes one waypoint at a time.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// A solid-cell query: true when the cell blocks a walker. Tests pass
/// a closure over a hand-built grid; the game passes its block test.
pub type BlockQuery<'a> = &'a dyn Fn(i32, i32, i32) -> bool;

/// One step on a route: the standing cell, or the wall face a climb
/// step presses against.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) struct Waypoint {
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) z: i32,
    /// The solid cell a climb step holds onto, as (dx, dz) from the
    /// waypoint.
    pub(crate) wall: Option<(i32, i32)>,
}

impl Waypoint {
    #[cfg(test)]
    fn stand(x: i32, y: i32, z: i32) -> Waypoint {
        Waypoint {
            x,
            y,
            z,
            wall: None,
        }
    }
}

/// The flat-move directions, fixed order for determinism.
const DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
/// Visited-node budget per search.
const NODE_BUDGET: u32 = 400;
/// Nodes farther than this ring from the start are not expanded.
const RANGE_CAP: i32 = 48;
/// The routed fall bound, in blocks.
const MAX_FALL: i32 = 4;
/// Flat move cost per block.
const COST_FLAT: f32 = 1.0;
/// Jump-up cost.
const COST_JUMP: f32 = 1.5;
/// Climb step cost.
const COST_CLIMB: f32 = 2.0;
/// Fall surcharge per dropped block.
const COST_FALL_BLOCK: f32 = 0.5;

/// Whether a walker stands at the cell: solid floor, clear feet and
/// head.
fn walkable(solid: BlockQuery, x: i32, y: i32, z: i32) -> bool {
    solid(x, y - 1, z) && !solid(x, y, z) && !solid(x, y + 1, z)
}

/// One open-list entry: the priority is (f, insertion order).
struct Open {
    f: f32,
    order: u32,
    pos: (i32, i32, i32),
}

impl PartialEq for Open {
    fn eq(&self, other: &Open) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Open {}

impl PartialOrd for Open {
    fn partial_cmp(&self, other: &Open) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Open {
    fn cmp(&self, other: &Open) -> Ordering {
        // Smaller f, then earlier insertion, pops first.
        other
            .f
            .total_cmp(&self.f)
            .then(other.order.cmp(&self.order))
    }
}

/// The per-node record: cost so far, parent, and the climb state.
struct Visit {
    g: f32,
    from: Option<(i32, i32, i32)>,
    wall: Option<(i32, i32)>,
    climbing: bool,
}

/// One expansion edge.
struct Edge {
    pos: (i32, i32, i32),
    cost: f32,
    wall: Option<(i32, i32)>,
    climbing: bool,
}

/// The neighbor edges of a node. Standing nodes walk, jump and drop;
/// climbing nodes climb on or dismount onto standing cells.
fn edges(solid: BlockQuery, p: (i32, i32, i32), climbing: bool, climb: bool) -> Vec<Edge> {
    let mut out = Vec::with_capacity(8);
    if !climbing {
        for (dx, dz) in DIRS {
            let (nx, nz) = (p.0 + dx, p.2 + dz);
            // Flat.
            if walkable(solid, nx, p.1, nz) {
                out.push(Edge {
                    pos: (nx, p.1, nz),
                    cost: COST_FLAT,
                    wall: None,
                    climbing: false,
                });
            }
            // Jump up one: standing target, headroom over the jump.
            if walkable(solid, nx, p.1 + 1, nz) && !solid(p.0, p.1 + 2, p.2) {
                out.push(Edge {
                    pos: (nx, p.1 + 1, nz),
                    cost: COST_JUMP,
                    wall: None,
                    climbing: false,
                });
            }
            // Drop: a clear fall shaft onto a standing cell.
            for n in 1..=MAX_FALL {
                let ty = p.1 - n;
                if !walkable(solid, nx, ty, nz) {
                    continue;
                }
                let shaft_clear = ((ty + 2)..=(p.1 + 1)).all(|j| !solid(nx, j, nz));
                if shaft_clear {
                    out.push(Edge {
                        pos: (nx, ty, nz),
                        cost: COST_FLAT + COST_FALL_BLOCK * n as f32,
                        wall: None,
                        climbing: false,
                    });
                }
            }
        }
    }
    if climb {
        // Climb up the current column against a solid face.
        if !solid(p.0, p.1 + 1, p.2) && !solid(p.0, p.1 + 2, p.2) {
            for (dx, dz) in DIRS {
                if solid(p.0 + dx, p.1, p.2 + dz) || solid(p.0 + dx, p.1 + 1, p.2 + dz) {
                    out.push(Edge {
                        pos: (p.0, p.1 + 1, p.2),
                        cost: COST_CLIMB,
                        wall: Some((dx, dz)),
                        climbing: true,
                    });
                    break;
                }
            }
        }
        // Dismount onto a standing cell beside or above the wall.
        if climbing {
            for (dx, dz) in DIRS {
                for dy in 0..=1 {
                    if walkable(solid, p.0 + dx, p.1 + dy, p.2 + dz) {
                        out.push(Edge {
                            pos: (p.0 + dx, p.1 + dy, p.2 + dz),
                            cost: COST_JUMP,
                            wall: None,
                            climbing: false,
                        });
                    }
                }
            }
        }
    }
    out
}

/// The bounded grid search from a standing cell to a target column.
/// Returns the route excluding the start, an empty route when the
/// column already holds, or none when the target column is
/// unreachable; a budget exhaustion keeps the best partial route.
pub(crate) fn find_path(
    solid: BlockQuery,
    from: (i32, i32, i32),
    to_col: (i32, i32),
    climb: bool,
) -> Option<Vec<Waypoint>> {
    // Resolve a standing start: a walker caught midair drops to the
    // landing below it.
    let mut start = from;
    if !walkable(solid, start.0, start.1, start.2) {
        let mut landed = false;
        for dy in 1..=8 {
            if solid(start.0, start.1 - dy, start.2) {
                break;
            }
            if walkable(solid, start.0, start.1 - dy, start.2) {
                start = (start.0, start.1 - dy, start.2);
                landed = true;
                break;
            }
        }
        if !landed {
            return None;
        }
    }
    if (start.0, start.2) == to_col {
        return Some(Vec::new());
    }
    let heuristic = |p: (i32, i32, i32)| -> f32 {
        (p.0 - to_col.0).abs() as f32 + (p.2 - to_col.1).abs() as f32
    };
    let mut visits: HashMap<(i32, i32, i32), Visit> = HashMap::new();
    let mut open: BinaryHeap<Open> = BinaryHeap::new();
    let mut order = 0u32;
    let h0 = heuristic(start);
    visits.insert(
        start,
        Visit {
            g: 0.0,
            from: None,
            wall: None,
            climbing: false,
        },
    );
    open.push(Open {
        f: h0,
        order,
        pos: start,
    });
    let mut expansions = 0u32;
    let mut best = (start, h0);
    let mut goal: Option<(i32, i32, i32)> = None;
    let mut budget_hit = false;
    while let Some(Open { pos, .. }) = open.pop() {
        if expansions >= NODE_BUDGET {
            budget_hit = true;
            break;
        }
        expansions += 1;
        let visit = &visits[&pos];
        if !visit.climbing && (pos.0, pos.2) == to_col {
            goal = Some(pos);
            break;
        }
        let h = heuristic(pos);
        if h < best.1 {
            best = (pos, h);
        }
        let climbing = visit.climbing;
        let g = visit.g;
        for edge in edges(solid, pos, climbing, climb) {
            let target = edge.pos;
            if (target.0 - start.0).abs().max((target.2 - start.2).abs()) > RANGE_CAP {
                continue;
            }
            let tentative = g + edge.cost;
            match visits.get(&target) {
                Some(v) if v.g <= tentative => continue,
                _ => {}
            }
            visits.insert(
                target,
                Visit {
                    g: tentative,
                    from: Some(pos),
                    wall: edge.wall,
                    climbing: edge.climbing,
                },
            );
            order += 1;
            open.push(Open {
                f: tentative + heuristic(target),
                order,
                pos: target,
            });
        }
    }
    // A definitive dead end has no route; a budget cut keeps the best
    // partial route toward the target.
    let end = match goal {
        Some(g) => g,
        None if budget_hit && best.0 != start => best.0,
        None => return None,
    };
    let mut route = Vec::new();
    let mut at = end;
    while let Some(v) = visits.get(&at) {
        let Some(from_pos) = v.from else {
            break;
        };
        route.push(Waypoint {
            x: at.0,
            y: at.1,
            z: at.2,
            wall: v.wall,
        });
        at = from_pos;
    }
    route.reverse();
    Some(route)
}

/// Horizontal progress under this counts as stuck, per tick.
const STUCK_STEP: f64 = 0.03;
/// Stuck ticks before a repath.
const STUCK_TICKS: i32 = 40;
/// Stuck repaths before the navigation gives up.
const STUCK_REPATHS: i32 = 3;
/// Ticks between path rechecks.
const RECHECK_TICKS: i32 = 20;
/// Waypoint reach tolerance, horizontal.
const WP_TOLERANCE: f64 = 0.7;
/// Waypoint reach tolerance, vertical, for standing cells.
const WP_Y_TOLERANCE: f64 = 1.5;

/// One navigation request: the destination column, the speed
/// modifier, the route, and the bookkeeping that keeps the route
/// honest. The mob's move control consumes `wanted` every tick.
pub struct Nav {
    want: Option<Want>,
    age: i32,
    path: Vec<Waypoint>,
    idx: usize,
    climb: bool,
    needs_search: bool,
    recheck_in: i32,
    last: (f64, f64),
    stuck_ticks: i32,
    stuck_repaths: i32,
}

struct Want {
    x: f64,
    z: f64,
    modifier: f64,
}

impl Nav {
    pub(crate) fn new() -> Nav {
        Nav {
            want: None,
            age: 0,
            path: Vec::new(),
            idx: 0,
            climb: false,
            needs_search: false,
            recheck_in: RECHECK_TICKS,
            last: (0.0, 0.0),
            stuck_ticks: 0,
            stuck_repaths: 0,
        }
    }

    /// Enables the climb moves (the wall-crawling capability).
    pub(crate) fn set_climb(&mut self, climb: bool) {
        self.climb = climb;
    }

    /// Walk toward (x, z) at `modifier` times the base speed. Starts a
    /// fresh give-up clock and a fresh search.
    pub(crate) fn move_to(&mut self, x: f64, z: f64, modifier: f64) {
        self.want = Some(Want { x, z, modifier });
        self.age = 0;
        self.path.clear();
        self.idx = 0;
        self.needs_search = true;
        self.stuck_ticks = 0;
        self.stuck_repaths = 0;
        self.recheck_in = RECHECK_TICKS;
    }

    /// Retargets without resetting the give-up clock. A destination
    /// within a block of the route end keeps the route.
    pub(crate) fn retarget(&mut self, x: f64, z: f64, modifier: f64) {
        if self.want.is_none() {
            self.move_to(x, z, modifier);
            return;
        }
        let keeps_route = self
            .path
            .last()
            .is_some_and(|wp| (wp.x as f64 - x).abs().max((wp.z as f64 - z).abs()) <= 1.0);
        self.want = Some(Want { x, z, modifier });
        if !keeps_route {
            self.needs_search = true;
        }
    }

    pub(crate) fn stop(&mut self) {
        self.want = None;
        self.age = 0;
        self.path.clear();
        self.idx = 0;
        self.needs_search = false;
        self.stuck_ticks = 0;
        self.stuck_repaths = 0;
    }

    /// The active request: the next waypoint center (or the raw
    /// destination before the first search), with the speed modifier.
    pub(crate) fn wanted(&self) -> Option<(f64, f64, f64)> {
        let w = self.want.as_ref()?;
        if self.idx < self.path.len() {
            let wp = self.path[self.idx];
            let (tx, tz) = match wp.wall {
                Some((dx, dz)) => ((wp.x + dx) as f64 + 0.5, (wp.z + dz) as f64 + 0.5),
                None => (wp.x as f64 + 0.5, wp.z as f64 + 0.5),
            };
            Some((tx, tz, w.modifier))
        } else {
            Some((w.x, w.z, w.modifier))
        }
    }

    /// Whether the walker stands at the destination column.
    pub(crate) fn arrived(&self, x: f64, z: f64) -> bool {
        match &self.want {
            None => true,
            Some(w) => {
                let (dx, dz) = (w.x - x, w.z - z);
                dx * dx + dz * dz < 0.25
            }
        }
    }

    /// Whether a request is active.
    pub(crate) fn in_progress(&self) -> bool {
        self.want.is_some()
    }

    /// One tick of the give-up clock; false past the limit.
    pub(crate) fn tick_age(&mut self, limit: i32) -> bool {
        self.age += 1;
        if self.age > limit {
            self.stop();
            false
        } else {
            true
        }
    }

    /// The per-tick housekeeping, run where the world is in hand:
    /// deferred searches, waypoint advance, stuck repaths, and the
    /// periodic route recheck.
    pub(crate) fn nav_tick(&mut self, x: f64, y: f64, z: f64, on_ground: bool, solid: BlockQuery) {
        let Some(want) = self.want.as_ref().map(|w| (w.x, w.z)) else {
            return;
        };
        if self.needs_search {
            self.run_search(x, y, z, solid);
            if self.want.is_none() {
                return;
            }
        }
        while self.idx < self.path.len() && self.reached(x, y, z, on_ground, self.path[self.idx]) {
            self.idx += 1;
        }
        if self.idx < self.path.len() {
            let (dx, dz) = (x - self.last.0, z - self.last.1);
            if dx * dx + dz * dz < STUCK_STEP * STUCK_STEP {
                self.stuck_ticks += 1;
            } else {
                self.stuck_ticks = 0;
            }
            if self.stuck_ticks >= STUCK_TICKS {
                self.stuck_ticks = 0;
                if self.stuck_repaths >= STUCK_REPATHS {
                    self.stop();
                    return;
                }
                self.stuck_repaths += 1;
                self.run_search(x, y, z, solid);
            }
        } else {
            self.stuck_ticks = 0;
            self.stuck_repaths = 0;
        }
        self.last = (x, z);
        self.recheck_in -= 1;
        if self.recheck_in <= 0 {
            self.recheck_in = RECHECK_TICKS;
            let end_ok = self.path.last().is_none_or(|wp| {
                (wp.x as f64 - want.0)
                    .abs()
                    .max((wp.z as f64 - want.1).abs())
                    <= 1.0
            });
            if !end_ok || !self.route_open(solid) {
                self.run_search(x, y, z, solid);
            }
        }
    }

    /// Runs the deferred search; an unreachable destination stops the
    /// navigation so goals fall back on their give-up logic.
    fn run_search(&mut self, x: f64, y: f64, z: f64, solid: BlockQuery) {
        let Some(w) = &self.want else {
            return;
        };
        let from = (x.floor() as i32, y.floor() as i32, z.floor() as i32);
        let to_col = (w.x.floor() as i32, w.z.floor() as i32);
        match find_path(solid, from, to_col, self.climb) {
            Some(path) => {
                self.path = path;
                self.idx = 0;
                self.needs_search = false;
                self.stuck_ticks = 0;
                self.recheck_in = RECHECK_TICKS;
            }
            None => self.stop(),
        }
    }

    /// Whether the walker's position counts as at the waypoint.
    fn reached(&self, x: f64, y: f64, z: f64, on_ground: bool, wp: Waypoint) -> bool {
        let (cx, cz) = (wp.x as f64 + 0.5, wp.z as f64 + 0.5);
        let near = (x - cx).abs() < WP_TOLERANCE && (z - cz).abs() < WP_TOLERANCE;
        match wp.wall {
            Some(_) => near && y >= wp.y as f64 - 0.75,
            None => near && on_ground && (y - wp.y as f64).abs() <= WP_Y_TOLERANCE,
        }
    }

    /// Whether the next waypoint still stands (or holds its wall).
    fn route_open(&self, solid: BlockQuery) -> bool {
        let Some(wp) = self.path.get(self.idx) else {
            return true;
        };
        match wp.wall {
            None => walkable(solid, wp.x, wp.y, wp.z),
            Some((dx, dz)) => {
                !solid(wp.x, wp.y, wp.z)
                    && !solid(wp.x, wp.y + 1, wp.z)
                    && (solid(wp.x + dx, wp.y, wp.z + dz) || solid(wp.x + dx, wp.y + 1, wp.z + dz))
            }
        }
    }

    /// The remaining waypoints, for tests.
    #[cfg(test)]
    pub(crate) fn route(&self) -> Vec<Waypoint> {
        self.path[self.idx.min(self.path.len())..].to_vec()
    }
}

impl Default for Nav {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashSet;

    /// A solid grid from cells, plus a floor helper.
    fn grid(cells: &[(i32, i32, i32)]) -> impl Fn(i32, i32, i32) -> bool + '_ {
        let set: HashSet<(i32, i32, i32)> = cells.iter().copied().collect();
        move |x, y, z| set.contains(&(x, y, z))
    }

    /// A flat floor y=99 over x in 0..=20, z in 0..=5.
    fn flat_floor() -> Vec<(i32, i32, i32)> {
        (0..=20)
            .flat_map(|x| (0..=5).map(move |z| (x, 99, z)))
            .collect()
    }

    #[test]
    fn straight_route_walks_the_line() {
        let cells = flat_floor();
        let q = grid(&cells);
        let route = find_path(&q, (0, 100, 2), (10, 2), false).unwrap();
        assert_eq!(route.len(), 10, "one waypoint per block");
        assert_eq!(route[0], Waypoint::stand(1, 100, 2));
        assert!(route.iter().all(|wp| wp.y == 100 && wp.wall.is_none()));
    }

    #[test]
    fn one_block_obstacle_hops() {
        let mut cells = flat_floor();
        cells.push((5, 100, 2));
        let q = grid(&cells);
        let route = find_path(&q, (0, 100, 2), (10, 2), false).unwrap();
        let hop = route.iter().find(|wp| wp.x == 5);
        assert!(
            hop.is_some_and(|wp| wp.y == 101 && wp.wall.is_none()),
            "the route steps up onto the block: {route:?}"
        );
    }

    #[test]
    fn ledge_route_drops_down() {
        // High floor x<=8, low floor x>=9: a two-block ledge at x=9.
        let mut cells: Vec<(i32, i32, i32)> = (0..=8)
            .flat_map(|x| (0..=5).map(move |z| (x, 99, z)))
            .collect();
        cells.extend((9..=20).flat_map(|x| (0..=5).map(move |z| (x, 97, z))));
        let q = grid(&cells);
        let route = find_path(&q, (2, 100, 2), (12, 2), false).unwrap();
        let drop = route.iter().find(|wp| wp.x == 9);
        assert!(
            drop.is_some_and(|wp| wp.y == 98),
            "the route drops off the ledge: {route:?}"
        );
        assert!(route.iter().all(|wp| wp.y <= 100));
    }

    #[test]
    fn enclosed_target_finds_no_route() {
        let mut cells = flat_floor();
        // A solid box around column (10, 2), floor to head.
        for x in 9..=11 {
            for z in 1..=3 {
                for y in 100..=102 {
                    if (x, z) != (10, 2) {
                        cells.push((x, y, z));
                    }
                }
            }
        }
        let q = grid(&cells);
        assert!(
            find_path(&q, (0, 100, 2), (10, 2), false).is_none(),
            "the enclosed column is unreachable"
        );
    }

    #[test]
    fn wall_routes_around() {
        let mut cells = flat_floor();
        // A 5-tall wall across z=0..=3 at x=5; the gap sits at z=4..5.
        for z in 0..=3 {
            for y in 100..=104 {
                cells.push((5, y, z));
            }
        }
        let q = grid(&cells);
        let route = find_path(&q, (0, 100, 2), (10, 2), false).unwrap();
        assert!(route.len() > 10, "the detour is longer than the line");
        assert!(
            !route.iter().any(|wp| wp.x == 5 && wp.z <= 3 && wp.y <= 104),
            "the route never crosses the wall: {route:?}"
        );
        assert_eq!(route.last().map(|wp| (wp.x, wp.z)), Some((10, 2)));
    }

    #[test]
    fn climb_wall_only_with_the_flag() {
        // A 3-tall wall sealing the corridor: z=0..=2 walled at x=5,
        // the floor's z range is 0..=2 so there is no way around.
        let mut cells: Vec<(i32, i32, i32)> = (0..=20)
            .flat_map(|x| (0..=2).map(move |z| (x, 99, z)))
            .collect();
        for z in 0..=2 {
            for y in 100..=102 {
                cells.push((5, y, z));
            }
        }
        let q = grid(&cells);
        assert!(
            find_path(&q, (2, 100, 1), (8, 1), false).is_none(),
            "without the climb flag the wall blocks the route"
        );
        let route = find_path(&q, (2, 100, 1), (8, 1), true).unwrap();
        assert!(
            route.iter().any(|wp| wp.wall.is_some()),
            "the route climbs: {route:?}"
        );
        assert_eq!(route.last().map(|wp| (wp.x, wp.z)), Some((8, 1)));
        // The dismount lands standing on the far floor.
        assert!(route
            .last()
            .is_some_and(|wp| wp.y == 100 && wp.wall.is_none()));
    }

    #[test]
    fn nav_consumes_waypoints_in_order() {
        let cells = flat_floor();
        let q = grid(&cells);
        let mut nav = Nav::new();
        nav.move_to(5.5, 2.5, 1.0);
        nav.nav_tick(0.5, 100.0, 2.5, true, &q);
        assert_eq!(nav.wanted().unwrap().0, 1.5, "the first waypoint");
        // Walk onto it: the next waypoint takes over.
        nav.nav_tick(1.5, 100.0, 2.5, true, &q);
        assert_eq!(nav.wanted().unwrap().0, 2.5, "the second waypoint");
        assert!(nav.in_progress());
        // Reaching the destination stops the request.
        nav.nav_tick(5.5, 100.0, 2.5, true, &q);
        assert!(nav.arrived(5.5, 2.5));
    }

    #[test]
    fn nav_repaths_when_the_target_moves() {
        let cells = flat_floor();
        let q = grid(&cells);
        let mut nav = Nav::new();
        nav.move_to(10.5, 2.5, 1.0);
        nav.nav_tick(0.5, 100.0, 2.5, true, &q);
        let first: Vec<_> = nav.route();
        assert_eq!(first.last().map(|wp| wp.x), Some(10));
        nav.retarget(2.5, 4.5, 1.0);
        nav.nav_tick(0.5, 100.0, 2.5, true, &q);
        let second = nav.route();
        assert_eq!(second.last().map(|wp| wp.x), Some(2));
        assert_ne!(first, second, "the route follows the new target");
    }

    #[test]
    fn nav_repaths_when_a_block_lands_on_the_route() {
        let cells = RefCell::new(flat_floor());
        let q = |x: i32, y: i32, z: i32| cells.borrow().contains(&(x, y, z));
        let mut nav = Nav::new();
        nav.move_to(10.5, 2.5, 1.0);
        nav.nav_tick(0.5, 100.0, 2.5, true, &q);
        assert!(nav.route().iter().all(|wp| wp.z == 2));
        // Walk two waypoints in, then a 2-tall wall lands on the next
        // waypoint: the recheck must reroute around the column.
        nav.nav_tick(1.5, 100.0, 2.5, true, &q);
        nav.nav_tick(2.5, 100.0, 2.5, true, &q);
        cells.borrow_mut().push((3, 100, 2));
        cells.borrow_mut().push((3, 101, 2));
        for _ in 0..RECHECK_TICKS {
            nav.nav_tick(2.5, 100.0, 2.5, true, &q);
        }
        let rerouted = nav.route();
        assert!(
            !rerouted.iter().any(|wp| wp.x == 3 && wp.z == 2),
            "the route leaves the blocked column: {rerouted:?}"
        );
        assert_eq!(rerouted.last().map(|wp| wp.x), Some(10));
    }

    #[test]
    fn stuck_navigation_gives_up_after_repeated_repaths() {
        let cells = flat_floor();
        let q = grid(&cells);
        let mut nav = Nav::new();
        nav.move_to(10.5, 2.5, 1.0);
        // The walker never moves: stuck repaths accumulate, then stop.
        for _ in 0..(STUCK_TICKS * (STUCK_REPATHS + 2)) {
            nav.nav_tick(0.5, 100.0, 2.5, true, &q);
            if !nav.in_progress() {
                break;
            }
        }
        assert!(!nav.in_progress(), "the stuck route gives up");
    }
}
