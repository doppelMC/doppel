//! Local piston-lifecycle tests: they drive `Game` directly through the
//! same Inbound events the network path delivers.

use crate::game::{Game, Inbound, Outbound};
use crate::WireChunk;

/// A game with three synthetic all-air chunks around the origin and one
/// viewer whose outbound frames we read back.
fn harness() -> (Game, std::sync::mpsc::Receiver<Outbound>) {
    let (_tx, rx) = std::sync::mpsc::channel::<Inbound>();
    let mut g = Game::new(rx, None, None);
    assert!(g.registry_for_test(), "pins/blocks.json not found");
    let wire = |x: i32| {
        let mut w = WireChunk {
            x,
            z: 0,
            heightmaps: Vec::new(),
            sections: Vec::new(),
            block_entities: Vec::new(),
            light: Default::default(),
        };
        for _ in 0..24 {
            w.sections.push(doppel_world::chunk_codec::WireSection {
                non_empty: 0,
                fluid: 0,
                block_states: doppel_world::chunk_codec::Container::Single(0),
                biomes: doppel_world::chunk_codec::Container::Single(0),
            });
        }
        w
    };
    for cx in [-1, 0, 1, 2] {
        g.seed_chunk_for_test(cx, 0, wire(cx));
    }
    let (tx_out, rx_out) = std::sync::mpsc::channel::<Outbound>();
    g.join_viewer_for_test(0, &[(-1, 0), (0, 0), (1, 0), (2, 0)], tx_out);
    (g, rx_out)
}

/// Runs one scripted command (`setblock ...` or `tick step N`).
pub fn cmd(g: &mut Game, s: &str) {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts[0] == "tick" {
        g.handle(Inbound::TickStep {
            conn: 0,
            steps: parts[2].parse().unwrap(),
        });
        return;
    }
    g.handle(Inbound::Setblock {
        conn: 0,
        x: parts[1].parse().unwrap(),
        y: parts[2].parse().unwrap(),
        z: parts[3].parse().unwrap(),
        name: parts[4].to_string(),
    });
}

/// One game tick; returns that tick's broadcasts decoded to `name[props]`
/// strings (0x56 entries, or the single 0x08 entry).
pub fn tick(g: &mut Game, rx: &std::sync::mpsc::Receiver<Outbound>) -> Vec<String> {
    g.tick_once_for_test();
    let mut out = Vec::new();
    while let Ok(frame) = rx.try_recv() {
        let Outbound::Frame { id, body } = frame else {
            continue;
        };
        if id != 0x56 && id != 0x08 {
            continue;
        }
        let mut o = 8usize;
        let mut count = 1usize;
        let mut single_local = 0u64;
        if id == 0x56 {
            count = read_varlong(&body, &mut o) as usize;
        } else {
            let x = i64::from_be_bytes(body[0..8].try_into().unwrap());
            single_local = (((x >> 38) as u64 & 0xf) << 8)
                | (((x >> 12) as u64 & 0xf) << 4)
                | (x as u64 & 0xf);
        }
        for _ in 0..count {
            let (local, state) = if id == 0x56 {
                let v = read_varlong(&body, &mut o);
                (v & 0xfff, (v >> 12) as u32)
            } else {
                (single_local, read_varlong(&body, &mut o) as u32)
            };
            out.push(format!("{:x}={}", local, g.state_label_for_test(state)));
        }
    }
    out
}

fn read_varlong(body: &[u8], o: &mut usize) -> u64 {
    let mut v: u64 = 0;
    let mut sh = 0u32;
    while *o < body.len() {
        let b = body[*o];
        *o += 1;
        v |= u64::from(b & 0x7f) << sh;
        sh += 7;
        if b & 0x80 == 0 {
            break;
        }
    }
    v
}

pub fn at(g: &Game, x: i32, y: i32, z: i32) -> String {
    g.block_label_for_test(x, y, z)
}

/// Straight push: extend, carriers, landing, retract — the full lifecycle
/// with vanilla's per-tick broadcast offsets.
#[test]
fn piston_extend_and_retract() {
    let (mut g, rx) = harness();
    for c in [
        "setblock 25 100 10 minecraft:stone",
        "setblock 24 100 10 minecraft:stone",
        "setblock 23 100 10 minecraft:piston[extended=false,facing=east]",
        "setblock 22 100 10 minecraft:lever[face=floor,powered=true]",
    ] {
        cmd(&mut g, c);
    }
    // T1: placements (deduped, one entry per position).
    let f = tick(&mut g, &rx);
    assert_eq!(
        f,
        vec![
            "6a4=minecraft:lever[face=floor,facing=north,powered=true]",
            "7a4=minecraft:piston[extended=false,facing=east]",
            "8a4=minecraft:stone[]",
            "9a4=minecraft:stone[]",
        ],
        "placement flush"
    );
    // T1's block-event phase extended the piston (same tick as power).
    assert_eq!(
        at(&g, 23, 100, 10),
        "minecraft:piston[extended=true,facing=east]",
        "extended in the same tick as the power change"
    );
    // T2: the extend edits broadcast (carriers + extended base).
    let f = tick(&mut g, &rx);
    assert_eq!(f.len(), 4, "carrier edits + base: {f:?}");
    assert_eq!(
        at(&g, 24, 100, 10),
        "minecraft:moving_piston[facing=east,type=normal]"
    );
    assert_eq!(
        at(&g, 25, 100, 10),
        "minecraft:moving_piston[facing=east,type=normal]"
    );
    assert_eq!(
        at(&g, 26, 100, 10),
        "minecraft:moving_piston[facing=east,type=normal]"
    );
    // T3: animation completes (BE phase) — broadcast defers.
    let f = tick(&mut g, &rx);
    assert!(f.is_empty(), "landing defers its broadcast: {f:?}");
    // T4: the landing broadcast.
    let f = tick(&mut g, &rx);
    assert_eq!(f.len(), 3, "head + two stones: {f:?}");
    assert_eq!(
        at(&g, 24, 100, 10),
        "minecraft:piston_head[facing=east,short=false,type=normal]"
    );
    assert_eq!(at(&g, 25, 100, 10), "minecraft:stone[]");
    assert_eq!(at(&g, 26, 100, 10), "minecraft:stone[]");

    // Power off + retract.
    cmd(
        &mut g,
        "setblock 22 100 10 minecraft:lever[face=floor,powered=false]",
    );
    let f = tick(&mut g, &rx);
    assert_eq!(
        f,
        vec!["6a4=minecraft:lever[face=floor,facing=north,powered=false]"]
    );
    assert_eq!(
        at(&g, 23, 100, 10),
        "minecraft:moving_piston[facing=east,type=normal]",
        "retract starts in the power-change tick"
    );
    assert_eq!(at(&g, 24, 100, 10), "minecraft:air[]", "head popped");
    let f = tick(&mut g, &rx);
    assert_eq!(f.len(), 2, "retract edits: {f:?}");
    let f = tick(&mut g, &rx);
    assert!(f.is_empty(), "landing defers: {f:?}");
    let f = tick(&mut g, &rx);
    assert_eq!(f.len(), 1, "piston re-lands: {f:?}");
    assert_eq!(
        at(&g, 23, 100, 10),
        "minecraft:piston[extended=false,facing=east]"
    );
    // Stones stay pushed (plain piston does not pull).
    assert_eq!(at(&g, 25, 100, 10), "minecraft:stone[]");
    assert_eq!(at(&g, 26, 100, 10), "minecraft:stone[]");
}

/// Sticky pull: the block two ahead comes back with the head.
#[test]
fn sticky_piston_pulls() {
    let (mut g, rx) = harness();
    for c in [
        "setblock 25 100 10 minecraft:stone",
        "setblock 24 100 10 minecraft:stone",
        "setblock 23 100 10 minecraft:sticky_piston[extended=false,facing=east]",
        "setblock 22 100 10 minecraft:lever[face=floor,powered=true]",
    ] {
        cmd(&mut g, c);
    }
    tick(&mut g, &rx); // placements + extend event
    tick(&mut g, &rx); // carrier broadcast
    tick(&mut g, &rx); // animation completes
    tick(&mut g, &rx); // landed: head@24, stone@25, stone@26
    assert_eq!(at(&g, 25, 100, 10), "minecraft:stone[]");
    cmd(
        &mut g,
        "setblock 22 100 10 minecraft:lever[face=floor,powered=false]",
    );
    tick(&mut g, &rx); // retract event: base -> carrier, pull resolves
    assert_eq!(
        at(&g, 23, 100, 10),
        "minecraft:moving_piston[facing=east,type=sticky]"
    );
    // The pulled stone becomes a carrier at the arm cell.
    assert_eq!(
        at(&g, 24, 100, 10),
        "minecraft:moving_piston[facing=east,type=normal]"
    );
    tick(&mut g, &rx);
    tick(&mut g, &rx);
    tick(&mut g, &rx); // landing broadcast
    assert_eq!(
        at(&g, 23, 100, 10),
        "minecraft:sticky_piston[extended=false,facing=east]"
    );
    assert_eq!(at(&g, 24, 100, 10), "minecraft:stone[]", "pulled back");
    assert_eq!(at(&g, 25, 100, 10), "minecraft:air[]");
    assert_eq!(at(&g, 26, 100, 10), "minecraft:stone[]");
}

/// 13 blocks in front: resolution fails, nothing moves.
#[test]
fn piston_push_limit() {
    let (mut g, rx) = harness();
    for i in 0..13 {
        cmd(
            &mut g,
            &format!("setblock {} 100 10 minecraft:stone", 24 + i),
        );
    }
    cmd(
        &mut g,
        "setblock 23 100 10 minecraft:piston[extended=false,facing=east]",
    );
    cmd(
        &mut g,
        "setblock 22 100 10 minecraft:lever[face=floor,powered=true]",
    );
    let f = tick(&mut g, &rx);
    assert_eq!(f.len(), 15, "placements only: {f:?}");
    // No block event: the dry run failed.
    assert_eq!(
        at(&g, 23, 100, 10),
        "minecraft:piston[extended=false,facing=east]",
        "no extend past 12 blocks"
    );
    let f = tick(&mut g, &rx);
    assert!(f.is_empty(), "no carriers: {f:?}");
    assert_eq!(at(&g, 24, 100, 10), "minecraft:stone[]");
}

/// A destructible block dead ahead pops instead of blocking.
#[test]
fn piston_destroys_head_on() {
    let (mut g, rx) = harness();
    cmd(&mut g, "setblock 24 100 10 minecraft:redstone_torch");
    cmd(
        &mut g,
        "setblock 23 100 10 minecraft:piston[extended=false,facing=east]",
    );
    cmd(
        &mut g,
        "setblock 22 100 10 minecraft:lever[face=floor,powered=true]",
    );
    tick(&mut g, &rx);
    assert_eq!(
        at(&g, 23, 100, 10),
        "minecraft:piston[extended=true,facing=east]",
        "popped block does not block the push"
    );
    // The torch cell now holds the arm carrier.
    assert_eq!(
        at(&g, 24, 100, 10),
        "minecraft:moving_piston[facing=east,type=normal]"
    );
    tick(&mut g, &rx);
    tick(&mut g, &rx);
    tick(&mut g, &rx);
    assert_eq!(
        at(&g, 24, 100, 10),
        "minecraft:piston_head[facing=east,short=false,type=normal]"
    );
}

/// An obsidian wall blocks the push: no event fires.
#[test]
fn piston_blocked_by_obsidian() {
    let (mut g, rx) = harness();
    cmd(&mut g, "setblock 24 100 10 minecraft:obsidian");
    cmd(
        &mut g,
        "setblock 23 100 10 minecraft:piston[extended=false,facing=east]",
    );
    cmd(
        &mut g,
        "setblock 22 100 10 minecraft:lever[face=floor,powered=true]",
    );
    let f = tick(&mut g, &rx);
    assert_eq!(f.len(), 3, "placements only: {f:?}");
    assert_eq!(
        at(&g, 23, 100, 10),
        "minecraft:piston[extended=false,facing=east]"
    );
    assert_eq!(at(&g, 24, 100, 10), "minecraft:obsidian[]");
}

/// Quasi-connectivity: power one block up-and-across, never adjacent to
/// the piston itself (a neighbor of pos.above() that is not the piston).
#[test]
fn piston_quasi_connectivity() {
    let (mut g, rx) = harness();
    cmd(
        &mut g,
        "setblock 23 100 10 minecraft:piston[extended=false,facing=east]",
    );
    cmd(
        &mut g,
        "setblock 23 101 9 minecraft:lever[face=floor,powered=true]",
    );
    let f = tick(&mut g, &rx);
    assert_eq!(f.len(), 2, "placements: {f:?}");
    assert_eq!(
        at(&g, 23, 100, 10),
        "minecraft:piston[extended=true,facing=east]",
        "quasi-connectivity extends the piston"
    );
}

/// Reproduces the parity_redstone circuit against the real Game to trace
/// the extra-broadcast divergence locally (run with WIRE_TRACE=1).
#[test]
fn circuit_trace() {
    let (mut g, _rx) = harness();
    for c in [
        "setblock 10 100 10 minecraft:lever[face=floor,powered=false]",
        "setblock 11 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "setblock 12 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "setblock 13 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "setblock 14 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "setblock 15 100 10 minecraft:redstone_torch",
    ] {
        cmd(&mut g, c);
        g.tick_once_for_test();
    }
    std::env::set_var("WIRE_TRACE", "1");
    cmd(
        &mut g,
        "setblock 10 100 10 minecraft:lever[face=floor,powered=true]",
    );
    for _ in 0..10 {
        g.tick_once_for_test();
    }
    std::env::remove_var("WIRE_TRACE");
}
