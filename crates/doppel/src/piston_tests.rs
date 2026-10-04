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
        for sy in 0..24 {
            // A stone layer at y=99 (section 10, local y=3) floors every
            // circuit: floor-mounted redstone needs support, and the
            // reference pops it without one. Storage packing is YZX.
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
                    entries: vec![0, 1],
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
    if parts.len() == 2 && parts[0] == "tick" && parts[1] == "freeze" {
        g.handle(Inbound::TickFreeze {
            conn: 0,
            frozen: true,
        });
        return;
    }
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
    cmd(&mut g, "setblock 23 100 9 minecraft:stone");
    cmd(
        &mut g,
        "setblock 23 101 9 minecraft:lever[face=floor,powered=true]",
    );
    let f = tick(&mut g, &rx);
    assert_eq!(f.len(), 3, "placements: {f:?}");
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
    // Full CI circuit tail: comparator + observer + stone swap + L-shape.
    for c in [
        "setblock 16 100 10 minecraft:repeater[facing=west,delay=1]",
        "tick step 1",
        "setblock 17 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 18 100 10 minecraft:comparator[facing=west,mode=compare,powered=false]",
        "tick step 1",
        "setblock 10 100 12 minecraft:lever[face=floor,facing=north,powered=true]",
        "tick step 1",
        "setblock 11 100 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 17 100 13 minecraft:lever[face=floor,facing=north,powered=true]",
        "tick step 1",
        "setblock 20 100 10 minecraft:observer[facing=east]",
        "tick step 1",
        "setblock 21 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 19 100 10 minecraft:stone",
        "tick step 1",
        "setblock 19 100 10 minecraft:oak_planks",
        "tick step 10",
    ] {
        cmd(&mut g, c);
        g.tick_once_for_test();
    }
    std::env::remove_var("WIRE_TRACE");
    // Final lever state: 8439 = powered true, 8440 = false.
    let lever = g.get_block(10, 100, 10).expect("lever present");
    eprintln!("[trace] final lever state: {lever:?}");
    assert!(
        lever.1.contains("powered=true"),
        "lever should end ON (vanilla parity), got {lever:?}"
    );
}

#[test]
fn ci_placement_path_repro() {
    let (mut g, rx) = harness();
    cmd(&mut g, "setblock 1 -60 1 minecraft:stone");
    g.tick_once_for_test();
    g.handle(Inbound::Tp {
        conn: 0,
        x: 1.0,
        y: -59.0,
        z: 3.0,
    });
    g.tick_once_for_test();
    g.handle(Inbound::Give {
        conn: 0,
        item: "minecraft:stone".to_string(),
        count: 64,
    });
    g.tick_once_for_test();
    let _ = rx;
    // The exact wire bytes the oracle bot sends.
    let mut body = Vec::new();
    body.push(0); // hand
    let packed: i64 = (1i64 << 38) | (1i64 << 12) | (-60i64 & 0xfff);
    body.extend_from_slice(&packed.to_be_bytes());
    body.push(1); // face up
    body.extend_from_slice(&0.5f32.to_be_bytes());
    body.extend_from_slice(&1.0f32.to_be_bytes());
    body.extend_from_slice(&0.5f32.to_be_bytes());
    body.push(0);
    body.push(0);
    body.push(1); // sequence
    let hit = match crate::placement::parse_use_item_on(&body) {
        Ok(h) => h,
        Err(e) => panic!("parse failed: {e:#}"),
    };
    g.handle(Inbound::UseItemOn {
        conn: 0,
        x: hit.x,
        y: hit.y,
        z: hit.z,
        face: hit.face,
        cursor_x: hit.cursor_x,
        cursor_y: hit.cursor_y,
        cursor_z: hit.cursor_z,
        hand: hit.hand,
        sequence: hit.sequence,
    });
    g.tick_once_for_test();
    assert_eq!(
        g.block_label_for_test(1, -59, 1),
        "minecraft:stone[]",
        "the clicked block lands"
    );
}

#[test]
fn play_event_accepts_bot_use_item_on() {
    let mut body = Vec::new();
    body.push(0); // hand
    let packed: i64 = (1i64 << 38) | (1i64 << 12) | (-60i64 & 0xfff);
    body.extend_from_slice(&packed.to_be_bytes());
    body.push(1); // face up
    body.extend_from_slice(&0.5f32.to_be_bytes());
    body.extend_from_slice(&1.0f32.to_be_bytes());
    body.extend_from_slice(&0.5f32.to_be_bytes());
    body.push(0);
    body.push(0);
    body.push(1); // sequence
    let event =
        crate::play_event(7, 0x42, &body).expect("the bot's use_item_on frame must translate");
    match event {
        Inbound::UseItemOn {
            conn,
            x,
            y,
            z,
            face,
            ..
        } => {
            assert_eq!(conn, 7);
            assert_eq!((x, y, z, face), (1, -60, 1, 1));
        }
        _ => panic!("wrong event variant"),
    }
}

/// Encodes a 0x07 chat_command body for `play_event` probes.
fn chat_command_body(cmd: &str) -> Vec<u8> {
    let mut b = Vec::new();
    doppel_protocol::write_string(&mut b, cmd);
    b
}

#[test]
fn play_event_parses_time_set() {
    let value = |cmd: &str| match crate::play_event(0, 0x07, &chat_command_body(cmd)) {
        Some(Inbound::TimeSet { value, .. }) => Some(value),
        _ => None,
    };
    assert_eq!(value("time set 6000"), Some(6000));
    assert_eq!(value("time set -1"), Some(-1));
    assert_eq!(value("time set 0"), Some(0));
    // Word times resolve through the mob-clock parser: day=1000 etc.
    assert_eq!(value("time set day"), Some(1000));
    assert_eq!(value("time set abc"), None);
    assert_eq!(value("time set"), None);
    assert_eq!(value("time add 100"), None);
    assert_eq!(value("time query daytime"), None);
}

#[test]
fn time_set_broadcasts_set_time_and_replies() {
    let (mut g, rx) = harness();
    g.handle(Inbound::TimeSet {
        conn: 0,
        value: 6000,
    });
    // The frames ride the next tick's connection flush, like every
    // scripted command reply.
    g.tick_once_for_test();
    let mut saw_set_time = false;
    let mut saw_reply = false;
    while let Ok(frame) = rx.try_recv() {
        let Outbound::Frame { id, body } = frame else {
            continue;
        };
        if id == 0x73 {
            assert_eq!(body.len(), 17, "gameTime + day time + trailing byte");
            assert_eq!(body[..8], 0i64.to_be_bytes(), "gameTime stays static");
            assert_eq!(body[8..16], 6000i64.to_be_bytes(), "day time lands");
            saw_set_time = true;
        }
        if id == 0x7c {
            saw_reply = true;
        }
    }
    assert!(saw_set_time, "time set pushes a set_time frame");
    assert!(saw_reply, "time set answers with command feedback");
    // Nineteen more ticks land on the tick-20 periodic broadcast.
    for _ in 0..19 {
        g.tick_once_for_test();
    }
    let mut periodic = false;
    while let Ok(frame) = rx.try_recv() {
        let Outbound::Frame { id, body } = frame else {
            continue;
        };
        if id == 0x73 {
            assert_eq!(body[8..16], 6000i64.to_be_bytes());
            periodic = true;
        }
    }
    assert!(
        periodic,
        "the periodic set_time carries the stored day time"
    );
}

#[test]
fn lever_state_ids() {
    let (_tx, _rx) = std::sync::mpsc::channel::<Inbound>();
    let g = Game::new(_rx, None, None);
    assert!(g.registry_for_test(), "pins/blocks.json not found");
    let reg = g.registry_snapshot_for_test();
    assert_eq!(
        reg.state_of(8439).map(|(n, _)| n.to_string()),
        Some("minecraft:lever".to_string())
    );
    let (n, p) = doppel_world::registry::BlockRegistry::split_state(
        "minecraft:lever[face=floor,facing=north,powered=true]",
    );
    assert_eq!(reg.state_id(n, p), Some(8439));
    let (n, p) = doppel_world::registry::BlockRegistry::split_state(
        "minecraft:lever[face=floor,facing=north,powered=false]",
    );
    assert_eq!(reg.state_id(n, p), Some(8440));
}

/// Simulates the SERVE path exactly: the real `run()` loop on its own
/// thread, Inbound events fed through the real mpsc channel in a tight
/// burst (like the connection reader forwarding 47 rapid command
/// packets), viewer wired for broadcasts. Decodes the captured stream
/// with the oracle's apply() logic. Reproduces the CI-side conditions
/// that produced "7 update packets, lever ends 8440".
#[test]
fn serve_loop_burst_sim() {
    let script: Vec<&str> = vec![
        "tick freeze",
        // Support platform: the reference pops floor-mounted redstone
        // components whose support is missing, so every circuit position
        // gets a stone footing before anything lands on it.
        "setblock 10 99 10 minecraft:stone",
        "setblock 11 99 10 minecraft:stone",
        "setblock 12 99 10 minecraft:stone",
        "setblock 13 99 10 minecraft:stone",
        "setblock 14 99 10 minecraft:stone",
        "setblock 15 99 10 minecraft:stone",
        "setblock 16 99 10 minecraft:stone",
        "setblock 17 99 10 minecraft:stone",
        "setblock 18 99 10 minecraft:stone",
        "setblock 19 99 10 minecraft:stone",
        "setblock 20 99 10 minecraft:stone",
        "setblock 21 99 10 minecraft:stone",
        "tick step 1",
        "setblock 12 99 11 minecraft:stone",
        "tick step 1",
        "setblock 10 99 12 minecraft:stone",
        "setblock 11 99 12 minecraft:stone",
        "setblock 12 99 12 minecraft:stone",
        "setblock 13 99 12 minecraft:stone",
        "setblock 14 99 12 minecraft:stone",
        "setblock 15 99 12 minecraft:stone",
        "setblock 16 99 12 minecraft:stone",
        "tick step 1",
        "setblock 13 99 13 minecraft:stone",
        "setblock 14 99 13 minecraft:stone",
        "setblock 15 99 13 minecraft:stone",
        "setblock 16 99 13 minecraft:stone",
        "setblock 17 99 13 minecraft:stone",
        "tick step 5",
        "setblock 10 100 10 minecraft:lever[face=floor,facing=north,powered=false]",
        "tick step 1",
        "setblock 11 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 12 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 13 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 14 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 15 100 10 minecraft:redstone_torch",
        "tick step 1",
        "setblock 10 100 10 minecraft:lever[face=floor,facing=north,powered=true]",
        "tick step 1",
        "setblock 16 100 10 minecraft:repeater[facing=west,delay=1]",
        "tick step 1",
        "setblock 17 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 18 100 10 minecraft:comparator[facing=west,mode=compare,powered=false]",
        "tick step 1",
        "setblock 10 100 12 minecraft:lever[face=floor,facing=north,powered=true]",
        "tick step 1",
        "setblock 11 100 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 12 100 12 minecraft:comparator[facing=west,mode=compare,powered=false]",
        "tick step 1",
        "setblock 13 100 12 minecraft:comparator[facing=west,mode=subtract,powered=false]",
        "tick step 1",
        "setblock 13 100 13 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 14 100 13 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 15 100 13 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 16 100 13 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 17 100 13 minecraft:lever[face=floor,facing=north,powered=true]",
        "tick step 1",
        "setblock 20 100 10 minecraft:observer[facing=east]",
        "tick step 1",
        "setblock 21 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 19 100 10 minecraft:stone",
        "tick step 1",
        "setblock 19 100 10 minecraft:oak_planks",
        "tick step 1",
        "setblock 12 100 11 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 12 100 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 13 100 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 14 100 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 15 100 12 minecraft:stone",
        "tick step 1",
        "setblock 15 101 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 13 101 12 minecraft:stone",
        "tick step 1",
        "setblock 16 101 12 minecraft:stone",
        "tick step 10",
    ];
    let (tx, rx) = std::sync::mpsc::channel::<Inbound>();
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
    // The real serve loop on a real thread.
    let runner = std::thread::spawn(move || g.run());
    // The burst: all events queued back-to-back, like the reader thread
    // forwarding a rapid command volley.
    for c in &script {
        let parts: Vec<&str> = c.split_whitespace().collect();
        if parts.len() == 2 && parts[0] == "tick" && parts[1] == "freeze" {
            tx.send(Inbound::TickFreeze {
                conn: 0,
                frozen: true,
            })
            .unwrap();
        } else if parts[0] == "tick" {
            tx.send(Inbound::TickStep {
                conn: 0,
                steps: parts[2].parse().unwrap(),
            })
            .unwrap();
        } else {
            tx.send(Inbound::Setblock {
                conn: 0,
                x: parts[1].parse().unwrap(),
                y: parts[2].parse().unwrap(),
                z: parts[3].parse().unwrap(),
                name: parts[4].to_string(),
            })
            .unwrap();
        }
    }
    // Hold the channel open across tick boundaries so the deadline-based
    // run loop flushes the burst before Disconnected ends it.
    std::thread::sleep(std::time::Duration::from_millis(150));
    drop(tx); // reader gone -> Disconnected -> run() exits after draining
    runner.join().unwrap();
    // Decode everything the viewer received, oracle apply() style.
    let mut writes: Vec<((i32, i32, i32), u32)> = Vec::new();
    let mut update_pkts = 0usize;
    for frame in rx_out.try_iter() {
        let Outbound::Frame { id, body } = frame else {
            continue;
        };
        if id == 0x08 && body.len() >= 9 {
            update_pkts += 1;
            let packed = i64::from_be_bytes(body[0..8].try_into().unwrap());
            let x = ((packed >> 38) & 0x3ff_ffff) << 38 >> 38;
            let z = ((packed >> 12) & 0x3ff_ffff) << 38 >> 38;
            let y = (packed & 0xfff) as i32;
            let mut st = 0u32;
            let mut sh = 0u32;
            let mut o = 8usize;
            while o < body.len() {
                let b = body[o];
                o += 1;
                st |= u32::from(b & 0x7f) << sh;
                sh += 7;
                if b & 0x80 == 0 {
                    break;
                }
            }
            writes.push(((x as i32, y, z as i32), st));
        } else if id == 0x56 && body.len() >= 8 {
            update_pkts += 1;
            let sec = i64::from_be_bytes(body[0..8].try_into().unwrap());
            let sx = (sec >> 42) & 0x3f_ffff;
            let sz = (sec >> 20) & 0x3f_ffff;
            let sy = sec & 0xf_ffff;
            let sx = (sx << 10) >> 10;
            let sz = (sz << 10) >> 10;
            let mut o = 8usize;
            let rd = |body: &[u8], o: &mut usize| -> u64 {
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
            };
            let count = rd(&body, &mut o);
            for _ in 0..count {
                let e = rd(&body, &mut o);
                let local = (e & 0xfff) as i32;
                let st = (e >> 12) as u32;
                let lx = (local >> 8) & 0xf;
                let lz = (local >> 4) & 0xf;
                let ly = local & 0xf;
                writes.push((
                    (
                        (sx * 16 + lx as i64) as i32,
                        (sy * 16 + ly as i64) as i32,
                        (sz * 16 + lz as i64) as i32,
                    ),
                    st,
                ));
            }
        }
    }
    let mut map = std::collections::BTreeMap::new();
    for (pos, st) in &writes {
        map.insert(*pos, *st);
    }
    eprintln!(
        "[sim] update packets: {}, writes: {}",
        update_pkts,
        writes.len()
    );
    for (i, (pos, st)) in writes.iter().enumerate() {
        if *pos == (10, 100, 10) {
            eprintln!("[sim] lever write #{}: {}", i, st);
        }
    }
    eprintln!("[sim] final lever map state: {:?}", map.get(&(10, 100, 10)));
    assert_eq!(
        map.get(&(10, 100, 10)),
        Some(&8439u32),
        "serve-loop burst must end with the lever ON (8439)"
    );
}

/// Replays the EXACT parity-redstone CI script through a real Game with a
/// registered viewer, captures the outbound packet stream, and decodes
/// every 0x08/0x56 exactly like the oracle's apply() (final map, last
/// write wins). Prints every write touching the lever at (10,100,10) so
/// the phantom-OFF broadcast (if any) is visible locally.
#[test]
fn broadcast_stream_trace() {
    let (mut g, rx) = harness();
    let script: Vec<&str> = vec![
        "tick freeze",
        "setblock 10 100 10 minecraft:lever[face=floor,facing=north,powered=false]",
        "tick step 1",
        "setblock 11 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 12 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 13 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 14 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 15 100 10 minecraft:redstone_torch",
        "tick step 1",
        "setblock 10 100 10 minecraft:lever[face=floor,facing=north,powered=true]",
        "tick step 1",
        "setblock 16 100 10 minecraft:repeater[facing=west,delay=1]",
        "tick step 1",
        "setblock 17 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 18 100 10 minecraft:comparator[facing=west,mode=compare,powered=false]",
        "tick step 1",
        "setblock 10 100 12 minecraft:lever[face=floor,facing=north,powered=true]",
        "tick step 1",
        "setblock 11 100 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 12 100 12 minecraft:comparator[facing=west,mode=compare,powered=false]",
        "tick step 1",
        "setblock 13 100 12 minecraft:comparator[facing=west,mode=subtract,powered=false]",
        "tick step 1",
        "setblock 13 100 13 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 14 100 13 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 15 100 13 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 16 100 13 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 17 100 13 minecraft:lever[face=floor,facing=north,powered=true]",
        "tick step 1",
        "setblock 20 100 10 minecraft:observer[facing=east]",
        "tick step 1",
        "setblock 21 100 10 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 19 100 10 minecraft:stone",
        "tick step 1",
        "setblock 19 100 10 minecraft:oak_planks",
        "tick step 1",
        "setblock 12 100 11 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 12 100 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 13 100 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 14 100 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 15 100 12 minecraft:stone",
        "tick step 1",
        "setblock 15 101 12 minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "tick step 1",
        "setblock 13 101 12 minecraft:stone",
        "tick step 1",
        "setblock 16 101 12 minecraft:stone",
        "tick step 10",
    ];
    // (pos, state) writes in stream order, mirroring the oracle decode.
    let mut writes: Vec<((i32, i32, i32), u32)> = Vec::new();
    let mut pkts = 0usize;
    let drain = |rx: &std::sync::mpsc::Receiver<Outbound>,
                 writes: &mut Vec<((i32, i32, i32), u32)>,
                 pkts: &mut usize| {
        while let Ok(frame) = rx.try_recv() {
            let Outbound::Frame { id, body } = frame else {
                continue;
            };
            *pkts += 1;
            if id == 0x08 && body.len() >= 9 {
                let packed = i64::from_be_bytes(body[0..8].try_into().unwrap());
                let x = ((packed >> 38) & 0x3ff_ffff) << 38 >> 38;
                let z = ((packed >> 12) & 0x3ff_ffff) << 38 >> 38;
                let y = (packed & 0xfff) as i32;
                let mut st = 0u32;
                let mut sh = 0u32;
                let mut o = 8usize;
                while o < body.len() {
                    let b = body[o];
                    o += 1;
                    st |= u32::from(b & 0x7f) << sh;
                    sh += 7;
                    if b & 0x80 == 0 {
                        break;
                    }
                }
                writes.push(((x as i32, y, z as i32), st));
            } else if id == 0x56 && body.len() >= 8 {
                let sec = i64::from_be_bytes(body[0..8].try_into().unwrap());
                let sx = (sec >> 42) & 0x3f_ffff;
                let sz = (sec >> 20) & 0x3f_ffff;
                let sy = sec & 0xf_ffff;
                let sx = (sx << 10) >> 10;
                let sz = (sz << 10) >> 10;
                let mut o = 8usize;
                let rd = |body: &[u8], o: &mut usize| -> u64 {
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
                };
                let count = rd(&body, &mut o);
                for _ in 0..count {
                    let e = rd(&body, &mut o);
                    let local = (e & 0xfff) as i32;
                    let st = (e >> 12) as u32;
                    let lx = (local >> 8) & 0xf;
                    let lz = (local >> 4) & 0xf;
                    let ly = local & 0xf;
                    writes.push((
                        (
                            (sx * 16 + lx as i64) as i32,
                            (sy * 16 + ly as i64) as i32,
                            (sz * 16 + lz as i64) as i32,
                        ),
                        st,
                    ));
                }
            }
        }
    };
    for (i, c) in script.iter().enumerate() {
        let before = writes.len();
        cmd(&mut g, c);
        // tick step N runs its ticks inside handle(); a setblock's updates
        // are flushed by the NEXT tick — run one to mirror the serve loop.
        if !c.starts_with("tick") {
            g.tick_once_for_test();
        }
        drain(&rx, &mut writes, &mut pkts);
        for (pos, st) in &writes[before..] {
            if *pos == (10, 100, 10) {
                eprintln!("[stream] after cmd #{} `{}`: lever -> {}", i, c, st);
            }
        }
    }
    // Flush anything left by the final tick step 10.
    g.tick_once_for_test();
    drain(&rx, &mut writes, &mut pkts);
    let mut map = std::collections::BTreeMap::new();
    for (pos, st) in &writes {
        map.insert(*pos, *st);
    }
    eprintln!(
        "[stream] total update packets: {}, writes: {}",
        pkts,
        writes.len()
    );
    eprintln!(
        "[stream] final lever map state: {:?}",
        map.get(&(10, 100, 10))
    );
    let reg = g.registry_snapshot_for_test();
    let mut finals: Vec<_> = map.iter().collect();
    finals.sort();
    for (pos, st) in finals {
        let label = reg
            .state_of(*st)
            .map(|(n, p)| format!("{n}[{p}]"))
            .unwrap_or_else(|| st.to_string());
        eprintln!("[stream] final {pos:?} = {label}");
    }
    eprintln!(
        "[stream] engine get_state_id: {:?}",
        g.block_label_for_test(10, 100, 10)
    );
    assert_eq!(
        map.get(&(10, 100, 10)),
        Some(&8439u32),
        "broadcast stream must end with the lever ON (8439)"
    );
}
