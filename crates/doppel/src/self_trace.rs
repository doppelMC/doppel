//! The self-trace golden: a scripted session driven through `Game` at a
//! fixed tick clock (no wall clock, no sockets), recording every outgoing
//! packet - id plus full body bytes - in order, per connection. The
//! committed recording is the per-commit referee for game-thread
//! refactors: a replay must be byte-identical, where the differential
//! oracle is only approximate.
//!
//! Regenerate with DOPPEL_REGEN_SELF_TRACE=1 after an INTENDED behavior
//! change; never to make an unintended diff pass.

use crate::game::{Game, Inbound, Outbound};
use crate::inventory::{ClickKind, ContainerClick, HashedStack};
use crate::WireChunk;

/// The committed recording, next to this file.
const GOLDEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/self_trace.golden");

/// A stone-floored world chunk (floor at y=99), the piston-test layout:
/// floor-mounted redstone needs support.
fn floor_chunk(x: i32) -> WireChunk {
    let mut w = WireChunk {
        x,
        z: 0,
        heightmaps: Vec::new(),
        sections: Vec::new(),
        block_entities: Vec::new(),
        light: Default::default(),
    };
    for sy in 0..24 {
        let block_states = if sy == 10 {
            // Stone at local y=3 (world y=99); YZX packing.
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
}

fn packed_pos(x: i32, y: i32, z: i32) -> i64 {
    (((x as i64) & 0x3ff_ffff) << 38) | (((z as i64) & 0x3ff_ffff) << 12) | (y as i64 & 0xfff)
}

/// The exact serverbound use_item_on bytes the oracle bot sends.
fn use_item_on_bytes(hand: i32, x: i32, y: i32, z: i32, face: i32, sequence: i32) -> Vec<u8> {
    let mut body = Vec::new();
    doppel_protocol::write_varint(&mut body, hand);
    body.extend_from_slice(&packed_pos(x, y, z).to_be_bytes());
    doppel_protocol::write_varint(&mut body, face);
    body.extend_from_slice(&0.5f32.to_be_bytes());
    body.extend_from_slice(&1.0f32.to_be_bytes());
    body.extend_from_slice(&0.5f32.to_be_bytes());
    body.push(0); // inside
    body.push(0); // world border
    doppel_protocol::write_varint(&mut body, sequence);
    body
}

/// The exact serverbound player_action bytes (action, pos, direction,
/// sequence share one body layout).
fn player_action_bytes(
    action: i32,
    x: i32,
    y: i32,
    z: i32,
    direction: i32,
    sequence: i32,
) -> Vec<u8> {
    let mut body = Vec::new();
    doppel_protocol::write_varint(&mut body, action);
    body.extend_from_slice(&packed_pos(x, y, z).to_be_bytes());
    doppel_protocol::write_varint(&mut body, direction);
    doppel_protocol::write_varint(&mut body, sequence);
    body
}

/// The scripted session recorder. Conn 0 is the actor (and a viewer);
/// conn 1 is a second viewer in range of the dig overlays (the digger
/// never receives its own destruction stages).
struct Tracer {
    game: Game,
    rx0: std::sync::mpsc::Receiver<Outbound>,
    rx1: std::sync::mpsc::Receiver<Outbound>,
    lines: Vec<String>,
    ticks: u32,
}

impl Tracer {
    fn new() -> Tracer {
        let (_tx, rx) = std::sync::mpsc::channel::<Inbound>();
        let mut game = Game::new(rx, None, None);
        assert!(game.registry_for_test(), "pins/blocks.json not found");
        for cx in [-1, 0, 1, 2] {
            game.seed_chunk_for_test(cx, 0, floor_chunk(cx));
        }
        let (tx0, rx0) = std::sync::mpsc::channel::<Outbound>();
        game.join_viewer_for_test(0, &[(-1, 0), (0, 0), (1, 0), (2, 0)], tx0);
        let (tx1, rx1) = std::sync::mpsc::channel::<Outbound>();
        game.join_viewer_for_test(1, &[(-1, 0), (0, 0), (1, 0), (2, 0)], tx1);
        Tracer {
            game,
            rx0,
            rx1,
            lines: Vec::new(),
            ticks: 0,
        }
    }

    /// Records one scripted step, then everything it queued.
    fn step(&mut self, label: &str, event: Inbound) {
        self.lines.push(format!("# {label}"));
        self.game.handle(event);
        self.drain();
    }

    /// One fixed-clock tick, then everything it queued.
    fn tick(&mut self) {
        self.ticks += 1;
        self.game.tick_once_for_test();
        self.lines.push(format!("T {}", self.ticks));
        self.drain();
    }

    fn ticks(&mut self, n: u32) {
        for _ in 0..n {
            self.tick();
        }
    }

    /// Drains both viewers in a fixed order (per-connection packet order
    /// is exact; the connection iteration at each send point is
    /// deterministic BTreeMap order).
    fn drain(&mut self) {
        while let Ok(frame) = self.rx0.try_recv() {
            if let Outbound::Frame { id, body } = frame {
                self.lines.push(format!("P0 {id:02x} {}", hex(&body)));
            }
        }
        while let Ok(frame) = self.rx1.try_recv() {
            if let Outbound::Frame { id, body } = frame {
                self.lines.push(format!("P1 {id:02x} {}", hex(&body)));
            }
        }
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn setblock(conn: u64, x: i32, y: i32, z: i32, name: &str) -> Inbound {
    Inbound::Setblock {
        conn,
        x,
        y,
        z,
        name: name.to_string(),
    }
}

fn click(menu: i32, state: i32, slot: i16, button: i8, kind: ClickKind) -> Inbound {
    Inbound::ContainerClick {
        conn: 0,
        click: ContainerClick {
            container_id: menu,
            state_id: state,
            slot_num: slot,
            button_num: button,
            kind,
            changed_slots: Vec::new(),
            carried: HashedStack::default(),
        },
    }
}

/// Runs the scripted session and returns the recording.
fn record_session() -> Vec<String> {
    let mut t = Tracer::new();

    // Setup: stand the actor near the work area, bring a witness in range.
    t.step(
        "tp actor to the work area",
        Inbound::Tp {
            conn: 0,
            x: 5.0,
            y: 101.0,
            z: 5.0,
        },
    );
    t.step(
        "tp witness next to the actor",
        Inbound::Tp {
            conn: 1,
            x: 5.0,
            y: 101.0,
            z: 6.0,
        },
    );
    t.step(
        "select hotbar slot 2",
        Inbound::SetCarriedItem { conn: 0, slot: 2 },
    );
    t.step("arm swing", Inbound::Punch { conn: 0 });
    t.step(
        "select hotbar slot 0",
        Inbound::SetCarriedItem { conn: 0, slot: 0 },
    );

    // A setblock: the write, then the tick-end broadcast.
    t.step("setblock dirt", setblock(0, 6, 100, 6, "minecraft:dirt"));
    t.tick();

    // A placement: give a stack, click the floor face with it.
    t.step(
        "give 64 stone",
        Inbound::Give {
            conn: 0,
            item: "minecraft:stone".to_string(),
            count: 64,
        },
    );
    t.step(
        "look straight ahead",
        Inbound::Rotated {
            conn: 0,
            yaw: 0.0,
            pitch: 0.0,
        },
    );
    let hit = crate::placement::parse_use_item_on(&use_item_on_bytes(0, 5, 99, 5, 1, 1))
        .expect("use_item_on bytes parse");
    t.step(
        "use_item_on: place stone on the floor",
        Inbound::UseItemOn {
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
        },
    );
    // The placement's spent stack syncs on the next tick's menu broadcast.
    t.tick();

    // A dig cycle: START, stage broadcasts across ticks, STOP past the
    // 0.7 finish threshold (immediate break), then the spawned drop.
    // Dirt bare-handed: 1.0/0.5/30 = one tick of progress in 15.
    let dig = |action: i32| {
        crate::dig::parse_player_action(&player_action_bytes(action, 6, 100, 6, 1, 2))
            .expect("player_action bytes parse")
    };
    t.step(
        "dig start (dirt)",
        Inbound::PlayerAction {
            conn: 0,
            act: dig(0),
        },
    );
    t.ticks(12);
    t.step(
        "dig stop past the threshold",
        Inbound::PlayerAction {
            conn: 0,
            act: dig(3),
        },
    );
    // The entangled window: a block write, a menu broadcast, and the
    // spawned drop's movement frames land in the same ticks.
    t.step(
        "setblock dirt beside the drop",
        setblock(0, 4, 100, 6, "minecraft:dirt"),
    );
    t.step(
        "give 8 dirt",
        Inbound::Give {
            conn: 0,
            item: "minecraft:dirt".to_string(),
            count: 8,
        },
    );
    // The drop settles: spawn pairing, fall, landing syncs.
    t.ticks(20);

    // Pickup: step onto the resting drop.
    t.step(
        "tp onto the drop",
        Inbound::Tp {
            conn: 0,
            x: 6.5,
            y: 100.0,
            z: 6.5,
        },
    );
    t.ticks(3);

    // A thrown drop: ACTION_DROP_ITEM spends one from the held stack.
    t.step(
        "drop one stone",
        Inbound::PlayerAction {
            conn: 0,
            act: dig(5),
        },
    );
    t.ticks(6);

    // A chest: block entity, open (menu + lid), one quick_move click,
    // close (lid settles).
    t.step(
        "setblock chest",
        setblock(
            0,
            8,
            100,
            6,
            "minecraft:chest[facing=north,type=single,waterlogged=false]",
        ),
    );
    t.tick();
    t.step(
        "opencontainer chest",
        Inbound::OpenContainer {
            conn: 0,
            x: 8,
            y: 100,
            z: 6,
        },
    );
    t.tick();
    t.step(
        "quick_move stone into the chest",
        click(1, 1, 54, 0, ClickKind::QuickMove),
    );
    t.tick();
    t.step(
        "container_close",
        Inbound::ContainerClose {
            conn: 0,
            container_id: 1,
        },
    );
    t.ticks(2);

    // A lever-flip redstone cascade: lever, wire chain, repeater (input
    // side on the wire, facing west), output wire; flip on, settle, flip
    // off, settle.
    for (i, spec) in [
        "minecraft:lever[face=floor,facing=north,powered=false]",
        "minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
        "minecraft:repeater[facing=west,delay=1,locked=false,powered=false]",
        "minecraft:redstone_wire[east=none,north=none,south=none,west=none]",
    ]
    .into_iter()
    .enumerate()
    {
        t.step("circuit placement", setblock(0, 5 + i as i32, 100, 8, spec));
        t.tick();
    }
    t.step(
        "lever ON",
        setblock(
            0,
            5,
            100,
            8,
            "minecraft:lever[face=floor,facing=north,powered=true]",
        ),
    );
    // The stepped-clock barrier drives the cascade inside handle().
    t.step("tick step 8", Inbound::TickStep { conn: 0, steps: 8 });
    t.ticks(1);
    t.step(
        "lever OFF",
        setblock(
            0,
            5,
            100,
            8,
            "minecraft:lever[face=floor,facing=north,powered=false]",
        ),
    );
    t.step("tick step 8", Inbound::TickStep { conn: 0, steps: 8 });
    t.ticks(1);

    t.lines
}

#[test]
fn self_trace_matches_golden() {
    let recorded = record_session().join("\n");
    let recorded = format!("{recorded}\n");
    if std::env::var_os("DOPPEL_REGEN_SELF_TRACE").is_some() {
        std::fs::write(GOLDEN, &recorded).expect("writing the golden recording");
        eprintln!("[self-trace] rewrote {}", GOLDEN);
        return;
    }
    let committed = std::fs::read_to_string(GOLDEN).unwrap_or_else(|e| {
        panic!("reading {GOLDEN}: {e} (regenerate with DOPPEL_REGEN_SELF_TRACE=1)")
    });
    // Line endings stay stable across git checkout configurations; the
    // packet bodies themselves compare byte-exact.
    let normalize = |s: &str| {
        s.lines()
            .map(|l| l.trim_end_matches('\r'))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let recorded = normalize(&recorded);
    let committed = normalize(&committed);
    if recorded != committed {
        // Show the first divergence point.
        let mut line = 1;
        let mut column = 0;
        for (a, b) in recorded.lines().zip(committed.lines()) {
            if a != b {
                break;
            }
            line += 1;
            column = a.len();
        }
        panic!(
            "self-trace diverges from the golden at line {line} column {column}\n\
             golden:      {}\n\
             recorded:    {}\n\
             (regenerate only after an INTENDED behavior change: \
              DOPPEL_REGEN_SELF_TRACE=1)",
            committed.lines().nth(line - 1).unwrap_or("<eof>"),
            recorded.lines().nth(line - 1).unwrap_or("<eof>"),
        );
    }
}
