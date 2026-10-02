//! Right-click block placement (`use_item_on`): the serverbound parse,
//! the block-item table, and the geometry helpers that turn a clicked face
//! plus the placer's view direction into a block-state spec. The game-side
//! handler lives in game.rs under the placement hooks banner.

use std::collections::HashMap;
use std::sync::OnceLock;

use anyhow::{bail, Context, Result};
use doppel_protocol::Reader;

use crate::game::{DIR_DOWN, DIR_EAST, DIR_NORTH, DIR_SOUTH, DIR_UP, DIR_WEST};

/// Serverbound `use_item_on`, play state. The 26.2 protocol notes list
/// 0x42; 26.3 inserts `punch` at 0x2e, shifting every later serverbound id
/// +1 (the same shift that moves set_carried_item 0x35 -> 0x36). The
/// registration order in the pinned 26.3 jar confirms 0x43.
pub const SERVERBOUND_USE_ITEM_ON: i32 = 0x43;

/// Serverbound `move_player_rot`: yaw f32, pitch f32, flags u8.
pub const SERVERBOUND_MOVE_PLAYER_ROT: i32 = 0x20;

// ---------------------------------------------------------------------
// Parse
// ---------------------------------------------------------------------

/// One decoded use_item_on. Cursor floats are the in-cell click point;
/// placement geometry uses the face only (sub-block hit shaping is future
/// work).
///
/// NOTE(placement): `sequence` would echo in a block_changed_ack; this
/// build sends none (client prediction self-corrects from the tick
/// broadcasts). 26.3 registration order puts that clientbound packet at
/// 0x04, unpinned on the wire.
pub struct UseItemOn {
    pub x: i32,
    pub y: i32,
    pub z: i32,
    /// Clicked face, Direction 3D data value (0=down .. 5=east).
    pub face: u8,
    pub cursor_x: f32,
    pub cursor_y: f32,
    pub cursor_z: f32,
    /// 0 = main hand, 1 = offhand.
    pub hand: u8,
    pub sequence: i32,
}

/// Parses a serverbound use_item_on body (after the packet id):
/// hand VarInt, block pos i64 (packed BlockPos), face VarInt, cursor
/// xyz f32, inside Bool, worldBorder Bool, sequence VarInt.
pub fn parse_use_item_on(body: &[u8]) -> Result<UseItemOn> {
    let mut r = Reader::new(body);
    let hand = r.read_varint().context("hand")?;
    if !(0..=1).contains(&hand) {
        bail!("use_item_on hand {hand}");
    }
    let packed = r.read_i64().context("block pos")?;
    let face = r.read_varint().context("face")?;
    if !(0..=5).contains(&face) {
        bail!("use_item_on face {face}");
    }
    let cursor_x = r.read_f32().context("cursor x")?;
    let cursor_y = r.read_f32().context("cursor y")?;
    let cursor_z = r.read_f32().context("cursor z")?;
    let _inside = r.read_u8().context("inside")?;
    let _world_border = r.read_u8().context("world border")?;
    let sequence = r.read_varint().context("sequence")?;
    if r.remaining() != 0 {
        bail!("{} trailing bytes in use_item_on", r.remaining());
    }
    let (x, y, z) = unpack_block_pos(packed);
    Ok(UseItemOn {
        x,
        y,
        z,
        face: face as u8,
        cursor_x,
        cursor_y,
        cursor_z,
        hand: hand as u8,
        sequence,
    })
}

/// Vanilla BlockPos packing on the wire:
/// (x & 0x3FFFFFF) << 38 | (z & 0x3FFFFFF) << 12 | y & 0xFFF, each field
/// sign-extended from its bit width.
fn unpack_block_pos(packed: i64) -> (i32, i32, i32) {
    let x = sign_extend(packed >> 38, 26) as i32;
    let z = sign_extend((packed >> 12) & 0x3ff_ffff, 26) as i32;
    let y = sign_extend(packed & 0xfff, 12) as i32;
    (x, y, z)
}

fn sign_extend(value: i64, bits: u32) -> i64 {
    let shift = 64 - bits;
    (value << shift) >> shift
}

// ---------------------------------------------------------------------
// Direction helpers (face ids are the Direction 3D data values)
// ---------------------------------------------------------------------

/// The face's unit step.
pub fn face_step(face: u8) -> (i32, i32, i32) {
    match face {
        DIR_DOWN => (0, -1, 0),
        DIR_UP => (0, 1, 0),
        DIR_NORTH => (0, 0, -1),
        DIR_SOUTH => (0, 0, 1),
        DIR_WEST => (-1, 0, 0),
        _ => (1, 0, 0),
    }
}

pub fn face_name(face: u8) -> &'static str {
    match face {
        DIR_DOWN => "down",
        DIR_UP => "up",
        DIR_NORTH => "north",
        DIR_SOUTH => "south",
        DIR_WEST => "west",
        _ => "east",
    }
}

/// Opposite 3D id (the enum orders each axis pair adjacently).
pub fn face_opposite(face: u8) -> u8 {
    face ^ 1
}

/// The horizontal quadrant the yaw points at: floor(yaw/90 + 0.5) & 3
/// indexes [south, west, north, east] (the 2D data-value order).
pub fn quadrant(yaw: f32) -> u8 {
    match (yaw / 90.0 + 0.5).floor() as i32 & 3 {
        0 => DIR_SOUTH,
        1 => DIR_WEST,
        2 => DIR_NORTH,
        _ => DIR_EAST,
    }
}

/// The direction closest to the view vector (the head of vanilla's
/// orderedByNearest): the dominant of |sin yaw| vs |cos yaw| on the
/// horizontal plane, with |sin pitch| scaled by cos pitch competing for
/// the vertical.
pub fn nearest_look(yaw: f32, pitch: f32) -> u8 {
    let yaw_sin = (-yaw.to_radians()).sin();
    let yaw_cos = (-yaw.to_radians()).cos();
    let pitch_sin = pitch.to_radians().sin();
    let pitch_cos = pitch.to_radians().cos();
    let vertical = if pitch_sin < 0.0 { DIR_UP } else { DIR_DOWN };
    let x_yaw = yaw_sin.abs();
    let z_yaw = yaw_cos.abs();
    let y_mag = pitch_sin.abs();
    let x_mag = x_yaw * pitch_cos;
    let z_mag = z_yaw * pitch_cos;
    if x_yaw > z_yaw {
        if y_mag > x_mag {
            vertical
        } else if yaw_sin > 0.0 {
            DIR_EAST
        } else {
            DIR_WEST
        }
    } else if y_mag > z_mag {
        vertical
    } else if yaw_cos > 0.0 {
        DIR_SOUTH
    } else {
        DIR_NORTH
    }
}

// ---------------------------------------------------------------------
// Block-item table
// ---------------------------------------------------------------------

/// How a block item resolves to a placed state.
#[derive(Clone, Copy)]
pub enum Form {
    /// No orientation props; the default state lands.
    Plain,
    /// Fresh redstone dust: all sides disconnected, no power.
    Wire,
    /// facing = the horizontal quadrant opposite the placer (chests,
    /// furnaces, repeaters, comparators, carved pumpkins). `props` carries
    /// the family's remaining full-prop fill (registry default fill misses
    /// keys like chest type).
    QuadrantOpposite { props: &'static str },
    /// facing = the nearest look direction (observer).
    Look { props: &'static str },
    /// facing = the opposite of the nearest look direction (pistons,
    /// dispensers, droppers, barrels, shulker boxes).
    LookOpposite { props: &'static str },
    /// Top face -> the standing block; side face -> the wall variant
    /// facing out of the clicked face (torches); bottom face refuses.
    StandingAndWall { wall: &'static str },
    /// face = floor/ceiling/wall from the clicked face; a wall mount
    /// faces out of the clicked face (levers).
    Attached,
    /// facing = opposite of the clicked face, vertical clamped to down
    /// (hopper funnels into the block it was clicked on).
    Funnel,
    /// axis = the clicked face's axis (logs).
    Pillar,
}

impl Form {
    /// The state spec for one placement, or None when the form cannot
    /// attach to the clicked face. `props` entries must carry the
    /// family's full prop set: the registry's partial-props fill has no
    /// defaults for keys like chest type, hopper enabled, or diode mode.
    pub fn spec(&self, block: &str, face: u8, yaw: f32, pitch: f32) -> Option<String> {
        let facing = |dir: u8| face_name(dir);
        Some(match self {
            Form::Plain => block.to_string(),
            Form::Wire => {
                "minecraft:redstone_wire[east=none,north=none,power=0,south=none,west=none]"
                    .to_string()
            }
            Form::QuadrantOpposite { props } => {
                format!(
                    "{block}[facing={}{}]",
                    facing(face_opposite(quadrant(yaw))),
                    props
                )
            }
            Form::Look { props } => {
                format!(
                    "{block}[facing={}{}]",
                    facing(nearest_look(yaw, pitch)),
                    props
                )
            }
            Form::LookOpposite { props } => format!(
                "{block}[facing={}{}]",
                facing(face_opposite(nearest_look(yaw, pitch))),
                props
            ),
            Form::StandingAndWall { wall } => match face {
                DIR_UP => block.to_string(),
                DIR_DOWN => return None,
                side => format!("{wall}[facing={}]", facing(side)),
            },
            Form::Attached => match face {
                DIR_UP => format!("{block}[face=floor,facing={}]", facing(quadrant(yaw))),
                DIR_DOWN => format!("{block}[face=ceiling,facing={}]", facing(quadrant(yaw))),
                side => format!("{block}[face=wall,facing={}]", facing(side)),
            },
            Form::Funnel => {
                let dir = if face == DIR_UP || face == DIR_DOWN {
                    DIR_DOWN
                } else {
                    face_opposite(face)
                };
                format!("{block}[enabled=true,facing={}]", facing(dir))
            }
            Form::Pillar => {
                let axis = match face {
                    DIR_UP | DIR_DOWN => "y",
                    DIR_NORTH | DIR_SOUTH => "z",
                    _ => "x",
                };
                format!("{block}[axis={axis}]")
            }
        })
    }
}

/// item name -> (block name, form) for the placeable families. Registry
/// ids resolve through ItemTable. The full vanilla items->blocks mapping
/// is a registry pin for later; slabs/stairs stay out until two-height
/// geometry exists.
const BLOCK_ITEMS: &[(&str, &str, Form)] = &[
    ("minecraft:stone", "minecraft:stone", Form::Plain),
    ("minecraft:granite", "minecraft:granite", Form::Plain),
    (
        "minecraft:polished_granite",
        "minecraft:polished_granite",
        Form::Plain,
    ),
    ("minecraft:diorite", "minecraft:diorite", Form::Plain),
    (
        "minecraft:polished_diorite",
        "minecraft:polished_diorite",
        Form::Plain,
    ),
    ("minecraft:andesite", "minecraft:andesite", Form::Plain),
    (
        "minecraft:polished_andesite",
        "minecraft:polished_andesite",
        Form::Plain,
    ),
    ("minecraft:deepslate", "minecraft:deepslate", Form::Plain),
    (
        "minecraft:cobbled_deepslate",
        "minecraft:cobbled_deepslate",
        Form::Plain,
    ),
    (
        "minecraft:grass_block",
        "minecraft:grass_block",
        Form::Plain,
    ),
    ("minecraft:dirt", "minecraft:dirt", Form::Plain),
    (
        "minecraft:cobblestone",
        "minecraft:cobblestone",
        Form::Plain,
    ),
    ("minecraft:oak_planks", "minecraft:oak_planks", Form::Plain),
    ("minecraft:glass", "minecraft:glass", Form::Plain),
    (
        "minecraft:crafting_table",
        "minecraft:crafting_table",
        Form::Plain,
    ),
    (
        "minecraft:slime_block",
        "minecraft:slime_block",
        Form::Plain,
    ),
    (
        "minecraft:honey_block",
        "minecraft:honey_block",
        Form::Plain,
    ),
    ("minecraft:oak_log", "minecraft:oak_log", Form::Pillar),
    ("minecraft:redstone", "minecraft:redstone_wire", Form::Wire),
    // Chest family: the double-chest pairing needs neighbor context; every
    // placement starts single.
    (
        "minecraft:chest",
        "minecraft:chest",
        Form::QuadrantOpposite {
            props: ",type=single,waterlogged=false",
        },
    ),
    (
        "minecraft:trapped_chest",
        "minecraft:trapped_chest",
        Form::QuadrantOpposite {
            props: ",type=single,waterlogged=false",
        },
    ),
    (
        "minecraft:ender_chest",
        "minecraft:ender_chest",
        Form::QuadrantOpposite {
            props: ",type=single,waterlogged=false",
        },
    ),
    (
        "minecraft:furnace",
        "minecraft:furnace",
        Form::QuadrantOpposite {
            props: ",lit=false",
        },
    ),
    (
        "minecraft:repeater",
        "minecraft:repeater",
        Form::QuadrantOpposite {
            props: ",delay=1,locked=false,powered=false",
        },
    ),
    (
        "minecraft:comparator",
        "minecraft:comparator",
        Form::QuadrantOpposite {
            props: ",mode=compare,powered=false",
        },
    ),
    (
        "minecraft:carved_pumpkin",
        "minecraft:carved_pumpkin",
        Form::QuadrantOpposite { props: "" },
    ),
    (
        "minecraft:observer",
        "minecraft:observer",
        Form::Look {
            props: ",powered=false",
        },
    ),
    (
        "minecraft:piston",
        "minecraft:piston",
        Form::LookOpposite {
            props: ",extended=false",
        },
    ),
    (
        "minecraft:sticky_piston",
        "minecraft:sticky_piston",
        Form::LookOpposite {
            props: ",extended=false",
        },
    ),
    (
        "minecraft:dispenser",
        "minecraft:dispenser",
        Form::LookOpposite {
            props: ",triggered=false",
        },
    ),
    (
        "minecraft:dropper",
        "minecraft:dropper",
        Form::LookOpposite {
            props: ",triggered=false",
        },
    ),
    (
        "minecraft:barrel",
        "minecraft:barrel",
        Form::LookOpposite { props: "" },
    ),
    (
        "minecraft:shulker_box",
        "minecraft:shulker_box",
        Form::LookOpposite { props: "" },
    ),
    ("minecraft:hopper", "minecraft:hopper", Form::Funnel),
    (
        "minecraft:torch",
        "minecraft:torch",
        Form::StandingAndWall {
            wall: "minecraft:wall_torch",
        },
    ),
    (
        "minecraft:redstone_torch",
        "minecraft:redstone_torch",
        Form::StandingAndWall {
            wall: "minecraft:redstone_wall_torch",
        },
    ),
    ("minecraft:lever", "minecraft:lever", Form::Attached),
];

/// (block name, form) for a placeable item registry id.
pub fn block_item_form(item: i32) -> Option<(&'static str, Form)> {
    static TABLE: OnceLock<HashMap<i32, (&'static str, Form)>> = OnceLock::new();
    TABLE
        .get_or_init(|| {
            BLOCK_ITEMS
                .iter()
                .filter_map(|(item_name, block, form)| {
                    crate::inventory::item_id(item_name).map(|id| (id, (*block, *form)))
                })
                .collect()
        })
        .get(&item)
        .copied()
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Game, Inbound, Outbound};
    use crate::inventory::{
        item_id, ClickKind, ContainerClick, HashedStack, ItemStack, PACKET_CONTAINER_SET_SLOT,
    };
    use crate::WireChunk;

    /// A game with stone-floored chunks (floor y=99) and one viewer, the
    /// piston-test layout.
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

    fn give(g: &mut Game, item: &str, count: i32) {
        g.handle(Inbound::Give {
            conn: 0,
            item: item.to_string(),
            count,
        });
    }

    fn look(g: &mut Game, yaw: f32, pitch: f32) {
        g.handle(Inbound::Rotated {
            conn: 0,
            yaw,
            pitch,
        });
    }

    fn use_on(g: &mut Game, x: i32, y: i32, z: i32, face: u8, hand: u8) {
        g.handle(Inbound::UseItemOn {
            conn: 0,
            x,
            y,
            z,
            face,
            cursor_x: 0.5,
            cursor_y: 1.0,
            cursor_z: 0.5,
            hand,
            sequence: 1,
        });
    }

    fn click_swap(g: &mut Game, menu_slot: i16, button: i8) {
        g.handle(Inbound::ContainerClick {
            conn: 0,
            click: ContainerClick {
                container_id: 0,
                state_id: 1,
                slot_num: menu_slot,
                button_num: button,
                kind: ClickKind::Swap,
                changed_slots: Vec::new(),
                carried: HashedStack::default(),
            },
        });
    }

    /// One game tick; returns that tick's block broadcasts as
    /// `localhex=name[props]` strings.
    fn tick(g: &mut Game, rx: &std::sync::mpsc::Receiver<Outbound>) -> Vec<String> {
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

    fn at(g: &Game, x: i32, y: i32, z: i32) -> String {
        g.block_label_for_test(x, y, z)
    }

    /// The next queued container_set_slot frame, decoded to
    /// (containerId, stateId, menu slot, stack).
    fn next_set_slot(
        rx: &std::sync::mpsc::Receiver<Outbound>,
    ) -> Option<(i32, i32, i16, Option<ItemStack>)> {
        while let Ok(frame) = rx.try_recv() {
            if let Outbound::Frame { id, body } = frame {
                if id != PACKET_CONTAINER_SET_SLOT {
                    continue;
                }
                let mut r = Reader::new(&body);
                let container_id = r.read_varint().ok()?;
                let state_id = r.read_varint().ok()?;
                let slot = r.read_u16().ok()? as i16;
                let stack = crate::inventory::decode_item_stack(&mut r).ok()?;
                return Some((container_id, state_id, slot, stack));
            }
        }
        None
    }

    // -- parse ---------------------------------------------------------

    fn packed_pos(x: i32, y: i32, z: i32) -> i64 {
        (((x as i64) & 0x3ff_ffff) << 38) | (((z as i64) & 0x3ff_ffff) << 12) | (y as i64 & 0xfff)
    }

    fn use_item_on_bytes(hand: i32, x: i32, y: i32, z: i32, face: i32, sequence: i32) -> Vec<u8> {
        let mut body = Vec::new();
        doppel_protocol::write_varint(&mut body, hand);
        body.extend(&packed_pos(x, y, z).to_be_bytes());
        doppel_protocol::write_varint(&mut body, face);
        body.extend(&0.5f32.to_be_bytes());
        body.extend(&1.0f32.to_be_bytes());
        body.extend(&0.5f32.to_be_bytes());
        body.push(0);
        body.push(0);
        doppel_protocol::write_varint(&mut body, sequence);
        body
    }

    #[test]
    fn parse_golden_and_negative_coords() {
        let hit = parse_use_item_on(&use_item_on_bytes(0, 10, 100, 7, 1, 42)).unwrap();
        assert_eq!((hit.x, hit.y, hit.z), (10, 100, 7));
        assert_eq!(hit.face, DIR_UP);
        assert_eq!(hit.hand, 0);
        assert_eq!(hit.sequence, 42);
        assert_eq!((hit.cursor_x, hit.cursor_y, hit.cursor_z), (0.5, 1.0, 0.5));
        let hit = parse_use_item_on(&use_item_on_bytes(1, -5, -70, -33, 5, 1)).unwrap();
        assert_eq!((hit.x, hit.y, hit.z), (-5, -70, -33));
        assert_eq!(hit.face, DIR_EAST);
        assert_eq!(hit.hand, 1);
    }

    #[test]
    fn parse_rejects_bad_face_hand_and_trailing() {
        assert!(parse_use_item_on(&use_item_on_bytes(0, 0, 0, 0, 6, 1)).is_err());
        assert!(parse_use_item_on(&use_item_on_bytes(2, 0, 0, 0, 1, 1)).is_err());
        let mut trailing = use_item_on_bytes(0, 0, 0, 0, 1, 1);
        trailing.push(0);
        assert!(parse_use_item_on(&trailing).is_err());
        assert!(parse_use_item_on(&[0x00]).is_err());
    }

    #[test]
    fn direction_helpers() {
        assert_eq!(quadrant(0.0), DIR_SOUTH);
        assert_eq!(quadrant(44.9), DIR_SOUTH);
        assert_eq!(quadrant(45.0), DIR_WEST);
        assert_eq!(quadrant(180.0), DIR_NORTH);
        assert_eq!(quadrant(-180.0), DIR_NORTH);
        assert_eq!(quadrant(-90.0), DIR_EAST);
        assert_eq!(nearest_look(0.0, 0.0), DIR_SOUTH);
        assert_eq!(nearest_look(90.0, 0.0), DIR_WEST);
        assert_eq!(nearest_look(-90.0, 0.0), DIR_EAST);
        assert_eq!(nearest_look(180.0, 0.0), DIR_NORTH);
        assert_eq!(nearest_look(0.0, 90.0), DIR_DOWN);
        assert_eq!(nearest_look(0.0, -90.0), DIR_UP);
        // The vertical wins only past 45 degrees (|sin p| vs |cos p|).
        assert_eq!(nearest_look(0.0, 40.0), DIR_SOUTH);
        assert_eq!(nearest_look(0.0, 50.0), DIR_DOWN);
        assert_eq!(nearest_look(-90.0, 20.0), DIR_EAST);
        assert_eq!(nearest_look(-90.0, 60.0), DIR_DOWN);
    }

    // -- placement -----------------------------------------------------

    #[test]
    fn places_on_every_face() {
        let (mut g, rx) = harness();
        give(&mut g, "minecraft:stone", 64);
        // Top of the floor: target is the circuit plane y=100.
        use_on(&mut g, 5, 99, 5, DIR_UP, 0);
        assert_eq!(at(&g, 5, 100, 5), "minecraft:stone[]");
        // The four sides of the placed block.
        use_on(&mut g, 5, 100, 5, DIR_NORTH, 0);
        assert_eq!(at(&g, 5, 100, 4), "minecraft:stone[]");
        use_on(&mut g, 5, 100, 5, DIR_SOUTH, 0);
        assert_eq!(at(&g, 5, 100, 6), "minecraft:stone[]");
        use_on(&mut g, 5, 100, 5, DIR_WEST, 0);
        assert_eq!(at(&g, 4, 100, 5), "minecraft:stone[]");
        use_on(&mut g, 5, 100, 5, DIR_EAST, 0);
        assert_eq!(at(&g, 6, 100, 5), "minecraft:stone[]");
        // Bottom face points into the occupied floor cell: refused.
        use_on(&mut g, 5, 100, 5, DIR_DOWN, 0);
        assert_eq!(at(&g, 5, 99, 5), "minecraft:stone[]");
        // Five placements consumed five items; the refused click did not.
        // Each placement queues its own slot sync; the last shows the
        // final count.
        let mut last_slot_sync = None;
        while let Some(frame) = next_set_slot(&rx) {
            last_slot_sync = Some(frame);
        }
        let (container_id, _, slot, stack) = last_slot_sync.expect("slot syncs queued");
        assert_eq!(container_id, 0);
        assert_eq!(slot, 36, "hotbar 0 presents as menu slot 36");
        let stack = stack.expect("59 left after five placements");
        assert_eq!(
            (stack.count(), stack.item()),
            (59, item_id("minecraft:stone").unwrap())
        );
        // The flush broadcasts the placements like any setblock.
        let frames = tick(&mut g, &rx);
        assert_eq!(frames.len(), 5, "{frames:?}");
    }

    #[test]
    fn occupied_target_refused_without_decrement() {
        let (mut g, rx) = harness();
        give(&mut g, "minecraft:stone", 2);
        use_on(&mut g, 5, 99, 5, DIR_UP, 0);
        assert_eq!(at(&g, 5, 100, 5), "minecraft:stone[]");
        let (_, _, _, stack) = next_set_slot(&rx).unwrap();
        assert_eq!(stack.map(|s| s.count()), Some(1));
        // Click the same floor face again: the target holds the first block.
        use_on(&mut g, 5, 99, 5, DIR_UP, 0);
        assert_eq!(at(&g, 5, 100, 5), "minecraft:stone[]");
        assert!(
            next_set_slot(&rx).is_none(),
            "no slot sync for a refused click"
        );
    }

    #[test]
    fn stack_decrements_to_empty() {
        let (mut g, rx) = harness();
        give(&mut g, "minecraft:stone", 1);
        use_on(&mut g, 5, 99, 5, DIR_UP, 0);
        assert_eq!(at(&g, 5, 100, 5), "minecraft:stone[]");
        let (container_id, _, slot, stack) = next_set_slot(&rx).unwrap();
        assert_eq!(container_id, 0);
        assert_eq!(slot, 36);
        assert_eq!(stack, None, "empty slot at count zero");
        // Placing again with an empty hand changes nothing.
        use_on(&mut g, 6, 99, 6, DIR_UP, 0);
        assert_eq!(at(&g, 6, 100, 6), "minecraft:air[]");
    }

    #[test]
    fn non_block_item_refuses() {
        let (mut g, rx) = harness();
        give(&mut g, "minecraft:stick", 5);
        use_on(&mut g, 5, 99, 5, DIR_UP, 0);
        assert_eq!(at(&g, 5, 100, 5), "minecraft:air[]");
        assert!(next_set_slot(&rx).is_none());
    }

    #[test]
    fn orientable_quadrant_and_look_families() {
        let (mut g, _rx) = harness();
        // Facing west (yaw 90): the chest front points back east.
        give(&mut g, "minecraft:chest", 1);
        look(&mut g, 90.0, 0.0);
        use_on(&mut g, 5, 99, 5, DIR_UP, 0);
        assert_eq!(
            at(&g, 5, 100, 5),
            "minecraft:chest[facing=east,type=single,waterlogged=false]"
        );
        // Pistons face the placer: looking south -> facing north.
        give(&mut g, "minecraft:piston", 1);
        look(&mut g, 0.0, 0.0);
        use_on(&mut g, 6, 99, 6, DIR_UP, 0);
        assert_eq!(
            at(&g, 6, 100, 6),
            "minecraft:piston[extended=false,facing=north]"
        );
        // Looking down: the piston points up.
        give(&mut g, "minecraft:piston", 1);
        look(&mut g, 0.0, 90.0);
        use_on(&mut g, 7, 99, 7, DIR_UP, 0);
        assert_eq!(
            at(&g, 7, 100, 7),
            "minecraft:piston[extended=false,facing=up]"
        );
        // The observer watches where the placer looks.
        give(&mut g, "minecraft:observer", 1);
        look(&mut g, -90.0, 0.0);
        use_on(&mut g, 8, 99, 8, DIR_UP, 0);
        assert_eq!(
            at(&g, 8, 100, 8),
            "minecraft:observer[facing=east,powered=false]"
        );
        // Repeater input side faces the placer (quadrant opposite).
        give(&mut g, "minecraft:repeater", 1);
        look(&mut g, 180.0, 0.0);
        use_on(&mut g, 9, 99, 9, DIR_UP, 0);
        assert_eq!(
            at(&g, 9, 100, 9),
            "minecraft:repeater[delay=1,facing=south,locked=false,powered=false]"
        );
        // The hopper funnels into the block it was clicked on.
        give(&mut g, "minecraft:hopper", 1);
        look(&mut g, 0.0, 0.0);
        use_on(&mut g, 10, 99, 10, DIR_UP, 0);
        assert_eq!(
            at(&g, 10, 100, 10),
            "minecraft:hopper[enabled=true,facing=down]"
        );
        // Log axis follows the clicked face.
        give(&mut g, "minecraft:oak_log", 1);
        use_on(&mut g, 11, 99, 11, DIR_UP, 0);
        assert_eq!(at(&g, 11, 100, 11), "minecraft:oak_log[axis=y]");
        give(&mut g, "minecraft:oak_log", 1);
        use_on(&mut g, 11, 100, 11, DIR_NORTH, 0);
        assert_eq!(at(&g, 11, 100, 10), "minecraft:oak_log[axis=z]");
        give(&mut g, "minecraft:oak_log", 1);
        use_on(&mut g, 11, 100, 11, DIR_EAST, 0);
        assert_eq!(at(&g, 12, 100, 11), "minecraft:oak_log[axis=x]");
    }

    #[test]
    fn torch_floor_wall_and_refused_ceiling() {
        let (mut g, _rx) = harness();
        give(&mut g, "minecraft:torch", 2);
        use_on(&mut g, 5, 99, 5, DIR_UP, 0);
        assert_eq!(at(&g, 5, 100, 5), "minecraft:torch[]");
        // Against the north side: a wall torch pointing north (the plain
        // wall torch carries no lit prop).
        use_on(&mut g, 5, 100, 5, DIR_NORTH, 0);
        assert_eq!(at(&g, 5, 100, 4), "minecraft:wall_torch[facing=north]");
        // Torches do not hang from ceilings: the floor's bottom face
        // targets air, and the form still refuses.
        use_on(&mut g, 5, 99, 5, DIR_DOWN, 0);
        assert_eq!(at(&g, 5, 98, 5), "minecraft:air[]");
        // The redstone variant shares the rule.
        give(&mut g, "minecraft:redstone_torch", 2);
        use_on(&mut g, 6, 99, 6, DIR_UP, 0);
        assert_eq!(at(&g, 6, 100, 6), "minecraft:redstone_torch[lit=true]");
        use_on(&mut g, 6, 100, 6, DIR_WEST, 0);
        assert_eq!(
            at(&g, 5, 100, 6),
            "minecraft:redstone_wall_torch[facing=west,lit=true]"
        );
    }

    #[test]
    fn lever_face_follows_clicked_face() {
        let (mut g, _rx) = harness();
        give(&mut g, "minecraft:lever", 3);
        look(&mut g, 0.0, 0.0);
        use_on(&mut g, 5, 99, 5, DIR_UP, 0);
        assert_eq!(
            at(&g, 5, 100, 5),
            "minecraft:lever[face=floor,facing=south,powered=false]"
        );
        use_on(&mut g, 5, 100, 5, DIR_EAST, 0);
        assert_eq!(
            at(&g, 6, 100, 5),
            "minecraft:lever[face=wall,facing=east,powered=false]"
        );
        // Ceiling mount: click the bottom of a floating scaffold block.
        g.handle(Inbound::Setblock {
            conn: 0,
            x: 7,
            y: 101,
            z: 5,
            name: "minecraft:stone".to_string(),
        });
        use_on(&mut g, 7, 101, 5, DIR_DOWN, 0);
        assert_eq!(
            at(&g, 7, 100, 5),
            "minecraft:lever[face=ceiling,facing=south,powered=false]"
        );
    }

    #[test]
    fn offhand_places_when_main_is_not_a_block() {
        let (mut g, rx) = harness();
        give(&mut g, "minecraft:stick", 3);
        give(&mut g, "minecraft:stone", 5);
        // Hotbar 1 (menu 37) holds the stone; swap it to the offhand.
        click_swap(&mut g, 37, 40);
        use_on(&mut g, 5, 99, 5, DIR_UP, 1);
        assert_eq!(at(&g, 5, 100, 5), "minecraft:stone[]");
        let (_, _, slot, stack) = next_set_slot(&rx).unwrap();
        assert_eq!(slot, 45, "offhand presents as menu slot 45");
        assert_eq!(stack.map(|s| s.count()), Some(4));
    }

    #[test]
    fn offhand_refused_when_main_holds_a_block() {
        let (mut g, rx) = harness();
        give(&mut g, "minecraft:stone", 5);
        give(&mut g, "minecraft:stick", 3);
        click_swap(&mut g, 37, 40);
        use_on(&mut g, 5, 99, 5, DIR_UP, 1);
        assert_eq!(at(&g, 5, 100, 5), "minecraft:air[]");
        assert!(next_set_slot(&rx).is_none());
    }

    /// NOTE(placement): an unsupported target is placed and popped by the
    /// next tick's survival check, consuming the item; vanilla refuses
    /// the placement upfront and keeps the stack.
    #[test]
    fn unsupported_wire_pops_next_tick() {
        let (mut g, rx) = harness();
        give(&mut g, "minecraft:stone", 2);
        use_on(&mut g, 6, 99, 6, DIR_UP, 0);
        use_on(&mut g, 6, 100, 6, DIR_UP, 0);
        assert_eq!(at(&g, 6, 101, 6), "minecraft:stone[]");
        let _ = tick(&mut g, &rx);
        // The north side of the pillar: air below the target cell.
        give(&mut g, "minecraft:redstone", 1);
        use_on(&mut g, 6, 101, 6, DIR_NORTH, 0);
        assert_eq!(
            at(&g, 6, 101, 5),
            "minecraft:redstone_wire[east=none,north=none,power=0,south=none,west=none]"
        );
        let (_, _, _, stack) = next_set_slot(&rx).unwrap();
        assert_eq!(stack, None, "the dust was consumed");
        let frames = tick(&mut g, &rx);
        // The pop overrides the placement in the same per-tick dedup.
        assert_eq!(frames, vec!["655=minecraft:air[]"]);
        assert_eq!(at(&g, 6, 101, 5), "minecraft:air[]");
    }

    #[test]
    fn glass_places_as_plain_block() {
        let (mut g, _rx) = harness();
        give(&mut g, "minecraft:glass", 1);
        use_on(&mut g, 5, 99, 5, DIR_UP, 0);
        assert_eq!(at(&g, 5, 100, 5), "minecraft:glass[]");
    }

    #[test]
    fn creative_placement_keeps_the_stack() {
        let (mut g, rx) = harness();
        give(&mut g, "minecraft:stone", 1);
        g.set_creative_for_test(0, true);
        use_on(&mut g, 5, 99, 5, DIR_UP, 0);
        assert_eq!(at(&g, 5, 100, 5), "minecraft:stone[]");
        assert!(
            next_set_slot(&rx).is_none(),
            "infinite materials keep the count"
        );
    }

    #[test]
    fn unknown_item_never_places() {
        let (mut g, _rx) = harness();
        // An item id outside the block-item table (diamond) does not place.
        give(&mut g, "minecraft:diamond", 5);
        use_on(&mut g, 5, 99, 5, DIR_UP, 0);
        assert_eq!(at(&g, 5, 100, 5), "minecraft:air[]");
    }
}
