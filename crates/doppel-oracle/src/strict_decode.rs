//! The strict decode gate: reference-derived decoders for every
//! clientbound frame phase by phase, with any over-read, under-read, or
//! unknown id a failure. Frames come from live capture sessions against
//! Doppel (full body dumps) and, for validation, from the vanilla blob
//! set with corrupted variants.

use anyhow::{Context, Result};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::bot::{self, CaptureOpts, CapturedPacket};
use crate::parity_break::{default_doppel_bin, wait_for_port};
use crate::{capture, vanilla};

const VANILLA_PORT: u16 = 25566;
const DOPPEL_PORT: u16 = 25565;

/// Port overrides for local runs that share the machine with another
/// gate's servers; CI uses the defaults. An overridden port implies a
/// private vanilla run directory, whose boot-time wipe would otherwise
/// hit the shared one.
fn vanilla_port() -> u16 {
    match std::env::var("STRICT_VANILLA_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(port) => {
            vanilla::default_run_dir("strict");
            port
        }
        None => VANILLA_PORT,
    }
}

fn doppel_port() -> u16 {
    std::env::var("STRICT_DOPPEL_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DOPPEL_PORT)
}

// ---------------------------------------------------------------------
// The cursor
// ---------------------------------------------------------------------

/// One decode failure: the byte position it happened at plus the reason.
pub struct Fail {
    pub pos: usize,
    pub msg: String,
}

type DResult<T> = Result<T, Fail>;

fn fail<T>(pos: usize, msg: impl Into<String>) -> DResult<T> {
    Err(Fail {
        pos,
        msg: msg.into(),
    })
}

/// A strict reader that records the offset of every length or count varint
/// it consumes, so the corruption harness can patch them.
pub struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
    counts: Vec<usize>,
}

impl<'a> Cursor<'a> {
    pub fn new(buf: &'a [u8]) -> Cursor<'a> {
        Cursor {
            buf,
            pos: 0,
            counts: Vec::new(),
        }
    }

    pub fn consumed(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn count_offsets(&self) -> &[usize] {
        &self.counts
    }

    fn take(&mut self, n: usize, what: &str) -> DResult<&'a [u8]> {
        let end = self.pos + n;
        if end > self.buf.len() {
            return fail(
                self.pos,
                format!(
                    "{what} needs {n} bytes, only {} left",
                    self.buf.len() - self.pos
                ),
            );
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    pub fn u8(&mut self, what: &str) -> DResult<u8> {
        Ok(self.take(1, what)?[0])
    }

    pub fn bool(&mut self, what: &str) -> DResult<bool> {
        let b = self.u8(what)?;
        // Both encoders write 0 or 1; the gate pins that here.
        if b > 1 {
            return fail(self.pos - 1, format!("{what} bool byte {b}, want 0 or 1"));
        }
        Ok(b == 1)
    }

    pub fn i16(&mut self, what: &str) -> DResult<i16> {
        let b = self.take(2, what)?;
        Ok(i16::from_be_bytes([b[0], b[1]]))
    }

    pub fn u16(&mut self, what: &str) -> DResult<u16> {
        let b = self.take(2, what)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    pub fn i32(&mut self, what: &str) -> DResult<i32> {
        let b = self.take(4, what)?;
        Ok(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn i64(&mut self, what: &str) -> DResult<i64> {
        let b = self.take(8, what)?;
        Ok(i64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn f32(&mut self, what: &str) -> DResult<f32> {
        let v = self.i32(what)?;
        Ok(f32::from_bits(v as u32))
    }

    pub fn f64(&mut self, what: &str) -> DResult<f64> {
        let v = self.i64(what)?;
        Ok(f64::from_bits(v as u64))
    }

    pub fn uuid(&mut self, what: &str) -> DResult<()> {
        self.take(16, what)?;
        Ok(())
    }

    /// Reads a VarInt, rejecting the over-long and truncated forms the
    /// reference's own reader rejects.
    pub fn varint(&mut self, what: &str) -> DResult<i32> {
        let mut value: u32 = 0;
        for i in 0..5 {
            let b = self.u8(what)?;
            value |= u32::from(b & 0x7f) << (7 * i);
            if b & 0x80 == 0 {
                return Ok(value as i32);
            }
        }
        fail(self.pos, format!("{what} varint longer than 5 bytes"))
    }

    pub fn varlong(&mut self, what: &str) -> DResult<i64> {
        let mut value: u64 = 0;
        for i in 0..10 {
            let b = self.u8(what)?;
            value |= u64::from(b & 0x7f) << (7 * i);
            if b & 0x80 == 0 {
                return Ok(value as i64);
            }
        }
        fail(self.pos, format!("{what} varlong longer than 10 bytes"))
    }

    /// A bounded collection count.
    pub fn count(&mut self, max: usize, what: &str) -> DResult<usize> {
        self.count_inner(max, what, false)
    }

    /// A bounded collection count for a sequence the decoder knows is
    /// the frame's tail. Only these offsets are recorded for corruption,
    /// because only there does a count change provably leave the parse
    /// short or with leftovers; a mid-frame count can shift the rest of
    /// the bytes into another valid parse of the same grammar.
    pub fn count_last(&mut self, max: usize, what: &str) -> DResult<usize> {
        self.count_inner(max, what, true)
    }

    fn count_inner(&mut self, max: usize, what: &str, record: bool) -> DResult<usize> {
        let at = self.pos;
        let n = self.varint(what)?;
        if n < 0 || n as usize > max {
            return fail(at, format!("{what} count {n} out of bounds (max {max})"));
        }
        if record {
            self.counts.push(at);
        }
        Ok(n as usize)
    }

    /// A protocol string.
    pub fn string(&mut self, max: usize, what: &str) -> DResult<()> {
        let at = self.pos;
        let len = self.varint(what)?;
        if len < 0 || len as usize > max {
            return fail(at, format!("{what} length {len} out of bounds (max {max})"));
        }
        self.take(len as usize, what).map(|_| ()).map_err(|_| Fail {
            pos: at,
            msg: format!("{what} body shorter than its length"),
        })
    }

    /// A length-prefixed byte array.
    pub fn byte_array(&mut self, max: usize, what: &str) -> DResult<()> {
        let at = self.pos;
        let len = self.varint(what)?;
        if len < 0 || len as usize > max {
            return fail(at, format!("{what} length {len} out of bounds (max {max})"));
        }
        self.take(len as usize, what).map(|_| ()).map_err(|_| Fail {
            pos: at,
            msg: format!("{what} shorter than its length"),
        })
    }

    /// A BitSet as the reference writes it: a varint byte length plus the
    /// little-endian-bit-order bytes.
    pub fn bitset(&mut self, what: &str) -> DResult<()> {
        self.byte_array(1024, what)
    }

    /// A packed BlockPos long.
    pub fn block_pos(&mut self) -> DResult<()> {
        self.i64("block pos")?;
        Ok(())
    }

    /// An identifier: a string with the reference's identifier bound.
    pub fn identifier(&mut self, what: &str) -> DResult<()> {
        self.string(32760, what)
    }
}

// ---------------------------------------------------------------------
// Shared skippers
// ---------------------------------------------------------------------

/// Network NBT: one tag with a nameless root. Depth is bounded so a
/// corrupt frame cannot recurse forever.
fn skip_nbt(c: &mut Cursor, depth: u8) -> DResult<()> {
    let tag = c.u8("nbt tag")?;
    skip_nbt_payload(c, tag, depth)
}

fn nbt_name(c: &mut Cursor) -> DResult<()> {
    let len = c.u16("nbt name length")? as usize;
    c.take(len, "nbt name").map(|_| ())
}

fn skip_nbt_payload(c: &mut Cursor, tag: u8, depth: u8) -> DResult<()> {
    if depth > 32 {
        return fail(c.consumed(), "nbt nesting deeper than 32");
    }
    match tag {
        0x01 => {
            c.u8("nbt byte")?;
        }
        0x02 => {
            c.i16("nbt short")?;
        }
        0x03 => {
            c.i32("nbt int")?;
        }
        0x04 => {
            c.i64("nbt long")?;
        }
        0x05 => {
            c.take(4, "nbt float")?;
        }
        0x06 => {
            c.take(8, "nbt double")?;
        }
        0x07 => {
            let len = c.i32("nbt byte array length")?;
            if len < 0 || len as usize > 1 << 20 {
                return fail(c.consumed(), format!("nbt byte array length {len}"));
            }
            c.take(len as usize, "nbt byte array")?;
        }
        0x08 => {
            let len = c.u16("nbt string length")? as usize;
            c.take(len, "nbt string")?;
        }
        0x09 => {
            let elem = c.u8("nbt list element tag")?;
            let len = c.i32("nbt list length")?;
            if len < 0 || len as usize > 1 << 16 {
                return fail(c.consumed(), format!("nbt list length {len}"));
            }
            for _ in 0..len {
                skip_nbt_payload(c, elem, depth + 1)?;
            }
        }
        0x0a => loop {
            let t = c.u8("nbt field tag")?;
            if t == 0x00 {
                break;
            }
            nbt_name(c)?;
            skip_nbt_payload(c, t, depth + 1)?;
        },
        0x0b => {
            let len = c.i32("nbt int array length")?;
            if len < 0 || len as usize > 1 << 20 {
                return fail(c.consumed(), format!("nbt int array length {len}"));
            }
            c.take(4 * len as usize, "nbt int array")?;
        }
        0x0c => {
            let len = c.i32("nbt long array length")?;
            if len < 0 || len as usize > 1 << 20 {
                return fail(c.consumed(), format!("nbt long array length {len}"));
            }
            c.take(8 * len as usize, "nbt long array")?;
        }
        other => return fail(c.consumed(), format!("unknown nbt tag {other}")),
    }
    Ok(())
}

/// A text component: anonymous-root NBT.
fn skip_component(c: &mut Cursor) -> DResult<()> {
    skip_nbt(c, 0)
}

/// A nullable compound tag: a lone end tag means absent.
fn skip_optional_compound(c: &mut Cursor) -> DResult<()> {
    let at = c.consumed();
    let tag = c.u8("compound tag id")?;
    if tag == 0x00 {
        return Ok(());
    }
    if tag != 0x0a {
        return fail(at, format!("compound tag id {tag}, want 10"));
    }
    walk_compound(c, 1)
}

/// A required compound tag: the root must be TAG_Compound.
fn skip_compound(c: &mut Cursor) -> DResult<()> {
    let at = c.consumed();
    let tag = c.u8("compound tag id")?;
    if tag != 0x0a {
        return fail(at, format!("compound tag id {tag}, want 10"));
    }
    walk_compound(c, 1)
}

fn walk_compound(c: &mut Cursor, depth: u8) -> DResult<()> {
    loop {
        let t = c.u8("compound field tag")?;
        if t == 0x00 {
            break;
        }
        nbt_name(c)?;
        skip_nbt_payload(c, t, depth)?;
    }
    Ok(())
}

/// The data component ids Doppel and the pinned build put on the wire.
mod component {
    pub const CUSTOM_DATA: i32 = 0;
    pub const MAX_STACK_SIZE: i32 = 1;
    pub const MAX_DAMAGE: i32 = 2;
    pub const DAMAGE: i32 = 3;
    pub const UNBREAKABLE: i32 = 4;
    pub const CUSTOM_NAME: i32 = 6;
    pub const ITEM_NAME: i32 = 9;
    pub const RARITY: i32 = 12;
    pub const REPAIR_COST: i32 = 19;
}

/// A DataComponentPatch: added (type, value) pairs then removed type ids.
/// Unknown component types fail instead of guessing a payload size.
fn skip_component_patch(c: &mut Cursor) -> DResult<()> {
    let added = c.count(256, "patch entries")?;
    for _ in 0..added {
        let ty = c.varint("component type id")?;
        match ty {
            component::MAX_STACK_SIZE
            | component::MAX_DAMAGE
            | component::DAMAGE
            | component::RARITY
            | component::REPAIR_COST => {
                c.varint("component value")?;
            }
            component::UNBREAKABLE => {}
            component::CUSTOM_DATA | component::CUSTOM_NAME | component::ITEM_NAME => {
                skip_nbt(c, 0)?;
            }
            other => {
                return fail(
                    c.consumed(),
                    format!("no decoder for data component {other}"),
                )
            }
        }
    }
    let removed = c.count(256, "removed components")?;
    for _ in 0..removed {
        c.varint("removed component type id")?;
    }
    Ok(())
}

/// ItemStack.OPTIONAL_STREAM_CODEC: count <= 0 is the empty stack.
fn skip_item_stack(c: &mut Cursor) -> DResult<()> {
    let at = c.consumed();
    let count = c.varint("stack count")?;
    if count == 0 {
        return Ok(());
    }
    if count < 0 {
        return fail(at, format!("stack count {count}"));
    }
    if count > 99 {
        return fail(at, format!("stack count {count} above the absolute cap 99"));
    }
    c.varint("item id")?;
    skip_component_patch(c)
}

/// ItemStackTemplate: item id, count, component patch.
fn skip_item_stack_template(c: &mut Cursor) -> DResult<()> {
    c.varint("item id")?;
    let count = c.varint("template count")?;
    if count < 1 {
        return fail(c.consumed(), format!("template count {count}"));
    }
    skip_component_patch(c)
}

/// A HolderSet: one varint, zero selects the tag form (identifier
/// follows), anything else is a direct set of that many holders minus
/// one.
fn skip_holder_set(c: &mut Cursor) -> DResult<()> {
    let v = c.varint("holder set")?;
    if v == 0 {
        c.identifier("holder set tag")?;
        return Ok(());
    }
    if v < 0 {
        return fail(c.consumed(), format!("holder set count {v}"));
    }
    for _ in 0..v - 1 {
        c.varint("holder id")?;
    }
    Ok(())
}

/// The LpVec3 packed movement vector: a zero byte, or six bytes plus an
/// optional varint scale continuation.
fn skip_lp_vec3(c: &mut Cursor) -> DResult<()> {
    let b0 = c.u8("lp vec3 marker")?;
    if b0 == 0 {
        return Ok(());
    }
    c.take(5, "lp vec3 body")?;
    if b0 & 0x04 != 0 {
        c.varint("lp vec3 scale")?;
    }
    Ok(())
}

/// A registry holder that allows a direct form: 0 means the direct payload
/// follows, otherwise the varint is the holder id plus one.
fn skip_holder_or_direct(
    c: &mut Cursor,
    direct: fn(&mut Cursor) -> DResult<()>,
    what: &str,
) -> DResult<()> {
    let id = c.varint(what)?;
    if id == 0 {
        return direct(c);
    }
    if id < 0 {
        return fail(c.consumed(), format!("{what} holder id {id}"));
    }
    Ok(())
}

fn direct_sound_event(c: &mut Cursor) -> DResult<()> {
    c.identifier("sound event id")?;
    if c.bool("sound event has range")? {
        c.f32("sound event range")?;
    }
    Ok(())
}

fn direct_trim_pattern(c: &mut Cursor) -> DResult<()> {
    c.identifier("trim pattern asset id")?;
    skip_component(c)?;
    c.bool("trim pattern decal")?;
    Ok(())
}

/// One particle: the registry id plus the per-type options payload.
fn skip_particle(c: &mut Cursor) -> DResult<()> {
    let at = c.consumed();
    let id = c.varint("particle type id")?;
    match id {
        // Payload-less SimpleParticleTypes (registration order).
        0 | 3 | 4 | 5 | 6 | 11 | 12 | 13 | 14 | 16 | 17 | 18 | 19 | 20 | 24 | 25 | 26 | 27 | 29
        | 30 | 31 | 32 | 33 | 34 | 35 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 47 | 49
        | 50 | 51 | 53 | 54 | 55 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71
        | 72 | 73 | 74 | 75 | 76 | 77 | 78 | 79 | 80 | 81 | 82 | 83 | 84 | 85 | 86 | 87 | 88
        | 89 | 90 | 91 | 92 | 93 | 94 | 95 | 96 | 97 | 98 | 99 | 100 | 101 | 102 | 103 | 104
        | 105 | 106 | 107 | 108 | 109 | 110 | 111 | 112 | 113 | 114 | 116 | 117 | 118 | 119
        | 120 | 122 | 123 | 124 | 126 | 127 => {}
        // block / block_marker / falling_dust / dust_pillar / block_crumble
        1 | 2 | 36 | 121 | 125 => {
            c.varint("particle block state")?;
        }
        // geyser / geyser_plume
        7 | 10 => {
            c.i32("geyser water blocks")?;
        }
        // geyser_base / geyser_poof
        8 | 9 => {
            c.i32("geyser water blocks")?;
            c.f32("geyser burst impulse")?;
        }
        // dragon_breath
        15 => {
            c.f32("particle power")?;
        }
        // dust
        21 => {
            c.i32("dust color")?;
            c.f32("dust scale")?;
        }
        // dust_color_transition
        22 => {
            c.i32("dust from color")?;
            c.i32("dust to color")?;
            c.f32("dust scale")?;
        }
        // effect / instant_effect
        23 | 56 => {
            c.i32("spell color")?;
            c.f32("spell power")?;
        }
        // entity_effect / tinted_leaves / flash
        28 | 46 | 52 => {
            c.i32("particle color")?;
        }
        // sculk_charge
        48 => {
            c.f32("sculk charge roll")?;
        }
        // item
        57 => {
            skip_item_stack_template(c)?;
        }
        // vibration: a position source dispatch (block or entity).
        58 => {
            let ty = c.varint("position source type id")?;
            match ty {
                0 => {
                    c.block_pos()?;
                }
                1 => {
                    c.varint("position source entity")?;
                    c.f32("position source delta")?;
                }
                other => return fail(c.consumed(), format!("position source type {other}")),
            }
            c.varint("vibration arrival ticks")?;
        }
        // trail
        59 => {
            c.f64("trail x")?;
            c.f64("trail y")?;
            c.f64("trail z")?;
            c.i32("trail color")?;
            c.varint("trail duration")?;
        }
        // shriek
        115 => {
            c.varint("shriek delay")?;
        }
        other => return fail(at, format!("particle type id {other} out of range")),
    }
    Ok(())
}

/// One SlotDisplay: the type id plus its fields, recursive where the type
/// nests further displays.
fn skip_slot_display(c: &mut Cursor, depth: u8) -> DResult<()> {
    if depth > 16 {
        return fail(c.consumed(), "slot display nesting deeper than 16");
    }
    let at = c.consumed();
    let ty = c.varint("slot display type id")?;
    match ty {
        0 | 1 => {} // empty, any_fuel
        2 => {
            skip_slot_display(c, depth + 1)?;
        }
        3 => {
            skip_slot_display(c, depth + 1)?;
            c.varint("component type id")?;
        }
        4 => {
            c.varint("slot display item")?;
        }
        5 => {
            skip_item_stack_template(c)?;
        }
        6 => {
            skip_holder_set(c)?;
        }
        7 => {
            skip_slot_display(c, depth + 1)?;
            skip_slot_display(c, depth + 1)?;
        }
        8 => {
            skip_slot_display(c, depth + 1)?;
            skip_slot_display(c, depth + 1)?;
            skip_holder_or_direct(c, direct_trim_pattern, "trim pattern")?;
        }
        9 => {
            skip_slot_display(c, depth + 1)?;
            skip_slot_display(c, depth + 1)?;
        }
        10 => {
            let n = c.count(256, "composite contents")?;
            for _ in 0..n {
                skip_slot_display(c, depth + 1)?;
            }
        }
        other => return fail(at, format!("slot display type id {other} out of range")),
    }
    Ok(())
}

/// One RecipeDisplay: the type id plus its slot displays.
fn skip_recipe_display(c: &mut Cursor) -> DResult<()> {
    let at = c.consumed();
    let ty = c.varint("recipe display type id")?;
    match ty {
        0 => {
            let n = c.count(256, "shapeless ingredients")?;
            for _ in 0..n {
                skip_slot_display(c, 0)?;
            }
            skip_slot_display(c, 0)?;
            skip_slot_display(c, 0)?;
        }
        1 => {
            c.varint("shaped width")?;
            c.varint("shaped height")?;
            let n = c.count(256, "shaped ingredients")?;
            for _ in 0..n {
                skip_slot_display(c, 0)?;
            }
            skip_slot_display(c, 0)?;
            skip_slot_display(c, 0)?;
        }
        2 => {
            skip_slot_display(c, 0)?;
            skip_slot_display(c, 0)?;
            skip_slot_display(c, 0)?;
            skip_slot_display(c, 0)?;
            c.varint("furnace duration")?;
            c.f32("furnace experience")?;
        }
        3 => {
            skip_slot_display(c, 0)?;
            skip_slot_display(c, 0)?;
            skip_slot_display(c, 0)?;
        }
        4 => {
            skip_slot_display(c, 0)?;
            skip_slot_display(c, 0)?;
            skip_slot_display(c, 0)?;
            skip_slot_display(c, 0)?;
            skip_slot_display(c, 0)?;
        }
        other => return fail(at, format!("recipe display type id {other} out of range")),
    }
    Ok(())
}

/// CommonPlayerSpawnInfo: dimension holder, dimension key, seed, game
/// modes, flags, death location, portal cooldown, sea level.
fn skip_spawn_info(c: &mut Cursor) -> DResult<()> {
    let dim = c.varint("dimension type holder")?;
    if dim < 0 {
        return fail(c.consumed(), format!("dimension type holder {dim}"));
    }
    c.identifier("dimension key")?;
    c.i64("seed")?;
    c.varint("game mode")?;
    c.varint("previous game mode")?;
    c.bool("is debug")?;
    c.bool("is flat")?;
    if c.bool("has death location")? {
        c.identifier("death dimension")?;
        c.block_pos()?;
    }
    c.varint("portal cooldown")?;
    c.varint("sea level")?;
    Ok(())
}

/// One advancement display: two components, the icon, the frame, a flags
/// int, and the optional background.
fn skip_advancement_display(c: &mut Cursor) -> DResult<()> {
    skip_component(c)?;
    skip_component(c)?;
    skip_item_stack_template(c)?;
    c.varint("advancement type")?;
    let flags = c.i32("advancement flags")?;
    if flags & !0x07 != 0 {
        return fail(
            c.consumed(),
            format!("advancement flags {flags} uses unknown bits"),
        );
    }
    if flags & 1 != 0 {
        c.identifier("advancement background")?;
    }
    Ok(())
}

/// One command tree node.
fn skip_command_node(c: &mut Cursor) -> DResult<()> {
    let flags = c.u8("command node flags")?;
    let ty = flags & 3;
    let children = c.count(8192, "command node children")?;
    for _ in 0..children {
        let child = c.varint("command child")?;
        if child < 0 {
            return fail(c.consumed(), format!("command child {child}"));
        }
    }
    if flags & 0x08 != 0 {
        c.varint("command redirect")?;
    }
    match ty {
        1 => {
            c.string(32760, "literal name")?;
        }
        2 => {
            c.string(32760, "argument name")?;
            let serializer = c.varint("argument type id")?;
            skip_argument_payload(c, serializer)?;
            if flags & 0x10 != 0 {
                c.identifier("suggestion provider")?;
            }
        }
        _ => {}
    }
    if flags & !0x3f != 0 {
        return fail(
            c.consumed(),
            format!("command node flags {flags} uses unknown bits"),
        );
    }
    Ok(())
}

/// The per-serializer payload of a command argument node. Ids follow the
/// command argument type registration order.
fn skip_argument_payload(c: &mut Cursor, id: i32) -> DResult<()> {
    let at = c.consumed();
    match id {
        0 => {}                                    // bool
        1 => number_flags(c, "float", Num::F32)?,  // float
        2 => number_flags(c, "double", Num::F64)?, // double
        3 => number_flags(c, "int", Num::I32)?,    // int
        4 => number_flags(c, "long", Num::I64)?,   // long
        5 => {
            c.varint("string type")?; // string
        }
        6 | 31 => {
            c.u8("argument flags")?; // entity, score_holder
        }
        43 => {
            c.i32("time min")?; // time
        }
        44..=48 => {
            c.identifier("argument registry key")?; // resource family
        }
        other => {
            // The rest of the registration order carries no payload.
            if !(7..=42).contains(&other) && !(49..=61).contains(&other) {
                return fail(at, format!("argument type id {other} out of range"));
            }
        }
    }
    Ok(())
}

enum Num {
    F32,
    F64,
    I32,
    I64,
}

fn number_flags(c: &mut Cursor, what: &str, kind: Num) -> DResult<()> {
    let flags = c.u8(what)?;
    if flags > 3 {
        return fail(c.consumed(), format!("{what} flags {flags}"));
    }
    for bit in [1u8, 2u8] {
        if flags & bit != 0 {
            match kind {
                Num::F32 => {
                    c.f32(what)?;
                }
                Num::F64 => {
                    c.f64(what)?;
                }
                Num::I32 => {
                    c.i32(what)?;
                }
                Num::I64 => {
                    c.i64(what)?;
                }
            }
        }
    }
    Ok(())
}

/// One SynchedEntityData value payload, by serializer id.
fn skip_entity_datum(c: &mut Cursor, serializer: i32) -> DResult<()> {
    let at = c.consumed();
    match serializer {
        0 => {
            c.u8("datum byte")?;
        }
        1 => {
            c.varint("datum int")?;
        }
        2 => {
            c.varlong("datum long")?;
        }
        3 => {
            c.f32("datum float")?;
        }
        4 => {
            c.string(32760, "datum string")?;
        }
        5 => {
            skip_component(c)?;
        }
        6 => {
            if c.bool("datum has component")? {
                skip_component(c)?;
            }
        }
        7 => {
            skip_item_stack(c)?;
        }
        8 => {
            c.bool("datum bool")?;
        }
        9 => {
            c.f32("datum pitch")?;
            c.f32("datum yaw")?;
            c.f32("datum roll")?;
        }
        10 | 11 => {
            if serializer == 11 && !c.bool("datum has pos")? {
                return Ok(());
            }
            c.block_pos()?;
        }
        12 => {
            c.varint("datum direction")?;
        }
        13 => {
            c.varint("datum living entity reference")?;
        }
        14 => {
            c.varint("datum block state")?;
        }
        15 => {
            c.varint("datum optional block state")?;
        }
        16 => {
            skip_particle(c)?;
        }
        17 => {
            let n = c.count(4096, "datum particles")?;
            for _ in 0..n {
                skip_particle(c)?;
            }
        }
        18 => {
            c.varint("villager type")?;
            c.varint("villager profession")?;
            c.varint("villager level")?;
        }
        19 => {
            c.varint("datum optional unsigned int")?;
        }
        20 => {
            c.varint("datum pose")?;
        }
        21..=32 => {
            c.varint("datum variant holder")?;
        }
        33 => {
            c.identifier("datum global pos dimension")?;
            c.block_pos()?;
        }
        34 => {
            c.varint("datum painting variant")?;
        }
        35..=38 => {
            c.varint("datum state")?;
        }
        39 => {
            c.f32("datum vec x")?;
            c.f32("datum vec y")?;
            c.f32("datum vec z")?;
        }
        40 => {
            c.f32("datum quaternion x")?;
            c.f32("datum quaternion y")?;
            c.f32("datum quaternion z")?;
            c.f32("datum quaternion w")?;
        }
        41 => {
            let n = c.count(16, "profile properties")?;
            for _ in 0..n {
                c.string(64, "profile property name")?;
                c.string(32760, "profile property value")?;
                if c.bool("profile property signed")? {
                    c.string(32760, "profile property signature")?;
                }
            }
        }
        42 | 43 => {
            c.varint("datum enum")?;
        }
        other => {
            return fail(
                at,
                format!("entity data serializer id {other} out of range"),
            )
        }
    }
    Ok(())
}

/// The full light payload of a chunk frame: four masks plus the two
/// arrays of 2048-byte layers.
fn skip_light(c: &mut Cursor) -> DResult<()> {
    c.bitset("sky mask")?;
    c.bitset("block mask")?;
    c.bitset("empty sky mask")?;
    c.bitset("empty block mask")?;
    for (i, what) in ["sky updates", "block updates"].iter().enumerate() {
        let what = *what;
        // The block-update list ends the chunk frame; the sky list does
        // not, so only its count is provably corruption-breaking.
        let n = if i == 1 {
            c.count_last(4096, what)?
        } else {
            c.count(4096, what)?
        };
        for _ in 0..n {
            // The reference bounds each layer at 2048 bytes.
            c.byte_array(2048, "light layer")?;
        }
    }
    Ok(())
}

/// One paletted container inside a chunk section blob.
fn skip_container(c: &mut Cursor, entry_count: usize, biomes: bool) -> DResult<()> {
    let at = c.consumed();
    let bits = c.u8("container bits")?;
    let (max_indirect, max_bits) = if biomes { (3, 8) } else { (8, 16) };
    if bits > max_bits {
        return fail(at, format!("container bits {bits}"));
    }
    match bits {
        0 => {
            c.varint("container single value")?;
        }
        _ => {
            if bits <= max_indirect {
                let size = c.count(65536, "palette size")?;
                for _ in 0..size {
                    c.varint("palette entry")?;
                }
            }
            let longs = entry_count.div_ceil(64 / bits as usize);
            c.take(longs * 8, "container storage")?;
        }
    }
    Ok(())
}

/// The whole level_chunk_with_light body.
fn decode_chunk(c: &mut Cursor) -> DResult<()> {
    c.i32("chunk x")?;
    c.i32("chunk z")?;
    let maps = c.count(8, "heightmap count")?;
    for _ in 0..maps {
        let at = c.consumed();
        let ty = c.varint("heightmap type")?;
        // The heightmap type ids follow the Heightmap.Types ordinals; the
        // client-sent set is WORLD_SURFACE (1) and MOTION_BLOCKING (4).
        if !(0..=5).contains(&ty) {
            return fail(at, format!("heightmap type {ty}"));
        }
        let longs = c.count(64, "heightmap longs")?;
        c.take(longs * 8, "heightmap data")?;
    }
    let data_len = c.count(2 << 20, "chunk data array")?;
    let data_start = c.consumed();
    c.take(data_len, "chunk data array")?;
    {
        let mut sc = Cursor::new(&c.buf[data_start..data_start + data_len]);
        while sc.remaining() > 0 {
            sc.i16("section non-empty count")?;
            sc.i16("section fluid count")?;
            skip_container(&mut sc, 4096, false)?;
            skip_container(&mut sc, 64, true)?;
        }
    }
    let entities = c.count(1024, "block entity count")?;
    for _ in 0..entities {
        c.u8("block entity packed xz")?;
        c.i16("block entity y")?;
        c.varint("block entity type")?;
        skip_optional_compound(c)?;
    }
    skip_light(c)
}

// ---------------------------------------------------------------------
// Known id tables (registration order per phase)
// ---------------------------------------------------------------------

const LOGIN_IDS: &[(i32, &str)] = &[
    (0x00, "login_disconnect"),
    (0x01, "hello"),
    (0x02, "login_finished"),
    (0x03, "login_compression"),
    (0x04, "custom_query"),
    (0x05, "cookie_request"),
];

const CONFIG_IDS: &[(i32, &str)] = &[
    (0x00, "cookie_request"),
    (0x01, "custom_payload"),
    (0x02, "disconnect"),
    (0x03, "finish_configuration"),
    (0x04, "keep_alive"),
    (0x05, "ping"),
    (0x06, "reset_chat"),
    (0x07, "registry_data"),
    (0x08, "resource_pack_pop"),
    (0x09, "resource_pack_push"),
    (0x0a, "post_effects"),
    (0x0b, "store_cookie"),
    (0x0c, "transfer"),
    (0x0d, "update_enabled_features"),
    (0x0e, "update_tags"),
    (0x0f, "select_known_packs"),
    (0x10, "custom_report_details"),
    (0x11, "server_links"),
    (0x12, "clear_dialog"),
    (0x13, "show_dialog"),
    (0x14, "code_of_conduct"),
];

const PLAY_IDS: &[(i32, &str)] = &[
    (0x00, "bundle_delimiter"),
    (0x01, "add_entity"),
    (0x02, "animate"),
    (0x03, "award_stats"),
    (0x04, "block_changed_ack"),
    (0x05, "block_destruction"),
    (0x06, "block_entity_data"),
    (0x07, "block_event"),
    (0x08, "block_update"),
    (0x09, "boss_event"),
    (0x0a, "change_difficulty"),
    (0x0b, "chunk_batch_finished"),
    (0x0c, "chunk_batch_start"),
    (0x0d, "chunks_biomes"),
    (0x0e, "clear_titles"),
    (0x0f, "command_suggestions"),
    (0x10, "commands"),
    (0x11, "container_close"),
    (0x12, "container_set_content"),
    (0x13, "container_set_data"),
    (0x14, "container_set_slot"),
    (0x15, "cookie_request"),
    (0x16, "cooldown"),
    (0x17, "custom_chat_completions"),
    (0x18, "custom_payload"),
    (0x19, "damage_event"),
    (0x1a, "debug_block_value"),
    (0x1b, "debug_chunk_value"),
    (0x1c, "debug_entity_value"),
    (0x1d, "debug_event"),
    (0x1e, "debug_sample"),
    (0x1f, "delete_chat"),
    (0x20, "disconnect"),
    (0x21, "disguised_chat"),
    (0x22, "entity_event"),
    (0x23, "entity_position_sync"),
    (0x24, "explode"),
    (0x25, "add_transient_block"),
    (0x26, "forget_level_chunk"),
    (0x27, "game_event"),
    (0x28, "game_rule_values"),
    (0x29, "game_test_highlight_pos"),
    (0x2a, "mount_screen_open"),
    (0x2b, "hurt_animation"),
    (0x2c, "initialize_border"),
    (0x2d, "keep_alive"),
    (0x2e, "level_chunk_with_light"),
    (0x2f, "level_event"),
    (0x30, "level_particles"),
    (0x31, "light_update"),
    (0x32, "login"),
    (0x33, "low_disk_space_warning"),
    (0x34, "map_item_data"),
    (0x35, "merchant_offers"),
    (0x36, "move_entity_pos"),
    (0x37, "move_entity_pos_rot"),
    (0x38, "move_minecart_along_track"),
    (0x39, "move_entity_rot"),
    (0x3a, "move_vehicle"),
    (0x3b, "open_book"),
    (0x3c, "open_screen"),
    (0x3d, "open_sign_editor"),
    (0x3e, "ping"),
    (0x3f, "pong_response"),
    (0x40, "place_ghost_recipe"),
    (0x41, "player_abilities"),
    (0x42, "player_chat"),
    (0x43, "player_combat_end"),
    (0x44, "player_combat_enter"),
    (0x45, "player_combat_kill"),
    (0x46, "player_info_remove"),
    (0x47, "player_info_update"),
    (0x48, "player_look_at"),
    (0x49, "player_position"),
    (0x4a, "player_rotation"),
    (0x4b, "recipe_book_add"),
    (0x4c, "recipe_book_remove"),
    (0x4d, "recipe_book_settings"),
    (0x4e, "remove_entities"),
    (0x4f, "remove_mob_effect"),
    (0x50, "reset_score"),
    (0x51, "resource_pack_pop"),
    (0x52, "resource_pack_push"),
    (0x53, "post_effects"),
    (0x54, "respawn"),
    (0x55, "rotate_head"),
    (0x56, "section_blocks_update"),
    (0x57, "select_advancements_tab"),
    (0x58, "server_data"),
    (0x59, "set_action_bar_text"),
    (0x5a, "set_border_center"),
    (0x5b, "set_border_lerp_size"),
    (0x5c, "set_border_size"),
    (0x5d, "set_border_warning_delay"),
    (0x5e, "set_border_warning_distance"),
    (0x5f, "set_camera"),
    (0x60, "set_chunk_cache_center"),
    (0x61, "set_chunk_cache_radius"),
    (0x62, "set_cursor_item"),
    (0x63, "set_default_spawn_position"),
    (0x64, "set_display_objective"),
    (0x65, "set_entity_data"),
    (0x66, "set_entity_link"),
    (0x67, "set_entity_motion"),
    (0x68, "set_equipment"),
    (0x69, "set_experience"),
    (0x6a, "set_health"),
    (0x6b, "set_held_slot"),
    (0x6c, "set_objective"),
    (0x6d, "set_passengers"),
    (0x6e, "set_player_inventory"),
    (0x6f, "set_player_team"),
    (0x70, "set_score"),
    (0x71, "set_simulation_distance"),
    (0x72, "set_subtitle_text"),
    (0x73, "set_time"),
    (0x74, "set_title_text"),
    (0x75, "set_titles_animation"),
    (0x76, "sound_entity"),
    (0x77, "sound"),
    (0x78, "start_configuration"),
    (0x79, "stop_sound"),
    (0x7a, "store_cookie"),
    (0x7b, "swing_animation"),
    (0x7c, "system_chat"),
    (0x7d, "tab_list"),
    (0x7e, "tag_query"),
    (0x7f, "take_item_entity"),
    (0x80, "teleport_entity"),
    (0x81, "test_instance_block_status"),
    (0x82, "ticking_state"),
    (0x83, "ticking_step"),
    (0x84, "transfer"),
    (0x85, "update_advancements"),
    (0x86, "update_attributes"),
    (0x87, "update_mob_effect"),
    (0x88, "update_recipes"),
    (0x89, "update_tags"),
    (0x8a, "projectile_power"),
    (0x8b, "custom_report_details"),
    (0x8c, "server_links"),
    (0x8d, "waypoint"),
    (0x8e, "clear_dialog"),
    (0x8f, "show_dialog"),
];

/// The ids with decoders, per phase. Every id Doppel can emit plus the
/// rest of each phase table where the shape is a fixed one.
const LOGIN_DECODED: &[i32] = &[0x00, 0x01, 0x02, 0x03, 0x04, 0x05];
const CONFIG_DECODED: &[i32] = &[
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x14,
];
const PLAY_DECODED: &[i32] = &[
    0x00, 0x01, 0x04, 0x05, 0x06, 0x07, 0x08, 0x0a, 0x0b, 0x0c, 0x10, 0x11, 0x12, 0x13, 0x14, 0x19,
    0x1f, 0x22, 0x23, 0x24, 0x26, 0x27, 0x2b, 0x2c, 0x2d, 0x2e, 0x32, 0x36, 0x37, 0x39, 0x3c, 0x41,
    0x45, 0x47, 0x49, 0x4b, 0x4d, 0x4e, 0x53, 0x54, 0x55, 0x56, 0x58, 0x60, 0x62, 0x63, 0x65, 0x67,
    0x68, 0x69, 0x6a, 0x6b, 0x6e, 0x73, 0x7c, 0x7f, 0x80, 0x82, 0x83, 0x85, 0x86, 0x88,
];

fn table(phase: Phase) -> &'static [(i32, &'static str)] {
    match phase {
        Phase::Login => LOGIN_IDS,
        Phase::Config => CONFIG_IDS,
        Phase::Play => PLAY_IDS,
    }
}

fn decoded_ids(phase: Phase) -> &'static [i32] {
    match phase {
        Phase::Login => LOGIN_DECODED,
        Phase::Config => CONFIG_DECODED,
        Phase::Play => PLAY_DECODED,
    }
}

fn name_of(phase: Phase, id: i32) -> Option<&'static str> {
    table(phase).iter().find(|(i, _)| *i == id).map(|(_, n)| *n)
}

fn has_decoder(phase: Phase, id: i32) -> bool {
    decoded_ids(phase).contains(&id)
}

// ---------------------------------------------------------------------
// check_frame
// ---------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Login,
    Config,
    Play,
}

impl Phase {
    fn label(&self) -> &'static str {
        match self {
            Phase::Login => "login",
            Phase::Config => "config",
            Phase::Play => "play",
        }
    }
}

/// One frame failure: everything the brief asks the report to carry.
#[derive(Clone, Debug)]
pub struct FrameFailure {
    pub phase: &'static str,
    pub id: i32,
    pub name: String,
    pub consumed: usize,
    pub len: usize,
    pub reason: String,
    pub frame: String,
}

/// Decodes one frame body strictly. Ok carries the consumed byte count
/// (always the body length on success).
pub fn check_frame(phase: Phase, id: i32, body: &[u8]) -> Result<usize, FrameFailure> {
    let wrap = |reason: String, consumed: usize| FrameFailure {
        phase: phase.label(),
        id,
        name: name_of(phase, id).unwrap_or("unknown").to_string(),
        consumed,
        len: body.len(),
        reason,
        frame: String::new(),
    };
    let Some(name) = name_of(phase, id) else {
        return Err(wrap(
            format!("unknown id {id:#04x} for the {} phase", phase.label()),
            0,
        ));
    };
    if !has_decoder(phase, id) {
        return Err(wrap(format!("{name} has no decoder"), 0));
    }
    let mut c = Cursor::new(body);
    match decode_body(phase, id, &mut c) {
        Ok(()) => {
            let consumed = c.consumed();
            if consumed != body.len() {
                return Err(wrap(
                    format!(
                        "frame carries {} leftover byte(s) the format does not read",
                        body.len() - consumed
                    ),
                    consumed,
                ));
            }
            Ok(consumed)
        }
        Err(f) => Err(wrap(format!("{} (byte {})", f.msg, f.pos), f.pos)),
    }
}

fn decode_body(phase: Phase, id: i32, c: &mut Cursor) -> DResult<()> {
    match phase {
        Phase::Login => decode_login(id, c),
        Phase::Config => decode_config(id, c),
        Phase::Play => decode_play(id, c),
    }
}

fn decode_login(id: i32, c: &mut Cursor) -> DResult<()> {
    match id {
        0x00 => skip_component(c),
        0x01 => {
            c.identifier("server id")?;
            c.byte_array(512, "public key")?;
            c.byte_array(20, "verify token")?;
            Ok(())
        }
        0x02 => {
            c.uuid("profile uuid")?;
            c.string(16, "profile name")?;
            let n = c.count(16, "profile properties")?;
            for _ in 0..n {
                c.string(64, "property name")?;
                c.string(32760, "property value")?;
                if c.bool("property signed")? {
                    c.string(32760, "property signature")?;
                }
            }
            c.uuid("session id")?;
            Ok(())
        }
        0x03 => {
            c.varint("compression threshold")?;
            Ok(())
        }
        0x04 => {
            c.varint("query transaction id")?;
            c.identifier("query channel")?;
            c.byte_array(1 << 20, "query data")?;
            Ok(())
        }
        0x05 => c.identifier("cookie key"),
        _ => fail(c.consumed(), "no decoder"),
    }
}

fn decode_config(id: i32, c: &mut Cursor) -> DResult<()> {
    match id {
        0x00 => c.identifier("cookie key"),
        0x01 => {
            c.identifier("payload channel")?;
            let rest = c.remaining();
            c.take(rest, "payload data")?;
            Ok(())
        }
        0x02 => skip_component(c),
        0x03 => {
            if c.remaining() != 0 {
                return fail(0, "finish_configuration body not empty");
            }
            Ok(())
        }
        0x04 => {
            c.i64("keep alive id")?;
            Ok(())
        }
        0x05 => {
            c.i32("ping id")?;
            Ok(())
        }
        0x06 => Ok(()),
        0x07 => {
            c.identifier("registry key")?;
            let n = c.count_last(1 << 16, "registry entries")?;
            for _ in 0..n {
                c.identifier("entry id")?;
                if c.bool("entry has data")? {
                    skip_nbt(c, 0)?;
                }
            }
            Ok(())
        }
        0x08 => {
            if c.bool("has pack id")? {
                c.uuid("pack id")?;
            }
            Ok(())
        }
        0x09 => {
            c.uuid("pack id")?;
            c.string(32760, "pack url")?;
            c.string(40, "pack hash")?;
            c.bool("pack required")?;
            if c.bool("pack has prompt")? {
                skip_component(c)?;
            }
            Ok(())
        }
        0x0a => Ok(()),
        0x0b => {
            c.identifier("cookie key")?;
            c.byte_array(5120, "cookie payload")?;
            Ok(())
        }
        0x0c => {
            c.string(32760, "transfer host")?;
            c.varint("transfer port")?;
            Ok(())
        }
        0x0d => {
            let n = c.count(1024, "enabled features")?;
            for _ in 0..n {
                c.identifier("feature id")?;
            }
            Ok(())
        }
        0x0e => {
            let n = c.count(1024, "tag registries")?;
            for _ in 0..n {
                c.identifier("registry key")?;
                let t = c.count(1 << 16, "registry tags")?;
                for _ in 0..t {
                    c.identifier("tag id")?;
                    let e = c.count(1 << 16, "tag entries")?;
                    for _ in 0..e {
                        c.varint("tag entry")?;
                    }
                }
            }
            Ok(())
        }
        0x0f => {
            let n = c.count(64, "known packs")?;
            for _ in 0..n {
                c.string(32760, "pack namespace")?;
                c.string(32760, "pack id")?;
                c.string(32760, "pack version")?;
            }
            Ok(())
        }
        0x10 => {
            let n = c.count(32, "report details")?;
            for _ in 0..n {
                c.string(128, "detail key")?;
                c.string(4096, "detail value")?;
            }
            Ok(())
        }
        0x11 => {
            let n = c.count(64, "server links")?;
            for _ in 0..n {
                if c.bool("link is known type")? {
                    let ty = c.varint("known link type")?;
                    if !(0..=9).contains(&ty) {
                        return fail(c.consumed(), format!("known link type {ty}"));
                    }
                } else {
                    skip_component(c)?;
                }
                c.string(32760, "link target")?;
            }
            Ok(())
        }
        0x12 => Ok(()),
        0x14 => {
            c.string(32760, "code of conduct")?;
            Ok(())
        }
        _ => fail(c.consumed(), "no decoder"),
    }
}

/// Play-frame dispatch by packet family.
fn decode_play(id: i32, c: &mut Cursor) -> DResult<()> {
    match id {
        0x04 | 0x05 | 0x06 | 0x07 | 0x08 | 0x26 | 0x27 | 0x56 | 0x2e => decode_world_frame(id, c),
        0x01 | 0x19 | 0x22 | 0x23 | 0x2b | 0x36 | 0x37 | 0x39 | 0x4e | 0x55 | 0x65 | 0x67
        | 0x68 | 0x7f | 0x86 => decode_entity_frame(id, c),
        0x11 | 0x12 | 0x13 | 0x14 | 0x3c | 0x62 | 0x6b | 0x6e => decode_inventory_frame(id, c),
        _ => decode_join_frame(id, c),
    }
}

fn decode_world_frame(id: i32, c: &mut Cursor) -> DResult<()> {
    match id {
        0x04 => {
            c.varint("sequence")?;
            Ok(())
        }
        0x05 => {
            c.varint("breaker entity id")?;
            c.block_pos()?;
            c.u8("destroy progress")?;
            Ok(())
        }
        0x06 => {
            c.block_pos()?;
            c.varint("block entity type")?;
            skip_compound(c)?;
            Ok(())
        }
        0x07 => {
            c.block_pos()?;
            c.u8("block event b0")?;
            c.u8("block event b1")?;
            c.varint("block id")?;
            Ok(())
        }
        0x08 => {
            c.block_pos()?;
            c.varint("block state")?;
            Ok(())
        }
        0x26 => {
            c.i64("chunk pos")?;
            Ok(())
        }
        0x27 => {
            c.u8("game event id")?;
            c.f32("game event param")?;
            Ok(())
        }
        0x56 => {
            c.i64("section pos")?;
            let n = c.count_last(1 << 16, "section update count")?;
            for _ in 0..n {
                c.varlong("packed section change")?;
            }
            Ok(())
        }
        0x2e => decode_chunk(c),
        _ => fail(c.consumed(), "family mismatch"),
    }
}

fn decode_entity_frame(id: i32, c: &mut Cursor) -> DResult<()> {
    match id {
        0x01 => {
            c.varint("entity id")?;
            c.uuid("entity uuid")?;
            c.varint("entity type")?;
            c.f64("entity x")?;
            c.f64("entity y")?;
            c.f64("entity z")?;
            skip_lp_vec3(c)?;
            c.u8("entity x rot")?;
            c.u8("entity y rot")?;
            c.u8("entity y head rot")?;
            c.varint("entity data")?;
            Ok(())
        }
        0x19 => {
            c.varint("target entity id")?;
            c.varint("damage type holder")?;
            c.varint("source cause id")?;
            c.varint("source direct id")?;
            if c.bool("has source position")? {
                c.f64("source x")?;
                c.f64("source y")?;
                c.f64("source z")?;
            }
            Ok(())
        }
        0x22 => {
            c.i32("entity id")?;
            c.u8("entity event")?;
            Ok(())
        }
        0x23 => {
            c.varint("entity id")?;
            let ty = c.varint("position path type")?;
            match ty {
                0 => {
                    c.f64("path x")?;
                    c.f64("path y")?;
                    c.f64("path z")?;
                }
                1 => {
                    let n = c.count(1024, "path steps")?;
                    for _ in 0..n {
                        c.f64("step x")?;
                        c.f64("step y")?;
                        c.f64("step z")?;
                        c.varint("step tick offset")?;
                    }
                }
                other => return fail(c.consumed(), format!("position path type {other}")),
            }
            c.f32("y rot")?;
            c.f32("x rot")?;
            c.bool("on ground")?;
            Ok(())
        }
        0x2b => {
            c.varint("entity id")?;
            c.f32("hurt yaw")?;
            Ok(())
        }
        0x36 | 0x37 => {
            c.varint("entity id")?;
            let props = c.varint("move properties")?;
            if props < 0 {
                return fail(c.consumed(), format!("move properties {props}"));
            }
            let steps = (props >> 1) as usize;
            if steps > 0 {
                if steps > c.remaining() / 7 {
                    return fail(
                        c.consumed(),
                        format!("vec delta steps {steps} larger than the frame allows"),
                    );
                }
                for _ in 0..steps {
                    c.varint("step ticks")?;
                    c.i16("step x")?;
                    c.i16("step y")?;
                    c.i16("step z")?;
                }
            } else {
                c.i16("delta x")?;
                c.i16("delta y")?;
                c.i16("delta z")?;
            }
            if id == 0x37 {
                c.u8("entity y rot")?;
                c.u8("entity x rot")?;
            }
            Ok(())
        }
        0x39 => {
            c.varint("entity id")?;
            c.bool("on ground")?;
            c.u8("entity y rot")?;
            c.u8("entity x rot")?;
            Ok(())
        }
        0x4e => {
            let n = c.count_last(1 << 16, "removed entities")?;
            for _ in 0..n {
                c.varint("removed entity id")?;
            }
            Ok(())
        }
        0x55 => {
            c.varint("entity id")?;
            c.u8("y head rot")?;
            Ok(())
        }
        0x65 => {
            c.varint("entity id")?;
            loop {
                let accessor = c.u8("entity data accessor id")?;
                if accessor == 0xff {
                    break;
                }
                let serializer = c.varint("entity data serializer id")?;
                skip_entity_datum(c, serializer)?;
            }
            Ok(())
        }
        0x67 => {
            c.varint("entity id")?;
            skip_lp_vec3(c)?;
            Ok(())
        }
        0x68 => {
            c.varint("entity id")?;
            loop {
                let slot = c.u8("equipment slot byte")?;
                let id = slot & 0x7f;
                if id > 5 {
                    return fail(c.consumed(), format!("equipment slot {id}"));
                }
                skip_item_stack(c)?;
                if slot & 0x80 == 0 {
                    break;
                }
            }
            Ok(())
        }
        0x7f => {
            c.varint("item entity id")?;
            c.varint("collector entity id")?;
            c.varint("pickup count")?;
            Ok(())
        }
        0x86 => {
            c.varint("entity id")?;
            let n = c.count_last(1024, "attributes")?;
            for _ in 0..n {
                c.varint("attribute holder id")?;
                c.f64("attribute base")?;
                let m = c.count(256, "attribute modifiers")?;
                for _ in 0..m {
                    c.identifier("modifier id")?;
                    c.f64("modifier amount")?;
                    c.varint("modifier operation")?;
                }
            }
            Ok(())
        }
        _ => fail(c.consumed(), "family mismatch"),
    }
}

fn decode_inventory_frame(id: i32, c: &mut Cursor) -> DResult<()> {
    match id {
        0x11 => {
            c.varint("container id")?;
            Ok(())
        }
        0x12 => {
            c.varint("container id")?;
            c.varint("state id")?;
            let n = c.count(65536, "container slots")?;
            for _ in 0..n {
                skip_item_stack(c)?;
            }
            skip_item_stack(c)?;
            Ok(())
        }
        0x13 => {
            c.varint("container id")?;
            c.i16("data id")?;
            c.i16("data value")?;
            Ok(())
        }
        0x14 => {
            c.varint("container id")?;
            c.varint("state id")?;
            c.i16("slot")?;
            skip_item_stack(c)?;
            Ok(())
        }
        0x3c => {
            c.varint("container id")?;
            c.varint("menu type")?;
            skip_component(c)?;
            Ok(())
        }
        0x62 => skip_item_stack(c),
        0x6b => {
            let slot = c.varint("held slot")?;
            if !(0..9).contains(&slot) {
                return fail(c.consumed(), format!("held slot {slot}"));
            }
            Ok(())
        }
        0x6e => {
            let slot = c.varint("inventory slot")?;
            if !(0..=45).contains(&slot) {
                return fail(c.consumed(), format!("inventory slot {slot}"));
            }
            skip_item_stack(c)?;
            Ok(())
        }
        _ => fail(c.consumed(), "family mismatch"),
    }
}

/// Join and steady-state frames (first split: session and player
/// state, second: recipe and advancement payloads).
fn decode_join_frame(id: i32, c: &mut Cursor) -> DResult<()> {
    match id {
        0x32 | 0x41 | 0x47 | 0x49 | 0x53 | 0x54 | 0x58 | 0x60 | 0x63 | 0x69 | 0x6a => {
            decode_state_frame(id, c)
        }
        0x4b | 0x4d | 0x82 | 0x83 | 0x85 | 0x88 => decode_payload_frame(id, c),
        _ => decode_steady_frame(id, c),
    }
}

fn decode_state_frame(id: i32, c: &mut Cursor) -> DResult<()> {
    match id {
        0x32 => {
            c.i32("player id")?;
            c.bool("hardcore")?;
            let n = c.count(64, "levels")?;
            for _ in 0..n {
                c.identifier("level key")?;
            }
            c.varint("max players")?;
            c.varint("chunk radius")?;
            c.varint("simulation distance")?;
            c.bool("reduced debug info")?;
            c.bool("show death screen")?;
            c.bool("do limited crafting")?;
            skip_spawn_info(c)?;
            c.bool("online mode")?;
            c.bool("enforces secure chat")?;
            Ok(())
        }
        0x41 => {
            let flags = c.u8("abilities flags")?;
            if flags & !0x0f != 0 {
                return fail(c.consumed(), format!("abilities flags {flags}"));
            }
            c.f32("flying speed")?;
            c.f32("walking speed")?;
            Ok(())
        }
        0x47 => {
            let actions = c.u8("player info actions")?;
            if actions == 0 {
                return fail(c.consumed(), "player info action set is empty");
            }
            let n = c.count_last(1024, "player info entries")?;
            for _ in 0..n {
                c.uuid("profile uuid")?;
                for bit in 0..8 {
                    if actions & (1 << bit) == 0 {
                        continue;
                    }
                    match bit {
                        0 => {
                            c.string(16, "player name")?;
                            let p = c.count(16, "player properties")?;
                            for _ in 0..p {
                                c.string(64, "property name")?;
                                c.string(32760, "property value")?;
                                if c.bool("property signed")? {
                                    c.string(32760, "property signature")?;
                                }
                            }
                        }
                        1 => {
                            if c.bool("has chat session")? {
                                c.uuid("chat session id")?;
                                c.i64("key expiry")?;
                                c.byte_array(512, "public key")?;
                                c.byte_array(4096, "key signature")?;
                            }
                        }
                        2 | 4 | 6 => {
                            c.varint("action value")?;
                        }
                        3 | 7 => {
                            c.bool("action value")?;
                        }
                        5 if c.bool("has display name")? => {
                            skip_component(c)?;
                        }
                        _ => {}
                    }
                }
            }
            Ok(())
        }
        0x49 => {
            c.varint("teleport id")?;
            c.f64("position x")?;
            c.f64("position y")?;
            c.f64("position z")?;
            c.f64("delta x")?;
            c.f64("delta y")?;
            c.f64("delta z")?;
            c.f32("y rot")?;
            c.f32("x rot")?;
            c.i32("relative flags")?;
            Ok(())
        }
        0x53 => {
            let n = c.count(1024, "post effects")?;
            for _ in 0..n {
                c.identifier("post effect id")?;
            }
            Ok(())
        }
        0x54 => {
            skip_spawn_info(c)?;
            c.u8("data to keep")?;
            Ok(())
        }
        0x58 => {
            skip_component(c)?;
            if c.bool("has server icon")? {
                c.byte_array(1 << 20, "server icon")?;
            }
            Ok(())
        }
        0x60 => {
            c.varint("chunk cache center x")?;
            c.varint("chunk cache center z")?;
            Ok(())
        }
        0x63 => {
            c.identifier("respawn dimension")?;
            c.block_pos()?;
            c.f32("respawn yaw")?;
            c.f32("respawn pitch")?;
            Ok(())
        }
        0x69 => {
            c.f32("experience progress")?;
            c.varint("experience level")?;
            c.varint("total experience")?;
            Ok(())
        }
        0x6a => {
            c.f32("health")?;
            c.varint("food")?;
            c.f32("saturation")?;
            Ok(())
        }
        _ => fail(c.consumed(), "family mismatch"),
    }
}

fn decode_payload_frame(id: i32, c: &mut Cursor) -> DResult<()> {
    match id {
        0x4b => {
            let n = c.count(1 << 16, "recipe book entries")?;
            for _ in 0..n {
                c.varint("recipe display id")?;
                skip_recipe_display(c)?;
                c.varint("recipe group")?;
                c.varint("recipe book category")?;
                if c.bool("has crafting requirements")? {
                    let r = c.count(256, "crafting requirements")?;
                    for _ in 0..r {
                        skip_holder_set(c)?;
                    }
                }
                c.u8("recipe book entry flags")?;
            }
            c.bool("replace")?;
            Ok(())
        }
        0x4d => {
            for what in [
                "crafting open",
                "crafting filtering",
                "furnace open",
                "furnace filtering",
                "blast furnace open",
                "blast furnace filtering",
                "smoker open",
                "smoker filtering",
            ] {
                c.bool(what)?;
            }
            Ok(())
        }
        0x82 => {
            c.f32("tick rate")?;
            c.bool("is frozen")?;
            Ok(())
        }
        0x83 => {
            c.varint("tick steps")?;
            Ok(())
        }
        0x85 => {
            c.bool("should reset")?;
            let n = c.count(1 << 16, "added advancements")?;
            for _ in 0..n {
                c.identifier("advancement id")?;
                if c.bool("has parent")? {
                    c.identifier("advancement parent")?;
                }
                if c.bool("has display")? {
                    skip_advancement_display(c)?;
                }
                let r = c.count(1 << 16, "advancement requirement lists")?;
                for _ in 0..r {
                    let e = c.count(1 << 16, "advancement requirements")?;
                    for _ in 0..e {
                        c.string(32760, "requirement name")?;
                    }
                }
                c.bool("sends telemetry")?;
                c.f32("advancement x")?;
                c.f32("advancement y")?;
            }
            let d = c.count(1 << 16, "removed advancements")?;
            for _ in 0..d {
                c.identifier("removed advancement id")?;
            }
            let p = c.count(1 << 16, "advancement progress entries")?;
            for _ in 0..p {
                c.identifier("progress advancement id")?;
                let m = c.count(1 << 16, "progress criteria")?;
                for _ in 0..m {
                    c.string(32760, "criterion name")?;
                    if c.bool("criterion obtained")? {
                        c.i64("criterion obtained at")?;
                    }
                }
            }
            c.bool("show advancements")?;
            Ok(())
        }
        0x88 => {
            let n = c.count(1024, "recipe property sets")?;
            for _ in 0..n {
                c.identifier("property set key")?;
                let e = c.count(1 << 16, "property set items")?;
                for _ in 0..e {
                    c.varint("item id")?;
                }
            }
            let s = c.count_last(1 << 16, "stonecutter recipes")?;
            for _ in 0..s {
                skip_holder_set(c)?;
                skip_slot_display(c, 0)?;
            }
            Ok(())
        }
        _ => fail(c.consumed(), "family mismatch"),
    }
}
/// The remaining join-burst and steady-state frames.
fn decode_steady_frame(id: i32, c: &mut Cursor) -> DResult<()> {
    match id {
        0x00 => {
            if c.remaining() != 0 {
                return fail(0, "bundle_delimiter body not empty");
            }
            Ok(())
        }
        0x0a => {
            let d = c.u8("difficulty")?;
            if d > 3 {
                return fail(c.consumed(), format!("difficulty {d}"));
            }
            c.bool("difficulty locked")?;
            Ok(())
        }
        0x0b => {
            c.varint("batch size")?;
            Ok(())
        }
        0x0c => {
            if c.remaining() != 0 {
                return fail(0, "chunk_batch_start body not empty");
            }
            Ok(())
        }
        0x10 => {
            let n = c.count(1 << 16, "command nodes")?;
            for _ in 0..n {
                skip_command_node(c)?;
            }
            let root = c.varint("command root index")?;
            if root < 0 || root as usize >= n {
                return fail(c.consumed(), format!("command root index {root}"));
            }
            Ok(())
        }
        0x1f => {
            // delete_chat: one signature id (uuid).
            c.uuid("signature id")
        }
        0x24 => {
            c.f64("explosion x")?;
            c.f64("explosion y")?;
            c.f64("explosion z")?;
            c.f32("explosion radius")?;
            c.i32("block count")?;
            if c.bool("has player knockback")? {
                c.f64("knockback x")?;
                c.f64("knockback y")?;
                c.f64("knockback z")?;
            }
            skip_particle(c)?;
            skip_holder_or_direct(c, direct_sound_event, "explosion sound")?;
            let n = c.count(1024, "block particles")?;
            for _ in 0..n {
                skip_particle(c)?;
                c.f32("particle scaling")?;
                c.f32("particle speed")?;
                c.varint("particle weight")?;
            }
            c.bool("play sound")?;
            Ok(())
        }
        0x2c => {
            c.f64("border center x")?;
            c.f64("border center z")?;
            c.f64("border old size")?;
            c.f64("border new size")?;
            c.varlong("border lerp time")?;
            c.varint("border absolute max size")?;
            c.varint("border warning blocks")?;
            c.varint("border warning time")?;
            Ok(())
        }
        0x2d => {
            c.i64("keep alive id")?;
            Ok(())
        }
        0x73 => {
            c.i64("game time")?;
            let n = c.count_last(64, "clock updates")?;
            for _ in 0..n {
                c.varint("clock holder id")?;
                c.varlong("clock total ticks")?;
                c.f32("clock partial tick")?;
                c.f32("clock rate")?;
            }
            Ok(())
        }
        0x45 => {
            c.varint("player id")?;
            skip_component(c)?;
            Ok(())
        }
        0x7c => {
            skip_component(c)?;
            c.bool("overlay")?;
            Ok(())
        }
        0x80 => {
            c.varint("entity id")?;
            c.f64("position x")?;
            c.f64("position y")?;
            c.f64("position z")?;
            c.f64("delta x")?;
            c.f64("delta y")?;
            c.f64("delta z")?;
            c.f32("y rot")?;
            c.f32("x rot")?;
            c.i32("relative flags")?;
            c.bool("on ground")?;
            Ok(())
        }
        _ => fail(c.consumed(), "family mismatch"),
    }
}
// ---------------------------------------------------------------------
// run_pass
// ---------------------------------------------------------------------

/// The frame-order phase machine: login until login_finished, config
/// until the empty config 0x03, play afterwards.
struct PhaseMachine {
    phase: Phase,
}

impl PhaseMachine {
    fn new() -> PhaseMachine {
        PhaseMachine {
            phase: Phase::Login,
        }
    }

    fn classify(&mut self, id: i32, body_len: usize) -> Phase {
        let phase = self.phase;
        match phase {
            Phase::Login => {
                if id == 0x02 {
                    self.phase = Phase::Config;
                }
            }
            Phase::Config => {
                if id == 0x03 && body_len == 0 {
                    self.phase = Phase::Play;
                }
            }
            Phase::Play => {}
        }
        phase
    }
}

/// The pass result over one capture set.
pub struct PassReport {
    pub frames: usize,
    pub decoded_ids: BTreeMap<&'static str, BTreeMap<i32, usize>>,
    pub failures: Vec<FrameFailure>,
}

impl PassReport {
    fn new() -> PassReport {
        let mut decoded_ids = BTreeMap::new();
        for phase in [Phase::Login, Phase::Config, Phase::Play] {
            decoded_ids.entry(phase.label()).or_default();
        }
        PassReport {
            frames: 0,
            decoded_ids,
            failures: Vec::new(),
        }
    }
}

/// Walks captured packets (with dumped bodies), decoding every frame.
fn run_over_packets(
    pkts: &[CapturedPacket],
    dir: &Path,
    label: &str,
    report: &mut PassReport,
) -> Result<()> {
    let mut machine = PhaseMachine::new();
    for p in pkts {
        if p.id < 0 {
            if let Some(note) = &p.note {
                println!("[strict] {label} transcript end: {note}");
            }
            break;
        }
        let body = match &p.file {
            Some(f) => std::fs::read(dir.join(f))
                .with_context(|| format!("reading dump {f} for session {label}"))?,
            None => hex::decode(&p.head_hex).context("head hex")?,
        };
        let phase = machine.classify(p.id, body.len());
        report.frames += 1;
        *report
            .decoded_ids
            .entry(phase.label())
            .or_default()
            .entry(p.id)
            .or_insert(0) += 1;
        if let Err(mut f) = check_frame(phase, p.id, &body) {
            f.frame = format!("{label}/{}", p.file.as_deref().unwrap_or("<head-only>"));
            report.failures.push(f);
        }
    }
    Ok(())
}

/// Runs the pass over an existing capture directory (manifest + dumps).
fn run_over_capture_dir(dir: &Path, report: &mut PassReport) -> Result<()> {
    let manifest: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json"))?)
            .with_context(|| format!("parsing {}", dir.join("manifest.json").display()))?;
    let mut machine = PhaseMachine::new();
    for e in &manifest {
        let id = e["id"].as_i64().context("manifest id")? as i32;
        let file = e["file"].as_str().context("manifest file")?.to_string();
        let body =
            std::fs::read(dir.join(&file)).with_context(|| format!("reading dump {file}"))?;
        let phase = machine.classify(id, body.len());
        report.frames += 1;
        *report
            .decoded_ids
            .entry(phase.label())
            .or_default()
            .entry(id)
            .or_insert(0) += 1;
        if let Err(mut f) = check_frame(phase, id, &body) {
            f.frame = format!("{}:{}", dir.display(), file);
            report.failures.push(f);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// validate_blobs
// ---------------------------------------------------------------------

/// Reads the varint at `at` and re-encodes it with a delta applied,
/// splicing the new encoding into a copy of the body.
fn patch_varint(body: &[u8], at: usize, delta: i64) -> Option<Vec<u8>> {
    let mut end = at;
    while end < body.len() {
        let b = body[end];
        end += 1;
        if b & 0x80 == 0 {
            break;
        }
        if end - at > 5 {
            return None;
        }
    }
    if end >= body.len() && body[body.len() - 1] & 0x80 != 0 {
        return None;
    }
    let raw = &body[at..end];
    let mut value: u64 = 0;
    for (i, b) in raw.iter().enumerate() {
        value |= u64::from(b & 0x7f) << (7 * i);
    }
    let patched = (value as i64).wrapping_add(delta) as u32;
    let mut out = body[..at].to_vec();
    let mut v = patched;
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if v == 0 {
            break;
        }
    }
    out.extend_from_slice(&body[end..]);
    Some(out)
}

/// The corruption harness: every vanilla blob frame with a decoder must
/// pass, and every corrupted variant must fail.
pub struct VariantStats {
    pub frames: usize,
    pub variants: usize,
    pub survivors: Vec<String>,
    pub good_failures: Vec<String>,
}

fn validate_blobs(blobs_dir: &Path) -> Result<VariantStats> {
    let manifest: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(blobs_dir.join("manifest.json"))?)?;
    let mut stats = VariantStats {
        frames: 0,
        variants: 0,
        survivors: Vec::new(),
        good_failures: Vec::new(),
    };
    let mut machine = PhaseMachine::new();
    for e in &manifest {
        let id = e["id"].as_i64().context("manifest entry has no id")? as i32;
        let file = e["file"]
            .as_str()
            .context("manifest entry has no file")?
            .to_string();
        let body = std::fs::read(blobs_dir.join(&file))?;
        let phase = machine.classify(id, body.len());
        if !has_decoder(phase, id) {
            continue;
        }
        stats.frames += 1;
        let mut counts = Vec::new();
        match check_frame(phase, id, &body) {
            Ok(_) => {
                // Re-run purely to harvest the recorded count offsets.
                let mut c = Cursor::new(&body);
                if decode_body(phase, id, &mut c).is_ok() {
                    counts = c.count_offsets().to_vec();
                }
            }
            Err(f) => stats.good_failures.push(format!(
                "{file} {name}: good frame failed: {reason}",
                name = f.name,
                reason = f.reason
            )),
        }
        // A config custom_payload's channel string is followed by data
        // defined as the rest of the frame, so no byte change can break
        // the decode; no variant of it proves anything.
        if matches!((phase, id), (Phase::Config, 0x01)) {
            continue;
        }
        let mut variants: Vec<(String, Vec<u8>)> = Vec::new();
        if !body.is_empty() {
            variants.push(("truncate-1".into(), body[..body.len() - 1].to_vec()));
        }
        if body.len() >= 2 {
            variants.push(("truncate-half".into(), body[..body.len() / 2].to_vec()));
        }
        let mut padded = body.clone();
        padded.push(0x00);
        variants.push(("pad-1".into(), padded));
        for at in counts {
            for (label, delta) in [("count+1", 1i64), ("count-1", -1i64)] {
                if let Some(patched) = patch_varint(&body, at, delta) {
                    variants.push((label.into(), patched));
                }
            }
        }
        for (label, variant) in variants {
            stats.variants += 1;
            if check_frame(phase, id, &variant).is_ok() {
                stats.survivors.push(format!("{file} {label}"));
            }
        }
    }
    Ok(stats)
}

// ---------------------------------------------------------------------
// The report
// ---------------------------------------------------------------------

fn coverage(phase: Phase) -> serde_json::Value {
    let decoded = decoded_ids(phase);
    let gaps: Vec<serde_json::Value> = table(phase)
        .iter()
        .filter(|(id, _)| !decoded.contains(id))
        .map(|(id, name)| json!({ "id": id, "name": name }))
        .collect();
    json!({
        "known": table(phase).len(),
        "decoded": decoded.len(),
        "gaps": gaps,
    })
}

fn write_report(
    dir: &Path,
    pass: &PassReport,
    variants: &VariantStats,
    emitted_undecoded: &[String],
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let ids = |phase: Phase| -> Vec<serde_json::Value> {
        pass.decoded_ids
            .get(phase.label())
            .map(|m| {
                m.iter()
                    .map(|(id, n)| {
                        json!({ "id": id, "name": name_of(phase, *id).unwrap_or("?"), "frames": n })
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let doc = json!({
        "frames": pass.frames,
        "failures": pass.failures.iter().map(|f| json!({
            "frame": f.frame,
            "phase": f.phase,
            "id": f.id,
            "name": f.name,
            "consumed": f.consumed,
            "len": f.len,
            "reason": f.reason,
        })).collect::<Vec<_>>(),
        "decoded_ids": {
            "login": ids(Phase::Login),
            "config": ids(Phase::Config),
            "play": ids(Phase::Play),
        },
        "coverage": {
            "login": coverage(Phase::Login),
            "config": coverage(Phase::Config),
            "play": coverage(Phase::Play),
        },
        "emitted_but_undecoded": emitted_undecoded,
        "variants": {
            "frames": variants.frames,
            "generated": variants.variants,
            "survivors": variants.survivors,
            "good_failures": variants.good_failures,
        },
    });
    std::fs::write(
        dir.join("report.json"),
        serde_json::to_string_pretty(&doc)? + "\n",
    )?;
    let mut text = String::new();
    text.push_str(&format!("frames decoded: {}\n", pass.frames));
    text.push_str(&format!("failures: {}\n", pass.failures.len()));
    for f in pass.failures.iter().take(40) {
        text.push_str(&format!(
            "  {} {} {name} id {id:#04x}: consumed {consumed} of {len}: {reason}\n",
            f.frame,
            f.phase,
            name = f.name,
            id = f.id,
            consumed = f.consumed,
            len = f.len,
            reason = f.reason
        ));
    }
    for phase in [Phase::Login, Phase::Config, Phase::Play] {
        let gaps: Vec<&str> = table(phase)
            .iter()
            .filter(|(id, _)| !decoded_ids(phase).contains(id))
            .map(|(_, name)| *name)
            .collect();
        let seen = pass
            .decoded_ids
            .get(phase.label())
            .map(|m| m.len())
            .unwrap_or(0);
        text.push_str(&format!(
            "{}: {} known ids, {} decoded, {} distinct seen, gaps: {}\n",
            phase.label(),
            table(phase).len(),
            decoded_ids(phase).len(),
            seen,
            if gaps.is_empty() {
                "none".to_string()
            } else {
                gaps.join(", ")
            }
        ));
    }
    text.push_str(&format!(
        "emitted-but-undecoded: {}\n",
        if emitted_undecoded.is_empty() {
            "none".to_string()
        } else {
            emitted_undecoded.join(", ")
        }
    ));
    text.push_str(&format!(
        "vanilla blob validation: {} frames, {} variants, {} survivors\n",
        variants.frames,
        variants.variants,
        variants.survivors.len()
    ));
    for s in variants.survivors.iter().take(20) {
        text.push_str(&format!("  survived: {s}\n"));
    }
    for s in variants.good_failures.iter().take(20) {
        text.push_str(&format!("  good frame failed: {s}\n"));
    }
    std::fs::write(dir.join("report.txt"), &text)?;
    println!("[strict] report at {}", dir.join("report.txt").display());
    Ok(())
}

/// Ids seen on the wire without a decoder: emitted-but-undecoded.
fn emitted_undecoded(pass: &PassReport) -> Vec<String> {
    let mut out = Vec::new();
    for (phase, ids) in &pass.decoded_ids {
        let phase = match *phase {
            "login" => Phase::Login,
            "config" => Phase::Config,
            _ => Phase::Play,
        };
        for id in ids.keys() {
            if !has_decoder(phase, *id) {
                out.push(format!("{phase:?} {id:#04x}"));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------
// The strict-decode command
// ---------------------------------------------------------------------

/// The actor session's command volley: block writes, a chest menu cycle,
/// and an item grant. The mob scenario runs in its own later session so
/// the actor survives to dig.
fn session_commands() -> Vec<String> {
    [
        "tp @s 1.5 -60.0 3.5",
        "setblock 0 -60 0 minecraft:stone",
        "setblock 1 -60 0 minecraft:dirt",
        "setblock 2 -60 0 minecraft:oak_planks",
        "setblock 3 -60 1 minecraft:torch",
        "setblock 9 -60 11 minecraft:stone",
        "setblock 10 -60 11 minecraft:torch",
        "setblock 4 -60 4 minecraft:chest[facing=north,type=single,waterlogged=false]",
        "give @s minecraft:stone 64",
        // The wire parser wants five tokens; the fifth is ignored.
        "opencontainer 4 -60 4 0",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// The mob session's command volley: the three summons under a frozen
/// clock, stepped in short barriers (a single long step outlives the
/// volley's reply window).
fn mob_commands() -> Vec<String> {
    let mut commands = vec![
        "tp @s 10.5 -60.0 10.5".to_string(),
        "tick freeze".to_string(),
        "summon minecraft:skeleton 15.5 -60 10.5".to_string(),
        "summon minecraft:spider 10.5 -60 14.5".to_string(),
        "summon minecraft:creeper 6.5 -60 6.5".to_string(),
    ];
    for _ in 0..7 {
        commands.push("tick step 100".to_string());
    }
    commands.push("tick unfreeze".to_string());
    commands
}

/// Prebuilt serverbound frames for the post-volley interactions: the
/// menu click and close, a hotbar select, and a dig cycle.
fn session_raw_packets() -> Vec<(i32, Vec<u8>)> {
    let mut click = Vec::new();
    doppel_protocol::write_varint(&mut click, 1); // container id
    doppel_protocol::write_varint(&mut click, 1); // state id
    click.extend_from_slice(&10i16.to_be_bytes()); // slot
    click.push(0); // button
    doppel_protocol::write_varint(&mut click, 0); // click kind: pickup
    doppel_protocol::write_varint(&mut click, 0); // changed slots
    click.push(0); // carried: absent
    let mut carried = Vec::new();
    carried.extend_from_slice(&1i16.to_be_bytes());
    let dig = |action: i32, pos: (i32, i32, i32), seq: i32| {
        let mut b = Vec::new();
        doppel_protocol::write_varint(&mut b, action);
        b.extend_from_slice(&bot::pack_block_pos(pos.0, pos.1, pos.2).to_be_bytes());
        doppel_protocol::write_varint(&mut b, 1); // direction: up
        doppel_protocol::write_varint(&mut b, seq);
        b
    };
    vec![
        (0x2c, Vec::new()),              // player_loaded
        (0x36, carried),                 // set_carried_item
        (0x12, click),                   // container_click
        (0x13, vec![1]),                 // container_close (id 1)
        (0x29, dig(0, (3, -60, 1), 20)), // start digging the torch
        (0x29, dig(0, (2, -60, 0), 21)), // start digging stone (stages)
        (0x29, dig(1, (2, -60, 0), 21)), // stop digging
    ]
}

/// Boots vanilla on this gate's port, snapshots the pristine world, and
/// captures the clean join blobs (no commands run, so the world the
/// actor session later edits is untouched).
fn capture_clean_blobs_on(
    pin: &doppel_protocol::Pin,
    jar: &Path,
    blobs_dir: &Path,
    pristine_world: &Path,
    port: u16,
) -> Result<()> {
    let server = vanilla::boot(pin, jar, port)?;
    std::thread::sleep(Duration::from_secs(2));
    if pristine_world.exists() {
        std::fs::remove_dir_all(pristine_world)?;
    }
    let world = vanilla::run_world()?;
    anyhow::ensure!(world.is_dir(), "vanilla world dir missing after boot");
    std::fs::create_dir_all(pristine_world)?;
    for entry in std::fs::read_dir(&world)? {
        let entry = entry?;
        if entry.file_name() == "session.lock" {
            continue;
        }
        let from = entry.path();
        let to = pristine_world.join(entry.file_name());
        if from.is_dir() {
            copy_dir(&from, &to)?;
        } else {
            std::fs::copy(&from, &to).with_context(|| format!("copying {}", from.display()))?;
        }
    }
    let login = capture::login_start_c("Doppel");
    let protocol = pin.protocol.unwrap_or(0);
    let v = bot::login_capture(
        "127.0.0.1",
        port,
        protocol,
        &login,
        &CaptureOpts {
            idle_timeout: Some(Duration::from_secs(5)),
            max_packets: Some(300),
            dump_dir: Some(blobs_dir),
            commands: &[],
            walk_chunks: None,
            raw_packets: &[],
        },
    )
    .context("capturing clean vanilla join")?;
    drop(server);
    let entries = capture::write_manifest(&v, blobs_dir)?;
    anyhow::ensure!(entries > 0, "no blobs captured from vanilla");
    println!(
        "[oracle] clean join blobs ({} packets) + pristine world at {}",
        v.iter().filter(|p| p.id >= 0).count(),
        pristine_world.display()
    );
    Ok(())
}

/// Recursively copies a directory.
fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir(&from, &to)?;
        } else {
            std::fs::copy(&from, &to).with_context(|| format!("copying {}", from.display()))?;
        }
    }
    Ok(())
}

/// Boots Doppel with the clean blobs and the pristine world, runs the
/// witness and actor sessions with full dumps, then the pass.
fn capture_doppel(
    blobs_dir: &Path,
    pristine_world: &Path,
    capture_root: &Path,
) -> Result<PassReport> {
    let bin = default_doppel_bin()?;
    let pin_path = doppel_protocol::pin_path()?;
    let port = doppel_port();
    let mut child = std::process::Command::new(&bin)
        .env("DOPPEL_ADDR", "127.0.0.1")
        .env("DOPPEL_PORT", port.to_string())
        .env("DOPPEL_PIN", &pin_path)
        .env("DOPPEL_BLOBS", blobs_dir)
        .env("DOPPEL_WORLD", pristine_world)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawning {}", bin.display()))?;
    let result = (|| -> Result<PassReport> {
        wait_for_port(port, Duration::from_secs(30))?;
        let protocol = doppel_protocol::load_pin()?.protocol.unwrap_or(0);
        for dir in ["wit", "act", "mob", "rej"] {
            let p = capture_root.join(dir);
            if p.exists() {
                std::fs::remove_dir_all(&p)?;
            }
        }
        // The witness joins first, walks the chunk stream, and observes
        // the other sessions' broadcasts (the dig destruction overlay
        // reaches every player but the digger).
        let wit_dir = capture_root.join("wit");
        let witness_login = capture::login_start_c("Doppelist2");
        let witness = std::thread::spawn(move || {
            bot::login_capture(
                "127.0.0.1",
                port,
                protocol,
                &witness_login,
                &CaptureOpts {
                    // The join replay's chunk-rebuild stalls exceed any
                    // short idle, so a long one carries the session; the
                    // cap, set just above the content, is what ends it.
                    idle_timeout: Some(Duration::from_secs(60)),
                    max_packets: Some(670),
                    dump_dir: Some(&wit_dir),
                    commands: &[],
                    walk_chunks: Some(4),
                    raw_packets: &[],
                },
            )
        });
        std::thread::sleep(Duration::from_secs(8));
        // The actor edits blocks, opens and clicks the chest, and digs;
        // the mob scenario waits for its own session so the actor lives
        // to finish the dig cycle.
        let act_dir = capture_root.join("act");
        let login = capture::login_start_c("Doppel");
        let commands = session_commands();
        let raw = session_raw_packets();
        let actor = bot::login_capture(
            "127.0.0.1",
            port,
            protocol,
            &login,
            &CaptureOpts {
                idle_timeout: Some(Duration::from_secs(45)),
                max_packets: Some(580),
                dump_dir: Some(&act_dir),
                commands: &commands,
                walk_chunks: None,
                raw_packets: &raw,
            },
        )
        .context("capturing actor session")?;
        // The mob scenario: summons under a frozen clock, stepped in
        // barriers, then the clock resumes and the survivors keep
        // moving until the cap.
        let mob_dir = capture_root.join("mob");
        let mob_login = capture::login_start_c("Doppelist3");
        let mob_cmds = mob_commands();
        let mobber = bot::login_capture(
            "127.0.0.1",
            port,
            protocol,
            &mob_login,
            &CaptureOpts {
                idle_timeout: Some(Duration::from_secs(45)),
                max_packets: Some(800),
                dump_dir: Some(&mob_dir),
                commands: &mob_cmds,
                walk_chunks: None,
                raw_packets: &[],
            },
        )
        .context("capturing mob session")?;
        let witness = witness
            .join()
            .map_err(|_| anyhow::anyhow!("witness thread panicked"))?
            .context("capturing witness session")?;
        // Manifests make the dumped sessions replayable by --capture-dir.
        capture::write_manifest(&witness, &capture_root.join("wit"))?;
        capture::write_manifest(&actor, &capture_root.join("act"))?;
        capture::write_manifest(&mobber, &capture_root.join("mob"))?;
        // The rejoin: the actor's disconnect saved its playerdata, so a
        // second join of the same name replays the burst rewritten with
        // the saved pose - the path the rewritten packets live on.
        let rej_dir = capture_root.join("rej");
        if rej_dir.exists() {
            std::fs::remove_dir_all(&rej_dir)?;
        }
        let rejoin = bot::login_capture(
            "127.0.0.1",
            port,
            protocol,
            &capture::login_start_c("Doppel"),
            &CaptureOpts {
                idle_timeout: Some(Duration::from_secs(20)),
                max_packets: Some(400),
                dump_dir: Some(&rej_dir),
                commands: &[],
                walk_chunks: None,
                raw_packets: &[],
            },
        )
        .context("capturing rejoin session")?;
        capture::write_manifest(&rejoin, &rej_dir)?;
        let mut report = PassReport::new();
        run_over_packets(&witness, &capture_root.join("wit"), "witness", &mut report)?;
        run_over_packets(&actor, &capture_root.join("act"), "actor", &mut report)?;
        run_over_packets(&mobber, &capture_root.join("mob"), "mob", &mut report)?;
        run_over_packets(&rejoin, &rej_dir, "rejoin", &mut report)?;
        Ok(report)
    })();
    let _ = child.kill();
    let _ = child.wait();
    result
}

/// The strict-decode entry point.
pub fn run(capture_dir: Option<&str>) -> Result<()> {
    let root = doppel_protocol::find_repo_root()?;
    let report_dir = root.join("target").join("oracle").join("strict-decode");

    let (pass, variants) = match capture_dir {
        Some(dir) => {
            let dir_path = PathBuf::from(dir);
            anyhow::ensure!(
                dir_path.join("manifest.json").is_file(),
                "{} has no manifest.json",
                dir_path.display()
            );
            let mut pass = PassReport::new();
            run_over_capture_dir(&dir_path, &mut pass)?;
            let variants = match std::env::var("STRICT_VALIDATE_DIR")
                .ok()
                .map(PathBuf::from)
                .or_else(|| default_blobs_dir(&root).ok())
            {
                Some(dir) => validate_blobs(&dir)?,
                None => VariantStats {
                    frames: 0,
                    variants: 0,
                    survivors: Vec::new(),
                    good_failures: Vec::new(),
                },
            };
            (pass, variants)
        }
        None => {
            let pin = doppel_protocol::load_pin()?;
            let jar = vanilla::ensure_jar(&pin)?;
            let blobs_dir = root.join("target").join("vanilla").join("blobs-strict");
            let pristine_world = root
                .join("target")
                .join("vanilla")
                .join("pristine-world-strict");
            let capture_root = root.join("target").join("oracle").join("strict-capture");
            for dir in [&blobs_dir, &pristine_world] {
                if dir.exists() {
                    std::fs::remove_dir_all(dir)?;
                }
            }
            capture_clean_blobs_on(&pin, &jar, &blobs_dir, &pristine_world, vanilla_port())?;
            let variants = validate_blobs(&blobs_dir)?;
            let pass = capture_doppel(&blobs_dir, &pristine_world, &capture_root)?;
            (pass, variants)
        }
    };

    let undecoded = emitted_undecoded(&pass);
    write_report(&report_dir, &pass, &variants, &undecoded)?;

    // Emitted-but-undecoded frames already fail individually above;
    // counting their ids again would double-report them.
    let mut failures = pass.failures.len();
    failures += variants.survivors.len();
    failures += variants.good_failures.len();
    if failures == 0 {
        println!(
            "PASS: strict decode: {} frames, 0 failures; {} blob variants, 0 survivors",
            pass.frames, variants.variants
        );
        Ok(())
    } else {
        println!(
            "FAIL: strict decode: {failures} failure(s) (see {})",
            report_dir.join("report.txt").display()
        );
        std::process::exit(1);
    }
}

fn default_blobs_dir(root: &Path) -> Result<PathBuf> {
    let dir = root.join("target").join("vanilla").join("blobs");
    anyhow::ensure!(
        dir.join("manifest.json").is_file(),
        "no blob set at {} - run the capture first",
        dir.display()
    );
    Ok(dir)
}

// ---------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------

#[cfg(test)]
#[path = "strict_decode_tests.rs"]
mod tests;
