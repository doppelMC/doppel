//! Lush-cave features: springs and the cave-surface ground patches with
//! the vegetation that grows on them.
//!
//! Every draw the reference features make is reproduced in shape and
//! order, because one feature's leftover stream moves every later
//! placement of that same feature.

use serde_json::Value;

use crate::decoration::{DecorRng, Decorator};

/// Whether a block name is one of the air family.
pub(crate) fn is_air_name(name: &str) -> bool {
    matches!(
        name,
        "minecraft:air" | "minecraft:cave_air" | "minecraft:void_air"
    )
}

/// Places a fluid pocket embedded in stone: the walls around the cell
/// must be the config's blocks with exactly the configured open sides,
/// and the draw stream stands still. The source fluid's legacy block is
/// the plain fluid block.
pub(crate) fn run_spring(
    d: &mut Decorator,
    v: &Value,
    _rng: &mut DecorRng,
    x: i32,
    y: i32,
    z: i32,
) -> bool {
    let Some(fluid) = v
        .get("state")
        .and_then(|s| s.get("id"))
        .and_then(Value::as_str)
    else {
        return false;
    };
    let Some(state) = d.state_id_of(fluid, "") else {
        return false;
    };
    let requires_below = v
        .get("requires_block_below")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let rock_count = int_field(v, "rock_count", 4);
    let hole_count = int_field(v, "hole_count", 1);
    let Some(valid) = v.get("valid_blocks").and_then(Value::as_array) else {
        return false;
    };
    let valid: Vec<&str> = valid.iter().filter_map(Value::as_str).collect();
    if valid.is_empty() {
        return false;
    }
    let name_at = |d: &Decorator, x: i32, y: i32, z: i32| -> String {
        d.block_name(d.block(x, y, z)).to_string()
    };
    let is_valid = move |name: &str| valid.contains(&name);
    if !is_valid(&name_at(d, x, y + 1, z)) {
        return false;
    }
    if requires_below && !is_valid(&name_at(d, x, y - 1, z)) {
        return false;
    }
    let here = name_at(d, x, y, z);
    if !is_air_name(&here) && !is_valid(&here) {
        return false;
    }
    let mut rocks = 0;
    let mut holes = 0;
    for (nx, ny, nz) in [
        (x - 1, y, z),
        (x + 1, y, z),
        (x, y, z - 1),
        (x, y, z + 1),
        (x, y - 1, z),
    ] {
        let name = name_at(d, nx, ny, nz);
        if is_valid(&name) {
            rocks += 1;
        }
        if is_air_name(&name) {
            holes += 1;
        }
    }
    if rocks == rock_count && holes == hole_count {
        d.set_block(x, y, z, state);
        return true;
    }
    false
}

/// A config integer with its vanilla default.
fn int_field(v: &Value, key: &str, default: i32) -> i32 {
    v.get(key)
        .and_then(Value::as_i64)
        .map_or(default, |n| n as i32)
}
