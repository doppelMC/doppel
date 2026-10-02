//! Comparator block (`minecraft:comparator`).
//!
//! Model: the back input is the diode
//! base read of the FACING-side block, REPLACED by the analog value when
//! that block `hasAnalogOutputSignal`; the two side inputs
//! The two side inputs take the
//! max of the control reads. `shouldTurnOn = input != 0 && (input > side
//! || (input == side && COMPARE))`; the emitted value is `0` when `side >
//! input`, `input - side` in SUBTRACT, else `input`. Output changes are
//! scheduled 2gt out (`getDelay` = 2; vanilla hard-codes the literal) and
//! re-evaluated against the live world at fire time — no pulse
//! stretching, no refresh ticker.
//!
//! Deferred (no engine model yet): item-frame
//! entities and the look-through-a-conductor branch (spec 1c), the
//! right-click MODE toggle (`useWithoutItem`), and TickPriority (the
//! scheduler is FIFO within a tick, so the HIGH/NORMAL distinction does
//! not change ordering here).

use doppel_world::registry::BlockRegistry;

use super::{prop_dir, Game, PendingKind, TickAction};

/// Horizontal facing prop -> (dx, dz) pointing at the INPUT side (diode
/// FACING runs from the output face toward the input face, so the input
/// block is `pos + FACING` and the front block is `pos - FACING`).
fn facing_offset(facing: &str) -> (i32, i32) {
    match facing {
        "north" => (0, -1),
        "south" => (0, 1),
        "west" => (-1, 0),
        "east" => (1, 0),
        _ => (0, 0),
    }
}

/// True when the read direction (reader -> source) equals the source
/// diode's `facing` — i.e. the reader sits at the diode's output face.
/// Answers `ownSignal` only into that block.
fn reads_front(read_dx: i32, read_dz: i32, props: &str) -> bool {
    (read_dx, read_dz) == facing_offset(prop_dir(props))
}

/// The should-turn-on decision core.
fn should_turn_on(input: i32, side: i32, compare: bool) -> bool {
    input != 0 && (input > side || (input == side && compare))
}

/// The output-signal calculation core.
fn output_signal(input: i32, side: i32, compare: bool) -> i32 {
    if input == 0 {
        return 0;
    }
    if side > input {
        return 0;
    }
    if compare {
        input
    } else {
        input - side
    }
}

/// `MODE` prop: compare unless `mode=subtract`.
fn mode_is_compare(props: &str) -> bool {
    !props.contains("mode=subtract")
}

/// State-based analog sources (`getAnalogOutputSignal` blocks whose value
/// derives purely from block state). Returns `None` for blocks without an
/// analog output — the caller then keeps the redstone read. Container and
/// block-entity sources (barrel/chest/furnace/jukebox/...) are deferred
/// and also read as `None`.
fn analog_output(name: &str, props: &str) -> Option<i32> {
    let level = |key: &str| BlockRegistry::prop_int(props, key);
    let signal = match name {
        // Cake: (7 - bites) * 2.
        "minecraft:cake" => (7 - level("bites").unwrap_or(0).clamp(0, 6)) * 2,
        // CandleCakeBlock: CakeBlock.FULL_CAKE_SIGNAL.
        "minecraft:candle_cake" => 14,
        // CauldronBlock (empty): analog-capable, default signal 0.
        "minecraft:cauldron" => 0,
        // LayeredCauldronBlock: LEVEL 1..3.
        "minecraft:water_cauldron" | "minecraft:powder_snow_cauldron" => {
            level("level").unwrap_or(0).clamp(0, 3)
        }
        // LavaCauldronBlock: constant 3.
        "minecraft:lava_cauldron" => 3,
        // ComposterBlock: LEVEL 0..8.
        "minecraft:composter" => level("level").unwrap_or(0).clamp(0, 8),
        // Respawn anchor: floor(CHARGE / 4 * 15).
        "minecraft:respawn_anchor" => (level("charges").unwrap_or(0).clamp(0, 4) * 15) / 4,
        // BeehiveBlock: HONEY_LEVEL 0..5 (bee_nest inherits it).
        "minecraft:beehive" | "minecraft:bee_nest" => level("honey_level").unwrap_or(0).clamp(0, 5),
        // CopperBulbBlock: LIT ? 15 : 0.
        "minecraft:copper_bulb" => i32::from(props.contains("lit=true")) * 15,
        // EndPortalFrameBlock: HAS_EYE ? 15 : 0.
        "minecraft:end_portal_frame" => i32::from(props.contains("eye=true")) * 15,
        _ => return None,
    };
    Some(signal.clamp(0, 15))
}

impl Game {
    /// The weak signal the block at
    /// (x, y, z) offers the reader at (x + read_dx, z + read_dz).
    /// `read_*` is the direction from reader toward the source (the
    /// spec's direction convention); diodes answer only their front
    /// block. Strongly-powered conductors are not modeled.
    fn comparator_signal_toward(&self, x: i32, y: i32, z: i32, read_dx: i32, read_dz: i32) -> i32 {
        let Some((name, props)) = self.get_block(x, y, z) else {
            return 0;
        };
        match name.as_str() {
            // The wire's power level.
            "minecraft:redstone_wire" => BlockRegistry::prop_int(&props, "power").unwrap_or(0),
            "minecraft:lever" if props.contains("powered=true") => 15,
            "minecraft:redstone_torch" | "minecraft:redstone_wall_torch"
                if !props.contains("lit=false") =>
            {
                15
            }
            "minecraft:redstone_block" => 15,
            // Into the front block: repeaters emit
            // the fixed 15, comparators the stored analog output
            // (the stored output
            // here the `comparator_outputs` map).
            "minecraft:repeater"
                if props.contains("powered=true") && reads_front(read_dx, read_dz, &props) =>
            {
                15
            }
            "minecraft:comparator"
                if props.contains("powered=true") && reads_front(read_dx, read_dz, &props) =>
            {
                self.comparator_outputs
                    .get(&(x, y, z))
                    .copied()
                    .unwrap_or(0)
            }
            _ => 0,
        }
    }

    /// The control-input read for
    /// the side reads: redstone block 15, wire POWER, lever 15 when on
    /// (attachment direction not modeled, matching the rest of the
    /// engine), diodes only into their front block. Torches read 0:
    /// A standing torch's direct signal is UP-only, so a horizontal
    /// control read sees nothing.
    fn control_signal(&self, x: i32, y: i32, z: i32, read_dx: i32, read_dz: i32) -> i32 {
        let Some((name, props)) = self.get_block(x, y, z) else {
            return 0;
        };
        match name.as_str() {
            "minecraft:redstone_block" => 15,
            "minecraft:lever" if props.contains("powered=true") => 15,
            "minecraft:redstone_wire" => BlockRegistry::prop_int(&props, "power").unwrap_or(0),
            "minecraft:repeater"
                if props.contains("powered=true") && reads_front(read_dx, read_dz, &props) =>
            {
                15
            }
            "minecraft:comparator"
                if props.contains("powered=true") && reads_front(read_dx, read_dz, &props) =>
            {
                self.comparator_outputs
                    .get(&(x, y, z))
                    .copied()
                    .unwrap_or(0)
            }
            _ => 0,
        }
    }

    /// The diode base read of the
    /// FACING-side block, REPLACED (not maxed) by the analog value of an
    /// analog-capable target. The section 1c look-through branch (solid
    /// conductor + item frame / 2-away analog) needs conductor and entity
    /// models and is deferred.
    fn comparator_input(&self, x: i32, y: i32, z: i32, facing: &str) -> i32 {
        let (dx, dz) = facing_offset(facing);
        if (dx, dz) == (0, 0) {
            return 0;
        }
        let (tx, tz) = (x + dx, z + dz);
        // --- containers hook (containers.rs): container fill sources ---
        if let Some(signal) = self.container_analog_output(tx, y, tz) {
            return signal;
        }
        let Some((name, props)) = self.get_block(tx, y, tz) else {
            return 0;
        };
        if let Some(analog) = analog_output(&name, &props) {
            return analog;
        }
        // Signal at the back block, maxed
        // with the wire POWER when below 15 — for wire targets both reads
        // agree, so signal_toward covers it.
        self.comparator_signal_toward(tx, y, tz, dx, dz)
    }

    /// Max of the two perpendicular side
    /// reads (clockwise and counter-clockwise of FACING).
    fn comparator_side_input(&self, x: i32, y: i32, z: i32, facing: &str) -> i32 {
        let (dx, dz) = facing_offset(facing);
        if (dx, dz) == (0, 0) {
            return 0;
        }
        // Horizontal clockwise rotation: north -> east -> south -> west.
        let (cw_dx, cw_dz) = (-dz, dx);
        let (ccw_dx, ccw_dz) = (dz, -dx);
        self.control_signal(x + cw_dx, y, z + cw_dz, cw_dx, cw_dz)
            .max(self.control_signal(x + ccw_dx, y, z + ccw_dz, ccw_dx, ccw_dz))
    }

    /// Schedule the 2gt refresh
    /// when the output value or the POWERED state is stale. Comparators
    /// are never locked, so there is no isLocked guard.
    pub(super) fn update_comparator(&mut self, x: i32, y: i32, z: i32, props: &str) {
        let facing = prop_dir(props);
        let compare = mode_is_compare(props);
        let input = self.comparator_input(x, y, z, facing);
        let side = self.comparator_side_input(x, y, z, facing);
        let output = output_signal(input, side, compare);
        let stored = self
            .comparator_outputs
            .get(&(x, y, z))
            .copied()
            .unwrap_or(0);
        let stale = output != stored
            || props.contains("powered=true") != should_turn_on(input, side, compare);
        if stale && !self.pending.contains(&((x, y, z), PendingKind::Comparator)) {
            self.pending.insert(((x, y, z), PendingKind::Comparator));
            // 2gt out; TickPriority HIGH vs NORMAL (shouldPrioritize) is
            // not modeled — the scheduled queue is FIFO within a tick.
            self.scheduled
                .push((self.tick + 2, (x, y, z), TickAction::ComparatorToggle));
        }
    }

    /// Store the new
    /// output value, flip POWERED per the live decision, and notify the
    /// front block. Recomputed from current world state at fire time — a
    /// sub-2gt input pulse that is already gone simply leaves the
    /// comparator off (no pulse stretching, no follow-up tick).
    pub(super) fn comparator_toggle(&mut self, x: i32, y: i32, z: i32) {
        self.pending.remove(&((x, y, z), PendingKind::Comparator));
        let Some((name, props)) = self.get_block(x, y, z) else {
            return;
        };
        if name != "minecraft:comparator" {
            return;
        }
        let facing = prop_dir(&props);
        let compare = mode_is_compare(&props);
        let input = self.comparator_input(x, y, z, facing);
        let side = self.comparator_side_input(x, y, z, facing);
        let output = output_signal(input, side, compare);
        // The stored value updates unconditionally (setOutputSignal); the
        // POWERED flip + front notification run when it changed, or
        // always in COMPARE mode.
        let old = self
            .comparator_outputs
            .insert((x, y, z), output)
            .unwrap_or(0);
        if old == output && !compare {
            return;
        }
        let on = should_turn_on(input, side, compare);
        if props.contains("powered=true") != on {
            let new_props =
                BlockRegistry::with_prop(&props, "powered", if on { "true" } else { "false" });
            let spec = format!("minecraft:comparator[{new_props}]");
            if let Some(state) = self.resolve_state(&spec) {
                // set_block's notify already covers updateNeighborsInFront
                // (it updates all six neighbors plus this position).
                self.set_block(x, y, z, state, true);
                return;
            }
        }
        // Front re-check without a state write: the stored value moved
        // while POWERED held, and downstream analog readers must notice.
        let (dx, dz) = facing_offset(facing);
        self.scheduled
            .push((self.tick, (x - dx, y, z - dz), TickAction::NeighborUpdate));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_formulas() {
        // shouldTurnOn: never at zero input; strict compare turns ties on
        // only in COMPARE mode.
        assert!(!should_turn_on(0, 0, true));
        assert!(!should_turn_on(0, 15, false));
        assert!(should_turn_on(1, 0, false));
        assert!(should_turn_on(15, 14, false));
        assert!(should_turn_on(5, 5, true));
        assert!(!should_turn_on(5, 5, false));
        assert!(!should_turn_on(3, 9, true));
        // Output: zero input -> 0; side above input -> 0; SUBTRACT -> the
        // difference; COMPARE -> the input.
        assert_eq!(output_signal(0, 0, true), 0);
        assert_eq!(output_signal(3, 9, false), 0);
        assert_eq!(output_signal(15, 6, false), 9);
        assert_eq!(output_signal(15, 15, false), 0);
        assert_eq!(output_signal(15, 6, true), 15);
        assert_eq!(output_signal(15, 15, true), 15);
    }

    #[test]
    fn analog_sources() {
        assert_eq!(analog_output("minecraft:cake", "bites=0"), Some(14));
        assert_eq!(analog_output("minecraft:cake", "bites=6"), Some(2));
        assert_eq!(
            analog_output("minecraft:candle_cake", "lit=false"),
            Some(14)
        );
        assert_eq!(analog_output("minecraft:cauldron", ""), Some(0));
        assert_eq!(
            analog_output("minecraft:water_cauldron", "level=3"),
            Some(3)
        );
        assert_eq!(analog_output("minecraft:lava_cauldron", ""), Some(3));
        assert_eq!(analog_output("minecraft:composter", "level=8"), Some(8));
        // Respawn anchor: floor(charge / 4 * 15) -> 0, 3, 7, 11, 15.
        assert_eq!(
            analog_output("minecraft:respawn_anchor", "charges=0"),
            Some(0)
        );
        assert_eq!(
            analog_output("minecraft:respawn_anchor", "charges=1"),
            Some(3)
        );
        assert_eq!(
            analog_output("minecraft:respawn_anchor", "charges=2"),
            Some(7)
        );
        assert_eq!(
            analog_output("minecraft:respawn_anchor", "charges=3"),
            Some(11)
        );
        assert_eq!(
            analog_output("minecraft:respawn_anchor", "charges=4"),
            Some(15)
        );
        assert_eq!(
            analog_output("minecraft:beehive", "facing=north,honey_level=5"),
            Some(5)
        );
        assert_eq!(
            analog_output("minecraft:copper_bulb", "lit=true,powered=true"),
            Some(15)
        );
        assert_eq!(
            analog_output("minecraft:copper_bulb", "lit=false,powered=true"),
            Some(0)
        );
        assert_eq!(
            analog_output("minecraft:end_portal_frame", "eye=true,facing=north"),
            Some(15)
        );
        // Non-analog blocks (including deferred containers) read None so
        // the redstone side input survives.
        assert_eq!(analog_output("minecraft:stone", ""), None);
        assert_eq!(analog_output("minecraft:barrel", ""), None);
        assert_eq!(
            analog_output("minecraft:chest", "facing=north,type=single"),
            None
        );
        assert_eq!(analog_output("minecraft:redstone_wire", "power=15"), None);
    }

    #[test]
    fn facing_math() {
        // Diode chain: same-FACING comparators feed each other; the read
        // vector equals the source's FACING only from its front block.
        assert!(reads_front(-1, 0, "facing=west,mode=compare,powered=true"));
        assert!(!reads_front(1, 0, "facing=west,mode=compare,powered=true"));
        assert!(reads_front(
            0,
            1,
            "facing=south,mode=subtract,powered=false"
        ));
        // Clockwise of west is north: rotation keeps side reads off the
        // front/back axis.
        assert_eq!(facing_offset("west"), (-1, 0));
        assert_eq!(facing_offset("north"), (0, -1));
        assert_eq!(facing_offset("east"), (1, 0));
        assert_eq!(facing_offset("south"), (0, 1));
        assert_eq!(facing_offset("sideways"), (0, 0));
    }
}
