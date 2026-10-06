//! The inbound event enum: one variant per serverbound event the
//! connection threads forward to the game thread.

use crate::game::{ConnId, GameMode, Outbound};
use std::sync::mpsc::Sender;

/// One `setblock` axis: an absolute block coordinate or a `~` offset
/// against the sender's feet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SetblockAxis {
    Abs(i32),
    Rel(f64),
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
    /// `setblock` with relative axes: a relative axis resolves against
    /// the sender's feet with a double offset, a bare `~` carries 0.
    SetblockRel {
        conn: ConnId,
        x: SetblockAxis,
        y: SetblockAxis,
        z: SetblockAxis,
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
    /// `gamemode <mode>`: switches the commanding player's mode.
    GameMode {
        conn: ConnId,
        mode: GameMode,
    },
    /// set_creative_mode_slot: a creative client pushing its picked stack.
    CreativeSlot {
        conn: ConnId,
        set: crate::inventory::CreativeSlotSet,
    },
    /// player_abilities: the client's flight toggle.
    PlayerAbilities {
        conn: ConnId,
        flying: bool,
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
    /// `summon <kind> [x y z]`: the deterministic spawn driver. Without
    /// coordinates the mob spawns at the sender's feet.
    Summon {
        conn: ConnId,
        kind: String,
        x: Option<f64>,
        y: Option<f64>,
        z: Option<f64>,
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
    // --- death hooks (death.rs) ---
    /// `client_command`: action 0 is PERFORM_RESPAWN (death.rs).
    ClientCommand {
        conn: ConnId,
        action: i32,
    },
    /// `kill` / `kill @s`: the sender kills themselves.
    Kill {
        conn: ConnId,
    },
    /// `gamerule keepInventory <bool>`: the death-drop rule.
    KeepInventory {
        conn: ConnId,
        enabled: bool,
    },
}
