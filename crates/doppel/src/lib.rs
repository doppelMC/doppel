//! Doppel server core: status (server-list ping) parity, the login ->
//! configuration -> play choreography, replaying registry/join/chunk blobs
//! captured from the vanilla oracle.

pub mod blobs;
pub mod creeper;
pub mod dig;
pub mod events;
pub mod explosion;
pub mod game;
pub mod inventory;
pub mod living;
pub mod pathing;
pub mod persistence;
pub mod placement;
pub mod projectile;
pub mod skeleton;
pub mod spawning;
pub mod spider;
pub mod wire;
pub mod zombie;

#[cfg(test)]
mod piston_tests;
#[cfg(test)]
mod self_trace;

use anyhow::{bail, Context, Result};
use blobs::Blobs;
use doppel_protocol::{frame_packet, read_packet, write_string, write_varint, Conn, Pin, Reader};
use doppel_world::WireChunk;
use serde_json::{json, Value};
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

const MAX_FRAME: usize = 1024 * 1024;
const DEFAULT_MAX_PLAYERS: i64 = 20;
const COMPRESSION_THRESHOLD: i32 = 256;

/// Shared world state: Anvil regions plus the learned palette maps.
pub struct WorldState {
    pub dir: doppel_world::WorldDir,
    pub boot: doppel_world::anvil_to_wire::PaletteBootstrap,
    /// The world root every persistence write targets.
    pub root: std::path::PathBuf,
    /// Loaded level meta; defaults when no level.dat exists.
    pub level: doppel_world::level::LevelMeta,
    /// True when level.dat exists but does not parse; saves leave the
    /// file alone instead of overwriting it with defaults.
    pub level_readonly: bool,
}

type SharedWorld = Arc<std::sync::Mutex<WorldState>>;

/// Rebuilds a chunk from Anvil storage when the world has it; the wire
/// capture bootstraps palette maps and supplies light. Falls back to the
/// capture itself for chunks missing on disk.
fn build_chunk(world: Option<&SharedWorld>, reference: &WireChunk) -> Result<WireChunk> {
    let Some(world) = world else {
        return Ok(reference.clone());
    };
    let mut w = world.lock().unwrap_or_else(|e| e.into_inner());
    match w.dir.chunk(reference.x, reference.z)? {
        Some(anvil) => {
            w.boot.learn(reference, &anvil);
            doppel_world::anvil_to_wire::convert(&anvil, reference, &w.boot)
        }
        None => Ok(reference.clone()),
    }
}

/// Replays the captured join burst, rebuilding chunks from storage and
/// rewriting the position and inventory packets with saved player state.
/// Returns the absolute position the client ends up standing at.
fn replay_join_burst(
    conn: &mut Conn<TcpStream>,
    blobs: Option<&Blobs>,
    world: Option<&SharedWorld>,
    name: &str,
) -> Result<(f64, f64, f64)> {
    let saved = world.and_then(|w| {
        let w = w.lock().unwrap_or_else(|e| e.into_inner());
        doppel_world::playerdata::load(&w.root, &blobs::offline_uuid(name))
            .ok()
            .flatten()
    });
    let saved_container: Vec<Option<inventory::ItemStack>> = saved
        .as_ref()
        .map(|data| saved_container_slots(&data.inventory))
        .unwrap_or_default();
    let mut join_pos = (0.0f64, 0.0f64, 0.0f64);
    if let Some(b) = blobs {
        for (id, body) in &b.play {
            let body = if *id == 0x2e {
                let chunk =
                    doppel_world::WireChunk::decode(body).context("decoding replayed chunk")?;
                build_chunk(world, &chunk)?.encode()
            } else if *id == 0x49 {
                match &saved {
                    Some(data) => rewrite_position(body, data.pos, data.yaw, data.pitch)
                        .unwrap_or_else(|| body.clone()),
                    None => body.clone(),
                }
            } else if *id == inventory::PACKET_CONTAINER_SET_CONTENT && saved.is_some() {
                rewrite_set_content(body, &saved_container).unwrap_or_else(|| body.clone())
            } else {
                body.clone()
            };
            if *id == 0x49 {
                let mut r = Reader::new(&body);
                if r.read_varint().is_ok() {
                    if let (Ok(x), Ok(y), Ok(z)) = (r.read_f64(), r.read_f64(), r.read_f64()) {
                        let relative = r.read_f64().is_err()
                            || r.read_f64().is_err()
                            || r.read_f64().is_err()
                            || r.read_f32().is_err()
                            || r.read_f32().is_err()
                            || !matches!(r.read_varint(), Ok(0));
                        if !relative {
                            join_pos = (x, y, z);
                        }
                    }
                }
            }
            conn.write_packet(*id, &body)?;
        }
    }
    Ok(join_pos)
}

/// Saved inventory slots in container-slot order; out-of-range slots are
/// dropped.
fn saved_container_slots(
    saved: &[doppel_world::playerdata::SavedSlot],
) -> Vec<Option<inventory::ItemStack>> {
    let mut slots = vec![None; inventory::TOTAL_SLOTS];
    for slot in saved {
        if slot.slot >= 0 && (slot.slot as usize) < inventory::TOTAL_SLOTS {
            slots[slot.slot as usize] = persistence::saved_to_stack(slot);
        }
    }
    slots
}

/// Rebuilds a player_position (0x49) body with a saved pose, preserving
/// the teleport id, deltas, and flags. None when the body does not parse
/// as the absolute form.
fn rewrite_position(body: &[u8], pos: [f64; 3], yaw: f32, pitch: f32) -> Option<Vec<u8>> {
    let mut r = Reader::new(body);
    let teleport_id = r.read_varint().ok()?;
    // The original position precedes the deltas.
    for _ in 0..3 {
        r.read_f64().ok()?;
    }
    let mut deltas = [0.0f64; 3];
    for d in &mut deltas {
        *d = r.read_f64().ok()?;
    }
    r.read_f32().ok()?;
    r.read_f32().ok()?;
    let flags = r.read_varint().ok()?;
    if flags != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(body.len());
    write_varint(&mut out, teleport_id);
    for v in pos {
        out.extend_from_slice(&v.to_be_bytes());
    }
    for d in deltas {
        out.extend_from_slice(&d.to_be_bytes());
    }
    out.extend_from_slice(&yaw.to_be_bytes());
    out.extend_from_slice(&pitch.to_be_bytes());
    write_varint(&mut out, flags);
    Some(out)
}

/// Rebuilds a container_set_content body for the player inventory menu
/// with the saved slots, preserving the container and state ids. None
/// when the body does not decode.
fn rewrite_set_content(
    body: &[u8],
    container_slots: &[Option<inventory::ItemStack>],
) -> Option<Vec<u8>> {
    let (container_id, state_id, ..) = inventory::decode_container_set_content(body).ok()?;
    if container_id != 0 {
        return None;
    }
    let mut menu = Vec::with_capacity(inventory::INVENTORY_MENU_SIZE);
    for slot in 0..inventory::INVENTORY_MENU_SIZE {
        let stack = inventory::menu_to_container(slot)
            .and_then(|c| container_slots.get(c).cloned().flatten());
        menu.push(stack);
    }
    Some(inventory::encode_container_set_content(
        container_id,
        state_id,
        &menu,
        None,
    ))
}

// ---------------------------------------------------------------------------
// Status (M0)
// ---------------------------------------------------------------------------

/// The status response body, mirroring vanilla's shape.
pub fn status_response(pin: &Pin, client_protocol: i32) -> Value {
    json!({
        "version": {
            "name": pin.version_name.clone().unwrap_or_else(|| pin.id.clone()),
            "protocol": pin.protocol.unwrap_or(client_protocol),
        },
        "players": {
            "max": DEFAULT_MAX_PLAYERS,
            "online": 0,
            "sample": [],
        },
        "description": "A Minecraft Server",
    })
}

fn encode_status_response(value: &Value) -> Vec<u8> {
    let mut payload = Vec::new();
    write_varint(&mut payload, 0x00);
    write_string(&mut payload, &value.to_string());
    frame_packet(&payload)
}

fn encode_pong(payload: i64) -> Vec<u8> {
    let mut p = Vec::new();
    write_varint(&mut p, 0x01);
    p.extend_from_slice(&payload.to_be_bytes());
    frame_packet(&p)
}

fn handle_status(stream: &mut TcpStream, pin: &Pin, client_protocol: i32) -> Result<()> {
    let (id, _body) = read_packet(stream, MAX_FRAME).context("status request")?;
    if id != 0x00 {
        bail!("expected status request (id 0), got id {id}");
    }
    let resp = encode_status_response(&status_response(pin, client_protocol));
    stream.write_all(&resp).context("writing status response")?;

    if let Ok((id, body)) = read_packet(stream, MAX_FRAME) {
        if id == 0x01 {
            let payload = Reader::new(&body).read_i64().context("ping payload")?;
            stream
                .write_all(&encode_pong(payload))
                .context("writing pong")?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Login -> configuration -> play
// ---------------------------------------------------------------------------

fn handle_login(
    stream: TcpStream,
    pin: &Pin,
    blobs: Option<&Blobs>,
    world: Option<&SharedWorld>,
    game_tx: &std::sync::mpsc::Sender<game::Inbound>,
) -> Result<()> {
    let mut conn = Conn::new(stream);

    // --- login state: hello (String name + bare UUID) ---
    let (id, body) = conn.read_packet().context("hello")?;
    if id != 0x00 {
        bail!("expected hello (id 0), got id {id}");
    }
    let mut r = Reader::new(&body);
    let name = r.read_string(16).context("hello username")?;
    let _client_uuid = r.read_bytes(16).context("hello profile uuid")?;
    let profile_uuid = blobs::offline_uuid(&name);

    // compression on, then login_finished (uuid, name, properties, sessionId)
    let mut threshold = Vec::new();
    write_varint(&mut threshold, COMPRESSION_THRESHOLD);
    conn.write_packet(0x03, &threshold)?; // login_compression
    conn.set_compression(COMPRESSION_THRESHOLD);

    let mut finished = Vec::new();
    finished.extend_from_slice(&profile_uuid);
    write_string(&mut finished, &name);
    write_varint(&mut finished, 0); // properties: none
    finished.extend_from_slice(&blobs::session_uuid());
    conn.write_packet(0x02, &finished)?; // login_finished

    // client confirms the state transition (login-state 0x03, empty)
    let (id, body) = conn.read_packet().context("login acknowledged")?;
    if id != 0x03 || !body.is_empty() {
        bail!(
            "expected login acknowledged (empty 0x03), got id {id} len {}",
            body.len()
        );
    }

    // --- configuration state ---
    let mut brand = Vec::new();
    write_string(&mut brand, "minecraft:brand");
    write_string(&mut brand, "vanilla");
    conn.write_packet(0x01, &brand)?;

    let mut features = Vec::new();
    write_varint(&mut features, 1);
    write_string(&mut features, "minecraft:vanilla");
    conn.write_packet(0x0d, &features)?; // update_enabled_features (26.3)

    let mut packs = Vec::new();
    write_varint(&mut packs, 1);
    write_string(&mut packs, "minecraft");
    write_string(&mut packs, "core");
    write_string(&mut packs, pin.version_name.as_deref().unwrap_or(&pin.id));
    conn.write_packet(0x0f, &packs)?; // select_known_packs (26.3)

    // The client may volunteer Client Information (0x00) and keep-alives at
    // any point; vanilla stores them without blocking. Wait for the actual
    // known-packs reply (0x07), skipping anything else.
    let (_id, _body) = loop {
        let (id, body) = conn.read_packet().context("known packs reply")?;
        if id == 0x07 {
            break (id, body);
        }
    };

    if let Some(b) = blobs {
        for registry in &b.registries {
            conn.write_packet(0x07, registry)?;
        }
        if let Some(tags) = &b.update_tags {
            conn.write_packet(0x0e, tags)?;
        }
    }

    conn.write_packet(0x03, &[])?; // finish_configuration: server first
    loop {
        let (id, body) = conn.read_packet().context("finish configuration reply")?;
        if id == 0x03 && body.is_empty() {
            break;
        }
    }

    // --- play state: replay the full captured join sequence in order ---
    // Chunk packets round-trip through the codec, and when a world
    // directory is configured, chunks are REBUILT from Anvil storage (the
    // capture only bootstraps the name->id palette maps and supplies light
    // data). Parity proves the Anvil-built bytes identical to vanilla's.
    //
    // The burst's player_position (0x49) carries the spawn coordinates
    // the reference stands every joining player at; tracking them keeps
    // position-dependent rules (reach, overlay radius) honest before the
    // first move or tp. Saved player state rewrites the burst's own
    // position and inventory packets, so the client joins directly on
    // the restored state instead of learning it afterwards.
    let join_pos = replay_join_burst(&mut conn, blobs, world, &name)?;

    // Steady state: the connection becomes an IO actor. The reader loop
    // forwards Inbound events to the game thread; a writer thread drains
    // Outbound frames onto a cloned socket. All world/streaming/keep-alive
    // decisions live in the game thread.
    let (tx_out, rx_out) = std::sync::mpsc::channel::<game::Outbound>();
    let conn_id = game::next_conn_id();
    let threshold = conn.compression_threshold();
    let write_stream = conn
        .get_ref()
        .try_clone()
        .context("cloning socket for writer")?;
    let writer = std::thread::spawn(move || {
        use std::io::Write as _;
        let mut stream = write_stream;
        for out in rx_out {
            match out {
                game::Outbound::Frame { id, body } => {
                    let frame = doppel_protocol::encode_frame(threshold, id, &body);
                    if stream.write_all(&frame).is_err() {
                        break;
                    }
                }
                game::Outbound::Disconnect => {
                    let _ = stream.shutdown(std::net::Shutdown::Both);
                    break;
                }
            }
        }
    });

    // The replayed join burst already delivered these chunks.
    let mut sent = Vec::new();
    if let Some(b) = blobs {
        for (pid, body) in &b.play {
            if *pid == 0x2e {
                if let Ok(chunk) = WireChunk::decode(body) {
                    sent.push((chunk.x, chunk.z));
                }
            }
        }
    }
    let _ = game_tx.send(game::Inbound::Joined {
        conn: conn_id,
        name: name.clone(),
        x: join_pos.0,
        y: join_pos.1,
        z: join_pos.2,
        sent,
        tx: tx_out,
    });

    // Reader: blocking reads, forwarding to the game thread. Reads continue
    // until the writer disconnects us (socket shutdown breaks read too).
    while let Ok((id, body)) = conn.read_packet() {
        let event = play_event(conn_id, id, &body);
        if let Some(event) = event {
            if game_tx.send(event).is_err() {
                break;
            }
        }
    }
    let _ = game_tx.send(game::Inbound::Left { conn: conn_id });
    let _ = writer.join();
    Ok(())
}

/// Translates one serverbound play packet into a game-thread event.
/// 26.x serverbound ids are stable across the 26.2→26.3 clientbound
/// shifts. Unknown packets are ignored, matching vanilla's tolerance for
/// forward-compat channels.
/// The chat-command arm of `play_event`: the harness drivers Doppel
/// exactly like vanilla, so every command it observes must translate.
fn chat_command_event(conn: game::ConnId, mut r: Reader) -> Option<game::Inbound> {
    // chat_command (unsigned, no leading slash). Minimal /tp so the
    // walk-parity bot can drive Doppel exactly like vanilla.
    let cmd = r.read_string(1024).ok()?;
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.len() == 5 && parts[0] == "tp" {
        if let (Ok(x), Ok(y), Ok(z)) = (
            parts[2].parse::<f64>(),
            parts[3].parse::<f64>(),
            parts[4].parse::<f64>(),
        ) {
            if parts[1] == "@s" {
                return Some(game::Inbound::Tp { conn, x, y, z });
            }
            return Some(game::Inbound::TpNamed {
                conn,
                name: parts[1].to_string(),
                x,
                y,
                z,
            });
        }
    }
    if parts.len() == 5 && parts[0] == "setblock" {
        if let (Ok(x), Ok(y), Ok(z)) = (
            parts[1].parse::<i32>(),
            parts[2].parse::<i32>(),
            parts[3].parse::<i32>(),
        ) {
            return Some(game::Inbound::Setblock {
                conn,
                x,
                y,
                z,
                name: parts[4].to_string(),
            });
        }
    }
    // `tick step N` is the differential harness's sequencing
    // barrier: later commands must land on later ticks.
    if parts.len() == 3 && parts[0] == "tick" && parts[1] == "step" {
        if let Ok(steps) = parts[2].parse::<u32>() {
            return Some(game::Inbound::TickStep { conn, steps });
        }
    }
    // `tick freeze` / `tick unfreeze` gate the wall-clock loop;
    // stepped ticks still advance a frozen clock.
    if parts.len() == 2 && parts[0] == "tick" {
        match parts[1] {
            "freeze" => {
                return Some(game::Inbound::TickFreeze { conn, frozen: true });
            }
            "unfreeze" => {
                return Some(game::Inbound::TickFreeze {
                    conn,
                    frozen: false,
                });
            }
            _ => {}
        }
    }
    // `time set <ticks>`: the game thread stores the day time
    // and answers with a set_time broadcast plus the pacing
    // reply.
    if parts.len() == 3 && parts[0] == "time" && parts[1] == "set" {
        if let Ok(value) = parts[2].parse::<i64>() {
            return Some(game::Inbound::TimeSet { conn, value });
        }
    }
    // `gamemode <mode>` for the commanding player switches the
    // mode the inventory, placement, and dig paths read.
    if parts.len() == 2 && parts[0] == "gamemode" {
        if let Some(mode) = game::GameMode::parse(parts[1]) {
            return Some(game::Inbound::GameMode { conn, mode });
        }
    }
    // `give @s <item> [count]`: the inventory test driver. The
    // item name resolves (or fails) game-side against the learned
    // id table, like setblock.
    if (parts.len() == 3 || parts.len() == 4) && parts[0] == "give" && parts[1] == "@s" {
        let count = if parts.len() == 4 {
            parts[3].parse::<i32>().ok()?
        } else {
            1
        };
        return Some(game::Inbound::Give {
            conn,
            item: parts[2].to_string(),
            count,
        });
    }
    // --- survival hooks (entities.rs) ---
    // `gamerule random_tick_speed N`: the random tick rate the
    // survival gate amplifies its decay window with (the rule is
    // snake_case on the wire).
    if parts.len() == 3 && parts[0] == "gamerule" && parts[1] == "random_tick_speed" {
        if let Ok(speed) = parts[2].parse::<usize>() {
            return Some(game::Inbound::GameRule {
                conn,
                tick_speed: speed,
            });
        }
    }
    // --- mob hooks (living.rs / spawning.rs) ---
    if let Some(inbound) = mob_command(conn, &parts) {
        return Some(inbound);
    }
    // The other spawner gamerules silence monster families this
    // build does not spawn; the reply must still come.
    if parts.len() == 3
        && parts[0] == "gamerule"
        && parts[1].starts_with("spawn_")
        && parts[2].parse::<bool>().is_ok()
    {
        return Some(game::Inbound::GameRuleNoop { conn });
    }
    // --- containers hooks (containers.rs) ---
    // `opencontainer x y z`: the container test driver.
    if parts.len() == 5 && parts[0] == "opencontainer" {
        if let (Ok(x), Ok(y), Ok(z)) = (
            parts[1].parse::<i32>(),
            parts[2].parse::<i32>(),
            parts[3].parse::<i32>(),
        ) {
            return Some(game::Inbound::OpenContainer { conn, x, y, z });
        }
    }
    None
}

fn play_event(conn: game::ConnId, id: i32, body: &[u8]) -> Option<game::Inbound> {
    let mut r = Reader::new(body);
    match id {
        0x1c => {
            let answer = r.read_i64().ok()?;
            Some(game::Inbound::KeepAliveAnswer { conn, id: answer })
        }
        // move_player_pos: x y z f64 + flags u8 (flags carry no state).
        0x1e => {
            let x = r.read_f64().ok()?;
            let y = r.read_f64().ok()?;
            let z = r.read_f64().ok()?;
            Some(game::Inbound::Moved {
                conn,
                x,
                y,
                z,
                yaw: None,
                pitch: None,
            })
        }
        // move_player_pos_rot: x y z f64, yaw pitch f32, flags u8.
        0x1f => {
            let x = r.read_f64().ok()?;
            let y = r.read_f64().ok()?;
            let z = r.read_f64().ok()?;
            let yaw = r.read_f32().ok()?;
            let pitch = r.read_f32().ok()?;
            Some(game::Inbound::Moved {
                conn,
                x,
                y,
                z,
                yaw: Some(yaw),
                pitch: Some(pitch),
            })
        }
        // move_player_rot: yaw pitch f32 + flags u8 — the view direction
        // placement geometry reads.
        0x20 => {
            let yaw = r.read_f32().ok()?;
            let pitch = r.read_f32().ok()?;
            Some(game::Inbound::Rotated { conn, yaw, pitch })
        }
        // chat_command (unsigned, no leading slash).
        0x07 => chat_command_event(conn, r),
        // --- inventory hooks (inventory.rs) ---
        // set_carried_item: hotbar select, one i16 slot. The id is the
        // 26.3 registration order (26.2's 0x35 + the inserted-punch shift)
        // — wire-verify TODO noted in inventory.rs.
        0x36 => {
            let slot = inventory::parse_set_carried_item(body).ok()?;
            Some(game::Inbound::SetCarriedItem { conn, slot })
        }
        // set_creative_mode_slot: creative clients push their picked
        // stacks.
        0x39 => {
            let set = inventory::parse_set_creative_slot(body).ok()?;
            Some(game::Inbound::CreativeSlot { conn, set })
        }
        // player_abilities: the client toggling flight.
        0x28 => {
            let flying = inventory::parse_player_abilities(body).ok()?;
            Some(game::Inbound::PlayerAbilities { conn, flying })
        }
        // container_click: clicks against an open menu (HashedStack
        // predictions decoded but not applied — see inventory.rs).
        0x12 => {
            let click = inventory::parse_container_click(body).ok()?;
            Some(game::Inbound::ContainerClick { conn, click })
        }
        // --- containers hooks (containers.rs) ---
        // container_close: the client closed a menu (one VarInt id).
        0x13 => {
            let id = game::containers::parse_container_close(body).ok()?;
            Some(game::Inbound::ContainerClose {
                conn,
                container_id: id,
            })
        }
        // --- placement hooks (placement.rs) ---
        // use_item_on: right-click a block face.
        0x42 => {
            let hit = match placement::parse_use_item_on(body) {
                Ok(hit) => hit,
                Err(e) => {
                    eprintln!("[doppel] use_item_on: {e:#}");
                    return None;
                }
            };
            Some(game::Inbound::UseItemOn {
                conn,
                x: hit.x,
                y: hit.y,
                z: hit.z,
                face: hit.face,
                cursor_x: hit.cursor_x,
                cursor_y: hit.cursor_y,
                cursor_z: hit.cursor_z,
                hand: hit.hand,
                sequence: hit.sequence,
            })
        }
        // --- breaking hooks (dig.rs) ---
        // player_action: dig lifecycle, drops, offhand swap.
        0x29 => {
            let act = match dig::parse_player_action(body) {
                Ok(act) => act,
                Err(e) => {
                    eprintln!("[doppel] player_action: {e:#}");
                    return None;
                }
            };
            Some(game::Inbound::PlayerAction { conn, act })
        }
        // punch: the arm swing, empty body.
        0x2e => {
            if !body.is_empty() {
                eprintln!("[doppel] punch: {} trailing bytes", body.len());
                return None;
            }
            Some(game::Inbound::Punch { conn })
        }
        _ => None,
    }
}

/// The mob-hook command parses: the part of the command volley that
/// living.rs and spawning.rs answer. `spawn_mobs` leads the generic
/// `spawn_*` noop parse so the real gate wins the race.
fn mob_command(conn: game::ConnId, parts: &[&str]) -> Option<game::Inbound> {
    // `gamerule spawn_mobs <bool>`: the natural-spawn gate.
    if parts.len() == 3 && parts[0] == "gamerule" && parts[1] == "spawn_mobs" {
        let enabled = parts[2].parse::<bool>().ok()?;
        return Some(game::Inbound::SpawnMobs { conn, enabled });
    }
    // `time set <word|ticks>`: the day clock the darkness and burn
    // checks read.
    if parts.len() == 3 && parts[0] == "time" && parts[1] == "set" {
        let value = match parts[2] {
            "day" => 1000,
            "noon" => 6000,
            "night" => 13000,
            "midnight" => 18000,
            word => word.parse::<i64>().ok()? % 24000,
        };
        return Some(game::Inbound::TimeSet { conn, value });
    }
    // `summon <kind> [x y z]`: the mobs gate's deterministic spawn
    // driver; the bare form spawns at the sender. Kind names carry no
    // namespace by default.
    if parts.len() == 2 && parts[0] == "summon" {
        return Some(game::Inbound::Summon {
            conn,
            kind: namespaced(parts[1]),
            x: None,
            y: None,
            z: None,
        });
    }
    if parts.len() == 5 && parts[0] == "summon" {
        if let (Ok(x), Ok(y), Ok(z)) = (
            parts[2].parse::<f64>(),
            parts[3].parse::<f64>(),
            parts[4].parse::<f64>(),
        ) {
            return Some(game::Inbound::Summon {
                conn,
                kind: namespaced(parts[1]),
                x: Some(x),
                y: Some(y),
                z: Some(z),
            });
        }
    }
    // `difficulty <word>`: peaceful removes monsters.
    if parts.len() == 2 && parts[0] == "difficulty" {
        let peaceful = match parts[1] {
            "peaceful" => true,
            "easy" | "normal" | "hard" => false,
            _ => return None,
        };
        return Some(game::Inbound::SetDifficulty { conn, peaceful });
    }
    None
}

/// Applies the default `minecraft:` namespace to a bare identifier.
fn namespaced(name: &str) -> String {
    if name.contains(':') {
        name.to_string()
    } else {
        format!("minecraft:{name}")
    }
}

// ---------------------------------------------------------------------------
// Connection dispatch
// ---------------------------------------------------------------------------

fn handle_conn(
    stream: TcpStream,
    pin: Pin,
    blobs: Option<Arc<Blobs>>,
    world: Option<SharedWorld>,
    game: GameChannels,
) {
    let peer = stream
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "unknown".into());
    if stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .is_err()
    {
        return;
    }
    let mut stream = stream;
    let result = (|| -> Result<()> {
        let (id, body) = read_packet(&mut stream, MAX_FRAME).context("handshake")?;
        if id != 0x00 {
            bail!("expected handshake (id 0), got id {id}");
        }
        let mut r = Reader::new(&body);
        let client_protocol = r.read_varint().context("protocol version")?;
        let _addr = r.read_string(1024).context("server address")?;
        let _port = r.read_u16().context("server port")?;
        let next_state = r.read_varint().context("next state")?;
        match next_state {
            1 => handle_status(&mut stream, &pin, client_protocol),
            2 => handle_login(stream, &pin, blobs.as_deref(), world.as_ref(), &game.tx),
            n => bail!("invalid next state {n}"),
        }
    })();
    match result {
        Ok(()) => {}
        Err(e) => eprintln!("[doppel] {peer}: {e:#}"),
    }
}

/// Accept loop. Binds nothing itself so tests can hand us an ephemeral port.
pub fn serve_on(
    listener: TcpListener,
    pin: Pin,
    blobs: Option<Arc<Blobs>>,
    world: Option<SharedWorld>,
) -> Result<()> {
    // The game thread: single owner of world/streaming/keep-alive state.
    // It owns the Game outright — no shared lock.
    let (tx, rx) = std::sync::mpsc::channel::<game::Inbound>();
    {
        let mut game = game::Game::new(rx, world.clone(), blobs.clone());
        std::thread::spawn(move || game.run());
    }
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let pin = pin.clone();
                let blobs = blobs.clone();
                let world = world.clone();
                let game = GameChannels { tx: tx.clone() };
                std::thread::spawn(move || handle_conn(stream, pin, blobs, world, game));
            }
            Err(e) => eprintln!("[doppel] accept error: {e}"),
        }
    }
    Ok(())
}

/// Per-connection handle into the game thread.
struct GameChannels {
    tx: std::sync::mpsc::Sender<game::Inbound>,
}

/// Convenience for the binary: load pin (+ blobs from DOPPEL_BLOBS if set),
/// bind, serve.
pub fn serve(addr: &str, pin_path: Option<&std::path::Path>) -> Result<()> {
    let pin = if let Some(path) = pin_path {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading pin {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?
    } else {
        doppel_protocol::load_pin()?
    };
    let blobs = match std::env::var("DOPPEL_BLOBS") {
        Ok(dir) => {
            let b = blobs::load(std::path::Path::new(&dir))?;
            println!(
                "[doppel] loaded blobs: {} registries, {} play packets",
                b.registries.len(),
                b.play.len()
            );
            Some(Arc::new(b))
        }
        Err(_) => {
            eprintln!(
                "[doppel] no DOPPEL_BLOBS set — login will complete config but send no world data"
            );
            None
        }
    };
    let world = match std::env::var("DOPPEL_WORLD") {
        Ok(dir) => {
            let root = std::path::PathBuf::from(&dir);
            // A fresh world dir has no region storage yet; the save path
            // creates it on first write, the read path needs it now.
            let _ = std::fs::create_dir_all(doppel_world::anvil_write::region_dir(&root));
            let (level, level_readonly) = match doppel_world::level::load(&root) {
                Ok(meta) => (meta.unwrap_or_default(), false),
                Err(e) => {
                    eprintln!("[doppel] level.dat unreadable, leaving it untouched: {e:#}");
                    (doppel_world::level::LevelMeta::default(), true)
                }
            };
            let state = WorldState {
                dir: doppel_world::WorldDir::open(&root)?,
                boot: Default::default(),
                root,
                level,
                level_readonly,
            };
            println!("[doppel] world storage: {dir}");
            Some(Arc::new(std::sync::Mutex::new(state)))
        }
        Err(_) => None,
    };
    let listener = TcpListener::bind(addr).with_context(|| format!("binding {addr}"))?;
    println!("[doppel] listening on {addr} (vanilla target: {})", pin.id);
    serve_on(listener, pin, blobs, world)
}

#[cfg(test)]
mod tests {
    use super::*;
    use doppel_protocol::write_varint as out_varint;

    fn position_body(teleport_id: i32, x: f64, y: f64, z: f64, yaw: f32, pitch: f32) -> Vec<u8> {
        let mut body = Vec::new();
        out_varint(&mut body, teleport_id);
        for v in [x, y, z, 0.0, 0.0, 0.0] {
            body.extend_from_slice(&v.to_be_bytes());
        }
        body.extend_from_slice(&yaw.to_be_bytes());
        body.extend_from_slice(&pitch.to_be_bytes());
        out_varint(&mut body, 0);
        body
    }

    #[test]
    fn rewrite_position_swaps_pose_and_keeps_shape() {
        let body = position_body(3, 1.0, 2.0, 3.0, 10.0, 20.0);
        let rewritten = rewrite_position(&body, [7.5, -60.0, 9.25], -90.0, 12.5).expect("rewrites");
        let mut r = Reader::new(&rewritten);
        assert_eq!(r.read_varint().unwrap(), 3, "teleport id kept");
        assert_eq!(
            (
                r.read_f64().unwrap(),
                r.read_f64().unwrap(),
                r.read_f64().unwrap()
            ),
            (7.5, -60.0, 9.25)
        );
        assert_eq!(
            (
                r.read_f64().unwrap(),
                r.read_f64().unwrap(),
                r.read_f64().unwrap()
            ),
            (0.0, 0.0, 0.0),
            "deltas kept"
        );
        assert_eq!(
            (r.read_f32().unwrap(), r.read_f32().unwrap()),
            (-90.0, 12.5)
        );
        assert_eq!(r.read_varint().unwrap(), 0, "flags kept");
        // A truncated body stays untouched.
        assert!(rewrite_position(&body[..8], [0.0; 3], 0.0, 0.0).is_none());
    }

    #[test]
    fn rewrite_set_content_carries_saved_slots() {
        let empty = vec![None; inventory::INVENTORY_MENU_SIZE];
        let body = inventory::encode_container_set_content(0, 1, &empty, None);
        let mut slots = vec![None; inventory::TOTAL_SLOTS];
        slots[0] = Some(inventory::ItemStack::new(1, 5));
        let rewritten = rewrite_set_content(&body, &slots).expect("rewrites");
        let (container_id, state_id, menu, carried) =
            inventory::decode_container_set_content(&rewritten).unwrap();
        assert_eq!((container_id, state_id), (0, 1));
        assert!(carried.is_none());
        // Hotbar container 0 presents as menu slot 36.
        assert_eq!(menu[36].as_ref().map(inventory::ItemStack::count), Some(5));
        assert!(menu.iter().enumerate().all(|(i, s)| i == 36 || s.is_none()));
        // A foreign container id stays untouched.
        let other = inventory::encode_container_set_content(2, 1, &empty, None);
        assert!(rewrite_set_content(&other, &slots).is_none());
    }

    #[test]
    fn saved_container_slots_drops_out_of_range() {
        let stack = |slot: i8| doppel_world::playerdata::SavedSlot {
            slot,
            id: "minecraft:stone".into(),
            count: 1,
            extra: Some(vec![0x01, 0x01, 0x00, 0x00]),
        };
        let slots = saved_container_slots(&[stack(0), stack(-106), stack(100)]);
        assert_eq!(slots.len(), inventory::TOTAL_SLOTS);
        assert!(slots[0].is_some(), "in-range slot restores");
        assert!(
            slots[1..].iter().all(Option::is_none),
            "foreign slot layouts drop instead of indexing wild"
        );
    }
}
