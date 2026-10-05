//! Doppel server core: status (server-list ping) parity, the login ->
//! configuration -> play choreography, replaying registry/join/chunk blobs
//! captured from the vanilla oracle.

pub mod blobs;
pub mod dig;
pub mod game;
pub mod inventory;
pub mod living;
pub mod pathing;
pub mod placement;
pub mod spawning;
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
    // first move or tp.
    let mut join_pos = (0.0f64, 0.0f64, 0.0f64);
    if let Some(b) = blobs {
        for (id, body) in &b.play {
            if *id == 0x49 {
                let mut r = Reader::new(body);
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
            let body = if *id == 0x2e {
                let chunk =
                    doppel_world::WireChunk::decode(body).context("decoding replayed chunk")?;
                build_chunk(world, &chunk)?.encode()
            } else {
                body.clone()
            };
            conn.write_packet(*id, &body)?;
        }
    }

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
            let state = WorldState {
                dir: doppel_world::WorldDir::open(std::path::Path::new(&dir))?,
                boot: Default::default(),
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
