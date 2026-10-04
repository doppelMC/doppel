//! A minimal wire-protocol client used to observe servers (vanilla and
//! Doppel) from the outside, exactly like a real client would.

use anyhow::{bail, Context, Result};
use doppel_protocol::{
    encode_handshake, encode_ping, encode_status_request, read_packet, Conn, Reader,
};
use serde::Serialize;
use serde_json::Value;
use std::io::Write;
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

const PING_PAYLOAD: i64 = 0x0064_6F70_7065_6C01; // "doppel"
/// Quiet beat on chunk packets that closes a capture leg: batch ends
/// are not stream completion (batches adapt their size to feedback, so
/// the first is tiny), and fresh chunks generate slower than the beat.
/// Entity chatter keeps reads busy, so quiet is measured on chunks.
const STREAM_QUIET: Duration = Duration::from_secs(6);
/// The join stream stalls for several seconds after its first batches;
/// firing the first locate on hop-length quiet would cut it in half.
const SPAWN_QUIET: Duration = Duration::from_secs(8);
/// Altitude for hop teleports. The 3D biome search can return a cell
/// far below the surface (the surface biome fills the column), so hops
/// land above the located column instead: chunk streaming is
/// column-based, and the server holds the player there because
/// movement is client-driven and this client never moves.
const HOP_TP_Y: i32 = 300;
/// Chunks a hop stream must deliver before quiet can close it. Fresh
/// biome chunks generate slower than the quiet beat for tens of seconds
/// at a time (mangrove swamps stall mid-stream), so quiet alone would
/// cut the leg at the first stall; a stream that never reaches this
/// closes on HOP_DEADLINE instead.
const HOP_MIN_LEG_CHUNKS: usize = 40;
/// Read window for a /locate reply. The biome search scans up to 6400
/// blocks in 3D; keep-alives bridge the window, and HOP_DEADLINE
/// bounds the total wait when they do.
const LOCATE_TIMEOUT: Duration = Duration::from_secs(30);
/// Wall-clock bound for one hop phase (locate reply or post-teleport
/// chunk batch), whether or not other packets keep arriving.
const HOP_DEADLINE: Duration = Duration::from_secs(60);
/// A late hop's stream trickles: by the second teleport the generator
/// runs saturated, chunks arrive fractionally in small batches, and
/// quiet never spans the floor. The settle deadline lets the trickle
/// build a sample before closing the leg with whatever arrived.
const SETTLE_DEADLINE: Duration = Duration::from_secs(100);

/// Connects to a server and returns its parsed status (server-list-ping)
/// JSON. Verifies the ping/pong round-trip on the way out.
pub fn status_ping(host: &str, port: u16, protocol_hint: i32, timeout: Duration) -> Result<Value> {
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .ok()
        .or_else(|| (host, port).to_socket_addrs().ok()?.next())
        .context("resolving server address")?;

    let mut stream = TcpStream::connect_timeout(&addr, timeout)
        .with_context(|| format!("connecting to {addr}"))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;

    stream.write_all(&encode_handshake(
        protocol_hint,
        host,
        port,
        1, // status
    ))?;
    stream.write_all(&encode_status_request())?;

    let (id, body) = read_packet(&mut stream, 1024 * 1024).context("status response")?;
    if id != 0x00 {
        bail!("expected status response (id 0), got id {id}");
    }
    let mut r = Reader::new(&body);
    let json_text = r.read_string(1024 * 1024).context("status JSON")?;
    let value: Value = serde_json::from_str(&json_text).context("parsing status JSON")?;

    // Verify the ping/pong round-trip as well.
    stream.write_all(&encode_ping(PING_PAYLOAD))?;
    let (id, body) = read_packet(&mut stream, 1024).context("pong")?;
    if id != 0x01 {
        bail!("expected pong (id 1), got id {id}");
    }
    let echoed = Reader::new(&body).read_i64().context("pong payload")?;
    if echoed != PING_PAYLOAD {
        bail!("pong payload mismatch: sent {PING_PAYLOAD}, got {echoed}");
    }

    Ok(value)
}

/// `status_ping` with retries - used right after booting a server.
pub fn status_ping_retry(
    host: &str,
    port: u16,
    protocol_hint: i32,
    attempts: u32,
    delay: Duration,
) -> Result<Value> {
    let mut last = None;
    for i in 0..attempts {
        match status_ping(host, port, protocol_hint, Duration::from_secs(5)) {
            Ok(v) => return Ok(v),
            Err(e) => {
                if i + 1 < attempts {
                    std::thread::sleep(delay);
                }
                last = Some(e);
            }
        }
    }
    Err(last.unwrap_or_else(|| anyhow::anyhow!("status ping failed without error")))
}

/// One observed server-to-client packet in a login transcript.
#[derive(Serialize)]
pub struct CapturedPacket {
    pub id: i32,
    pub t_ms: u128,
    pub body_len: usize,
    pub head_hex: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    pub note: Option<String>,
}

/// Serverbound Client Information body for the configuration state:
/// locale, view distance, chat visibility, chat colors, model customisation
/// bitmask, main hand, text filtering, server listings, particle status.
pub fn client_information_body() -> Vec<u8> {
    let mut b = Vec::new();
    doppel_protocol::write_string(&mut b, "en_US");
    b.push(8); // view distance
    doppel_protocol::write_varint(&mut b, 0); // chat visibility: full
    b.push(0x01); // chat colors: true
    b.push(0x7f); // model customisation: all layers
    doppel_protocol::write_varint(&mut b, 1); // main hand: right
    b.push(0x00); // text filtering: off
    b.push(0x01); // allow server listings: true
    doppel_protocol::write_varint(&mut b, 0); // particle status: all
    b
}

/// Sends the next scripted command and arms the short response timeout:
/// the session paces itself on the server's system_chat replies.
fn send_commanded(
    conn: &mut Conn<TcpStream>,
    commands: &[String],
    next_cmd: &mut usize,
) -> Result<()> {
    let cmd = &commands[*next_cmd];
    let mut body = Vec::new();
    doppel_protocol::write_string(&mut body, cmd.trim_start_matches('/'));
    conn.write_packet(0x07, &body)?;
    *next_cmd += 1;
    // While awaiting the response, a short read timeout keeps a missing
    // reply from ending the session at the idle threshold.
    conn.get_ref()
        .set_read_timeout(Some(std::time::Duration::from_millis(1500)))?;
    Ok(())
}

/// Sends `locate biome <name>` and arms the locate read window.
fn send_locate(conn: &mut Conn<TcpStream>, biome: &str) -> Result<()> {
    let mut body = Vec::new();
    doppel_protocol::write_string(&mut body, &format!("locate biome {biome}"));
    conn.write_packet(0x07, &body)?;
    conn.get_ref().set_read_timeout(Some(LOCATE_TIMEOUT))?;
    Ok(())
}

/// Sends the hop teleport (unsigned chat_command, serverbound play
/// 0x07): the located column at a safe altitude.
fn send_teleport(conn: &mut Conn<TcpStream>, (x, _, z): (i32, i32, i32)) -> Result<()> {
    let mut body = Vec::new();
    doppel_protocol::write_string(&mut body, &format!("tp @s {x} {HOP_TP_Y} {z}"));
    conn.write_packet(0x07, &body)
}

/// Packs a block position the wire way (x 26<<38 | z 26<<12 | y 12).
pub fn pack_block_pos(x: i32, y: i32, z: i32) -> i64 {
    (((x as i64) & 0x3ff_ffff) << 38) | (((z as i64) & 0x3ff_ffff) << 12) | ((y as i64) & 0xfff)
}

/// use_item_on body against the top face of the clicked block.
pub fn build_use_item_on_top(x: i32, y: i32, z: i32, sequence: i32) -> Vec<u8> {
    let mut b = Vec::new();
    doppel_protocol::write_varint(&mut b, 0); // hand: main
    b.extend_from_slice(&pack_block_pos(x, y, z).to_be_bytes());
    doppel_protocol::write_varint(&mut b, 1); // face: up
    b.extend_from_slice(&0.5f32.to_be_bytes());
    b.extend_from_slice(&1.0f32.to_be_bytes());
    b.extend_from_slice(&0.5f32.to_be_bytes());
    b.push(0); // inside
    b.push(0); // world border
    doppel_protocol::write_varint(&mut b, sequence);
    b
}

/// Options for a login capture session.
#[derive(Default)]
pub struct CaptureOpts<'a> {
    pub idle_timeout: Option<Duration>,
    pub max_packets: Option<usize>,
    pub dump_dir: Option<&'a std::path::Path>,
    pub commands: &'a [String],
    /// Prebuilt serverbound frames sent once the command volley
    /// completes (interactions with no command response to pace on).
    pub raw_packets: &'a [(i32, Vec<u8>)],
    /// After the join burst, walk this many chunks in +x (one
    /// move_player_pos per 400ms) to exercise chunk streaming.
    pub walk_chunks: Option<usize>,
}

impl CaptureOpts<'_> {
    fn idle_timeout(&self) -> Duration {
        self.idle_timeout.unwrap_or(Duration::from_secs(8))
    }

    fn max_packets(&self) -> usize {
        self.max_packets.unwrap_or(160)
    }
}

/// One position leg of a capture session: the packet index where it
/// starts. Legs after the first come from locate hops and carry the
/// reply text plus the teleport target parsed from it.
pub struct CaptureLeg {
    pub label: String,
    pub start: usize,
    /// Flattened reply text; empty for the spawn leg.
    pub reply: String,
    /// Located coordinates from the reply; absent when the locate found
    /// nothing. Hops teleport to the column at HOP_TP_Y, not this y.
    pub pos: Option<(i32, i32, i32)>,
}

/// A login capture split into position legs: the full packet stream
/// plus one leg per position (spawn first, then each locate hop).
pub struct LeggedCapture {
    pub packets: Vec<CapturedPacket>,
    pub legs: Vec<CaptureLeg>,
}

/// Connects as an offline-mode login client and records every packet the
/// server sends, driving the full confirmed choreography - login, ack,
/// client information, known packs, finish configuration - into the PLAY
/// state, capturing the join sequence. When `dump_dir` is set, every
/// packet's FULL body is also written to `pNNN.bin` there (the JSONL head
/// is truncated; the dumps carry full bodies).
/// A reactive chase probe: when the first add_entity of `entity_type`
/// arrives (after the command volley), teleport to `stand` blocks from
/// it along +x so hostile chase goals engage on a stationary player.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct ChaseFirst {
    pub entity_type: i32,
    pub stand: f64,
}

/// The chase probe's command for an add frame: `stand` blocks from the
/// entity along +x, when the frame is an add_entity of the wanted type.
fn chase_tp(chase: &ChaseFirst, body: &[u8]) -> Option<(String, Vec<u8>)> {
    let mut r = Reader::new(body);
    let _ = r.read_varint().ok()?;
    let _ = r.read_bytes(16).ok()?;
    let ty = r.read_varint().ok()?;
    if ty != chase.entity_type {
        return None;
    }
    let x = r.read_f64().ok()?;
    let y = r.read_f64().ok()?;
    let z = r.read_f64().ok()?;
    let cmd = format!("tp @s {:.1} {:.1} {:.1}", x + chase.stand, y, z);
    let mut cbody = Vec::new();
    doppel_protocol::write_string(&mut cbody, &cmd);
    Some((cmd, cbody))
}

/// `login_capture` with the reactive chase probe armed: once the
/// scripted volley completes and the first add_entity of the wanted type
/// arrives, teleport `stand` blocks from it along +x so hostile chase
/// goals engage on a stationary player.
pub fn login_capture_chase(
    host: &str,
    port: u16,
    protocol: i32,
    login_start_body: &[u8],
    opts: &CaptureOpts<'_>,
    chase: ChaseFirst,
) -> Result<Vec<CapturedPacket>> {
    login_capture_legs_chase(
        host,
        port,
        protocol,
        login_start_body,
        opts,
        &[],
        Some(chase),
    )
    .map(|c| c.packets)
}

pub fn login_capture(
    host: &str,
    port: u16,
    protocol: i32,
    login_start_body: &[u8],
    opts: &CaptureOpts<'_>,
) -> Result<Vec<CapturedPacket>> {
    Ok(login_capture_legs(host, port, protocol, login_start_body, opts, &[])?.packets)
}

/// `login_capture` plus locate hops: after the join burst, locate each
/// listed biome, teleport to the reply's coordinates, and capture the
/// chunk stream there as a further leg per biome. The session ends one
/// quiet settle beat after the last hop's chunk batch closes.
pub fn login_capture_legs(
    host: &str,
    port: u16,
    protocol: i32,
    login_start_body: &[u8],
    opts: &CaptureOpts<'_>,
    locate_biomes: &[&str],
) -> Result<LeggedCapture> {
    login_capture_legs_chase(
        host,
        port,
        protocol,
        login_start_body,
        opts,
        locate_biomes,
        None,
    )
}

fn login_capture_legs_chase(
    host: &str,
    port: u16,
    protocol: i32,
    login_start_body: &[u8],
    opts: &CaptureOpts<'_>,
    locate_biomes: &[&str],
    chase: Option<ChaseFirst>,
) -> Result<LeggedCapture> {
    let idle_timeout = opts.idle_timeout();
    let max_packets = opts.max_packets();
    let dump_dir = opts.dump_dir;
    let commands = opts.commands;
    let stream =
        TcpStream::connect((host, port)).with_context(|| format!("connecting to {host}:{port}"))?;
    stream.set_read_timeout(Some(idle_timeout))?;
    stream.set_write_timeout(Some(idle_timeout))?;
    stream.set_nodelay(true).ok();
    if let Some(dir) = dump_dir {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let mut conn = Conn::new(stream);

    // Handshake (packet 0x00) with next_state=2 (login), sent raw.
    // Unlike status pings, vanilla validates the protocol number here.
    let mut hs = Vec::new();
    doppel_protocol::write_varint(&mut hs, protocol);
    doppel_protocol::write_string(&mut hs, host);
    hs.extend_from_slice(&port.to_be_bytes());
    doppel_protocol::write_varint(&mut hs, 2);
    conn.write_packet(0x00, &hs)?;

    // Login Start (packet 0x00) with the caller's field layout.
    conn.write_packet(0x00, login_start_body)?;

    let mut packets = Vec::new();
    let mut compression_on = false;
    let mut config_started = false;
    let mut packs_answered = false;
    let mut play_started = false;
    let mut commands_pending = !commands.is_empty();
    let mut chase_done = false;
    let mut next_cmd = 0usize;
    let raw_packets = opts.raw_packets;
    let mut raw_sent = false;
    let walk = opts.walk_chunks;
    let mut steps_done = 0usize;
    let mut last_walk_at = std::time::Instant::now();
    let mut walk_base: Option<(f64, f64, f64)> = None;
    // Locate-hop state: Idle fires the first locate when the join burst
    // goes quiet, Locating waits on its system_chat reply, Streaming
    // waits on the post-teleport chunk batch, Settling waits out the
    // quiet beat that closes the leg.
    let mut hop = Hop::Idle;
    let mut hop_idx = 0usize;
    let mut hop_at: Option<std::time::Instant> = None;
    // Spawn-stream completion: entity traffic keeps the session busy past
    // the burst, so quiet is measured on chunk packets, not reads.
    let mut spawn_batch_ends = 0usize;
    let mut last_chunk_at: Option<std::time::Instant> = None;
    let mut ka_since_chunk = false;
    let mut leg_chunks = 0usize;
    let mut legs = vec![CaptureLeg {
        label: "spawn".to_string(),
        start: 0,
        reply: String::new(),
        pos: None,
    }];
    let started = std::time::Instant::now();
    // The budget counts captured packets: read timeouts retried inside
    // hop waits must not spend it, and every wait is wall-clock bounded
    // by HOP_DEADLINE or the idle timeout instead.
    while packets.len() < max_packets {
        let (id, body) = match conn.read_packet() {
            Ok(p) => p,
            Err(e) => {
                // A read timeout while idle with locates pending is a
                // server that went quiet before proving itself alive
                // again; firing a locate into that risks the reply
                // queuing behind a stuck tick loop until the connection
                // times out.
                if play_started && hop == Hop::Idle && hop_idx < locate_biomes.len() {
                    packets.push(CapturedPacket {
                        id: -1,
                        t_ms: started.elapsed().as_millis(),
                        body_len: 0,
                        head_hex: String::new(),
                        file: None,
                        note: Some(
                            "transcript ended: server went quiet before the first locate".into(),
                        ),
                    });
                    break;
                }
                // Read timeouts while a hop is mid-flight are silence, not
                // failure: the locate search blocks the server's main
                // thread, and a post-teleport batch may take seconds to
                // generate. HOP_DEADLINE bounds both.
                if matches!(hop, Hop::Locating | Hop::Streaming | Hop::Armed) {
                    if hop_at.is_some_and(|at| at.elapsed() > HOP_DEADLINE) {
                        packets.push(CapturedPacket {
                            id: -1,
                            t_ms: started.elapsed().as_millis(),
                            body_len: 0,
                            head_hex: String::new(),
                            file: None,
                            note: Some(format!(
                                "transcript ended: {} {} past hop deadline",
                                match hop {
                                    Hop::Locating => "locate",
                                    Hop::Armed => "arming locate for",
                                    _ => "chunk batch for",
                                },
                                locate_biomes[hop_idx]
                            )),
                        });
                        break;
                    }
                    continue;
                }
                // A settle closes on chunk quiet, not read quiet: entity
                // chatter keeps reads busy, and total silence just means
                // the same quiet check runs here instead. HOP_DEADLINE
                // bounds a stream that never finishes.
                if hop == Hop::Settling {
                    let quiet = last_chunk_at.is_some_and(|at| at.elapsed() >= STREAM_QUIET)
                        && leg_chunks >= HOP_MIN_LEG_CHUNKS;
                    let past = hop_at.is_some_and(|at| at.elapsed() > SETTLE_DEADLINE);
                    if quiet || past {
                        hop_idx += 1;
                        if hop_idx < locate_biomes.len() {
                            hop = Hop::Armed;
                            hop_at = Some(std::time::Instant::now());
                            continue;
                        }
                        packets.push(CapturedPacket {
                            id: -1,
                            t_ms: started.elapsed().as_millis(),
                            body_len: 0,
                            head_hex: String::new(),
                            file: None,
                            note: Some(if quiet {
                                "transcript ended: chunk quiet closed the last hop".into()
                            } else {
                                "transcript ended: last hop stream past deadline".into()
                            }),
                        });
                        break;
                    }
                    continue;
                }
                // A scripted command whose reply never came names itself
                // here: the volley stalls at it, and the stall reads as a
                // timeout once the server goes quiet.
                if next_cmd > 0 && next_cmd < commands.len() {
                    eprintln!(
                        "[bot] volley stalled after command {:?} (no reply)",
                        commands[next_cmd - 1]
                    );
                }
                // Record WHY the transcript ended: "idle timeout" (server
                // waiting on us) reads very differently from a framing error.
                packets.push(CapturedPacket {
                    id: -1,
                    t_ms: started.elapsed().as_millis(),
                    body_len: 0,
                    head_hex: String::new(),
                    file: None,
                    note: Some(format!("transcript ended: {e:#}")),
                });
                break;
            }
        };
        let mut note = None;
        // Set Compression arrives raw in login state and switches framing
        // for everything after it.
        if !compression_on && id == 0x03 {
            let threshold = Reader::new(&body)
                .read_varint()
                .context("set compression threshold")?;
            conn.set_compression(threshold);
            compression_on = true;
            note = Some(format!("set compression threshold={threshold}"));
        }
        // Login Success -> client confirms the configuration transition with
        // an empty Login Acknowledged (LOGIN-state serverbound 0x03), then
        // volunteers its Client Information (serverbound config 0x00).
        if !config_started && id == 0x02 {
            conn.write_packet(0x03, &[])?; // login_acknowledged (login state)
            conn.write_packet(0x00, &client_information_body())?; // config state
            config_started = true;
            note = Some("login success; acked + sent client information".into());
        }
        // Known Packs request (S->C 0x0f): reply serverbound config 0x07
        // with an empty array = "send me everything with full NBT".
        if !packs_answered && config_started && id == 0x0f {
            conn.write_packet(0x07, &[0x00])?;
            packs_answered = true;
            note = Some("known packs request; replied empty list".into());
        }
        // Finish Configuration (S->C 0x03, empty, config state): the server
        // sends first and waits for our empty serverbound 0x03 reply, after
        // which the connection enters the PLAY state.
        if packs_answered && !play_started && id == 0x03 && body.is_empty() {
            conn.write_packet(0x03, &[])?;
            play_started = true;
            note = Some("finish configuration; acked - entering play state".into());
        }
        // Keep-alive (S->C play 0x2d): echo the i64
        // challenge back as serverbound 0x1c so the connection survives
        // vanilla's 15s watchdog during long captures.
        if play_started && id == 0x2d {
            ka_since_chunk = true;
            conn.write_packet(0x1c, &body)?;
        }
        // Chunk batch flow control: after every batch_finished, the client
        // reports throughput (serverbound 0x0b, f32 desired chunks/tick);
        // the server sends no further batches until it hears one. Without
        // this, vanilla streams the join batch and then waits forever.
        if play_started && id == 0x2e {
            last_chunk_at = Some(std::time::Instant::now());
            ka_since_chunk = false;
            leg_chunks += 1;
        }
        if play_started && id == 0x0b && !body.is_empty() {
            spawn_batch_ends += 1;
            let mut feedback = Vec::with_capacity(4);
            feedback.extend_from_slice(&64.0f32.to_be_bytes());
            conn.write_packet(0x0b, &feedback)?;
        }
        // player_position (S->C 0x49, the join teleport): the client MUST
        // acknowledge it before the server accepts movement or chat, and
        // before chunk tracking follows the player to a hop position.
        // 26.3's accept_teleportation echoes the id AND the position (id
        // VarInt, pos f64 x3, yaw/pitch f32).
        if play_started && id == 0x49 && body.len() >= 4 {
            let mut r = Reader::new(&body);
            let teleport_id = r.read_varint().context("teleport id")?;
            let x = r.read_f64().context("teleport x")?;
            let y = r.read_f64().context("teleport y")?;
            let z = r.read_f64().context("teleport z")?;
            r.read_f64().ok(); // delta x
            r.read_f64().ok(); // delta y
            r.read_f64().ok(); // delta z
            let yaw = r.read_f32().unwrap_or(0.0);
            let pitch = r.read_f32().unwrap_or(0.0);
            let mut ack = Vec::with_capacity(40);
            doppel_protocol::write_varint(&mut ack, teleport_id);
            ack.extend_from_slice(&x.to_be_bytes());
            ack.extend_from_slice(&y.to_be_bytes());
            ack.extend_from_slice(&z.to_be_bytes());
            ack.extend_from_slice(&yaw.to_be_bytes());
            ack.extend_from_slice(&pitch.to_be_bytes());
            conn.write_packet(0x00, &ack)?;
            walk_base = Some((x, y, z));
            note = Some(format!(
                "teleport ({x:.1},{y:.1},{z:.1}) id={teleport_id}: acked"
            ));
        }
        // Once the chunk batch closes, run the scripted commands one at a
        // time (unsigned chat_command: serverbound play 0x07; the wire
        // string has NO leading slash - clients strip it before sending).
        // Each command waits for the server's system_chat (0x7c) response
        // before the next is sent: a blasted volley makes the reference's
        // command->tick grouping racy, and the final circuit states with
        // it. A short read timeout while awaiting keeps a missing response
        // from stalling the session.
        if play_started && commands_pending && id == 0x0b {
            commands_pending = false; // only trigger on the first batch end
            send_commanded(&mut conn, commands, &mut next_cmd)?;
            note = Some(format!("sent command: {}", commands[0]));
        }
        if id == 0x7c && next_cmd > 0 {
            if next_cmd < commands.len() {
                send_commanded(&mut conn, commands, &mut next_cmd)?;
                note = Some(format!("sent command: {}", commands[next_cmd - 1]));
            } else {
                // Volley complete: back to the session's idle threshold.
                conn.get_ref().set_read_timeout(Some(idle_timeout))?;
                if !raw_sent && !raw_packets.is_empty() {
                    raw_sent = true;
                    for (id, body) in raw_packets {
                        conn.write_packet(*id, body)?;
                    }
                    note = Some("sent raw interaction packets".to_string());
                }
            }
        }
        // Chase probe: after the scripted volley, the first add_entity of
        // the wanted type teleports the bot beside it so hostile chase
        // goals engage on a stationary player.
        if !chase_done && play_started && !commands_pending && next_cmd >= commands.len() {
            if let (Some(chase), 0x01) = (&chase, id) {
                if let Some((cmd, cbody)) = chase_tp(chase, &body) {
                    conn.write_packet(0x07, &cbody)?;
                    note = Some(format!("chase: {cmd}"));
                    chase_done = true;
                }
            }
        }
        // Locate hop replies: a system_chat carrying a locate result ends
        // the wait; other chat while locating is ignored.
        if id == 0x7c && hop == Hop::Locating {
            if let Some(reply) = parse_locate_reply(&body)? {
                match reply {
                    LocateReply::Found { pos, text } => {
                        send_teleport(&mut conn, pos)?;
                        legs.push(CaptureLeg {
                            label: locate_biomes[hop_idx].to_string(),
                            start: packets.len(),
                            reply: text,
                            pos: Some(pos),
                        });
                        leg_chunks = 0;
                        hop = Hop::Streaming;
                        hop_at = Some(std::time::Instant::now());
                        conn.get_ref().set_read_timeout(Some(idle_timeout))?;
                        note = Some(format!(
                            "locate replied; teleported to column ({}, {}) at y={HOP_TP_Y}",
                            pos.0, pos.2
                        ));
                    }
                    LocateReply::Failed { text } => {
                        legs.push(CaptureLeg {
                            label: locate_biomes[hop_idx].to_string(),
                            start: packets.len(),
                            reply: text,
                            pos: None,
                        });
                        if let Some(dir) = dump_dir {
                            let name = format!("p{:03}.bin", packets.len());
                            std::fs::write(dir.join(&name), &body).ok();
                        }
                        packets.push(CapturedPacket {
                            id: -1,
                            t_ms: started.elapsed().as_millis(),
                            body_len: 0,
                            head_hex: String::new(),
                            file: None,
                            note: Some(format!(
                                "transcript ended: locate {} found nothing",
                                locate_biomes[hop_idx]
                            )),
                        });
                        break;
                    }
                }
            }
        }
        if play_started && id == 0x0b && hop == Hop::Streaming {
            hop = Hop::Settling;
            hop_at = Some(std::time::Instant::now());
            note = Some("hop chunk batch closed; settling on chunk quiet".into());
        }
        // The spawn stream is complete once a batch has closed, no chunk
        // has arrived for a beat, and a keep-alive since the last chunk
        // proves the server's tick loop is draining again; entity
        // traffic keeps reads busy, so the checks run on every arriving
        // packet.
        if play_started
            && hop == Hop::Idle
            && hop_idx < locate_biomes.len()
            && spawn_batch_ends > 0
            && ka_since_chunk
            && last_chunk_at.is_some_and(|at| at.elapsed() >= SPAWN_QUIET)
        {
            send_locate(&mut conn, locate_biomes[hop_idx])?;
            hop = Hop::Locating;
            hop_at = Some(std::time::Instant::now());
            note = Some(format!(
                "spawn stream quiet after {spawn_batch_ends} batches; locating {}",
                locate_biomes[hop_idx]
            ));
        }
        // Settling closes on the same chunk quiet once the leg holds a
        // real sample, checked per packet; a starved stream closes on
        // the hop deadline instead of spinning to the packet cap.
        let settled = last_chunk_at.is_some_and(|at| at.elapsed() >= STREAM_QUIET)
            && leg_chunks >= HOP_MIN_LEG_CHUNKS;
        let starved = hop_at.is_some_and(|at| at.elapsed() > SETTLE_DEADLINE);
        if hop == Hop::Settling && (settled || starved) {
            hop_idx += 1;
            if hop_idx < locate_biomes.len() {
                hop = Hop::Armed;
                hop_at = Some(std::time::Instant::now());
                note = Some(if settled {
                    format!(
                        "chunk quiet closed the hop; arming locate {}",
                        locate_biomes[hop_idx]
                    )
                } else {
                    format!(
                        "hop deadline closed a starved stream; arming locate {}",
                        locate_biomes[hop_idx]
                    )
                });
            } else {
                packets.push(CapturedPacket {
                    id: -1,
                    t_ms: started.elapsed().as_millis(),
                    body_len: 0,
                    head_hex: String::new(),
                    file: None,
                    note: Some(if settled {
                        "transcript ended: chunk quiet closed the last hop".into()
                    } else {
                        "transcript ended: hop deadline closed the last starved stream".into()
                    }),
                });
                break;
            }
        }
        // Armed fires on the next keep-alive after the stream closed.
        if hop == Hop::Armed && ka_since_chunk {
            send_locate(&mut conn, locate_biomes[hop_idx])?;
            hop = Hop::Locating;
            hop_at = Some(std::time::Instant::now());
            note = Some(format!(
                "keep-alive armed; locating {}",
                locate_biomes[hop_idx]
            ));
        }
        if matches!(hop, Hop::Locating | Hop::Streaming | Hop::Armed)
            && hop_at.is_some_and(|at| at.elapsed() > HOP_DEADLINE)
        {
            packets.push(CapturedPacket {
                id: -1,
                t_ms: started.elapsed().as_millis(),
                body_len: 0,
                head_hex: String::new(),
                file: None,
                note: Some(format!(
                    "transcript ended: {} {} past hop deadline",
                    match hop {
                        Hop::Locating => "locate",
                        Hop::Armed => "arming locate for",
                        _ => "chunk batch for",
                    },
                    locate_biomes[hop_idx]
                )),
            });
            break;
        }
        // Walk pacing: cross one chunk per step via /tp - vanilla's
        // movement speed checks reject raw move packets this fast, but
        // teleports are legal and trigger the same chunk streaming.
        if let (Some(steps), Some((bx, by, bz))) = (walk, walk_base) {
            if play_started
                && !commands_pending
                && steps_done < steps
                && last_walk_at.elapsed() >= Duration::from_millis(400)
            {
                let x = bx + (steps_done as f64 + 1.0) * 16.0;
                let cmd = format!("tp @s {x} {by} {bz}");
                let mut body = Vec::new();
                doppel_protocol::write_string(&mut body, &cmd);
                conn.write_packet(0x07, &body)?;
                steps_done += 1;
                last_walk_at = std::time::Instant::now();
                note = Some(format!("tp-walk step {steps_done}: x={x:.1}"));
            }
        }
        // Keep plenty of headroom: decoder-error messages arrive inside
        // disconnect packets.
        let head = &body[..body.len().min(4096)];
        // A locate session runs minutes; per-packet dumps fall seconds
        // to tens of seconds behind the wire under entity traffic, and a
        // late keep-alive echo trips the server's 15s pending deadline.
        // Only chunk packets need bodies on disk.
        let mut file = None;
        if let Some(dir) = dump_dir {
            let dump_all = locate_biomes.is_empty();
            if dump_all || id == 0x2e {
                let name = format!("p{:03}.bin", packets.len());
                std::fs::write(dir.join(&name), &body)
                    .with_context(|| format!("dumping {name}"))?;
                let id_name = format!("p{:03}.id", packets.len());
                std::fs::write(dir.join(&id_name), format!("{id:#04x}"))
                    .with_context(|| format!("dumping {id_name}"))?;
                file = Some(name);
            }
        }
        packets.push(CapturedPacket {
            id,
            t_ms: started.elapsed().as_millis(),
            body_len: body.len(),
            head_hex: hex::encode(head),
            file,
            note,
        });
    }
    Ok(LeggedCapture { packets, legs })
}

/// Locate-hop phases inside a legged capture session.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Hop {
    Idle,
    Locating,
    Streaming,
    Settling,
    /// Chunk stream closed; waiting for a keep-alive round trip before
    /// the next locate. Chunk generation buries the server's tick loop
    /// (89 ticks behind at spawn), which delays the command reply past
    /// the keep-alive timeout; a keep-alive received after the last
    /// chunk proves the loop is draining again.
    Armed,
}

/// One parsed /locate reply: the target position, or the failure text.
pub enum LocateReply {
    Found { pos: (i32, i32, i32), text: String },
    Failed { text: String },
}

/// Parses a system_chat (0x7c) body carrying a /locate biome result.
/// Returns None for any other chat. The reply is a translatable
/// component; the coordinates live in its nested chat.coordinates node,
/// so no client-side lang table is needed to read them.
pub fn parse_locate_reply(body: &[u8]) -> Result<Option<LocateReply>> {
    let mut r = Reader::new(body);
    let tag = r.read_u8().context("system chat nbt root tag")?;
    let root = read_nbt(&mut r, tag, 0)?;
    let mut text = String::new();
    flatten_component(&root, &mut text);
    if has_translate(&root, "commands.locate.biome.not_found") {
        return Ok(Some(LocateReply::Failed { text }));
    }
    match find_coordinates(&root) {
        Some(pos) => Ok(Some(LocateReply::Found { pos, text })),
        // A success marker without readable coordinates is a locate reply
        // the harness cannot act on.
        None if has_translate(&root, "commands.locate.biome.success") => {
            Ok(Some(LocateReply::Failed { text }))
        }
        None => Ok(None),
    }
}

/// Reads "[x, y, z]" out of rendered locate text. Biome replies always
/// carry a numeric y; the "~" structure locates print fails to parse.
pub fn parse_locate_text(text: &str) -> Option<(i32, i32, i32)> {
    let open = text.find('[')?;
    let close = text[open..].find(']')? + open;
    let mut parts = text[open + 1..close].split(',');
    let x = parts.next()?.trim().parse().ok()?;
    let y = parts.next()?.trim().parse().ok()?;
    let z = parts.next()?.trim().parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((x, y, z))
}

/// Client-side templates for the translation keys /locate replies use,
/// so flattened text reads like the client renders it. Coordinate
/// extraction never depends on these strings.
fn template(key: &str) -> Option<&'static str> {
    Some(match key {
        "chat.coordinates" => "%s, %s, %s",
        "chat.square_brackets" => "[%s]",
        "commands.locate.biome.success" => "The nearest %s is at %s (%s blocks away)",
        "commands.locate.biome.not_found" => "Could not find a %s within reasonable distance",
        _ => return None,
    })
}

/// True when any component in the tree translates to `key`.
fn has_translate(root: &Nbt, key: &str) -> bool {
    match root {
        Nbt::Compound(fields) => fields.iter().any(|(k, v)| {
            k == "translate" && matches!(v, Nbt::Str(s) if s == key) || has_translate(v, key)
        }),
        Nbt::List(items) => items.iter().any(|i| has_translate(i, key)),
        _ => false,
    }
}

/// Unwraps the empty-name single-field compounds NbtOps boxes
/// heterogeneous `with` arguments into.
fn unboxed(v: &Nbt) -> &Nbt {
    match v {
        Nbt::Compound(fields) if fields.len() == 1 && fields[0].0.is_empty() => &fields[0].1,
        other => other,
    }
}

/// The first chat.coordinates node in the tree: (x, y, z) from its args.
fn find_coordinates(root: &Nbt) -> Option<(i32, i32, i32)> {
    let coord_arg = |v: &Nbt| match unboxed(v) {
        Nbt::Int(i) => Some(*i),
        Nbt::Str(s) => s.parse().ok(),
        _ => None,
    };
    match root {
        Nbt::Compound(fields) => {
            let is_coords = fields.iter().any(|(k, v)| {
                k == "translate" && matches!(v, Nbt::Str(s) if s == "chat.coordinates")
            });
            if is_coords {
                let Nbt::List(args) = fields.iter().find(|(k, _)| k == "with").map(|(_, v)| v)?
                else {
                    return None;
                };
                return Some((
                    coord_arg(args.first()?)?,
                    coord_arg(args.get(1)?)?,
                    coord_arg(args.get(2)?)?,
                ));
            }
            fields.iter().find_map(|(_, v)| find_coordinates(v))
        }
        Nbt::List(items) => items.iter().find_map(find_coordinates),
        _ => None,
    }
}

/// Flattens a text component to plain text: literal text, resolved
/// templates for the known keys, then extras. Unknown translation keys
/// contribute their arguments.
fn flatten_component(v: &Nbt, out: &mut String) {
    match v {
        Nbt::Compound(fields) => {
            if let [(name, inner)] = &fields[..] {
                if name.is_empty() {
                    return flatten_component(inner, out);
                }
            }
            let field = |name: &str| fields.iter().find(|(k, _)| k == name).map(|(_, v)| v);
            if let Some(Nbt::Str(s)) = field("text") {
                out.push_str(s);
            }
            if let Some(Nbt::Str(key)) = field("translate") {
                let args: &[Nbt] = match field("with") {
                    Some(Nbt::List(items)) => items,
                    _ => &[],
                };
                match template(key) {
                    Some(t) => {
                        let mut pieces = t.split("%s");
                        out.push_str(pieces.next().unwrap_or(""));
                        for (piece, arg) in
                            pieces.zip(args.iter().chain(std::iter::repeat(&Nbt::Other)))
                        {
                            flatten_component(arg, out);
                            out.push_str(piece);
                        }
                    }
                    None => {
                        for (i, arg) in args.iter().enumerate() {
                            if i > 0 {
                                out.push(' ');
                            }
                            flatten_component(arg, out);
                        }
                    }
                }
            }
            if let Some(Nbt::List(extra)) = field("extra") {
                for e in extra {
                    flatten_component(e, out);
                }
            }
        }
        Nbt::List(items) => {
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                flatten_component(item, out);
            }
        }
        Nbt::Str(s) => out.push_str(s),
        Nbt::Int(i) => out.push_str(&i.to_string()),
        Nbt::Other => {}
    }
}

/// Minimal network NBT value: enough of the tag set for text
/// components; other payloads are consumed and dropped.
enum Nbt {
    Compound(Vec<(String, Nbt)>),
    List(Vec<Nbt>),
    Str(String),
    Int(i32),
    Other,
}

fn be_i32(r: &mut Reader) -> Result<i32> {
    let b = r.read_bytes(4).context("nbt i32")?;
    Ok(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// NBT strings carry a u16 length, unlike protocol strings (varint).
fn nbt_string(r: &mut Reader) -> Result<String> {
    let len = r.read_u16().context("nbt string length")? as usize;
    if len > 8192 {
        bail!("nbt string length {len} out of range");
    }
    let bytes = r.read_bytes(len).context("nbt string body")?;
    String::from_utf8(bytes).context("nbt string is not valid UTF-8")
}

fn read_nbt(r: &mut Reader, tag: u8, depth: u8) -> Result<Nbt> {
    if depth > 16 {
        bail!("nbt nesting deeper than 16");
    }
    Ok(match tag {
        0x01 | 0x02 | 0x04 | 0x05 | 0x06 => {
            let n = match tag {
                0x01 => 1,
                0x02 => 2,
                0x05 => 4,
                _ => 8,
            };
            r.read_bytes(n).context("nbt scalar")?;
            Nbt::Other
        }
        0x03 => Nbt::Int(be_i32(r)?),
        0x07 => {
            let n = be_i32(r)?.max(0) as usize;
            r.read_bytes(n).context("nbt byte array")?;
            Nbt::Other
        }
        0x08 => Nbt::Str(nbt_string(r)?),
        0x09 => {
            let elem = r.read_u8().context("nbt list element tag")?;
            let len = be_i32(r)?;
            if !(0..=4096).contains(&len) {
                bail!("nbt list length {len} out of range");
            }
            let mut items = Vec::with_capacity(len as usize);
            if elem != 0x00 {
                for _ in 0..len {
                    items.push(read_nbt(r, elem, depth + 1)?);
                }
            }
            Nbt::List(items)
        }
        0x0a => {
            let mut fields = Vec::new();
            loop {
                let t = r.read_u8().context("nbt field tag")?;
                if t == 0x00 {
                    break;
                }
                let name = nbt_string(r)?;
                fields.push((name, read_nbt(r, t, depth + 1)?));
            }
            Nbt::Compound(fields)
        }
        0x0b => {
            let n = be_i32(r)?.max(0) as usize;
            r.read_bytes(4 * n).context("nbt int array")?;
            Nbt::Other
        }
        0x0c => {
            let n = be_i32(r)?.max(0) as usize;
            r.read_bytes(8 * n).context("nbt long array")?;
            Nbt::Other
        }
        other => bail!("unknown nbt tag {other}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locate_text_parses_coordinates() {
        assert_eq!(
            parse_locate_text(
                "The nearest minecraft:cherry_grove is at [123, 64, -456] (31 blocks away)"
            ),
            Some((123, 64, -456))
        );
        assert_eq!(
            parse_locate_text("Teleported Doppel to [1, 2, 3]"),
            Some((1, 2, 3))
        );
        assert_eq!(parse_locate_text("The nearest x is at [1, ~, 3]"), None);
        assert_eq!(parse_locate_text("no brackets at all"), None);
        assert_eq!(parse_locate_text("[1, 2, 3, 4]"), None);
        assert_eq!(parse_locate_text(""), None);
    }

    #[test]
    fn locate_reply_reads_live_cherry_fixture() {
        // Golden body of the real 26.3 system_chat reply, captured from a
        // live server at seed 42: mixed with-args arrive boxed in
        // empty-name single-field compounds, y as a string.
        let body = include_bytes!("../fixtures/locate-reply-cherry-grove.bin");
        match parse_locate_reply(body).expect("parse ok") {
            Some(LocateReply::Found { pos, text }) => {
                assert_eq!(pos, (-1958, 83, -573));
                assert_eq!(
                    text,
                    "The nearest minecraft:cherry_grove is at [-1958, 83, -573] (2004 blocks away)"
                );
                assert_eq!(parse_locate_text(&text), Some((-1958, 83, -573)));
            }
            _ => panic!("expected found"),
        }
    }

    #[test]
    fn locate_reply_reads_live_mangrove_fixture() {
        // Second live capture. The reported y tracks the search origin's
        // altitude (the surface biome fills the column), here the hop
        // altitude, and still arrives as a string.
        let body = include_bytes!("../fixtures/locate-reply-mangrove-swamp.bin");
        match parse_locate_reply(body).expect("parse ok") {
            Some(LocateReply::Found { pos, text }) => {
                assert_eq!(pos, (-1508, 300, -1025));
                assert_eq!(
                    text,
                    "The nearest minecraft:mangrove_swamp is at [-1508, 300, -1025] (633 blocks away)"
                );
                assert_eq!(parse_locate_text(&text), Some((-1508, 300, -1025)));
            }
            _ => panic!("expected found"),
        }
    }

    #[test]
    fn locate_reply_reads_nested_coordinates() {
        // Hand-built variant of the live shape: translate success with
        // [boxed name, brackets(boxed coordinates), boxed distance].
        let mut body = vec![0x0a];
        str_field(&mut body, "translate", "commands.locate.biome.success");
        body.extend_from_slice(&[0x09]);
        str_payload(&mut body, "with");
        body.extend_from_slice(&[0x0a, 0x00, 0x00, 0x00, 0x03]);
        boxed_str(&mut body, "minecraft:cherry_grove");
        str_field(&mut body, "translate", "chat.square_brackets");
        body.extend_from_slice(&[0x09]);
        str_payload(&mut body, "with");
        body.extend_from_slice(&[0x0a, 0x00, 0x00, 0x00, 0x01]);
        str_field(&mut body, "translate", "chat.coordinates");
        body.extend_from_slice(&[0x09]);
        str_payload(&mut body, "with");
        body.extend_from_slice(&[0x0a, 0x00, 0x00, 0x00, 0x03]);
        boxed_int(&mut body, 123);
        boxed_str(&mut body, "64");
        boxed_int(&mut body, -456);
        body.push(0x00);
        body.push(0x00);
        boxed_int(&mut body, 31);
        body.push(0x00);
        match parse_locate_reply(&body).expect("parse ok") {
            Some(LocateReply::Found { pos, text }) => {
                assert_eq!(pos, (123, 64, -456));
                assert_eq!(
                    text,
                    "The nearest minecraft:cherry_grove is at [123, 64, -456] (31 blocks away)"
                );
                assert_eq!(parse_locate_text(&text), Some((123, 64, -456)));
            }
            _ => panic!("expected found"),
        }
    }

    #[test]
    fn locate_reply_reports_not_found() {
        let mut body = vec![0x0a];
        str_field(&mut body, "translate", "commands.locate.biome.not_found");
        body.extend_from_slice(&[0x09]);
        str_payload(&mut body, "with");
        body.extend_from_slice(&[0x0a, 0x00, 0x00, 0x00, 0x01]);
        boxed_str(&mut body, "minecraft:the_end");
        body.push(0x00);
        match parse_locate_reply(&body).expect("parse ok") {
            Some(LocateReply::Failed { text }) => {
                assert_eq!(
                    text,
                    "Could not find a minecraft:the_end within reasonable distance"
                );
            }
            _ => panic!("expected failed"),
        }
    }

    #[test]
    fn locate_reply_ignores_other_chat() {
        let mut body = vec![0x0a];
        str_field(&mut body, "text", "Doppel joined the game");
        body.push(0x00);
        assert!(parse_locate_reply(&body).expect("parse ok").is_none());
    }

    // Network NBT writers for hand-built fixtures.
    fn str_payload(body: &mut Vec<u8>, s: &str) {
        body.extend_from_slice(&(s.len() as u16).to_be_bytes());
        body.extend_from_slice(s.as_bytes());
    }

    fn str_field(body: &mut Vec<u8>, name: &str, value: &str) {
        body.push(0x08);
        str_payload(body, name);
        str_payload(body, value);
    }

    fn boxed_str(body: &mut Vec<u8>, value: &str) {
        str_field(body, "", value);
        body.push(0x00);
    }

    fn boxed_int(body: &mut Vec<u8>, value: i32) {
        body.push(0x03);
        str_payload(body, "");
        i32_raw(body, value);
        body.push(0x00);
    }

    fn i32_raw(body: &mut Vec<u8>, v: i32) {
        body.extend_from_slice(&v.to_be_bytes());
    }
}
