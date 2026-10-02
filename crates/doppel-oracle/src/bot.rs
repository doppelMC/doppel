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

/// `status_ping` with retries — used right after booting a server.
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

/// Connects as an offline-mode login client and records every packet the
/// server sends, driving the full confirmed choreography — login, ack,
/// client information, known packs, finish configuration — into the PLAY
/// state, capturing the join sequence. When `dump_dir` is set, every
/// packet's FULL body is also written to `pNNN.bin` there (the JSONL head
/// is truncated; the dumps carry full bodies).
pub fn login_capture(
    host: &str,
    port: u16,
    protocol: i32,
    login_start_body: &[u8],
    opts: &CaptureOpts<'_>,
) -> Result<Vec<CapturedPacket>> {
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
    let mut next_cmd = 0usize;
    let raw_packets = opts.raw_packets;
    let mut raw_sent = false;
    let walk = opts.walk_chunks;
    let mut steps_done = 0usize;
    let mut last_walk_at = std::time::Instant::now();
    let mut walk_base: Option<(f64, f64, f64)> = None;
    let started = std::time::Instant::now();
    for _ in 0..max_packets {
        let (id, body) = match conn.read_packet() {
            Ok(p) => p,
            Err(e) => {
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
            note = Some("finish configuration; acked — entering play state".into());
        }
        // Keep-alive (S->C play 0x2d): echo the i64
        // challenge back as serverbound 0x1c so the connection survives
        // vanilla's 15s watchdog during long captures.
        if play_started && id == 0x2d {
            conn.write_packet(0x1c, &body)?;
        }
        // Chunk batch flow control: after every batch_finished, the client
        // reports throughput (serverbound 0x0b, f32 desired chunks/tick);
        // the server sends no further batches until it hears one. Without
        // this, vanilla streams the join batch and then waits forever.
        if play_started && id == 0x0b && !body.is_empty() {
            let mut feedback = Vec::with_capacity(4);
            feedback.extend_from_slice(&64.0f32.to_be_bytes());
            conn.write_packet(0x0b, &feedback)?;
        }
        // player_position (S->C 0x49, the join teleport): the client MUST
        // acknowledge it before the server accepts movement or chat. 26.3's
        // accept_teleportation echoes the id AND the full position (id
        // VarInt; pos f64×3; delta f64×3; yaw/pitch f32; relatives i32).
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
        // string has NO leading slash — clients strip it before sending).
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
                    for (i, (id, body)) in raw_packets.iter().enumerate() {
                        if i > 0 {
                            // Space the interactions apart so a decoder
                            // rejection is attributable to one packet.
                            std::thread::sleep(Duration::from_millis(1500));
                        }
                        conn.write_packet(*id, body)?;
                    }
                    note = Some("sent raw interaction packets".to_string());
                }
            }
        }
        // Walk pacing: cross one chunk per step via /tp — vanilla's
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
        let mut file = None;
        if let Some(dir) = dump_dir {
            let name = format!("p{:03}.bin", packets.len());
            std::fs::write(dir.join(&name), &body).with_context(|| format!("dumping {name}"))?;
            file = Some(name);
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
    Ok(packets)
}
