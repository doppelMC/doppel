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
    idle_timeout: Duration,
    max_packets: usize,
    dump_dir: Option<&std::path::Path>,
) -> Result<Vec<CapturedPacket>> {
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
