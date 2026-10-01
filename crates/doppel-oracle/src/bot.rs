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
    pub body_len: usize,
    pub head_hex: String,
    pub note: Option<String>,
}

/// Connects as an offline-mode login client and records every packet the
/// server sends from the moment of login until the stream goes idle.
/// This is protocol discovery: the transcript tells us the exact packet
/// sequence vanilla 26.3 uses, which we then implement.
pub fn login_capture(
    host: &str,
    port: u16,
    protocol: i32,
    username: &str,
    idle_timeout: Duration,
    max_packets: usize,
) -> Result<Vec<CapturedPacket>> {
    let stream =
        TcpStream::connect((host, port)).with_context(|| format!("connecting to {host}:{port}"))?;
    stream.set_read_timeout(Some(idle_timeout))?;
    stream.set_write_timeout(Some(idle_timeout))?;
    stream.set_nodelay(true).ok();
    let mut conn = Conn::new(stream);

    // Handshake (packet 0x00) with next_state=2 (login), sent raw.
    // Unlike status pings, vanilla validates the protocol number here.
    let mut hs = Vec::new();
    doppel_protocol::write_varint(&mut hs, protocol);
    doppel_protocol::write_string(&mut hs, host);
    hs.extend_from_slice(&port.to_be_bytes());
    doppel_protocol::write_varint(&mut hs, 2);
    conn.write_packet(0x00, &hs)?;

    // Login Start (packet 0x00): username + no UUID (boolean false).
    let mut ls = Vec::new();
    doppel_protocol::write_string(&mut ls, username);
    ls.push(0x00);
    conn.write_packet(0x00, &ls)?;

    let mut packets = Vec::new();
    let mut compression_on = false;
    for _ in 0..max_packets {
        let (id, body) = match conn.read_packet() {
            Ok(p) => p,
            Err(_) => break, // idle timeout, EOF, or disconnect: transcript over
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
        let head = &body[..body.len().min(64)];
        packets.push(CapturedPacket {
            id,
            body_len: body.len(),
            head_hex: hex::encode(head),
            note,
        });
    }
    Ok(packets)
}
