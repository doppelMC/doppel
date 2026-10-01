//! A minimal wire-protocol client used to observe servers (vanilla and
//! Doppel) from the outside, exactly like a real client would.

use anyhow::{bail, Context, Result};
use doppel_protocol::{encode_handshake, encode_ping, encode_status_request, read_packet, Reader};
use serde_json::Value;
use std::io::{Read, Write};
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
