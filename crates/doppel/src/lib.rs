//! Doppel server core. M0 scope: status (server-list ping) with full
//! handshake/ping handling, and a polite disconnect for login attempts.

use anyhow::{Context, Result};
use doppel_protocol::{frame_packet, read_packet, write_string, write_varint, Pin, Reader};
use serde_json::{json, Value};
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

const MAX_FRAME: usize = 1024 * 1024;
const DEFAULT_MAX_PLAYERS: i64 = 20;

/// The status response body, mirroring vanilla's shape.
/// `client_protocol` is used as a fallback when the pin has not yet been
/// healed by the oracle (vanilla always reports its own number; the parity
/// harness exists precisely to catch that difference).
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
        "description": {
            "text": "A Minecraft Server",
        },
    })
}

/// Encode a status response packet (id 0x00, JSON string body).
fn encode_status_response(value: &Value) -> Vec<u8> {
    let mut payload = Vec::new();
    write_varint(&mut payload, 0x00);
    write_string(&mut payload, &value.to_string());
    frame_packet(&payload)
}

/// Encode a login-state disconnect packet (id 0x00, chat component body).
fn encode_login_disconnect(text: &str) -> Vec<u8> {
    let mut payload = Vec::new();
    write_varint(&mut payload, 0x00);
    write_string(&mut payload, &json!({ "text": text }).to_string());
    frame_packet(&payload)
}

fn encode_pong(payload: i64) -> Vec<u8> {
    let mut p = Vec::new();
    write_varint(&mut p, 0x01);
    p.extend_from_slice(&payload.to_be_bytes());
    frame_packet(&p)
}

fn handle_conn(mut stream: TcpStream, pin: Pin) {
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
    let result = handle_conn_inner(&mut stream, &pin);
    match result {
        Ok(()) => {}
        Err(e) => eprintln!("[doppel] {peer}: {e:#}"),
    }
}

fn handle_conn_inner(stream: &mut TcpStream, pin: &Pin) -> Result<()> {
    // --- Handshake (state -1) ---
    let (id, body) = read_packet(stream, MAX_FRAME).context("handshake")?;
    if id != 0x00 {
        anyhow::bail!("expected handshake (id 0), got id {id}");
    }
    let mut r = Reader::new(&body);
    let client_protocol = r.read_varint().context("protocol version")?;
    let _addr = r.read_string(1024).context("server address")?;
    let _port = r.read_u16().context("server port")?;
    let next_state = r.read_varint().context("next state")?;
    if next_state == 1 {
        handle_status(stream, pin, client_protocol)
    } else if next_state == 2 {
        handle_login(stream)
    } else {
        anyhow::bail!("invalid next state {next_state}");
    }
}

fn handle_status(stream: &mut TcpStream, pin: &Pin, client_protocol: i32) -> Result<()> {
    // Status request (empty 0x00), then respond. A ping may follow.
    let (id, _body) = read_packet(stream, MAX_FRAME).context("status request")?;
    if id != 0x00 {
        anyhow::bail!("expected status request (id 0), got id {id}");
    }
    let resp = encode_status_response(&status_response(pin, client_protocol));
    stream.write_all(&resp).context("writing status response")?;

    // Optional ping/pong.
    if let Ok((id, body)) = read_packet(stream, MAX_FRAME) {
        if id == 0x01 {
            let mut r = Reader::new(&body);
            let payload = r.read_i64().context("ping payload")?;
            stream
                .write_all(&encode_pong(payload))
                .context("writing pong")?;
        }
    }
    Ok(())
}

fn handle_login(stream: &mut TcpStream) -> Result<()> {
    // Login start is read but ignored; we answer with a disconnect.
    let (_id, _body) = read_packet(stream, MAX_FRAME).context("login start")?;
    let disconnect = encode_login_disconnect("Doppel M0: login not implemented yet");
    stream
        .write_all(&disconnect)
        .context("writing login disconnect")?;
    Ok(())
}

/// Accept loop. Binds nothing itself so tests can hand us an ephemeral port.
pub fn serve_on(listener: TcpListener, pin: Pin) -> Result<()> {
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let pin = pin.clone();
                std::thread::spawn(move || handle_conn(stream, pin));
            }
            Err(e) => eprintln!("[doppel] accept error: {e}"),
        }
    }
    Ok(())
}

/// Convenience for the binary: load pin, bind, serve.
pub fn serve(addr: &str, pin_path: Option<&std::path::Path>) -> Result<()> {
    let pin = if let Some(path) = pin_path {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading pin {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?
    } else {
        doppel_protocol::load_pin()?
    };
    let listener = TcpListener::bind(addr).with_context(|| format!("binding {addr}"))?;
    println!("[doppel] listening on {addr} (vanilla target: {})", pin.id);
    serve_on(listener, pin)
}
