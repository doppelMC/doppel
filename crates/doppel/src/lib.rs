//! Doppel server core: status (server-list ping) parity, the login ->
//! configuration -> play choreography, replaying registry/join/chunk blobs
//! captured from the vanilla oracle.

pub mod blobs;

use anyhow::{bail, Context, Result};
use blobs::Blobs;
use doppel_protocol::{frame_packet, read_packet, write_string, write_varint, Conn, Pin, Reader};
use serde_json::{json, Value};
use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::time::Duration;

const MAX_FRAME: usize = 1024 * 1024;
const DEFAULT_MAX_PLAYERS: i64 = 20;
const COMPRESSION_THRESHOLD: i32 = 256;

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

fn handle_login(stream: TcpStream, pin: &Pin, blobs: Option<&Blobs>) -> Result<()> {
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
    if let Some(b) = blobs {
        for (id, body) in &b.play {
            conn.write_packet(*id, body)?;
        }
    }

    // Steady state: consume client traffic until the connection ends.
    while conn.read_packet().is_ok() {}
    Ok(())
}

// ---------------------------------------------------------------------------
// Connection dispatch
// ---------------------------------------------------------------------------

fn handle_conn(stream: TcpStream, pin: Pin, blobs: Option<Arc<Blobs>>) {
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
            2 => handle_login(stream, &pin, blobs.as_deref()),
            n => bail!("invalid next state {n}"),
        }
    })();
    match result {
        Ok(()) => {}
        Err(e) => eprintln!("[doppel] {peer}: {e:#}"),
    }
}

/// Accept loop. Binds nothing itself so tests can hand us an ephemeral port.
pub fn serve_on(listener: TcpListener, pin: Pin, blobs: Option<Arc<Blobs>>) -> Result<()> {
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let pin = pin.clone();
                let blobs = blobs.clone();
                std::thread::spawn(move || handle_conn(stream, pin, blobs));
            }
            Err(e) => eprintln!("[doppel] accept error: {e}"),
        }
    }
    Ok(())
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
    let listener = TcpListener::bind(addr).with_context(|| format!("binding {addr}"))?;
    println!("[doppel] listening on {addr} (vanilla target: {})", pin.id);
    serve_on(listener, pin, blobs)
}
