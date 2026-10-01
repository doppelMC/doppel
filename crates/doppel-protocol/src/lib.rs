//! Packet primitives shared by the Doppel server and the oracle harness.
//!
//! This crate is deliberately tiny and synchronous: VarInts, packet framing
//! over `impl Read`/`impl Write`, the handshake/status packet encodings, and
//! the version pin artifact. Everything here must behave byte-identically to
//! vanilla Java Edition, so every encoding has unit tests.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Minecraft protocol VarInt: LEB128-style, 7 bits per byte, max 5 bytes.
pub fn write_varint(buf: &mut Vec<u8>, value: i32) {
    let mut v = value as u32;
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if v == 0 {
            break;
        }
    }
}

/// Number of bytes `write_varint` will emit for this value.
pub fn varint_len(value: i32) -> usize {
    let mut tmp = Vec::with_capacity(5);
    write_varint(&mut tmp, value);
    tmp.len()
}

/// Writes a protocol string: VarInt byte-length prefix + UTF-8 bytes.
pub fn write_string(buf: &mut Vec<u8>, s: &str) {
    write_varint(buf, s.len() as i32);
    buf.extend_from_slice(s.as_bytes());
}

/// Wraps `payload` (packet id + body) in a length-prefixed frame.
pub fn frame_packet(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 5);
    write_varint(&mut out, payload.len() as i32);
    out.extend_from_slice(payload);
    out
}

/// Cursor over an already-buffered packet body.
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    pub fn read_u8(&mut self) -> Result<u8> {
        let b = *self
            .buf
            .get(self.pos)
            .context("packet truncated: expected u8")?;
        self.pos += 1;
        Ok(b)
    }

    pub fn read_varint(&mut self) -> Result<i32> {
        let mut value: u32 = 0;
        for i in 0..5 {
            let b = self.read_u8()?;
            value |= u32::from(b & 0x7f) << (7 * i);
            if b & 0x80 == 0 {
                return Ok(value as i32);
            }
        }
        bail!("VarInt longer than 5 bytes");
    }

    pub fn read_u16(&mut self) -> Result<u16> {
        let hi = self.read_u8()? as u16;
        let lo = self.read_u8()? as u16;
        Ok((hi << 8) | lo)
    }

    pub fn read_i64(&mut self) -> Result<i64> {
        let mut bytes = [0u8; 8];
        for b in &mut bytes {
            *b = self.read_u8()?;
        }
        Ok(i64::from_be_bytes(bytes))
    }

    pub fn read_string(&mut self, max_bytes: usize) -> Result<String> {
        let len = self.read_varint()?;
        if len < 0 || len as usize > max_bytes {
            bail!("string length {len} out of bounds (max {max_bytes})");
        }
        let start = self.pos;
        self.pos += len as usize;
        let bytes = self
            .buf
            .get(start..self.pos)
            .context("packet truncated: string body")?;
        String::from_utf8(bytes.to_vec()).context("string is not valid UTF-8")
    }
}

/// Reads one length-prefixed packet from a stream.
/// Returns `(packet_id, body_after_id)`.
pub fn read_packet(stream: &mut impl Read, max_frame: usize) -> Result<(i32, Vec<u8>)> {
    let len = read_varint_stream(stream).context("reading frame length")?;
    if len < 0 || len as usize > max_frame {
        bail!("frame length {len} exceeds limit {max_frame}");
    }
    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf).context("reading frame body")?;
    let mut r = Reader::new(&buf);
    let id = r.read_varint().context("reading packet id")?;
    Ok((id, r.buf[r.pos..].to_vec()))
}

/// Reads a bare VarInt directly from a stream (used for frame lengths).
pub fn read_varint_stream(stream: &mut impl Read) -> Result<i32> {
    let mut value: u32 = 0;
    for i in 0..5 {
        let mut b = [0u8; 1];
        stream
            .read_exact(&mut b)
            .context("stream closed mid-varint")?;
        value |= u32::from(b[0] & 0x7f) << (7 * i);
        if b[0] & 0x80 == 0 {
            return Ok(value as i32);
        }
    }
    bail!("stream VarInt longer than 5 bytes");
}

/// Builds the handshake packet (id 0x00) for the given next state (1=status, 2=login).
pub fn encode_handshake(protocol: i32, addr: &str, port: u16, next_state: i32) -> Vec<u8> {
    let mut payload = Vec::with_capacity(16);
    write_varint(&mut payload, 0x00);
    write_varint(&mut payload, protocol);
    write_string(&mut payload, addr);
    payload.extend_from_slice(&port.to_be_bytes());
    write_varint(&mut payload, next_state);
    frame_packet(&payload)
}

/// Empty status request packet (id 0x00).
pub fn encode_status_request() -> Vec<u8> {
    let mut payload = Vec::with_capacity(1);
    write_varint(&mut payload, 0x00);
    frame_packet(&payload)
}

/// Ping packet (id 0x01) with an arbitrary payload the server must echo.
pub fn encode_ping(payload: i64) -> Vec<u8> {
    let mut p = Vec::with_capacity(9);
    write_varint(&mut p, 0x01);
    p.extend_from_slice(&payload.to_be_bytes());
    frame_packet(&p)
}

/// A packet-framed connection that transparently handles the zlib
/// compression vanilla enables mid-login (Set Compression). Before any
/// threshold is set, frames are plain length-prefixed packets; afterwards
/// each frame is `VarInt frame_len | VarInt data_len | payload` where a
/// `data_len` of 0 means the payload is stored raw.
pub struct Conn<S: Read + Write> {
    stream: S,
    threshold: Option<i32>,
    max_frame: usize,
    max_decompressed: usize,
}

const DEFAULT_MAX_FRAME: usize = 8 * 1024 * 1024;
const DEFAULT_MAX_DECOMPRESSED: usize = 8 * 1024 * 1024;

impl<S: Read + Write> Conn<S> {
    pub fn new(stream: S) -> Self {
        Conn {
            stream,
            threshold: None,
            max_frame: DEFAULT_MAX_FRAME,
            max_decompressed: DEFAULT_MAX_DECOMPRESSED,
        }
    }

    /// Enables compression with the given threshold in bytes. Packets whose
    /// payload is >= threshold are zlib-compressed.
    pub fn set_compression(&mut self, threshold: i32) {
        self.threshold = Some(threshold);
    }

    pub fn into_inner(self) -> S {
        self.stream
    }

    /// Reads one packet, transparently decompressing if needed.
    /// Returns `(packet_id, body_after_id)`.
    pub fn read_packet(&mut self) -> Result<(i32, Vec<u8>)> {
        let frame_len = read_varint_stream(&mut self.stream).context("frame length")?;
        if frame_len < 0 || frame_len as usize > self.max_frame {
            bail!("frame length {frame_len} exceeds limit {}", self.max_frame);
        }
        let mut frame = vec![0u8; frame_len as usize];
        self.stream.read_exact(&mut frame).context("frame body")?;

        let payload: Vec<u8> = if self.threshold.is_some() {
            let mut r = Reader::new(&frame);
            let data_len = r.read_varint().context("data length")?;
            let rest = &frame[r.pos..];
            if data_len == 0 {
                rest.to_vec()
            } else {
                let data_len = data_len as usize;
                if data_len > self.max_decompressed {
                    bail!("decompressed size {data_len} exceeds limit");
                }
                let mut out = Vec::with_capacity(data_len.min(1024 * 1024));
                flate2::read::ZlibDecoder::new(rest)
                    .read_to_end(&mut out)
                    .context("decompressing packet")?;
                if out.len() != data_len {
                    bail!(
                        "decompressed size mismatch: declared {data_len}, got {}",
                        out.len()
                    );
                }
                out
            }
        } else {
            frame
        };

        let mut r = Reader::new(&payload);
        let id = r.read_varint().context("packet id")?;
        Ok((id, r.buf[r.pos..].to_vec()))
    }

    /// Writes one packet, compressing when enabled and large enough.
    pub fn write_packet(&mut self, id: i32, body: &[u8]) -> Result<()> {
        let mut payload = Vec::with_capacity(body.len() + 5);
        write_varint(&mut payload, id);
        payload.extend_from_slice(body);

        let frame: Vec<u8> = match self.threshold {
            Some(t) if payload.len() as i32 >= t => {
                let mut enc =
                    flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
                enc.write_all(&payload).context("compressing packet")?;
                let z = enc.finish().context("finishing compression")?;
                let mut f = Vec::with_capacity(z.len() + 5);
                write_varint(&mut f, payload.len() as i32);
                f.extend_from_slice(&z);
                f
            }
            Some(_) => {
                let mut f = Vec::with_capacity(payload.len() + 1);
                write_varint(&mut f, 0);
                f.extend_from_slice(&payload);
                f
            }
            None => payload,
        };

        let mut out = Vec::with_capacity(frame.len() + 5);
        write_varint(&mut out, frame.len() as i32);
        out.extend_from_slice(&frame);
        self.stream.write_all(&out).context("writing frame")?;
        Ok(())
    }
}

/// The version pin: which vanilla build Doppel targets. Generated by
/// `doppel-oracle pin`,
/// self-healed by `doppel-oracle parity-status` (the protocol number and
/// version name are discovered by asking the vanilla oracle itself).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pin {
    /// Piston version id, e.g. "26.3".
    pub id: String,
    /// ISO timestamp from the piston manifest.
    pub release_time: String,
    /// Download URL for the vanilla server jar.
    pub server_jar_url: String,
    /// SHA-1 of the server jar.
    pub server_jar_sha1: String,
    /// Major Java version required by this build.
    pub java_major: u32,
    /// Protocol version number, discovered from the vanilla oracle.
    #[serde(default)]
    pub protocol: Option<i32>,
    /// Version name as reported in vanilla's status response.
    #[serde(default)]
    pub version_name: Option<String>,
}

/// Locates the repo root by walking up from the cwd: a workspace manifest
/// plus either the pins directory (normal checkout) or `.git` (fresh clone
/// before the first pin exists).
pub fn find_repo_root() -> Result<PathBuf> {
    let mut dir: &Path = &std::env::current_dir().context("no cwd")?;
    loop {
        let has_manifest = dir.join("Cargo.toml").is_file();
        let marked = dir.join("pins").is_dir() || dir.join(".git").exists();
        if has_manifest && marked {
            return Ok(dir.to_path_buf());
        }
        dir = dir
            .parent()
            .context("reached filesystem root without finding the repo")?;
    }
}

pub fn pin_path() -> Result<PathBuf> {
    Ok(find_repo_root()?.join("pins").join("version.json"))
}

pub fn load_pin() -> Result<Pin> {
    let path = pin_path()?;
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading pin file {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

pub fn save_pin(pin: &Pin) -> Result<()> {
    let path = pin_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(pin).context("serializing pin")?;
    std::fs::write(&path, text + "\n").with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_varint(v: i32) {
        let mut buf = Vec::new();
        write_varint(&mut buf, v);
        assert_eq!(buf.len(), varint_len(v));
        let mut r = Reader::new(&buf);
        assert_eq!(r.read_varint().unwrap(), v);
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn varint_known_encodings() {
        let mut buf = Vec::new();
        write_varint(&mut buf, 0);
        assert_eq!(buf, vec![0x00]);
        let mut buf = Vec::new();
        write_varint(&mut buf, 1);
        assert_eq!(buf, vec![0x01]);
        let mut buf = Vec::new();
        write_varint(&mut buf, 128);
        assert_eq!(buf, vec![0x80, 0x01]);
    }

    #[test]
    fn varint_roundtrip_boundaries() {
        for v in [
            0,
            1,
            127,
            128,
            255,
            256,
            32_767,
            32_768,
            2_097_151,
            2_097_152,
            i32::MAX,
            -1,
            i32::MIN,
        ] {
            roundtrip_varint(v);
        }
    }

    #[test]
    fn varint_rejects_six_bytes() {
        let buf = vec![0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
        assert!(Reader::new(&buf).read_varint().is_err());
    }

    #[test]
    fn string_roundtrip() {
        let mut buf = Vec::new();
        write_string(&mut buf, "localhost");
        let mut r = Reader::new(&buf);
        assert_eq!(r.read_string(256).unwrap(), "localhost");
    }

    #[test]
    fn string_respects_max() {
        let mut buf = Vec::new();
        write_string(&mut buf, "a-longer-string");
        assert!(Reader::new(&buf).read_string(4).is_err());
    }

    #[test]
    fn frame_roundtrip_via_stream() {
        let packet = encode_handshake(767, "localhost", 25565, 1);
        // A stream containing handshake + status request back to back.
        let bytes: Vec<u8> = [packet.as_slice(), encode_status_request().as_slice()].concat();
        let mut stream: &[u8] = bytes.as_slice();
        let (id, body) = read_packet(&mut stream, 4096).unwrap();
        assert_eq!(id, 0x00);
        let mut r = Reader::new(&body);
        assert_eq!(r.read_varint().unwrap(), 767);
        assert_eq!(r.read_string(256).unwrap(), "localhost");
        assert_eq!(r.read_u16().unwrap(), 25565);
        assert_eq!(r.read_varint().unwrap(), 1);
        assert_eq!(r.remaining(), 0);
        let (id2, body2) = read_packet(&mut stream, 4096).unwrap();
        assert_eq!(id2, 0x00);
        assert!(body2.is_empty());
    }

    #[test]
    fn frame_length_limit_enforced() {
        let bytes = encode_handshake(1, "x", 1, 1);
        let mut stream: &[u8] = bytes.as_slice();
        assert!(read_packet(&mut stream, 4).is_err());
    }

    #[test]
    fn conn_roundtrip_uncompressed() {
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut c = Conn::new(&mut buf);
            c.write_packet(0x00, &[1, 2, 3]).unwrap();
            c.write_packet(0x05, b"hello").unwrap();
        }
        buf.set_position(0);
        let mut c = Conn::new(&mut buf);
        let (id, body) = c.read_packet().unwrap();
        assert_eq!(id, 0x00);
        assert_eq!(body, vec![1u8, 2, 3]);
        let (id, body) = c.read_packet().unwrap();
        assert_eq!(id, 0x05);
        assert_eq!(body, b"hello".to_vec());
    }

    #[test]
    fn conn_roundtrip_compressed() {
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut c = Conn::new(&mut buf);
            c.set_compression(64);
            // Below threshold: stored raw inside the compressed framing.
            c.write_packet(0x01, b"tiny").unwrap();
            // At/above threshold and compressible: zlib path.
            let big: Vec<u8> = (0..200u8).cycle().take(500).collect();
            c.write_packet(0x02, &big).unwrap();
        }
        buf.set_position(0);
        let mut c = Conn::new(&mut buf);
        c.set_compression(64); // reader must share the compression state
        let (id, body) = c.read_packet().unwrap();
        assert_eq!(id, 0x01);
        assert_eq!(body, b"tiny".to_vec());
        let (id, body) = c.read_packet().unwrap();
        assert_eq!(id, 0x02);
        let expected: Vec<u8> = (0..200u8).cycle().take(500).collect();
        assert_eq!(body, expected);
    }
}
