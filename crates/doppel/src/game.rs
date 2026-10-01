//! The game thread: single owner of world state, players, and streaming
//! decisions. Connection actors. Connection threads are IO actors that
//! forward inbound events and drain outbound frames; nothing here touches
//! a socket. This is the skeleton the block-modification tick loop hangs
//! from.

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use crate::blobs::Blobs;
use crate::WireChunk;

pub type ConnId = u64;

/// Events from connection actors to the game thread.
pub enum Inbound {
    /// A player finished the join burst; `sent` lists chunks already
    /// delivered by the replay.
    Joined {
        conn: ConnId,
        name: String,
        x: f64,
        y: f64,
        z: f64,
        sent: Vec<(i32, i32)>,
    },
    Moved {
        conn: ConnId,
        x: f64,
        y: f64,
        z: f64,
    },
    /// `tp @s x y z` (the walk-parity bot's vehicle).
    Tp {
        conn: ConnId,
        x: f64,
        y: f64,
        z: f64,
    },
    KeepAliveAnswer {
        conn: ConnId,
        id: i64,
    },
    Left {
        conn: ConnId,
    },
}

/// Frames the connection actor's writer thread puts on the wire.
pub enum Outbound {
    Frame { id: i32, body: Vec<u8> },
    Disconnect,
}

struct Player {
    name: String,
    x: f64,
    y: f64,
    z: f64,
    yaw: f32,
    pitch: f32,
    center: Option<(i32, i32)>,
    sent: std::collections::HashSet<(i32, i32)>,
    teleport_id: i32,
    pending_keep_alive: Option<(i64, Instant)>,
}

/// One cached, versioned chunk. `wire` is the sendable form; block
/// modification will mutate sections and bump `version` (invalidating the
/// encoded-frame cache).
pub struct CachedChunk {
    pub wire: WireChunk,
    pub version: u64,
}

pub struct Game {
    chunks: HashMap<(i32, i32), CachedChunk>,
    players: HashMap<ConnId, Player>,
    /// Inverse index: chunk column -> connections tracking it.
    viewers: HashMap<(i32, i32), Vec<ConnId>>,
    inbound: Receiver<Inbound>,
    outbounds: HashMap<ConnId, Sender<Outbound>>,
    world: Option<std::sync::Arc<std::sync::Mutex<crate::WorldState>>>,
    blobs: Option<std::sync::Arc<Blobs>>,
    next_conn: ConnId,
}

const VIEW_RADIUS: i32 = 4;
const KEEP_ALIVE_INTERVAL: Duration = Duration::from_secs(15);

impl Game {
    pub fn new(
        inbound: Receiver<Inbound>,
        world: Option<std::sync::Arc<std::sync::Mutex<crate::WorldState>>>,
        blobs: Option<std::sync::Arc<Blobs>>,
    ) -> Game {
        Game {
            chunks: HashMap::new(),
            players: HashMap::new(),
            viewers: HashMap::new(),
            inbound,
            outbounds: HashMap::new(),
            world,
            blobs,
            next_conn: 0,
        }
    }

    pub fn register(&mut self, tx: Sender<Outbound>) -> ConnId {
        let id = self.next_conn;
        self.next_conn += 1;
        self.outbounds.insert(id, tx);
        id
    }

    /// The event loop. The 1s recv timeout doubles as the coarse keep-alive
    /// tick; the real 20 TPS loop replaces this when simulation arrives.
    pub fn run(&mut self) {
        loop {
            match self.inbound.recv_timeout(Duration::from_secs(1)) {
                Ok(event) => self.handle(event),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
            }
            self.tick_keep_alives();
        }
    }

    fn handle(&mut self, event: Inbound) {
        match event {
            Inbound::Joined {
                conn,
                name,
                x,
                y,
                z,
                sent,
            } => {
                self.players.insert(
                    conn,
                    Player {
                        name,
                        x,
                        y,
                        z,
                        yaw: 0.0,
                        pitch: 0.0,
                        center: None,
                        sent: sent.into_iter().collect(),
                        teleport_id: 1,
                        pending_keep_alive: None,
                    },
                );
            }
            Inbound::Moved { conn, x, y, z } => {
                if let Some(p) = self.players.get_mut(&conn) {
                    p.x = x;
                    p.y = y;
                    p.z = z;
                }
                self.stream_if_moved(conn);
            }
            Inbound::Tp { conn, x, y, z } => {
                let Some(p) = self.players.get_mut(&conn) else {
                    return;
                };
                p.x = x;
                p.y = y;
                p.z = z;
                let mut sync = Vec::with_capacity(64);
                doppel_protocol::write_varint(&mut sync, p.teleport_id);
                sync.extend_from_slice(&x.to_be_bytes());
                sync.extend_from_slice(&y.to_be_bytes());
                sync.extend_from_slice(&z.to_be_bytes());
                sync.extend_from_slice(&0.0f64.to_be_bytes());
                sync.extend_from_slice(&0.0f64.to_be_bytes());
                sync.extend_from_slice(&0.0f64.to_be_bytes());
                sync.extend_from_slice(&p.yaw.to_be_bytes());
                sync.extend_from_slice(&p.pitch.to_be_bytes());
                sync.extend_from_slice(&0i32.to_be_bytes());
                p.teleport_id += 1;
                self.send(conn, 0x49, &sync);
                self.stream_if_moved(conn);
            }
            Inbound::KeepAliveAnswer { conn, id } => {
                if let Some(p) = self.players.get_mut(&conn) {
                    if let Some((challenge, _)) = p.pending_keep_alive {
                        if challenge == id {
                            p.pending_keep_alive = None;
                        }
                    }
                }
            }
            Inbound::Left { conn } => {
                let Some(p) = self.players.remove(&conn) else {
                    return;
                };
                self.outbounds.remove(&conn);
                for chunk in p.sent {
                    if let Some(v) = self.viewers.get_mut(&chunk) {
                        v.retain(|c| *c != conn);
                    }
                }
            }
        }
    }

    fn tick_keep_alives(&mut self) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let conns: Vec<ConnId> = self.players.keys().copied().collect();
        for conn in conns {
            let Some(p) = self.players.get_mut(&conn) else {
                continue;
            };
            match p.pending_keep_alive {
                None => {
                    let body = now_ms.to_be_bytes().to_vec();
                    p.pending_keep_alive = Some((now_ms, Instant::now()));
                    self.send(conn, 0x2d, &body);
                }
                Some((_, sent)) if sent.elapsed() > KEEP_ALIVE_INTERVAL => {
                    eprintln!("[game] {}: keep-alive timeout", p.name);
                    self.send(conn, i32::MAX, &[]); // sentinel; Disconnect below
                    let _ = self
                        .outbounds
                        .get(&conn)
                        .map(|tx| tx.send(Outbound::Disconnect));
                    self.handle(Inbound::Left { conn });
                }
                Some(_) => {}
            }
        }
    }

    fn send(&mut self, conn: ConnId, id: i32, body: &[u8]) {
        if let Some(tx) = self.outbounds.get(&conn) {
            let _ = tx.send(Outbound::Frame {
                id,
                body: body.to_vec(),
            });
        }
    }

    /// Vanilla's streaming choreography, verbatim from the capture: cache
    /// center, forgets, then one batch of entering chunks. All decisions
    /// happen up front (single &mut borrow); frames and loads are applied
    /// after the borrow ends.
    fn stream_if_moved(&mut self, conn: ConnId) {
        let Some(p) = self.players.get_mut(&conn) else {
            return;
        };
        let cx = p.x.floor().div_euclid(16.0) as i32;
        let cz = p.z.floor().div_euclid(16.0) as i32;
        if p.center == Some((cx, cz)) {
            return;
        }
        p.center = Some((cx, cz));

        let mut sends: Vec<(i32, Vec<u8>)> = Vec::new();
        let mut center = Vec::new();
        doppel_protocol::write_varint(&mut center, cx);
        doppel_protocol::write_varint(&mut center, cz);
        sends.push((0x60, center));

        let desired: std::collections::HashSet<(i32, i32)> = (-VIEW_RADIUS..=VIEW_RADIUS)
            .flat_map(move |dx| (-VIEW_RADIUS..=VIEW_RADIUS).map(move |dz| (cx + dx, cz + dz)))
            .collect();

        let leaving: Vec<(i32, i32)> = p.sent.difference(&desired).copied().collect();
        for (x, z) in leaving {
            let packed = ((x as i64 & 0x3ff_ffff) << 38) | ((z as i64 & 0x3ff_ffff) << 12);
            sends.push((0x26, packed.to_be_bytes().to_vec()));
            p.sent.remove(&(x, z));
        }

        let mut entering: Vec<(i32, i32)> = desired.difference(&p.sent).copied().collect();
        entering.sort_by_key(|(x, z)| (x - cx).abs() + (z - cz).abs());
        if entering.is_empty() {
            for (id, body) in sends {
                self.send(conn, id, &body);
            }
            return;
        }
        sends.push((0x0c, Vec::new()));

        // Load chunks (may lock the world), then emit everything.
        let (Some(world), Some(blobs)) = (self.world.clone(), self.blobs.clone()) else {
            for (id, body) in sends {
                self.send(conn, id, &body);
            }
            return;
        };
        let mut loaded: Vec<((i32, i32), Vec<u8>)> = Vec::new();
        for (x, z) in &entering {
            match self.load_chunk(&world, &blobs, *x, *z) {
                Ok(chunk) => loaded.push(((*x, *z), chunk.wire.encode())),
                Err(e) => eprintln!("[game] chunk ({x},{z}) skipped: {e:#}"),
            }
        }
        for ((x, z), body) in &loaded {
            sends.push((0x2e, body.clone()));
            if let Some(p) = self.players.get_mut(&conn) {
                p.sent.insert((*x, *z));
            }
            self.viewers.entry((*x, *z)).or_default().push(conn);
        }
        let mut finished = Vec::new();
        doppel_protocol::write_varint(&mut finished, loaded.len() as i32);
        sends.push((0x0b, finished));

        for (id, body) in sends {
            self.send(conn, id, &body);
        }
    }

    /// Loads a chunk into the cache (validated against the capture
    /// when vanilla sent one; reference-free conversion otherwise).
    fn load_chunk(
        &mut self,
        world: &std::sync::Arc<std::sync::Mutex<crate::WorldState>>,
        blobs: &std::sync::Arc<Blobs>,
        cx: i32,
        cz: i32,
    ) -> anyhow::Result<&CachedChunk> {
        if let std::collections::hash_map::Entry::Vacant(slot) = self.chunks.entry((cx, cz)) {
            let mut w = world.lock().unwrap_or_else(|e| e.into_inner());
            let Some(anvil) = w.dir.chunk(cx, cz)? else {
                anyhow::bail!("chunk not generated");
            };
            let reference = blobs.play.iter().find_map(|(id, body)| {
                if *id != 0x2e {
                    return None;
                }
                WireChunk::decode(body)
                    .ok()
                    .filter(|c| c.x == cx && c.z == cz)
            });
            let wire = match reference {
                Some(reference) => {
                    w.boot.learn(&reference, &anvil);
                    doppel_world::anvil_to_wire::convert(&anvil, &reference, &w.boot)?
                }
                None => doppel_world::anvil_to_wire::convert_uncaptured(&anvil, &w.boot)?,
            };
            drop(w);
            slot.insert(CachedChunk { wire, version: 0 });
        }
        Ok(self.chunks.get(&(cx, cz)).expect("present: occupied above"))
    }
}
