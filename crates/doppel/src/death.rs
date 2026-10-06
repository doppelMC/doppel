//! Player death and respawn: the health bar on the wire, the dying
//! sequence a killed client sees, and the respawn burst the
//! client_command gesture answers with. Frame order follows the vanilla
//! control transcript (the soak's SOAK_CONTROL=vanilla recording).

use crate::game::{ConnId, Game, Player};
use crate::inventory::{
    container_to_menu, encode_container_set_content, encode_container_set_slot, encode_game_event,
    encode_player_abilities, ItemStack, PACKET_CONTAINER_SET_CONTENT, PACKET_CONTAINER_SET_SLOT,
    PACKET_GAME_EVENT, PACKET_PLAYER_ABILITIES,
};
use crate::living::{encode_entity_event, EVENT_DEATH, MOB_TRACK_RANGE};
use doppel_protocol::write_varint;

/// `set_health`: registration order 107.
pub const PACKET_SET_HEALTH: i32 = 0x6a;
/// `player_combat_kill`: registration order 70.
pub const PACKET_PLAYER_COMBAT_KILL: i32 = 0x45;
/// `respawn`: registration order 85.
pub const PACKET_RESPAWN: i32 = 0x54;
/// `system_chat`: registration order 125.
pub const PACKET_SYSTEM_CHAT: i32 = 0x7c;

/// The food level every set_health carries; this build has no hunger
/// model.
pub const PLAYER_FOOD: i32 = 20;
/// The saturation level every set_health carries.
pub const PLAYER_SATURATION: f32 = 5.0;
/// A full health bar.
pub const PLAYER_MAX_HEALTH: f32 = 20.0;

/// The overworld dimension key, as the respawn spawn info carries it.
const DIMENSION_OVERWORLD: &str = "minecraft:overworld";
/// The overworld's sea level, as the respawn spawn info carries it.
const SEA_LEVEL: i32 = 63;
/// game_event: start loading chunks (the join burst's frame).
const GAME_EVENT_START_LOADING_CHUNKS: u8 = 0x0d;

/// What killed the player; each cause names the death message's killer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KillCause {
    /// Mob melee; the attacker's display name.
    Melee(&'static str),
    /// A projectile; the shooter's display name.
    Arrow(&'static str),
    /// An explosion; the source's display name.
    Explosion(&'static str),
    /// The kill command.
    KillCommand,
}

impl KillCause {
    /// The literal death message ("was slain by X" / "was killed"),
    /// vanilla's translatable form reduced to its rendered text.
    fn message(&self, name: &str) -> String {
        match self {
            KillCause::Melee(killer) | KillCause::Arrow(killer) | KillCause::Explosion(killer) => {
                format!("{name} was slain by {killer}")
            }
            KillCause::KillCommand => format!("{name} was killed"),
        }
    }
}

/// `set_health` body: f32 health, varint food, f32 saturation.
pub fn encode_set_health(health: f32) -> Vec<u8> {
    let mut body = Vec::with_capacity(9);
    body.extend_from_slice(&health.to_be_bytes());
    write_varint(&mut body, PLAYER_FOOD);
    body.extend_from_slice(&PLAYER_SATURATION.to_be_bytes());
    body
}

/// A literal-text chat component in the anonymous-root NBT form: one
/// TAG_String whose payload is the text (the command feedback shape).
fn encode_text_component(text: &str) -> Vec<u8> {
    let mut body = Vec::with_capacity(text.len() + 4);
    body.push(0x08);
    body.extend_from_slice(&(text.len() as u16).to_be_bytes());
    body.extend_from_slice(text.as_bytes());
    body
}

/// `system_chat` body: the component, then the overlay bool (false:
/// chat, not the action bar).
fn encode_system_chat(text: &str) -> Vec<u8> {
    let mut body = encode_text_component(text);
    body.push(0x00);
    body
}

/// `player_combat_kill` body: the player's entity id, then the death
/// message component (no overlay byte).
fn encode_player_combat_kill(player_id: i32, text: &str) -> Vec<u8> {
    let mut body = Vec::with_capacity(text.len() + 6);
    write_varint(&mut body, player_id);
    body.extend_from_slice(&encode_text_component(text));
    body
}

/// `respawn` body: the common spawn info plus the data-to-keep byte
/// (0: a death respawn keeps nothing).
fn encode_respawn(game_mode: i32, previous_game_mode: i32) -> Vec<u8> {
    let mut body = Vec::with_capacity(40);
    write_varint(&mut body, 0); // dimension type holder
    let dim = DIMENSION_OVERWORLD.as_bytes();
    write_varint(&mut body, dim.len() as i32);
    body.extend_from_slice(dim);
    body.extend_from_slice(&0i64.to_be_bytes()); // hashed seed
    write_varint(&mut body, game_mode);
    write_varint(&mut body, previous_game_mode);
    body.push(0x00); // not a debug world
    body.push(0x01); // a flat world
    body.push(0x00); // no death location
    write_varint(&mut body, 0); // portal cooldown
    write_varint(&mut body, SEA_LEVEL);
    body.push(0x00); // data to keep: nothing
    body
}

impl Game {
    /// One damage event against a player: the health bar moves, and a
    /// bar that reaches zero runs the death sequence. A dead player
    /// takes no further damage.
    pub(crate) fn damage_player(&mut self, conn: ConnId, damage: f32, cause: KillCause) {
        let outcome = {
            let Some(p) = self.players.get_mut(&conn) else {
                return;
            };
            if p.health <= 0.0 {
                return;
            }
            if p.health - damage <= 0.0 {
                // Fatal: kill_player owns the bar and the sequence.
                None
            } else {
                p.health -= damage;
                Some(p.health)
            }
        };
        match outcome {
            Some(_) => self.send_set_health(conn),
            None => self.kill_player(conn, cause),
        }
    }

    /// `set_health` for the player's current bar.
    pub(crate) fn send_set_health(&mut self, conn: ConnId) {
        let Some(p) = self.players.get(&conn) else {
            return;
        };
        let body = encode_set_health(p.health);
        self.send(conn, PACKET_SET_HEALTH, &body);
    }

    /// The dying sequence in the reference's order: the combat kill and
    /// its death message, the inventory consequence, the death entity
    /// event, set_health 0, and every mob target dropped. A player
    /// already dead stays dead: no second sequence (the reference's
    /// hurt path returns early on a dead entity).
    pub(crate) fn kill_player(&mut self, conn: ConnId, cause: KillCause) {
        let (name, entity_id, x, y, z) = {
            let Some(p) = self.players.get_mut(&conn) else {
                return;
            };
            if p.health <= 0.0 {
                return;
            }
            p.health = 0.0;
            (p.name.clone(), p.entity_id, p.x, p.y, p.z)
        };
        let text = cause.message(&name);
        self.send(
            conn,
            PACKET_PLAYER_COMBAT_KILL,
            &encode_player_combat_kill(entity_id, &text),
        );
        self.send(conn, PACKET_SYSTEM_CHAT, &encode_system_chat(&text));
        if !self.keep_inventory {
            self.drop_and_clear_inventory(conn, x, y, z);
        }
        self.send(
            conn,
            crate::living::PACKET_ENTITY_EVENT,
            &encode_entity_event(entity_id, EVENT_DEATH),
        );
        self.send(conn, PACKET_SET_HEALTH, &encode_set_health(0.0));
        for mob in self.mobs.mobs.iter_mut() {
            if mob.body.target == Some(conn) {
                mob.body.target = None;
            }
        }
    }

    /// The default inventory consequence: every non-empty slot spawns
    /// its stack as an item entity at the death position, then the slot
    /// clears (one container_set_slot per emptied menu slot). The drop
    /// scatter skips the reference's random velocities: drops land
    /// still, deterministic on the wire.
    fn drop_and_clear_inventory(&mut self, conn: ConnId, x: f64, y: f64, z: f64) {
        let held: Vec<(usize, ItemStack)> = {
            let Some(p) = self.players.get(&conn) else {
                return;
            };
            (0..crate::inventory::TOTAL_SLOTS)
                .filter_map(|slot| p.inv.inventory.get(slot).map(|stack| (slot, stack.clone())))
                .collect()
        };
        for (_, stack) in &held {
            self.spawn_item(x, y, z, (0.0, 0.0, 0.0), 0.0, stack.clone(), None);
        }
        let state_id = self
            .players
            .get_mut(&conn)
            .map(|p| p.inv.session.next_state_id());
        let Some(state_id) = state_id else {
            return;
        };
        for (slot, _) in &held {
            let Some(menu) = container_to_menu(*slot) else {
                continue;
            };
            let body = encode_container_set_slot(0, state_id, menu as i16, None);
            self.send(conn, PACKET_CONTAINER_SET_SLOT, &body);
        }
        if let Some(p) = self.players.get_mut(&conn) {
            for (slot, _) in &held {
                p.inv.inventory.set(*slot, None);
            }
            p.inv.pending_sync.clear();
        }
    }

    /// client_command varint 0 (PERFORM_RESPAWN): ignored while alive,
    /// like the reference's early return on health > 0.
    pub(crate) fn handle_client_command(&mut self, conn: ConnId, action: i32) {
        if action != 0 {
            return;
        }
        let dead = self.players.get(&conn).is_some_and(|p| p.health <= 0.0);
        if !dead {
            return;
        }
        self.respawn_player(conn);
    }

    /// The respawn burst in the reference's order: abilities, the
    /// respawn packet, the teleport-acknowledged position sync, the
    /// chunk-load game event and cache center, the inventory resend,
    /// set_health reset, then the chunk re-stream.
    fn respawn_player(&mut self, conn: ConnId) {
        let spawn = self.persistence.spawn;
        let (x, y, z) = (spawn.0 as f64 + 0.5, spawn.1 as f64, spawn.2 as f64 + 0.5);
        let (mode, flying, yaw, pitch, teleport_id) = {
            let Some(p) = self.players.get_mut(&conn) else {
                return;
            };
            p.health = PLAYER_MAX_HEALTH;
            p.x = x;
            p.y = y;
            p.z = z;
            let teleport_id = p.teleport_id;
            p.teleport_id += 1;
            (p.inv.mode, p.inv.flying, p.yaw, p.pitch, teleport_id)
        };
        let abilities = encode_player_abilities(mode.ability_flags(flying), 0.05, 0.1);
        self.send(conn, PACKET_PLAYER_ABILITIES, &abilities);
        self.send(conn, PACKET_RESPAWN, &encode_respawn(mode.id(), mode.id()));
        let sync = crate::game::position_sync_body(teleport_id, x, y, z, yaw, pitch);
        self.send(conn, crate::game::PACKET_PLAYER_POSITION, &sync);
        let event = encode_game_event(GAME_EVENT_START_LOADING_CHUNKS, 0.0);
        self.send(conn, PACKET_GAME_EVENT, &event);
        // The respawned client wiped its world: the mobs in range re-pair
        // ahead of the inventory and health reset, like the reference's
        // add_entity flood.
        let pairings: Vec<Vec<(i32, Vec<u8>)>> = self
            .mobs
            .mobs
            .iter()
            .filter(|m| {
                let (dx, dz) = (m.body.x - x, m.body.z - z);
                dx * dx + dz * dz <= MOB_TRACK_RANGE * MOB_TRACK_RANGE
            })
            .map(|m| m.pairing_frames())
            .collect();
        for frames in pairings {
            for (pid, body) in frames {
                self.send(conn, pid, &body);
            }
        }
        let (state_id, slots) = {
            let Some(p) = self.players.get_mut(&conn) else {
                return;
            };
            let state_id = p.inv.session.next_state_id();
            let slots = menu_slots(&p.inv.inventory);
            (state_id, slots)
        };
        let content = encode_container_set_content(0, state_id, &slots, None);
        self.send(conn, PACKET_CONTAINER_SET_CONTENT, &content);
        let health = encode_set_health(PLAYER_MAX_HEALTH);
        self.send(conn, PACKET_SET_HEALTH, &health);
        // The chunk stream restarts from the spawn view: no cached
        // center, no already-sent set, so the center packet, the batch,
        // and every entering chunk go out. Item entities re-pair after
        // the stream (their pairing rides the sent set).
        {
            let Some(p) = self.players.get_mut(&conn) else {
                return;
            };
            p.center = None;
            p.sent.clear();
        }
        self.stream_if_moved(conn);
        self.track_player_respawned(conn);
        self.track_player_view(conn);
    }
}

/// The inventory menu's slot list in menu order.
fn menu_slots(inventory: &crate::inventory::PlayerInventory) -> Vec<Option<ItemStack>> {
    (0..crate::inventory::INVENTORY_MENU_SIZE)
        .map(|menu| crate::inventory::menu_to_container(menu).and_then(|c| inventory.get(c)))
        .collect()
}

/// True when the player can act (movement, interaction, dig, placement).
pub(crate) fn player_alive(p: &Player) -> bool {
    p.health > 0.0
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Game, Inbound, Outbound};
    use doppel_world::WireChunk;
    use std::sync::mpsc;

    /// A grass-floored chunk (surface y=99) with one player.
    fn harness() -> (Game, mpsc::Receiver<Outbound>) {
        let (_tx, rx) = mpsc::channel::<Inbound>();
        let mut g = Game::new(rx, None, None);
        assert!(g.registry_for_test(), "pins/blocks.json not found");
        let grass = g.resolve_state("minecraft:grass_block").unwrap();
        let mut w = WireChunk {
            x: 0,
            z: 0,
            heightmaps: Vec::new(),
            sections: Vec::new(),
            block_entities: Vec::new(),
            light: Default::default(),
        };
        for sy in 0..24 {
            let block_states = if sy == 10 {
                let mut longs = vec![0u64; 256];
                for (l, slot) in longs.iter_mut().enumerate() {
                    for j in 0..16 {
                        let i = l * 16 + j;
                        let v: u64 = if (i >> 8) == 3 { 1 } else { 0 };
                        *slot |= v << (j * 4);
                    }
                }
                doppel_world::chunk_codec::Container::Palette {
                    bits: 4,
                    entries: vec![0, grass],
                    longs,
                }
            } else {
                doppel_world::chunk_codec::Container::Single(0)
            };
            w.sections.push(doppel_world::chunk_codec::WireSection {
                non_empty: if sy == 10 { 256 } else { 0 },
                fluid: 0,
                block_states,
                biomes: doppel_world::chunk_codec::Container::Single(0),
            });
        }
        g.seed_chunk_for_test(0, 0, w);
        let (tx_out, rx_out) = mpsc::channel::<Outbound>();
        g.join_viewer_for_test(0, &[(0, 0)], tx_out);
        (g, rx_out)
    }

    fn give_stone(g: &mut Game, count: i32, rx: &mpsc::Receiver<Outbound>) {
        g.handle(Inbound::Give {
            conn: 0,
            item: "minecraft:stone".into(),
            count,
        });
        g.flush_connections();
        while rx.try_recv().is_ok() {}
    }

    fn drain(rx: &mpsc::Receiver<Outbound>) -> Vec<(i32, Vec<u8>)> {
        let mut frames = Vec::new();
        while let Ok(out) = rx.try_recv() {
            if let Outbound::Frame { id, body } = out {
                frames.push((id, body));
            }
        }
        frames
    }

    fn flush_and_drain(g: &mut Game, rx: &mpsc::Receiver<Outbound>) -> Vec<(i32, Vec<u8>)> {
        g.flush_connections();
        drain(rx)
    }

    /// Damage moves the bar and puts set_health on the wire.
    #[test]
    fn damage_sends_set_health_bytes() {
        let (mut g, rx) = harness();
        g.damage_player(0, 3.0, KillCause::Melee("Zombie"));
        let frames = flush_and_drain(&mut g, &rx);
        assert_eq!(frames.len(), 1, "one frame for a non-fatal hit");
        let (id, body) = &frames[0];
        assert_eq!(*id, PACKET_SET_HEALTH);
        let mut want = Vec::new();
        want.extend_from_slice(&17.0f32.to_be_bytes());
        write_varint(&mut want, PLAYER_FOOD);
        want.extend_from_slice(&PLAYER_SATURATION.to_be_bytes());
        assert_eq!(body, &want);
    }

    /// A full bar of damage runs the dying sequence in the reference's
    /// frame order.
    #[test]
    fn fatal_damage_sends_dying_sequence() {
        let (mut g, rx) = harness();
        give_stone(&mut g, 16, &rx);
        g.damage_player(0, 20.0, KillCause::Melee("Zombie"));
        let frames = flush_and_drain(&mut g, &rx);
        let ids: Vec<i32> = frames.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            ids,
            vec![
                PACKET_PLAYER_COMBAT_KILL,
                PACKET_SYSTEM_CHAT,
                crate::game::entities::PACKET_ADD_ENTITY,
                crate::game::entities::PACKET_SET_ENTITY_DATA,
                PACKET_CONTAINER_SET_SLOT,
                crate::living::PACKET_ENTITY_EVENT,
                PACKET_SET_HEALTH,
            ],
            "dying sequence order"
        );
        let (_, kill) = &frames[0];
        assert_eq!(kill[0], 0x01, "player entity id varint");
        let text = String::from_utf8_lossy(&kill[3..kill.len()]).to_string();
        assert!(
            text.contains("was slain by Zombie"),
            "death message: {text}"
        );
        let (_, health) = frames.last().unwrap();
        assert_eq!(&health[..4], &0.0f32.to_be_bytes(), "health 0");
        // The inventory dropped and cleared.
        let got = g
            .player_inv_state_for_test(0)
            .expect("player")
            .inventory
            .get(0);
        assert!(got.is_none(), "inventory cleared");
        assert!(!g.survival.items.is_empty(), "ground drop spawned");
    }

    /// keepInventory=true keeps the inventory off the ground.
    #[test]
    fn keep_inventory_skips_drops() {
        let (mut g, rx) = harness();
        give_stone(&mut g, 16, &rx);
        g.keep_inventory = true;
        g.damage_player(0, 20.0, KillCause::KillCommand);
        let frames = flush_and_drain(&mut g, &rx);
        let ids: Vec<i32> = frames.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            ids,
            vec![
                PACKET_PLAYER_COMBAT_KILL,
                PACKET_SYSTEM_CHAT,
                crate::living::PACKET_ENTITY_EVENT,
                PACKET_SET_HEALTH,
            ],
            "no drop or clear frames"
        );
        let got = g
            .player_inv_state_for_test(0)
            .expect("player")
            .inventory
            .get(0);
        assert!(got.is_some(), "inventory kept");
    }

    /// The respawn gesture answers with the reference's frame order.
    #[test]
    fn respawn_sends_reference_order() {
        let (mut g, rx) = harness();
        g.damage_player(0, 20.0, KillCause::Melee("Zombie"));
        let _ = flush_and_drain(&mut g, &rx);
        g.handle(Inbound::ClientCommand { conn: 0, action: 0 });
        let frames = flush_and_drain(&mut g, &rx);
        let ids: Vec<i32> = frames.iter().map(|(id, _)| *id).collect();
        let head = [
            PACKET_PLAYER_ABILITIES,
            PACKET_RESPAWN,
            crate::game::PACKET_PLAYER_POSITION,
            PACKET_GAME_EVENT,
            PACKET_CONTAINER_SET_CONTENT,
            PACKET_SET_HEALTH,
        ];
        assert_eq!(&ids[..head.len()], &head[..], "respawn sequence order");
        // The chunk restart follows the health reset: the cache center
        // rides the stream, then the batch and the entering chunks (the
        // harness has no world store, so the stream carries the center
        // packet alone; the soak covers the chunk frames).
        let i_health = ids
            .iter()
            .position(|id| *id == PACKET_SET_HEALTH)
            .expect("set_health");
        let rest = &ids[i_health + 1..];
        assert_eq!(
            rest.first(),
            Some(&0x60),
            "set_chunk_cache_center after the health reset: {ids:?}"
        );
        let health = frames
            .iter()
            .find(|(id, _)| *id == PACKET_SET_HEALTH)
            .expect("set_health in respawn");
        assert_eq!(&health.1[..4], &20.0f32.to_be_bytes());
        // The respawn packet's last byte is the data-to-keep flag.
        let respawn = frames
            .iter()
            .find(|(id, _)| *id == PACKET_RESPAWN)
            .expect("respawn packet");
        assert_eq!(*respawn.1.last().unwrap(), 0x00, "data to keep: nothing");
    }

    /// An alive player's respawn gesture is ignored, like the reference.
    #[test]
    fn alive_client_command_is_silent() {
        let (mut g, rx) = harness();
        g.handle(Inbound::ClientCommand { conn: 0, action: 0 });
        let frames = flush_and_drain(&mut g, &rx);
        assert!(frames.is_empty(), "no frames for an alive respawn request");
    }

    /// A dead player takes no further damage and no second sequence.
    #[test]
    fn dead_player_takes_no_further_damage() {
        let (mut g, rx) = harness();
        g.damage_player(0, 20.0, KillCause::Melee("Zombie"));
        let _ = flush_and_drain(&mut g, &rx);
        g.damage_player(0, 3.0, KillCause::Melee("Zombie"));
        let frames = flush_and_drain(&mut g, &rx);
        assert!(frames.is_empty(), "no frames after death");
        // A kill command on the corpse re-runs nothing either.
        g.kill_player(0, KillCause::KillCommand);
        let frames = flush_and_drain(&mut g, &rx);
        assert!(frames.is_empty(), "no second dying sequence");
    }

    /// The kill command drives the same death and respawn cycle.
    #[test]
    fn kill_command_drives_the_cycle() {
        let (mut g, rx) = harness();
        g.handle(Inbound::Kill { conn: 0 });
        let frames = flush_and_drain(&mut g, &rx);
        let ids: Vec<i32> = frames.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            ids[..4],
            vec![
                PACKET_PLAYER_COMBAT_KILL,
                PACKET_SYSTEM_CHAT,
                crate::living::PACKET_ENTITY_EVENT,
                PACKET_SET_HEALTH,
            ][..],
            "kill command death sequence"
        );
        let (_, kill) = &frames[0];
        let text = String::from_utf8_lossy(&kill[3..]).to_string();
        assert!(text.contains("was killed"), "generic-kill message: {text}");
        assert_eq!(
            ids.last(),
            Some(&PACKET_SYSTEM_CHAT),
            "command feedback last"
        );
        g.handle(Inbound::ClientCommand { conn: 0, action: 0 });
        let frames = flush_and_drain(&mut g, &rx);
        assert!(
            frames.iter().any(|(id, _)| *id == PACKET_RESPAWN),
            "respawn after kill command"
        );
    }
}
