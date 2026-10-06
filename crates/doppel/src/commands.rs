//! The chat commands and their replies.

use crate::game::{ConnId, Game, GameMode, Inbound};

impl Game {
    /// Each arm applies its effect and answers with a feedback frame;
    /// the oracle harness paces its scripted volleys on those replies.
    pub(crate) fn apply_command(&mut self, event: Inbound) {
        match event {
            Inbound::Tp { conn, x, y, z } => {
                self.teleport_conn(conn, x, y, z);
                let name = self
                    .players
                    .get(&conn)
                    .map(|p| p.name.clone())
                    .unwrap_or_default();
                self.send_command_feedback(conn, &format!("Teleported {name} to {x}, {y}, {z}"));
            }
            Inbound::TpNamed {
                conn,
                name,
                x,
                y,
                z,
            } => {
                let target = self
                    .players
                    .iter()
                    .find(|(_, p)| p.name == name)
                    .map(|(&c, _)| c);
                if let Some(target) = target {
                    self.teleport_conn(target, x, y, z);
                }
                let name = self
                    .players
                    .iter()
                    .find(|(_, p)| p.name == name)
                    .map(|(_, p)| p.name.clone())
                    .unwrap_or_default();
                self.send_command_feedback(conn, &format!("Teleported {name} to {x}, {y}, {z}"));
            }
            Inbound::Setblock {
                conn,
                x,
                y,
                z,
                name,
            } => {
                self.setblock(conn, x, y, z, name);
                self.send_command_feedback(conn, &format!("Changed the block at {x}, {y}, {z}"));
            }
            Inbound::SetblockRel {
                conn,
                x,
                y,
                z,
                name,
            } => {
                let Some(p) = self.players.get(&conn) else {
                    return;
                };
                let (px, py, pz) = (p.x, p.y, p.z);
                let resolve = |axis: crate::events::SetblockAxis, pos: f64| match axis {
                    crate::events::SetblockAxis::Abs(v) => v,
                    crate::events::SetblockAxis::Rel(off) => {
                        (pos + off).floor().clamp(i32::MIN as f64, i32::MAX as f64) as i32
                    }
                };
                let rx = resolve(x, px);
                let ry = resolve(y, py);
                let rz = resolve(z, pz);
                self.setblock(conn, rx, ry, rz, name);
                self.send_command_feedback(conn, &format!("Changed the block at {rx}, {ry}, {rz}"));
            }
            Inbound::GameMode { conn, mode } => {
                // The reference's wire order, five frames: the mode
                // change itself syncs abilities and broadcasts the
                // UPDATE_GAME_MODE info update to every player
                // (changeGameModeForPlayer), then setGameMode sends
                // game_event followed by the abilities again, and the
                // command answers its feedback last. A same-mode switch
                // is refused before any of it: nothing reaches the wire,
                // not even the command feedback.
                let (flags, uuid) = {
                    let Some(p) = self.players.get_mut(&conn) else {
                        return;
                    };
                    if p.inv.mode == mode {
                        return;
                    }
                    p.inv.mode = mode;
                    p.inv.creative = mode.is_creative();
                    p.inv.mayfly = matches!(mode, GameMode::Creative | GameMode::Spectator);
                    if !p.inv.mayfly {
                        p.inv.flying = false;
                    } else if mode == GameMode::Spectator {
                        p.inv.flying = true;
                    }
                    (mode.ability_flags(p.inv.flying), p.uuid)
                };
                let abilities = crate::inventory::encode_player_abilities(flags, 0.05, 0.1);
                self.send(conn, crate::inventory::PACKET_PLAYER_ABILITIES, &abilities);
                let info = crate::inventory::encode_player_info_update_game_mode(&uuid, mode.id());
                let conns: Vec<ConnId> = self.players.keys().copied().collect();
                for c in conns {
                    self.send(c, crate::inventory::PACKET_PLAYER_INFO_UPDATE, &info);
                }
                self.send(
                    conn,
                    crate::inventory::PACKET_GAME_EVENT,
                    &crate::inventory::encode_game_event(
                        crate::inventory::GAME_EVENT_CHANGE_GAME_MODE,
                        mode.id() as f32,
                    ),
                );
                self.send(conn, crate::inventory::PACKET_PLAYER_ABILITIES, &abilities);
                self.send_command_feedback(
                    conn,
                    &format!("Set own game mode to {} Mode", mode.name()),
                );
            }
            Inbound::Give { conn, item, count } => {
                self.give_item(conn, &item, count);
                self.send_command_feedback(conn, &format!("Gave {count} {item}"));
            }
            Inbound::GameRule { conn, tick_speed } => {
                self.set_tick_speed(tick_speed);
                self.send_command_feedback(
                    conn,
                    &format!("Gamerule randomTickSpeed is now set to: {tick_speed}"),
                );
            }
            Inbound::TimeSet { conn, value } => {
                self.day_time = value;
                // The spawn cycle's darkness/burn timelines read the
                // spawning clock, so the command drives both.
                self.spawning.day_time = value.rem_euclid(24000) as u64;
                self.spawning.time_running = true;
                self.send_set_time();
                self.send_command_feedback(conn, &format!("Set the time to {value}"));
            }
            Inbound::GameRuleNoop { conn } => {
                self.send_command_feedback(conn, "Gamerule updated");
            }
            Inbound::TickStep { conn, steps } => {
                // Run the stepped ticks inline: commands queued behind this
                // event in the same channel batch land on later ticks,
                // matching vanilla's `tick step` barrier. The whole step
                // batch shares ONE connection flush: stepped ticks execute
                // inside a single server cycle, and the reference flushes
                // once after it.
                self.flush_suspended = true;
                for _ in 0..steps {
                    self.game_tick();
                }
                self.flush_suspended = false;
                self.send_command_feedback(conn, "");
            }
            Inbound::TickFreeze { conn, frozen } => {
                self.frozen = frozen;
                let what = if frozen { "frozen" } else { "resumed" };
                self.send_command_feedback(conn, &format!("Tick {what}"));
            }
            // --- death hooks (death.rs) ---
            Inbound::Kill { conn } => {
                self.kill_player(conn, crate::death::KillCause::KillCommand);
                let name = self
                    .players
                    .get(&conn)
                    .map(|p| p.name.clone())
                    .unwrap_or_default();
                self.send_command_feedback(conn, &format!("Killed {name}"));
            }
            Inbound::KeepInventory { conn, enabled } => {
                self.keep_inventory = enabled;
                self.send_command_feedback(
                    conn,
                    &format!("Gamerule keepInventory is now set to: {enabled}"),
                );
            }
            _ => {}
        }
    }
}
