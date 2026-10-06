// Minimal 26.3 (protocol 777) client on raw sockets. Packet ids and body
// shapes come from the repo pin, the oracle captures, strict_decode.rs,
// inventory.rs, dig.rs and placement.rs.
import net from 'node:net';
import zlib from 'node:zlib';
import crypto from 'node:crypto';

export const PROTOCOL = 777;

// Clientbound id tables, registration order per phase (strict_decode.rs).
export const LOGIN_IDS = {
  0x00: 'login_disconnect', 0x01: 'hello', 0x02: 'login_finished',
  0x03: 'login_compression', 0x04: 'custom_query', 0x05: 'cookie_request',
};
export const CONFIG_IDS = {
  0x00: 'cookie_request', 0x01: 'custom_payload', 0x02: 'disconnect',
  0x03: 'finish_configuration', 0x04: 'keep_alive', 0x05: 'ping',
  0x06: 'reset_chat', 0x07: 'registry_data', 0x08: 'resource_pack_pop',
  0x09: 'resource_pack_push', 0x0a: 'post_effects', 0x0b: 'store_cookie',
  0x0c: 'transfer', 0x0d: 'update_enabled_features', 0x0e: 'update_tags',
  0x0f: 'select_known_packs', 0x10: 'custom_report_details',
  0x11: 'server_links', 0x12: 'clear_dialog', 0x13: 'show_dialog',
  0x14: 'code_of_conduct',
};
export const PLAY_IDS = {
  0x00: 'bundle_delimiter', 0x01: 'add_entity', 0x02: 'animate',
  0x03: 'award_stats', 0x04: 'block_changed_ack', 0x05: 'block_destruction',
  0x06: 'block_entity_data', 0x07: 'block_event', 0x08: 'block_update',
  0x09: 'boss_event', 0x0a: 'change_difficulty', 0x0b: 'chunk_batch_finished',
  0x0c: 'chunk_batch_start', 0x0d: 'chunks_biomes', 0x0e: 'clear_titles',
  0x0f: 'command_suggestions', 0x10: 'commands', 0x11: 'container_close',
  0x12: 'container_set_content', 0x13: 'container_set_data',
  0x14: 'container_set_slot', 0x15: 'cookie_request', 0x16: 'cooldown',
  0x17: 'custom_chat_completions', 0x18: 'custom_payload',
  0x19: 'damage_event', 0x1a: 'debug_block_value', 0x1b: 'debug_chunk_value',
  0x1c: 'debug_entity_value', 0x1d: 'debug_event', 0x1e: 'debug_sample',
  0x1f: 'delete_chat', 0x20: 'disconnect', 0x21: 'disguised_chat',
  0x22: 'entity_event', 0x23: 'entity_position_sync', 0x24: 'explode',
  0x25: 'add_transient_block', 0x26: 'forget_level_chunk',
  0x27: 'game_event', 0x28: 'game_rule_values',
  0x29: 'game_test_highlight_pos', 0x2a: 'mount_screen_open',
  0x2b: 'hurt_animation', 0x2c: 'initialize_border', 0x2d: 'keep_alive',
  0x2e: 'level_chunk_with_light', 0x2f: 'level_event',
  0x30: 'level_particles', 0x31: 'light_update', 0x32: 'login',
  0x33: 'low_disk_space_warning', 0x34: 'map_item_data',
  0x35: 'merchant_offers', 0x36: 'move_entity_pos',
  0x37: 'move_entity_pos_rot', 0x38: 'move_minecart_along_track',
  0x39: 'move_entity_rot', 0x3a: 'move_vehicle', 0x3b: 'open_book',
  0x3c: 'open_screen', 0x3d: 'open_sign_editor', 0x3e: 'ping',
  0x3f: 'pong_response', 0x40: 'place_ghost_recipe',
  0x41: 'player_abilities', 0x42: 'player_chat', 0x43: 'player_combat_end',
  0x44: 'player_combat_enter', 0x45: 'player_combat_kill',
  0x46: 'player_info_remove', 0x47: 'player_info_update',
  0x48: 'player_look_at', 0x49: 'player_position', 0x4a: 'player_rotation',
  0x4b: 'recipe_book_add', 0x4c: 'recipe_book_remove',
  0x4d: 'recipe_book_settings', 0x4e: 'remove_entities',
  0x4f: 'remove_mob_effect', 0x50: 'reset_score', 0x51: 'resource_pack_pop',
  0x52: 'resource_pack_push', 0x53: 'post_effects', 0x54: 'respawn',
  0x55: 'rotate_head', 0x56: 'section_blocks_update',
  0x57: 'select_advancements_tab', 0x58: 'server_data',
  0x59: 'set_action_bar_text', 0x5a: 'set_border_center',
  0x5b: 'set_border_lerp_size', 0x5c: 'set_border_size',
  0x5d: 'set_border_warning_delay', 0x5e: 'set_border_warning_distance',
  0x5f: 'set_camera', 0x60: 'set_chunk_cache_center',
  0x61: 'set_chunk_cache_radius', 0x62: 'set_cursor_item',
  0x63: 'set_default_spawn_position', 0x64: 'set_display_objective',
  0x65: 'set_entity_data', 0x66: 'set_entity_link',
  0x67: 'set_entity_motion', 0x68: 'set_equipment',
  0x69: 'set_experience', 0x6a: 'set_health', 0x6b: 'set_held_slot',
  0x6c: 'set_objective', 0x6d: 'set_passengers',
  0x6e: 'set_player_inventory', 0x6f: 'set_player_team',
  0x70: 'set_score', 0x71: 'set_simulation_distance',
  0x72: 'set_subtitle_text', 0x73: 'set_time', 0x74: 'set_title_text',
  0x75: 'set_titles_animation', 0x76: 'sound_entity', 0x77: 'sound',
  0x78: 'start_configuration', 0x79: 'stop_sound', 0x7a: 'store_cookie',
  0x7b: 'swing_animation', 0x7c: 'system_chat', 0x7d: 'tab_list',
  0x7e: 'tag_query', 0x7f: 'take_item_entity', 0x80: 'teleport_entity',
  0x81: 'test_instance_block_status', 0x82: 'ticking_state',
  0x83: 'ticking_step', 0x84: 'transfer', 0x85: 'update_advancements',
  0x86: 'update_attributes', 0x87: 'update_mob_effect',
  0x88: 'update_recipes', 0x89: 'update_tags', 0x8a: 'projectile_power',
  0x8b: 'custom_report_details', 0x8c: 'server_links', 0x8d: 'waypoint',
  0x8e: 'clear_dialog', 0x8f: 'show_dialog',
};

// Serverbound play ids (repo pins; client_command 0x0c per the
// registration order fact noted in .harness/progress.md).
export const SB = {
  accept_teleportation: 0x00,
  chat_command: 0x07,
  chunk_batch_received: 0x0b,
  client_command: 0x0c,
  client_tick_end: 0x0d,
  container_click: 0x12,
  container_close: 0x13,
  keep_alive: 0x1c,
  move_player_pos: 0x1e,
  move_player_pos_rot: 0x1f,
  move_player_rot: 0x20,
  player_abilities: 0x28,
  player_action: 0x29,
  player_loaded: 0x2c,
  punch: 0x2e,
  set_carried_item: 0x36,
  set_creative_mode_slot: 0x39,
  use_item_on: 0x42,
};

// player_action ordinals (dig.rs).
export const ACTION = {
  start_destroy: 0, change_direction: 1, abort_destroy: 2, stop_destroy: 3,
  drop_all: 4, drop_item: 5, release_use: 6, swap_offhand: 7, stab: 8,
};

export class ParseError extends Error {}

export function writeVarint(out, v) {
  let u = v >>> 0; // two's complement 32-bit form
  while (u > 0x7f) {
    out.push((u & 0x7f) | 0x80);
    u >>>= 7;
  }
  out.push(u);
}

export function writeString(out, s) {
  const bytes = Buffer.from(s, 'utf8');
  writeVarint(out, bytes.length);
  out.push(...bytes);
}

export function readVarint(buf, off) {
  let value = 0;
  let shift = 0;
  for (let i = 0; i < 5; i++) {
    if (off + i >= buf.length) {
      throw new ParseError(`varint runs past the frame at ${off + i}`);
    }
    const b = buf[off + i];
    value |= (b & 0x7f) << shift;
    if ((b & 0x80) === 0) {
      return [value | 0, off + i + 1];
    }
    shift += 7;
  }
  throw new ParseError(`varint longer than 5 bytes at ${off}`);
}

// The offline-mode profile uuid: UUIDv3 (MD5) of "OfflinePlayer:<name>",
// matching blobs.rs.
export function offlineUuid(name) {
  const hash = crypto.createHash('md5')
    .update(`OfflinePlayer:${name}`).digest();
  hash[6] = (hash[6] & 0x0f) | 0x30;
  hash[8] = (hash[8] & 0x3f) | 0x80;
  return hash;
}

export function packBlockPos(x, y, z) {
  const mx = BigInt.asUintN(26, BigInt(x));
  const mz = BigInt.asUintN(26, BigInt(z));
  const my = BigInt.asUintN(12, BigInt(y));
  return BigInt.asIntN(64, (mx << 38n) | (mz << 12n) | my);
}

/**
 * One client connection through login, configuration and play. Emits
 * 'packet' (id, name, body), 'kicked' (raw reason head), 'end',
 * 'error'. Any framing, decompression, length or unknown-id failure
 * raises a ParseError via 'error' and closes.
 */
export class Client {
  constructor(host, port, username) {
    this.host = host;
    this.port = port;
    this.username = username;
    this.state = 'login';
    this.compression = -1;
    this.buffer = Buffer.alloc(0);
    this.handlers = {};
    for (const ev of ['packet', 'kicked', 'end', 'error', 'ready']) {
      this.handlers[ev] = [];
    }
    this.sock = new net.Socket();
    this.sock.setNoDelay(true);
    this.sock.on('data', (chunk) => this.onData(chunk));
    this.sock.on('close', () => this.emitEvent('end', {}));
    this.sock.on('error', (err) => this.emitEvent('error', err));
  }

  on(ev, fn) {
    this.handlers[ev].push(fn);
  }

  get destroyed() {
    return this.sock.destroyed;
  }

  destroy() {
    this.sock.destroy();
  }

  emitEvent(ev, arg) {
    for (const fn of this.handlers[ev] || []) {
      fn(arg);
    }
  }

  fail(err) {
    this.emitEvent('error', err);
    this.sock.destroy();
  }

  onData(chunk) {
    this.buffer = this.buffer.length === 0 ? chunk : Buffer.concat([this.buffer, chunk]);
    while (true) {
      let len;
      let next;
      try {
        [len, next] = readVarint(this.buffer, 0);
      } catch (e) {
        if (this.buffer.length < 5) {
          return; // partial length prefix; wait for more
        }
        this.fail(new ParseError(`frame length: ${e.message}`));
        return;
      }
      if (len < 0 || len > 32 * 1024 * 1024) {
        this.fail(new ParseError(`frame length ${len} out of range`));
        return;
      }
      if (this.buffer.length < next + len) {
        return; // partial frame; wait for more
      }
      let payload = this.buffer.subarray(next, next + len);
      this.buffer = this.buffer.subarray(next + len);
      if (this.compression >= 0) {
        let dataLen;
        let p;
        try {
          [dataLen, p] = readVarint(payload, 0);
        } catch (e) {
          this.fail(new ParseError(`compressed length: ${e.message}`));
          return;
        }
        if (dataLen < 0 || dataLen > 32 * 1024 * 1024) {
          this.fail(new ParseError(`compressed data length ${dataLen} out of range`));
          return;
        }
        const rest = payload.subarray(p);
        if (dataLen === 0) {
          if (rest.length >= this.compression) {
            this.fail(new ParseError(
              `uncompressed frame of ${rest.length} bytes at threshold ${this.compression}`));
            return;
          }
          payload = rest;
        } else {
          try {
            payload = zlib.inflateSync(rest);
          } catch (e) {
            this.fail(new ParseError(`zlib inflate: ${e.message}`));
            return;
          }
          if (payload.length !== dataLen) {
            this.fail(new ParseError(`inflated ${payload.length} bytes, frame says ${dataLen}`));
            return;
          }
        }
      }
      let id;
      let bodyOff;
      try {
        [id, bodyOff] = readVarint(payload, 0);
      } catch (e) {
        this.fail(new ParseError(`packet id: ${e.message}`));
        return;
      }
      const body = payload.subarray(bodyOff);
      try {
        this.dispatch(id, body);
      } catch (e) {
        if (e instanceof ParseError) {
          this.fail(e);
          return;
        }
        throw e;
      }
    }
  }

  nameFor(id) {
    const table = this.state === 'login' ? LOGIN_IDS
      : this.state === 'config' ? CONFIG_IDS : PLAY_IDS;
    return table[id];
  }

  dispatch(id, body) {
    const name = this.nameFor(id);
    if (name === undefined) {
      throw new ParseError(`unknown ${this.state}-state packet id 0x${id.toString(16)}`);
    }
    if (this.state === 'login') {
      if (id === 0x03) {
        const [threshold] = readVarint(body, 0);
        this.compression = threshold;
      } else if (id === 0x02) {
        // The ack may land after set_compression switched framing.
        this.writePacket(0x03, Buffer.alloc(0)); // login_acknowledged
        this.state = 'config';
        this.writePacket(0x00, clientInformationBody()); // client information
      } else if (id === 0x00) {
        this.emitEvent('kicked', { head: body.subarray(0, 256) });
      }
    } else if (this.state === 'config') {
      if (id === 0x0f) {
        this.writePacket(0x07, Buffer.from([0x00])); // known packs: empty list
      } else if (id === 0x03 && body.length === 0) {
        this.writePacket(0x03, Buffer.alloc(0)); // finish_configuration ack
        this.state = 'play';
        this.emitEvent('ready', {});
      } else if (id === 0x02) {
        this.emitEvent('kicked', { head: body.subarray(0, 256) });
      }
    } else {
      if (id === 0x2d) {
        this.writePacket(SB.keep_alive, body); // echo the i64 challenge
      } else if (id === 0x0b) {
        const feedback = Buffer.alloc(4);
        feedback.writeFloatBE(64.0, 0);
        this.writePacket(SB.chunk_batch_received, feedback);
      } else if (id === 0x20) {
        this.emitEvent('kicked', { head: body.subarray(0, 256) });
      }
    }
    this.emitEvent('packet', { id, name, body });
  }

  writePacketRaw(id, body) {
    const head = [];
    writeVarint(head, id);
    const frame = Buffer.concat([Buffer.from(head), body]);
    this.writeFrame(frame);
  }

  writePacket(id, body) {
    const head = [];
    writeVarint(head, id);
    const frame = Buffer.concat([Buffer.from(head), body]);
    if (this.compression < 0) {
      this.writeFrame(frame);
      return;
    }
    if (frame.length >= this.compression) {
      const deflated = zlib.deflateSync(frame, { level: 6 });
      const sizePrefix = [];
      writeVarint(sizePrefix, deflated.length);
      const lenPrefix = [];
      writeVarint(lenPrefix, sizePrefix.length + deflated.length);
      this.sock.write(Buffer.concat([Buffer.from(lenPrefix), Buffer.from(sizePrefix), deflated]));
    } else {
      const sizePrefix = Buffer.from([0x00]);
      const lenPrefix = [];
      writeVarint(lenPrefix, 1 + frame.length);
      this.sock.write(Buffer.concat([Buffer.from(lenPrefix), sizePrefix, frame]));
    }
  }

  writeFrame(frame) {
    const prefix = [];
    writeVarint(prefix, frame.length);
    this.sock.write(Buffer.concat([Buffer.from(prefix), frame]));
  }

  connectClient() {
    this.sock.connect(this.port, this.host, () => {
      const hs = [];
      writeVarint(hs, PROTOCOL);
      writeString(hs, this.host);
      const portBuf = Buffer.alloc(2);
      portBuf.writeUInt16BE(this.port, 0);
      hs.push(...portBuf);
      writeVarint(hs, 2); // next state: login
      this.writePacketRaw(0x00, Buffer.from(hs));
      const ls = [];
      writeString(ls, this.username);
      const uuid = offlineUuid(this.username);
      ls.push(...uuid);
      this.writePacketRaw(0x00, Buffer.from(ls));
    });
    return this;
  }
}

// Serverbound configuration Client Information (bot.rs field order).
export function clientInformationBody() {
  const b = [];
  writeString(b, 'en_US');
  b.push(8); // view distance
  writeVarint(b, 0); // chat visibility: full
  b.push(0x01); // chat colors
  b.push(0x7f); // model customisation
  writeVarint(b, 1); // main hand: right
  b.push(0x00); // text filtering
  b.push(0x01); // allow server listings
  writeVarint(b, 0); // particle status
  return Buffer.from(b);
}

export function chatCommandBody(cmd) {
  const b = [];
  writeString(b, cmd.replace(/^\//, ''));
  return Buffer.from(b);
}

// use_item_on body against the top face (placement.rs / bot.rs layout).
export function useItemOnBody(x, y, z, sequence, face = 1, hand = 0) {
  const parts = [];
  writeVarint(parts, hand);
  const pos = packBlockPos(x, y, z);
  const posBuf = Buffer.alloc(8);
  posBuf.writeBigInt64BE(BigInt.asIntN(64, pos), 0);
  parts.push(...posBuf);
  writeVarint(parts, face);
  const cursor = Buffer.alloc(12);
  cursor.writeFloatBE(0.5, 0);
  cursor.writeFloatBE(face === 1 ? 1.0 : 0.5, 4);
  cursor.writeFloatBE(0.5, 8);
  parts.push(...cursor);
  parts.push(0); // inside block: false
  parts.push(0); // world border: false
  writeVarint(parts, sequence);
  return Buffer.from(parts);
}

export function playerActionBody(action, x, y, z, direction, sequence) {
  const b = [];
  writeVarint(b, action);
  const posBuf = Buffer.alloc(8);
  posBuf.writeBigInt64BE(BigInt.asIntN(64, packBlockPos(x, y, z)), 0);
  b.push(...posBuf);
  writeVarint(b, direction);
  writeVarint(b, sequence);
  return Buffer.from(b);
}

export function acceptTeleportBody(id, x, y, z, yaw, pitch) {
  const b = [];
  writeVarint(b, id);
  const nums = Buffer.alloc(3 * 8 + 2 * 4);
  nums.writeDoubleBE(x, 0);
  nums.writeDoubleBE(y, 8);
  nums.writeDoubleBE(z, 16);
  nums.writeFloatBE(yaw, 24);
  nums.writeFloatBE(pitch, 28);
  b.push(...nums);
  return Buffer.from(b);
}

export function moveBody(x, y, z, yaw, pitch, flags) {
  if (yaw === null) {
    const nums = Buffer.alloc(24 + 1);
    nums.writeDoubleBE(x, 0);
    nums.writeDoubleBE(y, 8);
    nums.writeDoubleBE(z, 16);
    nums[24] = flags;
    return nums;
  }
  const nums = Buffer.alloc(3 * 8 + 2 * 4 + 1);
  nums.writeDoubleBE(x, 0);
  nums.writeDoubleBE(y, 8);
  nums.writeDoubleBE(z, 16);
  nums.writeFloatBE(yaw, 24);
  nums.writeFloatBE(pitch, 28);
  nums[32] = flags;
  return nums;
}

export function rotBody(yaw, pitch, flags) {
  const b = Buffer.alloc(8 + 1);
  b.writeFloatBE(yaw, 0);
  b.writeFloatBE(pitch, 4);
  b[8] = flags;
  return b;
}

export function setCarriedItemBody(slot) {
  const b = Buffer.alloc(2);
  b.writeInt16BE(slot, 0);
  return b;
}

export function containerCloseBody(containerId) {
  const b = [];
  writeVarint(b, containerId);
  return Buffer.from(b);
}

// container_click: pickup click on one slot, no changes, empty cursor
// (the proven strict-decode session frame).
export function containerClickBody(containerId, stateId, slot) {
  const b = [];
  writeVarint(b, containerId);
  writeVarint(b, stateId);
  const slotBuf = Buffer.alloc(2);
  slotBuf.writeInt16BE(slot, 0);
  b.push(...slotBuf);
  b.push(0); // button
  writeVarint(b, 0); // click kind: pickup
  writeVarint(b, 0); // changed slots
  b.push(0); // carried: absent
  return Buffer.from(b);
}

export function playerAbilitiesBody(flying) {
  return Buffer.from([flying ? 0x02 : 0x00]);
}

// set_creative_mode_slot writing a bare stack to a menu slot: i16 slot,
// then the untrusted stack codec (count varint, 0 = empty; item varint;
// patch +count/-count). No presence byte. Menu numbering: 0 is the
// crafting result, 36+ the hotbar.
export function creativeSetBody(menuSlot, itemId, count) {
  const b = [];
  const slot = Buffer.alloc(2);
  slot.writeInt16BE(menuSlot, 0);
  b.push(...slot);
  writeVarint(b, count);
  if (count > 0) {
    writeVarint(b, itemId);
    writeVarint(b, 0); // patch: no added components
    writeVarint(b, 0); // patch: no removed components
  }
  return Buffer.from(b);
}

export function clientCommandBody(action) {
  const b = [];
  writeVarint(b, action);
  return Buffer.from(b);
}

// Reads (f32 health, varint food, f32 saturation) out of set_health.
export function parseSetHealth(body) {
  if (body.length < 4) {
    throw new ParseError(`set_health body ${body.length} bytes`);
  }
  return body.readFloatBE(0);
}
