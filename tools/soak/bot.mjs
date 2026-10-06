// The soak bot: one phased player session over proto.mjs. Every phase
// drives the server the way a real client drives it (raw interaction
// packets, not command shortcuts) wherever the server implements the
// player path.
import {
  Client, SB, ACTION, ParseError, chatCommandBody, useItemOnBody,
  playerActionBody, acceptTeleportBody, moveBody, rotBody, setCarriedItemBody,
  containerClickBody, containerCloseBody, playerAbilitiesBody,
  clientCommandBody, creativeSetBody, parseSetHealth, readVarint,
} from './proto.mjs';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const now = () => Date.now();

// Walk speed in blocks per 50 ms step (4 m/s, under vanilla's budget).
const WALK_STEP = 0.2;
const JUMP_VY = 0.42;
const V_DRAG = 0.98;
const GRAVITY = 0.08;
// Zombie melee is 3.0 per hit and a full bar is 20.0, so seven damage
// events targeting us is the downed threshold the server tracks.
const DOWNED_HITS = 7;

class Bot {
  constructor(opts) {
    this.host = opts.host;
    this.port = opts.port;
    this.durationMs = opts.durationMs;
    this.log = opts.log || (() => {});
    this.items = opts.items || {};
    this.stats = opts.stats || {};
    this.stats.lastPacketAt = now();
    this.stats.phase = 'connect';
    this.stats.packetsSeen = 0;
    this.stats.hits = 0;
    this.stats.health = null;
    this.stats.deaths = 0;

    this.phaseList = [];
    this.lastPackets = [];
    this.error = null;
    this.red = false;
    this.ended = false;

    this.pos = null; // { x, y, z }
    this.yaw = 0;
    this.pitch = 0;
    this.onGround = true;
    this.vy = 0;
    this.jumping = false;
    this.entityId = null;
    this.gamemode = 'survival';
    this.dead = false;
    this.moveTarget = null;
    this.lookSweep = null;
    this.chatWaiters = [];
    this.openContainer = null;
    this.stateId = 0;
    this.sequence = 100;
    this.flying = false;
    this.residuals = [];

    this.client = new Client(this.host, this.port, 'Doppel');
  }

  setPhase(name) {
    this.stats.phase = name;
    this.phaseList.push(`${name}@${((now() - this.t0) / 1000).toFixed(1)}s`);
    this.log(`phase: ${name}`);
  }

  recordPacket(name) {
    this.lastPackets.push(name);
    if (this.lastPackets.length > 20) {
      this.lastPackets.shift();
    }
    this.stats.packetsSeen += 1;
    this.stats.lastPacketAt = now();
  }

  send(id, body) {
    if (!this.client.destroyed) {
      this.client.writePacket(id, body);
    }
  }

  cmd(text) {
    // One chat command, paced on the next system_chat reply (3 s cap).
    this.send(SB.chat_command, chatCommandBody(text));
    return new Promise((resolve) => {
      const waiter = { resolve, timer: null };
      waiter.timer = setTimeout(() => {
        this.chatWaiters = this.chatWaiters.filter((w) => w !== waiter);
        resolve(false);
      }, 3000);
      this.chatWaiters.push(waiter);
    });
  }

  stopMovement() {
    this.moveTarget = null;
    this.lookSweep = null;
    this.jumping = false;
    this.vy = 0;
  }

  async walkTo(x, z, { jump = false } = {}) {
    if (!this.pos) {
      return;
    }
    this.moveTarget = { x, z, jump };
    while (this.moveTarget && !this.red && !this.done) {
      await sleep(50);
    }
    this.moveTarget = null;
  }

  lookAround() {
    // One sweep: yaw 360 over twelve rot frames, pitch nodding. The
    // safety timer keeps the phase from hanging if the tick loop stops.
    return new Promise((resolve) => {
      let step = 0;
      const finish = () => {
        this.lookSweep = null;
        clearTimeout(safety);
        resolve();
      };
      const safety = setTimeout(finish, 5000);
      this.lookSweep = () => {
        if (step >= 12 || this.red) {
          finish();
          return null;
        }
        this.yaw = (step * 30) % 360;
        this.pitch = step % 2 === 0 ? 10 : -10;
        step += 1;
        return null;
      };
    });
  }

  jump() {
    if (this.onGround && !this.jumping) {
      this.vy = JUMP_VY;
      this.jumping = true;
      this.onGround = false;
    }
  }

  tick() {
    // Nothing moves before the play state; a tick_end sent in login or
    // configuration is an unknown packet there.
    if (this.client.state !== 'play') {
      return;
    }
    // Every client tick closes with client_tick_end: vanilla allows one
    // position packet per tick and resets that gate only on this packet.
    if (this.pos && !this.dead) {
      let moved = false;
      let dy = 0;
      if (this.jumping || !this.onGround) {
        dy = this.vy;
        if (!this.flying) {
          this.vy = this.vy * V_DRAG - GRAVITY;
          if (Math.abs(this.vy) < 0.003) {
            this.vy = 0;
          }
        }
        this.pos.y += dy;
        moved = true;
        if (this.flying && Math.abs(dy) < 0.01 && !this.jumping) {
          this.onGround = false;
        }
        if (this.pos.y <= this.groundY) {
          this.pos.y = this.groundY;
          this.onGround = true;
          this.jumping = false;
          this.vy = 0;
        }
      }
      if (this.moveTarget) {
        const t = this.moveTarget;
        const dx = t.x - this.pos.x;
        const dz = t.z - this.pos.z;
        const dist = Math.hypot(dx, dz);
        if (dist <= WALK_STEP) {
          this.pos.x = t.x;
          this.pos.z = t.z;
          this.moveTarget = null;
          moved = true;
        } else {
          this.pos.x += (dx / dist) * WALK_STEP;
          this.pos.z += (dz / dist) * WALK_STEP;
          this.yaw = (Math.atan2(-(dz), dx) * 180) / Math.PI;
          moved = true;
        }
        if (t.jump && this.onGround) {
          this.jump();
        }
      }
      if (this.lookSweep) {
        this.lookSweep();
        this.send(SB.move_player_rot, rotBody(this.yaw, this.pitch, groundFlag(this)));
      } else if (moved) {
        this.send(
          SB.move_player_pos_rot,
          moveBody(this.pos.x, this.pos.y, this.pos.z, this.yaw, this.pitch, groundFlag(this)),
        );
      }
    }
    this.send(SB.client_tick_end, Buffer.alloc(0));
  }

  fail(msg) {
    if (this.red || this.teardown) {
      return;
    }
    this.red = true;
    this.error = msg;
    this.log(`red: ${msg}`);
  }

  async run() {
    this.t0 = now();
    const c = this.client;
    c.on('packet', ({ id, name, body }) => this.onPacket(id, name, body));
    c.on('kicked', ({ head }) => {
      this.fail(`kicked by server: ${componentText(head)}`);
    });
    c.on('end', () => {
      this.ended = true;
      if (!this.red) {
        this.fail('connection ended');
      }
      this.finish();
    });
    c.on('error', (err) => {
      this.ended = true;
      this.fail(err instanceof ParseError ? `parse error: ${err.message}` : `error: ${err.message}`);
      this.finish();
    });

    // One movement batch plus client_tick_end per loop; 60 ms keeps the
    // cadence at or below the 20 Hz client rate.
    const tickTimer = setInterval(() => this.tick(), 60);
    try {
      await this.session();
    } catch (e) {
      this.fail(`session error: ${e.message}`);
    } finally {
      clearInterval(tickTimer);
    }
    if (!this.red && now() - this.t0 < this.durationMs) {
      this.fail('session returned before duration');
    }
    // Post-resolution socket close is teardown, not a session red.
    this.teardown = true;
    try {
      c.destroy();
    } catch {}
    return this.result();
  }

  result() {
    return {
      status: this.red ? 'red' : 'green',
      phase: this.stats.phase,
      error: this.error,
      phases: this.phaseList,
      lastPackets: this.lastPackets.slice(),
      packetsSeen: this.stats.packetsSeen,
      hits: this.stats.hits,
      health: this.stats.health,
      deaths: this.stats.deaths,
      position: this.pos,
      residuals: this.residuals.slice(),
    };
  }

  finish() {
    if (this.done) {
      return;
    }
    this.done = true;
    for (const w of this.waiters || []) {
      w.stop();
    }
    this.waiters = [];
    this.moveTarget = null;
    this.lookSweep = null;
    if (this.resolveSession) {
      this.resolveSession();
    }
  }

  onPacket(id, name, body) {
    this.recordPacket(name);
    if (name === 'system_chat') {
      for (const w of this.chatWaiters.splice(0)) {
        clearTimeout(w.timer);
        w.resolve(true);
      }
      return;
    }
    if (name === 'player_position' && body.length >= 4) {
      const [teleportId, off] = readVarint(body, 0);
      // id, pos x3, delta x3, yaw, pitch: the rotation sits behind the
      // three delta doubles.
      if (body.length >= off + 52) {
        const x = body.readDoubleBE(off);
        const y = body.readDoubleBE(off + 8);
        const z = body.readDoubleBE(off + 16);
        const yaw = body.readFloatBE(off + 48);
        const pitch = body.readFloatBE(off + 52);
        this.pos = { x, y, z };
        this.groundY = y;
        this.yaw = yaw;
        this.pitch = pitch;
        this.send(SB.accept_teleportation, acceptTeleportBody(teleportId, x, y, z, yaw, pitch));
      }
      return;
    }
    if (name === 'login' && this.entityId === null && body.length >= 4) {
      this.entityId = body.readInt32BE(0);
      return;
    }
    if (name === 'set_health' && body.length >= 4) {
      const health = parseSetHealth(body);
      this.stats.health = health;
      if (health <= 0 && !this.dead) {
        this.dead = true;
        this.stats.deaths += 1;
        this.stopMovement();
        this.log('death screen: health 0');
      }
      return;
    }
    if (name === 'damage_event' && body.length >= 1) {
      const [target] = readVarint(body, 0);
      if (this.entityId !== null && target === this.entityId) {
        this.stats.hits += 1;
      }
      return;
    }
    if (name === 'open_screen' && body.length >= 1) {
      const [containerId] = readVarint(body, 0);
      this.openContainer = containerId;
      return;
    }
    if (name === 'container_set_content' && body.length >= 2) {
      let off = 0;
      let containerId = 0;
      let stateId = 0;
      let count = 0;
      [containerId, off] = readVarint(body, off);
      [stateId, off] = readVarint(body, off);
      [count, off] = readVarint(body, off);
      if (this.openContainer !== null && containerId === this.openContainer) {
        this.stateId = stateId;
      }
      // The player inventory broadcast: menu slots run 0 (crafting
      // result) .. 44 (hotbar). A slot is [count varint][item varint]
      // [component patch]; count 0 is empty. Report the first non-empty
      // stack to track what the server thinks we hold.
      if (containerId === 0) {
        let p = off;
        let found = null;
        for (let i = 0; i < Math.min(count, 46) && p < body.length; i++) {
          let slotCount = 0;
          try {
            [slotCount, p] = readVarint(body, p);
          } catch {
            break;
          }
          if (slotCount === 0) {
            continue;
          }
          let item = 0;
          try {
            [item, p] = readVarint(body, p);
          } catch {
            break;
          }
          found = `menu slot ${i}: item ${item} x${slotCount}`;
          break;
        }
        const desc = found || 'all slots empty';
        if (desc !== this.lastInvDesc) {
          this.lastInvDesc = desc;
          this.log(`inventory: ${desc} (state ${stateId})`);
        }
      }
      return;
    }
    if (name === 'respawn') {
      this.respawnSeen = true;
      this.dead = false;
      this.stats.health = null;
      return;
    }
    if (name === 'change_difficulty' || name === 'game_event') {
      return;
    }
  }

  waitFor(predicate, timeoutMs, what) {
    return new Promise((resolve) => {
      const check = () => {
        if (predicate() || this.red) {
          clearInterval(timer);
          resolve(!this.red);
          return;
        }
        if (now() - start > timeoutMs) {
          clearInterval(timer);
          this.log(`timeout waiting for ${what}`);
          resolve(false);
        }
      };
      const start = now();
      const timer = setInterval(check, 50);
      this.waiters.push({ stop: () => { clearInterval(timer); resolve(false); } });
    });
  }

  async session() {
    this.waiters = [];
    const done = new Promise((resolve) => {
      this.resolveSession = resolve;
    });
    const deadline = this.t0 + this.durationMs;
    const timer = setTimeout(() => {
      if (!this.red) {
        this.setPhase('complete');
        this.log('duration reached');
      }
      this.finish();
    }, this.durationMs);

    this.client.connectClient();

    // 1. join: play state + teleport + first chunks.
    this.setPhase('join');
    const joined = await this.waitFor(
      () => this.pos !== null && this.stats.packetsSeen > 0 && this.seenChunk,
      60000,
      'join (teleport + chunks)',
    );
    if (!joined) {
      this.fail('join never completed (no teleport or chunks within 60s)');
      clearTimeout(timer);
      return;
    }
    this.send(SB.player_loaded, Buffer.alloc(0));

    await this.runPhases(deadline);
    // The duration timer is the resolver of `done`: it must stay armed
    // until it fires (or already fired), or the session promise leaks
    // and the runner's global timeout is the only way out.
    await done;
    clearTimeout(timer);
  }

  get seenChunk() {
    return this._seenChunk;
  }

  async runPhases(deadline) {
    // Fixed work area on the flat layer (y -60 feet, floor at -61).
    const A = { x: 8.5, y: -60.0, z: 8.5 };
    const FLOOR = { x0: 8, x1: 12, z0: 6, z1: 15 };
    const DIG = [
      { x: 10, y: -60, z: 6, name: 'minecraft:torch' },
      { x: 9, y: -60, z: 6, name: 'minecraft:dirt' },
      { x: 11, y: -60, z: 6, name: 'minecraft:stone' },
    ];
    const CHEST = { x: 10, y: -60, z: 15 };

    await this.cmd(`tp @s ${A.x} ${A.y} ${A.z}`);
    this.stopMovement();
    await sleep(300);
    this.pos = { ...A };
    this.groundY = A.y;

    // The floor is a command volley, not interactions: fire it with
    // spacing instead of pacing on replies, so a silent command source
    // cannot stretch setup past the session budget.
    for (let x = FLOOR.x0; x <= FLOOR.x1; x++) {
      for (let z = FLOOR.z0; z <= FLOOR.z1; z++) {
        this.send(SB.chat_command, chatCommandBody(`setblock ${x} -61 ${z} minecraft:stone`));
        await sleep(40);
      }
    }
    await sleep(500);
    for (const d of DIG) {
      await this.cmd(`setblock ${d.x} ${d.y} ${d.z} ${d.name}`);
    }
    await this.cmd(
      `setblock ${CHEST.x} ${CHEST.y} ${CHEST.z} minecraft:chest[facing=north,type=single,waterlogged=false]`,
    );

    // 2. walk: square loop with jumps and look-around, plus a creative
    // flight hop (the ability toggle and hover moves a player makes).
    this.setPhase('walk');
    for (let loop = 0; loop < 2 && !this.red; loop++) {
      await this.walkTo(12.5, 8.5, { jump: true });
      await this.walkTo(12.5, 12.5, { jump: loop === 0 });
      await this.walkTo(8.5, 12.5, { jump: true });
      await this.walkTo(8.5, 8.5);
    }
    await this.lookAround();
    await this.flightHop(A);

    // 3. dig: real dig lifecycle (instant torch, staged dirt, creative stone).
    this.setPhase('dig');
    await this.walkTo(10.5, 7.5);
    this.digStart(DIG[0]);
    await sleep(500);
    this.digStart(DIG[1]);
    await sleep(1600);
    this.digStop(DIG[1]);
    await this.cmd('gamemode creative');
    this.gamemode = 'creative';
    this.digStart(DIG[2]);
    await sleep(500);
    await this.cmd('gamemode survival');
    this.gamemode = 'survival';

    // 4. place: the give command grants a stack, then the creative hotbar
    // push pins stone to the selected slot. Picked-up dig drops shuffle
    // where a give lands, so the deterministic push is what the
    // use-item-on reads.
    this.setPhase('place');
    await this.cmd('give @s minecraft:stone 16');
    await this.pushStack('minecraft:stone', 16);
    await this.walkTo(9.5, 7.5);
    this.useOn(9, -61, 6);
    await sleep(800);
    this.useOn(9, -61, 7);
    await sleep(800);

    // 5. container: open the chest by right-clicking it, click a slot, close.
    this.setPhase('container');
    await this.walkTo(10.5, 13.5);
    this.useOn(CHEST.x, CHEST.y, CHEST.z);
    const opened = await this.waitFor(
      () => this.openContainer !== null,
      5000,
      'open_screen',
    );
    if (opened) {
      this.log(`container ${this.openContainer} opened (state ${this.stateId})`);
      await sleep(500);
      this.send(
        SB.container_click,
        containerClickBody(this.openContainer, this.stateId, 10),
      );
      await sleep(500);
      this.send(SB.container_close, containerCloseBody(this.openContainer));
      this.openContainer = null;
    } else {
      // A chest that does not open is a broken player path, not a
      // shortfall to note: the phase goes red.
      this.fail('container phase: use_item_on on the chest produced no open_screen');
    }

    // 6. drop the held stack.
    this.setPhase('drop');
    this.send(SB.player_action, playerActionBody(
      ACTION.drop_item, 9, -60, 6, 1, this.sequence++,
    ));
    await sleep(600);
    this.send(SB.player_action, playerActionBody(
      ACTION.drop_all, 9, -60, 6, 1, this.sequence++,
    ));
    await sleep(600);

    // 7-8. hit and death: summon a zombie, survive its melee until the
    // server tracks us down (vanilla: set_health 0; doppel: damage tally).
    this.setPhase('hit');
    await this.cmd('time set midnight');
    await this.walkTo(10.5, 10.5);
    this.stopMovement();
    await this.cmd('summon minecraft:zombie 12.5 -60 10.5');
    // The wait never eats the whole session: the continue phase keeps a
    // 20 s reserve even when the zombie is slow.
    const deathBudget = Math.max(15000, Math.min(90000, deadline - now() - 20000));
    let deathSeen = await this.waitFor(
      () => this.dead || this.stats.hits >= DOWNED_HITS,
      deathBudget,
      'zombie damage',
    );
    // Doppel tallies damage server-side only (no visible health drop),
    // so the hit count is the downed signal there. Vanilla reports
    // health: stand in until it truly reaches zero so the respawn is
    // the real one.
    if (deathSeen && !this.dead && this.stats.health !== null && this.stats.health > 0) {
      deathSeen = await this.waitFor(() => this.dead, 30000, 'health to reach zero');
    }
    this.log(`hits taken: ${this.stats.hits} health: ${this.stats.health}`);
    // The wait poll and a late hit can land together; re-test before
    // calling it a miss.
    if (!deathSeen && (this.dead || this.stats.hits >= DOWNED_HITS)) {
      deathSeen = true;
    }
    // The phase label says what the wire carried: a real death signal,
    // or (doppel) only the server-side damage tally. No signal at all
    // leaves the phase list honest and the residual tells the story.
    if (this.dead) {
      this.setPhase('death');
    } else if (this.stats.hits >= DOWNED_HITS) {
      this.setPhase('death(tally)');
    } else {
      this.residuals.push(`zombie phase ended without any death signal (${this.stats.hits} hits)`);
    }

    await sleep(1000);
    this.send(SB.client_command, clientCommandBody(0)); // PERFORM_RESPAWN
    const respawnConfirmed = await this.waitFor(
      () => this.respawnSeen && !this.dead,
      10000,
      'respawn confirmation',
    );
    // Doppel never answers a respawn request: the gesture went out, the
    // wire stayed quiet.
    this.setPhase(respawnConfirmed ? 'respawn' : 'respawn(sent)');
    if (!respawnConfirmed) {
      this.residuals.push('respawn request sent, no respawn packet received');
    }
    await sleep(1000);
    this.dead = false;
    // A vanilla respawn lands at world spawn; walk-area coordinates only
    // mean anything again after the teleport back.
    await this.cmd(`tp @s ${A.x} ${A.y} ${A.z}`);
    this.stopMovement();
    await sleep(300);
    this.pos = { ...A };
    this.groundY = A.y;

    // 9. continue: loop walk, dig and place until the duration ends.
    this.setPhase('continue');
    let round = 0;
    while (!this.red && now() < deadline - 15000) {
      round += 1;
      this.log(`continue round ${round}`);
      await this.walkTo(12.5, 8.5, { jump: true });
      await this.walkTo(12.5, 12.5);
      await this.walkTo(8.5, 12.5, { jump: true });
      await this.walkTo(8.5, 8.5);
      const target = { x: 9 + (round % 3), y: -60, z: 6 };
      await this.cmd(`setblock ${target.x} ${target.y} ${target.z} minecraft:dirt`);
      await this.pushStack('minecraft:stone', 4);
      await this.walkTo(10.5, 7.5);
      this.digStart(target);
      await sleep(1200);
      this.digStop(target);
      this.useOn(target.x, -61, target.z);
      await sleep(800);
    }
  }

  // The creative hotbar push: switch to creative, write the exact stack
  // to hotbar slot 0 (menu slot 36), switch back, select it. This is the
  // gesture a player makes picking an item from the creative menu, and
  // it makes the selected slot deterministic no matter what drops were
  // picked up.
  async pushStack(item, count) {
    const id = (this.items || {})[item];
    if (id === undefined) {
      this.residuals.push(`no item id for ${item}`);
      return;
    }
    await this.cmd('gamemode creative');
    this.gamemode = 'creative';
    await sleep(150);
    this.send(SB.set_creative_mode_slot, creativeSetBody(36, id, count));
    await sleep(150);
    await this.cmd('gamemode survival');
    this.gamemode = 'survival';
    this.send(SB.set_carried_item, setCarriedItemBody(0));
    await sleep(150);
  }

  async flightHop(A) {
    // The flight gesture a player makes in creative: toggle, rise with a
    // look sweep, descend, toggle off, back to survival. All position
    // frames flow from the one tick loop; this only drives state.
    await this.cmd('gamemode creative');
    this.gamemode = 'creative';
    await sleep(200);
    this.flying = true;
    this.send(SB.player_abilities, playerAbilitiesBody(true));
    this.vy = 0.3;
    this.jumping = false;
    this.onGround = false;
    const yawSweep = setInterval(() => {
      this.yaw = (this.yaw + 15) % 360;
    }, 180);
    await this.waitFor(
      () => this.pos.y >= this.groundY + 5 || this.red,
      10000,
      'flight ascent',
    );
    this.vy = -0.3;
    await this.waitFor(
      () => this.pos.y <= this.groundY + 0.01 || this.red,
      10000,
      'flight descent',
    );
    clearInterval(yawSweep);
    this.pos.y = this.groundY;
    this.flying = false;
    this.onGround = true;
    this.vy = 0;
    this.send(SB.player_abilities, playerAbilitiesBody(false));
    await this.cmd('gamemode survival');
    this.gamemode = 'survival';
  }

  useOn(x, y, z) {
    this.send(SB.use_item_on, useItemOnBody(x, y, z, this.sequence++));
    this.send(SB.punch, Buffer.alloc(0));
  }

  digStart(target) {
    this.send(SB.player_action, playerActionBody(
      ACTION.start_destroy, target.x, target.y, target.z, 1, this.sequence++,
    ));
    this.send(SB.punch, Buffer.alloc(0));
  }

  digStop(target) {
    this.send(SB.player_action, playerActionBody(
      ACTION.stop_destroy, target.x, target.y, target.z, 1, this.sequence++,
    ));
  }
}

function groundFlag(bot) {
  return bot.onGround ? 0x01 : 0x00;
}

// Best-effort readable text out of a disconnect component head.
function componentText(head) {
  if (!head) {
    return '(empty)';
  }
  const ascii = head.toString('latin1').replace(/[^\x20-\x7e]/g, ' ').trim();
  return ascii.slice(0, 160) || head.toString('hex').slice(0, 60);
}

export async function runSession(opts) {
  const bot = new Bot(opts);
  // The first chunk packet flips the join gate.
  const orig = bot.onPacket.bind(bot);
  bot.onPacket = (id, name, body) => {
    if (name === 'level_chunk_with_light') {
      bot._seenChunk = true;
    }
    orig(id, name, body);
  };
  return bot.run();
}
