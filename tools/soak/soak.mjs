// The soak runner: build, boot one server, run one bot session, judge,
// report, and tear everything down. Controls: SOAK_CONTROL=kill kills
// the server mid-session (must go red fast); SOAK_CONTROL=vanilla runs
// the same session against the pinned vanilla jar (must go green).
import { spawn, execSync } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { runSession } from './bot.mjs';
import { offlineUuid } from './proto.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = findRepoRoot(here);
const soakDir = path.join(repoRoot, 'target', 'soak');
const isWin = process.platform === 'win32';
const CONTROL = process.argv[2] || process.env.SOAK_CONTROL || 'none';
// The booted server, for teardown on failure paths that never reach
// main()'s own cleanup.
let server = null;

function findRepoRoot(start) {
  let dir = start;
  for (let i = 0; i < 6; i++) {
    if (fs.existsSync(path.join(dir, 'Cargo.toml')) && fs.existsSync(path.join(dir, 'pins'))) {
      return dir;
    }
    dir = path.dirname(dir);
  }
  throw new Error('repo root not found above tools/soak');
}

function log(msg) {
  console.log(`[soak] ${msg}`);
}

function refuseVanillaRunPath(p) {
  const norm = p.split(path.sep).join('/').toLowerCase();
  if (norm.includes('target/vanilla/run')) {
    throw new Error(`refusing to operate on ${p}`);
  }
}

function rmFresh(p) {
  refuseVanillaRunPath(p);
  fs.rmSync(p, { recursive: true, force: true });
  fs.mkdirSync(p, { recursive: true });
}

function killTree(child) {
  if (!child || child.exitCode !== null) {
    return;
  }
  if (isWin) {
    try {
      execSync(`taskkill /T /F /PID ${child.pid}`, { stdio: 'ignore' });
    } catch {
      try {
        child.kill('SIGKILL');
      } catch {}
    }
  } else {
    try {
      child.kill('SIGKILL');
    } catch {}
  }
}

function waitPort(port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const sock = net.connect({ host: '127.0.0.1', port }, () => {
        sock.destroy();
        resolve();
      });
      sock.on('error', () => {
        if (Date.now() > deadline) {
          reject(new Error(`port ${port} never opened within ${timeoutMs / 1000}s`));
        } else {
          setTimeout(attempt, 200);
        }
      });
    };
    attempt();
  });
}

function tail(text, lines = 40) {
  const all = String(text).split(/\r?\n/);
  return all.slice(-lines).join('\n');
}

class ServerLog {
  constructor() {
    this.chunks = [];
  }
  push(chunk) {
    this.chunks.push(chunk);
    if (this.chunks.length > 400) {
      this.chunks.splice(0, this.chunks.length - 400);
    }
    for (const line of String(chunk).split(/\r?\n/)) {
      if (line.trim()) {
        log(`server: ${line}`);
      }
    }
  }
  text() {
    return tail(this.chunks.join(''), 60);
  }
}

async function build() {
  const env = { ...process.env };
  // The local windows-gnu toolchain env; absent on CI, where the plain
  // toolchain applies.
  const zigcc = 'C:/Users/nour/tools/bin/zigcc.exe';
  if (!env.CC && fs.existsSync(zigcc)) {
    env.PATH = `C:/Users/nour/tools/bin;${env.PATH || ''}`;
    env.CC = zigcc;
    env.AR = 'C:/Users/nour/tools/bin/ar.exe';
    env.CARGO_TARGET_DIR = env.CARGO_TARGET_DIR || 'target-gnu';
    env.RUSTUP_TOOLCHAIN = env.RUSTUP_TOOLCHAIN || '1.98.0-x86_64-pc-windows-gnu';
  }
  log('cargo build --release');
  await new Promise((resolve, reject) => {
    const child = spawn('cargo', ['build', '--release'], { cwd: repoRoot, env, shell: isWin });
    child.stdout.on('data', (c) => process.stdout.write(c));
    child.stderr.on('data', (c) => process.stderr.write(c));
    child.on('exit', (code) => (code === 0 ? resolve() : reject(new Error(`cargo build exited ${code}`))));
    child.on('error', reject);
  });
}

function findDoppelBin() {
  const targetDirs = [process.env.CARGO_TARGET_DIR, 'target-gnu', 'target'];
  for (const dir of targetDirs) {
    if (!dir) {
      continue;
    }
    for (const name of ['doppel.exe', 'doppel']) {
      const p = path.join(repoRoot, dir, 'release', name);
      if (fs.existsSync(p)) {
        return p;
      }
    }
  }
  throw new Error('doppel binary not found; run the build first');
}

function resolveBlobs() {
  if (process.env.DOPPEL_BLOBS) {
    return process.env.DOPPEL_BLOBS;
  }
  for (const rel of ['captures/blobs', 'target/vanilla/blobs']) {
    const p = path.join(repoRoot, rel);
    if (fs.existsSync(path.join(p, 'manifest.json'))) {
      return p;
    }
  }
  throw new Error(
    'no blob set found: run `cargo run --release -p doppel-oracle -- pin` and '
    + '`capture-vanilla-login captures/vanilla-login.jsonl captures/blobs`, '
    + 'or set DOPPEL_BLOBS',
  );
}

function bootDoppel(port) {
  const bin = findDoppelBin();
  const world = path.join(soakDir, 'world');
  rmFresh(world);
  const logSink = new ServerLog();
  const env = {
    ...process.env,
    DOPPEL_ADDR: '127.0.0.1',
    DOPPEL_PORT: String(port),
    DOPPEL_PIN: path.join(repoRoot, 'pins', 'version.json'),
    DOPPEL_BLOBS: path.resolve(repoRoot, resolveBlobs()),
    DOPPEL_WORLD: world,
  };
  log(`booting ${bin} on ${port} (world ${world})`);
  const child = spawn(bin, [], { cwd: repoRoot, env, shell: false });
  child.stdout.on('data', (c) => logSink.push(c));
  child.stderr.on('data', (c) => logSink.push(c));
  return { child, logSink, world };
}

function readPin() {
  return JSON.parse(fs.readFileSync(path.join(repoRoot, 'pins', 'version.json'), 'utf8'));
}

// The name-to-id item registry pin, as-is.
function readItems() {
  return JSON.parse(fs.readFileSync(path.join(repoRoot, 'pins', 'items.json'), 'utf8'));
}

async function ensureVanillaJar(pin) {
  const dir = path.join(repoRoot, 'target', 'vanilla');
  fs.mkdirSync(dir, { recursive: true });
  const jar = path.join(dir, `server-${pin.id}.jar`);
  if (fs.existsSync(jar)) {
    return jar;
  }
  log(`downloading vanilla ${pin.id} jar...`);
  const resp = await fetch(pin.server_jar_url);
  if (!resp.ok) {
    throw new Error(`jar download failed: ${resp.status}`);
  }
  const bytes = Buffer.from(await resp.arrayBuffer());
  const sha1 = crypto.createHash('sha1').update(bytes).digest('hex');
  if (sha1 !== pin.server_jar_sha1) {
    throw new Error(`jar sha1 mismatch: expected ${pin.server_jar_sha1}, got ${sha1}`);
  }
  fs.writeFileSync(jar, bytes);
  return jar;
}

function uuidString(buf) {
  const hex = buf.toString('hex');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

async function bootVanilla(port, jar) {
  const runDir = path.join(soakDir, 'vanilla');
  rmFresh(runDir);
  fs.writeFileSync(path.join(runDir, 'eula.txt'), 'eula=true\n');
  fs.writeFileSync(path.join(runDir, 'server.properties'), [
    'online-mode=false',
    'white-list=false',
    `server-port=${port}`,
    'server-ip=127.0.0.1',
    'level-type=minecraft\\:flat',
    'generate-structures=false',
    'view-distance=4',
    'simulation-distance=4',
    'spawn-protection=0',
    'sync-chunk-writes=false',
    'motd=A Minecraft Server',
    '',
  ].join('\n'));
  const name = 'Doppel';
  const ops = [{ uuid: uuidString(offlineUuid(name)), name, level: 4, bypassesPlayerLimit: true }];
  fs.writeFileSync(path.join(runDir, 'ops.json'), JSON.stringify(ops, null, 2));

  const java = process.env.JAVA_BIN || 'java';
  const logSink = new ServerLog();
  log(`booting vanilla from ${jar} on ${port}`);
  const child = spawn(java, ['-Xms512M', '-Xmx2G', '-jar', jar, 'nogui'], {
    cwd: runDir,
    shell: false,
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  child.stdout.on('data', (c) => logSink.push(c));
  child.stderr.on('data', (c) => logSink.push(c));

  const done = new Promise((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error('vanilla did not reach Done within 300s')),
      300000,
    );
    const onData = (chunk) => {
      if (String(chunk).includes('Done (')) {
        clearTimeout(timer);
        resolve();
      }
    };
    child.stdout.on('data', onData);
    child.on('exit', (code) => {
      clearTimeout(timer);
      reject(new Error(`vanilla exited before ready (code ${code})`));
    });
  });
  try {
    await done;
  } catch (e) {
    killTree(child);
    throw e;
  }
  return { child, logSink, runDir };
}

async function stopVanilla(child) {
  if (child.exitCode !== null) {
    return;
  }
  try {
    child.stdin.write('stop\n');
  } catch {}
  await new Promise((resolve) => {
    const timer = setTimeout(resolve, 60000);
    child.on('exit', () => {
      clearTimeout(timer);
      resolve();
    });
  });
  killTree(child);
}

function writeReport(report) {
  fs.mkdirSync(soakDir, { recursive: true });
  if (report.serverFull) {
    fs.writeFileSync(path.join(soakDir, 'server.log'), report.serverFull);
  }
  const json = path.join(soakDir, 'report.json');
  const txt = path.join(soakDir, 'report.txt');
  fs.writeFileSync(json, JSON.stringify(report, null, 2));
  const lines = [
    `status: ${report.status}`,
    `control: ${report.control}`,
    `phase: ${report.phase}`,
    `uptime_s: ${report.uptimeS}`,
    `error: ${report.error || '(none)'}`,
    `packets_seen: ${report.packetsSeen}`,
    `hits: ${report.hits} health: ${report.health} deaths: ${report.deaths}`,
    `residuals: ${(report.residuals || []).join('; ') || '(none)'}`,
    `phases: ${(report.phases || []).join(', ')}`,
    `last_packets: ${(report.lastPackets || []).join(', ')}`,
    `position: ${JSON.stringify(report.position)}`,
    '',
    '--- server stdout/stderr tail ---',
    report.serverTail || '(none)',
  ];
  fs.writeFileSync(txt, lines.join('\n'));
  log(`report: ${json}`);
}

function clampInt(v, lo, hi, fallback) {
  const n = parseInt(v, 10);
  if (Number.isNaN(n)) {
    return fallback;
  }
  return Math.min(hi, Math.max(lo, n));
}

async function main() {
  const control = CONTROL;
  if (!['none', 'kill', 'vanilla'].includes(control)) {
    throw new Error(`unknown control "${control}" (none, kill, vanilla)`);
  }
  const port = parseInt(process.env.SOAK_DOPPEL_PORT || '25575', 10);
  if (port === 25565) {
    throw new Error('SOAK_DOPPEL_PORT=25565 is the live server port; refusing');
  }
  fs.mkdirSync(soakDir, { recursive: true });

  const pin = readPin();

  if (control === 'vanilla') {
    const jar = await ensureVanillaJar(pin);
    server = await bootVanilla(port, jar);
  } else {
    await build();
    server = bootDoppel(port);
  }

  const durationMs =
    control === 'vanilla'
      ? clampInt(process.env.SOAK_VANILLA_SECONDS, 30, 300, 90) * 1000
      : clampInt(process.env.SOAK_SECONDS, 180, 300, 240) * 1000;

  const stats = {};
  const sessionLog = [];
  const logLine = (msg) => {
    sessionLog.push(msg);
    log(`bot: ${msg}`);
  };

  // Server exit is a red in its own right: watch it for the whole run.
  let serverExit = null;
  server.child.on('exit', (code) => {
    serverExit = code;
  });

  await waitPort(port, control === 'vanilla' ? 300000 : 60000).catch((e) => {
    killTree(server.child);
    throw e;
  });
  log(`server accepting on ${port}; session ${durationMs / 1000}s`);

  let killTimer = null;
  if (control === 'kill') {
    killTimer = setTimeout(() => {
      log('kill control: killing the server tree');
      killTree(server.child);
    }, 45000);
  }

  const t0 = Date.now();
  const globalTimeoutMs = durationMs + 120000;

  const sessionPromise = runSession({
    host: '127.0.0.1',
    port,
    durationMs,
    log: logLine,
    stats,
    items: readItems(),
  });

  // The inbound watchdog participates in the race: a live-socket hang
  // must go red when it happens, not only if it still holds after the
  // session resolves.
  let watchdogClear = null;
  const watchdogPromise = new Promise((resolve) => {
    const timer = setInterval(() => {
      if (stats.lastPacketAt && Date.now() - stats.lastPacketAt > 60000) {
        clearInterval(timer);
        resolve({
          status: 'red',
          phase: stats.phase || 'unknown',
          error: `watchdog: ${Math.round((Date.now() - stats.lastPacketAt) / 1000)}s without inbound packets`,
          lastPackets: [],
          phases: [],
        });
      }
    }, 5000);
    watchdogClear = () => clearInterval(timer);
  });

  let result;
  let globalTimer;
  try {
    result = await Promise.race([
      sessionPromise,
      watchdogPromise,
      new Promise((resolve) => {
        globalTimer = setTimeout(() => resolve({
          status: 'red',
          phase: stats.phase || 'unknown',
          error: `global timeout after ${globalTimeoutMs / 1000}s`,
          lastPackets: [],
          phases: [],
        }), globalTimeoutMs);
      }),
    ]);
  } finally {
    watchdogClear();
    clearTimeout(globalTimer);
    if (killTimer) {
      clearTimeout(killTimer);
    }
  }

  const uptimeS = ((Date.now() - t0) / 1000).toFixed(1);

  // Post-session judgment beyond the bot's own conditions.
  let status = result.status;
  let error = result.error;
  if (status !== 'red') {
    if (Date.now() - (stats.lastPacketAt || t0) > 60000) {
      status = 'red';
      error = 'watchdog: 60s without inbound packets';
    } else if (serverExit !== null) {
      status = 'red';
      error = `server process exited (code ${serverExit})`;
    }
  }

  const report = {
    status,
    control,
    phase: result.phase || stats.phase,
    uptimeS: Number(uptimeS),
    error,
    phases: result.phases,
    lastPackets: result.lastPackets,
    packetsSeen: result.packetsSeen ?? stats.packetsSeen ?? 0,
    hits: result.hits ?? stats.hits ?? 0,
    health: result.health ?? stats.health ?? null,
    deaths: result.deaths ?? stats.deaths ?? 0,
    position: result.position ?? null,
    residuals: result.residuals || [],
    serverTail: server.logSink.text(),
    serverFull: server.logSink.chunks.join(''),
    serverExit,
    botLog: sessionLog.slice(-40),
  };
  writeReport(report);

  if (control === 'vanilla') {
    await stopVanilla(server.child);
  }
  killTree(server.child);

  console.log(`[soak] ${status}: ${error || 'all phases and duration completed'}`);
  process.exitCode = status === 'green' ? 0 : 1;
}

const running = main();
let cleaned = false;
async function cleanup(code) {
  if (cleaned) {
    return;
  }
  cleaned = true;
  if (server) {
    killTree(server.child);
  }
  try {
    await running;
  } catch {}
  process.exit(code);
}
process.on('SIGINT', () => cleanup(130));
process.on('unhandledRejection', (err) => {
  console.error('[soak] unhandled rejection:', err);
  cleanup(1);
});
running.catch((err) => {
  console.error(`[soak] ${err.message}`);
  if (server) {
    killTree(server.child);
  }
  try {
    writeReport({
    status: 'red',
    control: CONTROL,
    phase: 'boot',
    uptimeS: 0,
    error: err.message,
    phases: [],
    lastPackets: [],
    packetsSeen: 0,
    hits: 0,
    health: null,
    deaths: 0,
    position: null,
    serverTail: '(failed before the server booted; see the error above)',
    serverExit: null,
    botLog: [],
    });
  } catch (e) {
    console.error('[soak] report write failed:', e.message);
  }
  process.exitCode = 1;
});
