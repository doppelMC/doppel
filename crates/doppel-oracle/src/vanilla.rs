//! Vanilla server lifecycle management: jar download + SHA-1 verification,
//! headless boot with a deterministic configuration, and teardown.

use anyhow::{bail, Context, Result};
use doppel_protocol::{find_repo_root, Pin};
use sha1::{Digest, Sha1};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

const BOOT_TIMEOUT: Duration = Duration::from_secs(300);

pub fn vanilla_dir() -> Result<PathBuf> {
    Ok(find_repo_root()?.join("target").join("vanilla"))
}

/// The vanilla run directory: VANILLA_RUN_DIR when set, the shared
/// default otherwise. The boot below wipes this directory, so local
/// runs that share the machine with another gate or a live server
/// point it somewhere private.
pub fn run_dir() -> Result<PathBuf> {
    match std::env::var_os("VANILLA_RUN_DIR") {
        Some(dir) => Ok(PathBuf::from(dir)),
        None => Ok(vanilla_dir()?.join("run")),
    }
}

/// The world directory inside the vanilla run directory.
pub fn run_world() -> Result<PathBuf> {
    Ok(run_dir()?.join("world"))
}

/// Gives a gate a private run directory unless one is already set.
pub fn default_run_dir(tag: &str) {
    if std::env::var_os("VANILLA_RUN_DIR").is_none() {
        let dir = vanilla_dir()
            .map(|base| base.join(format!("run-{tag}")))
            .unwrap_or_else(|_| PathBuf::from(format!("run-{tag}")));
        std::env::set_var("VANILLA_RUN_DIR", dir);
    }
}

/// Downloads the pinned server jar if absent (and verifies its SHA-1).
pub fn ensure_jar(pin: &Pin) -> Result<PathBuf> {
    let dir = vanilla_dir()?;
    std::fs::create_dir_all(&dir)?;
    let jar = dir.join(format!("server-{}.jar", pin.id));

    if jar.exists() && sha1_file(&jar)? == pin.server_jar_sha1 {
        return Ok(jar);
    }
    println!("[oracle] downloading vanilla {} server jar...", pin.id);
    let resp = ureq::get(&pin.server_jar_url)
        .timeout(Duration::from_secs(600))
        .call()
        .with_context(|| format!("fetching {}", pin.server_jar_url))?;
    let tmp = dir.join("server.jar.part");
    let mut file = File::create(&tmp)?;
    let mut reader = resp.into_reader();
    std::io::copy(&mut reader, &mut file)?;
    drop(file);

    let got = sha1_file(&tmp)?;
    if got != pin.server_jar_sha1 {
        std::fs::remove_file(&tmp).ok();
        bail!(
            "server jar SHA-1 mismatch: expected {}, got {}",
            pin.server_jar_sha1,
            got
        );
    }
    std::fs::rename(&tmp, &jar)?;
    Ok(jar)
}

fn sha1_file(path: &std::path::Path) -> Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = Sha1::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Reports the major version of `java` on PATH (8 for "1.8.0_x", 25 for "25.0.1").
/// The java binary the oracle boots vanilla with: JAVA_BIN when set
/// (a specific JDK install), else whatever `java` resolves on PATH.
fn java_bin() -> String {
    std::env::var("JAVA_BIN").unwrap_or_else(|_| "java".to_string())
}

pub fn java_major_version() -> Result<u32> {
    let out = Command::new(java_bin())
        .arg("-version")
        .output()
        .context("running `java -version` — is a JDK/JRE installed and on PATH?")?;
    let text = String::from_utf8_lossy(&out.stderr);
    let line = text.lines().next().unwrap_or_default();
    let quoted = line
        .split('"')
        .nth(1)
        .with_context(|| format!("cannot parse `java -version` output: {line}"))?;
    let major = if let Some(rest) = quoted.strip_prefix("1.") {
        rest.split(['.', '_']).next().unwrap_or("0")
    } else {
        quoted.split(['.', '_']).next().unwrap_or("0")
    };
    major
        .parse()
        .with_context(|| format!("cannot parse java major version from {quoted:?}"))
}

/// A running vanilla server. Killing it on drop guarantees teardown even
/// when the harness errors out mid-test.
pub struct VanillaServer {
    child: Option<Child>,
    pub port: u16,
}

impl Drop for VanillaServer {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
            eprintln!("[oracle] vanilla (port {}) stopped", self.port);
        }
    }
}

/// Boots vanilla headless on `port` inside a fresh run directory and waits
/// for the "Done" log line.
pub fn boot(pin: &Pin, jar: &std::path::Path, port: u16) -> Result<VanillaServer> {
    // Minimal deterministic configuration: flat world, offline, small radius.
    // The MOTD is deliberately left at the vanilla default ("A Minecraft
    // Server") — matching the default is exactly what parity means.
    let properties = format!(
        "online-mode=false\n\
         white-list=false\n\
         server-port={port}\n\
         level-type=minecraft\\:flat\n\
         generate-structures=false\n\
         view-distance=4\n\
         simulation-distance=4\n\
         spawn-protection=0\n\
         sync-chunk-writes=false\n"
    );
    boot_with_properties(pin, jar, port, &properties)
}

/// Boots vanilla for the survival gate: the flat configuration plus
/// peaceful difficulty. Spawner gamerules only stop future spawns, so a
/// normal boot keeps whatever mob swarm built up before the scripted
/// volley lands - and a swarm can shove or kill the scenario's player.
/// Peaceful difficulty holds monsters out entirely; drops, pickup, and
/// random ticks are difficulty-independent.
pub fn boot_peaceful(pin: &Pin, jar: &std::path::Path, port: u16) -> Result<VanillaServer> {
    let properties = format!(
        "online-mode=false\n\
         white-list=false\n\
         server-port={port}\n\
         level-type=minecraft\\:flat\n\
         generate-structures=false\n\
         view-distance=4\n\
         simulation-distance=4\n\
         spawn-protection=0\n\
         sync-chunk-writes=false\n\
         difficulty=peaceful\n"
    );
    boot_with_properties(pin, jar, port, &properties)
}

/// Boots vanilla with a pinned seed and normal terrain generation, for
/// worldgen parity comparisons.
pub fn boot_seeded(
    pin: &Pin,
    jar: &std::path::Path,
    port: u16,
    seed: i64,
) -> Result<VanillaServer> {
    let properties = format!(
        "online-mode=false\n\
         white-list=false\n\
         server-port={port}\n\
         level-seed={seed}\n\
         level-type=minecraft\\:normal\n\
         generate-structures=false\n\
         view-distance=4\n\
         simulation-distance=4\n\
         spawn-protection=0\n\
         sync-chunk-writes=false\n"
    );
    boot_with_properties(pin, jar, port, &properties)
}

fn boot_with_properties(
    pin: &Pin,
    jar: &std::path::Path,
    port: u16,
    properties: &str,
) -> Result<VanillaServer> {
    let have = java_major_version()?;
    if have < pin.java_major {
        bail!(
            "this machine has Java {have}, but vanilla {} requires Java {}+ — \
             run `doppel-oracle parity-status` on CI, or install a newer JDK \
             (e.g. Temurin {})",
            pin.id,
            pin.java_major,
            pin.java_major
        );
    }

    let run_dir = run_dir()?;
    if run_dir.exists() {
        std::fs::remove_dir_all(&run_dir).context("cleaning vanilla run dir")?;
    }
    std::fs::create_dir_all(&run_dir)?;

    std::fs::write(run_dir.join("eula.txt"), "eula=true\n").context("writing eula.txt")?;

    std::fs::write(run_dir.join("server.properties"), properties)
        .context("writing server.properties")?;

    // Op the capture bots so scenarios can run commands (/tick, /setblock,
    // /data). The offline-mode profile UUIDs are deterministic (UUIDv3 of
    // "OfflinePlayer:<name>"), so the same entries work every run. The
    // witness bot needs it to teleport away from the (variable) world
    // spawn before the scenario runs.
    let ops = r#"[{"uuid": "97e9cb14-470c-3c15-a976-2b16dcd2e827", "name": "Doppel", "level": 4, "bypassesPlayerLimit": true}, {"uuid": "c82d9a9e-f1aa-33eb-9328-3cc5d6797842", "name": "Doppelist", "level": 4, "bypassesPlayerLimit": true}]
"#;
    std::fs::write(run_dir.join("ops.json"), ops).context("writing ops.json")?;

    eprintln!("[oracle] booting vanilla {} on port {port}...", pin.id);
    let mut child = Command::new(java_bin())
        .args(["-Xms512M", "-Xmx2G", "-jar"])
        .arg(jar)
        .arg("nogui")
        .current_dir(&run_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("spawning vanilla server")?;

    let stdout = child.stdout.take().expect("stdout piped above");
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            eprintln!("[vanilla] {line}");
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    // Constructed before the wait loop so that any early `bail!` below
    // drops it and kills the JVM instead of orphaning it on the port.
    let server = VanillaServer {
        child: Some(child),
        port,
    };

    let deadline = std::time::Instant::now() + BOOT_TIMEOUT;
    loop {
        let timeout = deadline.saturating_duration_since(std::time::Instant::now());
        if timeout.is_zero() {
            bail!(
                "vanilla did not reach 'Done' within {}s",
                BOOT_TIMEOUT.as_secs()
            );
        }
        match rx.recv_timeout(timeout) {
            Ok(line) => {
                // Vanilla logs `[HH:MM:SS] [Server thread/INFO]: Done (X.XXXs)! ...`
                if line.contains("Done (") {
                    eprintln!("[oracle] vanilla is ready");
                    return Ok(server);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                bail!(
                    "vanilla did not reach 'Done' within {}s",
                    BOOT_TIMEOUT.as_secs()
                );
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                bail!("vanilla exited before becoming ready (see log above)");
            }
        }
    }
}
