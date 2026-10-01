//! Extracts the block-state registry from the pinned vanilla jar by running
//! its own data generator (`--reports`), producing pins/blocks.json with
//! every (name, properties) -> state-id mapping. This is the registry
//! source of truth; the learned-table bootstrap retires once this lands.

use anyhow::{Context, Result};

pub fn run() -> Result<()> {
    let pin = doppel_protocol::load_pin()?;
    let jar = crate::vanilla::ensure_jar(&pin)?;
    let root = doppel_protocol::find_repo_root()?;
    let out_dir = root.join("target").join("vanilla").join("reports");

    if out_dir.join("blocks.json").exists() {
        println!("[registry] report already present");
    } else {
        std::fs::create_dir_all(&out_dir)?;
        eprintln!("[registry] running vanilla data generator (--reports)...");
        // The bundler jar dispatches to the data Main via this property.
        let status = std::process::Command::new("java")
            .env("JAVA_HOME", std::env::var("JAVA_HOME").unwrap_or_default())
            .arg("-DbundlerMainClass=net.minecraft.data.Main")
            .arg("-jar")
            .arg(&jar)
            .arg("--reports")
            .arg("--output")
            .arg(&out_dir)
            .current_dir(&out_dir)
            .status()
            .context("spawning data generator")?;
        anyhow::ensure!(status.success(), "data generator failed: {status}");
    }

    // Find the generated blocks.json (output layout varies: generated/ or direct).
    let candidates = [
        out_dir
            .join("generated")
            .join("reports")
            .join("blocks.json"),
        out_dir.join("reports").join("blocks.json"),
        out_dir.join("blocks.json"),
    ];
    let blocks = candidates
        .iter()
        .find(|p| p.exists())
        .context("blocks.json not found after generation")?;

    // Parse into the compact registry: [{name, props: "k=v,k=v", id}, ...]
    let raw = std::fs::read_to_string(blocks)?;
    let parsed: serde_json::Value = serde_json::from_str(&raw)?;
    let mut entries: Vec<serde_json::Value> = Vec::new();
    if let serde_json::Value::Object(blocks) = &parsed {
        for (name, def) in blocks {
            let Some(states) = def.get("states").and_then(|s| s.as_array()) else {
                continue;
            };
            for state in states {
                let id = state.get("id").and_then(|i| i.as_i64()).unwrap_or(-1);
                let props = state
                    .get("properties")
                    .map(|p| {
                        let pairs: Vec<String> = p
                            .as_object()
                            .map(|m| {
                                let mut v: Vec<(String, String)> = m
                                    .iter()
                                    .map(|(k, val)| {
                                        (k.clone(), val.as_str().unwrap_or("").to_string())
                                    })
                                    .collect();
                                v.sort();
                                v.into_iter().map(|(k, val)| format!("{k}={val}")).collect()
                            })
                            .unwrap_or_default();
                        pairs.join(",")
                    })
                    .unwrap_or_default();
                entries.push(serde_json::json!({
                    "name": name,
                    "props": props,
                    "id": id,
                }));
            }
        }
    }
    let pins = root.join("pins").join("blocks.json");
    std::fs::write(&pins, serde_json::to_string_pretty(&entries)? + "\n")
        .with_context(|| format!("writing {}", pins.display()))?;
    println!(
        "[registry] {} block states -> {}",
        entries.len(),
        pins.display()
    );
    Ok(())
}
