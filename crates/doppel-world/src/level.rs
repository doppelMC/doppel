//! level.dat: gzip NBT with a `Data` root compound. The loader fills
//! defaults for absent fields so fresh and foreign files both parse.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::anvil_write::DATA_VERSION;

/// The persisted world meta: spawn point, clocks, game rules.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelMeta {
    pub spawn: (i32, i32, i32),
    pub day_time: i64,
    pub game_time: i64,
    pub game_rules: BTreeMap<String, String>,
    pub data_version: i32,
}

impl Default for LevelMeta {
    fn default() -> Self {
        LevelMeta {
            spawn: (0, -60, 0),
            day_time: 0,
            game_time: 0,
            game_rules: BTreeMap::new(),
            data_version: DATA_VERSION,
        }
    }
}

#[derive(Serialize, Deserialize, Default)]
struct SpawnNbt {
    #[serde(default = "overworld")]
    dimension: String,
    #[serde(default)]
    pos: Vec<i32>,
    #[serde(default)]
    yaw: f32,
    #[serde(default)]
    pitch: f32,
}

fn overworld() -> String {
    "minecraft:overworld".into()
}

#[derive(Serialize, Deserialize, Default)]
struct ClockNbt {
    total_ticks: i64,
    #[serde(default)]
    paused: bool,
}

#[derive(Serialize, Deserialize, Default)]
struct LevelNbt {
    #[serde(rename = "DataVersion", default)]
    data_version: i32,
    #[serde(rename = "spawn", default)]
    spawn: SpawnNbt,
    #[serde(rename = "world_clocks", default)]
    world_clocks: BTreeMap<String, ClockNbt>,
    #[serde(rename = "Time", default)]
    time: i64,
    #[serde(rename = "GameRules", default)]
    game_rules: BTreeMap<String, String>,
    #[serde(rename = "version", default)]
    version: i32,
}

#[derive(Serialize, Deserialize)]
struct RootNbt {
    #[serde(rename = "Data")]
    data: LevelNbt,
}

fn to_nbt(meta: &LevelMeta) -> RootNbt {
    let mut clocks = BTreeMap::new();
    clocks.insert(
        "minecraft:overworld".to_string(),
        ClockNbt {
            total_ticks: meta.day_time,
            paused: false,
        },
    );
    RootNbt {
        data: LevelNbt {
            data_version: meta.data_version,
            spawn: SpawnNbt {
                dimension: "minecraft:overworld".into(),
                pos: vec![meta.spawn.0, meta.spawn.1, meta.spawn.2],
                yaw: 0.0,
                pitch: 0.0,
            },
            world_clocks: clocks,
            time: meta.game_time,
            game_rules: meta.game_rules.clone(),
            version: 19133,
        },
    }
}

fn from_nbt(root: RootNbt) -> LevelMeta {
    let data = root.data;
    let spawn = |i: usize| data.spawn.pos.get(i).copied().unwrap_or(0);
    let day_time = data
        .world_clocks
        .get("minecraft:overworld")
        .map(|c| c.total_ticks)
        .unwrap_or(0);
    LevelMeta {
        spawn: (spawn(0), spawn(1), spawn(2)),
        day_time,
        game_time: data.time,
        game_rules: data.game_rules,
        data_version: data.data_version,
    }
}

/// Saves the meta as `level.dat` under the world root, atomically.
pub fn save(root: &Path, meta: &LevelMeta) -> Result<()> {
    let nbt = fastnbt::to_bytes(&to_nbt(meta)).context("serializing level.dat")?;
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(&nbt).context("compressing level.dat")?;
    let bytes = enc.finish().context("finishing level.dat compression")?;
    let real = root.join("level.dat");
    let tmp = root.join("level.dat.tmp");
    std::fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;
    std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &real)
        .with_context(|| format!("swapping {} into place", real.display()))?;
    Ok(())
}

/// Loads `level.dat`; None when it does not exist.
pub fn load(root: &Path) -> Result<Option<LevelMeta>> {
    let Ok(compressed) = std::fs::read(root.join("level.dat")) else {
        return Ok(None);
    };
    let mut out = Vec::new();
    let mut dec = flate2::read::GzDecoder::new(&compressed[..]);
    dec.read_to_end(&mut out)
        .context("decompressing level.dat")?;
    let root: RootNbt = fastnbt::from_bytes(&out).context("parsing level.dat")?;
    Ok(Some(from_nbt(root)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("doppel-level-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn level_roundtrip() {
        let root = dir("roundtrip");
        let mut rules = BTreeMap::new();
        rules.insert("random_tick_speed".to_string(), "300".to_string());
        rules.insert("spawn_mobs".to_string(), "false".to_string());
        let meta = LevelMeta {
            spawn: (8, -60, -3),
            day_time: 6000,
            game_time: 123_456,
            game_rules: rules,
            data_version: DATA_VERSION,
        };
        save(&root, &meta).unwrap();
        assert_eq!(load(&root).unwrap(), Some(meta));
        // A fresh dir has no level file.
        let empty = dir("empty");
        assert_eq!(load(&empty).unwrap(), None);
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&empty);
    }

    #[test]
    fn level_defaults_when_fields_absent() {
        let root = dir("sparse");
        let nbt = fastnbt::to_bytes(&RootNbt {
            data: LevelNbt {
                time: 42,
                ..Default::default()
            },
        })
        .unwrap();
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&nbt).unwrap();
        std::fs::write(root.join("level.dat"), enc.finish().unwrap()).unwrap();
        let meta = load(&root).unwrap().expect("parses");
        assert_eq!(meta.game_time, 42);
        assert_eq!(meta.day_time, 0);
        assert_eq!(meta.spawn, (0, 0, 0));
        assert!(meta.game_rules.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }
}
