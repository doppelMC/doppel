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

#[derive(Serialize, Deserialize)]
#[serde(default)]
struct SpawnNbt {
    dimension: String,
    #[serde(deserialize_with = "pos_any_array")]
    pos: fastnbt::IntArray,
    yaw: f32,
    pitch: f32,
}

impl Default for SpawnNbt {
    fn default() -> Self {
        SpawnNbt {
            dimension: overworld(),
            pos: fastnbt::IntArray::new(Vec::new()),
            yaw: 0.0,
            pitch: 0.0,
        }
    }
}

fn overworld() -> String {
    "minecraft:overworld".into()
}

/// The reference writes the spawn position as a typed int array; the
/// earlier doppel writer emitted an int list. Both shapes load.
fn pos_any_array<'de, D>(deserializer: D) -> Result<fastnbt::IntArray, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let value = fastnbt::Value::deserialize(deserializer)?;
    let ints = match value {
        fastnbt::Value::IntArray(a) => a.into_inner(),
        fastnbt::Value::List(items) => items
            .into_iter()
            .filter_map(|v| match v {
                fastnbt::Value::Int(i) => Some(i),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    Ok(fastnbt::IntArray::new(ints))
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
                pos: fastnbt::IntArray::new(vec![meta.spawn.0, meta.spawn.1, meta.spawn.2]),
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
    let pick = |i: usize| data.spawn.pos.get(i).copied().unwrap_or(0);
    let day_time = data
        .world_clocks
        .get("minecraft:overworld")
        .map(|c| c.total_ticks)
        .unwrap_or(0);
    LevelMeta {
        spawn: (pick(0), pick(1), pick(2)),
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
    fn level_parses_a_vanilla_file() {
        // The reference writes spawn/pos as a typed int array, not a list;
        // this fixture is a real reference-generated level.dat.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/level-vanilla.dat");
        let root = dir("vanilla");
        std::fs::copy(&path, root.join("level.dat")).unwrap();
        let meta = load(&root).unwrap().expect("vanilla level.dat parses");
        assert_eq!(meta.spawn, (0, -60, 0));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn level_parses_a_list_written_position() {
        // The earlier doppel writer emitted the position as an int list;
        // those worlds still load.
        let mut spawn = std::collections::HashMap::new();
        spawn.insert(
            "pos".to_string(),
            fastnbt::Value::List(vec![
                fastnbt::Value::Int(7),
                fastnbt::Value::Int(-60),
                fastnbt::Value::Int(9),
            ]),
        );
        let mut data = std::collections::HashMap::new();
        data.insert("spawn".to_string(), fastnbt::Value::Compound(spawn));
        let mut root = std::collections::HashMap::new();
        root.insert("Data".to_string(), fastnbt::Value::Compound(data));
        let nbt = fastnbt::to_bytes(&root).unwrap();
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&nbt).unwrap();
        let root = dir("list");
        std::fs::write(root.join("level.dat"), enc.finish().unwrap()).unwrap();
        let meta = load(&root).unwrap().expect("list-written level.dat parses");
        assert_eq!(meta.spawn, (7, -60, 9));
        let _ = std::fs::remove_dir_all(&root);
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
