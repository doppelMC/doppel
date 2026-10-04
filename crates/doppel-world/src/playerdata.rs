//! Player data files: gzip NBT under `playerdata/<uuid>.dat`, written
//! through a tmp file and renamed into place. Field names follow the
//! player save format; missing fields load as defaults so foreign files
//! parse.

use std::io::{Read, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// One saved inventory slot. `extra` carries the stack's codec bytes when
/// the writer had them; loaders fall back to `id`/`count` without it.
#[derive(Clone, Debug, PartialEq)]
pub struct SavedSlot {
    pub slot: i8,
    pub id: String,
    pub count: i32,
    pub extra: Option<Vec<u8>>,
}

/// The persisted slice of a player: position, rotation, game mode, and
/// inventory slots.
#[derive(Clone, Debug, PartialEq)]
pub struct PlayerData {
    pub pos: [f64; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub game_mode: u8,
    pub inventory: Vec<SavedSlot>,
}

#[derive(Serialize, Deserialize)]
struct PlayerNbt {
    #[serde(rename = "Pos", default)]
    pos: Vec<f64>,
    #[serde(rename = "Rotation", default)]
    rotation: Vec<f32>,
    #[serde(rename = "playerGameType", default)]
    game_mode: i32,
    #[serde(rename = "Inventory", default)]
    inventory: Vec<SlotNbt>,
}

#[derive(Serialize, Deserialize)]
struct SlotNbt {
    #[serde(rename = "Slot", default)]
    slot: i8,
    #[serde(rename = "id")]
    id: String,
    #[serde(rename = "count", default = "one")]
    count: i32,
    #[serde(
        rename = "components",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    components: Option<fastnbt::ByteArray>,
}

fn one() -> i32 {
    1
}

/// Hyphenates 16 raw uuid bytes (8-4-4-4-12).
pub fn uuid_hyphenated(uuid: &[u8; 16]) -> String {
    let hex = |slice: &[u8]| -> String { slice.iter().map(|b| format!("{b:02x}")).collect() };
    format!(
        "{}-{}-{}-{}-{}",
        hex(&uuid[0..4]),
        hex(&uuid[4..6]),
        hex(&uuid[6..8]),
        hex(&uuid[8..10]),
        hex(&uuid[10..16])
    )
}

/// Parses a hyphenated uuid back to raw bytes; wrong shapes give None.
pub fn uuid_parse(text: &str) -> Option<[u8; 16]> {
    let mut bytes = Vec::with_capacity(16);
    let mut hex = text.chars().filter(|c| *c != '-');
    while bytes.len() < 16 {
        let hi = hex.next()?.to_digit(16)?;
        let lo = hex.next()?.to_digit(16)?;
        bytes.push((hi * 16 + lo) as u8);
    }
    if hex.next().is_some() {
        return None;
    }
    bytes.try_into().ok()
}

impl From<PlayerData> for PlayerNbt {
    fn from(data: PlayerData) -> PlayerNbt {
        PlayerNbt {
            pos: data.pos.to_vec(),
            rotation: vec![data.yaw, data.pitch],
            game_mode: data.game_mode as i32,
            inventory: data
                .inventory
                .into_iter()
                .map(|s| SlotNbt {
                    slot: s.slot,
                    id: s.id,
                    count: s.count,
                    components: s.extra.map(|bytes| {
                        fastnbt::ByteArray::new(bytes.into_iter().map(|b| b as i8).collect())
                    }),
                })
                .collect(),
        }
    }
}

fn gz_bytes(nbt: &[u8]) -> Result<Vec<u8>> {
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(nbt).context("compressing player nbt")?;
    enc.finish().context("finishing player nbt compression")
}

/// Saves player data under the world root. The write is atomic per file.
pub fn save(root: &Path, uuid: &[u8; 16], data: &PlayerData) -> Result<()> {
    let dir = root.join("playerdata");
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let nbt = fastnbt::to_bytes(&PlayerNbt::from(PlayerData {
        pos: data.pos,
        yaw: data.yaw,
        pitch: data.pitch,
        game_mode: data.game_mode,
        inventory: data.inventory.clone(),
    }))
    .context("serializing player nbt")?;
    let bytes = gz_bytes(&nbt)?;
    let real = dir.join(format!("{}.dat", uuid_hyphenated(uuid)));
    let tmp = dir.join(format!("{}.dat.tmp", uuid_hyphenated(uuid)));
    std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &real)
        .with_context(|| format!("swapping {} into place", real.display()))?;
    Ok(())
}

/// Loads player data; None when the file does not exist.
pub fn load(root: &Path, uuid: &[u8; 16]) -> Result<Option<PlayerData>> {
    let path = root
        .join("playerdata")
        .join(format!("{}.dat", uuid_hyphenated(uuid)));
    let Ok(compressed) = std::fs::read(&path) else {
        return Ok(None);
    };
    let mut out = Vec::new();
    let mut dec = flate2::read::GzDecoder::new(&compressed[..]);
    dec.read_to_end(&mut out)
        .with_context(|| format!("decompressing {}", path.display()))?;
    let nbt: PlayerNbt =
        fastnbt::from_bytes(&out).with_context(|| format!("parsing {}", path.display()))?;
    let pos = match (
        nbt.pos.first().copied(),
        nbt.pos.get(1).copied(),
        nbt.pos.get(2).copied(),
    ) {
        (Some(x), Some(y), Some(z)) => [x, y, z],
        _ => [0.0, 0.0, 0.0],
    };
    let rotation = |i: usize| nbt.rotation.get(i).copied().unwrap_or(0.0);
    Ok(Some(PlayerData {
        pos,
        yaw: rotation(0),
        pitch: rotation(1),
        game_mode: nbt.game_mode.clamp(0, 3) as u8,
        inventory: nbt
            .inventory
            .into_iter()
            .map(|s| SavedSlot {
                slot: s.slot,
                id: s.id,
                count: s.count,
                extra: s
                    .components
                    .map(|array| array.iter().map(|&b| b as u8).collect()),
            })
            .collect(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("doppel-playerdata-{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn playerdata_roundtrip() {
        let root = dir("roundtrip");
        let uuid = [
            0x97, 0xe9, 0xcb, 0x14, 0x47, 0x0c, 0x3c, 0x15, 0x97, 0x76, 0x2b, 0x16, 0xdc, 0xd2,
            0xe8, 0x27,
        ];
        let data = PlayerData {
            pos: [12.5, -60.0, 7.25],
            yaw: -90.0,
            pitch: 12.5,
            game_mode: 1,
            inventory: vec![
                SavedSlot {
                    slot: 0,
                    id: "minecraft:stone".into(),
                    count: 64,
                    extra: Some(vec![0x40, 0x01, 0x00, 0x00]),
                },
                SavedSlot {
                    slot: 40,
                    id: "minecraft:diamond_sword".into(),
                    count: 1,
                    extra: None,
                },
            ],
        };
        save(&root, &uuid, &data).unwrap();
        assert_eq!(load(&root, &uuid).unwrap(), Some(data));
        // A second uuid has no file.
        let other = [0u8; 16];
        assert_eq!(load(&root, &other).unwrap(), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn playerdata_tolerates_missing_fields() {
        let root = dir("sparse");
        let uuid = [1u8; 16];
        // Only the position: no rotation list at all, no gamemode, no
        // inventory.
        #[derive(Serialize)]
        struct PosOnly {
            #[serde(rename = "Pos")]
            pos: Vec<f64>,
        }
        let nbt = fastnbt::to_bytes(&PosOnly {
            pos: vec![1.0, 2.0, 3.0],
        })
        .unwrap();
        let dir_path = root.join("playerdata");
        std::fs::create_dir_all(&dir_path).unwrap();
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(&nbt).unwrap();
        std::fs::write(
            dir_path.join(format!("{}.dat", uuid_hyphenated(&uuid))),
            enc.finish().unwrap(),
        )
        .unwrap();
        let data = load(&root, &uuid).unwrap().expect("parses");
        assert_eq!(data.pos, [1.0, 2.0, 3.0]);
        assert_eq!((data.yaw, data.pitch), (0.0, 0.0));
        assert_eq!(data.game_mode, 0);
        assert!(data.inventory.is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn uuid_hyphenation_roundtrip() {
        let uuid = [
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88,
        ];
        let text = uuid_hyphenated(&uuid);
        assert_eq!(text, "12345678-9abc-def0-1122-334455667788");
        assert_eq!(uuid_parse(&text), Some(uuid));
        assert_eq!(uuid_parse("nope"), None);
    }
}
