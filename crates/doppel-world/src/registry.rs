//! The block-state registry: (name, sorted properties) -> global state id,
//! extracted from vanilla's own data generator. Keys are canonical:
//! properties sorted by name, `k=v` joined by commas.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone)]
pub struct BlockRegistry {
    /// "name|k=v,k=v" -> state id
    by_key: HashMap<String, u32>,
    /// state id -> ("name", "k=v,k=v")
    by_id: HashMap<u32, (String, String)>,
    /// name -> default state id
    defaults: HashMap<String, u32>,
}

fn key(name: &str, props: &str) -> String {
    if props.is_empty() {
        name.to_string()
    } else {
        format!("{name}|{props}")
    }
}

/// Canonicalizes a property string: sorts `k=v` pairs by key.
pub fn canonical_props(props: &str) -> String {
    let mut pairs: Vec<&str> = props.split(',').filter(|p| !p.is_empty()).collect();
    pairs.sort();
    pairs.join(",")
}

impl BlockRegistry {
    pub fn load(path: &Path) -> Result<BlockRegistry> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("reading registry {}", path.display()))?;
        let entries: Vec<serde_json::Value> =
            serde_json::from_str(&raw).context("parsing registry")?;
        let mut reg = BlockRegistry {
            by_key: HashMap::new(),
            by_id: HashMap::new(),
            defaults: HashMap::new(),
        };
        for e in &entries {
            let name = e["name"].as_str().unwrap_or_default();
            let props = canonical_props(e["props"].as_str().unwrap_or_default());
            let id = e["id"].as_i64().unwrap_or(-1) as u32;
            if id == u32::MAX {
                continue;
            }
            reg.by_key.insert(key(name, &props), id);
            reg.by_id.insert(id, (name.to_string(), props.clone()));
            reg.defaults.entry(name.to_string()).or_insert(id);
        }
        // The enumeration lists states in property-value order, so the
        // first state carries each property's first value, not its
        // default: substitute the known property defaults and prefer
        // that state when it exists.
        let names: Vec<String> = reg.defaults.keys().cloned().collect();
        for name in names {
            let Some(first) = reg.defaults.get(&name).copied() else {
                continue;
            };
            let Some((_, first_props)) = reg.by_id.get(&first).cloned() else {
                continue;
            };
            let mut pairs: Vec<String> = Vec::new();
            let mut changed = false;
            for pair in first_props.split(',').filter(|p| !p.is_empty()) {
                let Some((k, v)) = pair.split_once('=') else {
                    continue;
                };
                let default = prop_default(k);
                if default.is_empty() {
                    pairs.push(pair.to_string());
                } else {
                    changed |= default != v;
                    pairs.push(format!("{k}={default}"));
                }
            }
            if !changed {
                continue;
            }
            let merged = canonical_props(&pairs.join(","));
            if let Some(&id) = reg.by_key.get(&key(&name, &merged)) {
                reg.defaults.insert(name, id);
            }
        }
        // Pillar blocks override the first-listed state: their default
        // is the vertical axis, not the enumeration's leading axis=x.
        let pillars: Vec<String> = reg
            .defaults
            .iter()
            .filter(|(name, &first)| {
                reg.by_id
                    .get(&first)
                    .is_some_and(|(_, props)| props == "axis=x")
                    && reg.by_key.contains_key(&format!("{name}|axis=y"))
            })
            .map(|(name, _)| name.clone())
            .collect();
        for name in pillars {
            if let Some(&vertical) = reg.by_key.get(&format!("{name}|axis=y")) {
                reg.defaults.insert(name, vertical);
            }
        }
        Ok(reg)
    }

    /// Looks up a state id; empty props falls back to the block's default.
    /// Partial props are filled from the default state's props (vanilla
    /// setblock semantics), then the exact match resolves.
    pub fn state_id(&self, name: &str, props: &str) -> Option<u32> {
        let props = canonical_props(props);
        if props.is_empty() {
            return self.defaults.get(name).copied();
        }
        if let Some(id) = self.by_key.get(&key(name, &props)) {
            return Some(*id);
        }
        // Fill unspecified props from the property defaults first, then
        // from the enumeration's first state for anything unknown: the
        // generator lists states in property-value order, not with the
        // default first (a bare wire enumerates "up" before "none").
        let mut pairs: Vec<String> = Vec::new();
        let known = self.by_id.values().find(|(n, _)| n == name);
        if let Some((_, first_props)) = known {
            for pair in first_props.split(',').filter(|p| !p.is_empty()) {
                if let Some((k, _)) = pair.split_once('=') {
                    pairs.push(format!("{k}={}", prop_default(k)));
                }
            }
        }
        let base = pairs.join(",");
        let merged = merge_props(&base, &props);
        if let Some(id) = self.by_key.get(&key(name, &merged)) {
            return Some(*id);
        }
        // Fall back to the first state's values for props without a
        // known default.
        let default_id = *self.defaults.get(name)?;
        let (_, default_props) = self.by_id.get(&default_id)?;
        let merged = merge_props(default_props, &merged);
        self.by_key.get(&key(name, &merged)).copied()
    }

    /// Splits "name[k=v,k=v]" into parts.
    pub fn split_state(spec: &str) -> (&str, &str) {
        match spec.split_once('[') {
            Some((name, rest)) => (name, rest.trim_end_matches(']')),
            None => (spec, ""),
        }
    }

    /// Returns (name, props) for a state id.
    pub fn state_of(&self, id: u32) -> Option<(&str, &str)> {
        self.by_id.get(&id).map(|(n, p)| (n.as_str(), p.as_str()))
    }

    /// Reads one integer property from a props string.
    pub fn prop_int(props: &str, name: &str) -> Option<i32> {
        for pair in props.split(',') {
            if let Some((k, v)) = pair.split_once('=') {
                if k == name {
                    return v.parse().ok();
                }
            }
        }
        None
    }

    /// Returns props with one property replaced (canonical order kept).
    pub fn with_prop(props: &str, name: &str, value: &str) -> String {
        let mut pairs: Vec<String> = props
            .split(',')
            .filter(|p| !p.is_empty())
            .filter(|p| !p.starts_with(&format!("{name}=")))
            .map(str::to_string)
            .collect();
        pairs.push(format!("{name}={value}"));
        canonical_props(&pairs.join(","))
    }
}

/// The default value of a block-state property. Keyed per property, not
/// per block: defaults are uniform across the families the engine models.
fn prop_default(prop: &str) -> &'static str {
    match prop {
        "powered" => "false",
        "lit" => "true",
        "locked" => "false",
        "delay" => "1",
        "mode" => "compare",
        "facing" => "north",
        "face" => "floor",
        "power" => "0",
        "east" | "north" | "south" | "west" => "none",
        "extended" => "false",
        "waterlogged" => "false",
        "snowy" => "false",
        "distance" => "7",
        "persistent" => "false",
        _ => "",
    }
}

/// Merges explicit `k=v` pairs over a base props string (canonical order).
pub fn merge_props(base: &str, explicit: &str) -> String {
    let mut map: std::collections::BTreeMap<String, String> = base
        .split(',')
        .filter(|p| !p.is_empty())
        .filter_map(|p| {
            p.split_once('=')
                .map(|(k, v)| (k.to_string(), v.to_string()))
        })
        .collect();
    for pair in explicit.split(',').filter(|p| !p.is_empty()) {
        if let Some((k, v)) = pair.split_once('=') {
            map.insert(k.to_string(), v.to_string());
        }
    }
    map.into_iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pillar_defaults_stand_upright() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/blocks.json");
        let reg = BlockRegistry::load(&path).expect("block registry pins");
        // The state list names axis=x first; the pillar default is vertical.
        for name in [
            "minecraft:deepslate",
            "minecraft:oak_log",
            "minecraft:basalt",
        ] {
            let id = reg.state_id(name, "").unwrap();
            let props = reg.state_of(id).unwrap().1;
            assert_eq!(props, "axis=y", "{name} default props");
        }
        // Blocks without an axis property keep the first-listed state.
        let stone = reg.state_id("minecraft:stone", "").unwrap();
        assert_eq!(reg.state_of(stone).unwrap().1, "");
    }

    #[test]
    fn props_helpers() {
        assert_eq!(canonical_props("b=2,a=1"), "a=1,b=2");
        assert_eq!(BlockRegistry::with_prop("a=1,b=2", "b", "3"), "a=1,b=3");
        assert_eq!(BlockRegistry::with_prop("", "power", "7"), "power=7");
        assert_eq!(
            BlockRegistry::prop_int("power=7,lit=true", "power"),
            Some(7)
        );
        assert_eq!(BlockRegistry::prop_int("power=7", "lit"), None);
        let (n, p) = BlockRegistry::split_state("minecraft:redstone_wire[power=7]");
        assert_eq!((n, p), ("minecraft:redstone_wire", "power=7"));
        let (n2, p2) = BlockRegistry::split_state("minecraft:stone");
        assert_eq!((n2, p2), ("minecraft:stone", ""));
    }

    #[test]
    fn defaults_substitute_property_defaults() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pins/blocks.json");
        let reg = BlockRegistry::load(&path).expect("block registry pins");
        // The state list names snowy=true first; the default is not snowy.
        let grass = reg.state_id("minecraft:grass_block", "").unwrap();
        assert_eq!(reg.state_of(grass).unwrap().1, "snowy=false");
        // Leaves enumerate distance=1 first; the default distance is 7.
        let leaves = reg.state_id("minecraft:oak_leaves", "").unwrap();
        assert_eq!(
            reg.state_of(leaves).unwrap().1,
            "distance=7,persistent=false,waterlogged=false"
        );
        // Litter enumerates facing=east first; the default facing is north.
        let litter = reg.state_id("minecraft:leaf_litter", "").unwrap();
        assert_eq!(
            reg.state_of(litter).unwrap().1,
            "facing=north,segment_amount=1"
        );
        // Explicit props still resolve to the named state.
        let snowed = reg.state_id("minecraft:grass_block", "snowy=true").unwrap();
        assert_eq!(reg.state_of(snowed).unwrap().1, "snowy=true");
    }
}
