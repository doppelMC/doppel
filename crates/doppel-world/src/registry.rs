//! The block-state registry: (name, sorted properties) -> global state id,
//! extracted from vanilla's own data generator. Keys are canonical:
//! properties sorted by name, `k=v` joined by commas.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

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
            // The generator lists the default state first per block.
            reg.defaults.entry(name.to_string()).or_insert(id);
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
        // Fill unspecified props from the block's default state.
        let default_id = *self.defaults.get(name)?;
        let (_, default_props) = self.by_id.get(&default_id)?;
        let merged = merge_props(default_props, &props);
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
}
