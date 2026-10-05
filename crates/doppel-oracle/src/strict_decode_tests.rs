//! Unit tests for the strict decode gate.

use super::*;
use doppel_protocol::{write_string, write_varint};

fn body_ok(phase: Phase, id: i32, body: &[u8]) {
    match check_frame(phase, id, body) {
        Ok(n) => assert_eq!(n, body.len(), "consumed must equal the body length"),
        Err(f) => panic!("expected decode: {f:?}"),
    }
}

fn body_fails(phase: Phase, id: i32, body: &[u8]) -> FrameFailure {
    match check_frame(phase, id, body) {
        Ok(n) => panic!("expected failure, decoded {n} bytes"),
        Err(f) => f,
    }
}

#[test]
fn set_time_roundtrip() {
    // Steady state: game time, empty clock map.
    let mut b = Vec::new();
    b.extend_from_slice(&1000i64.to_be_bytes());
    write_varint(&mut b, 0);
    body_ok(Phase::Play, 0x73, &b);
    // One clock entry: holder 0, varlong day time, partial, rate.
    let mut b = Vec::new();
    b.extend_from_slice(&1000i64.to_be_bytes());
    write_varint(&mut b, 1);
    write_varint(&mut b, 0);
    doppel_protocol::write_varlong(&mut b, 6000);
    b.extend_from_slice(&0.0f32.to_be_bytes());
    b.extend_from_slice(&1.0f32.to_be_bytes());
    body_ok(Phase::Play, 0x73, &b);
}

#[test]
fn set_time_rejects_the_two_raw_longs_form() {
    // The regression shape: game time plus a raw day long, no map.
    let b = include_bytes!("../fixtures/strict_decode/set_time-two-raw-longs.bin");
    let f = body_fails(Phase::Play, 0x73, b);
    assert!(f.reason.contains("leftover"), "{}", f.reason);
    assert_eq!(f.consumed, 9, "the map count is the last decoded field");
}

#[test]
fn set_time_rejects_truncation_and_padding() {
    let mut b = Vec::new();
    b.extend_from_slice(&1000i64.to_be_bytes());
    write_varint(&mut b, 0);
    body_fails(Phase::Play, 0x73, &b[..8]);
    let mut padded = b.clone();
    padded.push(0x00);
    body_fails(Phase::Play, 0x73, &padded);
}

#[test]
fn system_chat_roundtrip() {
    // Anonymous-root TAG_String plus the overlay bool.
    let mut b = vec![0x08];
    b.extend_from_slice(&5u16.to_be_bytes());
    b.extend_from_slice(b"hello");
    b.push(0x00);
    body_ok(Phase::Play, 0x7c, &b);
    body_fails(Phase::Play, 0x7c, &b[..b.len() - 1]);
}

#[test]
fn keep_alive_roundtrip() {
    let b = 0x0064_6f70_7065_6c01i64.to_be_bytes();
    body_ok(Phase::Play, 0x2d, &b);
    body_fails(Phase::Play, 0x2d, &b[..7]);
    let mut padded = b.to_vec();
    padded.push(0);
    body_fails(Phase::Play, 0x2d, &padded);
}

#[test]
fn block_update_roundtrip() {
    let mut b = Vec::new();
    b.extend_from_slice(&bot::pack_block_pos(1, -60, 2).to_be_bytes());
    write_varint(&mut b, 1);
    body_ok(Phase::Play, 0x08, &b);
    // A second state varint is a leftover byte.
    let mut b2 = b.clone();
    write_varint(&mut b2, 0);
    let f = body_fails(Phase::Play, 0x08, &b2);
    assert!(f.reason.contains("leftover"), "{}", f.reason);
}

#[test]
fn container_set_content_roundtrip() {
    let mut b = Vec::new();
    write_varint(&mut b, 0); // container id
    write_varint(&mut b, 1); // state id
    write_varint(&mut b, 2); // slots
    write_varint(&mut b, 4); // a stack of 4
    write_varint(&mut b, 5); // ...of item 5
    write_varint(&mut b, 0); // patch: nothing added
    write_varint(&mut b, 0); // patch: nothing removed
    write_varint(&mut b, 0); // an empty slot
    write_varint(&mut b, 0); // carried: empty
    body_ok(Phase::Play, 0x12, &b);
    // A claimed slot count above the actual is an under-read.
    let mut bad = b.clone();
    bad[2] = 3;
    let f = body_fails(Phase::Play, 0x12, &bad);
    assert!(
        f.reason.contains("stack count") || f.reason.contains("bytes"),
        "{}",
        f.reason
    );
}

#[test]
fn damage_event_optional_position_flag() {
    // The position flag byte is always present.
    let mut b = Vec::new();
    write_varint(&mut b, 7);
    write_varint(&mut b, 35);
    write_varint(&mut b, 0);
    write_varint(&mut b, 0);
    b.push(0);
    body_ok(Phase::Play, 0x19, &b);
    let mut with = b.clone();
    with.pop();
    with.push(1);
    with.extend_from_slice(&0.5f64.to_be_bytes());
    with.extend_from_slice(&1.0f64.to_be_bytes());
    with.extend_from_slice(&1.5f64.to_be_bytes());
    body_ok(Phase::Play, 0x19, &with);
    // A set flag without the position is an under-read.
    let mut broken = b.clone();
    broken.pop();
    broken.push(1);
    body_fails(Phase::Play, 0x19, &broken);
}

#[test]
fn explode_roundtrip() {
    let mut b = Vec::new();
    for v in [0.5f64, 64.0, 0.5] {
        b.extend_from_slice(&v.to_be_bytes());
    }
    b.extend_from_slice(&3.0f32.to_be_bytes());
    b.extend_from_slice(&12i32.to_be_bytes()); // fixed-width count
    b.push(0); // no knockback
    write_varint(&mut b, 29); // explosion_emitter
    write_varint(&mut b, 672); // sound holder
    write_varint(&mut b, 1); // one weighted entry
    write_varint(&mut b, 69); // particle
    b.extend_from_slice(&0.5f32.to_be_bytes());
    b.extend_from_slice(&1.0f32.to_be_bytes());
    write_varint(&mut b, 0); // weight
    b.push(1); // play sound
    body_ok(Phase::Play, 0x24, &b);
    body_fails(Phase::Play, 0x24, &b[..b.len() - 1]);
}

#[test]
fn login_finished_and_config_packets() {
    let mut b = Vec::new();
    b.extend_from_slice(&[0u8; 16]);
    write_string(&mut b, "Doppel");
    write_varint(&mut b, 0);
    b.extend_from_slice(&[1u8; 16]);
    body_ok(Phase::Login, 0x02, &b);
    let mut brand = Vec::new();
    write_string(&mut brand, "minecraft:brand");
    write_string(&mut brand, "vanilla");
    body_ok(Phase::Config, 0x01, &brand);
    let mut packs = Vec::new();
    write_varint(&mut packs, 1);
    write_string(&mut packs, "minecraft");
    write_string(&mut packs, "core");
    write_string(&mut packs, "26.3");
    body_ok(Phase::Config, 0x0f, &packs);
    body_ok(Phase::Config, 0x03, &[]);
}

#[test]
fn unknown_id_and_gap_fail() {
    let f = body_fails(Phase::Play, 0x90, &[]);
    assert!(f.reason.contains("unknown id"), "{}", f.reason);
    let f = body_fails(Phase::Play, 0x09, &[]);
    assert!(f.reason.contains("no decoder"), "{}", f.reason);
}

#[test]
fn truncation_and_padding_fail_on_synthetic_frames() {
    let mut b = Vec::new();
    b.extend_from_slice(&bot::pack_block_pos(1, 2, 3).to_be_bytes());
    write_varint(&mut b, 9);
    // Truncation, padding, and every count patch must fail.
    body_fails(Phase::Play, 0x08, &b[..7]);
    body_fails(Phase::Play, 0x08, &[b.as_slice(), &[0u8][..]].concat());
}
