//! Boots the real server on an ephemeral port and talks the actual wire
//! protocol to it — handshake, status, ping/pong.

use doppel_protocol::{encode_handshake, encode_ping, encode_status_request, read_packet, Reader};
use std::io::Write;
use std::net::TcpStream;

fn test_pin() -> doppel_protocol::Pin {
    serde_json::from_str(
        r#"{
            "id": "26.3",
            "release_time": "2026-09-15T00:00:00+00:00",
            "server_jar_url": "https://example.invalid/server.jar",
            "server_jar_sha1": "da39a3ee5e6b4b0d3255bfef95601890afd80709",
            "java_major": 25,
            "protocol": 777,
            "version_name": "26.3"
        }"#,
    )
    .expect("valid pin fixture")
}

#[test]
fn status_ping_roundtrip() {
    let pin = test_pin();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn({
        let pin = pin.clone();
        move || {
            doppel::serve_on(listener, pin, None, None).unwrap();
        }
    });

    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .unwrap();
    stream
        .write_all(&encode_handshake(777, "127.0.0.1", port, 1))
        .unwrap();
    stream.write_all(&encode_status_request()).unwrap();

    let (id, body) = read_packet(&mut stream, 1024 * 1024).unwrap();
    assert_eq!(id, 0x00);
    let mut r = Reader::new(&body);
    let json: serde_json::Value =
        serde_json::from_str(&r.read_string(1024 * 1024).unwrap()).unwrap();

    assert_eq!(json["version"]["name"], "26.3");
    assert_eq!(json["version"]["protocol"], 777);
    assert_eq!(json["players"]["max"], 20);
    assert_eq!(json["players"]["online"], 0);
    assert_eq!(json["description"], "A Minecraft Server");

    let payload = 0x00D0_BB50_0000_0001i64;
    stream.write_all(&encode_ping(payload)).unwrap();
    let (id, body) = read_packet(&mut stream, 1024).unwrap();
    assert_eq!(id, 0x01);
    assert_eq!(Reader::new(&body).read_i64().unwrap(), payload);
}
