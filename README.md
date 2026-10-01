# Doppel

A Minecraft server written in Rust.

## Goals

- Fast, lightweight, multithreaded
- Full vanilla parity on the latest Java Edition version
- Simple to configure and run

## Status

Early development. Login and world join are implemented and verified
byte-for-byte against vanilla; the world is currently a static snapshot.

## Running

```
cargo run --release -p doppel
```

## Testing

```
cargo test --workspace
```

## License

GPL-3.0. Not affiliated with Mojang or Microsoft.
