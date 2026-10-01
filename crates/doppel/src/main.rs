fn main() -> anyhow::Result<()> {
    let host = std::env::var("DOPPEL_ADDR").unwrap_or_else(|_| "0.0.0.0".into());
    let port = std::env::var("DOPPEL_PORT").unwrap_or_else(|_| "25565".into());
    let addr = format!("{host}:{port}");
    let pin_path = std::env::var("DOPPEL_PIN").ok();
    doppel::serve(&addr, pin_path.as_deref().map(std::path::Path::new))
}
