fn main() -> anyhow::Result<()> {
    let addr = std::env::var("DOPPEL_ADDR").unwrap_or_else(|_| "0.0.0.0:25565".into());
    let pin_path = std::env::var("DOPPEL_PIN").ok();
    doppel::serve(&addr, pin_path.as_deref().map(std::path::Path::new))
}
