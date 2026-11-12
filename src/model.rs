use anyhow::{bail, Context, Result};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;

const HF_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// Model default untuk web (akurat tapi masih muat di GPU 6 GB).
pub const DEFAULT: &str = "large-v3-q5_0";

pub const NAMES: &[&str] = &[
    "large-v3-q5_0",
    "large-v3-turbo",
    "large-v3-turbo-q8_0",
    "large-v3-turbo-q5_0",
    "large-v3",
    "medium",
    "small",
    "base",
];

/// Nama model -> nama file ggml di Hugging Face.
fn model_file(name: &str) -> Option<&'static str> {
    match name {
        "large-v3" => Some("ggml-large-v3.bin"),
        "large-v3-q5_0" => Some("ggml-large-v3-q5_0.bin"),
        "large-v3-turbo" => Some("ggml-large-v3-turbo.bin"),
        "large-v3-turbo-q5_0" => Some("ggml-large-v3-turbo-q5_0.bin"),
        "large-v3-turbo-q8_0" => Some("ggml-large-v3-turbo-q8_0.bin"),
        "medium" => Some("ggml-medium.bin"),
        "small" => Some("ggml-small.bin"),
        "base" => Some("ggml-base.bin"),
        _ => None,
    }
}

pub fn supported() -> String {
    NAMES.join(", ")
}

fn models_dir() -> Result<PathBuf> {
    Ok(dirs::cache_dir()
        .context("tidak bisa menemukan direktori cache")?
        .join("subgen")
        .join("models"))
}

/// Apakah model sudah ada di cache lokal (tanpa mengunduh).
pub fn is_cached(name: &str) -> bool {
    match (model_file(name), models_dir()) {
        (Some(f), Ok(d)) => d.join(f).exists(),
        _ => false,
    }
}

/// Pastikan model ada di cache lokal; kalau belum, download dari Hugging Face.
pub fn ensure_model(name: &str) -> Result<PathBuf> {
    let file = match model_file(name) {
        Some(f) => f,
        None => bail!("model '{}' tidak dikenal. Pilihan: {}", name, supported()),
    };

    let dir = models_dir()?;
    fs::create_dir_all(&dir)?;

    let path = dir.join(file);
    if path.exists() {
        return Ok(path);
    }

    let url = format!("{HF_BASE}/{file}");
    eprintln!("Model belum ada, mengunduh {file} dari Hugging Face...");
    download(&url, &path)?;
    Ok(path)
}

fn download(url: &str, dest: &PathBuf) -> Result<()> {
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("gagal mengunduh {url}"))?;

    let total: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());
    let mut reader = resp.into_reader();

    // Tulis ke file .part dulu supaya download yang terputus tidak dianggap model valid.
    let tmp = dest.with_extension("bin.part");
    let mut out = File::create(&tmp)?;
    let mut buf = vec![0u8; 256 * 1024];
    let mut done: u64 = 0;
    let mut last_pct = u64::MAX;

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;

        match total {
            Some(t) if t > 0 => {
                let pct = done * 100 / t;
                if pct != last_pct {
                    eprint!("\r  {}% ({} / {} MB)", pct, done / 1_048_576, t / 1_048_576);
                    last_pct = pct;
                }
            }
            _ => eprint!("\r  {} MB", done / 1_048_576),
        }
    }
    eprintln!();

    out.flush()?;
    drop(out);
    fs::rename(&tmp, dest)?;
    Ok(())
}
