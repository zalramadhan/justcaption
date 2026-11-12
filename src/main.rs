use subgen::{audio, model, srt, transcribe};

use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "subgen", version, about = "Generate subtitle (.srt) lokal dengan Whisper (GPU)")]
struct Cli {
    /// File video/audio input (mp4, mov, mkv, mp3, wav, dll.)
    input: PathBuf,

    /// File .srt output (default: sama dengan input, ekstensi .srt)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Nama model: large-v3-turbo (default: large-v3-q5_0), large-v3-turbo-q8_0, large-v3-turbo-q5_0, large-v3, medium, small, base
    #[arg(short, long, default_value = "large-v3-q5_0")]
    model: String,

    /// Pakai file model ggml sendiri (mengabaikan --model)
    #[arg(long)]
    model_path: Option<PathBuf>,

    /// Kode bahasa
    #[arg(short, long, default_value = "id")]
    language: String,

    /// Beam size (1 = greedy/cepat, 5 = lebih akurat)
    #[arg(long, default_value_t = 5)]
    beam_size: i32,

    /// Jumlah thread CPU
    #[arg(long)]
    threads: Option<i32>,

    /// Prompt awal untuk membantu nama/istilah khusus, mis. "Percakapan tentang Rust dan Kubernetes."
    #[arg(long)]
    prompt: Option<String>,

    /// Maks. kata per segmen subtitle (0 = otomatis per kalimat, maks. 12)
    #[arg(long, default_value_t = 0)]
    max_words: usize,

    /// Maksimal karakter per baris subtitle
    #[arg(long, default_value_t = 42)]
    max_line_chars: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let started = Instant::now();

    if !cli.input.exists() {
        anyhow::bail!("file tidak ditemukan: {}", cli.input.display());
    }

    // 1. Model
    let model_path = match &cli.model_path {
        Some(p) => p.clone(),
        None => model::ensure_model(&cli.model)?,
    };
    eprintln!("Model   : {}", model_path.display());

    // 2. Audio
    eprintln!("Ekstrak audio dengan ffmpeg...");
    let samples = audio::extract(&cli.input)?;
    eprintln!("Durasi  : {:.1} detik", audio::duration_secs(&samples));

    // 3. Transkripsi
    let threads = cli.threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .min(8)
    });
    let opts = transcribe::Options {
        language: &cli.language,
        beam_size: cli.beam_size,
        threads,
        prompt: cli.prompt.as_deref(),
        max_words: cli.max_words,
    };
    let segments = transcribe::transcribe(&model_path, &samples, &opts)?;

    // 4. Tulis SRT
    let out_path = cli
        .output
        .clone()
        .unwrap_or_else(|| cli.input.with_extension("srt"));
    fs::write(&out_path, srt::to_srt(&segments, cli.max_line_chars))
        .with_context(|| format!("gagal menulis {}", out_path.display()))?;

    eprintln!(
        "Selesai : {} segmen -> {} ({:.1}s)",
        segments.len(),
        out_path.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
