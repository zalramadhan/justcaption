use anyhow::{bail, Result};
use std::io::ErrorKind;
use std::path::Path;
use std::process::{Command, Stdio};

/// Sample rate yang diharapkan Whisper.
pub const SAMPLE_RATE: usize = 16_000;

/// Ekstrak audio dari file video/audio apa pun menjadi PCM f32, mono, 16 kHz.
/// Memakai ffmpeg dan membaca hasilnya langsung dari stdout (tanpa file sementara).
pub fn extract(input: &Path) -> Result<Vec<f32>> {
    let output = Command::new("ffmpeg")
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(input)
        .args(["-vn", "-ac", "1", "-ar"])
        .arg(SAMPLE_RATE.to_string())
        .args(["-f", "f32le", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| {
            if e.kind() == ErrorKind::NotFound {
                anyhow::anyhow!("ffmpeg tidak ditemukan. Install dulu: brew install ffmpeg")
            } else {
                anyhow::Error::new(e).context("gagal menjalankan ffmpeg")
            }
        })?;

    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        bail!("ffmpeg gagal:\n{}", err.trim());
    }

    if output.stdout.is_empty() {
        bail!("tidak ada audio yang berhasil diekstrak (apakah file punya track audio?)");
    }

    let samples = output
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect::<Vec<f32>>();

    Ok(samples)
}

/// Durasi audio dalam detik.
pub fn duration_secs(samples: &[f32]) -> f64 {
    samples.len() as f64 / SAMPLE_RATE as f64
}
