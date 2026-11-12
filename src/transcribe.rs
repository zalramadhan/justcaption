use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
pub use whisper_rs::WhisperContext;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContextParameters};

/// Adaptor kecil: antar versi whisper-rs, getter segmen kadang mengembalikan
/// nilai langsung, kadang `Result`. Trait ini menyeragamkan keduanya.
trait Ok2<T> {
    fn ok2(self) -> Result<T>;
}

impl<T> Ok2<T> for T {
    fn ok2(self) -> Result<T> {
        Ok(self)
    }
}

impl<T, E> Ok2<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn ok2(self) -> Result<T> {
        self.map_err(anyhow::Error::from)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Word {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    /// Timing per kata (kosong kalau tidak tersedia / sudah diedit manual).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<Word>,
}

pub struct Options<'a> {
    pub language: &'a str,
    pub beam_size: i32,
    pub threads: i32,
    pub prompt: Option<&'a str>,
    /// Maks. kata per segmen (0 = otomatis per kalimat, maks. 12).
    pub max_words: usize,
}

/// Muat model ke GPU/CPU. Context bisa dipakai ulang untuk banyak transkripsi.
pub fn load_model(model_path: &Path) -> Result<WhisperContext> {
    let s = model_path
        .to_str()
        .context("path model bukan UTF-8 yang valid")?;
    WhisperContext::new_with_params(s, WhisperContextParameters::default())
        .with_context(|| format!("gagal memuat model: {s}"))
}

/// Jalankan transkripsi. `progress` dipanggil dengan persen 0..100.
/// Hasilnya sudah dikelompokkan jadi segmen kalimat dengan timing per kata.
pub fn run(
    ctx: &WhisperContext,
    audio: &[f32],
    opts: &Options,
    mut progress: impl FnMut(i32) + 'static,
) -> Result<Vec<Segment>> {
    let mut state = ctx.create_state().context("gagal membuat state whisper")?;

    let strategy = if opts.beam_size > 1 {
        SamplingStrategy::BeamSearch {
            beam_size: opts.beam_size,
            patience: -1.0,
        }
    } else {
        SamplingStrategy::Greedy { best_of: 1 }
    };

    let mut params = FullParams::new(strategy);
    params.set_language(Some(opts.language));
    params.set_translate(false);
    params.set_n_threads(opts.threads);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_suppress_blank(true);
    // Timestamp tingkat kata: satu "segmen" whisper = satu kata.
    params.set_token_timestamps(true);
    params.set_max_len(1);
    params.set_split_on_word(true);
    if let Some(p) = opts.prompt {
        params.set_initial_prompt(p);
    }
    params.set_progress_callback_safe(move |p: i32| progress(p));

    state
        .full(params, audio)
        .context("proses transkripsi gagal")?;

    let mut words: Vec<Word> = Vec::new();
    let n: i32 = state.full_n_segments().ok2()?;
    for i in 0..n {
        let text: String = state.full_get_segment_text(i).ok2()?;
        // Timestamp whisper dalam satuan 10 ms (centisecond).
        let t0: i64 = state.full_get_segment_t0(i).ok2()?;
        let t1: i64 = state.full_get_segment_t1(i).ok2()?;
        push_words(&mut words, &text, t0 * 10, t1 * 10);
    }

    refine_boundaries(&mut words, audio);
    Ok(group_words(words, opts.max_words))
}

/// Versi CLI: tampilkan progress di stderr dan gagal kalau tidak ada ucapan.
pub fn transcribe(model_path: &Path, audio: &[f32], opts: &Options) -> Result<Vec<Segment>> {
    let ctx = load_model(model_path)?;
    let segments = run(&ctx, audio, opts, |p| {
        eprint!("\r  transkripsi: {p}%   ");
    })?;
    eprintln!("\r  transkripsi: 100%   ");
    if segments.is_empty() {
        bail!("tidak ada ucapan yang terdeteksi");
    }
    Ok(segments)
}

/// Tambah satu kata; tanda baca yang berdiri sendiri digabung ke kata sebelumnya.
fn add_word(out: &mut Vec<Word>, text: &str, start: i64, end: i64) {
    let has_alnum = text.chars().any(|c| c.is_alphanumeric());
    if !has_alnum {
        if let Some(last) = out.last_mut() {
            last.text.push_str(text);
            last.end_ms = last.end_ms.max(end);
            return;
        }
    }
    out.push(Word {
        start_ms: start,
        end_ms: end.max(start),
        text: text.to_string(),
    });
}

/// Pecah teks satu segmen whisper menjadi kata (biasanya sudah satu kata).
fn push_words(out: &mut Vec<Word>, text: &str, start: i64, end: i64) {
    let toks: Vec<&str> = text.split_whitespace().collect();
    if toks.is_empty() {
        return;
    }
    let end = end.max(start);
    if toks.len() == 1 {
        add_word(out, toks[0], start, end);
        return;
    }
    let total: i64 = toks.iter().map(|t| t.chars().count() as i64 + 1).sum();
    let dur = end - start;
    let mut acc: i64 = 0;
    for t in toks {
        let a = start + dur * acc / total;
        acc += t.chars().count() as i64 + 1;
        let b = start + dur * acc / total;
        add_word(out, t, a, b);
    }
}

/// Koreksi batas tiap kata dengan energi audio (frame 10 ms) memakai ambang lokal per kata:
/// - awal kata jatuh di bagian sunyi -> geser maju ke onset suara (maks. 1,2 dtk);
/// - awal kata jatuh di dalam suara yang didahului sunyi -> geser mundur ke awal suara
///   (maks. 300 ms, tidak melewati akhir kata sebelumnya);
/// - akhir kata jatuh di bagian sunyi -> mundurkan ke suara terakhir + 50 ms (maks. 1,2 dtk).
/// Ucapan yang menyambung tanpa jeda sunyi tidak diubah: energi audio tidak bisa memisahkan kata di situ.
fn refine_boundaries(words: &mut [Word], audio: &[f32]) {
    const FRAME: usize = 160; // 10 ms pada 16 kHz
    let n = audio.len() / FRAME;
    if n < 20 {
        return;
    }
    // energi dB per frame, dihaluskan 3 frame
    let raw: Vec<f32> = (0..n)
        .map(|i| {
            let f = &audio[i * FRAME..(i + 1) * FRAME];
            let rms = (f.iter().map(|x| x * x).sum::<f32>() / FRAME as f32).sqrt();
            20.0 * (rms + 1e-6).log10()
        })
        .collect();
    let db: Vec<f32> = (0..n)
        .map(|i| {
            let a = i.saturating_sub(1);
            let b = (i + 1).min(n - 1);
            (raw[a] + raw[i] + raw[b]) / 3.0
        })
        .collect();

    let mut prev_end: usize = 0;
    for w in words.iter_mut() {
        let s = (w.start_ms.max(0) / 10) as usize;
        let e = ((w.end_ms.max(0) / 10) as usize).min(n);
        if s >= n || e <= s + 8 {
            prev_end = prev_end.max(e);
            continue;
        }

        // ambang lokal dari sekitar kata (+-1 detik): lantai noise vs suara keras
        let lo = s.saturating_sub(100);
        let hi = (e + 100).min(n);
        let mut win: Vec<f32> = db[lo..hi].to_vec();
        win.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let floor = win[win.len() * 15 / 100];
        let peak = win[win.len() * 90 / 100];
        if peak - floor < 12.0 {
            // tidak ada kontras suara/sunyi yang jelas: jangan diubah
            prev_end = e;
            continue;
        }
        let thr = (floor + 8.0).max(peak - 28.0);
        let voiced = |k: usize| db[k] > thr;

        let mut ns = s;
        let mut ne = e;

        if !voiced(s) {
            // awal di sunyi: cari onset (3 frame berturut-turut bersuara)
            let limit = (s + 120).min(e - 8);
            let mut k = s;
            while k + 3 <= limit {
                if voiced(k) && voiced(k + 1) && voiced(k + 2) {
                    ns = k;
                    break;
                }
                k += 1;
            }
        } else {
            // awal di dalam suara: mundur hanya kalau memang ada sunyi di belakangnya
            let bound = prev_end.max(s.saturating_sub(30));
            let mut j = s;
            while j > bound && voiced(j - 1) {
                j -= 1;
            }
            if j > bound {
                ns = j;
            }
        }

        if !voiced(e - 1) {
            // akhir di sunyi: mundur ke suara terakhir
            let floor_k = e.saturating_sub(120).max(ns + 8);
            let mut k = e;
            while k > floor_k && !voiced(k - 1) {
                k -= 1;
            }
            if k > floor_k {
                ne = (k + 5).min(e);
            }
        }

        if ne > ns + 8 {
            w.start_ms = (ns * 10) as i64;
            w.end_ms = (ne * 10) as i64;
            prev_end = ne;
        } else {
            prev_end = e;
        }
    }
}

/// Kelompokkan kata menjadi segmen yang enak diedit. Putus di akhir kalimat atau jeda >= 0,8 detik,
/// lalu tiap potongan dibagi rata menjadi segmen berisi maksimal `max_words` kata (0 = 12),
/// jadi 7 kata dengan batas 3 menjadi 3+2+2, bukan 3+3+1.
fn group_words(words: Vec<Word>, max_words: usize) -> Vec<Segment> {
    const GAP_MS: i64 = 800;
    let max_words = if max_words == 0 { 12 } else { max_words };

    let mut segs: Vec<Segment> = Vec::new();
    let mut run: Vec<Word> = Vec::new();

    for (i, w) in words.iter().enumerate() {
        run.push(w.clone());
        let ends_sentence = w
            .text
            .ends_with(|c: char| matches!(c, '.' | '?' | '!' | '…'));
        let gap = words
            .get(i + 1)
            .map(|n| n.start_ms - w.end_ms >= GAP_MS)
            .unwrap_or(true);
        if ends_sentence || gap {
            split_run(std::mem::take(&mut run), max_words, &mut segs);
        }
    }
    segs
}

fn split_run(run: Vec<Word>, max_words: usize, segs: &mut Vec<Segment>) {
    if run.is_empty() {
        return;
    }
    let k = (run.len() + max_words - 1) / max_words;
    let base = run.len() / k;
    let rem = run.len() % k;
    let mut it = run.into_iter();
    for c in 0..k {
        let size = base + if c < rem { 1 } else { 0 };
        let chunk: Vec<Word> = it.by_ref().take(size).collect();
        let text = chunk
            .iter()
            .map(|x| x.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        segs.push(Segment {
            start_ms: chunk[0].start_ms,
            end_ms: chunk[chunk.len() - 1].end_ms,
            text,
            words: chunk,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(s: i64, e: i64, t: &str) -> Word {
        Word {
            start_ms: s,
            end_ms: e,
            text: t.into(),
        }
    }

    #[test]
    fn groups_on_sentence_end() {
        let g = group_words(vec![
            w(0, 300, "Halo"),
            w(300, 600, "dunia."),
            w(700, 900, "Apa"),
            w(900, 1200, "kabar"),
        ], 12);
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].text, "Halo dunia.");
        assert_eq!(g[1].words.len(), 2);
    }

    #[test]
    fn respects_max_words() {
        let g = group_words(
            vec![w(0, 100, "a"), w(100, 200, "b"), w(200, 300, "c"), w(300, 400, "d"), w(400, 500, "e")],
            2,
        );
        assert_eq!(g.len(), 3);
        assert_eq!(g[0].text, "a b");
        assert_eq!(g[2].text, "e");
    }

    #[test]
    fn splits_evenly() {
        let ws: Vec<Word> = (0..7).map(|i| w(i * 100, i * 100 + 90, "x")).collect();
        let g = group_words(ws, 3);
        let sizes: Vec<usize> = g.iter().map(|x| x.words.len()).collect();
        assert_eq!(sizes, vec![3, 2, 2]);
    }

    /// Audio 16 kHz: sunyi (derau kecil) dengan blok "suara" pada rentang detik tertentu.
    fn synth(total_s: usize, speech: &[(f32, f32)]) -> Vec<f32> {
        let mut a = vec![0.0005f32; total_s * 16_000];
        for &(from, to) in speech {
            let end = ((to * 16_000.0) as usize).min(a.len());
            for i in (from * 16_000.0) as usize..end {
                a[i] = if i % 2 == 0 { 0.3 } else { -0.3 };
            }
        }
        a
    }

    fn word(s: i64, e: i64) -> Word {
        Word {
            start_ms: s,
            end_ms: e,
            text: "x".into(),
        }
    }

    fn near(a: i64, b: i64) -> bool {
        (a - b).abs() <= 10
    }

    #[test]
    fn snaps_early_start_forward() {
        let mut v = vec![word(400, 2000)];
        refine_boundaries(&mut v, &synth(3, &[(1.0, 2.0)]));
        assert!(near(v[0].start_ms, 1000), "{}", v[0].start_ms);
        assert!(near(v[0].end_ms, 2000), "{}", v[0].end_ms);
    }

    #[test]
    fn snaps_late_start_backward() {
        let mut v = vec![word(1100, 2000)];
        refine_boundaries(&mut v, &synth(3, &[(1.0, 2.0)]));
        assert!(near(v[0].start_ms, 1000), "{}", v[0].start_ms);
    }

    #[test]
    fn keeps_continuous_speech() {
        let mut v = vec![word(1000, 2000)];
        refine_boundaries(&mut v, &synth(3, &[(0.0, 3.0)]));
        assert_eq!((v[0].start_ms, v[0].end_ms), (1000, 2000));
    }

    #[test]
    fn trims_hanging_end() {
        let mut v = vec![word(1000, 2500)];
        refine_boundaries(&mut v, &synth(3, &[(1.0, 2.0)]));
        assert!(v[0].end_ms >= 2040 && v[0].end_ms <= 2060, "{}", v[0].end_ms);
    }

    #[test]
    fn refines_two_words_around_a_pause() {
        let mut v = vec![word(500, 1500), word(1600, 3000)];
        refine_boundaries(&mut v, &synth(4, &[(0.6, 1.4), (2.2, 2.9)]));
        assert!(near(v[0].start_ms, 600), "{}", v[0].start_ms);
        assert!(near(v[1].start_ms, 2200), "{}", v[1].start_ms);
    }

    #[test]
    fn merges_lone_punctuation() {
        let mut v = vec![];
        push_words(&mut v, " halo", 0, 300);
        push_words(&mut v, ",", 300, 320);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].text, "halo,");
        assert_eq!(v[0].end_ms, 320);
    }
}
