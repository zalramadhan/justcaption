//! Server web lokal: unggah video, transkripsi (Whisper), preview & edit caption.

use anyhow::{Context, Result};
use axum::{
    extract::{DefaultBodyLimit, Multipart, Path, Request, State},
    http::{header, HeaderValue, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Json, Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use subgen::transcribe::{self, Segment, WhisperContext};
use subgen::{audio, model};
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;
use tower::ServiceExt;
use tower_http::services::ServeFile;

#[derive(Parser)]
#[command(name = "subgen-web", version, about = "Editor subtitle web lokal (Whisper)")]
struct Cli {
    /// Port server
    #[arg(long, default_value_t = 3000)]
    port: u16,

    /// Alamat bind (default hanya lokal)
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Folder penyimpanan proyek (default: ~/.local/share/subgen)
    #[arg(long)]
    data_dir: Option<PathBuf>,

    /// Mode aplikasi: buka jendela tanpa tab/address bar (browser Chromium). Menutup jendela menghentikan server.
    #[arg(long)]
    app: bool,
}

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Status {
    Queued,
    Working,
    Done,
    Failed,
}

fn default_beam() -> i32 {
    5
}

fn default_max_words() -> usize {
    3
}

#[derive(Serialize, Deserialize, Clone)]
struct JobMeta {
    id: String,
    filename: String,
    media_file: String,
    created: u64,
    model: String,
    language: String,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default = "default_beam")]
    beam: i32,
    #[serde(default = "default_max_words")]
    max_words: usize,
    status: Status,
    #[serde(default)]
    progress: i32,
    #[serde(default)]
    message: String,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    duration: f64,
    /// Versi pemrosesan timing (>= 2: batas kata sudah dikoreksi dengan energi audio; 3: ambang lokal, dua arah).
    #[serde(default)]
    timing: u8,
    #[serde(default)]
    segments: Vec<Segment>,
}

struct AppState {
    data_dir: PathBuf,
    jobs: Mutex<HashMap<String, JobMeta>>,
    model_cache: Mutex<Option<(String, Arc<WhisperContext>)>>,
    /// Satu transkripsi pada satu waktu (GPU terbatas).
    gpu: Semaphore,
    /// Status export video per proyek (di memori).
    exports: Mutex<HashMap<String, ExportState>>,
    /// Satu export pada satu waktu.
    exp: Semaphore,
}

type Shared = Arc<AppState>;

// ---------- error ----------

struct AppError(StatusCode, String);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

fn bad(e: impl std::fmt::Display) -> AppError {
    AppError(StatusCode::BAD_REQUEST, e.to_string())
}
fn internal(e: impl std::fmt::Display) -> AppError {
    AppError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}
fn not_found() -> AppError {
    AppError(StatusCode::NOT_FOUND, "proyek tidak ditemukan".into())
}

// ---------- util ----------

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn new_id() -> String {
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:x}{:x}", n, COUNTER.fetch_add(1, Ordering::Relaxed))
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn ext_of(name: &str) -> String {
    std::path::Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            e.chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .take(5)
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|e| !e.is_empty())
        .unwrap_or_else(|| "mp4".to_string())
}

fn update(st: &AppState, id: &str, f: impl FnOnce(&mut JobMeta)) {
    if let Some(j) = st.jobs.lock().unwrap().get_mut(id) {
        f(j);
    }
}

fn persist(st: &AppState, id: &str) {
    let job = st.jobs.lock().unwrap().get(id).cloned();
    if let Some(job) = job {
        let dir = st.data_dir.join("jobs").join(id);
        let _ = std::fs::create_dir_all(&dir);
        if let Ok(bytes) = serde_json::to_vec_pretty(&job) {
            let _ = std::fs::write(dir.join("job.json"), bytes);
        }
    }
}

fn load_jobs(data_dir: &PathBuf) -> HashMap<String, JobMeta> {
    let mut map = HashMap::new();
    if let Ok(rd) = std::fs::read_dir(data_dir.join("jobs")) {
        for entry in rd.flatten() {
            let path = entry.path().join("job.json");
            if let Ok(bytes) = std::fs::read(&path) {
                if let Ok(mut j) = serde_json::from_slice::<JobMeta>(&bytes) {
                    if matches!(j.status, Status::Queued | Status::Working) {
                        j.status = Status::Failed;
                        j.error = Some("Terputus karena server berhenti.".into());
                    }
                    map.insert(j.id.clone(), j);
                }
            }
        }
    }
    map
}

// ---------- pemrosesan ----------

fn get_context(st: &AppState, name: &str) -> Result<Arc<WhisperContext>> {
    let mut cache = st.model_cache.lock().unwrap();
    if let Some((n, c)) = &*cache {
        if n == name {
            return Ok(c.clone());
        }
    }
    *cache = None; // bebaskan VRAM sebelum memuat model lain
    let path = model::ensure_model(name)?;
    let ctx = Arc::new(transcribe::load_model(&path)?);
    *cache = Some((name.to_string(), ctx.clone()));
    Ok(ctx)
}

fn process(st: &Shared, id: &str) -> Result<()> {
    let (media, model_name, language, prompt, beam, max_words) = {
        let jobs = st.jobs.lock().unwrap();
        let j = jobs.get(id).context("proyek sudah dihapus")?;
        (
            st.data_dir.join("jobs").join(id).join(&j.media_file),
            j.model.clone(),
            j.language.clone(),
            j.prompt.clone(),
            j.beam,
            j.max_words,
        )
    };

    update(st, id, |j| {
        j.status = Status::Working;
        j.progress = 0;
        j.message = if model::is_cached(&model_name) {
            "Memuat model…".into()
        } else {
            "Mengunduh model (pertama kali, bisa lama)…".into()
        };
    });
    let ctx = get_context(st, &model_name)?;

    update(st, id, |j| j.message = "Mengekstrak audio…".into());
    let samples = audio::extract(&media)?;
    let duration = audio::duration_secs(&samples);
    update(st, id, |j| {
        j.duration = duration;
        j.message = "Transkripsi…".into();
    });

    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .min(8);
    let opts = transcribe::Options {
        language: &language,
        beam_size: beam,
        threads,
        prompt: prompt.as_deref(),
        max_words,
    };

    let st2 = st.clone();
    let id2 = id.to_string();
    let segments = transcribe::run(&ctx, &samples, &opts, move |p| {
        update(&st2, &id2, |j| j.progress = p);
    })?;

    update(st, id, |j| {
        j.segments = segments;
        j.timing = 3;
        j.status = Status::Done;
        j.progress = 100;
        j.message = "Selesai".into();
    });
    persist(st, id);
    Ok(())
}

async fn run_job(st: Shared, id: String) {
    let _permit = st.gpu.acquire().await.unwrap();

    let st2 = st.clone();
    let id2 = id.clone();
    let result = tokio::task::spawn_blocking(move || process(&st2, &id2)).await;

    let failure: Option<String> = match result {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(format!("{e:#}")),
        Err(e) => Some(format!("proses berhenti tak terduga: {e}")),
    };
    if let Some(msg) = failure {
        update(&st, &id, |j| {
            j.status = Status::Failed;
            j.message = "Gagal".into();
            j.error = Some(msg);
        });
        persist(&st, &id);
    }
}

// ---------- handler ----------

async fn index() -> Html<&'static str> {
    Html(include_str!("../../web/index.html"))
}

async fn config() -> Json<Value> {
    Json(json!({ "models": model::NAMES, "default": model::DEFAULT }))
}

struct Upload {
    filename: String,
    media_file: String,
    model: String,
    language: String,
    prompt: Option<String>,
    beam: i32,
    max_words: usize,
}

async fn parse_upload(mp: &mut Multipart, dir: &std::path::Path) -> Result<Upload, AppError> {
    let mut up = Upload {
        filename: String::new(),
        media_file: String::new(),
        model: model::DEFAULT.to_string(),
        language: "id".to_string(),
        prompt: None,
        beam: 5,
        max_words: 3,
    };

    while let Some(mut field) = mp.next_field().await.map_err(bad)? {
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => {
                let orig = field.file_name().unwrap_or("video.mp4").to_string();
                let stored = format!("media.{}", ext_of(&orig));
                let mut f = tokio::fs::File::create(dir.join(&stored))
                    .await
                    .map_err(internal)?;
                while let Some(chunk) = field.chunk().await.map_err(bad)? {
                    f.write_all(&chunk).await.map_err(internal)?;
                }
                f.flush().await.map_err(internal)?;
                up.filename = orig;
                up.media_file = stored;
            }
            "model" => up.model = field.text().await.map_err(bad)?,
            "language" => up.language = field.text().await.map_err(bad)?,
            "prompt" => {
                let p = field.text().await.map_err(bad)?;
                let p = p.trim().chars().take(500).collect::<String>();
                up.prompt = if p.is_empty() { None } else { Some(p) };
            }
            "maxwords" => {
                let v = field.text().await.map_err(bad)?;
                up.max_words = v.trim().parse::<usize>().unwrap_or(3).min(30);
            }
            "beam" => {
                let b = field.text().await.map_err(bad)?;
                up.beam = b.trim().parse::<i32>().unwrap_or(5).clamp(1, 10);
            }
            _ => {}
        }
    }

    if up.media_file.is_empty() {
        return Err(bad("file video/audio wajib diisi"));
    }
    if !model::NAMES.contains(&up.model.as_str()) {
        return Err(bad(format!("model tidak dikenal: {}", up.model)));
    }
    let lang = up.language.trim().to_lowercase();
    up.language = if !lang.is_empty()
        && lang.len() <= 8
        && lang.chars().all(|c| c.is_ascii_alphabetic() || c == '-')
    {
        lang
    } else {
        "id".to_string()
    };
    Ok(up)
}

async fn create_job(
    State(st): State<Shared>,
    mut mp: Multipart,
) -> Result<Json<Value>, AppError> {
    let id = new_id();
    let dir = st.data_dir.join("jobs").join(&id);
    tokio::fs::create_dir_all(&dir).await.map_err(internal)?;

    let up = match parse_upload(&mut mp, &dir).await {
        Ok(u) => u,
        Err(e) => {
            let _ = tokio::fs::remove_dir_all(&dir).await;
            return Err(e);
        }
    };

    let job = JobMeta {
        id: id.clone(),
        filename: up.filename,
        media_file: up.media_file,
        created: now_secs(),
        model: up.model,
        language: up.language,
        prompt: up.prompt,
        beam: up.beam,
        max_words: up.max_words,
        status: Status::Queued,
        progress: 0,
        message: "Menunggu giliran…".into(),
        error: None,
        duration: 0.0,
        timing: 0,
        segments: vec![],
    };
    st.jobs.lock().unwrap().insert(id.clone(), job);
    persist(&st, &id);
    tokio::spawn(run_job(st.clone(), id.clone()));

    Ok(Json(json!({ "id": id })))
}

async fn list_jobs(State(st): State<Shared>) -> Json<Vec<Value>> {
    let mut v: Vec<Value> = st
        .jobs
        .lock()
        .unwrap()
        .values()
        .map(|j| {
            json!({
                "id": j.id,
                "filename": j.filename,
                "status": j.status,
                "progress": j.progress,
                "duration": j.duration,
                "created": j.created,
            })
        })
        .collect();
    v.sort_by(|a, b| b["created"].as_u64().cmp(&a["created"].as_u64()));
    Json(v)
}

async fn get_job(
    State(st): State<Shared>,
    Path(id): Path<String>,
) -> Result<Json<JobMeta>, AppError> {
    let job = st.jobs.lock().unwrap().get(&id).cloned();
    job.map(Json).ok_or_else(not_found)
}

#[derive(Deserialize)]
struct SaveBody {
    segments: Vec<Segment>,
}

async fn put_job(
    State(st): State<Shared>,
    Path(id): Path<String>,
    Json(body): Json<SaveBody>,
) -> Result<Json<Value>, AppError> {
    let mut segs: Vec<Segment> = body
        .segments
        .into_iter()
        .map(|mut s| {
            s.start_ms = s.start_ms.max(0);
            s.end_ms = s.end_ms.max(s.start_ms);
            s.text = s.text.trim().to_string();
            s
        })
        .filter(|s| !s.text.is_empty())
        .collect();
    segs.sort_by_key(|s| s.start_ms);

    {
        let mut jobs = st.jobs.lock().unwrap();
        let j = jobs.get_mut(&id).ok_or_else(not_found)?;
        if j.status != Status::Done {
            return Err(bad("proyek belum selesai diproses"));
        }
        j.segments = segs;
    }
    persist(&st, &id);
    Ok(Json(json!({ "ok": true })))
}

async fn delete_job(
    State(st): State<Shared>,
    Path(id): Path<String>,
) -> Result<StatusCode, AppError> {
    let removed = st.jobs.lock().unwrap().remove(&id);
    if removed.is_none() {
        return Err(not_found());
    }
    let _ = tokio::fs::remove_dir_all(st.data_dir.join("jobs").join(&id)).await;
    Ok(StatusCode::NO_CONTENT)
}

/// Stream video dengan dukungan HTTP Range (supaya bisa di-seek di browser).
async fn media(State(st): State<Shared>, Path(id): Path<String>, req: Request) -> Response {
    let path = {
        let jobs = st.jobs.lock().unwrap();
        match jobs.get(&id) {
            Some(j) => st.data_dir.join("jobs").join(&id).join(&j.media_file),
            None => return StatusCode::NOT_FOUND.into_response(),
        }
    };
    match ServeFile::new(path).oneshot(req).await {
        Ok(res) => res.into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

// ---------- export video (burn-in caption) ----------

#[derive(Serialize, Clone)]
struct ExportState {
    state: String, // idle | working | done | failed
    progress: i32,
    message: String,
    error: Option<String>,
    ext: String,
}

impl ExportState {
    fn idle() -> Self {
        ExportState {
            state: "idle".into(),
            progress: 0,
            message: String::new(),
            error: None,
            ext: String::new(),
        }
    }
}

fn export_update(st: &AppState, id: &str, f: impl FnOnce(&mut ExportState)) {
    if let Some(e) = st.exports.lock().unwrap().get_mut(id) {
        f(e);
    }
}

#[derive(Deserialize)]
struct ExportReq {
    /// Isi file .ass yang sudah dibuat di browser (gaya + animasi kata).
    ass: String,
    format: String,  // mp4-h264 | mp4-h265 | webm-vp9
    quality: String, // high | mid | low
    encoder: String, // nvenc | cpu
    #[serde(default)]
    scale: i32, // 0 = resolusi asli, selain itu tinggi target
}

/// Daftar family font yang terpasang di sistem (dipakai preview browser dan ffmpeg/libass).
async fn fonts() -> Json<Vec<String>> {
    let out = tokio::task::spawn_blocking(|| {
        Command::new("fc-list")
            .args([":", "family"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
    })
    .await
    .ok()
    .flatten()
    .unwrap_or_default();

    let mut set = std::collections::BTreeSet::new();
    for line in out.lines() {
        if let Some(f) = line.split(',').next() {
            let f = f.trim();
            if !f.is_empty() {
                set.insert(f.to_string());
            }
        }
    }
    Json(set.into_iter().collect())
}

fn ffmpeg_args(req: &ExportReq, enc: &str, input: &str, out: &str) -> Vec<String> {
    let q = req.quality.as_str();
    let pick = |hi: &'static str, mid: &'static str, lo: &'static str| -> &'static str {
        match q {
            "high" => hi,
            "mid" => mid,
            _ => lo,
        }
    };

    let mut vf = "ass=captions.ass".to_string();
    if req.scale >= 240 {
        vf.push_str(&format!(",scale=-2:{}", req.scale.min(4320)));
    }

    let mut a: Vec<String> = ["-y", "-loglevel", "error", "-nostats", "-progress", "pipe:1", "-i"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    a.push(input.to_string());
    a.extend(
        ["-map", "0:v:0", "-map", "0:a:0?", "-vf"]
            .iter()
            .map(|s| s.to_string()),
    );
    a.push(vf);

    let vargs: Vec<&str> = match (req.format.as_str(), enc) {
        ("mp4-h264", "nvenc") => vec![
            "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq",
            pick("19", "23", "29"), "-b:v", "0", "-pix_fmt", "yuv420p",
        ],
        ("mp4-h264", _) => vec![
            "-c:v", "libx264", "-preset", "medium", "-crf",
            pick("18", "22", "28"), "-pix_fmt", "yuv420p",
        ],
        ("mp4-h265", "nvenc") => vec![
            "-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-cq",
            pick("21", "25", "31"), "-b:v", "0", "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
        ],
        ("mp4-h265", _) => vec![
            "-c:v", "libx265", "-preset", "medium", "-crf",
            pick("20", "24", "30"), "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
            "-x265-params", "log-level=error",
        ],
        _ => vec![
            "-c:v", "libvpx-vp9", "-crf", pick("28", "32", "38"),
            "-b:v", "0", "-row-mt", "1", "-pix_fmt", "yuv420p",
        ],
    };
    a.extend(vargs.iter().map(|s| s.to_string()));

    let aargs: Vec<&str> = if req.format == "webm-vp9" {
        vec!["-c:a", "libopus", "-b:a", "128k"]
    } else {
        vec!["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart"]
    };
    a.extend(aargs.iter().map(|s| s.to_string()));
    a.push(out.to_string());
    a
}

/// Jalankan ffmpeg: bakar caption (filter `ass`) ke video. Mengembalikan ekstensi file hasil.
fn run_export(st: &Shared, id: &str, req: &ExportReq) -> Result<String> {
    let (job_dir, media_name, duration) = {
        let jobs = st.jobs.lock().unwrap();
        let j = jobs.get(id).context("proyek sudah dihapus")?;
        (
            st.data_dir.join("jobs").join(id),
            j.media_file.clone(),
            j.duration,
        )
    };
    std::fs::write(job_dir.join("captions.ass"), &req.ass).context("gagal menulis captions.ass")?;

    let ext = if req.format == "webm-vp9" { "webm" } else { "mp4" };
    let out_name = format!("export.{ext}");

    // NVENC tidak untuk WebM; kalau NVENC gagal, otomatis coba CPU.
    let first = if req.format == "webm-vp9" { "cpu" } else { req.encoder.as_str() };
    let mut attempts = vec![first];
    if first == "nvenc" {
        attempts.push("cpu");
    }

    let mut last_err = String::new();
    for enc in attempts {
        export_update(st, id, |e| {
            e.progress = 0;
            e.message = if enc == "nvenc" {
                "Encoding (GPU NVENC)…".into()
            } else {
                "Encoding (CPU)…".into()
            };
        });

        // cwd = folder proyek, jadi nama file ass tidak perlu di-escape di filter ffmpeg
        let mut child = Command::new("ffmpeg")
            .current_dir(&job_dir)
            .args(ffmpeg_args(req, enc, &media_name, &out_name))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    anyhow::anyhow!("ffmpeg tidak ditemukan. Install dulu: sudo pacman -S ffmpeg")
                } else {
                    anyhow::Error::new(e).context("gagal menjalankan ffmpeg")
                }
            })?;

        let mut stderr = child.stderr.take().context("stderr ffmpeg tidak tersedia")?;
        let drain = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = stderr.read_to_string(&mut s);
            s
        });

        if let Some(stdout) = child.stdout.take() {
            for line in BufReader::new(stdout).lines().map_while(|l| l.ok()) {
                // out_time_us / out_time_ms keduanya dalam mikrodetik
                let v = line
                    .strip_prefix("out_time_us=")
                    .or_else(|| line.strip_prefix("out_time_ms="));
                if let Some(v) = v {
                    if let Ok(us) = v.trim().parse::<i64>() {
                        if duration > 0.0 {
                            let pct = (us as f64 / 1_000_000.0 / duration * 100.0) as i32;
                            export_update(st, id, |e| e.progress = pct.clamp(0, 99));
                        }
                    }
                }
            }
        }

        let status = child.wait()?;
        let err = drain.join().unwrap_or_default();
        if status.success() {
            return Ok(ext.to_string());
        }
        last_err = err.trim().to_string();
    }
    anyhow::bail!("ffmpeg gagal:\n{}", last_err)
}

async fn start_export(
    State(st): State<Shared>,
    Path(id): Path<String>,
    Json(req): Json<ExportReq>,
) -> Result<Json<Value>, AppError> {
    if !["mp4-h264", "mp4-h265", "webm-vp9"].contains(&req.format.as_str()) {
        return Err(bad("format video tidak dikenal"));
    }
    if !["high", "mid", "low"].contains(&req.quality.as_str()) {
        return Err(bad("kualitas tidak dikenal"));
    }
    if !["nvenc", "cpu"].contains(&req.encoder.as_str()) {
        return Err(bad("encoder tidak dikenal"));
    }
    {
        let jobs = st.jobs.lock().unwrap();
        let j = jobs.get(&id).ok_or_else(not_found)?;
        if j.status != Status::Done {
            return Err(bad("proyek belum selesai diproses"));
        }
    }
    {
        let mut ex = st.exports.lock().unwrap();
        if ex.get(&id).map(|e| e.state == "working").unwrap_or(false) {
            return Err(AppError(
                StatusCode::CONFLICT,
                "export untuk proyek ini sedang berjalan".into(),
            ));
        }
        ex.insert(
            id.clone(),
            ExportState {
                state: "working".into(),
                progress: 0,
                message: "Menunggu giliran…".into(),
                error: None,
                ext: String::new(),
            },
        );
    }

    let st2 = st.clone();
    let id2 = id.clone();
    tokio::spawn(async move {
        let _permit = st2.exp.acquire().await.unwrap();
        let st3 = st2.clone();
        let id3 = id2.clone();
        let result = tokio::task::spawn_blocking(move || run_export(&st3, &id3, &req)).await;
        match result {
            Ok(Ok(ext)) => export_update(&st2, &id2, |e| {
                e.state = "done".into();
                e.progress = 100;
                e.message = "Selesai".into();
                e.ext = ext;
            }),
            Ok(Err(err)) => export_update(&st2, &id2, |e| {
                e.state = "failed".into();
                e.message = "Gagal".into();
                e.error = Some(format!("{err:#}"));
            }),
            Err(err) => export_update(&st2, &id2, |e| {
                e.state = "failed".into();
                e.message = "Gagal".into();
                e.error = Some(format!("proses berhenti tak terduga: {err}"));
            }),
        }
    });

    Ok(Json(json!({ "ok": true })))
}

async fn export_status(State(st): State<Shared>, Path(id): Path<String>) -> Json<ExportState> {
    let s = st.exports.lock().unwrap().get(&id).cloned();
    Json(s.unwrap_or_else(ExportState::idle))
}

async fn export_file(State(st): State<Shared>, Path(id): Path<String>, req: Request) -> Response {
    let (path, name) = {
        let jobs = st.jobs.lock().unwrap();
        let ex = st.exports.lock().unwrap();
        match (jobs.get(&id), ex.get(&id)) {
            (Some(j), Some(e)) if e.state == "done" => {
                let base: String = std::path::Path::new(&j.filename)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("video")
                    .chars()
                    .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
                    .collect();
                (
                    st.data_dir.join("jobs").join(&id).join(format!("export.{}", e.ext)),
                    format!("{base}-captioned.{}", e.ext),
                )
            }
            _ => return StatusCode::NOT_FOUND.into_response(),
        }
    };
    match ServeFile::new(path).oneshot(req).await {
        Ok(res) => {
            let mut res = res.into_response();
            if let Ok(v) = HeaderValue::from_str(&format!("attachment; filename=\"{name}\"")) {
                res.headers_mut().insert(header::CONTENT_DISPOSITION, v);
            }
            res
        }
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

/// Buka URL sebagai jendela aplikasi (mode `--app`) memakai browser berbasis Chromium
/// dengan profil khusus, supaya prosesnya terpisah dan bisa ditunggu sampai jendela ditutup.
/// Mengembalikan None kalau tidak ada browser Chromium (lalu dibuka lewat xdg-open sebagai tab biasa).
fn open_window(url: &str, profile: &std::path::Path) -> Option<std::process::Child> {
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(b) = std::env::var("SUBGEN_BROWSER") {
        if !b.trim().is_empty() {
            candidates.push(b);
        }
    }
    for b in [
        "chromium",
        "google-chrome-stable",
        "google-chrome",
        "brave",
        "brave-browser",
        "vivaldi-stable",
        "vivaldi",
        "microsoft-edge-stable",
        "thorium-browser",
    ] {
        candidates.push(b.to_string());
    }

    for b in candidates {
        let spawned = Command::new(&b)
            .arg(format!("--app={url}"))
            .arg(format!("--user-data-dir={}", profile.display()))
            .args([
                "--class=subgen",
                "--window-size=1440,900",
                "--no-first-run",
                "--no-default-browser-check",
                "--ozone-platform-hint=auto",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        if let Ok(child) = spawned {
            eprintln!("Jendela aplikasi dibuka dengan: {b}");
            return Some(child);
        }
    }

    eprintln!("Browser Chromium tidak ditemukan; membuka browser bawaan (tab biasa).");
    let _ = Command::new("xdg-open")
        .arg(url)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    None
}

/// Ada transkripsi atau export yang sedang berjalan/antre?
fn is_busy(st: &AppState) -> bool {
    let jobs_busy = st
        .jobs
        .lock()
        .unwrap()
        .values()
        .any(|j| matches!(j.status, Status::Queued | Status::Working));
    let exp_busy = st
        .exports
        .lock()
        .unwrap()
        .values()
        .any(|e| e.state == "working");
    jobs_busy || exp_busy
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let data_dir = match cli.data_dir {
        Some(d) => d,
        None => dirs::data_dir()
            .context("tidak bisa menemukan direktori data")?
            .join("subgen"),
    };
    std::fs::create_dir_all(data_dir.join("jobs"))?;
    let shown = data_dir.display().to_string();
    let profile = data_dir.join("browser-profile");

    let state: Shared = Arc::new(AppState {
        jobs: Mutex::new(load_jobs(&data_dir)),
        data_dir,
        model_cache: Mutex::new(None),
        gpu: Semaphore::new(1),
        exports: Mutex::new(HashMap::new()),
        exp: Semaphore::new(1),
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/api/config", get(config))
        .route("/api/jobs", get(list_jobs).post(create_job))
        .route(
            "/api/jobs/:id",
            get(get_job).put(put_job).delete(delete_job),
        )
        .route("/media/:id", get(media))
        .route("/api/fonts", get(fonts))
        .route("/api/jobs/:id/export", get(export_status).post(start_export))
        .route("/api/jobs/:id/export/file", get(export_file))
        .layer(DefaultBodyLimit::disable())
        .with_state(state.clone());

    let addr = format!("{}:{}", cli.host, cli.port);
    let open_host = if cli.host == "0.0.0.0" { "127.0.0.1" } else { cli.host.as_str() };
    let url = format!("http://{}:{}", open_host, cli.port);

    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) if cli.app && e.kind() == std::io::ErrorKind::AddrInUse => {
            // Server sudah berjalan (mis. aplikasi dibuka dua kali): cukup buka jendelanya.
            eprintln!("Server sudah berjalan di {url}; hanya membuka jendela.");
            tokio::task::spawn_blocking(move || {
                if let Some(mut c) = open_window(&url, &profile) {
                    let _ = c.wait();
                }
            })
            .await?;
            return Ok(());
        }
        Err(e) => {
            return Err(anyhow::Error::new(e)
                .context(format!("gagal membuka {addr} (port sudah dipakai?)")))
        }
    };
    eprintln!("subgen-web berjalan di {url}");
    eprintln!("Data proyek: {shown}");

    let shutdown = Arc::new(tokio::sync::Notify::new());
    if cli.app {
        if let Some(mut child) = open_window(&url, &profile) {
            let st = state.clone();
            let sd = shutdown.clone();
            tokio::task::spawn_blocking(move || {
                let t0 = std::time::Instant::now();
                let _ = child.wait();
                if t0.elapsed() < std::time::Duration::from_secs(5) {
                    eprintln!("Proses browser langsung selesai (launcher terpisah?). Server tetap berjalan; tekan Ctrl+C untuk berhenti.");
                    return;
                }
                // jangan menghilangkan pekerjaan yang sedang berjalan
                while is_busy(&st) {
                    eprintln!("Jendela ditutup, menunggu transkripsi/export selesai…");
                    std::thread::sleep(std::time::Duration::from_secs(3));
                }
                sd.notify_one();
            });
        }
    }

    let sd2 = shutdown.clone();
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = sd2.notified() => {}
            }
        })
        .await?;
    Ok(())
}
