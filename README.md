# subgen

Generate subtitle lokal dari video/audio (Bahasa Indonesia) dengan Whisper dari Hugging Face
(whisper.cpp), plus editor web: preview video + caption, edit teks & timing, atur kata per caption.

Dua program:

| Program | Fungsi |
|---|---|
| `subgen` | CLI: video -> file `.srt` |
| `subgen-web` | Server web lokal (http://127.0.0.1:3000): upload, progress, preview, editor, export |

## Menjalankan web

```bash
env LIBCLANG_PATH=/usr/lib/llvm18/lib cargo build --release --features cuda
./target/release/subgen-web            # buka http://127.0.0.1:3000
./target/release/subgen-web --port 8080 --data-dir ~/subgen-data
```

## Aplikasi desktop (tanpa buka web manual)

```bash
env LIBCLANG_PATH=/usr/lib/llvm18/lib cargo build --release --features cuda
./install.sh            # pasang ke menu aplikasi + ikon (tanpa sudo); ./install.sh uninstall untuk menghapus
```

Lalu buka **Subgen** dari menu aplikasi (bisa di-pin ke taskbar), atau jalankan `subgen-web --app`.

- Jendela dibuka dalam mode app (tanpa tab/address bar) memakai browser berbasis Chromium yang terpasang
  (chromium, chrome, brave, vivaldi, edge, thorium). Pilih browser tertentu dengan `SUBGEN_BROWSER=brave subgen-web --app`.
  Firefox tidak punya mode app; kalau tidak ada Chromium, dibuka sebagai tab biasa dan server berjalan sampai Ctrl+C.
- Menutup jendela menghentikan server. Kalau masih ada transkripsi atau export yang berjalan, server menunggu
  sampai selesai dulu.
- Membuka aplikasi dua kali tidak membuat server ganda: jendela kedua memakai server yang sudah berjalan.
- Jendela memakai profil browser tersendiri (`~/.local/share/subgen/browser-profile`), jadi pengaturan gaya teks
  di browser biasa tidak ikut terbawa dan perlu diatur sekali lagi.

- Proyek (video + hasil edit) disimpan di `~/.local/share/subgen` dan tetap ada setelah server dimatikan.
- Satu transkripsi berjalan pada satu waktu; sisanya antre. Model di-cache di VRAM antar proyek.
- Server hanya bind ke `127.0.0.1` (tidak terbuka ke jaringan).
- Preview memakai pemutar browser, jadi hanya codec yang didukung browser (mp4 H.264, webm).
  Transkripsi tetap jalan untuk format apa pun yang dibaca ffmpeg.

### Editor

- **Maks. kata per segmen** (default 3): saat unggah, kata dikelompokkan jadi segmen maksimal N kata.
  Di editor, mengubah angka ini langsung menyusun ulang semua segmen dari data kata (bisa memecah maupun
  menggabung; edit batas segmen manual akan hilang). Perubahan permanen setelah **Simpan**.
- **Sinkronisasi**: saat transkripsi, batas tiap kata dikoreksi dengan energi audio memakai ambang lokal:
  awal kata digeser maju ke onset suara bila Whisper menaruhnya di jeda sunyi, digeser mundur bila terlambat
  setelah jeda, dan akhir kata dimundurkan ke suara terakhir. Ucapan yang menyambung tanpa jeda tidak diubah
  (energi audio tidak bisa memisahkan kata di situ). Proyek lama perlu digenerate ulang untuk mendapat koreksi ini.
  Sisa selisih diatur dengan **Geser waktu caption (ms)**: `+` = caption muncul lebih lambat, `-` = lebih cepat.
  Berlaku di preview dan export (tidak mengubah data tersimpan).
- Caption selalu **satu baris horizontal**: tidak ada batas karakter dan ukuran font tidak dikecilkan otomatis.
  Kalau terlalu lebar untuk video, kurangi **Maks. kata per segmen**.
- **Kata muncul satu per satu** (default aktif): tiap kata fade/slide masuk tepat saat diucapkan, mengikuti timing
  per kata. Ini hanya efek preview; file `.srt`/`.vtt` tidak bisa membawa animasi.
- **Gaya teks**: font (daftar font yang terpasang di sistem, dibaca lewat `fc-list`), ukuran (% tinggi video),
  warna teks, warna outline, tebal, huruf (normal / BESAR / kecil / Kapital Tiap Kata), jarak huruf (kerning),
  dan posisi dari bawah. Disimpan di browser dan dipakai di preview, export `.srt`/`.vtt` (huruf besar/kecil),
  export `.ass`, dan export video. Browser dan ffmpeg memakai font sistem yang sama, jadi tampilannya konsisten.
- **Export `.ass`**: membawa gaya teks dan animasi kata (satu event per kata; kata baru fade-in 150 ms).
- **Export video + caption** (caption dibakar ke video lewat ffmpeg + libass): format MP4 H.264, MP4 H.265, atau
  WebM VP9; kualitas tinggi/sedang/kecil; encoder GPU (NVENC, otomatis fallback ke CPU kalau gagal) atau CPU;
  resolusi asli / 1080p / 720p / 480p. Memakai segmen yang sedang tampil di editor (termasuk yang belum disimpan)
  dan geser waktu yang diatur. Hasil diunduh dari tautan setelah selesai. Butuh `ffmpeg` dengan libass
  (paket `ffmpeg` bawaan Arch sudah ada) dan `fontconfig`.
- Segmen: edit teks, edit waktu mulai/selesai (`01:23.450`), putar dari segmen, **Pisah** di posisi video,
  **Gabung** dengan segmen berikutnya, **Hapus**, **+ Segmen**.
- `Ctrl+S` menyimpan. Export `.srt` / `.vtt` memakai pengaturan kata per caption saat ini.

## Pilihan model (disesuaikan untuk GTX 1660 Super 6 GB VRAM + RAM 8 GB)

| Model (`-m`) | Ukuran file | Catatan |
|---|---|---|
| `large-v3-q5_0` (default) | ~1,1 GB | Akurasi paling tinggi dengan memori kecil, lebih lambat dari turbo. |
| `large-v3-turbo` | ~1,6 GB | Akurasi sedikit di bawah large-v3, jauh lebih cepat. |
| `large-v3-turbo-q8_0` | ~0,9 GB | Turbo kuantisasi, hemat memori. |
| `large-v3-turbo-q5_0` | ~0,6 GB | Paling ringan. |
| `large-v3` | ~3,1 GB | Mepet di 6 GB VRAM + RAM 8 GB, tidak disarankan. |

## Prasyarat (CachyOS / Arch Linux)

```bash
sudo pacman -S --needed base-devel rustup cmake clang ffmpeg cuda gcc14
rustup default stable
```

- Driver NVIDIA harus sudah aktif (cek dengan `nvidia-smi`). CachyOS biasanya memasangnya saat instalasi.
- `clang` dibutuhkan bindgen (libclang), `cuda` memasang nvcc di `/opt/cuda`.
- `gcc14` dibutuhkan karena nvcc sering menolak GCC terbaru di Arch.

Atur environment variable. CachyOS memakai **fish** secara default, jadi pakai sintaks fish
(jangan tempel `export PATH=...:$PATH` bergaya bash ke fish, itu merusak PATH):

```fish
fish_add_path /opt/cuda/bin
set -Ux CUDA_PATH /opt/cuda
set -Ux CUDAHOSTCXX /usr/bin/g++-14   # host compiler yang didukung nvcc
set -Ux CUDAARCHS 75                  # GTX 1660 Super (Turing), mempercepat build
```

Kalau memakai bash/zsh, tambahkan ke `~/.bashrc` / `~/.zshrc`:

```bash
export PATH=/opt/cuda/bin:$PATH
export CUDA_PATH=/opt/cuda
export CUDAHOSTCXX=/usr/bin/g++-14
export CUDAARCHS=75
```

Cek: `nvidia-smi`, `nvcc --version`, `g++-14 --version`, `cargo --version`.

Tidak mau pakai CUDA? Alternatifnya Vulkan:
`sudo pacman -S vulkan-headers vulkan-icd-loader shaderc`, lalu build dengan `--features vulkan`.

## Build & jalankan

```bash
cargo test
env LIBCLANG_PATH=/usr/lib/llvm18/lib cargo build --release --features cuda

# model diunduh otomatis saat pertama kali jalan
./target/release/subgen video.mp4                    # model default: large-v3-q5_0
./target/release/subgen video.mp4 -m large-v3-turbo
./target/release/subgen video.mp4 -o hasil.srt
./target/release/subgen video.mp4 --prompt "Podcast teknologi, istilah: Rust, Kubernetes, Hugging Face."
./target/release/subgen video.mp4 --beam-size 1
```

Model disimpan di `~/.cache/subgen/models`.

## Struktur

| File | Isi |
|---|---|
| `src/audio.rs` | ffmpeg -> PCM f32 mono 16 kHz |
| `src/model.rs` | cache & download model dari Hugging Face |
| `src/transcribe.rs` | whisper-rs, timestamp per kata, pengelompokan segmen |
| `src/bin/subgen-web.rs` | server axum (upload, job, media Range, simpan edit) |
| `web/index.html` | UI (vanilla JS, ditanam ke binary saat build) |
| `src/srt.rs` | format SRT + word wrap + unit test |
| `src/main.rs` | CLI (clap) |

## Troubleshooting

- Error kompilasi di `transcribe.rs` soal API whisper-rs: versi crate berubah-ubah, kirim pesan errornya.
- Error `unsupported GNU version` dari nvcc: pastikan `gcc14` terpasang dan `CUDAHOSTCXX` ter-set, lalu `cargo clean` dan build ulang.
- Error `Size of _IO_FILE` / `whisper_full_params` saat build: libclang 22 terlalu baru untuk bindgen 0.71. Pasang `llvm18 clang18` dan build dengan `LIBCLANG_PATH=/usr/lib/llvm18/lib`.
- Error `libclang not found`: pastikan paket `clang` terpasang.
- Error `nvcc not found`: pastikan `/opt/cuda/bin` ada di PATH.
- Out of memory di GPU: pakai model kuantisasi atau `--beam-size 1`.
- Video sangat panjang (>1 jam) memakan RAM lebih banyak, akan ditangani dengan chunking di tahap 4.
- Build pertama memakai semua core CPU (cmake/nvcc mengompilasi whisper.cpp). Batasi dengan
  `env CMAKE_BUILD_PARALLEL_LEVEL=4 nice -n 19 cargo build -j4 ...`. Pakai variabel environment yang sama
  setiap build supaya `whisper-rs-sys` tidak dikompilasi ulang.
