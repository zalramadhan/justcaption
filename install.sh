#!/usr/bin/env bash
# Pasang Subgen sebagai aplikasi desktop (menu aplikasi + ikon), tanpa sudo.
#   ./install.sh            pasang
#   ./install.sh uninstall  hapus
set -euo pipefail
cd "$(dirname "$0")"

BIN_DST="$HOME/.local/bin/subgen-web"
ICON_DST="$HOME/.local/share/icons/hicolor/scalable/apps/subgen.svg"
DESK_DST="$HOME/.local/share/applications/subgen.desktop"

refresh() {
  update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
  command -v kbuildsycoca6 >/dev/null 2>&1 && kbuildsycoca6 >/dev/null 2>&1 || true
}

if [ "${1:-}" = "uninstall" ]; then
  rm -f "$BIN_DST" "$ICON_DST" "$DESK_DST"
  refresh
  echo "Subgen dihapus (data proyek di ~/.local/share/subgen tidak disentuh)."
  exit 0
fi

SRC="target/release/subgen-web"
if [ ! -x "$SRC" ]; then
  echo "Binary belum ada. Build dulu:"
  echo "  env LIBCLANG_PATH=/usr/lib/llvm18/lib cargo build --release --features cuda"
  exit 1
fi

install -Dm755 "$SRC" "$BIN_DST"
install -Dm644 packaging/subgen.svg "$ICON_DST"
mkdir -p "$(dirname "$DESK_DST")"
sed "s|@BIN@|$BIN_DST|" packaging/subgen.desktop > "$DESK_DST"
chmod 644 "$DESK_DST"
refresh

echo "Terpasang. Cari \"Subgen\" di menu aplikasi (bisa di-pin ke taskbar)."
echo "Atau jalankan: $BIN_DST --app"
