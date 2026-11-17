#!/bin/sh
# Qwen Station @VERSION@ for Linux. One file: it installs the app for the current user and starts it.
#
#   sh Qwen-Station-Setup-@VERSION@-linux.run              install and start
#   sh Qwen-Station-Setup-@VERSION@-linux.run --no-start   install only
#   Every other option goes on to the app, for example: --setup --location /data/qwen-station
#
# Nothing here needs root rights. It installs the app into ~/.local/share/qwen-station/app, the command
# ~/.local/bin/qwen-station and a menu entry. Remove them again with: qwen-station --uninstall
set -eu

VERSION="@VERSION@"
START=1
for argument in "$@"; do
  case "$argument" in
    --no-start) START=0 ;;
    --help|-h) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  esac
done

fail() { printf 'Qwen Station setup: %s\n' "$1" >&2; exit 1; }
[ "$(uname -s)" = Linux ] || fail "this installer is for Linux."
[ "$(uname -m)" = x86_64 ] || fail "the engine needs an x86_64 machine, this one is $(uname -m)."
command -v python3 >/dev/null 2>&1 || fail "python3 is missing. Install Python 3.10 or newer, then run this file again."
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' ||
  fail "Python 3.10 or newer is needed, found $(python3 -V 2>&1)."
command -v tar >/dev/null 2>&1 || fail "tar is missing."

# Keep these paths in step with linux_paths() in qwen_station/installer.py.
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
APP="$DATA/qwen-station/app"
BIN="$HOME/.local/bin"
DESKTOP="$DATA/applications/qwen-station.desktop"
ICON="$DATA/icons/hicolor/256x256/apps/qwen-station.png"

PAYLOAD_LINE=$(awk '/^__PAYLOAD__$/ { print NR + 1; exit }' "$0")
rm -rf "$APP.new"
mkdir -p "$APP.new" "$BIN" "$(dirname "$DESKTOP")" "$(dirname "$ICON")"
tail -n +"$PAYLOAD_LINE" "$0" | tar -xzf - -C "$APP.new" || fail "the file is damaged. Download it again."
rm -rf "$APP"
mv "$APP.new" "$APP"
cp "$APP/qwen-station.png" "$ICON"

cat > "$BIN/qwen-station" <<EOF
#!/bin/sh
exec python3 "$APP/run-station.py" "\$@"
EOF
chmod +x "$BIN/qwen-station"

cat > "$DESKTOP" <<EOF
[Desktop Entry]
Type=Application
Name=Qwen Station
Comment=Start and monitor Qwen3.8 on vLLM
Exec=$BIN/qwen-station
Icon=qwen-station
Terminal=false
Categories=Development;Science;
EOF

printf 'Qwen Station %s is installed. Start it with: qwen-station\n' "$VERSION"
case ":$PATH:" in
  *":$BIN:"*) ;;
  *) printf '%s is not in your PATH. Use %s/qwen-station or the menu entry.\n' "$BIN" "$BIN" ;;
esac
[ "$START" = 1 ] || exit 0

# Everything except --no-start goes on to the app.
for argument in "$@"; do
  shift
  [ "$argument" = --no-start ] || set -- "$@" "$argument"
done
# Without a desktop there is no window to click in: start the setup right away, its progress shows here.
if [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && [ "$#" = 0 ]; then
  set -- --setup
fi
exec "$BIN/qwen-station" "$@"
exit 0
__PAYLOAD__
