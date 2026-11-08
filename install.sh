#!/bin/bash
# Install (or update) Crosshair, the hand-tracking target game, on a Mac. No Homebrew, no password.
#   curl -fsSL https://raw.githubusercontent.com/htmiyagi/crosshair/main/install.sh | bash
#
#   1. asks "Install Crosshair?" in a normal Mac window
#   2. installs "uv" (it installs Python itself) into your home folder, if needed
#   3. downloads Crosshair into ~/crosshair (your players, scores and games are kept when updating)
#   4. sets up its own Python 3.12 and packages (a few minutes the first time)
#   5. puts a Crosshair icon on your Desktop: double-click it to play
set -e

REPO="https://github.com/htmiyagi/crosshair"
DIR="${CROSSHAIR_DIR:-$HOME/crosshair}"
QUIET=0
[ "$1" = "--no-launch" ] && QUIET=1

say() { printf '\n\033[1;36m%s\033[0m\n' "$1"; }

ask() {   # a real Mac window with two buttons; prints the one clicked
  osascript - "$1" "$2" "$3" <<'APPLESCRIPT' 2>/dev/null || true
on run argv
  set {msg, b1, b2} to argv
  return button returned of (display dialog msg with title "Crosshair" buttons {b1, b2} default button b2)
end run
APPLESCRIPT
}

if [ "$QUIET" = 0 ]; then
  answer="$(ask "Install Crosshair?

A target game you play by pointing at your webcam. It helps Hoyu's research on hand-tracking cursors.

It downloads about 1 GB and takes about 5 minutes. No password needed." "Cancel" "Install")"
  [ "$answer" = "Install" ] || { echo "Cancelled."; exit 0; }
fi

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  say "Installing uv (it installs Python for Crosshair)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

if [ -d "$DIR/.git" ] && command -v git >/dev/null 2>&1 && git -C "$DIR" rev-parse >/dev/null 2>&1; then
  say "Updating Crosshair (git)..."
  git -C "$DIR" pull --ff-only || say "(couldn't update with git; keeping your copy as it is)"
else
  say "Downloading Crosshair..."
  TMP="$(mktemp -d)"
  curl -fsSL "$REPO/archive/refs/heads/main.zip" -o "$TMP/crosshair.zip"
  unzip -q "$TMP/crosshair.zip" -d "$TMP"
  mkdir -p "$DIR"
  cp -R "$TMP"/crosshair-main/. "$DIR"/          # data/ and runs/ aren't in the download, so yours are kept
  rm -rf "$TMP"
fi

cd "$DIR"
say "Setting up Python 3.12 and Crosshair's packages (a few minutes the first time)..."
[ -x .venv/bin/python ] || "$UV" venv --python 3.12 .venv
"$UV" pip install --python .venv/bin/python -r requirements.txt

printf '#!/bin/bash\ncd "%s" || exit 1\n.venv/bin/python game.py\n' "$DIR" > "$HOME/Desktop/Crosshair.command"
chmod +x "$HOME/Desktop/Crosshair.command"

say "Crosshair is installed!"
if [ "$QUIET" = 0 ]; then
  answer="$(ask "Crosshair is installed!

From now on, double-click Crosshair on your Desktop to play.

The first time, macOS asks to use the camera: click Allow (then quit Terminal with Cmd+Q and open Crosshair again). The very first start can take a minute while it gets ready." "Later" "Play now")"
  if [ "$answer" = "Play now" ]; then open "$HOME/Desktop/Crosshair.command"; fi
fi
