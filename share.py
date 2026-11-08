"""SEND GAMES: upload your Crosshair games to Hoyu's Google Drive, for his research.

What gets sent (only after you click "Send" in the box that asks):
  - your recorded games (data/session_*.npz): hand POSITIONS only (21 points per frame), the targets, the cursor.
    No camera images, ever.
  - your rows from the leaderboard (score, which cursor lead, your guess), so the "can you feel it?" part works.
    Never your passcode.
Only games you haven't sent before are uploaded (the list is kept in runs/sent.json), so pressing it again is safe.

Where it goes: a small Google script Hoyu owns (crosshair_upload.gs) saves the files into his Drive folder
"Crosshair data/<your name>/". His Google Drive app syncs that to his Mac, into crosshair/data_shared/.

Hoyu: paste your script's web app URL below (and the same SECRET as in the script).
Test from Terminal:  python share.py <player name>
"""

import base64
import glob
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

SHARE_URL = "https://script.google.com/macros/s/AKfycbx-i08GucihETxFSHRs6M478LO_AbcCe4OswcmeKOERPIVxRMDLckSkeId7EBOwN_Qk/exec"                      # e.g. https://script.google.com/macros/s/AKfy.../exec
SECRET = "crosshair-501st-lab"                # must match SECRET in crosshair_upload.gs

LAB = Path(__file__).resolve().parent
SENT_FILE = LAB / "runs" / "sent.json"
BOARD_FILE = LAB / "runs" / "leaderboard.json"


def games_of(player):
    """This player's recordings (newest last)."""
    out = []
    for f in sorted(glob.glob(str(LAB / "data" / "session_*.npz"))):
        try:
            import numpy as np
            d = np.load(f)
            meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
        except Exception:
            continue
        if meta.get("player") == player:
            out.append(Path(f))
    return out


def unsent(player):
    sent = set(json.loads(SENT_FILE.read_text())) if SENT_FILE.exists() else set()
    return [f for f in games_of(player) if f.name not in sent]


def ask_consent(n, name):
    """A normal Mac box: send or not. True only if they click Send."""
    msg = (f"Send your {n} new Crosshair game{'s' if n != 1 else ''} to Hoyu for his research?\\n\\n"
           "Only hand POSITIONS (dots, no camera images), the targets, and your scores and guesses. "
           "They are saved under your player name: " + name.replace(chr(34), ""))
    script = (f'button returned of (display dialog "{msg}" with title "Crosshair" '
              f'buttons {{"Not now", "Send"}} default button "Send")')
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=300)
        return out.stdout.strip() == "Send"
    except Exception:
        return False


def upload(player, name, data):
    body = json.dumps({"secret": SECRET, "player": player, "name": name,
                       "data": base64.b64encode(data).decode()}).encode()
    req = urllib.request.Request(SHARE_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        reply = json.loads(r.read().decode() or "{}")
    if not reply.get("ok"):
        raise RuntimeError(reply.get("error", "the upload was refused"))


def send(player, display_name=None, progress=print, ask=True):
    """Upload this player's unsent games. Returns a short message for the screen."""
    if not SHARE_URL:
        return "Sending isn't set up yet (no SHARE_URL in share.py)."
    todo = unsent(player)
    if not todo:
        return "Nothing new to send: all your games are already with Hoyu."
    if ask and not ask_consent(len(todo), display_name or player):
        return "Not sent."
    sent = set(json.loads(SENT_FILE.read_text())) if SENT_FILE.exists() else set()
    done = 0
    try:
        for f in todo:
            progress(f"Sending game {done + 1} of {len(todo)}...")
            upload(player, f.name, f.read_bytes())
            sent.add(f.name)
            done += 1
            SENT_FILE.parent.mkdir(exist_ok=True)
            SENT_FILE.write_text(json.dumps(sorted(sent)))
        runs = json.loads(BOARD_FILE.read_text()) if BOARD_FILE.exists() else []
        mine = [{k: v for k, v in r.items() if "pass" not in k.lower()} for r in runs if r.get("player") == player]
        upload(player, f"board_{player}.json", json.dumps(mine).encode())
    except Exception as err:
        return f"Sent {done} of {len(todo)}, then it failed ({err}). Try again later: it picks up where it stopped."
    return f"Sent {done} game{'s' if done != 1 else ''} to Hoyu. Thank you!"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python share.py <player name as shown in the game>")
    print(send(sys.argv[1]))
