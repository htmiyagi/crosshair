"""SEND GAMES: upload your Crosshair games to Hoyu's Google Drive, for his research.

What gets sent (only after you click "Send" in the box that asks):
  - your recorded games (data/session_*.npz): hand POSITIONS only (21 points per frame), the targets, the cursor.
    No camera images, ever.
  - your rows from the leaderboard (score, which cursor lead, your guess), so the "can you feel it?" part works.
    Never your passcode.
Only games you haven't sent before are uploaded (the list is kept in runs/sent.json), so pressing it again is safe.
They go up as ONE zip per send (not one upload per game): far lighter on Google's daily Drive limit. Each zip also
carries a small summary of every game, which Hoyu's Google Sheet turns into one row per game.

Where it goes: a small Google script Hoyu owns (crosshair_upload.gs) saves the files into his Drive folder
"Crosshair data/<your name>/". His Google Drive app syncs that to his Mac, into crosshair/data_shared/.

Hoyu: paste your script's web app URL below (and the same SECRET as in the script).
Test from Terminal:  python share.py <player name>
"""

import base64
import glob
import json
import zipfile
import subprocess
import sys
import time
import urllib.request
from io import BytesIO
from pathlib import Path

SHARE_URL = "https://script.google.com/macros/s/AKfycbx-i08GucihETxFSHRs6M478LO_AbcCe4OswcmeKOERPIVxRMDLckSkeId7EBOwN_Qk/exec"
SECRET = "crosshair-501st-lab"     # must match SECRET in crosshair_upload.gs

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


class DailyLimit(Exception):
    """Google's daily Drive limit for the script is used up: try again tomorrow."""


def upload(player, name, data, tries=4, games=None):
    """One file. If Google says 'too much, slow down', wait and try again (5 s, 20 s, 60 s)."""
    body = json.dumps({"secret": SECRET, "player": player, "name": name, "games": games or [],
                       "data": base64.b64encode(data).decode()}).encode()
    for attempt in range(tries):
        req = urllib.request.Request(SHARE_URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            reply = json.loads(r.read().decode() or "{}")
        if reply.get("ok"):
            return
        error = reply.get("error", "the upload was refused")
        if "limit" in error.lower() or "too many" in error.lower() or "service error" in error.lower():
            if attempt < tries - 1:
                time.sleep((5, 20, 60)[attempt])
                continue
            raise DailyLimit(error)
        raise RuntimeError(error)


MAX_PACK = 12_000_000          # bytes per zip (Google accepts about 50 MB per upload; stay well under)


def packages(files):
    """Split the games into zips of at most MAX_PACK bytes (usually all in one)."""
    out, cur, size = [], [], 0
    for f in files:
        n = f.stat().st_size
        if cur and size + n > MAX_PACK:
            out.append(cur)
            cur, size = [], 0
        cur.append(f)
        size += n
    if cur:
        out.append(cur)
    return out


def summary(path, rows):
    """One game in a few words, for Hoyu's Google Sheet (the zip carries the real data)."""
    import numpy as np
    d = np.load(path)
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    row = next((r for r in rows if r.get("attempt") == meta.get("attempt")), {})
    return {"file": path.name, "attempt": meta.get("attempt"), "played": row.get("date", ""),
            "lead": meta.get("lead", ""), "guess": row.get("guess") or "", "score": meta.get("score"),
            "bits_per_s": row.get("tp"), "targets": meta.get("targets", "normal"),
            "finished": bool(meta.get("finished"))}


def make_zip(player, files, board, rows):
    """The games + the player's scores/guesses + a summary of each game, as one zip (games are already
    compressed, so the zip just stores them)."""
    games = [summary(f, rows) for f in files]
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for f in files:
            z.write(f, f.name)
        z.writestr(f"board_{player}.json", board)
        z.writestr("summary.json", json.dumps(games))
    return buf.getvalue(), games


def send(player, display_name=None, progress=print, ask=True):
    """Upload this player's unsent games. Returns a short message for the screen."""
    if not SHARE_URL:
        return "Sending isn't set up yet (no SHARE_URL in share.py)."
    todo = unsent(player)
    if todo and ask and not ask_consent(len(todo), display_name or player):
        return "Not sent."
    sent = set(json.loads(SENT_FILE.read_text())) if SENT_FILE.exists() else set()
    runs = json.loads(BOARD_FILE.read_text()) if BOARD_FILE.exists() else []
    mine = [{k: v for k, v in r.items() if "pass" not in k.lower()} for r in runs if r.get("player") == player]
    board = json.dumps(mine).encode()
    done = 0
    try:
        for k, pack in enumerate(packages(todo)):
            progress(f"Sending {len(pack)} game{'s' if len(pack) != 1 else ''} "
                     f"({done + len(pack)} of {len(todo)})...")
            data, games = make_zip(player, pack, board, mine)
            upload(player, f"pack_{player}_{time.strftime('%Y%m%d_%H%M%S')}_{k}.zip", data, games=games)
            sent.update(f.name for f in pack)
            done += len(pack)
            SENT_FILE.parent.mkdir(exist_ok=True)
            SENT_FILE.write_text(json.dumps(sorted(sent)))
    except DailyLimit:
        return (f"Sent {done} of {len(todo)}. Google's daily limit is reached: press SEND again tomorrow, "
                "it picks up where it stopped.")
    except Exception as err:
        return f"Sent {done} of {len(todo)}, then it failed ({err}). Try again later: it picks up where it stopped."
    if not todo:
        return "Nothing new to send: all your games are already with Hoyu."
    return f"Sent {done} game{'s' if done != 1 else ''} to Hoyu. Thank you!"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python share.py <player name as shown in the game>")
    print(send(sys.argv[1]))
