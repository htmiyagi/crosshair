"""Friends' games, unpacked game by game.

Friends press SEND GAMES TO HOYU; each send arrives as ONE zip in Google Drive (data_shared/<player>/pack_*.zip,
synced to this Mac). This unpacks every zip into

    data_friends/<player>/session_....npz      one file per game, exactly like your own data/
    data_friends/<player>/board_<player>.json  their scores and guesses (the newest copy)

Already-unpacked games are skipped, so it's instant after the first time. Lessons call games() and boards(); you can
also run it by hand to see what's arrived:   python friends.py
"""

import glob
import json
import zipfile
from pathlib import Path

LAB = Path(__file__).resolve().parent
SHARED = LAB / "data_shared"
OUT = LAB / "data_friends"


def unpack():
    """Unzip anything new. Returns how many games were added."""
    added = 0
    for z in sorted(glob.glob(str(SHARED / "**" / "*.zip"), recursive=True)):
        player = Path(z).parent.name
        dest = OUT / player
        try:
            with zipfile.ZipFile(z) as pack:
                for name in pack.namelist():
                    target = dest / Path(name).name
                    if name.endswith(".npz") and not target.exists():
                        dest.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(pack.read(name))
                        added += 1
                    elif name.startswith("board_") and name.endswith(".json"):
                        dest.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(pack.read(name))      # zips are read oldest first: newest wins
        except (zipfile.BadZipFile, OSError) as err:
            print(f"(skipped {Path(z).name}: {err}; it may still be syncing)")
    return added


def games(skip=()):
    """Every friend's game file (unpacked zips + the older one-file-per-game uploads), minus names in `skip`."""
    unpack()
    found = sorted(glob.glob(str(OUT / "**" / "*.npz"), recursive=True)) + \
        sorted(glob.glob(str(SHARED / "**" / "*.npz"), recursive=True))
    seen, out = set(skip), []
    for f in found:
        if Path(f).name not in seen:
            seen.add(Path(f).name)
            out.append(f)
    return out


def boards():
    """Every friend's leaderboard rows (scores, leads, guesses)."""
    rows = []
    for f in glob.glob(str(OUT / "**" / "board_*.json"), recursive=True) + \
            glob.glob(str(SHARED / "**" / "board_*.json"), recursive=True):
        try:
            rows += json.loads(Path(f).read_text())
        except (OSError, ValueError):
            pass
    return rows


if __name__ == "__main__":
    n = unpack()
    print(f"{n} new game{'s' if n != 1 else ''} unpacked into data_friends/")
    for d in sorted(OUT.glob("*")) if OUT.exists() else []:
        print(f"   {d.name:16s} {len(list(d.glob('*.npz')))} games")
