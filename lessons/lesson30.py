"""Lesson 30: hunting phantom clicks in games where you never clicked.

In Crosshair you aim by HOVERING, never by pinching. So if FIVES's click rules fire anywhere in these recordings,
every single click is a PHANTOM. That makes your ~70 recorded games a free test set of "this should never click".

What this does, for every game:
  1. Replays your recorded hand through FIVES's REAL click code (gestures.py from your FIVES folder, with the
     settings from your FIVES config.py): same gap measure, same 3-frame confirm, same speed limit, and the depth
     veto too for games that recorded z (new games do; older ones only saved x and y).
  2. Every "down" it fires = a phantom. For each one it records:
       speed        how fast your fingertip was moving (picture widths per second, FIVES's measure)
       gap          how close thumb and index got (below 0.20 = FIVES calls it a pinch)
       lasted       how many frames the phantom click was held
       since back   frames since the hand came back after being lost (your "it panics when it finds the hand" idea)
       curl         for each finger: wrist->tip distance / wrist->knuckle distance (small = curled in)
       hand size    how big the hand is in the picture (big = close to the camera)
       depth gap    thumb vs index depth (only games with z)
  3. THE COMPARISON. "Phantoms happen at speed 1.5" means nothing on its own: maybe your hand is usually at 1.5.
     So every number is compared with ALL your normal frames: "this phantom was faster than 92% of all frames".
     If phantoms sit near 50%, that factor doesn't matter. Near 90%+ (or 10%-), it stands out.
  4. Replays again with other settings (confirm frames, speed limit, start line) and counts the phantoms each time.

Run from the crosshair folder (inside your FIVES folder):  python lessons/lesson30.py
"""

import glob
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

FIVES = Path(__file__).resolve().parents[2]          # lessons -> crosshair -> fives
sys.path.insert(0, str(FIVES))
import gestures as g                                  # FIVES's real click code
try:
    import config as fives_config                     # your FIVES settings
except Exception as err:
    print(f"(couldn't read FIVES config.py: {err}; using FIVES's usual settings)")
    fives_config = None


def setting(name, default):
    return getattr(fives_config, name, default) if fives_config is not None else default


START = setting("PINCH_START", 0.20)
END = setting("PINCH_END", 0.25)
RELEASE = setting("PINCH_RELEASE_MARGIN", 0.10)
DRAG_END = setting("PINCH_DRAG_END", 0.40)
CONFIRM = setting("PINCH_CONFIRM_FRAMES", 3)
MAX_SPEED = setting("PINCH_MAX_SPEED", 2.0)
MAX_DEPTH = setting("PINCH_MAX_DEPTH", 0.35)

WRIST = 0
FINGERS = {"index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}   # (knuckle, tip)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def curls(p):
    """For each finger: wrist->tip / wrist->knuckle. Straight finger ~1.8-2, curled in ~1 or less."""
    return {f: dist(p[WRIST], p[tip]) / max(dist(p[WRIST], p[k]), 1e-6) for f, (k, tip) in FINGERS.items()}


def replay(pts, t, z, start=START, confirm=CONFIRM, max_speed=MAX_SPEED, max_depth=MAX_DEPTH):
    """Run FIVES's click rules over one game, the same way eyes.py does every frame.
    Returns (phantoms, every_frame): lists of dicts with the numbers above."""
    pinch = g.PinchDetector(start, END, RELEASE, DRAG_END, confirm)
    recent_tips = deque(maxlen=4)                     # eyes.py: raw fingertip over the last few frames
    since_back = 0
    phantoms, frames = [], []
    current = None
    for i in range(len(pts)):
        if np.isnan(pts[i]).any():                    # hand lost (eyes.py lets go of any pinch here)
            pinch.pinched = False
            since_back = 0
            if current is not None:
                current = None
            continue
        since_back += 1
        points = [(float(x), float(y)) for x, y in pts[i]]
        ratio = g.pinch_ratio(points)
        recent_tips.append((t[i], *points[g.INDEX_TIP]))
        speed = 0.0
        if len(recent_tips) >= 2 and recent_tips[-1][0] > recent_tips[0][0]:
            (t0, x0, y0), (t1, x1, y1) = recent_tips[0], recent_tips[-1]
            speed = math.hypot(x1 - x0, y1 - y0) / (t1 - t0)
        depth = None
        if z is not None and not np.isnan(z[i]).any():
            depth = g.depth_gap([float(v) for v in z[i]], points)
        too_fast = max_speed > 0 and speed > max_speed
        too_deep = max_depth > 0 and depth is not None and depth > max_depth
        info = {"speed": speed, "gap": ratio, "since back": since_back, "hand size": g.hand_size(points),
                "depth gap": depth, **curls(points)}
        frames.append(info)
        was = pinch.pinched
        event = pinch.update(ratio, holding=False, allow_start=not too_fast and not too_deep)
        if event == "down":
            current = dict(info, frame=i, time=float(t[i] - t[0]), gap=pinch.low, lasted=0)
            phantoms.append(current)
        if pinch.pinched and current is not None:
            current["lasted"] += 1
        if was and not pinch.pinched:
            current = None
    return phantoms, frames


# ---------- load every recorded game ----------
files = sorted(glob.glob("data/*.npz")) + sorted(glob.glob("data_final/*.npz")) + sorted(glob.glob("data_still/*.npz"))
games, skipped = [], []
for f in files:
    d = np.load(f)
    if "pts" not in d.files:                          # e.g. older recordings saved in a different format
        skipped.append(Path(f).name)
        continue
    pts = d["pts"].astype(np.float64)
    t = d["t"] if "t" in d.files else np.arange(len(pts)) / 30
    z = d["z"].astype(np.float64) if "z" in d.files else None
    games.append((Path(f).name, pts, t, z))
if skipped:
    print(f"(skipped {len(skipped)} recordings with no hand points, e.g. {skipped[0]})")
with_z = sum(z is not None for _, _, _, z in games)
print(f"{len(games)} games ({with_z} with depth z).  FIVES settings: start {START}, confirm {CONFIRM} frames, "
      f"speed limit {MAX_SPEED}, depth limit {MAX_DEPTH}\n")

phantoms, frames, minutes = [], [], 0.0
for name, pts, t, z in games:
    ph, fr = replay(pts, t, z)
    for p in ph:
        p["game"] = name
    phantoms += ph
    frames += fr
    minutes += len(fr) / 30 / 60

print(f"PHANTOM CLICKS WITH YOUR CURRENT SETTINGS: {len(phantoms)} in {minutes:.0f} minutes of hand tracking "
      f"({len(phantoms) / max(minutes, 1e-9):.2f} per minute)")
if not phantoms:
    print("None at all! Then your phantoms come from something these games don't contain (see the end).")

# ---------- each phantom ----------
COLS = ["speed", "gap", "lasted", "since back", "index", "middle", "ring", "pinky", "hand size", "depth gap"]
if phantoms:
    print("\nEVERY PHANTOM (game, seconds into it, then the numbers):")
    print(f"   {'game':22s} {'sec':>5s} " + " ".join(f"{c:>9s}" for c in COLS))
    for p in phantoms[:60]:
        vals = [p.get(c) for c in COLS]
        print(f"   {p['game'][:22]:22s} {p['time']:5.1f} "
              + " ".join(f"{'-':>9s}" if v is None else f"{v:9.2f}" for v in vals))
    if len(phantoms) > 60:
        print(f"   ... and {len(phantoms) - 60} more")

# ---------- the comparison: phantoms vs ALL your frames ----------
if phantoms:
    print("\nDOES ANYTHING STAND OUT?  (each phantom's value, as a percentile of ALL your normal frames)")
    print("   50% = totally ordinary.  90%+ = unusually HIGH for you.  10%- = unusually LOW for you.")
    print(f"   {'factor':11s} {'all frames (median)':>20s} {'phantoms (median)':>18s} {'phantom percentile':>19s}")
    pct = {}
    for c in ["speed", "since back", "index", "middle", "ring", "pinky", "hand size", "depth gap"]:
        base = np.array([fr[c] for fr in frames if fr[c] is not None], float)
        ph = np.array([p[c] for p in phantoms if p.get(c) is not None], float)
        if len(base) == 0 or len(ph) == 0:
            continue
        where = np.array([(base < v).mean() * 100 for v in ph])
        pct[c] = where
        flag = "  <- HIGH" if np.median(where) >= 80 else "  <- LOW" if np.median(where) <= 20 else ""
        print(f"   {c:11s} {np.median(base):20.2f} {np.median(ph):18.2f} {np.median(where):18.0f}%{flag}")
    print("   (index curl will look LOW almost by definition: a phantom means the index tip got near the thumb.")
    print("    The interesting ones are the OTHER fingers, speed, and 'since back'.)")

# ---------- what if the settings were different? ----------
def count(start=START, confirm=CONFIRM, max_speed=MAX_SPEED):
    return sum(len(replay(pts, t, z, start, confirm, max_speed)[0]) for _, pts, t, z in games)


print("\nWHAT IF?  Change ONE setting at a time (the others stay as yours) and count the phantoms again:")
now = len(phantoms)
print(f"   your settings now: {now} phantoms")
print("   confirm frames (each extra frame = ~33 ms slower on EVERY real click):")
for c in (1, 2, 3, 4, 5):
    print(f"      {c}   {count(confirm=c):5d}{'   <- you now' if c == CONFIRM else ''}")
print("   speed limit (no new click while the hand moves faster than this):")
for v in (0, 3.0, 2.0, 1.5, 1.0, 0.7):
    print(f"      {'off' if v == 0 else v:<4}{count(max_speed=v):5d}{'   <- you now' if v == MAX_SPEED else ''}")
print("   start line (how tightly you must pinch):")
for v in (0.25, 0.20, 0.15, 0.12):
    print(f"      {v:<4}{count(start=v):5d}{'   <- you now' if v == START else ''}")
print("   Fewer phantoms is easy (just make clicking hard). These games have NO real clicks, so they can't show")
print("   the cost. Lower speed limit = can't click while moving; tighter start line = must pinch harder.")

# ---------- picture ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if phantoms:
        show = ["speed", "since back", "middle", "ring"]
        fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.8))
        for ax, c in zip(axes, show):
            base = np.array([fr[c] for fr in frames], float)
            ph = np.array([p[c] for p in phantoms], float)
            hi = np.percentile(base, 99.5) if c != "since back" else min(np.percentile(base, 99), 120)
            bins = np.linspace(base.min(), max(hi, ph.max()), 40)
            ax.hist(base, bins=bins, density=True, color="lightgray", label="all your frames")
            ax.hist(ph, bins=bins, density=True, color="tab:red", alpha=0.6, label="phantom clicks")
            ax.set_title(c if c in ("speed", "since back") else f"{c} finger curl (small = curled)")
            ax.legend(fontsize=8)
        fig.suptitle("Where do phantom clicks happen, compared with normal? (red far from gray = a cause)")
        fig.tight_layout()
        fig.savefig("lesson30.png", dpi=120)
        print("\nSaved lesson30.png  (open lesson30.png)")
except ImportError:
    pass
