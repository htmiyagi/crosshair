"""Fitts' law: the standard way pointing research scores how good a pointer is.

Used by the game (throughput on the results screen) and by lessons/lesson28.py.

THE IDEA (Paul Fitts, 1954): how long it takes to hit a target depends on two things only:
    D = how FAR away it is           (farther = slower)
    W = how BIG it is                (smaller = slower)
Combined into one number, the INDEX OF DIFFICULTY, measured in "bits":
    ID = log2(D / W + 1)
Doubling the distance, or halving the size, adds about 1 bit: one more "halving" of uncertainty your hand
has to do. Then the time to hit the target grows in a straight line with the difficulty:
    MT = a + b * ID        (MT = movement time; a and b are found from your data)
THROUGHPUT = ID / MT, averaged over targets, in bits per second: how much "pointing difficulty" you get through
per second. It's the single score pointing papers compare (a mouse is often around 4-5 bits/s in studies,
mid-air hand pointing usually lower), so a cursor change that raises throughput made pointing genuinely better.

Here:
    D  = from where the cursor was when the target appeared, to the target's center (window pixels)
    W  = the target's smaller side (window pixels)
    MT = time from the target appearing to it closing, MINUS the lock-on time (the dwell), which is the same
         for every target and isn't pointing
"""

import math

import numpy as np


def to_window(p, margin, w, h):
    """Camera point (0..1) -> game window pixels (same mapping as the game)."""
    span = 1 - 2 * margin
    x = min(max((p[0] - margin) / span, 0.0), 1.0)
    y = min(max((p[1] - margin) / span, 0.0), 1.0)
    return x * (w - 1), y * (h - 1)


def trials(t, cursor, rects, spawn_t, close_t, margin, w, h, dwell):
    """One (ID, MT, D, W) per closed target. Skips targets the cursor was already on when they appeared."""
    t = np.asarray(t, float)
    cursor = np.asarray(cursor, float)
    out = []
    for rect, s, c in zip(rects, spawn_t, close_t):
        if not np.isfinite(c) or not np.isfinite(s):
            continue
        i = int(np.searchsorted(t, s))
        i = min(i, len(t) - 1)
        while i > 0 and not np.isfinite(cursor[i]).all():   # no hand at that instant: use the last known spot
            i -= 1
        if not np.isfinite(cursor[i]).all():
            continue
        sx, sy = to_window(cursor[i], margin, w, h)
        x, y, rw, rh = rect
        D = math.hypot(x + rw / 2 - sx, y + rh / 2 - sy)
        W = min(rw, rh)
        mt = (c - s) - dwell
        if D < W / 2 or mt <= 0.05:                         # started on the target, or impossibly fast
            continue
        out.append((math.log2(D / W + 1), mt, D, W))
    return out


def summary(tr):
    """Throughput (bits/s) plus the straight-line fit MT = a + b * ID and how well it fits (R squared)."""
    if len(tr) < 3:
        return None
    ids = np.array([x[0] for x in tr])
    mts = np.array([x[1] for x in tr])
    b, a = np.polyfit(ids, mts, 1)
    pred = a + b * ids
    r2 = 1 - ((mts - pred) ** 2).sum() / max(((mts - mts.mean()) ** 2).sum(), 1e-9)
    return {"throughput": float(np.mean(ids / mts)), "a": float(a), "b": float(b), "r2": float(r2), "n": len(tr)}


def binned(tr, width=0.5, min_count=5):
    """The standard way to fit Fitts' law: group targets by difficulty, average each group, fit the line through
    the AVERAGES. Single attempts are noisy (a slow reaction, an overshoot); averages show the real trend.
    Returns {"a", "b", "r2", "bins": [(average ID, average MT, how many targets)]}."""
    if not tr:
        return None
    ids = np.array([x[0] for x in tr])
    mts = np.array([x[1] for x in tr])
    edges = np.arange(np.floor(ids.min() / width) * width, ids.max() + width, width)
    bins = []
    for lo in edges:
        m = (ids >= lo) & (ids < lo + width)
        if m.sum() >= min_count:                       # skip near-empty groups: their averages are noise too
            bins.append((float(ids[m].mean()), float(mts[m].mean()), int(m.sum())))
    if len(bins) < 3:
        return None
    bx = np.array([b[0] for b in bins])
    by = np.array([b[1] for b in bins])
    b, a = np.polyfit(bx, by, 1)
    r2 = 1 - ((by - (a + b * bx)) ** 2).sum() / max(((by - by.mean()) ** 2).sum(), 1e-9)
    return {"a": float(a), "b": float(b), "r2": float(r2), "bins": bins}


def settle_jitter(t, cursor, close_t, margin, w, h, window=0.6):
    """How shaky the cursor is while you're SETTLING onto each target: the last `window` seconds before it closed
    (the lock-on and the final adjustments). Jitter = average |p[i] - 2 p[i-1] + p[i-2]| in window pixels:
    how much the cursor's path bends frame to frame. A smooth, steady cursor scores low.
    Returns one number per closed target."""
    t = np.asarray(t, float)
    cursor = np.asarray(cursor, float)
    out = []
    for c in close_t:
        if not np.isfinite(c):
            continue
        idx = np.where((t >= c - window) & (t <= c))[0]
        if len(idx) < 5:
            continue
        p = np.array([to_window(cursor[i], margin, w, h) if np.isfinite(cursor[i]).all() else (np.nan, np.nan)
                      for i in idx])
        bend = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1)
        bend = bend[np.isfinite(bend)]
        if len(bend):
            out.append(float(bend.mean()))
    return out


def from_recording(path, default_margin=0.22, default_dwell=0.30):
    """Fitts' trials from one saved game (data/*.npz). Returns (trials, meta) or (None, meta) if it has no targets."""
    import json
    d = np.load(path)
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    if "target_rect" not in d.files or "cursor" not in d.files:
        return None, meta
    tr = trials(d["t"], d["cursor"], d["target_rect"], d["target_spawn_t"], d["target_close_t"],
                meta.get("margin", default_margin), meta.get("window_w", 960), meta.get("window_h", 720),
                meta.get("dwell", default_dwell))
    return tr, meta


def jitter_from_recording(path, default_margin=0.22):
    """Settle jitter (one number per closed target) for one saved game, or [] if it has no targets."""
    import json
    d = np.load(path)
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    if "target_close_t" not in d.files or "cursor" not in d.files:
        return []
    return settle_jitter(d["t"], d["cursor"], d["target_close_t"], meta.get("margin", default_margin),
                         meta.get("window_w", 960), meta.get("window_h", 720))
