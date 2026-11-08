"""Lesson 33: can the webcam tell pen-ON-paper from pen-LIFTED? And pen grip from a relaxed hand?

Uses your recordings from depth_rec.py (data_depth/). Two questions:

1. ON vs LIFTED, from HAND SIZE ("the virtual sheet of glass")
   Closer to the camera = bigger in the picture. We try several ways to measure hand size, because some bones
   change less when your hand turns:
       wrist->middle knuckle   what FIVES uses now
       palm width              index knuckle <-> pinky knuckle
       whole palm              average of wrist->each knuckle + palm width (more bones = less noise)
   and one that isn't size at all:
       tip depth (z)           MediaPipe's depth guess for the index tip, relative to the wrist
   For each: the best single line ("bigger than this = ON"), and how often that line is right. 50% = useless,
   100% = perfect. "Smoothed" = the median of the last 5 frames, like a real-time filter would do.
   Also: is your "paper" drifting? (ON hand size in the first round vs the last.)

2. PEN GRIP vs RELAXED, from hand SHAPE (Hand Interfaces style)
   A small model looks at the 21 points (relative to the wrist, scaled by palm size, so WHERE your hand is and
   how BIG it looks don't matter, only its shape). Trained on your first rounds, tested on the last 2 rounds
   it never saw. Classification, like the letters, but 2 classes.

Run from the crosshair folder:  python lessons/lesson33.py
"""

import glob

import numpy as np
import torch
from torch import nn

ON, LIFTED, RELAXED, POINTING = 1, 0, 2, 3


def dist(p, a, b):
    return np.hypot(*(p[:, a] - p[:, b]).T)


def measures(p, z):
    whole = np.mean([dist(p, 0, 5), dist(p, 0, 9), dist(p, 0, 13), dist(p, 0, 17), dist(p, 5, 17)], axis=0)
    return {"wrist->middle knuckle": dist(p, 0, 9),
            "palm width": dist(p, 5, 17),
            "whole palm": whole,
            "tip depth (z)": -(z[:, 8] - z[:, 0])}       # minus: more negative z = closer, so bigger = closer


def causal_median(v, n=5):
    out = np.copy(v)
    for i in range(len(v)):
        w = v[max(0, i - n + 1): i + 1]
        w = w[~np.isnan(w)]
        out[i] = np.median(w) if len(w) else np.nan
    return out


def best_line(on, off):
    """Best threshold separating two groups (either direction). Returns (accuracy, threshold)."""
    vals = np.sort(np.concatenate([on, off]))
    best = (0.5, None)
    for th in vals[:: max(1, len(vals) // 400)]:
        acc = 0.5 * ((on >= th).mean() + (off < th).mean())     # balanced: both groups count equally
        acc = max(acc, 1 - acc)
        if acc > best[0]:
            best = (acc, th)
    return best


files = sorted(glob.glob("data_depth/*.npz"))
if not files:
    raise SystemExit("No recordings yet: run  python depth_rec.py  first.")
print(f"{len(files)} recording(s)\n")

all_meas, all_labels, shape_x, shape_y, shape_round = {}, [], [], [], []
timeline = None
for f in files:
    d = np.load(f)
    p, z, label, rnd = d["pts"].astype(float), d["z"].astype(float), d["label"], d["round"]
    ok = ~np.isnan(p).any(axis=(1, 2))
    m = measures(p, z)
    for k, v in m.items():
        all_meas.setdefault(k, {"raw": [], "smoothed": []})
        all_meas[k]["raw"].append(v)
        all_meas[k]["smoothed"].append(causal_median(v))
    all_labels.append(label)
    timeline = (d["t"] - d["t"][0], m["whole palm"], label)          # the picture shows the LATEST recording
    # shape features for question 2
    rel = (p - p[:, :1]) / m["whole palm"][:, None, None]
    shape_x.append(rel.reshape(len(p), -1)[ok])
    shape_y.append(label[ok])
    shape_round.append(rnd[ok])
labels = np.concatenate(all_labels)

# ---------- question 1 ----------
def table(pick, title):
    print(title)
    print(f"   {'measure':22s} {'ON median':>10s} {'LIFTED median':>14s} {'change':>7s} {'raw':>6s} {'smoothed':>9s}")
    lab = np.concatenate([all_labels[i] for i in pick])
    for k, v in all_meas.items():
        raw = np.concatenate([v["raw"][i] for i in pick])
        sm = np.concatenate([v["smoothed"][i] for i in pick])
        on, off = raw[(lab == ON) & ~np.isnan(raw)], raw[(lab == LIFTED) & ~np.isnan(raw)]
        son, soff = sm[(lab == ON) & ~np.isnan(sm)], sm[(lab == LIFTED) & ~np.isnan(sm)]
        if len(on) < 10 or len(off) < 10:
            continue
        change = (np.median(on) / np.median(off) - 1) * 100 if k != "tip depth (z)" else float("nan")
        print(f"   {k:22s} {np.median(on):10.3f} {np.median(off):14.3f} "
              f"{'' if np.isnan(change) else f'{change:+6.0f}%':>7s} {best_line(on, off)[0] * 100:5.0f}% "
              f"{best_line(son, soff)[0] * 100:8.0f}%")


print("1. PEN ON THE PAPER vs LIFTED  (best single line; 50% = useless, 100% = perfect)")
for i, f in enumerate(files):
    table([i], f"\n   recording {i + 1}: {f.split('/')[-1]}")
if len(files) > 1:
    table(list(range(len(files))), "\n   ALL recordings together (one line for every session: does it carry over?)")

d0 = np.load(files[-1])
p, lab, rnd = d0["pts"].astype(float), d0["label"], d0["round"]
w = measures(p, d0["z"].astype(float))["whole palm"]
print("\n   Is the paper drifting? 'whole palm' size while ON, round by round (latest recording):")
print("   " + "  ".join(f"r{r + 1} {np.nanmedian(w[(lab == ON) & (rnd == r)]):.3f}"
                       for r in np.unique(rnd) if ((lab == ON) & (rnd == r)).any()))
print("   (if it creeps up or down, your 'paper' moves as you write, and the line would need to follow it)")

# ---------- question 2 ----------
X, Y, R = np.concatenate(shape_x), np.concatenate(shape_y), np.concatenate(shape_round)
keep = Y >= 0
X, Y, R = X[keep], np.isin(Y[keep], [ON, LIFTED]).astype(np.int64), R[keep]   # 1 = pen grip (on or lifted)
last = R >= R.max() - 1
if last.all() or not last.any() or len(np.unique(Y[~last])) < 2:
    print("\n2. (not enough rounds to test the grip model)")
else:
    torch.manual_seed(0)
    xt, yt = torch.tensor(X[~last], dtype=torch.float32), torch.tensor(Y[~last])
    xs, ys = torch.tensor(X[last], dtype=torch.float32), torch.tensor(Y[last])
    model = nn.Sequential(nn.Linear(42, 32), nn.ReLU(), nn.Linear(32, 2))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(600):
        loss = nn.functional.cross_entropy(model(xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        guess = model(xs).argmax(1).numpy()
    yn = ys.numpy()
    print(f"\n2. PEN GRIP vs NOT (relaxed / pointing)  (trained on rounds 1-{R.max() - 1}, tested on the last 2 rounds)")
    print(f"   right {np.mean(guess == yn) * 100:.0f}% of frames")
    print(f"   pen grip frames called 'not pen': {np.mean(guess[yn == 1] == 0) * 100:.0f}%   "
          f"(ink would DROP OUT)")
    print(f"   not-pen frames called 'pen grip': {np.mean(guess[yn == 0] == 1) * 100:.0f}%   "
          f"(PHANTOM ink)")
    import os
    os.makedirs("runs", exist_ok=True)
    torch.save(model.state_dict(), "runs/grip_model.pt")            # for later: the pen-grip detector

# ---------- picture ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, size, lab = timeline
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
    colors = {ON: "tab:green", LIFTED: "tab:blue", RELAXED: "tab:gray", POINTING: "tab:pink", -1: "white"}
    for k, c in colors.items():
        if k >= 0:
            m = lab == k
            a1.scatter(t[m], size[m], s=4, color=c, label={ON: "ON the paper", LIFTED: "LIFTED", RELAXED: "RELAXED", POINTING: "POINTING"}[k])
    a1.set_xlabel("seconds")
    a1.set_ylabel("hand size ('whole palm')")
    a1.set_title(f"Hand size over time (latest recording: {files[-1].split('/')[-1]})")
    a1.legend(markerscale=4)
    raw, latest = all_meas["whole palm"]["smoothed"][-1], all_labels[-1]
    bins = np.linspace(np.nanpercentile(raw, 1), np.nanpercentile(raw, 99), 40)
    a2.hist(raw[latest == ON], bins=bins, alpha=0.6, color="tab:green", label="ON")
    a2.hist(raw[latest == LIFTED], bins=bins, alpha=0.6, color="tab:blue", label="LIFTED")
    a2.set_title("ON vs LIFTED (smoothed): two separate hills = it works")
    a2.legend()
    fig.tight_layout()
    fig.savefig("lesson33.png", dpi=120)
    print("\nSaved lesson33.png  (open lesson33.png)")
except ImportError:
    pass
