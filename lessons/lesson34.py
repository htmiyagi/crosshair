"""Lesson 34: making the pen-grip detector better. What should the model LOOK AT?

Lesson 33's grip model got 68%: it called your pen grip "relaxed" 44% of the time. Before collecting more data,
check whether we're feeding it the right information. Same model, five different INPUTS ("features"):

  A  2D shape          what lesson 33 used: the 21 points relative to the wrist, scaled by palm size
  B  + upright         same, but first ROTATED so your wrist->middle knuckle always points straight up
                       (a tilted hand and an upright hand become the same shape)
  C  + depth (z)       B plus MediaPipe's z for every point. From the front, a pen grip and a loose curled hand
                       look alike in 2D; in 3D they differ (where the fingertips point)
  D  hand-made         a few numbers a person would use to describe a pen grip, all size/position-proof:
                         - each finger's curl (lesson 30's measure)
                         - the TRIPOD: how close thumb, index and middle tips are (that's how you hold a pen)
                         - thumb tip to index knuckle (a pen rests there)
  E  everything        C + D

FAIRER TEST: instead of one split, each ROUND takes a turn as the test (train on the other 5, test on it), and we
average. That's cross-validation by round, the same idea as by game in Crosshair. Less luck in the number.
Also: the first 0.7 s of each phase is skipped (you're still moving into the pose), and "smoothed" = majority vote
over the last 5 frames, like a real-time filter.

Run from the crosshair folder:  python lessons/lesson34.py
"""

import glob

import numpy as np
import torch
from torch import nn

ON, LIFTED, RELAXED, POINTING = 1, 0, 2, 3
SETTLE = 0.7          # seconds skipped at the start of each phase
FINGERS = {"index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}


def palm(p):
    d = lambda a, b: np.linalg.norm(p[:, a, :2] - p[:, b, :2], axis=1)
    return np.mean([d(0, 5), d(0, 9), d(0, 13), d(0, 17), d(5, 17)], axis=0)


def upright(p):
    """Rotate each frame so wrist -> middle knuckle points straight up (x, y only)."""
    v = p[:, 9, :2] - p[:, 0, :2]
    ang = np.arctan2(v[:, 0], -v[:, 1])                      # angle away from "up"
    c, s = np.cos(-ang), np.sin(-ang)
    rel = p[:, :, :2] - p[:, :1, :2]
    x = rel[..., 0] * c[:, None] - rel[..., 1] * s[:, None]
    y = rel[..., 0] * s[:, None] + rel[..., 1] * c[:, None]
    return np.stack([x, y], -1)


def handmade(p):
    size = palm(p)[:, None]
    d = lambda a, b: np.linalg.norm(p[:, a] - p[:, b], axis=1)
    out = [d(0, tip) / np.maximum(d(0, k), 1e-6) for k, tip in FINGERS.values()]          # curls
    out += [d(4, 8) / size[:, 0], d(4, 12) / size[:, 0], d(8, 12) / size[:, 0]]            # tripod
    out += [d(4, 5) / size[:, 0], d(4, 6) / size[:, 0]]                                     # thumb near index base
    return np.stack(out, 1)


def features(p2, z, which):
    size = palm(p2)[:, None, None]
    flat = lambda a: a.reshape(len(a), -1)
    A = flat((p2 - p2[:, :1]) / size)
    B = flat(upright(p2) / size)
    C = np.concatenate([B, (z - z[:, :1]) / size[:, :, 0]], 1)
    p3 = np.concatenate([p2, z[..., None]], -1)
    D = handmade(p3)
    return {"A": A, "B": B, "C": C, "D": D, "E": np.concatenate([C, D], 1)}[which]


def smooth_votes(guess, n=5):
    out = np.copy(guess)
    for i in range(len(guess)):
        out[i] = round(np.mean(guess[max(0, i - n + 1): i + 1]))
    return out


def train_test(xtr, ytr, xte):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6                   # same scale for every input number
    xt = torch.tensor((xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(ytr)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(xtr.shape[1], 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 2))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    model.train()
    for _ in range(500):
        loss = nn.functional.cross_entropy(model(xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor((xte - mu) / sd, dtype=torch.float32)).argmax(1).numpy()


# ---------- load: pen grip (ON + LIFTED) = 1, anything else (RELAXED, POINTING) = 0 ----------
files = sorted(glob.glob("data_depth/*.npz"))
if not files:
    raise SystemExit("No recordings yet: run  python depth_rec.py  first.")
P, Z, Y, R, K = [], [], [], [], []
for k, f in enumerate(files):
    d = np.load(f)
    p, z, lab, rnd, t = d["pts"].astype(float), d["z"].astype(float), d["label"], d["round"], d["t"]
    start = np.zeros(len(lab), bool)            # frames within SETTLE seconds of a phase change
    change = np.nonzero(np.diff(lab) != 0)[0] + 1
    for c in change:
        start |= (t >= t[c]) & (t < t[c] + SETTLE)
    ok = (lab >= 0) & ~start & ~np.isnan(p).any(axis=(1, 2)) & ~np.isnan(z).any(1)
    P.append(p[ok])
    Z.append(z[ok])
    Y.append(np.isin(lab[ok], [ON, LIFTED]).astype(np.int64))
    K.append(lab[ok])                           # the exact pose, to see WHICH non-pen pose gets mistaken
    R.append(rnd[ok] + 100 * k)                 # each recording's rounds are separate test folds
P, Z, Y, R, K = (np.concatenate(a) for a in (P, Z, Y, R, K))
folds = np.unique(R)
print(f"{len(files)} recording(s), {len(Y)} frames ({Y.mean() * 100:.0f}% pen grip), {len(folds)} rounds as test folds\n")

NAMES = {"A": "2D shape (lesson 33)", "B": "+ upright", "C": "+ depth (z)", "D": "hand-made features",
         "E": "everything"}
has_point = (K == POINTING).any()
print(f"   {'input':22s} {'right':>6s} {'smoothed':>9s} {'grip->not':>10s} {'relaxed->grip':>14s}"
      + (f" {'POINTING->grip':>15s}" if has_point else ""))
print(f"   {'':22s} {'':>6s} {'':>9s} {'(drops)':>10s} {'(phantom)':>14s}" + (f" {'(phantom)':>15s}" if has_point else ""))
results = {}
for key, name in NAMES.items():
    X = features(P, Z, key)
    guess, sguess = np.zeros_like(Y), np.zeros_like(Y)
    for f in folds:
        te = R == f
        g = train_test(X[~te], Y[~te], X[te])
        guess[te], sguess[te] = g, smooth_votes(g)
    results[key] = sguess
    drop = np.mean(sguess[Y == 1] == 0) * 100
    phantom = np.mean(sguess[K == RELAXED] == 1) * 100
    point = np.mean(sguess[K == POINTING] == 1) * 100 if has_point else None
    print(f"   {name:22s} {np.mean(guess == Y) * 100:5.0f}% {np.mean(sguess == Y) * 100:8.0f}% "
          f"{drop:9.0f}% {phantom:13.0f}%" + (f" {point:14.0f}%" if has_point else ""))

# which hand-made numbers actually differ between grip and relaxed?
D = features(P, Z, "D")
labels = ["index curl", "middle curl", "ring curl", "pinky curl", "thumb-index tips", "thumb-middle tips",
          "index-middle tips", "thumb-index knuckle", "thumb-index middle joint"]
other = POINTING if has_point else RELAXED
print(f"\n   which hand-made numbers separate the pen grip from {'POINTING' if has_point else 'RELAXED'}?"
      " ('gap' = how far apart, in spreads)")
for i, lab in enumerate(labels):
    g, r = D[Y == 1, i], D[K == other, i]
    gap = abs(np.median(g) - np.median(r)) / (0.5 * (np.std(g) + np.std(r)) + 1e-9)
    print(f"      {lab:26s} grip {np.median(g):5.2f}   {'pointing' if has_point else 'relaxed'} {np.median(r):5.2f}"
          f"   gap {gap:4.1f}  {'#' * int(gap * 5)}")
print("   (gap above ~1.5 = clearly different; near 0 = the same in both)")


# ---------- save the pen for FIVES ----------
# The final pen = hand-made features (small, and they work as well as everything) trained on ALL the data,
# plus the depth lines from lesson 33 ("whole palm" hand size), saved as plain numbers so FIVES needs no torch.
def train_final(x, y):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt, yt = torch.tensor((x - mu) / sd, dtype=torch.float32), torch.tensor(y)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(x.shape[1], 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 2))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    model.train()
    for _ in range(500):
        loss = nn.functional.cross_entropy(model(xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, mu, sd


model, mu, sd = train_final(features(P, Z, "D"), Y)
size = palm(P)
on, off = size[K == ON], size[K == LIFTED]
best_acc, best_th = 0.0, None
for th in np.quantile(np.concatenate([on, off]), np.linspace(0.02, 0.98, 400)):
    acc = 0.5 * ((on >= th).mean() + (off < th).mean())
    if acc > best_acc:
        best_acc, best_th = acc, th
import os
os.makedirs("runs", exist_ok=True)
w = {k: v.detach().numpy() for k, v in model.state_dict().items()}
np.savez("runs/grip_pen.npz", W1=w["0.weight"], b1=w["0.bias"], W2=w["3.weight"], b2=w["3.bias"], mu=mu, sd=sd,
         on_line=best_th * 1.04, off_line=best_th * 0.96)
print(f"\nSaved runs/grip_pen.npz for FIVES: pen grip model + depth line {best_th:.3f} "
      f"(ink on above {best_th * 1.04:.3f}, off below {best_th * 0.96:.3f}; that line is right {best_acc * 100:.0f}% of frames)")
