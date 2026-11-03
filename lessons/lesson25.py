"""Lesson 25: the honest check. Does CNN + dropout really beat everything?

Lesson 24 (3 folds): CNN hand + dropout 17.41 px vs MLP tip 18.35 vs damped speed 20.13. Before believing it:
  1. ALL 6 FOLDS, not 3, so one lucky split can't decide it.
  2. FAIRNESS: the MLP gets the same treatment (dropout, calm learning rate 0.001, mirrored data).
  3. LONGER training (5000 steps): dropout's curve was still going down at 3000.
  4. A new idea: PREDICT THE CORRECTION. Instead of the whole answer, the model predicts only what damped speed
     gets wrong:  guess = damped speed's guess + model's correction.  If the model learns nothing, it's exactly
     damped speed; anything it learns is pure improvement on top.
  5. THE LOCKBOX: if a data_final/ folder exists (games you never used), the winner is trained once on all of
     data/ and scored on those games, once. That's the final exam.

Run:  python lessons/lesson25.py          (6 folds, ~30-40 min on the GPU)
      python lessons/lesson25.py --quick  (3 folds)
"""

import copy
import glob
import sys
import time

import numpy as np
import torch
from torch import nn

HISTORY = 8
AHEAD = 2
GLITCH = 0.15
FOLDS = 3 if "--quick" in sys.argv else 6
HUBER = 0.1          # Huber loss: misses bigger than this (~15 px) count less harshly
SCREEN_PX = 1512
SCALE = 10
STEPS = 5000
CHECK_EVERY = 50
MAX_GAP = 0.1
FILTERS = 32         # how many "questions" each conv layer asks
WIDTH = 3            # how many frames (or stops) the magnifying glass covers

TIP, KNUCKLE, WRIST = 8, 5, 0
HAND = [TIP, KNUCKLE, WRIST]
SPIKE_MIN, SPIKE_BACK, TOO_FAR = 0.03, 0.5, 1.0
MOVING, TURN_DEG, SPEED_CHANGE = 4.0, 30, 1.3

# Which chip does the math: "mps" = Apple's GPU, "cpu" = the regular processor.
DEVICE = torch.device("mps" if torch.backends.mps.is_available() and "--cpu" not in sys.argv else "cpu")


# ---------- data (same as lesson 15) ----------
def flag_frames(pts):
    tip, knuckle = pts[:, TIP, :].astype(np.float64), pts[:, KNUCKLE, :].astype(np.float64)
    bad = np.zeros(len(tip), bool)
    out = np.linalg.norm(tip[1:-1] - tip[:-2], axis=1)
    back = np.linalg.norm(tip[1:-1] - tip[2:], axis=1)
    across = np.linalg.norm(tip[2:] - tip[:-2], axis=1)
    smaller = np.minimum(out, back)
    bad[1:-1] |= (smaller > SPIKE_MIN) & (across < SPIKE_BACK * smaller)
    rel = tip - knuckle
    length = np.linalg.norm(rel, axis=1)
    change = np.linalg.norm(rel[1:] - rel[:-1], axis=1)
    bad[1:] |= change > TOO_FAR * np.maximum(length[1:], length[:-1])
    return bad


def make_examples(pts, t):
    """Like before, plus each example's STREAK: how many clean frames in a row led up to 'now'."""
    tip = pts[:, TIP, :]
    bad = flag_frames(pts)
    dt = np.diff(t, prepend=t[0] - 1 / 30)
    jump = np.r_[0, np.abs(np.diff(tip, axis=0)).max(axis=1)]
    clean = ~np.isnan(tip).any(axis=1) & ~bad & (dt <= MAX_GAP) & ~(jump > GLITCH)
    streak = np.zeros(len(tip), int)
    for i in range(len(tip)):
        streak[i] = (streak[i - 1] + 1 if i > 0 else 1) if clean[i] else 0
    hands, answers, streaks = [], [], []
    for i in range(HISTORY - 1, len(tip) - AHEAD):
        a, b = i - HISTORY + 1, i + AHEAD + 1
        path = tip[a:b]
        if np.isnan(path).any() or np.abs(path[1:] - path[:-1]).max() > GLITCH:
            continue
        if bad[a:b].any() or dt[a + 1 : b].max() > MAX_GAP:
            continue
        hands.append(pts[a : i + 1][:, HAND])
        answers.append(tip[i + AHEAD])
        streaks.append(streak[i])
    return hands, answers, streaks


def load(file_list):
    H, Y, S = [], [], []
    for f in file_list:
        d = np.load(f)
        pts = d["pts"].astype(np.float32)
        t = d["t"] if "t" in d.files else np.arange(len(pts)) / 30
        h, y, s = make_examples(pts, t)
        H += h
        Y += y
        S += s
    hands = torch.tensor(np.array(H), dtype=torch.float32)
    y = torch.tensor(np.array(Y), dtype=torch.float32)
    now = hands[:, -1, 0, :]
    return ((hands - now[:, None, None, :]) * SCALE).to(DEVICE), ((y - now) * SCALE).to(DEVICE), np.array(S)


def to_px(v):
    return v * SCREEN_PX / SCALE


# ---------- the models ----------
def mirror(h, y):
    """Flip left <-> right: negate every x. Returns the ORIGINAL + mirrored examples together (twice the data)."""
    h2, y2 = h.clone(), y.clone()
    h2[..., 0] *= -1
    y2[:, 0] *= -1
    return torch.cat([h, h2]), torch.cat([y, y2])


class CNN(nn.Module):
    """CNN on tip + knuckle + wrist (8 frames), with dropout."""

    def __init__(self, drop=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, FILTERS, kernel_size=WIDTH), nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(FILTERS, FILTERS, kernel_size=WIDTH), nn.ReLU(), nn.Dropout(drop),
        )
        self.head = nn.Sequential(nn.Linear(FILTERS * 4, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 2))

    def forward(self, hands):
        x = hands[:, -8:].reshape(len(hands), 8, -1).permute(0, 2, 1)
        return self.head(self.conv(x).reshape(len(hands), -1))


class MLP(nn.Module):
    """Fingertip only, last 8 frames, optional dropout."""

    def __init__(self, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Dropout(drop),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 2))

    def forward(self, hands):
        return self.net(hands[:, -8:, 0, :].reshape(len(hands), -1))


class Correction(nn.Module):
    """damped speed's guess + a CNN's correction on top."""

    def __init__(self, damp_k):
        super().__init__()
        self.damp_k = damp_k
        self.cnn = CNN(0.3)

    def forward(self, hands):
        damped = (hands[:, -1, 0] - hands[:, -2, 0]) * self.damp_k
        return damped + self.cnn(hands)


def mirror(h, y):
    h2, y2 = h.clone(), y.clone()
    h2[..., 0] *= -1
    y2[:, 0] *= -1
    return torch.cat([h, h2]), torch.cat([y, y2])


# name: (build(damp_k), learning rate, loss, mirrored?)
VARIANTS = {
    "MLP tip (old way)":         (lambda k: MLP(0.0), 0.01, "mse", False),
    "MLP tip + dropout (fair)":  (lambda k: MLP(0.3), 0.001, "huber", True),
    "CNN hand + dropout":        (lambda k: CNN(0.3), 0.001, "huber", True),
    "damped + CNN correction":   (lambda k: Correction(k), 0.001, "huber", True),
}
WINNER = "CNN hand + dropout"          # the one that goes to the lockbox (change it if the table says otherwise)


def px_err(model, h, y):
    model.eval()
    with torch.no_grad():
        e = to_px((model(h) - y).norm(dim=1)).cpu().numpy()
    model.train()
    return e


def fit(build, lr, kind, x, y, vx, vy):
    torch.manual_seed(0)
    model = build.to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_err, best_step = copy.deepcopy(model.state_dict()), float("inf"), 0
    for step in range(STEPS + 1):
        if step % CHECK_EVERY == 0:
            v = px_err(model, vx, vy).mean()
            if v < best_err:
                best_err, best, best_step = v, copy.deepcopy(model.state_dict()), step
        if step == STEPS:
            break
        guess = model(x)
        loss = nn.functional.huber_loss(guess, y, delta=HUBER) if kind == "huber" else nn.functional.mse_loss(guess, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.load_state_dict(best)
    return model, best_step


def damp_of(h, y):
    m1 = h[:, -1, 0] - h[:, -2, 0]
    return ((m1 * y).sum() / (m1 * m1).sum()).item()


FOLDS = 3 if "--quick" in sys.argv else 6
files = sorted(glob.glob("data/*.npz"))
order = list(np.random.default_rng(0).permutation(len(files)))
files = [files[i] for i in order]
folds = [files[k::6] for k in range(6)][:FOLDS]
print(f"{len(files)} recordings, {FOLDS} of 6 folds, {STEPS} steps, running on: {DEVICE}\n")

errs = {"damped speed": []} | {n: [] for n in VARIANTS}
steps = {n: [] for n in VARIANTS}
start = time.time()
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    th, ty, _ = load(test_files)
    vh, vy, _ = load(rest[-3:])
    fh, fy, _ = load(rest[:-3])
    mh, my = mirror(fh, fy)
    dk = damp_of(fh, fy)
    errs["damped speed"].append(to_px(((th[:, -1, 0] - th[:, -2, 0]) * dk - ty).norm(dim=1)).cpu().numpy())
    for name, (build, lr, kind, mirrored) in VARIANTS.items():
        xh, xy = (mh, my) if mirrored else (fh, fy)
        model, step = fit(build(dk), lr, kind, xh, xy, vh, vy)
        steps[name].append(step)
        errs[name].append(px_err(model, th, ty))
    print(f"fold {k + 1}: " + " | ".join(f"{n.split(' (')[0]} {errs[n][-1].mean():.1f}" for n in errs)
          + f"   ({time.time() - start:.0f} s)")

ref = [e.mean() for e in errs["MLP tip + dropout (fair)"]]
print(f"\n{'model':26s}  overall   beats fair MLP   kept at step")
for name, per_fold in errs.items():
    e = np.concatenate(per_fold)
    vs = "      -       " if name in ("damped speed", "MLP tip + dropout (fair)") else \
        f"    {sum(f.mean() < r for f, r in zip(per_fold, ref))} of {len(ref)}     "
    print(f"{name:26s}  {e.mean():6.2f}  {vs}  {steps.get(name, '')}")

# ---------- the lockbox ----------
final = sorted(glob.glob("data_final/*.npz"))
if not final:
    print("\nNo data_final/ folder yet: play 5 games, move them there, and run this again for the final exam.")
else:
    print(f"\nLOCKBOX: training on all {len(files)} games, scoring once on {len(final)} untouched games...")
    th, ty, _ = load(final)
    vh, vy, _ = load(files[-3:])
    fh, fy, _ = load(files[:-3])
    mh, my = mirror(fh, fy)
    dk = damp_of(fh, fy)
    damped = to_px(((th[:, -1, 0] - th[:, -2, 0]) * dk - ty).norm(dim=1)).mean().item()
    results = {"damped speed": damped}
    for name in ("MLP tip + dropout (fair)", WINNER):
        build, lr, kind, mirrored = VARIANTS[name]
        model, _ = fit(build(dk), lr, kind, mh if mirrored else fh, my if mirrored else fy, vh, vy)
        results[name] = px_err(model, th, ty).mean()
        if name == WINNER:
            import os
            os.makedirs("runs", exist_ok=True)
            torch.save(model.state_dict(), "runs/best_cnn.pt")
    for name, e in results.items():
        print(f"   {name:26s} {e:6.2f} px")
    print("   (winner's weights saved to runs/best_cnn.pt, ready for FIVES)")
