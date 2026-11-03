"""Lesson 26: would MORE DATA make the CNN better? (fast learning curve)

Lesson 19's learning curve used the old, memorizing setup. Now that dropout stopped the memorizing, ask again with
the real model: CNN hand + dropout, learning rate 0.001, Huber, mirrored, 5000 steps (exactly lesson 25's winner).

It trains on 1/3, 2/3 and all of the training games (same 3 test folds each time) and watches the test error.
    still dropping from 2/3 to all  -> more games will help: worth recording more (and friends)
    flat                            -> this model has what it needs from this kind of data

Only the CNN, 3 sizes, 3 folds: ~6-10 minutes on the GPU.

Run:  python lessons/lesson26.py
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


FRACTIONS = [1 / 3, 2 / 3, 1.0]
files = sorted(glob.glob("data/*.npz"))
order = list(np.random.default_rng(0).permutation(len(files)))
files = [files[i] for i in order]
folds = [files[k::6] for k in range(6)][:3]
print(f"{len(files)} recordings, 3 folds, sizes 1/3, 2/3, all, running on: {DEVICE}\n")

results = {f: [] for f in FRACTIONS}
damped = {f: [] for f in FRACTIONS}
n_games = {}
start = time.time()
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    val_files, pool = rest[-3:], rest[:-3]
    th, ty, _ = load(test_files)
    vh, vy, _ = load(val_files)
    for frac in FRACTIONS:
        use = pool[: max(1, round(len(pool) * frac))]
        n_games.setdefault(frac, len(use))
        fh, fy, _ = load(use)
        mh, my = mirror(fh, fy)
        dk = damp_of(fh, fy)
        damped[frac].append(to_px(((th[:, -1, 0] - th[:, -2, 0]) * dk - ty).norm(dim=1)).mean().item())
        model, step = fit(CNN(0.3), 0.001, "huber", mh, my, vh, vy)
        results[frac].append(px_err(model, th, ty).mean())
        print(f"fold {k + 1}, {len(use):2d} games: CNN {results[frac][-1]:.2f} px (kept at step {step})"
              f"   ({time.time() - start:.0f} s)")

print(f"\n{'training games':>16s}  {'CNN + dropout':>14s}  {'damped speed':>13s}  {'CNN better by':>14s}")
for f in FRACTIONS:
    c, d = np.mean(results[f]), np.mean(damped[f])
    print(f"{n_games[f]:16d}  {c:14.2f}  {d:13.2f}  {(d - c) / d * 100:13.1f}%")
a, b, c = (np.mean(results[f]) for f in FRACTIONS)
print(f"\nfirst third -> two thirds: {a - b:+.2f} px      two thirds -> all: {b - c:+.2f} px")
if b - c > 0.15:
    print("Still dropping: more games should help. Recording more (and friends' games) is worth it.")
elif b - c > -0.15:
    print("About flat: more of the same games won't change much. Different data (other people) might.")
else:
    print("Got worse with more data: probably noise between folds. Treat it as flat.")
