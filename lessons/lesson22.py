"""Lesson 22: detective work. Does the CNN fall apart right after a tracking hiccup?

Lesson 20 (examples needing 16 clean frames): CNN + Huber 18.3 px, clearly better than damped speed (20.3).
Lesson 21 (examples needing only 8 clean frames): the same CNN ~20.3, no better than damped speed.
The extra examples in lesson 21 are mostly the ones RIGHT AFTER a hiccup: the hand was lost, a glitch was thrown
out, or the camera paused. Suspect: the CNN does badly there.

So every test example gets a tag: how many clean frames in a row came before "now" (its "streak").
    streak  8-11   just after a hiccup (the first ~0.3 s)
    streak 12-15
    streak 16+     settled (the only kind lesson 20 tested)
Then we compare models in each group.

Second question: if the CNN is bad after hiccups, is it the KNUCKLE + WRIST (they may jump when MediaPipe re-finds
the hand) or the CNN itself? So there's also a CNN that only sees the fingertip. If "CNN tip" is fine after hiccups
but "CNN hand" isn't, the extra hand points are the problem.

Models (all mirrored + Huber except the baselines): damped speed, MLP tip, CNN tip, CNN hand.
6 folds (or --quick for 3). Same shuffled folds as lessons 20-21.

Run:  python lessons/lesson22.py
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
STEPS = 2000
CHECK_EVERY = 100
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
class CNN(nn.Module):
    """CNN on tip + knuckle + wrist. frames = how many recent frames it sees; moves = feed changes, not positions."""

    def __init__(self, frames=8, moves=False, points=3):
        super().__init__()
        self.frames, self.moves, self.points = frames, moves, points
        steps = frames - 1 if moves else frames                     # n positions -> n-1 moves between them
        self.conv = nn.Sequential(
            nn.Conv1d(points * 2, FILTERS, kernel_size=WIDTH), nn.ReLU(),
            nn.Conv1d(FILTERS, FILTERS, kernel_size=WIDTH), nn.ReLU(),
        )
        stops = steps - 2 * (WIDTH - 1)
        self.head = nn.Sequential(nn.Linear(FILTERS * stops, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, hands):
        x = hands[:, -self.frames:, : self.points]                  # (n, frames, points, 2)
        if self.moves:
            x = x[:, 1:] - x[:, :-1]                                # how far each point moved each frame
        x = x.reshape(len(hands), x.shape[1], -1).permute(0, 2, 1)  # (n, 6 channels, time)
        return self.head(self.conv(x).reshape(len(hands), -1))


def mirror(h, y):
    """Flip left <-> right: negate every x. Returns the ORIGINAL + mirrored examples together (twice the data)."""
    h2, y2 = h.clone(), y.clone()
    h2[..., 0] *= -1
    y2[:, 0] *= -1
    return torch.cat([h, h2]), torch.cat([y, y2])


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, hands):
        return self.net(hands[:, -8:, 0, :].reshape(len(hands), -1))


# name: (build, mirrored?, loss)
VARIANTS = {
    "MLP tip":          (lambda: MLP(), False, "mse"),
    "CNN tip + Huber":  (lambda: CNN(8, points=1), True, "huber"),
    "CNN hand + Huber": (lambda: CNN(8, points=3), True, "huber"),
}


def train(build, kind, x, y, vx, vy):
    torch.manual_seed(0)
    model = build().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best, best_err, best_step = copy.deepcopy(model.state_dict()), float("inf"), 0
    for step in range(STEPS):
        guess = model(x)
        loss = nn.functional.huber_loss(guess, y, delta=HUBER) if kind == "huber" else nn.functional.mse_loss(guess, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % CHECK_EVERY == 0:
            with torch.no_grad():
                err = (model(vx) - vy).norm(dim=1).mean().item()
            if err < best_err:
                best_err, best, best_step = err, copy.deepcopy(model.state_dict()), step
    model.load_state_dict(best)
    return model, best_step


GROUPS = [("8-11 hiccup", 8, 11), ("12-15", 12, 15), ("16+ settled", 16, 10**9)]

files = sorted(glob.glob("data/*.npz"))
order = list(np.random.default_rng(0).permutation(len(files)))
files = [files[i] for i in order]
folds = [files[k::FOLDS] for k in range(FOLDS)]
print(f"{len(files)} recordings, {FOLDS} folds, running on: {DEVICE}\n")

errs = {"damped speed": []} | {n: [] for n in VARIANTS}
streaks, stops = [], {n: [] for n in VARIANTS}
start = time.time()
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    th, ty, ts = load(test_files)
    vh, vy, _ = load(rest[-1:])
    fh, fy, _ = load(rest[:-1])
    mh, my = mirror(fh, fy)
    m1 = fh[:, -1, 0] - fh[:, -2, 0]
    damp_k = ((m1 * fy).sum() / (m1 * m1).sum()).item()
    errs["damped speed"].append(to_px(((th[:, -1, 0] - th[:, -2, 0]) * damp_k - ty).norm(dim=1)).cpu().numpy())
    line = []
    for name, (build, mirrored, kind) in VARIANTS.items():
        xh, xy = (mh, my) if mirrored else (fh, fy)
        model, step = train(build, kind, xh, xy, vh, vy)
        stops[name].append(step)
        with torch.no_grad():
            e = to_px((model(th) - ty).norm(dim=1)).cpu().numpy()
        errs[name].append(e)
        line.append(f"{name} {e.mean():.1f}")
    streaks.append(ts)
    print(f"fold {k + 1}: damped {errs['damped speed'][-1].mean():.1f} | " + " | ".join(line)
          + f"   ({time.time() - start:.0f} s)")

S = np.concatenate(streaks)
print("\n(test px by streak = clean frames in a row before now; lower is better)")
print(f"{'model':18s}  {'all':>6s}" + "".join(f"  {g[0]:>13s}" for g in GROUPS))
print(f"{'share of examples':18s}  {'100%':>6s}" + "".join(f"  {((S >= lo) & (S <= hi)).mean() * 100:12.0f}%" for _, lo, hi in GROUPS))
for name, per_fold in errs.items():
    e = np.concatenate(per_fold)
    print(f"{name:18s}  {e.mean():6.2f}" + "".join(f"  {e[(S >= lo) & (S <= hi)].mean():13.2f}" for _, lo, hi in GROUPS))
print("\nearly stopping: the step each model was kept at, per fold (1900 = trained to the end, 0 = barely trained)")
for name, s in stops.items():
    print(f"   {name:18s} {s}")
print("\nIf 'CNN hand' is much worse than damped in the first group but fine in the last, hiccups are the problem."
      "\nIf 'CNN tip' is fine there too, it's the knuckle + wrist points that jump.")
