"""Lesson 23: training curves. Were the CNNs really trained?

Lesson 22 showed early stopping kept the models at step 100-300 out of 2000: they were barely trained.
Before trusting any comparison, look at HOW training goes. A training curve plots two things over the steps:
    train error   how wrong the model is on the games it learns from (should keep going down)
    val error     how wrong it is on games it never learns from (what we actually care about)
Patterns to recognize:
    both drop, then val turns UP while train keeps dropping   -> overfitting (memorizing)
    both jump around wildly                                   -> learning rate too high (steps too big)
    both drop slowly and steadily                             -> healthy (maybe needs more steps)

Changes from before:
  - three LEARNING RATES for the CNN: 0.01 (what we used), 0.003, 0.001
  - THREE validation games instead of one, so the "best moment" isn't a random dip in one game
  - 3000 steps, checked every 25
  - test error is recorded too, ONLY for the plot. The model we keep is still chosen by validation.

Uses the first 3 of the 6 folds from lessons 20-22 (to keep it ~10 minutes). Saves training_curves.png.

Run:  python lessons/lesson23.py
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
STEPS = 3000
CHECK_EVERY = 25
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


# name: (build, learning rate)    all CNN hand + Huber + mirrored; MLP tip + MSE as the reference
VARIANTS = {
    "MLP tip, lr 0.01":     (lambda: MLP(), 0.01),
    "CNN hand, lr 0.01":    (lambda: CNN(8, points=3), 0.01),
    "CNN hand, lr 0.003":   (lambda: CNN(8, points=3), 0.003),
    "CNN hand, lr 0.001":   (lambda: CNN(8, points=3), 0.001),
}


def px_err(model, h, y):
    with torch.no_grad():
        return to_px((model(h) - y).norm(dim=1)).mean().item()


def train_curve(build, lr, mlp, x, y, vx, vy, tx, ty, sx, sy):
    """Train, recording (step, train px, val px, test px) every CHECK_EVERY steps. Keeps the best-on-validation model."""
    torch.manual_seed(0)
    model = build().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_err, best_step = copy.deepcopy(model.state_dict()), float("inf"), 0
    curve = []
    for step in range(STEPS + 1):
        if step % CHECK_EVERY == 0:
            v = px_err(model, vx, vy)
            curve.append((step, px_err(model, sx, sy), v, px_err(model, tx, ty)))
            if v < best_err:
                best_err, best, best_step = v, copy.deepcopy(model.state_dict()), step
        if step == STEPS:
            break
        guess = model(x)
        loss = nn.functional.mse_loss(guess, y) if mlp else nn.functional.huber_loss(guess, y, delta=HUBER)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.load_state_dict(best)
    return model, best_step, np.array(curve)


files = sorted(glob.glob("data/*.npz"))
order = list(np.random.default_rng(0).permutation(len(files)))
files = [files[i] for i in order]
folds = [files[k::6] for k in range(6)][:3]
print(f"{len(files)} recordings, first 3 of 6 folds, running on: {DEVICE}\n")

curves = {n: [] for n in VARIANTS}
summary = {n: {"best_step": [], "test_at_best": [], "test_at_end": []} for n in VARIANTS}
damped = []
start = time.time()
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    val_files, train_files = rest[-3:], rest[:-3]              # THREE validation games now
    th, ty, _ = load(test_files)
    vh, vy, _ = load(val_files)
    fh, fy, _ = load(train_files)
    mh, my = mirror(fh, fy)
    pick = torch.randperm(len(fh), generator=torch.Generator().manual_seed(0))[:20000].to(DEVICE)
    sh, sy = fh[pick], fy[pick]                                # a sample of training examples, for the train curve
    m1 = fh[:, -1, 0] - fh[:, -2, 0]
    damp_k = ((m1 * fy).sum() / (m1 * m1).sum()).item()
    damped.append(to_px(((th[:, -1, 0] - th[:, -2, 0]) * damp_k - ty).norm(dim=1)).mean().item())
    for name, (build, lr) in VARIANTS.items():
        mlp = name.startswith("MLP")
        xh, xy = (fh, fy) if mlp else (mh, my)
        model, step, curve = train_curve(build, lr, mlp, xh, xy, vh, vy, th, ty, sh, sy)
        curves[name].append(curve)
        summary[name]["best_step"].append(step)
        summary[name]["test_at_best"].append(px_err(model, th, ty))
        summary[name]["test_at_end"].append(curve[-1, 3])
    print(f"fold {k + 1} done   ({time.time() - start:.0f} s so far)")

print(f"\ndamped speed (reference): {np.mean(damped):.2f} px")
print(f"\n{'model':20s}  {'kept at step':>18s}  {'test px (kept)':>15s}  {'test px (step 3000)':>20s}")
for name, r in summary.items():
    print(f"{name:20s}  {str(r['best_step']):>18s}  {np.mean(r['test_at_best']):15.2f}  {np.mean(r['test_at_end']):20.2f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(4.2 * len(VARIANTS), 4), sharey=True)
    for ax, name in zip(axes, VARIANTS):
        c = np.mean(curves[name], axis=0)                      # average over the folds
        ax.plot(c[:, 0], c[:, 1], label="train", color="tab:blue")
        ax.plot(c[:, 0], c[:, 2], label="validation", color="tab:orange")
        ax.plot(c[:, 0], c[:, 3], label="test", color="tab:green", alpha=0.5, ls="--")
        ax.axhline(np.mean(damped), color="gray", lw=0.8, ls=":", label="damped speed")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("error (px)")
    lo = min(np.mean(curves[n], axis=0)[:, 1:].min() for n in VARIANTS)
    axes[0].set_ylim(lo - 1, np.mean(damped) + 6)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("training_curves.png", dpi=110)
    print("\nSaved training_curves.png  (open it:  open training_curves.png)")
except ImportError:
    print("\n(pip install matplotlib for the plot)")
