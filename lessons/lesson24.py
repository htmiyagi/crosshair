"""Lesson 24: stop the memorizing. Dropout, weight decay, and rotate/scale augmentation.

Lesson 23's curves: the train error kept dropping while the test error climbed after a few hundred steps. The
model was memorizing the training games. Three standard fixes, each alone and all together, all on CNN hand +
Huber + mirrored at the calm learning rate 0.001:

  DROPOUT 0.3      During training, every step, 30% of the neurons are switched off at random. The model can't lean
                   on any single neuron (a memorized shortcut), so it has to learn patterns that hold up broadly.
                   IMPORTANT: dropout is only for training. Before testing we call model.eval(), which turns it
                   off (and model.train() turns it back on). Forgetting this is a classic bug.
  WEIGHT DECAY     A small, constant pull of every weight toward zero (lesson 8). Simpler models memorize less.
  ROTATE / SCALE   Augmentation, like mirroring, but fresh every step: each training example is randomly rotated
                   (up to 12 degrees) and resized (90-110%), together with its answer. A hand movement tilted a bit
                   or a bit bigger is still a real hand movement, so the model sees endless "new" examples.

Same 3 folds and 3 validation games as lesson 23. Plots training_curves_24.png.
Reference from lesson 23: MLP tip 18.35 px, CNN hand lr 0.001 18.85 px, damped speed 20.13 px.

Run:  python lessons/lesson24.py
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
def mirror(h, y):
    """Flip left <-> right: negate every x. Returns the ORIGINAL + mirrored examples together (twice the data)."""
    h2, y2 = h.clone(), y.clone()
    h2[..., 0] *= -1
    y2[:, 0] *= -1
    return torch.cat([h, h2]), torch.cat([y, y2])


class CNN(nn.Module):
    """CNN on tip + knuckle + wrist (8 frames), with optional dropout."""

    def __init__(self, drop=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, FILTERS, kernel_size=WIDTH), nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(FILTERS, FILTERS, kernel_size=WIDTH), nn.ReLU(), nn.Dropout(drop),
        )
        self.head = nn.Sequential(nn.Linear(FILTERS * 4, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 2))

    def forward(self, hands):
        x = hands[:, -8:].reshape(len(hands), 8, -1).permute(0, 2, 1)     # (n, 6 channels, 8 frames)
        return self.head(self.conv(x).reshape(len(hands), -1))


def rotate_scale(h, y):
    """A fresh random tilt (up to 12 degrees) and size (90-110%) for every example, applied to hand AND answer."""
    n = len(h)
    angle = (torch.rand(n, device=h.device) * 2 - 1) * np.radians(12)
    size = 0.9 + 0.2 * torch.rand(n, device=h.device)
    c, s = torch.cos(angle) * size, torch.sin(angle) * size
    R = torch.stack([torch.stack([c, -s], 1), torch.stack([s, c], 1)], 1)    # (n, 2, 2) rotation x size
    return torch.einsum("nfpk,njk->nfpj", h, R), torch.einsum("nk,njk->nj", y, R)


# name: (dropout, weight decay, rotate/scale?)
VARIANTS = {
    "no fix (lesson 23)": (0.0, 0.0, False),
    "+ dropout 0.3":      (0.3, 0.0, False),
    "+ weight decay":     (0.0, 0.1, False),
    "+ rotate/scale":     (0.0, 0.0, True),
    "all three":          (0.3, 0.1, True),
}
LR = 0.001


def px_err(model, h, y):
    model.eval()                                        # dropout OFF for measuring
    with torch.no_grad():
        e = to_px((model(h) - y).norm(dim=1)).mean().item()
    model.train()                                       # dropout back ON for training
    return e


def train_curve(drop, wd, aug, x, y, vx, vy, tx, ty, sx, sy):
    torch.manual_seed(0)
    model = CNN(drop).to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=wd)
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
        bx, by = rotate_scale(x, y) if aug else (x, y)
        loss = nn.functional.huber_loss(model(bx), by, delta=HUBER)
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
summary = {n: {"best_step": [], "kept": [], "end": []} for n in VARIANTS}
start = time.time()
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    th, ty, _ = load(test_files)
    vh, vy, _ = load(rest[-3:])
    fh, fy, _ = load(rest[:-3])
    mh, my = mirror(fh, fy)
    pick = torch.randperm(len(fh), generator=torch.Generator().manual_seed(0))[:20000].to(DEVICE)
    sh, sy = fh[pick], fy[pick]
    for name, (drop, wd, aug) in VARIANTS.items():
        model, step, curve = train_curve(drop, wd, aug, mh, my, vh, vy, th, ty, sh, sy)
        curves[name].append(curve)
        summary[name]["best_step"].append(step)
        summary[name]["kept"].append(px_err(model, th, ty))
        summary[name]["end"].append(curve[-1, 3])
    print(f"fold {k + 1} done   ({time.time() - start:.0f} s so far)")

print("\nreference (lesson 23): MLP tip 18.35 px, damped speed 20.13 px")
print(f"\n{'CNN hand, lr 0.001':20s}  {'kept at step':>18s}  {'test px (kept)':>15s}  {'test px (end)':>14s}  {'climb':>6s}")
for name, r in summary.items():
    kept, end = np.mean(r["kept"]), np.mean(r["end"])
    print(f"{name:20s}  {str(r['best_step']):>18s}  {kept:15.2f}  {end:14.2f}  {end - kept:+6.2f}")
print("\n'climb' = how much worse it got by the end of training. Near 0 = no more memorizing.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(4 * len(VARIANTS), 4), sharey=True)
    for ax, name in zip(axes, VARIANTS):
        c = np.mean(curves[name], axis=0)
        ax.plot(c[:, 0], c[:, 1], label="train", color="tab:blue")
        ax.plot(c[:, 0], c[:, 2], label="validation", color="tab:orange")
        ax.plot(c[:, 0], c[:, 3], label="test", color="tab:green", ls="--")
        ax.axhline(18.35, color="gray", lw=0.8, ls=":", label="MLP tip (lesson 23)")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("error (px)")
    lo = min(np.mean(curves[n], axis=0)[:, 1:].min() for n in VARIANTS)
    axes[0].set_ylim(lo - 1, 27)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("training_curves_24.png", dpi=110)
    print("\nSaved training_curves_24.png  (open it:  open training_curves_24.png)")
except ImportError:
    print("\n(pip install matplotlib for the plot)")
