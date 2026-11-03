"""Lesson 17: the gated CNN. Lesson 15's mixture of experts, with CNN experts from lesson 16.

Lesson 16: the tip CNN is best at steady movement, the hand CNN is best at turns and speeding up.
So: keep both as experts, and add a gate that decides, example by example, how much to trust each.

                      +-> tip CNN  -> tip guess  --+
    last 8 frames ----+-> hand CNN -> hand guess --+->  (1 - g) * tip guess + g * hand guess
                      +-> gate CNN -> g (0..1) ----+

The gate is a small CNN too (16 magnifying glasses), so it can spot "a turn might be starting" in the hand data.
All three parts are trained TOGETHER on the same loss: nobody tells the gate when to open.

Compared (6-fold cross-validation, clean examples, early stopping, GPU):
    MLP tip, CNN tip, CNN hand     from lesson 16, for reference
    gated CNN                      the new one

New column "beats MLP tip": in how many of the 6 folds the model beats the fingertip MLP. A real improvement
should win most folds, not just have a better average (one lucky fold can move an average).

Run:  python lessons/lesson17.py      (a few minutes on the GPU)
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
GAMES_PER_FOLD = 3
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
    tip = pts[:, TIP, :]
    bad = flag_frames(pts)
    dt = np.diff(t, prepend=t[0] - 1 / 30)
    hands, answers = [], []
    for i in range(HISTORY - 1, len(tip) - AHEAD):
        a, b = i - HISTORY + 1, i + AHEAD + 1
        path = tip[a:b]
        if np.isnan(path).any() or np.abs(path[1:] - path[:-1]).max() > GLITCH:
            continue
        if bad[a:b].any() or dt[a + 1 : b].max() > MAX_GAP:
            continue
        hands.append(pts[a : i + 1][:, HAND])
        answers.append(tip[i + AHEAD])
    return hands, answers


def load(file_list):
    H, Y = [], []
    for f in file_list:
        d = np.load(f)
        pts = d["pts"].astype(np.float32)
        t = d["t"] if "t" in d.files else np.arange(len(pts)) / 30
        h, y = make_examples(pts, t)
        H += h
        Y += y
    hands = torch.tensor(np.array(H), dtype=torch.float32)          # (n, 8 frames, 3 points, 2)
    y = torch.tensor(np.array(Y), dtype=torch.float32)
    now = hands[:, -1, 0, :]
    return ((hands - now[:, None, None, :]) * SCALE).to(DEVICE), ((y - now) * SCALE).to(DEVICE)


def to_px(v):
    return v * SCREEN_PX / SCALE


# ---------- the models ----------
class MLP(nn.Module):
    def __init__(self, n_points):
        super().__init__()
        self.n_points = n_points
        self.net = nn.Sequential(nn.Linear(HISTORY * n_points * 2, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(),
                                 nn.Linear(64, 2))

    def forward(self, hands):
        x = hands[:, :, : self.n_points, :].reshape(len(hands), -1)    # flatten: every number its own slot
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, n_points):
        super().__init__()
        self.n_points = n_points
        channels = n_points * 2                                        # x and y for each point
        self.conv = nn.Sequential(
            nn.Conv1d(channels, FILTERS, kernel_size=WIDTH), nn.ReLU(),   # 8 frames -> 6 stops
            nn.Conv1d(FILTERS, FILTERS, kernel_size=WIDTH), nn.ReLU(),    # 6 stops  -> 4 stops
        )
        stops = HISTORY - 2 * (WIDTH - 1)
        self.head = nn.Sequential(nn.Linear(FILTERS * stops, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, hands):
        x = hands[:, :, : self.n_points, :].reshape(len(hands), HISTORY, -1)   # (n, time, channels)
        x = x.permute(0, 2, 1)                                         # -> (n, channels, time), as Conv1d wants
        scores = self.conv(x)                                          # (n, FILTERS, stops): the rows of scores
        return self.head(scores.reshape(len(hands), -1))               # small MLP reads all the score rows


class GatedCNN(nn.Module):
    """Two CNN experts + a small CNN gate that blends them."""

    def __init__(self):
        super().__init__()
        self.tip_expert = CNN(1)
        self.hand_expert = CNN(3)
        self.gate = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=WIDTH), nn.ReLU(),           # 16 glasses on the hand data
            nn.Flatten(),                                             # (n, 16 filters x 6 stops) -> one list
            nn.Linear(16 * (HISTORY - WIDTH + 1), 1),                 # one number per example
        )

    def gate_value(self, hands):
        x = hands.reshape(len(hands), HISTORY, -1).permute(0, 2, 1)   # (n, 6 channels, 8 frames)
        return torch.sigmoid(self.gate(x))                            # squash into 0..1, shape (n, 1)

    def forward(self, hands):
        g = self.gate_value(hands)
        return (1 - g) * self.tip_expert(hands) + g * self.hand_expert(hands)


VARIANTS = {
    "MLP tip":   lambda: MLP(1),
    "CNN tip":   lambda: CNN(1),
    "CNN hand":  lambda: CNN(3),
    "gated CNN": GatedCNN,
}


def count_weights(model):
    return sum(p.numel() for p in model.parameters())


def train(build, x, y, vx, vy):
    torch.manual_seed(0)
    model = build().to(DEVICE)              # the model's weights live on the GPU too
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best, best_err = copy.deepcopy(model.state_dict()), float("inf")
    for step in range(STEPS):
        loss = nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % CHECK_EVERY == 0:
            with torch.no_grad():
                err = (model(vx) - vy).norm(dim=1).mean().item()
            if err < best_err:
                best_err, best = err, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model


def bucket_of(hands, y):
    tip = hands[:, :, 0, :]
    move_now = tip[:, -1] - tip[:, -2]
    move_next = y / AHEAD
    speed_now = to_px(move_now.norm(dim=1)).cpu().numpy()
    speed_next = to_px(move_next.norm(dim=1)).cpu().numpy()
    cos = nn.functional.cosine_similarity(move_now, move_next, dim=1).clamp(-1, 1).cpu().numpy()
    angle = np.degrees(np.arccos(cos))
    labels = np.full(len(y), "steady", dtype=object)
    slow_now, slow_next = speed_now < MOVING, speed_next < MOVING
    both = ~slow_now & ~slow_next
    labels[both & (speed_next > speed_now * SPEED_CHANGE)] = "speeding up"
    labels[both & (speed_next < speed_now / SPEED_CHANGE)] = "slowing down"
    labels[both & (angle > TURN_DEG)] = "turning"
    labels[~slow_now & slow_next] = "stopping"
    labels[slow_now & ~slow_next] = "starting"
    labels[slow_now & slow_next] = "still"
    return labels


# ---------- cross-validation ----------
files = sorted(glob.glob("data/*.npz"))
folds = [files[i : i + GAMES_PER_FOLD] for i in range(0, len(files), GAMES_PER_FOLD)]
print(f"{len(files)} recordings, {len(folds)} folds, running on: {DEVICE}\n")
print("weights in each model:")
for name, build in VARIANTS.items():
    print(f"   {name:9s} {count_weights(build()):6d}")
print()

errs = {"damped speed": []} | {name: [] for name in VARIANTS}
labels, gates = [], []
start = time.time()
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    th, ty = load(test_files)
    if len(ty) == 0:
        continue
    fh, fy = load(rest[:-1])
    vh, vy = load(rest[-1:])
    ah, ay = load(rest)
    m1 = ah[:, -1, 0] - ah[:, -2, 0]
    damp_k = ((m1 * ay).sum() / (m1 * m1).sum()).item()
    errs["damped speed"].append(to_px(((th[:, -1, 0] - th[:, -2, 0]) * damp_k - ty).norm(dim=1)).cpu().numpy())
    for name, build in VARIANTS.items():
        model = train(build, fh, fy, vh, vy)
        with torch.no_grad():
            errs[name].append(to_px((model(th) - ty).norm(dim=1)).cpu().numpy())
            if isinstance(model, GatedCNN):
                gates.append(model.gate_value(th).squeeze(1).cpu().numpy())
    labels.append(bucket_of(th, ty))
    print(f"fold {k + 1} done   ({time.time() - start:.0f} s so far)")

labels = np.concatenate(labels)
damp_folds = [e.mean() for e in errs["damped speed"]]
mlp_folds = [e.mean() for e in errs["MLP tip"]]
cols = ["steady", "speeding up", "slowing down", "turning"]
print("\n(clean test examples, px; lower is better)")
print(f"{'model':13s}  overall   beats damped   beats MLP tip   " + "  ".join(f"{c:>12s}" for c in cols))
for name, per_fold in errs.items():
    e = np.concatenate(per_fold)
    n = len(per_fold)
    vs_damp = "     -      " if name == "damped speed" else f"   {sum(f.mean() < d for f, d in zip(per_fold, damp_folds))} of {n}    "
    vs_mlp = "      -       " if name in ("damped speed", "MLP tip") else f"    {sum(f.mean() < m for f, m in zip(per_fold, mlp_folds))} of {n}     "
    print(f"{name:13s}  {e.mean():6.1f}  {vs_damp}  {vs_mlp}  " + "  ".join(f"{e[labels == c].mean():12.1f}" for c in cols))

g = np.concatenate(gates)
print("\nWhat did the CNN gate learn? Average g (0 = trusts tip CNN, 1 = trusts hand CNN):")
for name in ["still", "starting", "steady", "speeding up", "slowing down", "turning", "stopping"]:
    m = labels == name
    if m.any():
        print(f"   {name:14s} {g[m].mean():.2f}  {'#' * int(round(g[m].mean() * 40))}")
print(f"\nHow decisive is it? {np.mean(g < 0.2) * 100:.0f}% of examples mostly trust the tip (g < 0.2), "
      f"{np.mean(g > 0.8) * 100:.0f}% mostly trust the hand (g > 0.8).")
print("(Lesson 15's MLP gate stayed around 0.3 everywhere: leaning, never committing.)")
