"""Lesson 20: squeezing the motion model. Three ideas, alone and combined.

Starting point: your best model, CNN hand + mirrored (lesson 19). Three changes to try:

  MOVES        Feed the model frame-to-frame CHANGES (how far each point moved each frame) instead of positions.
               Speed and turning are then right there in the numbers instead of hidden in differences the model
               has to work out itself.
  16 FRAMES    See 16 frames (about half a second) instead of 8, so the whole shape of a movement fits:
               where it started, the speed-up, the start of a curve.
  HUBER LOSS   MSE squares every miss, so a few wild, unpredictable moments (a sudden change of mind) pull the
               whole model toward them. Huber treats small misses like MSE but big ones more gently:
               it focuses on what CAN be predicted.

FAIRNESS: every model is graded on exactly the same examples. Examples carry 16 frames; models that use 8 just
look at the last 8. (So the numbers here are a bit different from lesson 19: needing 16 clean frames in a row
drops some examples.)

All models train on mirrored data (lesson 19's free win) except the two baselines.
3 folds by default (~20-30 min on the GPU), or  --full  for 6 folds (twice as long, more trustworthy).

Run:  python lessons/lesson20.py
"""

import copy
import glob
import sys
import time

import numpy as np
import torch
from torch import nn

HISTORY = 16         # examples carry 16 frames; each model uses the last 8 or all 16
AHEAD = 2
GLITCH = 0.15
FOLDS = 6 if "--full" in sys.argv else 3
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
    """Baseline: fingertip only, last 8 positions."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, hands):
        return self.net(hands[:, -8:, 0, :].reshape(len(hands), -1))


class CNN(nn.Module):
    """CNN on tip + knuckle + wrist. frames = how many recent frames it sees; moves = feed changes, not positions."""

    def __init__(self, frames=8, moves=False):
        super().__init__()
        self.frames, self.moves = frames, moves
        steps = frames - 1 if moves else frames                     # n positions -> n-1 moves between them
        self.conv = nn.Sequential(
            nn.Conv1d(6, FILTERS, kernel_size=WIDTH), nn.ReLU(),
            nn.Conv1d(FILTERS, FILTERS, kernel_size=WIDTH), nn.ReLU(),
        )
        stops = steps - 2 * (WIDTH - 1)
        self.head = nn.Sequential(nn.Linear(FILTERS * stops, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, hands):
        x = hands[:, -self.frames:]                                 # (n, frames, 3 points, 2)
        if self.moves:
            x = x[:, 1:] - x[:, :-1]                                # how far each point moved each frame
        x = x.reshape(len(hands), x.shape[1], -1).permute(0, 2, 1)  # (n, 6 channels, time)
        return self.head(self.conv(x).reshape(len(hands), -1))


# name: (build, mirrored training data?, loss)
VARIANTS = {
    "MLP tip (baseline)":      (lambda: MLP(), False, "mse"),
    "CNN hand (best so far)":  (lambda: CNN(8), True, "mse"),
    #"  + moves":               (lambda: CNN(8, moves=True), True, "mse"),
   # "  + 16 frames":           (lambda: CNN(16), True, "mse"),
    "  + Huber loss":          (lambda: CNN(8), True, "huber"),
    "  Huber + moves":         (lambda: CNN(8, moves=True), True, "huber"),
    #"  all three":             (lambda: CNN(16, moves=True), True, "huber"),
}


def loss_fn(kind, guess, truth):
    if kind == "huber":
        return nn.functional.huber_loss(guess, truth, delta=HUBER)
    return nn.functional.mse_loss(guess, truth)


def train(build, kind, x, y, vx, vy):
    torch.manual_seed(0)
    model = build().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best, best_err = copy.deepcopy(model.state_dict()), float("inf")
    for step in range(STEPS):
        loss = loss_fn(kind, model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % CHECK_EVERY == 0:
            with torch.no_grad():
                err = (model(vx) - vy).norm(dim=1).mean().item()    # early stopping always uses real px error
            if err < best_err:
                best_err, best = err, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model


def mirror(h, y):
    """Flip left <-> right: negate every x. Returns the ORIGINAL + mirrored examples together (twice the data)."""
    h2, y2 = h.clone(), y.clone()
    h2[..., 0] *= -1
    y2[:, 0] *= -1
    return torch.cat([h, h2]), torch.cat([y, y2])


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
order = list(np.random.default_rng(0).permutation(len(files)))
files = [files[i] for i in order]
folds = [files[k::FOLDS] for k in range(FOLDS)]
print(f"{len(files)} recordings, {FOLDS} folds, running on: {DEVICE}\n")

errs = {"damped speed": []} | {name: [] for name in VARIANTS}
labels = []
start = time.time()
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    th, ty = load(test_files)
    vh, vy = load(rest[-1:])
    fh, fy = load(rest[:-1])
    mh, my = mirror(fh, fy)
    m1 = fh[:, -1, 0] - fh[:, -2, 0]
    damp_k = ((m1 * fy).sum() / (m1 * m1).sum()).item()
    errs["damped speed"].append(to_px(((th[:, -1, 0] - th[:, -2, 0]) * damp_k - ty).norm(dim=1)).cpu().numpy())
    for name, (build, mirrored, kind) in VARIANTS.items():
        xh, xy = (mh, my) if mirrored else (fh, fy)
        model = train(build, kind, xh, xy, vh, vy)
        with torch.no_grad():
            errs[name].append(to_px((model(th) - ty).norm(dim=1)).cpu().numpy())
    labels.append(bucket_of(th, ty))
    print(f"fold {k + 1} done   ({len(ty)} test examples, {time.time() - start:.0f} s so far)")

labels = np.concatenate(labels)
ref = [e.mean() for e in errs["CNN hand (best so far)"]]
cols = ["steady", "speeding up", "slowing down", "turning"]
print("\n(clean test examples, px; lower is better)")
print(f"{'model':24s}  overall   beats 'best so far'   " + "  ".join(f"{c:>12s}" for c in cols))
for name, per_fold in errs.items():
    e = np.concatenate(per_fold)
    vs = "        -          " if name in ("damped speed", "CNN hand (best so far)") else \
        f"       {sum(f.mean() < r for f, r in zip(per_fold, ref))} of {len(per_fold)}       "
    print(f"{name:24s}  {e.mean():6.2f}  {vs}  " + "  ".join(f"{e[labels == c].mean():12.1f}" for c in cols))
print("\nA change is worth keeping if it lowers 'overall' AND wins most folds against 'best so far'.")
