"""Lesson 21: Huber for normal moves, MSE for fast ones. Can we get both?

Lesson 20: Huber loss made the CNN better on steady / slowing / turning, but WORSE on speeding up. Huber goes
easy on big misses, and big misses happen when the hand is fast. So: be strict (MSE) when the hand is fast or
speeding up, gentle (Huber) otherwise.

The key: we can tell "fast or speeding up" from the PAST frames, so it's usable at prediction time.
At 30 fps a reach takes several frames to speed up, so once it starts, the last few frames show it.
    speed  = how far the fingertip moved in the last frame (px)
    accel  = last move minus the average of the two before (px; positive = speeding up)
    "hot"  = speed above a threshold OR accel above a threshold

Two ways to use it:
  SWITCH       train two models (one MSE, one Huber). Each prediction: hot -> MSE model's guess, else Huber's.
               The thresholds are picked on the validation game (never the test games), from a few candidates.
  SPEED-AWARE  ONE model, trained with a mixed loss: MSE on hot examples, Huber on the rest.

Compared on the same 6 folds (or --quick for 3): damped speed, CNN + MSE, CNN + Huber, switch, speed-aware.
All CNNs: tip + knuckle + wrist, 8 frames, trained on mirrored data.

Run:  python lessons/lesson21.py
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


def speed_accel(h):
    """Fingertip speed and acceleration from the past frames (normalized units). h: (n, frames, 3, 2)."""
    m1 = (h[:, -1, 0] - h[:, -2, 0]).norm(dim=1)
    m2 = (h[:, -2, 0] - h[:, -3, 0]).norm(dim=1)
    m3 = (h[:, -3, 0] - h[:, -4, 0]).norm(dim=1)
    return m1, m1 - (m2 + m3) / 2


def hot(h, speed_cut, accel_cut):
    s, a = speed_accel(h)
    return (s > speed_cut) | (a > accel_cut)


def loss_fn(kind, guess, truth, h=None, cuts=None):
    if kind == "huber":
        return nn.functional.huber_loss(guess, truth, delta=HUBER)
    if kind == "speed-aware":                            # MSE on hot examples, Huber on the rest
        mse = ((guess - truth) ** 2).mean(dim=1)
        hub = nn.functional.huber_loss(guess, truth, delta=HUBER, reduction="none").mean(dim=1)
        return torch.where(hot(h, *cuts), mse, hub).mean()
    return nn.functional.mse_loss(guess, truth)


def train(kind, x, y, vx, vy, cuts=None):
    torch.manual_seed(0)
    model = CNN(8).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best, best_err = copy.deepcopy(model.state_dict()), float("inf")
    for step in range(STEPS):
        loss = loss_fn(kind, model(x), y, x, cuts)
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


QUANTILES = [0.5, 0.6, 0.7, 0.8, 0.9]          # threshold candidates: "faster than X% of training moments"

# ---------- cross-validation ----------
files = sorted(glob.glob("data/*.npz"))
order = list(np.random.default_rng(0).permutation(len(files)))
files = [files[i] for i in order]
folds = [files[k::FOLDS] for k in range(FOLDS)]
print(f"{len(files)} recordings, {FOLDS} folds, running on: {DEVICE}\n")

names = ["damped speed", "CNN + MSE", "CNN + Huber", "switch (MSE when hot)", "speed-aware loss"]
errs = {n: [] for n in names}
labels, hot_share, picked = [], [], []
start = time.time()
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    th, ty = load(test_files)
    vh, vy = load(rest[-1:])
    fh, fy = load(rest[:-1])
    mh, my = mirror(fh, fy)
    s, a = speed_accel(fh)
    cands = [(torch.quantile(s, q).item(), torch.quantile(a, q).item()) for q in QUANTILES]

    m1 = fh[:, -1, 0] - fh[:, -2, 0]
    damp_k = ((m1 * fy).sum() / (m1 * m1).sum()).item()
    errs["damped speed"].append(to_px(((th[:, -1, 0] - th[:, -2, 0]) * damp_k - ty).norm(dim=1)).cpu().numpy())

    mse_model = train("mse", mh, my, vh, vy)
    hub_model = train("huber", mh, my, vh, vy)
    aware_model = train("speed-aware", mh, my, vh, vy, cuts=cands[2])           # 70th percentile cuts
    with torch.no_grad():
        # pick the switch thresholds on the VALIDATION game (all speed x accel candidate pairs)
        gm, gh = mse_model(vh), hub_model(vh)
        best_cut, best_err = None, float("inf")
        for sc, _ in cands:
            for _, ac in cands:
                guess = torch.where(hot(vh, sc, ac)[:, None], gm, gh)
                err = (guess - vy).norm(dim=1).mean().item()
                if err < best_err:
                    best_cut, best_err = (sc, ac), err
        picked.append(best_cut)
        tm, thb = mse_model(th), hub_model(th)
        is_hot = hot(th, *best_cut)
        hot_share.append(is_hot.float().mean().item())
        errs["CNN + MSE"].append(to_px((tm - ty).norm(dim=1)).cpu().numpy())
        errs["CNN + Huber"].append(to_px((thb - ty).norm(dim=1)).cpu().numpy())
        errs["switch (MSE when hot)"].append(to_px((torch.where(is_hot[:, None], tm, thb) - ty).norm(dim=1)).cpu().numpy())
        errs["speed-aware loss"].append(to_px((aware_model(th) - ty).norm(dim=1)).cpu().numpy())
    labels.append(bucket_of(th, ty))
    print(f"fold {k + 1} done   ({time.time() - start:.0f} s so far)")

labels = np.concatenate(labels)
ref = [e.mean() for e in errs["CNN + Huber"]]
cols = ["steady", "speeding up", "slowing down", "turning"]
print("\n(clean test examples, px; lower is better)")
print(f"{'model':22s}  overall   beats Huber   " + "  ".join(f"{c:>12s}" for c in cols))
for name in names:
    e = np.concatenate(errs[name])
    vs = "     -     " if name in ("damped speed", "CNN + Huber") else \
        f"   {sum(f.mean() < r for f, r in zip(errs[name], ref))} of {len(ref)}   "
    print(f"{name:22s}  {e.mean():6.2f}  {vs}  " + "  ".join(f"{e[labels == c].mean():12.1f}" for c in cols))
sc = np.mean([p[0] for p in picked]) * SCREEN_PX / SCALE
ac = np.mean([p[1] for p in picked]) * SCREEN_PX / SCALE
print(f"\nswitch: 'hot' = fingertip moved more than ~{sc:.0f} px last frame, or sped up by ~{ac:.0f} px"
      f"  ->  {np.mean(hot_share) * 100:.0f}% of moments used the MSE model")
print("The goal: MSE's 'speeding up' number together with Huber's numbers everywhere else.")
