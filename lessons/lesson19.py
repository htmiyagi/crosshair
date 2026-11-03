"""Lesson 19: how much data do I need? A learning curve, plus free extra data (mirroring).

LEARNING CURVE: train each model on 25%, 50%, 75% and 100% of the training games, always testing on the same unseen
games, and watch the error as data grows.
    still going DOWN at 100%  -> this model is data-hungry: more games will help
    FLAT by 100%              -> this model has learned what it can from this kind of data

AUGMENTATION (mirroring): flip every training example left <-> right, as if a mirror-image you played. It's free
extra data: twice as many examples. Real ML does this constantly (flipping photos of cats: still a cat).
Only TRAINING data gets mirrored. Testing is always on your real, unflipped games, so the comparison stays fair.
Inputs are relative to the fingertip "now", so mirroring is just: x -> -x (for every point, and for the answer).

Models: MLP tip, CNN hand, transformer (from lessons 16-18), each with and without mirroring.
Uses 3 folds (a third of the games tested each round) to keep it under ~20 minutes on the GPU.
Saves learning_curve.png in the crosshair folder.

Run:  python lessons/lesson19.py
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
FOLDS = 3            # 3 rounds: each time a third of the games is the test set
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


EMBED = 32           # numbers per frame inside the transformer


class Attention(nn.Module):
    """One attention head, written out step by step."""

    def __init__(self, d):
        super().__init__()
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.last_weights = None                    # saved so we can look at them afterwards

    def forward(self, x):                           # x: (examples, 8 frames, d)
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = q @ k.transpose(1, 2)              # (n, 8, 8): how well each frame's query matches each key
        scores = scores / k.shape[-1] ** 0.5        # scale down, so softmax doesn't get too extreme
        weights = torch.softmax(scores, dim=-1)     # each row: 8 weights that add up to 1
        self.last_weights = weights.detach()
        return self.out(weights @ v)                # each frame: weighted average of everyone's values


class Block(nn.Module):
    """Attention, then a small MLP, each with a skip connection (ResNet's trick) and a LayerNorm."""

    def __init__(self, d):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attention(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))            # gather info from other frames
        x = x + self.mlp(self.norm2(x))             # think about it, frame by frame
        return x


class TinyTransformer(nn.Module):
    def __init__(self, n_points=3, blocks=2):
        super().__init__()
        self.n_points = n_points
        self.embed = nn.Linear(n_points * 2, EMBED)                    # 6 numbers -> 32 per frame
        self.position = nn.Parameter(torch.zeros(1, HISTORY, EMBED))   # a learned tag for each of the 8 slots
        self.blocks = nn.ModuleList([Block(EMBED) for _ in range(blocks)])
        self.norm = nn.LayerNorm(EMBED)
        self.head = nn.Linear(EMBED, 2)                                # "now" frame -> guess (x, y)

    def forward(self, hands):
        x = hands[:, :, : self.n_points, :].reshape(len(hands), HISTORY, -1)   # (n, 8 frames, 6)
        x = self.embed(x) + self.position
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, -1]))                         # read the answer from the last frame

    def now_attention(self):
        """What the last frame ('now') paid attention to, averaged over the blocks: (n, 8)."""
        return torch.stack([b.attn.last_weights[:, -1] for b in self.blocks]).mean(0)


VARIANTS = {
    "MLP tip":     lambda: MLP(1),
    "CNN hand":    lambda: CNN(3),
    "transformer": lambda: TinyTransformer(3),
}


def count_weights(model):
    return sum(p.numel() for p in model.parameters())


def train(build, x, y, vx, vy):
    torch.manual_seed(0)
    model = build().to(DEVICE)              # the model's weights live on the GPU too
    lr = 0.002 if isinstance(model, TinyTransformer) else 0.01   # transformers like a gentler learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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


# ---------- the learning curve ----------
def mirror(h, y):
    """Flip left <-> right: negate every x. Returns the ORIGINAL + mirrored examples together (twice the data)."""
    h2, y2 = h.clone(), y.clone()
    h2[..., 0] *= -1
    y2[:, 0] *= -1
    return torch.cat([h, h2]), torch.cat([y, y2])


FRACTIONS = [0.25, 0.5, 0.75, 1.0]
files = sorted(glob.glob("data/*.npz"))
rng = np.random.default_rng(0)
order = list(rng.permutation(len(files)))                 # shuffle once, so folds mix old and new games
files = [files[i] for i in order]
folds = [files[k::FOLDS] for k in range(FOLDS)]
print(f"{len(files)} recordings, {FOLDS} folds, running on: {DEVICE}\n")

results = {}            # (model, mirrored, fraction) -> list of test errors (px), one per fold
n_games = {}            # fraction -> how many training games that is (first fold)
start = time.time()
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    val_files, pool = rest[-1:], rest[:-1]                 # one game for early stopping, the rest to draw from
    th, ty = load(test_files)
    vh, vy = load(val_files)
    for frac in FRACTIONS:
        use = pool[: max(1, round(len(pool) * frac))]
        n_games.setdefault(frac, len(use))
        fh, fy = load(use)
        for mirrored in (False, True):
            xh, xy = mirror(fh, fy) if mirrored else (fh, fy)
            for name, build in VARIANTS.items():
                model = train(build, xh, xy, vh, vy)
                with torch.no_grad():
                    err = to_px((model(th) - ty).norm(dim=1)).mean().item()
                results.setdefault((name, mirrored, frac), []).append(err)
        print(f"fold {k + 1}: {int(frac * 100):3d}% of games ({len(use)}) done   ({time.time() - start:.0f} s so far)")

print("\n(test px, lower is better; columns = how many training games)")
print(f"{'model':28s}" + "".join(f"{n_games[f]:>9d} g" for f in FRACTIONS) + "    last step")
for name in VARIANTS:
    for mirrored in (False, True):
        errs = [np.mean(results[(name, mirrored, f)]) for f in FRACTIONS]
        drop = errs[-2] - errs[-1]
        trend = "still dropping" if drop > 0.15 else ("flat" if drop > -0.15 else "got worse")
        label = name + ("  + mirrored" if mirrored else "")
        print(f"{label:28s}" + "".join(f"{e:11.1f}" for e in errs) + f"    {drop:+5.2f} px  ({trend})")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"MLP tip": "tab:gray", "CNN hand": "tab:blue", "transformer": "tab:red"}
    xs = [n_games[f] for f in FRACTIONS]
    for name in VARIANTS:
        for mirrored in (False, True):
            ys = [np.mean(results[(name, mirrored, f)]) for f in FRACTIONS]
            ax.plot(xs, ys, "o--" if mirrored else "o-", color=colors.get(name, None),
                    label=name + (" + mirrored" if mirrored else ""), alpha=0.6 if mirrored else 1.0)
    ax.set_xlabel("training games")
    ax.set_ylabel("test error (px, lower is better)")
    ax.set_title("Learning curve: does more data help?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("learning_curve.png", dpi=120)
    print("\nSaved learning_curve.png  (open it:  open learning_curve.png)")
except ImportError:
    print("\n(pip install matplotlib to also get the plot)")
print("Lines still sloping down at the right edge = that model wants more data.")
