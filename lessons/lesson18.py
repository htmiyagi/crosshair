"""Lesson 18: a tiny transformer, with attention built by hand.

A transformer reads the 8 frames like 8 people in a room. Each frame can look at EVERY other frame and decide how
much each one matters to it. That "how much" is called ATTENTION. It's the core of every LLM (there, the "frames"
are words).

How one attention step works, for each frame:
  1. Turn the frame's numbers (tip, knuckle, wrist x/y = 6 numbers) into a longer list, its "embedding" (32 numbers).
  2. Add a POSITION tag (also learned), so the model knows which frame came first. Attention alone has no sense of order.
  3. Every frame makes three lists from its embedding:
        query  "what am I looking for?"
        key    "what do I have?"
        value  "what will I share if you pick me?"
  4. Frame A's query is compared with every frame's key (multiply and add = a "dot product"). A big result = a good match.
  5. softmax turns those 8 match scores into 8 weights that add up to 1: the ATTENTION weights.
  6. Frame A's new embedding = the weighted average of everyone's values. It now carries info from the frames it cared about.
After attention, a small MLP refines each frame on its own. That pair (attention + MLP) is one "transformer block".
Real transformers stack many blocks and run several attentions side by side ("heads"). Ours: 2 blocks, 1 head.

The guess comes from the LAST frame ("now"), after it has gathered what it needs from the others.

At the end we print what "now" pays attention to, split by what the hand was doing. If it learned something real,
turns might look further back (to see where the curve started) than steady moves do.

Compared: damped speed, MLP tip, CNN hand (your best), transformer hand.
Same setup as lesson 17. Folds: 3 games each (default) or  --quick  for 6 games each (half the time).

Run:  python lessons/lesson18.py            or   python lessons/lesson18.py --quick
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
GAMES_PER_FOLD = 6 if "--quick" in sys.argv else 3
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


# ---------- cross-validation ----------
files = sorted(glob.glob("data/*.npz"))
folds = [files[i : i + GAMES_PER_FOLD] for i in range(0, len(files), GAMES_PER_FOLD)]
print(f"{len(files)} recordings, {len(folds)} folds, running on: {DEVICE}\n")
print("weights in each model:")
for name, build in VARIANTS.items():
    print(f"   {name:9s} {count_weights(build()):6d}")
print()

errs = {"damped speed": []} | {name: [] for name in VARIANTS}
labels, attn = [], []
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
            if isinstance(model, TinyTransformer):
                model(th)                                   # run once more so the attention weights are saved
                attn.append(model.now_attention().cpu().numpy())
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

a = np.concatenate(attn)                              # (examples, 8): "now"'s attention on each frame
print("\nWhat does 'now' pay attention to? (columns = frames, oldest -> now; each row adds up to 100%)")
print("                 " + "  ".join(f"{('t-' + str(HISTORY - 1 - i)) if i < HISTORY - 1 else 'now':>4s}" for i in range(HISTORY)))
for name in ["still", "starting", "steady", "speeding up", "slowing down", "turning", "stopping"]:
    m = labels == name
    if m.any():
        row = a[m].mean(0) * 100
        print(f"   {name:13s} " + "  ".join(f"{v:4.0f}" for v in row) + "   " + "".join(" .:-=+*#%@"[min(9, int(v / 6))] for v in row))
print("\nIf turning rows put more weight on older frames than steady rows, attention found something real.")
