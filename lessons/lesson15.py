"""Lesson 15: get the hand's turning win WITHOUT its steady-movement loss.

Lesson 14: knuckle + wrist made turns ~10% better but steady movement ~15% worse. Overall: a wash.
Two ways to let the model use the hand only when it's useful:

  A  TIP + SMALL CORRECTION
     guess = tip_model(fingertip) + correction(whole hand)
     The correction is penalized for being big (weight decay), so it stays near zero unless the hand clearly
     says something. Default behavior = the tip-only model.

  B  MIXTURE OF EXPERTS (a "gate")
     Two experts: tip_model(fingertip) and hand_model(whole hand).
     A tiny third model, the GATE, looks at the hand and outputs a number g between 0 and 1:
         guess = (1 - g) * tip_guess + g * hand_guess
     g = 0 means "trust the fingertip expert", g = 1 means "trust the hand expert".
     Nobody tells the gate when to open. It learns that by itself, because opening at the right moments lowers
     the loss. Big LLMs use this same idea (with many experts) to decide which part of the network handles
     each word.

At the end we print the gate's AVERAGE g in each bucket. If it learned what you guessed, it should be more open
during turns than during steady movement.

"hand" here = fingertip + knuckle + wrist (the 3 points from lesson 14; less noisy than all 21).
Same setup as lesson 14: 6-fold cross-validation, glitch filter, time gaps skipped, clean examples only.

Run:  python lessons/lesson15.py      (a few minutes)
"""

import copy
import glob

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
CORRECTION_DECAY = 0.1     # option A: how hard the correction is pushed toward zero

TIP, KNUCKLE, WRIST = 8, 5, 0
HAND = [TIP, KNUCKLE, WRIST]
SPIKE_MIN, SPIKE_BACK, TOO_FAR = 0.03, 0.5, 1.0
MOVING, TURN_DEG, SPEED_CHANGE = 4.0, 30, 1.3
BUCKETS = ["still", "starting", "steady", "speeding up", "slowing down", "turning", "stopping"]


# ---------- data (same as lesson 14) ----------
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
    hands = torch.tensor(np.array(H), dtype=torch.float32)          # (n, 8, 3, 2): 8 frames, 3 points
    y = torch.tensor(np.array(Y), dtype=torch.float32)
    now = hands[:, -1, 0, :]                                        # fingertip now (point 0 of HAND)
    hands = (hands - now[:, None, None, :]) * SCALE
    tip_x = hands[:, :, 0, :].reshape(len(hands), -1)               # 16 numbers
    hand_x = hands.reshape(len(hands), -1)                          # 48 numbers
    return tip_x, hand_x, (y - now) * SCALE


def to_px(v):
    return v * SCREEN_PX / SCALE


# ---------- the models ----------
def mlp(n_in, n_out=2, hidden=64):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_out))


class TipOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.tip = mlp(16)

    def forward(self, tip_x, hand_x):
        return self.tip(tip_x)


class HandOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.hand = mlp(48)

    def forward(self, tip_x, hand_x):
        return self.hand(hand_x)


class TipPlusCorrection(nn.Module):
    """Option A: the tip model's guess + a small correction from the hand."""

    def __init__(self):
        super().__init__()
        self.tip = mlp(16)
        self.correction = mlp(48, hidden=32)

    def forward(self, tip_x, hand_x):
        return self.tip(tip_x) + self.correction(hand_x)


class Mixture(nn.Module):
    """Option B: two experts + a gate that decides how much to trust each."""

    def __init__(self):
        super().__init__()
        self.tip = mlp(16)
        self.hand = mlp(48)
        self.gate = nn.Sequential(nn.Linear(48, 16), nn.ReLU(), nn.Linear(16, 1))

    def gate_value(self, hand_x):
        return torch.sigmoid(self.gate(hand_x))          # squashes any number into 0..1

    def forward(self, tip_x, hand_x):
        g = self.gate_value(hand_x)                      # shape (n, 1): one dial per example
        return (1 - g) * self.tip(tip_x) + g * self.hand(hand_x)


VARIANTS = {
    "tip only": TipOnly,
    "hand only": HandOnly,
    "A: tip + correction": TipPlusCorrection,
    "B: mixture (gate)": Mixture,
}


def train(build, data, val):
    torch.manual_seed(0)
    model = build()
    if isinstance(model, TipPlusCorrection):
        # weight decay ONLY on the correction, so it's the part that gets pushed toward zero
        groups = [{"params": model.tip.parameters(), "weight_decay": 0.0},
                  {"params": model.correction.parameters(), "weight_decay": CORRECTION_DECAY}]
    else:
        groups = [{"params": model.parameters(), "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=0.01)
    tx, hx, y = data
    vtx, vhx, vy = val
    best, best_err = copy.deepcopy(model.state_dict()), float("inf")
    for step in range(STEPS):
        loss = nn.functional.mse_loss(model(tx, hx), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % CHECK_EVERY == 0:
            with torch.no_grad():
                err = (model(vtx, vhx) - vy).norm(dim=1).mean().item()
            if err < best_err:
                best_err, best = err, copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return model


def bucket_of(tip_x, y):
    move_now = tip_x[:, -2:] - tip_x[:, -4:-2]
    move_next = y / AHEAD
    speed_now = to_px(move_now.norm(dim=1)).numpy()
    speed_next = to_px(move_next.norm(dim=1)).numpy()
    cos = nn.functional.cosine_similarity(move_now, move_next, dim=1).clamp(-1, 1).numpy()
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
print(f"{len(files)} recordings, {len(folds)} folds, {len(VARIANTS)} models\n")

errs = {"damped speed": []} | {name: [] for name in VARIANTS}
labels, gates = [], []
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    test = load(test_files)
    if len(test[2]) == 0:
        continue
    fit, val, everything = load(rest[:-1]), load(rest[-1:]), load(rest)
    m1 = everything[0][:, -2:] - everything[0][:, -4:-2]
    damp_k = ((m1 * everything[2]).sum() / (m1 * m1).sum()).item()
    errs["damped speed"].append(to_px(((test[0][:, -2:] - test[0][:, -4:-2]) * damp_k - test[2]).norm(dim=1)).numpy())
    for name, build in VARIANTS.items():
        model = train(build, fit, val)
        with torch.no_grad():
            errs[name].append(to_px((model(test[0], test[1]) - test[2]).norm(dim=1)).numpy())
            if isinstance(model, Mixture):
                gates.append(model.gate_value(test[1]).squeeze(1).numpy())
    labels.append(bucket_of(test[0], test[2]))
    print(f"fold {k + 1} done")

labels = np.concatenate(labels)
damp_folds = [e.mean() for e in errs["damped speed"]]
cols = ["steady", "speeding up", "slowing down", "turning"]
print("\n(clean test examples, px; lower is better)")
print(f"{'model':21s}  overall   beats damped   " + "  ".join(f"{c:>12s}" for c in cols))
for name, per_fold in errs.items():
    e = np.concatenate(per_fold)
    wins = "      -     " if name == "damped speed" else \
        f"   {sum(f.mean() < d for f, d in zip(per_fold, damp_folds))} of {len(per_fold)}    "
    print(f"{name:21s}  {e.mean():6.1f}  {wins}  " + "  ".join(f"{e[labels == c].mean():12.1f}" for c in cols))

g = np.concatenate(gates)
print("\nWhat did the gate learn? Average g (0 = trusts fingertip, 1 = trusts hand):")
for name in BUCKETS:
    m = labels == name
    if m.any():
        print(f"   {name:14s} {g[m].mean():.2f}  {'#' * int(round(g[m].mean() * 40))}")
print("\nIf the gate is more open for 'turning' than 'steady', it discovered your idea by itself.")
