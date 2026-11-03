"""Lesson 14: does seeing MORE OF THE HAND help predict turns?

Lesson 13: on clean data, the fingertip-only MLP barely beats damped speed (13.4 vs 14.1 px), and most of the
error left is turning and speeding up. Idea: your hand often rotates or leans BEFORE the fingertip changes
direction, so the knuckle and wrist might give an early warning the fingertip can't.

Same everything as lesson 13 (6-fold cross-validation, glitch filter, MLP + early stopping, CLEAN examples only),
but the model gets different inputs:

  tip only               the 8 fingertip positions (what we had)              16 numbers
  tip + knuckle + wrist  same 8 frames, 3 points each                         48 numbers
  ... + timing           plus the real time between frames (camera gaps)      56 numbers
  whole hand + timing    all 21 MediaPipe points, 8 frames, plus timing       344 numbers

Everything is relative to the fingertip "now" (so position on screen doesn't matter), x SCALE.
Timing is (seconds between frames) x 30, so a normal frame is ~1.0.

Also new: the game only saves frames where it saw a hand, so a lost hand shows up as a TIME GAP, not a NaN.
Examples spanning a gap longer than 0.1 s are now skipped (for every predictor, so it's still fair).

Run:  python lessons/lesson14.py      (a few minutes: 4 models x 6 folds)
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
MAX_GAP = 0.1        # seconds: a longer gap between saved frames = the hand was lost

TIP, KNUCKLE, WRIST = 8, 5, 0
SPIKE_MIN, SPIKE_BACK, TOO_FAR = 0.03, 0.5, 1.0          # glitch rules, same as lesson 13
MOVING, TURN_DEG, SPEED_CHANGE = 4.0, 30, 1.3             # buckets, same as lesson 11

VARIANTS = {
    "tip only":              ([TIP], False),
    "tip + knuckle + wrist": ([TIP, KNUCKLE, WRIST], False),
    "  ... + timing":        ([TIP, KNUCKLE, WRIST], True),
    "whole hand + timing":   (list(range(21)), True),
}


def flag_frames(pts):
    """Lesson 13's two glitch rules. True = glitchy frame."""
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
    """Clean examples only. Returns hands (n, HISTORY, 21, 2), gaps (n, HISTORY), answers (n, 2)."""
    tip = pts[:, TIP, :]
    bad = flag_frames(pts)
    dt = np.diff(t, prepend=t[0] - 1 / 30)
    hands, gaps, answers = [], [], []
    for i in range(HISTORY - 1, len(tip) - AHEAD):
        a, b = i - HISTORY + 1, i + AHEAD + 1
        path = tip[a:b]
        if np.isnan(path).any() or np.abs(path[1:] - path[:-1]).max() > GLITCH:
            continue
        if bad[a:b].any() or dt[a + 1 : b].max() > MAX_GAP:
            continue
        hands.append(pts[a : i + 1])
        gaps.append(dt[a : i + 1])
        answers.append(tip[i + AHEAD])
    return hands, gaps, answers


def load(file_list):
    H, T, Y = [], [], []
    for f in file_list:
        d = np.load(f)
        pts = d["pts"].astype(np.float32)
        t = d["t"] if "t" in d.files else np.arange(len(pts)) / 30
        h, g, y = make_examples(pts, t)
        H += h
        T += g
        Y += y
    hands = torch.tensor(np.array(H), dtype=torch.float32)            # (n, 8, 21, 2)
    gaps = torch.tensor(np.array(T), dtype=torch.float32) * 30        # (n, 8), ~1.0 per normal frame
    y = torch.tensor(np.array(Y), dtype=torch.float32)
    now = hands[:, -1, TIP, :]                                        # fingertip now
    hands = (hands - now[:, None, None, :]) * SCALE                   # relative to fingertip now
    return hands, gaps, (y - now) * SCALE


def inputs(hands, gaps, points, timing):
    x = hands[:, :, points, :].reshape(len(hands), -1)
    return torch.cat([x, gaps], dim=1) if timing else x


def to_px(v):
    return v * SCREEN_PX / SCALE


def train(x, y, vx, vy):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(),
                          nn.Linear(64, 64), nn.ReLU(),
                          nn.Linear(64, 2))
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
    move_now = hands[:, -1, TIP] - hands[:, -2, TIP]
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


files = sorted(glob.glob("data/*.npz"))
folds = [files[i : i + GAMES_PER_FOLD] for i in range(0, len(files), GAMES_PER_FOLD)]
print(f"{len(files)} recordings, {len(folds)} folds, {len(VARIANTS)} models\n")

errs = {"damped speed": []} | {name: [] for name in VARIANTS}     # per fold: array of test errors (px)
labels = []
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    th, tg, ty = load(test_files)
    if len(ty) == 0:
        continue
    fh, fg, fy = load(rest[:-1])
    vh, vg, vy = load(rest[-1:])
    ah, ag, ay = load(rest)
    m1 = ah[:, -1, TIP] - ah[:, -2, TIP]
    damp_k = ((m1 * ay).sum() / (m1 * m1).sum()).item()
    errs["damped speed"].append(to_px(((th[:, -1, TIP] - th[:, -2, TIP]) * damp_k - ty).norm(dim=1)).numpy())
    for name, (points, timing) in VARIANTS.items():
        net = train(inputs(fh, fg, points, timing), fy, inputs(vh, vg, points, timing), vy)
        with torch.no_grad():
            errs[name].append(to_px((net(inputs(th, tg, points, timing)) - ty).norm(dim=1)).numpy())
    labels.append(bucket_of(th, ty))
    print(f"fold {k + 1} done   ({len(ty)} clean test examples)")

labels = np.concatenate(labels)
damp_folds = [e.mean() for e in errs["damped speed"]]
cols = ["steady", "speeding up", "slowing down", "turning"]
print("\n(all clean test examples, px; lower is better)")
print(f"{'inputs':23s}  overall   beats damped   " + "  ".join(f"{c:>12s}" for c in cols))
for name, per_fold in errs.items():
    e = np.concatenate(per_fold)
    wins = "      -     " if name == "damped speed" else \
        f"   {sum(f.mean() < d for f, d in zip(per_fold, damp_folds))} of {len(per_fold)}    "
    by = "  ".join(f"{e[labels == c].mean():12.1f}" for c in cols)
    print(f"{name:23s}  {e.mean():6.1f}  {wins}  {by}")
print("\nThe question: does any row beat 'tip only' clearly, especially in the turning column?")
