"""Lesson 13: a smarter glitch detector. How much of the error was really tracker glitches?

Lesson 12's pictures showed many of the worst "turns" were the tracker snapping to the wrong spot.
The old filter only caught HUGE jumps (0.15 of the picture in one frame). Two new rules:

  1. SPIKE       a frame that is far from BOTH its neighbors, while the neighbors are close to each other.
                 That's the out-and-back zigzag: the dot jumps away for one frame and snaps back.
                 Real fingers can't do that in 1/30 of a second.

  2. TOO FAR     the fingertip moves, compared to its KNUCKLE (point 5), by more than a finger-length in one
                 frame. Moving only your finger is fine: the tip swings around the knuckle and can only get
                 about one finger-length away. A jump farther than that would need a longer finger.
                 The finger-length is measured in each frame, so it works close to or far from the camera.

Honesty rules (so we don't "improve" the score by deleting hard examples):
  - the rules only look at the hand positions, never at the model's error
  - the same rules decide for training AND testing
  - we report how many examples were removed, and how much of the OLD error they held

How it works: every example from the old filter is kept for grading, but marked "glitchy" if any of its frames
(the 8 it sees + the answer frames) is flagged. The model trains only on clean examples. Then we split the error.

Run:  python lessons/lesson13.py      (a minute or two)
"""

import copy
import glob

import numpy as np
import torch
from torch import nn

HISTORY = 8
AHEAD = 2
GLITCH = 0.15        # the old filter (still used: these are thrown out completely, as before)
GAMES_PER_FOLD = 3
SCREEN_PX = 1512
SCALE = 10
STEPS = 2000
CHECK_EVERY = 100

TIP, KNUCKLE = 8, 5
SPIKE_MIN = 0.03     # rule 1: the jump out must be at least this big (~45 px) to count as a spike
SPIKE_BACK = 0.5     # ...and the neighbors must be closer to each other than half of that jump
TOO_FAR = 1.0        # rule 2: tip-vs-knuckle change bigger than this many finger-lengths in one frame

MOVING, TURN_DEG, SPEED_CHANGE = 4.0, 30, 1.3     # buckets, same as lesson 11


def flag_frames(pts):
    """Return (spike flags, too-far flags), one True/False per frame."""
    tip, knuckle = pts[:, TIP, :].astype(np.float64), pts[:, KNUCKLE, :].astype(np.float64)
    n = len(tip)
    spike = np.zeros(n, bool)
    too_far = np.zeros(n, bool)

    # rule 1: spike
    out = np.linalg.norm(tip[1:-1] - tip[:-2], axis=1)            # jump from the frame before
    back = np.linalg.norm(tip[1:-1] - tip[2:], axis=1)            # jump to the frame after
    across = np.linalg.norm(tip[2:] - tip[:-2], axis=1)           # before -> after, skipping it
    smaller = np.minimum(out, back)
    spike[1:-1] = (smaller > SPIKE_MIN) & (across < SPIKE_BACK * smaller)

    # rule 2: tip moved too far relative to its knuckle
    rel = tip - knuckle                                           # the finger, as an arrow from knuckle to tip
    finger_len = np.linalg.norm(rel, axis=1)
    change = np.linalg.norm(rel[1:] - rel[:-1], axis=1)
    length = np.maximum(finger_len[1:], finger_len[:-1])
    too_far[1:] = change > TOO_FAR * length

    spike &= ~np.isnan(tip).any(axis=1)
    too_far &= ~np.isnan(tip).any(axis=1)
    return spike, too_far


def make_examples(pts):
    tip = pts[:, TIP, :]
    spike, too_far = flag_frames(pts)
    bad = spike | too_far
    inputs, answers, glitchy = [], [], []
    for i in range(HISTORY - 1, len(tip) - AHEAD):
        path = tip[i - HISTORY + 1 : i + AHEAD + 1]
        if np.isnan(path).any() or np.abs(path[1:] - path[:-1]).max() > GLITCH:
            continue
        inputs.append(tip[i - HISTORY + 1 : i + 1].reshape(-1))
        answers.append(tip[i + AHEAD])
        glitchy.append(bad[i - HISTORY + 1 : i + AHEAD + 1].any())
    return inputs, answers, glitchy, spike.sum(), too_far.sum(), (spike & too_far).sum()


counts = {"frames": 0, "spike": 0, "too_far": 0, "both": 0}


def load(file_list, tally=False):
    xs, ys, gs = [], [], []
    for f in file_list:
        pts = np.load(f)["pts"]
        ins, ans, gl, s, t, b = make_examples(pts)
        xs += ins
        ys += ans
        gs += gl
        if tally:
            counts["frames"] += len(pts)
            counts["spike"] += s
            counts["too_far"] += t
            counts["both"] += b
    x = torch.tensor(np.array(xs), dtype=torch.float32).reshape(-1, 2 * HISTORY)
    y = torch.tensor(np.array(ys), dtype=torch.float32).reshape(-1, 2)
    now = x[:, -2:]
    return (x - now.repeat(1, HISTORY)) * SCALE, (y - now) * SCALE, np.array(gs, bool)


def to_px(v):
    return v * SCREEN_PX / SCALE


def mlp():
    return nn.Sequential(nn.Linear(2 * HISTORY, 64), nn.ReLU(),
                         nn.Linear(64, 64), nn.ReLU(),
                         nn.Linear(64, 2))


def train(x, y, vx, vy):
    torch.manual_seed(0)
    model = mlp()
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


def bucket_of(x, y):
    move_now = -x[:, -4:-2]
    move_next = y / AHEAD
    speed_now = to_px(move_now.norm(dim=1)).numpy()
    speed_next = to_px(move_next.norm(dim=1)).numpy()
    cos = nn.functional.cosine_similarity(move_now, move_next, dim=1).clamp(-1, 1).numpy()
    angle = np.degrees(np.arccos(cos))
    labels = np.full(len(x), "steady", dtype=object)
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
print(f"{len(files)} recordings, {len(folds)} folds\n")

L, E, D, G = [], [], [], []      # bucket labels, model errors, damped errors, glitchy flags (all test examples)
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    tx, ty, tg = load(test_files, tally=True)
    if len(tx) == 0:
        continue
    fx, fy, fg = load(rest[:-1])
    vx, vy, vg = load(rest[-1:])
    net = train(fx[~fg], fy[~fg], vx[~vg], vy[~vg])            # train and early-stop on CLEAN examples only
    x, y, g = load(rest)
    m1 = -x[~g][:, -4:-2]
    damp_k = ((m1 * y[~g]).sum() / (m1 * m1).sum()).item()
    with torch.no_grad():
        E.append(to_px((net(tx) - ty).norm(dim=1)).numpy())
    D.append(to_px((-tx[:, -4:-2] * damp_k - ty).norm(dim=1)).numpy())
    L.append(bucket_of(tx, ty))
    G.append(tg)
    print(f"fold {k + 1} done")

labels, errs, derrs, glitchy = np.concatenate(L), np.concatenate(E), np.concatenate(D), np.concatenate(G)
clean = ~glitchy

print(f"\nFLAGGED FRAMES (out of {counts['frames']}):")
print(f"   spike (out-and-back):        {counts['spike']:5d}  ({counts['spike'] / counts['frames'] * 100:.1f}%)")
print(f"   too far from the knuckle:    {counts['too_far']:5d}  ({counts['too_far'] / counts['frames'] * 100:.1f}%)")
print(f"   caught by both rules:        {counts['both']:5d}")
print(f"\nTEST EXAMPLES: {len(errs)} total, {glitchy.sum()} touch a flagged frame ({glitchy.mean() * 100:.1f}%)")
print(f"   ...and those {glitchy.mean() * 100:.1f}% hold {errs[glitchy].sum() / errs.sum() * 100:.1f}% of all the model's error")
print(f"\n                       model avg px    damped avg px")
print(f"   all (old filter)      {errs.mean():6.1f}          {derrs.mean():6.1f}")
print(f"   glitchy only          {errs[glitchy].mean():6.1f}          {derrs[glitchy].mean():6.1f}")
print(f"   CLEAN only            {errs[clean].mean():6.1f}          {derrs[clean].mean():6.1f}   <- the real score")

ce, cd, cl = errs[clean], derrs[clean], labels[clean]
print("\nCLEAN examples only:")
print("what the hand did   share of frames   model avg px   damped avg px   SHARE OF ALL ERROR")
for name in ["still", "starting", "steady", "speeding up", "slowing down", "turning", "stopping"]:
    m = cl == name
    if not m.any():
        continue
    share = ce[m].sum() / ce.sum() * 100
    print(f"{name:16s}    {m.mean() * 100:6.1f}%          {ce[m].mean():6.1f}         {cd[m].mean():6.1f}"
          f"          {share:5.1f}%  {'#' * int(round(share / 2))}")
worst = np.sort(ce)[::-1]
n = max(1, len(worst) // 10)
print(f"\nclean: the worst 10% hold {worst[:n].sum() / ce.sum() * 100:.1f}% of the error (was 40.4% before)")
