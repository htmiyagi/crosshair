"""Lesson 11: WHEN does the model make mistakes?

Lesson 10 showed tracking noise is only ~1 px, so ~15 px of the model's 16.5 px error is real movement it
didn't see coming. Here we sort every test prediction into a bucket by what the hand was ACTUALLY doing
(we can peek at the future here, because we're grading, not predicting):

  still         not moving now, not moving 2 frames later
  starting      not moving now, but moving 2 frames later      (nothing in the past hints at this)
  stopping      moving now, stopped 2 frames later
  turning       moving, but the next 2 frames go off at an angle from the current direction
  speeding up   same direction, getting faster
  slowing down  same direction, getting slower
  steady        same direction, about the same speed

For each bucket: how common it is, the model's average error there, and the column that matters most:
what SHARE of all the error lives in that bucket. A rare bucket with huge errors can still own most of it.

Same 6-fold cross-validation and MLP + early stopping as lesson 9, so every prediction is on unseen games.

Run from the crosshair folder:  python lessons/lesson11.py      (takes a minute or two)
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

MOVING = 4.0        # px per frame: slower than this counts as "not moving"
TURN_DEG = 30       # direction change bigger than this = turning
SPEED_CHANGE = 1.3  # 30% faster / slower = speeding up / slowing down


def make_examples(tip):
    inputs, answers = [], []
    for i in range(HISTORY - 1, len(tip) - AHEAD):
        path = tip[i - HISTORY + 1 : i + AHEAD + 1]
        if np.isnan(path).any() or np.abs(path[1:] - path[:-1]).max() > GLITCH:
            continue
        inputs.append(tip[i - HISTORY + 1 : i + 1].reshape(-1))
        answers.append(tip[i + AHEAD])
    return inputs, answers


def load(file_list):
    xs, ys = [], []
    for f in file_list:
        ins, ans = make_examples(np.load(f)["pts"][:, 8, :])
        xs += ins
        ys += ans
    x = torch.tensor(np.array(xs), dtype=torch.float32).reshape(-1, 2 * HISTORY)
    y = torch.tensor(np.array(ys), dtype=torch.float32).reshape(-1, 2)
    now = x[:, -2:]
    return (x - now.repeat(1, HISTORY)) * SCALE, (y - now) * SCALE


def to_px(v):
    """Normalized units -> rough screen pixels."""
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
    """Label each example by what the hand actually did. Returns an array of bucket names."""
    move_now = -x[:, -4:-2]                         # last move: now (0) minus 1 frame ago
    move_next = y / AHEAD                           # average move per frame over the next 2 frames
    speed_now = to_px(move_now.norm(dim=1)).numpy()
    speed_next = to_px(move_next.norm(dim=1)).numpy()
    cos = nn.functional.cosine_similarity(move_now, move_next, dim=1).clamp(-1, 1).numpy()
    angle = np.degrees(np.arccos(cos))

    labels = np.full(len(x), "steady", dtype=object)
    slow_now, slow_next = speed_now < MOVING, speed_next < MOVING
    moving_both = ~slow_now & ~slow_next
    labels[moving_both & (speed_next > speed_now * SPEED_CHANGE)] = "speeding up"
    labels[moving_both & (speed_next < speed_now / SPEED_CHANGE)] = "slowing down"
    labels[moving_both & (angle > TURN_DEG)] = "turning"          # turning wins over speed changes
    labels[~slow_now & slow_next] = "stopping"
    labels[slow_now & ~slow_next] = "starting"
    labels[slow_now & slow_next] = "still"
    return labels


files = sorted(glob.glob("data/*.npz"))
folds = [files[i : i + GAMES_PER_FOLD] for i in range(0, len(files), GAMES_PER_FOLD)]
print(f"{len(files)} recordings, {len(folds)} folds\n")

all_labels, model_errs, damp_errs = [], [], []
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    tx, ty = load(test_files)
    if len(tx) == 0:
        continue
    x, y = load(rest)
    m1 = -x[:, -4:-2]
    damp_k = ((m1 * y).sum() / (m1 * m1).sum()).item()      # the damped-speed baseline, as in lesson 9
    net = train(*load(rest[:-1]), *load(rest[-1:]))
    with torch.no_grad():
        model_errs.append(to_px((net(tx) - ty).norm(dim=1)).numpy())
    damp_errs.append(to_px((-tx[:, -4:-2] * damp_k - ty).norm(dim=1)).numpy())
    all_labels.append(bucket_of(tx, ty))
    print(f"fold {k + 1} done")

labels = np.concatenate(all_labels)
errs = np.concatenate(model_errs)
derrs = np.concatenate(damp_errs)
total = errs.sum()

print(f"\n{len(errs)} test predictions, model average {errs.mean():.1f} px\n")
print("what the hand did   share of frames   model avg px   damped avg px   SHARE OF ALL ERROR")
order = ["still", "starting", "steady", "speeding up", "slowing down", "turning", "stopping"]
for name in order:
    mask = labels == name
    if not mask.any():
        continue
    share_err = errs[mask].sum() / total * 100
    bar = "#" * int(round(share_err / 2))
    print(f"{name:16s}    {mask.mean() * 100:6.1f}%          {errs[mask].mean():6.1f}         "
          f"{derrs[mask].mean():6.1f}          {share_err:5.1f}%  {bar}")

# How concentrated is the error? If a few predictions are terrible, they dominate.
worst = np.sort(errs)[::-1]
for pct in (1, 5, 10):
    n = max(1, len(worst) * pct // 100)
    print(f"\nthe worst {pct:2d}% of predictions hold {worst[:n].sum() / total * 100:5.1f}% of all error"
          f"   (each off by {worst[n - 1]:.0f}+ px)", end="")
print("\n")
for name in order:
    mask = labels == name
    top = errs >= worst[max(1, len(worst) // 10) - 1]
    if mask.any():
        print(f"   of the worst 10%: {(top & mask).sum() / top.sum() * 100:5.1f}% are '{name}'")
