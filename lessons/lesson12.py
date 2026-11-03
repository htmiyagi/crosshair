"""Lesson 12: look at the worst mistakes with your own eyes. Real turns, or tracking glitches?

Same 6-fold cross-validation as lessons 9 and 11. This time we remember WHERE each test prediction came from
(which game, which frame), take the 20 worst, and draw each one:

  gray line     the fingertip a bit before and after (context)
  blue line     the 8 frames the model saw (ending at the big blue dot = "now")
  green star    where the finger really was 2 frames later (the answer)
  red X         where the model guessed
  orange line   the WRIST over the same frames (faint)

How to spot a glitch: the fingertip suddenly jumps and snaps BACK, and/or the fingertip jumps but the wrist
doesn't move at all. A real turn is smooth, and the wrist moves along with it.

Each panel's title also shows "tip/wrist": how far the fingertip moved over the 2 answer frames divided by how
far the wrist moved. Real hand movements are usually under ~3. A huge number = only the fingertip "moved".

Saves worst_mistakes.png in the crosshair folder.
Needs matplotlib:  pip install matplotlib
Run:  python lessons/lesson12.py
"""

import copy
import glob
import re

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
SHOW = 20            # how many worst mistakes to draw
CONTEXT = 6          # extra frames drawn before and after, in gray
TIP, WRIST = 8, 0    # MediaPipe point numbers


def make_examples(tip):
    """Same as before, but also returns the frame number 'now' for each example."""
    inputs, answers, where = [], [], []
    for i in range(HISTORY - 1, len(tip) - AHEAD):
        path = tip[i - HISTORY + 1 : i + AHEAD + 1]
        if np.isnan(path).any() or np.abs(path[1:] - path[:-1]).max() > GLITCH:
            continue
        inputs.append(tip[i - HISTORY + 1 : i + 1].reshape(-1))
        answers.append(tip[i + AHEAD])
        where.append(i)
    return inputs, answers, where


def load(file_list):
    xs, ys, info = [], [], []
    for f in file_list:
        ins, ans, where = make_examples(np.load(f)["pts"][:, TIP, :])
        xs += ins
        ys += ans
        info += [(f, i) for i in where]
    x = torch.tensor(np.array(xs), dtype=torch.float32).reshape(-1, 2 * HISTORY)
    y = torch.tensor(np.array(ys), dtype=torch.float32).reshape(-1, 2)
    now = x[:, -2:]
    return (x - now.repeat(1, HISTORY)) * SCALE, (y - now) * SCALE, info


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


files = sorted(glob.glob("data/*.npz"))
folds = [files[i : i + GAMES_PER_FOLD] for i in range(0, len(files), GAMES_PER_FOLD)]
print(f"{len(files)} recordings, {len(folds)} folds")

mistakes = []        # (error px, file, frame, guess in camera coords)
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    tx, ty, tinfo = load(test_files)
    if len(tx) == 0:
        continue
    fx, fy, _ = load(rest[:-1])
    vx, vy, _ = load(rest[-1:])
    net = train(fx, fy, vx, vy)
    with torch.no_grad():
        guess = net(tx)
    errs = (guess - ty).norm(dim=1) * SCREEN_PX / SCALE
    for e, g, (f, i) in zip(errs.tolist(), guess.numpy(), tinfo):
        mistakes.append((e, f, i, g / SCALE))        # guess back to camera units, relative to "now"
    print(f"fold {k + 1} done")

mistakes.sort(key=lambda m: -m[0])
worst = mistakes[:SHOW]

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib isn't installed. Run:  pip install matplotlib   then run this again.")

cols = 5
rows = (len(worst) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.8))
ratios = []
cache = {}
for ax, (err, f, i, g) in zip(axes.flat, worst):
    if f not in cache:
        cache[f] = np.load(f)["pts"]
    pts = cache[f]
    tip, wrist = pts[:, TIP, :], pts[:, WRIST, :]
    a, b = max(0, i - HISTORY + 1 - CONTEXT), min(len(tip), i + AHEAD + CONTEXT + 1)
    now = tip[i]

    ax.plot(tip[a:b, 0], tip[a:b, 1], "-", color="0.75", lw=1)                       # context
    ax.plot(tip[i - HISTORY + 1 : i + 1, 0], tip[i - HISTORY + 1 : i + 1, 1], "o-", color="tab:blue", ms=3)
    ax.plot(*now, "o", color="tab:blue", ms=8)                                       # now
    ax.plot(tip[i + 1 : i + AHEAD + 1, 0], tip[i + 1 : i + AHEAD + 1, 1], "--", color="tab:green")
    ax.plot(*tip[i + AHEAD], "*", color="tab:green", ms=14)                          # the answer
    ax.plot(*(now + g), "x", color="tab:red", ms=11, mew=2.5)                        # the guess
    # wrist, shifted so its "now" sits on the fingertip's "now" (we care about its shape, not where it is)
    w = wrist[a:b] - wrist[i] + now
    ax.plot(w[:, 0], w[:, 1], "-", color="tab:orange", alpha=0.45, lw=1.5)

    tip_move = np.linalg.norm(tip[i + AHEAD] - tip[i])
    wrist_move = np.linalg.norm(wrist[i + AHEAD] - wrist[i])
    ratio = tip_move / max(wrist_move, 1e-6)
    ratios.append(ratio)
    name = re.search(r"\d{8}_\d{4}", f)
    ax.set_title(f"{err:.0f} px   tip/wrist {ratio:.1f}\n{name.group(0) if name else f}  frame {i}", fontsize=8)
    ax.set_aspect("equal")
    ax.invert_yaxis()                     # camera y goes DOWN, like the screen
    ax.set_xticks([])
    ax.set_yticks([])
for ax in axes.flat[len(worst):]:
    ax.axis("off")

fig.suptitle("20 worst predictions   blue = what the model saw, green star = truth, red X = guess, "
             "orange = wrist", fontsize=11)
fig.tight_layout()
fig.savefig("worst_mistakes.png", dpi=110)
print("\nSaved worst_mistakes.png  (open it:  open worst_mistakes.png)")
print(f"tip/wrist ratios of the {SHOW} worst: " + ", ".join(f"{r:.1f}" for r in ratios))
print(f"{sum(r > 3 for r in ratios)} of {SHOW} have the fingertip moving 3x+ more than the wrist (glitch suspects)")
