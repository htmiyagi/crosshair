"""Lesson 9: stronger baselines. Does the model beat a GOOD non-ML guess, not just a naive one?

"Constant speed" (last move x 2) is a weak opponent: one frame of movement is jittery. If the model only
learned "average the recent speed", it would beat that baseline without understanding hands at all.
So here are four non-ML guessers, from naive to smart, against the two best models from lessons 7 and 8,
all on the same 6-fold cross-validation (train on 5 groups of games, test on the unseen 6th, rotate):

  constant speed        last move x 2                                  (the old baseline)
  smoothed speed        average of the last 3 moves x 2                (less jitter)
  damped speed          last move x k, with the best k fit on the training games (no learning beyond 1 number)
  constant acceleration last move x 2 + change-in-move x 3             (physics: speeding up / slowing down)

Positions here are relative to "now" (now = 0), so with p1, p2, p3 = 1, 2, 3 frames ago:
  last move = 0 - p1,  move before = p1 - p2,  and so on.

Run from the crosshair folder:  python lessons/lesson9.py      (takes a few minutes)
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


def px(guess, truth):
    return (guess - truth).norm(dim=1).mean().item() * SCREEN_PX / SCALE


def moves(x):
    """The last three frame-to-frame moves (newest first), from the relative positions."""
    p1, p2, p3 = x[:, -4:-2], x[:, -6:-4], x[:, -8:-6]       # 1, 2, 3 frames ago (now is 0)
    return 0 - p1, p1 - p2, p2 - p3


def baselines(x, damp_k):
    m1, m2, m3 = moves(x)
    return {
        "constant speed": m1 * AHEAD,
        "smoothed speed": (m1 + m2 + m3) / 3 * AHEAD,
        "damped speed": m1 * damp_k,
        "constant acceleration": m1 * AHEAD + (m1 - m2) * 3,   # 2 frames ahead: v*2 + a*(1+2)
    }


def fit_damp(x, y):
    """The single best k for 'last move x k' on the training games (a least-squares fit)."""
    m1 = moves(x)[0]
    return ((m1 * y).sum() / (m1 * m1).sum()).item()


def mlp():
    return nn.Sequential(nn.Linear(2 * HISTORY, 64), nn.ReLU(),
                         nn.Linear(64, 64), nn.ReLU(),
                         nn.Linear(64, 2))


def train(model, x, y, vx=None, vy=None):
    torch.manual_seed(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best, best_err = copy.deepcopy(model.state_dict()), float("inf")
    for step in range(STEPS):
        loss = nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if vx is not None and step % CHECK_EVERY == 0:      # early stopping on the validation game
            with torch.no_grad():
                err = px(model(vx), vy)
            if err < best_err:
                best_err, best = err, copy.deepcopy(model.state_dict())
    if vx is not None:
        model.load_state_dict(best)
    return model


files = sorted(glob.glob("data/*.npz"))
folds = [files[i : i + GAMES_PER_FOLD] for i in range(0, len(files), GAMES_PER_FOLD)]
print(f"{len(files)} recordings, {len(folds)} folds\n")

scores = {}                                 # predictor name -> test error (px) in each fold
ks = []
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    x, y = load(rest)                       # all training games (for the baselines and the linear model)
    tx, ty = load(test_files)
    if len(tx) == 0:
        continue
    damp_k = fit_damp(x, y)
    ks.append(damp_k)
    for name, guess in baselines(tx, damp_k).items():
        scores.setdefault(name, []).append(px(guess, ty))
    linear = train(nn.Linear(2 * HISTORY, 2), x, y)
    fx, fy = load(rest[:-1])                # MLP: one training game held aside for early stopping
    vx, vy = load(rest[-1:])
    net = train(mlp(), fx, fy, vx, vy)
    with torch.no_grad():
        scores.setdefault("linear model", []).append(px(linear(tx), ty))
        scores.setdefault("MLP + early stopping", []).append(px(net(tx), ty))
    print(f"fold {k + 1} done   (best damping k = {damp_k:.2f}; plain constant speed would be {AHEAD})")

base_names = ["constant speed", "smoothed speed", "damped speed", "constant acceleration"]
best_base = min(base_names, key=lambda n: np.mean(scores[n]))
print("\npredictor                 avg test px   folds the MLP beats it   MLP better by")
for name, errs in scores.items():
    mlp_errs = scores["MLP + early stopping"]
    if name == "MLP + early stopping":
        line = "         (itself)"
    else:
        beats = sum(m < e for m, e in zip(mlp_errs, errs))
        gain = (np.mean(errs) - np.mean(mlp_errs)) / np.mean(errs) * 100
        line = f"         {beats} of {len(errs)}              {gain:+6.1f}%"
    tag = "   <- best non-ML guess" if name == best_base else ""
    print(f"{name:24s}  {np.mean(errs):8.1f}   {line}{tag}")
print(f"\nThe question that matters: does the MLP beat '{best_base}', the strongest guesser without ML?")
