"""Lesson 8: stop the MLP from memorizing (overfitting), and compare fairly with cross-validation.

The MLP was much better on its training games than on new ones: it memorized quirks of specific games.
Three standard fixes, tried one at a time, all scored with the same 6-fold cross-validation as lesson 7:

  weight decay     a small penalty on big knob values, so the model prefers simple explanations
  early stopping   hold one training game aside as a "validation" game; keep the version of the model
                   that did best on it, instead of the one that trained longest (and memorized most)
  smaller model    fewer knobs = less room to memorize

Plus one combo: "linear + MLP". The linear model already works; the MLP only adds a small correction
on top, so it starts from something good instead of from scratch.

Run from the crosshair folder:  python lessons/lesson8.py      (takes a few minutes)
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
CHECK_EVERY = 100    # early stopping: how often to check the validation game


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


class LinearPlusMLP(nn.Module):
    """The linear model's guess, plus a small learned correction from an MLP."""

    def __init__(self, hidden=32):
        super().__init__()
        self.linear = nn.Linear(2 * HISTORY, 2)
        self.mlp = nn.Sequential(nn.Linear(2 * HISTORY, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.linear(x) + self.mlp(x)


def mlp(hidden):
    return nn.Sequential(nn.Linear(2 * HISTORY, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 2))


# name: (how to build the model, weight decay, early stopping?)
VARIANTS = {
    "linear":                     (lambda: nn.Linear(2 * HISTORY, 2), 0.0, False),
    "MLP 64 (as before)":         (lambda: mlp(64), 0.0, False),
    "MLP 64 + weight decay":      (lambda: mlp(64), 0.01, False),
    "MLP 64 + early stopping":    (lambda: mlp(64), 0.0, True),
    "MLP 16 (smaller)":           (lambda: mlp(16), 0.0, True),
    "linear + MLP (both fixes)":  (lambda: LinearPlusMLP(32), 0.01, True),
}


def train(build, wd, early, x, y, vx, vy):
    torch.manual_seed(0)
    model = build()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=wd)
    best, best_err = copy.deepcopy(model.state_dict()), float("inf")
    for step in range(STEPS):
        loss = nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if early and step % CHECK_EVERY == 0:
            with torch.no_grad():
                err = px(model(vx), vy)
            if err < best_err:                         # best so far on the validation game: remember it
                best_err, best = err, copy.deepcopy(model.state_dict())
    if early:
        model.load_state_dict(best)
    return model


files = sorted(glob.glob("data/*.npz"))
folds = [files[i : i + GAMES_PER_FOLD] for i in range(0, len(files), GAMES_PER_FOLD)]
print(f"{len(files)} recordings, {len(folds)} folds, {len(VARIANTS)} models to compare\n")

results = {name: {"test": [], "train": [], "gain": []} for name in VARIANTS}
for k, test_files in enumerate(folds):
    rest = [f for f in files if f not in test_files]
    train_files, val_files = rest[:-1], rest[-1:]      # one training game set aside for early stopping
    x, y = load(train_files)
    vx, vy = load(val_files)
    tx, ty = load(test_files)
    if len(tx) == 0:
        continue
    const = px(-tx[:, -4:-2] * AHEAD, ty)
    for name, (build, wd, early) in VARIANTS.items():
        model = train(build, wd, early, x, y, vx, vy)
        with torch.no_grad():
            test_err, train_err = px(model(tx), ty), px(model(x), y)
        results[name]["test"].append(test_err)
        results[name]["train"].append(train_err)
        results[name]["gain"].append((const - test_err) / const * 100)
    print(f"fold {k + 1} done")

print("\nmodel                         wins vs const   avg improvement   avg test px   avg train px   gap")
for name, r in results.items():
    wins = sum(g > 0 for g in r["gain"])
    gap = np.mean(r["test"]) - np.mean(r["train"])
    print(f"{name:28s}  {wins} of {len(r['gain'])}          {np.mean(r['gain']):+6.1f}%        "
          f"{np.mean(r['test']):6.1f}        {np.mean(r['train']):6.1f}     {gap:5.1f}")
print("\n'gap' = how much worse on new games than on training games. Big gap = memorizing.")
