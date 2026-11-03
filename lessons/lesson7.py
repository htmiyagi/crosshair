"""Lesson 7: cross-validation. Is the linear model's win real, or did we get lucky with which games were tested?

Lesson 6 tested on ONE choice of held-out games (the last 3). A different choice could tell a different story.
Cross-validation rotates it: split the games into groups ("folds"), and for each fold, train on all the OTHER
games and test on that fold. Every game gets a turn as unseen test data. Then we look at:
  - how often the model beats constant speed
  - the average improvement across all folds

Run from the crosshair folder:  python lessons/lesson7.py
"""

import glob
import re

import numpy as np
import torch
from torch import nn

HISTORY = 8          # how many past frames the model sees
AHEAD = 2            # predict this many frames ahead (2 frames ~ 60 ms)
GLITCH = 0.15        # a jump bigger than this in one frame = tracking glitch
GAMES_PER_FOLD = 3   # how many games are held out each round
SCREEN_PX = 1512
SCALE = 10


def make_examples(tip):
    """One recording's fingertip track -> examples (same as lesson 6)."""
    inputs, answers = [], []
    for i in range(HISTORY - 1, len(tip) - AHEAD):
        path = tip[i - HISTORY + 1 : i + AHEAD + 1]
        if np.isnan(path).any() or np.abs(path[1:] - path[:-1]).max() > GLITCH:
            continue
        inputs.append(tip[i - HISTORY + 1 : i + 1].reshape(-1))
        answers.append(tip[i + AHEAD])
    return inputs, answers


def load(file_list):
    """Examples from several recordings, normalized (relative to 'now', x SCALE)."""
    xs, ys = [], []
    for f in file_list:
        ins, ans = make_examples(np.load(f)["pts"][:, 8, :])
        xs += ins
        ys += ans
    x = torch.tensor(np.array(xs), dtype=torch.float32)
    y = torch.tensor(np.array(ys), dtype=torch.float32)
    now = x[:, -2:]
    return (x - now.repeat(1, HISTORY)) * SCALE, (y - now) * SCALE


def px(guess, truth):
    return (guess - truth).norm(dim=1).mean().item() * SCREEN_PX / SCALE


def train_linear(x, y, steps=2000, lr=0.01):
    torch.manual_seed(0)
    model = nn.Linear(2 * HISTORY, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


files = sorted(glob.glob("data/*.npz"))
folds = [files[i : i + GAMES_PER_FOLD] for i in range(0, len(files), GAMES_PER_FOLD)]
print(f"{len(files)} recordings in {len(folds)} folds\n")
print("fold  test games                                constant speed   linear model   better by")

wins, gains = 0, []
for k, test_files in enumerate(folds):
    train_files = [f for f in files if f not in test_files]
    train_x, train_y = load(train_files)
    test_x, test_y = load(test_files)
    if len(test_x) == 0:
        continue
    model = train_linear(train_x, train_y)
    with torch.no_grad():
        model_err = px(model(test_x), test_y)
    const_err = px(-test_x[:, -4:-2] * AHEAD, test_y)
    gain = (const_err - model_err) / const_err * 100      # % better than constant speed (negative = worse)
    wins += model_err < const_err
    gains.append(gain)
    names = ", ".join(re.search(r"\d{8}_\d{4}", f).group(0) if re.search(r"\d{8}_\d{4}", f) else f
                      for f in test_files)                 # just the date-time part of each file name
    print(f"{k + 1:3d}   {names:41s} {const_err:8.1f} px     {model_err:8.1f} px   {gain:+6.1f}%")

print(f"\nlinear model beat constant speed in {wins} of {len(gains)} folds")
print(f"average improvement: {np.mean(gains):+.1f}%  (spread: {min(gains):+.1f}% to {max(gains):+.1f}%)")
