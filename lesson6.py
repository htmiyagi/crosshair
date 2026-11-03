"""Lesson 6: train on ALL recordings, clean out glitches, and test on games the model never saw.

Same idea as lesson 5 (last 8 fingertip positions -> where it is 2 frames later), but:
  1. every recording in data/ is used, not just one
  2. examples are built inside each recording, so no example mixes two different games
  3. examples with a tracking glitch (a "teleport") are thrown out
  4. the test set is whole games the model never trained on (a fairer test than "the end of one game")

Run from the crosshair folder:  python lessons/lesson6.py
"""

import glob

import numpy as np
import torch
from torch import nn

torch.manual_seed(0)

HISTORY = 8          # how many past frames the model sees
AHEAD = 2            # predict this many frames ahead (2 frames ~ 60 ms)
GLITCH = 0.15        # a jump bigger than this (fraction of the picture) in one frame = tracking glitch
TEST_GAMES = 3       # the last few recordings are kept for testing only
SCREEN_PX = 1512     # to turn errors into rough screen pixels
SCALE = 10           # inputs and answers are multiplied by this (normalization, see lesson 5)


def make_examples(tip):
    """Turn one recording's fingertip track (frames x 2) into examples. Returns (inputs, answers, skipped)."""
    inputs, answers, skipped = [], [], 0
    for i in range(HISTORY - 1, len(tip) - AHEAD):
        path = tip[i - HISTORY + 1 : i + AHEAD + 1]          # the history frames AND the answer frame
        if np.isnan(path).any():
            continue                                          # hand was lost somewhere in here
        jumps = path[1:] - path[:-1]                          # how far the finger moved each frame
        if np.abs(jumps).max() > GLITCH:
            skipped += 1
            continue                                          # a teleport: tracking glitch, not a real move
        history = tip[i - HISTORY + 1 : i + 1]
        inputs.append(history.reshape(-1))
        answers.append(tip[i + AHEAD])
    return inputs, answers, skipped


# --- 1. load every recording, build examples per game
files = sorted(glob.glob("data/*.npz"))
train_files, test_files = files[:-TEST_GAMES], files[-TEST_GAMES:]
print(f"{len(files)} recordings: {len(train_files)} for training, {len(test_files)} for testing")


def load(file_list):
    all_in, all_ans, total_skipped = [], [], 0
    for f in file_list:
        tip = np.load(f)["pts"][:, 8, :]                      # point 8 = index fingertip
        ins, ans, skipped = make_examples(tip)
        all_in += ins
        all_ans += ans
        total_skipped += skipped
    x = torch.tensor(np.array(all_in), dtype=torch.float32)
    y = torch.tensor(np.array(all_ans), dtype=torch.float32)
    return x, y, total_skipped


train_x, train_y, skip_train = load(train_files)
test_x, test_y, skip_test = load(test_files)
print(f"examples: {len(train_x)} train, {len(test_x)} test  (glitches thrown out: {skip_train + skip_test})")


# --- 2. normalize: everything relative to where the finger is now, x SCALE
def normalize(x, y):
    now = x[:, -2:]
    return (x - now.repeat(1, HISTORY)) * SCALE, (y - now) * SCALE


train_x, train_y = normalize(train_x, train_y)
test_x, test_y = normalize(test_x, test_y)


def px(guess, truth):
    """Average distance between guess and truth, in rough screen pixels."""
    return (guess - truth).norm(dim=1).mean().item() * SCREEN_PX / SCALE


# --- 3. the baselines (no learning)
prev = test_x[:, -4:-2]                                       # one frame back, relative to now
no_prediction = torch.zeros_like(test_y)                      # "you'll stay where you are"
constant_speed = -prev * AHEAD                                # "you'll keep the same speed"


# --- 4. train a model and report its test error
def train(model, steps=3000, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        loss = nn.functional.mse_loss(model(train_x), train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print(f"   step {step:4d}  train loss {loss.item():.5f}")
    with torch.no_grad():
        return px(model(train_x), train_y), px(model(test_x), test_y)


print("\nlinear model:")
linear_train, linear_test = train(nn.Linear(2 * HISTORY, 2))

print("\nMLP (hidden layers):")
mlp_train, mlp_test = train(nn.Sequential(
    nn.Linear(2 * HISTORY, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 2),
))

# --- 5. the scoreboard (all on the held-out test games)
print("\nTEST GAMES (never seen in training)")
print(f"   no prediction:   {px(no_prediction, test_y):6.1f} px")
print(f"   constant speed:  {px(constant_speed, test_y):6.1f} px   <- the one to beat")
print(f"   linear model:    {linear_test:6.1f} px   (on its training games: {linear_train:.1f})")
print(f"   MLP:             {mlp_test:6.1f} px   (on its training games: {mlp_train:.1f})")