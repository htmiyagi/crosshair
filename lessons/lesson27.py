"""Lesson 27: the deploy model. Getting the CNN ready for FIVES.

Four upgrades over lesson 25's winner, and a check that numbers alone can't do:

  1. MORE DATA   the 5 lockbox games are used up as a test, so they join training (data/ + data_final/).
  2. LONGER      8000 steps instead of 5000 (it was still improving).
  3. ENSEMBLE    3 copies of the CNN, each from a different random start; the guess is their AVERAGE.
                 Their random wobbles partly cancel, so the average is usually a bit more accurate AND smoother.
  4. JITTER      for a cursor, smoothness matters as much as accuracy: a guess that shakes feels worse even if
                 it's closer on average. Jitter = how much the predicted path changes direction frame to frame
                 (its "acceleration", in px). We compare it with damped speed and with your real fingertip.
  5. HOW FAR AHEAD   everything is trained twice: 1 frame ahead (~33 ms, close to FIVES's 40 ms lead) and
                 2 frames ahead (~66 ms, what all the lessons used).

Honest test: 4 games are set aside before anything else and used ONLY for the final numbers (a fresh lockbox).
3 more are the validation games for early stopping. Everything else is training data.
At the end, the 1-frame and 2-frame ensembles are saved to runs/deploy_cnn_1.pt and runs/deploy_cnn_2.pt.

Run:  python lessons/lesson27.py      (~15-25 min on the GPU)
"""

import copy
import glob
import os
import sys
import time

import numpy as np
import torch
from torch import nn

FRAMES = 8
GLITCH = 0.15
MAX_GAP = 0.1
SCREEN_PX = 1512
SCALE = 10
HUBER = 0.1
FILTERS = 32
WIDTH = 3
DROPOUT = 0.3
LR = 0.001
STEPS = 8000
CHECK_EVERY = 50
SEEDS = [0, 1, 2]                    # the ensemble: 3 models
TIP, KNUCKLE, WRIST = 8, 5, 0
HAND = [TIP, KNUCKLE, WRIST]
SPIKE_MIN, SPIKE_BACK, TOO_FAR = 0.03, 0.5, 1.0
DEVICE = torch.device("mps" if torch.backends.mps.is_available() and "--cpu" not in sys.argv else "cpu")


# ---------- data (same cleaning as lessons 13-26) ----------
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


def make_examples(pts, t, ahead, game_id):
    """Examples plus WHERE each came from (game, frame), so we can find runs of consecutive frames for jitter."""
    tip = pts[:, TIP, :]
    bad = flag_frames(pts)
    dt = np.diff(t, prepend=t[0] - 1 / 30)
    hands, answers, where = [], [], []
    for i in range(FRAMES - 1, len(tip) - ahead):
        a, b = i - FRAMES + 1, i + ahead + 1
        path = tip[a:b]
        if np.isnan(path).any() or np.abs(path[1:] - path[:-1]).max() > GLITCH:
            continue
        if bad[a:b].any() or dt[a + 1 : b].max() > MAX_GAP:
            continue
        hands.append(pts[a : i + 1][:, HAND])
        answers.append(tip[i + ahead])
        where.append((game_id, i))
    return hands, answers, where


def load(file_list, ahead):
    H, Y, W = [], [], []
    for g, f in enumerate(file_list):
        d = np.load(f)
        pts = d["pts"].astype(np.float32)
        t = d["t"] if "t" in d.files else np.arange(len(pts)) / 30
        h, y, w = make_examples(pts, t, ahead, g)
        H += h
        Y += y
        W += w
    hands = torch.tensor(np.array(H), dtype=torch.float32)
    y = torch.tensor(np.array(Y), dtype=torch.float32)
    now = hands[:, -1, 0, :]
    return ((hands - now[:, None, None, :]) * SCALE).to(DEVICE), ((y - now) * SCALE).to(DEVICE), np.array(W), now.to(DEVICE)


def to_px(v):
    return v * SCREEN_PX / SCALE


def mirror(h, y):
    h2, y2 = h.clone(), y.clone()
    h2[..., 0] *= -1
    y2[:, 0] *= -1
    return torch.cat([h, h2]), torch.cat([y, y2])


# ---------- the model (same as best_model.py) ----------
class HandCNN(nn.Module):
    def __init__(self, drop=DROPOUT):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, FILTERS, kernel_size=WIDTH), nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(FILTERS, FILTERS, kernel_size=WIDTH), nn.ReLU(), nn.Dropout(drop),
        )
        self.head = nn.Sequential(nn.Linear(FILTERS * 4, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 2))

    def forward(self, hands):
        x = hands.reshape(len(hands), FRAMES, -1).permute(0, 2, 1)
        return self.head(self.conv(x).reshape(len(hands), -1))


def predict(models, h):
    """Average guess of all the models (dropout off)."""
    with torch.no_grad():
        for m in models:
            m.eval()
        return torch.stack([m(h) for m in models]).mean(0)


def fit(seed, x, y, vx, vy):
    torch.manual_seed(seed)
    model = HandCNN().to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_err, best_step = copy.deepcopy(model.state_dict()), float("inf"), 0
    for step in range(STEPS + 1):
        if step % CHECK_EVERY == 0:
            model.eval()
            with torch.no_grad():
                v = (model(vx) - vy).norm(dim=1).mean().item()
            model.train()
            if v < best_err:
                best_err, best, best_step = v, copy.deepcopy(model.state_dict()), step
        if step == STEPS:
            break
        loss = nn.functional.huber_loss(model(x), y, delta=HUBER)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.load_state_dict(best)
    model.eval()
    return model, best_step


def jitter(points, where):
    """Average 'acceleration' of a path, in px: |p[t] - 2 p[t-1] + p[t-2]| over runs of consecutive frames."""
    p = points.cpu().numpy()
    ok = ((where[2:, 0] == where[:-2, 0]) & (where[2:, 1] - where[:-2, 1] == 2))   # same game, frames t-2..t
    acc = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1)[ok]
    return acc.mean() * SCREEN_PX


# ---------- run ----------
files = sorted(glob.glob("data/*.npz")) + sorted(glob.glob("data_final/*.npz"))
order = list(np.random.default_rng(1).permutation(len(files)))
files = [files[i] for i in order]
test_files, val_files, train_files = files[:4], files[4:7], files[7:]
print(f"{len(files)} games: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} fresh test "
      f"(lockbox), {STEPS} steps, ensemble of {len(SEEDS)}, running on: {DEVICE}\n")

os.makedirs("runs", exist_ok=True)
start = time.time()
report = []
for ahead in (1, 2):
    fh, fy, _, _ = load(train_files, ahead)
    vh, vy, _, _ = load(val_files, ahead)
    th, ty, tw, tnow = load(test_files, ahead)
    mh, my = mirror(fh, fy)
    m1 = fh[:, -1, 0] - fh[:, -2, 0]
    damp_k = ((m1 * fy).sum() / (m1 * m1).sum()).item()

    models = []
    for seed in SEEDS:
        model, step = fit(seed, mh, my, vh, vy)
        models.append(model)
        print(f"{ahead} frame{'s' if ahead > 1 else ''} ahead: model {seed + 1}/{len(SEEDS)} kept at step {step}"
              f"   ({time.time() - start:.0f} s)")

    guesses = {
        "no prediction": torch.zeros_like(ty),
        "damped speed": (th[:, -1, 0] - th[:, -2, 0]) * damp_k,
        "one CNN": predict(models[:1], th),
        f"ensemble of {len(SEEDS)}": predict(models, th),
    }
    truth_path = tnow + ty / SCALE                      # where the fingertip really went (camera units)
    real_jitter = jitter(truth_path, tw)
    for name, g in guesses.items():
        err = to_px((g - ty).norm(dim=1)).mean().item()
        path = tnow + g / SCALE                         # the cursor path this predictor would draw
        report.append((ahead, name, err, jitter(path, tw), real_jitter))

    torch.save({"models": [m.state_dict() for m in models], "ahead": ahead, "frames": FRAMES, "points": HAND,
                "scale": SCALE, "dropout": DROPOUT, "filters": FILTERS, "width": WIDTH},
               f"runs/deploy_cnn_{ahead}.pt")

print("\n(fresh test games; error and jitter in px, lower is better)")
print(f"{'ahead':>6s}  {'predictor':16s}  {'error':>7s}  {'jitter':>7s}   (your real fingertip's jitter)")
for ahead, name, err, jit, real in report:
    print(f"{ahead * 33:4d}ms  {name:16s}  {err:7.2f}  {jit:7.2f}   ({real:.2f})")
print("\nGood for FIVES = low error AND jitter close to your real fingertip's (more than that = a shakier cursor).")
print("Saved runs/deploy_cnn_1.pt and runs/deploy_cnn_2.pt (each holds the 3 ensemble models).")
