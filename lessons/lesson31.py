"""Lesson 31: teach a CNN to read handwritten letters (from the strokes, not a picture).

NEW IDEA: CLASSIFICATION. Every model so far answered "WHERE?" (two numbers: the fingertip's x and y). That's
called REGRESSION. This one answers "WHICH ONE?" (is this an a, b, c, ...?). That's called CLASSIFICATION.
Three things change:
  1. The model outputs one SCORE per letter (35 scores), not 2 numbers.
  2. SOFTMAX turns the scores into chances that add up to 100%: "a 91%, o 6%, d 2%, ...".
  3. The loss is CROSS-ENTROPY: how surprised the model was by the right answer. If it said "a: 91%" and it was
     an a, loss is small. If it said "a: 1%", loss is huge. Training = being less surprised by the truth.

THE DATA: UJI Pen Characters. 60 people wrote every character twice with a stylus on a tablet. Each sample is
its STROKES: lists of (x, y) points, in drawing order, one list per pen-down. Like your fingertip path in Crosshair.

THE TASK (same as the dataset's authors, so we can compare): 35 classes = 26 letters (upper and lower case count
as the same letter) + digits 1-9 (zero counts as o, because they look the same). They got 8.47% error.

GETTING STROKES READY FOR A CNN (a CNN needs every sample the same length):
  - Put the strokes end to end, and RESAMPLE to exactly 64 evenly spaced points along the ink (a fast stroke and
    a slow stroke give the same 64 points; that matters for air-writing, where your speed is all over the place).
  - Center it and scale it to the same size, keeping its shape (an "l" stays tall and thin).
  - Each point gets 5 numbers: x, y, the step to the next point (dx, dy), and "pen lifted before this point" (1/0).
    That last one tells the model where one stroke ended and the next began (the two strokes of an x).

FAIR TEST: split by WRITER, like we split by game. The 20 "tst" writers are never seen in training, so the score
means "how well does it read someone new" (which is exactly your situation: it'll never have seen YOUR hand).

AUGMENTATION: while training, each letter gets randomly tilted, stretched, slanted and wobbled a little. Real
air-writing is shakier and more slanted than tablet writing, so this teaches the model not to care about that.
(Same idea as mirroring in lesson 19: free extra data.)

Saves the trained model to runs/letters_cnn.pt and a picture of test letters with its guesses (letters31.png).
Run from the crosshair folder:  python lessons/lesson31.py      (a few minutes on the GPU)
Needs letters/ujipenchars2.txt
"""

import math
import os
import sys
import time

import numpy as np
import torch
from torch import nn

POINTS = 64
STEPS = int(os.environ.get("STEPS", 4000))
BATCH = 256
LR = 0.001
CHECK_EVERY = 100
DEVICE = torch.device("mps" if torch.backends.mps.is_available() and "--cpu" not in sys.argv else "cpu")
CLASSES = list("abcdefghijklmnopqrstuvwxyz123456789")       # 0 counts as o; upper case counts as lower


# ---------- read the file ----------
def read_uji(path):
    """Returns a list of (character, writer, strokes); strokes = list of (n, 2) arrays."""
    samples, char, writer, strokes = [], None, None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("//"):
                continue
            if parts[0] == "WORD":
                if char is not None:
                    samples.append((char, writer, strokes))
                char, writer, strokes = parts[1], parts[2].rsplit("-", 1)[0], []
            elif parts[0] == "POINTS":
                nums = np.array(parts[3:], float).reshape(-1, 2)
                strokes.append(nums)
    if char is not None:
        samples.append((char, writer, strokes))
    return samples


def label_of(ch):
    c = ch.lower()
    if c == "0":
        c = "o"
    return CLASSES.index(c) if c in CLASSES else None


# ---------- strokes -> 64 points x 5 numbers ----------
def resample(strokes, n=POINTS):
    """Evenly spaced points along the ink. Returns (n, 2) points and (n,) 'pen lifted before this point'."""
    strokes = [s for s in strokes if len(s) > 0]
    lengths = [max(np.linalg.norm(np.diff(s, axis=0), axis=1).sum(), 1e-6) for s in strokes]
    total = sum(lengths)
    # give each stroke a share of the 64 points, by how much ink it has (at least 2 each)
    share = [max(2, int(round(n * L / total))) for L in lengths]
    while sum(share) > n:
        share[int(np.argmax(share))] -= 1
    while sum(share) < n:
        share[int(np.argmax(lengths))] += 1
    pts, lift = [], []
    for k, (s, m) in enumerate(zip(strokes, share)):
        if len(s) == 1:
            s = np.vstack([s, s])
        seg = np.linalg.norm(np.diff(s, axis=0), axis=1)
        along = np.concatenate([[0], np.cumsum(seg)])
        want = np.linspace(0, along[-1], m)
        pts.append(np.stack([np.interp(want, along, s[:, 0]), np.interp(want, along, s[:, 1])], 1))
        lift += [1.0 if (k > 0 and i == 0) else 0.0 for i in range(m)]
    return np.concatenate(pts), np.array(lift)


def features(pts, lift):
    """Center + scale (keeping the shape), then x, y, dx, dy, lift."""
    lo, hi = pts.min(0), pts.max(0)
    size = max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
    p = (pts - (lo + hi) / 2) / size * 2                    # about -1..1 on the longer side
    step = np.diff(p, axis=0, append=p[-1:])
    step[:-1][lift[1:] > 0.5] = 0                               # no "step" across a pen lift (that's not ink)
    return np.concatenate([p, step, lift[:, None]], 1).astype(np.float32)   # (64, 5)


def augment(strokes, rng):
    """A randomly tilted, stretched, slanted, wobbly copy (training only)."""
    angle = rng.uniform(-0.25, 0.25)                        # about +-15 degrees
    sx, sy = rng.uniform(0.8, 1.2, 2)
    shear = rng.uniform(-0.3, 0.3)
    m = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]) \
        @ np.array([[sx, shear], [0, sy]])
    allp = np.concatenate(strokes)
    size = np.ptp(allp, 0).max() + 1e-6
    out = []
    for s in strokes:
        s2 = (s - allp.mean(0)) @ m.T
        s2 = s2 + rng.normal(0, 0.01 * size, s2.shape)      # a little hand wobble
        out.append(s2)
    return out


def build(samples, rng=None):
    X = []
    for _, _, strokes in samples:
        s = augment(strokes, rng) if rng is not None else strokes
        X.append(features(*resample(s)))
    return torch.tensor(np.array(X))


# ---------- the model: a 1D CNN, like Crosshair's, but deeper and with 35 outputs ----------
class LetterCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES), drop=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(5, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),       # 64 points -> 32
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),      # 32 -> 16
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(128 * 16, 128), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(128, n_classes))

    def forward(self, x):                      # x: (n, 64 points, 5 numbers)
        h = self.conv(x.permute(0, 2, 1))      # Conv1d wants (n, channels, points)
        return self.head(h.reshape(len(x), -1))   # 35 scores ("logits") per letter


if __name__ == "__main__":
    path = "letters/ujipenchars2.txt"
    samples = [(c, w, s) for c, w, s in read_uji(path) if label_of(c) is not None]
    trn = [s for s in samples if s[1].startswith("trn")]
    tst = [s for s in samples if s[1].startswith("tst")]
    writers = sorted({w for _, w, _ in trn})
    val_writers = set(writers[-5:])                         # 5 training writers held out for early stopping
    fit = [s for s in trn if s[1] not in val_writers]
    val = [s for s in trn if s[1] in val_writers]
    print(f"{len(samples)} samples, {len(CLASSES)} classes. Train {len(fit)} ({len(writers) - 5} writers), "
          f"validation {len(val)} (5 writers), test {len(tst)} ({len({w for _, w, _ in tst})} NEW writers). "
          f"On {DEVICE}\n")

    y_fit = torch.tensor([label_of(c) for c, _, _ in fit])
    xv, yv = build(val).to(DEVICE), torch.tensor([label_of(c) for c, _, _ in val]).to(DEVICE)
    xt, yt = build(tst).to(DEVICE), torch.tensor([label_of(c) for c, _, _ in tst]).to(DEVICE)

    # a pool of augmented copies, rebuilt every few hundred steps (making them every step would be slow)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    model = LetterCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_err, best_step = None, 1.0, 0
    start = time.time()
    for step in range(STEPS + 1):
        if step % 500 == 0:
            pool = torch.cat([build(fit), build(fit, rng), build(fit, rng)]).to(DEVICE)   # clean + 2 wobbly copies
            pool_y = torch.cat([y_fit] * 3).to(DEVICE)
        if step % CHECK_EVERY == 0:
            model.eval()
            with torch.no_grad():
                err = (model(xv).argmax(1) != yv).float().mean().item()
            model.train()
            if err < best_err:
                best_err, best_step = err, step
                best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if step % 1000 == 0:
                print(f"   step {step:5d}   validation error {err * 100:5.1f}%   ({time.time() - start:.0f} s)")
        if step == STEPS:
            break
        idx = torch.randint(len(pool), (BATCH,), device=DEVICE)
        loss = nn.functional.cross_entropy(model(pool[idx]), pool_y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        scores = model(xt)
        guess = scores.argmax(1)
        top3 = (scores.topk(3, dim=1).indices == yt[:, None]).any(1)
    err = (guess != yt).float().mean().item()
    print(f"\nkept step {best_step}.  TEST (20 writers it never saw):")
    print(f"   error        {err * 100:5.2f}%     (the dataset's authors: 8.47%)")
    print(f"   top-3 error  {(~top3).float().mean().item() * 100:5.2f}%     (right answer not even in its top 3 guesses)")

    # which letters does it mix up?
    g, t = guess.cpu().numpy(), yt.cpu().numpy()
    pairs = {}
    for a, b in zip(t, g):
        if a != b:
            pairs[(CLASSES[a], CLASSES[b])] = pairs.get((CLASSES[a], CLASSES[b]), 0) + 1
    print("\n   most common mix-ups (real -> guessed):")
    for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:10]:
        print(f"      {a} -> {b}   {n} times")
    per = [(c, (g[t == i] == i).mean()) for i, c in enumerate(CLASSES)]
    print("\n   hardest letters:  " + "  ".join(f"{c} {acc * 100:.0f}%" for c, acc in sorted(per, key=lambda x: x[1])[:8]))

    os.makedirs("runs", exist_ok=True)
    torch.save({"model": best, "classes": CLASSES, "points": POINTS}, "runs/letters_cnn.pt")
    print("\nSaved runs/letters_cnn.pt")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pick = np.random.default_rng(3).choice(len(tst), 24, replace=False)
        fig, axes = plt.subplots(3, 8, figsize=(14, 6))
        probs = torch.softmax(scores, 1).cpu().numpy()
        for ax, i in zip(axes.flat, pick):
            ch, _, strokes = tst[i]
            for s in strokes:
                ax.plot(s[:, 0], -s[:, 1], "k-", lw=2)
            ok = g[i] == t[i]
            ax.set_title(f"'{ch}' -> {CLASSES[g[i]]} ({probs[i, g[i]] * 100:.0f}%)", color="green" if ok else "red",
                         fontsize=10)
            ax.set_aspect("equal")
            ax.axis("off")
        fig.suptitle("Test letters from people it never saw: real -> guessed (how sure). Red = wrong.")
        fig.tight_layout()
        fig.savefig("letters31.png", dpi=110)
        print("Saved letters31.png  (open letters31.png)")
    except ImportError:
        pass
