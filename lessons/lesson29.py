"""Lesson 29: teach the CNN not to shake.

Your study (lesson 28) found: the CNN is the FASTEST lead (21% over no lead) but also the SHAKIEST
(about 2x the settle jitter). Can we keep the speed and lose the shake?

Two ways to get a steadier cursor:

  A  BOLT-ON SMOOTHING (the engineering fix)
     Train the CNN exactly as before, then smooth its guesses afterwards: each frame the push moves only part of
     the way toward the new guess (an "exponential moving average", EMA). Simple, but the smoother sees nothing
     about the hand, so it just adds lag everywhere, including when you're braking.

  B  A JITTER PENALTY IN THE LOSS (the research-y fix)
     Change what the model is TRAINED to want. So far the loss only said "be close to where the finger goes":
         loss = huber(guess, truth)
     Now it also says "and don't make the cursor shake":
         loss = huber(guess, truth)  +  LAMBDA * jitter
     jitter = how much the predicted cursor path bends frame to frame: |p[t] - 2 p[t-1] + p[t-2]|
     (same measure as lessons 27-28; a straight or smoothly curving path scores 0, a zigzag scores high).
     LAMBDA is the dial: 0 = the old model, bigger = cares more about steadiness than accuracy.
     The model gets to DECIDE where being steady is cheap (holding still, settling) and where it isn't (fast moves),
     because it sees the hand. That's the hope, anyway. This lesson tests it.

To compute jitter, the model needs to see 3 CONSECUTIVE frames of the same game at once, so we find every
"triple" (frames t-2, t-1, t in a row) and score the predicted path through them.

THE RESULT IS A TRADE-OFF CURVE, not one number: for each setting, (error, jitter). Lower-left is better on both.
If B's curve sits below-left of A's, the model learned something smoothing can't: being steady WITHOUT lag.
That kind of plot (a "Pareto front") is how research papers compare methods that trade two things off.

1 frame ahead (~33 ms, what FIVES and the game use). One model per setting (no ensemble) so it's a fair, faster
comparison. Every setting's model is saved to runs/steady_cnn_<lambda>.pt in the same format as deploy_cnn_1.pt,
so you can try the winner in the game.

Run:  python lessons/lesson29.py      (~10-15 min on the GPU)
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
AHEAD = 1
GLITCH = 0.15
MAX_GAP = 0.1
SCREEN_PX = 1512
SCALE = 10
HUBER = 0.1
FILTERS = 32
WIDTH = 3
DROPOUT = 0.3
LR = 0.001
STEPS = int(os.environ.get("STEPS", 6000))
CHECK_EVERY = 50
LAMBDAS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0]           # B: how much the loss cares about jitter
EMAS = [1.0, 0.7, 0.5, 0.35, 0.25]            # A: fraction of the way toward the new guess each frame (1 = none)
TIP, KNUCKLE, WRIST = 8, 5, 0
HAND = [TIP, KNUCKLE, WRIST]
SPIKE_MIN, SPIKE_BACK, TOO_FAR = 0.03, 0.5, 1.0
DEVICE = torch.device("mps" if torch.backends.mps.is_available() and "--cpu" not in sys.argv else "cpu")


# ---------- data (same cleaning as lessons 13-28) ----------
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


def make_examples(pts, t, game_id):
    tip = pts[:, TIP, :]
    bad = flag_frames(pts)
    dt = np.diff(t, prepend=t[0] - 1 / 30)
    hands, answers, where = [], [], []
    for i in range(FRAMES - 1, len(tip) - AHEAD):
        a, b = i - FRAMES + 1, i + AHEAD + 1
        path = tip[a:b]
        if np.isnan(path).any() or np.abs(path[1:] - path[:-1]).max() > GLITCH:
            continue
        if bad[a:b].any() or dt[a + 1 : b].max() > MAX_GAP:
            continue
        hands.append(pts[a : i + 1][:, HAND])
        answers.append(tip[i + AHEAD])
        where.append((game_id, i))
    return hands, answers, where


def triples(where):
    """Indices (k-2, k-1, k) of examples that are 3 consecutive frames of the same game."""
    w = np.array(where)
    ok = (w[2:, 0] == w[:-2, 0]) & (w[2:, 1] - w[:-2, 1] == 2)
    k = np.nonzero(ok)[0] + 2
    return torch.tensor(np.stack([k - 2, k - 1, k], 1), dtype=torch.long)


def load(file_list):
    H, Y, W = [], [], []
    for g, f in enumerate(file_list):
        d = np.load(f)
        pts = d["pts"].astype(np.float32)
        t = d["t"] if "t" in d.files else np.arange(len(pts)) / 30
        h, y, w = make_examples(pts, t, g)
        H += h
        Y += y
        W += w
    hands = torch.tensor(np.array(H), dtype=torch.float32)
    y = torch.tensor(np.array(Y), dtype=torch.float32)
    now = hands[:, -1, 0, :]
    x = (hands - now[:, None, None, :]) * SCALE
    return {"x": x.to(DEVICE), "y": ((y - now) * SCALE).to(DEVICE), "now": (now * SCALE).to(DEVICE),
            "tri": triples(W).to(DEVICE), "where": np.array(W)}


def mirror(d):
    """Original + left<->right flipped copy. The flipped copy's triples point into the second half."""
    n = len(d["x"])
    x2, y2, now2 = d["x"].clone(), d["y"].clone(), d["now"].clone()
    x2[..., 0] *= -1
    y2[:, 0] *= -1
    now2[:, 0] *= -1
    return {"x": torch.cat([d["x"], x2]), "y": torch.cat([d["y"], y2]), "now": torch.cat([d["now"], now2]),
            "tri": torch.cat([d["tri"], d["tri"] + n])}


def to_px(v):
    return v * SCREEN_PX / SCALE


# ---------- the model (same as best_model.py / lesson 27) ----------
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


def path_jitter(guess, d):
    """Jitter of the predicted cursor path, per triple (scaled units). cursor = now + guess."""
    p = d["now"] + guess
    a, b, c = d["tri"][:, 0], d["tri"][:, 1], d["tri"][:, 2]
    return (p[c] - 2 * p[b] + p[a]).norm(dim=1)


def objective(model, d, lam):
    g = model(d["x"])
    loss = nn.functional.huber_loss(g, d["y"], delta=HUBER)
    if lam > 0:
        loss = loss + lam * HUBER * path_jitter(g, d).mean()    # x HUBER: puts both parts on a similar scale
    return loss


def fit(lam, train, val):
    torch.manual_seed(0)
    model = HandCNN().to(DEVICE)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best, best_score, best_step = copy.deepcopy(model.state_dict()), float("inf"), 0
    for step in range(STEPS + 1):
        if step % CHECK_EVERY == 0:              # early stopping on the SAME goal (accuracy + lam * steadiness)
            model.eval()
            with torch.no_grad():
                score = objective(model, val, lam).item()
            model.train()
            if score < best_score:
                best_score, best, best_step = score, copy.deepcopy(model.state_dict()), step
        if step == STEPS:
            break
        loss = objective(model, train, lam)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.load_state_dict(best)
    model.eval()
    return model, best_step


def ema(guess, where, alpha):
    """Option A: smooth the guesses over time, restarting at every break in the frames."""
    g = guess.cpu().numpy().copy()
    out = np.empty_like(g)
    for k in range(len(g)):
        fresh = k == 0 or where[k, 0] != where[k - 1, 0] or where[k, 1] != where[k - 1, 1] + 1
        out[k] = g[k] if fresh else out[k - 1] + alpha * (g[k] - out[k - 1])
    return torch.tensor(out, device=guess.device)


def score(guess, d, mask=None):
    err = to_px((guess - d["y"]).norm(dim=1))
    jit = to_px(path_jitter(guess, d))
    return err.mean().item(), jit.mean().item()


# ---------- run ----------
files = sorted(glob.glob("data/*.npz")) + sorted(glob.glob("data_final/*.npz"))
order = list(np.random.default_rng(1).permutation(len(files)))
files = [files[i] for i in order]
test_files, val_files, train_files = files[:5], files[5:9], files[9:]
train, val, test = load(train_files), load(val_files), load(test_files)
mtrain = mirror(train)
print(f"{len(files)} games: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test. "
      f"{len(train['x'])} training examples ({len(train['tri'])} consecutive triples), {STEPS} steps, on {DEVICE}\n")

# reference points
truth = test["y"]
real = score(truth, test)
m1 = train["x"][:, -1, 0] - train["x"][:, -2, 0]
damp_k = ((m1 * train["y"]).sum() / (m1 * m1).sum()).item()
damped = score((test["x"][:, -1, 0] - test["x"][:, -2, 0]) * damp_k, test)
still = score(torch.zeros_like(truth), test)

start = time.time()
results_b, results_a = [], []
os.makedirs("runs", exist_ok=True)
for lam in LAMBDAS:
    model, step = fit(lam, mtrain, val)
    with torch.no_grad():
        g = model(test["x"])
    results_b.append((lam, *score(g, test)))
    torch.save({"models": [model.state_dict()], "ahead": AHEAD, "frames": FRAMES, "points": HAND, "scale": SCALE,
                "dropout": DROPOUT, "filters": FILTERS, "width": WIDTH, "jitter_lambda": lam},
               f"runs/steady_cnn_{lam:g}.pt")
    print(f"B  lambda {lam:<4g} kept step {step:5d}   error {results_b[-1][1]:6.2f} px   "
          f"jitter {results_b[-1][2]:5.2f} px   ({time.time() - start:.0f} s)")
    if lam == 0:
        plain = g                                     # option A starts from the ordinary model
        for alpha in EMAS:
            results_a.append((alpha, *score(ema(g, test["where"], alpha), test)))

print("\n(test games, 1 frame ahead; px, lower is better on both)")
print(f"   {'':34s} {'error':>7s}  {'jitter':>7s}")
print(f"   {'no lead (cursor = finger now)':34s} {still[0]:7.2f}  {still[1]:7.2f}")
print(f"   {'damped speed':34s} {damped[0]:7.2f}  {damped[1]:7.2f}")
print(f"   {'your real fingertip (perfect guess)':34s} {0:7.2f}  {real[1]:7.2f}")
print("\n   A: ordinary CNN + bolt-on smoothing")
for alpha, e, j in results_a:
    print(f"      EMA {alpha:<4g} {'(none)' if alpha == 1 else '':22s} {e:7.2f}  {j:7.2f}")
print("\n   B: CNN trained with a jitter penalty")
for lam, e, j in results_b:
    print(f"      lambda {lam:<4g} {'(the old model)' if lam == 0 else '':19s} {e:7.2f}  {j:7.2f}")

# Who wins at the same steadiness? For each B model, find what A's error would be at B's jitter (interpolate A).
aj = np.array([j for _, _, j in results_a])
ae = np.array([e for _, e, _ in results_a])
order_a = np.argsort(aj)
print("\n   SAME STEADINESS, WHO'S MORE ACCURATE?  (B vs A's curve at the same jitter)")
for lam, e, j in results_b[1:]:
    if aj.min() <= j <= aj.max():
        a_err = np.interp(j, aj[order_a], ae[order_a])
        print(f"      lambda {lam:<4g} jitter {j:5.2f}:  B error {e:6.2f}  vs  A error {a_err:6.2f}   "
              f"-> {'B' if e < a_err else 'A'} wins by {abs(a_err - e):.2f} px")
    else:
        print(f"      lambda {lam:<4g} jitter {j:5.2f}:  outside A's range (A can't get that steady, or that shaky)")
print("\nIf B wins: the model learned WHERE to be steady (it sees the hand), which plain smoothing can't.")
print("Models saved: runs/steady_cnn_<lambda>.pt  (try one in the game: see leads.py CNN_PATH)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([j for _, _, j in results_a], [e for _, e, _ in results_a], "o-", color="tab:gray",
            label="A: CNN + bolt-on smoothing")
    for alpha, e, j in results_a:
        ax.annotate(f"EMA {alpha:g}", (j, e), textcoords="offset points", xytext=(5, -10), fontsize=8, color="gray")
    ax.plot([j for _, _, j in results_b], [e for _, e, _ in results_b], "o-", color="tab:blue",
            label="B: CNN trained with a jitter penalty")
    for lam, e, j in results_b:
        ax.annotate(f"λ {lam:g}", (j, e), textcoords="offset points", xytext=(5, 5), fontsize=8, color="tab:blue")
    ax.scatter([damped[1]], [damped[0]], color="tab:orange", marker="s", s=60, label="damped speed", zorder=3)
    ax.axvline(real[1], color="green", ls=":", label="your real fingertip's jitter")
    ax.set_xlabel("jitter (px per frame)  ← steadier")
    ax.set_ylabel("error (px)  ← more accurate")
    ax.set_title("Accuracy vs steadiness: lower-left is better")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("lesson29.png", dpi=120)
    print("Saved lesson29.png  (open lesson29.png)")
except ImportError:
    pass
