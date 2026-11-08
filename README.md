# Crosshair

**Predicting hand motion to make webcam hand tracking precise and lag-free.**

Crosshair is an experiment in predicting where a fingertip will be 60 ms in the future, so a
hand-tracking cursor feels like it keeps up with your hand instead of trailing behind it.

It's the research side of [FIVES](https://github.com/htmiyagi/fives), a webcam hand-tracking interface
that replaces the mouse with gestures. FIVES tracks the hand at ~28 fps with ~21 ms of processing, so the
cursor always shows where the finger *was*. A prediction model is one way to close that gap.

> **Status:** a small CNN beats a damped-velocity baseline by **15%** on held-out games, and it's now the cursor
> lead in FIVES. In a blind study (lessons 28-29) it made pointing **21% faster** than no lead; a version trained with a
> jitter penalty ("STEADY") looks like it keeps that speed with much less shake (early: only a few games so far). Next: more players (see "Play it" below).

## How it works

<img width="1902" height="1458" alt="Adobe Express - Screen Recording 2026-10-30 at 12 14 29 AM" src="https://github.com/user-attachments/assets/4868186a-1620-4f1c-b391-255800c651a4" />

```
webcam ─► MediaPipe (21 hand points) ─► fingertip, knuckle, wrist ─► last 8 frames ─► model ─► fingertip 2 frames (~60 ms) ahead
```

1. **Collect data with a game.** `game.py` is a target-acquisition game: lock the reticle onto targets as fast as
   you can. It has player accounts, a score and timer, and a leaderboard, so friends can contribute data.
   Every game records the raw MediaPipe hand landmarks (positions only, no images), the targets, and who played.
2. **Clean it.** Frames where tracking was lost, or that break two physical rules (a one-frame "spike" out and
   back, or the fingertip jumping farther from its knuckle than a finger is long), are left out.
3. **Train and evaluate fairly:** cross-validation across whole games, early stopping on held-out
   validation games, and a final "lockbox" of games no model saw during development.
4. **Compare against baselines that need no learning**, including the strongest one found: *damped velocity*
   (the last move × one fitted number).

## Results

Error = average distance between the predicted and actual fingertip, in screen pixels (1512 px wide).
36 games for cross-validation (6 folds), plus 5 untouched lockbox games.

| Predictor | 6-fold CV | Lockbox |
|---|---|---|
| Damped velocity (baseline) | 20.44 px | 30.40 px |
| MLP, fingertip only, dropout | 18.67 px | 27.02 px |
| **CNN, fingertip + knuckle + wrist, dropout** | **17.46 px** | **25.76 px** |

The CNN beat the MLP in all 6 folds. The lockbox games were faster (harder) than earlier ones, but the ranking held.
The final model and its training recipe are in [`best_model.py`](best_model.py).

## What I found along the way

The model took two days of experiments (`lessons/`). The path mattered more than the final number:

- **Noise isn't the problem.** Holding a finger still, MediaPipe's jitter is only ~1 px, so almost all of the
  error is real movement the model didn't see coming (lesson 10).
- **Tracker glitches hid a lot of error.** 5% of examples touched a glitch frame and held 17% of all error.
  A physics-based filter (spikes, fingertip-to-knuckle distance) fixed that (lessons 11-13).
- **My first "win" was mostly fake.** On clean data, the early MLP was basically damped velocity in disguise.
  It had looked better only because glitches threw off the baseline's fit (lesson 13).
- **The hand helps on turns.** Adding knuckle and wrist made turns and speed-ups ~10% better but steady
  movement worse (lesson 14). A mixture-of-experts gate learned *on its own* to trust the hand more during turns
  (lesson 15).
- **A transformer rediscovered damped velocity.** Its attention weights showed it looking almost only at the
  previous frame, in every situation. Not enough data for attention to find more (lesson 18).
- **Mirroring is free data.** Flipping recordings left ↔ right was worth about as much as doubling the games (lesson 19).
- **Three folds can lie.** A combination that looked best on 3 folds fell apart on 6 (lesson 20).
- **The CNN was never bad; its training was.** Training curves showed the learning rate was too high (jumpy) and the
  model memorized games after a few hundred steps. A calmer learning rate plus dropout fixed both, and the CNN
  went from tied with damped velocity to 15% better (lessons 22-25).
- **More data still helps.** The learning curve is still dropping at 27 training games (lesson 26).

Applied to FIVES: the damped-velocity "lead" from these baselines is already shipped there, and you can feel the lag drop.

## Play it (one line, for anyone)

Open Terminal, paste this, press Enter:

```bash
curl -fsSL https://raw.githubusercontent.com/htmiyagi/crosshair/main/install.sh | bash
```

A window asks **Install Crosshair?** After about 5 minutes there's a **Crosshair** icon on your Desktop: double-click it
to play. The first time, macOS asks to use the camera (click Allow, then quit Terminal with Cmd + Q and open it again).
New players are in the blind study automatically. When you're done, click **SEND GAMES TO HOYU** (bottom right of the
menu): it asks first, then uploads your games (hand positions only, no camera images) for the research.

## Run it (development)

Needs Python **3.12** (MediaPipe doesn't support 3.13 yet) and a webcam. Tested on macOS.

```bash
git clone https://github.com/htmiyagi/crosshair.git
cd crosshair
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python game.py                       # sign in or make a player, then play (q quits a game early)
python lessons/lesson25.py           # the full comparison: 6-fold CV + lockbox (data_final/)
python best_model.py                 # print the winning model
```

The menu (bottom right) switches the color theme and the next-target preview. On macOS, allow your terminal to
use the camera the first time. Recordings go to `data/` and player accounts to `runs/`; neither is uploaded.
Recordings go to data/ and player accounts to runs/; neither is uploaded. The recordings behind the results table aren’t included in this repo, so to run lessons/lesson25.py or best_model.py, play some games with game.py first (and move 5 of them into data_final/ for the lockbox).

## Files

| File | What it does |
|---|---|
| `game.py` | the data-collection game: targets, scope-style cursor, score, timer |
| `arcade.py` | menus, player accounts (passcodes stored scrambled), leaderboard, themes |
| `best_model.py` | the winning model and its training recipe, on its own |
| `common.py` | camera, MediaPipe setup, shared settings |
| `features.py`, `model.py`, `train.py` | the original training pipeline (used by `game.py --model`) |
| `fake_data.py` | synthetic recordings for testing without a camera |
| `lessons/` | the experiments, in order (see below) |

**Lessons:** 1-5 PyTorch basics · 6-9 cross-validation, overfitting, baselines · 10 noise floor · 11-13 error
breakdown and glitch filter · 14-15 hand points and mixture of experts · 16-17 CNNs · 18 transformer and attention ·
19 learning curve and mirroring · 20-21 inputs and loss functions · 22-24 debugging training · 25 final comparison ·
26 learning curve for the final model.

## How this was built

In addition to help project FIVES, I built this to help me learn basic ML Concepts and PyTorch with help from Claude to conduct my experiments in a a lesson style workflow. Lessons 1-5 are my own, written step by step while learning tensors, training loops, and normalization.  I chose which questions to test, ran every experiment, read the results, and decided what to try next (for example, suspecting that the worst errors were small tracker glitches, noticing that the game measured reacting rather than planning, which led to the next-target preview, and asking whether the model could be strict only on fast moves).
