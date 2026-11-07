"""Lesson 28: Fitts' law. Scoring your cursor the way pointing research does.

So far everything was "pixels off, 60 ms ahead". That's about the model. Research on pointing devices (mice,
touchpads, VR hands) asks a different question: how fast can a PERSON hit things with it? And it answers with
one famous rule, Fitts' law (1954):

    difficulty   ID = log2(D / W + 1)       D = distance to the target, W = its size
    time         MT = a + b * ID            a straight line: harder targets take longer, by a steady amount
    score        throughput = ID / MT       "bits per second": how much pointing difficulty you handle per second

Why log2? Think of aiming as narrowing down where the target is: each halving of the remaining distance-to-size
ratio is one "bit". A target twice as far (or half as big) needs about one more bit, so about b more seconds.

This lesson reads every recorded game (they all saved where targets were, when they appeared and closed, and where
the cursor was) and:
  1. measures D, W and MT for each target  (see fitts.py, which the game uses too)
  2. fits the straight line MT = a + b*ID and reports R squared (1 = perfect line, 0 = no relationship),
     two ways: through every single target (noisy), and the STANDARD way: group targets by difficulty
     (every 0.5 bits), average each group, and fit the line through the averages
  3. reports your throughput overall, per game over time, with vs without the next-target preview, and per player
  4. plots time vs difficulty (fitts.png), and the same split by cursor lead (fitts_by_lead.png)
  5. for games played with a cursor lead: throughput AND settle jitter per lead, per player, and your guesses

Run:  python lessons/lesson28.py
"""

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # so "import fitts" finds the main folder
import fitts                                                          # noqa: E402

files = sorted(glob.glob("data/*.npz")) + sorted(glob.glob("data_final/*.npz"))
games = []
for f in files:
    tr, meta = fitts.from_recording(f)
    if tr:
        games.append((Path(f).stem, tr, meta))
print(f"{len(files)} recordings, {len(games)} with targets recorded\n")
if not games:
    raise SystemExit("No games with target data yet.")

every = [x for _, tr, _ in games for x in tr]
s = fitts.summary(every)
print(f"ALL GAMES: {s['n']} targets")
print(f"   MT = {s['a']:.2f} + {s['b']:.2f} x ID   (seconds)    R squared = {s['r2']:.2f}")
print(f"   throughput: {s['throughput']:.2f} bits/s")
print(f"   (a = time that doesn't depend on difficulty: reacting, starting to move.  b = extra seconds per bit.)")
bn = fitts.binned(every)
if bn:
    print(f"\nSTANDARD FIT (targets grouped every 0.5 bits, line through the group averages):")
    print(f"   MT = {bn['a']:.2f} + {bn['b']:.2f} x ID    R squared = {bn['r2']:.2f}")
    for bid, bmt, n in bn["bins"]:
        print(f"      ID {bid:4.2f} bits: {bmt:.2f} s average  ({n} targets)")
print()

print("BY GAME (oldest first): does your throughput go up as you practice?")
for name, tr, meta in games:
    g = fitts.summary(tr)
    if g:
        tag = "  preview" if meta.get("preview") else ""
        who = meta.get("player", "")
        when = re.search(r"\d{8}_\d{4}", name)
        print(f"   {when.group(0) if when else name:14s} {g['throughput']:5.2f} bits/s   ({g['n']:2d} targets){tag}  {who}")

for label, want in (("next-target preview OFF", False), ("next-target preview ON", True)):
    tr = [x for _, t, m in games if bool(m.get("preview")) == want for x in t]
    g = fitts.summary(tr)
    if g:
        print(f"\n{label}: {g['throughput']:.2f} bits/s over {g['n']} targets   (MT = {g['a']:.2f} + {g['b']:.2f} x ID)")

# ---- the user study: does the cursor lead change how well people point? ----
CONDS = ["off", "damped", "cnn", "steady"]     # steady = lesson 29's CNN trained not to shake
with_lead = [(n, t, m) for n, t, m in games if m.get("lead") in CONDS]
if with_lead:
    print("\nBY CURSOR LEAD (games played with the lead setting; STUDY mode rotates these):")
    for c in CONDS:
        g = fitts.summary([x for _, t, m in with_lead if m["lead"] == c for x in t])
        b = fitts.binned([x for _, t, m in with_lead if m["lead"] == c for x in t])
        if g:
            extra = f"   MT = {b['a']:.2f} + {b['b']:.2f} x ID (R² {b['r2']:.2f})" if b else ""
            n_games = sum(1 for _, _, m in with_lead if m["lead"] == c)
            print(f"   {c:7s} {g['throughput']:5.2f} bits/s   ({n_games} games, {g['n']} targets){extra}")
    # Fair comparison: each player against THEMSELVES (some people are just faster; that shouldn't count)
    print("\n   per player (each person compared with themselves):")
    for p in sorted({m.get("player") for _, _, m in with_lead}):
        row = []
        for c in CONDS:
            g = fitts.summary([x for _, t, m in with_lead if m.get("player") == p and m["lead"] == c for x in t])
            row.append(f"{c} {g['throughput']:.2f}" if g else f"{c}   -  ")
        print(f"   {str(p):12s} " + "   ".join(row))
    print("   (A real difference should show up for MOST players, not just on average.)")

    # Hard targets only: tiny/far ones, where LANDING is most of the time (the CNN's weak spot so far)
    print("\n   HARD TARGETS ONLY (ID 4.5 bits or more, from any game; TARGETS: HARD makes lots of these):")
    for c in CONDS:
        tr = [x for _, t, m in with_lead if m["lead"] == c for x in t if x[0] >= 4.5]
        if len(tr) >= 5:
            mt = np.mean([x[1] for x in tr])
            tp = np.mean([x[0] / x[1] for x in tr])
            print(f"   {c:7s} {tp:5.2f} bits/s   average {mt:.2f} s per target   ({len(tr)} targets)")
        else:
            print(f"   {c:7s}   -    (only {len(tr)} hard targets so far)")

    # Jitter: how shaky the cursor is while settling onto a target (the last 0.6 s before it closed)
    jit = {n: fitts.jitter_from_recording(f) for f in files for n in [Path(f).stem]}
    print("\n   SETTLE JITTER (how shaky the cursor is in the last 0.6 s before each target closes; lower = steadier):")
    for c in CONDS:
        vals = [v for n, _, m in with_lead if m["lead"] == c for v in jit.get(n, [])]
        if vals:
            print(f"   {c:7s} {np.mean(vals):5.2f} px per frame   ({len(vals)} targets)")
    print("   per player:")
    for p in sorted({m.get("player") for _, _, m in with_lead}):
        row = []
        for c in CONDS:
            vals = [v for n, _, m in with_lead if m.get("player") == p and m["lead"] == c for v in jit.get(n, [])]
            row.append(f"{c} {np.mean(vals):.2f}" if vals else f"{c}   -  ")
        print(f"   {str(p):12s} " + "   ".join(row))
    print("   Throughput = how FAST you point. Jitter = how STEADY it feels. A lead can win one and lose the other.")

    # ROUND 2: only the games since STEADY joined the rotation, so all three were played at the same skill level
    steady_names = [n for n, _, m in with_lead if m["lead"] == "steady"]
    if steady_names:
        first = min(steady_names)                                   # file names start with the date and time
        r2 = [(n, t, m) for n, t, m in with_lead if n >= first]
        print(f"\nROUND 2 (the {len(r2)} games since STEADY joined; fair, because they're mixed in time):")
        print(f"   {'':8s} {'bits/s':>7s} {'hard bits/s':>12s} {'settle jitter':>14s}   games")
        for c in ["off", "cnn", "steady"]:
            tr = [x for n, t, m in r2 if m["lead"] == c for x in t]
            g = fitts.summary(tr)
            hard = [x[0] / x[1] for x in tr if x[0] >= 4.5]
            vals = [v for n, _, m in r2 if m["lead"] == c for v in jit.get(n, [])]
            if g:
                print(f"   {c:8s} {g['throughput']:7.2f} {np.mean(hard) if len(hard) >= 5 else float('nan'):12.2f} "
                      f"{np.mean(vals) if vals else float('nan'):14.2f}   {sum(1 for _, _, m in r2 if m['lead'] == c)}")
        print("   The hope: STEADY keeps CNN's speed (bits/s) with less jitter, closer to OFF's.")
else:
    print("\nNo games with a cursor lead yet: set LEAD in the game menu (STUDY is best) and play some rounds.")

# ---- can you FEEL the difference? (guesses made on the results screen of study games) ----
board = json.loads(Path("runs/leaderboard.json").read_text()) if Path("runs/leaderboard.json").exists() else []
guesses = [r for r in board if r.get("blind") and r.get("guess") in CONDS]
if guesses:
    right = sum(r["guess"] == r["lead"] for r in guesses)
    print(f"\nCAN YOU FEEL IT? {right} of {len(guesses)} guesses right ({right / len(guesses) * 100:.0f}%; "
          f"pure luck would be about 33%)")
    print("   rows = the real lead, columns = what was guessed")
    print("            " + "".join(f"{c:>8s}" for c in CONDS))
    for real in CONDS:
        print(f"   {real:8s} " + "".join(f"{sum(1 for r in guesses if r['lead'] == real and r['guess'] == g):8d}"
                                      for g in CONDS))
    print("   (Lots on the diagonal = you can tell them apart. CNN and DAMPED mixed up but OFF spotted = lead is"
          " noticeable, the kind of lead less so.)")

players = sorted({m.get("player") for _, _, m in games if m.get("player")})
if len(players) > 1:
    print("\nBY PLAYER:")
    for p in players:
        g = fitts.summary([x for _, t, m in games if m.get("player") == p for x in t])
        if g:
            print(f"   {p:12s} {g['throughput']:.2f} bits/s ({g['n']} targets)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = np.array([x[0] for x in every])
    mts = np.array([x[1] for x in every])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ids, mts, s=8, alpha=0.2, color="tab:blue", label="one target")
    xs = np.linspace(ids.min(), ids.max(), 50)
    ax.plot(xs, s["a"] + s["b"] * xs, color="tab:blue", lw=1, ls="--", alpha=0.6,
            label=f"fit through every target   (R² = {s['r2']:.2f})")
    if bn:
        bx = [b[0] for b in bn["bins"]]
        by = [b[1] for b in bn["bins"]]
        ax.scatter(bx, by, s=70, color="tab:red", zorder=3, label="group averages (every 0.5 bits)")
        ax.plot(xs, bn["a"] + bn["b"] * xs, color="tab:red", lw=2,
                label=f"MT = {bn['a']:.2f} + {bn['b']:.2f} ID   (R² = {bn['r2']:.2f})")
    ax.set_xlabel("index of difficulty, ID = log2(D/W + 1)  (bits)")
    ax.set_ylabel("movement time, MT (s)")
    ax.set_title(f"Fitts' law for your hand cursor: {s['throughput']:.2f} bits/s")
    ax.set_ylim(0, np.percentile(mts, 98) * 1.1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("fitts.png", dpi=120)
    print("\nSaved fitts.png  (open it:  open fitts.png)")
except ImportError:
    print("\n(pip install matplotlib for the plot)")

# ---- the same, split by cursor lead (fitts_by_lead.png) ----
try:
    if with_lead:
        COLORS = {"off": "tab:gray", "damped": "tab:orange", "cnn": "tab:blue", "steady": "tab:green"}
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 5))
        # 1. Fitts' line per lead: lower line = faster at every difficulty
        for c in CONDS:
            tr = [x for _, t, m in with_lead if m["lead"] == c for x in t]
            b = fitts.binned(tr)
            g = fitts.summary(tr)
            if b and g:
                a1.scatter([q[0] for q in b["bins"]], [q[1] for q in b["bins"]], s=50, color=COLORS[c], zorder=3)
                a1.plot(xs, b["a"] + b["b"] * xs, color=COLORS[c], lw=2,
                        label=f"{c.upper():6s} {g['throughput']:.2f} bits/s  ({len(tr)} targets)")
        a1.set_xlabel("difficulty ID (bits)")
        a1.set_ylabel("movement time (s)")
        a1.set_title("Fitts' law by lead (lower line = faster)")
        a1.grid(alpha=0.3)
        if a1.get_legend_handles_labels()[0]:
            a1.legend()
        # 2. every game in the order you played it, colored by lead: practice vs lead
        names = [m.get("player") for _, _, m in with_lead]
        me = max(set(names), key=names.count)          # the player with the most lead games (you)
        order = [(m.get("player"), n, t, m) for n, t, m in with_lead
                 if m.get("player") == me and m.get("targets", "normal") == "normal"]   # hard games score differently
        for i, (_, n, t, m) in enumerate(order):
            g = fitts.summary(t)
            if g:
                a2.scatter(i + 1, g["throughput"], s=60, color=COLORS[m["lead"]], zorder=3)
        tps = [fitts.summary(t)["throughput"] if fitts.summary(t) else np.nan for _, _, t, _ in order]
        if len(tps) >= 3:
            k = np.arange(1, len(tps) + 1)
            ok = ~np.isnan(tps)
            slope, icpt = np.polyfit(k[ok], np.array(tps)[ok], 1)
            a2.plot(k, icpt + slope * k, color="black", lw=1, ls="--", label=f"practice trend ({slope:+.3f} per game)")
            # Take out the practice trend: is each lead ABOVE or BELOW where practice alone says you'd be?
            print(f"\nAFTER REMOVING PRACTICE ({me}: getting {slope:+.3f} bits/s better per game on average):")
            for c in CONDS:
                res = [tps[i] - (icpt + slope * (i + 1)) for i, o in enumerate(order) if o[3]["lead"] == c and ok[i]]
                if res:
                    print(f"   {c:7s} {np.mean(res):+.2f} bits/s vs the trend   ({len(res)} games)")
            print("   (+ = better than practice alone predicts. This is the fair answer to 'I PB every time it's CNN'.)")
        for c in CONDS:
            a2.scatter([], [], color=COLORS[c], label=c.upper())
        a2.xaxis.get_major_locator().set_params(integer=True)
        a2.set_xlabel("game number (in the order played)")
        a2.set_ylabel("throughput (bits/s)")
        a2.set_title(f"{me}'s games: lead or just practice?")
        a2.grid(alpha=0.3)
        a2.legend()
        # 3. settle jitter per lead, with each game as a dot
        for j, c in enumerate(CONDS):
            per_game = [np.mean(jit[n]) for n, _, m in with_lead if m["lead"] == c and jit.get(n)]
            if per_game:
                a3.bar(j, np.mean(per_game), color=COLORS[c], alpha=0.5)
                a3.scatter(np.full(len(per_game), j) + np.random.default_rng(j).uniform(-0.15, 0.15, len(per_game)),
                           per_game, color=COLORS[c], s=30, zorder=3)
        a3.set_xticks(range(len(CONDS)), [c.upper() for c in CONDS])
        a3.set_ylabel("settle jitter (px per frame, lower = steadier)")
        a3.set_title("How shaky while settling on a target")
        a3.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig("fitts_by_lead.png", dpi=120)
        print("Saved fitts_by_lead.png  (open it:  open fitts_by_lead.png)")
except NameError:
    pass

# ---- round 2 picture (fitts_round2.png): off vs cnn vs steady, only games since STEADY joined ----
try:
    if with_lead and steady_names:
        R2 = ["off", "cnn", "steady"]
        COL2 = {"off": "tab:gray", "cnn": "tab:blue", "steady": "tab:green"}
        fig, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4.5))
        for j, c in enumerate(R2):
            games_c = [(n, t) for n, t, m in r2 if m["lead"] == c]
            tps = [fitts.summary(t)["throughput"] for _, t in games_c if fitts.summary(t)]
            jits = [np.mean(jit[n]) for n, _ in games_c if jit.get(n)]
            spread = np.random.default_rng(j)
            if tps:
                b1.bar(j, np.mean(tps), color=COL2[c], alpha=0.5)
                b1.scatter(j + spread.uniform(-0.15, 0.15, len(tps)), tps, color=COL2[c], s=40, zorder=3)
                b1.text(j, np.mean(tps), f"{np.mean(tps):.2f}", ha="center", va="bottom", fontsize=11)
            if jits:
                b2.bar(j, np.mean(jits), color=COL2[c], alpha=0.5)
                b2.scatter(j + spread.uniform(-0.15, 0.15, len(jits)), jits, color=COL2[c], s=40, zorder=3)
                b2.text(j, np.mean(jits), f"{np.mean(jits):.1f}", ha="center", va="bottom", fontsize=11)
        for ax, title, ylab in ((b1, "Speed (higher = better)", "throughput (bits/s)"),
                                (b2, "Settle jitter (lower = steadier)", "px per frame")):
            ax.set_xticks(range(3), [c.upper() for c in R2])
            ax.set_title(title)
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3, axis="y")
        fig.suptitle(f"Round 2: does the jitter-trained CNN keep the speed?  ({len(r2)} games, each dot = one game)")
        fig.tight_layout()
        fig.savefig("fitts_round2.png", dpi=120)
        print("Saved fitts_round2.png  (open it:  open fitts_round2.png)")
except NameError:
    pass
