"""The window-closing game. It RECORDS training data and MEASURES how fast you are.

    python game.py                 # play with plain tracking (records data)
    python game.py --model         # play with the trained model predicting your finger
    python game.py --seed 1        # same windows in the same order (fair before/after comparison)
    python game.py --raw           # no smoothing on the dot (see exactly what MediaPipe sees)
    python game.py --no-precise    # no precision pointer (the dot is always 1:1 with your finger)
    python game.py --margin 0.15   # how much of the picture maps to the window (FIVES CURSOR_MARGIN)
    python game.py --no-glide      # the dot jumps once per camera frame instead of sliding

It opens on the arcade menu (arcade.py): sign in or make a player, then press START. Every game has a score,
a timer, and goes on the leaderboard with your attempt number.

How to play: point with your index finger. Little windows pop up; hover the dot
over one until its ring fills to close it (drifting off drains the ring slowly). Small windows test precision, far ones
test speed. Every game saves your hand movement to data/ (training data) and your
average time to runs/scores.csv (the benchmark).

Keys in the game window: q = quit early (still saves your hand data, but isn't put on the leaderboard).

Each recording (data/session_*.npz) holds:
  t, pts           camera time and the 21 raw hand points of every frame (NaN = no hand)
  cursor           where the dot was (camera 0..1, after smoothing / precision), NaN = no hand
  target_rect      every window as x, y, w, h in game-window pixels
  target_cam       the same windows in camera units (x0, y0, x1, y1), so they line up with pts
  target_spawn_t   camera time each window appeared
  target_close_t   camera time it was closed (NaN = never closed)
  meta             game settings (window size, margin, dwell, ...) and the player, as JSON text

The dot you SEE is smoothed (One Euro filter, like FIVES) and ignores one-frame glitches.
What gets RECORDED is always the raw MediaPipe points, so your training data stays honest.
"""

import argparse
import csv
import json
import math
import random
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np

import arcade as A
import common as C
import fitts
import features as F

W, H = 960, 720
WIN = "crosshair"
TOP = 80              # the score bar covers the top, so windows start below it
COUNTDOWN = 3         # seconds of 3-2-1 before the timer starts
DWELL = 0.30          # seconds to hover a window to close it
DRAIN = 0.60          # off the target, the lock-on ring drains over this long (instead of resetting at once)
STICKY = 8            # once you're locking on, the target counts as this many pixels bigger on every side
# OpenCV colors are (blue, green, red): the same tactical look as the menu.
# AMBER is the theme's accent color; play() sets it from the menu's theme at the start of every game.
BG = (12, 9, 7)
AMBER = (64, 64, 255)
WHITE = (234, 225, 216)
DIM = (90, 75, 62)
PANEL = (21, 16, 12)
BLUE = AMBER          # (old name, still used by the model's faint dot)


class OneEuro:
    """Smart smoothing (same as FIVES): heavy when the point is nearly still, light when it moves fast.
    min_cutoff: steadiness when still (lower = steadier). beta: how fast it loosens up (higher = less lag)."""

    def __init__(self, min_cutoff=1.5, beta=8.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, p, t):
        x, y = float(p[0]), float(p[1])
        if self.x is None:
            self.x, self.y, self.t = x, y, t
            return np.array([x, y], np.float32)
        dt = max(t - self.t, 1e-3)
        self.t = t
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx += a_d * ((x - self.x) / dt - self.dx)
        self.dy += a_d * ((y - self.y) / dt - self.dy)
        a = self._alpha(self.min_cutoff + self.beta * math.hypot(self.dx, self.dy), dt)
        self.x += a * (x - self.x)
        self.y += a * (y - self.y)
        return np.array([self.x, self.y], np.float32)

    def reset(self):
        self.x = self.y = self.t = None
        self.dx = self.dy = 0.0


class PrismPointer:
    """Precise when slow, pinned when fast (the "PRISM" technique from VR research).

      slow, careful hand -> the cursor moves only a fraction as far (min_scale..1): precise aiming
      fast hand          -> the cursor moves 1:1 AND catches back up to your fingertip,
                            so any gap built up while aiming slowly disappears
      still hand         -> nothing moves (no slow drift behind your back)

    Works on camera positions (0..1). update() returns the offset (cursor - fingertip); add it to
    every hand point and the rest of FIVES works unchanged. The offset is usually 0, so the cursor
    is right under your finger; it only opens up a little while you aim slowly.
    """

    def __init__(self, precise_speed=0.35, min_scale=0.35, catchup_speed=0.9, lo=0.0, hi=1.0):
        self.precise_speed = precise_speed      # below this hand speed (picture widths / s): slowed down
        self.min_scale = min_scale              # the slowest the cursor goes, as a fraction of your hand
        self.catchup_speed = catchup_speed      # at this speed the gap closes fastest
        self.lo, self.hi = lo, hi
        self.off = (0.0, 0.0)
        self.prev = None
        self.t = None
        self.last_scale = 1.0

    def scale(self, speed: float) -> float:
        """How far the cursor moves for each bit your hand moves, at this hand speed."""
        return min(1.0, max(self.min_scale, speed / max(1e-6, self.precise_speed)))

    def catchup(self, speed: float) -> float:
        """What fraction of the gap to close this frame (0 when slow, up to 0.5 when fast)."""
        span = max(1e-6, self.catchup_speed - self.precise_speed)
        return 0.5 * min(1.0, max(0.0, (speed - self.precise_speed) / span))

    def update(self, hand, t, frozen=False):
        """hand = smoothed fingertip (0..1). frozen = keep the cursor still (open palm). Returns the offset."""
        if self.prev is not None:
            dt = max(t - self.t, 1e-3)
            dx, dy = hand[0] - self.prev[0], hand[1] - self.prev[1]
            if frozen:                          # cursor holds still while the hand moves: absorb it
                ox, oy = self.off[0] - dx, self.off[1] - dy
            else:
                speed = math.hypot(dx, dy) / dt
                k = self.last_scale = self.scale(speed)
                ox, oy = self.off[0] + (k - 1) * dx, self.off[1] + (k - 1) * dy
                c = self.catchup(speed)
                ox, oy = ox * (1 - c), oy * (1 - c)
            # keep the cursor inside the usable area (pushing past the edge just slides the gap)
            cx = min(self.hi, max(self.lo, hand[0] + ox))
            cy = min(self.hi, max(self.lo, hand[1] + oy))
            self.off = (cx - hand[0], cy - hand[1])
        self.prev, self.t = hand, t
        return self.off

    def hand_gone(self):
        """The hand left the picture: when it comes back the cursor is under your finger again."""
        self.prev = None
        self.off = (0.0, 0.0)


GLITCH_JUMP = 0.2     # a fingertip "teleporting" more than 20% of the picture in one frame is a tracking glitch


def new_target(rng, last, hard=False):
    """A random window. One in three is small (precision), and it's never right next to the last one.
    hard=True (TARGETS: HARD in the menu): every target is tiny and far away, so most are 4-5.5 bits of
    difficulty, where landing (the careful settle) takes most of the time."""
    if hard:
        side = rng.choice((16, 20, 26))
        w, h, gap = side, side, 380
    else:
        small = rng.random() < 1 / 3
        w, h = (34, 26) if small else (90, 66)
        gap = 180
    for _ in range(1000):
        x, y = rng.randint(40, W - 40 - w), rng.randint(TOP + 20, H - 40 - h)
        if last is None or np.hypot(x - last[0], y - last[1]) > gap:
            return (x, y, w, h)
    return (x, y, w, h)


def make_backdrop():
    """The game's background: the menu's grid and scanlines (drawn once, copied every frame)."""
    return cv2.cvtColor(np.asarray(A.backdrop(W, H)), cv2.COLOR_RGB2BGR)


def draw_target(img, rect, fill, number, color=None):
    """A target: corner brackets that close in while you lock on, a lock-on ring, and a label.
    color: used to draw the faint NEXT preview (no label)."""
    x, y, w, h = rect
    cx, cy = x + w // 2, y + h // 2
    aa = cv2.LINE_AA
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    faint = color is not None
    color = color or (AMBER if fill > 0 else WHITE)
    o = int(10 * (1 - fill))                            # brackets start a bit outside and close in
    L = max(7, min(w, h) // 3)
    for bx, by, dx, dy in ((x - o, y - o, 1, 1), (x + w + o, y - o, -1, 1),
                           (x - o, y + h + o, 1, -1), (x + w + o, y + h + o, -1, -1)):
        cv2.line(img, (bx, by), (bx + dx * L, by), color, 2, aa)
        cv2.line(img, (bx, by), (bx, by + dy * L), color, 2, aa)
    s = 4 if w < 60 else 6                              # small diamond in the middle
    cv2.polylines(img, [np.array([(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)])], True, color, 1, aa)
    if fill > 0:
        r = max(w, h) // 2 + 14
        cv2.ellipse(img, (cx, cy), (r, r), -90, 0, int(360 * fill), AMBER, 2, aa)
    if faint:                                           # the NEXT preview: no label
        return
    tag = f"T-{number:02d}" + ("  x1.5" if w < 60 else "")
    cv2.putText(img, tag, (x - o, y - o - 6), cv2.FONT_HERSHEY_PLAIN, 0.9, DIM if fill == 0 else AMBER, 1, aa)


def draw_preview(img, rect, number):
    """The NEXT target: the same target, drawn in a faint version of the theme color, so you can plan."""
    faint = tuple(int(a * 0.38 + b * 0.62) for a, b in zip(AMBER, BG))
    draw_target(img, rect, 0.0, number, color=faint)


def draw_reticle(img, p, hot):
    """Your cursor: a scope reticle. Amber when it's on a target."""
    color = AMBER if hot else WHITE
    aa = cv2.LINE_AA
    cv2.circle(img, p, 15, color, 1, aa)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(img, (p[0] + dx * 6, p[1] + dy * 6), (p[0] + dx * 24, p[1] + dy * 24), color, 1, aa)
    cv2.circle(img, p, 2, color, -1, aa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true", help="steer with the trained model's prediction")
    ap.add_argument("--windows", type=int, default=40, help="windows per game")
    ap.add_argument("--seed", type=int, default=None, help="same number = same windows, for fair comparisons")
    ap.add_argument("--raw", action="store_true", help="no smoothing / glitch guard on the dot you see")
    ap.add_argument("--cutoff", type=float, default=1.5, help="One Euro min_cutoff (FIVES uses 1.5)")
    ap.add_argument("--beta", type=float, default=8.0, help="One Euro beta (FIVES uses 8.0)")
    ap.add_argument("--no-precise", action="store_true", help="turn the precision pointer off")
    ap.add_argument("--min-scale", type=float, default=0.35, help="precision: slowest cursor speed (FIVES PRECISE_MIN_SCALE)")
    ap.add_argument("--precise-speed", type=float, default=0.35, help="precision: below this hand speed it slows (FIVES PRECISE_SPEED)")
    ap.add_argument("--margin", type=float, default=0.22, help="edge margin of the picture (FIVES CURSOR_MARGIN)")
    ap.add_argument("--no-glide", action="store_true", help="no sliding between camera frames")
    args = ap.parse_args()

    predictor = None
    if args.model:
        import torch
        from model import load
        path = C.RUNS_DIR / "cursor_predictor.pt"
        if not path.exists():
            raise SystemExit("No trained model yet. Run  python train.py  first.")
        predictor, info = load(path)
        torch.set_num_threads(1)
        print(f"Using {path.name} (test error {info['test_error']:.4f})")

    C.MARGIN = args.margin                      # same mapping as FIVES' CURSOR_MARGIN
    landmarker = C.make_landmarker()
    cam = C.Camera()
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    players = A.Players()
    menu = A.Menu(WIN, W, H, players, args.windows)
    clock = {"last_ms": -1, "seq": 0}           # MediaPipe needs timestamps that only go up, across games
    last_run = None
    try:
        while True:
            action, player = menu.run(last_run)
            if action != "play":
                break
            attempt = players.next_attempt(player)
            lead = A.lead_for(player, attempt)
            game = play(args, cam, landmarker, predictor, clock, players.name(player), attempt,
                        lead=lead, blind=A.lead_mode == "study")
            last_run = finish(args, game, predictor, player, players.name(player), attempt)
            if game["window_closed"]:
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()
        landmarker.close()


def play(args, cam, landmarker, predictor, clock, name, attempt, lead="off", blind=False):
    """One game. Returns everything recorded plus the score. lead = "off", "damped", "cnn" or "steady" (see leads.py)."""
    global AMBER, BLUE
    AMBER = BLUE = tuple(int(v) for v in A.ACCENT[::-1])       # the theme color, RGB -> BGR
    if predictor is not None:
        import torch
    smoother = None if args.raw else OneEuro(args.cutoff, args.beta)
    accel = None if args.no_precise else PrismPointer(args.precise_speed, args.min_scale, 0.9, C.MARGIN, 1 - C.MARGIN)
    last_raw = None                             # last good raw fingertip, for spotting glitches
    glitches = held = 0
    rng = random.Random(args.seed)
    times, frames, cursors = [], [], []         # the recording (hand points + where the dot was)
    targets = []                                # every window: rect, when it appeared, when it closed
    shown = None                                # where the dot was last drawn (window pixels), for gliding
    GLIDE_STEPS = 3                             # in-between pictures per camera frame
    recent = deque(maxlen=F.HISTORY)            # (t, signal) for the model
    target, last_target = None, None
    spawned = None
    lock = 0.0                                  # lock-on progress for the current target, 0..1
    last_frame = time.monotonic()
    preview = A.preview                         # show the NEXT target faintly, so you can plan your move
    hard = A.targets == "hard"                  # tiny, far targets (menu: TARGETS)
    leader = None                               # the cursor lead for this game (FIVES' damped speed or CNN)
    if lead != "off":
        import leads
        try:
            leader = (leads.LeadPredictor(0.04, 0.04) if lead == "damped"
                      else leads.CNNLead(leads.STEADY_PATH if lead == "steady" else leads.CNN_PATH,
                                         **leads.CNN_SETTINGS))
        except Exception as err:                # e.g. no runs/deploy_cnn_1.pt yet: play without a lead
            print(f"Lead '{lead}' not available ({err}); playing with no lead.")
            lead = "off"
    upcoming = None                             # that next target
    close_times = []
    score, popups = 0, []                       # popups: (x, y, points, when) for the "+667" that floats up
    hand_since = None                           # when the hand showed up (starts the countdown)
    started = finished_at = None                # timer: from the end of the countdown to the last window
    quit_early = window_closed = False
    backdrop = make_backdrop()
    print(f"{name}, attempt #{attempt}: point at the windows to close them. q quits.")

    while len(close_times) < args.windows:
        frame, stamp, clock["seq"] = cam.read(clock["seq"])
        if frame is None:
            continue
        frame = cv2.flip(frame, 1)              # mirror, so moving right moves the dot right
        ms = max(int(stamp * 1000), clock["last_ms"] + 1)
        clock["last_ms"] = ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(image, ms)

        canvas = backdrop.copy()
        pointer = None
        if result.hand_landmarks:
            pts = np.array([(lm.x, lm.y) for lm in result.hand_landmarks[0]], np.float32)
            times.append(stamp)
            frames.append(pts)
            sig = F.signal(pts[None])[0]
            if recent and stamp - recent[-1][0] > F.MAX_GAP:
                recent.clear()
            recent.append((stamp, sig))
            tip = pts[C.INDEX_TIP]
            if smoother is not None:            # glitch guard: only for the dot you see
                jumped = last_raw is not None and np.abs(tip - last_raw).max() > GLITCH_JUMP
                if jumped and held < 2:         # a teleport: hold the dot still for up to 2 frames
                    glitches += 1
                    held += 1
                    tip = last_raw
                else:                           # normal (or it stayed there: it was real, follow it)
                    last_raw, held = tip, 0
            aim = tip                           # where the dot should go before smoothing
            if predictor is not None:
                cv2.circle(canvas, C.to_window(tip, W, H), 5, DIM, -1)     # faint: raw finger
                if len(recent) == F.HISTORY:
                    wt = np.array([r[0] for r in recent])
                    ws = np.array([r[1] for r in recent])
                    x = torch.from_numpy(F.window_features(wt, ws)[None])
                    with torch.no_grad():
                        aim = tip + predictor(x)[0].numpy()
            pointer = aim if smoother is None else smoother.update(aim, stamp)
            if accel is not None:               # slow hand = precise, fast hand = pinned (like FIVES)
                off = accel.update(pointer, stamp)
                pointer = np.array([pointer[0] + off[0], pointer[1] + off[1]], np.float32)
            if leader is not None:              # the lead: draw the cursor a little ahead (like FIVES)
                if lead == "damped":
                    off = leader.update(pointer, stamp)
                else:
                    off = leader.update([(float(p[0]), float(p[1])) for p in pts], stamp)
                pointer = np.array([pointer[0] + off[0], pointer[1] + off[1]], np.float32)
        else:
            times.append(stamp)
            frames.append(np.full((21, 2), np.nan, np.float32))
            recent.clear()
            if smoother is not None:
                smoother.reset()                # hand lost: start fresh when it comes back
            last_raw, held = None, 0
            if accel is not None:
                accel.hand_gone()               # bring your hand back: the dot is under your finger again
            if leader is not None:
                leader.hand_gone()

        cursors.append(np.array(pointer, np.float32) if pointer is not None else np.full(2, np.nan, np.float32))
        now = time.monotonic()

        # Before the timer: wait for a hand, then count down 3-2-1.
        if started is None:
            if pointer is not None and hand_since is None:
                hand_since = now
            if hand_since is not None and now - hand_since >= COUNTDOWN:
                started = now
        if started is not None and target is None:
            target, spawned, lock = upcoming or new_target(rng, last_target, hard), now, 0.0
            upcoming = new_target(rng, target, hard) if preview and len(close_times) + 1 < args.windows else None
            targets.append({"rect": target, "spawn_t": stamp, "close_t": float("nan")})
        dt = min(now - last_frame, 0.1)                 # time since the last frame (capped, in case of a hiccup)
        last_frame = now
        on_target = False
        if pointer is not None:
            px, py = C.to_window(pointer, W, H)
        if target is not None:
            x, y, w, h = target
            grow = STICKY if lock > 0 else 0            # sticky edges: harder to leave than to enter
            inside = (pointer is not None and x - grow <= px <= x + w + grow and y - grow <= py <= y + h + grow)
            if inside:
                on_target = True
                lock += dt / DWELL
                if lock >= 1:
                    took = now - spawned
                    close_times.append(took)
                    targets[-1]["close_t"] = stamp
                    pts_won = A.window_points(took, small=w < 60)
                    score += pts_won
                    popups.append((x + w // 2, y + h // 2, pts_won, now))
                    last_target, target = target, None
                    lock = 0.0
            else:
                lock = max(0.0, lock - dt / DRAIN)      # drifted off: drain slowly instead of resetting
        if upcoming is not None and target is not None:
            draw_preview(canvas, upcoming, len(close_times) + 2)
        if target is not None:
            draw_target(canvas, target, min(lock, 1.0), len(close_times) + 1)

        elapsed = 0.0 if started is None else now - started
        A.draw_hud(canvas, name if blind else f"{name} · {lead}", attempt, score, elapsed, len(close_times), args.windows, popups, now)
        popups = [p for p in popups if now - p[3] < 0.7]
        if started is None:
            if hand_since is None:
                A.draw_big_message(canvas, "ACQUIRING HAND", "hold your index finger up in front of the camera")
            else:
                A.draw_big_message(canvas, str(COUNTDOWN - int(now - hand_since)), "HAND LOCKED · GET READY")
        if pointer is None and started is not None:
            cv2.putText(canvas, "SIGNAL LOST", (W // 2 - 70, H - 30), cv2.FONT_HERSHEY_PLAIN, 1.4, AMBER, 1, cv2.LINE_AA)

        # Glide (like FIVES): slide the dot to the new spot in a few small steps instead of one jump.
        # Hovering/closing still uses the real spot, decided above.
        if pointer is None:
            shown, steps = None, [None]
        elif shown is None or args.no_glide:
            steps = [(px, py)]
        else:
            steps = [(int(shown[0] + (px - shown[0]) * k / GLIDE_STEPS),
                      int(shown[1] + (py - shown[1]) * k / GLIDE_STEPS)) for k in range(1, GLIDE_STEPS + 1)]
        for i, spot in enumerate(steps):
            img = canvas.copy() if len(steps) > 1 else canvas
            if spot is not None:
                draw_reticle(img, spot, on_target)
            cv2.imshow(WIN, A.glow(img))
            if cv2.waitKey(1 if i == len(steps) - 1 else 6) & 0xFF == ord("q"):
                quit_early = True
                break
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            quit_early = window_closed = True
        if pointer is not None:
            shown = (px, py)
        if quit_early:
            break
    finished_at = time.monotonic()

    return {"times": times, "frames": frames, "cursors": cursors, "targets": targets, "glitches": glitches,
            "close_times": close_times, "score": score, "accel": accel is not None, "preview": preview, "hard": hard,
            "lead": lead, "blind": blind,
            "time": 0.0 if started is None else finished_at - started,
            "finished": len(close_times) >= args.windows, "window_closed": window_closed}


def finish(args, game, predictor, player, name, attempt):
    """Save the recording, the benchmark line, and the leaderboard entry. Returns the leaderboard entry."""
    times, targets, close_times = game["times"], game["targets"], game["close_times"]
    mode = ("model" if predictor is not None else "plain") + ("_raw" if args.raw else "") + ("_precise" if game["accel"] else "")
    if game["glitches"]:
        print(f"Ignored {game['glitches']} glitchy frames on screen (they're still in the recording, raw)")
    stamp_name = time.strftime("%Y%m%d_%H%M%S")
    if len(times) > 30:
        C.DATA_DIR.mkdir(exist_ok=True)
        out = C.DATA_DIR / f"session_{stamp_name}_{mode}.npz"
        span = 1 - 2 * C.MARGIN                 # window pixels -> camera units (the inverse of C.to_window)
        to_cam = lambda px, n: C.MARGIN + px / (n - 1) * span
        rects = np.array([tg["rect"] for tg in targets], np.float32).reshape(-1, 4)
        cam = np.array([(to_cam(x, W), to_cam(y, H), to_cam(x + w, W), to_cam(y + h, H))
                        for x, y, w, h in rects], np.float32).reshape(-1, 4)
        meta = {"window_w": W, "window_h": H, "margin": C.MARGIN, "dwell": DWELL, "precise": game["accel"],
                "min_scale": args.min_scale, "precise_speed": args.precise_speed, "raw": args.raw,
                "glide": not args.no_glide, "seed": args.seed, "model": predictor is not None,
                "player": player, "attempt": attempt, "score": game["score"], "finished": game["finished"],
                "preview": game["preview"], "drain": DRAIN, "sticky": STICKY, "lead": game["lead"],
                "targets": "hard" if game["hard"] else "normal"}
        np.savez_compressed(out, t=np.array(times), pts=np.array(game["frames"]), cursor=np.array(game["cursors"]),
                            target_rect=rects, target_cam=cam,
                            target_spawn_t=np.array([tg["spawn_t"] for tg in targets], np.float64),
                            target_close_t=np.array([tg["close_t"] for tg in targets], np.float64),
                            meta=np.array(json.dumps(meta)))
        closed = int(np.sum(~np.isnan([tg["close_t"] for tg in targets]))) if targets else 0
        print(f"Saved {len(times)} frames of hand movement and {len(targets)} windows ({closed} closed) to data/{out.name}")
    avg = float(np.mean(close_times)) if close_times else 0.0
    # Fitts' law throughput (bits per second): the standard pointing score. See fitts.py.
    tr = fitts.trials(times, game["cursors"], [tg["rect"] for tg in targets], [tg["spawn_t"] for tg in targets],
                      [tg["close_t"] for tg in targets], C.MARGIN, W, H, DWELL) if targets else []
    fit = fitts.summary(tr)
    tp = round(fit["throughput"], 2) if fit else None
    if close_times:
        C.RUNS_DIR.mkdir(exist_ok=True)
        scores = C.RUNS_DIR / "scores.csv"
        new = not scores.exists()
        with open(scores, "a", newline="") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["time", "mode", "seed", "windows", "avg_seconds"])
            wr.writerow([stamp_name, mode, args.seed, len(close_times), f"{avg:.3f}"])
    run = {"player": player, "name": name, "attempt": attempt, "score": game["score"],
           "time": round(game["time"], 2), "avg": round(avg, 3), "closed": len(close_times),
           "windows": args.windows, "finished": game["finished"], "mode": mode, "preview": game["preview"], "tp": tp,
           "targets": "hard" if game["hard"] else "normal",
           "lead": game["lead"], "blind": game["blind"], "guess": None, "revealed": False,
           "date": time.strftime("%Y-%m-%d %H:%M")}
    A.add_run(run)
    print(f"{name} attempt #{attempt}: {game['score']:,} points in {A.clock(game['time'])}"
          + (f", throughput {tp:.2f} bits/s" if tp is not None else "")
          + ("" if game["blind"] else f", lead: {game['lead']}")
          + ("" if game["finished"] else "  (quit early: not on the leaderboard)"))
    return run


if __name__ == "__main__":
    main()
