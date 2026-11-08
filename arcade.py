"""The arcade around the game: player accounts, scoring, the leaderboard, and the menu screens.

Everything is saved on this computer in runs/:
  runs/players.json       players: name, passcode (scrambled with a salt, never stored as typed), attempt count
  runs/leaderboard.json   every finished or quit game: who, which attempt, score, time

The passcodes are for friends sharing one laptop, not real security: anyone with the files can edit them.

The screens are drawn with Pillow (for real fonts), then shown in the OpenCV window. Use the mouse or
trackpad to click buttons; type with the keyboard.
"""

import hashlib
import json
import os
import secrets
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import common as C

PLAYERS_FILE = C.RUNS_DIR / "players.json"
BOARD_FILE = C.RUNS_DIR / "leaderboard.json"

# Colors (RGB): a tactical scope / HUD look. Near-black, amber accents.
BG = (7, 9, 12)
SURFACE = (12, 16, 21)
INK = (216, 225, 234)
MUTED = (111, 127, 144)
LINE = (31, 42, 54)
GRID = (16, 22, 29)
SOFT = (22, 30, 39)
ACCENT_INK = (8, 6, 6)

# The accent color is a theme you can switch in the menu (bottom right). One color at a time.
THEMES = {
    "red":   {"accent": (255, 64, 64),   "good": (230, 236, 242), "bad": (255, 196, 80)},
    "green": {"accent": (70, 236, 130),  "good": (200, 255, 222), "bad": (255, 92, 77)},
    "blue":  {"accent": (100, 200, 255), "good": (120, 236, 170), "bad": (255, 92, 77)},
}
SETTINGS_FILE = C.RUNS_DIR / "settings.json"
ACCENT = GOOD = BAD = None
theme = None
preview = True        # show the next target faintly (switch in the menu, bottom right)
# Cursor lead (switch in the menu): "off", "damped", "cnn", "steady" (the CNN trained not to shake), or "study". STUDY picks one for each game in a rotating,
# per-player order and only reveals it on the results screen: a blind comparison for Fitts' law (lesson 28).
LEAD_MODES = ["study", "off", "damped", "cnn", "steady"]
CONDITIONS = ["off", "cnn", "steady"]      # what STUDY rotates (round 2: does training out the jitter help?)
lead_mode = "study"
targets = "normal"    # "normal" or "hard" (tiny, far targets: tests landing). Switch in the menu.


def set_theme(name):
    global ACCENT, GOOD, BAD, theme
    theme = name if name in THEMES else "red"
    t = THEMES[theme]
    ACCENT, GOOD, BAD = t["accent"], t["good"], t["bad"]


def load_theme():
    global preview, lead_mode, targets
    settings = _load_early(SETTINGS_FILE)
    set_theme(settings.get("theme", "red"))
    preview = settings.get("preview", True)
    lead_mode = settings.get("lead", "study") if settings.get("lead", "study") in LEAD_MODES else "study"
    targets = "hard" if settings.get("targets") == "hard" else "normal"


def _save_settings():
    _save(SETTINGS_FILE, {"theme": theme, "preview": preview, "lead": lead_mode, "targets": targets})


def next_lead():
    global lead_mode
    lead_mode = LEAD_MODES[(LEAD_MODES.index(lead_mode) + 1) % len(LEAD_MODES)]
    _save_settings()


def lead_for(player, attempt):
    """Which lead this game uses. In STUDY mode: rotate off -> cnn -> steady, each player starting at a different
    point (so no condition always gets the 'fresh' or the 'practiced' games: counterbalancing)."""
    if lead_mode != "study":
        return lead_mode
    start = sum(map(ord, player)) % len(CONDITIONS)
    return CONDITIONS[(start + attempt) % len(CONDITIONS)]


def next_theme():
    names = list(THEMES)
    set_theme(names[(names.index(theme) + 1) % len(names)])
    _save_settings()


def toggle_targets():
    global targets
    targets = "normal" if targets == "hard" else "hard"
    _save_settings()


def toggle_preview():
    global preview
    preview = not preview
    _save_settings()


def _load_early(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


set_theme("red")

SMALL_BONUS = 1.5     # small targets are worth this much more


# ---------------------------------------------------------------- scoring
def window_points(seconds, small):
    """Points for one target. Faster = more: 0.5 s -> 667, 1 s -> 500, 2 s -> 333."""
    return int(round(1000 / (1 + seconds) * (SMALL_BONUS if small else 1)))


def clock(seconds):
    """83.4 -> '1:23.4'"""
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m)}:{s:04.1f}"


# ---------------------------------------------------------------- saving
def _load(path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save(path, data):
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)                                   # swap in one step, so a crash can't half-write it


def key_of(name):
    return "".join(c for c in name.lower() if c.isalnum())[:20]


def _scramble(code, salt):
    return hashlib.pbkdf2_hmac("sha256", code.encode(), bytes.fromhex(salt), 100_000).hex()


class Players:
    def __init__(self):
        self.data = _load(PLAYERS_FILE, {})

    def keys(self):
        return sorted(self.data, key=lambda k: self.data[k]["name"].lower())

    def name(self, key):
        return self.data[key]["name"]

    def attempts(self, key):
        return self.data[key].get("attempts", 0)

    def create(self, name, code):
        """Returns an error message, or None if it worked."""
        name = " ".join(name.split())[:16]
        key = key_of(name)
        if not key:
            return "Type a name first."
        if key in self.data:
            return f"{self.data[key]['name']} already exists. Pick it from the list."
        if len(code) < 3:
            return "Passcode needs at least 3 characters."
        salt = secrets.token_hex(8)
        self.data[key] = {"name": name, "salt": salt, "code": _scramble(code, salt), "attempts": 0,
                          "created": time.strftime("%Y-%m-%d %H:%M")}
        _save(PLAYERS_FILE, self.data)
        return None

    def check(self, key, code):
        p = self.data.get(key)
        return p is not None and secrets.compare_digest(p["code"], _scramble(code, p["salt"]))

    def next_attempt(self, key):
        self.data[key]["attempts"] = self.attempts(key) + 1
        _save(PLAYERS_FILE, self.data)
        return self.data[key]["attempts"]


def add_run(run):
    runs = _load(BOARD_FILE, [])
    runs.append(run)
    _save(BOARD_FILE, runs)


def all_runs():
    return _load(BOARD_FILE, [])


def update_run(player, attempt, **fields):
    """Change one saved game (found by player + attempt), e.g. to store a guess or whether it's revealed."""
    runs = all_runs()
    for r in runs:
        if r.get("player") == player and r.get("attempt") == attempt:
            r.update(fields)
    _save(BOARD_FILE, runs)


def ranked(runs, best_only):
    """Finished runs, highest score first. best_only = each player's best run only."""
    done = [r for r in runs if r.get("finished")]
    done.sort(key=lambda r: (-r["score"], r["time"]))
    if best_only:
        seen, out = set(), []
        for r in done:
            if r["player"] not in seen:
                seen.add(r["player"])
                out.append(r)
        return out
    return done


# ---------------------------------------------------------------- drawing
FONT_FILES = {
    "display": ["/System/Library/Fonts/Supplemental/Impact.ttf", "/Library/Fonts/Impact.ttf"],
    "body": ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"],
    "mono": ["/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf"],
}
_fonts = {}


def font(kind, size):
    if (kind, size) not in _fonts:
        f = None
        for path in FONT_FILES[kind]:
            if os.path.exists(path):
                try:
                    f = ImageFont.truetype(path, size)
                    break
                except OSError:
                    pass
        if f is None:                                   # not on a Mac: Pillow's built-in font
            try:
                f = ImageFont.load_default(size=size)
            except TypeError:
                f = ImageFont.load_default()
        _fonts[(kind, size)] = f
    return _fonts[(kind, size)]


GLOW = 0.55          # how strong the glow is (0 = none). Only bright things glow; the background stays dark.


def glow(bgr, strength=None):
    """A soft halo around bright things (a 'bloom'), done on a small copy so it's cheap."""
    strength = GLOW if strength is None else strength
    if strength <= 0:
        return bgr
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    bright = cv2.subtract(small, np.full_like(small, 70))          # keep only the bright parts
    halo = cv2.GaussianBlur(bright, (0, 0), 2.5)
    halo = cv2.resize(halo, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(bgr, 1.0, halo, strength, 0)


_backdrops = {}


def backdrop(w, h):
    """Near-black with a faint grid, scanlines, and range marks along the edges. Made once, then reused."""
    if (w, h) not in _backdrops:
        img = Image.new("RGB", (w, h), BG)
        d = ImageDraw.Draw(img)
        for x in range(0, w, 48):
            d.line((x, 0, x, h), fill=GRID)
        for y in range(0, h, 48):
            d.line((0, y, w, y), fill=GRID)
        for x in range(0, w, 12):                       # tick marks along the top and bottom edges
            d.line((x, h - (8 if x % 48 == 0 else 4), x, h), fill=LINE)
        arr = np.asarray(img).copy()
        arr[::3] = (arr[::3] * 0.82).astype(np.uint8)   # scanlines: every third row a bit darker
        _backdrops[(w, h)] = Image.fromarray(arr)
    return _backdrops[(w, h)]


def reticle_icon(d, cx, cy, r, color):
    """A small scope reticle: ring, four posts with a gap in the middle, centre dot."""
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=3)
    gap = r * 0.28
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        d.line((cx + dx * gap, cy + dy * gap, cx + dx * (r + 6), cy + dy * (r + 6)), fill=color, width=3)
    d.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=color)


class Canvas:
    """A Pillow picture plus a list of clickable buttons."""

    def __init__(self, w, h):
        self.img = backdrop(w, h).copy()
        self.d = ImageDraw.Draw(self.img)
        self.buttons = []                               # (x0, y0, x1, y1, action)

    def brackets(self, x0, y0, x1, y1, color=None, size=14, width=2):
        """Corner brackets, like a scope's target frame."""
        color = color or ACCENT
        for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            self.d.line((cx, cy, cx + dx * size, cy), fill=color, width=width)
            self.d.line((cx, cy, cx, cy + dy * size), fill=color, width=width)

    def text(self, xy, s, kind="body", size=16, color=INK, anchor="la"):
        self.d.text(xy, s, font=font(kind, size), fill=color, anchor=anchor)

    def width(self, s, kind="body", size=16):
        return self.d.textlength(s, font=font(kind, size))

    def panel(self, x0, y0, x1, y1):
        self.d.rectangle((x0, y0, x1, y1), fill=SURFACE, outline=LINE)
        self.brackets(x0, y0, x1, y1)

    def button(self, x, y, label, action, primary=False, size=16, pressed=False, pad=14):
        w = self.width(label, "body", size) + 2 * pad
        h = size + 16
        fill = ACCENT if primary else (SOFT if pressed else SURFACE)
        ink = ACCENT_INK if primary else (ACCENT if pressed else INK)
        self.d.rectangle((x, y, x + w, y + h), fill=fill, outline=ACCENT if (primary or pressed) else LINE)
        if primary:
            self.d.polygon(((x + w - 9, y + h), (x + w, y + h - 9), (x + w, y + h)), fill=ACCENT_INK)   # clipped corner
        self.text((x + w / 2, y + h / 2), label, "body", size, ink, "mm")
        self.buttons.append((x, y, x + w, y + h, action))
        return x + w

    def field(self, x, y, w, value, placeholder, focused, secret, action):
        h = 38
        self.d.rectangle((x, y, x + w, y + h), fill=BG, outline=ACCENT if focused else LINE,
                         width=2 if focused else 1)
        shown = ("*" * len(value)) if secret else value
        if shown:
            self.text((x + 10, y + h / 2), shown + ("_" if focused else ""), "mono", 15, INK, "lm")
        else:
            self.text((x + 10, y + h / 2), ("_" if focused else "") + placeholder, "mono", 14, MUTED, "lm")
        self.buttons.append((x, y, x + w, y + h, action))

    def hit(self, px, py):
        for x0, y0, x1, y1, action in reversed(self.buttons):
            if x0 <= px <= x1 and y0 <= py <= y1:
                return action
        return None

    def bgr(self):
        return glow(cv2.cvtColor(np.asarray(self.img), cv2.COLOR_RGB2BGR))


def _rgb(canvas_bgr):
    return Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))


def draw_hud(canvas_bgr, name, attempt, score, seconds, closed, total, popups, now):
    """The game's top bar (operator, score, timer, target ticks) and the hit popups."""
    h, w = canvas_bgr.shape[:2]
    bar = 70
    img = _rgb(canvas_bgr)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, w, bar), fill=BG)
    d.line((0, bar, w, bar), fill=LINE)
    d.text((20, 16), "OPERATOR", font=font("mono", 10), fill=MUTED, anchor="lm")
    d.text((20, 38), f"{name.upper()}", font=font("display", 24), fill=INK, anchor="lm")
    d.text((22 + d.textlength(name.upper(), font=font("display", 24)) + 8, 40), f"#{attempt:02d}",
           font=font("mono", 13), fill=MUTED, anchor="lm")
    d.text((w / 2, 14), "SCORE", font=font("mono", 10), fill=MUTED, anchor="mm")
    d.text((w / 2, 40), f"{score:,}", font=font("display", 36), fill=INK, anchor="mm")
    d.text((w - 20, 16), "TIME", font=font("mono", 10), fill=MUTED, anchor="rm")
    d.text((w - 20, 40), clock(seconds), font=font("mono", 24), fill=ACCENT, anchor="rm")
    if total:                                            # one tick per target, lit when cleared
        tick_w = (w - 40) / total
        for i in range(total):
            x = 20 + i * tick_w
            d.rectangle((x + 1, bar - 7, x + tick_w - 2, bar - 3), fill=ACCENT if i < closed else LINE)
    for (x, y, pts, born) in popups:
        age = now - born
        if age < 0.7:
            yy = y - 50 * age
            d.text((x, yy - 18), "HIT", font=font("mono", 12), fill=ACCENT, anchor="mm")
            d.text((x, yy + 4), f"+{pts}", font=font("display", 30), fill=GOOD, anchor="mm")
    canvas_bgr[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw_big_message(canvas_bgr, big, small=""):
    """Countdown numbers (inside a reticle) and 'acquiring hand'."""
    h, w = canvas_bgr.shape[:2]
    img = _rgb(canvas_bgr)
    d = ImageDraw.Draw(img)
    cx, cy = w / 2, h / 2 + 20
    if len(big) <= 2:
        r = 110
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=LINE, width=2)
        d.arc((cx - r, cy - r, cx + r, cy + r), -90, -90 + 120 * (4 - int(big or 0)), fill=ACCENT, width=4)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            d.line((cx + dx * (r - 14), cy + dy * (r - 14), cx + dx * (r + 18), cy + dy * (r + 18)), fill=ACCENT, width=2)
        d.text((cx, cy), big, font=font("display", 120), fill=ACCENT, anchor="mm")
    else:
        d.text((cx, cy), big, font=font("display", 56), fill=ACCENT, anchor="mm")
    if small:
        d.text((cx, cy + (150 if len(big) <= 2 else 60)), small, font=font("mono", 14), fill=MUTED, anchor="mm")
    canvas_bgr[:] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------- the menu
class Menu:
    """All the screens between games. menu.run() returns ("play", player_key) or ("quit", None)."""

    TABS = [("play", "PLAY"), ("board", "LEADERBOARD"), ("profile", "PROFILE"), ("study", "STUDY LOG")]

    def __init__(self, win, w, h, players, windows):
        self.win, self.w, self.h = win, w, h
        self.players = players
        self.windows = windows
        self.me = None
        self.tab = "play"
        self.last_run = None                # shows the results screen right after a game
        self.best_only = True
        self.target = None                  # profile picked, waiting for its passcode
        self.fields = {"code": "", "new_name": "", "new_code": ""}
        self.focus = None
        self.err = ""
        self.note = ""                      # a short message at the bottom-left (e.g. sending games)
        self.sending = False
        self.clicks = []
        load_theme()
        cv2.setMouseCallback(win, self._mouse)

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicks.append((x, y))

    # ---------- screens
    def draw(self):
        c = Canvas(self.w, self.h)
        reticle_icon(c.d, 72, 66, 26, ACCENT)
        c.text((114, 26), "CROSS", "display", 64, INK)
        c.text((114 + c.width("CROSS", "display", 64), 26), "HAIR", "display", 64, ACCENT)
        c.text((116, 104), f"// TARGET ACQUISITION TRAINER · {self.windows} TARGETS · HIGHEST SCORE WINS",
               "mono", 11, MUTED)
        c.text((40, self.h - 14), self.note.upper() if self.note else "SYS ONLINE · HAND TRACKING READY", "mono", 10,
               ACCENT if self.note else MUTED, "ls")
        x = self.w - 40
        share_item = (("SENDING...  " if self.sending else "SEND GAMES TO HOYU  >", "send"),) if self.me else ()
        for label, action in share_item + ((f"THEME: {theme.upper()}  >", "theme"),
                              (f"PREVIEW: {'ON' if preview else 'OFF'}  >", "preview"),
                              (f"LEAD: {lead_mode.upper()}  >", "lead"),
                              (f"TARGETS: {targets.upper()}  >", "targets")):
            tw = c.width(label, "mono", 10)
            c.text((x, self.h - 14), label, "mono", 10, ACCENT, "rs")
            c.buttons.append((x - tw - 4, self.h - 30, x + 4, self.h - 6, action))
            x -= tw + 24
        c.text((x, self.h - 14), time.strftime("%H:%M"), "mono", 10, MUTED, "rs")
        if self.me:
            x = self.w - 40
            bx = x - (c.width("Switch", "body", 14) + 28)
            c.button(bx, 38, "Switch", "logout", size=14)
            name = self.players.name(self.me)
            c.text((bx - 12, 52), name, "body", 16, INK, "rm")
            c.text((bx - 16 - c.width(name, "body", 16), 52), "Playing as", "body", 16, MUTED, "rm")

        x = 40
        for key, label in self.TABS:
            if key in ("profile", "study") and not self.me:
                continue
            on = self.tab == key or (key == "play" and self.tab == "results")
            shown = f"[ {label} ]" if on else f"  {label}  "
            tw = c.width(shown, "mono", 15)
            c.text((x, 150), shown, "mono", 15, ACCENT if on else MUTED)
            if on:
                c.d.rectangle((x, 172, x + tw, 175), fill=ACCENT)
            c.buttons.append((x - 4, 140, x + tw + 4, 176, "tab:" + key))
            x += tw + 18
        c.d.line((40, 176, self.w - 40, 176), fill=LINE, width=2)

        {"play": self.draw_play, "board": self.draw_board, "profile": self.draw_profile,
         "results": self.draw_results, "study": self.draw_study}[self.tab](c)
        return c

    def draw_play(self, c):
        x0, y0, x1 = 40, 200, self.w - 40
        if not self.me:
            c.panel(x0, y0, x1, self.h - 40)
            c.text((x0 + 24, y0 + 22), "WHO'S PLAYING?", "display", 32, INK)
            keys = self.players.keys()
            c.text((x0 + 24, y0 + 66), "Click your name, then type your passcode." if keys
                   else "No players yet. Make the first one below.", "body", 15, MUTED)
            bx, by = x0 + 24, y0 + 96
            for k in keys:
                label = self.players.name(k)
                if bx + c.width(label, "body", 15) + 40 > x1 - 24:
                    bx, by = x0 + 24, by + 44
                bx = c.button(bx, by, label, "pick:" + k, size=15, pressed=(k == self.target)) + 8
            y = by + (56 if keys else 0)
            if self.target:
                c.field(x0 + 24, y, 300, self.fields["code"], f"Passcode for {self.players.name(self.target)}",
                        self.focus == "code", True, "focus:code")
                c.button(x0 + 336, y + 2, "Sign in", "login", primary=True)
                y += 56
            c.d.line((x0 + 24, y, x1 - 24, y), fill=LINE)
            y += 18
            c.text((x0 + 24, y), "NEW PLAYER", "display", 24, INK)
            y += 40
            c.field(x0 + 24, y, 250, self.fields["new_name"], "Name", self.focus == "new_name", False, "focus:new_name")
            c.field(x0 + 286, y, 250, self.fields["new_code"], "Passcode", self.focus == "new_code", True,
                    "focus:new_code")
            c.button(x0 + 548, y + 2, "Create player", "create", primary=True)
            c.text((x0 + 24, y + 54), self.err, "body", 14, BAD)
            c.text((x0 + 24, self.h - 70), "Tab = next box · Enter = go · Esc = quit", "mono", 12, MUTED)
            return
        best = max([r["score"] for r in all_runs() if r["player"] == self.me and r.get("finished")], default=None)
        c.panel(x0, y0, x1, self.h - 40)
        c.text((x0 + 24, y0 + 22), f"READY, {self.players.name(self.me).upper()}?", "display", 40, INK)
        c.text((x0 + 24, y0 + 78), f"Next up: attempt #{self.players.attempts(self.me) + 1}"
               + (f"   ·   your best: {best:,}" if best is not None else ""), "body", 16, MUTED)
        bx = c.button(x0 + 24, y0 + 120, "START", "play", primary=True, size=22, pad=30)
        c.text((bx + 16, y0 + 139), "or press Enter", "body", 14, MUTED, "lm")
        lines = ["Point with your index finger. Hold the reticle on a target until the lock-on ring fills.",
                 f"Faster locks score more. Small targets are worth {SMALL_BONUS:g}x.",
                 "The timer starts after the 3-2-1 countdown."
                 + ("  Lead: STUDY (a hidden condition each game)." if lead_mode == "study" else f"  Lead: {lead_mode.upper()}."),
                 "q quits a game early (it still records your hand, but doesn't go on the leaderboard)."]
        for i, s in enumerate(lines):
            c.text((x0 + 24, y0 + 200 + 28 * i), s, "body", 15, MUTED)

    def draw_board(self, c):
        x0, y0, x1 = 40, 200, self.w - 40
        c.panel(x0, y0, x1, self.h - 40)
        c.text((x0 + 24, y0 + 22), "STANDINGS", "display", 32, INK)
        bx = x1 - 24 - 230
        bx = c.button(bx, y0 + 24, "Best per player", "best:1", size=13, pressed=self.best_only) + 6
        c.button(bx, y0 + 24, "All runs", "best:0", size=13, pressed=not self.best_only)
        rows = ranked(all_runs(), self.best_only)[:10]
        if not rows:
            c.text((x0 + 24, y0 + 90), "No finished games yet. Be the first.", "body", 16, MUTED)
            return
        cols = [(x0 + 24, "#", "la"), (x0 + 80, "PLAYER", "la"), (x0 + 380, "ATTEMPT", "ra"),
                (x0 + 560, "SCORE", "ra"), (x0 + 700, "TIME", "ra"), (x1 - 24, "AVG / TARGET", "ra")]
        hy = y0 + 76
        for x, label, a in cols:
            c.text((x, hy), label, "mono", 11, MUTED, a)
        c.d.line((x0 + 24, hy + 20, x1 - 24, hy + 20), fill=LINE, width=2)
        c.text((x0 + 24, self.h - 64), "P = preview on  ·  H = hard targets  ·  lead: O = off, D = damped, C = CNN, T = steady CNN, S = study (hidden)",
               "mono", 10, MUTED)
        y = hy + 28
        for i, r in enumerate(rows):
            if r["player"] == self.me:
                c.d.rectangle((x0 + 20, y - 2, x1 - 20, y + 30), fill=SOFT)
            c.text((x0 + 24, y + 14), str(i + 1), "display", 24, ACCENT if i == 0 else INK, "lm")
            c.text((x0 + 80, y + 14), r["name"], "body", 16, INK, "lm")
            tag = (" P" if r.get("preview") else "") + (" H" if r.get("targets") == "hard" else "") + (" S" if r.get("blind") else
                   {"off": " O", "damped": " D", "cnn": " C", "steady": " T"}.get(r.get("lead"), ""))
            c.text((x0 + 380, y + 14), f"#{r['attempt']}{tag}", "mono", 15, MUTED, "rm")
            c.text((x0 + 560, y + 14), f"{r['score']:,}", "display", 24, INK, "rm")
            c.text((x0 + 700, y + 14), clock(r["time"]), "mono", 15, INK, "rm")
            c.text((x1 - 24, y + 14), f"{r['avg']:.2f} s", "mono", 15, MUTED, "rm")
            c.d.line((x0 + 24, y + 31, x1 - 24, y + 31), fill=LINE)
            y += 34

    def draw_profile(self, c):
        x0, y0, x1 = 40, 200, self.w - 40
        c.panel(x0, y0, x1, self.h - 40)
        mine = [r for r in all_runs() if r["player"] == self.me]
        done = [r for r in mine if r.get("finished")]
        c.text((x0 + 24, y0 + 22), self.players.name(self.me).upper(), "display", 40, INK)
        stats = [("ATTEMPTS", str(self.players.attempts(self.me))),
                 ("BEST SCORE", f"{max(r['score'] for r in done):,}" if done else "–"),
                 ("BEST TIME", clock(min(r["time"] for r in done)) if done else "–"),
                 ("BEST THROUGHPUT", (f"{max(r['tp'] for r in done if r.get('tp')):.2f} bits/s"
                                      if any(r.get("tp") for r in done) else "–"))]
        for i, (label, value) in enumerate(stats):
            x = x0 + 24 + i * 210
            c.text((x, y0 + 84), label, "mono", 11, MUTED)
            c.text((x, y0 + 102), value, "display", 36, ACCENT if i == 1 else INK)
        y = y0 + 170
        c.text((x0 + 24, y), "RECENT GAMES", "mono", 11, MUTED)
        y += 24
        if not mine:
            c.text((x0 + 24, y), "No games yet.", "body", 15, MUTED)
        for r in reversed(mine[-8:]):
            c.text((x0 + 24, y), f"#{r['attempt']}", "mono", 14, MUTED)
            c.text((x0 + 90, y), f"{r['score']:,}", "body", 15, INK)
            c.text((x0 + 200, y), clock(r["time"]), "mono", 14, INK)
            c.text((x0 + 300, y), "finished" if r.get("finished") else f"quit at {r['closed']}/{r['windows']}",
                   "body", 14, MUTED if r.get("finished") else BAD)
            c.text((x1 - 24, y), r["date"], "mono", 12, MUTED, "ra")
            y += 30

    def draw_results(self, c):
        r = self.last_run
        x0, y0, x1 = 40, 200, self.w - 40
        c.panel(x0, y0, x1, self.h - 40)
        c.text((x0 + 24, y0 + 22), f"ATTEMPT #{r['attempt']} · {r['name'].upper()}", "mono", 13, MUTED)
        c.text((x0 + 24, y0 + 44), f"{r['score']:,}", "display", 110, ACCENT if r.get("finished") else INK)
        c.text((x0 + 26, y0 + 172), f"TIME {clock(r['time'])}   ·   {r['avg']:.2f} s PER TARGET   ·   "
               f"{r['closed']}/{r['windows']} TARGETS"
               + (f"   ·   {r['tp']:.2f} BITS/S" if r.get("tp") is not None else ""), "mono", 14, INK)
        if r.get("finished"):
            board = ranked(all_runs(), best_only=False)
            rank = next((i + 1 for i, b in enumerate(board) if b == r), None)
            mine = [b["score"] for b in board if b["player"] == r["player"] and b != r]
            msg, color = (f"#{rank} of {len(board)} finished games", INK) if rank else ("", INK)
            if not mine or r["score"] > max(mine):
                msg, color = "NEW PERSONAL BEST" + (f"  ·  #{rank} overall" if rank else ""), GOOD
            c.text((x0 + 24, y0 + 214), msg, "display", 28, color)
        else:
            c.text((x0 + 24, y0 + 214), "ABORTED · NOT ON THE LEADERBOARD", "display", 28, BAD)
        if r.get("lead") and not r.get("blind"):
            c.text((x0 + 24, y0 + 252), f"CURSOR LEAD THIS ROUND: {r['lead'].upper()}", "mono", 12, MUTED)
        elif r.get("blind"):
            gy = y0 + 256
            if r.get("guess") is None:                  # ask BEFORE anything is revealed
                c.text((x0 + 24, gy), "STUDY ROUND: WHICH CURSOR LEAD DO YOU THINK THAT WAS?", "mono", 12, ACCENT)
                bx = x0 + 24
                for g in CONDITIONS + ["not sure"]:
                    bx = c.button(bx, gy + 18, g.upper(), "guess:" + g, size=13, pad=12) + 8
            else:
                c.text((x0 + 24, gy), f"YOUR GUESS: {r['guess'].upper()}", "mono", 12, MUTED)
                if r.get("revealed"):
                    right = r["guess"] == r["lead"]
                    c.text((x0 + 220, gy), f"ACTUAL: {r['lead'].upper()}  ({'right' if right else 'wrong'})",
                           "mono", 12, GOOD if right else BAD)
                    c.button(x0 + 24, gy + 18, "Hide it again", f"reveal:{r['attempt']}", size=13, pad=12)
                else:
                    bx = c.button(x0 + 24, gy + 18, "Reveal", f"reveal:{r['attempt']}", size=13, pad=12) + 10
                    c.text((bx, gy + 32), "or keep it hidden (you can reveal any game later in the STUDY LOG)",
                           "mono", 11, MUTED, "lm")
        by = y0 + 330 if r.get("blind") else y0 + 280
        bx = c.button(x0 + 24, by, "PLAY AGAIN", "play", primary=True, size=20, pad=24) + 10
        bx = c.button(bx, by, "Leaderboard", "tab:board", size=16, pad=18) + 10
        c.button(bx, by, "Switch player", "logout", size=16, pad=18)
        c.text((x0 + 24, self.h - 70), "Enter = play again · Esc = quit · your hand movement was saved to data/ for training",
               "mono", 12, MUTED)

    def draw_study(self, c):
        """Every STUDY game: your guess, and a show/hide toggle for the real lead, one game at a time."""
        x0, y0, x1 = 40, 200, self.w - 40
        c.panel(x0, y0, x1, self.h - 40)
        c.text((x0 + 24, y0 + 22), "STUDY LOG", "display", 32, INK)
        mine = [r for r in all_runs() if r.get("player") == self.me and r.get("blind")]
        guessed = [r for r in mine if r.get("guess") not in (None, "not sure")]
        shown = [r for r in guessed if r.get("revealed")]
        right = sum(r["guess"] == r["lead"] for r in shown)
        c.text((x0 + 24, y0 + 66), f"{len(mine)} study games  ·  {len(guessed)} guesses  ·  "
               + (f"{right} of {len(shown)} revealed guesses right (chance would be about 1 in 3)" if shown
                  else "none revealed yet"), "body", 14, MUTED)
        if not mine:
            c.text((x0 + 24, y0 + 110), "No study games yet. Set LEAD: STUDY (bottom right) and play.", "body", 15, MUTED)
            return
        cols = [(x0 + 24, "GAME", "la"), (x0 + 120, "DATE", "la"), (x0 + 330, "SCORE", "ra"), (x0 + 450, "BITS/S", "ra"),
                (x0 + 500, "YOUR GUESS", "la"), (x0 + 650, "ACTUAL LEAD", "la")]
        hy = y0 + 100
        for x, label, an in cols:
            c.text((x, hy), label, "mono", 11, MUTED, an)
        c.d.line((x0 + 24, hy + 20, x1 - 24, hy + 20), fill=LINE, width=2)
        y = hy + 28
        for r in list(reversed(mine))[:11]:
            c.text((x0 + 24, y + 14), f"#{r['attempt']}", "mono", 14, INK, "lm")
            c.text((x0 + 120, y + 14), r.get("date", ""), "mono", 12, MUTED, "lm")
            c.text((x0 + 330, y + 14), f"{r['score']:,}", "body", 15, INK, "rm")
            c.text((x0 + 450, y + 14), f"{r['tp']:.2f}" if r.get("tp") else "-", "mono", 14, INK, "rm")
            c.text((x0 + 500, y + 14), (r.get("guess") or "-").upper(), "mono", 13, INK, "lm")
            if r.get("revealed"):
                ok = r.get("guess") == r["lead"]
                col = GOOD if ok else (BAD if r.get("guess") not in (None, "not sure") else INK)
                c.text((x0 + 650, y + 14), r["lead"].upper(), "mono", 13, col, "lm")
                c.button(x1 - 90, y + 2, "hide", f"reveal:{r['attempt']}", size=11, pad=10)
            else:
                c.text((x0 + 650, y + 14), "hidden", "mono", 13, MUTED, "lm")
                c.button(x1 - 90, y + 2, "show", f"reveal:{r['attempt']}", size=11, pad=10)
            c.d.line((x0 + 24, y + 31, x1 - 24, y + 31), fill=LINE)
            y += 34

    # ---------- actions
    def do(self, action):
        self.err = ""
        if action is None:
            self.focus = None
            return None
        kind, _, arg = action.partition(":")
        if kind == "theme":
            next_theme()
        elif kind == "preview":
            toggle_preview()
        elif kind == "lead":
            next_lead()
        elif kind == "targets":
            toggle_targets()
        elif kind == "send" and self.me and not self.sending:
            self.start_sending()
        elif kind == "guess" and self.last_run is not None:
            update_run(self.last_run["player"], self.last_run["attempt"], guess=arg)
            self.last_run["guess"] = arg
        elif kind == "reveal":
            attempt = int(arg)
            run = next((r for r in all_runs() if r.get("player") == self.me and r.get("attempt") == attempt), None)
            if run is not None:
                flip = not run.get("revealed", False)                     # toggle: reveal / hide again
                update_run(self.me, attempt, revealed=flip)
                if self.last_run is not None and self.last_run.get("attempt") == attempt:
                    self.last_run["revealed"] = flip
        elif kind == "tab":
            self.tab = arg
        elif kind == "best":
            self.best_only = arg == "1"
        elif kind == "pick":
            self.target, self.focus, self.fields["code"] = arg, "code", ""
        elif kind == "focus":
            self.focus = arg
        elif kind == "login":
            if self.players.check(self.target, self.fields["code"]):
                self.me, self.target, self.focus = self.target, None, None
                self.fields = {k: "" for k in self.fields}
                self.tab = "play"
            else:
                self.err, self.fields["code"] = "Wrong passcode.", ""
        elif kind == "create":
            err = self.players.create(self.fields["new_name"], self.fields["new_code"])
            if err:
                self.err = err
            else:
                self.me, self.focus = key_of(" ".join(self.fields["new_name"].split())[:16]), None
                self.fields = {k: "" for k in self.fields}
                self.tab = "play"
        elif kind == "logout":
            self.me, self.last_run, self.tab, self.target = None, None, "play", None
        elif kind == "play" and self.me:
            return ("play", self.me)
        return None

    def start_sending(self):
        """SEND GAMES: asks first (a normal Mac box), then uploads in the background (see share.py)."""
        import share
        me, name = self.me, self.players.name(self.me)
        self.sending, self.note = True, "Asking..."

        def job():
            try:
                self.note = share.send(me, name, progress=lambda m: setattr(self, "note", m))
            except Exception as err:
                self.note = f"Couldn't send: {err}"
            self.sending = False
        threading.Thread(target=job, daemon=True).start()

    def key(self, k):
        """Keyboard. Returns an action to run, or None."""
        if k in (13, 10):                                       # Enter
            if self.focus == "code":
                return "login"
            if self.focus in ("new_name", "new_code"):
                return "create"
            if self.me and self.tab in ("play", "results"):
                return "play"
            return None
        if k == 9:                                              # Tab: next box
            order = (["code"] if self.target else []) + ["new_name", "new_code"]
            if not self.me:
                self.focus = order[(order.index(self.focus) + 1) % len(order)] if self.focus in order else order[0]
            return None
        if self.focus is None:
            return None
        if k in (8, 127):                                       # Backspace
            self.fields[self.focus] = self.fields[self.focus][:-1]
        elif 32 <= k < 127 and len(self.fields[self.focus]) < 16:
            self.fields[self.focus] += chr(k)
        return None

    def run(self, last_run=None):
        if last_run is not None:
            self.last_run, self.tab = last_run, "results"
        self.clicks.clear()
        while True:
            c = self.draw()
            cv2.imshow(self.win, c.bgr())
            k = cv2.waitKey(30) & 0xFF
            if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
                return ("quit", None)                          # window closed
            if k == 27 and self.focus is None:                 # Esc (when not typing) = quit
                return ("quit", None)
            if k == 27:
                self.focus = None
            elif k != 255:
                if self.focus is None and k == ord("q"):
                    return ("quit", None)
                action = self.key(k)
                if action:
                    result = self.do(action)
                    if result:
                        return result
            while self.clicks:
                result = self.do(c.hit(*self.clicks.pop(0)))
                if result:
                    return result
