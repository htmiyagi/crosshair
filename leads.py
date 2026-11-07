"""The three cursor "lead" conditions, the same ones FIVES uses, so the game can compare them with Fitts' law.

    off     the cursor is where your (smoothed) fingertip is
    damped  pushed ahead by your fingertip's speed x 40 ms (LeadPredictor)
    cnn     pushed ahead by where the Crosshair CNN predicts your fingertip goes next (CNNLead),
            faded out when your hand is still so it doesn't add jitter
    steady  the same, but with the CNN that was TRAINED not to shake (lesson 29's jitter penalty)

LeadPredictor is copied from FIVES' gestures.py and CNNLead from FIVES' cnn_lead.py, with FIVES' current settings.
"""

from collections import deque

import torch
from torch import nn

CNN_PATH = "runs/deploy_cnn_1.pt"     # the 1-frame-ahead ensemble from lesson 27
STEADY_PATH = "runs/steady_cnn_0.3.pt"   # lesson 29: the jitter-penalty CNN (lambda 0.3); try 0.1 by changing this
CNN_SETTINGS = {"scale": 1.0, "most": 0.04, "still": 0.04, "full": 0.07, "calm": 1.0}   # what you tuned in FIVES


class LeadPredictor:
    """Draws the cursor slightly AHEAD of your finger, to cancel out lag (from the Crosshair experiments).

    The camera, the hand tracker and the smoothing all take time, so the cursor trails your real finger.
    This guesses where the finger is going: its current speed x a short time ahead ("damped speed", the
    simple guesser that was almost as good as the neural networks in Crosshair).

      seconds  how far ahead to guess (0.04 = 40 ms). More = less lag, but more overshoot when you stop.
      most     the furthest it may lead, as a fraction of the picture, so a glitch can't fling the cursor.

    update() returns an offset (dx, dy) to add to the cursor. Works on camera positions (0..1).
    """

    def __init__(self, seconds=0.04, most=0.04):
        self.seconds, self.most = seconds, most
        self.prev = None
        self.t = None
        self.v = (0.0, 0.0)

    def update(self, p, t):
        if self.prev is not None and t > self.t:
            dt = t - self.t
            vx, vy = (p[0] - self.prev[0]) / dt, (p[1] - self.prev[1]) / dt
            self.v = (0.5 * self.v[0] + 0.5 * vx, 0.5 * self.v[1] + 0.5 * vy)     # calm the speed a little
        self.prev, self.t = p, t
        dx, dy = self.v[0] * self.seconds, self.v[1] * self.seconds
        size = (dx * dx + dy * dy) ** 0.5
        if size > self.most:                                   # never lead by more than `most`
            dx, dy = dx * self.most / size, dy * self.most / size
        return dx, dy

    def hand_gone(self):
        self.prev, self.t, self.v = None, None, (0.0, 0.0)


class HandCNN(nn.Module):
    """Same layers as Crosshair's best_model.py. It has to match exactly, or the saved weights won't load."""

    def __init__(self, frames, filters, width, drop):
        super().__init__()
        self.frames = frames
        self.conv = nn.Sequential(
            nn.Conv1d(6, filters, kernel_size=width), nn.ReLU(), nn.Dropout(drop),
            nn.Conv1d(filters, filters, kernel_size=width), nn.ReLU(), nn.Dropout(drop),
        )
        stops = frames - 2 * (width - 1)
        self.head = nn.Sequential(nn.Linear(filters * stops, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 2))

    def forward(self, hands):
        x = hands.reshape(len(hands), self.frames, -1).permute(0, 2, 1)
        return self.head(self.conv(x).reshape(len(hands), -1))


class CNNLead:
    """Keeps a short history of the hand and asks the CNN where the fingertip goes next."""

    def __init__(self, path, scale=1.0, most=0.04, max_gap=0.1, still=0.04, full=0.2, calm=0.5):
        torch.set_num_threads(1)                     # one tiny prediction per frame: one CPU thread is plenty
        saved = torch.load(path, map_location="cpu")
        self.frames = saved["frames"]
        self.points = saved["points"]                # MediaPipe point numbers, e.g. [8, 5, 0]
        self.size = saved["scale"]                   # inputs/outputs were relative to "now", x this
        self.models = []
        for state in saved["models"]:
            m = HandCNN(saved["frames"], saved["filters"], saved["width"], saved["dropout"])
            m.load_state_dict(state)
            m.eval()                                 # dropout off: we're predicting, not training
            self.models.append(m)
        self.scale, self.most, self.max_gap = scale, most, max_gap
        # Fade-out when still: the CNN reads RAW points, which wobble a little even when your hand doesn't move,
        # and that wobble would be added on top of FIVES's steady (smoothed) cursor. So:
        #   hand slower than `still` (picture widths per second)  -> no push at all
        #   between `still` and `full`                            -> blend in gradually
        #   faster than `full`                                     -> the full push
        # and `calm` (0..1) smooths the push itself: each frame it moves only that fraction toward the new guess.
        self.still, self.full, self.calm = still, full, calm
        self.smoothed = (0.0, 0.0)
        self.history = deque(maxlen=self.frames)     # (t, [(x, y) of each of the 3 points])
        self.last_offset = (0.0, 0.0)
        with torch.inference_mode():                 # warm-up: the very first prediction is slow, do it now
            for m in self.models:
                m(torch.zeros(1, self.frames, len(self.points), 2))

    def update(self, points, t):
        """points = the 21 (x, y) hand points this frame (camera units). Returns the lead offset (dx, dy)."""
        if self.history and t - self.history[-1][0] > self.max_gap:
            self.history.clear()                     # a pause or a lost hand: start a fresh history
        self.history.append((t, [points[i] for i in self.points]))
        if len(self.history) < self.frames:
            self.smoothed = (0.0, 0.0)
            return (0.0, 0.0)
        hands = torch.tensor([pts for _, pts in self.history], dtype=torch.float32)   # (8 frames, 3 points, 2)
        now = hands[-1, 0]
        x = ((hands - now) * self.size)[None]        # relative to the fingertip now, x scale, as in training
        with torch.inference_mode():
            guess = torch.stack([m(x) for m in self.models]).mean(0)[0] / self.size  # camera units, from "now"
        dx, dy = float(guess[0]) * self.scale, float(guess[1]) * self.scale
        # How fast is the hand REALLY moving? Net fingertip movement over the last 4 frames (wobble mostly cancels).
        (t0, p0), (t1, p1) = self.history[-4], self.history[-1]
        speed = ((p1[0][0] - p0[0][0]) ** 2 + (p1[0][1] - p0[0][1]) ** 2) ** 0.5 / max(t1 - t0, 1e-3)
        weight = min(1.0, max(0.0, (speed - self.still) / max(self.full - self.still, 1e-6)))
        dx, dy = dx * weight, dy * weight
        sx, sy = self.smoothed
        dx, dy = sx + self.calm * (dx - sx), sy + self.calm * (dy - sy)   # smooth the push a little
        self.smoothed = (dx, dy)
        size = (dx * dx + dy * dy) ** 0.5
        if size > self.most:                         # never lead by more than `most`, so a glitch can't fling it
            dx, dy = dx * self.most / size, dy * self.most / size
        self.last_offset = (dx, dy)
        return dx, dy

    def hand_gone(self):
        self.history.clear()
        self.smoothed = (0.0, 0.0)
