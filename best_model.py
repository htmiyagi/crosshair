"""The model that won Crosshair's experiments (lessons 16-26), written out on its own as a reference.

It is NOT wired into the game or into FIVES yet. FIVES currently uses the simpler "damped speed" lead.
The plan is to swap this in once there is more (and more varied) training data: lesson 26's learning curve
showed this model still improving as data grows.

RESULT (lesson 25): 17.46 px average error predicting the fingertip 2 frames (~60 ms) ahead,
    15% better than a damped-velocity baseline and 6.5% better than a regularized MLP,
    winning all 6 cross-validation folds, and the same ranking on 5 held-out "lockbox" games.

INPUT: the last 8 frames of 3 MediaPipe hand points (index fingertip 8, index knuckle 5, wrist 0),
    each as (x, y) in camera units (0..1), shape (examples, 8 frames, 3 points, 2).
    Everything is made relative to the fingertip's position "now" and multiplied by 10 (see `prepare`).
OUTPUT: where the fingertip will be 2 frames later, in the same relative x10 units (undo with `to_camera`).

TRAINING RECIPE (each piece was tested on its own in the lessons):
    - Huber loss (delta 0.1)        big, unpredictable misses pull the model around less (lesson 20)
    - dropout 0.3                   stopped the memorizing seen in the training curves (lessons 23-24)
    - Adam, learning rate 0.001     0.01 bounced around and stopped improving early (lesson 23)
    - mirror augmentation           flip left <-> right: worth about as much as doubling the data (lesson 19)
    - 5000 full-batch steps, keeping the version that scores best on 3 held-out validation games
    - examples touching a tracking glitch are left out (spike / too-far-from-knuckle rules, lesson 13)
"""

import torch
from torch import nn

FRAMES = 8          # how many past frames the model sees
POINTS = (8, 5, 0)  # MediaPipe points: index fingertip, index knuckle, wrist
SCALE = 10          # inputs and outputs are relative to "now", times this
FILTERS = 32        # "magnifying glasses" per conv layer
WIDTH = 3           # each one looks at 3 neighboring frames
DROPOUT = 0.3
HUBER_DELTA = 0.1
LEARNING_RATE = 0.001
STEPS = 5000


class HandCNN(nn.Module):
    """Two 1D convolution layers slide along time, then a small MLP turns their scores into a guess."""

    def __init__(self, drop=DROPOUT):
        super().__init__()
        channels = len(POINTS) * 2                                   # x and y of each point = 6
        self.conv = nn.Sequential(
            nn.Conv1d(channels, FILTERS, kernel_size=WIDTH), nn.ReLU(), nn.Dropout(drop),   # 8 frames -> 6 stops
            nn.Conv1d(FILTERS, FILTERS, kernel_size=WIDTH), nn.ReLU(), nn.Dropout(drop),    # 6 stops  -> 4 stops
        )
        stops = FRAMES - 2 * (WIDTH - 1)
        self.head = nn.Sequential(nn.Linear(FILTERS * stops, 64), nn.ReLU(), nn.Dropout(drop), nn.Linear(64, 2))

    def forward(self, hands):
        """hands: (n, 8 frames, 3 points, 2), already prepared. Returns (n, 2)."""
        x = hands.reshape(len(hands), FRAMES, -1).permute(0, 2, 1)  # (n, 6 channels, 8 frames), as Conv1d wants
        return self.head(self.conv(x).reshape(len(hands), -1))


def prepare(hands):
    """Raw camera points (n, 8, 3, 2) -> model input: relative to the fingertip 'now', x SCALE."""
    now = hands[:, -1, 0, :]
    return (hands - now[:, None, None, :]) * SCALE


def to_camera(guess, hands):
    """Model output -> predicted fingertip position in camera units."""
    return hands[:, -1, 0, :] + guess / SCALE


def mirror(hands, answers):
    """Augmentation: the original examples plus a left <-> right flipped copy (inputs are relative, so flip = negate x)."""
    h2, a2 = hands.clone(), answers.clone()
    h2[..., 0] *= -1
    a2[:, 0] *= -1
    return torch.cat([hands, h2]), torch.cat([answers, a2])


def loss_fn(guess, truth):
    return nn.functional.huber_loss(guess, truth, delta=HUBER_DELTA)


if __name__ == "__main__":
    model = HandCNN()
    print(model)
    print(f"{sum(p.numel() for p in model.parameters()):,} weights")
    model.eval()                                                     # dropout off when predicting
    fake = prepare(torch.rand(4, FRAMES, len(POINTS), 2))
    print("output shape:", tuple(model(fake).shape))
