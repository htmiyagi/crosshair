"""Lesson 10: the noise floor. How much of the model's error is impossible to remove?

The "answer" the model is graded on is MediaPipe's fingertip 2 frames later. That answer shakes a little
even when your finger doesn't move (camera noise, the tracker re-guessing every frame). No model can predict
random shake, so it sets a FLOOR: even a perfect predictor would be off by about that much.

To measure it: hold your finger still for a few seconds. If the finger isn't moving, any movement the tracker
reports is pure noise.

Do it twice, because there are two kinds of "still":
  python lessons/lesson10.py --label desk     rest your hand/wrist ON the desk, point up (tracking noise only)
  python lessons/lesson10.py --label air      hold your finger in the air like you're using it (noise + tremor)

Recordings go to data_still/ (NOT data/, so they never end up in training).
Re-check an old recording without the camera:  python lessons/lesson10.py --file data_still/still_air_....npz
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so "import common" finds the main folder

SCREEN_PX = 1512       # same pixel conversion as lessons 6-9, so the numbers compare directly
AHEAD = 2
MODEL_PX = 16.5        # your best model's test error from lesson 9 (MLP + early stopping)
SECONDS = 6
COUNTDOWN = 3
OUT_DIR = Path("data_still")


def record(label):
    """Show the camera, count down, then record the fingertip for SECONDS. Returns (times, tips)."""
    import cv2
    import mediapipe as mp
    import common as C

    landmarker = C.make_landmarker()
    cam = C.Camera()
    times, tips = [], []
    seq, last_ms = 0, -1
    start = time.monotonic()
    print(f"Get your finger in view. Recording starts in {COUNTDOWN} s and lasts {SECONDS} s. Stay STILL.")
    try:
        while True:
            frame, stamp, seq = cam.read(seq)
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)
            ms = max(int(stamp * 1000), last_ms + 1)
            last_ms = ms
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, ms)

            elapsed = time.monotonic() - start
            recording = elapsed >= COUNTDOWN
            h, w = frame.shape[:2]
            if result.hand_landmarks:
                lm = result.hand_landmarks[0][8]                       # point 8 = index fingertip
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 6, (60, 170, 255), -1)
                if recording:
                    times.append(stamp)
                    tips.append((lm.x, lm.y))
            elif recording:
                times.append(stamp)
                tips.append((np.nan, np.nan))                          # hand lost this frame

            if recording:
                msg, color = f"RECORDING - hold still  {SECONDS - (elapsed - COUNTDOWN):.1f}s", (60, 60, 255)
            else:
                msg, color = f"get ready ({label})  {COUNTDOWN - elapsed:.1f}s", (200, 200, 200)
            cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow("noise floor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q") or elapsed >= COUNTDOWN + SECONDS:
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()
        landmarker.close()
    return np.array(times), np.array(tips, dtype=np.float64)


def px(dist):
    return dist * SCREEN_PX


def analyze(tips, label):
    good = ~np.isnan(tips).any(axis=1)
    print(f"\n{label}: {len(tips)} frames, hand found in {good.sum()}")
    if good.sum() < 30:
        print("Not enough frames with a hand. Try again with your finger clearly in view.")
        return
    t = tips[good]

    # 1. How far does the dot wander from its average spot? (the size of the shake)
    center = t.mean(axis=0)
    wander = np.linalg.norm(t - center, axis=1)

    # 2. "No prediction" (guess: it stays where it is now), graded on the frame 2 later.
    #    For a still finger this is the RIGHT guess, so its error is pure noise.
    still_err = np.linalg.norm(tips[AHEAD:] - tips[:-AHEAD], axis=1)
    still_err = still_err[~np.isnan(still_err)]

    # 3. Constant speed on a still finger: it treats the shake as real movement and projects it forward.
    now, prev, later = tips[1:-AHEAD], tips[:-AHEAD - 1], tips[1 + AHEAD:]
    const_err = np.linalg.norm(now + (now - prev) * AHEAD - later, axis=1)
    const_err = const_err[~np.isnan(const_err)]

    # 4. Slow drift: is the hand creeping over the seconds? (that's real movement, not noise)
    drift = np.linalg.norm(t[-10:].mean(axis=0) - t[:10].mean(axis=0))

    floor = px(still_err.mean())
    print(f"   shake (avg distance from the average spot):   {px(wander.mean()):5.1f} px")
    print(f"   'stays put' guess, graded 2 frames later:     {floor:5.1f} px   <- the noise floor")
    print(f"   constant speed on a still finger:             {px(const_err.mean()):5.1f} px   (it chases the shake)")
    print(f"   slow drift from start to end:                 {px(drift):5.1f} px   (real movement, ignore if small)")
    print(f"\n   your model's error while MOVING:              {MODEL_PX:5.1f} px")
    print(f"   -> about {floor / MODEL_PX * 100:.0f}% of the model's error is noise no model can remove,")
    print(f"      leaving ~{max(MODEL_PX - floor, 0):.1f} px that better models or data could still chip away at.")


parser = argparse.ArgumentParser()
parser.add_argument("--label", default="air", help="desk or air (just goes in the file name)")
parser.add_argument("--file", help="analyze a saved recording instead of recording a new one")
args = parser.parse_args()

if args.file:
    d = np.load(args.file)
    analyze(d["tips"], Path(args.file).stem)
else:
    times, tips = record(args.label)
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"still_{args.label}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(out, t=times, tips=tips)
    print(f"Saved {out}")
    analyze(tips, args.label)
