"""Why does Apple Vision read nothing? Try several ways of drawing the same 60 letters and compare.

Run from the crosshair folder:  python lessons/vision_check.py
"""

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lesson31 import label_of, read_uji   # noqa: E402

import Vision                              # noqa: E402
from Foundation import NSData              # noqa: E402


def draw(strokes, w, h, letter_h, ink):
    """Letter `letter_h` pixels tall, at the left of a w x h white strip."""
    allp = np.concatenate(strokes)
    lo, hi = allp.min(0), allp.max(0)
    scale = letter_h / max(hi[1] - lo[1], (hi[0] - lo[0]) * 0.7, 1e-6)
    img = Image.new("L", (w, h), 255)
    pen = ImageDraw.Draw(img)
    x0, y0 = h * 0.5, (h - (hi[1] - lo[1]) * scale) / 2
    for s in strokes:
        pts = [(float((x - lo[0]) * scale + x0), float((y - lo[1]) * scale + y0)) for x, y in s]
        pen.line(pts if len(pts) > 1 else pts * 2, fill=0, width=ink, joint="curve")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def read(png, accurate=True, min_height=None, correction=False):
    data = NSData.dataWithBytes_length_(png, len(png))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate if accurate
                             else Vision.VNRequestTextRecognitionLevelFast)
    req.setUsesLanguageCorrection_(correction)
    if min_height is not None:
        req.setMinimumTextHeight_(min_height)
    ok, err = handler.performRequests_error_([req], None)
    if err is not None:
        return None, str(err)
    res = req.results() or []
    return (str(res[0].topCandidates_(1)[0].string()) if res else ""), None


samples = [(c, w, s) for c, w, s in read_uji("letters/ujipenchars2.txt") if label_of(c) is not None]
tst = [s for s in samples if s[1].startswith("tst")]
pick = [tst[i] for i in np.random.default_rng(0).choice(len(tst), 60, replace=False)]

TRIES = {
    "square 256, big letter (lesson 32)": dict(w=256, h=256, letter_h=176, ink=7),
    "wide strip, letter 40 px":           dict(w=600, h=100, letter_h=40, ink=3),
    "wide strip, letter 64 px":           dict(w=800, h=140, letter_h=64, ink=5),
    "wide strip, letter 100 px":          dict(w=1000, h=200, letter_h=100, ink=7),
    "square 256, letter 64 px":           dict(w=256, h=256, letter_h=64, ink=5),
}
print("how often Apple reads ANYTHING, and how often it's the right letter (60 letters each):\n")
first_error = None
for name, kw in TRIES.items():
    for accurate in (True, False):
        any_text = right = 0
        for ch, _, strokes in pick:
            text, err = read(draw(strokes, **kw), accurate=accurate, min_height=0.0)
            if err and first_error is None:
                first_error = err
            if text:
                any_text += 1
                first = next((c for c in text if c.isalnum()), None)
                right += first is not None and label_of(first) == label_of(ch)
        print(f"   {name:36s} {'accurate' if accurate else 'fast    '}   read something {any_text:2d}/60   right {right:2d}/60")
if first_error:
    print("\nApple reported an error:", first_error)

# save one example of each, so we can SEE what Apple got
for i, (name, kw) in enumerate(TRIES.items()):
    Image.open(io.BytesIO(draw(pick[0][2], **kw))).save(f"vision_try{i}.png")
print(f"\nSaved vision_try0..{len(TRIES) - 1}.png (the letter '{pick[0][0]}' drawn each way)")
