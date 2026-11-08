"""Lesson 32: your letter CNN vs Apple's handwriting reader (Apple Vision).

Apple Vision is what powers Live Text on your Mac (selecting text in a photo). It's trained on huge amounts of
handwriting that Apple never released. It reads PICTURES, not strokes, so for each test letter we:
  1. draw the strokes as black ink on a white square (like a scan of the tablet),
  2. ask Apple Vision "what does this say?",
  3. take the first character of its answer and score it the same way as lesson 31
     (upper/lower case count as the same letter, zero counts as o).

Same 2480 test letters, same 20 writers neither model trained on. So it's a fair fight, like CNN vs damped.

One thing to keep in mind: Apple's reader is built for WORDS and sentences. A lone letter gives it no context
(a lone "l" vs "1" vs "I" is a coin flip even for people), so single letters are its weakest case. Your CNN was
trained for exactly this. Whole words would be a different contest (later, with your own air-writing).

Needs (once):  pip install pyobjc-framework-Vision pillow
Run from the crosshair folder:  python lessons/lesson32.py        (a few minutes: one Apple call per letter)
                                python lessons/lesson32.py 300    (quick try on the first 300 letters)
"""

import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lesson31 import CLASSES, LetterCNN, build, label_of, read_uji   # noqa: E402

SIZE = 256          # picture size in pixels
MARGIN = 40         # white border around the letter
INK = 7             # line thickness in pixels


def draw(strokes):
    """Strokes -> PNG bytes: black ink on white, letter scaled to fit, shape kept."""
    allp = np.concatenate(strokes)
    lo, hi = allp.min(0), allp.max(0)
    scale = (SIZE - 2 * MARGIN) / max(hi[0] - lo[0], hi[1] - lo[1], 1e-6)
    mid = (lo + hi) / 2
    img = Image.new("L", (SIZE, SIZE), 255)
    pen = ImageDraw.Draw(img)
    for s in strokes:
        pts = [(float((x - mid[0]) * scale + SIZE / 2), float((y - mid[1]) * scale + SIZE / 2)) for x, y in s]
        if len(pts) == 1:
            pts = pts * 2
        pen.line(pts, fill=0, width=INK, joint="curve")
        for x, y in (pts[0], pts[-1]):                       # round line ends
            pen.ellipse((x - INK / 2, y - INK / 2, x + INK / 2, y + INK / 2), fill=0)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue(), img


def apple_reader():
    """Returns a function: PNG bytes -> list of (text, confidence), Apple's best guesses first."""
    import Vision
    from Foundation import NSData

    def read(png):
        data = NSData.dataWithBytes_length_(png, len(png))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)            # single letters: there's no word to correct
        request.setRecognitionLanguages_(["en-US"])
        request.setMinimumTextHeight_(0.1)
        handler.performRequests_error_([request], None)
        guesses = []
        for obs in request.results() or []:
            for cand in obs.topCandidates_(3):
                guesses.append((str(cand.string()), float(cand.confidence())))
        guesses.sort(key=lambda g: -g[1])
        return guesses
    return read


def to_class(text):
    """Apple's answer -> one of our 35 classes (or None if it's not a letter/digit we use)."""
    for ch in text.strip():
        if ch.isalnum():
            return label_of(ch)
    return None


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    samples = [(c, w, s) for c, w, s in read_uji("letters/ujipenchars2.txt") if label_of(c) is not None]
    tst = [s for s in samples if s[1].startswith("tst")][:limit]
    truth = np.array([label_of(c) for c, _, _ in tst])

    # your CNN from lesson 31
    saved = torch.load("runs/letters_cnn.pt", map_location="cpu")
    cnn = LetterCNN()
    cnn.load_state_dict(saved["model"])
    cnn.eval()
    with torch.no_grad():
        scores = cnn(build(tst))
    cnn_guess = scores.argmax(1).numpy()
    cnn_top3 = (scores.topk(3, 1).indices.numpy() == truth[:, None]).any(1)

    # Apple Vision
    read = apple_reader()
    apple_guess, apple_top3, apple_raw, pictures = [], [], [], []
    start = time.time()
    for i, (ch, _, strokes) in enumerate(tst):
        png, img = draw(strokes)
        guesses = read(png)
        classes = [to_class(t) for t, _ in guesses]
        first = classes[0] if classes else None
        apple_guess.append(-1 if first is None else first)
        apple_top3.append(truth[i] in [c for c in classes[:3] if c is not None])
        apple_raw.append(guesses[0][0] if guesses else "(nothing)")
        pictures.append(img)
        if (i + 1) % 250 == 0:
            print(f"   {i + 1}/{len(tst)} letters read by Apple   ({time.time() - start:.0f} s)")
    apple_guess = np.array(apple_guess)
    apple_top3 = np.array(apple_top3)

    cnn_ok, apple_ok = cnn_guess == truth, apple_guess == truth
    print(f"\n{len(tst)} test letters from 20 writers neither model trained on:")
    print(f"   {'':16s} {'error':>7s} {'top-3 error':>12s}")
    print(f"   {'your CNN':16s} {(~cnn_ok).mean() * 100:6.2f}% {(~cnn_top3).mean() * 100:11.2f}%")
    print(f"   {'Apple Vision':16s} {(~apple_ok).mean() * 100:6.2f}% {(~apple_top3).mean() * 100:11.2f}%")
    print(f"   Apple read nothing at all for {(apple_guess == -1).mean() * 100:.1f}% of letters")

    print("\nWHO GETS WHICH LETTERS RIGHT?")
    print(f"   both right        {(cnn_ok & apple_ok).mean() * 100:5.1f}%")
    print(f"   only your CNN     {(cnn_ok & ~apple_ok).mean() * 100:5.1f}%")
    print(f"   only Apple        {(~cnn_ok & apple_ok).mean() * 100:5.1f}%")
    print(f"   both wrong        {(~cnn_ok & ~apple_ok).mean() * 100:5.1f}%")
    print("   (if 'only Apple' and 'only CNN' are both big, they make DIFFERENT mistakes, and combining them helps)")

    per = []
    for k, c in enumerate(CLASSES):
        m = truth == k
        if m.any():
            per.append((c, apple_ok[m].mean(), cnn_ok[m].mean()))
    print("\n   Apple's hardest letters:  " + "  ".join(f"{c} {a * 100:.0f}%" for c, a, _ in sorted(per, key=lambda x: x[1])[:8]))
    print("   (and your CNN on those:   " + "  ".join(f"{c} {b * 100:.0f}%" for c, a, b in sorted(per, key=lambda x: x[1])[:8]) + ")")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        wrong = np.nonzero(~apple_ok)[0]
        pick = np.random.default_rng(0).choice(wrong, min(24, len(wrong)), replace=False) if len(wrong) else []
        fig, axes = plt.subplots(3, 8, figsize=(14, 6))
        for ax in axes.flat:
            ax.axis("off")
        for ax, i in zip(axes.flat, pick):
            ax.imshow(pictures[i], cmap="gray")
            ax.set_title(f"'{tst[i][0]}': Apple '{apple_raw[i][:6]}'\nCNN {CLASSES[cnn_guess[i]]}"
                         f" {'✓' if cnn_ok[i] else '✗'}", fontsize=9)
        fig.suptitle("Letters Apple Vision got WRONG (what Apple read, and what your CNN said)")
        fig.tight_layout()
        fig.savefig("letters32.png", dpi=110)
        print("\nSaved letters32.png  (open letters32.png)")
    except ImportError:
        pass
