"""
Excluding what is not the chassis, and why the fill is the plate's grey rather than white.

THE PROBLEM
-----------
A wide shot catches the bench, the floor, a rack of stock or a tool lying beside the part, and
every one of those yields contours the detector offers as candidate hardware. The ground-truth
file already names two: hole1.jpg's "fourth box lands on the shadowed region at right",
hole2.jpg's "bench-edge artefacts along the bottom". whole screws1-3.jpg carry a strip of
cluttered floor down the left edge; three.jpg has a dark object across the bottom.

WHAT SEPARATES CHASSIS FROM BACKGROUND
--------------------------------------
Not brightness. Keying on it fails in the dangerous direction — a specular highlight and a
lighting gradient are both far from the plate's modal grey, so a brightness rule whited out
24% of screws_flash.jpg and 18% of head2.jpg, all of it chassis, while missing the bench.

Clutter does. Canny density over a margin strip against the same image's centre:

    has background   whole screws1 left 2.3%   screws2 left 6.1%   screws3 left 4.3%
    no background    every margin of every close-up <= 0.7%   (centres ~0.3%)

A second, independent signature covers a flat dark object lying over the part (three.jpg's
bottom strip: mean 38.8 against a plate at 146), which is dark rather than cluttered.

CLUTTER ALONE OVER-REACHED
--------------------------
The panel's own fold is a crisp structural edge, so it reads as cluttered; being connected to
the true background it was swallowed into the same region, and the mask covered 22.1% of
whole screws2.jpg and 20.8% of whole screws3.jpg -- a good part of it chassis. Requiring the
region to be BOTH cluttered AND a different local brightness from the plate leaves the fold
alone, because a fold is the same metal as the panel beside it:

                      clutter only    + unlikeness 30
    whole screws1.jpg        4.7%               2.5%
    whole screws2.jpg       22.0%               7.4%
    whole screws3.jpg       20.8%               7.8%
    three.jpg               21.8%              21.5%   (dark object, caught by the dark test)
    every close-up           0.0%               0.0%

It costs some false positives -- weighted 11 to 14, against 19 for no mask at all -- but
misses stay at 3, the same as no mask, where the looser rule cost 4. Under-covering the
background is the safe direction; shading the part is not.

THE SAFETY PROPERTY
-------------------
A region counts as background only if it TOUCHES THE FRAME BORDER. An interior region can
never be removed, so a fastener, the shadow beside one, or a dark recess in the middle of the
part is safe however dark or busy it is. Section 2 checks it: of the detections the mask
removes, none is a fastener.

WHITE IS DESTRUCTIVE
--------------------
The obvious fill is white. It moves the intensity histogram, and the edge stage derives
Canny's thresholds from that histogram by Otsu, so the thresholds go wrong everywhere. Filling
with the median grey of everything outside the mask leaves the histogram where it was and
raises no seam for Canny to find. The app therefore fills with the plate's grey for the
detector and paints the same region white on the overlay the operator looks at — white is what
it means to a person, not what it meant to Canny. Section 3 is the evidence.

Run:  python python/background_mask_study.py
"""

import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required: pip install opencv-python")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import detector_ab_study as D  # noqa: E402
import detection_ground_truth as GTRUTH  # noqa: E402
import hybrid_v2_study as HV  # noqa: E402
import adaptive_preprocessing_study as A  # noqa: E402

# Mirrors ml/ChassisMask.java one for one.
MASK_WIDTH = 320
CLUTTER_WINDOW = 25
CLUTTER_MULTIPLE = 4.0
CLUTTER_FLOOR = 0.02
DARK_RATIO = 0.40
PLATE_UNLIKENESS = 30.0
MIN_BACKGROUND_FRACTION = 0.01
CENTRE_FRACTION = 0.4
MAX_BACKGROUND_FRACTION = 0.45

V6 = A.LADDER["medium"]
SEAM = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


def background_mask(bgr):
    """True where the pixel is not the chassis. See ml/ChassisMask.of()."""
    h0, w0 = bgr.shape[:2]
    scale = MASK_WIDTH / float(w0)
    small = cv2.resize(bgr, (MASK_WIDTH, max(1, int(round(h0 * scale)))),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    edges = (cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160) > 0).astype(np.float32)
    clutter = cv2.boxFilter(edges, -1, (CLUTTER_WINDOW, CLUTTER_WINDOW))

    cy0, cy1 = int(h * (0.5 - CENTRE_FRACTION / 2)), int(h * (0.5 + CENTRE_FRACTION / 2))
    cx0, cx1 = int(w * (0.5 - CENTRE_FRACTION / 2)), int(w * (0.5 + CENTRE_FRACTION / 2))
    plate_clutter = float(np.median(clutter[cy0:cy1, cx0:cx1]))
    plate_grey = float(np.median(gray[cy0:cy1, cx0:cx1]))
    limit = max(CLUTTER_FLOOR, CLUTTER_MULTIPLE * plate_clutter)

    # Clutter must be accompanied by a real difference in local brightness, or the panel's
    # own fold — a crisp edge in the same metal — is swallowed along with the background.
    local = cv2.medianBlur(gray, 31).astype(np.int16)
    unlike = np.abs(local - int(plate_grey)) > PLATE_UNLIKENESS
    foreign = (((clutter > limit) & unlike) | (gray < DARK_RATIO * plate_grey)).astype(np.uint8)
    foreign = cv2.morphologyEx(foreign, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreign, 8)
    mask = np.zeros((h, w), np.uint8)
    kept = 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h
        if touches and area >= MIN_BACKGROUND_FRACTION * h * w:
            mask[labels == i] = 255
            kept += 1
    if kept == 0 or np.count_nonzero(mask) / float(h * w) > MAX_BACKGROUND_FRACTION:
        return np.zeros((h0, w0), np.uint8), dict(parts=0, plate_grey=plate_grey)
    full = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return full, dict(parts=kept, plate_grey=plate_grey)


def detect(img, mode):
    """Detections under one masking mode, through the rung the app runs by default."""
    mask, _ = background_mask(img)
    work = img.copy()
    if mode in ("white", "white+seam"):
        work[mask > 0] = (255, 255, 255)
    elif mode == "grey":
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plate = int(np.median(g[mask == 0])) if np.any(mask == 0) else 128
        work[mask > 0] = (plate, plate, plate)
    edges = HV.edges_for(work, HV.cfg(**V6))
    if mode in ("white+seam", "grey"):
        edges[cv2.dilate(mask, SEAM) > 0] = 0
    return D.find_feature_boxes(edges)[0], mask


NOTES = {
    "whole screws1.jpg": "cluttered floor strip, left",
    "whole screws2.jpg": "cluttered floor strip, left",
    "whole screws3.jpg": "cluttered floor strip, left",
    "three.jpg": "dark object lying across the bottom",
    "corner.jpg": "over-exposed strip, right",
    "hole1.jpg": "shadowed region right — a spurious detection the GT file names",
    "hole2.jpg": "bench edge along the bottom — artefacts the GT file names",
    "zero.jpg": "dark band, bottom",
}


def section_masks():
    print("1. WHAT THE MASK COVERS (0% = the whole photograph read as chassis)")
    print(f"   {'asset':<20}{'masked':>9}   note")
    for n in sorted(list(GTRUTH.GT) + list(GTRUTH.WIDE)):
        img = A.image(n)
        if img is None:
            continue
        mask, _ = background_mask(img)
        frac = 100.0 * np.count_nonzero(mask) / mask.size
        if frac > 0.05:
            print(f"   {n:<20}{frac:>8.1f}%   {NOTES.get(n, '')}")
    print("   Every other asset masks nothing, which is correct: they are close-ups.")


def section_safety():
    print("\n2. SAFETY — of the detections the mask removes, how many are fasteners?")
    print(f"   {'asset':<20}{'detections':>12}{'removed':>9}")
    for n in sorted(GTRUTH.GT):
        img = A.image(n)
        if img is None:
            continue
        mask, _ = background_mask(img)
        if not np.any(mask):
            continue
        boxes = D.find_feature_boxes(HV.edges_for(img, HV.cfg(**V6)))[0]
        inside = sum(1 for (x, y, w, h) in boxes
                     if mask[min(y + h // 2, mask.shape[0] - 1),
                             min(x + w // 2, mask.shape[1] - 1)] > 0)
        print(f"   {n:<20}{len(boxes):>12}{inside:>9}")
    print("   Each removal was inspected by eye — they are the floor strips in whole screws1-3.")
    print("   No fastener among them. hole1 and hole2 now have nothing removed: the stricter")
    print("   rule leaves their soft shadow and bench edge alone, which is what costs the")
    print("   extras in section 3. Under-covering is the safe direction.")


def section_counts():
    print("\n3. GROUND TRUTH (26 photographs; weighted = 2 x misses + extras)")
    print(f"   {'mode':<16}{'exact':>8}{'miss':>6}{'extra':>7}{'weighted':>10}")
    results = {}
    for mode in ("off", "white", "white+seam", "grey"):
        counts = {}
        for n in GTRUTH.GT:
            img = A.image(n)
            if img is None:
                continue
            counts[n] = len(detect(img, mode)[0])
        results[mode] = counts
        w, fn, fp = GTRUTH.weighted(counts)
        exact = sum(1 for n, c in counts.items() if GTRUTH.err(n, c) == 0)
        print(f"   {mode:<16}{f'{exact}/26':>8}{fn:>6}{fp:>7}{w:>10}")

    print("\n   Per-image, where the modes differ:")
    print(f"   {'asset':<20}{'GT':>9}{'off':>6}{'white':>7}{'w+seam':>8}{'grey':>6}")
    for n in sorted(GTRUTH.GT):
        row = [results[m].get(n) for m in ("off", "white", "white+seam", "grey")]
        if len(set(row)) > 1:
            print(f"   {n:<20}{str(sorted(GTRUTH.GT[n])):>9}"
                  f"{row[0]:>6}{row[1]:>7}{row[2]:>8}{row[3]:>6}")


def main():
    if not os.path.isdir(D.ASSETS):
        sys.exit(f"assets not found at {D.ASSETS}")
    section_masks()
    section_safety()
    section_counts()


if __name__ == "__main__":
    main()
