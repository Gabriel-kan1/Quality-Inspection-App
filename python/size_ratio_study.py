"""
How much a fastener's detected size varies between two photographs of the same part — the
noise floor `FeatureLocalisation.SIZE_TOLERANCE` has to sit above.

WHY THIS EXISTS
---------------
The dimension check used to need a printed square marker in frame to establish mm/px, and
without one it reported nothing. The alignment gate now holds every inspection to roughly
the reference's framing, and `LayoutMatcher` recovers the residual scale exactly, so a
slot's size can be compared with the detection filling it *in pixels* — reference box
against inspection box, scaled by the fitted transform — with no marker at all.

That comparison is only as good as the detector's repeatability. Boxes come out of Canny →
dilate → close → contour → merge, and the same bolt head photographed twice does not come
out the same number of pixels wide twice. A tolerance set below that jitter flags every
part; one set far above it misses the wrong bolt. This script measures the jitter.

METHOD
------
Ground truth is `pair_ground_truth.py`'s: a SIFT + RANSAC homography between two photos,
verified for sanity, computed from image texture with no knowledge of the detector. Mapping
one photo's detection centres through it and taking mutual nearest neighbours yields "these
two boxes are the same physical thing" independently of anything under test.

For each such correspondence, the study computes exactly what the app will:

    ratio = sqrt( area_b / (area_a * s^2) )

where `s` is the isotropic scale of a similarity (and separately an affine) fitted over the
pair's correspondences — the same models `LayoutMatcher` fits. A ratio of 1 means the
detection is the size the reference predicts; the distribution of |ln ratio| across genuine
same-hardware pairs is the noise. The raw, unnormalised pixel ratio is reported alongside to
show what the alignment-scale correction is worth.

WHAT THIS CANNOT MEASURE
------------------------
Correspondences are of *detections*, not of confirmed fasteners: a scratch that the detector
finds in both photos counts too. That makes the sample slightly pessimistic — scratches have
no fixed extent — which is the safe direction for a tolerance. And, as in the localisation
study, the pairs are the ones the homography can prove, so the hardest cases (different
hardware, same part) are absent by construction.

Run:  python python/size_ratio_study.py
      python python/size_ratio_study.py --verbose

Needs opencv-python and numpy, plus the gitignored assets in
app/src/main/assets/chassis_images/.
"""

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required: pip install opencv-python")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector_ab_study import (  # noqa: E402
    ASSETS, compute_edges, find_feature_boxes, interchangeable_groups, load_working_image,
)
from localisation_tolerance_study import (  # noqa: E402
    MIN_CORRESPONDENCES, PAIRING_RADIUS_FRAC, apply_h, correspond, fit_affine,
    fit_similarity, homography,
)

# Candidate values for FeatureLocalisation.SIZE_TOLERANCE, as a fraction of linear size.
CANDIDATES = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 1.00)
# The Java constants under test.
CURRENT_TOLERANCE = 0.50
CURRENT_MIN_PX = 30
# Boxes are binned by the shorter side of the smaller of the two, in working-image pixels.
# Below CURRENT_MIN_PX the detector's boxes are fragments and merge chains far more often
# than they are hardware, and their sizes do not repeat; the app refuses to judge a slot
# whose reference box is that small rather than flag it.
SIZE_BINS = ((0, CURRENT_MIN_PX, f"<{CURRENT_MIN_PX}px"),
             (CURRENT_MIN_PX, 10 ** 6, f">={CURRENT_MIN_PX}px"))


def boxes_of(name):
    """Every detection as fractional [cx, cy, w, h] plus its pixel size, via the shipped pipeline."""
    img = load_working_image(Path(ASSETS) / name)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    edges, _ = compute_edges(img)
    boxes, _ = find_feature_boxes(edges)
    frac = np.array([[(x + bw / 2.0) / w, (y + bh / 2.0) / h, bw / w, bh / h]
                     for x, y, bw, bh in boxes], dtype=np.float64)
    px = np.array([[bw, bh] for _, _, bw, bh in boxes], dtype=np.float64)
    return frac, px


def analyse(pair, H, verbose):
    frac_a, px_a = boxes_of(pair[0])
    frac_b, px_b = boxes_of(pair[1])
    if frac_a is None or frac_b is None or len(frac_a) == 0 or len(frac_b) == 0:
        return None
    pred = apply_h(H, frac_a[:, :2])
    matches = correspond(pred, frac_b[:, :2], PAIRING_RADIUS_FRAC)
    if len(matches) < MIN_CORRESPONDENCES:
        return None

    ia = [i for i, _ in matches]
    ib = [j for _, j in matches]
    src, dst = frac_a[ia], frac_b[ib]
    u = src[:, :2] - src[:, :2].mean(axis=0)
    v = dst[:, :2] - dst[:, :2].mean(axis=0)

    area_a = src[:, 2] * src[:, 3]
    area_b = dst[:, 2] * dst[:, 3]
    row = {"pair": pair, "n": len(matches)}
    for model, fit in (("similarity", fit_similarity), ("affine", fit_affine)):
        _, scale = fit(u, v)
        if not scale or scale < 1e-9:
            return None
        # The exact quantity the app computes: found area over the area the reference box
        # would have at the alignment's scale, square-rooted to a linear ratio.
        row[model] = np.sqrt(area_b / (area_a * scale * scale))
        row[model + "_scale"] = scale

    # Without the correction — what a raw pixel comparison would see, given only that the
    # alignment gate held framing to its 0.85-1.35 band.
    row["raw"] = np.sqrt((px_b[ib, 0] * px_b[ib, 1]) / (px_a[ia, 0] * px_a[ia, 1]))
    row["px_a"] = px_a[ia]
    row["px_b"] = px_b[ib]

    if verbose:
        dev = np.abs(np.log(row["similarity"]))
        print(f"    {pair[0]} + {pair[1]}: {len(matches)} boxes, "
              f"scale {row['similarity_scale']:.3f}, "
              f"size ratio median {np.median(row['similarity']):.3f}, "
              f"worst {np.exp(dev.max()):.2f}x")
    return row


def report(rows, key, title):
    devs, short = [], []
    for r in rows:
        for k in range(r["n"]):
            devs.append(abs(np.log(r[key][k])))
            short.append(min(r["px_a"][k].min(), r["px_b"][k].min()))
    devs, short = np.array(devs), np.array(short)

    print(f"\n=== {title}: {len(devs)} same-detection pairs over {len(rows)} photo pairs ===")
    print("  |ln(found / expected)|, shown as the factor either way; 1.00 is a perfect match")
    print("  short side       n    p50    p90    p95   " + "  ".join(
        f">{1 + t:.2f}x" for t in CANDIDATES))
    judged = None
    for lo, hi, label in SIZE_BINS + ((0, 10 ** 6, "all"),):
        m = (short >= lo) & (short < hi)
        d = devs[m]
        if not len(d):
            continue
        # The app's test: |ln ratio| > ln(1 + t). Symmetric in log space, so a box 30% too
        # small is as far out as one 30% too large, i.e. 1/1.3 rather than 0.7.
        flagged = "  ".join(f"{100.0 * (d > np.log1p(t)).mean():5.0f}%" for t in CANDIDATES)
        print(f"  {label:<12} {len(d):5d}  x{np.exp(np.median(d)):.2f}  "
              f"x{np.exp(np.percentile(d, 90)):.2f}  x{np.exp(np.percentile(d, 95)):.2f}  {flagged}")
        if lo == CURRENT_MIN_PX:
            judged = d
    return judged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true", help="list every measured pair")
    args = ap.parse_args()

    if not os.path.isdir(ASSETS):
        sys.exit(f"assets not found at {ASSETS}")
    factory, norm = (lambda: cv2.SIFT_create(3000)), cv2.NORM_L2

    names = sorted(f for f in os.listdir(ASSETS) if f.lower().endswith((".jpg", ".png")))
    print(f"{len(names)} assets, {len(names) * (len(names) - 1) // 2} pairs, "
          f"ground truth by SIFT + RANSAC homography\n")

    rows = []
    for a, b in itertools.combinations(names, 2):
        H = homography(a, b, factory, norm)
        if H is None:
            continue
        row = analyse((a, b), H, args.verbose)
        if row is not None:
            rows.append(row)
    if not rows:
        sys.exit("no pair yielded enough correspondences to measure")

    scales = np.array([r["similarity_scale"] for r in rows])
    print(f"\n{len(rows)} pairs measured; alignment scale between them ranges "
          f"{scales.min():.2f}-{scales.max():.2f} (median {np.median(scales):.2f})")

    px = np.concatenate([r["px_a"] for r in rows])
    print(f"  detection boxes measured: median {np.median(px[:, 0]):.0f}x"
          f"{np.median(px[:, 1]):.0f} px at the 1280 working size, "
          f"smallest {px.min(axis=0)[0]:.0f}x{px.min(axis=0)[1]:.0f}")

    report(rows, "raw", "ALL verified pairs, RAW pixel ratio, no scale correction")
    report(rows, "similarity", "ALL verified pairs, corrected by the SIMILARITY fit's scale")
    report(rows, "affine", "ALL verified pairs, corrected by the AFFINE fit's scale")

    # What the app sees: the alignment gate refuses a frame whose framing differs from the
    # reference by more than its band, so pairs the gate would not let through (a tight crop
    # against a whole-part shot, at scale x2.5) do not describe inspection conditions.
    inter = set()
    for group in interchangeable_groups().values():
        inter.update(itertools.combinations(group, 2))
    gated = [r for r in rows if tuple(r["pair"]) in inter]
    judged = report(gated, "similarity",
                    "GATE-INTERCHANGEABLE pairs only, corrected by the SIMILARITY fit's scale")

    print("\n=== Verdict ===")
    if judged is not None and len(judged):
        p95 = np.exp(np.percentile(judged, 95))
        flagged = 100.0 * (judged > np.log1p(CURRENT_TOLERANCE)).mean()
        print(f"  judged population (gate-interchangeable, >= {CURRENT_MIN_PX}px): "
              f"p95 x{p95:.2f}; the shipping tolerance x{1 + CURRENT_TOLERANCE:.2f} "
              f"flags {flagged:.1f}% of genuine same-hardware boxes")
    print("  The tolerance has to clear the judged p95 to stop flagging good parts, and")
    print("  everything between the median and it is size change a wrong fastener can hide")
    print("  inside: adjacent metric sizes differ by x1.14-1.25, which is inside the noise.")
    print("  Below the pixel floor the boxes do not repeat at all, so the app refuses to")
    print("  judge such slots rather than report their jitter as a finding.")


if __name__ == "__main__":
    main()
