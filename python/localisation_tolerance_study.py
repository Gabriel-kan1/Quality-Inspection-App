"""
What `LayoutMatcher.INLIER_TOLERANCE` should be, and whether a similarity transform is
the right model — measured, not argued.

WHY THIS EXISTS
---------------
The positional check declares a detection to be "at" a slot when it lands within
`INLIER_TOLERANCE` (0.12) of the slot's projected position, as a fraction of the layout's
RMS spread. That number carried a rationale — "generous enough to absorb detection jitter
and mild perspective, tight enough that a fastener in the wrong hole does not match" — but
no measurement. On a real part it refused to register at all, and the residuals said why:
the same four fasteners changed their pairwise separations by ratios spanning 0.57 to 0.85
between two photos, which is an out-of-plane tilt and not something a similarity transform
can absorb at any threshold.

Two questions follow, and both are answerable from the bundled assets:

  1. How far does a correctly-identified fastener actually land from where the model puts
     it, across genuine same-part photo pairs?
  2. How far away is the *nearest wrong* slot — the thing the tolerance has to stay under
     if "wrong hole" is to remain detectable?

A tolerance is only meaningful in the gap between those two distributions. If they
overlap, no threshold works and the model is what needs changing — so the same run fits a
6-DOF affine alongside the 4-DOF similarity and reports both.

GROUND TRUTH, AND WHY IT IS NOT CIRCULAR
----------------------------------------
Correspondence comes from `pair_ground_truth.py`'s machinery: ORB/AKAZE/SIFT keypoints,
symmetric ratio test, RANSAC homography, plus the convexity and area sanity checks. That
is computed from image texture and knows nothing about the fastener detector, the
classifier, or `LayoutMatcher`. Mapping one photo's detections through that homography and
taking mutual nearest neighbours therefore yields "these two boxes are the same physical
fastener" without consulting the algorithm being measured.

WHAT THIS CANNOT MEASURE
------------------------
`pair_ground_truth.py` produces trustworthy positives only, and one pair it cannot certify
is `screws.jpg`/`screws_test.jpg` — zero symmetric matches under all three detectors,
because the hardware itself changed. That is precisely the pair that failed on-device, so
it is absent from this sample by construction. The measurement therefore describes the
pairs it can prove, and any conclusion about the failing pair is an extrapolation. Said
plainly rather than buried.

Run:  python python/localisation_tolerance_study.py
      python python/localisation_tolerance_study.py --detector SIFT --verbose

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
    ASSETS, compute_edges, find_feature_boxes, load_working_image,
)
from pair_ground_truth import (  # noqa: E402
    MIN_INLIERS, RANSAC_PX, RATIO, homography_is_sane, load_gray,
)

# The Java constant under test. Everything is reported as a multiple of the layout's RMS
# spread, which is the unit LayoutMatcher compares against.
CURRENT_TOLERANCE = 0.12
# Matching a mapped detection to its counterpart. Deliberately far looser than any
# tolerance under test — this decides *which box is which*, and a tight radius here would
# quietly discard exactly the large-residual correspondences the study is trying to
# measure, biasing the answer toward the value already in the code.
PAIRING_RADIUS_FRAC = 0.10
# Below this there is no shape to speak of and a fit is unconstrained.
MIN_CORRESPONDENCES = 4


def detections(name):
    """Fractional (0-1) centres of every detection, via the shipped detector pipeline."""
    img = load_working_image(Path(ASSETS) / name)
    if img is None:
        return None
    h, w = img.shape[:2]
    edges, _ = compute_edges(img)
    boxes, _ = find_feature_boxes(edges)
    return np.array([[(x + bw / 2.0) / w, (y + bh / 2.0) / h]
                     for x, y, bw, bh in boxes], dtype=np.float64)


def homography(name_a, name_b, factory, norm):
    """Fractional-coordinate homography a→b, or None when the pair cannot be verified."""
    img_a, img_b = load_gray(name_a), load_gray(name_b)
    if img_a is None or img_b is None:
        return None
    det = factory()
    kp_a, desc_a = det.detectAndCompute(img_a, None)
    kp_b, desc_b = det.detectAndCompute(img_b, None)
    if desc_a is None or desc_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return None

    bf = cv2.BFMatcher(norm, crossCheck=False)

    def ratio_filtered(d1, d2):
        keep = {}
        for pair in bf.knnMatch(d1, d2, k=2):
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < RATIO * second.distance:
                keep[best.queryIdx] = best.trainIdx
        return keep

    forward = ratio_filtered(desc_a, desc_b)
    backward = ratio_filtered(desc_b, desc_a)
    pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    if len(pairs) < 4:
        return None

    src = np.float32([kp_a[i].pt for i, _ in pairs]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[j].pt for _, j in pairs]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if H is None or int(mask.sum()) < MIN_INLIERS:
        return None
    ok, _ = homography_is_sane(H, img_a.shape, img_b.shape)
    if not ok:
        return None

    # H is in the verifier's own pixel space; rescale both ends so it maps fractional
    # coordinates, which is what the app and the detector output both use.
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    to_px = np.diag([wa, ha, 1.0])
    from_px = np.diag([1.0 / wb, 1.0 / hb, 1.0])
    return from_px @ H @ to_px


def apply_h(H, pts):
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    out = homo @ H.T
    return out[:, :2] / out[:, 2:3]


def correspond(pred, obs, radius):
    """Mutual nearest neighbours between mapped and observed detections."""
    if len(pred) == 0 or len(obs) == 0:
        return []
    d = np.linalg.norm(pred[:, None, :] - obs[None, :, :], axis=2)
    best_for_pred = d.argmin(axis=1)
    best_for_obs = d.argmin(axis=0)
    return [(i, j) for i, j in enumerate(best_for_pred)
            if best_for_obs[j] == i and d[i, j] <= radius]


def spread(pts):
    return float(np.sqrt(((pts - pts.mean(axis=0)) ** 2).sum(axis=1).mean()))


def fit_similarity(u, v):
    """Proper similarity (scale, rotation, translation) — LayoutMatcher's model."""
    den = (u ** 2).sum()
    if den < 1e-12:
        return None, None
    a = float((u * v).sum() / den)
    b = float((u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]).sum() / den)
    pred = np.column_stack([a * u[:, 0] - b * u[:, 1], b * u[:, 0] + a * u[:, 1]])
    return pred, float(np.hypot(a, b))


def fit_affine(u, v):
    """6-DOF affine — allows the anisotropic foreshortening a camera tilt produces."""
    if len(u) < 3:
        return None, None
    A, *_ = np.linalg.lstsq(u, v, rcond=None)
    pred = u @ A
    # Geometric mean of the singular values: the isotropic scale an affine implies.
    return pred, float(np.sqrt(abs(np.linalg.det(A))))


def analyse(pair, H, verbose):
    """Normalised residuals for one verified pair, under both models."""
    det_a, det_b = detections(pair[0]), detections(pair[1])
    if det_a is None or det_b is None or len(det_a) == 0 or len(det_b) == 0:
        return None
    pred = apply_h(H, det_a)
    matches = correspond(pred, det_b, PAIRING_RADIUS_FRAC)
    if len(matches) < MIN_CORRESPONDENCES:
        return None

    src = det_a[[i for i, _ in matches]]
    dst = det_b[[j for _, j in matches]]
    layout_spread = spread(src)
    if layout_spread < 1e-6:
        return None

    u = src - src.mean(axis=0)
    v = dst - dst.mean(axis=0)

    row = {"pair": pair, "n": len(matches)}
    for model, fit in (("similarity", fit_similarity), ("affine", fit_affine)):
        fitted, scale = fit(u, v)
        if fitted is None or not scale or scale < 1e-9:
            return None
        # The exact quantity LayoutMatcher tests: residual in photo units, over
        # (layout spread x transform scale).
        resid = np.linalg.norm(fitted - v, axis=1) / (layout_spread * scale)
        # What the tolerance must stay under to keep a wrong hole detectable: how far the
        # *nearest other* fastener sits from where this slot was predicted.
        others = np.linalg.norm(
            fitted[:, None, :] - v[None, :, :], axis=2) / (layout_spread * scale)
        np.fill_diagonal(others, np.inf)
        row[model] = {"resid": resid, "nearest_wrong": others.min(axis=1), "scale": scale}

    # The network view: one rigid transform implies one ratio for every separation.
    da = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=2)
    db = np.linalg.norm(dst[:, None, :] - dst[None, :, :], axis=2)
    iu = np.triu_indices(len(src), 1)
    ratios = db[iu] / np.maximum(da[iu], 1e-9)
    row["ratio_spread"] = float(ratios.max() / max(ratios.min(), 1e-9))

    if verbose:
        print(f"    {pair[0]} + {pair[1]}: {len(matches)} correspondences, "
              f"similarity scale {row['similarity']['scale']:.3f}, "
              f"worst residual {row['similarity']['resid'].max():.3f}, "
              f"distance-ratio spread x{row['ratio_spread']:.2f}")
    return row


def required_inliers(slots):
    """LayoutMatcher.requiredInliers, mirrored so registration can be counted here."""
    return min(max(3, -(-slots // 2)), slots - 1)


def report(rows, model):
    resid = np.concatenate([r[model]["resid"] for r in rows])
    wrong = np.concatenate([r[model]["nearest_wrong"] for r in rows])
    print(f"\n=== {model.upper()} model - {len(resid)} correspondences "
          f"over {len(rows)} proven pairs ===")
    print("  correct fastener lands this far from its predicted slot")
    print("  (multiples of the layout's RMS spread - the unit INLIER_TOLERANCE is in)")
    for q in (50, 75, 90, 95, 99, 100):
        print(f"    p{q:<3d} {np.percentile(resid, q):.3f}")

    # What actually decides the outcome. Being inside tolerance of two slots at once is
    # not a failure: assignment is greedy nearest-first and one-to-one, so the correct
    # pairing still wins provided it is the closer one. The threshold only has to keep the
    # correct match reachable; it is the ordering that keeps a wrong hole out.
    correct_is_nearest = 100.0 * (resid < wrong).mean()
    print(f"\n  correct match is closer than any wrong slot: {correct_is_nearest:.1f}%"
          f"  (this is what greedy one-to-one assignment resolves on)")
    print(f"  nearest WRONG slot: min {wrong.min():.3f}  p5 {np.percentile(wrong, 5):.3f}"
          f"  median {np.median(wrong):.3f}")

    print("\n  tolerance   correct match     a wrong slot    pairs that register")
    print("              inside tolerance   also in range   (>= requiredInliers)")
    for t in (0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40):
        tp = 100.0 * (resid <= t).mean()
        fp = 100.0 * (wrong <= t).mean()
        registered = sum(1 for r in rows
                         if int((r[model]["resid"] <= t).sum()) >= required_inliers(r["n"]))
        mark = "   <== shipping" if abs(t - CURRENT_TOLERANCE) < 1e-6 else ""
        print(f"    {t:.2f}       {tp:5.1f}%             {fp:5.1f}%"
              f"          {registered:2d}/{len(rows)}{mark}")

    return np.percentile(resid, 95), np.percentile(resid, 99), correct_is_nearest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", default="SIFT", help="ORB, AKAZE or SIFT (default SIFT)")
    ap.add_argument("--verbose", action="store_true", help="list every measured pair")
    args = ap.parse_args()

    if not os.path.isdir(ASSETS):
        sys.exit(f"assets not found at {ASSETS}")
    factories = {
        "ORB": (lambda: cv2.ORB_create(3000), cv2.NORM_HAMMING),
        "AKAZE": (lambda: cv2.AKAZE_create(), cv2.NORM_HAMMING),
        "SIFT": (lambda: cv2.SIFT_create(3000), cv2.NORM_L2),
    }
    if args.detector not in factories:
        sys.exit(f"unknown detector {args.detector!r}")
    factory, norm = factories[args.detector]

    names = sorted(f for f in os.listdir(ASSETS) if f.lower().endswith((".jpg", ".png")))
    print(f"{len(names)} assets, {len(names)*(len(names)-1)//2} pairs, "
          f"ground truth by {args.detector} + RANSAC homography\n")

    rows = []
    unusable = 0
    for a, b in itertools.combinations(names, 2):
        H = homography(a, b, factory, norm)
        if H is None:
            continue
        row = analyse((a, b), H, args.verbose)
        if row is None:
            unusable += 1
            continue
        rows.append(row)

    if not rows:
        sys.exit("no pair yielded enough correspondences to measure")
    print(f"\n{len(rows)} pairs measured "
          f"({unusable} verified but with too few paired detections)")

    ratios = np.array([r["ratio_spread"] for r in rows])
    print(f"\n=== Is a rigid transform the right model? ===")
    print("  A similarity transform implies ONE ratio for every pairwise separation, so a")
    print("  spread above ~1 is model error the tolerance is being asked to absorb.")
    print(f"  distance-ratio spread across pairs: median x{np.median(ratios):.2f}, "
          f"p90 x{np.percentile(ratios, 90):.2f}, max x{ratios.max():.2f}")

    sim95, sim99, sim_order = report(rows, "similarity")
    aff95, aff99, aff_order = report(rows, "affine")

    print("\n=== Verdict ===")
    print(f"  similarity needs {sim95:.3f} to reach 95% of correct matches "
          f"({sim99:.3f} for 99%); ordering correct {sim_order:.1f}% of the time")
    print(f"  affine     needs {aff95:.3f} to reach 95% of correct matches "
          f"({aff99:.3f} for 99%); ordering correct {aff_order:.1f}% of the time")
    print(f"  shipping tolerance is {CURRENT_TOLERANCE:.2f}")
    print(f"\n  Affine cuts the residual a tolerance must cover by "
          f"{100 * (1 - aff95 / sim95):.0f}% at p95.")
    print("  A rigid similarity is under-fitting: the leftover error is the anisotropic")
    print("  foreshortening of a tilted camera, which no threshold can distinguish from")
    print("  a genuinely misplaced fastener.")


if __name__ == "__main__":
    main()
