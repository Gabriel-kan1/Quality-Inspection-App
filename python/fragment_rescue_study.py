"""
Letting a fastener's own fragments survive the dynamic area floor.

THE DEFECT
----------
`findFeatureBoxes` filters contours by area BEFORE `mergeNearbyBoxes` reunites the pieces of
a fragmented feature. So a low-contrast outline that Canny broke into pieces has those pieces
judged individually against a floor sized for whole fasteners, and they are discarded — even
where the merge's own guards would happily have rejoined them.

Found on `screws_none.jpg`, which holds two identical studs. The shipped stack boxes one at
68x68 and the other at 44x49 — a 47% disagreement between two instances of the same hardware
in one photograph, at one distance, under one detector. Traced to the contour:

    slot 5 (correct)   collar 51x51 area 890  + pin 50x44 area 900   -> both >= floor 468
    slot 3 (cut short) collar 44x49 area 856  + pin 44x40 area 366   -> pin DISCARDED
                                              + pin 23x22 area 356   -> pin DISCARDED

The pin at slot 3 sits at 168.8 grey against a 171.2 plate (2.5 levels, Michelson 0.007)
where slot 5's is 11.2 levels, so Canny fragments it and dilate/close cannot bridge the gaps.
Both discarded fragments touch the surviving collar box (gap 0 px) and the merge's fill-ratio
guard would have accepted them at 0.94, well above its 0.7 threshold. The box should have
been 67x70.

The consequence is not a miscount — the fastener is still one box — but a wrong SIZE, which
is what the marker-free dimension check compares. Inspected against `screws_flash.jpg`, where
the same stud is well lit and boxes at 76x68, slot 3 read +62% and was flagged while its
identical twin read +8%.

THE FIX
-------
A contour below the floor that touches one above it is a fragment of that feature, not noise
of its own. Such a fragment is carried into the merge, but may never become a box of its own:
the anchor flag is OR-ed through every merge, and any group that never reached a box above
the floor is dropped at the end. So the change can grow an existing box and can never add a
detection — which is the property that makes it safe to measure against count ground truth.

Deliberately NOT chosen: applying the area floor after the merge. That lets a chain of
sub-floor noise fragments merge into each other and cross the floor collectively, which is
the snowballing failure `mergeNearbyBoxes`' fill-ratio guard was written to stop.

WHAT HAD TO BE RE-MEASURED
--------------------------
The geometry filter is the scorer every method in EdgeMethodLab is judged by, so a change
here moves every number in hybrid_v2_study section 6, detector_ab_study and size_ratio_study
at once. All of them are re-run below, for the shipped stack and all three adaptive rungs.

Run:  python python/fragment_rescue_study.py
      python python/fragment_rescue_study.py --quick    (skips the pair study, section 3)

Needs opencv-python and numpy, plus the gitignored assets in
app/src/main/assets/chassis_images/.
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

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
from localisation_tolerance_study import (  # noqa: E402
    MIN_CORRESPONDENCES, PAIRING_RADIUS_FRAC, apply_h, correspond, fit_similarity, homography,
)

# The edge stages the filter serves. "shipped" is compute_edges itself.
STAGES = {"shipped": None, "light": A.LADDER["light"],
          "medium (v6)": A.LADDER["medium"], "strong": A.LADDER["strong"]}
MIN_JUDGED_PX = 30          # FeatureLocalisation.MIN_SIZE_JUDGED_PX
SIZE_TOLERANCE = 0.50       # FeatureLocalisation.SIZE_TOLERANCE


def edges_of(name, cfg):
    img = A.image(name)
    return (D.compute_edges(img)[0] if cfg is None else HV.edges_for(img, HV.cfg(**cfg))), img


def boxes_of(name, cfg, rescue):
    edges, img = edges_of(name, cfg)
    return D.find_feature_boxes(edges, rescue=rescue)[0], img


# ─────────────────────────────────────────────
# 0. The motivating case
# ─────────────────────────────────────────────
def section_case():
    print("0. THE MOTIVATING CASE — two identical studs in screws_none.jpg")
    print("   (slot 3 at frac 0.405,0.433; slot 5 at 0.463,0.333; boxes in working pixels)")
    print(f"   {'stage':<14}{'rescue':<8}{'slot 3':>12}{'slot 5':>12}{'slot3/slot5':>13}{'boxes':>7}")
    for stage, cfg in STAGES.items():
        for rescue in (False, True):
            bxs, img = boxes_of("screws_none.jpg", cfg, rescue)
            h, w = img.shape[:2]

            def at(cx, cy):
                b = min(bxs, key=lambda r: np.hypot((r[0]+r[2]/2)/w - cx, (r[1]+r[3]/2)/h - cy))
                return b, np.sqrt(b[2]*b[3])
            b3, g3 = at(0.405, 0.433)
            b5, g5 = at(0.463, 0.333)
            print(f"   {stage:<14}{str(rescue):<8}{f'{b3[2]}x{b3[3]}':>12}{f'{b5[2]}x{b5[3]}':>12}"
                  f"{g3/g5:>13.2f}{len(bxs):>7}")
    print("   Two instances of one fastener should agree; 1.00 is correct.\n")


# ─────────────────────────────────────────────
# 1. Ground-truth counts
# ─────────────────────────────────────────────
def section_counts():
    print("1. GROUND TRUTH (26 reference-distance photographs; weighted = 2 x misses + extras)")
    print(f"   {'stage':<14}{'rescue':<8}{'exact':>8}{'miss':>6}{'extra':>7}{'weighted':>10}"
          f"{'boxes changed':>15}")
    for stage, cfg in STAGES.items():
        base = {}
        for rescue in (False, True):
            counts, changed = {}, 0
            for n in GTRUTH.GT:
                bxs, _ = boxes_of(n, cfg, rescue)
                counts[n] = len(bxs)
                if not rescue:
                    base[n] = sorted(bxs)
                else:
                    changed += sum(1 for b in sorted(bxs) if b not in base[n])
            w, fn, fp = GTRUTH.weighted(counts)
            exact = sum(1 for n, c in counts.items() if GTRUTH.err(n, c) == 0)
            print(f"   {stage:<14}{str(rescue):<8}{f'{exact}/26':>8}{fn:>6}{fp:>7}{w:>10}"
                  f"{(changed if rescue else 0):>15}")
    print("   The change can only grow a box, never add one, so a count that moves means a")
    print("   grown box absorbed a neighbour it now touches.\n")


# ─────────────────────────────────────────────
# 2. Stability between interchangeable photographs
# ─────────────────────────────────────────────
def section_stability():
    print("2. STABILITY (disagreement between photographs the alignment gate treats as equal)")
    print(f"   {'stage':<14}{'rescue':<8}{'stability':>11}{'totalDet':>10}")
    groups = D.interchangeable_groups()
    every = sorted(p.name for p in Path(D.ASSETS).glob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"))
    for stage, cfg in STAGES.items():
        for rescue in (False, True):
            res, total = {}, 0
            for a in every:
                if A.image(a) is None:
                    continue
                n = len(boxes_of(a, cfg, rescue)[0])
                res[a] = {"count": n}
                total += n
            stab, _ = D.stability(res, groups)
            print(f"   {stage:<14}{str(rescue):<8}{stab:>11.3f}{total:>10}")
    print()


# ─────────────────────────────────────────────
# 3. Size repeatability — the number this change exists to move
# ─────────────────────────────────────────────
def frac(bxs, w, h):
    return np.array([[(x+bw/2)/w, (y+bh/2)/h, bw/w, bh/h] for x, y, bw, bh in bxs]).reshape(-1, 4)


def section_sizes():
    print("3. SIZE REPEATABILITY on gate-interchangeable pairs (size_ratio_study's method:")
    print("   SIFT+RANSAC homography, mutual nearest neighbours, similarity-corrected ratio)")
    factory, norm = (lambda: cv2.SIFT_create(3000)), cv2.NORM_L2
    pairs = set()
    for group in D.interchangeable_groups().values():
        pairs.update(itertools.combinations(group, 2))
    H = {p: homography(p[0], p[1], factory, norm) for p in sorted(pairs)}
    usable = [p for p in sorted(pairs) if H[p] is not None]
    print(f"   {len(usable)} of {len(pairs)} pairs verified\n")
    print(f"   {'stage':<14}{'rescue':<8}{'n':>5}{'p50':>8}{'p90':>8}{'p95':>8}"
          f"{'% beyond x1.5':>15}")
    for stage, cfg in STAGES.items():
        for rescue in (False, True):
            devs = []
            for a, b in usable:
                ba, ia = boxes_of(a, cfg, rescue)
                bb, ib = boxes_of(b, cfg, rescue)
                if not len(ba) or not len(bb):
                    continue
                fa = frac(ba, ia.shape[1], ia.shape[0])
                fb = frac(bb, ib.shape[1], ib.shape[0])
                m = correspond(apply_h(H[(a, b)], fa[:, :2]), fb[:, :2], PAIRING_RADIUS_FRAC)
                if len(m) < MIN_CORRESPONDENCES:
                    continue
                i_, j_ = [i for i, _ in m], [j for _, j in m]
                u = fa[i_, :2] - fa[i_, :2].mean(axis=0)
                v = fb[j_, :2] - fb[j_, :2].mean(axis=0)
                _, s = fit_similarity(u, v)
                if not s or s < 1e-9:
                    continue
                for i, j in m:
                    # Same pixel floor the app applies before judging a slot's size.
                    if min(ba[i][2], ba[i][3]) < MIN_JUDGED_PX:
                        continue
                    r = np.sqrt((fb[j, 2]*fb[j, 3]) / (fa[i, 2]*fa[i, 3]*s*s))
                    devs.append(abs(np.log(r)))
            d = np.array(devs)
            if not len(d):
                print(f"   {stage:<14}{str(rescue):<8}{0:>5}")
                continue
            flagged = 100.0 * (d > np.log1p(SIZE_TOLERANCE)).mean()
            print(f"   {stage:<14}{str(rescue):<8}{len(d):>5}"
                  f"{np.exp(np.percentile(d, 50)):>7.2f}x{np.exp(np.percentile(d, 90)):>7.2f}x"
                  f"{np.exp(np.percentile(d, 95)):>7.2f}x{flagged:>14.1f}%")
    print("   Lower is better throughout: this is the noise floor SIZE_TOLERANCE sits above.\n")


# ─────────────────────────────────────────────
# 4. Cost
# ─────────────────────────────────────────────
def section_cost(repeats=5):
    print(f"4. GEOMETRY STAGE COST (median of {repeats} passes over the six calibration assets)")
    maps = [edges_of(n, A.LADDER["medium"])[0] for n in HV.NAMES if A.image(n) is not None]
    for rescue in (False, True):
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for e in maps:
                D.find_feature_boxes(e, rescue=rescue)
            samples.append(1000 * (time.perf_counter() - t0) / len(maps))
        print(f"   rescue={str(rescue):<6}{sorted(samples)[len(samples)//2]:>7.2f} ms/img")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip the pair study (section 3)")
    args = ap.parse_args()
    if not os.path.isdir(D.ASSETS):
        sys.exit(f"assets not found at {D.ASSETS}")
    section_case()
    section_counts()
    section_stability()
    if not args.quick:
        section_sizes()
    section_cost()


if __name__ == "__main__":
    main()
