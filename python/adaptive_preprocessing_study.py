"""
Where the adaptive preprocessing selector's constants come from.

The edge stage denoises before it differentiates, and how hard it denoises is a trade: too
little and machining texture, grain and scratches fragment into contours that merge with
real hardware or pass as it; too much and a faint edge — a low-contrast head, the far side
of a shadowed stud — is smoothed away and the box shrinks or vanishes. One setting cannot be
right for a clean, sharp, tripod-steady photograph and a grainy hand-held one.

The shipped stack already tried to adapt: `computeAdaptiveBilateralD` buckets the VARIANCE
of the Laplacian into a bilateral diameter. hybrid_v2_study found that step "firing backwards"
— it handed the least smoothing to the two sharpest photographs, because Laplacian variance
is dominated by edges, not by noise. A sharp, clean, high-resolution photo has a huge
Laplacian variance and no noise at all.

This study replaces it with two numbers and a ladder:

  * NOISE  — a MAD-based estimate of the noise standard deviation (Donoho's robust
             estimator on the Laplacian's absolute response). A median ignores the sparse
             large responses that edges produce and reads the bulk, which is what noise is.
  * EDGE CONTRAST — the 99th percentile of the same response: how strong the strongest
             edges are. A soft or blurred photograph has weak edges, and smoothing it
             harder loses what little it had.
  * LADDER — three rungs of one pipeline (Hybrid v6's, the most accurate against the
             ground truth). See the comment on LADDER for why the rungs move CLAHE's clip
             limit rather than the filter apertures a first draft moved.

Then it asks the question the app actually has: given a reference layout whose boxes are
known, can an inspection pick its strength by which one reproduces the reference's SIZES
best? A shadow merged into a box or a fastener half-cropped is a size error before it is
anything else, and the reference — refined by the sizes seen in past inspections — says what
size each slot should be.

RESULT
------
The noise estimate tracks added noise almost one-for-one (clean assets 2-4, textured plates
12-19; +5/+10/+20 of synthetic noise reads 8.6/14.6/27.2), where the Laplacian variance the
shipped stack bucketed spans 22-2877 on the CLEAN assets alone. Against ground truth
(weighted = 2 x misses + extras, 26 photographs):

                   clean   s=5   s=7  s=10  s=20   blur5  blur7  blur9  blur13
    light            254   235   228   291   364     113     31     24      36
    medium (= v6)     19    22    23    31    26      27     28     40     101
    strong            28    22    22    20    16      25     33     54     111

(Re-measured after fragment_rescue_study's change to the shared geometry filter. Only the
light rung and one medium cell moved, by a count or two; every threshold below is unchanged.)

So: STRONG from a noise estimate of about 12 (the s=7 crossover; also where the textured
plates sit, on which the two rungs tie), LIGHT below an edge contrast of about 20 (blur9),
MEDIUM otherwise — and LIGHT is not even a candidate above a contrast of 40, because on
anything sharp it is catastrophic (25 misses, 205 extras on the clean set) in a way the size
score cannot see: it judges the boxes at the slots, not the extras between them.

With that eligibility rule and a margin of 0.08 (size jitter between genuine same-hardware
boxes is a median 0.058 in the same units, size_ratio_study), size-agreement selection was
never worse than the noise rung and beat every fixed rung in every condition, L1 vs ground
truth over the interchangeable pairs:

                  clean  s=10  s=20  blur7  blur9  blur13
    chosen            8     2     2      6      7       8
    noise rung only   8     4     5      8     14       8
    best fixed rung   8     4     5      8      6       8

Below 0.08 the rule made bad overrides (a light rung winning on size while over-detecting);
above 0.10 it stopped helping under blur. A fourth, heavier-bilateral rung (d=9, the knob the
first draft moved) was tried as a candidate-only entry: chosen at most once per condition and
neutral on counts. Not adopted — for lack of benefit, not cost: bilateral 9 is about 12% more
per rung (23.5 against 20.9 ms). What made the first draft's strong rung expensive (48 ms) was
its 7px median, not its bilateral; OpenCV's fast median path stops at 5px.

The honest headline is that v6 is already robust to noise — the per-image gain from the
noise rung alone is modest (s=10: 20 vs 31) — and the larger wins are the soft-photo rung
and letting the reference's sizes arbitrate.

SECTIONS
--------
  1. The estimators on every asset, and how each tracks synthetic noise and blur. This is
     what decides whether MAD-sigma is a noise measure at all.
  2. Each ladder rung against the ground truth counts at several synthetic noise levels.
     The rung that is best at each level is what the noise thresholds have to reproduce.
  3. Size-agreement selection on the gate-interchangeable pairs: photo A is the reference,
     photo B the inspection (clean, noisy, blurred). For each rung on B, boxes are
     paired to A's through a SIFT+RANSAC homography — ground truth independent of the
     detector — and scored by how far their sizes sit from A's. The rung that agrees best
     is compared, on count accuracy, with the noise-chosen rung and every fixed rung.
  4. Cost per rung.
  5. (--alternatives) The rungs that were NOT adopted, on the same conditions: the first
     draft's aperture ladder, a heavier bilateral, and the CLAHE-clip variants the final
     rungs were chosen from. This is the evidence for the LADDER comment below.

Run:  python python/adaptive_preprocessing_study.py
      python python/adaptive_preprocessing_study.py --quick          (skips section 3)
      python python/adaptive_preprocessing_study.py --alternatives   (adds section 5)

Needs opencv-python and numpy, plus the gitignored assets in
app/src/main/assets/chassis_images/. Desktop OpenCV against the app's Android build; treat
small numeric differences as noise, as every other study here does.
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
from localisation_tolerance_study import (  # noqa: E402
    MIN_CORRESPONDENCES, PAIRING_RADIUS_FRAC, apply_h, correspond, fit_similarity, homography,
)

# ─────────────────────────────────────────────
# The ladder. Hybrid v6's pipeline, moved along the one axis that measured as a lever.
#
# The first draft stepped the three denoising apertures (bilateral 3→11, median 3→9,
# Gaussian 3→9). Measured against ground truth that ladder was wrong in both directions:
# the "strong" rung missed 15 fasteners on CLEAN photographs and "max" 41, because at the
# 1280px working size a 7px median erases small hardware outright, and "light" produced 175
# extras. What moved the count under noise was CLAHE's clip limit — local contrast
# equalisation amplifies grain in flat regions exactly as readily as it lifts faint edges,
# and under noise the grain wins — and what moved it under blur was the same knob the other
# way, plus the Gaussian aperture. So the rungs share the bilateral and median apertures
# (and therefore cost the same) and differ in how hard they push contrast.
# ─────────────────────────────────────────────
V6 = dict(high_scale=1.1, low_ratio=0.4, l2=True, dilate_k=5, close_k=9)
LADDER = {
    # Soft photographs: lift what little edge there is, blur it as little as possible.
    "light":  dict(V6, bilat=5, median=5, gauss=3, clahe_clip=2.0),
    # == Hybrid v6 exactly; the rung every photograph the estimators cannot fault gets.
    "medium": dict(V6, bilat=5, median=5, gauss=5),
    # Noisy photographs: half the contrast boost, and Canny's high bound back at 1.0x Otsu
    # so the edges that survive are not thinned further.
    "strong": dict(V6, bilat=5, median=5, gauss=5, clahe_clip=0.5, high_scale=1.0),
}
RUNGS = list(LADDER)

# Java-side constants under test (ImageQuality / AdaptivePreprocessing).
MIN_JUDGED_PX = 30            # FeatureLocalisation.MIN_SIZE_JUDGED_PX
DEVIATION_CAP = np.log(2.0)   # a box twice or half the expected size is as bad as absent
MISSING_PENALTY = DEVIATION_CAP

# For the Laplacian kernel [0 1 0; 1 -4 1; 0 1 0], white noise of standard deviation s has
# response standard deviation s*sqrt(20), and the median of the absolute value of a
# zero-mean normal is 0.6745 of its sigma. So sigma = median(|L|) / (0.6745 * sqrt(20)).
MAD_TO_SIGMA = 1.0 / (0.6745 * np.sqrt(20.0))


def image(name):
    return HV.image(name)


def gray_of(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ─────────────────────────────────────────────
# 1. Estimators
# ─────────────────────────────────────────────
def assess(gray):
    """The three numbers ImageQuality.assess() computes, on the working-resolution gray."""
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    # convertScaleAbs saturates at 255, exactly as the Java side does; nothing here reaches it.
    a = cv2.convertScaleAbs(lap)
    median = float(np.median(a))
    p99 = float(np.percentile(a, 99))
    lapvar = float(np.std(cv2.Laplacian(gray, cv2.CV_64F))) ** 2
    return dict(noise=median * MAD_TO_SIGMA, contrast=p99, lapvar=lapvar)


def noisy(img, s, seed=11):
    if s <= 0:
        return img
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, s, img.shape), 0, 255).astype(np.uint8)


def blurred(img, k):
    return img if k <= 1 else cv2.GaussianBlur(img, (k, k), 0)


def section_estimators():
    print("1. ESTIMATORS ON EVERY ASSET (working resolution, gray)")
    print(f"   {'asset':<20}{'lapvar':>9}{'noise':>8}{'p99':>6}   "
          f"noise after +s=5/10/20      p99 after blur k=5/9")
    rows = {}
    for name in sorted(GTRUTH.GT) + sorted(GTRUTH.WIDE):
        img = image(name)
        if img is None:
            continue
        g = gray_of(img)
        q = assess(g)
        rows[name] = q
        added = [assess(gray_of(noisy(img, s)))["noise"] for s in (5, 10, 20)]
        soft = [assess(gray_of(blurred(img, k)))["contrast"] for k in (5, 9)]
        print(f"   {name:<20}{q['lapvar']:>9.0f}{q['noise']:>8.2f}{q['contrast']:>6.0f}   "
              f"{added[0]:>6.2f} {added[1]:>6.2f} {added[2]:>6.2f}"
              f"{'':12}{soft[0]:>5.0f} {soft[1]:>5.0f}")

    clean = np.array([r["noise"] for r in rows.values()])
    print(f"\n   clean assets: noise median {np.median(clean):.2f}, "
          f"min {clean.min():.2f}, max {clean.max():.2f}")
    # The claim under test: sigma tracks ADDED noise roughly one-for-one, while the old
    # Laplacian variance is a function of the photograph's content.
    for s in (5, 10, 20):
        est = np.array([assess(gray_of(noisy(image(n), s)))["noise"] for n in rows])
        print(f"   +s={s:<3} noise estimate median {np.median(est):.2f} "
              f"(min {est.min():.2f}, max {est.max():.2f})")
    lv = np.array([r["lapvar"] for r in rows.values()])
    print(f"   Laplacian variance on the CLEAN assets already spans {lv.min():.0f}-{lv.max():.0f}"
          f" — the old bucket boundaries (200-700) sit inside the content range, not the noise range.")
    return rows


# ─────────────────────────────────────────────
# 2. Ladder against ground truth, by noise level
# ─────────────────────────────────────────────
def boxes(img, p):
    return D.find_feature_boxes(HV.edges_for(img, HV.cfg(**p)))[0]


def gt_score(counts):
    w, fn, fp = GTRUTH.weighted(counts)
    exact = sum(1 for n, c in counts.items() if GTRUTH.err(n, c) == 0)
    return dict(exact=exact, fn=fn, fp=fp, weighted=w)


def section_ladder(quality):
    print("\n2. LADDER AGAINST GROUND TRUTH (26 reference-distance photographs)")
    print("   weighted = 2 x misses + extras, the ranking hybrid_v2_study used; lower is better")
    conditions = [("clean", lambda i: i)] + [
        (f"noise s={s}", (lambda s: lambda i: noisy(i, s))(s)) for s in (5, 7, 10, 20)
    ] + [(f"blur k={k}", (lambda k: lambda i: blurred(i, k))(k)) for k in (5, 7, 9, 13)]
    configs = dict(LADDER)
    configs["shipped"] = None  # the stack the app runs today, for scale
    best = {}
    for label, fn in conditions:
        imgs = {n: fn(image(n)) for n in GTRUTH.GT if image(n) is not None}
        est = np.median([assess(gray_of(i))["noise"] for i in imgs.values()])
        ctr = np.median([assess(gray_of(i))["contrast"] for i in imgs.values()])
        print(f"\n   {label}  (median noise estimate {est:.2f}, median p99 {ctr:.0f})")
        print(f"   {'rung':<10}{'exact':>7}{'miss':>6}{'extra':>7}{'weighted':>10}")
        scores = {}
        for rung, p in configs.items():
            if p is None:
                counts = {n: len(D.find_feature_boxes(D.compute_edges(i)[0])[0])
                          for n, i in imgs.items()}
            else:
                counts = {n: len(boxes(i, p)) for n, i in imgs.items()}
            s = gt_score(counts)
            scores[rung] = s
            print(f"   {rung:<10}{s['exact']:>4}/26{s['fn']:>6}{s['fp']:>7}{s['weighted']:>10}")
        ranked = sorted(RUNGS, key=lambda r: (scores[r]["weighted"], -scores[r]["exact"]))
        best[label] = (est, ctr, ranked[0], scores)
        print(f"   -> best rung: {ranked[0]}")
    return best


# ─────────────────────────────────────────────
# 3. Size-agreement selection on same-scene pairs
# ─────────────────────────────────────────────
def frac_boxes(bx, w, h):
    return np.array([[(x + bw / 2.0) / w, (y + bh / 2.0) / h, bw / w, bh / h]
                     for x, y, bw, bh in bx], dtype=np.float64).reshape(-1, 4)


def agreement(ref, ref_px, H, cand):
    """
    Mean capped size deviation of the candidate's boxes from the reference's, over every
    reference box big enough to judge, with unmatched reference boxes charged the cap.

    A mean rather than a median on purpose: the failure this exists to catch is ONE box that
    swallowed a shadow, and a median over four slots would not see it.
    """
    judged = [i for i in range(len(ref)) if min(ref_px[i]) >= MIN_JUDGED_PX]
    if not judged or len(cand) == 0:
        return None, 0
    pred = apply_h(H, ref[:, :2])
    matches = correspond(pred, cand[:, :2], PAIRING_RADIUS_FRAC)
    if len(matches) < MIN_CORRESPONDENCES:
        return None, len(matches)
    ia = [i for i, _ in matches]
    ib = [j for _, j in matches]
    u = ref[ia, :2] - ref[ia, :2].mean(axis=0)
    v = cand[ib, :2] - cand[ib, :2].mean(axis=0)
    _, scale = fit_similarity(u, v)
    if not scale or scale < 1e-9:
        return None, len(matches)
    matched = dict(matches)
    devs = []
    for i in judged:
        j = matched.get(i)
        if j is None:
            devs.append(MISSING_PENALTY)
            continue
        ratio = np.sqrt((cand[j, 2] * cand[j, 3]) / (ref[i, 2] * ref[i, 3] * scale * scale))
        devs.append(min(abs(np.log(ratio)), DEVIATION_CAP))
    return float(np.mean(devs)), len(matches)


def noise_rung(noise, contrast, thresholds):
    """The pure decision ImageQuality.tierFor() makes, with thresholds under test."""
    strong_at, soft_below, _ = thresholds
    if noise >= strong_at:
        return "strong"
    if contrast < soft_below:
        return "light"
    return "medium"


def eligible(contrast, thresholds):
    """Rungs the size test may choose between — AdaptivePreprocessing's candidate set."""
    _, _, light_below = thresholds
    return ["medium", "strong"] + (["light"] if contrast < light_below else [])


def section_selection(thresholds, margin):
    print("\n3. SIZE-AGREEMENT SELECTION ON GATE-INTERCHANGEABLE PAIRS")
    print("   A = reference (its own noise-chosen rung), B = inspection under every rung.")
    print("   Each B rung is scored by how well its box sizes reproduce A's; the best is chosen")
    print("   unless the noise-chosen rung is within the margin of it.")
    factory, norm = (lambda: cv2.SIFT_create(3000)), cv2.NORM_L2
    pairs = []
    for group in D.interchangeable_groups().values():
        pairs.extend(itertools.permutations(group, 2))
    pairs = [p for p in pairs if p[0] in GTRUTH.GT and p[1] in GTRUTH.GT]
    print(f"   {len(pairs)} ordered pairs")

    conditions = [("B clean", lambda i: i),
                  ("B + noise s=10", lambda i: noisy(i, 10, seed=7)),
                  ("B + noise s=20", lambda i: noisy(i, 20, seed=7)),
                  ("B blurred k=7", lambda i: blurred(i, 7)),
                  ("B blurred k=9", lambda i: blurred(i, 9)),
                  ("B blurred k=13", lambda i: blurred(i, 13))]
    hcache = {}
    for label, degrade in conditions:
        tallies = {k: [] for k in ["chosen", "noise-only", "medium"] + RUNGS}
        switched = agreed = decidable = 0
        for a, b in pairs:
            key = (a, b)
            if key not in hcache:
                hcache[key] = homography(a, b, factory, norm)
            H = hcache[key]
            if H is None:
                continue
            img_a = image(a)
            ha, wa = img_a.shape[:2]
            qa = assess(gray_of(img_a))
            ref_rung = noise_rung(qa["noise"], qa["contrast"], thresholds)
            bx_a = boxes(img_a, LADDER[ref_rung])
            ref = frac_boxes(bx_a, wa, ha)
            ref_px = np.array([[bw, bh] for _, _, bw, bh in bx_a], dtype=np.float64).reshape(-1, 2)

            img_b = degrade(image(b))
            hb, wb = img_b.shape[:2]
            qb = assess(gray_of(img_b))
            prior = noise_rung(qb["noise"], qb["contrast"], thresholds)
            scores, counts = {}, {}
            for rung in RUNGS:
                if rung not in eligible(qb["contrast"], thresholds):
                    continue
                bx_b = boxes(img_b, LADDER[rung])
                counts[rung] = len(bx_b)
                scores[rung], _ = agreement(ref, ref_px, H, frac_boxes(bx_b, wb, hb))
            usable = {r: v for r, v in scores.items() if v is not None}
            if not usable:
                continue
            decidable += 1
            best = min(usable, key=usable.get)
            chosen = prior
            if prior not in usable or usable[best] < usable[prior] - margin:
                chosen = best
            if chosen != prior:
                switched += 1
            if best == prior:
                agreed += 1
            tallies["chosen"].append(GTRUTH.err(b, counts[chosen]))
            tallies["noise-only"].append(GTRUTH.err(b, counts[prior]))
            for rung in RUNGS:
                if rung in counts:
                    tallies[rung].append(GTRUTH.err(b, counts[rung]))
        print(f"\n   {label}: {decidable} decidable pairs; size agreement overrode the noise"
              f" rung {switched} times, agreed outright {agreed} times")
        print(f"   {'policy':<12}{'L1 vs GT':>10}{'exact':>7}")
        for k in ["chosen", "noise-only"] + RUNGS:
            errs = np.array(tallies[k])
            if len(errs) == 0:
                continue
            # A fixed rung is only tallied on the pairs it was eligible for, so the light
            # row is over fewer pairs than the others and says so.
            print(f"   {k:<12}{errs.sum():>10}{int((errs == 0).sum()):>4}/{len(errs)}")


def section_merge_case():
    """The first real case the size check explained: screws_none vs screws_flash, slot 3."""
    print("\n   The screws_none / screws_flash merge (reference box hugged the cap, the flash")
    print("   shot's box swallowed shaft and shadow, +91%): box sizes per rung on screws_flash")
    for name in ("screws_none.jpg", "screws_flash.jpg"):
        img = image(name)
        if img is None:
            continue
        line = []
        for rung in RUNGS:
            bx = sorted(boxes(img, LADDER[rung]), key=lambda r: (r[1], r[0]))
            line.append(f"{rung}: " + " ".join(f"{bw}x{bh}" for _, _, bw, bh in bx))
        print(f"   {name}\n     " + "\n     ".join(line))


# ─────────────────────────────────────────────
# 4. Cost
# ─────────────────────────────────────────────
def section_cost(repeats=5):
    print(f"\n4. EDGE STAGE COST (median of {repeats} passes over the six calibration assets)")
    imgs = [image(n) for n in HV.NAMES if image(n) is not None]
    t0 = time.perf_counter()
    for im in imgs:
        assess(gray_of(im))
    print(f"   assess():  {1000 * (time.perf_counter() - t0) / len(imgs):>7.1f} ms/img")
    for rung, p in LADDER.items():
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for im in imgs:
                HV.edges_for(im, HV.cfg(**p))
            samples.append(1000 * (time.perf_counter() - t0) / len(imgs))
        print(f"   {rung:<10}{sorted(samples)[len(samples) // 2]:>7.1f} ms/img")


# ─────────────────────────────────────────────
# 5. The rungs that were not adopted
# ─────────────────────────────────────────────
ALTERNATIVES = {
    # The first draft: strength as filter apertures. Wrong in both directions.
    "draft light 3/3/3":     dict(V6, bilat=3, median=3, gauss=3),
    "draft strong 9/7/7":    dict(V6, bilat=9, median=7, gauss=7),
    "draft max 11/9/9":      dict(V6, bilat=11, median=9, gauss=9),
    # Heavier bilateral only, apertures otherwise v6's: fewer extras under noise, but
    # loses fasteners on clean photographs and costs several times more.
    "bilateral 9":           dict(V6, bilat=9, median=5, gauss=5),
    # The CLAHE lever, and the two steps to the adopted strong rung.
    "clahe 0.5":             dict(V6, bilat=5, median=5, gauss=5, clahe_clip=0.5),
    "clahe 0.5 otsu1.0 (*)": LADDER["strong"],
    # Toward the adopted light rung: the median aperture and the clip both matter.
    "5/3/3":                 dict(V6, bilat=5, median=3, gauss=3),
    "5/5/5 clahe 2.0":       dict(V6, bilat=5, median=5, gauss=5, clahe_clip=2.0),
    "5/5/3 clahe 2.0 (*)":   LADDER["light"],
    "medium = v6 (*)":       LADDER["medium"],
}


def section_alternatives():
    print("\n5. RUNGS NOT ADOPTED (weighted / misses / extras; (*) marks the three adopted)")
    conditions = [("clean", lambda i: i), ("s=10", lambda i: noisy(i, 10)),
                  ("s=20", lambda i: noisy(i, 20)), ("s=30", lambda i: noisy(i, 30)),
                  ("blur5", lambda i: blurred(i, 5)), ("blur9", lambda i: blurred(i, 9)),
                  ("blur13", lambda i: blurred(i, 13))]
    imgs = {c: {n: fn(image(n)) for n in GTRUTH.GT} for c, fn in conditions}
    print(f"   {'candidate':<24}" + "".join(f"{c:>14}" for c, _ in conditions) + f"{'ms/img':>8}")
    six = [image(n) for n in HV.NAMES if image(n) is not None]
    for label, p in ALTERNATIVES.items():
        cells = []
        for c, _ in conditions:
            s = gt_score({n: len(boxes(i, p)) for n, i in imgs[c].items()})
            cells.append(f"{s['weighted']:>4}/{s['fn']:>3}/{s['fp']:>4}")
        t0 = time.perf_counter()
        for im in six:
            HV.edges_for(im, HV.cfg(**p))
        ms = 1000 * (time.perf_counter() - t0) / len(six)
        print(f"   {label:<24}" + "".join(f"{x:>14}" for x in cells) + f"{ms:>8.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip the pair study (section 3)")
    ap.add_argument("--alternatives", action="store_true",
                    help="also score the rungs that were not adopted (section 5)")
    ap.add_argument("--thresholds", type=float, nargs=3,
                    metavar=("STRONG", "SOFT", "LIGHT_ELIGIBLE"),
                    default=(12.0, 20.0, 40.0),
                    help="noise sigma from which the strong rung is chosen; the p99 edge "
                         "contrast below which the light rung is; and the contrast below "
                         "which light is a candidate at all")
    ap.add_argument("--margin", type=float, default=0.08,
                    help="ln-units by which another rung must beat the noise-chosen one")
    args = ap.parse_args()
    if not os.path.isdir(D.ASSETS):
        sys.exit(f"assets not found at {D.ASSETS}")

    HV.verify_base()
    print()
    quality = section_estimators()
    section_ladder(quality)
    section_merge_case()
    if not args.quick:
        section_selection(tuple(args.thresholds), args.margin)
    section_cost()
    if args.alternatives:
        section_alternatives()


if __name__ == "__main__":
    main()
