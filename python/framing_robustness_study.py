"""
Why the alignment gate said TOO FAR to two near-identical photographs, and what fixes it.

THE FAILURE
-----------
`whole screws1.jpg` as the reference, `whole screws2.jpg` as the frame: same plate, same
distance, a few degrees of camera angle apart. The gate read the framing as x3.33 and said
TOO FAR - MOVE CLOSER. Swap them and it says TOO CLOSE at x0.30. The framing search's
correlation at each candidate scale for that pair:

    scale   0.30  0.35  0.40  ...  0.90  0.95  1.00
    peak    0.48  0.44  0.39  ...  0.43  0.47  0.31      truth is ~1.00, the lowest value

Two effects, and each alone would be survivable:

  * The reference shrunk to 30% is a 96-pixel blob of plate texture that correlates ~0.48
    with *something* in any textured scene. Matching a photograph against itself gives ~0.54
    at that scale. Small templates have a high chance floor, because there are many places
    to try and little structure to disagree.
  * At the true scale the two frames do not line up pixel for pixel: the search models shift
    and zoom only, and a few degrees of rotation or perspective across a 320-pixel frame
    drops the gradient-map correlation to 0.31 - below the small-template chance floor.

So a spurious tiny-template peak wins, the ratio is read as 1/0.30, and the operator is
told to move. `TemplateMatcher` already documents that the ratio is unreliable across
viewpoints; this pins down the mechanism and measures the candidate remedies.

CANDIDATES
----------
  blur     Gaussian-blur the gradient maps before correlating, so a few pixels of
           misregistration cost a little correlation instead of all of it. Downsampling
           further would do the same but was measured (part B of the template study) to
           lose the fastener edges; blur keeps them, merely widens them.
  null     Score each scale by its peak MINUS the peak the same template finds in the
           scene rotated by 180 degrees - the same texture statistics with no true match,
           i.e. an empirical chance floor per template size. A tiny blob scores about the
           same either way and nets out to nothing; a real match stands above its floor.
  floor    Do not sweep below a framing of 0.5 / above 2.0. The gate rejects everything
           outside 0.85-1.35 anyway; it only needs the direction, not a number like x3.3,
           and the far ends of the sweep are where the chance floor is highest.
  tie      When several scales peak within a hair of each other, take the one nearest 1.0.
  prior    Charge each scale's peak  lambda * |ln scale|  before comparing, so a small
           template's chance-level peak has to beat the unchanged-framing reading by a
           margin that grows with how much of a change it claims. A soft version of "tie".

RESULT
------
"prior" is the one that helps without hurting: at lambda 0.08-0.12 it recovers one more
exact crop (9/10 from 8/10), turns two more gate verdicts right (117/129 from 115/129),
breaks none, and passes all three whole-screws pairs; 0.15 starts to over-pull toward 1.0.
Blur costs a verdict or two, the floor costs three, "null" is a disaster (the plate is
close to symmetric under a half turn, so the rotated scene is a partial true match and
subtracting it kills the real peaks), and "tie" trades one pair for another. Shipped as
TemplateMatcher.SCALE_PRIOR = 0.10.

MEASURED AGAINST
----------------
  1. The ten exact ground-truth cases of the template study: a known sub-region cropped from
     one photograph of a scene, found in a different photograph of the same scene under
     different light. Scale and position are known exactly. Must stay at 10/10.
  2. Synthetic scale recovery: crops of the reference itself. Must stay perfect.
  3. Every same-scene pair in the catalogue, with the framing ratio each pair should read
     from the eyeballed shooting distances. Pairs shot from the same distance should pass
     the gate's 0.85-1.35 band; pairs a clear step apart should be rejected the right way.
     Distances are ordinal, so pairs within 10% of a band edge are left undecided.

Run:  python python/framing_robustness_study.py
Needs opencv-python and numpy, plus the gitignored assets in
app/src/main/assets/chassis_images/.
"""

import itertools
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required: pip install opencv-python")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from template_matching_study import (  # noqa: E402
    CASES, CATALOGUE, LIGHTING_PAIRS, MIN_TEMPLATE_EDGE, OK_OFFSET, OK_SCALE, WORK_WIDTH,
    fit, load, rep_sobel,
)

# The gate's band, mirrored from AlignmentGate.
GATE_MIN, GATE_MAX = 0.85, 1.35
# Expected ratios within this factor of a band edge are not decidable from eyeballed
# distances, either side: a pair expected at 0.80 might be 0.90 in truth.
EDGE_SLACK = 1.10

FULL_SCALES = np.round(np.arange(0.30, 1.00 + 1e-9, 0.05), 2)
FLOORED_SCALES = np.round(np.arange(0.50, 1.00 + 1e-9, 0.05), 2)


def prepared(img, blur):
    mag = rep_sobel(img)
    if blur > 1:
        mag = cv2.GaussianBlur(mag, (blur, blur), 0)
    return mag


def search_in(scene, templ, scales, use_null, tie_eps=0.0, prior=0.0):
    """
    {score, scale, centreX, centreY, peak} of the best placement under the chosen scorer.

    With tie_eps > 0, every scale whose score is within that much of the maximum is a
    candidate and the one nearest 1.0 wins: the operator is told to frame like the ghost, so
    when the search cannot separate "unchanged" from "a third of the size" the gate should
    assume unchanged and let the band do its job on clear departures.
    """
    found = []
    aspect = templ.shape[0] / templ.shape[1]
    flipped = cv2.rotate(scene, cv2.ROTATE_180) if use_null else None
    for s in scales:
        tw = int(round(scene.shape[1] * s))
        th = int(round(tw * aspect))
        if (tw < MIN_TEMPLATE_EDGE or th < MIN_TEMPLATE_EDGE
                or tw > scene.shape[1] or th > scene.shape[0]):
            continue
        t = cv2.resize(templ, (tw, th), interpolation=cv2.INTER_AREA)
        r = cv2.matchTemplate(scene, t, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(r)
        score = peak
        if use_null:
            _, chance, _, _ = cv2.minMaxLoc(cv2.matchTemplate(flipped, t, cv2.TM_CCOEFF_NORMED))
            score = peak - chance
        score -= prior * abs(np.log(s))
        found.append((float(score), float(s),
                      (loc[0] + tw / 2) / scene.shape[1], (loc[1] + th / 2) / scene.shape[0],
                      float(peak)))
    if not found:
        return None
    top = max(f[0] for f in found)
    contenders = [f for f in found if f[0] >= top - tie_eps]
    return min(contenders, key=lambda f: abs(np.log(f[1])))


def evaluate(ref, frame, scales, use_null, tie_eps=0.0, prior=0.0):
    """(framing scale, offsetX, offsetY, raw peak) as the gate would read it."""
    closer = search_in(ref, frame, scales, use_null, tie_eps, prior)
    wider = search_in(frame, ref, scales, use_null, tie_eps, prior)
    if closer is None and wider is None:
        return None
    if wider is None or (closer is not None and closer[0] >= wider[0]):
        return closer[1], 0.5 - closer[2], 0.5 - closer[3], closer[4]
    return 1.0 / wider[1], wider[2] - 0.5, wider[3] - 0.5, wider[4]


def ground_truth(blur):
    """Exact cases: a crop of A found in B. Returns (recovered, total)."""
    ok = n = 0
    for a, b in LIGHTING_PAIRS:
        try:
            A, B = fit(load(a), WORK_WIDTH), fit(load(b), WORK_WIDTH)
        except FileNotFoundError:
            continue
        for s, cx, cy in CASES:
            h, w = int(A.shape[0] * s), int(A.shape[1] * s)
            x0 = int(np.clip(cx * A.shape[1] - w / 2, 0, A.shape[1] - w))
            y0 = int(np.clip(cy * A.shape[0] - h / 2, 0, A.shape[0] - h))
            crop = A[y0:y0 + h, x0:x0 + w]
            truth = (s, (x0 + w / 2) / A.shape[1], (y0 + h / 2) / A.shape[0])
            yield truth, prepared(B, blur), prepared(crop, blur)


def synthetic(blur):
    for name in ("screws_flash.jpg", "screws.jpg"):
        try:
            ref = fit(load(name), WORK_WIDTH)
        except FileNotFoundError:
            continue
        for f in (1.00, 0.85, 0.70, 0.55, 0.40):
            h, w = int(ref.shape[0] * f), int(ref.shape[1] * f)
            crop = ref[(ref.shape[0] - h) // 2:(ref.shape[0] - h) // 2 + h,
                       (ref.shape[1] - w) // 2:(ref.shape[1] - w) // 2 + w]
            yield f, prepared(ref, blur), prepared(crop, blur)


def verdict(scale):
    if scale < GATE_MIN:
        return "close"
    if scale > GATE_MAX:
        return "far"
    return "pass"


def expected_verdict(ratio):
    """What the gate should say for an expected ratio, or None when too near an edge to call."""
    if GATE_MIN * EDGE_SLACK <= ratio <= GATE_MAX / EDGE_SLACK:
        return "pass"
    if ratio <= GATE_MIN / EDGE_SLACK:
        return "close"
    if ratio >= GATE_MAX * EDGE_SLACK:
        return "far"
    return None


def real_pairs():
    names = [n for n in CATALOGUE if os.path.exists(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "app", "src", "main", "assets",
        "chassis_images", n))]
    for a, b in itertools.combinations(sorted(names), 2):
        (ga, da), (gb, db) = CATALOGUE[a], CATALOGUE[b]
        if ga != gb:
            continue
        yield a, b, db / da


def run(label, blur, scales, use_null, cache, tie_eps=0.0, prior=0.0):
    gt_ok = gt_n = 0
    for truth, scene, templ in ground_truth(blur):
        got = evaluate(scene, templ, scales, use_null, tie_eps, prior)
        # The crop is inside the scene, so the "closer" reading is the one under test.
        gt_n += 1
        if got is not None and abs(got[0] - truth[0]) <= OK_SCALE:
            cx, cy = 0.5 - got[1], 0.5 - got[2]
            if abs(cx - truth[1]) <= OK_OFFSET and abs(cy - truth[2]) <= OK_OFFSET:
                gt_ok += 1

    syn_ok = syn_n = 0
    for f, scene, templ in synthetic(blur):
        got = evaluate(scene, templ, scales, use_null, tie_eps, prior)
        syn_n += 1
        syn_ok += got is not None and abs(got[0] - f) <= OK_SCALE

    decided = correct = 0
    log_err = []
    whole = None
    for a, b, ratio in real_pairs():
        key = (a, blur)
        if key not in cache:
            cache[key] = prepared(fit(load(a), WORK_WIDTH), blur)
        key_b = (b, blur)
        if key_b not in cache:
            cache[key_b] = prepared(fit(load(b), WORK_WIDTH), blur)
        got = evaluate(cache[key], cache[key_b], scales, use_null, tie_eps, prior)
        if got is None:
            continue
        log_err.append(abs(np.log(got[0] / ratio)))
        want = expected_verdict(ratio)
        if want is not None:
            decided += 1
            correct += verdict(got[0]) == want
        if (a, b) == ("whole screws1.jpg", "whole screws2.jpg"):
            whole = got[0]

    print(f"  {label:<28} {gt_ok:2d}/{gt_n:<3d}   {syn_ok:2d}/{syn_n:<3d}   "
          f"{correct:3d}/{decided:<4d}  {np.median(log_err):.2f}     "
          f"{'x%.2f %s' % (whole, verdict(whole)) if whole else '-'}")
    return correct, decided


def main():
    print("Framing search variants, scored against exact ground truth and the catalogue\n")
    print("  variant                      crops    synth    gate verdict   median   whole screws 1 vs 2")
    print("                               10 exact 10 exact right/decided |ln err|  (should pass)")
    print("  " + "-" * 96)
    cache = {}
    for label, blur, scales, use_null, tie, prior in [
        ("shipped before: peak only", 1, FULL_SCALES, False, 0.0, 0.0),
        ("floor 0.5", 1, FLOORED_SCALES, False, 0.0, 0.0),
        ("blur 7", 7, FULL_SCALES, False, 0.0, 0.0),
        ("blur 11", 11, FULL_SCALES, False, 0.0, 0.0),
        ("null (chance floor)", 1, FULL_SCALES, True, 0.0, 0.0),
        ("tie->1.0, eps 0.02", 1, FULL_SCALES, False, 0.02, 0.0),
        ("tie->1.0, eps 0.03", 1, FULL_SCALES, False, 0.03, 0.0),
        ("prior 0.05", 1, FULL_SCALES, False, 0.0, 0.05),
        ("prior 0.08", 1, FULL_SCALES, False, 0.0, 0.08),
        ("prior 0.10  <== shipped", 1, FULL_SCALES, False, 0.0, 0.10),
        ("prior 0.12", 1, FULL_SCALES, False, 0.0, 0.12),
        ("prior 0.15", 1, FULL_SCALES, False, 0.0, 0.15),
        ("prior 0.20", 1, FULL_SCALES, False, 0.0, 0.20),
        ("prior 0.10 + blur 7", 7, FULL_SCALES, False, 0.0, 0.10),
    ]:
        run(label, blur, scales, use_null, cache, tie, prior)

    print("\n  'gate verdict' counts same-scene pairs whose expected ratio (from the eyeballed")
    print("  shooting distances) is clearly inside or clearly outside the 0.85-1.35 band, and")
    print("  whether the search would have the gate say the same. Pairs near a band edge are")
    print("  left out. |ln err| is over every same-scene pair, band or not.")


if __name__ == "__main__":
    main()
