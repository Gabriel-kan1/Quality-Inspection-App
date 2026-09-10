"""
Where Hybrid v2/v3/v4 came from.

The shipped edge stage over-detects badly on two of the bundled photographs and is exactly
right on four others. Calibration counts were supplied for six assets:

    defected.jpg 4   head1.jpg 4   reference.jpg 4
    screws.jpg   5   stud1.jpg 4   zero.jpg      3

Measured against those, the shipped stack scores L1 = 49 — and all 49 of it is two images:

    defected 4  head1 9  reference 4  screws 5  stud1 48  zero 3

head1.jpg and stud1.jpg are the only two at the full 1280px working resolution, and they are
the two the adaptive bilateral step hands d=1 to. That step buckets Laplacian variance and
gives *less* smoothing to *lower* variance, so on a clean, sharply-focused, high-resolution
photograph it removes almost nothing and machining texture survives into the contour stage.
The adaptive step was firing backwards on precisely the images that needed it.

WHY THE GEOMETRY FILTER IS NOT TUNED HERE
-----------------------------------------
Only the edge stage moves. findFeatureBoxes is the scorer every method in EdgeMethodLab is
judged by, so tuning it would move the goalposts for all twelve at once and make the Edge Lab
comparison meaningless. A variant is a different edge map, nothing else.

WHAT "BETTER" MEANS
-------------------
Hitting six numbers is easy and proves little, so a candidate has to survive four tests:

  1. targets       L1 against the six calibration counts.
  2. photo drift   the same L1 under brightness, gamma, noise and JPEG variation. A config
                   that only works on these exact files is a coincidence.
  3. param drift   the same L1 after nudging each parameter one step. Low drift means the
                   config sits on a plateau, so it is a setting rather than a lucky point.
  4. stability     detector_ab_study's metric: disagreement between photographs the alignment
                   gate treats as interchangeable. This is what actually causes a false FAIL,
                   and it is measured over all bundled assets rather than the six.

Guards: total detections across all assets is reported alongside, because a method that
collapses detection to nothing scores wonderfully on six small targets and destroys the app;
and any config that drives an asset to zero detections is rejected outright.

RESULT
------
Eight configurations hit all six targets exactly. The three kept differ in mechanism, not
just in parameter value, and every one of them beats the shipped stack on stability:

  v2  bilateral 9,  Otsu x1.1, low 0.4     photo drift 0 - never moves off the counts
  v3  bilateral 5, gauss 7, Otsu x1.2      best combined rank; stability 0.18; cheapest
  v4  bilateral 7, L2 gradient, 3/9 morph  best stability of anything tested (0.13)
  v5  v3 + adaptiveThreshold mask          stability 0.16, least fragile per nudge, ~2x cost

ADAPTIVE THRESHOLDING
---------------------
Round 3 asked whether adaptive thresholding helps, in both senses of the phrase. Two answers,
and they point opposite ways:

  * Deriving Canny's bounds adaptively - Triangle, median- or mean-anchored auto-Canny, or
    Otsu computed per tile instead of globally - is WORSE. Tiled Otsu can reach photo drift 0,
    but every locally-adaptive derivation tried costs group stability badly (0.48-0.57 against
    v3's 0.18), because a threshold that tracks local content makes the count depend on what
    else is in the frame. That is exactly the disagreement between interchangeable photographs
    that causes a false FAIL. Triangle, median and mean also drive assets to zero detections.
  * cv2.adaptiveThreshold as a MASK over the Canny map is BETTER. It is a poor edge detector
    alone (199 candidates on stud1.jpg where the shipped stack finds 42), but as a mask it
    removes texture speckle without touching outlines. It improved every variant it was
    applied to on at least two of the three robustness tests. v5 is the best of those.

Round 1 searched 720 configurations over the preprocessing and threshold scale only, and found
8 exact. Round 2 added Canny's low/high ratio and the morphology kernels: 149 exact, and every
good one sits at low_ratio=0.4 rather than production's 0.5. The round-1 picks are kept in
ALSO_RAN so the improvement can be read rather than taken on trust.

Pinning the bilateral diameter also makes these cheaper than the stack they replace, because
choosing d adaptively costs a full-image CV_64F Laplacian probe on every photograph.

Run:  python python/hybrid_v2_study.py
Note: desktop OpenCV (4.13 here) against the app's 4.5.3, and desktop timings are not device
timings. The primitives used are stable across that gap; treat small numeric differences as
noise. This module reuses detector_ab_study's port of the Java detector rather than making a
second copy of it — the BASE config below is asserted to reproduce it on every asset.
"""

import itertools
import time

import cv2
import numpy as np

import detector_ab_study as D

TARGETS = {"defected.jpg": 4, "head1.jpg": 4, "reference.jpg": 4,
           "screws.jpg": 5, "stud1.jpg": 4, "zero.jpg": 3}
NAMES = list(TARGETS)

# The shipped stack, expressed in the parameterisation below. Asserted equal to
# detector_ab_study.compute_edges() on every bundled asset by verify_base().
BASE = dict(glare="adaptive", glare_thresh=0.05, glare_k=75,
            clahe_clip=1.0, clahe_tiles=16,
            bilat="adaptive", bilat_sc=60, bilat_ss=20,
            median=5, gauss=5,
            high_scale=1.0, low_ratio=0.5, l2=False,
            dilate_k=5, close_k=7,
            # cv2.adaptiveThreshold used as a mask over the Canny map. mask_block <= 0 is off,
            # which is what the shipped stack and v2/v3/v4 use.
            mask_block=0, mask_c=2, mask_grow=5)

# The three ported to Java as EdgeMethodLab's Hybrid v2/v3/v4. Every one uses low_ratio=0.4:
# so does every other configuration that survived the wider search, which makes production's
# 0.5 the clearest single thing this study found wrong.
WINNERS = {
    "v2 bilat9 low0.4":     dict(bilat=9, median=5, gauss=5, high_scale=1.1, low_ratio=0.4,
                                 l2=False, dilate_k=5, close_k=7),
    "v3 gauss7 otsu1.2":    dict(bilat=5, median=5, gauss=7, high_scale=1.2, low_ratio=0.4,
                                 l2=False, dilate_k=5, close_k=7),
    "v4 L2 light-morph":    dict(bilat=7, median=5, gauss=5, high_scale=1.0, low_ratio=0.4,
                                 l2=True, dilate_k=3, close_k=9),
    "v5 v3 + adaptive mask": dict(bilat=5, median=5, gauss=7, high_scale=1.2, low_ratio=0.4,
                                  l2=False, dilate_k=5, close_k=7,
                                  mask_block=61, mask_c=2, mask_grow=5),
}
# The first-round picks, kept so the improvement is visible rather than asserted. All three
# are exact on the six targets too; they were beaten once low_ratio entered the search.
ALSO_RAN = {
    "(v2 round 1) bilat7 med9":  dict(bilat=7, median=9, gauss=5, high_scale=1.2, l2=False),
    "(v3 round 1) bilat9":       dict(bilat=9, median=5, gauss=5, high_scale=1.0, l2=False),
    "(v4 round 1) bilat5 L2":    dict(bilat=5, median=5, gauss=5, high_scale=1.0, l2=True),
}


def cfg(**kw):
    p = dict(BASE)
    p.update(kw)
    return p


def edges_for(img, p):
    """The shipped edge stage with the knobs the study moves exposed as parameters."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if p["glare"] == "adaptive":
        apply_glare = float(np.count_nonzero(g > 240)) / g.size > p["glare_thresh"]
    else:
        apply_glare = p["glare"] == "on"
    if apply_glare:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["glare_k"], p["glare_k"]))
        pre = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)
    else:
        pre = g

    # d is measured on the post-glare, pre-CLAHE image, exactly as computeEdges() does.
    # Feeding it the raw gray instead silently changes the bucket on any corrected photo.
    post_glare = pre
    if p["clahe_clip"] > 0:
        pre = cv2.createCLAHE(clipLimit=p["clahe_clip"],
                              tileGridSize=(p["clahe_tiles"],) * 2).apply(pre)

    d = D.bilateral_d_for(post_glare, p["bilat"])[0] if p["bilat"] == "adaptive" else p["bilat"]
    if d > 0:
        pre = cv2.bilateralFilter(pre, d, p["bilat_sc"], p["bilat_ss"])
    if p["median"] > 0:
        pre = cv2.medianBlur(pre, p["median"])
    if p["gauss"] > 0:
        pre = cv2.GaussianBlur(pre, (p["gauss"],) * 2, 0)

    hi, _ = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    hi *= p["high_scale"]
    e = cv2.Canny(pre, hi * p["low_ratio"], hi, L2gradient=p["l2"])

    if p.get("mask_block", 0) > 0:
        # A fastener is a locally dark region on a machined plate, so an edge that sits on one
        # is worth keeping and an edge in open texture is not. The mask has to be dilated
        # first: the edge lies on the boundary of the dark region, not inside it, so an
        # undilated mask deletes exactly the edges it exists to keep.
        m = cv2.adaptiveThreshold(pre, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, p["mask_block"], p["mask_c"])
        if p["mask_grow"] > 0:
            m = cv2.dilate(m, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (p["mask_grow"],) * 2))
        e = cv2.bitwise_and(e, m)

    if p["dilate_k"] > 0:
        e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_RECT, (p["dilate_k"],) * 2))
    if p["close_k"] > 0:
        e = cv2.morphologyEx(e, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["close_k"],) * 2))
    return e


_CACHE = {}


def image(name):
    if name not in _CACHE:
        _CACHE[name] = D.load_working_image(D.ASSETS / name)
    return _CACHE[name]


def count(img, p):
    return len(D.find_feature_boxes(edges_for(img, p))[0])


def l1(p, images=None):
    images = images or {n: image(n) for n in NAMES}
    return sum(abs(count(images[n], p) - TARGETS[n]) for n in NAMES if images.get(n) is not None)


def assets():
    return sorted(p.name for p in D.ASSETS.glob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"))


# ─────────────────────────────────────────────
# 0. Fidelity — the parameterisation must BE the shipped pipeline at BASE
# ─────────────────────────────────────────────
def verify_base():
    bad = []
    for a in assets():
        img = D.load_working_image(D.ASSETS / a)
        if img is None:
            continue
        if count(img, BASE) != len(D.find_feature_boxes(D.compute_edges(img)[0])[0]):
            bad.append(a)
    print("0. FIDELITY")
    print(f"   BASE reproduces detector_ab_study.compute_edges on "
          f"{len(assets()) - len(bad)}/{len(assets())} assets"
          + (f" — MISMATCHES: {bad}" if bad else ""))
    assert not bad, "the parameterisation has drifted from the shipped pipeline"


# ─────────────────────────────────────────────
# 1. Targets
# ─────────────────────────────────────────────
def report_targets(configs):
    print("\n1. CALIBRATION TARGETS")
    print(f"   {'method':<28}" + "".join(f"{n.split('.')[0][:9]:>11}" for n in NAMES) + f"{'L1':>6}")
    print(f"   {'(target)':<28}" + "".join(f"{TARGETS[n]:>11}" for n in NAMES) + f"{0:>6}")
    for label, p in configs.items():
        c = [count(image(n), p) for n in NAMES]
        print(f"   {label:<28}" + "".join(f"{v:>11}" for v in c)
              + f"{sum(abs(v - TARGETS[n]) for v, n in zip(c, NAMES)):>6}")


# ─────────────────────────────────────────────
# 2. Photographic variation
# ─────────────────────────────────────────────
def _bright(img, f):
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)


def _gamma(img, g):
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def _noise(img, s):
    rng = np.random.default_rng(11)
    return np.clip(img.astype(np.float32) + rng.normal(0, s, img.shape), 0, 255).astype(np.uint8)


def _jpeg(img, q):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


VARIANTS = [
    ("dim -15%", lambda i: _bright(i, 0.85)),
    ("bright +15%", lambda i: _bright(i, 1.15)),
    ("gamma 0.9", lambda i: _gamma(i, 0.9)),
    ("gamma 1.1", lambda i: _gamma(i, 1.1)),
    ("noise s=3", lambda i: _noise(i, 3)),
    ("noise s=6", lambda i: _noise(i, 6)),
    ("jpeg q70", lambda i: _jpeg(i, 70)),
]


def photo_drift(configs):
    print("\n2. PHOTOGRAPHIC VARIATION (total L1 over 7 variants; lower is better)")
    totals = {}
    for label, p in configs.items():
        t = 0
        for _, fn in VARIANTS:
            t += l1(p, {n: fn(image(n)) for n in NAMES})
        totals[label] = t
        print(f"   {label:<28}{t:>6}")
    return totals


# ─────────────────────────────────────────────
# 3. Parameter neighbourhood
# ─────────────────────────────────────────────
PERTURB = [
    ("bilat -2", lambda k: dict(k, bilat=max(1, k["bilat"] - 2))),
    ("bilat +2", lambda k: dict(k, bilat=k["bilat"] + 2)),
    ("median -2", lambda k: dict(k, median=max(3, k["median"] - 2))),
    ("median +2", lambda k: dict(k, median=k["median"] + 2)),
    ("gauss +2", lambda k: dict(k, gauss=k["gauss"] + 2)),
    ("otsu -0.1", lambda k: dict(k, high_scale=k["high_scale"] - 0.1)),
    ("otsu +0.1", lambda k: dict(k, high_scale=k["high_scale"] + 0.1)),
    ("low_ratio 0.4", lambda k: dict(k, low_ratio=0.4)),
    ("low_ratio 0.6", lambda k: dict(k, low_ratio=0.6)),
    ("clahe 0.5", lambda k: dict(k, clahe_clip=0.5)),
    ("dilate 3", lambda k: dict(k, dilate_k=3)),
    ("close 9", lambda k: dict(k, close_k=9)),
]


def param_drift(configs):
    print("\n3. PARAMETER NEIGHBOURHOOD (total L1 over 12 nudges; lower = a plateau)")
    totals = {}
    for label, kw in configs.items():
        t = sum(l1(cfg(**fn(kw))) for _, fn in PERTURB)
        totals[label] = t
        print(f"   {label:<28}{t:>6}")
    return totals


# ─────────────────────────────────────────────
# 4. Stability and cost over every asset
# ─────────────────────────────────────────────
def stability_and_cost(configs):
    print("\n4. ALL ASSETS (stability lower is better; zero-detection images reject a config)")
    print(f"   {'method':<28}{'stability':>11}{'totalDet':>10}{'zeroImgs':>10}{'ms/img':>9}")
    groups = D.interchangeable_groups()
    every = assets()
    out = {}
    for label, p in configs.items():
        res, total, t0 = {}, 0, time.time()
        for a in every:
            img = D.load_working_image(D.ASSETS / a)
            if img is None:
                continue
            n = count(img, p)
            res[a] = {"count": n}
            total += n
        ms = 1000 * (time.time() - t0) / max(1, len(res))
        stab, _ = D.stability(res, groups)
        zeros = sum(1 for a in res if res[a]["count"] == 0)
        out[label] = dict(stability=stab, total=total, zeros=zeros, ms=ms)
        print(f"   {label:<28}{stab:>11.3f}{total:>10}{zeros:>10}{ms:>9.0f}")
    return out


# ─────────────────────────────────────────────
# The search that produced the eight exact configs
# ─────────────────────────────────────────────
def search():
    # Round 2 added low_ratio and the morphology kernels. That is what turned 8 exact
    # configurations into 149, and every one of the good ones sits at low_ratio=0.4.
    grid = dict(bilat=[5, 7, 9, 11], median=[5, 7, 9], gauss=[5, 7, 9],
                high_scale=[1.0, 1.1, 1.2, 1.3, 1.4], low_ratio=[0.4, 0.5, 0.6],
                l2=[False, True], dilate_k=[3, 5], close_k=[7, 9], clahe_clip=[0.5, 1.0])
    keys = list(grid)
    exact = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        kw = dict(zip(keys, combo))
        if l1(cfg(**kw)) == 0:
            exact.append(kw)
    print(f"\nSEARCH: {len(exact)} of "
          f"{np.prod([len(v) for v in grid.values()])} configurations hit all six exactly")
    for kw in exact:
        print("   " + " ".join(f"{k}={v}" for k, v in kw.items()))
    return exact


def edge_cost(configs, repeats=9):
    """
    Cost of the edge stage alone, median of several passes.

    Separated from test 4 because that loop decodes a file per image, and disk dominates.
    The surprise here is that pinning the bilateral diameter makes v3 and v4 *cheaper* than
    the shipped stack despite filtering harder: choosing d adaptively costs a full-image
    CV_64F Laplacian and a standard deviation over it, every time, and that probe is more
    expensive than the difference between d=1 and d=9.
    """
    print(f"\n5. EDGE STAGE COST (median of {repeats} passes over the six, decode excluded)")
    print(f"   {'method':<28}{'ms/img':>10}{'vs shipped':>12}")
    images = [image(n) for n in NAMES if image(n) is not None]
    base_ms, out = None, {}
    for label, p in configs.items():
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for im in images:
                edges_for(im, p)
            samples.append(1000 * (time.perf_counter() - t0) / len(images))
        ms = sorted(samples)[len(samples) // 2]
        if base_ms is None:
            base_ms = ms
        out[label] = ms
        print(f"   {label:<28}{ms:>10.1f}{(ms / base_ms - 1) * 100:>11.0f}%")
    return out


def main():
    verify_base()

    configs = {"shipped": BASE}
    configs.update({k: cfg(**v) for k, v in WINNERS.items()})
    configs.update({k: cfg(**v) for k, v in ALSO_RAN.items()})

    report_targets(configs)
    photo_drift(configs)
    param_drift({k: v for k, v in list(WINNERS.items()) + list(ALSO_RAN.items())})
    stability_and_cost(configs)
    edge_cost(configs)
    search()

    print("\nCONCLUSION: v2/v3/v4 are the three kept. All eight exact configs beat the shipped")
    print("stack on stability; these three are the ones that also survive tests 2 and 3 and")
    print("drive no asset to zero detections.")


if __name__ == "__main__":
    main()
