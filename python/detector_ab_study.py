"""
A/B harness for HardwareFeatureExtractor's geometry pipeline.

Ports the *current Java* detector faithfully (not EdgeComputingImprove.py, which has
diverged), then re-runs it over app/src/main/assets/chassis_images/ with individual
changes switched on, so a proposed change is judged on measured behaviour rather than
on how convincing its rationale reads.

WHAT "BETTER" MEANS HERE
------------------------
The app decides pass/fail by counting detections on the inspection photo, counting them
again on the reference photo, and failing the part on any per-type difference. The
alignment gate only lets a photo through when it shows the same area at the same
distance as the reference. So the quantity that decides the false-FAIL rate is the
spread of detection counts across photos the gate would consider interchangeable.

Groups are taken from scale_probe_study.py's CATALOGUE, which labels each asset with
(area, relative shooting distance). Photos sharing both are what the gate treats as
equivalent, so disagreement inside such a group is a false FAIL waiting to happen.

Two guards on the metric:
  * defected.jpg is excluded — it differs from reference.jpg on purpose, and counting
    that as instability would reward a detector for missing real defects.
  * Stability is always reported next to the detection level. A detector that finds one
    box per image scores a perfect spread and is worthless, so a variant only wins if
    it holds detection counts up while bringing spread down.

Run:  python python/detector_ab_study.py
Note: desktop OpenCV (4.13 here) against the app's 4.14.0 Android build. The primitives used — Canny,
morphology, findContours, bilateralFilter — are stable across that gap, but treat small
numeric differences as noise rather than signal.
"""

import time
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "app/src/main/assets/chassis_images"

# ─────────────────────────────────────────────
# CONFIG — mirrors HardwareFeatureExtractor's block one-for-one.
# ─────────────────────────────────────────────
WORKING_MAX_DIM = 1280
GLARE_RATIO_THRESHOLD = 0.05
GLARE_KERNEL_SIZE = 75
CLAHE_CLIP_LIMIT = 1.0
BILATERAL_SIGMA_COLOR = 60
BILATERAL_SIGMA_SPACE = 20
CANNY_LOW_RATIO = 0.5
DILATE_KERNEL_SIZE = 5
CLOSE_KERNEL_SIZE = 7

MAX_ASPECT_RATIO = 2.5
MIN_ASPECT_RATIO = 0.4
DYNAMIC_AREA_MULTIPLIER = 0.15
OUTLIER_AREA_FACTOR = 10.0
PROXIMITY_THRESHOLD = 10
FILL_RATIO_THRESHOLD = 0.7
MAX_DETECTIONS = 300
# Whether a contour below the dynamic area floor may still join a box that passed it.
# Mirrors HardwareFeatureExtractor.RESCUE_ADJACENT_FRAGMENTS; see fragment_rescue_study.py.
RESCUE_FRAGMENTS = True

# (area, relative shooting distance) — copied from scale_probe_study.py.
CATALOGUE = {
    "screws_test.jpg": ("fasteners", 0.8),
    "stud1.jpg": ("fasteners", 1.0),
    "stud2.jpg": ("fasteners", 1.0),
    "stud3.jpg": ("fasteners", 1.0),
    "stud4.jpg": ("fasteners", 1.0),
    "screws.jpg": ("fasteners", 1.2),
    "screws-1.jpg": ("fasteners", 1.2),
    "reference.jpg": ("fasteners", 1.6),
    "defected.jpg": ("fasteners", 1.6),
    "screws2.jpg": ("fasteners", 2.5),
    "screws_two.jpg": ("fasteners", 2.5),
    "screws_three.jpg": ("fasteners", 2.8),
    "screws_flash.jpg": ("fasteners", 2.8),
    "screws_none.jpg": ("fasteners", 4.0),
    "whole screws1.jpg": ("fasteners", 10.0),
    "whole screws2.jpg": ("fasteners", 10.0),
    "whole screws3.jpg": ("fasteners", 10.0),
    "head1.jpg": ("heads", 1.0),
    "head2.jpg": ("heads", 1.0),
    "head3.jpg": ("heads", 1.0),
    "head4.jpg": ("heads", 1.0),
    "hole1.jpg": ("bracket", 1.0),
    "hole2.jpg": ("bracket", 1.2),
    "corner.jpg": ("corner", 1.0),
    "zero.jpg": ("zero", 1.0),
    "one.jpg": ("one", 1.0),
    "three.jpg": ("three", 1.0),
    "whole.jpg": ("plate", 1.0),
    "whole_light.jpg": ("plate", 1.0),
}

# Differs from its group on purpose; excluded from the stability metric.
DEFECT_ASSETS = {"defected.jpg"}


def interchangeable_groups():
    """Assets the alignment gate would treat as equivalent: same area, same distance."""
    groups = {}
    for name, (area, distance) in CATALOGUE.items():
        if name in DEFECT_ASSETS:
            continue
        groups.setdefault((area, distance), []).append(name)
    return {k: sorted(v) for k, v in groups.items() if len(v) >= 2}


def load_working_image(path):
    """Mirrors loadBitmap(): longest edge scaled to exactly WORKING_MAX_DIM."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    longest = max(w, h)
    if longest > WORKING_MAX_DIM:
        f = WORKING_MAX_DIM / longest
        img = cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))),
                         interpolation=cv2.INTER_AREA)
    return img


def bilateral_d_for(gray, mode):
    """
    computeAdaptiveBilateralD(), plus the variants under test.

    mode='adaptive' is the shipping behaviour: bucket the Laplacian variance into d=1..6.
    An integer mode pins d to that value for every image.
    """
    if isinstance(mode, int):
        return mode, float(np.std(cv2.Laplacian(gray, cv2.CV_64F))) ** 2
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    noise_variance = float(np.std(lap)) ** 2
    for limit, d in ((200, 1), (300, 2), (400, 3), (600, 4), (700, 5)):
        if noise_variance < limit:
            return d, noise_variance
    return 6, noise_variance


def compute_edges(img, bilateral_mode="adaptive"):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    glare_ratio = float(np.count_nonzero(gray > 240)) / gray.size
    if glare_ratio > GLARE_RATIO_THRESHOLD:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (GLARE_KERNEL_SIZE, GLARE_KERNEL_SIZE))
        preprocessed = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    else:
        preprocessed = gray.copy()

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(16, 16))
    img_clahe = clahe.apply(preprocessed)

    d, noise_variance = bilateral_d_for(preprocessed, bilateral_mode)
    img_filtered = cv2.bilateralFilter(img_clahe, d, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)

    img_smoothed = cv2.medianBlur(img_filtered, 5)
    img_smoothed = cv2.GaussianBlur(img_smoothed, (5, 5), 0)

    otsu_high, _ = cv2.threshold(img_smoothed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edges = cv2.Canny(img_smoothed, otsu_high * CANNY_LOW_RATIO, otsu_high)

    dilated = cv2.dilate(edges, cv2.getStructuringElement(
        cv2.MORPH_RECT, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE)))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)))

    return closed, {
        "glare_applied": glare_ratio > GLARE_RATIO_THRESHOLD,
        "bilateral_d": d,
        "noise_variance": noise_variance,
        "bilateral_changed_pixels": int(np.count_nonzero(img_clahe != img_filtered)),
    }


def remove_nested_indices(boxes, apply_aspect_guard=True):
    """Indices removeNestedBoxes() keeps, so a caller can carry per-box state through it."""
    out = []
    for i, r1 in enumerate(boxes):
        inside = False
        for j, r2 in enumerate(boxes):
            if i == j:
                continue
            if apply_aspect_guard:
                aspect = r2[2] / r2[3]
                if aspect > MAX_ASPECT_RATIO or aspect < MIN_ASPECT_RATIO:
                    continue
            if (r1[0] >= r2[0] and r1[1] >= r2[1]
                    and r1[0] + r1[2] <= r2[0] + r2[2]
                    and r1[1] + r1[3] <= r2[1] + r2[3]):
                inside = True
                break
        if not inside:
            out.append(i)
    return out


def remove_nested(boxes, apply_aspect_guard=True):
    """removeNestedBoxes(). apply_aspect_guard=False drops the guard under test."""
    return [boxes[i] for i in remove_nested_indices(boxes, apply_aspect_guard)]


def adjacent(r1, r2, threshold=PROXIMITY_THRESHOLD):
    """The merge stage's own proximity test, as a predicate."""
    h_gap = max(0, max(r1[0], r2[0]) - min(r1[0] + r1[2], r2[0] + r2[2]))
    v_gap = max(0, max(r1[1], r2[1]) - min(r1[1] + r1[3], r2[1] + r2[3]))
    return h_gap <= threshold and v_gap <= threshold


def merge_nearby(boxes, threshold=PROXIMITY_THRESHOLD, anchors=None):
    """
    mergeNearbyBoxes(). With `anchors` (one bool per box, True = passed the area floor),
    the flag is OR-ed through every merge and returned alongside the boxes, so the caller
    can drop groups made only of rescued fragments.
    """
    working = list(boxes)
    content = [float(r[2] * r[3]) for r in boxes]
    anchor = [True] * len(boxes) if anchors is None else list(anchors)
    changed = True
    while changed:
        changed = False
        nxt, nxt_content, nxt_anchor = [], [], []
        skipped = [False] * len(working)
        for i in range(len(working)):
            if skipped[i]:
                continue
            r1, c1, a1 = working[i], content[i], anchor[i]
            skipped[i] = True
            for j in range(i + 1, len(working)):
                if skipped[j]:
                    continue
                if adjacent(r1, working[j], threshold):
                    r2 = working[j]
                    nx, ny = min(r1[0], r2[0]), min(r1[1], r2[1])
                    nw = max(r1[0] + r1[2], r2[0] + r2[2]) - nx
                    nh = max(r1[1] + r1[3], r2[1] + r2[3]) - ny
                    merged_content = c1 + content[j]
                    if merged_content >= nw * nh * FILL_RATIO_THRESHOLD:
                        r1, c1 = (nx, ny, nw, nh), merged_content
                        a1 = a1 or anchor[j]
                        skipped[j] = True
                        changed = True
            nxt.append(r1)
            nxt_content.append(c1)
            nxt_anchor.append(a1)
        working, content, anchor = nxt, nxt_content, nxt_anchor
    return (working, anchor) if anchors is not None else working


def find_feature_boxes(edges, cap_early=False, aspect_guard=True, rescue=None):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_boxes, valid_areas = [], []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect = w / h
        if aspect > MAX_ASPECT_RATIO or aspect < MIN_ASPECT_RATIO:
            continue
        valid_boxes.append((x, y, w, h))
        valid_areas.append(cv2.contourArea(c))

    stats = {"contours": len(contours), "passed_geometry": len(valid_boxes),
             "passed_area": 0, "capped": False}
    if not valid_boxes:
        return [], stats

    ascending = sorted(valid_areas)
    median_area = ascending[len(ascending) // 2]
    ceiling = median_area * OUTLIER_AREA_FACTOR
    pool = [a for a in valid_areas if a <= ceiling]
    if len(pool) < 3:
        pool = list(valid_areas)
    pool.sort(reverse=True)
    reference_area = sum(pool[:3]) / min(3, len(pool))
    dynamic_min_area = max(25.0, reference_area * DYNAMIC_AREA_MULTIPLIER)

    sized = [b for b, a in zip(valid_boxes, valid_areas) if a >= dynamic_min_area]
    stats["passed_area"] = len(sized)

    # A contour below the floor that TOUCHES one above it is a fragment of that feature
    # rather than noise of its own: Canny breaks a low-contrast outline into pieces, and the
    # floor runs before the merge that would have rejoined them. Such a fragment is carried
    # into the merge but can never become a box of its own. See fragment_rescue_study.py.
    do_rescue = RESCUE_FRAGMENTS if rescue is None else rescue
    rescued = []
    if do_rescue and sized:
        rescued = [b for b, a in zip(valid_boxes, valid_areas)
                   if a < dynamic_min_area and any(adjacent(b, s) for s in sized)]
    stats["rescued"] = len(rescued)

    if cap_early and len(sized) > MAX_DETECTIONS:
        # Bound the input to the O(n^2) stages instead of only their output.
        sized = sorted(sized, key=lambda r: r[2] * r[3], reverse=True)[:MAX_DETECTIONS]
        rescued = []
        stats["capped"] = True

    candidates = sized + rescued
    anchors = [True] * len(sized) + [False] * len(rescued)
    kept = remove_nested_indices(candidates, apply_aspect_guard=aspect_guard)
    de_nested = [candidates[i] for i in kept]
    de_anchors = [anchors[i] for i in kept]
    boxes, box_anchors = merge_nearby(de_nested, anchors=de_anchors)
    # A group of rescued fragments that never reached a box above the floor stays dropped.
    boxes = [b for b, a in zip(boxes, box_anchors) if a]

    if len(boxes) > MAX_DETECTIONS:
        boxes = sorted(boxes, key=lambda r: r[2] * r[3], reverse=True)[:MAX_DETECTIONS]
        stats["capped"] = True

    return boxes, stats


def run_variant(images, cfg):
    results = {}
    for name, img in images.items():
        t0 = time.perf_counter()
        edges, edge_stats = compute_edges(img, cfg["bilateral"])
        boxes, stats = find_feature_boxes(
            edges, cap_early=cfg["cap_early"], aspect_guard=cfg["aspect_guard"])
        results[name] = {"count": len(boxes), "ms": (time.perf_counter() - t0) * 1000,
                         **stats, **edge_stats}
    return results


def stability(results, groups):
    """Mean relative spread across interchangeable groups; 0 = every group agrees."""
    rows, rel = [], []
    for (area, distance), members in sorted(groups.items()):
        counts = [results[m]["count"] for m in members if m in results]
        if len(counts) < 2:
            continue
        spread = max(counts) - min(counts)
        mean = float(np.mean(counts))
        r = spread / mean if mean else 0.0
        rel.append(r)
        rows.append((f"{area}@{distance}", counts, spread, r))
    return (float(np.mean(rel)) if rel else 0.0), rows


def pathological_case():
    """
    Worst case for the merge stages: dense high-frequency noise, which is what a
    low-contrast or badly-lit photo degenerates into. Used to check whether capping
    before the O(n^2) stages actually bounds anything.
    """
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, (960, 1280), dtype=np.uint8)


def main():
    groups = interchangeable_groups()
    paths = sorted(p for p in ASSETS.glob("*.jpg"))
    images = {}
    for p in paths:
        img = load_working_image(p)
        if img is not None:
            images[p.name] = img
    print(f"Loaded {len(images)} assets; {len(groups)} interchangeable groups "
          f"(same area, same distance, >=2 members).\n")

    variants = {
        "V0_baseline_adaptive_d": dict(cap_early=False, aspect_guard=True, bilateral="adaptive"),
        "V1_cap_early": dict(cap_early=True, aspect_guard=True, bilateral="adaptive"),
        "V2_no_dead_guard": dict(cap_early=False, aspect_guard=False, bilateral="adaptive"),
    }
    for d in (0, 3, 5, 7, 9):
        label = "d0_auto_from_sigma" if d == 0 else f"d{d}_fixed"
        variants[f"V3_{label}"] = dict(cap_early=False, aspect_guard=True, bilateral=d)

    all_results = {v: run_variant(images, cfg) for v, cfg in variants.items()}
    base = all_results["V0_baseline_adaptive_d"]
    base_stab, base_rows = stability(base, groups)

    print("=" * 92)
    print("BASELINE (V0) — current shipping behaviour, per interchangeable group")
    print("=" * 92)
    for label, counts, spread, rel in base_rows:
        print(f"  {label:<22} counts={str(counts):<28} spread={spread:<5} relative={rel:.2f}")
    print(f"\n  mean relative spread = {base_stab:.3f}   (0 = every group agrees)")
    bt = [r["ms"] for r in base.values()]
    print(f"  total boxes={sum(r['count'] for r in base.values())}  "
          f"time mean={np.mean(bt):.0f}ms max={max(bt):.0f}ms  "
          f"cap hit={sum(1 for r in base.values() if r['capped'])}/{len(base)}")
    unchanged = sum(1 for r in base.values() if r["bilateral_changed_pixels"] == 0)
    print(f"  images where the bilateral stage changed nothing: {unchanged}/{len(base)}")

    print("\n" + "=" * 92)
    print(f"{'variant':<26}{'boxes':>7}{'changed':>9}{'relspread':>11}{'mean ms':>9}{'max ms':>9}")
    print("=" * 92)
    for variant, res in all_results.items():
        stab, _ = stability(res, groups)
        changed = sum(1 for n in res if res[n]["count"] != base[n]["count"])
        t = [r["ms"] for r in res.values()]
        print(f"{variant:<26}{sum(r['count'] for r in res.values()):>7}{changed:>9}"
              f"{stab:>11.3f}{np.mean(t):>9.0f}{max(t):>9.0f}")

    print("\n" + "=" * 92)
    print("PATHOLOGICAL INPUT — dense noise, the case MAX_DETECTIONS exists for")
    print("=" * 92)
    noise = pathological_case()
    dilated = cv2.dilate(noise, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    edges = cv2.Canny(dilated, 50, 150)
    for label, cap_early in (("cap after merge (current)", False), ("cap before merge", True)):
        t0 = time.perf_counter()
        boxes, stats = find_feature_boxes(edges, cap_early=cap_early)
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  {label:<28} boxes_into_merge={stats['passed_area']:<7} "
              f"out={len(boxes):<5} {elapsed:>9.0f}ms")


if __name__ == "__main__":
    main()
