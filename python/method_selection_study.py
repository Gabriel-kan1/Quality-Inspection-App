"""
Can the edge method be chosen per photograph by looking at what each method *produced*,
rather than by thresholds on proxies (noise, contrast, feature size)?

The adaptive stage decides from measurements of the photograph and only chooses among three
strengths of v6; every other method in the Edge Lab is never on the ballot. This study asks
two questions against detection_ground_truth.py (hand counts, 26 reference-distance photos):

  1. HEADROOM — per photograph, how good is the best method? If an oracle picking the best
     method per photo barely beats v6, no selector is worth building.
  2. A REFERENCE-FREE SELECTOR — consensus. Run every candidate method; a box that several
     independent methods also find is probably hardware, a box only one method finds is
     probably that method's artefact. Score each method by F1 against the boxes a majority
     agrees on, and run the best. Nothing is tuned per photo and no threshold is on the
     photograph's statistics.

Every photo is chassis-masked as the app masks it. Counts only: the ground truth has no box
positions, so a method can score exact with a wrong box — that limit applies to every number
here, and to every study before it.

Run:  python method_selection_study.py
"""
import cv2
import numpy as np

import adaptive_preprocessing_study as A
import background_mask_study as B
import detector_ab_study as D
import detection_ground_truth as GT
import hybrid_v2_study as HV
from small_feature_rung_study import masked


def clahe_bilateral_canny(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.createCLAHE(1.0, (16, 16)).apply(g)
    f = cv2.bilateralFilter(eq, 5, 60, 20)
    hi, _ = cv2.threshold(f, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return cv2.Canny(f, hi * 0.5, hi)


METHODS = {
    "shipped": lambda im: HV.edges_for(im, HV.cfg()),
    "clahe-bil-canny": clahe_bilateral_canny,
    "v2": lambda im: HV.edges_for(im, HV.cfg(**HV.WINNERS["v2 bilat9 low0.4"])),
    "v3": lambda im: HV.edges_for(im, HV.cfg(**HV.WINNERS["v3 gauss7 otsu1.2"])),
    "v4": lambda im: HV.edges_for(im, HV.cfg(**HV.WINNERS["v4 L2 light-morph"])),
    "light": lambda im: HV.edges_for(im, HV.cfg(**A.LADDER["light"])),
    "v6": lambda im: HV.edges_for(im, HV.cfg(**A.LADDER["medium"])),
    "strong": lambda im: HV.edges_for(im, HV.cfg(**A.LADDER["strong"])),
}


def iou(a, b):
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter) if inter else 0.0


def same(a, b):
    """Same physical feature: overlapping boxes, or one's centre inside the other."""
    if iou(a, b) >= 0.3:
        return True
    cx, cy = a[0] + a[2] / 2, a[1] + a[3] / 2
    return b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3] and \
        max(a[2] * a[3], b[2] * b[3]) <= 4 * min(a[2] * a[3], b[2] * b[3])


def consensus_pick(boxes_by_method):
    names = list(boxes_by_method)
    quorum = len(names) // 2 + 1
    # A box is agreed if a majority of methods (itself included) found the same feature.
    agreed = []
    for m in names:
        for b in boxes_by_method[m]:
            votes = sum(1 for o in names if any(same(b, c) for c in boxes_by_method[o]))
            if votes >= quorum and not any(same(b, c) for c in agreed):
                agreed.append(b)
    scores = {}
    for m in names:
        bx = boxes_by_method[m]
        tp_p = sum(1 for b in bx if any(same(b, c) for c in agreed))
        tp_r = sum(1 for c in agreed if any(same(b, c) for b in bx))
        p = tp_p / len(bx) if bx else 0.0
        r = tp_r / len(agreed) if agreed else (1.0 if not bx else 0.0)
        scores[m] = 2 * p * r / (p + r) if p + r else 0.0
    # Ties go to v6, the method with the best record, so the selector moves only on evidence.
    best = max(names, key=lambda m: (round(scores[m], 6), m == "v6"))
    return best, scores, len(agreed)


def main():
    counts, picks, agreed_n = {}, {}, {}
    for name in sorted(GT.GT) + sorted(GT.WIDE):
        img = A.image(name)
        if img is None:
            continue
        work = masked(img)
        boxes = {m: D.find_feature_boxes(f(work))[0] for m, f in METHODS.items()}
        counts[name] = {m: len(b) for m, b in boxes.items()}
        picks[name], _, agreed_n[name] = consensus_pick(boxes)

    names = list(METHODS)
    print(f"{'photo':<22}" + "".join(f"{m[:9]:>10}" for m in names) + "   GT      consensus")
    for n, c in counts.items():
        gt = sorted(GT.GT.get(n) or GT.WIDE.get(n))
        row = "".join(f"{c[m]:>10}" for m in names)
        print(f"{n[:21]:<22}{row}   {str(gt):<8}{picks[n]} (agreed {agreed_n[n]})")

    gt26 = [n for n in counts if n in GT.GT]
    print("\nWEIGHTED over the 26 (2 x misses + extras; lower is better)")
    for m in names:
        w, fn, fp = GT.weighted({n: counts[n][m] for n in gt26})
        ex = sum(1 for n in gt26 if GT.err(n, counts[n][m]) == 0)
        print(f"  {m:<16} {w:>5}  ({fn} misses, {fp} extras, {ex}/26 exact)")
    w, fn, fp = GT.weighted({n: counts[n][picks[n]] for n in gt26})
    ex = sum(1 for n in gt26 if GT.err(n, counts[n][picks[n]]) == 0)
    print(f"  {'CONSENSUS PICK':<16} {w:>5}  ({fn} misses, {fp} extras, {ex}/26 exact)")
    oracle = {n: min((2 * GT.split(n, counts[n][m])[0] + GT.split(n, counts[n][m])[1], m)
                     for m in names)[1] for n in gt26}
    w, fn, fp = GT.weighted({n: counts[n][oracle[n]] for n in gt26})
    print(f"  {'ORACLE (bound)':<16} {w:>5}  ({fn} misses, {fp} extras)")
    print("\n  consensus picked: " + ", ".join(
        f"{m} x{sum(1 for n in gt26 if picks[n] == m)}" for m in names
        if any(picks[n] == m for n in gt26)))
    for n in GT.WIDE:
        if n in counts:
            print(f"  {n}: consensus -> {picks[n]} = {counts[n][picks[n]]} (GT {sorted(GT.WIDE[n])})")


if __name__ == "__main__":
    main()
