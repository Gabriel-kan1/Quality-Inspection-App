"""
Would a feature-size signal pick the adaptive rung better than image quality alone?

Prompted by corner.jpg: noise 2.0, edge contrast 94, so ImageQuality picks MEDIUM (v6) — and
v6 finds 4 of its ~10 small fasteners plus two structural blobs, where LIGHT finds 9. The
hypothesis: when the medium pass's own boxes are small, v6's dilate 5 / close 9 is too coarse
for the hardware, and LIGHT should run.

Candidate rule, applied after ImageQuality's own (noise first, so STRONG is never displaced):
    if the rule says MEDIUM and median short side of the medium pass's boxes < T:  LIGHT
Swept over T. Scored on the 26 reference-distance photographs (weighted = 2 x misses + extras,
the number every other study ranks on) and reported separately on the 3 WIDE shots, whose
counts are approximate. Every image is chassis-masked the way the app masks it
(background_mask_study.background_mask, filled with the plate's grey).

Run:  python small_feature_rung_study.py
"""
import cv2
import numpy as np

import adaptive_preprocessing_study as A
import background_mask_study as B
import detector_ab_study as D
import detection_ground_truth as GT
import hybrid_v2_study as HV


def masked(img):
    mask, _ = B.background_mask(img)
    if not mask.any():
        return img
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p = int(np.median(g[mask == 0]))
    out = img.copy()
    out[mask > 0] = (p, p, p)
    return out


def quality_rung(q):
    if q["noise"] >= 12:
        return "strong"
    if q["contrast"] < 20:
        return "light"
    return "medium"


def main():
    rows = {}
    for name in sorted(GT.GT) + sorted(GT.WIDE):
        img = A.image(name)
        if img is None:
            continue
        work = masked(img)
        q = A.assess(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY))
        counts, med_boxes = {}, None
        for rung, p in A.LADDER.items():
            bx = D.find_feature_boxes(HV.edges_for(work, HV.cfg(**p)))[0]
            counts[rung] = len(bx)
            if rung == "medium":
                med_boxes = bx
        short = [min(b[2], b[3]) for b in med_boxes]
        rows[name] = dict(q=q, counts=counts, base=quality_rung(q),
                          med_short=float(np.median(short)) if short else 0.0)

    print("PER PHOTO (masked)                  noise contrast  medShort  light medium strong  GT   base")
    for n, r in rows.items():
        c = r["counts"]
        gt = sorted(GT.GT.get(n) or GT.WIDE.get(n))
        tag = " (WIDE)" if n in GT.WIDE else ""
        print(f"  {n + tag:<34}{r['q']['noise']:>6.1f}{r['q']['contrast']:>8.0f}"
              f"{r['med_short']:>10.1f}{c['light']:>7}{c['medium']:>7}{c['strong']:>7}"
              f"  {str(gt):<7}{r['base']}")

    def score(threshold):
        chosen = {}
        for n, r in rows.items():
            rung = r["base"]
            if rung == "medium" and threshold and r["med_short"] < threshold:
                rung = "light"
            chosen[n] = r["counts"][rung]
        w, fn, fp = GT.weighted({n: c for n, c in chosen.items() if n in GT.GT})
        exact = sum(1 for n, c in chosen.items() if n in GT.GT and GT.err(n, c) == 0)
        wide = {n: chosen[n] for n in GT.WIDE if n in chosen}
        switched = [n for n, r in rows.items() if r["base"] == "medium"
                    and threshold and r["med_short"] < threshold]
        return w, fn, fp, exact, wide, switched

    print("\nRULE SWEEP  (T = 0 is today's image-quality rule)")
    print("   T   weighted misses extras exact/26   wide counts (corner/whole/whole_light)  switched to light")
    for t in (0, 20, 25, 30, 35, 40, 50, 60):
        w, fn, fp, ex, wide, sw = score(t)
        wd = "/".join(str(wide.get(k, "-")) for k in ("corner.jpg", "whole.jpg", "whole_light.jpg"))
        print(f"  {t:>3} {w:>9} {fn:>6} {fp:>6} {ex:>8}      {wd:<40}{', '.join(sw)}")

    print("\nFOR SCALE: every photo on one rung")
    for rung in A.LADDER:
        w, fn, fp = GT.weighted({n: r["counts"][rung] for n, r in rows.items() if n in GT.GT})
        print(f"  {rung:<7} weighted {w} ({fn} misses, {fp} extras)")


if __name__ == "__main__":
    main()
