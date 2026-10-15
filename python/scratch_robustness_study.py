"""
Can the detector be made robust to a scratched plate?

python/photos/l_plate_scratched.jpg is an L-shaped aluminium plate with five pan-head screws
(two large, two medium, one small) and a surface scratched in every direction. On the device
no edge method in the catalogue isolates all five without inventing others:

  * the two large heads sit against the plate's inside edge, and dilate/close welds their
    outlines onto it, so the screws and the plate become one contour that is either refused
    as structural or boxed as a strip across the part;
  * a long diagonal scratch runs into the right-hand screw and is welded on the same way;
  * every short scratch long enough to survive the area floor is a box of its own.

This study scores candidate fixes as stages added to the existing methods rather than as
new methods, so a fix that works can be offered to every method in the catalogue.

DATA
----
  1. The L-plate, with the five screws' positions marked by hand (GT_L, original pixels).
     Scored positionally: a screw counts as found only if one box isolates it (IoU >= 0.3
     with a square of the head's diameter). A count can be right with every box wrong —
     Hybrid v8 "finds 5" here with one box over three screws and two on the plate edge.
  2. The 26 hand-counted reference-distance photographs (detection_ground_truth.GT), so a fix
     that helps the L-plate is not bought with misses everywhere else.
  3. The same 26 with synthetic scratches drawn on (thin, slightly curved, bright or dark,
     blurred), at increasing density, seeded. A stress test, not a substitute for real plates.

Every image goes through what the app does before any method: ChassisMask's background, filled
with the plate's grey (ChassisMask.fillWithPlate). No seam is cut — see verify_fidelity().

Run:  python python/scratch_robustness_study.py
"""

import math
import sys
import time
import zlib

import cv2
import numpy as np

import adaptive_preprocessing_study as A
import background_mask_study as B
import detector_ab_study as D
import detection_ground_truth as GT
import hybrid_v2_study as HV

PHOTOS = D.ASSETS.parent.parent.parent.parent.parent / "python" / "photos"
L_PLATE = PHOTOS / "l_plate_scratched.jpg"
# (x, y, radius) of each screw head in the original 1500x2000 photograph, marked by eye and
# checked against a Hough circle fit: large left, large right, medium top, small, medium right.
GT_L = [(497, 1160, 71), (756, 1148, 72), (1086, 296, 50), (912, 738, 38), (1086, 1072, 52)]
MATCH_IOU = 0.3


# ─────────────────────────────────────────────
# The pipeline, with the hook points the fixes plug into
# ─────────────────────────────────────────────

def hv(name):
    return HV.cfg(**{"v6": A.LADDER["medium"], "v8": A.LADDER["strong"],
                     "v7": A.LADDER["light"], "v2": HV.WINNERS["v2 bilat9 low0.4"],
                     "v4": HV.WINNERS["v4 L2 light-morph"], "v1": {}}[name])


def smoothed_and_canny(img, p):
    """HV.edges_for up to Canny, returning the image Canny saw as well as its edges."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if float(np.count_nonzero(g > 240)) / g.size > p["glare_thresh"]:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["glare_k"], p["glare_k"]))
        g = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)
    post_glare = g
    pre = cv2.createCLAHE(clipLimit=p["clahe_clip"], tileGridSize=(p["clahe_tiles"],) * 2).apply(g)
    d = D.bilateral_d_for(post_glare, p["bilat"])[0] if p["bilat"] == "adaptive" else p["bilat"]
    pre = cv2.bilateralFilter(pre, d, p["bilat_sc"], p["bilat_ss"])
    pre = cv2.medianBlur(pre, p["median"])
    pre = cv2.GaussianBlur(pre, (p["gauss"],) * 2, 0)
    hi, _ = cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    hi *= p["high_scale"]
    return pre, cv2.Canny(pre, hi * p["low_ratio"], hi, L2gradient=p["l2"])


def morph(e, p):
    e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_RECT, (p["dilate_k"],) * 2))
    return cv2.morphologyEx(e, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["close_k"],) * 2))


def masked_input(img):
    """ChassisMask as the app applies it: background filled with the plate's grey."""
    mask, _ = B.background_mask(img)
    work = img.copy()
    if np.any(mask):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plate = int(np.median(g[mask == 0])) if np.any(mask == 0) else 128
        work[mask > 0] = (plate, plate, plate)
    return work, mask


def run(img, method="v6", raw_fix=None, contour_fix=None, seam=False, open_k=0):
    """
    Boxes for one image. raw_fix(canny, smoothed) edits the raw Canny map before the
    morphology joins anything; contour_fix(contour) -> bool vetoes a contour before the
    geometry filter sees it.
    """
    work, mask = masked_input(img)
    p = hv(method)
    pre, e = smoothed_and_canny(work, p)
    if raw_fix is not None:
        e = raw_fix(e, pre)
    e = morph(e, p)
    if open_k:
        # Opening with a disk wider than any line the morphology produced: the plate's outline
        # and every scratch are strokes 5-9 px wide after dilate/close and are erased, a fastener
        # is a thick blob of texture and survives. This is what breaks the weld between a head
        # and the plate edge it sits against, which no shape test after the fact can undo.
        e = cv2.morphologyEx(e, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    if seam and np.any(mask):
        # What background_mask_study measured and ChassisMask's comment reports, but the app
        # does not do: erase edges within the SEAM kernel of the painted-out background.
        e[cv2.dilate(mask, B.SEAM) > 0] = 0
    if contour_fix is not None:
        contours, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        keep = np.zeros_like(e)
        cv2.drawContours(keep, [c for c in contours if contour_fix(c)], -1, 255, cv2.FILLED)
        e = cv2.bitwise_and(e, keep)
    return D.find_feature_boxes(e)[0]


def verify_fidelity():
    """
    At no fix, run() must equal the shipped edge stage on the grey-filled image. This is what
    HardwareFeatureExtractor does — ChassisMask.fillWithPlate and nothing more. Note that
    background_mask_study.detect(..., "grey") ALSO cuts the seam out of the edge map; the app
    does not, and on the L-plate that difference is the whole story (the cut erases the plate
    edge the large screws weld onto, so the desktop found 4 of 5 where the device found 2).
    """
    for name in sorted(GT.GT):
        img = HV.image(name)
        work, _ = masked_input(img)
        ours = sorted(run(img, "v6"))
        theirs = sorted(D.find_feature_boxes(HV.edges_for(work, hv("v6")))[0])
        assert ours == theirs, f"{name}: {len(ours)} vs {len(theirs)}"


# ─────────────────────────────────────────────
# Candidate fixes
# ─────────────────────────────────────────────

def hough_erase(min_len=40, gap=4, width=5, thresh=30):
    """
    Erase long straight runs from the raw Canny map before morphology. A scratch and the
    plate's own edge are straight for tens of pixels; a screw head's outline is a circle and
    its recess is shorter than min_len. Erasing them before dilate/close is what stops a head
    being welded onto the edge it sits beside.
    """
    def fix(e, _pre):
        lines = cv2.HoughLinesP(e, 1, np.pi / 180, thresh, minLineLength=min_len, maxLineGap=gap)
        if lines is not None:
            e = e.copy()
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(e, (int(x1), int(y1)), (int(x2), int(y2)), 0, width)
        return e
    return fix


def coherence_gate(sigma=6.0, limit=0.6):
    """
    Drop edge pixels where the image's local orientation is coherent over a window the size
    of a small head. Along a scratch or a plate edge every gradient in the window points the
    same way; across a screw head the rim and the recess point every way.
    """
    def fix(e, pre):
        f = pre.astype(np.float32)
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
        jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
        jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
        tr = jxx + jyy + 1e-6
        root = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
        coh = root / tr                     # (l1 - l2) / (l1 + l2), 0 isotropic .. 1 a line
        out = e.copy()
        out[coh > limit] = 0
        return out
    return fix


def ridge_inpaint(scales=(1.0, 2.0), limit=6.0):
    """
    Find thin bright or dark lines with the Hessian (one large eigenvalue, one small) and
    paint them out of the image before Canny. Works on the grey image, so a scratch never
    becomes an edge at all, rather than being removed after the fact.
    """
    def ridges(gray):
        f = gray.astype(np.float32)
        best = np.zeros_like(f)
        for s in scales:
            b = cv2.GaussianBlur(f, (0, 0), s)
            dxx = cv2.Sobel(b, cv2.CV_32F, 2, 0, ksize=3)
            dyy = cv2.Sobel(b, cv2.CV_32F, 0, 2, ksize=3)
            dxy = cv2.Sobel(b, cv2.CV_32F, 1, 1, ksize=3)
            root = np.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
            l1 = 0.5 * (dxx + dyy + root)
            l2 = 0.5 * (dxx + dyy - root)
            big = np.where(np.abs(l1) > np.abs(l2), l1, l2)
            small = np.where(np.abs(l1) > np.abs(l2), l2, l1)
            line = (np.abs(big) - np.abs(small)) * s * s     # anisotropy, scale-normalised
            best = np.maximum(best, line)
        return best > limit
    return ridges


def run_ridge(img, method="v6", **kw):
    """ridge_inpaint applied to the input image, then the method as usual."""
    work, mask = masked_input(img)
    g = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    r = ridge_inpaint(**kw)(g)
    r = cv2.dilate(r.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    painted = cv2.inpaint(work, r, 3, cv2.INPAINT_TELEA)
    p = hv(method)
    pre, e = smoothed_and_canny(painted, p)
    e = morph(e, p)
    return D.find_feature_boxes(e)[0]


def rotated_elongation(limit=2.5, min_solidity=0.0):
    """
    The aspect band, measured on the contour's minimum-area rectangle instead of its upright
    bounding box. The shipped filter tests the upright box, so a scratch at 45 degrees has
    aspect ~1 and passes; rotated, it is as elongated as it looks.
    """
    def fix(c):
        if len(c) < 5:
            return True
        (_, _), (w, h), _ = cv2.minAreaRect(c)
        if min(w, h) < 1:
            return False
        if max(w, h) / min(w, h) > limit:
            return False
        if min_solidity > 0:
            hull = cv2.contourArea(cv2.convexHull(c))
            if hull > 0 and cv2.contourArea(c) / hull < min_solidity:
                return False
        return True
    return fix


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def l_plate_working():
    img = D.load_working_image(L_PLATE)
    f = img.shape[0] / 2000.0
    gts = [(x * f - r * f, y * f - r * f, 2 * r * f, 2 * r * f) for x, y, r in GT_L]
    return img, gts


def score_l(boxes, gts):
    """(found, merged, extras): screws isolated by a box, boxes over 2+ heads, unmatched boxes."""
    used = set()
    found = 0
    for g in gts:
        best, bi = 0.0, -1
        for i, b in enumerate(boxes):
            if i in used:
                continue
            v = D_iou(b, g)
            if v > best:
                best, bi = v, i
        if best >= MATCH_IOU:
            found += 1
            used.add(bi)
    centres = [(g[0] + g[2] / 2, g[1] + g[3] / 2) for g in gts]
    merged = sum(1 for b in boxes
                 if sum(b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]
                        for cx, cy in centres) >= 2)
    return found, merged, len(boxes) - len(used)


def D_iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter) if inter else 0.0


def scratched(img, n, seed):
    """n synthetic scratches: thin polylines, bright or dark, slightly curved, blurred."""
    if n == 0:
        return img
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    delta = np.zeros((h, w), np.float32)
    for _ in range(n):
        length = rng.uniform(40, 0.45 * max(h, w))
        angle = rng.uniform(0, np.pi)
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        pts = [(x, y)]
        for _ in range(3):
            angle += rng.normal(0, 0.12)
            x += math.cos(angle) * length / 3
            y += math.sin(angle) * length / 3
            pts.append((x, y))
        amp = rng.uniform(25, 60) if rng.random() < 0.7 else -rng.uniform(20, 40)
        layer = np.zeros((h, w), np.float32)
        cv2.polylines(layer, [np.int32(pts)], False, 1.0, int(rng.integers(1, 3)), cv2.LINE_AA)
        delta += amp * layer
    delta = cv2.GaussianBlur(delta, (0, 0), 0.7)
    out = img.astype(np.float32) + delta[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def weighted_over_gt(detect, density=0, seeds=(1,)):
    tot = fn_t = fp_t = 0
    for s in seeds:
        counts = {n: len(detect(scratched(HV.image(n), density, zlib.crc32(f"{n}/{s}".encode()))))
                  for n in sorted(GT.GT)}
        w, fn, fp = GT.weighted(counts)
        tot += w
        fn_t += fn
        fp_t += fp
    k = len(seeds)
    return tot / k, fn_t / k, fp_t / k


# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────

def variants():
    v = {"(none)": dict()}
    for L in (30, 40, 60):
        v[f"hough erase len{L}"] = dict(raw_fix=hough_erase(min_len=L))
    for s, lim in ((4, 0.6), (6, 0.6), (6, 0.7), (8, 0.7)):
        v[f"coherence s{s} >{lim}"] = dict(raw_fix=coherence_gate(s, lim))
    v["rotated aspect 2.5"] = dict(contour_fix=rotated_elongation(2.5))
    v["rotated aspect 2.0"] = dict(contour_fix=rotated_elongation(2.0))
    v["rotated 2.5 + solidity .6"] = dict(contour_fix=rotated_elongation(2.5, 0.6))
    v["hough40 + rotated 2.5"] = dict(raw_fix=hough_erase(min_len=40),
                                      contour_fix=rotated_elongation(2.5))
    v["seam cut"] = dict(seam=True)
    v["seam cut + rotated 2.5"] = dict(seam=True, contour_fix=rotated_elongation(2.5))
    v["seam + hough60 + rot 2.5"] = dict(seam=True, raw_fix=hough_erase(min_len=60),
                                         contour_fix=rotated_elongation(2.5))
    return v


def main():
    verify_fidelity()
    print("fidelity: run() == background_mask_study.detect(grey) under v6 on all 26  OK\n")
    img, gts = l_plate_working()
    density = (0, 30, 80)
    methods = ["v6", "v1", "v8", "v4"] if "--all" in sys.argv else ["v6"]
    for m in methods:
        print(f"=== base method {m} ===")
        print(f"{'fix':28s} {'L-plate found/5':>15s} {'merged':>6s} {'extras':>6s}   "
              + "   ".join(f"GT26 +{d:<3d} w(fn,fp)" for d in density) + "     ms")
        for name, kw in variants().items():
            t0 = time.perf_counter()
            boxes = run(img, m, **kw)
            ms = (time.perf_counter() - t0) * 1000
            f, mg, ex = score_l(boxes, gts)
            cols = []
            for d in density:
                w, fn, fp = weighted_over_gt(lambda im: run(im, m, **kw), d,
                                             seeds=(1,) if d == 0 else (1, 2))
                cols.append(f"{w:6.1f} ({fn:4.1f},{fp:5.1f})")
            print(f"{name:28s} {f:>15d} {mg:>6d} {ex:>6d}   " + "   ".join(cols) + f"  {ms:5.0f}")
        for lim in (4.0, 8.0):
            t0 = time.perf_counter()
            boxes = run_ridge(img, m, limit=lim)
            ms = (time.perf_counter() - t0) * 1000
            f, mg, ex = score_l(boxes, gts)
            cols = []
            for d in density:
                w, fn, fp = weighted_over_gt(lambda im: run_ridge(im, m, limit=lim), d,
                                             seeds=(1,) if d == 0 else (1, 2))
                cols.append(f"{w:6.1f} ({fn:4.1f},{fp:5.1f})")
            print(f"{'ridge inpaint >' + str(lim):28s} {f:>15d} {mg:>6d} {ex:>6d}   "
                  + "   ".join(cols) + f"  {ms:5.0f}")
        print()


if __name__ == "__main__":
    main()
