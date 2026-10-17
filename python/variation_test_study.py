"""
Does an edge method carry a reference through lighting changes and catch a defect?

Mirrors the app end to end for one fixed edge method at a time, on a folder of test photos
named "<config> <variant>.jpg":

  1. Reference setup. The method's boxes on "<config> normal.jpg" are labelled the way an
     operator would: a box over a known feature (FEATURES below) gets its type, anything else
     is left unnamed and so excluded. The part's library is built from those crops alone, as
     ReferenceSetupFragment does (a new part's library starts empty).
  2. Inspection. Each other photo is run through the same method; every box is cropped from
     the full-resolution photo with the app's padding, embedded with the shipped DINOv2-S
     encoder, and classified against that library with the app's default thresholds (0.74
     similarity, 0.02 margin). Unknowns are skipped, as the verdict does.
  3. Verdict. PASS when every type's count matches the reference's, FAIL otherwise — the
     count half of ResultsFragment.verdictFor. The positional check is not modelled.

A lighting variant should PASS and "defected" should FAIL. Photos named only for an angle
(left, right, bot, front) are reported apart: framing was not what this set was for.

All eighteen Edge Lab methods are here. Hybrid v1 to v10 run on the chassis-masked photo
unless --no-mask is given; v11 to v18 always run on the photo as taken, as the app runs them.
v11 to v18 are ports of EdgeMethodLab's Java (claheBilateralCanny, dropOversize, grayWorld,
gaussianFlat, medianFlat, roundRegions, houghDiscs), so a result here should match the
Edge Lab card up to OpenCV version differences.

    python python/variation_test_study.py "D:/path/to/Test photos"
    python python/variation_test_study.py "D:/path/to/Test photos" --methods v6 v11
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detector_ab_study as D  # noqa: E402
import hybrid_v2_study as HV  # noqa: E402
import adaptive_preprocessing_study as A  # noqa: E402
from small_feature_rung_study import masked  # noqa: E402

MODEL = Path(__file__).resolve().parent.parent / "app/src/main/assets/models/feature_encoder.onnx"
SIMILARITY_THRESHOLD = 0.74
MARGIN_THRESHOLD = 0.02
CROP_PADDING_RATIO = 0.15
MIN_CROP_PADDING = 2

# Where each configuration's hardware sits in its "normal" photo, as fractions (x, y), by
# type. Read off the photographs; the operator labels these and leaves anything else unnamed.
FEATURES = {
    "3 holes": [("threaded_hole", 0.19, 0.18), ("bolt_head", 0.22, 0.41),
                ("bolt_head", 0.22, 0.91)],
    "3 thread holes": [("threaded_hole", 0.56, 0.17), ("threaded_hole", 0.16, 0.44),
                       ("threaded_hole", 0.16, 0.86)],
    "4 screws": [("screw", 0.47, 0.42), ("screw", 0.38, 0.65),
                 ("stud", 0.67, 0.38), ("stud", 0.58, 0.60)],
}
LIGHTS = ("white", "warm", "yellow")


def canny_otsu(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    hi, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return cv2.Canny(g, hi * 0.5, hi)


def clahe_bilateral_canny(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.createCLAHE(1.0, (16, 16)).apply(g)
    f = cv2.bilateralFilter(eq, 5, 60, 20)
    hi, _ = cv2.threshold(f, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return cv2.Canny(f, hi * 0.5, hi)


# ── Hybrid v11 to v18's building blocks, mirroring EdgeMethodLab's ────────────────────────
OVERSIZE_FRACTION = 0.15
V3 = HV.cfg(**HV.WINNERS["v3 gauss7 otsu1.2"])
V4 = HV.cfg(**HV.WINNERS["v4 L2 light-morph"])


def gray_of(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def bgr_of(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def clahe_bilateral_canny_k(g, otsu_scale, gauss_k=0, dilate_k=0, close_k=0):
    """EdgeMethodLab.claheBilateralCanny: Hybrid v9's chain with its knobs."""
    f = cv2.bilateralFilter(cv2.createCLAHE(1.0, (16, 16)).apply(g), 5, 60, 20)
    if gauss_k > 0:
        f = cv2.GaussianBlur(f, (gauss_k, gauss_k), 0)
    hi, _ = cv2.threshold(f, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    hi *= otsu_scale
    out = cv2.Canny(f, hi * 0.5, hi)
    if dilate_k > 0:
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k)))
    if close_k > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    return out


def drop_oversize(m, fraction=OVERSIZE_FRACTION):
    """EdgeMethodLab.dropOversize: zero every 8-connected piece wider or taller than
    `fraction` of the map's longest side."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    limit = fraction * max(m.shape[:2])
    drop = (stats[:, cv2.CC_STAT_WIDTH] > limit) | (stats[:, cv2.CC_STAT_HEIGHT] > limit)
    drop[0] = False
    if drop.any():
        m = m.copy()
        m[drop[labels]] = 0
    return m


def gray_world(img):
    """EdgeMethodLab.grayWorld: each colour channel scaled so the three average the same."""
    mean = img.reshape(-1, 3).mean(axis=0)
    grey = mean.mean()
    out = img.astype(np.float32) * (grey / np.maximum(mean, 1.0))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def divide_by(g, bg):
    """128 * gray / background, saturated to 8 bits."""
    den = np.maximum(bg.astype(np.float32), 1.0)
    return np.clip(np.rint(128.0 * g.astype(np.float32) / den), 0, 255).astype(np.uint8)


def gaussian_flat(g, sigma_fraction):
    bg = cv2.GaussianBlur(g.astype(np.float32), (0, 0), max(g.shape[:2]) * sigma_fraction)
    return divide_by(g, bg)


def median_flat(g, window_fraction):
    s = 4
    small = cv2.resize(g, (g.shape[1] // s, g.shape[0] // s), interpolation=cv2.INTER_AREA)
    k = int(max(g.shape[:2]) * window_fraction / s) | 1
    small = cv2.medianBlur(small, max(3, k))
    bg = cv2.resize(small, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    return divide_by(g, bg)


def round_regions(g):
    """EdgeMethodLab.roundRegions: roundish, fastener-sized MSER regions, filled as hulls."""
    flat = cv2.GaussianBlur(gaussian_flat(g, 0.03), (5, 5), 0)
    area = g.shape[0] * g.shape[1]
    mser = cv2.MSER_create(5, int(area * 0.00015), int(area * 0.012))
    regions, _ = mser.detectRegions(flat)
    keep = []
    for r in regions:
        hull = cv2.convexHull(r.reshape(-1, 1, 2))
        a, per = cv2.contourArea(hull), cv2.arcLength(hull, True)
        if per > 0 and 4 * np.pi * a / (per * per) >= 0.5:
            keep.append(hull)
    out = np.zeros(g.shape[:2], np.uint8)
    if keep:
        cv2.drawContours(out, keep, -1, 255, -1)
    return out


def hough_discs(g):
    """EdgeMethodLab.houghDiscs: Hough circles on the flattened photo, each drawn filled."""
    flat = cv2.medianBlur(gaussian_flat(g, 0.03), 5)
    md = min(g.shape[:2])
    circles = cv2.HoughCircles(flat, cv2.HOUGH_GRADIENT, 1.5, md * 0.06, param1=120,
                               param2=40, minRadius=int(md * 0.012), maxRadius=int(md * 0.08))
    out = np.zeros(g.shape[:2], np.uint8)
    if circles is not None:
        for x, y, r in circles[0]:
            cv2.circle(out, (int(round(x)), int(round(y))), int(round(r)), 255, -1)
    return out


# The app's eighteen, under their Hybrid names. Each takes a working image and returns a map.
# v1 to v10 get the masked image (unless --no-mask); v11 to v18 always run on the photo as
# taken, as the app runs them (ChassisMask.unmaskedOr).
METHODS = {
    "v1": lambda im: HV.edges_for(im, HV.cfg()),
    "v2": lambda im: HV.edges_for(im, HV.cfg(**HV.WINNERS["v2 bilat9 low0.4"])),
    "v3": lambda im: HV.edges_for(im, HV.cfg(**HV.WINNERS["v3 gauss7 otsu1.2"])),
    "v4": lambda im: HV.edges_for(im, HV.cfg(**HV.WINNERS["v4 L2 light-morph"])),
    "v5": lambda im: HV.edges_for(im, HV.cfg(**HV.WINNERS["v5 v3 + adaptive mask"])),
    "v6": lambda im: HV.edges_for(im, HV.cfg(**A.LADDER["medium"])),
    "v7": lambda im: HV.edges_for(im, HV.cfg(**A.LADDER["light"])),
    "v8": lambda im: HV.edges_for(im, HV.cfg(**A.LADDER["strong"])),
    "v9": clahe_bilateral_canny,
    "v10": canny_otsu,
    "v11": lambda im: drop_oversize(clahe_bilateral_canny_k(gray_of(im), 1.3)),
    "v12": lambda im: drop_oversize(HV.edges_for(im, V4)),
    "v13": lambda im: drop_oversize(HV.edges_for(gray_world(im), V4)),
    "v14": lambda im: clahe_bilateral_canny_k(gray_of(im), 1.0, gauss_k=5, dilate_k=3, close_k=5),
    "v15": lambda im: HV.edges_for(bgr_of(median_flat(gray_of(im), 0.06)), V3),
    "v16": lambda im: drop_oversize(HV.edges_for(
        bgr_of(gaussian_flat(gray_of(gray_world(im)), 0.03)), V4)),
    "v17": lambda im: round_regions(gray_of(im)),
    "v18": lambda im: cv2.bitwise_or(drop_oversize(HV.edges_for(im, V4)), hough_discs(gray_of(im))),
}
UNMASKED = {f"v{i}" for i in range(11, 19)}


class Encoder:
    """The shipped encoder, fed exactly as FeatureEmbedder feeds it."""
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self):
        self.sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def embed(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = 224 / min(w, h)
        rw, rh = max(224, round(w * s)), max(224, round(h * s))
        r = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
        ox, oy = (rw - 224) // 2, (rh - 224) // 2
        x = ((r[oy:oy + 224, ox:ox + 224].astype(np.float32) / 255.0) - self.MEAN) / self.STD
        v = self.sess.run(None, {self.inp: np.transpose(x, (2, 0, 1))[None, ...]})[0].reshape(-1)
        return v / np.linalg.norm(v)


class Photo:
    def __init__(self, path):
        self.name = os.path.basename(path)
        self.full = cv2.imread(path, cv2.IMREAD_COLOR)
        self.work = D.load_working_image(Path(path))
        self.masked = masked(self.work)
        self.scale = self.full.shape[1] / self.work.shape[1]

    def crop(self, box):
        """The app's crop: padded by 15% of the box's longest side, cut from the source."""
        x, y, w, h = box
        pad = max(MIN_CROP_PADDING, round(max(w, h) * CROP_PADDING_RATIO))
        s = self.scale
        x0, y0 = max(0, round((x - pad) * s)), max(0, round((y - pad) * s))
        x1 = min(self.full.shape[1], round((x + w + pad) * s))
        y1 = min(self.full.shape[0], round((y + h + pad) * s))
        return self.full[y0:y1, x0:x1]


def classify(v, protos):
    if not protos:
        return "unknown", 0.0
    scored = sorted(((float(v @ p), t) for t, p in protos.items()), reverse=True)
    best, t = scored[0]
    if best < SIMILARITY_THRESHOLD:
        return "unknown", best
    if len(scored) > 1 and best - scored[1][0] <= MARGIN_THRESHOLD:
        return "unknown", best
    return t, best


def counts(labels):
    out = {}
    for t in labels:
        if t != "unknown":
            out[t] = out.get(t, 0) + 1
    return out


def run(folder, method_ids, encoder):
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    photos = {}

    def photo(f):
        if f not in photos:
            photos[f] = Photo(f)
        return photos[f]

    results = {}
    for config, feats in FEATURES.items():
        group = [f for f in files if os.path.basename(f).startswith(config + " ")
                 and not (config == "3 holes" and os.path.basename(f).startswith("3 holes "
                          ) is False)]
        group = [f for f in group if os.path.basename(f)[len(config) + 1:-4].split()[0]
                 in ("normal", "defected") + LIGHTS + ("left", "right", "bot", "front")]
        ref_file = os.path.join(folder, config + " normal.jpg")
        ref = photo(ref_file)
        for mid in method_ids:
            fn = METHODS[mid]
            source = (lambda ph: ph.work) if mid in UNMASKED else (lambda ph: ph.masked)
            boxes = D.find_feature_boxes(fn(source(ref)))[0]
            H, W = ref.work.shape[:2]
            # Operator labelling: each known feature claims the box it falls in.
            labelled, missing = [], []
            for t, fx, fy in feats:
                px, py = fx * W, fy * H
                hit = [b for b in boxes if b[0] - 6 <= px <= b[0] + b[2] + 6
                       and b[1] - 6 <= py <= b[1] + b[3] + 6]
                if hit:
                    labelled.append((t, min(hit, key=lambda b: b[2] * b[3])))
                else:
                    missing.append(t)
            emb = {}
            for t, b in labelled:
                emb.setdefault(t, []).append(encoder.embed(ref.crop(b)))
            protos = {t: (lambda m: m / np.linalg.norm(m))(np.mean(v, axis=0))
                      for t, v in emb.items()}
            expected = {}
            for t, _, _ in feats:
                expected[t] = expected.get(t, 0) + 1
            ref_ok = not missing
            ref_counts = counts([classify(encoder.embed(ref.crop(b)), protos)[0] for b in boxes])
            row = {"ref_boxes": len(boxes), "ref_missing": missing, "ref_counts": ref_counts,
                   "photos": {}}
            for f in group:
                if f == ref_file:
                    continue
                p = photo(f)
                bx = D.find_feature_boxes(fn(source(p)))[0]
                labels = [classify(encoder.embed(p.crop(b)), protos) for b in bx]
                c = counts([l for l, _ in labels])
                verdict = "PASS" if ref_ok and c == ref_counts else "FAIL"
                row["photos"][os.path.basename(f)[len(config) + 1:-4]] = (verdict, len(bx), c)
            row["ref_ok"] = ref_ok and ref_counts == expected
            results[(config, mid)] = row
    return results


def report(results, method_ids):
    for config in FEATURES:
        print(f"\n=== {config} ===")
        for mid in method_ids:
            r = results[(config, mid)]
            ph = r["photos"]
            light = {k: v for k, v in ph.items() if k.split()[0] in LIGHTS}
            passed = sum(1 for v in light.values() if v[0] == "PASS")
            defect = ph.get("defected", ("-",))[0]
            angles = {k: v for k, v in ph.items() if k.split()[0] in ("left", "right", "bot",
                                                                     "front")}
            ang = sum(1 for v in angles.values() if v[0] == "PASS")
            ref = "ref OK " if r["ref_ok"] else f"ref BAD {r['ref_counts']} missing {r['ref_missing']}"
            caught = "caught" if defect == "FAIL" else "MISSED"
            print(f"  {mid:5s} {ref:45s} lighting {passed:2d}/{len(light)} pass   "
                  f"defect {caught}   angles {ang}/{len(angles)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--no-mask", action="store_true",
                    help="skip the chassis mask: on these photos it paints out shadowed hardware")
    args = ap.parse_args()
    if args.no_mask:
        global masked
        masked = lambda img: img  # noqa: E731
    enc = Encoder()
    res = run(args.folder, args.methods, enc)
    report(res, args.methods)
    if args.detail:
        for (config, mid), r in res.items():
            print(f"\n{config} / {mid}: reference {r['ref_boxes']} boxes -> {r['ref_counts']}")
            for k, (v, n, c) in sorted(r["photos"].items()):
                print(f"   {k:14s} {v}  {n:3d} boxes  {c}")


if __name__ == "__main__":
    main()
