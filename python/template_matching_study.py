"""Why AlignmentGate measures framing by template matching on gradients.

Reproduces the five experiments behind ml/TemplateMatcher and the framing half of
ml/AlignmentGate. Mirrors the Java exactly: same working width, same scale sweep, same
Sobel preparation, same TM_CCOEFF_NORMED.

    python python/template_matching_study.py            # all five
    python python/template_matching_study.py --part A   # one of A B C D E

Needs opencv-python and numpy; part E additionally needs onnxruntime and the encoder
asset. Also needs the chassis images, which are gitignored
(app/src/main/assets/chassis_images/). Files missing from disk are skipped.

  A  Representation. Which image representation survives a lighting change, measured
     with exact ground truth: a known sub-region cropped out of one photo of a scene,
     searched for in a DIFFERENT photo of the same scene under different light.
  B  Working width. The same question at five widths.
  C  Scale recovery. First a synthetic control isolating scale alone, then real pairs
     where the viewpoint also differs. The gap between the two is the method's limit.
  D  Score as an area test. Whether the correlation peak can tell the reference area
     from a different one. It cannot; this is why the encoder keeps that question.
  E  Reference quality. The area half is only as good as the reference photo. Ranks every
     asset by cosine similarity against one reference, so an inverted ranking - unrelated
     close-ups outscoring the reference's own subject - is visible.

Findings are recorded in TemplateMatcher's class javadoc and in
Report Structures/Findings_Alignment_Gate_Framing_Measurement.md.
"""
import argparse
import os
import sys

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency hint only
    sys.exit(f"missing dependency: {exc}. pip install opencv-python numpy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "app", "src", "main", "assets", "chassis_images")

# ── mirrors ml/TemplateMatcher ───────────────────────────────────
WORK_WIDTH = 320
SCALE_MIN, SCALE_MAX, SCALE_STEP = 0.30, 1.00, 0.05
MIN_TEMPLATE_EDGE = 16
SCALES = np.round(np.arange(SCALE_MIN, SCALE_MAX + 1e-9, SCALE_STEP), 2)

# Matched pairs: same scene, different light. The only ground truth in the asset set
# that isolates illumination from viewpoint.
LIGHTING_PAIRS = [("whole.jpg", "whole_light.jpg"), ("screws_flash.jpg", "screws_none.jpg")]

# (scale, centre x, centre y) of the region cropped out of the first image of a pair
CASES = [(0.40, 0.35, 0.40), (0.40, 0.65, 0.60), (0.55, 0.50, 0.50),
         (0.70, 0.45, 0.55), (0.85, 0.50, 0.50)]

OK_SCALE, OK_OFFSET = 0.10, 0.08

# Shooting distances, eyeballed off the assets. Ordinal, not metrology — a frame at
# distance d against a reference at d_ref covers d/d_ref of the reference's extent.
# Kept in step with python/scale_probe_study.py.
CATALOGUE = {
    "screws_test.jpg": ("fasteners", 0.8), "stud1.jpg": ("fasteners", 1.0),
    "stud2.jpg": ("fasteners", 1.0), "stud3.jpg": ("fasteners", 1.0),
    "stud4.jpg": ("fasteners", 1.0), "screws.jpg": ("fasteners", 1.2),
    "screws-1.jpg": ("fasteners", 1.2), "reference.jpg": ("fasteners", 1.6),
    "defected.jpg": ("fasteners", 1.6), "screws2.jpg": ("fasteners", 2.5),
    "screws_two.jpg": ("fasteners", 2.5), "screws_three.jpg": ("fasteners", 2.8),
    "screws_flash.jpg": ("fasteners", 2.8), "screws_none.jpg": ("fasteners", 4.0),
    "whole screws1.jpg": ("fasteners", 10.0), "whole screws2.jpg": ("fasteners", 10.0),
    "whole screws3.jpg": ("fasteners", 10.0),
    "head1.jpg": ("heads", 1.0), "head2.jpg": ("heads", 1.0),
    "head3.jpg": ("heads", 1.0), "head4.jpg": ("heads", 1.0),
    "hole1.jpg": ("bracket", 1.0), "hole2.jpg": ("bracket", 1.2),
    "corner.jpg": ("corner", 1.0), "zero.jpg": ("zero", 1.0),
    "one.jpg": ("one", 1.0), "three.jpg": ("three", 1.0),
    "whole.jpg": ("plate", 1.0), "whole_light.jpg": ("plate", 1.0),
}


def load(name):
    img = cv2.imread(os.path.join(IMG_DIR, name), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(name)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fit(img, width):
    return cv2.resize(img, (width, max(1, round(width * img.shape[0] / img.shape[1]))),
                      interpolation=cv2.INTER_AREA)


# ── representations ──────────────────────────────────────────────

def rep_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def rep_clahe(img):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(rep_gray(img))


def rep_sobel(img):
    """What ml/TemplateMatcher.prepare() produces."""
    g = cv2.GaussianBlur(rep_gray(img), (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def rep_canny(img):
    g = cv2.GaussianBlur(rep_gray(img), (5, 5), 0)
    v = np.median(g)
    return cv2.Canny(g, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))


REPS = [("raw gray", rep_gray), ("CLAHE", rep_clahe),
        ("Sobel mag", rep_sobel), ("Canny", rep_canny)]


# ── the search, mirroring TemplateMatcher.searchIn ───────────────

def search_in(scene, templ):
    """{peak, scale, centreX, centreY} of the best placement, or None."""
    best = None
    aspect = templ.shape[0] / templ.shape[1]
    for s in SCALES:
        tw = int(round(scene.shape[1] * s))
        th = int(round(tw * aspect))
        if (tw < MIN_TEMPLATE_EDGE or th < MIN_TEMPLATE_EDGE
                or tw > scene.shape[1] or th > scene.shape[0]):
            continue
        t = cv2.resize(templ, (tw, th), interpolation=cv2.INTER_AREA)
        r = cv2.matchTemplate(scene, t, cv2.TM_CCOEFF_NORMED)
        _, peak, _, loc = cv2.minMaxLoc(r)
        if best is None or peak > best[0]:
            best = (float(peak), float(s),
                    (loc[0] + tw / 2) / scene.shape[1], (loc[1] + th / 2) / scene.shape[0])
    return best


def evaluate(ref_prepared, frame_prepared):
    """What AlignmentGate sees: (score, framing scale, offsetX, offsetY)."""
    closer = search_in(ref_prepared, frame_prepared)
    wider = search_in(frame_prepared, ref_prepared)
    if closer is None and wider is None:
        return None
    if wider is None or (closer is not None and closer[0] >= wider[0]):
        return closer[0], closer[1], 0.5 - closer[2], 0.5 - closer[3]
    scale = 1.0 / wider[1] if wider[1] > 0 else float("nan")
    return wider[0], scale, wider[2] - 0.5, wider[3] - 0.5


def ground_truth_cases(a, b, width, rep):
    """Yields (truth_scale, truth_cx, truth_cy, scene, template) for one lighting pair."""
    A, B = fit(load(a), width), fit(load(b), width)
    for s, cx, cy in CASES:
        h, w = int(A.shape[0] * s), int(A.shape[1] * s)
        x0 = int(np.clip(cx * A.shape[1] - w / 2, 0, A.shape[1] - w))
        y0 = int(np.clip(cy * A.shape[0] - h / 2, 0, A.shape[0] - h))
        crop = A[y0:y0 + h, x0:x0 + w]
        yield ((s, (x0 + w / 2) / A.shape[1], (y0 + h / 2) / A.shape[0]),
               rep(B), rep(crop))


# ── A. representation ────────────────────────────────────────────

def part_a():
    print("\nA. Representation — recovering a known crop across a real lighting change")
    print("   (%d cases over %d matched pairs, at %d px)\n" % (
        len(CASES), len(LIGHTING_PAIRS), WORK_WIDTH))
    print("%-12s %-10s %-15s %-15s" % ("representation", "correct", "mean |dscale|", "mean |doffset|"))
    print("-" * 56)
    for name, fn in REPS:
        ok = n = 0
        ds = do = 0.0
        for a, b in LIGHTING_PAIRS:
            try:
                cases = list(ground_truth_cases(a, b, WORK_WIDTH, fn))
            except FileNotFoundError:
                continue
            for (ts, tx, ty), scene, templ in cases:
                got = search_in(scene, templ)
                n += 1
                if got is None:
                    continue
                e_s = abs(got[1] - ts)
                e_o = max(abs(got[2] - tx), abs(got[3] - ty))
                ds += e_s
                do += e_o
                ok += (e_s <= OK_SCALE and e_o <= OK_OFFSET)
        if n:
            print("%-12s %-10s %-15.3f %-15.3f"
                  % (name, "%d/%d" % (ok, n), ds / n, do / n))
    print("\n   correct = scale within %.2f and centre within %.0f%% of the frame."
          % (OK_SCALE, OK_OFFSET * 100))


# ── B. working width ─────────────────────────────────────────────

def part_b():
    print("\nB. Working width — the same cases, Sobel only\n")
    print("%-8s %-10s %-15s %-15s" % ("width", "correct", "mean |dscale|", "mean |doffset|"))
    print("-" * 50)
    for width in (192, 256, 320, 480, 640):
        ok = n = 0
        ds = do = 0.0
        for a, b in LIGHTING_PAIRS:
            try:
                cases = list(ground_truth_cases(a, b, width, rep_sobel))
            except FileNotFoundError:
                continue
            for (ts, tx, ty), scene, templ in cases:
                got = search_in(scene, templ)
                n += 1
                if got is None:
                    continue
                e_s = abs(got[1] - ts)
                e_o = max(abs(got[2] - tx), abs(got[3] - ty))
                ds += e_s
                do += e_o
                ok += (e_s <= OK_SCALE and e_o <= OK_OFFSET)
        if n:
            mark = "  <- shipped" if width == WORK_WIDTH else ""
            print("%-8d %-10s %-15.3f %-15.3f%s"
                  % (width, "%d/%d" % (ok, n), ds / n, do / n, mark))


# ── C. scale recovery, synthetic vs real ─────────────────────────

def part_c():
    print("\nC. Scale recovery — synthetic control, then real pairs\n")
    print("   Synthetic: the frame is a crop of the reference itself, so only scale differs.")
    print("   %-18s %-10s %s" % ("reference", "correct", "note"))
    print("   " + "-" * 58)
    for name in ("screws_flash.jpg", "screws.jpg"):
        try:
            ref = fit(load(name), WORK_WIDTH)
        except FileNotFoundError:
            continue
        prepared = rep_sobel(ref)
        ok = n = 0
        for f in (1.00, 0.85, 0.70, 0.55, 0.40):
            h, w = int(ref.shape[0] * f), int(ref.shape[1] * f)
            crop = ref[(ref.shape[0] - h) // 2:(ref.shape[0] - h) // 2 + h,
                       (ref.shape[1] - w) // 2:(ref.shape[1] - w) // 2 + w]
            got = search_in(prepared, rep_sobel(crop))
            n += 1
            ok += got is not None and abs(got[1] - f) <= OK_SCALE
        print("   %-18s %-10s %s" % (name, "%d/%d" % (ok, n), "lighting and viewpoint identical"))

    print("\n   Real: a different photograph, so viewpoint differs too.")
    print("   %-18s %-10s %s" % ("reference", "correct", "note"))
    print("   " + "-" * 58)
    for refname in ("screws_flash.jpg", "screws.jpg"):
        try:
            ref = rep_sobel(fit(load(refname), WORK_WIDTH))
        except FileNotFoundError:
            continue
        group, d_ref = CATALOGUE[refname]
        ok = n = 0
        for name, (g, d) in CATALOGUE.items():
            if name == refname or g != group:
                continue
            expect = d / d_ref
            try:
                frame = rep_sobel(fit(load(name), WORK_WIDTH))
            except FileNotFoundError:
                continue
            got = evaluate(ref, frame)
            if got is None:
                continue
            n += 1
            ok += abs(got[1] - expect) <= 0.15
        if n:
            print("   %-18s %-10s %s" % (refname, "%d/%d" % (ok, n),
                                         "within 0.15 of the distance ratio"))
    print("\n   The gap between the two blocks is the method's limit: NCC models translation")
    print("   and uniform scale, not camera angle. Distances are eyeballed, so the real")
    print("   block understates accuracy somewhat — treat it as a floor, not a point estimate.")


# ── D. the score is not an area test ─────────────────────────────

def part_d():
    print("\nD. Correlation peak as an area test\n")
    print("%-18s %-14s %-14s %-12s" % ("reference", "same-area min", "other-area max", "separation"))
    print("-" * 62)
    for refname in ("screws_flash.jpg", "screws.jpg"):
        try:
            ref = rep_sobel(fit(load(refname), WORK_WIDTH))
        except FileNotFoundError:
            continue
        group = CATALOGUE[refname][0]
        same, diff = [], []
        for name, (g, _) in CATALOGUE.items():
            if name == refname:
                continue
            try:
                frame = rep_sobel(fit(load(name), WORK_WIDTH))
            except FileNotFoundError:
                continue
            got = evaluate(ref, frame)
            if got is None:
                continue
            (same if g == group else diff).append(got[0])
        if same and diff:
            sep = min(same) - max(diff)
            print("%-18s %-14.3f %-14.3f %-12.3f %s"
                  % (refname, min(same), max(diff), sep,
                     "OVERLAP" if sep < 0 else ""))
    print("\n   A negative separation means the worst same-area score sits below the best")
    print("   different-area score, so no threshold on this value separates the two. The")
    print("   encoder keeps the area question for that reason.")


def part_e(refname="screws_flash.jpg"):
    """Ranks every asset against one reference by encoder cosine similarity."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import scale_probe_study as sps
    except Exception as exc:
        print("\nE. Reference quality - skipped (%s)" % exc)
        return

    print("\nE. Reference quality - cosine similarity against %s\n" % refname)
    try:
        ref = sps.embed(refname, 1.0)
    except Exception as exc:
        print("   skipped (%s)" % exc)
        return

    group = CATALOGUE.get(refname, ("?", None))[0]
    rows = []
    for name, (g, _) in CATALOGUE.items():
        if name == refname:
            continue
        try:
            v = sps.embed(name, 1.0)
        except Exception:
            continue
        rows.append((float(v @ ref), name, g == group))

    print("   %-18s %-7s %-10s" % ("asset", "sim", "area"))
    print("   " + "-" * 40)
    for sim, name, is_same in sorted(rows, reverse=True):
        flag = ""
        if not is_same and sim >= 0.70:
            flag = "  <- different area, clears 0.70"
        print("   %-18s %-7.3f %-10s%s"
              % (name, sim, "SAME" if is_same else "different", flag))

    same = [s for s, _, k in rows if k]
    diff = [s for s, _, k in rows if not k]
    if same and diff:
        print("\n   same-area range %.3f-%.3f, different-area range %.3f-%.3f"
              % (min(same), max(same), min(diff), max(diff)))
        print("   separation: %.3f" % (min(same) - max(diff)))
        print("\n   A wide reference whose subject occupies a few percent of the frame has its")
        print("   embedding dominated by featureless plate, so unrelated close-ups of the same")
        print("   material score as well as its own subject. No threshold repairs an inverted")
        print("   ranking - the reference photo has to change.")


PARTS = {"A": part_a, "B": part_b, "C": part_c, "D": part_d, "E": part_e}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=sorted(PARTS), help="run only one part")
    args = ap.parse_args()
    for key in ([args.part] if args.part else sorted(PARTS)):
        PARTS[key]()
    print()
