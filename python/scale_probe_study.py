"""Why AlignmentGate gates on shooting distance, and how that distance is measured.

The gate's cosine similarity is partly scale tolerant: the right area photographed
from 2.5x the reference distance still clears the threshold. This script measures
three candidate distance cues over the bundled chassis assets and reports which one
separates "correctly framed" from "too far / too close".

    python python/scale_probe_study.py

Needs onnxruntime, opencv-python and numpy, plus the assets that are gitignored
(app/src/main/assets/chassis_images/). Pairs missing from disk are skipped.

Findings, reproduced by --all:

  1. Median detected-blob diameter is not a distance cue here. On centre crops of one
     image, where the true scale ratio is known exactly, it wanders by 3x: the
     contours that survive filtering are surface grain, not hardware.
  2. Texture spatial frequency is not one either. It flattens at close range where
     sensor noise dominates the spectrum, and moves the wrong way on corner.jpg.
  3. Re-scoring the frame's centre crop against the reference IS one. Correctly framed
     frames lose 0.070 on average by cropping in; too-far frames gain 0.156. Mirroring
     it (frame against a cropped reference) catches too-close the same way.

The gate ships (3) as AlignmentGate.SCALE_GAIN_TOLERANCE.
"""
import argparse
import os
import sys

import numpy as np

try:
    import cv2
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - dependency hint only
    sys.exit(f"missing dependency: {exc}. pip install onnxruntime opencv-python numpy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "app", "src", "main", "assets")
IMG_DIR = os.path.join(ASSETS, "chassis_images")
MODEL = os.path.join(ASSETS, "models", "feature_encoder.onnx")

# Mirrors FeatureEmbedder: 224x224, shorter side resized, centre cropped, ImageNet norm.
INPUT = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

PROBE = 0.5           # AlignmentGate.PROBE_CROP
TOLERANCE = 0.10      # AlignmentGate.SCALE_GAIN_TOLERANCE
THRESHOLD = 0.67      # AlignmentGate.MATCH_THRESHOLD

# area group, relative shooting distance (1.0 = the framing a close-up reference uses).
# Distances are read off the assets by eye; they are ordinal, not metrology, which is
# why the tolerance below is deliberately loose.
CATALOGUE = {
    "screws_test.jpg":   ("fasteners", 0.8),
    "stud1.jpg":         ("fasteners", 1.0),
    "stud2.jpg":         ("fasteners", 1.0),
    "stud3.jpg":         ("fasteners", 1.0),
    "stud4.jpg":         ("fasteners", 1.0),
    "screws.jpg":        ("fasteners", 1.2),
    "screws-1.jpg":      ("fasteners", 1.2),
    "reference.jpg":     ("fasteners", 1.6),
    "defected.jpg":      ("fasteners", 1.6),
    "screws2.jpg":       ("fasteners", 2.5),
    "screws_two.jpg":    ("fasteners", 2.5),
    "screws_three.jpg":  ("fasteners", 2.8),
    "screws_flash.jpg":  ("fasteners", 2.8),
    "screws_none.jpg":   ("fasteners", 4.0),
    "whole screws1.jpg": ("fasteners", 10.0),
    "whole screws2.jpg": ("fasteners", 10.0),
    "whole screws3.jpg": ("fasteners", 10.0),
    "head1.jpg":         ("heads", 1.0),
    "head2.jpg":         ("heads", 1.0),
    "head3.jpg":         ("heads", 1.0),
    "head4.jpg":         ("heads", 1.0),
    "hole1.jpg":         ("bracket", 1.0),
    "hole2.jpg":         ("bracket", 1.2),
    "corner.jpg":        ("corner", 1.0),
    "zero.jpg":          ("zero", 1.0),
    "one.jpg":           ("one", 1.0),
    "three.jpg":         ("three", 1.0),
    "whole.jpg":         ("plate", 1.0),
    "whole_light.jpg":   ("plate", 1.0),
}

# References an operator would plausibly save: sharp, well framed, various distances.
REFERENCES = ["stud1.jpg", "stud3.jpg", "screws.jpg", "screws_test.jpg", "head1.jpg",
              "hole1.jpg", "reference.jpg", "screws_two.jpg", "whole screws1.jpg"]

DISTANCE_TOLERANCE = 1.6  # within this ratio, two shots count as the same framing
ACCEPT, FAR, CLOSE, OTHER = "ACCEPT", "TOO_FAR", "TOO_CLOSE", "OTHER_AREA"


def label(ref, frame):
    ref_area, ref_distance = CATALOGUE[ref]
    frame_area, frame_distance = CATALOGUE[frame]
    if ref_area != frame_area:
        return OTHER
    ratio = frame_distance / ref_distance
    if ratio > DISTANCE_TOLERANCE:
        return FAR
    if ratio < 1 / DISTANCE_TOLERANCE:
        return CLOSE
    return ACCEPT


# ── encoder ──────────────────────────────────────────────────────

_session = None
_embeddings = {}


def session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    return _session


def load(name):
    image = cv2.imread(os.path.join(IMG_DIR, name), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(name)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def centre_crop(image, fraction):
    """Keep `fraction` of each side. fraction < 1 simulates stepping closer."""
    if fraction >= 1.0:
        return image
    h, w = image.shape[:2]
    ch, cw = int(round(h * fraction)), int(round(w * fraction))
    return image[(h - ch) // 2:(h - ch) // 2 + ch, (w - cw) // 2:(w - cw) // 2 + cw]


def preprocess(image):
    h, w = image.shape[:2]
    scale = INPUT / min(w, h)
    rw, rh = max(INPUT, round(w * scale)), max(INPUT, round(h * scale))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    ox, oy = (rw - INPUT) // 2, (rh - INPUT) // 2
    crop = resized[oy:oy + INPUT, ox:ox + INPUT].astype(np.float32) / 255.0
    return np.transpose((crop - MEAN) / STD, (2, 0, 1))[None]


def embed(name, fraction=1.0):
    key = (name, round(fraction, 3))
    if key not in _embeddings:
        raw = session().run(None, {session().get_inputs()[0].name:
                                   preprocess(centre_crop(load(name), fraction))})[0][0]
        _embeddings[key] = raw / (np.linalg.norm(raw) + 1e-12)
    return _embeddings[key]


# ── the two cues that did not work ───────────────────────────────

def _framed_gray(image, size):
    h, w = image.shape[:2]
    scale = size / min(w, h)
    rw, rh = max(size, round(w * scale)), max(size, round(h * scale))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    ox, oy = (rw - size) // 2, (rh - size) // 2
    return cv2.cvtColor(resized[oy:oy + size, ox:ox + size], cv2.COLOR_RGB2GRAY)


def blob_scale(image, size=448):
    """Median blob diameter as a fraction of frame width — cue 1."""
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(_framed_gray(image, size))
    gray = cv2.bilateralFilter(gray, 7, 60, 20)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.dilate(cv2.Canny(gray, otsu * 0.5, otsu), np.ones((3, 3), np.uint8))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diameters = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 12:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if not w or not h or not 0.4 < w / h < 2.5 or max(w, h) > 0.5 * size:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0 or 4 * np.pi * area / perimeter ** 2 < 0.45:
            continue
        diameters.append(2 * np.sqrt(area / np.pi))
    return float(np.median(diameters)) / size if diameters else float("nan")


def spectral_scale(image, size=INPUT):
    """Spectral centroid of the radially averaged power spectrum — cue 2."""
    gray = _framed_gray(image, size).astype(np.float32) / 255.0
    windowed = (gray - gray.mean()) * np.outer(np.hanning(size), np.hanning(size))
    power = np.fft.fftshift(np.abs(np.fft.fft2(windowed)) ** 2)
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2).astype(int)
    bins = size // 2
    inside = radius < bins
    profile = (np.bincount(radius[inside], power[inside], bins)
               / np.maximum(np.bincount(radius[inside], None, bins), 1))
    k = np.arange(bins)[3:]           # drop DC and the illumination gradient
    weights = profile[3:]
    return float((k * weights).sum() / (weights.sum() + 1e-12)) / size


def report_failed_cues():
    ladder = [1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.22]
    print("\nCUES 1 AND 2 — measured on centre crops of one image, where the true scale")
    print("ratio is the crop fraction itself. A working cue holds its ratio column at 1.00.")
    for name in ["whole screws1.jpg", "screws_none.jpg", "stud1.jpg", "corner.jpg"]:
        if not os.path.exists(os.path.join(IMG_DIR, name)):
            continue
        image = load(name)
        print(f"\n  {name}")
        print(f"    {'crop':>6} {'blob d':>8} {'ratio':>8} {'spectral':>10} {'ratio':>8}")
        base_blob = base_spectral = None
        for fraction in ladder:
            cropped = centre_crop(image, fraction)
            blob, spectral = blob_scale(cropped), spectral_scale(cropped)
            if base_blob is None:
                base_blob, base_spectral = blob, spectral
            print(f"    {fraction:6.2f} {blob:8.4f} {blob / base_blob * fraction:8.2f} "
                  f"{spectral:10.5f} {spectral / base_spectral / fraction:8.2f}")


# ── the cue that worked ──────────────────────────────────────────

def measure():
    available = [n for n in CATALOGUE
                 if os.path.exists(os.path.join(IMG_DIR, n))]
    missing = sorted(set(CATALOGUE) - set(available))
    if missing:
        print(f"skipping {len(missing)} asset(s) not on disk: {', '.join(missing)}")
    rows = []
    for ref in [r for r in REFERENCES if r in available]:
        ref_full, ref_zoom = embed(ref, 1.0), embed(ref, PROBE)
        for frame in available:
            if frame == ref:
                continue
            frame_full = embed(frame, 1.0)
            base = float(np.dot(ref_full, frame_full))
            rows.append(dict(
                ref=ref, frame=frame, label=label(ref, frame), base=base,
                far=float(np.dot(ref_full, embed(frame, PROBE))) - base,
                close=float(np.dot(ref_zoom, frame_full)) - base))
    return rows


def gate(row, threshold=THRESHOLD, tolerance=TOLERANCE, probe=True):
    if row["base"] < threshold:
        return False
    return not probe or (row["far"] <= tolerance and row["close"] <= tolerance)


def report_probe(rows):
    print("\nCUE 3 — how the probe gains distribute, by label:")
    for key in ("far", "close"):
        print(f"  {'crop the frame' if key == 'far' else 'crop the reference':<22}"
              f"{'n':>5}{'mean':>9}{'p10':>9}{'p90':>9}")
        for name in (ACCEPT, FAR, CLOSE, OTHER):
            values = [r[key] for r in rows if r["label"] == name]
            if values:
                print(f"    {name:<20}{len(values):5d}{np.mean(values):+9.3f}"
                      f"{np.percentile(values, 10):+9.3f}{np.percentile(values, 90):+9.3f}")

    print("\nFalse accepts / false rejects, similarity alone vs similarity + probe:")
    print(f"  {'threshold':>10}{'sim only':>14}{'+ probe':>12}")
    for threshold in np.arange(0.60, 0.81, 0.02):
        plain = tally(rows, threshold, probe=False)
        probed = tally(rows, threshold, probe=True)
        print(f"  {threshold:10.2f}{plain[0]:8d} /{plain[1]:3d}{probed[0]:8d} /{probed[1]:3d}")

    print(f"\nRemaining false accepts at the shipped setting "
          f"(sim >= {THRESHOLD}, gain <= {TOLERANCE}):")
    for row in sorted((r for r in rows if gate(r) and r["label"] != ACCEPT),
                      key=lambda r: -r["base"]):
        print(f"    {row['ref']:<18} vs {row['frame']:<18} [{row['label']}] "
              f"base {row['base']:.3f} far {row['far']:+.3f} close {row['close']:+.3f}")
    rejected = [r for r in rows if not gate(r) and r["label"] == ACCEPT]
    print(f"  Correctly framed frames wrongly rejected: {len(rejected)}")
    for row in rejected:
        print(f"    {row['ref']:<18} vs {row['frame']:<18} "
              f"base {row['base']:.3f} far {row['far']:+.3f} close {row['close']:+.3f}")


def tally(rows, threshold, probe):
    accepts = sum(1 for r in rows if gate(r, threshold, probe=probe) and r["label"] != ACCEPT)
    rejects = sum(1 for r in rows if not gate(r, threshold, probe=probe) and r["label"] == ACCEPT)
    return accepts, rejects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="also re-measure the two cues that did not work")
    args = parser.parse_args()

    if not os.path.exists(MODEL):
        sys.exit(f"encoder not found at {MODEL}")

    rows = measure()
    print(f"\n{len(rows)} labelled reference/frame pairs.")
    report_probe(rows)
    if args.all:
        report_failed_cues()


if __name__ == "__main__":
    main()
