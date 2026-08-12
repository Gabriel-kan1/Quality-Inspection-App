"""Objective same-area labels for the chassis assets, by geometric verification.

AlignmentGate's accuracy was previously quoted against labels produced by looking at
the images and deciding which showed the same area. That measures agreement with a
judgement call, not with the truth. This script replaces the judgement with a proof:
for every pair of assets it matches features, fits a homography with RANSAC, and
accepts the pair only if the homography is backed by enough inliers and maps the
first frame onto a convex, non-degenerate, plausibly-sized quad in the second.

    python python/pair_ground_truth.py                 # ORB + AKAZE + SIFT, gate at 0.80
    python python/pair_ground_truth.py --threshold 0.65
    python python/pair_ground_truth.py --detector SIFT --verbose

Needs opencv-python, numpy and onnxruntime, plus the gitignored assets in
app/src/main/assets/chassis_images/. Missing files are skipped.

WHAT THIS CAN AND CANNOT LABEL
------------------------------
A verified homography is strong evidence of "same area". A failure is only weak
evidence of "different area", because the hardware may simply have changed — which is
exactly the case the gate exists to handle. So this produces trustworthy POSITIVES
only, and negatives still need a human. Concretely, screws.jpg against screws_test.jpg
(the same four-hole cluster, swapped fasteners, one hole empty) yields zero symmetric
matches under all three detectors, so the pair the gate was built for is the one pair
this script cannot certify.

FINDINGS
--------
  1. The detector choice is not the problem. ORB, AKAZE and SIFT agree on which pairs
     verify (Jaccard 0.84-0.95), and SIFT — the most robust of the three — finds 0 raw
     matches on screws.jpg/screws_test.jpg. Feature matching is not merely badly tuned
     for this task; the correspondences do not exist once the hardware changes.
  2. What it does verify, it verifies decisively: 19 pairs across 6 clusters, with
     inlier counts from 21 to 3000.
  3. Those clusters never merge across shooting distance. stud1-4, screws_test/two,
     and screws/screws-1 are three separate components despite all showing the same
     four-hole cluster. Feature matching answers "near-duplicate photo", not "same
     area of the chassis" — which is the narrower question, and the reason the gate
     cannot be built on it.
  4. Scored against those 19 proven pairs, the encoder gate accepts 19/19 at a
     threshold of 0.65 and 15/19 at 0.80. The four it rejects at 0.80 include
     screws_test/screws_two, proven same-area by 435, 183 and 467 inliers.
"""
import argparse
import itertools
import os
import sys
from collections import defaultdict

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is required: pip install opencv-python")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "app", "src", "main", "assets", "chassis_images")
MODEL = os.path.join(ROOT, "app", "src", "main", "assets", "models", "feature_encoder.onnx")

# Verification is deliberately generous on detail and strict on geometry: this runs
# offline, so there is no reason to work at the gate's real-time resolution.
LONG_SIDE = 1200
RATIO = 0.75
MIN_INLIERS = 15
RANSAC_PX = 5.0
MAX_FEATURES = 3000

# FeatureEmbedder.preprocess() — kept in step with the Java so the similarities here
# are the ones the app computes.
INPUT_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def detectors(which):
    all_of = {
        "ORB": (lambda: cv2.ORB_create(MAX_FEATURES), cv2.NORM_HAMMING),
        "AKAZE": (lambda: cv2.AKAZE_create(), cv2.NORM_HAMMING),
        "SIFT": (lambda: cv2.SIFT_create(MAX_FEATURES), cv2.NORM_L2),
    }
    if which == "all":
        return all_of
    if which not in all_of:
        sys.exit(f"unknown detector {which!r}; choose from {', '.join(all_of)} or 'all'")
    return {which: all_of[which]}


def load_gray(name):
    img = cv2.imread(os.path.join(ASSETS, name), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    longest = max(img.shape[:2])
    if longest > LONG_SIDE:
        f = LONG_SIDE / longest
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    # These plates are low contrast; CLAHE materially raises keypoint counts on the
    # matte surface without inventing structure that RANSAC could latch onto.
    return cv2.createCLAHE(2.0, (8, 8)).apply(img)


def homography_is_sane(H, src_shape, dst_shape):
    """Reject homographies that are degenerate, mirrored, or absurdly distorted.

    RANSAC will happily return a transform fitted to 15 coincidental matches. Requiring
    the mapped frame to stay convex and plausibly sized is what turns "a homography was
    found" into "these images overlap".
    """
    if H is None:
        return False, "none"
    h, w = src_shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H).reshape(4, 2)
    if not np.all(np.isfinite(quad)):
        return False, "non-finite"

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - a[1]) - (a[1] - o[1]) * (b[0] - a[0])

    turns = [cross(quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]) for i in range(4)]
    if not (all(t > 0 for t in turns) or all(t < 0 for t in turns)):
        return False, "non-convex"

    area = 0.5 * abs((quad[2][0] - quad[0][0]) * (quad[3][1] - quad[1][1])
                     - (quad[2][1] - quad[0][1]) * (quad[3][0] - quad[1][0]))
    ratio = area / float(dst_shape[0] * dst_shape[1])
    if not (0.02 <= ratio <= 25.0):
        return False, f"area x{ratio:.2f}"

    sides = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    if min(sides) <= 1e-3 or max(sides) / min(sides) > 12.0:
        return False, "degenerate"
    return True, f"area x{ratio:.2f}"


def verify(desc_a, desc_b, kp_a, kp_b, norm, shape_a, shape_b):
    """Symmetric ratio test then RANSAC. Returns (raw matches, inliers, reason)."""
    if desc_a is None or desc_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return 0, 0, "no descriptors"
    bf = cv2.BFMatcher(norm, crossCheck=False)

    def ratio_filtered(d1, d2):
        keep = {}
        for pair in bf.knnMatch(d1, d2, k=2):
            if len(pair) < 2:
                continue
            best, second = pair
            if best.distance < RATIO * second.distance:
                keep[best.queryIdx] = best.trainIdx
        return keep

    forward = ratio_filtered(desc_a, desc_b)
    backward = ratio_filtered(desc_b, desc_a)
    # Mutual-best only. Chassis hardware is repetitive — many near-identical screw
    # heads — and one-to-many matches otherwise inflate the count with noise.
    pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    if len(pairs) < 4:
        return len(pairs), 0, "too few matches"

    src = np.float32([kp_a[i].pt for i, _ in pairs]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[j].pt for _, j in pairs]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if H is None:
        return len(pairs), 0, "no homography"
    inliers = int(mask.sum())
    ok, why = homography_is_sane(H, shape_a, shape_b)
    return len(pairs), inliers, why if ok else "rejected: " + why


def embed_all(names):
    """Cosine-normalised encoder embeddings, or None if onnxruntime/model is absent."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("! onnxruntime not installed — skipping the gate comparison\n")
        return None
    if not os.path.exists(MODEL):
        print(f"! {MODEL} not found — skipping the gate comparison\n")
        return None

    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    out = {}
    for name in names:
        img = cv2.imread(os.path.join(ASSETS, name), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = INPUT_SIZE / min(w, h)
        rw, rh = max(INPUT_SIZE, round(w * scale)), max(INPUT_SIZE, round(h * scale))
        resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
        ox, oy = (rw - INPUT_SIZE) // 2, (rh - INPUT_SIZE) // 2
        crop = resized[oy:oy + INPUT_SIZE, ox:ox + INPUT_SIZE]
        x = ((crop.astype(np.float32) / 255.0) - MEAN) / STD
        v = sess.run(None, {input_name: np.transpose(x, (2, 0, 1))[None, ...]})[0].reshape(-1)
        out[name] = v / np.linalg.norm(v)
    return out


def components(names, edges):
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key in edges:
        a, b = key.split("|")
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = defaultdict(list)
    for n in names:
        groups[find(n)].append(n)
    return sorted(groups.values(), key=len, reverse=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="AlignmentGate.MATCH_THRESHOLD to score against (default 0.80)")
    ap.add_argument("--detector", default="all", help="ORB, AKAZE, SIFT or all")
    ap.add_argument("--verbose", action="store_true", help="list every verified pair")
    args = ap.parse_args()

    if not os.path.isdir(ASSETS):
        sys.exit(f"assets not found at {ASSETS}")
    names = sorted(f for f in os.listdir(ASSETS) if f.lower().endswith((".jpg", ".png")))
    if len(names) < 2:
        sys.exit("need at least two images")
    print(f"{len(names)} images, {len(names)*(len(names)-1)//2} pairs\n")

    chosen = detectors(args.detector)
    verified = {}
    detail = {}
    for det_name, (factory, norm) in chosen.items():
        det = factory()
        feats = {}
        for n in names:
            img = load_gray(n)
            if img is None:
                continue
            kp, desc = det.detectAndCompute(img, None)
            feats[n] = (kp, desc, img.shape)
        counts = [len(feats[n][0]) for n in feats]
        accepted = set()
        for a, b in itertools.combinations(sorted(feats), 2):
            kp_a, desc_a, shape_a = feats[a]
            kp_b, desc_b, shape_b = feats[b]
            raw, inliers, why = verify(desc_a, desc_b, kp_a, kp_b, norm, shape_a, shape_b)
            key = f"{a}|{b}"
            detail.setdefault(key, {})[det_name] = (raw, inliers, why)
            if inliers >= MIN_INLIERS and not why.startswith(("rejected", "too few", "no ")):
                accepted.add(key)
        verified[det_name] = accepted
        print(f"  {det_name:6s} keypoints median {int(np.median(counts)):5d}   "
              f"verified {len(accepted):3d} pairs")

    union = set().union(*verified.values())
    consensus = set.intersection(*verified.values()) if len(verified) > 1 else union

    if len(verified) > 1:
        print("\n=== Detector agreement ===")
        for a, b in itertools.combinations(verified, 2):
            i, u = verified[a] & verified[b], verified[a] | verified[b]
            print(f"  {a:6s} vs {b:6s}  Jaccard {len(i)/max(1,len(u)):.2f}"
                  f"  ({len(i)} shared of {len(u)})")
        print(f"  all agree on {len(consensus)}; at least one accepts {len(union)}")

    print("\n=== Areas implied by the verified pairs ===")
    for group in components(names, consensus):
        if len(group) > 1:
            print(f"  [{len(group)}] " + ", ".join(group))
    alone = [g[0] for g in components(names, consensus) if len(g) == 1]
    print(f"  matched to nothing ({len(alone)}): " + ", ".join(alone))

    if args.verbose:
        print("\n=== Verified pairs ===")
        for key in sorted(union):
            a, b = key.split("|")
            cells = "  ".join(
                f"{d}:{detail[key][d][1]}" for d in verified if key in verified[d])
            print(f"  {a} + {b}   {cells}")

    vectors = embed_all(names)
    if vectors is None:
        return

    def sim(key):
        a, b = key.split("|")
        return float(vectors[a] @ vectors[b])

    print(f"\n=== Encoder gate at threshold {args.threshold:.2f}, "
          f"scored against proven pairs ===")
    rejected = []
    for key in sorted(union, key=sim, reverse=True):
        a, b = key.split("|")
        s = sim(key)
        if s < args.threshold:
            rejected.append((s, a, b))
    print(f"  accepts {len(union)-len(rejected)}/{len(union)} "
          f"= {100*(len(union)-len(rejected))/len(union):.1f}% objective recall")
    for s, a, b in rejected:
        print(f"    FALSE REJECT {s:.4f}  {a} + {b}")

    everything = [f"{a}|{b}" for a, b in itertools.combinations(names, 2)]
    unproven = [sim(k) for k in everything if k not in union]
    proven = [sim(k) for k in union]
    print(f"\n  proven same-area   min {min(proven):.3f}  median {np.median(proven):.3f}")
    print(f"  unproven           max {max(unproven):.3f}  median {np.median(unproven):.3f}")
    print("  (unproven pairs scoring high are not necessarily errors — see the module"
          " docstring)")

    print("\n=== Threshold sweep against proven pairs ===")
    for t in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        n = sum(1 for k in union if sim(k) >= t)
        flag = "  <== current" if abs(t - args.threshold) < 0.005 else ""
        print(f"  {t:.2f}  {n:2d}/{len(union)}  {'#' * n}{flag}")


if __name__ == "__main__":
    main()
