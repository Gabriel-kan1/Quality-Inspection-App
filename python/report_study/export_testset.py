"""Runs every Hybrid method over the 40 test photos and exports boxes + classifications.

The reference for each region is its "normal" photo under the method the app suggests (REF_METHOD:
v18, v8, v1), labelled at the FEATURES positions as an operator would; one library per region is
built from those crops, as the app builds one per part. Every inspection photo is then run
under all eighteen methods, unmasked (the final app has no chassis mask), and every box is
classified against that library by DINOv2-S (0.74 / 0.02) and by the old CLIP int8 encoder at
its own CLIP-era thresholds (0.85 / 0.03).

argv: <python dir> <test photo dir> <clip onnx> <out json>
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, sys.argv[1])
import variation_test_study as V  # noqa: E402
import detector_ab_study as D  # noqa: E402

FOLDER, CLIP_PATH, OUT = sys.argv[2], sys.argv[3], sys.argv[4]
V.masked = lambda img: img  # the final app runs every method on the photo as taken
METHODS = [f"v{i}" for i in range(1, 19)]
# The method FIND BEST METHOD picks on each reference: nearest the feature count, every
# feature boxed (ReferenceSetupFragment / AdaptivePreprocessing.closestToCount).
REF_METHOD = {"3 holes": "v1", "3 thread holes": "v8", "4 screws": "v18"}
THRESH = {"dino": (0.74, 0.02), "clip": (0.85, 0.03)}


class Clip:
    MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

    def __init__(self, path):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def embed(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = 224 / min(w, h)
        rw, rh = max(224, round(w * s)), max(224, round(h * s))
        r = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
        ox, oy = (rw - 224) // 2, (rh - 224) // 2
        x = ((r[oy:oy + 224, ox:ox + 224].astype(np.float32) / 255.0) - self.MEAN) / self.STD
        v = self.sess.run(None, {self.inp: np.transpose(x, (2, 0, 1))[None]})[0].reshape(-1)
        return v / np.linalg.norm(v)


enc = {"dino": V.Encoder(), "clip": Clip(CLIP_PATH)}


def classify(v, protos, sim, margin):
    scored = sorted(((float(v @ p), t) for t, p in protos.items()), reverse=True)
    best, t = scored[0]
    second = scored[1][0] if len(scored) > 1 else -1.0
    if best < sim or (len(scored) > 1 and best - second <= margin):
        typ = "unknown"
    else:
        typ = t
    return {"type": typ, "nearest": t, "score": round(best, 4), "second": round(second, 4)}


def kind_of(variant):
    w = variant.split()[0]
    return "defected" if w == "defected" else ("lighting" if w in V.LIGHTS else "angle")


out = {"regions": {}, "gate": {}}
t0 = time.time()
all_photos = {}
for cfg, feats in V.FEATURES.items():
    files = sorted(f for f in glob.glob(os.path.join(FOLDER, cfg + " *.jpg"))
                   if not (cfg == "3 holes" and "thread" in os.path.basename(f)))
    ref_file = os.path.join(FOLDER, cfg + " normal.jpg")
    ref = V.Photo(ref_file)
    H, W = ref.work.shape[:2]
    boxes = D.find_feature_boxes(V.METHODS[REF_METHOD[cfg]](ref.work))[0]
    labelled = []
    for t, fx, fy in feats:
        px, py = fx * W, fy * H
        hit = [b for b in boxes if b[0] - 6 <= px <= b[0] + b[2] + 6 and b[1] - 6 <= py <= b[1] + b[3] + 6]
        assert hit, f"reference misses {t} in {cfg}"
        labelled.append((t, min(hit, key=lambda b: b[2] * b[3])))
    protos = {}
    for e in enc:
        emb = {}
        for t, b in labelled:
            emb.setdefault(t, []).append(enc[e].embed(ref.crop(b)))
        protos[e] = {t: (lambda m: m / np.linalg.norm(m))(np.mean(v, axis=0)) for t, v in emb.items()}
    region = {"reference": {"method": REF_METHOD[cfg], "boxes": len(boxes),
                            "features": [{"type": t, "rect": [b[0] / W, b[1] / H, (b[0] + b[2]) / W, (b[1] + b[3]) / H]}
                                         for t, b in labelled]},
              "photos": {}}
    for f in files:
        name = os.path.basename(f)[:-4]
        all_photos[name] = f
        if f == ref_file:
            continue
        variant = name[len(cfg) + 1:]
        p = V.Photo(f)
        h, w = p.work.shape[:2]
        cache = {}
        per_method = {}
        for m in METHODS:
            src = p.work
            bx = D.find_feature_boxes(V.METHODS[m](src))[0]
            dets = []
            for b in bx:
                key = tuple(int(v) for v in b)
                if key not in cache:
                    crop = p.crop(b)
                    cache[key] = {e: classify(enc[e].embed(crop), protos[e], *THRESH[e]) for e in enc}
                dets.append({"rect": [b[0] / w, b[1] / h, (b[0] + b[2]) / w, (b[1] + b[3]) / h], **cache[key]})
            per_method[m] = dets
        region["photos"][variant] = {"kind": kind_of(variant), "methods": per_method}
        print(f"{cfg} / {variant}: {sum(len(v) for v in per_method.values())} boxes, {len(cache)} crops, {time.time() - t0:.0f}s", flush=True)
    out["regions"][cfg] = region

# Alignment gate's area test: whole-photo DINOv2 embedding, as the gate scores a frame.
names = sorted(all_photos)
embs = {n: enc["dino"].embed(cv2.imread(all_photos[n])) for n in names}
out["gate"] = {a: {b: round(float(embs[a] @ embs[b]), 4) for b in names} for a in names}
json.dump(out, open(OUT, "w"))
print("done", len(names), "photos", f"{time.time() - t0:.0f}s")
