"""
Encoder comparison for the few-shot reference library.

Answers one question: which image encoder spreads this project's five feature
types furthest apart in embedding space? Everything downstream of the encoder
(prototype averaging, cosine scoring, thresholds) is identical across encoders,
so the encoder is the only variable that matters for classification accuracy.

The scoring math here mirrors com.example.ml.ReferenceLibrary exactly:
  - each example embedding is L2-normalised before averaging (addExample)
  - prototype(type) = mean of that type's normalised examples
  - classify() = cosine(query, prototype) per type, then
    accept only if best >= SIMILARITY_THRESHOLD and (best - second) > MARGIN

Encoders compared:
  clip_int8   the ONNX file the app shipped before the DINOv2 swap
  clip_fp32   openai/clip-vit-base-patch32 via transformers -- CLIP's ceiling,
              isolates how much the int8 export costs
  dinov2_s    facebook/dinov2-small, CLS token (torch fp32 reference)
  dinov2_s_cp facebook/dinov2-small, CLS + mean-patch concat (often stronger
              on fine-grained texture than CLS alone)
  dino_onnx_* the exported ONNX graphs from export_encoder_onnx.py, fp32 and
              int8. These are what actually ships, so they are what matters --
              int8 cost CLIP ~9 points, so it is verified, not assumed.
  mnv3_s      torchvision MobileNetV3-Small ImageNet features -- the cheap
              baseline. If it scores near CLIP, CLIP is not earning its size.

Usage:  python python/compare_encoders.py
"""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import silhouette_score

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
CROPS_DIR = REPO / "app/src/main/assets/reference_crops"
CLIP_ONNX = REPO / "app/src/main/assets/models/clip_image_encoder.onnx"

# ReferenceLibrary.FEATURE_TYPES -- the app ignores any other asset subfolder
# (ReferenceLibraryProvider.bootstrapFromAssets iterates this list), so the
# stray reference_crops/unknown/ folder is excluded here too.
FEATURE_TYPES = ["bolt_head", "hole", "rivet_head", "screw", "stud"]

# ReferenceLibrary.DEFAULT_* -- the thresholds currently shipping.
SIMILARITY_THRESHOLD = 0.85
MARGIN_THRESHOLD = 0.03

INPUT_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────
# PREPROCESSING (mirrors ClipEmbedder.preprocess)
# ─────────────────────────────────────────────
def preprocess(path, mean, std):
    """Resize shorter side -> 224, centre-crop 224x224, normalise, pack NCHW."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = INPUT_SIZE / min(w, h)
    rw = max(INPUT_SIZE, round(w * scale))
    rh = max(INPUT_SIZE, round(h * scale))
    img = img.resize((rw, rh), Image.BILINEAR)
    off_x = (rw - INPUT_SIZE) // 2
    off_y = (rh - INPUT_SIZE) // 2
    img = img.crop((off_x, off_y, off_x + INPUT_SIZE, off_y + INPUT_SIZE))

    arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC
    arr = (arr - np.array(mean, np.float32)) / np.array(std, np.float32)
    return np.transpose(arr, (2, 0, 1))[None, ...].copy()     # 1CHW


# ─────────────────────────────────────────────
# ENCODERS -- each returns (name, embed_fn, param_millions)
# ─────────────────────────────────────────────
def load_clip_int8():
    import onnxruntime as ort
    sess = ort.InferenceSession(str(CLIP_ONNX), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    def embed(path):
        x = preprocess(path, CLIP_MEAN, CLIP_STD)
        return sess.run(None, {in_name: x})[0][0]

    # Size on disk is what matters for an already-quantised artefact.
    mib = CLIP_ONNX.stat().st_size / 1048576
    return "clip_int8", embed, None, mib


def load_clip_fp32():
    from transformers import CLIPVisionModelWithProjection
    model = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32").eval()
    params = sum(p.numel() for p in model.parameters())

    @torch.no_grad()
    def embed(path):
        x = torch.from_numpy(preprocess(path, CLIP_MEAN, CLIP_STD))
        return model(pixel_values=x).image_embeds[0].numpy()

    return "clip_fp32", embed, params, None


def _load_dinov2(concat_patches):
    from transformers import AutoModel
    model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    params = sum(p.numel() for p in model.parameters())

    @torch.no_grad()
    def embed(path):
        x = torch.from_numpy(preprocess(path, IMAGENET_MEAN, IMAGENET_STD))
        h = model(pixel_values=x).last_hidden_state[0]   # [1 + n_patches, 384]
        cls = h[0]
        if not concat_patches:
            return cls.numpy()
        return torch.cat([cls, h[1:].mean(dim=0)]).numpy()

    name = "dinov2_s_cp" if concat_patches else "dinov2_s"
    return name, embed, params, None


def load_mobilenet():
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).eval()
    params = sum(p.numel() for p in model.parameters())

    @torch.no_grad()
    def embed(path):
        x = torch.from_numpy(preprocess(path, IMAGENET_MEAN, IMAGENET_STD))
        f = model.features(x)
        return torch.flatten(model.avgpool(f), 1)[0].numpy()

    return "mnv3_s", embed, params, None


def load_onnx(name, path, mean, std):
    """Generic ONNX encoder -- used for the exported DINOv2 graphs."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    def embed(p):
        return sess.run(None, {in_name: preprocess(p, mean, std)})[0][0]

    return name, embed, None, path.stat().st_size / 1048576


EXPORT_DIR = REPO / "python" / "onnx_export"

ENCODERS = [
    load_clip_int8,
    load_clip_fp32,
    lambda: _load_dinov2(False),
    lambda: _load_dinov2(True),
    lambda: load_onnx("dino_onnx_fp32", EXPORT_DIR / "dinov2_small_cls_fp32.onnx",
                      IMAGENET_MEAN, IMAGENET_STD),
    lambda: load_onnx("dino_onnx_int8", EXPORT_DIR / "dinov2_small_cls_int8.onnx",
                      IMAGENET_MEAN, IMAGENET_STD),
    load_mobilenet,
]


# ─────────────────────────────────────────────
# LIBRARY MATH (mirrors ReferenceLibrary)
# ─────────────────────────────────────────────
def l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_prototypes(embeddings_by_type, skip=None):
    """prototype(type) = mean of L2-normalised examples, optionally holding one out."""
    protos = {}
    for t, items in embeddings_by_type.items():
        kept = [e for i, e in enumerate(items) if skip != (t, i)]
        if kept:
            protos[t] = np.mean([l2(e) for e in kept], axis=0)
    return protos


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def classify(query, protos):
    """Returns (best_type, best_score, second_type, second_score)."""
    scores = sorted(((cosine(query, p), t) for t, p in protos.items()), reverse=True)
    best_s, best_t = scores[0]
    if len(scores) > 1:
        second_s, second_t = scores[1]
    else:
        second_s, second_t = -1.0, None
    return best_t, best_s, second_t, second_s


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def collect_crops():
    crops = {}
    for t in FEATURE_TYPES:
        d = CROPS_DIR / t
        if not d.is_dir():
            continue
        files = sorted(
            p for p in d.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if files:
            crops[t] = files
    return crops


def evaluate(name, embed, crops):
    """Leave-one-out prototype classification + separation stats."""
    embs, t0 = {}, time.perf_counter()
    n_crops = 0
    for t, files in crops.items():
        embs[t] = [embed(p) for p in files]
        n_crops += len(files)
    ms_per_crop = (time.perf_counter() - t0) * 1000 / max(1, n_crops)

    # --- leave-one-out ---
    top1 = 0
    accepted_correct = 0
    below = ambiguous = 0
    margins, best_scores = [], []
    confusion = {}
    total = 0

    for t, items in embs.items():
        for i, q in enumerate(items):
            protos = build_prototypes(embs, skip=(t, i))
            if t not in protos:           # single-example class: nothing left
                continue
            total += 1
            bt, bs, st, ss = classify(q, protos)
            best_scores.append(bs)
            margin = bs - ss if st is not None else float("inf")
            margins.append(margin)

            if bt == t:
                top1 += 1
            else:
                confusion[(t, bt)] = confusion.get((t, bt), 0) + 1

            # Exactly ReferenceLibrary.classify(): threshold, then margin.
            if bs < SIMILARITY_THRESHOLD:
                below += 1
            elif st is not None and margin <= MARGIN_THRESHOLD:
                ambiguous += 1
            elif bt == t:
                accepted_correct += 1

    # --- separation: mean intra-class vs mean inter-class cosine ---
    intra, inter = [], []
    flat = [(t, l2(e)) for t, items in embs.items() for e in items]
    for a in range(len(flat)):
        for b in range(a + 1, len(flat)):
            (ta, ea), (tb, eb) = flat[a], flat[b]
            (intra if ta == tb else inter).append(cosine(ea, eb))

    X = np.array([e for _, e in flat])
    y = [t for t, _ in flat]
    try:
        sil = silhouette_score(X, y, metric="cosine")
    except Exception:
        sil = float("nan")

    finite_margins = [m for m in margins if np.isfinite(m)]
    return {
        "name": name,
        "dim": len(next(iter(embs.values()))[0]),
        "ms": ms_per_crop,
        "top1": top1 / total if total else 0.0,
        "accepted": accepted_correct / total if total else 0.0,
        "below": below / total if total else 0.0,
        "ambiguous": ambiguous / total if total else 0.0,
        "intra": float(np.mean(intra)) if intra else float("nan"),
        "inter": float(np.mean(inter)) if inter else float("nan"),
        "sep": float(np.mean(intra) - np.mean(inter)) if intra and inter else float("nan"),
        "sil": sil,
        "score_lo": float(np.min(best_scores)) if best_scores else float("nan"),
        "score_hi": float(np.max(best_scores)) if best_scores else float("nan"),
        "margin_med": float(np.median(finite_margins)) if finite_margins else float("nan"),
        "margin_min": float(np.min(finite_margins)) if finite_margins else float("nan"),
        "confusion": confusion,
        "total": total,
    }


def main():
    crops = collect_crops()
    if not crops:
        print(f"No reference crops found under {CROPS_DIR}")
        return

    print("Reference crops")
    for t, files in crops.items():
        warn = "   <- too few for a stable prototype" if len(files) < 4 else ""
        print(f"  {t:12s} {len(files):3d}{warn}")
    print(f"  {'TOTAL':12s} {sum(len(f) for f in crops.values()):3d}\n")

    results = []
    for loader in ENCODERS:
        try:
            name, embed, params, disk_mib = loader()
        except Exception as e:
            print(f"  skipped an encoder ({type(e).__name__}: {e})")
            continue
        print(f"Embedding with {name} ...")
        r = evaluate(name, embed, crops)
        # Realistic on-device size: int8 ~= 1 byte/param, or measured file size.
        r["mib"] = disk_mib if disk_mib is not None else params / 1048576
        r["params_m"] = params / 1e6 if params else float("nan")
        results.append(r)

    # ── report ───────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("LEAVE-ONE-OUT PROTOTYPE CLASSIFICATION"
          f"  (n={results[0]['total']} crops, app thresholds: "
          f"sim>={SIMILARITY_THRESHOLD}, margin>{MARGIN_THRESHOLD})")
    print("=" * 100)
    hdr = (f"{'encoder':12s} {'dim':>5s} {'int8 MiB':>9s} {'ms/crop':>8s} "
           f"{'top-1':>7s} {'accept':>7s} {'unk:lo':>7s} {'unk:amb':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda r: -r["top1"]):
        print(f"{r['name']:12s} {r['dim']:5d} {r['mib']:9.1f} {r['ms']:8.1f} "
              f"{r['top1']:6.1%} {r['accepted']:6.1%} {r['below']:6.1%} "
              f"{r['ambiguous']:7.1%}")

    print("\n" + "=" * 100)
    print("EMBEDDING-SPACE SEPARATION  (higher sep/silhouette = classes further apart)")
    print("=" * 100)
    hdr2 = (f"{'encoder':12s} {'intra':>7s} {'inter':>7s} {'sep':>7s} {'silhou':>7s} "
            f"{'best score range':>18s} {'margin med':>11s} {'margin min':>11s}")
    print(hdr2)
    print("-" * len(hdr2))
    for r in sorted(results, key=lambda r: -r["sep"]):
        rng = f"{r['score_lo']:.3f}-{r['score_hi']:.3f}"
        print(f"{r['name']:12s} {r['intra']:7.3f} {r['inter']:7.3f} {r['sep']:7.3f} "
              f"{r['sil']:7.3f} {rng:>18s} {r['margin_med']:11.3f} {r['margin_min']:11.3f}")

    print("\n" + "=" * 100)
    print("CONFUSIONS  (true -> predicted)")
    print("=" * 100)
    for r in sorted(results, key=lambda r: -r["top1"]):
        if not r["confusion"]:
            print(f"{r['name']:12s} none")
            continue
        pairs = ", ".join(
            f"{a}->{b} x{n}"
            for (a, b), n in sorted(r["confusion"].items(), key=lambda kv: -kv[1])
        )
        print(f"{r['name']:12s} {pairs}")

    print("\n" + "=" * 100)
    print("THRESHOLD CALIBRATION")
    print("=" * 100)
    print(f"The app hard-codes sim>={SIMILARITY_THRESHOLD} / margin>{MARGIN_THRESHOLD}. "
          "Those are CLIP-specific numbers;\nany encoder swap needs its own values. "
          "Suggested starting points per encoder\n(threshold = 5th percentile of "
          "correct best-scores, margin = half the median correct margin):")
    print()
    for r in sorted(results, key=lambda r: -r["top1"]):
        print(f"  {r['name']:12s} sim>={max(0.0, r['score_lo']):.2f}  "
              f"margin>{r['margin_med'] / 2:.3f}")


if __name__ == "__main__":
    main()
