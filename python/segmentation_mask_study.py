"""
Learned background removal against the hand-tuned chassis mask.

THE QUESTION
------------
ml/ChassisMask keys on clutter and local brightness, with the rules fitted by hand
(background_mask_study.py has the five that were tried before it). Consumer tools -- Canva,
Windows Photos/Paint, iOS "lift subject" -- do not use rules at all: they run a segmentation
network trained on hand-cut photographs. This study puts the open-source members of that
family on the same 26 hand-counted photographs, through the same detector, scored the same way.

THE CONTENDERS
--------------
    current       ml/ChassisMask, mirrored by background_mask_study.background_mask
    u2netp        U^2-Net small, 4.6 MB -- the phone-sized salient-object model (rembg)
    u2net         U^2-Net full, 176 MB
    isnet         IS-Net / DIS general-use, 178 MB
    birefnet      BiRefNet lite (Swin-T), 224 MB -- the current state of the art in rembg
    slimsam       SlimSAM-77 (SAM pruned to ~38 MB), PROMPTED with a 3x3 grid of points in the
                  central 40% of the frame -- "segment the thing in the middle", which is what
                  the alignment gate already makes the operator frame
    sam-base      SAM ViT-B, 375 MB, the same prompt -- the upper bound for the prompted family
    dinov2        the DINOv2-S already shipped, patch tokens instead of CLS: every 14 px patch
                  scored by cosine similarity to the mean patch of the central 40%, split by
                  Otsu. No new model at all, only a re-export.

    mobilesam     MobileSAM (Ultralytics), 39 MB      } the phone-sized SAMs, same prompt
    effsam-ti     EfficientSAM-Ti (TorchScript), 39 MB }
    sam2-tiny     SAM 2.1 Hiera-tiny, ~150 MB          }
    edgetam       EdgeTAM, ~55 MB                      }
    sam-base-int8 SAM ViT-B with dynamic int8 Linears, 103 MB

The salient-object models answer "what is the subject of this photograph?", which is
ill-defined when the part fills the frame. SAM and DINOv2 are told where the part is.

COMPOSITES
----------
    agree:a+b     background only where EVERY named model says so
    gated:m/f     the SUBJECT GATE: if model m's largest foreground region is >= 20% of the
                  frame, the photograph is a part on a backdrop and everything outside that
                  region (holes filled) is background, cap raised to 85%; otherwise method f
                  (default 'current', or 'none'). On a close-up a salient model names the
                  fasteners as the subject -- several blobs, none over 6.7% of the frame here
                  -- so the gate never fires on one.

The setting carried forward is  gated:u2netp/agree:mobilesam+sam2-tiny  -- U^2-Netp for a
part on a backdrop (1.jpg, the L-bracket on white), the two small SAMs agreeing otherwise.

THE AREA FLOOR (--floor)
------------------------
The detector's dynamic area floor is set from the three largest contours. Removing the
background can remove exactly those, and the floor then collapses and lets plate texture
through. 'masked' recomputes the floor after masking (the app today); 'premask' carries the
unmasked photograph's floor over; 'gated' (default) carries it over only on the subject-gate
branch. 'premask' everywhere cost SAM-base a fastener on hole1, which is why it is not the
default; the gate branch is where the collapse is certain (1.jpg 9 -> 17 boxes).

ONE POST-PROCESS FOR ALL OF THEM
--------------------------------
Every method's background is passed through the app's own safety rule before it is used:
only regions that TOUCH THE FRAME BORDER and cover >= 1% are removed, and a mask over 45% of
the frame is refused outright. So the comparison is between ways of proposing background,
not between safety policies, and no method can delete the middle of the part. The raw
coverage (before the rule) is printed alongside, because it says how the model reads the
scene.

Removed pixels are filled with the plate's median grey and the seam is cut, as in the app.
Edges are v6 (the medium rung), as in background_mask_study section 3.

Run:  python python/segmentation_mask_study.py [--methods a,b,...] [--sheet OUTDIR]
                                                [--floor masked|premask|gated] [--fill grey|cut]
                                                [--images PATH ...] [--propagate METHOD]
--images runs on photographs outside the hand-counted set (masks and sheets only).
Model files are looked up in $SEG_MODELS (default: ./seg_models): u2netp/u2net/isnet-general-use/
birefnet-general-lite (BiRefNet-general-bb_swin_v1_tiny-epoch_232) .onnx from
github.com/danielgatis/rembg/releases/tag/v0.0.0, mobile_sam.pt from the Ultralytics assets
release, efficient_sam_vitt_torchscript.pt from github.com/yformer/EfficientSAM; SAM, SAM 2.1,
EdgeTAM and DINOv2 come from Hugging Face. Needs torch, transformers, timm, onnxruntime and
ultralytics. Masks are cached under $SEG_MODELS/cache.
"""

import argparse
import os
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import detector_ab_study as D  # noqa: E402
import detection_ground_truth as GTRUTH  # noqa: E402
import hybrid_v2_study as HV  # noqa: E402
import adaptive_preprocessing_study as A  # noqa: E402
import background_mask_study as BM  # noqa: E402

MODELS = os.environ.get("SEG_MODELS", os.path.join(os.path.dirname(__file__), "seg_models"))
CACHE = os.path.join(MODELS, "cache")

MIN_BACKGROUND_FRACTION = BM.MIN_BACKGROUND_FRACTION
MAX_BACKGROUND_FRACTION = BM.MAX_BACKGROUND_FRACTION
CENTRE_FRACTION = BM.CENTRE_FRACTION

METHODS = ["current", "u2netp", "u2net", "isnet", "birefnet", "slimsam", "sam-base", "dinov2",
           "mobilesam", "effsam-ti", "sam2-tiny", "edgetam"]
SIZES = {"current": "0", "u2netp": "4.6 MB", "u2net": "176 MB", "isnet": "178 MB",
         "birefnet": "224 MB", "slimsam": "~38 MB", "sam-base": "375 MB", "dinov2": "shipped",
         "mobilesam": "39 MB", "effsam-ti": "39 MB", "sam2-tiny": "~150 MB", "edgetam": "~55 MB",
         "sam-base-int8": "~100 MB"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ─────────────────────────────────────────────
# Foreground probability, one function per family. Each returns float32 HxW in [0, 1].
# ─────────────────────────────────────────────
_SESSIONS = {}


def _ort(name):
    import onnxruntime as ort
    if name not in _SESSIONS:
        so = ort.SessionOptions()
        # BiRefNet's Swin attention at 1024^2 exhausts the default arena on a 16 GB laptop.
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        _SESSIONS[name] = ort.InferenceSession(os.path.join(MODELS, name), so,
                                               providers=["CPUExecutionProvider"])
    return _SESSIONS[name]


def rembg_family(bgr, model):
    """Preprocessing exactly as rembg's sessions do it."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if model in ("u2netp", "u2net"):
        size, mean, std, sigmoid = 320, IMAGENET_MEAN, IMAGENET_STD, False
    elif model == "isnet":
        size, mean, std, sigmoid = 1024, np.full(3, 0.5, np.float32), np.ones(3, np.float32), False
    else:  # birefnet
        size, mean, std, sigmoid = 1024, IMAGENET_MEAN, IMAGENET_STD, True
    x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    x = (x / max(float(x.max()), 1e-6) - mean) / std
    x = x.transpose(2, 0, 1)[None]
    file = {"u2netp": "u2netp.onnx", "u2net": "u2net.onnx", "isnet": "isnet-general-use.onnx",
            "birefnet": "birefnet-general-lite.onnx"}[model]
    s = _ort(file)
    pred = s.run(None, {s.get_inputs()[0].name: x})[0][0, 0]
    if sigmoid:
        pred = 1.0 / (1.0 + np.exp(-pred))
    lo, hi = float(pred.min()), float(pred.max())
    pred = (pred - lo) / max(hi - lo, 1e-6)
    return cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


_SAM = {}


def centre_points(w, h, n=3):
    xs = np.linspace(w * (0.5 - CENTRE_FRACTION / 2), w * (0.5 + CENTRE_FRACTION / 2), n)
    ys = np.linspace(h * (0.5 - CENTRE_FRACTION / 2), h * (0.5 + CENTRE_FRACTION / 2), n)
    return [[float(x), float(y)] for y in ys for x in xs]


def sam_family(bgr, repo, int8=False):
    import torch
    from transformers import SamModel, SamProcessor
    key = repo + ("#int8" if int8 else "")
    if key not in _SAM:
        model = SamModel.from_pretrained(repo).eval()
        if int8:
            # Dynamic int8 on every Linear -- the same scheme the shipped DINOv2 encoder uses
            # (weights int8, activations quantised on the fly). ViT-B is almost all Linear.
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear},
                                                           dtype=torch.qint8)
        _SAM[key] = (model, SamProcessor.from_pretrained(repo))
    model, proc = _SAM[key]
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pts = centre_points(w, h)
    inputs = proc(rgb, input_points=[[pts]], input_labels=[[[1] * len(pts)]],
                  return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(), binarize=False)[0][0]   # (3, H, W) logits
    return pick_mask(torch.sigmoid(masks).numpy(), out.iou_scores[0, 0].cpu().numpy())


def pick_mask(probs, scores):
    """SAM's three masks are nested granularities. With nine points spread over the centre
    the part is the whole-object reading; take the best-scored mask that is not the entire
    frame (a mask of everything says nothing about background)."""
    order = np.argsort(-scores)
    for i in order:
        if (probs[i] > 0.5).mean() < 0.98:
            return probs[i].astype(np.float32)
    return probs[order[0]].astype(np.float32)


def sam2_family(bgr, repo, cls):
    """SAM 2.1-tiny and EdgeTAM: same prompt, SAM 2's processor (points nest one deeper)."""
    import torch
    import transformers
    from transformers import Sam2Processor
    if repo not in _SAM:
        _SAM[repo] = (getattr(transformers, cls).from_pretrained(repo).eval(),
                      Sam2Processor.from_pretrained(repo))
    model, proc = _SAM[repo]
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pts = centre_points(w, h)
    inputs = proc(images=rgb, input_points=[[pts]], input_labels=[[[1] * len(pts)]],
                  return_tensors="pt")
    sizes = inputs.pop("original_sizes")     # EdgeTAM's timm backbone rejects it as a kwarg
    with torch.no_grad():
        out = model(**inputs, multimask_output=True)
    masks = proc.post_process_masks(out.pred_masks.cpu(), sizes,
                                    binarize=False)[0][0]               # (3, H, W) logits
    return pick_mask(torch.sigmoid(masks).numpy(), out.iou_scores[0, 0].cpu().numpy())


def efficientsam(bgr):
    """EfficientSAM-Ti, the authors' TorchScript export; resizes to 1024 internally."""
    import torch
    if "effsam" not in _SAM:
        _SAM["effsam"] = torch.jit.load(
            os.path.join(MODELS, "efficient_sam_vitt_torchscript.pt")).eval()
    model = _SAM["effsam"]
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    pts = centre_points(w, h)
    img = torch.from_numpy(rgb.transpose(2, 0, 1))[None]
    points = torch.tensor(pts, dtype=torch.float32)[None, None]          # (1, 1, N, 2)
    labels = torch.ones((1, 1, len(pts)), dtype=torch.float32)
    with torch.no_grad():
        logits, iou = model(img, points, labels)                          # (1,1,3,H,W), (1,1,3)
    return pick_mask(torch.sigmoid(logits[0, 0]).numpy(), iou[0, 0].numpy())


def mobilesam(bgr):
    """MobileSAM through Ultralytics. Its predictor returns one binarised mask per prompt
    (multimask off -- SAM's own recommendation when several points pin the object), so
    there is no choice to make and the probability is 0 or 1."""
    from ultralytics import SAM
    if "mobilesam" not in _SAM:
        _SAM["mobilesam"] = SAM(os.path.join(MODELS, "mobile_sam.pt"))
    h, w = bgr.shape[:2]
    pts = centre_points(w, h)
    res = _SAM["mobilesam"].predict(bgr, points=[pts], labels=[[1] * len(pts)], verbose=False)
    m = res[0].masks.data[0].cpu().numpy().astype(np.float32)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m


_DINO = {}


def dinov2(bgr, side=448):
    """Patch similarity to the centre's mean patch, the shipped encoder's own features."""
    import torch
    from transformers import AutoModel
    if "m" not in _DINO:
        _DINO["m"] = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Keep aspect: long side `side`, both sides multiples of the 14 px patch.
    f = side / max(h, w)
    th, tw = max(14, int(round(h * f / 14)) * 14), max(14, int(round(w * f / 14)) * 14)
    x = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
    with torch.no_grad():
        tok = _DINO["m"](pixel_values=torch.from_numpy(x)).last_hidden_state[0, 1:].numpy()
    gh, gw = th // 14, tw // 14
    tok = tok / np.linalg.norm(tok, axis=1, keepdims=True)
    grid = tok.reshape(gh, gw, -1)
    cy0, cy1 = int(gh * (0.5 - CENTRE_FRACTION / 2)), int(np.ceil(gh * (0.5 + CENTRE_FRACTION / 2)))
    cx0, cx1 = int(gw * (0.5 - CENTRE_FRACTION / 2)), int(np.ceil(gw * (0.5 + CENTRE_FRACTION / 2)))
    proto = grid[cy0:cy1, cx0:cx1].reshape(-1, grid.shape[2]).mean(0)
    proto /= np.linalg.norm(proto)
    sim = grid @ proto                                       # (gh, gw) cosine
    s8 = np.clip((sim - sim.min()) / max(sim.max() - sim.min(), 1e-6) * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Map to a probability centred on the Otsu split so the common 0.5 cut lands on it.
    prob = 1.0 / (1.0 + np.exp(-(s8.astype(np.float32) - t) / 12.0))
    return cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)


def foreground(name, bgr, method):
    """Cached, because the big models take seconds a photograph on CPU."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{method}__{name}.npy")
    if os.path.exists(path):
        return np.load(path)
    t0 = time.time()
    if method in ("u2netp", "u2net", "isnet", "birefnet"):
        p = rembg_family(bgr, method)
    elif method == "slimsam":
        p = sam_family(bgr, "Zigeng/SlimSAM-uniform-77")
    elif method == "sam-base":
        p = sam_family(bgr, "facebook/sam-vit-base")
    elif method == "sam-base-int8":
        p = sam_family(bgr, "facebook/sam-vit-base", int8=True)
    elif method == "mobilesam":
        p = mobilesam(bgr)
    elif method == "effsam-ti":
        p = efficientsam(bgr)
    elif method == "sam2-tiny":
        p = sam2_family(bgr, "facebook/sam2.1-hiera-tiny", "Sam2Model")
    elif method == "edgetam":
        p = sam2_family(bgr, "yonigozlan/EdgeTAM-hf", "EdgeTamModel")
    elif method == "dinov2":
        p = dinov2(bgr)
    else:
        raise ValueError(method)
    TIMES.setdefault(method, []).append(time.time() - t0)
    p = p.astype(np.float16)
    np.save(path, p)
    return p


TIMES = {}


# ─────────────────────────────────────────────
# The app's safety rule, applied to every proposal
# ─────────────────────────────────────────────
def safe_background(bg, cap=None):
    """Border-touching components >= 1%; refuse a mask over 45% (or `cap`). uint8 0/255."""
    cap = MAX_BACKGROUND_FRACTION if cap is None else cap
    h, w = bg.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats((bg > 0).astype(np.uint8), 8)
    out = np.zeros_like(bg)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h
        if touches and area >= MIN_BACKGROUND_FRACTION * h * w:
            out[labels == i] = 255
    if np.count_nonzero(out) / float(h * w) > cap:
        return np.zeros_like(bg), True
    return out, False


# A salient-object model is trusted only when its subject is ONE region this large. On a
# close-up it names the fasteners as the subject -- several blobs, the largest <= 6.7% of
# the frame on every photograph measured -- and the plate as background, which must never
# be acted on. A part lying on a backdrop comes out as one region of 40%+.
SUBJECT_MIN_FRACTION = 0.20
# With the subject gate passed, the backdrop may legitimately be most of the frame.
GATED_CAP = 0.85


def subject_gate(name, img, model):
    """Background = everything outside the largest foreground region (its holes filled, so
    a dark recess inside the part is never background), or None if the gate refuses."""
    fg = (foreground(name, img, model).astype(np.float32) >= 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, 4]))
    if stats[i, 4] < SUBJECT_MIN_FRACTION * fg.size:
        return None
    part = (labels == i).astype(np.uint8)
    # Fill holes: flood the background from the border; whatever it cannot reach is part.
    h, w = part.shape
    flood = (1 - part).copy()
    reach = np.zeros((h + 2, w + 2), np.uint8)
    for (x, y) in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if flood[y, x]:
            cv2.floodFill(flood, reach, (x, y), 2)
    part[flood != 2] = 1
    return np.where(part == 0, 255, 0).astype(np.uint8)


def masks_for(name, img, method):
    """(raw background, safe background, refused?) as uint8 0/255 at working size."""
    if method == "current":
        m, _ = BM.background_mask(img)
        return m, m, False
    if method.startswith("gated:"):
        # 'gated:u2netp' -> the subject gate, falling back to the current mask when it
        # refuses; 'gated:u2netp/none' falls back to nothing, 'gated:u2netp/sam-base' (or
        # any other method) to that method -- the combination: one model for a part on a
        # backdrop, another for a part that fills the frame.
        model, _, fallback = method[6:].partition("/")
        bg = subject_gate(name, img, model)
        if bg is None:
            if fallback == "none":
                z = np.zeros(img.shape[:2], np.uint8)
                return z, z, False
            return masks_for(name, img, fallback or "current")
        safe, refused = safe_background(bg, GATED_CAP)
        return bg, safe, refused
    raw = raw_background(name, img, method)
    safe, refused = safe_background(raw)
    return raw, safe, refused


def raw_background(name, img, method):
    """uint8 0/255 where the method calls background. 'agree:a+b+...' is background only
    where EVERY named model says so -- their mistakes fall on different photographs, so
    requiring agreement should cancel them while the true background, which all of them
    find, survives."""
    if method.startswith("gated:"):
        return masks_for(name, img, method)[0]
    if method.startswith("agree:"):
        parts = [raw_background(name, img, m) > 0 for m in method[6:].split("+")]
        return np.where(np.logical_and.reduce(parts), 255, 0).astype(np.uint8)
    p = foreground(name, img, method).astype(np.float32)
    return np.where(p < 0.5, 255, 0).astype(np.uint8)


FILL = "grey"


def detect_with(img, mask, floor_mode=None):
    """FILL='grey' is what the app does. FILL='cut' derives edges from the UNMASKED photo and
    only deletes edges under the mask, so Canny's Otsu thresholds are identical to no mask
    and a count can change only because a detection was removed -- the mask judged alone."""
    work = img.copy()
    if np.any(mask) and FILL == "grey":
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plate = int(np.median(g[mask == 0])) if np.any(mask == 0) else 128
        work[mask > 0] = (plate, plate, plate)
    edges = HV.edges_for(work, HV.cfg(**BM.V6))
    if not np.any(mask):
        return D.find_feature_boxes(edges)[0]
    edges[cv2.dilate(mask, BM.SEAM) > 0] = 0
    floor = None
    if (floor_mode or FLOOR) == "premask":
        # The dynamic area floor is set from the three largest contours. Removing the
        # background can remove those very contours (hole1's bench, 1.jpg's backdrop edge),
        # and the floor then collapses and lets plate texture through. Taking the floor from
        # the UNMASKED photograph keeps it where it was; a photograph the mask leaves alone
        # never reaches this line, so it cannot change.
        unmasked = HV.edges_for(img, HV.cfg(**BM.V6))
        floor = D.find_feature_boxes(unmasked)[1]["dynamic_min_area"]
    return D.find_feature_boxes(edges, min_area=floor)[0]


def floor_for(name, img, method):
    """FLOOR='gated': the pre-mask floor only when the subject gate fired, i.e. when a
    part-on-backdrop shot had its backdrop removed. That is the one case the collapse of
    the floor is certain to follow (1.jpg: 9 boxes -> 17 of mostly scratches), and the
    fallback branch -- whose floor the fix made worse on hole1 under SAM-base -- keeps the
    floor it always had."""
    if FLOOR != "gated":
        return FLOOR
    if method.startswith("gated:"):
        model = method[6:].partition("/")[0]
        if subject_gate(name, img, model) is not None:
            return "premask"
    return "masked"


# 'masked' = the floor is recomputed from what survives the mask (the app today, and what
# every table before the floor fix was scored with); 'gated' = premask on the subject-gate
# branch only (see floor_for);
# 'premask' = the floor of the unmasked photograph is carried over.
FLOOR = "gated"


# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────
def names():
    return sorted(list(GTRUTH.GT) + list(GTRUTH.WIDE))


def run(methods, imgs=None):
    if imgs is None:
        imgs = {n: A.image(n) for n in names()}
    imgs = {n: im for n, im in imgs.items() if im is not None}
    res = {}
    for m in methods:
        res[m] = {}
        for n, im in imgs.items():
            raw, safe, refused = masks_for(n, im, m)
            boxes = detect_with(im, safe, floor_for(n, im, m)) if n in GTRUTH.GT else []
            res[m][n] = dict(raw=100.0 * np.count_nonzero(raw) / raw.size,
                             safe=100.0 * np.count_nonzero(safe) / safe.size,
                             refused=refused, count=len(boxes), safe_mask=safe)
        print(f"   ... {m} done", flush=True)
    return imgs, res


def lab(m):
    """Short column label; 'agree:mobilesam+sam2-tiny' -> '&mob+s2t'."""
    for a, b in (("agree:", "&"), ("mobilesam", "mob"), ("sam2-tiny", "s2t"),
                 ("effsam-ti", "eff"), ("edgetam", "edg"), ("sam-base-int8", "sb-int8"), ("gated:", "G:")):
        m = m.replace(a, b)
    return m


def report(imgs, res, methods):
    if not any(n in GTRUTH.GT for n in imgs):
        coverage(imgs, res, methods)     # outside the hand-counted set: masks only
        return
    print("\n1. GROUND TRUTH (26 photographs; weighted = 2 x misses + extras)")
    base = {n: len(detect_with(imgs[n], np.zeros(imgs[n].shape[:2], np.uint8)))
            for n in GTRUTH.GT if n in imgs}
    w, fn, fp = GTRUTH.weighted(base)
    print(f"   {'method':<14}{'size':>9}{'weighted':>10}{'miss':>6}{'extra':>7}"
          f"{'masked>0':>10}{'refused':>9}{'s/img':>7}")
    print(f"   {'no mask':<14}{'':>9}{w:>10}{fn:>6}{fp:>7}")
    for m in methods:
        counts = {n: r["count"] for n, r in res[m].items() if n in GTRUTH.GT}
        w, fn, fp = GTRUTH.weighted(counts)
        fired = sum(1 for n, r in res[m].items() if r["safe"] > 0.05)
        refused = sum(1 for r in res[m].values() if r["refused"])
        t = f"{np.mean(TIMES[m]):.1f}" if m in TIMES else "cached" if m != "current" else ""
        print(f"   {lab(m):<14}{SIZES.get(m, ''):>9}{w:>10}{fn:>6}{fp:>7}{fired:>10}{refused:>9}{t:>7}")
    coverage(imgs, res, methods)
    counts_table(imgs, res, methods, base)


def coverage(imgs, res, methods):
    print("\n2. COVERAGE, % of frame removed after the safety rule (raw model reading in brackets)")
    hdr = "".join(f"{lab(m)[:14]:>15}" for m in methods)
    print(f"   {'asset':<19}{hdr}")
    for n in imgs:
        cells = []
        for m in methods:
            r = res[m][n]
            cell = f"{r['safe']:.1f}"
            if m != "current":
                cell += f" ({r['raw']:.0f})" if not r["refused"] else f" R({r['raw']:.0f})"
            cells.append(f"{cell:>15}")
        print(f"   {n:<19}{''.join(cells)}")
    print("   R = refused by the 45% cap (raw coverage too large to be background)")


def counts_table(imgs, res, methods, base):
    print("\n3. COUNTS where any method differs from no mask")
    print(f"   {'asset':<19}{'GT':>8}{'none':>6}" + "".join(f"{lab(m)[:13]:>14}" for m in methods))
    for n in sorted(GTRUTH.GT):
        if n not in imgs:
            continue
        row = [res[m][n]["count"] for m in methods]
        if any(c != base[n] for c in row):
            print(f"   {n:<19}{str(sorted(GTRUTH.GT[n])):>8}{base[n]:>6}"
                  + "".join(f"{c:>14}" for c in row))


def contact_sheet(imgs, res, methods, outdir, which):
    """One row per photograph, one column per method; removed area tinted red."""
    os.makedirs(outdir, exist_ok=True)
    tw = 260
    rows = []
    for n in which:
        im = imgs[n]
        th = int(im.shape[0] * tw / im.shape[1])
        tiles = []
        for m in methods:
            t = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
            mk = cv2.resize(res[m][n]["safe_mask"], (tw, th), interpolation=cv2.INTER_NEAREST) > 0
            raw_bg = None
            if m != "current":
                raw_bg = cv2.resize(raw_background(n, im, m), (tw, th),
                                    interpolation=cv2.INTER_NEAREST) > 0
            if raw_bg is not None:   # what the model thought, faint blue
                t[raw_bg & ~mk] = (0.6 * t[raw_bg & ~mk] + 0.4 * np.array([255, 120, 0])).astype(np.uint8)
            t[mk] = (0.45 * t[mk] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
            cv2.putText(t, lab(m), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(t, lab(m), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40)))
        row = np.hstack(tiles)
        cv2.putText(row, n, (6, row.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        rows.append(row)
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", help="write contact sheets (red = removed, blue = model said "
                                    "background but the safety rule kept it) to this dir")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--fill", choices=["grey", "cut"], default="grey")
    ap.add_argument("--floor", choices=["masked", "premask", "gated"], default="gated")
    ap.add_argument("--images", nargs="+", metavar="PATH",
                    help="run on these photographs instead of the bundled assets (masks and "
                         "contact sheets only -- they have no hand counts)")
    ap.add_argument("--propagate", metavar="METHOD",
                    help="also carry METHOD's reference outline onto same-scene photos")
    args = ap.parse_args()
    global FILL, FLOOR
    FILL = args.fill
    FLOOR = args.floor
    methods = args.methods.split(",")
    given = None
    if args.images:
        given = {os.path.basename(p): D.load_working_image(p) for p in args.images}
    imgs, res = run(methods, given)
    report(imgs, res, methods)
    if args.propagate:
        propagate(imgs, args.propagate)
    if args.sheet:
        all_names = list(imgs) if given else [n for n in names() if n in imgs]
        for i in range(0, len(all_names), 8):
            chunk = all_names[i:i + 8]
            sheet = contact_sheet(imgs, res, methods, args.sheet, chunk)
            path = os.path.join(args.sheet, f"sheet_{i // 8 + 1}.jpg")
            cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"   wrote {path}")



# ─────────────────────────────────────────────
# Option 3: segment the reference once, carry it to every inspection
# ─────────────────────────────────────────────
def homography(a, b, min_inliers=50):
    """SIFT + RANSAC, a -> b. An UPPER BOUND on what the app can do: the app aligns by
    LayoutMatcher (similarity/affine over a handful of fasteners), not a full homography
    over thousands of texture points."""
    sift = cv2.SIFT_create(4000)
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a, b))
    ka, da = sift.detectAndCompute(ga, None)
    kb, db = sift.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0
    good = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None, 0
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
    n = int(inl.sum()) if inl is not None else 0
    return (H if n >= min_inliers else None), n


def propagate(imgs, source):
    """For every verified same-scene pair (ref -> insp): the reference's background (from
    `source`, standing in for an outline the operator approved at setup) warped into the
    inspection photo, then the same safety rule. Scored against the inspection's own
    segmentation by IoU, and through the detector against the hand counts."""
    print(f"\n4. PROPAGATED FROM THE REFERENCE (reference outline = {lab(source)})")
    print(f"   {'reference':<19}{'inspection':<19}{'inliers':>8}{'warped%':>9}{'own%':>7}"
          f"{'IoU':>6}{'GT':>8}{'own':>5}{'warped':>7}")
    names_bg = [n for n in imgs if np.any(masks_for(n, imgs[n], source)[1])]
    for ref in names_bg:
        for insp in imgs:
            if insp == ref:
                continue
            H, n = homography(imgs[ref], imgs[insp])
            if H is None:
                continue
            h, w = imgs[insp].shape[:2]
            ref_bg = masks_for(ref, imgs[ref], source)[1]
            warped = cv2.warpPerspective(ref_bg, H, (w, h), flags=cv2.INTER_NEAREST)
            warped, _ = safe_background(warped)
            own = masks_for(insp, imgs[insp], source)[1]
            union = np.count_nonzero((warped > 0) | (own > 0))
            iou = np.count_nonzero((warped > 0) & (own > 0)) / union if union else 1.0
            gt = str(sorted(GTRUTH.GT[insp])) if insp in GTRUTH.GT else "wide"
            c_own = len(detect_with(imgs[insp], own)) if insp in GTRUTH.GT else ""
            c_w = len(detect_with(imgs[insp], warped)) if insp in GTRUTH.GT else ""
            print(f"   {ref:<19}{insp:<19}{n:>8}{100 * np.count_nonzero(warped) / warped.size:>8.1f}%"
                  f"{100 * np.count_nonzero(own) / own.size:>6.1f}%{iou:>6.2f}{gt:>8}{c_own!s:>5}{c_w!s:>7}")


if __name__ == "__main__":
    main()
