"""Summarises the 40-photo study: detection, classification, verdicts, selection, gate.

argv: <export json> <java study json> <out json>
"""
import json
import sys
from collections import Counter, defaultdict

exp = json.load(open(sys.argv[1]))
res = json.load(open(sys.argv[2]))
METHODS = [f"v{i}" for i in range(1, 19)]
REGION_NAME = {"4 screws": "Region 1", "3 thread holes": "Region 2", "3 holes": "Region 3"}
N_FEATURES = {r: len(exp["regions"][r]["reference"]["features"]) for r in exp["regions"]}
summary = {}


def conf(rows):
    tp = sum(1 for k, v in rows if k == "defected" and v == "FAIL")
    fn = sum(1 for k, v in rows if k == "defected" and v == "PASS")
    fp = sum(1 for k, v in rows if k == "lighting" and v == "FAIL")
    tn = sum(1 for k, v in rows if k == "lighting" and v == "PASS")
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, acc=acc, prec=prec, rec=rec, f1=f1,
                false_alarm=fp / max(1, fp + tn), false_pass=fn / max(1, tp + fn))


# ── detection (position-matched against the labelled reference features) ──────────────────
det = {}
for m in METHODS:
    found = total = extras = exact = photos = 0
    per_region = {}
    for r, photos_ in res.items():
        fr = tr = er = 0
        for v, p in photos_.items():
            if p["kind"] == "defected":
                continue
            d = p["dino"]["methods"][m]
            fr += d["matched"]; tr += N_FEATURES[r]; er += d["extras"]
            exact += int(d["matched"] == N_FEATURES[r] and d["extras"] == 0)
            photos += 1
        per_region[REGION_NAME[r]] = dict(found=fr, total=tr, extras=er)
        found += fr; total += tr; extras += er
    det[m] = dict(found=found, total=total, recall=found / total, extras=extras, exact=exact, photos=photos,
                  regions=per_region)
summary["detection"] = det

# ── classification: of reference features that got a box, how many carry the right label ──
cls = {}
for enc in ("dino", "clip"):
    cls[enc] = {}
    for m in METHODS:
        agree = matched = 0
        for r, photos_ in res.items():
            for v, p in photos_.items():
                if p["kind"] == "defected":
                    continue
                d = p[enc]["methods"][m]
                agree += d["labelsAgreeing"]; matched += d["matched"]
        cls[enc][m] = dict(agree=agree, matched=matched, acc=agree / matched if matched else 0.0)
summary["classification"] = cls

# ── verdicts ──────────────────────────────────────────────────────────────────────────────
ver = {}
for enc in ("dino", "clip"):
    ver[enc] = {}
    for m in METHODS + ["adaptive"]:
        rows_full, rows_count, angle_full = [], [], []
        regions = {}
        for r, photos_ in res.items():
            rr = []
            for v, p in photos_.items():
                mm = p[enc]["chosen"] if m == "adaptive" else m
                d = p[enc]["methods"][mm]
                if p["kind"] == "angle":
                    angle_full.append(d["verdict"])
                    continue
                rows_full.append((p["kind"], d["verdict"]))
                rows_count.append((p["kind"], d["countVerdict"]))
                rr.append((p["kind"], d["verdict"]))
            regions[REGION_NAME[r]] = conf(rr)
        ver[enc][m] = dict(full=conf(rows_full), count=conf(rows_count),
                           angle_pass=sum(1 for x in angle_full if x == "PASS"), angle_n=len(angle_full),
                           regions=regions)
summary["verdicts"] = ver

# ── what the adaptive stage chose ─────────────────────────────────────────────────────────
choices = {}
for enc in ("dino",):
    for r, photos_ in res.items():
        choices[REGION_NAME[r]] = {v: p[enc]["chosen"] for v, p in photos_.items()}
summary["choices"] = choices
summary["choice_counts"] = {reg: dict(Counter(c.values())) for reg, c in choices.items()}

# ── positional check availability and what it changed ───────────────────────────────────
pos = defaultdict(lambda: Counter())
for r, photos_ in res.items():
    for v, p in photos_.items():
        for m in METHODS:
            d = p["dino"]["methods"][m]
            pos[REGION_NAME[r]]["aligned" if d["aligned"] else "unaligned"] += 1
            if d["countVerdict"] == "PASS" and d["verdict"] == "FAIL":
                pos[REGION_NAME[r]][f"positional_failed_{p['kind']}"] += 1
summary["positional"] = {k: dict(v) for k, v in pos.items()}

# ── alignment gate: area test, whole-photo DINOv2 cosine at 0.70 ─────────────────────────
gate = exp["gate"]
refs = {"4 screws": "4 screws normal", "3 thread holes": "3 thread holes normal", "3 holes": "3 holes normal"}
def region_of(n):
    return next(r for r in ("3 thread holes", "3 holes", "4 screws") if n.startswith(r + " "))
same, diff = [], []
for r, ref in refs.items():
    for n, s in gate[ref].items():
        if n == ref:
            continue
        (same if region_of(n) == r else diff).append((n, s))
summary["gate"] = dict(
    same_min=min(s for _, s in same), same_mean=sum(s for _, s in same) / len(same),
    diff_max=max(s for _, s in diff), diff_mean=sum(s for _, s in diff) / len(diff),
    false_reject_070=sum(1 for _, s in same if s < 0.70), same_n=len(same),
    false_accept_070=sum(1 for _, s in diff if s >= 0.70), diff_n=len(diff),
    same_below=[(n, s) for n, s in same if s < 0.70])
# ── review prompts the app would raise, under the method it selects for each photo ────────
prompts = dict(inspections=0, asked=0, on_feature=0, nearest_right=0, off_feature=0, capped=0,
               per_kind=defaultdict(list))
for r, photos_ in res.items():
    feats = exp["regions"][r]["reference"]["features"]
    for v, p in photos_.items():
        m = p["dino"]["chosen"]
        dets = exp["regions"][r]["photos"][v]["methods"][m]
        box_feature = {b: i for i, b in enumerate(p["dino"]["methods"][m]["boxForFeature"]) if b >= 0}
        cands = [i for i, d in enumerate(dets) if 0.30 <= d["dino"]["score"] < 0.74]
        prompts["inspections"] += 1
        prompts["capped"] += int(len(cands) > 8)
        asked = cands[:8]
        prompts["asked"] += len(asked)
        prompts["per_kind"][p["kind"]].append(len(asked))
        for i in asked:
            if i in box_feature:
                prompts["on_feature"] += 1
                prompts["nearest_right"] += int(dets[i]["dino"]["nearest"] == feats[box_feature[i]]["type"])
            else:
                prompts["off_feature"] += 1
prompts["per_kind"] = {k: dict(n=len(v), total=sum(v), max=max(v)) for k, v in prompts["per_kind"].items()}
summary["prompts"] = prompts
json.dump(summary, open(sys.argv[3], "w"), indent=1)
print("PROMPTS", prompts)

# ── print ─────────────────────────────────────────────────────────────────────────────────
print("DETECTION (34 non-defective photos; features found / extras on part / exact)")
for m in METHODS:
    d = det[m]
    print(f"  {m:4s} found {d['found']:3d}/{d['total']} ({d['recall']:.1%})  extras {d['extras']:4d}  exact {d['exact']:2d}/{d['photos']}")
print("CLASSIFICATION (matched features labelled correctly)")
for m in METHODS:
    print(f"  {m:4s} dino {cls['dino'][m]['acc']:.1%} ({cls['dino'][m]['agree']}/{cls['dino'][m]['matched']})"
          f"   clip {cls['clip'][m]['acc']:.1%} ({cls['clip'][m]['agree']}/{cls['clip'][m]['matched']})")
for enc in ("dino", "clip"):
    print(f"VERDICTS {enc} (27 lighting + 3 defects; full = counts + positional)")
    for m in METHODS + ["adaptive"]:
        f, c = ver[enc][m]["full"], ver[enc][m]["count"]
        print(f"  {m:8s} full TP{f['tp']} FN{f['fn']} FP{f['fp']} TN{f['tn']} acc {f['acc']:.1%} F1 {f['f1']:.1%}"
              f" | count-only TN{c['tn']} TP{c['tp']}  | angles {ver[enc][m]['angle_pass']}/{ver[enc][m]['angle_n']}")
print("CHOICES", summary["choice_counts"])
print("POSITIONAL", summary["positional"])
print("GATE", {k: v for k, v in summary["gate"].items() if k != "same_below"}, summary["gate"]["same_below"])
