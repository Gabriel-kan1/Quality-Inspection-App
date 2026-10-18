"""
What the system does with a fastener type it has not been taught, and how that changes once the
operator labels it.

Region 2 (three Type A PEM inserts) is inspected three times over the same detections:

  before    the part's library holds only the bundled example crops (bolt head, hole, rivet head,
            screw, stud), as when a new part is first set up and no insert has been labelled.
  one       the operator labels one insert on the reference, adding a "type_a_pem" type with
            that single crop.
  three     the operator labels all three inserts on the reference.

Detections are the application's own: for each photo, the boxes of the method the app's selection
chose (testset_java.json), mapped to the reference's three insert positions by its position fit.
Every box is cropped and embedded exactly as the app does (variation_test_study.Photo/Encoder) and
classified with the shipped thresholds. The expected counts are what the reference photo itself is
classified as under the same library, as an operator accepting the suggestions would set them.

    python python/report_study/learning_study.py "<test photo dir>" python/report_study/results
"""
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import variation_test_study as V  # noqa: E402

PHOTOS, RESULTS = sys.argv[1], sys.argv[2]
REGION = "3 thread holes"
CROPS = Path(HERE).parent.parent / "app/src/main/assets/reference_crops"
DEFAULT_TYPES = ["bolt_head", "hole", "rivet_head", "screw", "stud"]
NEW = "type_a_pem"

E = json.load(open(os.path.join(RESULTS, "testset_export.json")))["regions"][REGION]
J = json.load(open(os.path.join(RESULTS, "testset_java.json")))[REGION]
enc = V.Encoder()


def proto(vectors):
    m = np.mean(vectors, axis=0)
    return m / np.linalg.norm(m)


# The bundled library: every example crop of each generic type.
default = {}
for t in DEFAULT_TYPES:
    vs = []
    for f in sorted(glob.glob(str(CROPS / t / "*.jpg"))):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is not None:
            vs.append(enc.embed(img))
    default[t] = vs


def library(pem_vectors):
    lib = {t: proto(v) for t, v in default.items() if v}
    if pem_vectors:
        lib[NEW] = proto(pem_vectors)
    return lib


def crop_vec(photo, rect):
    H, W = photo.work.shape[:2]
    x0, y0, x1, y1 = rect
    box = (round(x0 * W), round(y0 * H), round((x1 - x0) * W), round((y1 - y0) * H))
    return enc.embed(photo.crop(box))


ref_photo = V.Photo(os.path.join(PHOTOS, f"{REGION} normal.jpg"))
ref_vecs = [crop_vec(ref_photo, f["rect"]) for f in E["reference"]["features"]]
LIBS = {"before": library([]), "one": library(ref_vecs[:1]), "three": library(ref_vecs)}


def counts(labels):
    out = {}
    for t in labels:
        if t != "unknown":
            out[t] = out.get(t, 0) + 1
    return out


results = {}
for cond, lib in LIBS.items():
    ref_labels = [V.classify(v, lib)[0] for v in ref_vecs]
    expected = counts(ref_labels)
    rows = {}
    for name, info in J.items():
        p = V.Photo(os.path.join(PHOTOS, f"{REGION} {name}.jpg"))
        chosen = info["dino"]["chosen"]
        boxes = E["photos"][name]["methods"][chosen]
        vecs = [crop_vec(p, b["rect"]) for b in boxes]
        labels = [V.classify(v, lib) for v in vecs]
        on_feature = info["dino"]["methods"][chosen]["boxForFeature"]
        feat_labels = [labels[i][0] if i >= 0 else None for i in on_feature]
        verdict = "PASS" if counts([l for l, _ in labels]) == expected else "FAIL"
        rows[name] = dict(kind=info["kind"], method=chosen, verdict=verdict,
                          feature_labels=feat_labels,
                          feature_scores=[round(labels[i][1], 3) if i >= 0 else None for i in on_feature],
                          counts=counts([l for l, _ in labels]))
    results[cond] = dict(reference_labels=ref_labels, expected=expected, photos=rows)

# Summary: insert positions labelled as the new type, lighting passes, and the defect.
summary = {}
for cond, r in results.items():
    found = correct = 0
    by_label = {}
    for name, row in r["photos"].items():
        if row["kind"] == "defected":
            continue
        for lab in row["feature_labels"]:
            if lab is None:
                continue
            found += 1
            correct += lab == NEW
            by_label[lab] = by_label.get(lab, 0) + 1
    lighting = [row for row in r["photos"].values() if row["kind"] == "lighting"]
    defect = r["photos"]["defected"]
    summary[cond] = dict(expected=r["expected"], inserts_found=found, labelled_as_pem=correct,
                         label_counts=by_label,
                         lighting_passed=sum(row["verdict"] == "PASS" for row in lighting),
                         lighting_n=len(lighting), defect_verdict=defect["verdict"],
                         defect_counts=defect["counts"])

json.dump(dict(summary=summary, detail=results), open(os.path.join(RESULTS, "learning_study.json"), "w"),
          indent=1)
for cond, s in summary.items():
    print(cond, json.dumps(s))
