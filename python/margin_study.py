"""
What `ReferenceLibrary.marginThreshold` should be, fitted from labelled inspections rather
than asserted.

WHY THIS EXISTS
---------------
The shipping default is 0.02, which is not a fitted value: it is numerically identical to
`ThresholdTuner.MARGIN_MIN`, the floor that exists so tuning can never switch the
ambiguity check off. The sweep that blessed it ran over the 34 bundled reference crops -
a set that contains no near-ties at all, so every margin from 0.02 upward scores the same
on it and the sweep had no signal to prefer a larger one.

Real inspections do contain near-ties. Every classification the app makes is written to
`ClassificationEvent` with its score, its runner-up, and - once an operator reviews it -
what the hardware actually was. That is the population the threshold is supposed to
separate, so that is what this fits against.

    python python/margin_study.py                    # pull from the attached device
    python python/margin_study.py --db path/to.sqlite
    python python/margin_study.py --crops            # also run the bundled-crop baseline

TWO THINGS IT WILL TELL YOU THAT ARE EASY TO MISS
-------------------------------------------------
1. Whether the labels are trustworthy yet. Confirmations are not a random sample of what
   the classifier did: they come from the review queue (which only sees "unknown"), from
   spot checks, and from the operator tapping a box they thought looked wrong. Only the
   spot checks are unbiased. If the confirmed population is dominated by the other two,
   the separation this prints is optimistic and the number should not be shipped - so the
   split is reported before any recommendation.

2. Whether a threshold exists at all. If the worst gap among correct answers sits below
   the best gap among errors, the distributions overlap and no margin separates them; the
   answer is then a better encoder or a merged type, not a tuned constant.

Needs numpy. `--crops` additionally needs onnxruntime and opencv-python.
"""

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CROPS = REPO / "app/src/main/assets/reference_crops"
MODEL = REPO / "app/src/main/assets/models/feature_encoder.onnx"
PACKAGE = "com.aistudio.qualityinspection.bzpqrk"
DB_NAME = "shop_quality_inspection_db"

# ReferenceLibrary.FEATURE_TYPES - the app ignores any other subfolder.
FEATURE_TYPES = ["bolt_head", "hole", "rivet_head", "screw", "stud"]

SIMILARITY_THRESHOLD = 0.74
MARGIN_THRESHOLD = 0.02
HIGH_CONFIDENCE = 0.88
SWEEP = (0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.09, 0.10, 0.12, 0.15, 0.20)
# Unbiased confirmations required *inside* the accepted band before a value is recommended.
# Not a total spot-check count: samples drawn from the confident tail say nothing about
# where the boundary belongs, so counting them lets the gate pass on irrelevant evidence.
MIN_BAND_SPOT_CHECKS = 10


# ─────────────────────────────────────────────
# Source 1 - labelled inspection history off the device
# ─────────────────────────────────────────────

def pull_database(dest_dir):
    """Copies the app database off an attached device, WAL included.

    `adb shell run-as ... cat` corrupts the file: the shell translates line endings and a
    SQLite page is binary. `exec-out` is the byte-exact channel, and the -wal file has to
    come with it or every write since the last checkpoint is invisible.
    """
    adb = shutil.which("adb")
    if adb is None:
        local = Path(os.environ.get("LOCALAPPDATA", "")) / "Android/Sdk/platform-tools/adb.exe"
        home = Path.home() / "AppData/Local/Android/Sdk/platform-tools/adb.exe"
        adb = str(local) if local.exists() else (str(home) if home.exists() else None)
    if adb is None:
        sys.exit("adb not found on PATH - pass --db with a pulled database instead")

    dest = Path(dest_dir) / "history.sqlite"
    for suffix in ("", "-wal", "-shm"):
        out = Path(str(dest) + suffix)
        with open(out, "wb") as fh:
            r = subprocess.run([adb, "exec-out", "run-as", PACKAGE,
                                "cat", f"databases/{DB_NAME}{suffix}"],
                               stdout=fh, stderr=subprocess.PIPE)
        if r.returncode != 0 and suffix == "":
            sys.exit(f"could not read the database from the device: "
                     f"{r.stderr.decode(errors='replace').strip()}")
        if out.stat().st_size == 0:
            out.unlink(missing_ok=True)
    return dest


def shipping_encoder_id():
    """FeatureEmbedder.ENCODER_ID, read from the source so the two cannot drift apart."""
    java = REPO / "app/src/main/java/com/example/ml/FeatureEmbedder.java"
    try:
        for line in java.read_text(encoding="utf-8", errors="replace").splitlines():
            if "ENCODER_ID" in line and "=" in line and '"' in line:
                return line.split('"')[1]
    except OSError:
        pass
    return None


def load_history(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        encoders = list(con.execute(
            "select encoderId, count(*) from ClassificationEvent group by encoderId"))
    except sqlite3.DatabaseError as e:
        sys.exit(f"{db_path} is not readable as the app database ({e}). If you copied it with"
                 " `adb shell ... cat`, re-copy with `adb exec-out` - the shell corrupts it.")
    if not encoders:
        sys.exit("no classification events recorded yet")

    # Only the encoder in force now. Scores from two encoders do not share a numeric range,
    # so a margin replayed across both is meaningless - the same rule ThresholdTuner applies.
    #
    # Read from the Java rather than inferred from the data: the retired encoder usually has
    # *more* rows, simply because it ran for longer, so anything majority-based picks the
    # wrong one and silently fits the margin to a library the app no longer uses.
    current = shipping_encoder_id()
    print("encoders present: " + ", ".join(f"{e}={n}" for e, n in encoders))
    if current is None or not any(e == current for e, _ in encoders):
        newest = list(con.execute("select encoderId from ClassificationEvent"
                                  " order by timestamp desc limit 1"))
        current = newest[0][0] if newest else encoders[0][0]
        print(f"! FeatureEmbedder.ENCODER_ID not found in the events; "
              f"falling back to the most recent, {current}")
    print(f"using {current}\n")

    rows = list(con.execute(
        """select predictedType, predictedScore, runnerUpType, runnerUpScore, reason,
                  confirmedType, reviewSource, partId
           from ClassificationEvent where encoderId = ?""", (current,)))
    con.close()
    return rows


def report_sampling(rows):
    """How the confirmed labels were obtained - decides whether to trust what follows."""
    accepted = [r for r in rows if r[4] == "ACCEPTED"]
    confirmed = [r for r in accepted if r[5] is not None]
    print("=== Are these labels a fair sample? ===")
    print(f"  accepted predictions: {len(accepted)}, of which confirmed: {len(confirmed)}")

    by_source = {}
    for r in confirmed:
        by_source[r[6]] = by_source.get(r[6], 0) + 1
    for src, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
        note = ""
        if src == "spot_check":
            note = "  <- the only unbiased source"
        elif src == "overlay_tap":
            note = "  <- operator chose it because it looked wrong"
        elif src == "reference_setup":
            note = "  <- many labels from one photo; correlated, not independent"
        elif src == "noise_audit":
            note = "  <- gate-suppressed crops put back for checking"
        print(f"    {src:<14s} {n:4d}{note}")

    band = [r for r in accepted if r[1] is not None and r[1] <= HIGH_CONFIDENCE]
    band_confirmed = [r for r in band if r[5] is not None]
    band_spot = [r for r in band_confirmed if r[6] == "spot_check"]
    print(f"\n  in the {SIMILARITY_THRESHOLD:.2f}-{HIGH_CONFIDENCE:.2f} band:"
          f" {len(band)} accepted, {len(band_confirmed)} confirmed,"
          f" {len(band_spot)} of them by spot check")
    if band and not band_spot:
        print("  ! no spot check has ever landed in that band. Until some do, the"
              " correct-answer\n    distribution below is measured on crops the operator"
              " or the classifier pre-selected,\n    and its lower tail is optimistic.")
    # What gates the recommendation is unbiased coverage *of the band the errors live in*,
    # not the total spot-check count. Spot checks concentrated at high confidence say nothing
    # about where the boundary belongs, however many of them there are.
    return len(band_spot), len(confirmed)


def analyse(rows):
    """Runner-up gap for accepted predictions, split by whether the human agreed."""
    usable = [r for r in rows
              if r[4] == "ACCEPTED" and r[5] is not None and r[2] is not None
              and r[1] is not None and r[3] is not None]
    right = [r for r in usable if r[0] == r[5]]
    wrong = [r for r in usable if r[0] != r[5]]
    if not usable:
        sys.exit("no confirmed accepted predictions with a runner-up yet - nothing to fit")

    gaps_right = np.array([r[1] - r[3] for r in right])
    gaps_wrong = np.array([r[1] - r[3] for r in wrong])

    print(f"\n=== Runner-up gap, accepted predictions (n={len(usable)}) ===")
    if len(gaps_right):
        print(f"  operator AGREED  (n={len(gaps_right)})")
        for q in (0, 5, 10, 25, 50):
            print(f"    p{q:<3d} {np.percentile(gaps_right, q):.3f}")
    if len(gaps_wrong):
        print(f"\n  operator DISAGREED - silent misclassifications  (n={len(gaps_wrong)})")
        for r in sorted(wrong, key=lambda r: r[1] - r[3]):
            print(f"    said {r[0]:<12s} was {r[5]:<12s} score {r[1]:.3f}"
                  f"  gap {r[1] - r[3]:.3f}   ({r[6]})")
    else:
        print("\n  no silent misclassification has been found yet - nothing to separate,"
              "\n  so no margin can be fitted. Keep collecting.")
        return None

    print("\n=== What each margin would have done ===")
    print("  margin   silent errors caught   correct answers sent to review")
    for m in SWEEP:
        caught = 100.0 * (gaps_wrong <= m).mean()
        lost = 100.0 * (gaps_right <= m).mean() if len(gaps_right) else 0.0
        mark = "   <== shipping" if abs(m - MARGIN_THRESHOLD) < 1e-9 else ""
        print(f"   {m:.3f}      {caught:5.1f}%                 {lost:5.1f}%{mark}")

    worst_error = gaps_wrong.max()
    best_correct = gaps_right.min() if len(gaps_right) else float("inf")
    print(f"\n  worst silent error gap {worst_error:.3f}"
          f" | best correct-answer gap {best_correct:.3f}")
    if worst_error < best_correct:
        rec = (worst_error + best_correct) / 2
        print(f"  -> separable. Any margin in ({worst_error:.3f}, {best_correct:.3f})"
              f" catches every known error at no measured cost; midpoint {rec:.3f}.")
        return rec
    print("  -> NOT separable: the distributions overlap, so no margin divides them. The"
          "\n     fix is upstream - a better encoder, or two types that should be merged.")
    return None


# ─────────────────────────────────────────────
# Source 2 - leave-one-out over the bundled crops (the original calibration)
# ─────────────────────────────────────────────

def crop_baseline():
    try:
        import cv2
        import onnxruntime as ort
    except ImportError:
        print("\n(--crops needs opencv-python and onnxruntime; skipping)")
        return
    if not MODEL.exists():
        print(f"\n({MODEL} not found; skipping the crop baseline)")
        return

    INPUT = 224
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)
    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    def embed(path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        s = INPUT / min(w, h)
        rw, rh = max(INPUT, round(w * s)), max(INPUT, round(h * s))
        r = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
        ox, oy = (rw - INPUT) // 2, (rh - INPUT) // 2
        x = ((r[oy:oy + INPUT, ox:ox + INPUT].astype(np.float32) / 255.0) - MEAN) / STD
        v = sess.run(None, {inp: np.transpose(x, (2, 0, 1))[None, ...]})[0].reshape(-1)
        return v / np.linalg.norm(v)

    embs = {}
    for t in FEATURE_TYPES:
        d = CROPS / t
        if not d.is_dir():
            continue
        vecs = [embed(d / f) for f in sorted(os.listdir(d))
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        vecs = [v for v in vecs if v is not None]
        if vecs:
            embs[t] = vecs

    correct, wrong = [], []
    for t, items in embs.items():
        for i, q in enumerate(items):
            protos = {tt: np.mean([v for j, v in enumerate(vv) if not (tt == t and j == i)],
                                  axis=0)
                      for tt, vv in embs.items()
                      if [v for j, v in enumerate(vv) if not (tt == t and j == i)]}
            if len(protos) < 2:
                continue
            scored = sorted(((float(q @ p), tt) for tt, p in protos.items()), reverse=True)
            gap = scored[0][0] - scored[1][0]
            (correct if scored[0][1] == t else wrong).append((gap, scored[0][0]))

    print(f"\n=== Bundled-crop baseline (the original calibration set) ===")
    g = np.array([c[0] for c in correct])
    print(f"  correct answers n={len(g)}: min gap {g.min():.3f}, median {np.median(g):.3f}")
    for gap, score in wrong:
        rejected = " (already rejected by the score check)" if score < SIMILARITY_THRESHOLD \
            else ""
        print(f"  error: score {score:.3f} gap {gap:.3f}{rejected}")
    print("  This set contains no near-ties, which is why it cannot choose a margin:")
    print("  every value from 0.02 up scores identically on it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="a pulled app database; default pulls from the device")
    ap.add_argument("--crops", action="store_true",
                    help="also run leave-one-out over the bundled reference crops")
    args = ap.parse_args()

    tmp = None
    try:
        if args.db:
            db = Path(args.db)
            if not db.exists():
                sys.exit(f"{db} not found")
        else:
            tmp = tempfile.mkdtemp(prefix="margin_study_")
            db = pull_database(tmp)

        rows = load_history(db)
        band_spot, confirmed = report_sampling(rows)
        recommended = analyse(rows)

        if args.crops:
            crop_baseline()

        print("\n=== Verdict ===")
        print(f"  shipping margin {MARGIN_THRESHOLD:.3f}"
              f" (= ThresholdTuner.MARGIN_MIN, a floor rather than a fit)")
        if recommended is None:
            print("  not enough evidence to move it yet.")
        elif band_spot < MIN_BAND_SPOT_CHECKS:
            print(f"  measurement suggests ~{recommended:.3f}, but only {band_spot} of the"
                  f" confirmations in the\n  {SIMILARITY_THRESHOLD:.2f}-{HIGH_CONFIDENCE:.2f}"
                  f" band came from a spot check, and that is the band the errors are in."
                  f"\n  The rest were chosen by the operator because they looked wrong, so the"
                  f"\n  correct-answer floor above is optimistic. Provisional - re-run later.")
        else:
            print(f"  fit from {confirmed} confirmations, {band_spot} of them unbiased samples"
                  f"\n  inside the band that matters: set the margin to {recommended:.3f}.")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
