# AI-Powered Mobile Quality Inspection for Manufacturing

Final-year project #93. An Android tablet app that inspects manufactured parts for correct
hardware — bolts, screws, studs, rivets and the holes they sit in — aimed at SMEs who
currently do this by eye and record it on paper.

It answers three questions about a photographed part:

1. **What hardware is on it?** — every fastener is detected and classified.
2. **Is anything missing, extra, or in the wrong place?** — detections are matched against a
   reference layout the operator defined, so a report names *which* fastener is wrong and
   where it sits on the part.
3. **Can this be filed?** — a pass/fail verdict and a shareable PDF report.

**Fully offline.** There is no network code anywhere in the app and no `INTERNET`
permission; `CAMERA` is the only permission requested. All inference runs on-device.

---

## How it works

Two decoupled stages. Stage 1 finds *where* things are, stage 2 decides *what* they are.

```
photo ──> [1] classical CV detection ──> crops ──> [2] few-shot classifier ──> types
                (OpenCV, no ML)                        (DINOv2-S, ONNX)
                                                                │
   reference layout ────> [3] positional check <────────────────┘
   (operator-labelled)      (distance-network alignment)
                                    │
                                    v
                        verdict + PDF report
```

**Stage 1 — detection (`ml`/`HardwareFeatureExtractor`).** Pure OpenCV, no ML. Grayscale →
glare detection (black-hat if >5% blown out) → CLAHE → bilateral filter whose diameter is
chosen from a Laplacian-variance noise estimate → Otsu-adaptive Canny → dilate/close →
contours → geometry filters → nested-box removal → proximity merge. Every step adapts to the
photo rather than using one fixed setting, because a setting tuned for a dim noisy photo
destroys detail in a bright clean one.

**Stage 2 — classification (`ml/FeatureEmbedder`, `ml/ReferenceLibrary`).** A **DINOv2-S**
ONNX encoder (int8, 24 MB) turns each crop into a 384-d embedding, compared by cosine
similarity against one prototype per type. A prototype is the mean of that type's
L2-normalised examples, so a new type needs a handful of labelled crops, not a retraining
run. Classification is by *comparison*, not by a fixed label set — which is what lets the
same pipeline handle a fastener the model was never trained on.

> Was CLIP ViT-B/32 until commit `c3c3a76`. CLIP squeezed all five types into a 0.06-wide
> similarity band. Benchmark: `python/compare_encoders.py`.

**Stage 3 — localisation (`ml/FeatureLocalisation`, `ml/LayoutMatcher`, `ml/DistanceNetwork`).**
Counting answers "how many bolt heads"; localisation answers "which one is missing, and where".
Detections are registered against the operator's labelled reference layout, then each expected
slot is reported as `PRESENT`, `MISSING`, `DISPLACED` or `WRONG_TYPE`. This is what catches a
*swap* — two screws expected, two found, one in the wrong hole — which count comparison cannot
see at all.

Registration works on the **network of separations** between features rather than on positions:
a distance matrix describes the part, not the photograph, since it survives shift and rotation
and scales by one common factor. A correspondence set is grown by mutual agreement — each new
feature must match its expected separation from *every* member already accepted — and a
transform is then fitted by least squares over that set and verified by inlier count.
Separations are invariant to reflection, so the network proposes and the transform disposes.

The transform is **affine** where the layout can afford it (5+ slots) and a proper similarity
below that. Measured over 18 RANSAC-verified asset pairs, a rigid transform leaves a p95
residual of 0.141 of the layout's spread against affine's 0.087 — the leftover is the
anisotropic foreshortening of a camera that is not square to the part, which no threshold can
tell apart from a misplaced fastener. Affine costs one more inlier before an alignment counts
as evidence, which is why small layouts keep the poorer model.

**Capture gating (`ml/AlignmentGate`).** The camera will not fire until the frame actually
shows the reference area, at the right distance, held steady for three seconds. Each frame is
scored against the reference photo by the same encoder; a "scale probe" (re-scoring the
frame's centre crop) separates *right area, wrong distance* from correct framing and tells the
operator to move closer or back off. Without this, a badly framed photo produces a confident
and wrong verdict.

**Supervised learning loop.** Nothing enters the library without a human — there is no
pseudo-labelling, deliberately. Every detection is logged; `ActiveLearningPolicy` picks ≤8
crops to ask about (unknowns nearest the decision boundary, plus one random *spot check* drawn
uniformly from **all** accepted predictions — gating that on high confidence exempted the band
holding 88 of 141 acceptances and every known silent error); the inspector labels them; the crop joins that type's prototype
immediately; and `ThresholdTuner` re-fits the similarity/margin thresholds once a part has
20+ confirmations. The spot check is load-bearing: without it every label would come from
crops the classifier already flagged, so the tuner would be fitting on its own failures.

---

## Current status

Legend: ✅ done · 🟡 partial · ❌ not started

### Against the project brief

| Brief goal | State | Notes |
|---|---|---|
| Inspect bolts, rivets, screws, studs, holes | ✅ | Five feature types shipped |
| **Nuts** (named in the brief) | ❌ | No `nut` type exists |
| Localisation — *which* fastener is wrong and where | ✅ | Slot-level `PRESENT`/`MISSING`/`DISPLACED`/`WRONG_TYPE` |
| Alignment checking | ✅ | Real, via the positional check + capture gate. *The 1.8s "ALIGNED (98%)" mock is gone.* |
| Dimensions (pixel→mm) | 🟡 | Stage A only — see below |
| Variable lighting robustness | 🟡 | Implemented adaptively; the `Edge Lab` tab now makes it *observable* per photograph, but still **never measured** across a lighting set |
| Few-shot learning from rare examples | ✅ | 97.1% leave-one-out on 34 curated crops |
| Automatic documentation | ✅ | PDF (platform `PdfDocument`, no third-party library), plus JSON + CSV export, all via share intent |
| **Barcode → report linking** | ✅ | Reports carry a QR pointer and **History > SCAN reads one back** to open that inspection (ZXing, both directions). Linking a scanned code to a *part* is still not done — parts are chosen from the registry |
| Camera capture | ✅ | CameraX, live-gated, auto-shutter on a 3s hold |
| Evaluation / precision & recall | 🟡 | Component studies exist; no app-level per-type P/R |

### Done

- **Two-stage pipeline** — adaptive OpenCV detection + DINOv2-S few-shot classification.
- **Camera capture** with live alignment gating, three-second hold and automatic shutter.
- **Alignment gate** — embedding similarity + scale probe for distance + motion gate, all
  fitted against 252 labelled asset pairs with RANSAC-verified ground truth.
- **Positional/slot check** with one-to-one inlier counting, a model-dependent inlier floor,
  and outright refusal of layouts under 4 slots (a check that structurally cannot fail is
  worse than none). Verified end-to-end on a 12-slot layout: 11 of 12 positions matched with
  residuals of 0.0006–0.023 and 2.9% network distortion.
- **Severity scale** — every type is `CRITICAL`, `ADVISORY` or `IGNORED`, deciding whether it
  reaches the quantitative table, the trend notes, or neither. Snapshotted at save time so a
  later re-grade cannot rewrite a filed report.
- **Supervised active-learning loop** — review queue, spot checks, per-part threshold tuning.
- **Source-resolution crops** (`ml/SourceCropper`) — crops are cut from the original 4032×3024
  photo via `BitmapRegionDecoder`, not the 1280px working copy, with proportional padding.
- **Room database v11** with real migrations (no destructive fallback on new schema changes),
  foreign keys on every child table and indices on the columns actually queried. The delete
  rule is deliberately split: `FeatureResult` and `SlotResult` cascade because they *are* the
  report, while `ClassificationEvent` is `SET NULL` because it is human-labelled training data
  that outlives the report it was attached to — cascading there would cost the classifier its
  history for a part as a side effect of housekeeping.
- **PDF reporting** + share intent, and per-part advisory trend notes.
- **Immutable report identity** — every filed inspection carries a uuid, an operator, and a
  SHA-256 digest over its own canonical form, so a later edit is detectable and a re-inspection
  appends rather than overwrites. Legacy rows are backfilled with identity but deliberately
  left *unsealed* rather than back-dated, and the report says so.
- **Machine-readable export** — full record as JSON, roll-up row as CSV, beside the PDF and out
  through the same FileProvider. Previously the PDF was the only artefact, so the numbers had
  to be retyped off a page to reach a spreadsheet.
- **Traceability QR** on every report — a fixed 26-byte pointer (record id, verdict, digest
  prefix) as Base32, rendered with ZXing. It replaces a footer that read "Factory QC Manager
  digital signature valid." on every report the app had ever produced, which was decoration
  claiming an assurance that did not exist.
- **Report code on screen and on paper** — the traceability card on the report screen shows
  the same QR the PDF carries, so a code can be scanned straight off one tablet by another
  without exporting and opening a file first.
- **Report code lookup** — `History > SCAN` reads the QR off a printed or on-screen report,
  resolves the uuid against the local database and opens that inspection, warning first if the
  printout's digest no longer matches the filed record. A code from another device resolves to
  nothing and says so, because the code identifies a record rather than containing one.
- **Edge Method Lab** (`Edge Lab` tab) — a debugging screen that runs one photograph through
  nine edge/feature operators at the same working resolution and scores every map through the
  detector's own geometry filter: Canny at fixed thresholds, Canny with Otsu-derived
  thresholds, Sobel and Scharr gradient magnitude, Laplacian of Gaussian, morphological
  gradient, adaptive Gaussian threshold, a CLAHE+bilateral+Canny partial, and the shipped
  hybrid itself — called through `HardwareFeatureExtractor.computeEdges` rather than copied,
  so the screen cannot drift from the code it is evidence about. The edge stage was previously
  invisible: an inspection showed only the boxes that came out the far end, where a missed
  fastener and one rejected on aspect ratio look identical.
- **Hybrid v2/v3/v4** — three tuned replacements for the detector's edge stage, found by
  `python/hybrid_v2_study.py` and shown in the Edge Lab beside the stack they would replace.
  The shipped stack was exactly right on four calibration photographs and wildly over-
  detecting on two (`stud1.jpg` 48 against a target of 4), because the adaptive bilateral step
  buckets Laplacian variance and so gave the *least* smoothing to the two sharp,
  full-resolution photographs that needed the most. All three variants hit all six calibration
  counts exactly — confirmed on-device, not just in the study — and all three are **cheaper**
  than the stack they replace, because pinning the bilateral diameter drops the full-image
  CV_64F Laplacian probe that choosing one adaptively costs.

  | | exact of 26 | misses | extras | group stability | edge cost |
  |---|---|---|---|---|---|
  | shipped | 9 | **0** | 254 | 0.638 | — |
  | v2 bilateral 9, Otsu x1.1 | 12 | 5 | 32 | 0.261 | −15% |
  | v3 bilateral 5, gauss 7, Otsu x1.2 | 12 | 3 | 33 | 0.178 | **−51%** |
  | v4 bilateral 7, L2 gradient | 16 | 3 | 15 | **0.125** | −30% |
  | v5 v3 + adaptiveThreshold mask | 12 | 4 | 33 | 0.156 | +35% |
  | v6 bilateral 5, Otsu x1.1, L2 | **20** | 3 | **13** | 0.182 | −26% |

  *exact / misses / extras* are scored against `python/detection_ground_truth.py` — a
  hand-made count for every bundled photograph, which replaced the six calibration counts once
  it became clear the proxies could not separate the variants. Three findings from making it:
  `screws.jpg` has four screws, not the five the calibration target said (the fifth box is a
  scratch); the shipped stack's worst failure is not a calibration image at all (`three.jpg`:
  three rivets, twelve boxes); and the three misses no variant recovers are all low-contrast
  heads barely darker than the plate — a limit of a global threshold, not a tuning gap.
  **v4 was far ahead of v2/v3/v5 all along** and the proxies could not show it. v6 is exact on
  all fourteen stud and screw photographs, at the cost of a narrower parameter plateau and worse
  group stability than v4; the shipped stack never misses a fastener, it buries them.

  *photo drift* is the calibration error under brightness, gamma, noise and JPEG variation;
  *param drift* the same after nudging each parameter one step; *group stability* the
  disagreement between photographs the alignment gate treats as interchangeable, which is what
  actually causes a false FAIL. The clearest single finding: every one of the 149 exact
  configurations sets Canny's low threshold to 0.4x the high one, not production's 0.5.

  On **adaptive thresholding**, the study answers both senses of the phrase and they point
  opposite ways. Deriving Canny's bounds adaptively — Triangle, median- or mean-anchored
  auto-Canny, or Otsu computed per tile rather than globally — is *worse*: tiled Otsu can
  reach photo drift 0, but every locally-adaptive derivation costs group stability badly
  (0.48–0.57 against v3's 0.18), because a threshold that tracks local content makes the count
  depend on what else is in the frame. Using `cv2.adaptiveThreshold` as a *mask* over the Canny
  map is *better* — it is a poor edge detector alone (199 candidates where the shipped stack
  finds 42) but an effective speckle filter, and that is what v5 is — on the proxies. Against
  ground truth it is no better than v3. Three further discriminators tried against ground
  truth — a component solidity filter, a morphological open, and Canny bounds from a gradient
  percentile — were all inert; none appears in any top-ranked configuration.

  **Not adopted by default** — the detector runs the original stack until one is chosen in
  the Edge Lab (USE FOR INSPECTIONS).
- **The operator declares what is on the part.** Reference setup is: choose the photo, enter
  how many of each feature type the part has, confirm — and confirm stays disabled until both
  are there. The counts are stored per part (`ExpectedFeature`, DB v14) and are the expected
  side of every inspection's count comparison, read verbatim — no detector decides them. The
  photo is detected once, on the setup screen, for what counts cannot give: each feature's
  position and box size, shown on the preview and saved as the part's layout so the positional
  check and the dimensional measurement work as before. The classifier names each box (nearest
  prototype), so there is no labelling dialog; if it finds fewer positions than were declared
  the screen says so, and the counts still stand. The two presets ship declared:
  `CHASSIS-8821-B` one hole, two studs, one screw; `PART-CHASSIS-A03` one hole, two studs, one
  bolt head.
- **347 unit tests**, all passing.
- **Reproducible studies** in `python/` — localisation tolerance and transform model
  (`localisation_tolerance_study.py`), classifier margin fitted from real inspection history
  (`margin_study.py`), the edge-stage tuning search behind Hybrid v2/v3/v4
  (`hybrid_v2_study.py`) with its hand-made per-photograph ground truth
  (`detection_ground_truth.py`), encoder comparison, detector A/B harness, scale
  probe study, and objective same-area pair ground truth.

### In progress / partial

| Area | What exists | What's missing |
|---|---|---|
| **Dimensions** | `ScaleCalibration` finds a printed square marker geometrically; `FeatureMeasurement` reports mm sizes when one is in frame | Stage B: a single uniform mm/px is applied everywhere, valid only for a perpendicular camera and coplanar features. Homography rectification from the marker's four corners is designed but not built |
| **Evaluation** | Encoder LOO, detector stability A/B, gate threshold sweeps | No per-type precision/recall for the shipped app. Brief Phase 4 is effectively empty |
| **Lighting robustness** | Glare suppression and adaptive denoising implemented | Never quantified, despite matched pairs sitting in assets (`whole.jpg`/`whole_light.jpg`, `screws_flash.jpg`/`screws_none.jpg`) |
| **Layout drift** | Fixed for operator-labelled references | References made before labelling, or via **SKIP SETUP**, still re-classify and still drift |

### Not started

- **Barcode → *part* linking.** `ReportScanFragment` reads report codes, but nothing maps a
  scanned label to a part. Parts are deliberately chosen from the registry rather than typed —
  free text produced silent near-duplicates, each with its own reference photo and history — so
  a scan should *match* an existing part, never create one.
- **Nut** as a feature type.
- **Balanced/macro accuracy** in `ThresholdTuner` — it optimises plain accuracy, which is
  misleading on imbalanced classes, a challenge the brief names explicitly.
- **Linear probe** instead of nearest-centroid classification.
- **Type rename/delete** — types can only be re-graded. `FeatureResult` stores the display
  name, so a rename would orphan a part's advisory baseline.
- **`removeExample()`** — correcting a crop twice leaves the stale label in the wrong
  prototype permanently.

### Known bugs

- **"USE BUILT-IN PRESET" points at a missing asset.** `ReferenceSetupFragment` loads
  `chassis_hardware_preset.jpg`, which is not in `assets/`. It fails silently → zero
  detections → a spurious "INSPECTION PASSED".
- **Symmetric layouts cannot be pinned.** Slot identity is geometric, so four bolt heads on a
  perfect square have several equally good alignments. The count of missing positions and the
  verdict stay correct; *which* slot is named is arbitrary.

### Suggested order of work

**Priority rule: software proof-of-concepts first.** Work that can be developed, measured and
written up from the bundled assets and the existing `python/` harnesses outranks work that
needs a physical rig — a device, a printed marker, a barcode label — to make progress on. The
hardware items are not less important to the brief; they are less able to produce evidence for
the report right now, and they block on things outside the repo.

*Software, measurable from assets:*

1. **Evaluation set + per-type precision/recall** (Phase 4) — the single biggest gap in the
   report. Hundreds of labelled crops already exist on-device, and `compare_encoders.py`
   already mirrors `ReferenceLibrary`'s scoring maths. Confusion matrix per type.
2. **Lighting study** using matched pairs already in `assets/chassis_images/` — turns
   "implemented, never measured" into a result with no new photography.
3. **Balanced accuracy in the tuner** — small change, named brief challenge, and it makes
   every subsequent tuning figure defensible.
4. **Linear probe** vs. the nearest-centroid baseline on the same crops.
5. **Fix the preset-button asset bug** — it is a silent false PASS.
6. **`removeExample()`** + crop→type tracking.
7. **Dimensions Stage B** — homography rectification, removing the coplanarity assumption.

*Hardware-dependent, deprioritised:*

8. **Barcode → part linking** — needs printed labels and a device; produces little report evidence. The decoder already ships, and must match registered parts rather than create them.
9. **`nut` type** — needs photographs that do not exist in `assets/` yet.
10. **On-device performance profiling** — latency, memory, battery on real hardware.

---

## Building and running

**Prerequisites:** Android Studio, and a device or emulator on API 24+.

```bash
JAVA_HOME="/c/Program Files/Android/Android Studio/jbr" ./gradlew assembleDebug
```

Run the unit tests:

```bash
./gradlew testDebugUnitTest
```

In Android Studio: **Open** the project directory and let it resolve the Gradle sync.

Notes:

- ABI filters limit the build to `x86_64` and `arm64-v8a`. OpenCV and ONNX Runtime ship
  native libraries for four ABIs, which pushed the debug APK past 350 MB.
- No API keys or `.env` file are needed. The app makes no network calls.
- The ONNX model is kept uncompressed (`noCompress += "onnx"`) so ONNX Runtime can memory-map
  it from the APK rather than decompressing it into RAM.

### Using the app

1. **Home** → enter or pick a Part ID.
2. **Reference setup** — choose a reference photo, then label every detected crop. Those
   labels become both the prototype library and the expected slot layout, so this step is not
   optional decoration; a reference saved without it defines far less than it appears to.
3. **Inspection** — point the camera at the part. The traffic light turns green once framing,
   distance and steadiness are all within tolerance, and the shutter fires automatically.
4. **Results** — counts per type, positional findings, advisory trend notes, verdict.
5. **Report** — generate and share a PDF.

---

## Repository layout

```
app/src/main/java/com/example/
  ml/                    detection, embedding, matching, learning
    HardwareFeatureExtractor  stage 1: OpenCV detection pipeline
    FeatureEmbedder           stage 2: DINOv2-S ONNX encoder
    ReferenceLibrary          prototypes, cosine classification
    FeatureLocalisation       stage 3: slot-level findings
    LayoutMatcher             consensus growth + similarity-transform registration
    DistanceNetwork           pairwise separations, correspondence growth
    AlignmentGate             live capture gating
    ActiveLearningPolicy      review queue selection
    ThresholdTuner            per-part threshold fitting
    ScaleCalibration          marker-based pixel→mm
    SourceCropper             full-resolution crop extraction
  ui/fragment/           one fragment per screen (Navigation Component)
  data/                  Room entities, DAOs, migrations
  report/                PDF layout, advisory trend notes, export + digest, QR pointer
app/src/main/assets/
  models/                feature_encoder.onnx (DINOv2-S, int8)
  reference_crops/       bundled seed crops, one folder per type
  chassis_images/        test photographs
python/                  offline studies and the ONNX export script
app/src/test/            254 unit tests
```

Note: `HardwareFeatureExtractor` currently lives under `ui/fragment/` despite being pure
CV/ML with no UI. It belongs in `ml/`.

---

## Where state lives

| What | Where |
|---|---|
| Inspections, feature results, slot results, classification events | Room `shop_quality_inspection_db`, version 9 |
| Prototypes | `filesDir/clip_lib/<partId>.json` |
| Fitted thresholds | `filesDir/thresholds/<partId>.json` |
| Feature type catalog + severity | `filesDir/feature_types.json` (**global**, not per-part) |
| Reference layouts | `filesDir/layouts/<partId>.json` |
| Training crops | `filesDir/learning_crops/<partId>/*.png` (pruned to a disk budget) |
| Reports, captures | `getExternalFilesDir(...)` |

Learning is **per-part** except the type catalog: labels on one part do nothing for another,
and each part must independently reach 20 confirmations before threshold tuning engages.

### Encoder identity is load-bearing

`FeatureEmbedder.ENCODER_ID` is `dinov2-s14-cls-384-tightcrop`. Prototypes only mean anything
against queries from the same encoder, and cosine comparison silently truncates to the shorter
vector — so a stale library returns *confident nonsense* rather than failing. Libraries,
fitted thresholds and logged events all carry the encoder id and are discarded on mismatch.
Any change to the model **or to crop framing** must bump this id.

---

## Technical notes for the final report

Material here is written to be lifted into the dissertation. Each item states the problem, the
alternatives tried, what shipped, and — where one exists — the measurement that settled it.
**Provenance is marked**, because a figure quoted in a viva needs to be re-derivable.

### Measured results

| # | Result | Value | Source | Status |
|---|---|---|---|---|
| M1 | DINOv2-S vs CLIP ViT-B/32, leave-one-out accuracy over 34 curated crops | **97.1%** vs **79.4%** | `python/compare_encoders.py` | Carried from prior study, not re-run today |
| M2 | int8 export cost to CLIP's accuracy | ~9 points | `python/compare_encoders.py` | Carried |
| M3 | Alignment gate threshold sweep, 252 labelled pairs | 0.70 → 5 false accepts, 0 false rejects; 0.74 → 2 FA but 5 FR | `python/scale_probe_study.py` | Carried, tabulated in `AlignmentGate` javadoc |
| M4 | Scale probe separation: crop-in gain, correctly framed vs too far | mean **−0.070** vs **+0.156** | `python/scale_probe_study.py` | Carried |
| M5 | Crop framing: feature occupancy before vs after proportional padding | 6–44% → **52–58%** across the size range | Measured on bundled crop set | Carried |
| M6 | On-device crop resolution, working copy vs source decode | ~38×30 px → **120–169 px** | Measured on-device | Carried |
| M7 | Unit test suite | **151 tests, 0 failures** | `./gradlew testDebugUnitTest --rerun` | **Re-verified** |
| M9 | Registration residual, similarity vs affine, 18 RANSAC-verified asset pairs (158 correspondences) | p95 **0.141** vs **0.087** of layout spread; pairs registering at tolerance 0.12: **16/18** vs **18/18** | `python/localisation_tolerance_study.py` | **New** |
| M10 | Is a rigid transform the right model? Pairwise distance-ratio spread across the same pairs (1.0 = rigid) | median **×1.28**, p90 ×1.89, max ×2.80 | `python/localisation_tolerance_study.py` | **New** |
| M11 | Correct match is nearer than any wrong slot (what one-to-one assignment resolves on) | **100%** of 158 correspondences, both models | `python/localisation_tolerance_study.py` | **New** |
| M12 | Silent misclassification rate among *accepted* predictions, real inspection history | **4 of 35** confirmed — every one `stud` where the operator said `screw` | `python/margin_study.py` | **New** |
| M13 | Runner-up gap separates silent errors from correct calls | errors **0.020/0.039/0.054/0.064**, correct min **0.087** → separable, midpoint **~0.075** | `python/margin_study.py` | **New**, provisional — only 1 unbiased sample in the band |
| M14 | Spot-check coverage of the 0.74–0.88 confidence band before the fix | **0 of 88** accepted predictions ever sampled; all known errors lived there. After the fix, spot checks reach it | `python/margin_study.py` | **New** |
| M15 | Localisation end-to-end on a 12-slot layout (`whole screws1.jpg` → `whole screws2.jpg`, affine path) | **11 of 12 positions matched**, residuals 0.0006–0.023, network distortion **2.9%** of the part | On-device, `LayoutMatcher` log | **New** |
| M8 | Seed crop set composition | bolt_head 6, hole 9, rivet_head 2, screw 7, stud 10 (=34), unknown 3 | `assets/reference_crops/` | **Re-verified** |

**Gap to close:** there is no per-type precision/recall for the shipped end-to-end system.
M1 measures the *encoder* on curated crops, which is not the same claim. Say so explicitly in
the report rather than letting M1 stand in for system accuracy.

### Design decisions worth a paragraph each

**Two decoupled stages rather than an end-to-end detector.** A YOLO-style detector would need
hundreds of labelled boxes per part and a retraining run whenever an SME introduces a new
fastener. Splitting *where* (classical CV) from *what* (few-shot embedding comparison) means a
new type costs a handful of labelled crops and takes effect on the next crop, with no
gradients computed on-device. The trade is that stage 1 has no notion of the objects it is
finding, so recall is bounded by contour geometry.

**Classification by comparison, not by a fixed label set.** A fixed-head classifier cannot
represent a type it was never trained on. Cosine similarity against per-type prototypes lets
the operator define the label space at setup time. The honest limit — worth stating plainly —
is that "generality" here is a claim about *coverage effort*, not about the model recognising
new categories unaided.

**Encoder identity as a correctness invariant.** `cosine()` silently truncates to the shorter
vector, so a library built with one encoder and queried with another returns confident
nonsense rather than an error. Every persisted artefact (libraries, thresholds, logged events)
carries `ENCODER_ID` and is discarded on mismatch. Crop framing counts as part of that
identity, because changing it moves every embedding — hence the `-tightcrop` suffix. This is a
good example of a failure mode that is invisible at runtime and only preventable by design.

**No pseudo-labelling — rejected twice, on specific grounds.** On a mean prototype a
high-confidence crop is by construction nearly parallel to that prototype, so adding it is
close to a no-op, while diluting the rare human corrections that carry real signal. Confidence
gating also selects *against* the novel views that would broaden coverage. And it must never
feed threshold tuning, which would be circular. Note this contradicts the mid-year report,
which described automatic pseudo-labelling above a 0.95 upper threshold — see Corrections.

**Spot checks are load-bearing, not decoration.** The review queue selects crops near the
decision boundary. If every label came from crops the classifier already flagged, the
threshold tuner would be fitting on its own failures. One random spot check of an *accepted*
prediction per round breaks that loop. This is not hypothetical: the margin once collapsed
to 0 without it.

**`MIN_ACCURACY` exists because beating a bad baseline is not evidence.** A fit once scored
34% against a 21% baseline and was adopted. A relative-improvement gate alone is insufficient;
an absolute floor is also required.

**Positional registration must ignore type.** Consensus growth, inlier counting and slot
assignment compare position only. This is what lets a stud in a screw's hole read as *that
slot, wrong hardware* rather than as a missing screw plus an unexpected stud. Type agreement
only orders which seeds are tried. Requiring type agreement to seed made alignment starve
exactly as substitutions grew — the failure scaled with the defect it was meant to catch.

**Agreeing on distances is necessary for a match, not sufficient.** A detection displaced
along the line to its neighbour changes that separation by its full error; one displaced
across it barely changes it at all. So the network's agreement test must run *tighter* than
the positional gate it feeds, or it becomes the more permissive of the two — and it is the
one deciding what the transform is fitted over. A fastener a fifth out of position was
admitted to the consensus at the edge of a 12% threshold, least squares stretched to meet it,
and the defect was reported as correctly fitted. Conservative growth, generous verification.

**Type must never break an alignment tie.** Four fasteners at the corners of a parallelogram
map onto themselves under a half turn, so "every fastener swapped" and "photographed upside
down" have identical positional support. Preferring the type-consistent alignment took the
second reading every time, reporting a fully swapped part as all present. Ties go to the
smallest rotation instead. A good worked example of a plausible-sounding heuristic that was
guaranteed to hide the exact defect it sat next to.

**A verification that cannot fail is worse than none.** Layouts under 4 slots are refused
outright: with 3 slots, the only alignment clearing a 3-inlier floor is the one where all 3
are present, so the check could report "all verified" and nothing else. Likewise the inlier
floor of `max(3, half the slots)` — two correspondences supply exactly the 4 constraints a
similarity transform needs, so a floor of 2 is a tautology, not a threshold.

**Honest refusal over confident output.** Three places return "unavailable" rather than a
guess: unclassifiable crops become `unknown`, un-registerable layouts report no positional
claim, and an inspection with no real reference is `UNVERIFIED` rather than `PASSED`. For a
quality-inspection tool a false PASS is the costly error, so the system is biased toward
visibly declining to answer.

### Rejected alternatives (with the evidence)

These are worth a section of their own — a rejected approach with a measured reason is
dissertation material; a bare decision is not.

| Approach | Why rejected | Evidence |
|---|---|---|
| ORB keypoints + RANSAC homography for the alignment gate | The chassis plate is near-featureless, so keypoints land on the hardware — exactly what changes between reference and inspection | 8 of ~1000 keypoints survived on a true match vs 2 on a completely different area: signal absent, not weak |
| Detected-feature geometry for the gate | Detector merges clustered fasteners into one box; a different area scored closer than a genuine match | Measured on bundled assets |
| Median detected-blob diameter as a distance cue | Wanders by 3× across crops of one image — surviving contours are surface grain, not hardware | `scale_probe_study.py` |
| Texture spatial-frequency centroid as a distance cue | Flattens at close range where sensor noise dominates; moves the wrong way on `corner.jpg` | `scale_probe_study.py` |
| CLIP ViT-B/32 encoder | Compressed all five types into a 0.06-wide similarity band | M1 |
| Fixed 40 px crop padding | Made framing a function of shooting distance; encoder described background for small features | M5 |
| Raising the gate threshold to fix shooting distance | Takes genuine same-distance shots with it, and cannot tell the operator which way to move | M3 |

### Threats to validity — state these before an examiner finds them

| Limitation | Detail | Status |
|---|---|---|
| No system-level accuracy figure | M1 measures the encoder on curated crops, not the shipped pipeline on real photos | Open — priority 1 |
| Small evaluation set | 34 crops, and `rivet_head` has only 2 — leave-one-out on 2 examples is close to meaningless for that class | Open; report per-class counts alongside any accuracy figure |
| Lighting robustness unquantified | Adaptive glare/noise handling is implemented but never measured, despite matched pairs sitting in assets | Open — priority 2 |
| Tuner metric is plain accuracy | Misleading under class imbalance, which this dataset has | Open — priority 3 |
| Gate ground truth is partly unprovable | RANSAC verification yields trustworthy positives only; the one pair the gate was built for (`screws.jpg`/`screws_test.jpg`) cannot be certified by it | Documented in `pair_ground_truth.py` |
| Symmetric layouts | Slot identity is geometric; verdict and missing-count hold, named slot is arbitrary | Known limit; needs an asymmetric fiducial |
| Dimensions assume coplanarity | Single uniform mm/px; tilt biases measurements by camera pose | Stage B designed, not built |
| Emulator-only performance | No latency, memory or battery figures from real hardware | Open |

### Corrections to the mid-year report

The mid-year technical report describes the system as it was, and the following claims are no
longer accurate. Revise before building on them.

| Prior claim | Current position |
|---|---|
| "OpenAI's CLIP model was chosen" as the encoder | Replaced by DINOv2-S at commit `c3c3a76`, on the measured grounds in M1 |
| Automatic pseudo-labelling above a ≈0.95 upper threshold expands the library unattended | **Removed deliberately.** Nothing enters the library without a human; the reasoning is in the design-decisions section above |
| Dual thresholds of ≈0.95 / ≈0.80 | Shipped values are similarity `0.74`, margin `0.02`, high-confidence `0.88`, and they are re-fitted per part by `ThresholdTuner` |
| Non-critical artifacts go to a "filter list" | Generalised into a three-level severity scale (`CRITICAL` / `ADVISORY` / `IGNORED`) |
| Inspection outcome is determined by count comparison against the manifest | Count comparison **plus** a positional check that catches swaps and displacement, which counting cannot see |

Two items are ready-made narrative arcs: the mid-year report named "finding a good balance
point [for the confidence line]" as an open question carried into the second half — the
threshold sweeps in M3 and the per-part `ThresholdTuner` answer it. And it predicted that
similar-looking types "may end up closer together in embedding space"; M1 quantifies exactly
that as CLIP's 0.06-wide band, and the encoder swap resolves it.

---

## Further documentation

`CLAUDE.md` in this directory holds the detailed engineering context: calibrated constants and
their provenance, decisions that should not be relitigated, and the specific traps in the Room
migrations. Read it before changing thresholds or the detection geometry.
