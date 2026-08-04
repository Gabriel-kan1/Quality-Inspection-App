FEATURE ENCODER — MODEL FILE
============================

    app/src/main/assets/models/feature_encoder.onnx     (23 MiB, int8)

Loaded by com.example.ml.FeatureEmbedder. The app still builds and runs without
it — classification falls back to "unknown" and FeatureEmbedder logs a warning.

WHAT IT IS
----------
DINOv2 ViT-S/14 (facebook/dinov2-small), vision only, reduced to its CLS token
and dynamically quantised to int8. Regenerate it with:

    python python/export_encoder_onnx.py

WHY NOT CLIP
------------
This app previously shipped an int8 CLIP ViT-B/32 vision encoder. It was
replaced because it measurably could not separate these five feature types.
Measured by python/compare_encoders.py — leave-one-out prototype classification
over the bundled reference crops:

                        top-1     class separation   median runner-up margin
    CLIP ViT-B/32 int8  79.4%     0.060              0.036
    DINOv2-S int8       97.1%     0.271              0.175

Separation = mean intra-class cosine minus mean inter-class cosine. CLIP fitted
all five types into a 0.06-wide cosine band, so the runner-up margin check had
to be set at 0.03 — right at the median of *correct* answers — which discarded
roughly half of them as "too close to call". DINOv2 gives ~5x the headroom.

Two other findings from that comparison:
  - int8 cost CLIP ~9 points of top-1 (88.2% fp32 -> 79.4% int8). It costs
    DINOv2 almost nothing (97.1% at both precisions), which is why shipping
    int8 here is safe. Do not assume that for a future encoder — measure it.
  - CLIP's text tower was never used (the Python prototype sets
    TEXT_PROMPT_WEIGHT = 0.0), so its one distinguishing capability was dead
    weight. Nothing is lost by moving to a vision-only encoder.

INPUT / OUTPUT CONTRACT (see FeatureEmbedder.java)
--------------------------------------------------
- Input : "pixel_values", float32, fixed shape [1, 3, 224, 224], NCHW,
          ImageNet-normalised (mean 0.485/0.456/0.406, std 0.229/0.224/0.225).
          NOT CLIP's normalisation constants — they differ.
- Output: "image_embeds", float32 [1, 384] — the CLS token.

The export deliberately emits a SINGLE output. A stock HuggingFace export
exposes both last_hidden_state and pooler_output, and the old
ClipEmbedder.pickOutputName() would have silently picked whichever matched
first — possibly not the one that was benchmarked. One output, one name.

IF YOU SWAP THE ENCODER AGAIN
-----------------------------
Prototypes are only comparable to query embeddings from the SAME encoder, and
cosine() tolerates mismatched dimensions by truncating — so a stale library
produces plausible-looking garbage rather than an error. Bump ENCODER_ID in
FeatureEmbedder; ReferenceLibrary tags every saved library with it and rejects
libraries written by a different encoder. Also re-run the threshold sweep:
ReferenceLibrary's defaults are calibrated per-encoder, not universal.

Keep noCompress "onnx" in app/build.gradle.kts so ONNX Runtime can memory-map it.
