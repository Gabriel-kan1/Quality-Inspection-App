CLIP IMAGE ENCODER — MODEL FILE GOES HERE
=========================================

The app expects an ONNX CLIP (ViT-B/32) VISION encoder at:

    app/src/main/assets/models/clip_image_encoder.onnx

It is loaded by com.example.ml.ClipEmbedder. The app will still build and run
without it — classification simply falls back to "unknown" until the file is
present (ClipEmbedder logs a warning).

WHY IT IS NOT COMMITTED
-----------------------
The model is ~45-350 MB depending on quantisation, which is too large to keep
in the repo. Drop it in locally (git ignores large binaries here) or wire a
Gradle download task.

HOW TO GET ONE (any CLIP ViT-B/32 vision export works)
------------------------------------------------------
The reference library is built on-device with THIS SAME encoder, so the model
only has to be self-consistent — it does NOT have to match the desktop
sentence-transformers output byte-for-byte. Options:

1. Prebuilt (easiest): download a CLIP ViT-B/32 vision ONNX, e.g. from
   Hugging Face "Xenova/clip-vit-base-patch32" (onnx/vision_model.onnx or a
   quantised variant), rename to clip_image_encoder.onnx, and place it here.

2. Export it yourself (Python, one-off):
       pip install optimum[exporters] onnx onnxruntime
       optimum-cli export onnx \
         --model openai/clip-vit-base-patch32 \
         --task feature-extraction clip_onnx/
   then use the vision encoder graph, optionally int8-quantise with
   onnxruntime.quantization, and copy it here as clip_image_encoder.onnx.

INPUT / OUTPUT CONTRACT (see ClipEmbedder.java)
-----------------------------------------------
- Input : one float32 tensor, shape [1, 3, 224, 224], NCHW, CLIP-normalised
          (mean 0.48145466/0.4578275/0.40821073, std 0.26862954/0.26130258/0.27577711).
          The input NAME is read dynamically from the model, so it can be
          "pixel_values", "input", etc.
- Output: the first output tensor, flattened to a 1-D embedding vector
          (512-d projected image_embeds OR 768-d pooled output — both fine).

Keep noCompress "onnx" in app/build.gradle.kts so ONNX Runtime can memory-map it.
