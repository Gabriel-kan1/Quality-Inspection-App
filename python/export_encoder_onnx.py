"""
Exports the DINOv2-small vision encoder to ONNX for on-device use by
com.example.ml.FeatureEmbedder.

Two deliberate choices make the Android side unambiguous:

  1. The wrapper returns ONLY the CLS token, as a single [1, 384] output named
     "image_embeds". A stock HF export exposes both last_hidden_state and
     pooler_output, and FeatureEmbedder.pickOutputName() would silently pick
     whichever matched first -- possibly not the one that was benchmarked.
     One output, one name, no guessing.

  2. Fixed [1, 3, 224, 224] input shape (no dynamic axes). FeatureEmbedder
     always feeds exactly that, and it lets the 37x37 -> 16x16 position-embedding
     interpolation be constant-folded into the graph instead of computed per run.

Emits both fp32 and int8 so they can be accuracy-compared before shipping --
int8 cost the previous CLIP encoder ~9 points, so it is not assumed to be free.

Usage:  python python/export_encoder_onnx.py
"""

from pathlib import Path

import torch
from transformers import AutoModel

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "python" / "onnx_export"
MODEL_ID = "facebook/dinov2-small"
INPUT_SIZE = 224


class ClsTokenEncoder(torch.nn.Module):
    """DINOv2 backbone reduced to a single CLS-token embedding output."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values):
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state
        return hidden[:, 0]          # [B, 384] -- the CLS token


def export_fp32(path):
    backbone = AutoModel.from_pretrained(MODEL_ID).eval()
    model = ClsTokenEncoder(backbone).eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    with torch.no_grad():
        reference = model(dummy)
    print(f"  torch output shape {tuple(reference.shape)}")

    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    return dummy, reference


def quantize_int8(src, dst):
    """
    Dynamic int8, MatMul only.

    Conv is deliberately excluded. Quantising it emits a ConvInteger node, and the
    onnxruntime-android package has no kernel for that op -- the model then fails at
    session creation on device with ORT_NOT_IMPLEMENTED while loading fine on desktop
    onnxruntime, which does implement it. Desktop parity is not the bar here; the
    mobile operator set is.

    Cost of leaving it out is negligible: the excluded layer is the single patch-embedding
    conv (~226K of 22.1M parameters), so nearly all the weight still gets quantised.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )


def verify(path, dummy, reference):
    """Confirms the ONNX graph agrees with torch, and reports the I/O contract."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ins = [(i.name, i.shape) for i in sess.get_inputs()]
    outs = [(o.name, o.shape) for o in sess.get_outputs()]
    got = sess.run(None, {sess.get_inputs()[0].name: dummy.numpy()})[0]

    ref = reference.numpy()
    cos = float(np.dot(ref[0], got[0]) /
                (np.linalg.norm(ref[0]) * np.linalg.norm(got[0])))
    mib = path.stat().st_size / 1048576
    print(f"  {path.name:34s} {mib:6.1f} MiB  in={ins}  out={outs}")
    print(f"  {'':34s} cosine vs torch fp32 = {cos:.6f}")
    return cos


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32 = OUT_DIR / "dinov2_small_cls_fp32.onnx"
    int8 = OUT_DIR / "dinov2_small_cls_int8.onnx"

    print(f"Exporting {MODEL_ID} (CLS token) -> {fp32.name}")
    dummy, reference = export_fp32(fp32)
    verify(fp32, dummy, reference)

    print(f"\nQuantising (dynamic int8) -> {int8.name}")
    quantize_int8(fp32, int8)
    verify(int8, dummy, reference)

    print("\nBoth written to", OUT_DIR)
    print("Next: python python/compare_encoders.py  (benchmarks these against CLIP)")


if __name__ == "__main__":
    main()
