"""One-time export of visual-prompt YOLOE; no Python is used by the Rust runtime.

Install scripts/requirements-yoloe-export.txt in a separate environment first.
Exports are target-independent: replacing a target never requires re-exporting.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import ultralytics
from ultralytics import YOLOE


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, visuals):
        return self.model.predict(images, vpe=visuals, return_vpe=True)


class Detector(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, embeddings):
        predictions = self.model.predict(images, vpe=embeddings)
        # YOLOE-seg export returns (decoded boxes/scores/mask coefficients, prototypes).
        # One visual class: cx, cy, w, h, score; discard segmentation coefficients.
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        return predictions[:, :5, :].transpose(1, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yoloe-v8s-seg.pt")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("models/yoloe"))
    args = parser.parse_args()
    assert ultralytics.__version__ == "8.3.203", "Use the pinned export environment"
    assert args.size >= 64 and args.size % 32 == 0
    torch.set_num_threads(2)
    torch.manual_seed(17)
    model = YOLOE(args.model).model.cpu().float().eval()
    head = model.model[-1]
    assert hasattr(head, "savpe") and not getattr(head, "is_fused", False)
    assert not hasattr(head, "lrpc"), "Prompt-free/fused models are not supported"
    head.nc = 1
    head.export = True
    head.format = "onnx"
    head.dynamic = False
    args.output.mkdir(parents=True, exist_ok=True)
    image = torch.rand(1, 3, args.size, args.size)
    visuals = torch.ones(1, 1, args.size // 8, args.size // 8)
    encoder, detector = Encoder(model).eval(), Detector(model).eval()
    with torch.no_grad():
        embedding = encoder(image, visuals)
        expected = detector(image, embedding)
        assert tuple(embedding.shape) == (1, 1, 512)
        assert expected.ndim == 3 and expected.shape[2] == 5
        for wrapper, inputs, name, input_names, output_name in [
            (encoder, (image, visuals), "encoder", ["images", "visuals"], "embeddings"),
            (detector, (image, embedding), "detector", ["images", "embeddings"], "detections"),
        ]:
            path = args.output / f"{name}.onnx"
            torch.onnx.export(wrapper, inputs, str(path), input_names=input_names,
                              output_names=[output_name], opset_version=17,
                              dynamo=False, do_constant_folding=True)
            onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    enc = ort.InferenceSession(str(args.output / "encoder.onnx"), options, providers=["CPUExecutionProvider"])
    det = ort.InferenceSession(str(args.output / "detector.onnx"), options, providers=["CPUExecutionProvider"])
    actual_embedding = enc.run(None, {"images": image.numpy(), "visuals": visuals.numpy()})[0]
    actual = det.run(None, {"images": image.numpy(), "embeddings": actual_embedding})[0]
    np.testing.assert_allclose(actual_embedding, embedding.numpy(), atol=2e-4, rtol=2e-3)
    np.testing.assert_allclose(actual, expected.numpy(), atol=2e-2, rtol=2e-3)
    # Verify the prompt is a genuine runtime input, not constant-folded into a model.
    changed = det.run(None, {"images": image.numpy(), "embeddings": -actual_embedding})[0]
    assert np.max(np.abs(changed[..., 4] - actual[..., 4])) > 1e-5
    metadata = {"format": "mobile-tracker-yoloe-v1", "ultralytics": ultralytics.__version__,
                "checkpoint": args.model, "size": args.size, "embedding_size": 512,
                "preprocessing": "RGB float32 /255, centered square letterbox value 114",
                "visuals": [1, 1, args.size // 8, args.size // 8],
                "detections": "[1,N,5] cx,cy,w,h,probability in letterbox pixels; no NMS",
                "sha256": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                           for name in ("encoder.onnx", "detector.onnx")}}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Export + PyTorch/ORT numerical parity + mutable-prompt checks passed: {args.output}")


if __name__ == "__main__":
    main()
