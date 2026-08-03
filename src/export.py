"""
Export trained YOLOv8 weights to ONNX for edge deployment.
Static shape (batch=1, 640x640) — no dynamic axes, matching our
single-frame inference use case.
"""
from ultralytics import YOLO
import argparse
import numpy as np
import onnx
from pathlib import Path


def export_to_onnx(weights_path, output_path=None, imgsz=640, opset=12, simplify=True):
    model = YOLO(weights_path)

    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,      # static shape — deliberate choice, see reasoning above
        simplify=simplify,  # runs onnx-simplifier: folds constants, removes redundant ops
        opset=opset,        # opset 12 has broad ONNX Runtime version compatibility
        half=False,         # FP32 — keep FP16 as a documented future optimization, not default
    )

    # Verify the exported graph is structurally valid before trusting it
    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX export structurally valid: {export_path}")
    print(f"Input shape: {onnx_model.graph.input[0].type.tensor_type.shape}")

    return export_path


def verify_parity(pt_weights, onnx_path, sample_image, imgsz=640, atol=1e-3):
    """Numerical parity check: run the same real image through the .pt and
    .onnx models (both via YOLO()'s auto-backend detection, so both paths
    share identical pre/post-processing — isolating the comparison to the
    model itself) and compare final detection outputs directly.

    Forces device="cpu" for both models. This check only needs numerical
    correctness, not speed, and CPU sidesteps two real environment issues
    found on this machine: (1) a preceding CPU-mode export() call corrupts
    CUDA device state within the same process, breaking auto device
    selection for the .pt predict call; (2) Ultralytics auto-installed
    onnxruntime-gpu (since CUDA is visible), but it requires CUDA 13.x +
    cuDNN 9.x while this env has CUDA 12.1 (torch's cu121 wheel) — a real,
    unresolvable-here version mismatch. CPU-only avoids both entirely.
    """
    pt_model = YOLO(pt_weights)
    onnx_model = YOLO(onnx_path)

    pt_result = pt_model.predict(sample_image, imgsz=imgsz, device="cpu", verbose=False)[0]
    onnx_result = onnx_model.predict(sample_image, imgsz=imgsz, device="cpu", verbose=False)[0]

    pt_boxes = pt_result.boxes.data.cpu().numpy()
    onnx_boxes = onnx_result.boxes.data.cpu().numpy()

    if pt_boxes.shape != onnx_boxes.shape:
        print(f"PARITY CHECK: FAIL — detection count/shape differs (pt={pt_boxes.shape}, onnx={onnx_boxes.shape})")
        return False

    if pt_boxes.size == 0:
        print("PARITY CHECK: SKIPPED — no detections on this sample image from either model")
        return None

    max_diff = float(np.abs(pt_boxes - onnx_boxes).max())
    passed = bool(np.allclose(pt_boxes, onnx_boxes, atol=atol))
    print(f"PARITY CHECK: {'PASS' if passed else 'FAIL'} — max abs difference {max_diff:.6f} (atol={atol})")
    return passed


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    default_weights = str(PROJECT_ROOT / "results" / "training_runs" / "yolov8n_kaggle_run" / "weights" / "best.pt")
    default_sample = str(PROJECT_ROOT / "data" / "raw" / "train" / "images" / "India_000027.jpg")

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=default_weights)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--sample-image", default=default_sample)
    args = parser.parse_args()

    onnx_path = export_to_onnx(args.weights, imgsz=args.imgsz, opset=args.opset)
    verify_parity(args.weights, onnx_path, args.sample_image, imgsz=args.imgsz)