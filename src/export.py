"""
Export trained YOLOv8 weights to ONNX for edge deployment.
Static shape (batch=1, 640x640) — no dynamic axes, matching our
single-frame inference use case.
"""
from ultralytics import YOLO
import argparse
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
    print(f"ONNX export verified: {export_path}")
    print(f"Input shape: {onnx_model.graph.input[0].type.tensor_type.shape}")

    return export_path


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent 
    default_weights = str(PROJECT_ROOT / "results" / "training_runs" / "yolov8n_kaggle_run" / "weights" / "best.pt")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=default_weights)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()
    
    export_to_onnx(args.weights, imgsz=args.imgsz, opset=args.opset)