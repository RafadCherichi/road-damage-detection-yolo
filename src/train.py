"""Phase 3: fine-tune YOLOv8n (COCO-pretrained) on the RDD2022 India+Japan
subset. Reads configs/model_config.yaml (architecture/weights/imgsz) and
configs/train_config.yaml (loss/augmentation/schedule) and runs a single
model.train() call — no custom training loop.
"""

import argparse
import os
from pathlib import Path

# Must be set before torch is imported. Our env mixes conda-forge numpy/
# matplotlib (links LLVM's libomp.dll) with a pip-installed torch wheel
# (bundles Intel's libiomp5md.dll) — loading both in one process trips
# OMP Error #15 (duplicate OpenMP runtime). This is the standard, widely
# used workaround for exactly this conda-forge + pip-torch combination.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a last.pt checkpoint to resume an interrupted run from "
        "(e.g. results/training_runs/yolov8n_baseline-3/weights/last.pt)",
    )
    args = parser.parse_args()

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available — Ultralytics will raise if device=0 is requested.")

    if args.resume:
        # Resuming restores model/optimizer/epoch/augmentation state from the
        # checkpoint itself. train_config.yaml/model_config.yaml are not
        # re-read here on purpose — re-applying them could conflict with the
        # checkpoint's own saved args. Passing an explicit path (not bare
        # resume=True) avoids Ultralytics' fallback "most recently modified
        # last.pt anywhere under cwd" search, which gets ambiguous once more
        # than one run exists.
        model = YOLO(args.resume)
        model.train(resume=args.resume)
        return

    train_cfg = load_yaml(PROJECT_ROOT / args.config)
    model_cfg = load_yaml(PROJECT_ROOT / args.model_config)

    train_args = dict(train_cfg)
    train_args["data"] = str(PROJECT_ROOT / train_cfg["data"])
    train_args["imgsz"] = model_cfg["imgsz"]
    train_args.setdefault("project", str(PROJECT_ROOT / "results" / "training_runs"))
    train_args.setdefault("name", f"yolov8{model_cfg['scale']}_baseline")

    model = YOLO(model_cfg["weights"])
    model.train(**train_args)


if __name__ == "__main__":
    main()
