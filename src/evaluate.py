"""Evaluate a trained YOLOv8 checkpoint on the RDD2022 India+Japan valid
split. Runs Ultralytics' built-in validator and writes a clean per-class
summary CSV to results/metrics/, since Ultralytics' own run folder buries
the numbers inside its own structure.

Plots default OFF (--plots to enable): on this machine, plots=True
(confusion matrix / PR curve rendering) crashed non-deterministically
3/3 attempts with no Python traceback, at a different point each time —
consistent with a real race condition from this env's known dual-OpenMP-
runtime situation (KMP_DUPLICATE_LIB_OK=TRUE suppresses Intel's safety
check but doesn't fix the underlying conflict; Intel's own docs warn this
"may cause crashes ... or silently produce incorrect results"). The
numeric metrics computation itself is unaffected and has been reproduced
twice with identical results. If you need the plots, generating them on
Kaggle/Colab (same environment training already runs in, without this
laptop's specific OpenMP conflict) is the recommended path, or pass
--plots here to try anyway.
"""

import argparse
import csv
import os
from pathlib import Path

# Same fix as train.py: conda-forge numpy/matplotlib (LLVM libomp.dll) vs
# pip-installed torch (Intel libiomp5md.dll) trips OMP Error #15 if not set
# before torch is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=str(PROJECT_ROOT / "results" / "training_runs" / "yolov8n_kaggle_run" / "weights" / "best.pt"),
    )
    parser.add_argument("--data", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="DataLoader workers. Defaults to 0, not Ultralytics' default of 8 — "
        "on this machine, 8 spawned worker subprocesses each re-import torch/CUDA "
        "(Windows uses spawn, not fork), which was slow/RAM-heavy enough to look "
        "hung during a real run. See configs/train_config.yaml's workers comment "
        "for the full history of this exact issue during training.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate confusion matrix / PR curve plots. Off by default — "
        "see module docstring for why this crashes non-deterministically "
        "on this machine.",
    )
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        workers=args.workers,
        plots=args.plots,
        project=str(PROJECT_ROOT / "results" / "metrics"),
        name="yolov8n_kaggle_eval",
    )

    names = metrics.names
    class_indices = list(metrics.box.ap_class_index)
    rows = [
        {
            "class": "all",
            "precision": metrics.box.mp,
            "recall": metrics.box.mr,
            "map50": metrics.box.map50,
            "map50-95": metrics.box.map,
        }
    ]
    for i, cls_idx in enumerate(class_indices):
        rows.append(
            {
                "class": names[cls_idx],
                "precision": metrics.box.p[i],
                "recall": metrics.box.r[i],
                "map50": metrics.box.ap50[i],
                "map50-95": metrics.box.ap[i],
            }
        )

    out_path = PROJECT_ROOT / "results" / "metrics" / "metrics_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "map50", "map50-95"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved summary to {out_path}")
    print(f"{'class':<12}{'precision':>10}{'recall':>10}{'map50':>10}{'map50-95':>10}")
    for row in rows:
        print(f"{row['class']:<12}{row['precision']:>10.3f}{row['recall']:>10.3f}{row['map50']:>10.3f}{row['map50-95']:>10.3f}")


if __name__ == "__main__":
    main()
