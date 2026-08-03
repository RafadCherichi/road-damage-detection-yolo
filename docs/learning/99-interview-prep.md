# 99 — Interview Prep

This file is never "done" — append to it every time a new concept or
result creates a new plausible interview question (per `CLAUDE.md`'s
standing rule). Current status: covers the real evaluation results, the
Kaggle/Colab pivot, and the real ONNX parity result.

## "Walk me through your project"

A ready 60-90 second script:

> "This is a road damage detector for autonomous-vehicle perception —
> YOLOv8n fine-tuned to find four types of road damage (longitudinal
> cracks, transverse cracks, alligator cracks, potholes) from the RDD2022
> dataset, filtered to an India+Japan subset for visual diversity without
> needing the full 6-country, ~47K-image set.
>
> Architecture-wise, YOLOv8 is anchor-free — instead of matching against
> pre-defined box shapes, each spatial location directly predicts
> distances to an object's four edges, which sidesteps a whole class of
> tuning problems around picking good anchor priors. I started from
> COCO-pretrained weights (transfer learning) rather than training from
> scratch, since detection-pretrained weights transfer directly to a new
> detection task, and our ~14.6K images is a small fraction of COCO's
> ~118K pretraining images.
>
> The interesting part of this project is less the model and more the
> real engineering obstacles: local training on a 4GB VRAM / 8GB RAM
> laptop hit a chain of real crashes — a deprecated Focal Loss API, a
> Windows-multiprocessing RAM exhaustion bug, a MixUp-specific CPU-RAM
> crash — each diagnosed from source, not guessed at. Training ultimately
> moved to Kaggle Notebooks after a Colab session disconnected mid-run.
>
> For explainability, I tried EigenCAM first — it's fast and needs no
> gradients — but it consistently highlighted sky and glare instead of
> the actual damage on outdoor road photos, since it has no concept of
> class, just 'what varies most.' I built a custom per-detection Grad-CAM
> instead, which backprops from each specific detection's own class logit
> through the correct multi-scale feature layer.
>
> Final model gets 0.718 mAP50 overall, with the most common class
> (alligator cracks) at 0.858 mAP50, exported to a static-shape ONNX
> graph for edge deployment."

## Numbers to memorize

| Fact | Value |
|---|---|
| Dataset | RDD2022, India+Japan subset |
| Total images | 14,594 (train 12,425 / valid 1,052 / test 1,117) |
| Classes | D00 longitudinal, D10 transverse, D20 alligator, D40 pothole |
| Class imbalance | D20:D10 ≈ 2:1 (5,617 vs 2,814 train annotations) |
| Model | YOLOv8n — 3,006,428 params, 8.1 GFLOPs, 73 fused layers |
| Pretrained from | COCO (`yolov8n.pt`), full fine-tuning, no frozen layers |
| **mAP50 (all classes)** | **0.718** |
| **mAP50-95 (all classes)** | **0.386** |
| Best class | D20 (Alligator): 0.858 mAP50, 0.539 mAP50-95 |
| Weakest relative localization | D10 (Transverse): largest mAP50→mAP50-95 gap (~53%) |
| Precision / Recall (all) | 0.671 / 0.662 |
| Training compute | Local (crashed) → Colab T4 (disconnected) → Kaggle Notebooks (final) |
| Classification loss | Plain BCE (Focal Loss/`fl_gamma` not available in installed Ultralytics 8.4.x) |
| Box loss | CIoU |
| Deployment format | Static-shape ONNX (batch=1, 640×640, opset 12) |
| ONNX/PyTorch parity | **PASS** — max abs difference 0.000061 (atol=0.001) |

## Four real "tell me about a challenge" stories

**1. The RAM-crash chain (local training).**
Local training on a 4GB VRAM / 8GB RAM laptop failed three separate ways
in sequence, each requiring root-cause diagnosis rather than trial and
error: (a) a `fl_gamma` hyperparameter that turned out to not exist
anymore in the installed Ultralytics version — verified by reading
`v8DetectionLoss`'s actual source rather than assuming a stale mental
model was still correct; (b) `WinError 1455` from Windows' multiprocessing
`spawn` re-importing torch/CUDA per DataLoader worker, exhausting 8GB RAM
with the default `workers=8`; (c) a MixUp-specific CPU-RAM
`ArrayMemoryError`, traced to MixUp needing two full mosaic-composited
images in memory simultaneously, compounded by end-of-epoch validation
stacking its own memory on top. Each fix was verified independently
before moving to the next crash. **Why this is a good story:** it shows
methodical debugging from source code and logs, not guessing — and
knowing when a local constraint (not a code bug) is the real limiting
factor.

**2. The Focal Loss deprecation discovery.**
The original plan (Decision 2.1, following `docs/blueprint.md`) was to
use `fl_gamma` for the dataset's ~2:1 class imbalance. When it crashed
immediately with "not a valid YOLO argument," the investigation went
straight to `ultralytics/utils/loss.py` rather than trying random
alternate spellings — confirming `v8DetectionLoss` now uses plain BCE,
with `FocalLoss`/`VarifocalLoss` classes present but unused in the
standard training path. Decision: accept plain BCE for this moderate
imbalance rather than build custom loss-callback code for a small
expected gain. **Why this is a good story:** demonstrates reading library
source to get a real answer instead of assuming outdated documentation
still holds, and making a deliberate, justified scope call rather than
over-engineering a fix.

**3. The Colab→Kaggle compute pivot.**
After moving training off the crash-prone local laptop to Colab's free
T4 GPU, that Colab session disconnected before training finished — a
common free-tier failure mode. Rather than retry the same fragile setup,
training moved to Kaggle Notebooks, a second free-GPU option with longer
uninterrupted sessions, using the same `train.py`/configs unmodified. The
final tracked weights (`yolov8n_kaggle_run`) come from that run.
**Why this is a good story:** shows adapting to real infrastructure
failures (not just code bugs) under a genuine $0-budget constraint,
without losing momentum or silently working around it — this exact
sequence is now documented plainly in the README rather than smoothed
over into a falsely clean narrative.

**4. The ONNX "verified" claim that wasn't, and the two bugs found fixing it.**
An audit caught that the README claimed ONNX output was "numerically
verified against the `.pt` model," but the code only ran
`onnx.checker.check_model()` — a structural check with no numerical
comparison at all. Writing the real check (`verify_parity()`) surfaced
two genuine environment bugs in the process: chaining a CPU-mode
`export()` call with a GPU `predict()` call in the same process corrupted
CUDA device state (`AssertionError: Invalid device id`), and Ultralytics
auto-installed `onnxruntime-gpu`, which required CUDA 13.x/cuDNN 9.x
against this environment's actual CUDA 12.1 — a real, unresolvable
version mismatch. Both were fixed by forcing `device="cpu"` for the
parity check itself, since numerical correctness doesn't require GPU
execution. The check then genuinely passed: max absolute difference
0.000061. **Why this is a good story:** shows the difference between a
claim that *sounds* verified and one that actually is, plus systematic
isolation of two distinct real bugs (one at a time, via controlled
fresh-process tests) rather than a single conflated fix.

## Where deeper Q&A lives, by topic
Each concept file ends with its own 2-3 interview Q&As — this file
doesn't duplicate them, just points to them:
- Anchor-free detection, NMS, IoU/GIoU/DIoU/CIoU: `01`, `02`, `03`
- Transfer learning: `04`
- mAP/precision/recall: `05`
- Mosaic/MixUp, Focal Loss: `06`, `07`
- SAHI (documented, not implemented): `08`
- EigenCAM and why it failed here: `09`
- ONNX deployment: `10`
