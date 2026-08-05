## Phase 1 — Environment & Data
Dataset: RDD2022 (Kaggle, ~47K images, 4 damage classes: D00 longitudinal
crack, D10 transverse crack, D20 alligator crack, D40 pothole). Pascal VOC
XML annotations, must convert to YOLO TXT (normalized coords + class index).

Decisions to present when we reach this phase:
- 1.1 Environment Manager: Conda vs venv vs Docker (Conda recommended —
  reproducible, standard in ML, zero DevOps overhead)
- 1.2 Country Subset: All 6 countries vs India+Japan vs US-only
  (India+Japan recommended — diversity without crushing our 4GB VRAM budget)
- 1.3 Annotation Conversion: Manual script vs Roboflow free vs FiftyOne
  (Roboflow free recommended — spend learning time on model, not data scripts)

## Phase 2 — Preprocessing & Augmentation
Decisions:
- 2.1 Class Imbalance: Oversample vs Focal Loss vs Weighted Sampler vs
  Combination (Focal Loss recommended — YOLO supports natively).
  **SUPERSEDED, see `docs/learning/07-focal-loss.md`'s correction
  section:** the installed Ultralytics version (8.4.x) doesn't wire Focal
  Loss into the standard training path at all (`fl_gamma` isn't a valid
  argument; `v8DetectionLoss` uses plain `nn.BCEWithLogitsLoss`) — this
  was discovered only when training crashed on it. Actual final decision:
  plain BCE, accepted as a defensible tradeoff given the moderate (~2:1)
  imbalance. Not a case of the recommendation being wrong at the time it
  was written — the underlying library changed the option away.
- 2.2 Augmentation: YOLO built-in vs Albumentations vs Both composed
  (Both recommended — mosaic/mixup for distribution shift, Albumentations
  for weather realism)

## Phase 3 — Model Selection & Training
Decisions:
- 3.1 Architecture (KEYSTONE): YOLOv8 vs YOLOv9 vs YOLOv10 vs RT-DETR vs
  Faster R-CNN (YOLOv8 recommended — proven, mature ecosystem, real-time)
- 3.2 Model Scale: nano vs small vs medium vs large (GPU is RTX 3050 4GB VRAM
  — nano or small only, NOT medium/large as originally planned for 16GB)
- 3.3 Transfer Learning: COCO pretrained vs ImageNet-only vs scratch
  (COCO recommended)
- 4.1 Experiment Tracking: TensorBoard vs WandB vs MLflow
  (TensorBoard recommended — zero setup, local).
  **NEVER FINALIZED** — no experiment tracker was actually wired up;
  `src/train.py` relies solely on Ultralytics' own automatic
  `results.csv`/plots per run. Flagged here rather than silently assumed
  done.
- 4.2 Optimizer: SGD+cosine vs AdamW+OneCycleLR vs auto
  (SGD+cosine recommended — YOLO default, battle-tested).
  **NEVER FINALIZED** — `configs/train_config.yaml` ships with
  `optimizer: auto` (Ultralytics' own default), explicitly marked
  "provisional Ultralytics default — revisit at Phase 3/4 decision gate"
  in that file's comments. The gate was never actually reached.
- 4.3 Small Object Strategy: train as-is vs SAHI at inference vs multi-scale
  training vs custom head (SAHI recommended — road cracks are tiny, zero
  training cost).
  **NOT IMPLEMENTED** — see `docs/learning/08-sahi.md`, which documents
  the concept and the real latency/throughput tradeoff against this
  project's "real-time" framing, but SAHI itself was never built. Current
  actual strategy is "train as-is."

## Phase 5 — Evaluation & Explainability
Decisions:
- 5.1 Explainability: Grad-CAM vs EigenCAM vs SHAP vs Attention Rollout
  (EigenCAM recommended — more stable than Grad-CAM for detection models).
  **SUPERSEDED, see `docs/learning/09-eigencam.md`:** EigenCAM was
  actually tried first (per this recommendation) and failed observably on
  real road-scene images — sky/glare regions dominated its PCA-based
  saliency, consistently highlighting the wrong region (real evidence:
  `results/eigencam_outputs/*.jpg`). Actual final decision: a custom
  per-detection Grad-CAM (`src/explainability.py`), built specifically in
  response to this documented failure, not a preference swap.
- 5.2 Deployment Format: PyTorch .pt vs ONNX vs TensorRT vs OpenVINO
  (ONNX recommended — cross-platform, industry standard)

## Must-Know Concepts (teach deeply when reached, skip boilerplate)
Anchor-free detection, Non-Maximum Suppression, IoU/GIoU/DIoU/CIoU,
Transfer Learning, mAP derivation, Mosaic & MixUp, Focal Loss, SAHI,
EigenCAM, ONNX export pipeline.

## Repository Structure
data/{raw,processed,splits}, notebooks/, src/{data_utils,augmentation,
train,evaluate,inference,explainability}.py, configs/{train,model}.yaml,
results/{metrics,visualizations,eigencam_outputs,inference_samples}/,
.gitignore, requirements.txt, environment.yml, README.md
