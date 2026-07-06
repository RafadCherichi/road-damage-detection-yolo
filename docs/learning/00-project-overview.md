# 00 — Project Overview

## Problem Statement, in plain language
Imagine you're a self-driving car deciding where it's safe to drive. Before
the car can decide "avoid that spot," it first needs to *see* the damage —
a crack, a pothole — in a camera frame, in real time, as the car is moving.
That "look at an image and draw a box around the damage, with a label"
step is called **object detection** (locating *and* classifying things in
an image, as opposed to just classifying the whole image as one label).
This project builds and trains one of those object detectors, specialized
for four kinds of road damage, as a portfolio piece demonstrating applied
computer-vision skills for automotive/EV hiring (full framing in the root
`CLAUDE.md`).

## Architecture / Pipeline, step by step
Below is the full path from raw downloaded data to a deployed model. Each
arrow is a step we've either already done or will do next.

```
RDD2022 dataset (Kaggle, sujityp/rdd2022 upload — already in YOLO format)
        |
        v
  data/raw/{train,valid,test}/{images,labels}   <- kept only India+Japan
        |                                          (excluded: augmented_image_*
        |                                           files, since their filenames
        |                                           don't say which country they're
        |                                           from, so we can't honor the
        |                                           India+Japan-only choice for them)
        v
  configs/data.yaml   (says: 4 classes, D00/D10/D20/D40, in this order — caveated below)
        |
        v
  EDA (notebooks/01_eda.ipynb) -> class balance, bbox sizes, resolutions, co-occurrence
        |
        v
  YOLOv8 (nano/small size, pretrained on COCO) fine-tuning  <- capped by RTX 3050, 4GB VRAM
        |
        v
  Evaluation (mAP metric) + EigenCAM explainability
        |
        v
  ONNX export for deployment
```

## Tech Stack — what each piece is and why it's here
| Layer | Choice | Plain-language "what is this" | Why we picked it |
|---|---|---|---|
| Environment manager | Conda (`rdd-yolo` env) | A **conda environment** is an isolated folder of Python + libraries, versioned separately from your system Python, so this project's dependencies can't silently break some other project's. | Reproducible, the ML-community standard, no DevOps overhead (no Docker needed). |
| Deep learning framework | PyTorch 2.2.0, `+cu121` build | **PyTorch** is the library that does the actual tensor math (matrix multiplies, gradients) on the GPU. The `cu121` suffix means "built against CUDA 12.1" — **CUDA** is NVIDIA's toolkit that lets code run on the GPU instead of the CPU, which matters because GPUs do the thousands of parallel multiplications neural networks need far faster than a CPU can. | We installed this via `pip` from a manually-downloaded wheel, not conda's `pytorch` channel — conda's channel repeatedly failed mid-download (`IncompleteRead` on the 1.2GB package, 3 attempts) over an unstable connection. A pip wheel is just a single file, so it sidestepped the broken multi-part download entirely. |
| Detector architecture | YOLOv8 (Ultralytics library) | **YOLO** ("You Only Look Once") is a family of object detectors that look at the whole image once and directly predict all the boxes+labels in one pass, instead of scanning the image piece by piece — that's *why* it's fast enough for real-time use. | Proven, mature ecosystem, genuinely real-time on modest hardware. |
| Dataset | RDD2022, India+Japan subset only | RDD2022 is a public dataset of road photos from 6 countries, each photo hand-annotated with damage locations. | More visual variety than a single country, without needing the compute budget of all 6 countries (~47K images) on a 4GB laptop GPU — think of VRAM like the cargo space in a moving truck: more variety of "furniture" (images) needs more space, and 4GB is a small truck. |
| Annotation format | YOLO TXT (already in this upload) | Each image has a matching `.txt` file — one line per damage box, formatted as `class_id x_center y_center width height`, all as fractions of the image size (so it works regardless of the image's actual pixel dimensions). The older, more verbose alternative is **Pascal VOC XML** (one full XML file per image, coordinates in raw pixels). | No conversion work needed — this specific Kaggle upload already ships pre-converted to YOLO TXT, unlike the original RDD2022 release. |

## Why India+Japan specifically (not all 6 countries, not just one)
Per `docs/blueprint.md` decision 1.2: enough geographic and visual
diversity (different road surfaces, camera angles, damage styles) to make
the model generalize better than training on one country, but without the
GPU memory cost of the full 6-country, ~47K-image set.

## Class Mapping Caveat — read this before trusting `configs/data.yaml`
Here's the problem in plain terms: every label file just says a class is
"`0`", "`1`", "`2`", or "`3`" — a bare number, not a name. Something has to
tell us that, say, "0 means longitudinal crack." Normally that "something"
is a `data.yaml` or `classes.txt` file shipped with the dataset. This
specific archive (`sujityp/rdd2022` on Kaggle) ships with **none of those**
— confirmed by directly inspecting every file after extraction. So the
mapping below is *inferred*, not read off a label:

**Mapping used:** `0=D00 (longitudinal crack), 1=D10 (transverse crack),
2=D20 (alligator crack), 3=D40 (pothole)`.

Why we trust it anyway — three independent, converging clues:
1. The [RDD2022 paper](https://arxiv.org/pdf/2209.08538) and the
   [official sekilab GitHub repo](https://github.com/sekilab/RoadDamageDetector)
   both always list the four classes in this same order: D00, D10, D20, D40
   (ascending by the numeric damage code).
2. A *different* team's RDD2022-to-YOLO conversion repo,
   [sivakanth1/Detecting_Road_Damage](https://github.com/sivakanth1/Detecting_Road_Damage),
   ships an actual `data.yaml` we could read, and it encodes
   `names: ['D00', 'D10', 'D20', 'D40']` — i.e. index 0 through 3 in that
   exact order.
3. A tool called Roboflow — a very common way people convert this exact
   dataset to YOLO format, and the tool `blueprint.md` itself expects we'll
   likely use — alphabetizes class names by default, and D00<D10<D20<D40
   happens to sort correctly either way.
4. As a sanity check against our *own* data: class `2` (D20, alligator
   crack) is the single most common annotation in our India+Japan train
   split (5,617 out of ~15,938 total). That lines up with alligator
   cracking being reported as the dominant damage type specifically in
   India/Japan RDD2022 data — if the mapping were scrambled, we'd have no
   reason to expect that particular class to come out on top.

What we *couldn't* do: Kaggle's dataset page renders its content with
JavaScript, so it couldn't be fetched and read directly to check whether
the uploader documented this mapping somewhere on the page itself. If the
EDA notebook's class-distribution numbers ever look inconsistent with
published RDD2022 per-class statistics, this mapping is the first thing to
re-examine.

## Where this connects to interview prep
Being able to explain *why* a class mapping needed to be reverse-engineered
(rather than just trusted) is itself a good interview story about data
diligence — see `docs/learning/99-interview-prep.md` as that file grows.
