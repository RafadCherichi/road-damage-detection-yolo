"""Albumentations weather-simulation pipeline (Phase 2 Decision 2.2 — the
"Both" half: Ultralytics' built-in mosaic/mixup, configured in
configs/train_config.yaml, is composed with this Albumentations pipeline
for weather realism).

Wiring this into an actual training run is deferred to src/train.py
(Phase 3, not written yet). Ultralytics doesn't expose a config-driven hook
for custom Albumentations transforms — the standard approach is to replace
ultralytics.data.augment.Albumentations' transform list with the Compose
built here before calling model.train().
"""

from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image, ImageDraw


def get_weather_augmentations() -> A.Compose:
    """Pipeline simulating rain, fog, and lighting shifts.

    Weather-effects only, deliberately no geometric transforms (rotation/
    scale/crop) — those are already Ultralytics' job (mosaic + scale in
    configs/train_config.yaml). RandomRain/RandomFog/RandomBrightnessContrast
    are all ImageOnlyTransform: they never move a box, so there's no
    bbox_params here — passing bboxes through Albumentations' bbox pipeline
    for a transform list that can't touch them would just be unused
    machinery (and trips Albumentations' own "no transform to process
    bboxes" warning). Boxes stay valid as-is; callers can reuse the
    original label file unchanged after applying this transform.
    """
    return A.Compose(
        [
            # Rain and fog together isn't physically realistic, so they're
            # mutually exclusive; brightness/contrast layers independently.
            A.OneOf(
                [
                    A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1.0),
                    A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=1.0),
                ],
                p=0.3,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        ],
    )


def _load_yolo_labels(label_path: Path):
    bboxes, class_labels = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            cls_id, xc, yc, w, h = line.split()
            bboxes.append([float(xc), float(yc), float(w), float(h)])
            class_labels.append(int(cls_id))
    return bboxes, class_labels


def _draw_boxes(image: np.ndarray, bboxes, color=(230, 25, 75)) -> Image.Image:
    im = Image.fromarray(image)
    draw = ImageDraw.Draw(im)
    h, w = image.shape[:2]
    for xc, yc, bw, bh in bboxes:
        x0, y0 = (xc - bw / 2) * w, (yc - bh / 2) * h
        x1, y1 = (xc + bw / 2) * w, (yc + bh / 2) * h
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
    return im


if __name__ == "__main__":
    # Demo/sanity check: apply the pipeline to one real train image and save
    # a before/after comparison, matching the EDA notebook's convention of
    # saving figures to results/visualizations/.
    project_root = Path(__file__).resolve().parent.parent
    image_path = project_root / "data" / "raw" / "train" / "images" / "India_000027.jpg"
    label_path = project_root / "data" / "raw" / "train" / "labels" / "India_000027.txt"
    out_dir = project_root / "results" / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    image = np.array(Image.open(image_path).convert("RGB"))
    bboxes, _ = _load_yolo_labels(label_path)

    transform = get_weather_augmentations()
    augmented = transform(image=image)

    # Boxes are unchanged by a weather-only pipeline, so the same real
    # bboxes are drawn on both the before and after images.
    before = _draw_boxes(image, bboxes)
    after = _draw_boxes(augmented["image"], bboxes)

    comparison = Image.new("RGB", (before.width + after.width, before.height))
    comparison.paste(before, (0, 0))
    comparison.paste(after, (before.width, 0))
    comparison.save(out_dir / "weather_augmentation_demo.png")
    print(f"Saved before/after comparison to {out_dir / 'weather_augmentation_demo.png'}")
