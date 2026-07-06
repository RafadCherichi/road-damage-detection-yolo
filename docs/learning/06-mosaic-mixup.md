# 06 — Mosaic & MixUp Augmentation

## The intuition, before any jargon
Imagine studying for a test with flashcards. A normal flashcard shows one
clean photo. **Mosaic** augmentation is like someone hands you a page cut
into 4 quarters, and pasted into each quarter is a *different* flashcard —
sometimes cropped at an odd angle, sometimes small, sometimes right at the
edge of the page. You still have to find and name everything on the page,
but now you're forced to recognize objects regardless of their size,
position, or what's cropped next to them. That's exactly what Mosaic does
to training images: it stitches 4 real photos into 1 composite, and the
model has to detect every object in every quadrant.

**MixUp** is a different trick: instead of cutting-and-pasting quadrants,
it's like holding two transparent flashcards up to the light *at the same
time*, so you see both images faintly overlaid on each other. The model has
to still recognize the real objects even when the picture is a hard-to-read
blend of two scenes.

Both techniques exist for the same underlying reason: real datasets are
finite, so instead of only ever showing the model the exact photos we
collected, we synthetically create new, harder, more varied training
scenes out of the photos we already have — effectively getting more
"practice problems" for free.

## Mosaic — the geometry, worked step by step
Mosaic builds one new training image (typically the model's standard input
size, e.g. 640×640) out of 4 source images, each resized and placed into
one quadrant. Every bounding box has to be transformed to match: if a
source image is resized by scale factor $s$ and placed at pixel offset
$(dx, dy)$ in the new canvas, each box coordinate (in pixels) becomes:

$$x' = s \cdot x + dx \qquad y' = s \cdot y + dy \qquad w' = s \cdot w \qquad h' = s \cdot h$$

**Term-by-term breakdown:**
- $x, y, w, h$ — the box's original center coordinates and size, in pixels,
  in its *source* image.
- $s$ — the scale factor applied when resizing the source image to fit
  exactly into its assigned quadrant (e.g. if a source image must shrink
  to a quarter of its width to fit, $s = 0.25$ for that dimension; Mosaic
  typically uses a uniform $s$ for both axes).
- $dx, dy$ — the pixel offset of that quadrant's corner within the new,
  larger mosaic canvas (e.g. the top-left quadrant has $dx=dy=0$; the
  top-right quadrant's $dx$ equals the canvas half-width).
- $x', y', w', h'$ — the box's new coordinates and size, in pixels, in the
  finished mosaic canvas — what actually gets written to the training
  label.

### Worked example 1 — illustrative numbers
Take a training image with a box at normalized coordinates
$(x_c=0.5,\ y_c=0.5,\ w=0.3,\ h=0.2)$ in its original 640×640 photo.

1. Convert to pixels in the original image: $x_c = 320\text{px},\ y_c =
   320\text{px},\ w = 192\text{px},\ h = 128\text{px}$.
2. Suppose this image is placed in the mosaic's top-left quadrant, which is
   320×320 pixels (a quarter of the 640×640 canvas). To fit a 640×640 image
   into a 320×320 quadrant, it must be resized by scale factor $s = 0.5$.
3. Apply the scale (top-left quadrant offset is $dx=dy=0$, so it's just the
   scaling): $x_c' = 0.5 \times 320 = 160\text{px},\ y_c' = 160\text{px},\
   w' = 0.5 \times 192 = 96\text{px},\ h' = 0.5 \times 128 = 64\text{px}$.
4. Re-normalize against the full 640×640 mosaic canvas: $x_c = 160/640 =
   0.25,\ y_c = 0.25,\ w = 96/640 = 0.15,\ h = 64/640 = 0.1$.

The box went from occupying $30\%\times20\%$ of its original image to just
$15\%\times10\%$ of the mosaic image — exactly half as large in each
dimension.

### Worked example 2 — tied to our actual project data
Rather than an invented box, take a **real annotation** from
`data/raw/train/labels/India_000027.txt`, one of our actual India-subset
training images: a D20 (alligator crack) box on a real 720×720 photo,
label line `2 0.1986 0.8118 0.3639 0.2181`.

1. Convert to pixels in the real 720×720 image: $x_c = 0.1986 \times 720
   \approx 143.0\text{px},\ y_c = 0.8118 \times 720 \approx 584.5\text{px},\
   w = 0.3639 \times 720 \approx 262.0\text{px},\ h = 0.2181 \times 720
   \approx 157.0\text{px}$.
2. Place this image in the mosaic's top-left 320×320 quadrant: to fit the
   720×720 photo into a 320×320 quadrant, $s = 320/720 \approx 0.4444$.
3. Apply the scale ($dx=dy=0$): $x_c' \approx 0.4444 \times 143.0 \approx
   63.6\text{px},\ y_c' \approx 0.4444 \times 584.5 \approx 259.8\text{px},\
   w' \approx 0.4444 \times 262.0 \approx 116.4\text{px},\ h' \approx
   0.4444 \times 157.0 \approx 69.8\text{px}$.
4. Re-normalize against the full 640×640 mosaic canvas: $x_c \approx
   63.6/640 \approx 0.0993,\ y_c \approx 259.8/640 \approx 0.4059,\ w
   \approx 116.4/640 \approx 0.1819,\ h \approx 69.8/640 \approx 0.1090$.

Compare: the original normalized size was $w=0.3639, h=0.2181$; after
mosaic placement it's $w \approx 0.1819, h \approx 0.1090$ — again, almost
*exactly* half in each dimension, even though this source image (720×720)
had a completely different resolution from Worked Example 1's (640×640).
That's not a coincidence: as long as a source image is resized to exactly
fill its quadrant, $s = \text{quadrant\_size} / \text{source\_image\_size}$,
and the quadrant is half the mosaic canvas's width and height, so the
normalized box size always shrinks by a factor of $0.5$ *regardless of the
source image's original resolution*. This is precisely why our Phase 1
EDA bbox-size heatmap (which already showed most road-damage boxes are
small relative to their image) matters here: mosaic will roughly halve
every box's normalized footprint again, on top of whatever was already
small, and can push a marginal crack below a size the model can still
detect. This is exactly why Phase 2 Decision 2.2 flagged tuning mosaic
conservatively (e.g. Ultralytics' `close_mosaic` setting, which disables
mosaic for the final N training epochs so the model spends its last
epochs calibrating on true-scale objects, plus a moderated `scale`
augmentation range) rather than using aggressive defaults blindly.

## MixUp — the math

**Correction, discovered during the first real training run:** the theory
below is still accurate, but MixUp is currently **disabled**
(`mixup: 0.0` in `configs/train_config.yaml`, was `0.15`) for this
project's baseline run. The `yolov8n_baseline-3` run crashed partway
through epoch 2 with a CPU-RAM `ArrayMemoryError` specifically during a
MixUp event: YOLOv8 chains Mosaic → MixUp, so one MixUp blend needs *two
full mosaic-composited images* in memory simultaneously (each itself built
from up to 4 source images) — a sharp memory spike on an 8GB RAM laptop
that's already tight. Mosaic alone ran a full clean epoch first, so MixUp
specifically is the implicated operation. This is a resource constraint,
not a flaw in the technique — the math and reasoning below are unchanged
and still the reason MixUp is worth re-enabling later once memory headroom
(e.g. after confirming `batch: 4` trains stably) is better understood.

MixUp blends two whole images pixel-by-pixel:

$$I_{\text{mix}} = \lambda \cdot I_A + (1-\lambda) \cdot I_B$$

**Term-by-term breakdown:**
- $I_A, I_B$ — the two source images, as arrays of pixel values.
- $\lambda$ (lambda) — the blend weight, between $0$ and $1$, sampled
  randomly per mixed pair from a Beta distribution. It's tuned so $\lambda$
  usually lands close to $0$ or $1$, meaning most mixes are *mostly* one
  image with just a faint trace of the other, rather than a 50/50 blend
  every time.
- $I_{\text{mix}}$ — the resulting blended image actually fed to the
  model. Crucially, the **labels are not blended** — both $I_A$'s and
  $I_B$'s full bounding boxes stay in the training target as-is, so the
  model is asked to "find the real objects even though the picture itself
  is a partial, noisy overlay."

### Worked example — using two real images from our dataset
Take one real pixel value from two actual training photos (converted to
grayscale for simplicity), rather than invented numbers: in
`India_000027.jpg`, the pixel at the real D20 box's center (computed above,
pixel coordinates $(143, 584)$ in the 720×720 image) has grayscale value
$101$. In `Japan_000001.jpg`, the pixel at its image center has grayscale
value $140$.

With $\lambda = 0.4$:

$$I_{\text{mix}} = 0.4 \times 101 + 0.6 \times 140 = 40.4 + 84.0 = 124.4$$

That pixel now shows an intermediate gray value ($124.4$, between the two
originals) that doesn't cleanly match either source image's true
appearance at that location. This pushes the model away from relying on
crisp, clean pixel patterns alone, which builds robustness to real-world
visual noise (glare, overlapping shadows, etc.) — relevant for road
scenes, and notice the blend used one real India-subset image and one
real Japan-subset image, exactly the kind of cross-country mixing that
happens routinely once training pools both subsets together.

## Diagram
```
MOSAIC (spatial composition, 4 images -> 1 canvas)     MIXUP (pixel blend, 2 images -> 1)
+-----------+-----------+                              +-----------------+
| Image A   | Image B   |                               (o)  <- image A   
| (crack)   | (pothole) |                                 \  (faded/blended)
+-----------+-----------+                                (o)  <- image B, overlaid
| Image C   | Image D   |                              +-----------------+
| (bg-only) | (crack)   |                              blended = 0.4*A + 0.6*B
+-----------+-----------+                              labels: boxes from BOTH kept
each box rescaled/offset to its quadrant
```

## Why it matters for THIS project
Beyond the small-bbox caution above, mosaic has a second, complementary
effect relevant to Phase 2 Decision 2.1 (Focal Loss for class imbalance):
because every mosaic tile randomly recombines 4 images from the dataset,
images containing the rarer D10 class get reshuffled into more different
visual contexts over the course of training than they would if the model
only ever saw them in their one original photo — a mild, free diversity
boost for the minority class. Mosaic (data composition) and Focal Loss
(loss weighting) attack the same underlying class-imbalance problem from
two different angles, which is why both were chosen together rather than
treating them as redundant.

## Interview questions

**Q: Mosaic literally shrinks each object in the composite image — how can
that possibly help small-object detection?**
A: It's a genuine trade-off, not a free lunch. Mosaic's benefit is
diversity — one forward pass effectively trains on 4 images' worth of
objects, contexts, and scales, which reduces overfitting to fixed
backgrounds/positions. But yes, it does shrink each individual object
within the frame. That's exactly why this project disables mosaic for the
final training epochs (`close_mosaic`) — get the diversity benefit while
it's most useful early in training, then let the model spend its last
epochs calibrating on true, undistorted object scale.

**Q: What's the actual difference between Mosaic and MixUp, and why use
both instead of just one?**
A: Mosaic is spatial composition — cut and place 4 images into quadrants,
so each object's true appearance is preserved but its scale/position/
surrounding context changes. MixUp is pixel-value blending — overlay 2
images' raw pixel values, preserving each object's original scale and
position but degrading visual clarity. They guard against two different
kinds of overfitting: Mosaic against overfitting to fixed scale/position/
background, MixUp against overfitting to only ever seeing crisp, clean,
non-overlapping visual signals. Using both covers more failure modes than
either alone.

**Q: If you noticed recall on the smallest-box class dropping during
training, how would you adjust the augmentation config?**
A: First check whether that class's boxes are already the smallest in the
EDA bbox-size heatmap (in this project, that data already exists). If so,
reduce mosaic's aggressiveness — lower the `scale` augmentation range so
images aren't shrunk as much before tiling, reduce mosaic's sampling
probability, or move the `close_mosaic` cutoff earlier so more epochs
train on true-scale objects — rather than assuming the fix is simply "add
more augmentation."
