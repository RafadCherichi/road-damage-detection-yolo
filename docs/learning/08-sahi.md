# 08 — SAHI (Slice-Aided Hyper Inference)

**Status note, upfront:** unlike the other files in this folder, SAHI is
**not implemented** in this project. It's `docs/blueprint.md`'s Phase 4.3
recommendation for small-object detection, documented here as background
knowledge and a concrete next step — not a claim that it's running. Same
honesty pattern already used for the Focal Loss/`fl_gamma` correction in
`docs/learning/07-focal-loss.md`.

## The intuition, before any jargon
Imagine trying to find tiny towns on a huge world map by eye. Look at the
whole map at once and most small towns are practically invisible —
sub-pixel dots lost in the noise. But take a magnifying glass and scan
the map section by section, and suddenly those same towns are perfectly
readable, because each section gets zoomed in. The catch: if you slice
the map into non-overlapping sections, a town sitting right on a section
boundary might get cut in half and missed by both sections. So you scan
with slightly *overlapping* sections, then tape all your findings back
onto the master map at their correct global coordinates.

That's **SAHI**: instead of feeding a detector one full (possibly huge)
image, slice it into overlapping tiles, run the detector on each tile at
full resolution — so a small object that was a handful of pixels in the
full image becomes a much larger fraction of a tile — then remap every
tile's detections back into the original image's coordinate space and
merge overlapping duplicates with NMS (`docs/learning/02-non-max-suppression.md`).
Crucially, this is an **inference-time** technique — no retraining
required, since it's just choosing what pixels to hand the same trained
model at once.

## The math — but the "why" comes first
For an image of size $W \times H$, sliced into tiles of size $w \times h$
with overlap fraction $o$:

$$n_x = \left\lceil \frac{W}{w(1-o)} \right\rceil \qquad n_y = \left\lceil \frac{H}{h(1-o)} \right\rceil$$

**Term-by-term breakdown:**
- $W, H$ — the original image's width and height, in pixels.
- $w, h$ — the tile size fed to the detector (commonly matching the
  model's training resolution, e.g. $640 \times 640$).
- $o$ — the overlap fraction between adjacent tiles (commonly $0.1$–$0.2$).
  This exists specifically so an object sitting near a tile boundary still
  appears *whole* in at least one tile, rather than being split in half by
  both neighbors and missed by each.
- $n_x, n_y$ — the number of tiles needed along each axis. $w(1-o)$ is the
  effective *stride* between tile start positions — smaller than $w$
  itself, because of the overlap.

Remapping a detection from tile-local to global image coordinates is a
simple offset, using each tile's known top-left position $(i \cdot
w(1-o),\ j \cdot h(1-o))$:

$$x_{\text{global}} = x_{\text{local}} + i \cdot w(1-o) \qquad y_{\text{global}} = y_{\text{local}} + j \cdot h(1-o)$$

### Worked example 1 — illustrative numbers
A $1280 \times 1280$ image, tiles of $640 \times 640$, overlap $o = 0.2$.

1. Effective stride: $w(1-o) = 640 \times 0.8 = 512$ pixels.
2. $n_x = \lceil 1280/512 \rceil = \lceil 2.5 \rceil = 3$ tiles needed along
   each axis (real implementations shift the last tile inward to stay
   within the image bounds rather than reading past the edge).
3. Total tiles: $3 \times 3 = 9$, each run through the detector separately
   (plus, in the standard SAHI recipe, one extra pass on the full
   downscaled image, to still catch genuinely large objects that a tile
   might cut in half no matter the overlap).
4. A detection found at local coordinates $(50, 60)$ inside tile $(i=1,
   j=2)$ maps to global coordinates $(50 + 1 \times 512,\ 60 + 2 \times
   512) = (562, 1084)$.

### Worked example 2 — tied to our actual project data
This one is honest about what we do and don't have measured. Our Phase 1
EDA's `results/visualizations/bbox_size_heatmap.png` and
`resolution_histogram.png` already established two real facts: (1) many
road-damage boxes cluster at the small end of the size distribution, and
(2) RDD2022 images come from different phone cameras at varying
resolutions, not a single fixed size (e.g. `India_000027.jpg`, used
throughout these docs, is $720\times720$). At our model's training/
inference resolution (`imgsz: 640` in `configs/model_config.yaml`), a
small crack that's already a small fraction of a $720\times720$ source
image shrinks further once resized to $640\times640$ — the same
underlying small-object risk already flagged in
`docs/learning/06-mosaic-mixup.md` for mosaic's compounding shrink effect,
but this time at inference, not training. SAHI is the inference-side
answer to that same risk: instead of shrinking the whole image to fit the
model, slice it so each tile *is* the model's native resolution, and a
small crack keeps its original pixel footprint instead of being
downscaled away.

## Diagram
```
Full image (small crack near a tile boundary)      With 20% overlap
+------------------+------------------+           +----------------+
|                  |                  |           |   Tile A       |
|        [crack]===|===               |           | [crack]========|==+
|                  |                  |           |                |  |
+------------------+------------------+           +----------------+  |
        (non-overlapping: crack split,                 Tile B ========+
         each half looks incomplete,                (crack fully visible
         may be missed by both tiles)                in at least one tile)
```

## Why it matters for THIS project
SAHI would directly target exactly the risk our own EDA already flagged
(small boxes) and would need zero retraining — pure inference-time
config. It's not implemented here for a genuine, undecided trade-off worth
naming honestly rather than glossing over: SAHI means running the
detector $n_x \times n_y + 1$ times per image instead of once, which is a
real latency cost, in tension with this project's "real-time" framing
(`README.md`'s headline). Whether that trade-off is worth it depends on
the deployment target's actual latency budget — exactly the kind of
decision `CLAUDE.md`'s standing rule would gate behind presenting options
first, which is why this stayed a documented future-work item
(`README.md`'s Future Work) rather than something implemented without
that discussion happening.

## Concept Card
- **(a) In general:** slice a large image into overlapping tiles, run
  the detector on each tile at full resolution, then remap and merge
  detections back into the original image's coordinates — an
  inference-time technique for recovering small-object detail lost to
  downscaling.
- **(b) Used here:** **not implemented** — no file in `src/` performs
  tiled inference; every prediction call runs on the whole image resized
  to `imgsz: 640` (`configs/model_config.yaml`). There is no code
  reference to point to, by design (see status note at the top of this
  file).
- **(c) When "train as-is" (the actual current approach) is the wrong
  call instead:** if this project's real per-class results
  (`docs/pm-perspective.md`) showed small/thin classes like D10
  systematically failing on genuinely tiny objects in high-resolution
  source photos — rather than the moderate, broad-based weakness actually
  observed across precision, mAP, and localization — SAHI's latency cost
  would be easier to justify. As measured, the weakness isn't clearly a
  small-object-scale problem specifically, so training as-is remains the
  defensible default until that's tested directly.

## Interview questions

**Q: How does SAHI improve small-object recall without retraining the
model at all?**
A: It changes what pixels the already-trained model sees per forward
pass, not the model itself. By slicing a large image into overlapping
tiles at (or near) the model's native training resolution, a small object
that would have been downscaled into a handful of pixels in the full
image instead keeps its original pixel footprint inside its tile — the
same trained weights simply have an easier detection problem to solve.

**Q: What's the actual cost of using SAHI, and when would you not use it?**
A: Latency and compute: instead of one forward pass per image, SAHI runs
one pass per tile (plus typically one on the full image), so inference
time scales roughly with tile count. For a real-time system with a tight
per-frame latency budget, that multiplicative cost can be disqualifying.
It's most worth it when small-object recall is the binding constraint and
some extra latency is acceptable — e.g. offline batch analysis of road
survey footage, rather than a live in-vehicle feed.

**Q: Besides SAHI, what's another way to address small objects, and how
would you choose between them?**
A: Training at a higher input resolution (larger `imgsz`) also gives
small objects more effective pixels, without SAHI's multi-pass inference
cost — but it increases training-time memory usage and doesn't help a
model that's already trained and deployed. SAHI is the right choice when
you can't or don't want to retrain (e.g. a model already in production);
higher training resolution is the right choice when you're still in the
training phase and have the VRAM budget for it.
