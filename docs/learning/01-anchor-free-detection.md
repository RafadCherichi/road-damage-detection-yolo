# 01 — Anchor-Free Detection

## The intuition, before any jargon
Imagine you're asked to point out every pothole in a photo, but you're only
allowed to describe each one using a small set of pre-printed stencil
shapes — "small square," "wide rectangle," "tall rectangle" — and you have
to say *which stencil* plus *how much to stretch it* to match the real
pothole. That's **anchor-based detection** (used by YOLOv3 through YOLOv5):
at every location in the image, the model is handed a fixed set of
pre-defined box shapes (**anchors**, e.g. 3 different width/height
combinations per grid cell), and its job is just to say "shape #2, but 20%
wider and 10% shorter" relative to that stencil.

The problem: those stencil shapes have to be pre-decided *before training
even starts*, usually by clustering the shapes of boxes in your training
set (k-means on widths/heights). If your dataset's real objects don't
match those stencils well — which is a real risk for a new dataset like
ours — the model spends effort correcting a bad starting guess for every
single prediction, and you've added a whole extra tuning step (get the
anchor shapes right) before training even begins.

**Anchor-free detection** (used by YOLOv8) throws the stencils away
entirely. Instead, at every location, the model directly answers a
simpler question: "if there's an object centered near here, how far away
is its left edge, top edge, right edge, and bottom edge?" No pre-defined
shapes, no clustering step, no stencil to match — just four distances,
predicted fresh for every location, for every image.

## The math — but the "why" comes first

### Anchor-based: adjust a stencil
Each anchor has a fixed prior width/height ($p_w, p_h$). The network
predicts small offsets ($t_x, t_y, t_w, t_h$) that stretch/shift that prior
into the final box, relative to grid cell corner $(c_x, c_y)$:

$$b_x = \sigma(t_x) + c_x \qquad b_y = \sigma(t_y) + c_y \qquad b_w = p_w e^{t_w} \qquad b_h = p_h e^{t_h}$$

**Term-by-term breakdown:**
- $c_x, c_y$ — the grid cell's top-left corner, in grid units.
- $\sigma(t_x), \sigma(t_y)$ — a sigmoid-squashed offset (kept between 0
  and 1) that places the box center somewhere inside this specific grid
  cell, not outside it.
- $p_w, p_h$ — the anchor's pre-defined prior width/height — the "stencil."
- $t_w, t_h$ — the predicted log-scale stretch factor applied to that
  stencil. Using $e^{t_w}$ (not just $t_w$) guarantees the result is always
  positive — a width can never come out negative.
- $b_x, b_y, b_w, b_h$ — the final predicted box, in grid units (later
  rescaled to pixels).

### Anchor-free: predict raw distances
No prior box at all. Each location $(x, y)$ (called an **anchor point** —
just a coordinate, not a shape) directly predicts how far the box's four
edges are from that point:

$$x_0 = x - l \qquad y_0 = y - t \qquad x_1 = x + r \qquad y_1 = y + b$$

**Term-by-term breakdown:**
- $(x, y)$ — the anchor point: a specific pixel location on the image,
  corresponding to one cell of the model's output feature map.
- $l, t, r, b$ — the four predicted distances (left, top, right, bottom)
  from that anchor point to each edge of the box. These are the *only*
  numbers the network predicts for box shape — no width/height stencil
  involved.
- $(x_0, y_0), (x_1, y_1)$ — the box's resulting top-left and bottom-right
  corners.

(YOLOv8 additionally represents $l,t,r,b$ each as a small learned
probability distribution over discrete distance bins rather than one raw
number — that refinement is **Distribution Focal Loss (DFL)**, already
noted in `docs/learning/07-focal-loss.md` as a separate mechanism from
classification Focal Loss. The core anchor-free idea — predict distances
to edges, not stencil adjustments — is unchanged either way.)

### Worked example 1 — illustrative numbers

**Anchor-based:** Grid cell $(c_x, c_y) = (3, 4)$, anchor prior
$(p_w, p_h) = (10, 10)$ (grid units), predicted offsets
$t_x=0.2, t_y=-0.1, t_w=0.4, t_h=0.3$ (values arriving already
sigmoid/exp-ready for clarity).
1. $b_x = \sigma(0.2) + 3 \approx 0.550 + 3 = 3.550$
2. $b_y = \sigma(-0.1) + 4 \approx 0.475 + 4 = 4.475$
3. $b_w = 10 \times e^{0.4} \approx 10 \times 1.492 = 14.92$
4. $b_h = 10 \times e^{0.3} \approx 10 \times 1.350 = 13.50$

The stencil (10×10) got stretched to roughly 14.92×13.50 — the model's job
was only ever "how much to stretch/shift the stencil," never "what shape
should this be from nothing."

**Anchor-free:** anchor point $(x, y) = (112, 144)$ pixels, predicted
distances $l=40, t=30, r=35, b=25$ pixels.
1. $x_0 = 112 - 40 = 72$
2. $y_0 = 144 - 30 = 114$
3. $x_1 = 112 + 35 = 147$
4. $y_1 = 144 + 25 = 169$

Final box: corners $(72, 114)$ to $(147, 169)$ — built entirely from four
predicted distances, no stencil involved at any step.

### Worked example 2 — tied to our actual project data
Take the same real D20 (alligator crack) annotation used elsewhere in
these docs: `India_000027.txt`, label `2 0.1986 0.8118 0.3639 0.2181` on a
720×720 image. In pixels: $x_c \approx 143.0, y_c \approx 584.5, w \approx
262.0, h \approx 157.0$, giving corners $x_0=12.0, y_0=506.0, x_1=274.0,
y_1=663.0$.

YOLOv8's smallest-stride detection level (stride $s=8$) assigns an anchor
point to the center of whichever grid cell contains the box's true center.
Grid cell index: $\lfloor 143.0/8 \rfloor = 17$, $\lfloor 584.5/8 \rfloor
= 73$. That cell's center in pixels: $(17.5 \times 8,\ 73.5 \times 8) =
(140.0,\ 588.0)$ — this is the anchor point $(x, y)$ actually used.

Computing the four target distances this anchor point would need to
predict to exactly reconstruct our real box:
1. $l = x - x_0 = 140.0 - 12.0 = 128.0$
2. $t = y - y_0 = 588.0 - 506.0 = 82.0$
3. $r = x_1 - x = 274.0 - 140.0 = 134.0$
4. $b = y_1 - y = 663.0 - 588.0 = 75.0$

Check by reversing the formula: $x_0 = 140.0 - 128.0 = 12.0$ ✓,
$x_1 = 140.0 + 134.0 = 274.0$ ✓ — matches our real box exactly. Notice
$l \ne r$ ($128.0$ vs $134.0$) and $t \ne b$ ($82.0$ vs $75.0$): the anchor
point isn't exactly centered in the real box (it's snapped to the nearest
grid-cell center), so the four distances are genuinely asymmetric — the
model has to learn each one independently, which is exactly what
anchor-free regression is designed to handle.

## Diagram
```
ANCHOR-BASED (YOLOv3-v5)              ANCHOR-FREE (YOLOv8)
+-------------------+                 +-------------------+
| grid cell (c_x,c_y)|                | anchor point (x,y)|
|  [===] [--]  (oo)  |                |         .          |
|  3 preset stencils |                |         .  <- predicts
|  model picks one + |                |    l <--*--> r      |
|  stretches it      |                |         .          |
+-------------------+                 |         t          |
                                       |         .  b       |
predict: t_x,t_y,t_w,t_h               +-------------------+
(offsets from a stencil)               predict: l, t, r, b
                                        (raw distances to edges)
```

## Why it matters for THIS project
Our EDA bbox-size heatmap already showed road-damage boxes vary a lot in
shape — thin, elongated cracks and blockier potholes coexist in the same
4-class dataset. Anchor-based detection would require us to run k-means on
our own box shapes to pick good anchor priors, and a badly-fit anchor set
would silently hurt small/oddly-shaped classes like D10 (transverse
crack) the most. YOLOv8's anchor-free head removes that whole tuning step:
one less hyperparameter surface to get wrong on a dataset this size, and
one less thing to defend/explain if results look off. YOLOv8's head is
also *decoupled* (separate branches for "what class" vs "where exactly"),
a related but distinct architectural choice from being anchor-free —
worth knowing the two are independent ideas that happen to appear
together in YOLOv8.

## Concept Card
- **(a) In general:** anchor-free detection predicts each box as raw
  distances from a point to its four edges; anchor-based detection
  predicts offsets from a pre-clustered set of stencil shapes.
- **(b) Used here:** not a separate toggle — it's inherent to choosing
  YOLOv8 at all. `configs/model_config.yaml`'s `architecture: yolov8`
  line is the actual decision point; every `YOLO()` instantiation in
  `src/train.py`, `evaluate.py`, `inference.py`, and `explainability.py`
  inherits the anchor-free head automatically from that one choice.
- **(c) When anchor-based would win:** if this project's box-shape
  distribution were tight and well-understood in advance (e.g. a
  single, consistent object class/aspect ratio), a well-fit anchor set
  can converge faster since the model starts closer to the right answer.
  Our 4-class, visually varied dataset (per the EDA bbox-size heatmap) is
  exactly the opposite case — anchor-free avoids needing to get that
  clustering right at all.

## Interview questions

**Q: Why did YOLO move from anchor-based (v3–v5) to anchor-free (v8)?**
A: Anchor-based detection requires pre-computing a fixed set of box
"stencils" (usually via k-means clustering on the training set's box
shapes) before training even starts. If those stencils don't match your
actual data's shape distribution well, every prediction starts from a
worse initial guess, and you've added an extra tuning step with its own
failure mode. Anchor-free detection predicts box edges directly as
distances from a point, removing that whole pre-tuning step and its
associated risk, at the cost of the network having to learn a slightly
less constrained regression target.

**Q: What's a concrete failure mode of anchor-based detection that
anchor-free avoids?**
A: If your dataset has an unusual aspect-ratio distribution (e.g. our
transverse cracks are long and thin) that wasn't well represented in
whatever anchor-clustering process was used, the model has to fight its
own stencils on every prediction for that class, systematically
under- or over-shooting. Anchor-free removes that mismatch risk entirely
since there's no stencil to begin with.

**Q: Is "anchor-free" the same thing as YOLOv8's "decoupled head"?**
A: No — they're independent design choices that happen to both appear in
YOLOv8. Anchor-free is about *how box coordinates are parameterized*
(distances from a point vs. offsets from a stencil). Decoupled head is
about *whether classification and box-regression share the same final
layers* (YOLOv8 splits them into separate branches). You could have an
anchor-based model with a decoupled head, or an anchor-free model with a
shared head — YOLOv8 just uses both improvements together.
