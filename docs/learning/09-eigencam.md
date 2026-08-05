# 09 — EigenCAM (and why it failed on this project)

## The intuition, before any jargon
Imagine handing someone a photo and asking, "without telling you what to
look for, just point at whatever part of this photo changes the most as
I show you similar photos." On a clean, controlled studio photo, that
might genuinely land on the interesting object. But hand them a hundred
outdoor road photos, and the part that varies the most from photo to
photo is often the *sky* — sometimes bright, sometimes overcast,
sometimes glaring off wet pavement — while the actual road damage in each
photo is a comparatively small, consistent, subtle detail. The person
would keep pointing at the sky, not because it's wrong about "what
varies most," but because you never told them *what you actually cared
about*.

That's exactly **EigenCAM**: a way of visualizing "what does a CNN layer
pay attention to" using pure statistics (PCA) on the layer's activations,
with **no class label and no backpropagation at all** — genuinely fast
and architecture-agnostic, but blind to the one thing that actually
matters for an explanation: *which class, in which specific detection,
are we trying to explain?*

## The math — but the "why" comes first
Take a convolutional layer's output feature map $A$ with $C$ channels
over an $H \times W$ spatial grid. Reshape it so each of the $H \times W$
spatial locations is one row, with $C$ channel-values as columns, giving
matrix $M \in \mathbb{R}^{(HW) \times C}$. EigenCAM finds the single
direction in that $C$-dimensional channel space along which the
activations vary the *most* across the image — the first principal
component — and uses each location's alignment with that direction as
the saliency value:

$$L_{\text{EigenCAM}} = M v_1$$

**Term-by-term breakdown:**
- $M$ — the reshaped feature map: one row per spatial location, one
  column per channel.
- $v_1$ — the first principal component (top eigenvector of $M^\top M$,
  found via SVD): the one direction in channel-space that captures more
  of the activations' spread than any other direction.
- $L_{\text{EigenCAM}}$ — the resulting per-location saliency map: how
  strongly each spatial location's activation pattern aligns with $v_1$,
  reshaped back to $H \times W$ and overlaid as a heatmap.

Critically: nothing here ever looks at a class score or a detected box.
$v_1$ is found purely from how much the activations *statistically vary*
across the image — it has no notion of "crack" or "background," only
"what's the biggest source of variation, whatever that happens to be."

### Worked example 1 — illustrative numbers
A tiny feature map, 4 spatial locations, 2 channels each:

| Location | Channel 1 | Channel 2 |
|---|---|---|
| (0,0) | 10 | 5 |
| (0,1) | 2 | 5 |
| (1,0) | 8 | 5 |
| (1,1) | 4 | 5 |

1. Channel 2 is identical ($5$) everywhere — zero variance, contributes
   nothing to "what varies."
2. Channel 1 varies a lot ($10, 2, 8, 4$; mean $= 6$). Since channel 2
   contributes zero spread, the direction of maximum variance $v_1$
   points almost entirely along the channel-1 axis: $v_1 \approx [1, 0]$
   — no need to solve a full eigenvalue problem here, since one channel's
   variance is trivially zero.
3. Mean-center channel 1: $4, -4, 2, -2$. Projecting onto $v_1 = [1,0]$
   just returns these same values (channel 2's contribution is exactly
   $0$ regardless).
4. Min-max normalize to $[0,1]$ (min $=-4$, max $=4$): location $(0,0)
   \to 1.0$ (brightest), $(1,0) \to 0.75$, $(1,1) \to 0.25$, $(0,1) \to
   0.0$ (dimmest).

The heatmap simply lights up wherever channel 1's value is highest —
exactly the mechanism, whether or not channel 1 happens to correspond to
anything semantically meaningful.

### Worked example 2 — tied to our actual project data
This is a real, observed failure, not a hypothetical: `results/eigencam_outputs/`
contains 5 real heatmaps generated on actual validation images (e.g.
`heatmap_India_000011.jpg`) during this project's development. Per
`README.md`'s own documented finding: *"On outdoor road scenes, sky/glare
regions dominate feature variance, so EigenCAM consistently highlighted
the sky instead of the road."* This is precisely Worked Example 1's
mechanism playing out at full scale: across a batch of real outdoor road
photos, sky brightness and glare vary enormously from photo to photo (and
even within a photo, sky-to-pavement is a huge activation swing), while
an actual crack's activation signature is comparatively small and
localized. PCA has no way to prefer "the subtle, class-relevant thing"
over "the big, dominant-but-irrelevant thing" — it was never given the
concept of "class-relevant" at all. This is exactly what motivated
`src/explainability.py`'s custom per-detection Grad-CAM: it fixes the
problem by backpropagating from one specific detection's actual class
logit, so the heatmap answers "what made the model say *crack* right
here," not "what's statistically the loudest signal anywhere in this
photo."

## Diagram
```
EigenCAM (no class, no backward pass)         Grad-CAM (this project's fix)
                                          
  feature map A [C,H,W]                        feature map A [C,H,W]
        |                                             |
        v                                             v
  reshape to [H*W, C]                         backprop from ONE detection's
        |                                     class logit (docs/learning/
        v                                     01-anchor-free-detection.md's
  PCA -> top eigenvector v1                   anchor-point machinery)
  (biggest variance direction,                        |
   no idea what's "interesting")                      v
        |                                     gradient-weighted activation
        v                                     map, conditioned on that
  project every pixel onto v1                 SPECIFIC class + detection
        |                                             |
        v                                             v
  heatmap: often = sky/glare              heatmap: correctly isolates
  (biggest variance, wrong reason)         the actual damage region
```

## Why it matters for THIS project
This file is the direct origin story for this project's headline
explainability feature. EigenCAM was the natural first thing to try — the
fastest, simplest CAM method, needing no gradients — and it failed
specifically, observably, and repeatably on outdoor road photos, with
real saved evidence in `results/eigencam_outputs/`. That concrete failure
is what justified building `src/explainability.py`'s custom
per-detection Grad-CAM instead of settling for "good enough" — the
project's Grad-CAM approach isn't an arbitrary architectural preference,
it's a direct, evidence-backed response to a method that demonstrably
didn't work for this specific domain.

## Concept Card
- **(a) In general:** EigenCAM visualizes CNN attention via PCA on
  activations alone — no class label, no backward pass, fast and
  architecture-agnostic, but blind to which class or detection is
  actually being explained.
- **(b) Used here:** **tried and rejected** — the 5 real images in
  `results/eigencam_outputs/*.jpg` are its actual output on this
  project's validation images, kept as documented evidence of the
  failure, not as the chosen explainability method. The method that
  replaced it, `src/explainability.py`'s `DetectionGradCAM` class, is
  what's actually used in this project today.
- **(c) When EigenCAM would have been the better choice instead:** on
  imagery without a single dominant, class-irrelevant variance source —
  e.g. controlled/studio photos, or a task where "what does this layer
  generally attend to" (not tied to one specific class) is the actual
  question being asked. Outdoor road photos, with sky/glare as a huge,
  irrelevant variance source, are close to a worst case for this method.

## Interview questions

**Q: Why is EigenCAM sometimes described as "class-agnostic," and is that
a strength or a weakness?**
A: Both, depending on the use case. It's a strength because it needs no
class label, no target detection, and no backward pass — just a forward
pass and an SVD, so it's fast and works on any CNN layer regardless of
architecture. It's a weakness the moment you actually need to know *why
this specific class was predicted here* — since EigenCAM has no concept
of class at all, it can only show "what activation pattern dominates,"
which on our road-scene data was consistently sky/glare, not the damage
we actually cared about.

**Q: Why did EigenCAM specifically fail on road-damage images rather than
working reasonably everywhere?**
A: Outdoor road photos have a large, consistent source of high-variance
signal — sky brightness, glare, lighting shifts — that's unrelated to the
actual damage. PCA finds whatever direction has the most variance,
full stop; on this kind of imagery, that's almost always the sky/lighting
axis, not the comparatively subtle, localized crack signal. The failure
is a direct, predictable consequence of the method's core mechanism, not
a bug or a one-off unlucky case.

**Q: How does the custom Grad-CAM in this project specifically fix
EigenCAM's failure mode?**
A: By backpropagating from one specific accepted detection's own class
logit (see `src/explainability.py` and
`docs/learning/01-anchor-free-detection.md` for how a detection maps to
an anchor point and scale), the resulting gradient-weighted heatmap is
mathematically tied to "what evidence supports *this* detection being
*this* class" — sky pixels get no special treatment unless they actually
influenced that specific class score, which they generally don't.
