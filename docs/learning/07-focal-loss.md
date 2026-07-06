# 07 — Focal Loss

## The intuition, before any jargon
Picture a strength coach watching you do 100 reps of an exercise. Ninety-five
of those reps are ones you already do perfectly — the coach barely needs to
watch them, they add nothing to your training. Five reps are the ones where
your form breaks down. A good coach spends nearly all their attention on
those five, not evenly across all 100.

Standard training loss doesn't work like a good coach — it works like a
coach who watches every rep with equal attention. In our dataset, roughly
**39% of training images have no damage in them at all** (empty label
files) — these are "easy reps": the model quickly learns to say "no damage
here" with high confidence, and once it does, grading that prediction
teaches it almost nothing new. Meanwhile the genuinely hard cases — a faint
D10 transverse crack that looks a lot like a shadow, or a rare class with
fewer examples — get *the same weight* in a standard loss as those 9,500
easy backgrounds, even though they're what the model actually still needs
to learn from. **Focal Loss** is the fix: a loss function that
automatically pays less attention to examples the model already nails, and
proportionally more attention to the ones it's still unsure about.

The term "focal" is literal — it focuses training effort onto the hard
examples, like a lens focusing light onto one point instead of spreading it
evenly.

## The math — but the "why" comes first
Standard classification loss, **cross-entropy (CE)**, for one example is:

$$\text{CE}(p_t) = -\log(p_t)$$

Here $p_t$ is *the model's predicted probability for the correct answer*.
If the true answer is "pothole" and the model outputs 90% confidence for
"pothole," then $p_t = 0.9$. Notice: the more confident and correct the
model is, the closer $p_t$ gets to 1, and $-\log(p_t)$ gets close to 0 (low
loss, as expected — it did well). The less confident or more wrong, the
smaller $p_t$ is, and $-\log(p_t)$ grows (high loss).

The problem CE doesn't solve: it doesn't grow *fast enough* when $p_t$ is
already high. Focal Loss adds two multipliers to fix that:

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \log(p_t)$$

**Term-by-term breakdown:**
- $p_t$ — the model's predicted probability for the *true* class of this
  example (same as in CE above).
- $\log(p_t)$ — the standard cross-entropy term: small when the model is
  confidently correct, large when it's wrong or unsure.
- $\gamma$ (gamma, the **focusing parameter**, typically $2$) — controls
  how aggressively easy examples get down-weighted. Larger $\gamma$ =
  more aggressive focusing.
- $(1 - p_t)^{\gamma}$ — the **focusing term**. If the model is already
  confident and correct ($p_t$ close to $1$), then $(1-p_t)$ is close to
  $0$, and raising a small number to a power makes it *even smaller* —
  the loss for that easy example gets crushed toward zero. If the model
  is unsure or wrong ($p_t$ small), $(1-p_t)$ stays close to $1$, and the
  loss barely shrinks — hard examples keep close to their full,
  undiminished loss.
- $\alpha_t$ (alpha, the **class-balancing weight**) — a separate,
  per-class scalar that directly rebalances by how frequent that class is
  in the training data, independent of how "hard" any single example is.
  This is the knob that encodes raw class-frequency imbalance (as opposed
  to $\gamma$, which encodes per-example difficulty).

Setting $\gamma = 0$ turns $(1-p_t)^{0}$ into $1$ for every example —
meaning Focal Loss becomes *exactly* standard cross-entropy (times
$\alpha_t$). Focal Loss doesn't replace CE, it generalizes it; CE is just
the "$\gamma=0$, don't focus" special case.

### Worked example 1 — narrating every step, not just the numbers
Let $\gamma = 2$ and, for this first pass, $\alpha_t = 1$ (i.e. ignore
class balancing for a moment and isolate what focusing alone does).
Compare an easy background image against a hard, ambiguous crack image.

**Easy example (background, correctly and confidently classified):**
$p_t = 0.99$
1. Standard CE: $-\log(0.99) = 0.01005$ — already a very small loss.
2. Focusing term: $(1 - 0.99)^{2} = (0.01)^{2} = 0.0001$.
3. Focal loss: $\text{FL} = 1 \times 0.0001 \times 0.01005 \approx 0.000001$.
   The focusing term shrunk an already-small loss by another 10,000x. This
   example now contributes almost nothing to the gradient — exactly what
   we want, since the model has nothing left to learn from it.

**Hard example (ambiguous crack, model unsure):**
$p_t = 0.4$
1. Standard CE: $-\log(0.4) = 0.9163$.
2. Focusing term: $(1 - 0.4)^{2} = (0.6)^{2} = 0.36$.
3. Focal loss: $\text{FL} = 1 \times 0.36 \times 0.9163 \approx 0.330$. The
   focusing term only shrank this loss to about a third of its CE value —
   it stays large because the model is still unsure.

**Why this matters, compared side-by-side:** under plain CE, the hard
example already got about $0.9163 / 0.01005 \approx 91\times$ more loss
than the easy one. Under Focal Loss, that gap widens to
$0.330 / 0.000001 \approx 330{,}000\times$. Focal Loss doesn't just
prioritize the hard example — it makes the easy, already-solved example's
contribution *functionally disappear*, freeing up essentially all of the
training signal for the cases that still matter.

### Worked example 2 — tied to our actual project data
Now bring in $\alpha_t$ using our *real* Phase-1 EDA train-split counts:
D00 = 3,767, D10 = 2,814, D20 = 5,617, D40 = 3,740 annotations. D10 is the
rarest class, so a common inverse-frequency scheme sets each class's
weight relative to the rarest one:

$$\alpha_c = \frac{n_{\min}}{n_c}$$

where $n_{\min} = 2{,}814$ (the D10 count) and $n_c$ is that class's own
count. Computing this for D10 and D20:

$$\alpha_{D10} = \frac{2{,}814}{2{,}814} = 1.0 \qquad\qquad \alpha_{D20} = \frac{2{,}814}{5{,}617} \approx 0.501$$

D10, being the rarest, gets full weight ($1.0$); D20, being about twice as
common, gets roughly half weight ($0.501$) — the weight directly encodes
how much rarer D10 is than D20 in our actual dataset.

Now suppose the model is equally unsure about a D10 example and a D20
example — both get $p_t = 0.5$ — with $\gamma = 2$ as before:

$$\text{FL}_{D10} = -(1.0)(1-0.5)^{2}\log(0.5) = -(1.0)(0.25)(-0.6931) \approx 0.1733$$

$$\text{FL}_{D20} = -(0.501)(1-0.5)^{2}\log(0.5) = -(0.501)(0.25)(-0.6931) \approx 0.0868$$

$$\frac{\text{FL}_{D10}}{\text{FL}_{D20}} = \frac{0.1733}{0.0868} \approx 1.996$$

That ratio, $\approx 2.0$, is not a coincidence — it's almost exactly our
real D20:D10 annotation ratio ($5{,}617/2{,}814 \approx 1.996$). At equal
model confidence, $\alpha_t$ directly injects our dataset's actual
imbalance ratio into the loss, so the rarer class's mistakes count about
twice as much toward the gradient — on top of whatever extra weight
$\gamma$'s focusing already adds if that example is also individually
hard.

## Diagram
```
Loss
 |                                            CE (gamma = 0)
 |...                                       ,,'
 |    ...                                ,,'
 |       ...                          ,,'
 |          ...                    ,,'          FL (gamma = 2)
 |             ...              ,,'         _.-'
 |                ...       ,,'        _.-''
 |                   ... ,,'      _.-''
 |______________________''__.-''___________________ p_t (model confidence
 0                0.5                1.0            in the correct class)

At p_t near 1 (easy, confident-correct):  FL curve flattens near 0 fast.
At p_t near 0 (hard, wrong/unsure):       FL curve stays close to CE.
```
(Kept as ASCII here since this is a curve *shape* illustration, not a
formula — the equations themselves are the LaTeX above.)

## Why it matters for THIS project
Two Phase-1 EDA numbers originally motivated choosing Focal Loss over
oversampling or a custom weighted sampler (see `docs/blueprint.md` Decision
2.1):
- **39% of train images are empty-label backgrounds.** Without focusing,
  this flood of easy negatives would dominate the average loss and drown
  out learning signal from the actual damage classes.
- **D10 (transverse crack) is the least-represented class** (2,814
  annotations vs. D20's 5,617 in train) — a moderate ~2:1 imbalance, not
  severe enough to justify a custom weighted sampler, but exactly the kind
  of "some classes are harder to learn because they're rarer" situation
  Focal Loss's focusing behavior helps with.

**Correction, discovered when the first real training run crashed:**
everything above is still correct as *background theory*, but it is **not
what actually runs in this project**. `fl_gamma` was assumed to be a live
Ultralytics hyperparameter based on how earlier YOLO releases worked; the
installed `ultralytics==8.4.84` was checked directly
(`ultralytics/utils/loss.py`) and its `v8DetectionLoss` — the class
actually used for standard detection training — computes classification
loss with plain `nn.BCEWithLogitsLoss`. The `FocalLoss`/`VarifocalLoss`
classes still exist in that file but are never instantiated there in this
version; `fl_gamma` isn't a recognized training argument at all anymore
and crashes immediately if passed.

There is a newer, different lever — `v8DetectionLoss` reads an optional
`model.class_weights` tensor and multiplies it into the classification
loss, which is actually a closer match to this file's $\alpha_t$ (a true
per-class weight) than $\gamma$-based focusing ever was. But it has no
YAML exposure; it can only be set from Python (e.g. a training callback),
which is real custom code, not a config value.

**Decision:** for this project's baseline training run, we use Ultralytics'
plain BCE loss as-is — no explicit rebalancing — since our imbalance is
moderate (~2:1), not severe, and YOLOv8's `TaskAlignedAssigner` already
handles foreground/background matching differently from the
anchor-based detectors Focal Loss was originally designed to fix. If
validation results later show D10 specifically underperforming, the
`class_weights` callback above is the concrete next thing to try — not
`fl_gamma`, which is not just renamed but genuinely gone.

## Interview questions

**Q: Why choose Focal Loss over oversampling or class-weighted sampling for
a 2:1 imbalance?**
A: Class weighting/oversampling rebalance *how often* each class is seen,
but don't address *within-class* difficulty — a flood of easy backgrounds
still dominates training regardless of how the rarer class is resampled,
and with only ~2,814 D10 instances, duplicating images risks overfitting on
repeated pixels rather than learning genuinely new signal. Focal Loss
instead reweights by how hard each individual example currently is,
addressing both the background/foreground imbalance and the moderate
class imbalance in one mechanism. (For this specific project: we later
discovered the installed Ultralytics version doesn't actually expose Focal
Loss as a hyperparameter — see the correction above — so we're currently
running plain BCE and treating this as background theory plus a documented
fallback plan, not as this project's active mechanism.)

**Q: What happens to Focal Loss if $\gamma = 0$?**
A: It becomes exactly standard cross-entropy (scaled by $\alpha_t$) —
$(1-p_t)^{0} = 1$ for every example, so the focusing multiplier
disappears. This shows Focal Loss is a generalization of CE, not a
separate, incompatible idea; $\gamma$ is a dial between "treat every
example equally" ($0$) and "focus almost entirely on hard examples"
(higher values, e.g. $2$–$5$).

**Q: How does Focal Loss fit into YOLOv8's overall loss function?**
A: YOLOv8's total loss is a weighted sum of three parts: box regression
(CIoU loss), classification confidence (a BCE loss — historically
focal-modulated via `fl_gamma` in older YOLO releases, though the
installed version here uses plain BCE, confirmed by reading
`ultralytics/utils/loss.py` directly), and Distribution Focal Loss (DFL)
for the box coordinate regression. Note DFL is a *different* mechanism
despite the similar name — it's about representing box coordinates as a
learned probability distribution over discretized bins, not about
down-weighting easy examples, and DFL is still active in this version
regardless of the classification-side Focal Loss question.
