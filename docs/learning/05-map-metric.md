# 05 — Precision, Recall, and mAP

## The intuition, before any jargon
Imagine a metal detector on a beach. Two failure modes matter, and they
trade off against each other. First: how much of what it beeps at is
*actually* treasure, not a bottle cap? That's **precision** — "of
everything I flagged, how much was real?" Second: of all the treasure
actually buried in the sand, how much did it find? That's **recall** —
"of everything real, how much did I catch?" A detector tuned to beep at
literally anything metallic has perfect recall (finds every coin) but
terrible precision (mostly bottle caps). A detector tuned to only beep
when extremely confident has great precision but misses buried coins —
poor recall. Every object detector, including ours, faces exactly this
same trade-off, controlled by a confidence threshold: how sure does the
model need to be before it "beeps"?

**mAP (mean Average Precision)** is the single number that summarizes
this whole trade-off curve — precision at every possible recall level,
averaged into one score — rather than picking one arbitrary threshold and
reporting just that one point.

## The math — but the "why" comes first
Precision and recall are both built from three counts: **TP** (true
positive — a predicted box that correctly matches a real one), **FP**
(false positive — a predicted box with nothing real there, or the wrong
class), and **FN** (false negative — a real piece of damage the model
missed entirely).

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}$$

**Term-by-term breakdown:**
- $TP + FP$ — every box the model predicted, period. Precision asks what
  fraction of *those* were actually correct.
- $TP + FN$ — every box that should have been predicted (the real ground
  truth count). Recall asks what fraction of *those* the model actually
  found.

A single detector doesn't have one precision/recall number — it has one
*pair* for every possible confidence threshold, since raising or lowering
the threshold changes which boxes count as "predicted" at all. Plotting
precision against recall as the threshold sweeps from strict to loose
traces the **PR curve**. Average Precision is the area under it:

$$AP = \int_0^1 p(r)\, dr \qquad\qquad mAP = \frac{1}{C}\sum_{c=1}^{C} AP_c$$

**Term-by-term breakdown:**
- $p(r)$ — precision as a function of recall (the PR curve itself, using
  the standard "interpolated" precision: at each recall level, the
  *highest* precision achieved at that recall or beyond, which removes
  the small zigzags a single unlucky false positive would otherwise cause).
- $AP$ — the area under that curve for one class: a single number
  capturing "how good is the whole precision/recall trade-off," not just
  one threshold's snapshot.
- $mAP$ — Average Precision, averaged again across all $C$ classes.

**mAP50 vs. mAP50-95** — one more axis: how strict is "correctly
localized"? A prediction only counts as a match if its IoU with the real
box clears a threshold (see `docs/learning/03-iou-variants.md`).
$mAP_{50}$ uses one lenient threshold (IoU ≥ 0.5). $mAP_{50\text{-}95}$
averages mAP across ten thresholds from 0.5 to 0.95:

$$mAP_{50\text{-}95} = \frac{1}{10}\sum_{t \in \{0.50, 0.55, \dots, 0.95\}} mAP_t$$

A model can score well on $mAP_{50}$ (roughly the right place) while
scoring much lower on $mAP_{50\text{-}95}$ (rarely *tightly* boxed) — the
gap between the two numbers is itself informative, not just noise.

### Worked example 1 — illustrative numbers
Suppose one class has 5 real ground-truth boxes, and the model produces 5
predictions, sorted by confidence, that turn out to be
TP, TP, FP, TP, FP (in that order — the model's 3rd and 5th most
confident predictions were wrong).

| Rank | Result | Cumulative TP | Cumulative FP | Precision | Recall |
|---|---|---|---|---|---|
| 1 | TP | 1 | 0 | $1/1=1.00$ | $1/5=0.20$ |
| 2 | TP | 2 | 0 | $2/2=1.00$ | $2/5=0.40$ |
| 3 | FP | 2 | 1 | $2/3=0.667$ | $2/5=0.40$ |
| 4 | TP | 3 | 1 | $3/4=0.75$ | $3/5=0.60$ |
| 5 | FP | 3 | 2 | $3/5=0.60$ | $3/5=0.60$ |

Interpolating (taking the max precision at each recall level or beyond)
and summing rectangle areas as recall increases:
1. Recall $0 \to 0.2$: interpolated precision $=1.00$ (the highest
   precision anywhere at recall $\ge 0.2$). Area $= 0.2 \times 1.00 = 0.20$.
2. Recall $0.2 \to 0.4$: interpolated precision $=1.00$. Area
   $= 0.2 \times 1.00 = 0.20$.
3. Recall $0.4 \to 0.6$: interpolated precision $= \max(0.75, 0.60) =
   0.75$. Area $= 0.2 \times 0.75 = 0.15$.
4. Recall never exceeds $0.6$ (2 of the 5 real boxes were never found at
   all — false negatives), so nothing more is added.

$$AP = 0.20 + 0.20 + 0.15 = 0.55$$

Notice the single false positive at rank 3 didn't directly punish AP the
way a raw (non-interpolated) precision curve would suggest — interpolation
is specifically designed to smooth out that kind of one-off noise.

### Worked example 2 — tied to our actual project data
Real results from `src/evaluate.py` against `yolov8n_kaggle_run` on the
valid split (`results/metrics/metrics_summary.csv`):

| Class | mAP50 | mAP50-95 | Gap (relative) |
|---|---|---|---|
| D20 (Alligator) | 0.858 | 0.539 | $(0.858-0.539)/0.858 \approx 37.2\%$ |
| D10 (Transverse) | 0.653 | 0.309 | $(0.653-0.309)/0.653 \approx 52.7\%$ |

Both classes lose real ground going from lenient (IoU ≥ 0.5) to strict
(averaged over IoU ≥ 0.5 through 0.95) scoring — expected, since
$mAP_{50\text{-}95}$ is a harder bar by construction. But D10's *relative*
gap (52.7%) is meaningfully larger than D20's (37.2%): even in the cases
where the model does find a D10 crack, the predicted box tends to be
looser/less tightly localized than a D20 box tends to be. This lines up
with something already documented in `docs/learning/01-anchor-free-detection.md`
and the Phase 1 EDA: D10 (transverse crack) is thin and elongated, which
is inherently harder to box tightly than D20's blockier alligator-crack
pattern — the mAP50-vs-mAP50-95 gap isn't just an abstract statistic here,
it's numeric confirmation of a shape-difficulty effect we'd already
suspected from the raw data.

## Diagram
```
Precision
   1.0 |*\
       |  \___
       |      \___          <- PR curve: starts high (strict threshold,
   0.5 |          \___         few but confident predictions), falls as
       |              \__      threshold loosens (more predictions,
       |                 \_    more false positives creep in)
   0.0 |___________________\__
       0.0      0.5        1.0   Recall

AP = area under this curve (one number per class)
mAP = AP averaged across all 4 classes (D00, D10, D20, D40)

IoU strictness axis (separate from the above):
mAP50      : count a box as correct if IoU >= 0.50  (lenient)
mAP50-95   : average mAP over IoU >= 0.50, 0.55, ..., 0.95 (strict)
```

## Why it matters for THIS project
This project frames itself as feeding an autonomous-vehicle path planner
(`CLAUDE.md`'s WHY) — which changes how precision and recall should
actually be weighed, not just measured. Missing a real pothole (a false
negative) is a worse outcome for a path planner than flagging a harmless
shadow as possible damage (a false positive): the former risks physical
damage or an unsafe maneuver, the latter just costs a moment of
unnecessary caution. That argues for favoring **recall** over precision
when picking an operating confidence threshold for deployment, not simply
reporting the threshold-agnostic mAP number and calling it done. The real
per-class gap between $mAP_{50}$ and $mAP_{50\text{-}95}$ (worked example
2) also flags D10 specifically as the class most in need of tighter
localization — a concrete, data-backed target for any future
improvement work, rather than a vague "the model could be better" claim.

## Interview questions

**Q: What's the practical difference between mAP50 and mAP50-95, and why
report both?**
A: mAP50 only requires a predicted box to overlap the real one by at
least 50% IoU to count as correct — a fairly forgiving bar. mAP50-95
averages performance across ten IoU thresholds from 0.50 to 0.95, so it
also rewards *tight*, precise localization, not just "roughly in the
right place." A model can look strong on mAP50 while being noticeably
weaker on mAP50-95, which reveals a real localization-quality gap that a
single number would hide.

**Q: For a safety-critical detection task, would you optimize for
precision or recall?**
A: It depends on the cost of each error type, not a fixed rule. For a
road-damage detector feeding a path-planning system, a missed pothole
(false negative) risks physical harm, while a false alarm (false
positive) just costs a brief unnecessary caution. That asymmetry argues
for favoring recall — accepting more false positives to catch more real
damage — by choosing a lower confidence threshold at deployment, informed
by the PR curve rather than an arbitrary default.

**Q: If a class has a much bigger mAP50-to-mAP50-95 gap than the other
classes, what does that tell you, and what would you check next?**
A: It means that class's detections are relatively "found but loosely
boxed" more often than other classes' — the model recognizes the object
but doesn't localize it tightly. The next step is checking whether that
class's ground-truth boxes have a distinguishing shape or size (in this
project, D10's thin, elongated boxes are the concrete example), since
harder-to-box shapes are a more likely explanation than "the model is
generally worse at this class."
