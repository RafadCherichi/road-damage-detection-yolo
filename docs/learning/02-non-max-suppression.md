# 02 — Non-Maximum Suppression (NMS)

## The intuition, before any jargon
Imagine three people all photographing the same pothole from slightly
different angles, then each drawing a box around it on a shared printout.
You'd end up with three overlapping boxes for one real pothole. If a
fourth box appears around a *genuinely different* pothole ten feet away,
you obviously want to keep that one too. The task "clean up the drawing so
each real pothole gets exactly one box" is exactly what a detector needs
done to its raw output: an anchor-free (or anchor-based) detector predicts
a box at *every* grid location near an object, so a single real pothole
typically produces dozens of near-duplicate, overlapping candidate boxes,
each with its own confidence score.

**Non-Maximum Suppression (NMS)** is the cleanup rule: keep the
highest-confidence box for each real object, and throw away any other box
that overlaps it too much (probably describing the same object) — while
keeping boxes that don't overlap much (probably describing a different,
nearby object).

## The math — but the "why" comes first
NMS is a greedy, iterative algorithm, not a single formula — but it leans
entirely on one number, **IoU** (Intersection over Union, the overlap
measure covered in `docs/learning/03-iou-variants.md`):

$$\text{IoU}(b_1, b_2) = \frac{|b_1 \cap b_2|}{|b_1 \cup b_2|}$$

The algorithm, over a set of candidate boxes $\mathcal{B}$ with confidence
scores $s$, and an overlap threshold $\tau$ (commonly $0.5$–$0.7$):

1. Pick the remaining box with the highest score: $b^* = \arg\max_{b \in
   \mathcal{B}} s_b$. Move it to the kept list.
2. Remove every remaining box whose IoU with $b^*$ exceeds $\tau$:
   $\mathcal{B} \leftarrow \mathcal{B} \setminus \{ b \in \mathcal{B} :
   \text{IoU}(b, b^*) > \tau \}$.
3. Repeat until $\mathcal{B}$ is empty.

**Term-by-term breakdown:**
- $b^*$ — the current "winner": the highest-confidence box not yet
  processed. It always survives — it's the reference every remaining box
  gets compared against this round.
- $\tau$ — the overlap threshold. High IoU with $b^*$ (above $\tau$) means
  "this is almost certainly describing the same real object as $b^*$, just
  a slightly worse duplicate" — so it gets suppressed. Low IoU means
  "this is probably a different object" — so it survives to be judged in
  a later round on its own merits.
- The loop repeats because after removing duplicates of $b^*$, the
  *next*-highest remaining score becomes the new $b^*$ for a different
  object.

### Worked example 1 — illustrative numbers
Two boxes, both claiming the same object: Box A = $(x_0,y_0,x_1,y_1) =
(0,0,10,10)$ (score $0.9$), Box B = $(2,2,12,12)$ (score $0.6$), threshold
$\tau = 0.5$.
1. Intersection: overlap region is $x \in [2,10]$, $y \in [2,10]$ → width
   $8$, height $8$, area $= 64$.
2. Areas: $|A| = 10 \times 10 = 100$, $|B| = 10 \times 10 = 100$.
3. Union: $|A| + |B| - \text{intersection} = 100 + 100 - 64 = 136$.
4. $\text{IoU}(A,B) = 64 / 136 \approx 0.471$.

$0.471 < \tau=0.5$, so in this case B would actually **survive** — a
reminder that the threshold is a real, tunable decision boundary, not just
a formality: a slightly looser or tighter $\tau$ changes the outcome here.

### Worked example 2 — tied to our actual project data
Since no model has been trained yet in this project (Phase 3, `train.py`
comes next), there are no real predicted boxes to run NMS on. To make this
concrete without inventing disconnected numbers, we simulate three
plausible raw detections around the real spatial region of the D20 box
from `India_000027.txt` (real box, pixel corners $(12.0, 506.0)$–$(274.0,
663.0)$ on the real 720×720 image) — this is exactly the kind of
near-duplicate cluster a trained detector would produce around that real
crack, plus one distinct box for a hypothetical second, separate pothole
nearby. Threshold $\tau = 0.5$.

| Box | Corners $(x_0,y_0,x_1,y_1)$ | Score | Area |
|---|---|---|---|
| A | $(20, 515, 270, 655)$ | $0.91$ | $250 \times 140 = 35{,}000$ |
| B | $(35, 525, 260, 645)$ | $0.77$ | $225 \times 120 = 27{,}000$ |
| C | $(310, 520, 430, 610)$ | $0.62$ | $120 \times 90 = 10{,}800$ |

**Round 1:** highest score is A ($0.91$) → keep A, compare the rest to it.
- $\text{IoU}(A,B)$: intersection $x \in [35,260], y \in [525,645]$ → width
  $225$, height $120$, area $27{,}000$ (B sits entirely inside A). Union
  $= 35{,}000 + 27{,}000 - 27{,}000 = 35{,}000$.
  $\text{IoU} = 27{,}000/35{,}000 \approx 0.771$. That's $> 0.5$ →
  **suppress B** (almost certainly the same crack as A, just a slightly
  worse duplicate box).
- $\text{IoU}(A,C)$: A spans $x \in [20,270]$, C spans $x \in [310,430]$ —
  these don't overlap in $x$ at all, so intersection area $= 0$.
  $\text{IoU} = 0 < 0.5$ → C survives this round.

**Round 2:** only C remains → keep C (nothing left to compare it against).

**Final kept boxes: A and C.** B was correctly identified as a redundant
duplicate of the same real crack; C, a genuinely separate detection, was
correctly preserved even though it came in with a lower score than the
one that got suppressed.

## Diagram
```
BEFORE NMS (raw detector output)         AFTER NMS
                                          
  .--A(0.91)--.                            .--A(0.91)--.
  |  .-B(0.77)-|.                          |            |
  |  |         ||          -- NMS -->      |            |
  '--'---------''                          '------------'
        (same crack, duplicate)
                                                                .--C(0.62)--.
              .--C(0.62)--.                                    |            |
              |            |                                   |            |
              '------------'                                   '------------'
        (different object, kept)
```

## Why it matters for THIS project
Every YOLOv8 inference call runs NMS automatically (Ultralytics exposes it
as the `iou` argument to `model.predict()`), so this isn't optional
plumbing we build — it's a threshold we may need to *tune* at evaluation
time. Given our EDA already showed D20 (alligator crack) is our most
common and often clustered class, a threshold that's too loose risks
merging genuinely separate nearby cracks into one box; too strict risks
keeping near-duplicate boxes around a single crack and inflating false
positives in our precision/recall numbers. Understanding the mechanism
means a bad `iou` setting shows up as an explainable, fixable number, not
a mysterious accuracy drop.

## Interview questions

**Q: Why is a suppressed box's score irrelevant to whether it survives?**
A: NMS only ever compares *overlap* (IoU) against the current winner, not
score directly — score only decides *processing order* (who becomes
$b^*$ each round). A very high-confidence box can still get suppressed if
it overlaps an even-higher-confidence box past the threshold; a
low-confidence box can survive if it's spatially distinct from every box
processed so far, exactly like Box C surviving with the lowest score of
the three in Worked Example 2.

**Q: What happens if the IoU threshold is set too low or too high?**
A: Too low (aggressive suppression) risks merging two real, nearby objects
into a single kept box, since even moderate overlap gets treated as
"duplicate." Too high (permissive) risks keeping several near-duplicate
boxes around one real object, inflating false positives. The right value
is a genuine tuning decision, usually chosen by checking validation
mAP at a few threshold values.

**Q: NMS is greedy and processes boxes one at a time — what's the
practical downside of that?**
A: It's inherently sequential (each round's outcome depends on which box
won the previous round), so it doesn't parallelize as cleanly as the rest
of a CNN's forward pass, and it can occasionally make a locally-reasonable
but globally-suboptimal choice (e.g., suppressing a box that would have
been a better match for a different, not-yet-considered object).
Alternatives like Soft-NMS (decay score instead of hard removal) address
part of this, but classic greedy NMS remains the default because it's fast
and good enough in practice.
