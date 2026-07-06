# 03 — IoU, GIoU, DIoU, CIoU

## The intuition, before any jargon
Imagine laying two transparent rectangular stickers on top of each other —
one is where the model *thinks* the crack is, one is where the crack
*actually* is. The most natural way to score "how good is this guess?" is:
how much of the combined sticker area do they actually share? That's
**Intersection over Union (IoU)**: overlap area divided by total covered
area. A perfect prediction gives IoU $=1$ (identical stickers); a
completely wrong one gives IoU $=0$ (no overlap at all).

Here's the catch that motivated three follow-up variants: if two boxes
don't overlap *at all*, IoU is exactly $0$ whether they're one pixel apart
or on opposite sides of the image. As a training signal, that's a dead
zone — the model gets zero gradient information telling it "you're
close, keep nudging this direction" versus "you're wildly wrong." GIoU,
DIoU, and CIoU each add a correction term specifically to fix that dead
zone, using progressively more geometric information (how far apart, in
what direction, what shape).

## The math — but the "why" comes first

**IoU** — the base measure:

$$\text{IoU} = \frac{|B \cap B_{gt}|}{|B \cup B_{gt}|}$$

**Term-by-term breakdown:** $B$ is the predicted box, $B_{gt}$ is the
ground-truth box, $|B \cap B_{gt}|$ is their overlapping area, and
$|B \cup B_{gt}|$ is the total area covered by either box.

**GIoU** (Generalized IoU) — penalizes wasted space in the smallest box
that contains both:

$$\text{GIoU} = \text{IoU} - \frac{|C \setminus (B \cup B_{gt})|}{|C|}$$

**Term-by-term breakdown:** $C$ is the smallest enclosing box that fully
contains both $B$ and $B_{gt}$. $|C \setminus (B \cup B_{gt})|$ is the
"empty" area inside $C$ that neither box covers — the more space the two
boxes waste inside their smallest common bounding box, the more GIoU gets
penalized below plain IoU, even at IoU $=0$, since $C$ still shrinks as
the boxes get closer together.

**DIoU** (Distance IoU) — penalizes how far apart the box *centers* are,
directly:

$$\text{DIoU} = \text{IoU} - \frac{\rho^2(b, b_{gt})}{c^2}$$

**Term-by-term breakdown:** $b, b_{gt}$ are the center points of the
predicted and ground-truth boxes. $\rho^2(b, b_{gt})$ is the squared
Euclidean distance between those centers. $c$ is the diagonal length of
the enclosing box $C$ (same $C$ as GIoU) — used to normalize the distance
so it's scale-independent. This gives a direct, more targeted "get closer"
signal than GIoU's more indirect wasted-space penalty.

**CIoU** (Complete IoU) — adds one more term, penalizing mismatched aspect
ratio (width-to-height shape):

$$\text{CIoU} = \text{DIoU} - \alpha v \qquad
v = \frac{4}{\pi^2}\left(\arctan\frac{w_{gt}}{h_{gt}} - \arctan\frac{w}{h}\right)^2 \qquad
\alpha = \frac{v}{(1-\text{IoU}) + v}$$

**Term-by-term breakdown:** $w, h$ are the predicted box's width/height;
$w_{gt}, h_{gt}$ are the ground-truth box's. $v$ measures how differently
*shaped* (not sized — arctan of a ratio ignores absolute scale) the two
boxes are; it's $0$ when the aspect ratios match exactly. $\alpha$ is a
weight that automatically turns this penalty down when IoU is already
high (the boxes mostly overlap, so shape mismatch matters less) and up
when IoU is low.

### Worked example 1 — illustrative numbers
Box $P = (0,0,4,4)$ (a $4\times4$ square), Box $Q = (2,2,6,6)$ (a
$4\times4$ square, shifted).
1. Intersection: $x \in [2,4], y \in [2,4]$ → $2 \times 2 = 4$.
2. Areas: $|P| = |Q| = 16$. Union $= 16+16-4 = 28$.
3. $\text{IoU} = 4/28 \approx 0.1429$.
4. Enclosing box $C$: $x \in [0,6], y \in [0,6]$ → area $= 36$.
5. $\text{GIoU} = 0.1429 - (36-28)/36 = 0.1429 - 0.2222 \approx -0.0794$ —
   negative, even though the boxes do overlap, because a third of $C$'s
   area is wasted space neither box touches.
6. Centers: $P=(2,2)$, $Q=(4,4)$. $\rho^2 = (4-2)^2+(4-2)^2 = 8$.
   Diagonal $c^2 = 6^2+6^2 = 72$.
7. $\text{DIoU} = 0.1429 - 8/72 \approx 0.1429 - 0.1111 \approx 0.0317$.
8. Both boxes are perfect squares (aspect ratio $1{:}1$ each), so
   $\arctan(4/4) - \arctan(4/4) = 0 \Rightarrow v = 0 \Rightarrow \alpha =
   0$. $\text{CIoU} = \text{DIoU} - 0 = 0.0317$ — identical to DIoU here.
   The aspect-ratio penalty only ever activates when the two boxes are
   genuinely differently *shaped*, not just differently placed.

### Worked example 2 — tied to our actual project data
Ground truth: the real D20 box from `India_000027.txt`, pixel corners
$(12.0, 506.0)$–$(274.0, 663.0)$ (width $262$, height $157$, area
$41{,}134$). Since no model has been trained yet, we pair it with a
plausible hypothetical predicted box — corners $(180, 560)$–$(400, 700)$
(width $220$, height $140$, area $30{,}800$) — to make the calculation
concrete.

1. Intersection: $x \in [180,274], y \in [560,663]$ → width $94$, height
   $103$, area $9{,}682$.
2. Union $= 41{,}134 + 30{,}800 - 9{,}682 = 62{,}252$.
3. $\text{IoU} = 9{,}682/62{,}252 \approx 0.1555$.
4. Enclosing box $C$: $x \in [12,400], y \in [506,700]$ → width $388$,
   height $194$, area $75{,}272$.
5. $\text{GIoU} = 0.1555 - (75{,}272-62{,}252)/75{,}272 = 0.1555 -
   0.1730 \approx -0.0175$ — again negative: the enclosing box has a lot
   of empty space, since the predicted box sits noticeably down-and-right
   of the real crack.
6. Centers: GT $=(143.0, 584.5)$, predicted $=(290.0, 630.0)$.
   $\rho^2 = (290.0-143.0)^2 + (630.0-584.5)^2 = 147.0^2 + 45.5^2 =
   21{,}609 + 2{,}070.25 = 23{,}679.25$. Diagonal $c^2 = 388^2+194^2 =
   150{,}544+37{,}636 = 188{,}180$.
7. $\text{DIoU} = 0.1555 - 23{,}679.25/188{,}180 \approx 0.1555 - 0.1258
   \approx 0.0297$.
8. Aspect ratios: GT $w/h = 262/157 \approx 1.669$, predicted $w/h =
   220/140 \approx 1.571$ — close but not identical, unlike Worked
   Example 1. $v \approx 0.000375$ (small, since the ratios are close),
   $\alpha \approx 0.000443$. $\text{CIoU} \approx 0.0297 -
   (0.000443 \times 0.000375) \approx 0.0297$ — essentially unchanged
   from DIoU to this precision. This is the same lesson as Worked Example
   1, just less extreme: the aspect-ratio penalty only meaningfully
   changes the score when width/height *shape* genuinely mismatches, and
   here the predicted box happens to be roughly the right shape, just in
   the wrong place — so DIoU's distance penalty is doing essentially all
   of the corrective work.

## Diagram
```
        C (smallest enclosing box)
   +---------------------------+
   |   B_gt            .       |
   |  +------+          .      |
   |  | /////|--+       .      |  ///  = intersection (B ^ B_gt)
   |  |//////|B |       .      |  ...  = "wasted" space GIoU penalizes
   |  +------+--+       .      |         (inside C, outside B u B_gt)
   |       ..................  |
   +---------------------------+
IoU  = overlap / union
GIoU = IoU  - (wasted space in C) / |C|
DIoU = IoU  - (center distance)^2 / (C diagonal)^2
CIoU = DIoU - (aspect-ratio mismatch penalty)
```

## Why it matters for THIS project
`docs/learning/07-focal-loss.md` already notes that YOLOv8's box
regression loss is a **CIoU loss**, not plain IoU — this file is the
missing piece explaining *why* that specific variant was chosen: our real
worked example above shows plain IoU alone (or even GIoU) can go
negative or stay near-zero for boxes that are in roughly the right
neighborhood but not yet overlapping well, giving weak gradient signal
right when a still-learning model needs the clearest possible "which way
to move" information. DIoU's direct center-distance term and CIoU's added
shape penalty both exist to make that early-training gradient signal
stronger and more directional — relevant here since our own EDA showed
many road-damage boxes are small, so a predicted box that's merely
"close" rather than "overlapping" is a very common case this project's
model will actually encounter, especially early in training.

## Interview questions

**Q: Why isn't plain IoU good enough as a loss function?**
A: If two boxes don't overlap at all, IoU is exactly $0$ regardless of
whether they're one pixel or a thousand pixels apart — no gradient
signal distinguishes "almost right" from "wildly wrong." GIoU, DIoU, and
CIoU each add a term that stays informative even at IoU $=0$, using the
smallest enclosing box's wasted space (GIoU), center-to-center distance
(DIoU), or both plus aspect-ratio mismatch (CIoU).

**Q: What's the practical difference between DIoU and CIoU?**
A: DIoU only penalizes how far apart the two boxes' centers are. CIoU adds
an extra term on top of that, penalizing when the predicted box's
width-to-height ratio doesn't match the ground truth's, even if the
centers are already close. CIoU matters most when a model predicts
roughly the right location but the wrong *shape* — e.g., a box that's too
wide and too short — a case DIoU alone wouldn't specifically correct.

**Q: In Worked Example 1, CIoU came out identical to DIoU — is CIoU
sometimes pointless?**
A: Not pointless — it just correctly contributes nothing extra when
$v = 0$, i.e., when the predicted and ground-truth boxes already have the
exact same aspect ratio. The penalty is designed to activate in
proportion to actual shape mismatch; two same-shaped squares (or, in our
real project example, two nearly-same-shaped rectangles) is exactly the
case where it *should* stay small or zero — it only grows when a model
predicts a genuinely wrong-shaped box.
