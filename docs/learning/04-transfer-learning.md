# 04 — Transfer Learning

## The intuition, before any jargon
Imagine you need to teach three people to spot road damage in photos.

Person A has spent years as a general "object spotter" — given any photo,
they can already draw a box around cars, people, dogs, furniture, whatever
is in it, and say what each thing is. They've just never specifically been
asked about cracks or potholes before.

Person B has only ever played a different game: look at a whole photo and
say one word for it ("that's a cat," "that's a kitchen") — they've never
had to point at *where* something is, only *what* the whole image shows.

Person C has never looked at a photo in their life.

If you have one afternoon to teach all three "find road cracks and
potholes, and draw a box around each one," who needs the least new
training? Obviously Person A — they already have the *mechanical skill* of
scanning an image and outputting precise boxes; you're just teaching them
one new *category* of thing to look for. Person B has to learn an entirely
new skill (pointing at locations, not just describing the whole photo) on
top of the new category. Person C has to learn everything from nothing.

That's **transfer learning**: instead of randomly initializing a neural
network and making it learn every visual skill from zero, start from a
network already trained on a different-but-related task, then keep
training ("fine-tune") it on your smaller, narrower task. Person A =
COCO-pretrained detector, Person B = ImageNet-pretrained classifier,
Person C = training from scratch.

## The math — but the "why" comes first
Whether a network is randomly initialized or pretrained, the actual
training rule (gradient descent) is identical — every weight gets nudged
in the direction that reduces the loss:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

**Term-by-term breakdown:**
- $\theta_t$ — all of the network's weights, at training step $t$.
- $\mathcal{L}(\theta_t)$ — the loss (how wrong the model currently is),
  evaluated with the current weights.
- $\nabla_\theta \mathcal{L}(\theta_t)$ — the gradient: the direction each
  weight should move to reduce the loss the most.
- $\eta$ (eta) — the learning rate: how big a step to take in that
  direction.
- $\theta_{t+1}$ — the updated weights after one training step.

**Transfer learning changes exactly one thing in this equation: the
starting point, $\theta_0$.** Training from scratch sets
$\theta_0 \sim \mathcal{N}(0, \sigma^2)$ — small random numbers, so the
network starts out knowing nothing about edges, shapes, or textures.
Transfer learning instead sets $\theta_0 = \theta_{\text{pretrained}}$ —
weights already tuned on a large, different dataset. Every subsequent
gradient-descent step is the same math; you're just starting the walk from
a point much closer to "useful" instead of from a random point in a huge
weight space.

A related, optional idea is **freezing**: choosing not to update some
layers at all during fine-tuning, usually the earliest ones (which tend to
learn generic things like edges and textures, regardless of the task):

$$\theta_{t+1}^{(l)} = \begin{cases} \theta_t^{(l)} & \text{if layer } l \text{ is frozen} \\ \theta_t^{(l)} - \eta \nabla_{\theta^{(l)}} \mathcal{L}(\theta_t) & \text{otherwise} \end{cases}$$

**Term-by-term breakdown:**
- $\theta_t^{(l)}$ — just the weights belonging to layer $l$.
- "Frozen" — that layer's weights are excluded from the update; they stay
  exactly as they were in the pretrained checkpoint, forever, for the rest
  of training.
- Non-frozen layers update normally, via the same gradient-descent rule as
  before.

### Worked example 1 — illustrative numbers
Take a toy 4-layer network with 100, 200, 50, and 20 parameters in each
layer (370 total). Suppose we freeze the first two layers (thought to hold
generic, reusable features) and only fine-tune the last two:

1. Frozen parameter count: $100 + 200 = 300$ — these values never change
   from their pretrained starting point.
2. Trainable parameter count: $50 + 20 = 70$.
3. Fraction of the network actually being re-learned:
   $70 / 370 \approx 18.9\%$.

So freezing turns "learn a 370-parameter function from your data" into
"learn a 70-parameter function from your data" — roughly a **5.3x
reduction** ($370/70 \approx 5.3$) in how much the optimizer has to
discover purely from your (possibly small) dataset, because 81% of the
network's decision-making capacity is simply inherited, untouched, from
whatever task it was pretrained on.

### Worked example 2 — tied to our actual project data
Ultralytics' published spec for `yolov8n.pt` lists it at roughly **3.2
million parameters**. Its COCO pretraining set has about **118,000
images** across 80 object classes. Our actual RDD2022 India+Japan subset
totals (train 12,425 + valid 1,052 + test 1,117 =) **14,594 images**
across 4 damage classes.

$$\frac{118{,}000}{14{,}594} \approx 8.08$$

COCO pretraining exposed this exact architecture to roughly **8x more
images** than our entire project dataset contains — and critically, COCO's
pretraining task (find objects, draw boxes, classify them) is *structurally
identical* to ours, just with different category labels (car/dog/chair
instead of D00/D10/D20/D40). That structural match is what makes nearly
all of that 8x head start transfer directly: the backbone already knows
how to detect edges, textures, and object-shaped regions and output precise
box coordinates. Fine-tuning mostly has to teach it what *road damage*
specifically looks like, not how to localize things in an image at all.

This is also why we're using **full fine-tuning, not a frozen backbone**
(Ultralytics' default `freeze=None` — no layers frozen): with 14,594 real
images, we have enough data to safely adjust every layer, including the
backbone's early filters, toward asphalt/crack-specific textures, rather
than only retraining a classification head on top of frozen, generic COCO
features. Freezing is more valuable when fine-tuning data is much scarcer
than ours (hundreds of images, not tens of thousands).

## Diagram
```
COCO pretraining                      Our fine-tuning
(80 classes, ~118K images)            (4 classes, ~14.6K images)
        |                                     |
        v                                     v
 [ random init ] --gradient descent--> [ yolov8n.pt weights ]
                                               |
                                               | theta_0 = theta_pretrained
                                               v
                                     [ fine-tune on RDD2022 ]
                                       (ALL layers trainable,
                                        freeze=None)
                                               |
                                               v
                                  [ road-damage detector ]
```

## Why it matters for THIS project
This is Phase 3 Decision 3.3, locked in alongside 3.1 (YOLOv8) and 3.2
(nano scale) — `configs/model_config.yaml`'s `weights: yolov8n.pt` line is
what actually sets $\theta_0 = \theta_{\text{pretrained}}$ for every
training run. With only 14,594 images, training from scratch would mean
learning basic visual primitives (edges, shapes, localization) *and*
road-damage-specific patterns from the same small pool — COCO pretraining
means we only have to teach the second half. Combined with Focal Loss
(`fl_gamma`, `docs/learning/07-focal-loss.md`) and the mosaic/mixup/
Albumentations augmentation stack (`docs/learning/06-mosaic-mixup.md`),
transfer learning is the third leg of the same underlying constraint this
whole project is built around: get a good detector out of a comparatively
small, moderately imbalanced dataset on a single 4GB laptop GPU.

## Interview questions

**Q: Why COCO-pretrained instead of an ImageNet-pretrained backbone for an
object detection task?**
A: Pretraining transfers best when the pretext task matches the target
task structurally, not just visually. ImageNet pretraining only ever asks
"what's the single label for this whole image?" — it never teaches a
network to output precise box coordinates. COCO pretraining is *already*
an object-detection task (locate + classify), so the entire localization
machinery, not just low-level filters, transfers directly. For a detector,
detection-pretrained weights beat classification-pretrained weights, even
though ImageNet is arguably more visually diverse.

**Q: Why fine-tune the whole network instead of freezing the backbone and
only training the head?**
A: Freezing trades data-efficiency for capacity — it's the right call when
your fine-tuning set is very small (hundreds of images) and you're worried
about overfitting if you let every parameter move. With 14,594 images, we
have enough signal to safely adjust the backbone's early filters toward
road-specific textures (asphalt, cracks) rather than leaving them frozen at
whatever generic COCO features they started with, which should give
better final accuracy at the cost of somewhat slower initial convergence.

**Q: If this dataset were 100x smaller (say, ~150 images), would you change
this decision?**
A: Yes — at that size I'd lean toward freezing most or all of the backbone
(`freeze` argument in Ultralytics) and fine-tuning mainly the detection
head, since with that little data, letting the whole network move risks
catastrophic forgetting of the useful pretrained features and overfitting
to those specific 150 images rather than learning generalizable
road-damage patterns.
