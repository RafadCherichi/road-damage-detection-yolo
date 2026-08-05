# 10 — ONNX Export and Deployment Verification

## The intuition, before any jargon
Imagine translating a recipe from one language to another so a chef who
doesn't speak the original language can still cook it. A *good*
translation isn't done just because it's grammatically valid in the new
language — it's only actually trustworthy once someone cooks both
versions and confirms the dish tastes the same. **ONNX** is that
translation for a trained neural network: it converts a PyTorch model
into a framework-agnostic graph format that a different, lighter-weight
"chef" — **ONNX Runtime** — can run without PyTorch installed at all.
And exactly like the recipe, checking that the *translation itself* is
grammatically well-formed (structurally valid ONNX) is a completely
different, much weaker claim than checking that it actually *produces
the same result* (numerical parity) — this project's own history is a
real example of that gap being missed and then fixed.

## The math — but the "why" comes first
Deciding **static vs. dynamic** export shape is the first real choice.
A static-shape graph is compiled for exactly one input size (here,
batch=1, $640\times640$) — every operation's tensor dimensions are known
in advance, which lets the exporter fold constants and fuse operators
much more aggressively than a graph that has to stay flexible for
arbitrary batch sizes or resolutions. The trade-off is inflexibility: a
static graph can't handle a different input shape at all without
re-exporting. For this project's single-frame, fixed-resolution
inference use case, that inflexibility costs nothing real.

The actual trustworthiness question is answered by a **numerical parity
check** — not just "does the graph parse," but "does it compute the same
answer." The standard tolerance check (what `numpy.allclose` computes
under the hood) is:

$$|a - b| \le \text{atol} + \text{rtol} \cdot |b|$$

**Term-by-term breakdown:**
- $a$ — one backend's output value (e.g. the PyTorch model's).
- $b$ — the other backend's output value for the *same* input (e.g. the
  ONNX Runtime model's), treated as the reference for the relative term.
- $\text{atol}$ (absolute tolerance) — a fixed floor of allowed
  difference, dominant when values are near zero.
- $\text{rtol}$ (relative tolerance) — scales with the magnitude of $b$,
  so a tiny absolute difference is judged more strictly when $b$ is small
  and more leniently when $b$ is large. `numpy.allclose`'s default is
  $\text{rtol}=10^{-5}$ if not overridden.

This is a genuinely different, *stronger* claim than `onnx.checker.check_model()`,
which only confirms the exported graph is schema-valid — well-formed
operator names, matching tensor shapes at each node — with **no
guarantee at all that its computed output matches the original model**.
Conflating the two is a real gap, not a pedantic distinction: this
project's own `src/export.py` originally called a model "verified" after
only the structural check, before an audit caught that no numerical
comparison had ever actually run.

### Worked example 1 — illustrative numbers
Suppose one output value from the PyTorch model is $a = 87.65$ (a box
coordinate, in pixels) and the corresponding ONNX Runtime output is
$b = 87.6499$, with $\text{atol}=0.001$ and the default $\text{rtol}=10^{-5}$.

1. Raw difference: $|87.65 - 87.6499| = 0.0001$.
2. Allowed threshold: $0.001 + 10^{-5} \times 87.6499 \approx 0.001 +
   0.000876 = 0.001876$.
3. $0.0001 \le 0.001876$ — passes comfortably.

Now a confidence score, $a = 0.523$, $b = 0.5232$:
1. Raw difference: $|0.523 - 0.5232| = 0.0002$.
2. Allowed threshold: $0.001 + 10^{-5} \times 0.5232 \approx 0.0010052$.
3. $0.0002 \le 0.0010052$ — also passes.

### Worked example 2 — tied to our actual project data
This project's real check (`src/export.py`'s `verify_parity()`), run on
`yolov8n_kaggle_run/weights/best.pt` vs. its freshly exported
`best.onnx`, using a real sample image
(`data/raw/train/images/India_000027.jpg`):

$$\text{PARITY CHECK: PASS — max abs difference } 0.000061 \text{ (atol=0.001)}$$

To see how comfortable a margin that really is, check it against the
*loosest* applicable threshold in this tensor — detection outputs mix
small values (confidence scores, roughly $0$–$1$) with large ones (pixel
coordinates, up to $640$ at this input resolution). Even at the largest
plausible magnitude:

$$\text{threshold} = 0.001 + 10^{-5} \times 640 = 0.001 + 0.0064 = 0.0074$$

The real observed difference ($0.000061$) is about **120x smaller** than
even this most-permissive threshold in the tensor — this isn't a export
that barely scraped by, it's numerical agreement essentially at the
floor of ordinary FP32 floating-point noise between two different
kernel implementations (PyTorch's native ops vs. ONNX Runtime's).

## Diagram
```
PyTorch model (.pt)                    ONNX graph (.onnx)
       |                                       |
       | model.export(dynamic=False,           |
       |   imgsz=640, opset=12)                |
       +----------------->  static graph  -----+
       |                                       |
       v                                       v
  predict(image)                        predict(image)
  [PyTorch backend]                    [ONNX Runtime backend]
       |                                       |
       +---------------> compare <-------------+
                    |a - b| <= atol + rtol*|b|
                            |
                            v
              PASS (this project: max diff 0.000061)
```

## Why it matters for THIS project
This file exists because of a real, audit-caught gap: the README claimed
ONNX output was "numerically verified against the original `.pt` model,"
but the actual code only ran `onnx.checker.check_model()` — a structural
check that says nothing about numerical correctness. Writing and running
the real parity check wasn't just a formality; it's exactly the kind of
claim that's cheap to state and easy to leave unverified, and doing the
check surfaced two genuine environment bugs along the way (a CUDA-state
corruption from chaining a CPU-mode export with a GPU predict call in one
process, and an `onnxruntime-gpu`/CUDA-version mismatch) that would have
silently blocked real deployment testing later. The project's "static
ONNX, no PyTorch dependency" deployment story is only actually true now
that this check has run and passed with real numbers to point to.

## Concept Card
- **(a) In general:** ONNX converts a trained model into a
  framework-agnostic graph a different runtime can execute; a static
  export fixes the input shape for more aggressive graph optimization at
  the cost of shape flexibility; structural validity and numerical
  parity are two genuinely different, independently-necessary checks.
- **(b) Used here:** `src/export.py`'s `export_to_onnx()` (static shape,
  `dynamic=False`, `imgsz=640`, `opset=12`) and `verify_parity()` (the
  real numerical check, `device="cpu"`, `atol=1e-3`) — real result: PASS,
  max abs difference 0.000061, on `yolov8n_kaggle_run/weights/best.onnx`.
- **(c) When dynamic-shape export would have been the better choice:** if
  this project's deployment target needed to serve variable batch sizes
  or resolutions from one exported graph (e.g. a shared inference server
  handling requests of different sizes) — this project's actual use case
  (single-frame, fixed-resolution inference) never needs that flexibility,
  which is why static shape was chosen with no real cost.

## Interview questions

**Q: What's the difference between checking an ONNX export is "valid" and
checking it's "correct," and why do both matter?**
A: `onnx.checker.check_model()` validates the graph is well-formed —
correct operator names, consistent tensor shapes — but it never actually
runs the model, so it can't catch a silent numerical bug (e.g. an
operator that got approximated slightly differently during export).
Numerical parity verification actually executes both the original and
exported models on the same real input and compares outputs directly.
A model can pass the first check and still be numerically wrong; only
the second check catches that.

**Q: Why export with a static shape instead of a dynamic one?**
A: Static shape lets the ONNX exporter and runtime apply more aggressive
graph optimizations — constant folding, operator fusion — since every
tensor dimension is fixed and known ahead of time, rather than needing to
stay flexible for arbitrary batch sizes or resolutions. This project's
deployment target only ever processes one frame at a fixed resolution, so
the flexibility a dynamic graph would offer isn't needed, and giving it
up costs nothing real in exchange for the optimization headroom.

**Q: If your parity check had failed instead of passed, what would you
have checked first?**
A: First, whether the divergence is uniform across the whole output
(suggesting a systematic issue, like a precision or preprocessing
mismatch) or concentrated in specific outputs (suggesting one particular
operator was exported incorrectly or approximated). Second, whether both
backends are actually receiving identical preprocessed input — a
mismatched letterbox/normalization step between the two prediction calls
would produce a real difference that has nothing to do with the exported
graph itself. Only after ruling those out would I suspect the export
process itself (e.g. an unsupported op silently substituted during
conversion).
