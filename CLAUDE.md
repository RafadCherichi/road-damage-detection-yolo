# Road Damage Detection (YOLOv8) — Project Context

## WHY
Portfolio project: real-time multi-class road defect detection for autonomous
vehicle path planning, built to demonstrate applied CV skills for automotive/EV
hiring. Part of a 7-project Applied AI portfolio spanning Auto/EV/CV/LLM domains.

## WHAT
YOLOv8 fine-tuned on the RDD2022 dataset (road damage: cracks, potholes),
with EigenCAM explainability and ONNX export for deployment.

## CONSTRAINTS (non-negotiable)
- Single local Windows laptop only. No cloud GPUs, no Colab, no AWS/GCP/Azure.
- GPU: NVIDIA RTX 3050 laptop, 4GB VRAM. This caps model scale (nano/small
  YOLOv8 variants only, not medium/large) and batch size.
- $0 budget. 100% open-source tools only.
- Terminal: Anaconda Prompt (conda + claude both work here).

## HOW WE WORK
- I am learning, not just shipping. For every non-trivial technical decision
  (architecture, loss function, augmentation strategy, deployment format),
  STOP and present me 2-4 options with pros/cons and your recommendation.
  Wait for my choice before implementing.
- Follow the 80/20 rule: teach deeply the concepts in docs/must-know-concepts.md
  when we reach them, skip explaining boilerplate.
- Use Plan Mode for anything touching the training/data pipeline — show me
  the plan before executing.
- STANDING RULE: whenever you teach a must-know concept (from
  docs/blueprint.md's list, or any other concept central to a technical
  decision), don't just explain it in the terminal — also write or append
  a dedicated file to docs/learning/. Structure:
  ```
  docs/learning/
    00-project-overview.md       — problem statement, architecture diagram,
                                     pipeline summary, tech stack, why each
                                     major choice was made
    01-anchor-free-detection.md
    02-non-max-suppression.md
    03-iou-variants.md
    04-transfer-learning.md
    05-map-metric.md
    06-mosaic-mixup.md
    07-focal-loss.md
    08-sahi.md
    09-eigencam.md
    10-onnx-deployment.md
    99-interview-prep.md         — updated continuously: likely interview
                                     questions about this project + strong
                                     answers, common follow-up questions,
                                     what to say if asked "walk me through
                                     your architecture," numbers/results to
                                     memorize
  ```
  Each concept file must include, in order: (1) plain-language intuition
  first, (2) the underlying math/algorithm with a worked numerical example,
  (3) a diagram (ASCII or Mermaid), (4) why it matters for this specific
  project, (5) 2-3 likely interview questions with model answers at the
  bottom. 99-interview-prep.md is never "done" — append to it every time a
  new concept or result creates a new plausible interview question.
- STANDING WRITING-STYLE RULE for every docs/learning/ file: write to
  genuinely teach, not to dump jargon.
  - Use real technical vocabulary (IoU, backpropagation, etc.) — I need
    correct terms for interviews — but always give a plain-language
    explanation of a term first, or right alongside it, never after.
  - Every math/algorithm section must open with a real-world analogy or
    physical intuition BEFORE the formula. Formula comes second, after
    the "why," never first.
  - Assume I'm smart but new to this specific concept — teach it the way
    you'd teach linear algebra or ML basics to a capable beginner
    starting from zero on this exact topic.
  - Worked numerical examples must narrate the reasoning behind every
    step, not just the arithmetic — explain why each step happens, not
    only what number comes out.
  - End every concept file by tying it back explicitly to THIS project —
    not a generic example, but where this exact concept shows up in our
    road damage detector.
  - All mathematical content must use proper LaTeX (inline `$...$`,
    block/display `$$...$$`) — never equations spelled out in plain words
    or ASCII approximations. Every concept file with underlying math must
    include, in this order: (1) the formal equation in LaTeX, (2) a
    term-by-term breakdown of what each symbol/variable means in plain
    language directly under the equation, (3) at least one fully worked
    numerical example with real numbers (not abstract variables), showing
    every intermediate calculation step, (4) where relevant, a second
    worked example using our actual project data (e.g. our real class
    imbalance ratio for Focal Loss, our real bbox sizes for an IoU
    example).

## COMMANDS (to be filled in as we build them)
- conda activate rdd-yolo — activate project environment
- python src/train.py --config configs/train_config.yaml — train
- python src/evaluate.py — run eval suite

## ARCHITECTURE
Full phased pipeline and decision tree will live in docs/blueprint.md
(to be added next).
