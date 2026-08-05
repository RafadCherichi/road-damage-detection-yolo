# PM Perspective: What This Detector Is Actually Good For

This document translates the technical build (see `README.md`,
`docs/blueprint.md`, `docs/learning/`) into business terms. It doesn't
re-explain anchor-free detection, mAP, or ONNX export — it asks the
questions a public-works PM or ops lead deciding whether to actually use
this tool would ask. All numbers below are cited from, not re-derived
from, `results/metrics/metrics_summary.csv` (via `src/evaluate.py`) and
`README.md`'s Results table — see those for full methodology.

## 1. Business Reframe

**What it actually is:** not "a YOLOv8 model" — a triage tool that flags
candidate road-damage locations from photos, so a limited inspection or
repair budget gets pointed at the right segments first instead of relying
on citizen complaints or blind periodic sweeps alone.

**Who it's for:**
- **City/municipal public works departments** deciding which road
  segments get scheduled for repair this budget cycle.
- **Road maintenance contractors** scoping and prioritizing work orders
  across a large road network.
- **Insurance and infrastructure assessors** doing pavement-condition
  surveys for asset-management or claims purposes.

**The business question it answers:** *"Given a batch of road photos,
which segments most likely have damage worth sending a human to look
at, and how much should I trust that ranking?"* The second half matters
as much as the first — every class in this tool's output has a real,
measured, and uneven reliability (Section 2), and a repair-budget
decision made on an over-trusted number is worse than one made with the
real number in hand.

**What it does NOT answer:**
- **Repair cost or damage severity.** This detects *presence* and
  *class* (crack type or pothole), not depth, area extent, or urgency
  beyond what class alone implies. Two D00 (longitudinal crack)
  detections with similar box sizes could represent a hairline cosmetic
  crack and a structurally serious one — this tool cannot tell them
  apart. Cost/severity estimation is a separate step this tool doesn't
  attempt.
- **Safety-certified, real-time path-planning reliability.** `README.md`
  frames the broader motivation as feeding an autonomous-vehicle path
  planner. The real numbers (Section 2) — 0.662 overall recall, meaning
  roughly **1 in 3 real damage instances goes undetected** — are nowhere
  near the reliability bar a safety-critical, real-time driving system
  would need. This tool is honestly positioned as a **municipal
  planning/triage aid**, not a driving-safety system, given the numbers
  actually measured.
- **Generalization outside India+Japan road/pavement types.** Trained
  and evaluated exclusively on the RDD2022 India+Japan subset. Whether
  detection quality holds up on a different country's pavement material,
  lane-marking conventions, or camera/mounting setup has never been
  tested. Do not assume it transfers.
- **Change over time.** This is a single-pass detector on a set of
  photos, not a monitoring system. It has no concept of "this crack grew
  since last month" — that would require re-running on a cadence and
  comparing results, which isn't built.
- **An exhaustive, zero-miss damage inventory.** With ~66% recall, using
  this tool's silence on a segment as proof "there's no damage here" is
  a real misuse — it's a prioritization aid for what to look at *first*,
  not a certified inventory of everything that's wrong.

## 2. Success Metrics Beyond Accuracy

Real results from `src/evaluate.py` on the valid split
(`results/metrics/metrics_summary.csv`), translated into what each number
means for someone actually using this:

| Metric | Business-facing question it answers | What the number says |
|---|---|---|
| Recall, all classes (0.662) | If I rely on this tool's flags, how much real damage will I miss entirely? | About **1 in 3** real damage instances produces no detection at all. This is a triage aid, not a substitute for periodic manual inspection or an existing citizen-complaint channel. |
| Precision, all classes (0.671) | Of everything flagged, how often is it a wasted trip? | Roughly **1 in 3** flagged detections is a false alarm. Cheap to filter (a human glances at the image), but real — don't auto-dispatch a crew off a raw detection count without a review step. |
| D20 (Alligator) — 0.741P / 0.784R / 0.858 mAP50 | Which class's flags can I trust the most? | The strongest class on every axis measured — best candidate to weight heavily in any repair-priority ranking. |
| D00 (Longitudinal) — **0.591 recall**, the lowest of the four classes | Which damage type is most likely to be silently missed? | Longitudinal cracks specifically — a "clean" report should not be read as "definitely no longitudinal cracking present." |
| D10 (Transverse) — **0.627 precision, 0.653 mAP50**, both the lowest of the four, and the largest relative mAP50→mAP50-95 gap (52.7%) | Which class's flags need the most scrutiny, and which class's box coordinates are least trustworthy? | Transverse cracks are both the least reliably *found* and, even when found, the most loosely *boxed* of any class (see `docs/learning/05-map-metric.md`'s worked example on this exact gap). Any future feature that estimates severity from box size should not trust D10 boxes without a wide error margin. |
| D40 (Pothole) — 0.634P / 0.622R, no standout weakness but no standout strength either | How much can I rely on this for the damage type most likely to already generate citizen complaints? | Moderate on every axis. Given potholes are often independently reported (311-style complaint lines), cross-referencing this tool's flags against existing complaint data is a more complete picture than either source alone. |
| ONNX export latency (README states "~20ms/frame") | Is this actually fast enough for the deployment target I have in mind? | **Not independently benchmarked in this project.** Unlike the mAP table and the ONNX/PyTorch parity check (both real, verified numbers as of this writing), this specific latency figure was not measured as part of this project's evaluation work. Flagging this honestly rather than treating it as equally trustworthy — do not make a real-time deployment decision on this number without benchmarking it first. |

**One thing this project does not currently measure at all:** whether
acting on this tool's priority ranking (repairing the flagged segments)
actually reduces future damage reports or resurvey findings. Everything
above is a one-time snapshot evaluation on a held-out valid split, not a
closed-loop measurement of real-world repair outcomes.

## 3. Ship vs. Review Scoping Call

| Class | Real numbers | Call | Why |
|---|---|---|---|
| **D20 (Alligator)** | 0.741P / 0.784R / 0.858 mAP50 / 0.539 mAP50-95 | **Ship as a priority signal** | Strongest class measured on every axis — precision, recall, and both mAP thresholds. Reasonable to weight heavily in an automated repair-priority score with lighter human review. |
| **D40 (Pothole)** | 0.634P / 0.622R / 0.690 mAP50 | **Ship with review** | No specific standout weakness, but recall of 0.622 means real miss risk on a damage type that's often safety-urgent and publicly visible — recommend cross-checking against existing citizen-complaint channels rather than relying on this alone. |
| **D00 (Longitudinal)** | 0.681P / **0.591R** / 0.670 mAP50 | **Needs review before use** | Lowest recall of all four classes — this is the damage type most likely to be silently under-reported by the tool. A "no D00 detected" result on a segment should not be treated as confirmation of no longitudinal cracking. |
| **D10 (Transverse)** | **0.627P** / 0.650R / **0.653 mAP50** / worst localization gap (52.7%) | **Needs review before use** | Weakest class on precision, both mAP thresholds, and box-localization tightness. Both a "less trustworthy to find" and "less trustworthy to measure" class — the most caution-warranted of the four for any downstream decision. |

**The scoping rule that falls out of this:** no class in this model
clears a bar high enough for fully automated, un-reviewed action (even
D20's numbers mean roughly 1 in 5 real alligator-crack instances is
missed) — this tool's honest role across all four classes is
*prioritization and triage*, with a human reviewing flagged images before
any budget or dispatch decision is finalized. D20 flags deserve the most
weight; D00 and D10 flags deserve the most skepticism, for different
reasons (D00 under-detects, D10 both under-detects and mis-locates).

## 4. Risk Register

| Risk | Likelihood (observed) | Impact | Mitigation |
|---|---|---|---|
| Real damage missed entirely (recall gap, worst on D00 at 0.591) | Moderate-high — affects roughly 1 in 3 instances overall | **High** — a skipped repair-budget line item for a real, existing problem | Use as a triage aid alongside existing citizen-complaint/manual-inspection channels, not a sole source of truth; periodic re-scans provide a second chance to catch what one pass misses |
| False positives creating wasted inspection trips (precision gap, worst on D10 at 0.627) | Moderate — roughly 1 in 3 flags overall, more on D10 | Low-moderate — a few minutes of human review time, not a safety issue | Mandatory human image review before any dispatch decision, especially for D10 flags |
| Untested geographic/pavement generalization (India+Japan only) | Unknown — never measured outside this subset | **High if deployed elsewhere without re-validation** — could silently underperform on unfamiliar pavement/markings | Do not deploy to a new country/region without first validating on a local labeled sample; treat India+Japan performance as non-transferable by default |
| No severity/cost estimation exists | N/A — by design, out of scope | Moderate — a PM could over-interpret detection confidence as urgency | Pair with a separate human or future model-based severity/cost step; do not treat detection confidence as a severity proxy |
| D10 box localization is loose even when detection succeeds (52.7% relative mAP50→mAP50-95 gap) | Confirmed, measured | Moderate — only matters if a future feature derives area/severity from box size | Flag any box-size-derived metric for D10 as low-confidence if that feature gets built; not a risk for detection-presence-only use today |
| Small/thin cracks in high-resolution source photos may be undercounted (SAHI not implemented, see `docs/learning/08-sahi.md`) | Plausible, not directly measured — inference resizes to a fixed 640×640 | Moderate, resolution-dependent | Pre-tile unusually high-resolution source images before inference until SAHI or an equivalent is built; low risk if source photos are already close to 640×640 |
| ONNX "~20ms/frame" latency claim was never independently benchmarked in this project | N/A — unmeasured, not unverified-and-wrong, just unmeasured | Moderate if used to justify a real-time deployment decision | Benchmark before quoting this number in any decision-facing context — the same standard already applied to the mAP table and the ONNX/PyTorch parity check |
| Non-deterministic plotting crash during local evaluation (dev environment only, see `README.md` Lessons Learned) | Confirmed, 3/3 attempts on this specific laptop | Low — numeric metrics are unaffected and reproduced identically across runs | Anyone re-running `src/evaluate.py` locally on similar hardware should expect to use its default `plots=False` |

---

*This document is a translation layer, not a new analysis — every claim
above traces back to `results/metrics/metrics_summary.csv`,
`README.md`'s Results and Lessons Learned sections, and
`docs/learning/05-map-metric.md`, `08-sahi.md`, and `09-eigencam.md`.*
