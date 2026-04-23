# Three-way Behavioral Analysis: corr_only vs corrrank vs tiebreak (epoch 1, SLAKE test)

**Date:** 2026-04-23
**Setup:** All three methods evaluated on SLAKE English test (1061 questions, greedy decode) at matched epoch 1. Per-question generations from
- `slake_test_groupB.json` → `corr_s42`
- `slake_test_tiebreak_s42_epochs.json` → `tiebreak_s42_ep1`
- `slake_test_corrrank.json` → `corrrank_s42`

All three methods trained with the same recipe (SLAKE organ-only, 1888/188 split, LoRA r=16, 8 rollouts, token-F1 reward, 5 planned epochs). Only difference: how advantage is computed.

## TL;DR

1. **The F1 advantage of tiebreak (0.5200 vs corr_only 0.4086) decomposes into (a) not truncating, (b) avoiding degenerate wait-loops in baselines, (c) a mild Yes-bias that helps on GT=Yes pathology questions.**
2. **Tiebreak is not "No by default" on pathology questions** — it over-predicts Yes in both GT=Yes (89%) and GT=No (48%) directions relative to the baselines.
3. **Baselines exhibit catastrophic repetition loops** (literal string repeats until token budget exhausts) that tiebreak does not.
4. **The one subtype where tiebreak underperforms corr_only is color identification** (−0.036 F1). Every other subtype (yes/no, identify, locate, modality, count, other) is +0.08 to +0.18.

## Outcome distribution (3-way)

Scoring each rollout correct (F1 ≥ 0.5) or wrong, per the existing F1 threshold:

| Bucket | Count | % of 1061 |
|---|---|---|
| all three wrong | 457 | 43.1% |
| all three right | 343 | 32.3% |
| **tiebreak-only right** | **129** | **12.2%** |
| corr+tiebreak right, corrrank wrong | 58 | 5.5% |
| corrrank+tiebreak right, corr wrong | 34 | 3.2% |
| corr+corrrank right, **tiebreak wrong** | **17** | **1.6%** |
| corr-only right | 12 | 1.1% |
| corrrank-only right | 11 | 1.0% |

Tiebreak-only-right vs baseline-only-right (strictly): **129 vs 17 = 7.6× asymmetric.**

F1 means: corr_only 0.4086, corrrank 0.3807, tiebreak 0.5200.

## Length + format

| Method | Total chars (mean) | Think chars (mean) | No `<answer>` tag |
|---|---|---|---|
| corr_only | 1346 | 1014 | **24.3%** |
| corrrank | 1583 | 1169 | **29.4%** |
| tiebreak_ep1 | **395** | **347** | **0.8%** |

Tiebreak is 3-4× shorter and almost always extracts. Corr_only and corrrank burn 25-30% of their budget on outputs that truncate before the tag.

## Reasoning-content heuristics

Mean mentions per response of:

| Method | anatomy/path terms | visual descriptors (left/right/dark/etc.) | "wait"/hedge phrases |
|---|---|---|---|
| corr_only | 12.70 | 8.20 | 6.90 |
| corrrank | 14.58 | 10.52 | 6.43 |
| tiebreak_ep1 | **3.86** | **2.39** | **0.56** |

Corrrank has the *most* medical reasoning content (more anatomy, more descriptors, most length). Tiebreak has 3–4× less of all three. Corr_only sits in the middle.

## Wait-loops are the proximate cause of truncation

In corr_only and corrrank, the probability of truncation is an almost-deterministic function of wait-phrase count:

**corr_only — truncation rate by wait count:**

| n "wait"s | n rollouts | truncation rate |
|---|---|---|
| 0 | 328 | 0.6% |
| 1 | 194 | 1.0% |
| 2 | 82 | 3.7% |
| 3 | 46 | 13.0% |
| 6 | 16 | 31.2% |
| 10 | 17 | 52.9% |
| 13 | 17 | 88.2% |
| 15+ | 218 | **95.4%** |

Mean wait count in truncated responses: **19.17** vs **1.74** in completed (11× ratio). Corrrank same pattern: **100%** truncation at 11, 14 waits.

Tiebreak mostly has 0 waits (887/1061 = 83.6%) and truncates 0.8%.

### Example wait-loop from corr_only

Same response, one question, 8 visible "Wait" cycles, 1500+ chars, never reaches `<answer>`:

> *"Wait, CT scans can show various conditions... Wait, the user is asking what diseases are included. But the image itself might not show... Wait, maybe the question is about what's visible. Wait, maybe the image is a normal chest CT? But no, the question is about diseases. Wait, maybe the answer is..."* [truncated mid-token]

### Pathological repetition from corrrank

For *"What disease is shown on the left of the brain?"* the corrrank response is this, repeated verbatim ~20 times:

> *"the left side of the image is the left hemisphere. Wait, no, the left side of the image is the left hemisphere."*

Until token budget exhausts. This is not "reasoning that wandered too long" — this is a degenerate repetition failure. Tiebreak does not exhibit this mode.

## F1 by question subtype

Classification is heuristic (based on question prefix + keywords):

| Subtype | n | corr_only | corrrank | tiebreak | Δ vs corr |
|---|---|---|---|---|---|
| yes/no | 330 | 0.591 | 0.540 | **0.716** | +0.125 |
| locate | 231 | 0.376 | 0.328 | **0.458** | +0.082 |
| identify | 210 | 0.187 | 0.201 | **0.296** | +0.109 |
| modality | 83 | 0.493 | 0.491 | **0.671** | +0.177 |
| count | 52 | 0.404 | 0.423 | **0.481** | +0.077 |
| other | 128 | 0.350 | 0.318 | **0.486** | +0.136 |
| **color** | 27 | **0.212** | 0.153 | 0.175 | **−0.036** |

- Modality-identification (what scanner is this?) is tiebreak's biggest gain — a one-look visual question where commitment is exactly right.
- **Color is the one subtype tiebreak loses.** Color identification benefits from sustained visual examination, which tiebreak's compressed reasoning doesn't support.
- Identify (which organ/which disease) has the lowest F1 across all methods — these are the genuinely hard open-ended items.

## First-GT-mention timing

For non-yes/no questions: in each response, find the first occurrence of the GT term (or its first keyword).

| Method | abs position | % of total response | mentioned at all |
|---|---|---|---|
| corr_only | char 228 | 17% | 52% |
| corrrank | char 271 | 17% | 54% |
| tiebreak | char 122 | 34% | 48% |

All three **mention the correct word roughly as often** (~50%). The difference is in what happens after first mention: corr_only hits it at char 228 then writes 1100 more chars, often forgetting to emit `<answer>`. Tiebreak hits it at char 122 and commits within another 270 chars.

**Reading:** the model's first-pass reasoning is reasonable across all methods. The baseline methods know the answer but wander past it.

## Pathology-judgment behavior (211 yes/no questions about abnormality, disease, pathology)

| Method | GT=Yes: correct | GT=No: correct | GT=Yes: says "Yes" | GT=No: says "Yes" | GT=Yes: blank answer |
|---|---|---|---|---|---|
| corr_only | 70.9% | 50.5% | 71% | 31% | 16% |
| corrrank | 63.6% | 45.5% | 64% | 22% | 28% |
| **tiebreak_ep1** | **89.1%** | 51.5% | **89%** | **48%** | **0%** |

**Tiebreak has a Yes-bias on pathology questions, not a No-bias.** It says Yes more often than both baselines in both directions.

Two mechanisms split the GT=Yes advantage:

- **~2/3 of the gain comes from not-blanking.** Baselines leave 16-28% of GT=Yes answers blank; if those completed at the baseline's Yes-rate, corr_only would sit at ~83% vs tiebreak's 89%.
- **~1/3 comes from actual preference for Yes.** When tiebreak does commit to an answer, it's slightly more likely to call an image abnormal.

On GT=No the F1 parity hides a style difference: tiebreak says "Yes" on 48% of normal images (vs 31% for corr_only, 22% for corrrank). Tiebreak is **more likely to call normal images abnormal**, not less. The F1 parity comes from fewer blanks balancing out the more frequent false-Yes.

### Hallucinated pathology in CoT — not the same as answer-level hallucination

Separate measurement: on the 38 GT=No questions asking about abnormality, count responses with ≥3 pathology-related terms in the `<think>` block:

| Method | Correct | 3+ pathology terms in CoT |
|---|---|---|
| corr_only | 42.1% | **76.3%** |
| corrrank | 28.9% | **84.2%** |
| tiebreak | **55.3%** | **21.1%** |

These two findings reconcile: tiebreak's responses are so short (mean 395 chars) that they don't *contain* many terms of any kind. But when tiebreak DOES commit to a Yes verdict, it doesn't need long diagnostic reasoning to do so — the 48% Yes-rate on GT=No shows the bias exists regardless of CoT length.

## Representative deep reads

### "What is the largest organ in the picture?" GT: Bladder [OPEN / identify]

- **corr_only** [F1 0.00]: 20+ "wait" cycles between *"the skin is the largest organ"* and *"but skin isn't in the picture."* Never emits `<answer>`. Truncated mid-sentence.
- **corrrank** [F1 0.00]: Same trap, different wording, same loop. Never commits.
- **tiebreak** [F1 1.00]: *"The largest structure here is the bladder. The bladder is the largest organ visible."* → `<answer>bladder</answer>`. 280 chars total.

### "What disease is/are shown on the left of brain?" GT: Brain Edema, Brain Non-enhancing Tumor [OPEN / identify]

- **corr_only** [0.00]: Reasons to "hemorrhage," commits.
- **corrrank** [0.00]: Stuck repeating *"the left side of the image is the left hemisphere"* until budget exhausts.
- **tiebreak** [0.67]: *"the image shows a large bright area on the left, which is typical of a brain tumor."* → `<answer>brain tumor</answer>`. Hits one of the two GT terms → partial credit.

### "What modality is used to take this image?" GT: X-Ray [OPEN / modality]

- **corr_only** [F1 0.00]: Reasons correctly, emits `<answer>X - ray</answer>` (spaced hyphen → 3 tokens → 0 F1 overlap with "X-Ray").
- **corrrank** [F1 1.00]: Same reasoning, emits `<answer>X-ray</answer>` → matches → 1.0.
- **tiebreak** [F1 1.00]: *"It's a chest X-ray, right? So the answer should be X-ray."* → `<answer>X-ray</answer>`. 4× shorter.

### "Is the lung abnormal?" GT: Yes [CLOSED / yes_no — a tiebreak LOSS case]

- **corr_only** [1.00]: *"right lung might have a mass? Wait, no, let me recheck… there's an area that could be a sign of pathology."* → `<answer>Yes</answer>`.
- **corrrank** [1.00]: Similar hedging, lands on `<answer>Yes</answer>`.
- **tiebreak** [0.00]: *"The lung fields seem normal. No obvious abnormality."* → `<answer>No</answer>`.

This is the minority "dropped medical reasoning" failure mode — 17 cases out of 1061. Tiebreak's terse commit misses the subtle-pathology call that the baselines' "wait, let me check..." pattern happens to catch.

## What this means for the method

The probe's role is not "predict correctness." It's "filter against degenerate wait-loop rollouts at T=1.0 sampling." The mechanism:

1. At T=1.0 × 8 rollouts, rollouts that wait-loop truncate (F1 = 0); rollouts that commit complete (scorable F1).
2. Under pure corr_only advantage, 8 rollouts tying at 0 F1 produce collapsed gradient → no training signal.
3. Under tiebreak, faith breaks ties. Committed rollouts have coherent attention (probe-preferred); wait-looping rollouts have diffuse/chaotic attention (probe-disfavored). Committed → positive advantage; wait-looping → negative advantage.
4. Training amplifies "commit early" as a learned behavior.

**The probe doesn't need to know whether an answer is correct — it needs to distinguish coherent attention from degenerate attention.** Correctness correlates with coherence on average, but the direct selection pressure is on coherence.

### Implications for generalization

- **Domain-general mechanism.** The wait-loop-vs-commit distinction exists in any T-sampled CoT generation, not just medical VQA. A probe trained on coherence-vs-incoherence attention patterns should work across domains.
- **Base-model first-pass accuracy is the ceiling.** Tiebreak converts blanks into first-pass answers. If first-pass is good, tiebreak helps; if first-pass is bad, tiebreak just emits wrong answers faster.
- **The regression on color questions is telling.** Color is one of the few SLAKE question types where the first-pass judgment is often wrong and careful examination would catch it. Expected failure mode: wherever committed first-pass underperforms considered reasoning, tiebreak is worse. This also predicts the "dropped medical reasoning" pattern at late training.
- **The Yes-bias on pathology questions** is a side effect of "commit on first impression" — first-impression of a medical image containing abnormality is often Yes because models are trained on skewed data. Worth monitoring in the paper as a limitation: tiebreak may slightly over-diagnose on ambiguous normal images.

## Caveats

- Heuristic subtype classification misses nuance (color questions can be yes/no, some "locate" questions are really identify).
- Pathology-term hallucination is a heuristic count; does not verify whether CoT claims are actually false.
- Sample sizes for some buckets are small (color n=27; two-option questions n=7).
- This is epoch 1 only; late-checkpoint behavior (the existing `tiebreaker_grpo_results.md` training-evolution section) differs — the commitment behavior amplifies further into image-description-only responses.
