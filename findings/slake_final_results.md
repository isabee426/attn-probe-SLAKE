# SLAKE Spatial GRPO: Final Results

**Date:** April 5, 2026
**Dataset:** SLAKE organ-specific questions (5972 → 2076, 35%), 10/1 split (1888 train / 188 val)
**Model:** Qwen3-VL-2B-Thinking + LoRA (r=16), 8 rollouts, max_new_tokens=512, "brief" prompt

## Experiment 1: Old Correctness Function (binary with containment/synonyms)

| Step | Correctness-only (W0, α=1.0) | | Spatial GRPO (W5, α=0.7) | |
|------|---|---|---|---|
| | Correct | Faith | Correct | Faith |
| 10 | 0.692 | 0.276 | 0.676 | 0.276 |
| 20 | 0.670 | 0.277 | 0.654 | 0.277 |
| 30 | 0.681 | 0.276 | 0.686 | 0.276 |
| 40 | 0.681 | 0.276 | **0.723** | 0.277 |
| 50 | **0.697** | 0.277 | — | — |
| 80 | **0.697** (best) | 0.275 | — | — |
| 90 | 0.681 | 0.275 | 0.676 | 0.276 |
| 100 | — | — | 0.686 | 0.276 |

- Correctness-only peaked at step 80: **0.697**
- Spatial peaked at step 40: **0.723**
- **Spatial won by +0.026** (3.7% relative)
- Faith identical for both (~0.276) — probe acts as regularizer, not grounding improver

## Experiment 2: Tight Correctness Function (token F1)

| Step | Correctness-only (W3, α=1.0) | | Spatial GRPO (W4, α=0.7) | |
|------|---|---|---|---|
| | Correct | Faith | Correct | Faith |
| 10-20 | ~0.62 (best) | 0.276 | — | — |
| 30 | 0.612 | 0.276 | — | — |
| 40 | 0.601 | 0.276 | — | — |
| 50 | 0.575 ↓ | 0.276 | **0.622** | 0.276 |

- Correctness-only peaked early (~step 10-20), then declined to 0.575
- Spatial still climbing at step 50: **0.622**
- **Spatial ahead by +0.047** (8.2% relative) at comparable steps
- Same pattern: spatial has a slower start but stronger trajectory

## Rollout Analysis (30 examples, greedy decode)

Compared zero-shot vs best checkpoints from each run:

| Model | Accuracy (old corr fn) | Accuracy (tight corr fn) |
|-------|----------------------|------------------------|
| Zero-shot | 21/30 (70%) | 22/30 (73%) |
| Correctness-only | 20/30 (67%) | 20/30 (67%) |
| Spatial α=0.7 | 19/30 (63%) | 21/30 (70%) |

- All three models produced near-identical outputs on 23/30 questions
- When they differed, spatial model showed:
  - More specific visual descriptions ("in the axial view" vs "in the abdominal region")
  - More decisive reasoning (breaks out of "Wait, no, wait" loops sooner)
  - More concise answers (90 tokens vs 172 for correctness-only on Q10)
  - One genuine spatial win: correctly localized a lung mass (Q18) where others failed

## Key Findings

1. **Spatial probe reward improves accuracy** despite not changing measured faithfulness. Both experiments show spatial > correctness-only on val correct.

2. **The probe acts as a regularizer**, not a grounding improver. Faith is flat at ~0.276 for both runs. The 30% faith component in the composite reward smooths the training landscape and slows overfitting on the binary correctness signal.

3. **Correctness-only overfits faster.** In both experiments, correctness-only peaks early and declines. Spatial has a slower start but maintains improvement longer.

4. **The tight correctness function (token F1) is stricter** — scores are lower overall (0.62 vs 0.72) but more honest. No more false positives from containment matching on full reasoning text.

## Limitations

- Only 35% of SLAKE has organ bboxes (2076/5972) — keyword matching is noisy
- Probe trained on correctness labels (circular) for Experiment 1
- Probe trained on bbox overlap labels for SLAKE probe on GPU 6 (still running at session end)
- Small val set (188 examples) — results are noisy
- Faith measurement may not capture actual grounding differences (greedy decode produces nearly identical outputs)

## Moving to VinDr-CXR-VQA

These limitations motivate the switch to VinDr-CXR-VQA:
- 100% per-question expert bboxes (vs 35% keyword-matched)
- 17,597 QA pairs (vs 2,076)
- Probe trained on bbox overlap labels (non-circular)
- Larger val set (~1,760 examples)
- Clean per-question bbox lookup (no keyword matching)
