# Rank-Based Tiebreaker GRPO: Live Results

**Date:** April 21, 2026
**Dataset:** SLAKE organ-only (5972 → 2076, 35%), 10/1 split (1888 train / 188 val), SLAKE English test set (1061 Q)
**Model:** Qwen3-VL-2B-Thinking + LoRA (r=16), 8 rollouts, max_new_tokens=512
**Metric:** Token F1 (strict — extract `<answer>` or post-`</think>`, token-overlap F1; yes/no and numbers use exact match)

## Summary

The *faith-tiebreaker rank advantage* (new construction) beats both correctness-only GRPO and the standard composite-reward GRPO on held-out SLAKE test, despite using no faith signal in the reward magnitude. Full and corr both overfit the organ-only training distribution by ~0.10 F1 on test; tiebreaker generalizes (test F1 > val F1).

## Method

Three reward/advantage constructions:

| Method | Reward | Advantage |
|---|---|---|
| `corr_only` (α=1.0) | `correctness` | `reward − mean(reward)` |
| `full` (α=0.7) | `0.7·correctness + 0.3·faith` | `reward − mean(reward)` |
| **`tiebreak`** | `correctness` (logged; not used in advantage) | `rank(correctness, faith) − mean(rank)`, lex sort |
| `corrrank` (ablation) | `correctness` | `rank(correctness) − mean(rank)`, no tiebreaker |

`faith` = internal spatial attention probe (AUROC 0.68–0.78 on SLAKE) + lookback lens signal, z-score-normalized.

Tiebreaker advantage construction (in `train_grpo.py:1091`):

```python
sorted_idx = sorted(fmt_idx, key=lambda i: (rollouts[i]["correctness"], rollouts[i]["faithfulness"]), reverse=True)
N = len(sorted_idx)
mean_rank = (N - 1) / 2.0
advantages[i] = (N - 1 - pos) - mean_rank  # centered rank, range [-(N-1)/2, +(N-1)/2]
```

Faith never enters reward magnitude. It only disambiguates ties in correctness-sort.

## SLAKE test set results (1061 English questions, full test set, greedy decode)

Best-val-correct checkpoint used for each run.

| Model | Overall F1 | Exact | Closed Q F1 | Open Q F1 | Val peak | Val→Test gap |
|---|---|---|---|---|---|---|
| zero_shot | (pending final report) | — | — | — | — | — |
| corr_s42 | **0.3919** | 396/1061 | 0.5365 | 0.2986 | 0.4844 | −0.0925 |
| full_s42 | **0.4126** | 419/1061 | 0.5888 | 0.2991 | 0.5126 | −0.1000 |
| **tiebreak_s456** | **0.5340** | 562/1061 | 0.7372 | 0.4030 | 0.4928 | **+0.0412** |

### Deltas (tiebreak_s456 vs baselines)

| Comparison | Absolute F1 | Relative |
|---|---|---|
| vs. full_s42 | +0.1214 | +29.4% |
| vs. corr_s42 | +0.1421 | +36.3% |

**Gap split:** Tiebreak gains +0.15 on closed Q, +0.10 on open Q vs full. Gain is larger on closed than open, suggesting tiebreaker learns discriminative binary decisions especially well.

### The generalization pattern

- **corr and full overfit**: val peaks ~0.48–0.51, test drops to 0.39–0.41. Policies specialize to the organ-only training distribution.
- **Tiebreaker does not overfit**: val peak 0.49, test **higher** at 0.53. Policy learned the underlying task rather than the training-distribution artifacts.

Mechanism hypothesis: rank-based advantage with faith-tiebreaker preserves the pure-correctness objective exactly (no reward contamination from faith signal), so the policy does not chase faith patterns that bind to training-distribution artifacts. Corr_only has the same invariance but lacks the dense gradient signal from tiebreaker, so converges to a lower-quality solution.

## Val trajectories (correctness only, token F1)

| Step | corr_s42 | corr_s456 | full_s42 | tiebreak_s456 | tiebreak_s42 |
|---|---|---|---|---|---|
| 10 | 0.3151 | 0.2581 | 0.2780 | 0.2806 | 0.3000 |
| 40 | 0.3381 | 0.3021 | 0.3092 | 0.2955 | 0.3893 |
| 70 | 0.3524 | 0.3596 | 0.3945 | 0.3997 | — |
| 100 | 0.4106 | 0.3822 | 0.4498 | 0.4521 | — |
| 130 | 0.4442 | 0.4013 | 0.4757 | 0.4651 | **0.5148** |
| 160 | — | — | 0.4815 | 0.4844 | — |
| 170 | 0.4792 | 0.4332 | 0.5020 | **0.4928** | — |
| 180 | 0.4844 | 0.4406 | **0.5126** | — | — |

### Matched-seed comparisons

**Seed 456 (weak seed), step 140:**
- corr_s456: 0.4222
- tiebreak_s456: 0.4766 — **Δ +0.054 absolute, +12.9% relative**

**Seed 42 (strong seed), step 130:**
- corr_s42: 0.4442
- full_s42: 0.4757
- tiebreak_s42: **0.5148** — **Δ +0.071 over corr, +0.039 over full**

### Cross-method peak (sweep leaderboard by val)

1. tiebreak_s42: 0.5148 (step 130)
2. full_s42: 0.5126 (step 180)
3. tiebreak_s456: 0.4928 (step 170)
4. corr_s42: 0.4844 (step 180)
5. corr_s456: 0.4406 (step 180)

## Mechanism analysis

Three reasons correctness-only GRPO underperforms:

1. **Coarse signal.** Token F1 takes few discrete values per batch. Mean-subtracted advantage gives small magnitudes when rewards cluster near the mean.
2. **Zero-advantage batches.** When all 8 rollouts tie on correctness, advantage = 0 for everyone → batch skipped. Training stalls on hard examples exactly where it needs to learn.
3. **Coarse signal + tied batches = lost examples.** Dense gradient direction is only available on a fraction of training examples.

Tiebreaker addresses (2) directly: ties in correctness are broken by faith, producing non-zero advantages. Rank-based advantage also addresses (1): ranks span fixed range [−(N−1)/2, +(N−1)/2], giving consistent gradient magnitude regardless of reward distribution.

`corrrank` ablation (running): rank-based advantage on correctness alone (no tiebreaker). Isolates whether rank structure alone is responsible for the gain, or whether the faith-tiebreaker specifically matters. Early data (step 50–60) shows corrrank at ~0.31, well below tiebreak at matched step.

## Related work

- **GOPO** (Group Ordinal Policy Optimization, arXiv 2602.03876, Feb 2026): rank-based advantage in GRPO with single scalar reward. No compositional or tiebreaker mechanism.
- **FaithRL** (arXiv 2602.05897, Feb 2026): step-level faithfulness reward via external PRM + truncated resampling. Different domain (text QA), different mechanism (additive composite + contrastive).

Our construction — compositional lex-rank advantage with ordinal auxiliary tiebreaker — occupies the gap between these two lines of work.

## Caveats (as of 2026-04-21)

- n=1 seed with completed test eval (tiebreak_s456). tiebreak_s42 finishing training, test eval pending.
- corrrank ablation not yet produced val > step 60 data. Cannot fully separate rank effect from tiebreaker effect yet.
- Full fine-tune not tested (LoRA only). Reviewer concern: may the effect be LoRA-specific.
- Only SLAKE organ-only training distribution. VQARAD/PathVQA OOD eval not yet run on tiebreaker.

## Next steps

1. Complete tiebreak_s42 training to convergence (ETA +2 days).
2. Run SLAKE test eval on tiebreak_s42 and corrrank_s{42,456} checkpoints.
3. Run cross-dataset eval (VQARAD + PathVQA) on all best_correct checkpoints.
4. Possible: 3rd seed (s123) for full sweep replication.

## Paths (remote, vlaa-01)

- Code: `/data3/ishaplan/cse40_slake_repro/final_version/src/faithscan/train_grpo.py`
- Configs: `/data3/ishaplan/cse40_slake_repro/slake_reproduction/configs/{correctness_only,spatial_grpo_a07_fullprobe,tiebreaker_slake,correctness_rank_slake}_*.yaml`
- Logs: `/data3/ishaplan/cse40_slake_repro/slake_reproduction/results/{corr,full,tiebreak,corrrank}_*.log`
- Checkpoints: `/data3/ishaplan/slake_reproduction/checkpoints_5ep/{correctness_only,spatial_grpo_fullprobe,tiebreaker,correctness_rank}/seed{42,456}/`
- SLAKE test eval: `/data3/ishaplan/cse40_slake_repro/slake_reproduction/results/slake_test_4way.json`
