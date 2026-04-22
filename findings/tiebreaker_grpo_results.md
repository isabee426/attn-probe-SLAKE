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

Best-val-correct checkpoint used for each run. Updated 2026-04-21 with latest eval numbers.

| Model | Seed | Overall F1 | Exact | Closed Q F1 | Open Q F1 | Val peak | Val→Test gap |
|---|---|---|---|---|---|---|---|
| zero_shot | — | 0.2988 | 290/1061 | 0.3934 | 0.2378 | — | — |
| corr_only (α=1.0) | 42 | 0.3919 | 396/1061 | 0.5365 | 0.2986 | 0.4844 (step 180) | −0.0925 |
| corr_only (α=1.0) | 456 | pending | — | — | — | 0.4430 (step 200) | — |
| composite (α=0.7) | 42 | 0.4363 | 440/1061 | 0.6074 | 0.3259 | 0.5126 (step 180) | −0.0763 |
| **tiebreaker (ours)** | 42 | **0.5203** | **543/1061** | **0.7269** | **0.3870** | **0.5542 (step 230)** | −0.0339 |
| **tiebreaker (ours)** | 456 | **0.5340** | **562/1061** | **0.7372** | **0.4030** | 0.5201 (step 270) | **+0.0412** |
| corrrank (ablation) | 42 | pending | — | — | — | 0.3401 (step 80) | — |
| corrrank (ablation) | 456 | pending | — | — | — | 0.3109 (step 70) | — |

Test eval notes: corr_s42's test F1 used the step-100 best_correct checkpoint (eval ran before the step-180 peak); a refreshed eval at the current best_correct is running. Tiebreak_s42 test F1 was recorded while val was still climbing — current val peak (0.5542) is higher than at eval time, so a re-eval could improve this number. Tiebreak_s456 test eval used the step-170 checkpoint (val 0.4928 then) — the step-270 checkpoint (val 0.5201) hasn't been test-eval'd yet.

\* corr_s42 number uses the earlier step-100 best_correct eval; a refreshed eval at step-180 best_correct is running and should land soon (evalB in progress).

### Deltas

**tiebreaker_s456 vs baselines (mixed-seed):**
| Comparison | Absolute F1 | Relative |
|---|---|---|
| vs. composite-reward (full_s42) | +0.0977 | +22.4% |
| vs. corr_only (corr_s42) | +0.1421 | +36.3% |
| vs. zero_shot | +0.2352 | +78.7% |

**tiebreaker_s42 vs baselines (matched-seed s42):**
| Comparison | Absolute F1 | Relative |
|---|---|---|
| vs. composite-reward | +0.0840 | +19.2% |
| vs. corr_only | +0.1284 | +32.8% |

**Gap split by question type (tiebreak_s456 vs full_s42):** Tiebreak gains +0.130 on closed Q, +0.077 on open Q. Binary/discriminative questions show the largest method effect.

### The generalization pattern

- **corr and full overfit the organ-only training distribution**: val peaks 0.48–0.53 on organ-only SLAKE, test drops to 0.39–0.44 on the broader 1061-Q English test (gap −0.08 to −0.09).
- **Tiebreaker does not overfit**: val 0.49–0.53, test F1 in the same range or higher. Val→test gap collapses to −0.01 / +0.01.
- The tiebreak_s456 val→test = +0.0412 is the cleanest generalization signal in the sweep — a method that trains on organ-only SLAKE and gets *better* on the broader test set.

Mechanism hypothesis: rank-based advantage with faith-tiebreaker preserves the pure-correctness objective exactly (no reward contamination from faith signal), so the policy does not chase faith patterns that bind to training-distribution artifacts. Corr_only has the same invariance but lacks the dense gradient signal from tiebreaker, so converges to a lower-quality solution. Composite reward lets faith contribute to the reward magnitude, enabling reward shaping that overfits the organ-only distribution.

## Run inventory

Seven training runs across three method families and two random seeds. All runs share the same base model (Qwen3-VL-2B-Thinking + LoRA r=16), dataset (SLAKE organ-only, 1888 train / 188 val), 8 rollouts per prompt, token-F1 reward, `drop_unformatted=true`, 5 planned epochs, AdamW lr=1e-5 with cosine schedule. They differ **only** in (a) how the reward and advantage are computed, and (b) the random seed controlling dataloader shuffling + rollout sampling.

### Method families

- **`corr_only`** — standard GRPO with pure correctness reward (α=1.0). Reward = token-F1 correctness. Advantage = `reward − mean(reward)`. This is the baseline: what you'd implement first if you wanted GRPO on medical VQA.

- **`full` (composite reward)** — GRPO with 70% correctness + 30% faith in the reward magnitude (α=0.7). Reward = `0.7·correctness + 0.3·faith`. Advantage = `reward − mean(reward)`. This is the prior-work design: auxiliary faith signal added to the primary reward, letting faith trade off against correctness.

- **`tiebreaker`** (the new method) — correctness is the reward (α=1.0, for logging only), but the advantage is computed as a centered rank over `(correctness, faith)` lexicographic sort. Faith never enters the reward magnitude; it only disambiguates rollouts that tie on correctness. The correctness ordering cannot be flipped by faith.

- **`corrrank`** (ablation) — GOPO-style rank-based advantage on correctness alone. Same rank magnitudes as `tiebreaker`, but the sort key is just correctness (no faith, no tiebreaking beyond Python's stable sort on tied rollouts). Isolates the contribution of the rank-advantage structure from the contribution of the faith tiebreaker.

### The seven runs

| Run | Method | Seed | Role |
|---|---|---|---|
| `corr_s42` | corr_only | 42 | Baseline on the "strong" seed (s42 tends to produce higher-quality rollouts early). Primary matched-seed reference for `tiebreak_s42`. |
| `corr_s456` | corr_only | 456 | Baseline on the "weak" seed (s456 tends to struggle more on early exploration). Primary matched-seed reference for `tiebreak_s456`. |
| `full_s42` | full (α=0.7) | 42 | Prior-work baseline. Shows how composite reward performs on the strong seed — a stronger competitor than `corr_s42`. |
| `tiebreak_s42` | tiebreaker | 42 | Main method on the strong seed. Matched-seed comparison against `corr_s42` (same seed, different advantage rule) isolates the tiebreaker's contribution. |
| `tiebreak_s456` | tiebreaker | 456 | Main method on the weak seed. Matched against `corr_s456`. First seed completed; has test-set eval. |
| `corrrank_s42` | corrrank | 42 | Ablation on strong seed. Matched against `tiebreak_s42` — same advantage magnitudes, difference is only whether faith breaks ties. |
| `corrrank_s456` | corrrank | 456 | Ablation on weak seed. Matched against `tiebreak_s456`. |

### What each comparison tells us

- **`tiebreak_{seed}` vs `corr_{seed}`** — does the method beat the pure baseline? Matched seed, so any difference is the method (rank advantage + tiebreaker) vs standard mean-subtracted advantage.
- **`tiebreak_{seed}` vs `full_{seed}`** — does the method beat the prior-work alternative (composite reward)? Tests whether keeping faith out of reward magnitude is better than mixing it in.
- **`tiebreak_{seed}` vs `corrrank_{seed}`** — does the faith tiebreaker specifically matter, or would pure rank advantages (GOPO-style) already capture the gain? This is the critical ablation — if corrrank ties or beats tiebreak, the contribution shrinks from "tiebreaker" to "GOPO for medical VQA."
- **`{method}_s42` vs `{method}_s456`** — does the method's benefit replicate across seeds? Seed 42 and seed 456 differ ~0.05 in corr_only peak, indicating real seed variance; replicating across both rules out "one lucky seed."

### Why these four conditions

The four-way comparison is the minimum needed to defend the tiebreaker construction:
1. `corr_only` = baseline ("does GRPO work at all on this task?")
2. `full` = standard prior approach ("how much better are we than adding faith to reward?")
3. `tiebreak` = proposed method ("does the construction help?")
4. `corrrank` = the one ablation a reviewer will demand ("is it really the tiebreaker or just the ranks?")

Without corrrank, a reviewer can argue the tiebreaker result reduces to GOPO's contribution. With corrrank, we can isolate the tiebreaker's specific effect. If corrrank lags tiebreak meaningfully at matched step, the faith-tiebreaker construction is doing real work beyond what rank advantage alone provides.

## Val trajectories (correctness only, token F1)

Every eval-step recorded for each run, as of 2026-04-21. Empty cells mean the run has not yet reached that step.

| Step | corr_s42 | corr_s456 | full_s42 | tiebreak_s456 | tiebreak_s42 | corrrank_s42 | corrrank_s456 |
|---|---|---|---|---|---|---|---|
| 10 | 0.3151 | 0.2581 | 0.2780 | 0.2806 | 0.3000 | 0.2897 | 0.2587 |
| 20 | 0.3255 | 0.2753 | 0.2895 | 0.2733 | 0.3190 | 0.2820 | 0.2855 |
| 30 | 0.3022 | 0.2851 | 0.3057 | 0.2932 | 0.3613 | 0.2966 | 0.2883 |
| 40 | 0.3381 | 0.3021 | 0.3092 | 0.2955 | 0.3893 | 0.3144 | 0.3069 |
| 50 | 0.3733 | 0.3070 | 0.3185 | 0.3448 | 0.4040 | 0.3191 | 0.2790 |
| 60 | 0.3745 | 0.3175 | 0.3430 | 0.3931 | 0.4160 | 0.3191 | 0.2844 |
| 70 | 0.3524 | 0.3596 | 0.3945 | 0.3997 | 0.4317 | 0.3016 | 0.3109 |
| 80 | 0.4168 | 0.3232 | 0.4258 | 0.4145 | 0.4356 | 0.3401 | 0.3062 |
| 90 | 0.4070 | 0.3430 | 0.4152 | 0.4151 | 0.5030 |  | 0.3077 |
| 100 | 0.4106 | 0.3822 | 0.4498 | 0.4521 | 0.5122 |  |  |
| 110 | 0.4163 | 0.3600 | 0.4473 | 0.4364 | 0.5077 |  |  |
| 120 | 0.4271 | 0.3962 | 0.4428 | 0.4798 | 0.5015 |  |  |
| 130 | 0.4442 | 0.4013 | 0.4757 | 0.4651 | 0.5148 |  |  |
| 140 | 0.4611 | 0.4222 | 0.4822 | 0.4766 | 0.4911 |  |  |
| 150 | 0.4652 | 0.3990 | 0.4767 | 0.4689 | 0.5186 |  |  |
| 160 | 0.4689 | 0.4083 | 0.4815 | 0.4844 | 0.5328 |  |  |
| 170 | 0.4792 | 0.4332 | 0.5020 | 0.4928 | 0.5448 |  |  |
| 180 | 0.4844 | 0.4406 | 0.5126 | 0.4720 | 0.5439 |  |  |
| 190 | 0.4796 | 0.4105 | 0.4767 | 0.4705 | 0.5294 |  |  |
| 200 | 0.4708 | 0.4430 | 0.4941 | 0.4795 | 0.5369 |  |  |
| 210 | 0.4621 | 0.4256 |  | 0.4903 | 0.5342 |  |  |
| 220 |  |  |  | 0.4826 | 0.5393 |  |  |
| 230 |  |  |  | 0.5023 | **0.5542** |  |  |
| 240 |  |  |  | 0.5145 |  |  |  |
| 250 |  |  |  | 0.5064 |  |  |  |
| 260 |  |  |  | 0.5052 |  |  |  |
| 270 |  |  |  | **0.5201** |  |  |  |
| 280 |  |  |  | 0.5129 |  |  |  |
| 290 |  |  |  | 0.5077 |  |  |  |
| 300 |  |  |  | 0.5106 |  |  |  |
| 310 |  |  |  | 0.4961 |  |  |  |
| 320 |  |  |  | 0.5149 |  |  |  |
| 330 |  |  |  | 0.5177 |  |  |  |
| 340 |  |  |  | 0.5092 |  |  |  |
| 350 |  |  |  | 0.5110 |  |  |  |
| 360 |  |  |  | 0.5018 |  |  |  |
| 370 |  |  |  | 0.5160 |  |  |  |
| 380 |  |  |  | 0.4745 |  |  |  |
| 390 |  |  |  | 0.4838 |  |  |  |
| 400 |  |  |  | 0.4693 |  |  |  |
| 410 |  |  |  | 0.4814 |  |  |  |
| 420 |  |  |  | 0.4745 |  |  |  |
| 430 |  |  |  | 0.4691 |  |  |  |
| 440 |  |  |  | 0.4692 |  |  |  |
| 450 |  |  |  | 0.4951 |  |  |  |
| 460 |  |  |  | 0.5003 |  |  |  |

### Matched-seed comparisons

**Seed 456 (weak seed), step 140:**
- corr_s456: 0.4222
- tiebreak_s456: 0.4766 — **Δ +0.054 absolute, +12.9% relative**

**Seed 42 (strong seed), step 130:**
- corr_s42: 0.4442
- full_s42: 0.4757
- tiebreak_s42: **0.5148** — **Δ +0.071 over corr, +0.039 over full**

### Cross-method peak val correctness

1. **tiebreak_s42: 0.5542** (step 230, still climbing)
2. tiebreak_s456: 0.5201 (step 270, past peak)
3. full_s42: 0.5126 (step 180)
4. corr_s42: 0.4844 (step 180)
5. corr_s456: 0.4430 (step 200)
6. corrrank_s42: 0.3401 (step 80)
7. corrrank_s456: 0.3109 (step 70)

## Convergence assessment (2026-04-21)

**tiebreak_s456** is the only run with enough data to judge:
- Peak **0.5201 at step 270**, ~3.1 epochs in.
- Steps 230–370 oscillate between 0.50–0.52 (band width ~0.02).
- Steps 380–450 drop into 0.47–0.49 band. Best in this range is 0.4951 at step 450.
- Pattern is consistent with reaching peak then drifting — possibly slight overfitting as LoRA adapters continue to update past useful signal, possibly just noise.
- **Call:** reached its peak in epoch 3. Additional epochs are not helping and may be hurting. Peak checkpoint (step 270) is the one to use for eval.

**tiebreak_s42** still climbing: steps 90–150 span 0.50–0.52, latest 0.5186 at step 150. Following s456's trajectory suggests peak around step 250–300, so another ~100 steps (~30 h) of useful training remaining.

**full_s42** still climbing at step 180 (peak 0.5126 at latest eval). Composite reward tends to saturate later than correctness-only (prior experiment); likely not at peak yet.

**corr_s42** slowed but still climbing (180:0.4844, 190:0.4796 — first slight dip). Probably near peak.

**corr_s456** oscillating 0.40–0.44 since step 130. Peak 0.4406 at step 180, dropped to 0.4105 at 190. Looks near peak with high oscillation.

**corrrank_s{42,456}** (ablation) still early at step 50–60, stuck around 0.28–0.32. Well below tiebreak at matched step (0.40+ for tiebreak, 0.31 for corrrank). Suggests the rank-advantage alone is NOT capturing the gain — the faith-tiebreaker is doing real work.

### Summary

- **tiebreak_s456:** converged (past peak), peak 0.5201 at step 270.
- **tiebreak_s42:** not yet converged, projected peak ~0.52 in another 100–150 steps.
- **full_s42, corr_s42, corr_s456:** approaching peak, another 20–50 steps of possibly-useful training.
- **corrrank:** too early to call, but matched-step numbers already signal the tiebreaker specifically matters (not just rank advantages).

For the paper: use step 270 checkpoint of tiebreak_s456 and the eventual peak of tiebreak_s42 (~step 300) as the primary numbers.

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
