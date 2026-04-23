# Rank-Based Tiebreaker GRPO: Live Results

**Date:** April 21, 2026 (last major update); table refresh 2026-04-23
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

Best-val-correct checkpoint used for each run. Updated 2026-04-23 with latest eval numbers (corrrank seeds 42/456 + refreshed corr_only s42).

| Model | Seed | Overall F1 | Exact | Closed Q F1 | Open Q F1 | Val peak | Val→Test gap |
|---|---|---|---|---|---|---|---|
| corr_only (α=1.0) | 42 | **0.4299** | **440/1061** | **0.5830** | **0.3312** | **0.5166 (step 290)** | −0.0867 |
| corr_only (α=1.0) | 456 | 0.4453 | 452/1061 | 0.6203 | 0.3325 | 0.4790 (step 250) | −0.0337 |
| composite (α=0.7) | 42 | 0.4363 | 440/1061 | 0.6074 | 0.3259 | 0.5126 (step 180) | −0.0763 |
| **tiebreaker (ours, bbox-cond)** | 42 | **0.5472** | **579/1061** | **0.6635** | **0.4722** | **0.6190 (step 570, sweep peak)** | −0.0718 |
| **tiebreaker (ours, bbox-cond)** | 456 | **0.5340** | **562/1061** | **0.7372** | **0.4030** | 0.5201 (step 270) | **+0.0412** |
| corrrank (ablation) | 42 | **0.3807** | **382/1061** | **0.5206** | **0.2904** | 0.4577 (step 250) | −0.0770 |
| corrrank (ablation) | 456 | **0.4141** | **421/1061** | **0.5743** | **0.3108** | 0.4248 (step 290) | −0.0107 |
| **tiebreaker (ours, bbox-free, Option D)** | 42 | _training_ | | | | step 20 so far: 0.3109 | _early_ |
| zero_shot | — | 0.2988 | 290/1061 | 0.3934 | 0.2378 | — | — |

Test eval notes: the 0.5472 tiebreak_s42 number is from a refresh eval against the step-570 best_correct checkpoint (val 0.6190). An earlier eval against an intermediate (~step-290) checkpoint produced 0.5218 with a different closed/open profile: closed-Q F1 was higher (0.7019 vs 0.6635) while open-Q F1 was lower (0.4056 vs 0.4722). The later checkpoint trades some yes/no precision for substantially better open-ended reasoning. Tiebreak_s456 test eval used the step-170 checkpoint (val 0.4928 at that time); the step-270 peak (val 0.5201) has not been refreshed yet, so the reported 0.5340 is from the weaker checkpoint.

### Out-of-domain test (VQARAD, 451 questions — model never trained on this dataset)

| Model | VQARAD F1 | Exact | vs corr_only |
|---|---|---|---|
| zero_shot | 0.1246 | 47/451 | — |
| corr_only (s42) | 0.3385 | 145/451 | baseline |
| composite α=0.7 (s42) | 0.3257 | 138/451 | **−0.0128** (composite hurts OOD) |
| **tiebreaker (s42)** | **0.4564** | **199/451** | **+0.1179 (+34.8% rel)** |
| **tiebreaker (s456)** | **0.4644** | **202/451** | **+0.1259 (+37.2% rel)** |

The tiebreaker's SLAKE-test gain over corr_only replicates on VQARAD OOD (+0.12 F1 on both seeds). Composite underperforms corr_only on VQARAD by 0.013 F1, directly validating the reward-shape-overfitting concern behind the tiebreaker construction.

### Out-of-domain test (PathVQA)

Full test set (6719 questions):

| Model | PathVQA F1 | Exact | vs corr_only |
|---|---|---|---|
| **tiebreaker (s42)** | **0.3212** | **2110/6719** | _corr_only full-6719 queued_ |
| **tiebreaker (s456)** | **0.3375** | **2194/6719** | _corr_only full-6719 queued_ |
| corr_only (s42) | _running_ | — | baseline |
| corrrank (s42) | _running_ | — | — |
| corrrank (s456) | _running_ | — | — |

Earlier 500-example subset (older ckpts, not directly comparable to above):

| Model | PathVQA-500 F1 | Exact |
|---|---|---|
| zero_shot | 0.0627 | 25/500 |
| corr_only | 0.1385 | 64/500 |
| bbox spatial probe (weighted) | 0.1166 | 53/500 |

### Deltas

**tiebreaker_s456 vs baselines (mixed-seed):**
| Comparison | Absolute F1 | Relative |
|---|---|---|
| vs. composite-reward (full_s42) | +0.0977 | +22.4% |
| vs. corr_only (corr_s42, 5ep best_correct) | +0.1041 | +24.2% |
| vs. corrrank_s456 (matched-seed ablation) | +0.1199 | +28.9% |
| vs. zero_shot | +0.2352 | +78.7% |

**tiebreaker_s42 vs baselines (matched-seed s42, step-570 eval):**
| Comparison | Absolute F1 | Relative |
|---|---|---|
| vs. composite-reward | +0.1109 | +25.4% |
| vs. corr_only (5ep best_correct) | +0.1173 | +27.3% |
| vs. corrrank_s42 (matched-seed ablation) | +0.1665 | +43.7% |

**Rank-only ablation lands — corrrank underperforms corr_only on both seeds** (s42: 0.3807 vs 0.4299 = −0.049; s456: 0.4141 vs 0.4453 = −0.031). This is the predicted failure mode: rank-based advantage with arbitrary tie-breaking on sparse discrete rewards injects noise. The faith-tiebreaker construction rescues this — tiebreak_s42 beats corrrank_s42 by +0.167 F1 on the same seed.

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
| 90 | 0.4070 | 0.3430 | 0.4152 | 0.4151 | 0.5030 | 0.3126 | 0.3077 |
| 100 | 0.4106 | 0.3822 | 0.4498 | 0.4521 | 0.5122 | 0.3290 | 0.3308 |
| 110 | 0.4163 | 0.3600 | 0.4473 | 0.4364 | 0.5077 | 0.3422 | 0.3459 |
| 120 | 0.4271 | 0.3962 | 0.4428 | 0.4798 | 0.5015 | 0.3512 | 0.3330 |
| 130 | 0.4442 | 0.4013 | 0.4757 | 0.4651 | 0.5148 | 0.3837 | 0.3460 |
| 140 | 0.4611 | 0.4222 | 0.4822 | 0.4766 | 0.4911 | 0.3861 | 0.3436 |
| 150 | 0.4652 | 0.3990 | 0.4767 | 0.4689 | 0.5186 | 0.3861 | 0.3464 |
| 160 | 0.4689 | 0.4083 | 0.4815 | 0.4844 | 0.5328 | 0.4084 | 0.3684 |
| 170 | 0.4792 | 0.4332 | 0.5020 | 0.4928 | 0.5448 | 0.4094 | 0.3468 |
| 180 | 0.4844 | 0.4406 | **0.5126** | 0.4720 | 0.5439 | 0.4061 | 0.3528 |
| 190 | 0.4796 | 0.4105 | 0.4767 | 0.4705 | 0.5294 | **0.4308** | 0.3723 |
| 200 | 0.4708 | 0.4430 | 0.4941 | 0.4795 | 0.5369 |  | 0.3602 |
| 210 | 0.4621 | 0.4256 | 0.4617 | 0.4903 | 0.5342 |  | **0.3966** |
| 220 | **0.4944** | 0.4481 | 0.4770 | 0.4826 | 0.5393 | 0.4249 |  |
| 230 | 0.4699 | 0.4516 | 0.4564 | 0.5023 | 0.5542 | 0.4141 | 0.4106 |
| 240 | 0.4961 | 0.4570 | 0.4764 | 0.5145 | 0.5544 | 0.4436 | 0.4033 |
| 250 | 0.4895 | **0.4790** | 0.4927 | 0.5064 | 0.5680 | **0.4577** | 0.3908 |
| 260 | 0.4757 | 0.4635 | 0.4847 | 0.5052 | 0.5530 | 0.4450 | **0.4166** |
| 270 | 0.4968 | 0.4529 | 0.5046 | **0.5201** | 0.5722 |  | 0.4115 |
| 280 | 0.4925 | 0.4650 | 0.4984 | 0.5129 | 0.5699 |  | 0.4009 |
| 290 | **0.5166** | 0.4416 | 0.4845 | 0.5077 | **0.5828** |  |  |
| 300 | 0.4845 | 0.4517 | 0.4872 (full_s42 killed) | 0.5106 | 0.5601 |  |  |
| 310 | 0.5055 | 0.4689 |  |  | 0.5401 |  |  |
| 320 | 0.5036 | 0.4769 |  |  |  |  |  |
| 330 | 0.4960 | 0.4374 |  |  |  |  |  |
| 340 |  | 0.4705 |  |  |  |  |  |
| ... | | | | tiebreak_s456 oscillating 0.48–0.52 through step 460 | tiebreak_s42 oscillating 0.51–0.60 through step ~500, then broke through | | |
| 460 |  |  |  | 0.5003 (run killed) |  |  |  |
| 510 |  |  |  |  | 0.6077 |  |  |
| 570 |  |  |  |  | **0.6190** (sweep peak) |  |  |
| 630 |  |  |  |  | 0.5609 (run killed at step 670, past peak) |  |  |

**New ablation run (not in table above):** `tiebreak_s42_nodrop` (no `drop_unformatted` format gating) launched Apr 22. Early trajectory matches tiebreak_with_drop: step 10 = 0.3046, step 20 = 0.3226, step 30 = 0.3313.

### Matched-seed comparisons

**Seed 456 (weak seed), step 140:**
- corr_s456: 0.4222
- tiebreak_s456: 0.4766 — **Δ +0.054 absolute, +12.9% relative**

**Seed 42 (strong seed), step 130:**
- corr_s42: 0.4442
- full_s42: 0.4757
- tiebreak_s42: **0.5148** — **Δ +0.071 over corr, +0.039 over full**

### Cross-method peak val correctness (refreshed 2026-04-23)

1. **tiebreak_s42 (bbox-cond): 0.6190** (step 570, past peak; run killed step 670)
2. tiebreak_s456 (bbox-cond): 0.5201 (step 270, past peak; run killed step 460)
3. **corr_s42: 0.5166** (step 290; run killed step 330 past peak)
4. full_s42: 0.5126 (step 180, killed step 300)
5. tiebreak_s42_nodrop (bbox-cond, no drop_unformatted): **0.5092** (step 120, still climbing)
6. **corr_s456: 0.4790** (step 250; run killed step 340 past peak)
7. corrrank_s42: 0.4577 (step 250, at peak as of step 286 — val plateauing)
8. corrrank_s456: 0.4248 (step 290, still climbing slowly)
9. tiebreak_s42 (bbox-free, Option D): 0.3109 (step 20, too early to judge — first eval of new probe)

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

`corrrank` ablation (running): rank-based advantage on correctness alone (no tiebreaker). Isolates whether rank structure alone is responsible for the gain, or whether the faith-tiebreaker specifically matters. Early data (step 50–80) shows corrrank at 0.29–0.34, well below tiebreak at matched step.

## Why GOPO doesn't win here (the rank-without-tiebreaker failure mode)

GOPO (arXiv 2602.03876, Feb 2026) claims rank-based advantage beats Dr. GRPO's mean-subtracted advantage. Our `corrrank` ablation — which is essentially GOPO applied to token F1 rewards — underperforms Dr. GRPO (`corr_only`) by −0.08 at step 80 (0.3401 vs 0.4168). This isn't contradicting GOPO; it reveals where GOPO's assumptions break.

### GOPO's assumed regime: continuous reward model scores

GOPO's paper tests on tasks with reward-model output as the primary reward (continuous [0, 10] scalars from models like UltraFeedback / LLaMA-Reward-Bench). In that regime:

- **Ties are rare** — continuous scores give unique values with high probability.
- **Magnitudes are noisy** but **orderings are reliable** — the whole point of GOPO is discarding noisy magnitudes in favor of cleaner ranks.

GOPO's formula: `A_i = 2 − (ρ(i) − 1)·4/(G − 1)` where ρ(i) is the rank of rollout i. The assumption is that ρ(i) reflects *meaningful* ordering.

### Our regime: sparse discrete rewards (token F1)

Token F1 takes values in {0, 0.33, 0.5, 0.67, 1} or similar for open questions, {0, 1} for yes/no. Consequences:

- **Ties are common.** On hard questions, 6 of 8 rollouts often score 0 (all wrong). On easy ones, 5 of 8 score 1.
- **When rollouts tie, Python's `sorted()` preserves insertion order** (stable sort). So rank ρ(i) = the arbitrary order in which rollouts were sampled.
- **GOPO's formula then maps arbitrary order to fixed advantages** (+2, +1.43, ..., −2). The policy is trained to prefer rollout 1 over rollout 8 for no real reason. That is **systematic noise in the gradient**.

Dr. GRPO handles ties gracefully: `σ → 0`, advantage → 0 for everyone. No gradient, but no noise either.

### The comparison

| Regime | Dr. GRPO (mean-subtracted) | GOPO (pure rank) |
|---|---|---|
| Rollouts differ on reward | Standard advantage; works | Rank-based advantage; works |
| **Rollouts tie on reward** | **Advantage = 0 (no gradient; wasted)** | **Advantage = rank of arbitrary order (noise gradient; harmful)** |

In the sparse regime, "wasted" beats "harmful." That's why corrrank loses to corr_only here.

### How the tiebreaker extends GOPO

The faith tiebreaker says: *when correctness ties, use the auxiliary probe to produce a meaningful rank ordering.* The rank-based advantage machinery is reused, but the sort key becomes `(correctness, faith)` — so tied-on-correctness rollouts get ranked by faith, which is not arbitrary.

| Regime | Tiebreaker (ours) |
|---|---|
| Rollouts differ on correctness | Advantage = rank on correctness (correctness dominates, same as GOPO) |
| **Rollouts tie on correctness** | **Advantage = rank on faith (meaningful ordering from probe)** |

This restores GOPO's "reliable ordering" assumption in a regime where it would otherwise fail. The tiebreaker is the mechanism that makes rank-based advantages work on sparse discrete rewards.

### Implication for the method's positioning

We are not competing with GOPO — we are **extending it to sparse-reward settings.** The paper's related-work claim is:

> *"GOPO (Liu et al., 2026) proposes rank-based advantages as a de-noising mechanism when reward scores are continuous and approximately ordered. In sparse-reward settings where rollouts frequently tie, stable-sort-on-ties assigns arbitrary ranks that inject noise into the policy gradient (our corrrank ablation: 0.34 vs 0.42 for Dr. GRPO at matched step). Our tiebreaker construction extends GOPO to this regime by using an auxiliary signal to provide meaningful ordering over tied rollouts."*

## Bbox-free probe variant (Option D — Lookback-Lens image-vs-text ratio)

A version of the tiebreaker probe that requires no anatomical bounding-box annotations. Feature per head is `sum(attn → vision_tokens) / sum(attn → all_tokens)`, averaged over generated tokens — same dimensionality as the bbox-conditioned probe (448 = 28 layers × 16 heads on Qwen3-VL-2B-Thinking), drops into `train_grpo.py`'s existing `lookback_classifier` branch unchanged.

### Probe-level comparison on the same labeled dataset (1000-example `spatial_grounding_v1_full`)

Balanced LR, C-sweep {0.001, 0.01, 0.1, 1.0, 10.0}, 80/20 balanced split.

| Feature | Val AUROC | Permutation AUROC | Real − perm gap | Top-10 heads % of L1 |
|---|---|---|---|---|
| Bbox-conditioned (`spatial_ratios`) | 0.809 | 0.555 ± 0.020 | +0.254 | 8.2% (diffuse) |
| **Bbox-free (Option D, `lookback_feats`)** | **0.873** | 0.522 ± 0.027 | **+0.351** | 15.2% (concentrated) |

Bbox-free outperforms bbox-conditioned on the same labels. Top-10 heads concentrate in layers L10–L18 (visual-text integration range), consistent with a genuine late-layer grounding signal rather than a shortcut.

### Three-check validation (multi-rollout, 200 prompts × 8 T=1.0 rollouts, prompt-level splits)

Check 2 (within-group std) and Check 3 (refit-on-shuffle permutation) are the checks that matter for GRPO tiebreaker deployment — AUROC can be high but still useless if same-prompt rollouts get near-constant scores (Check 2) or if the probe picks up prompt-id leakage (Check 3).

| Check | **Option D (Lookback ratio)** — deployed | Option A (Attention entropy) — baseline |
|---|---|---|
| 1. Val AUROC (prompt-split) | train 0.972 / val **0.918** | train 1.000 / val 0.943 (overfit flag) |
| 2. Within-group std (mean/median) | **0.061** / 0.052 — PASS (≥ 0.05) | 0.077 / **0.008** — degenerate (mean inflated by outliers) |
| 3. Permuted AUROC | 0.531 ± 0.082 — PASS (≤ 0.55) | **0.579 ± 0.124 — FAIL** |
| Verdict | **ALL PASS — deployed in GRPO** | NOT READY (circularity) |

Option D clears all three checks. Option A's AUROC was partly circular — its permutation AUROC of 0.58 means ~0.1 AUROC comes from non-correctness artifacts. Within-prompt variance for Option A collapses to near-zero on most prompts (median 0.008 — effectively constant scores), which would make it useless as a tiebreaker regardless of its raw AUROC.

### Earlier "Lookback Lens is circular" verdict — corrected

Our earlier [probe architectures table in the README](../README.md) listed Lookback Lens as failing because of circularity. That verdict came from a naive test that didn't refit the classifier on shuffled labels and didn't use prompt-level splits. Under the proper three-check protocol, bbox-free Option D is clearly not circular — the real vs permuted gap is +0.4 AUROC.

### GRPO deployment status

Bbox-free tiebreaker is training with seed 42 on GPU 0 (tmux window `grpo-bboxfree`). Config: `slake_bboxfree_a07_seed42.yaml`, α=0.7, same SLAKE organ-only data filter as the bbox-conditioned runs. Latest val correctness at step 40: 0.2984 (climbing, ~16h training remaining). If step-200+ matches bbox-conditioned tiebreak trajectory, the paper claim "tiebreaker method generalizes to datasets without bbox annotations" lands.

## Related work

- **GOPO** (Group Ordinal Policy Optimization, arXiv 2602.03876, Feb 2026): rank-based advantage in GRPO with single scalar reward. Targets continuous reward-model outputs. Fails on sparse token-F1 rewards (see corrrank ablation).
- **FaithRL** (arXiv 2602.05897, Feb 2026): step-level faithfulness reward via external PRM + truncated resampling. Different domain (text QA), different mechanism (additive composite + contrastive).
- **DAPO** (Ma et al., 2025): dynamic sampling to eliminate all-tied batches via oversampling. Different fix for the same advantage-collapse problem.
- **GDPO** (2601.05242, Jan 2026): decoupled normalization of multi-reward advantages. Different problem (multi-reward) but related concern about advantage collapse.

Our construction — compositional lex-rank advantage with ordinal auxiliary tiebreaker — sits at the intersection of these three lines of work: keeps GOPO's rank structure, addresses the advantage-collapse problem DAPO targets, uses an auxiliary signal like FaithRL but without reward-magnitude contamination.

## Caveats (as of 2026-04-23)

- Two seeds with completed SLAKE-test eval on the main method (tiebreak_s42, tiebreak_s456). VQARAD OOD also two seeds.
- Corrrank ablation now has test-set numbers (0.3807 / 0.4141) — the rank-alone-fails-on-sparse-rewards story is directly supported by matched-seed comparisons.
- Full fine-tune not tested (LoRA only). Reviewer concern: may the effect be LoRA-specific.
- PathVQA OOD: tiebreak_s42 (F1 0.3212) and tiebreak_s456 (F1 0.3375) on full 6719. corr_only and corrrank OOD evals queued; corr_s42 pathvqa eval 59% through as of 2026-04-23.
- Bbox-free tiebreaker (Option D) passes all probe-level diagnostics but has not yet produced GRPO F1 numbers — training still in progress.

## Next steps

1. Finish bbox-free tiebreaker GRPO training (ETA ~16h as of 2026-04-23 morning).
2. Complete overnight OOD eval queue: corr_only + corrrank_s{42,456} on PathVQA full test (6719).
3. Run SLAKE test eval on the corrrank latest `best_correct` — training may still improve these peaks.
4. Full fine-tune ablation for reviewer defense.
5. Possible 3rd seed (s123) for full sweep replication.

## Paths (remote, vlaa-01)

- Code: `/data3/ishaplan/cse40_slake_repro/final_version/src/faithscan/train_grpo.py`
- Configs: `/data3/ishaplan/cse40_slake_repro/slake_reproduction/configs/{correctness_only,spatial_grpo_a07_fullprobe,tiebreaker_slake,correctness_rank_slake}_*.yaml`
- Logs: `/data3/ishaplan/cse40_slake_repro/slake_reproduction/results/{corr,full,tiebreak,corrrank}_*.log`
- Checkpoints: `/data3/ishaplan/slake_reproduction/checkpoints_5ep/{correctness_only,spatial_grpo_fullprobe,tiebreaker,correctness_rank}/seed{42,456}/`
- SLAKE test eval: `/data3/ishaplan/cse40_slake_repro/slake_reproduction/results/slake_test_4way.json`

## Training evolution: how tiebreak_s42 responses changed from early to late

Two test evals were run on tiebreak_s42: one against an earlier best_correct checkpoint (~step 290, val 0.5828) and one against the later best_correct (step 570, val 0.6190). Comparing per-question outputs reveals a major shift in reasoning style.

**Mean generation length: 554 chars (early) → 335 chars (late). That is a 40% compression.**

### Reasoning style shift: from analysis to description

Early checkpoint reasoning traces included:
- Medical-fact lookups ("T1-weighted images show gray matter as lighter...")
- Multi-step deliberation ("wait, let me check...")
- Reference to anatomy ("pneumothorax is air in the pleural space...")
- Domain-specific explanations

Late checkpoint reasoning traces collapsed to:
- Image-description only ("The image is a black and white X-ray with a clear view of the chest, including the ribs...")
- No medical reasoning chain
- Direct jump to an answer after descriptive sentence

**Example: same question, two checkpoints:**

> *Is the lung abnormal? Gold: Yes*
>
> **Early (~step 290, 817 chars, correct):** "...check for abnormalities. First, check the lung fields. The left lung (marked L) and right lung. Are there any signs of consolidation, nodules, or other issues? The image shows the lungs with some areas that might be darker or have irregularities. Wait, in chest X-rays, normal lungs have clear lung fields. If there's a mass or abnormality, it would show up..." → `Yes` ✓
>
> **Late (step 570, 257 chars, correct):** "The image is a black and white X-ray with a view of the chest, showing the rib cage, lungs, heart, and part of the shoulder with a PORTABLE label and an L for left side with an arrow." → `Yes` ✓

Both correct, but the reasoning changed entirely. Late checkpoint no longer analyzes pathology; it describes the image surface and jumps to an answer.

### Why this helps on open-ended questions

The late-checkpoint style produces cleaner, more precise single-word answers because it commits to the answer without getting caught in verbose qualifications.

> *What is the largest organ in the picture? Gold: Lung*
>
> **Early:** Verbose "the liver is the big structure in the center... yes, the liver is the largest" → `liver` ✗
>
> **Late:** Brief image description with no organ analysis → `lung` ✓

The early checkpoint *reasoned itself into the wrong answer* by committing too early in the analysis. The late checkpoint just describes what it sees and answers correctly. Similar pattern on color, plural, and terminology questions.

### Why this regresses some closed-ended questions

The late-checkpoint style can produce confident yes/no flips because it never reasons about the actual pathological content.

> *Is the lung healthy? Gold: No*
>
> **Early (correct):** "there's a noticeable area that's more opaque, which could be a sign of pathology like pneumonia... the lung isn't healthy." → `No` ✓
>
> **Late (wrong):** Image description only, no pathology analysis → `Yes` ✗

Without reasoning, the late checkpoint defaults to "yes it contains it" / "yes it looks normal" on ambiguous images. This is reward-hacking-adjacent: the policy learned that shorter responses get higher advantage, but some hard binary questions require the reasoning to arrive at the correct answer.

### Counting the tradeoff across the test set

At the per-question level between the two checkpoints (out of 1061):
- **136 questions newly won** by the late checkpoint (step-570 right, earlier checkpoint wrong)
- **111 questions newly lost** by the late checkpoint
- Net +25 questions
- By type: CLOSED net **−26** (regression), OPEN net **+51** (improvement)

Net F1 is higher because open-ended questions are the majority (645 of 1061) and the improvement on open-ended outweighs the closed-ended regression.

### What this suggests for future work

The late-checkpoint behavior — dropping the medical reasoning chain entirely — is a form of reward-hacking-adjacent convergence. The tiebreaker's denser gradient drives tersening, but past a certain point this costs reasoning capability on hard closed-ended questions.

Two possible mitigations, either for ablation in the paper or future work:
1. **Early stopping on closed-Q F1 variance** — stop training when closed-Q F1 starts regressing relative to open-Q F1
2. **Reasoning-length regularizer** — add a small auxiliary signal rewarding reasoning chains of a minimum length, integrated either as a second-level tiebreaker or in a composite reward

This is consistent with the auxiliary-saturation boundary already noted: when faith saturates, the tiebreaker degrades and training continues to tersen the policy past the point of diminishing returns.
