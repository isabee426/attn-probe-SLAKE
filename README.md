# Spatial Attention Probe as Auxiliary Reward for Medical VQA GRPO

Training a classical logistic-regression probe on per-head attention features of a vision-language model, then using that probe as an ordinal tiebreaker inside GRPO to reduce hallucinations and improve medical VQA accuracy on SLAKE.

## Main result

Full SLAKE English test set (1061 questions, including non-organ questions the model was never trained on):

| Model | Seed | Test F1 | Exact | Closed Q | Open Q | Val peak (step) | Val→Test |
|---|---|---|---|---|---|---|---|
| corr_only (α=1.0) | 42 | 0.4086 | 417/1061 | 0.5720 | 0.3032 | 0.4944 (220) | −0.086 |
| corr_only (α=1.0) | 456 | 0.4453 | 452/1061 | 0.6203 | 0.3325 | 0.4516 (230) | −0.006 |
| composite (α=0.7) | 42 | 0.4363 | 440/1061 | 0.6074 | 0.3259 | 0.5126 (180) | −0.076 |
| **tiebreaker (ours)** | 42 | **0.5472** | **579/1061** | **0.6635** | **0.4722** | **0.6190 (570, sweep peak)** | −0.072 |
| **tiebreaker (ours)** | 456 | **0.5340** | **562/1061** | **0.7372** | **0.4030** | 0.5201 (270) | **+0.041** |
| zero_shot | — | 0.2988 | 290/1061 | 0.3934 | 0.2378 | — | — |

### Out-of-domain (VQARAD test, 451 questions — model never trained on this dataset)

| Model | VQARAD F1 | Exact | vs corr_only baseline |
|---|---|---|---|
| zero_shot | 0.1246 | 47/451 | — |
| corr_only (s42) | 0.3385 | 145/451 | baseline |
| composite α=0.7 (s42) | 0.3257 | 138/451 | **−0.013** (composite hurts OOD) |
| **tiebreaker (s42)** | **0.4564** | **199/451** | **+0.118 (+34.8% relative)** |
| **tiebreaker (s456)** | **0.4644** | **202/451** | **+0.126 (+37.2% relative)** |

**SLAKE test deltas (tiebreaker vs baselines, matched seed 42):**
- tiebreak_s42 vs composite (s42): **+0.1109 absolute F1, +25.4% relative**
- tiebreak_s42 vs corr_only (s42): **+0.1386 absolute F1, +33.9% relative**

**Mixed-seed comparisons (tiebreak_s456 vs s42 baselines):**
- vs composite: **+0.0977 absolute F1, +22.4% relative**
- vs corr_only: **+0.1254 absolute F1, +30.7% relative**

**Checkpoint selection note (tiebreak_s42):** the step-570 checkpoint (val 0.6190) has a different closed/open profile than the earlier step-290 checkpoint (val 0.5828): closed-Q F1 regressed slightly (0.7019 → 0.6635), open-Q F1 rose substantially (0.4056 → 0.4722). Net test F1 is higher (0.5472 vs 0.5218) because open-Q is the majority of the test set. The tiebreaker's deeper-training checkpoints trade some yes/no precision for better open-ended reasoning.

**Generalization pattern:** composite and corr_only overfit the organ-only training distribution (val → test drops 0.08–0.09 F1). Tiebreaker does not: val → test is flat or rising. The pattern replicates on OOD: composite underperforms corr_only on VQARAD (−0.013 F1), while tiebreaker gains +0.12 F1 over corr_only across both seeds.

## The method

| Method | Reward | Advantage |
|---|---|---|
| `corr_only` | correctness (token F1) | (reward − group mean) / group std |
| `composite` | 0.7·correctness + 0.3·faith | (reward − group mean) / group std |
| **`tiebreaker`** | correctness (logged; not used in advantage) | centered rank over (correctness, faith) lex-sort |

Faith signal never enters reward magnitude. It only disambiguates rollouts that tie on correctness. Correctness ordering is strictly preserved — no reward-hacking pathway.

```python
sorted_idx = sorted(fmt_idx, key=lambda i: (rollouts[i]["correctness"], rollouts[i]["faithfulness"]), reverse=True)
N = len(sorted_idx)
mean_rank = (N - 1) / 2.0
advantages[i] = (N - 1 - pos) - mean_rank  # ranks centered at 0
```

## Probe architecture (classical ML)

The auxiliary signal is a **logistic regression classifier** on per-head attention features. No neural network, no fine-tuning — a classical binary classifier trained from scratch per sweep.

### Feature extraction

For each (image, question, rollout) triple:
1. Run the VLM forward pass and collect attention weights at every layer × head.
2. For each head, compute the sum of attention from generated tokens to image patches within the ground-truth organ bounding box.
3. Normalize by total attention to image patches to get a **bbox attention ratio** per head.
4. Stack across layers × heads to get the feature vector.

Feature dimensionality: **28 layers × 16 heads = 448 features** per rollout.

### Probe variants

| Probe | Training data | Target | AUROC | Pearson r with correctness | Predict_proba gap |
|---|---|---|---|---|---|
| **Corrprobe** (weak) | 154 examples, balanced positives/negatives | correctness label (0/1) | 0.667 | 0.636 | 0.83 vs 0.32 (gap 0.51) |
| **Fullprobe** (strong) | 572 examples, balanced from 1276 candidates | correctness label (0/1) | 0.968 | 0.844 | 0.91 vs 0.15 (gap 0.76) |

Both probes:
- **Model class:** logistic regression (scikit-learn, default L2 penalty)
- **Loss:** binary cross-entropy
- **Regularization:** L2, default λ
- **Feature scaling:** z-score normalized per head across training set
- **Class balance:** upsampling/undersampling to equal positive/negative counts
- **Train/val split:** 80/20 within the labeled set
- **Held-out evaluation:** on VLM rollouts from the main training corpus

### At inference time in GRPO

During rollout generation:
1. Collect attention weights for each of the 8 rollouts.
2. Compute 448-dim feature vector per rollout.
3. Run logistic regression forward pass → probability of correctness ∈ [0, 1].
4. z-score normalize across the group; optional sigmoid.
5. Use as the auxiliary (faith) signal in the tiebreaker advantage.

### Why logistic regression (not a neural probe)

- **Interpretability:** coefficients map directly to per-head importance; can inspect which heads matter.
- **Speed:** forward pass is one matrix-vector multiply per rollout. Orders of magnitude faster than a neural probe.
- **Small-data regime:** 154–572 training examples is enough for LR but not for a deep probe.
- **Bias-variance trade-off:** the probe only needs to be *ordinally* informative on correctness ties. A logistic regressor captures linear separability in attention space, which is enough for this purpose.

## Configuration

| Parameter | Value |
|---|---|
| Base model | Qwen3-VL-2B-Thinking |
| PEFT | LoRA r=16, α=32, dropout=0.05, target q/k/v/o projections |
| Trainable params | 6,422,528 / 2.13B total (0.30%) |
| Dataset | SLAKE (English, organ-only subset: 5972 → 2076 examples, 35%) |
| Train/val split | 10:1 ratio → 1888 train, 188 val |
| Test sets | 1061 English test (full), 364 organ-only test (prior experiment) |
| GRPO rollouts per prompt | 8 |
| Max new tokens | 512 |
| Temperature | 1.0 (sampling), 0.0 (greedy eval) |
| α (composite only) | 0.7 (70% correctness + 30% faith) |
| Faith normalization | Global running z-score + sigmoid |
| Primary reward metric | Token F1 (strict: set-overlap after lowercase + punct strip) |
| Yes/no / numbers | Exact first-token match |
| Advantage baseline | EBPO shrinkage of group mean |
| Optimizer | AdamW, lr=1e-5, cosine schedule, weight decay 0.01, β=(0.9, 0.95), grad clip 1.0 |
| Gradient accumulation | 8 rollout-example batches per optimizer step |
| Format gating | `drop_unformatted=true` (unformatted rollouts excluded from gradient) |
| Epochs | 5 (current sweep) |
| Typical wall clock per run | ~3 days, 1× RTX 4090 (24 GB) |

## Why the tiebreaker works

GRPO's group-relative advantage becomes gradient-starved on coarse rewards like token F1: when all 8 rollouts tie on correctness (>50% of batches in our setting), σ → 0 and advantage → 0 for everyone. Two failure modes are fixed:

1. **Coarse-signal collapse.** Rank-based advantage gives fixed-magnitude gradient regardless of reward distribution (GOPO-style).
2. **Tied-group noise.** When rollouts tie on correctness, the faith tiebreaker produces meaningful ordering. Wrong rollouts that looked at the right region get positive advantage over wrong rollouts that hallucinated from nowhere.

Because the correctness ordering is never flipped by faith, there is no reward-hacking pathway. The faith probe can be noisy or biased (AUROC 0.67–0.97 in our runs) — it only needs to correlate with quality among correctness-tied rollouts.

## Prior experiment (2-epoch reproduction)

An earlier 2-epoch comparison on the **organ-only** subset of the SLAKE test set (364 questions — only questions about organs the model was trained on) produced these numbers:

| Model | Token F1 (364 organ-only) | Exact (>0.5) |
|---|---|---|
| Zero-shot | 0.275 | 92/364 (25.3%) |
| Corr-only | 0.447 | 156/364 (42.9%) |
| Corrprobe (r=0.636) | 0.424 | 148/364 (40.7%) |
| Fullprobe (r=0.844) | 0.445 | 154/364 (42.3%) |

On the in-distribution subset, corr-only and the composite-reward probes reached the same accuracy (0.447 vs 0.445), suggesting the probe as a composite reward improved training stability but not final accuracy.

The new tiebreaker experiment replaces this composite-reward approach with the rank-based tiebreaker, and the gap emerges on the broader 1061-Q test set where baselines overfit to organ-specific patterns. For the full tiebreaker results (val trajectories, matched-seed comparisons, corrrank ablation, GOPO comparison) see [`findings/tiebreaker_grpo_results.md`](findings/tiebreaker_grpo_results.md). For a per-question behavioral analysis of the current sweep (head-to-head tables, unique-wins examples, failure-mode categorization, generation-length statistics) see [`findings/tiebreak_behavioral_analysis.md`](findings/tiebreak_behavioral_analysis.md).

## Discussion

### Is the probe circular?

No. The probe maps **attention patterns** to correctness. When used as tiebreaker, it tells the model "attend like this" — not "be correct." The model cannot game it by memorizing answers; it has to shift its attention distribution. The Pearson correlation (r = 0.636–0.844) indicates a complementary signal from attention space rather than a copy of the correctness signal.

### Why does the tiebreaker generalize when composite does not?

The tiebreaker never puts faith into the reward magnitude. Faith only disambiguates correctness ties. The pure-correctness objective is preserved exactly. Composite reward (α=0.7) allows tradeoffs between correctness and faith, which helps in-distribution training but over-specializes the policy to faith patterns that bind to the training distribution.

### Why logistic regression, not a deep probe?

The tiebreaker mechanism uses only *ordinal* information from the probe — whether rollout A is more grounded than rollout B. A logistic regressor captures linear separability in attention-feature space, which is sufficient for this ordering. Deep probes would require more training data than we have for in-session feature collection and carry higher variance on a small calibration set.

### `drop_unformatted` = true is required

Without format gating, probe conditions collapse (e.g., corrprobe crashed to 0.233 val F1 in epoch 2 of prior experiments). With gating, both conditions reach equivalent in-distribution accuracy. Format-learning gradient dominates early training; `drop_unformatted` isolates the probe's contribution.

## Reproduction

```bash
git clone https://github.com/isabee426/attn-probe-SLAKE.git
cd attn-probe-SLAKE
export PYTHONPATH=src:/path/to/faithscan_vqarad/src

# 1. Train the logistic regression probe from extracted features
python scripts/retrain_probe_balanced.py \
    --features /path/to/spatial_features.npz \
    --output checkpoints/spatial_probe/

# 2. GRPO training (5 epochs)
CUDA_VISIBLE_DEVICES=0 python -m faithscan.train_grpo --config configs/reproduction/correctness_only_seed42.yaml
CUDA_VISIBLE_DEVICES=1 python -m faithscan.train_grpo --config configs/reproduction/spatial_grpo_a07_corrlabels_seed42.yaml
CUDA_VISIBLE_DEVICES=2 python -m faithscan.train_grpo --config configs/tiebreaker_slake_seed42.yaml

# 3. Test set evaluation (full 1061 English test)
CUDA_VISIBLE_DEVICES=0 python scripts/eval_test_set.py \
    --corr-ckpt checkpoints/correctness_only/seed42/best_correct \
    --fullprobe-ckpt checkpoints/spatial_grpo_fullprobe/a07_seed42/best_correct \
    --output results/test_set.json
```

## File structure

```
attn-probe-SLAKE/
├── README.md
├── src/faithscan/
│   ├── train_grpo.py                   # GRPO training (tiebreaker + rank advantage flags)
│   ├── reward.py                       # Correctness (token F1) + composite reward
│   ├── data/dataset.py                 # SLAKE / VQA-RAD / PathVQA loaders
│   └── models/
│       ├── lookback_lens.py            # Spatial attention probe + faith computation
│       └── dhcp_probe.py               # Cross-modal attention probe (legacy)
├── configs/
│   ├── original/                       # April 4 configs
│   ├── reproduction/                   # 3 conditions × 2 seeds (2-epoch prior)
│   └── tiebreaker_slake_seed*.yaml     # 5-epoch tiebreaker sweep (current)
├── scripts/
│   ├── train_spatial_probe.py          # Feature extraction + LR probe training
│   ├── retrain_probe_balanced.py       # Class-balanced probe retraining
│   ├── eval_test_set.py                # SLAKE test eval (strict token F1)
│   ├── compare_checkpoints.py          # Zero-shot vs trained comparison
│   └── rollout_analysis.py             # Per-question rollout diffing
└── findings/                           # Full experiment logs and analyses
    ├── tiebreaker_grpo_results.md      # Current — tiebreaker result details, val trajectories, ablations
    ├── tiebreak_s456_open_responses.md # 645 open-ended test responses with full reasoning traces
    ├── reproduction_findings.md        # Prior 2-epoch behavioral analysis + unique-wins tables
    ├── slake_final_results.md          # Earlier SLAKE sweep
    └── ...
```

## References

- **GOPO** (arXiv:2602.03876, Feb 2026) — Group Ordinal Policy Optimization. Rank-based GRPO advantages for single scalar rewards. We extend to compositional sparse-reward settings via the tiebreaker.
- **FaithRL** (arXiv:2602.05897, Feb 2026) — Step-level faithfulness reward via external PRM + truncated resampling. Different mechanism (additive composite), different domain (text QA).
- **Dr. GRPO** (arXiv:2503.02948) — Unbiased GRPO reference. Our baseline uses this variant of the advantage.
- **Med-R1** (arXiv:2503.13939) — GRPO on Qwen2-VL-2B for medical VQA. Reports +29.94% over SFT under fuzzy correctness; we compare under strict token F1.
- **DAPO** (2025) — Dynamic sampling fix for advantage collapse. Different fix for the same problem.
- **Lookback Lens** (EMNLP 2024) — Per-head attention-ratio-based hallucination detector; inspired our per-head bbox feature construction.
- **DHCP** (arXiv:2411.18659) — Cross-modal attention hallucination detection.
- **VLAA-Thinker** (arXiv:2504.11468) — SFT-then-RL dynamics in visual reasoning models.
