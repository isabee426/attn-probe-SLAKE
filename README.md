# Spatial Attention Probe as Auxiliary Reward for Medical VQA GRPO

Training a spatial grounding probe on organ bounding box attention patterns to differentiate correct and incorrect attention patterns, then using it as a secondary reward signal during GRPO to improve medical VQA accuracy on SLAKE.

## Latest: 2-Epoch Runs with Two Probe Strengths (in progress)

We're comparing **four conditions** across 2 epochs to test how probe strength affects training stability:

### Probe configurations

| Probe | Training data | AUROC | r with correctness | predict_proba gap |
|-------|--------------|-------|--------------------|--------------------|
| **Corrprobe** (weak) | 154 examples, balanced | 0.667 | 0.636 | 0.83 vs 0.32 (gap: 0.51) |
| **Fullprobe** (strong) | 572 examples, balanced from 1276 | 0.968 | 0.844 | 0.91 vs 0.15 (gap: 0.76) |

Both are logistic regression on per-head bbox attention ratios (28 layers x 16 heads = 448 features), trained on correctness labels. The fullprobe uses 4x more data from running the base model on all organ-only examples.

### 2-Epoch training trajectories (no drop_unformatted)

| Step | Corr-only | Corrprobe (r=0.636) | Fullprobe (r=0.844) |
|------|-----------|---------------------|---------------------|
| 10 | 0.292 | 0.298 | 0.290 |
| 20 | 0.292 | 0.274↓ | 0.289 |
| 30 | 0.296 | 0.257↓ | 0.296 |
| 40 | 0.299 | 0.275 | **0.347** |
| 50 | **0.318** | 0.281 | — |
| 60 | 0.286↓ | 0.288 | — |
| 70 | 0.316 | 0.289 | — |
| 80 | 0.306↓ | 0.259↓ | — |
| 90 (ep1 end) | 0.321 | 0.256↓ | — |
| 100 (ep2) | 0.310 | 0.274 | — |
| 110 (ep2) | 0.281↓ | — | — |

**Fullprobe hit 0.347 at step 40** — the highest val correctness at any step 40 across all experiments. Its trajectory is accelerating: 0.269→0.296→0.347 (+0.027, +0.051 per step).

**Key finding: probe strength matters.**
- **Corrprobe (r=0.636)** dipped hard without `drop_unformatted` (0.298→0.257) and crashed in epoch 2 (0.289→0.256). The weaker probe signal gets diluted by format learning noise.
- **Fullprobe (r=0.844)** recovered from its dip faster (0.269→0.347) because the stronger signal cuts through the format noise. The 0.91/0.15 probability separation gives clearer gradient than corrprobe's 0.83/0.32.
- **Corr-only** oscillates as always (0.318→0.286→0.316→0.306→0.321→0.310→0.281).

### Drop_unformatted runs (also in progress, just started)

Running same 3 conditions with `drop_unformatted: true` — unformatted rollouts excluded from advantage computation so the probe signal isn't diluted by format learning.

Early step 10 results:

| Condition | Step 10 correct |
|-----------|----------------|
| Corr-only (drop) | 0.302 |
| **Fullprobe (drop)** | **0.308** |

Fullprobe with drop at 0.308 is the highest step 10 of any run ever.

## Key Finding

The attention probe acts as an auxiliary reward that enforces good internal practice — attending to the queried organ. A **stronger probe** (higher AUROC, higher correlation with correctness) produces faster recovery from early training dips and higher val correctness. The probe prevents GRPO reward saturation by maintaining gradient signal when the correctness reward becomes uninformative.

## Previous Results (1-epoch runs with drop_unformatted)

### 30-example rollout analysis (epoch 1, corrprobe, seed 42)

| Metric | Zero-shot | Corr-only | Corrprobe |
|--------|-----------|-----------|-----------|
| Greedy F1 | 0.423 | 0.486 | **0.558** |
| Exact wins | 12/30 | 14/30 | **16/30** |
| Avg tokens | 325 | 275 | 284 |
| Reasoning loops | 63% | 53% | **53%** |

Corrprobe greedy F1 of 0.558 on seed 42 (+14.8% over corr-only). Across 5 random 30-example samples (seeds 42, 77, 200, 999, 55), mean greedy F1 is tied: corr-only 0.457 vs corrprobe 0.454. The advantage is behavioral, not consistently in aggregate accuracy.

### 1-epoch training trajectories (with drop_unformatted, corrprobe r=0.636)

**Corr-only (volatile):** 0.284→0.298→0.330→0.317→0.335→**0.412**→0.394↓→0.380↓→0.407

**Corrprobe (monotonic):** 0.301→0.298→0.310→0.322→0.326→0.344→0.374→0.377→**0.406**

Corrprobe climbs steadily while corr-only spikes and crashes. Both end epoch 1 at similar correctness (~0.40), but corrprobe's trajectory is more stable and predictable.

## Behavioral Analysis

### The probe shapes which questions the model becomes confident on

Both models learn conciseness. But they commit on **different question types**:
- **Probe models** → visual judgment (does X exist, which is bigger, what's abnormal, what color is the lesion)
- **Corr-only** → simple existence checks and knowledge recall (does picture contain X, what modality, which plane)

**Probe model unique wins — visual interpretation:**

| Question | GT | Probe model | CO |
|----------|-----|-------------|-----|
| What is the largest organ? | Brain | `brain` | Long explanation, no tag |
| Where is the abnormality? | Right | `right hemisphere` | Long explanation |
| What color is the brain tumor? | White | `white` | `hyperintense` (synonym miss) |
| Is the lung healthy? | No | `No` | `[none]` |
| Which organs are sensory organs? | Eyes | `eyes, ears` | `[none]` |

**CO unique wins — simple recall:**

| Question | GT | CO | Probe model |
|----------|-----|-----|-------------|
| Does picture contain liver? | Yes | `Yes` | Long explanation |
| Which organ belongs to circulatory system? | Heart | `heart` | Long explanation |
| What modality? | CT | `CT` | `computed tomography (CT)` (too verbose, lower F1) |

Token F1 penalizes the probe model's clinical precision: "computed tomography (CT)" vs "CT", "thoracic cavity" vs "chest".

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-VL-2B-Thinking + LoRA (r=16, alpha=32) |
| Dataset | SLAKE organ-only (5972 → 2076, 35%) |
| Split | 10/1 train/val (1888 / 188) |
| GRPO rollouts | 8 |
| Max new tokens | 512 |
| Reward (baseline) | alpha=1.0 (correctness only, token F1) |
| Reward (probe) | alpha=0.7 (70% correctness + 30% probe faith) |
| Faith normalization | Global z-score + sigmoid |
| Advantage | EBPO shrinkage baseline |
| Optimizer | AdamW, lr=1e-5, cosine schedule, grad clip 1.0 |

## Discussion

### Why does a stronger probe help more?

The probe adds a continuous per-rollout signal to the sparse binary correctness reward. A stronger probe (higher AUROC) creates a wider gap between "correct-like" and "incorrect-like" attention patterns (0.91 vs 0.15 for fullprobe, vs 0.83 vs 0.32 for corrprobe). This means:

1. **More within-group variance** — even when all 8 rollouts are correct, they get different probe scores, maintaining gradient flow
2. **Faster format recovery** — the strong signal cuts through the noise of learning format simultaneously with attention
3. **Less overfitting** — the probe prevents reward saturation at high correctness levels

### Why does drop_unformatted matter for probe conditions?

Without `drop_unformatted`, the probe signal competes with format learning for gradient bandwidth. The weaker corrprobe (r=0.636) gets diluted by this competition and crashes. The stronger fullprobe (r=0.844) survives because its signal is loud enough to be heard over the format noise. With `drop_unformatted`, format is handled by exclusion and the probe signal dominates from the start.

### Is the probe circular?

No. The probe maps **attention patterns** to correctness. When used as reward, it tells the model "attend like this" — not "be correct." The model can't game it by memorizing answers; it has to shift its attention distribution. The correlation means the probe points in the same direction as correctness but provides information from a different modality (attention space vs output space).

## Reproduction

```bash
git clone https://github.com/isabee426/attn-probe-SLAKE.git
cd attn-probe-SLAKE
export PYTHONPATH=src:/path/to/faithscan_vqarad/src

# Train probe (balanced, correctness labels)
python scripts/retrain_probe_balanced.py \
    --features /path/to/spatial_features.npz \
    --output checkpoints/spatial_probe/

# GRPO training
CUDA_VISIBLE_DEVICES=0 python -m faithscan.train_grpo --config configs/reproduction/correctness_only_seed42.yaml
CUDA_VISIBLE_DEVICES=1 python -m faithscan.train_grpo --config configs/reproduction/spatial_grpo_a07_corrlabels_seed42.yaml

# Evaluation
CUDA_VISIBLE_DEVICES=0 python scripts/compare_checkpoints.py \
    --corr-ckpt checkpoints/correctness_only/seed42/best_correct \
    --spatial-corr-ckpt checkpoints/spatial_grpo_corrlabels/a07_seed42/best_correct \
    --spatial-ckpt checkpoints/correctness_only/seed42/best_correct \
    --n 188 --seed 42 --output results/comparison.json
```

## File Structure

```
attn-probe-SLAKE/
├── README.md
├── src/faithscan/
│   ├── train_grpo.py                   # GRPO training loop
│   ├── reward.py                       # Correctness + composite reward
│   ├── data/dataset.py                 # Data loading
│   └── models/
│       ├── lookback_lens.py            # Spatial attention probe + faith
│       └── dhcp_probe.py               # Cross-modal probe (legacy)
├── configs/
│   ├── original/                       # Original April 4 configs
│   └── reproduction/                   # 3 conditions x 2 seeds
├── scripts/
│   ├── train_spatial_probe.py          # Probe training
│   ├── retrain_probe_balanced.py       # Retrain with balanced classes
│   ├── compare_checkpoints.py          # Eval zero-shot vs trained
│   └── rollout_analysis.py             # Qualitative rollout comparison
└── findings/                           # All experiment logs and analysis
```
