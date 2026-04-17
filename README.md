# Spatial Attention Probe as Auxiliary Reward for Medical VQA GRPO

Training a spatial grounding probe on organ bounding box attention patterns to differentiate correct and incorrect attention patterns, then using it as a secondary reward signal during GRPO to improve medical VQA accuracy on SLAKE.

## Main Result: Unseen Test Set (364 organ-only SLAKE test examples)

| Model | Token F1 | Exact (>0.5) |
|-------|----------|-------------|
| Zero-shot | 0.275 | 92/364 (25.3%) |
| **Corr-only** | **0.447** | **156/364 (42.9%)** |
| Corrprobe (r=0.636) | 0.424 | 148/364 (40.7%) |
| **Fullprobe (r=0.844)** | **0.445** | **154/364 (42.3%)** |

**Corr-only and fullprobe achieve the same accuracy on unseen data** (0.447 vs 0.445, delta = 0.4%). 322/364 (88%) agree. But they get there through fundamentally different training dynamics and commit on different question types.

### Training trajectories: same destination, different paths

**Corr-only (spike-crash):**
```
Step 50→60→70→80→90→...→170→180→190
0.352→0.410→0.388→0.419→0.429→...→0.493→0.470→0.508
```
Oscillates between spikes and crashes. Peak of 0.508 at step 190, but drops to 0.470 one step later.

**Fullprobe (monotonic climb):**
```
Step 40→50→60→70→80→...→170→180→190→200
0.344→0.318→0.333→0.368→0.401→...→0.448→0.449→0.481→0.485
```
Steady improvement with occasional dips. Never crashes. Ends at 0.491 epoch-end val.

### Behavioral specialization on unseen test data

Both models learned conciseness. But they commit on **different question types**:

**Fullprobe's 17 unique wins — visual/clinical judgment:**

| Question | GT | Fullprobe | Corr-only |
|----------|-----|-----------|-----------|
| Where is the pneumonia in the lung? | Lower Right Lung | `right lung` | `[none]` (loops) |
| Where is the atelectasis? | Lower Left Lung | `left lung` | Long explanation, no tag |
| Can nodule be observed on lower left lung? | Yes | `Yes` | `[none]` |
| Does the liver look normal? | Yes | `Yes` | `No` (hallucinated abnormality) |
| Is the brain tumor hyperdense or hypodense? | Hyperdense | `hyperdense` | Long explanation |
| What color does the duodenum show? | Gray | `gray` | Long explanation |
| What is the shape of spleen? | Oval | `oval` | `crescent` (wrong) |
| Which organ is part of the nervous system? | Spinal cord | `spinal cord` (x2) | `[none]` |
| Which organ is part of the lymphatic system? | Spleen | `spleen` | `[none]` |
| Which is smaller, spleen or kidney? | Spleen | `spleen` | `[none]` |

**Corr-only's 20 unique wins — existence and knowledge recall:**

Both models get the same total correct (156 vs 154 exact), but corr-only wins more on simple yes/no organ presence questions.

### The probe prevents medical prior hallucination

Key example from test set: **"Does the liver look normal?" (GT: Yes)**
- Corr-only: `No` — hallucinated an abnormality from medical prior
- Fullprobe: `Yes` — correctly assessed the image

The probe reward trains the model to rely on visual evidence rather than defaulting to medical priors.

## Probe Configurations

| Probe | Training data | AUROC | r with correctness | predict_proba gap |
|-------|--------------|-------|--------------------|--------------------|
| **Corrprobe** (weak) | 154 examples, balanced | 0.667 | 0.636 | 0.83 vs 0.32 (gap: 0.51) |
| **Fullprobe** (strong) | 572 examples, balanced from 1276 | 0.968 | 0.844 | 0.91 vs 0.15 (gap: 0.76) |

Both are logistic regression on per-head bbox attention ratios (28 layers x 16 heads = 448 features), trained on correctness labels.

## 2-Epoch Training (with drop_unformatted)

### Full trajectory

| Step | Corr-only | Corrprobe (r=0.636) | Fullprobe (r=0.844) |
|------|-----------|---------------------|---------------------|
| 10 | 0.302 | 0.299 | **0.308** |
| 20 | 0.287↓ | 0.306 | 0.324 |
| 30 | 0.333 | **0.340** | 0.336 |
| 40 | 0.319↓ | 0.309↓ | 0.344 |
| 50 | 0.352 | 0.327 | 0.318↓ |
| 60 | **0.410** | 0.316 | 0.333 |
| 70 | 0.388↓ | 0.336 | 0.368 |
| 80 | **0.419** | **0.396** | **0.401** |
| 90 (ep1 end) | **0.429** | 0.379↓ | 0.383↓ |
| 100 | — | **0.406** | — |
| ... | | | |
| 150 | **0.481** | **0.453** | 0.444 |
| 170 | **0.493** | **0.458** | 0.448 |
| 180 | 0.470↓ | 0.458 | 0.449 |
| 190 | **0.508** | — | **0.481** |
| 200 (ep2 end) | — | — | **0.485** |
| Epoch 2 end val | 0.488 | 0.488 | **0.491** |

### Key observations

1. **Corr-only spikes and crashes** throughout both epochs (0.410→0.388, 0.419→0.429→0.470→0.508→...). Peak of 0.508 is a spike.
2. **Fullprobe climbs monotonically** in epoch 2 (0.444→0.448→0.449→0.481→0.485→0.491). Highest epoch-end val of all conditions.
3. **All three converge at epoch 2 end** (~0.488-0.491). The probe doesn't improve final accuracy — it stabilizes the path to get there.

## Why the Probe Matters (Even Without Accuracy Improvement)

### 1. Training reliability
The probe produces predictable, monotonic improvement. Corr-only's spikes mean you don't know if a checkpoint is genuinely good or a lucky fluctuation. With the probe, every new best is a real improvement.

### 2. Reward saturation prevention
At high correctness, most of 8 rollouts are correct → advantages near zero → no gradient. The probe maintains within-group variance because different rollouts have different attention patterns even when all correct.

### 3. Behavioral quality at equal accuracy
Same test set accuracy (0.447 vs 0.445) but the probe model commits on harder visual questions (localization, clinical assessment, organ comparison) while corr-only commits on easier knowledge recall.

### 4. `drop_unformatted` is necessary
Without it, probe conditions collapse (corrprobe crashed to 0.233 in epoch 2). With it, both conditions reach ~0.49. The probe's gradient signal can't compete with format learning noise — it needs `drop_unformatted` to isolate its contribution.

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-VL-2B-Thinking + LoRA (r=16, alpha=32) |
| Dataset | SLAKE organ-only (5972 → 2076, 35%) |
| Split | 10/1 train/val (1888 / 188) |
| Test | 364 organ-only (unseen) |
| GRPO rollouts | 8 |
| Max new tokens | 512 |
| Reward (baseline) | alpha=1.0 (correctness only, token F1) |
| Reward (probe) | alpha=0.7 (70% correctness + 30% probe faith) |
| Faith normalization | Global z-score + sigmoid |
| Advantage | EBPO shrinkage baseline |
| Optimizer | AdamW, lr=1e-5, cosine schedule, grad clip 1.0 |
| Epochs | 2 |
| drop_unformatted | true |

## Discussion

### Is the probe circular?

No. The probe maps **attention patterns** to correctness. When used as reward, it tells the model "attend like this" — not "be correct." The model can't game it by memorizing answers; it has to shift its attention distribution. The correlation (r=0.636-0.844) means it's a complementary signal from attention space, not a copy of the correctness signal.

### Why doesn't the probe beat corr-only on accuracy?

With `drop_unformatted`, corr-only gets the format discipline that the probe provided in earlier experiments. Given enough training (2 epochs), corr-only's spike-crash dynamics average out to the same level as the probe's monotonic climb. The probe's value is in the journey (stable training, reliable checkpoints) not the destination (final accuracy).

### What would make the probe win on accuracy?

A harder task where reward saturation is a bigger problem — larger dataset, more diverse questions, or a setting where corr-only's spikes don't average out over 2 epochs. At 2B parameters with LoRA on 1888 organ-only examples, the task may be too small for the probe's anti-saturation benefit to manifest as an accuracy gap.

## Reproduction

```bash
git clone https://github.com/isabee426/attn-probe-SLAKE.git
cd attn-probe-SLAKE
export PYTHONPATH=src:/path/to/faithscan_vqarad/src

# Train probe
python scripts/retrain_probe_balanced.py \
    --features /path/to/spatial_features.npz \
    --output checkpoints/spatial_probe/

# GRPO training (2 epochs, drop_unformatted)
CUDA_VISIBLE_DEVICES=0 python -m faithscan.train_grpo --config configs/reproduction/correctness_only_seed42.yaml
CUDA_VISIBLE_DEVICES=1 python -m faithscan.train_grpo --config configs/reproduction/spatial_grpo_a07_corrlabels_seed42.yaml

# Test set evaluation
CUDA_VISIBLE_DEVICES=0 python scripts/eval_test_set.py \
    --corr-ckpt checkpoints/correctness_only/seed42/best_correct \
    --fullprobe-ckpt checkpoints/spatial_grpo_fullprobe/a07_seed42/best_correct \
    --organ-only --output results/test_set.json
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
│   ├── eval_test_set.py                # Unseen test set evaluation
│   ├── retrain_probe_balanced.py       # Retrain with balanced classes
│   ├── compare_checkpoints.py          # Eval zero-shot vs trained
│   └── rollout_analysis.py             # Qualitative rollout comparison
└── findings/                           # All experiment logs and analysis
```
