# Spatial Attention Probe as Auxiliary Reward for Medical VQA GRPO

Training a spatial grounding probe on organ bounding box attention patterns, then using it as a secondary reward signal during GRPO to improve medical VQA accuracy on SLAKE.

## Key Finding

A spatial attention probe reward (30% weight) improves accuracy over correctness-only GRPO by **+3.7% relative** (0.723 vs 0.697) with the binary correctness function, and **+8.2% relative** (0.622 vs 0.575) with strict token F1 — despite not measurably changing the model's spatial grounding behavior (faith ~0.276 for both).

The probe acts as an **implicit regularizer**, not a grounding improver. It prevents the model from overfitting to the sparse correctness signal and encourages more decisive, concise reasoning.

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-VL-2B-Thinking + LoRA (r=16, alpha=32) |
| Dataset | SLAKE organ-only (5972 → 2076, 35%) |
| Split | 10/1 train/val (1888 / 188) |
| GRPO rollouts | 8 |
| Max new tokens | 512 |
| Prompt | "Answer this medical question. Think step by step, then give your final answer in \<answer\>...\</answer\> tags." |
| Reward (baseline) | alpha=1.0 (correctness only) |
| Reward (spatial) | alpha=0.7 (70% correctness + 30% spatial probe) |
| Faith normalization | Global z-score + sigmoid (Welford's algorithm) |
| Advantage | EBPO shrinkage baseline |
| Optimizer | AdamW, lr=1e-5, cosine schedule, grad clip 1.0 |

## Results

### Experiment 1: Binary Correctness (containment + synonyms)

| Step | Correctness-only (alpha=1.0) Correct | Correctness-only (alpha=1.0) Faith | Spatial GRPO (alpha=0.7) Correct | Spatial GRPO (alpha=0.7) Faith |
|------|---|---|---|---|
| 10 | 0.692 | 0.276 | 0.676 | 0.276 |
| 20 | 0.670 | 0.277 | 0.654 | 0.277 |
| 30 | 0.681 | 0.276 | 0.686 | 0.276 |
| 40 | 0.681 | 0.276 | **0.723** | 0.277 |
| 50 | **0.697** | 0.277 | — | — |
| 80 | **0.697** (peak) | 0.275 | — | — |
| 90 | 0.681 | 0.275 | 0.676 | 0.276 |
| 100 | — | — | 0.686 | 0.276 |

- Correctness-only peaked at step 80: **0.697**
- Spatial peaked at step 40: **0.723**
- **Spatial won by +0.026** (+3.7% relative)
- Faith identical for both (~0.276) — probe acts as regularizer

### Experiment 2: Strict Token F1

| Step | Correctness-only Correct | Correctness-only Faith | Spatial GRPO Correct | Spatial GRPO Faith |
|------|---|---|---|---|
| 10-20 | ~0.62 (peak) | 0.276 | — | — |
| 30 | 0.612 | 0.276 | — | — |
| 40 | 0.601 | 0.276 | — | — |
| 50 | 0.575 ↓ | 0.276 | **0.622** | 0.276 |

- Correctness-only peaked early (~step 10-20), then declined to 0.575
- Spatial still climbing at step 50: **0.622**
- **Spatial ahead by +0.047** (+8.2% relative)
- Correctness-only overfits faster; spatial has slower start but sustained improvement

### Training Dynamics

| Step | Correctness-only Reward | Correctness-only Correct | Correctness-only Faith | Spatial GRPO Reward | Spatial GRPO Correct | Spatial GRPO Faith |
|------|---|---|---|---|---|---|
| ~9 | 0.634 | 0.650 | 0.351 | 0.536 | 0.537 | 0.343 |
| ~18-19 | 0.304 | 0.263 | 0.336 | 0.414 | 0.400 | 0.355 |
| ~28 | 0.696 | 0.700 | 0.377 | 0.530 | 0.525 | 0.400 |
| ~37-39 | 0.650 | 0.662 | 0.259 | 0.679 | 0.688 | 0.370 |
| ~42 | 0.465 | 0.450 | 0.307 | 0.569 | 0.562 | 0.415 |

Training faith is higher and varies more for spatial (0.34-0.42) vs correctness-only (0.26-0.38), confirming probe signal reaches the model during training.

## Rollout Analysis (30 examples, greedy decode)

### Scores


All three models produced near-identical outputs on **23/30 questions**. Differences concentrate in 7 questions.

### Key Examples

**Q5 — "Which organ is part of the digestive system?" (GT: Colon, Small Bowel)**

- **Zero-shot**: Goes straight to "stomach" from memory — never considers what's actually visible
- **Correctness-only**: Identifies "colon" and gives a long explanation but answer is too verbose → wrong on token F1
- **Spatial**: Also identifies "colon" but gives a **concise answer** → correct

The spatial model learned to be more concise. Shorter answers score better on token F1.

**Q13 — "What disease on the right of brain?" (GT: Brain Edema, Brain Non-enhancing Tumor)**

- **Zero-shot**: 512-token loop, never concludes. "Maybe hemorrhagic stroke? Maybe tumor? Wait..."
- **Correctness-only**: Same 512-token loop, slightly different wording
- **Spatial**: Reasons for 473 tokens but **actually commits to an answer**: "the disease is a brain tumor" → correct

The spatial model **breaks out of reasoning loops sooner**. The probe reward may encourage the model to commit to what it sees rather than endlessly doubting itself.

**Q10 — "Does the picture contain spleen?" (GT: Yes)**

- **Zero-shot**: 152 tokens of careful reasoning
- **Correctness-only**: 172 tokens
- **Spatial**: **90 tokens** — much more confident and concise

Same correct answer, but spatial is nearly 2x shorter.

**Q18 — "What part of the lung is the mass in?" (GT: Right Lung)**

- **Zero-shot**: "The mass is in the left lung" → wrong
- **Correctness-only**: "The mass is in the left lung" → wrong
- **Spatial**: "The mass is in the right lung. I need to check the image. The right lung has a noticeable mass, while the left lung seems normal." → **correct**

Genuine spatial win — the model explicitly checks the image and correctly localizes.

### Emerging Pattern

The spatial probe reward makes the model:

1. **More concise** (Q5, Q10 — shorter answers)
2. **More decisive** (Q13 — commits instead of looping)
3. **More visually grounded** (Q18 — explicitly checks image regions)
4. **Slightly less careful with edge cases** (Q17 — uses clinical terms that don't token-match)

## Spatial Grounding Probe

Logistic regression on per-head bbox attention ratios (28 layers x 16 heads = 448 features).

**Training**: Extract spatial attention from base Qwen3-VL-2B-Thinking on SLAKE organ-only examples. For each example, compute per-head ratio of attention inside vs outside the target organ's bounding box. Train classifier to predict correctness from these ratios.

**Probe stats** (original, correctness labels):
- 300 examples, 77 correct (25.7%)
- Val AUROC: ~0.67
- Probe score ↔ correctness: r=0.42

**Probe stats** (balanced, correctness labels):
- 154 examples (77 pos / 77 neg), balanced
- Val AUROC: 0.667
- Probe score ↔ correctness: r=0.636
- Mean predict_proba: positives=0.83, negatives=0.32

**Probe stats** (bbox overlap labels, non-circular):
- 1000 examples, 365 positive (36.5%)
- Val AUROC: 1.000
- Probe score ↔ correctness: r=-0.02

The bbox overlap probe has perfect AUROC (trivially predicts attention-in-bbox from attention features) but zero correlation with correctness — a pure spatial signal with no correctness leakage.

## Reproduction

### Prerequisites

```bash
# Clone
git clone https://github.com/isabee426/attn-probe-SLAKE.git
cd attn-probe-SLAKE

# Requires: faithscan_vqarad package on PYTHONPATH
# SLAKE dataset at /data3/ishaplan/slake_full/Slake1.0
# Python environment with: torch, transformers, peft, qwen-vl-utils, scikit-learn
```

### Step 0: Train spatial probe

```bash
# Bbox overlap probe (non-circular)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_spatial_probe.py \
    --slake-dir /data3/ishaplan/slake_full/Slake1.0 \
    --output checkpoints/spatial_probe/ \
    --organ-only --labels bbox_overlap --max-examples 1000

# Correctness-labeled probe (original setup)
# Option A: Retrain from original features (instant, no GPU needed)
python scripts/retrain_probe_balanced.py \
    --features /data3/ishaplan/final_version/checkpoints/spatial_grounding/spatial_features.npz \
    --output checkpoints/spatial_probe_corrlabels/

# Option B: Extract fresh features + train
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train_spatial_probe.py \
    --slake-dir /data3/ishaplan/slake_full/Slake1.0 \
    --output checkpoints/spatial_probe_corrlabels/ \
    --organ-only --labels correctness --max-examples 1000
```

### Step 1: GRPO training (6 runs)

Three conditions x two seeds. Each run takes ~20h on an A5000 (25GB).

```bash
export PYTHONPATH=src:/path/to/faithscan_vqarad/src

# Correctness-only baseline (alpha=1.0)
CUDA_VISIBLE_DEVICES=0 python -m faithscan.train_grpo --config configs/reproduction/correctness_only_seed42.yaml
CUDA_VISIBLE_DEVICES=1 python -m faithscan.train_grpo --config configs/reproduction/correctness_only_seed123.yaml

# Spatial GRPO with bbox overlap probe (alpha=0.7)
CUDA_VISIBLE_DEVICES=2 python -m faithscan.train_grpo --config configs/reproduction/spatial_grpo_a07_seed42.yaml
CUDA_VISIBLE_DEVICES=3 python -m faithscan.train_grpo --config configs/reproduction/spatial_grpo_a07_seed123.yaml

# Spatial GRPO with correctness-labeled probe (alpha=0.7)
CUDA_VISIBLE_DEVICES=4 python -m faithscan.train_grpo --config configs/reproduction/spatial_grpo_a07_corrlabels_seed42.yaml
CUDA_VISIBLE_DEVICES=5 python -m faithscan.train_grpo --config configs/reproduction/spatial_grpo_a07_corrlabels_seed123.yaml
```

### Step 2: Evaluation

```bash
# Compare checkpoints (zero-shot vs corr-only vs spatial-bbox vs spatial-corr)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/compare_checkpoints.py \
    --corr-ckpt checkpoints/correctness_only/seed42/best \
    --spatial-ckpt checkpoints/spatial_grpo/a07_seed42/best \
    --spatial-corr-ckpt checkpoints/spatial_grpo_corrlabels/a07_seed42/best \
    --n 188 --seed 42 --output results/comparison_seed42.json

# Qualitative rollout analysis
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/rollout_analysis.py \
    --corr-ckpt checkpoints/correctness_only/seed42/best \
    --spatial-ckpt checkpoints/spatial_grpo/a07_seed42/best \
    --n 30 --seed 42 --output results/rollout_analysis.json
```

### Full pipeline (tmux launcher)

```bash
# On vlaa-01.be.ucsc.edu:
TMUX_TMPDIR=/data3/ishaplan/tmp tmux new -s slake_repro
bash run_all.sh all    # probe → 6 GRPO runs → eval
```

## Discussion

### Why does the probe improve accuracy without changing grounding?

1. **Regularization**: The 30% faith component smooths the training landscape. Correctness is sparse (0 or 1); spatial probe score varies continuously across rollouts — richer gradient signal.

2. **Anti-overfitting**: Correctness-only peaks early and declines (step 10-20 with token F1). Spatial model has a slower start but sustains improvement through step 50+. The probe prevents the model from exploiting format/vocabulary shortcuts.

3. **Reward shaping**: The probe implicitly rewards shorter, more decisive responses. Models that attend to the right organ and commit to a short answer get higher spatial faith. Longer "Wait, no, wait" loops attend more to generated text than to the image, lowering faith.

### Is the correctness-labeled probe circular?

No. The probe learns a mapping from **attention patterns** to correctness. When used as reward, it tells the model "attend like this" — not "be correct." The model can't game it by memorizing answers; it has to shift its attention distribution. The moderate correlation (r=0.42) confirms it's a complementary signal, not a copy — it explains only 18% of the variance in correctness.

The bbox overlap probe ablation strengthens this: if both probe types yield similar gains, the spatial attention signal itself matters, regardless of label source.

## File Structure

```
attn-probe-SLAKE/
├── README.md
├── run_all.sh                          # Full pipeline launcher
├── launch_grpo_after_probe.sh          # GRPO launcher (waits for probes)
├── launch_corrlabel_grpo.sh            # Corrlabels GRPO launcher
├── src/faithscan/
│   ├── train_grpo.py                   # GRPO training loop
│   ├── reward.py                       # Correctness + composite reward
│   ├── data/dataset.py                 # Data loading (wraps faithscan_vqarad)
│   └── models/
│       ├── lookback_lens.py            # Spatial attention probe + faith computation
│       └── dhcp_probe.py               # Cross-modal attention probe (legacy)
├── configs/
│   ├── original/                       # Original April 4 configs
│   │   ├── correctness_only_slake_10to1_seed42.yaml
│   │   └── spatial_grpo_slake_organonly_a07_seed42.yaml
│   └── reproduction/                   # Reproduction configs (3 conditions x 2 seeds)
│       ├── correctness_only_seed42.yaml
│       ├── correctness_only_seed123.yaml
│       ├── spatial_grpo_a07_seed42.yaml
│       ├── spatial_grpo_a07_seed123.yaml
│       ├── spatial_grpo_a07_corrlabels_seed42.yaml
│       └── spatial_grpo_a07_corrlabels_seed123.yaml
├── scripts/
│   ├── train_spatial_probe.py          # Probe training (bbox or correctness labels)
│   ├── train_spatial_grounding.py      # Original probe training script
│   ├── retrain_probe_balanced.py       # Retrain from saved features with balanced classes
│   ├── train_spatial_classifier_from_saved.py  # Hyperparameter sweep on saved features
│   ├── compare_checkpoints.py          # Evaluate zero-shot vs trained models
│   └── rollout_analysis.py             # Qualitative rollout comparison
└── findings/
    ├── slake_final_results.md          # Original April 4-5 results
    ├── rollout_analysis.md             # Qualitative analysis of 30 examples
    ├── research_progress.md            # Full project history
    └── reproduction_findings.md        # Reproduction run tracking
```

## Citation

If you use this work, please cite:

```
Spatial Attention Probe as Auxiliary Reward for Medical VQA GRPO
SLAKE organ-only experiments, April 2026
```
