# Spatial Attention Probe as Auxiliary Reward for Medical VQA GRPO

Training a spatial grounding probe on organ bounding box attention patterns to differentiate correct and incorrect attention patterns, then using it as a signal during GRPO to improve medical VQA accuracy on SLAKE.

---

## 2026-04-21 Update: Rank-Based Tiebreaker Advantage

A new construction — using the probe as an **ordinal tiebreaker** in rank-based advantages rather than adding it to the reward magnitude — beats both correctness-only and composite-reward GRPO on the full SLAKE English test set (1061 questions, including non-organ questions the model was never trained on).

### Main result (full SLAKE English test set, 1061 Q)

| Model | Overall F1 | Exact | Closed Q F1 | Open Q F1 | Val peak | Val→Test gap |
|---|---|---|---|---|---|---|
| corr_only (α=1.0) | 0.3919 | 396/1061 | 0.5365 | 0.2986 | 0.4844 | −0.0925 |
| composite (α=0.7) | 0.4126 | 419/1061 | 0.5888 | 0.2991 | 0.5126 | −0.1000 |
| **tiebreaker (ours)** | **0.5340** | **562/1061** | **0.7372** | **0.4030** | 0.4928 | **+0.0412** |

- vs composite-reward: **+0.1214 absolute F1, +29% relative**
- vs correctness-only: **+0.1421 absolute F1, +36% relative**

**Composite-reward and correctness-only both overfit the organ-only training distribution** — val peaks ~0.48–0.51 on organ-only SLAKE, test drops to 0.39–0.41 on the broader English test. The tiebreaker does the opposite — val peak 0.49, test F1 **higher** at 0.53. It learned the underlying task rather than reward-shape artifacts.

### The method

Three reward/advantage constructions in the sweep:

| Method | Reward | Advantage |
|---|---|---|
| `corr_only` | `correctness` | `reward − mean(reward)` |
| `composite` (prior work) | `0.7·correctness + 0.3·faith` | `reward − mean(reward)` |
| **`tiebreaker`** | `correctness` (logged; not used in advantage) | `rank(correctness, faith) − mean(rank)`, lex sort |

The tiebreaker replaces the advantage computation only. Faith never enters the reward magnitude. It sorts formatted rollouts by `(correctness, faith)` descending — correctness always dominates the ordering, faith only disambiguates ties. Centered rank position becomes the advantage.

```python
sorted_idx = sorted(fmt_idx, key=lambda i: (rollouts[i]["correctness"], rollouts[i]["faithfulness"]), reverse=True)
N = len(sorted_idx)
mean_rank = (N - 1) / 2.0
advantages[i] = (N - 1 - pos) - mean_rank  # ranks centered at 0
```

### Why this works

GRPO's group-relative advantage becomes gradient-starved on coarse rewards like token F1, because most rollouts tie (all wrong, all 0.5, etc.) and produce zero advantage. The tiebreaker fixes two problems:

1. **Coarse signal.** Rank-based advantage gives fixed-magnitude gradient regardless of reward distribution (GOPO-style).
2. **Zero-advantage batches.** When all rollouts tie on correctness, the faith tiebreaker produces meaningful ordering — the wrong rollouts that looked at the right region get positive advantage over the wrong rollouts that hallucinated from nowhere.

The correctness ordering is never flipped by faith, so there is no reward-hacking pathway. Faith can be noisy or biased (AUROC ≈ 0.78 is "good enough") — it only needs to correlate with quality on tied examples.

### Details + matched-seed numbers

See [`findings/tiebreaker_grpo_results.md`](findings/tiebreaker_grpo_results.md) for the full breakdown including val trajectories, matched-seed comparisons, the `corrrank` ablation (rank advantage without tiebreaker), and related work (GOPO, FaithRL).

---

## Prior Experiment: 2-Epoch Reproduction (organ-only test, 364 Q)

The result below is from an earlier 2-epoch comparison on the 364-question organ-only test subset (the questions the model was directly trained on). This is not directly comparable to the 1061-question full test above — the 1061 set includes broader question types the model was never trained on.

| Model | Token F1 (364 organ-only) | Exact (>0.5) |
|-------|----------|-------------|
| Zero-shot | 0.275 | 92/364 (25.3%) |
| **Corr-only** | **0.447** | **156/364 (42.9%)** |
| Corrprobe (r=0.636) | 0.424 | 148/364 (40.7%) |
| **Fullprobe (r=0.844)** | **0.445** | **154/364 (42.3%)** |

Corr-only and fullprobe reach the same accuracy on the in-distribution organ-only test (0.447 vs 0.445). 322/364 (88%) agree. They get there through different training dynamics and commit on different question types.

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

### Behavioral specialization on the organ-only test

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

**Corr-only's 20 unique wins — existence, knowledge recall, and some visual:**

| Question | GT | Corr-only | Fullprobe |
|----------|-----|-----------|-----------|
| Which organ is abnormal, heart or lung? | Lung | `lung` | `[none]` or `heart` (wrong) |
| Where is the nodule? | Upper Right Lung | `right lung` | `left lung` (wrong) |
| Does the picture contain liver? | No | `No` | `Yes` (hallucinated presence) |
| What disease is shown on left of lung? | Pulmonary Mass | `mass` | Long explanation |
| Can pneumonia be observed on right lung? | Yes | `Yes` | Long explanation |
| Is there an esophagus in this image? | No | `No` | `[none]` |
| Which is smaller, bladder or rectum? | Rectum | `rectum` | `[none]` |
| Does picture contain spinal cord? | Yes | `Yes` | Long explanation |

Corr-only wins on a mix of existence checks, localization, and some visual questions that fullprobe either loops on or gets wrong. Both models have blind spots — they just differ on which questions.

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

1. **Corr-only spikes and crashes** throughout both epochs. Peak of 0.508 is a spike.
2. **Fullprobe climbs monotonically** in epoch 2. Highest epoch-end val.
3. **All three converge at epoch 2 end** (~0.488–0.491) on organ-only val. The composite reward doesn't improve final in-distribution accuracy — it stabilizes the path to get there. (The tiebreaker, at the top of this README, does improve accuracy, especially on the broader non-organ test.)

## Why the Probe Matters (Even Without Accuracy Improvement)

### 1. Training reliability
The probe produces predictable, monotonic improvement. Corr-only's spikes mean you don't know if a checkpoint is genuinely good or a lucky fluctuation. With the probe, every new best is a real improvement.

### 2. Reward saturation prevention
At high correctness, most of 8 rollouts are correct → advantages near zero → no gradient. The probe maintains within-group variance because different rollouts have different attention patterns even when all correct. The **tiebreaker construction** exploits this explicitly: when correctness ties, faith disambiguates.

### 3. Behavioral quality at equal accuracy
On organ-only test, same accuracy (0.447 vs 0.445) but the probe model commits on harder visual questions. On the broader 1061-Q test, **the tiebreaker model also wins substantially on overall accuracy** (+0.14 F1), not just behavior.

### 4. `drop_unformatted` is necessary
Without it, probe conditions collapse (corrprobe crashed to 0.233 in epoch 2). With it, both conditions reach ~0.49 on organ-only val. The probe's gradient signal can't compete with format learning noise — it needs `drop_unformatted` to isolate its contribution.

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-VL-2B-Thinking + LoRA (r=16, alpha=32) |
| Dataset | SLAKE organ-only (5972 → 2076, 35%) |
| Split | 10/1 train/val (1888 / 188) |
| Test (organ-only, prior) | 364 organ-only (unseen) |
| Test (full, new result) | 1061 English (full SLAKE test) |
| GRPO rollouts | 8 |
| Max new tokens | 512 |
| Reward (corr_only) | α=1.0 (correctness only, token F1) |
| Reward (composite) | α=0.7 (70% correctness + 30% probe faith) |
| Advantage (tiebreaker, new) | `rank(correctness, faith)` lex-sort, centered |
| Faith normalization | Global z-score + sigmoid |
| Optimizer | AdamW, lr=1e-5, cosine schedule, grad clip 1.0 |
| Epochs | 2 (prior experiment), 5 (new sweep) |
| drop_unformatted | true |

## Discussion

### Is the probe circular?

No. The probe maps **attention patterns** to correctness. When used as reward (or tiebreaker), it tells the model "attend like this" — not "be correct." The model can't game it by memorizing answers; it has to shift its attention distribution. The correlation (r=0.636–0.844) means it's a complementary signal from attention space, not a copy of the correctness signal.

### Why does the tiebreaker generalize better than composite reward?

The tiebreaker never puts faith into the reward magnitude. Faith only disambiguates correctness ties. This preserves the pure-correctness objective exactly — the policy cannot trade correctness for faith, so it cannot over-specialize to faith patterns that bind to the training distribution. Composite reward (α=0.7) allows such trades, which helps in-distribution training but hurts on broader test questions.

### What would make the tiebreaker win even more?

A harder task where reward saturation is a bigger problem — larger dataset, more diverse questions, or a setting where the primary reward ties frequently. The tiebreaker's advantage is largest when the primary reward is coarse and ties are common — exactly the regime GRPO on token-F1 rewards lives in.

## Reproduction

```bash
git clone https://github.com/isabee426/attn-probe-SLAKE.git
cd attn-probe-SLAKE
export PYTHONPATH=src:/path/to/faithscan_vqarad/src

# Train probe
python scripts/retrain_probe_balanced.py \
    --features /path/to/spatial_features.npz \
    --output checkpoints/spatial_probe/

# GRPO training (5 epochs)
CUDA_VISIBLE_DEVICES=0 python -m faithscan.train_grpo --config configs/reproduction/correctness_only_seed42.yaml
CUDA_VISIBLE_DEVICES=1 python -m faithscan.train_grpo --config configs/reproduction/spatial_grpo_a07_corrlabels_seed42.yaml
CUDA_VISIBLE_DEVICES=2 python -m faithscan.train_grpo --config configs/tiebreaker_slake_seed42.yaml

# Test set evaluation (full 1061 English test)
CUDA_VISIBLE_DEVICES=0 python scripts/eval_test_set.py \
    --corr-ckpt checkpoints/correctness_only/seed42/best_correct \
    --fullprobe-ckpt checkpoints/spatial_grpo_fullprobe/a07_seed42/best_correct \
    --output results/test_set.json
```

## File Structure

```
attn-probe-SLAKE/
├── README.md
├── src/faithscan/
│   ├── train_grpo.py                   # GRPO training loop (tiebreaker + rank advantage flags)
│   ├── reward.py                       # Correctness + composite reward
│   ├── data/dataset.py                 # Data loading
│   └── models/
│       ├── lookback_lens.py            # Spatial attention probe + faith
│       └── dhcp_probe.py               # Cross-modal probe (legacy)
├── configs/
│   ├── original/                       # Original April 4 configs
│   ├── reproduction/                   # 3 conditions x 2 seeds (2-epoch prior experiment)
│   └── tiebreaker_slake_seed*.yaml     # New 5-epoch tiebreaker sweep
├── scripts/
│   ├── train_spatial_probe.py          # Probe training
│   ├── eval_test_set.py                # Unseen test set evaluation
│   ├── retrain_probe_balanced.py       # Retrain with balanced classes
│   ├── compare_checkpoints.py          # Eval zero-shot vs trained
│   └── rollout_analysis.py             # Qualitative rollout comparison
└── findings/                           # All experiment logs and analysis
    ├── tiebreaker_grpo_results.md      # New — tiebreaker result details
    ├── slake_final_results.md
    ├── reproduction_findings.md
    └── ...
```

## References

- **GOPO** (arXiv:2602.03876, Feb 2026) — Group Ordinal Policy Optimization. Closest prior work on rank-based GRPO advantages. Single scalar reward; no compositional/tiebreaker mechanism.
- **FaithRL** (arXiv:2602.05897, Feb 2026) — Step-level faithfulness reward via external PRM + truncated resampling. Different domain (text QA), different mechanism (additive composite).
- Dr. GRPO (arXiv:2503.02948)
- Med-R1 (arXiv:2503.13939)
- VLAA-Thinker (arXiv:2504.11468)
- DHCP (arXiv:2411.18659)
- Lookback Lens (EMNLP 2024)
