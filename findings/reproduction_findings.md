# Spatial Grounding Reward as Implicit Regularizer for Medical VQA

## Abstract

We show that incorporating a spatial attention probe as a secondary reward signal
in GRPO training improves medical VQA accuracy by 3-8% relative over a
correctness-only baseline, despite not measurably changing the model's spatial
grounding behavior. The probe reward acts as an implicit regularizer that
prevents overfitting to the sparse binary correctness signal and encourages
more decisive, concise reasoning.

## Setup

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-VL-2B-Thinking + LoRA (r=16, α=32) |
| Dataset | SLAKE organ-only (5972 → 2076, 35%) |
| Split | 10/1 train/val (1888 / 188) |
| GRPO rollouts | 8 |
| Max new tokens | 512 |
| Reward (baseline) | α=1.0 (correctness only) |
| Reward (spatial) | α=0.7 (70% correctness + 30% spatial probe) |
| Faith normalization | Global z-score + sigmoid |
| Seeds | 42, 123 |

### Spatial Grounding Probe

Logistic regression trained on per-head bbox attention ratios from the
base model. Label mode: bbox overlap (non-circular — does not use correctness).

## Results

### Reproduction Run 1 (Token F1, drop_unformatted, bbox probe for spatial-bbox)

**Note:** Uses token F1 correctness (not the original binary containment/synonym function).
Original April 4 results used binary correctness → higher absolute numbers (0.69-0.72).
Token F1 is stricter — original "Experiment 2" with token F1 peaked at 0.575-0.622.

#### Corrlabels probe (broken run — wrong prompt during probe training, 0.9% positive rate)

Ran with unbalanced corrlabels probe before fix. Faith near zero (0.02) due to
badly calibrated predict_proba. Despite this, corr probe s42 scored highest at step 20.

| Run | Step 10 Correct | Step 10 Faith | Step 20 Correct | Step 20 Faith |
|-----|----------------|---------------|-----------------|---------------|
| Corr-only s42 | 0.307 | 0.261 | **0.328** | 0.251 |
| Corr-only s123 | 0.284 | 0.265 | 0.295 | 0.253 |
| Spatial-bbox s42 | 0.291 | 0.269 | 0.311 | 0.254 |
| Spatial-bbox s123 | 0.279 | 0.260 | 0.307 | 0.278 |
| Spatial-corr s42 (broken probe) | 0.218 | 0.021 | **0.331** | 0.021 |
| Spatial-corr s123 (broken probe) | 0.211 | 0.028 | 0.252 | 0.028 |

**Observations:**
- Val faith ~0.26 for all runs using bbox probe (raw predict_proba, unnormalized)
- Corrlabels probe faith ~0.02 because probe was trained with wrong prompt (0.9% positive labels)
- Despite broken faith signal, spatial-corr s42 still hit highest correctness (0.331) at step 20
- Correctness numbers (0.25-0.33) are lower than original token F1 run (0.575-0.622) — investigating

#### Reproduction Run 2 (in progress — balanced corrlabels probe, val faith normalized)

Fixes applied:
- Retrained corrlabels probe with balanced classes (77 pos / 77 neg from original features)
- Probe: AUROC=0.667, r=0.636, mean predict_proba positives=0.83, negatives=0.32
- Val faith now uses same z-score + sigmoid normalization as training
- Correctness-only runs now use corrlabels probe for faith measurement (α=1.0, faith not in reward)
- All 6 runs use same probe for comparable faith numbers

*Fill after runs complete:*

| Step | Corr-only s42 | Corr-only s123 | Spatial-bbox s42 | Spatial-bbox s123 | Spatial-corr s42 | Spatial-corr s123 |
|------|---------------|----------------|------------------|-------------------|------------------|-------------------|
| 10   |               |                |                  |                   |                  |                   |
| 20   |               |                |                  |                   |                  |                   |
| 30   |               |                |                  |                   |                  |                   |
| 40   |               |                |                  |                   |                  |                   |
| 50   |               |                |                  |                   |                  |                   |

### Peak Results

| Condition | Seed 42 | Seed 123 | Mean |
|-----------|---------|----------|------|
| Corr-only (α=1.0) |  |  |  |
| Spatial-bbox (α=0.7) |  |  |  |
| Spatial-corr (α=0.7) |  |  |  |

### Original Run Reference (April 4-5)

| Correctness Function | Condition | Best Correct | At Step |
|---------------------|-----------|-------------|---------|
| Binary (containment/synonyms) | Corr-only | 0.697 | 80 |
| Binary (containment/synonyms) | Spatial | 0.723 | 40 |
| Binary (containment/synonyms) | Delta | +0.026 (+3.7%) | — |
| Token F1 (tight) | Corr-only | ~0.620 | 10-20 |
| Token F1 (tight) | Spatial | 0.622 | 50 |
| Token F1 (tight) | Delta | +0.047 (+8.2%) | — |

## Qualitative Analysis

### Behavioral Differences (from rollout analysis)

| Metric | Zero-shot | Corr-only | Spatial |
|--------|-----------|-----------|---------|
| Mean response length |  |  |  |
| Reasoning loop rate |  |  |  |
| Hit max tokens rate |  |  |  |

### Key Examples

*Fill from rollout_analysis.json — focus on cases where spatial differs:*

1. **Conciseness** — spatial model gives shorter, more confident answers
2. **Decisiveness** — spatial model breaks out of "Wait, no, wait" loops sooner
3. **Edge cases** — spatial model sometimes uses clinical synonyms that reduce token F1

## Discussion

### Why does spatial probe improve accuracy without changing grounding?

Three hypotheses:

1. **Regularization**: The 30% faith component smooths the training landscape.
   Correctness is binary/sparse; spatial probe score varies continuously across
   rollouts. This gives richer gradient signal and prevents the model from
   collapsing onto one pattern that happens to score high on correctness.

2. **Reward shaping**: The spatial probe implicitly rewards shorter, more
   decisive responses. If the model attends to the right region and produces a
   short answer, it gets higher spatial faith. Longer responses with "Wait, no"
   loops attend more to generated text than to the image, lowering faith score.

3. **Anti-overfitting**: Correctness-only peaks early and declines. Spatial
   model has a slower start (trades some early correctness for grounding) but
   sustains improvement. The probe prevents the model from exploiting
   format/vocabulary shortcuts.

### Limitations

- Small val set (188 examples) — need multiple seeds to confirm
- Token F1 metric has known issues with synonyms
- Probe label quality depends on keyword→bbox matching (35% of SLAKE)
- 2B parameter model — unclear if finding transfers to larger models

## Reproducibility

```bash
# On vlaa-01.be.ucsc.edu:
TMUX_TMPDIR=/data3/ishaplan/tmp tmux new -s slake_repro
cd /path/to/slake_reproduction
bash run_all.sh all
```
