# Experiment Timeline and Current State

**Last updated:** April 17, 2026

## Timeline

### Phase 1: Original SLAKE experiments (April 4-5, 2026)
- Ran first correctness-only vs spatial GRPO comparison on SLAKE organ-only
- 1 epoch, `drop_unformatted`, binary correctness function (containment/synonyms)
- Result: spatial +3.7% over corr-only (0.723 vs 0.697)
- Also ran with tight token F1: spatial +8.2% (0.622 vs 0.575)

### Phase 2: Reproduction attempt (April 9-12)
- Created `slake_reproduction/` folder, rsync'd to vlaa-01 server
- Trained new balanced corrprobe (r=0.636) and bbox probe (r=-0.02)
- Ran 6 x 1-epoch runs (3 conditions × 2 seeds) with `drop_unformatted`
- Key finding: corrprobe trajectory is monotonic, corr-only spike-crashes
- Seed 42 rollout: corrprobe greedy F1 0.558 vs corr-only 0.486 (+14.8%)
- Across 5 seeds: mean tied (0.457 vs 0.454) — seed 42 was lucky

### Phase 3: Full probe training (April 13)
- Trained fullprobe on all 2076 organ-only examples (vs 300 original)
- Result: AUROC 0.968, r=0.844, gap 0.91/0.15 (vs 0.83/0.32 for corrprobe)

### Phase 4: 2-epoch runs without drop_unformatted (April 13-14)
- Ran 3 conditions × 1 seed (42) × 2 epochs, no drop_unformatted
- **Fullprobe hit 0.347 at step 40** — highest step 40 ever
- But full eval showed training val was unreliable (0.347 training → 0.277 full eval)
- **Finding: drop_unformatted is necessary for reliable checkpoints**
- Corrprobe crashed in epoch 2 (0.289→0.256)
- Fullprobe survived because stronger signal

### Phase 5: 2-epoch runs with drop_unformatted (April 14-16)
- Ran same 3 conditions × 2 epochs WITH drop_unformatted
- All three converged to similar val correct (~0.488-0.491) at epoch 2 end
- Corr-only spike-crashes throughout (peaks at 0.508, drops to 0.470, spikes to 0.419)
- Fullprobe monotonically climbs in epoch 2 (0.444→0.448→0.449→0.481→0.485)
- Training val matches full eval with drop_unformatted

### Phase 6: Unseen test set eval (April 16-17)
- Evaluated epoch 2 end checkpoints on 364 unseen organ-only test examples
- **Result: corr-only 0.447, fullprobe 0.445, corrprobe 0.424**
- Gap of 0.002 between corr-only and fullprobe (tied)
- Behavioral split on disagreements: fullprobe commits on descriptive/observational, corr-only on diagnostic/decisive

### Phase 7: Cross-dataset generalization (April 17, in progress)
- Running zero-shot eval on VQA-RAD and PathVQA test sets
- Testing whether fullprobe's descriptive advantage transfers outside SLAKE

## Current Status

### Active runs (all finished training):
- All 2-epoch drop runs complete (corr_drop, corrprobe_drop, fullprobe_drop)
- VQA-RAD eval: running on GPU 1
- PathVQA eval: running on GPU 2

### Saved checkpoints (`/data3/ishaplan/slake_reproduction/saved_checkpoints/`):
- `corr_epoch2_end`, `corrprobe_epoch2_end`, `fullprobe_epoch2_end` — epoch 2 end
- `correctness_only_seed42_step190_0508` — corr-only's 0.508 spike checkpoint
- `correctness_only_seed42_best_correct` — step 170 snapshot
- `spatial_grpo_corrlabels_a07_seed42_best_correct`
- `spatial_grpo_fullprobe_a07_seed42_best_correct`

## Key Findings

### 1. Probe doesn't beat corr-only on accuracy (with drop_unformatted)
On SLAKE test set: 0.447 vs 0.445 (effectively tied). `drop_unformatted` gives corr-only enough format discipline that the probe's main advantage disappears over 2 epochs.

### 2. Training dynamics are fundamentally different
- Corr-only: spike-crash-spike-crash (0.410→0.388→0.419→0.429→0.470→0.508)
- Fullprobe: monotonic climb (0.444→0.448→0.449→0.481→0.485)
- Same destination, completely different paths

### 3. drop_unformatted is necessary
Without it:
- Corrprobe (r=0.636) collapses to 0.233 in epoch 2
- Fullprobe (r=0.844) survives but training val doesn't match full eval
- Corr-only oscillates but is more robust to no-drop

### 4. Probe strength matters
- r=-0.02 (bbox): doesn't help, performs same as corr-only
- r=0.636 (corrprobe): helps but fragile, needs drop_unformatted
- r=0.844 (fullprobe): helps most, more robust

### 5. Behavioral specialization on unseen test data
- Fullprobe → descriptive/observational (localization, visual properties, clinical assessment)
- Corr-only → diagnostic/decisive (which organ is abnormal, what disease, yes/no existence)
- 88% agreement on test set, disagreements split almost evenly (22 vs 20 unique wins each)

## What Would Make the Probe Win on Accuracy

1. **Harder task** where reward saturation is more severe
2. **Larger model** where attention patterns have more capacity
3. **Less compute** — corr-only needs 2 epochs to catch up, probe is faster to converge
4. **Different metric** — token F1 penalizes clinical precision ("computed tomography (CT)" vs "CT")

## Open Questions

1. Does the descriptive/observational advantage transfer to PathVQA/VQA-RAD? (eval running now)
2. Would the probe produce bigger gains at 7B or 14B model scale?
3. What happens at 3+ epochs — does fullprobe's monotonic climb continue, or plateau?
4. Does the probe help more on sampled decoding vs greedy? (first run hint: corrprobe sampled F1 was lower, but greedy was higher)
