# Research Progress: Closed-Loop Hallucination Detection & Mitigation for Medical VQA

## Current Runs (April 4) — Organ-Only SLAKE, 10/1 Split, 8 Rollouts

**Setup:** Qwen3-VL-2B-Thinking, SLAKE filtered to organ-specific questions (5972 → 2076, 35%), 10/1 split (1888 train / 188 val), unified "brief" prompt, max_new_tokens=512, 8 rollouts, faith normalized via global z-score + sigmoid.

### Validation (greedy decode, spatial faith measured for both)

| Step | Correctness-only (α=1.0) | | Spatial GRPO (α=0.7) | |
|------|--------------------------|---|----------------------|---|
| | Correct | Faith | Correct | Faith |
| 10 | **0.692** | 0.276 | 0.676 | 0.276 |
| 20 | 0.670 | 0.277 | 0.654 | 0.277 |
| 30 | 0.681 | 0.276 | 0.686 | 0.276 |
| 40 | pending | — | **0.723** | 0.277 |

### Training (recent rolling average from progress bar)

| Step | Correctness-only (α=1.0) | | | Spatial GRPO (α=0.7) | | |
|------|---|---|---|---|---|---|
| | Reward | Correct | Faith | Reward | Correct | Faith |
| ~9 | 0.634 | 0.650 | 0.351 | 0.536 | 0.537 | 0.343 |
| ~18-19 | 0.304 | 0.263 | 0.336 | 0.414 | 0.400 | 0.355 |
| ~28 | 0.696 | 0.700 | 0.377 | 0.530 | 0.525 | 0.400 |
| ~37-39 | 0.650 | 0.662 | 0.259 | 0.679 | 0.688 | 0.370 |
| ~42 | — | — | — | 0.569 | 0.562 | 0.415 |

**Observations:**
- Spatial run has stronger upward trajectory on val correctness (0.654 → 0.686 → 0.723) despite slower start
- Val faith is flat at ~0.277 for both — probe may be acting as regularizer rather than changing grounding
- Training faith is higher and varies more for spatial run (0.34-0.42) vs correctness-only (0.26-0.38), confirming probe signal reaches model during training
- Need to inspect actual rollout reasoning chains to see if grounding behavior differs qualitatively

**Fixes applied this session:**
- Unified prompt akaggle competitions listcross rollout/log-prob/eval (was 3 different prompts)
- max_new_tokens 256 → 512 (was truncating 100% of outputs)
- Dropped union bbox fallback (76% of examples were getting noise signal)
- Organ-only filter (every example has real organ bbox)
- 10/1 train/val split
- Faith normalization: min-max on 4 rollouts → global z-score + sigmoid on 8 rollouts
- Spatial faith measured during validation for BOTH runs (was showing dummy 0.5)

---

## TL;DR

We're building a pipeline that detects hallucinations from a VLM's internal states, then feeds that signal back as a GRPO reward to train the model to hallucinate less. After two detector approaches failed as reward functions (FaithSCAN saturated, HALP cancelled out in GRPO), we identified a cross-modal attention probe (DHCP) that solves both problems. Next step: get correctness-only GRPO working on Qwen2-VL-2B, then layer the DHCP detector on top.

---

## Phase 1: FaithSCAN (Post-Generation Hidden-State Detector)

**What it is:** A 4-branch cross-attention detector that fuses vision, question, answer hidden states + token-level confidence stats (entropy, log-likelihood, logit margin) from layers [7, 14, 21] of Qwen2-VL. Trained with BCE loss on correct-vs-hallucinated labels derived from VQA-RAD.

**Standalone results:**
- AUROC: **0.927** on VQA-RAD val (7B), **0.927** on 2B
- F1: 0.556 (limited by class imbalance — only 12/359 val examples are hallucinated)

**What went wrong as a GRPO reward:**
- Baseline faithfulness score was already **0.88-0.93** for almost all examples — the detector thought everything was faithful
- Gap between correct (0.208) and wrong (0.170) answers was only **0.038** — too small for meaningful gradient signal
- When used as a frozen reward, the model learned adversarial hidden-state patterns that scored high on FaithSCAN without actually grounding in the image — faithfulness hit **0.999** while independent HALP score dropped from 0.370 to 0.189
- Result: **zero improvement on greedy eval** across all alpha values and seeds

Note: The dataset has since been expanded, (SLAKE + vqarad) and faithscan had a similar problem even when hallucinations and non-hallucinations were dispersed evenly. This paper uses correctness as part of the definition of a hallucination so I decided to do something else.

**Cited:** FaithSCAN architecture adapted from arXiv:2502.04171

---

## Phase 2: HALP (Pre-Generation Probe)

**What it is:** A 2-layer MLP trained on internal states *before* any answer tokens are decoded. Predicts hallucination from the question+image representation alone.

**Standalone results:**
- AUROC: **0.748** on VQA-RAD val
- Better calibrated than FaithSCAN (baseline score 0.370 vs FaithSCAN's 0.88+)

**What went wrong as a GRPO reward:**
- HALP produces **one score per question**, shared across all 8 rollouts in a GRPO batch
- GRPO computes advantages by subtracting the mean reward within each group
- Same HALP score for all 8 rollouts → **signal cancels out completely** → zero gradient from the faithfulness component
- This is a fundamental incompatibility: pre-generation probes cannot discriminate between rollouts

**Cited:** HALP from arXiv:2603.05465

---

## Phase 3: Combined Reward Experiments

**Setup:** Mixed HALP + FaithSCAN reward with KL penalty on Qwen2-VL-2B, full fine-tuning

| Config | Val Reward | Train Correct | Train Faith | Notes |
|--------|-----------|---------------|-------------|-------|
| FaithSCAN-only + KL | 0.347 | 0.338 | 0.163 | Collapsed without HALP |
| 70% HALP + 30% Faith (beta=0.7) | **0.384** | 0.338 | **0.331** | Best stability |
| 50% HALP + 50% Faith (beta=0.5) | 0.372 | 0.293 | 0.277 | Reasonable |
| 30% HALP + 70% Faith (beta=0.3) | 0.363 | 0.282 | 0.255 | OK |

**Key findings:**
1. HALP stabilizes training even though its gradient signal cancels — the KL penalty against the frozen reference model does the heavy lifting
2. Faithfulness improved from 0.186 → 0.331 (beta=0.7), nearly doubled
3. But **correctness dropped below baseline** (0.429 → 0.338) in all configs
4. All runs degraded after epoch 1 — early stopping was optimal

---

## Phase 4: Exhaustive GRPO Sweeps (the grind)

### 7B + LoRA (v1, v2)
- LoRA r=16 and r=64 on quantized Qwen2-VL-7B
- **Result:** Greedy eval 0.578 ± 0.01 across ALL configs — identical to baseline
- LoRA_B weights were ~5e-6 magnitude — the adapters weren't changing the model at all
- **Conclusion:** 7B is already near-ceiling on VQA-RAD, no headroom for GRPO

### 2B + LoRA (v3)
- LoRA r=64 on Qwen2-VL-2B (no quantization)
- **Result:** Greedy eval 0.490 vs baseline 0.492 — still no change
- Same problem: LoRA weights too small to override frozen base weights

### 2B + Full Fine-Tuning (v3_full)
- Full parameter updates on all 2B weights
- Multiple reward functions: token F1, binary+semantic, format reward
- 4 alpha values × 3 seeds × multiple LR/KL configurations
- **Result:** Best greedy eval 0.538 vs baseline 0.535 — within noise

### Combined Multi-Dataset
- VQA-RAD + PathVQA + SLAKE (6353 examples)
- **Result:** Val correct collapsed to 0.12 — catastrophic forgetting

**The pattern across ALL runs:** GRPO training rewards go up, but greedy eval never changes. The model learns to produce better *sampled* outputs but its single best answer stays the same.

---

## The Breakthrough Insight: Why Frozen Detectors Fail

Reading Zhang et al. (arXiv:2411.18659, "Detecting Hallucinations by Cross-modal Attention Pattern") and Jiang et al. (arXiv:2411.16724, "Devils in Middle Layers") revealed the core issue:

1. **FaithSCAN uses hidden states** — these are high-dimensional and easy to game. The model can adjust internal representations to look faithful without actually attending to the image.

2. **HALP is pre-generation** — fundamentally cannot discriminate between different answers to the same question.

3. **Cross-modal attention is the right signal** — it directly measures whether answer tokens attend to relevant image regions. This is:
   - Per-rollout (different attention patterns for each answer)
   - Hard to game (attention to image regions reflects actual visual grounding)
   - Interpretable (which heads/layers matter for faithfulness)

---

## Phase 5: DHCP Probe (Current Direction)

**Architecture:** Lightweight MLP (336 → 256 → 128 → 1) trained on cross-modal attention features — answer tokens attending to image tokens across all 28 layers × 12 heads of Qwen2-VL-2B.

**Why it solves our problems:**

| Problem | FaithSCAN | HALP | DHCP |
|---------|-----------|------|------|
| Per-rollout discrimination | Shared hidden states | One score per question | Different attention per answer |
| Reward hacking | Easy to game via hidden states | N/A (cancels out) | Must actually attend to image |
| Calibration | Saturated at 0.99 | OK (0.37 baseline) | TBD |

**Based on:**
- Zhang et al. "DHCP" (arXiv:2411.18659) — 93%+ detection using all heads/layers with MLP
- Jiang et al. "Devils in Middle Layers" (arXiv:2411.16724) — middle layers critical for visual processing
- Kang et al. "Your LVLM Only Needs A Few Attention Heads" (CVPR 2025) — attention heads encode grounding

---

## Next Steps

### Step 1: Get correctness-only GRPO working on 2B
- The base problem: GRPO doesn't change greedy eval. Fix this first before adding any detector.
- Approach: Format reward + binary accuracy (Med-R1 style), full fine-tuning, aggressive early stopping
- Target: Greedy correctness > 0.50 (baseline 0.429 with format prompt)
- **Cited:** Med-R1 (arXiv:2503.13939) showed +29.94% on 2B with this exact setup

### Step 2: Train DHCP probe
- Extract cross-modal attention features from Qwen2-VL-2B on VQA-RAD
- Train the 3-layer MLP probe on correct vs hallucinated labels
- Target: AUROC > 0.85

### Step 3: Add DHCP as composite reward
- Composite: alpha * accuracy + (1-alpha) * (1 - dhcp_halluc_prob)
- Hypothesis: DHCP signal improves faithfulness without degrading accuracy (unlike FaithSCAN which collapsed)
- Alpha sweep: 0.5, 0.7, 0.85, 1.0 × 3 seeds

### Step 4: Evaluate
- Greedy decode + majority vote (5x, 8x)
- Compare: baseline vs correctness-only GRPO vs composite GRPO
- Per-example response analysis (CSV with question, GT, model answer, scores)
- Statistical significance: paired bootstrap across seeds

---

## Key References

| Paper | Year | Relevance |
|-------|------|-----------|
| Dr. GRPO (arXiv:2503.02948) | 2025 | No length/std normalization — our RL algorithm |
| Med-R1 (arXiv:2503.13939) | 2025 | GRPO on Qwen2-VL-2B for medical VQA, +29.94% |
| MedVLM-R1 (arXiv:2502.19634) | 2025 | 600 samples, 2B beats 72B |
| VLAA-Thinker (arXiv:2504.11468) | 2025 | SFT degrades RL; mixed perception+cognition reward |
| DHCP (arXiv:2411.18659) | 2025 | Cross-modal attention probe, 93%+ accuracy |
| Devils in Middle Layers (arXiv:2411.16724) | 2025 | Middle layers critical for visual processing |
| HALP (arXiv:2603.05465) | 2025 | Pre-gen probing baseline |
| FaithSCAN (arXiv:2502.04171) | 2025 | Post-gen hidden-state detector |
| RARL (arXiv:2506.06600) | 2025 | GRPO + LoRA dual-reward, closest to our work |
| REFLEX-Med (OpenReview) | 2025 | Curriculum GRPO + visual fidelity rewards |
