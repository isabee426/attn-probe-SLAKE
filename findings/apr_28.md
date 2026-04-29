# Session Discoveries (Apr 27–28, 2026)

Post-resume sanity check on the slake_repro tmux state, plus first test-set comparison of the lenfloor variants. Ordered by paper impact.

## 1. Step-190 test eval: lenfloor variants underperform their no-lenfloor counterparts (single-step snapshot only)

First side-by-side of the two lenfloor runs on the SLAKE held-out test set (n=1061, greedy decode), at the matched step-190 checkpoint:

| Variant | seed | step | val_correct (organ-only, n=188) | **Test F1 (n=1061)** | Exact |
|---|---|---|---|---|---|
| tiebreak_lenfloor | 42 | 190 | 0.4541 | **0.4159** | 419/1061 |
| bboxfree_lenfloor | 42 | 190 | 0.4274 | **0.3661** | 363/1061 |
| Δ (TB − BF) | | | +0.027 | **+0.050** | |

For comparison, the **non-lenfloor** runs at step 190 on the same test split:

| Variant | seed | step | Test F1 |
|---|---|---|---|
| tiebreak (no lenfloor) | 42 | 190 | 0.5362 |
| bboxfree_tiebreak (no lenfloor) | 42 | 190 | 0.4582 |
| Δ | | | +0.078 |

**Important caveat:** step 190 is *not* a fair comparison point for the lenfloor variants. Both lenfloor runs peak much later (TB-lenfloor val peaks at 0.5793 at step 810 — see §3). Step-190 only tells us the lenfloor regularizer slows early learning. Final-checkpoint test eval is still pending (TB at step 810 once W7 finishes, BF when W5 finishes).

**What it does say cleanly:** the tiebreak-over-bboxfree lead is preserved with lenfloor on (Δ = +0.050 at matched step). The bboxfree gap doesn't go away when lenfloor is added.

## 2. The bbox-free generalization claim still doesn't have proof

Head-to-head val peaks across all four sweep cells at current best-so-far:

| | tiebreak (bbox-cond) | bboxfree | Δ (TB − BF) |
|---|---|---|---|
| **No lenfloor** | val 0.6190 (step 570) → test 0.5472 | val 0.5506 (step 620) → test 0.4582 (step-190 eval, ckpt overfit by 620) | **+0.068 val** |
| **With lenfloor** | val **0.5793 (step 810)** | val 0.5182 (step 370) | **+0.061 val** |

The +0.06–0.07 val gap holds whether lenfloor is on or off. At probe level, bbox-free actually slightly *outperforms* bbox-conditioned (AUROC 0.873 vs ~0.85 on the 1000-example labeled set, see [tiebreaker_grpo_results.md:366](tiebreaker_grpo_results.md#L366)) — but as a GRPO tiebreaker signal, bbox-conditioned consistently wins. The probe-level diagnostic and the downstream GRPO outcome diverge.

**Paper risk:** the bbox-free story is what carries cross-domain generalization in the §5 plan ([three_month_plan.md:177](three_month_plan.md#L177)) — most general-VL benchmarks won't have bboxes. If the gap is real and persists at final-checkpoint test eval, the bbox-free pitch is weaker than presented in the proposal. Need: bboxfree seed-456 replication, OOD evals (PathVQA + VQARAD) under bboxfree, and a final-checkpoint test number that closes (or doesn't close) the gap.

## 3. tiebreak_lenfloor hit a new val peak: 0.5793 at step 810

Updates the figure I'd been carrying (0.5676 at step 520). Run is now in epoch 5/5, finishing in ~7-8 hours. Faith score sustained at ~0.97.

This is *below* tiebreak (no lenfloor)'s val peak of 0.6190 at step 570. So **lenfloor lowers val ceiling for tiebreak by ~0.04** but the run is still climbing as it enters its final epoch.

**Action:** when W7 completes, run test-set eval at the step-810 best_correct checkpoint. That's the headline tiebreak_lenfloor test number.

## 4. bboxfree_tiebreak_s42_nodrop killed in epoch 4 — overfit confirmed

Best val_correct was 0.5506 at step 620 (end of epoch 3). Epoch 4 vals dropped through 0.50, 0.48, 0.44, recovering only to 0.45–0.46 by step 770. Faith stayed steady ~0.67 throughout, so the attention features didn't degrade — just policy overfit. Killed at step ~770, freed GPU 0.

**Implication:** for bboxfree without lenfloor, **3 epochs is the cap**. Final test eval should be at the step-620 best_correct checkpoint, not later.

## 5. Outstanding evals after current runs finish

When W5 (bboxfree_lenfloor) and W7 (tiebreak_lenfloor) complete:

1. Test eval on **W7 step-810 best_correct** (final TB-lenfloor headline).
2. Test eval on **W5 final best_correct** (final BF-lenfloor headline).
3. Test eval on **W2 step-620 best_correct** (final BF-no-lenfloor headline; the step-190 number we have is from a worse checkpoint).
4. PathVQA + VQARAD OOD on the four lenfloor / no-lenfloor cells, for an apples-to-apples cross-domain table.
5. Bboxfree seed-456 — currently *not started*. Without it the bboxfree story is single-seed.

## 6. Three-way head-to-head: corr_only baseline vs BF vs TB (n=300/300/200, prelim, all seed42_nodrop best_correct)

First-ever direct BF-vs-TB comparison with a matched corr_only baseline, all at the **best_correct checkpoint** of each `seed42_nodrop` run. Subset-eval (n=300 SLAKE/PathVQA, n=200 VQARAD), matched seed/config across every cell.

| Dataset | corr_only baseline | bboxfree (BF) | tiebreaker bbox (TB) | Δ TB-BF | Δ BF-corr | Δ TB-corr |
|---|---|---|---|---|---|---|
| SLAKE (n=300) | 0.2771 (76/300) | 0.4768 (142/300) | 0.4871 (142/300) | **+0.0103** | +0.1997 | +0.2100 |
| PathVQA (n=300) | 0.0735 (20/300) | 0.2315 (65/300) | 0.3592 (106/300) | **+0.1277** | +0.1580 | +0.2858 |
| VQARAD (n=200) | 0.1208 (20/200) | 0.3190 (62/200) | 0.4745 (93/200) | **+0.1555** | +0.1982 | +0.3537 |

(Cells show F1 (Exact/n). Result files: `results/quick_{slake,pathvqa,vqarad}_bf_vs_tb_s42_n*.json` and `results/quick_{slake,pathvqa,vqarad}_corr_only_s42_nodrop_n*.json`. JSON label mapping: `corr_only` field in BF-vs-TB JSONs = bboxfree best_correct; `corrprobe`/`fullprobe` field = tiebreaker best_correct; `corr_only` in the corr-only JSONs = correctness_only/seed42_nodrop best_correct.)

**Headline finding:** the tiebreaker advantage is **~order-of-magnitude larger OOD than in-domain**. BF and TB are statistically indistinguishable on SLAKE (Δ +0.01, well within sampling noise on n=300), but TB beats BF by **+0.13 on PathVQA and +0.16 on VQARAD**. Bbox-conditioned tiebreaker generalizes; bbox-free does not.

**Confirms:** the `seed42_nodrop` best_correct corr_only checkpoint is broken across the board (SLAKE 0.28, VQARAD 0.12 — barely above zero_shot's 0.30/0.12). Use the drop-on variant for representative corr_only baselines (SLAKE 0.43, VQARAD 0.34 from prior eval).

**Caveat:** subset sizes are small (n=200–300). Full-test confirmation needed for paper-grade numbers, but the OOD divergence is too large to be a sampling artifact.

## 7. New training: bboxfree_tiebreak_s42_drop launched

Started at 14:55 PDT on GPU 1. Same config as W2 (the killed bboxfree_tiebreak_s42_nodrop) except `drop_unformatted: true` and a new checkpoint dir. Hypothesis: the 38% truncation rate seen in BF nodrop responses is what's killing in-domain perf; drop_unformatted may close the gap to TB the way it lifted corr_only_drop from 0.30 → 0.43 historically. ETA ~30h to epoch 5.

Config: `configs/slake_bboxfree_tiebreak_s42_drop.yaml`
Checkpoint dir: `/data3/ishaplan/slake_reproduction/checkpoints_5ep/bboxfree_tiebreak/seed42_drop/`

## 8. Summary of what changed since [apr22](apr22_session_discoveries.md)

- Tiebreak (no-lenfloor) test result is unchanged (0.5472 at step 570).
- New cells: tiebreak_lenfloor and bboxfree_lenfloor have step-190 test numbers and ongoing training.
- W7 (tiebreak_lenfloor) is the only run still actively improving; everything else has peaked.
- The bbox-free gap is now confirmed across two reward-shaping settings (with and without lenfloor), strengthening the negative read on bbox-free as a GRPO signal.
