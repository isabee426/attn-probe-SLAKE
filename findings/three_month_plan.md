# Three-Month Plan (2026-04-23 → 2026-07-26)

**Deadline:** July 26, 2026
**Total:** ~13 weeks
**Scope:** Generalize the bbox-free probe + tiebreaker method from medical VQA (SLAKE) to general-domain visual-language tasks, produce a baselines matrix, draft the paper.

## Status vs. TA asks

| # | Ask | Status | Notes |
|---|---|---|---|
| 1a | Discard bboxes — general attention patterns predict correctness | ✅ **in progress** | Option D (Lookback-Lens per-head) passes 3-check; bbox-free tiebreak GRPO running (GPU 0). Pattern probe (Family 1+5) mid-extraction (GPU 7). |
| 1b | Probe input independent of bboxes (visual math datasets) | ✅ **in progress** | Pattern probe designed with this constraint — operates on raw attention tensor, no spatial anchor. Awaiting 3-check verdict. |
| 2a | Find general VL datasets | ❌ **blocked** | Needs @Hardy Chen. Pre-proposing MathVista / ChartQA / A-OKVQA. |
| 2b | Medical as extension, official metrics | 🟡 **partial** | On fuzzy token F1 per SLAKE convention (matches Med-R1). Need to verify VQA-RAD / PathVQA tokenization matches. |
| 3a | GRPO variants (vanilla, Dr. GRPO, DAPO) + ours | 🟡 **partial** | Have vanilla corr_only, Dr. GRPO-like mean-subtracted advantage, GOPO-equivalent (`correctness_rank`). DAPO not implemented. |
| 3b | Tiebreaker methods (GOPO + more) | 🟡 **partial** | GOPO = `correctness_rank` ablation. Literature sweep for others needed (BoN-RS, BOND, ProcessRM-tiebreaker, FaithRL tiebreaker). |
| 3c | Composite-reward ablations (ours, our composite, FaithRL) | 🟡 **partial** | Have composite α=0.7 (`full_s42`). FaithRL not implemented. |
| 4 | Reward hacking analysis | ⚪ **not started** | Low-priority per TA. |

## Phase 1 — Lock bbox-free on medical (Apr 23 → May 14, 3 weeks)

Goal: bbox-free tiebreak story on SLAKE is publication-quality and replicated on 2+ seeds.

### Week 1 (now, Apr 23 → Apr 30)

Mostly in-flight already. Minimal new launches; let running jobs land and consolidate results.

- [live] bbox-free tiebreak nodrop on GPU 0 finishes first few val cycles, establishes trajectory vs `tiebreak_s42_nodrop` (bbox-cond)
- [live] Length-floor ablation queued on GPU 7 (auto-launches after pattern probe wraps)
- [live] Pattern probe 3-check validator (~90 min from now)
- [live] 3 PathVQA OOD eval chains finish → refresh OOD table
- [live] Queued SLAKE eval of `tiebreak_s42_nodrop` step-150 snapshot
- [todo] Refresh README + `tiebreaker_grpo_results.md` as each result lands
- [todo] Snapshot `full_s42` and `corr_s42` 5ep peak checkpoints (mirrors what we did for `tiebreak_s42_nodrop` and `corrrank_*`)

### Week 2 (Apr 30 → May 7)

- [todo] Based on pattern-probe 3-check verdict: either launch dual-probe GRPO (Lookback + pattern as dual tiebreaker keys), or run length-floor seed sweep (s42 + s456)
- [todo] Launch `slake_bboxfree_tiebreak_s456_nodrop` (replication seed)
- [todo] Launch `correctness_only_s42_nodrop` + `correctness_rank_s42_nodrop` baselines (configs already written and synced)

### Week 3 (May 7 → May 14)

- [todo] Re-eval corrrank + tiebreak nodrop checkpoints on SLAKE test after training finishes (current table numbers are stale)
- [todo] VQA-RAD + PathVQA official metric verification: read each paper's eval code, confirm tokenization matches our `compute_correctness`
- [todo] Draft paper §4 (Experiments on medical VQA): method description, SLAKE + OOD tables, ablation grid
- [todo] Commit Phase-1 writeup as `findings/phase1_medical_complete.md`

## Phase 2 — Port to general domain (May 14 → Jun 10, 4 weeks)

### Week 4 (May 14 → May 21) — Dataset prep

- [blocked] **Confirm dataset with Hardy.** Likely candidates:
  - **MathVista** — visual math, very common reviewer-expected comparison
  - **ChartQA** — charts + natural-language questions, good for "reasoning over visual data"
  - **A-OKVQA** — open-ended world-knowledge VQA
  - **ScienceQA** — multimodal science questions with CoT annotations
- [todo] Port data loader → `faithscan/data/<dataset>.py`. Match the SLAKE loader interface: `question`, `image`, `answer`, optional `think_gt`
- [todo] Adapt answer extraction for the chosen format (math may need LaTeX normalization; charts need numeric extraction)
- [todo] Verify the correctness function — if the official metric differs substantially from token F1, add the official scorer

### Week 5 (May 21 → May 28) — Probe validation on general domain

- [todo] Run greedy-decode extraction on the general dataset → build a feature dataset analogous to `spatial_grounding_v1_full/spatial_features.npz`
- [todo] Train Option D (Lookback-Lens) probe on the general-domain data. Run 3-check.
- [todo] Train Pattern probe (Family 1+5) on the general-domain data. Run 3-check.
- [todo] **Critical test:** does either probe clear the checks on a non-medical domain? Expected:
  - Lookback may collapse on text-heavy tasks (math) where image attention is naturally low across all rollouts
  - Pattern probe's reasoning-dynamics features should survive better — they don't reference image tokens specifically
- [todo] Cross-domain transfer: probe trained on SLAKE evaluated on the general domain (and vice-versa). AUROC gap = the "domain-specific vs domain-general" answer

### Week 6 (May 28 → Jun 4) — First GRPO on general domain

- [todo] Launch tiebreaker GRPO on general domain with best probe from week 5
- [todo] Launch corr_only baseline on same dataset
- [todo] Launch corrrank baseline (GOPO ablation)
- [todo] Monitor val trajectory; if the reasoning-compression failure mode appears on general-domain, the length-floor ablation becomes a critical Phase-3 comparison

### Week 7 (Jun 4 → Jun 10) — Length-floor + consolidation

- [todo] Length-floor ablation on general domain
- [todo] Write paper §5 (Generalization): in-domain vs cross-domain probe AUROC, general-domain GRPO table
- [todo] Commit `findings/phase2_general_domain.md`

## Phase 3 — Baselines matrix (Jun 10 → Jul 15, 5 weeks)

Target matrix:

| GRPO variant | + nothing (baseline) | + tiebreaker (ours) | + length-floor (ours) |
|---|---|---|---|
| Vanilla GRPO | existing `corr_only` | run | run |
| Dr. GRPO | implement | run | run |
| DAPO | implement | run | run |

Plus composite-reward ablations (based on Dr. GRPO):

| Method | Status |
|---|---|
| Ours (Dr. GRPO + tiebreaker) | will have from top matrix |
| Our composite (Dr. GRPO + α=0.7 reward) | already have `full_s42` analog |
| FaithRL (step-level PRM + truncated resampling) | implement |

### Week 8 (Jun 10 → Jun 17) — DAPO implementation

- [todo] Implement dynamic sampling: when all G rollouts tie, oversample until at least K differ. Add config flag `dynamic_sampling: true` + `min_distinct_rewards: K`
- [todo] Verify against DAPO paper's reported behavior on a small sanity-check

### Week 9 (Jun 17 → Jun 24) — FaithRL or cut

- [todo] Evaluate FaithRL implementation cost. If >1 week, cut to "future work" and document.
- [todo] Otherwise: implement step-level PRM scoring + truncated resampling hook
- [todo] **Decision checkpoint:** commit to either full FaithRL or drop. Update plan accordingly.

### Week 10 (Jun 24 → Jul 1) — Baseline runs (no probe)

- [todo] Vanilla GRPO, Dr. GRPO, DAPO on general-domain — 1 seed each, ~3 days each
- [todo] Log full trajectories so we can add training-curve plots to the paper

### Week 11 (Jul 1 → Jul 8) — + ours variants

- [todo] {Dr. GRPO, DAPO} + tiebreaker + length-floor — the main paper comparison rows
- [todo] If a variant looks strongest, replicate on 2nd seed

### Week 12 (Jul 8 → Jul 15) — Composite-reward ablations

- [todo] Composite α=0.7 on Dr. GRPO backbone
- [todo] FaithRL (if implemented in week 9)
- [todo] Final replications on strongest combinations

## Phase 4 — Analysis + writing (Jul 15 → Jul 26, 1.5 weeks)

### Week 13 (Jul 15 → Jul 22) — Analysis

- [todo] Reward-hacking analysis on composite reward: measure faith trajectory vs correctness trajectory during training. If faith rises while correctness plateaus, we have hacking evidence.
- [todo] Behavioral analysis extensions: repeat the wait-loop / commitment analysis from `three_way_behavioral_analysis.md` on general-domain outputs
- [todo] Cross-domain feature audit: which heads matter on medical? on general? overlap?

### Week 14 (Jul 22 → Jul 26) — Writing finalization

- [todo] Paper draft: consolidated tables, figures, ablation rows
- [todo] Reviewer-question buffer: anticipate "what about X" and have data ready
- [todo] Final replications of anything critical

## Dependencies / blockers

Things I cannot start alone:

- **Hardy Chen's general-domain dataset list.** Pre-proposing MathVista + ChartQA. Ping today.
- **TA priority call on tiebreaker methods beyond GOPO.** Draft list: BoN-RS (rejection sampling), BOND (iterated BoN with distillation), ProcessRM-as-tiebreaker, FaithRL-tiebreaker. Ask which 2-3 matter.
- **Official-metric confirmation** for VQA-RAD / PathVQA / general-domain candidates. Can self-resolve by reading the respective paper repos.

## Scope honesty

What's realistic:

- ✅ **Bbox-free tiebreak story on medical, 2 seeds, paper-ready.** Definitely.
- ✅ **Probe + GRPO generalized to ONE general-domain dataset.** Definitely, if we commit by Apr 30.
- 🟡 **Full 3×3 GRPO-variant baseline matrix.** Tight. DAPO + FaithRL each take a week. Have ~5 weeks for all baseline runs (week 8-12). Expect to drop FaithRL to "future work" or use 1-seed only for some cells.
- 🟡 **Two general-domain datasets.** Tight. Medical may end up as "evaluate zero-shot OOD" rather than "retrain on medical."
- ❌ **Reward-hacking analysis as a major section.** Low-priority per TA; keep as a subsection not a pillar.

## First-week action items (this week)

1. [me] Ping Hardy for general-domain dataset list. Pre-propose MathVista + ChartQA.
2. [me, auto] Finish pattern-probe 3-check + commit findings on what works (~1.5h wait).
3. [me, background] Draft DAPO dynamic-sampling implementation as a config flag in `train_grpo.py` (~2 days).
4. [me] Verify fuzzy token F1 matches official SLAKE / VQA-RAD / PathVQA metrics by reading each repo's eval code (~1 day).
5. [me] Draft literature-review addendum on tiebreaker-adjacent methods. Ask TA to prioritize 2-3 for implementation.
6. [me, ~30 min] Snapshot milestone checkpoints for `full_s42`, `corr_s42` 5ep peaks for symmetry with `tiebreak_s42_nodrop` / `corrrank_*` snapshots.

## Commitments that set the paper's story arc

If Phase 1 (medical) lands cleanly, the paper has these movements:

1. **Method setup (§3):** rank-based advantage + attention-pattern probe as tiebreaker
2. **Medical results (§4):** SLAKE — bbox-cond tiebreaker (0.55 F1) → bbox-free tiebreaker (~same) → ablations (length-floor, dual-probe); OOD — PathVQA, VQA-RAD
3. **Generalization (§5):** identical method on one general-domain dataset, with the probe's cross-domain AUROC demonstrating the attention-pattern signal is not medical-specific
4. **Baselines (§6):** vanilla / Dr. GRPO / DAPO × our method — isolates which part of the GRPO advantage structure interacts best with the probe
5. **Analysis (§7):** wait-loop mechanism, commitment-vs-hallucination, reward-hacking evidence in composite-reward variants

If Phase 2 generalization doesn't land or the probe collapses on math, fallback: the paper is a **medical VQA methods paper** instead of a "general method" paper, and the pitch becomes "bbox-free tiebreaker for medical VQA specifically." Less ambitious but still submittable.
