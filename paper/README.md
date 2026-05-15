# Paper skeleton — GroundLM @ EMNLP 2026

Workshop short paper draft for [GroundLM @ EMNLP 2026](https://openreview.net/group?id=EMNLP/2026/Workshop/GroundLM) (Budapest, Oct 24–29, 2026; submission deadline July 26, 2026).

## Files

- `main.tex` — 4-page short paper draft with TODO placeholders for numbers
- `references.bib` — bibliography; entries verified May 2026
- `README.md` — this file

## Build

```bash
# Once acl.sty is downloaded from https://github.com/acl-org/acl-style-files
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

For now, `main.tex` uses a fallback `geometry` setup so it builds with stock LaTeX. Switch to `\usepackage[review]{acl}` once you confirm the GroundLM template.

## TODO checklist before submission

1. **Numbers** — replace every `\todo{...}` once `eval_paper_table.sh` finishes. Source of truth: `/data3/ishaplan/cse40_slake_repro/slake_reproduction/results/paper_table/*.json`.
2. **Author block** — anonymize for review (current entry is a placeholder).
3. **Probe details** — confirm L×H = 28×16 = 448 dim feature for Qwen3-VL-2B.
4. **Probe training corpus** — describe exactly which rollouts were used to fit the LR classifier and what the held-out set looks like.
5. **Optional Figure 1** — bar chart of probe AUROC (bbox-free vs bbox-cond vs permuted). Can drop for space if needed.
6. **Anonymous-author bib entries** — six entries currently say `author = {Anonymous}`. Replace with real authors once arxiv pages are confirmed.
7. **EBPO citation** — find the original baseline-shrinkage paper that `train_grpo.py` cites; the current entry is a placeholder.
8. **Verify GroundLM-specific style** — workshop may require its own template; check OpenReview page closer to deadline.

## Section page budget (target: 4 pages content)

| Section | Target | Notes |
|---|---|---|
| Abstract | 0.1 | Hit "grounding / faithfully / efficiently" |
| Introduction | 0.7 | 3 contributions, lead with the tiebreaker-vs-additive insight |
| Method | 1.0 | Probe eq + tiebreaker algorithm + drop_unformatted |
| Experiments | 1.5 | Setup + main table + ablation + probe AUROC |
| Related Work | 0.5 | 4 paragraphs (Lookback / GOPO / Perceval / DAPO/NGRPO) |
| Conclusion | 0.15 | Single paragraph |
| Limitations | 0.05 | Bullet list, unnumbered |

References: unlimited (don't count toward 4 pages).

## What to cut if over length

- Drop Figure 1 (probe AUROC table-only)
- Cut composite_a07 row from main table (it's a weak baseline anyway)
- Compress Related Work to 3 paragraphs
- Move drop_unformatted ablation to a footnote
