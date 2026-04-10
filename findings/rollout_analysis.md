# Rollout Analysis: Zero-shot vs Correctness-only vs Spatial GRPO

**Date:** April 4, 2026
**Checkpoints:** Step 30-40 best checkpoints from organ-only SLAKE (10/1 split, 8 rollouts)
**Eval:** 30 organ-specific SLAKE val examples, greedy decode, max_new_tokens=512

## Scores

| Model | Correct | Wrong | Accuracy |
|-------|---------|-------|----------|
| Zero-shot | 21 | 9 | **70%** |
| Correctness-only (α=1.0) | 20 | 10 | 67% |
| Spatial GRPO (α=0.7) | 19 | 11 | 63% |

## The Big Picture: The Three Models Are 90% Identical

Out of 30 questions, **23 have word-for-word identical reasoning across all three models.** The LoRA at step 30-40 has barely moved the weights. When the models do differ, it's in subtle wording changes, not fundamentally different reasoning.

## Per-Question Disagreements

| Q | Question | GT | Zero-shot | Corr-only | Spatial |
|---|----------|-----|-----------|-----------|---------|
| 1 | What diseases? | Liver Cancer | CORRECT | WRONG | WRONG |
| 5 | Digestive system organs? | Colon, Small Bowel | WRONG | **CORRECT** | WRONG |
| 7 | Left lung color? | Black | CORRECT | WRONG | WRONG |
| 13 | Disease on right of brain? | Brain Edema, Tumor | WRONG | CORRECT* | WRONG |
| 18 | What part of lung is mass in? | Right Lung | WRONG | WRONG | **CORRECT** |
| 20 | Right lung color? | Black | CORRECT | WRONG | **CORRECT** |
| 28 | Where is atelectasis? | Lower Right Lung | CORRECT | CORRECT | WRONG |

*Q13 correctness-only "correct" is likely a lucky containment match in a 512-token ramble.

## Pattern 1: Identical Outputs (23/30 questions)

Most questions produce the same thinking chain and answer across all three models. Examples:

**Q2 — "Does the picture contain liver?" (All CORRECT)**
All three say "The liver is a large organ on the left side" and answer "Yes." Only difference: spatial says "in the axial view" instead of "in the abdominal region."

**Q6 — "Where is the cardiomegaly?" (All CORRECT)**
Word-for-word identical across all three, down to "the heart is more prominent, especially the left side."

**Q25 — "What organ is the gray part of the image?" (All CORRECT)**
Identical reasoning, identical "Heart" answer.

## Pattern 2: Spatial Model Gives More Specific Visual Descriptions

When outputs do differ, the spatial model tends to describe what it sees more specifically:

**Q2 — "Does picture contain liver?"**
- Zero-shot/Correctness: "liver is a large organ on the left side of the image (**in the abdominal region**)"
- Spatial: "liver is a large organ on the left side of the image (**in the axial view**)" — refers to the actual imaging plane

**Q16 — "Where is the abnormality?" (All CORRECT)**
- Zero-shot/Correctness: "there's a mass or lesion in the liver... maybe a tumor or cyst"
- Spatial: "the liver shows a lesion... **a bright area in the liver**, which might be a cyst or a tumor" — actually describes the visual appearance (bright area)

**Q10 — "Does picture contain spleen?" (All CORRECT)**
- Zero-shot: 152 tokens, confident
- Spatial: 198 tokens, more detailed: **"the left side has a structure that might be the spleen"** — actively searching the image rather than asserting from memory

**Q22 — "How many hearts?" (All CORRECT)**
- Zero-shot: "let's look at the image"
- Spatial: **"Got it, let's look at the image"** — minor wording difference, slightly more confident

## Pattern 3: The 512-Token Reasoning Loops

8 questions (Q1, Q3, Q7, Q11, Q12, Q13, Q20, Q28) trigger "Wait, no, wait" loops where the model gets stuck. All three models do this equally — the spatial probe doesn't prevent or worsen looping.

**Q3 — "Which is bigger, kidney or liver?" (GT: Kidney, All WRONG)**
All three models start with "the kidneys are larger than the liver" (from the image), then doubt themselves with textbook knowledge: "Wait, the liver is usually larger" → infinite loop. **This is the core visual grounding problem in one example: the model can't reconcile what it sees with what it knows.**

**Q7 — "What color is the left lung?" (GT: Black)**
Zero-shot navigates through X-ray physics confusion and gets it right. Both trained models get confused about black = dense or black = air and loop until 512. **Training may have slightly destabilized the model's reasoning on novel visual questions.**

## Pattern 4: Key Disagreements Tell the Real Story

### Q5 — "Which organ is part of the digestive system?" (GT: Colon, Small Bowel)

The most revealing example:

- **Zero-shot**: "The stomach is a common digestive organ... So the answer should be stomach." → **Stomach** (WRONG — textbook default, didn't look)
- **Correctness-only**: "the large intestine (colon) might be visible... the colon (large intestine) is visible" → **Colon** (CORRECT — actually looked at the image and identified a specific structure)
- **Spatial**: "The liver is a major one... the liver is the large organ on the left side" → **Liver** (WRONG — fixated on the most visually prominent organ)

**Interpretation:** Correctness-only training helped the model look more carefully at what's actually in the image (colon). Spatial training made the model fixate on the biggest, most attention-grabbing organ (liver) — which makes sense because the spatial probe rewards attention to large organ regions. **The probe has a perverse incentive: it rewards looking at big organs, not the right organ.**

### Q18 — "What part of the lung is the mass in?" (GT: Right Lung)

- **Zero-shot**: "The mass is in the left lung" → Left lung (WRONG)
- **Correctness-only**: "The mass is in the left lung" → Left lung (WRONG)
- **Spatial**: "The mass is in the right lung. I need to check the image. The right lung has a noticeable mass, while the left lung seems normal." → **Right lung** (CORRECT)

**The spatial model correctly localized the mass when others didn't.** Its reasoning explicitly checks the image: "the right lung has a noticeable mass, while the left lung seems normal." This is the kind of spatial grounding the probe is designed to encourage.

### Q20 — "What color does the right lung show?" (GT: Black)

- **Zero-shot**: Long ramble but eventually says dark/blackish (512 tokens, CORRECT)
- **Correctness-only**: Gets stuck repeating "the right lung is on the left side of the image" 10+ times — pure repetition loop (512 tokens, WRONG)
- **Spatial**: Reasons through it more cleanly: "the lungs appear as dark areas (because they're filled with air, which absorbs less X-ray, so the X-ray passes through and the area appears dark)" → **"Dark (or black, as it is air-filled)"** (401 tokens, CORRECT)

The spatial model avoided the reasoning loop. Its explanation is more grounded in visual physics.

### Q28 — "Where is the atelectasis in the lung?" (GT: Lower Right Lung)

- **Zero-shot**: Concise, identifies left lung (143 tokens, CORRECT)
- **Correctness-only**: Even more concise (105 tokens, CORRECT)
- **Spatial**: Gets into a left/right confusion loop about PA view orientation — "the L is on the left side... wait, the right lung is on the left side of the image" — hits 512 tokens (WRONG)

**Ironic: the spatially-trained model overthinks spatial localization.** The probe may be making the model second-guess spatial relationships it would otherwise handle confidently.

## Summary

1. **Models are 90% identical at step 30-40** — LoRA hasn't moved weights enough to change behavior on most questions. 23/30 questions produce the same output.

2. **When spatial differs, it tends toward more specific visual descriptions** — "in the axial view," "a bright area in the liver," "the right lung has a noticeable mass." These are qualitatively more grounded even when they don't change the answer.

3. **The spatial probe has a perverse incentive (Q5)** — rewarding attention to large organ regions biases the model toward naming the biggest organ rather than the correct one. The probe rewards *where* you look, not *what you conclude* from looking.

4. **Spatial has genuine wins on visual-spatial questions (Q18, Q20)** — correctly localizing a lung mass and identifying lung color where others failed. These are questions where looking at the right region genuinely helps.

5. **The "Wait, no, wait" loop is the real enemy** — accounts for most errors across all three models. Neither training approach fixes this. It's a fundamental limitation of the reasoning architecture at 512 tokens.

6. **Net: Zero-shot 70%, Correctness-only 67%, Spatial 63%** — both trained models slightly worse on this 30-question sample. Too small to be conclusive, but the qualitative differences in reasoning style are real.

## Implications for Next Steps

- The spatial probe signal is real but weak at step 30-40. More training may amplify the differences.
- The perverse incentive (big organ bias) suggests the probe needs to be organ-specific, not just region-attention. Scoring attention to the *queried* organ specifically, not any organ.
- The reasoning loops are the bigger problem. Addressing the 512-token budget (or the model's tendency to doubt its visual observations) would help more than probe tuning.
- The qualitative grounding differences (more specific visual descriptions) are worth tracking even if they don't yet translate to accuracy gains.
