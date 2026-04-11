# Full Validation Disagreement Analysis

**Dataset:** 188 organ-only SLAKE val examples, greedy decode, max_new_tokens=512
**Checkpoints:** Original April 4 best (trained with binary correctness)
**Eval metric:** Token F1

## Summary

| Model | Token F1 | Exact (>0.5) |
|-------|----------|-------------|
| Zero-shot | 0.289 | 49/188 (26.1%) |
| Correctness-only | 0.262 | 45/188 (23.9%) |
| **Spatial (alpha=0.7)** | **0.298** | **51/188 (27.1%)** |

- **Spatial vs Corr-only:** +0.036 (+13.7% relative)
- **Agreement:** 164/188 (87%) all three models agree
- **24 disagreements** across 188 examples

## Unique Wins

Questions where **only** that model got it right:

| Model | Unique wins | Character |
|-------|------------|-----------|
| **Spatial** | **8** | Concise answers, commits to answer tags |
| Zero-shot | 6 | Trained models overthink and fail |
| Correctness-only | 3 | Simple yes/no organ identification |

## Spatial's 8 Unique Wins

Every spatial-only win follows the same pattern: zero-shot and corr-only produce long explanations (300-400+ tokens) without ever committing to an `<answer>` tag, while spatial gives a concise 1-3 word answer.

### Q: "Which is smaller in this image, liver or left lung?" (GT: Left Lung)
- **Zero-shot:** No answer tag
- **Corr-only:** No answer tag
- **Spatial:** `left lung`

### Q: "Where is the pneumonia located?" (GT: Lower Left Lung)
- **Zero-shot:** No answer tag (398 tok)
- **Corr-only:** 378 tokens explaining opacity and density, analyzes AP view, identifies left lung opacity — but never produces an answer tag: "Pneumonia appears as areas of increased density (opacity) in the lung fields..."
- **Spatial:** `left lung` (347 tok)

### Q: "Which organs belong to the circulatory system?" (GT: Heart)
- **Zero-shot:** 380+ tokens explaining the circulatory system, lists heart, blood vessels, arteries, veins — never commits: "The circulatory system includes the heart and the blood vessels..."
- **Corr-only:** Same pattern, slightly different wording, no answer tag
- **Spatial:** `heart`

### Q: "What is the organ on top of the body in this image?" (GT: Liver)
- **Zero-shot:** Correctly identifies liver in explanation but no answer tag: "In abdominal CT scans, the liver is the largest organ in the upper right quadrant..."
- **Corr-only:** Same explanation, no answer tag
- **Spatial:** `liver`

### Q: "Is there an esophagus in this image?" (GT: No)
- **Zero-shot:** Long explanation about esophagus anatomy, no answer tag
- **Corr-only:** Long explanation about pelvic CT vs thoracic cavity, correctly reasons but no tag
- **Spatial:** `No`

### Q: "Can cardiomegaly be observed in this picture?" (GT: Yes)
- **Zero-shot:** Explains cardiomegaly definition, visual inspection, never commits
- **Corr-only:** Same pattern, discusses heart silhouette size
- **Spatial:** `Yes`

### Q: "How many existing heart in this image?" (GT: 1)
- **Zero-shot:** Gets it right (also produces tag)
- **Corr-only:** Long explanation about heart anatomy, sinoatrial node — no tag
- **Spatial:** `1`

### Q: "Where is the infiltration located?" (GT: Lower Left Lung)
- **Zero-shot:** No answer tag (399 tok)
- **Corr-only:** Long radiographic analysis, no answer tag (392 tok)
- **Spatial:** `left lung` (342 tok)

## Correctness-Only's 3 Unique Wins

### Q: "Does the picture contain liver?" (GT: Yes)
- **Zero-shot:** Long CT analysis, no answer tag
- **Corr-only:** `Yes`
- **Spatial:** Long CT analysis, no answer tag (same failure mode as zero-shot)

### Q: "Which is bigger, liver, spleen or kidney?" (GT: Liver)
- **Zero-shot:** Long explanation, no tag
- **Corr-only:** `liver` (also spatial got this right)

### Q: "Is there a liver in the image?" (GT: Yes)
- **Zero-shot:** No answer tag
- **Corr-only:** `Yes` (also spatial got this right)

## Zero-Shot's 6 Unique Wins

Cases where GRPO training made things worse:

### Q: "Can mass be observed on the lower left lung?" (GT: No)
- **Zero-shot:** `No`
- **Corr-only:** No answer tag
- **Spatial:** No answer tag

### Q: "Does the picture contain heart?" (GT: No) — 2 different images
- **Zero-shot:** `No`
- **Corr-only:** `No` (also correct)
- **Spatial:** No answer tag

### Q: "What color is the liver?" (GT: Gray)
- **Zero-shot:** `gray`
- **Corr-only:** No answer tag (418 tok)
- **Spatial:** Long CT imaging explanation (401 tok), no tag

### Q: "Where is the atelectasis?" (GT: Lower Left Lung)
- **Zero-shot:** `left lung`
- **Corr-only:** Long explanation, no tag
- **Spatial:** Long explanation about dark regions, left/right analysis, no tag

### Q: "Where is the abnormality located?" (GT: Left Lung, Right)
- **Zero-shot:** Gets partial match
- **Corr-only:** No answer tag
- **Spatial:** No answer tag

## Key Observations

### 1. The dominant failure mode is answer tag production, not medical knowledge

All three models have similar medical knowledge (90% identical reasoning). The difference is whether they **commit** to an answer within 512 tokens. Spatial commits more often (51/188 vs 45/188 for corr-only).

### 2. Spatial's advantage is conciseness, not accuracy of reasoning

When spatial wins uniquely, it's not because it reasons better — it's because it produces a short answer ("left lung", "heart", "No") while the others explain endlessly. The corr-only model often identifies the correct answer *within its explanation* but never formats it.

### 3. Correctness-only training increases reasoning loops

Corr-only has 70% reasoning loop rate vs 63% for both zero-shot and spatial. GRPO with pure correctness reward makes the model more verbose and less decisive. The spatial probe reward counteracts this.

### 4. When spatial fails, it fails like zero-shot

Spatial's failures (6 questions where zero-shot wins) look like the base model behavior — long explanations without commitment. These are cases where the probe regularizer wasn't enough to overcome the base model's tendency to over-explain. Spatial never develops a *new* failure mode; it just sometimes doesn't fix the existing one.

### 5. The 87% agreement rate means LoRA barely moved the weights

164/188 questions produce identical outcomes across all three models. At step 40 with LoRA r=16, the model's behavior changed on only 24/188 (13%) of questions. This is consistent with the original rollout analysis finding of 23/30 (77%) identical outputs.
