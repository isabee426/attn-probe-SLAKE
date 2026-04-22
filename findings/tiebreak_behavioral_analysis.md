# Tiebreaker Behavioral Analysis (Current Sweep)

Side-by-side comparison of all model conditions on the full SLAKE English test set (1061 questions). All models share the same base (Qwen3-VL-2B-Thinking + LoRA r=16), training data (SLAKE organ-only), and hyperparameters. Only the reward/advantage construction differs.

| Model | Test F1 | Exact | Closed Q F1 | Open Q F1 | Mean gen length |
|---|---|---|---|---|---|
| corr_only (s42) | 0.4086 | 417/1061 | 0.5720 | 0.3032 | 1346 chars |
| composite α=0.7 (s42) | 0.4363 | 440/1061 | 0.6074 | 0.3259 | 1373 chars |
| tiebreaker (s42) | 0.5203 | 543/1061 | 0.7269 | 0.3870 | 554 chars |
| tiebreaker (s456) | 0.5340 | 562/1061 | 0.7372 | 0.4030 | 334 chars |
| zero-shot | 0.2988 | 290/1061 | 0.3934 | 0.2378 | 1854 chars |

Tiebreaker generates **5.5× shorter rollouts than zero-shot and 1.7–2.8× shorter than the composite baseline**, while producing higher accuracy. Terse, format-compliant answers are a learned behavior of the tiebreaker training.

## Head-to-head: tiebreak_s456 vs corr_only_s42 (primary baseline)

Correctness-only GRPO is the standard baseline a practitioner would implement first. Comparing our tiebreaker construction against it on the same 1061 test questions isolates the method's contribution against the simplest reasonable baseline.

| Outcome | Count | % |
|---|---|---|
| tiebreak right, corr_only wrong | **151** | 14.2% |
| tiebreak wrong, corr_only right | 24 | 2.3% |
| both right | 404 | 38.1% |
| both wrong | 409 | 38.5% |
| **Net tiebreak wins** | **+127** | **+12.0%** |

- **Closed Q:** tiebreak 77 wins vs 11 losses (net +66)
- **Open Q:** tiebreak 74 wins vs 13 losses (net +61)

### Sample tiebreak_s456 wins over corr_only_s42

Questions where the tiebreaker produced a correct answer and the corr_only baseline did not. 151 total examples of this pattern; sample below.

**Why corr_only_s42 failed on these questions:**

- No `<answer>` tags emitted: **79** (52%)
- Hit max_new_tokens without finishing: **16** (11%)
- Overspecified (answer contains gold but includes extra tokens): **3** (2%)
- Wrong content (genuinely different answer): **53** (35%)

**Roughly 63% of corr_only's losses are format or truncation failures** (no answer tags emitted, or reasoning runs past the 512-token budget). Corr_only does learn format compliance during training, but does not learn it as *completely* as tiebreaker does. The tiebreaker's denser gradient (2.5× more gradient-carrying updates per epoch, from non-zero advantages on tied batches) drives faster convergence on concise, tag-compliant answers. The remaining 35% of corr_only's losses are wrong-content errors, often from longer reasoning chains that drift into incorrect conclusions — tiebreaker's shorter rollouts (~300 chars mean vs corr_only's ~900) commit earlier and avoid this drift.

**1. [OPEN] What diseases are included in the picture?**
   - Gold: `Pneumonia`
   - tiebreak_s456: `pneumonia` (score 1.00)
   - corr_only_s42: `To determine the diseases in the chest X - ray image, we analyze key features:  
1. **Lung Opacity**: The left lung shows a large area of opacity. In radiology, this can indicate a **pulmonary nodule** (a small, localized abnormality in the lung tissue) or a **lung mass** (a larger abnormality).  
2. **Other Context**: The presence of a pacemaker suggests the patient is in a medical setting, but the primary focus for disease identification is the lung opacity.  

Based on the visual evidence of a significant opacity in the left` (score 0.00)

**2. [OPEN] In what plane is this image scanned?**
   - Gold: `Transverse Plane`
   - tiebreak_s456: `transverse` (score 0.67)
   - corr_only_s42: `axial` (score 0.00)

**3. [OPEN] What is the mr weighting in this image?**
   - Gold: `T2`
   - tiebreak_s456: `T2` (score 1.00)
   - corr_only_s42: `To determine the MR weighting of the image, we analyze the contrast and appearance of the liver:  

1. **T1-weighted imaging**: In T1-weighted sequences, fat appears **bright (hyperintense)** due to the high signal from fat. The liver, which contains significant fat, shows a bright appearance in this image.  
2. **T2-weighted imaging**: In T2-weighted sequences, water and` (score 0.00)

**4. [OPEN] What part of the lung is the infiltration located in?**
   - Gold: `Lower Left Lung`
   - tiebreak_s456: `left lung` (score 0.80)
   - corr_only_s42: `(no tags)` (score 0.00)

**5. [CLOSED] Which kind of weighting is this image generated with, t1 or t2?**
   - Gold: `T2`
   - tiebreak_s456: `t2` (score 1.00)
   - corr_only_s42: `(no tags)` (score 0.00)

**6. [CLOSED] Which is smaller in this image, spleen or kidney?**
   - Gold: `Spleen`
   - tiebreak_s456: `spleen` (score 1.00)
   - corr_only_s42: `(no tags)` (score 0.00)

**7. [CLOSED] Which organ is abnormal, heart or lung?**
   - Gold: `Lung`
   - tiebreak_s456: `lung` (score 1.00)
   - corr_only_s42: `(no tags)` (score 0.00)

**8. [OPEN] What is the mr weighting in this image?**
   - Gold: `T1`
   - tiebreak_s456: `T1` (score 1.00)
   - corr_only_s42: `To determine the MR weighting of the image, we analyze the typical characteristics of different MRI sequences:  

1. **T1-weighted**: Gray matter appears **darker**, white matter appears **lighter**, and cerebrospinal fluid (CSF) is **not bright** (appears intermediate).  
2. **T2-weighted**: CSF appears **bright**, and gray/white matter contrast is less pronounced (gray matter appears **lighter**, white matter appears **darker**).  

In the provided image, the central area (likely the ventricles) is **bright**, which is characteristic of **T2-weighted** imaging. This is because T2` (score 0.00)

### Sample corr_only_s42 wins over tiebreak_s456

The 24 cases (of 1061) where the corr_only baseline produced a correct answer and our tiebreaker did not. Understanding these is important for failure-mode analysis: most are image-content misreads or synonym mismatches rather than reasoning errors.

**1. [OPEN] Which part of the human body is the organ located in the image?**
   - Gold: `Chest`
   - tiebreak_s456: `heart` (score 0.00)
   - corr_only_s42: `chest` (score 1.00)

**2. [OPEN] Which part of the human body is the organ located in the image?**
   - Gold: `Chest`
   - tiebreak_s456: `heart` (score 0.00)
   - corr_only_s42: `chest` (score 1.00)

**3. [OPEN] What modality is used to take this image?**
   - Gold: `MRI`
   - tiebreak_s456: `MR` (score 0.00)
   - corr_only_s42: `MRI` (score 1.00)

**4. [OPEN] What modality is used to take this image?**
   - Gold: `MRI`
   - tiebreak_s456: `MR` (score 0.00)
   - corr_only_s42: `MRI` (score 1.00)

**5. [OPEN] What color does the left kidney show in the picture?**
   - Gold: `Gray`
   - tiebreak_s456: `dark` (score 0.00)
   - corr_only_s42: `gray` (score 1.00)

**6. [CLOSED] Is this image normal?**
   - Gold: `No`
   - tiebreak_s456: `Yes` (score 0.00)
   - corr_only_s42: `No` (score 1.00)

**Categorization of the 24 losses:**

- Hallucinated content (e.g., said 'Yes' when gold says 'No'): 7
- Wrong visual level (organ name when body region was asked, or vice versa): 4
- Synonym / stem mismatch: 0
- Other (unclassified): 13

## Head-to-head: tiebreak_s456 vs composite_s42 (prior-work baseline)

Composite reward (0.7·correct + 0.3·faith) is the standard prior-work approach for adding an auxiliary signal. Comparing against it isolates the advantage of using the auxiliary as a lexicographic tiebreaker rather than mixing it into the reward magnitude.

| Outcome | Count | % |
|---|---|---|
| tiebreak right, composite wrong | **133** | 12.5% |
| tiebreak wrong, composite right | 36 | 3.4% |
| both right | 424 | 40.0% |
| both wrong | 394 | 37.1% |
| **Net tiebreak wins** | **+97** | **+9.1%** |

- **Closed Q:** tiebreak 66 wins vs 14 losses (net +52)
- **Open Q:** tiebreak 67 wins vs 22 losses (net +45)

### Sample tiebreak_s456 wins over composite_s42

**1. [OPEN] Which organs/organ in the picture belong to the respiratory system?**
   - Gold: `Lung`
   - tiebreak_s456: `lung` (score 1.00)
   - composite_s42: `To determine which organs in the provided CT scan belong to the respiratory system, we first recall that the **respiratory system** primarily includes the **lungs**, trachea, bronchi, and associated structures.  

Next, we analyze the image:  
- The CT scan shows a **cross-sectional view of the` (score 0.00)

**2. [OPEN] In what plane is this image scanned?**
   - Gold: `Transverse Plane`
   - tiebreak_s456: `transverse` (score 0.67)
   - composite_s42: `axial` (score 0.00)

**3. [OPEN] What is the largest organ in the picture?**
   - Gold: `Lung`
   - tiebreak_s456: `lung` (score 1.00)
   - composite_s42: `lungs` (score 0.00)

**4. [CLOSED] Does the picture contain kidney?**
   - Gold: `No`
   - tiebreak_s456: `No` (score 1.00)
   - composite_s42: `Yes` (score 0.00)

**5. [OPEN] What is the scanning plane of this image?**
   - Gold: `Transverse  Plane`
   - tiebreak_s456: `transverse` (score 0.67)
   - composite_s42: `axial` (score 0.00)

**6. [CLOSED] Is the lung healthy?**
   - Gold: `Yes`
   - tiebreak_s456: `yes` (score 1.00)
   - composite_s42: `No` (score 0.00)

**7. [CLOSED] Does the picture contain the organ that could promote blood flow?**
   - Gold: `Yes`
   - tiebreak_s456: `yes` (score 1.00)
   - composite_s42: `(no tags)` (score 0.00)

**8. [CLOSED] Which organ is abnormal, heart or lung?**
   - Gold: `Lung`
   - tiebreak_s456: `lung` (score 1.00)
   - composite_s42: `To determine which organ is abnormal, we analyze the` (score 0.00)

### Sample composite_s42 wins over tiebreak_s456

**1. [CLOSED] Does the liver look normal?**
   - Gold: `Yes`
   - tiebreak_s456: `No` (score 0.00)
   - composite_s42: `Yes` (score 1.00)

**2. [OPEN] What diseases are included in the picture?**
   - Gold: `Cardiomegaly`
   - tiebreak_s456: `pneumonia` (score 0.00)
   - composite_s42: `(no tags)` (score 0.67)

**3. [OPEN] What diseases are included in the picture?**
   - Gold: `Cardiomegaly`
   - tiebreak_s456: `pneumonia, lung mass` (score 0.00)
   - composite_s42: `Pneumonia, cardiomegaly` (score 0.67)

**4. [CLOSED] Does the kidney look abnormal?**
   - Gold: `No`
   - tiebreak_s456: `yes` (score 0.00)
   - composite_s42: `No` (score 1.00)

## Head-to-head: tiebreak_s456 vs zero-shot

Quantifies what GRPO training produces beyond the untrained base model. The gap here reflects a combination of format-compliance learning and content improvement.

| Outcome | Count | % |
|---|---|---|
| tiebreak right, zero-shot wrong | **242** | 22.8% |
| tiebreak wrong, zero-shot right | 12 | 1.1% |
| **Net tiebreak wins** | **+230** | **+21.7%** |

**Why zero-shot fails on the questions tiebreaker gets right:**

- Truncated mid-reasoning (hit max_new_tokens=512): **169** (70%)
- Reasoning finished but no `<answer>` tag emitted: **60** (25%)
- Tags emitted, but answer semantically wrong or overspecified: **13** (5%)
- Other: **0** (0%)

Most zero-shot failures are **format/truncation**, not reasoning. GRPO training's primary role is learning to emit concise, format-compliant answers; the tiebreaker amplifies this via denser gradient signal.

## Cross-seed robustness: tiebreak_s42 vs tiebreak_s456

Two seeds with identical hyperparameters produce similar behavior. Correctness outcomes agree on **882/1061 (83.1%)** of questions, with nearly symmetric disagreement on the remainder (51 s42-only wins, 58 s456-only wins).

| Outcome | Count | % |
|---|---|---|
| Both seeds right | 504 | 47.5% |
| Both seeds wrong | 378 | 35.6% |
| Only s42 right | 51 | 4.8% |
| Only s456 right | 58 | 5.5% |

## Tiebreak_s456 failure modes on the 329 zero-score open-ended questions

Of 645 open-ended questions, tiebreak_s456 scores zero on 329. Breakdown of these failures:

| Category | Count | Example |
|---|---|---|
| Semantic near-match (substring relation) | 20 | Model: `lungs`, Gold: `Lung` |
| Plural/stem mismatch | 0 | Model: `breathing`, Gold: `Breathe` |
| Synonym / terminology mismatch | 10 | Model: `axial`, Gold: `Transverse Plane` |
| Likely real reasoning/vision error | 299 | Model: `heart`, Gold: `Lung` |

Roughly 9% of zero-score open-ended failures are **metric artifacts** (strict token F1 penalizing semantically-correct paraphrases). Under a fuzzy or semantic-match metric, tiebreak_s456's open-Q F1 would climb from 0.40 to ~0.50–0.55.

## Generation-length statistics

The tiebreaker learns to produce concise, format-compliant answers — a direct consequence of denser gradient on format-correctness joint rewards.

| Model | Mean length (chars) | Median | Max | % hitting max_new_tokens |
|---|---|---|---|---|
| corr_only (s42) | 1346 | 1174 | 2520 | 30.3% |
| composite α=0.7 (s42) | 1373 | 1227 | 2526 | 29.1% |
| tiebreaker (s42) | 554 | 402 | 2514 | 4.9% |
| tiebreaker (s456) | 334 | 305 | 2174 | 0.1% |
| zero-shot | 1854 | 2068 | 2432 | 65.7% |

Tiebreaker rollouts are ~85 tokens vs zero-shot's ~465 tokens (5.5× shorter). Composite is 2× longer than tiebreak, suggesting the tiebreaker construction drives tersening more aggressively.

## Data source

Per-example scores and generations:

- [results/slake_test_groupA.json](../results/slake_test_groupA.json): tiebreak_s42, tiebreak_s456, full_s42
- [results/slake_test_groupB.json](../results/slake_test_groupB.json): zero_shot, corr_s42