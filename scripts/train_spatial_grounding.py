"""Train spatial grounding probe using SLAKE bounding box annotations.

Loads SLAKE from the original directory structure (with detection.json per image).
For questions that mention a specific organ, checks if the model's attention
concentrates on that organ's bounding box region.

Usage:
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src python scripts/train_spatial_grounding.py \
        --slake-dir /data3/ishaplan/slake_full/Slake1.0
"""

import torch
import numpy as np
import json
import re
import math
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pickle
from pathlib import Path

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from faithscan.models.lookback_lens import extract_lookback_from_generate
from faithscan.reward import compute_correctness, extract_answer_tag

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Organs we can match in questions
ORGAN_KEYWORDS = {
    "liver": "Liver",
    "kidney": ["Left Kidney", "Right Kidney"],
    "left kidney": "Left Kidney",
    "right kidney": "Right Kidney",
    "spleen": "Spleen",
    "heart": "Heart",
    "lung": ["Left Lung", "Right Lung"],
    "left lung": "Left Lung",
    "right lung": "Right Lung",
    "brain": "Brain",
    "stomach": "Stomach",
    "gallbladder": "Gallbladder",
    "pancreas": "Pancreas",
    "bladder": "Bladder",
}


def find_organ_in_text(text: str, available_organs: list[str]) -> str | None:
    """Find which organ is mentioned in text (question or answer)."""
    t_lower = text.lower()
    available_lower = {o.lower(): o for o in available_organs}

    for keyword, organ_name in ORGAN_KEYWORDS.items():
        if keyword in t_lower:
            if isinstance(organ_name, list):
                for on in organ_name:
                    if on.lower() in available_lower:
                        return available_lower[on.lower()]
            elif organ_name.lower() in available_lower:
                return available_lower[organ_name.lower()]

    for org_lower, org_original in available_lower.items():
        if org_lower in t_lower:
            return org_original

    return None


def load_slake_with_detections(slake_dir: str, split: str = "train"):
    """Load SLAKE questions matched with bounding box annotations."""
    slake_path = Path(slake_dir)

    # Load train/val/test split
    split_file = slake_path / f"{split}.json"
    if not split_file.exists():
        # Try alternate names
        alt_names = {"train": "train.json", "val": "validate.json", "test": "test.json"}
        split_file = slake_path / alt_names.get(split, f"{split}.json")

    if not split_file.exists():
        logger.warning(f"Split file not found: {split_file}")
        return []

    with open(split_file) as f:
        questions = json.load(f)

    # Filter English only
    questions = [q for q in questions if q.get("q_lang") == "en"]

    examples = []
    for q in questions:
        img_name = q["img_name"]  # e.g., "xmlab0/source.jpg"
        img_dir = img_name.split("/")[0]  # e.g., "xmlab0"

        # Load detection.json for this image
        det_path = slake_path / "imgs" / img_dir / "detection.json"
        if not det_path.exists():
            continue

        with open(det_path) as f:
            detections = json.load(f)

        # Parse detections: list of {organ: [x, y, w, h]}
        organ_bboxes = {}
        for det in detections:
            for organ_name, bbox in det.items():
                organ_bboxes[organ_name] = bbox

        if not organ_bboxes:
            continue

        # Find which organ this question is about.
        # Try: question first, then answer, then union of all bboxes.
        organs = list(organ_bboxes.keys())
        target_organ = find_organ_in_text(q["question"], organs)
        bbox_source = "question"

        if target_organ is None:
            target_organ = find_organ_in_text(q.get("answer", ""), organs)
            bbox_source = "answer"

        if target_organ is not None:
            target_bbox = organ_bboxes[target_organ]
        else:
            # Union of all bboxes: bounding box that covers all organs.
            all_bboxes = list(organ_bboxes.values())
            x_min = min(b[0] for b in all_bboxes)
            y_min = min(b[1] for b in all_bboxes)
            x_max = max(b[0] + b[2] for b in all_bboxes)
            y_max = max(b[1] + b[3] for b in all_bboxes)
            target_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            target_organ = "union"
            bbox_source = "union"

        img_path = slake_path / "imgs" / img_dir / "source.jpg"

        examples.append({
            "question": q["question"],
            "answer": q["answer"],
            "image_path": str(img_path),
            "organ_bboxes": organ_bboxes,
            "target_organ": target_organ,
            "target_bbox": target_bbox,
            "bbox_source": bbox_source,
            "img_dir": img_dir,
        })

    n_question = sum(1 for e in examples if e["bbox_source"] == "question")
    n_answer = sum(1 for e in examples if e["bbox_source"] == "answer")
    n_union = sum(1 for e in examples if e["bbox_source"] == "union")
    logger.info(f"Loaded {len(examples)} questions: "
                f"{n_question} matched by question, {n_answer} by answer, "
                f"{n_union} using union bbox")
    return examples


def compute_spatial_focus(
    vision_lookback_ratio: torch.Tensor,
    bbox: list,
    image_size: tuple,
    patch_size: int = 14,
) -> np.ndarray:
    """Compute per-head ratio of attention inside vs outside bounding box.

    Returns (n_layers * n_heads,) spatial ratio vector.
    > 0.5 means more attention to bbox region.
    """
    n_layers, n_heads, n_vis_tokens = vision_lookback_ratio.shape

    # Image dimensions
    w, h = image_size
    grid_w = max(w // patch_size, 1)
    grid_h = max(h // patch_size, 1)
    total_patches = grid_w * grid_h

    # bbox [x, y, w, h] in absolute pixels → patch grid
    bx, by, bw, bh = bbox
    px_start = max(int(bx / patch_size), 0)
    py_start = max(int(by / patch_size), 0)
    px_end = min(int((bx + bw) / patch_size) + 1, grid_w)
    py_end = min(int((by + bh) / patch_size) + 1, grid_h)

    bbox_patch_indices = set()
    for py in range(py_start, py_end):
        for px in range(px_start, px_end):
            idx = py * grid_w + px
            if idx < n_vis_tokens:
                bbox_patch_indices.add(idx)

    if not bbox_patch_indices or len(bbox_patch_indices) >= n_vis_tokens:
        return None

    bbox_idx = list(bbox_patch_indices)
    other_idx = [i for i in range(min(n_vis_tokens, total_patches))
                 if i not in bbox_patch_indices]

    if not other_idx:
        return None

    # Per-head attention to bbox vs other
    in_bbox = vision_lookback_ratio[:, :, bbox_idx].mean(dim=2)   # (L, H)
    out_bbox = vision_lookback_ratio[:, :, other_idx].mean(dim=2)  # (L, H)

    ratio = in_bbox / (in_bbox + out_bbox + 1e-10)  # (L, H)
    return ratio.flatten().numpy()  # (L*H,)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Thinking")
    parser.add_argument("--slake-dir", default="/data3/ishaplan/slake_full/Slake1.0")
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="checkpoints/spatial_grounding/")
    parser.add_argument("--organ-only", action="store_true",
                        help="Only use questions with organ-specific bbox (skip union)")
    args = parser.parse_args()

    device = "cuda"
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # Load SLAKE with bbox annotations
    examples = []
    for split in ["train", "validate"]:
        examples.extend(load_slake_with_detections(args.slake_dir, split))

    # Filter examples
    if args.organ_only:
        organ_examples = [e for e in examples
                          if e["target_bbox"] is not None and e["bbox_source"] != "union"]
    else:
        organ_examples = [e for e in examples if e["target_bbox"] is not None]
    logger.info(f"Using {len(organ_examples)} examples with spatial annotations "
                f"(out of {len(examples)} total)")

    if not organ_examples:
        logger.error("No organ-specific examples found!")
        return

    organ_examples = organ_examples[:args.max_examples]

    all_spatial_ratios = []
    all_lookback_feats = []
    all_correct = []
    focused_correct = 0
    focused_wrong = 0
    unfocused_correct = 0
    unfocused_wrong = 0

    for i, ex in enumerate(tqdm(organ_examples, desc="Collecting")):
        try:
            image = Image.open(ex["image_path"]).convert("RGB")
        except Exception as e:
            continue

        try:
            result = extract_lookback_from_generate(
                model, processor, image, ex["question"], device,
                max_new_tokens=args.max_new_tokens, do_sample=False,
            )
        except Exception as e:
            logger.debug(f"Skip {i}: {e}")
            continue

        raw = result["answer_text"]
        extracted = extract_answer_tag(raw)
        c = compute_correctness(extracted if extracted else raw, ex["answer"])
        correct = 1.0 if c > 0.5 else 0.0

        # Compute spatial focus on target organ.
        # Use spatial_vision_attn which has shape (n_layers, n_heads, n_vision_tokens)
        # — actual per-vision-token attention from generated tokens.
        import math
        spatial_attn = result["spatial_vision_attn"]
        n_vis_tokens = spatial_attn.shape[2]
        img_w, img_h = image.size
        grid_side = int(math.sqrt(n_vis_tokens))
        effective_patch_size = max(img_w, img_h) / grid_side if grid_side > 0 else 14

        if i == 0:
            logger.info(f"Debug: image={img_w}x{img_h}, vis_tokens={n_vis_tokens}, "
                        f"grid={grid_side}x{grid_side}, patch_size={int(effective_patch_size)}, "
                        f"bbox={ex['target_bbox']}, organ={ex['target_organ']}")

        spatial = compute_spatial_focus(
            spatial_attn,
            ex["target_bbox"],
            image.size,
            patch_size=int(effective_patch_size),
        )

        if spatial is None:
            continue

        mean_focus = spatial.mean()
        focused = mean_focus > 0.5

        all_spatial_ratios.append(spatial)
        all_lookback_feats.append(
            result["vision_lookback_ratio"].mean(dim=2).flatten().numpy()
        )
        all_correct.append(correct)

        if focused and correct > 0.5:
            focused_correct += 1
        elif focused and correct <= 0.5:
            focused_wrong += 1
        elif not focused and correct > 0.5:
            unfocused_correct += 1
        else:
            unfocused_wrong += 1

        if (i + 1) % 30 == 0:
            n = len(all_correct)
            print(f"\n--- After {n} examples ---")
            print(f"Accuracy: {sum(all_correct)/n:.3f}")
            print(f"Mean spatial focus: {np.mean([s.mean() for s in all_spatial_ratios]):.4f}")
            print(f"  Focused + Correct:   {focused_correct}")
            print(f"  Focused + Wrong:     {focused_wrong}")
            print(f"  Unfocused + Correct: {unfocused_correct}")
            print(f"  Unfocused + Wrong:   {unfocused_wrong}")

            X = np.array(all_spatial_ratios)
            y = np.array(all_correct)
            if len(np.unique(y)) > 1:
                # Spatial ratio → correctness
                try:
                    clf = LogisticRegression(max_iter=1000, C=1.0)
                    clf.fit(X, y)
                    probs = clf.predict_proba(X)[:, 1]
                    auroc = roc_auc_score(y, probs)
                    print(f"Spatial ratio → correctness: AUROC={auroc:.3f}")
                except Exception:
                    pass

                # Correlation
                focus_scores = np.array([s.mean() for s in all_spatial_ratios])
                corr = np.corrcoef(focus_scores, y)[0, 1]
                print(f"Spatial focus ↔ correctness: r={corr:.3f}")

                # Top correlated heads
                correlations = np.array([
                    np.corrcoef(X[:, h], y)[0, 1] for h in range(X.shape[1])
                ])
                top_pos = np.argsort(correlations)[-3:][::-1]
                top_neg = np.argsort(correlations)[:3]
                print(f"Top spatial heads (correct = focused on organ):")
                for idx in top_pos:
                    layer, head = idx // 12, idx % 12
                    print(f"  Layer {layer} Head {head}: r={correlations[idx]:.3f}")
                print(f"Top anti-correlated:")
                for idx in top_neg:
                    layer, head = idx // 12, idx % 12
                    print(f"  Layer {layer} Head {head}: r={correlations[idx]:.3f}")
            print()

    # Train/val split and final classifier
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    X = np.array(all_spatial_ratios)
    y = np.array(all_correct)

    n = len(y)
    logger.info(f"Final: {n} examples, {int(y.sum())} correct")
    logger.info(f"  Focused + Correct:   {focused_correct}")

    if len(np.unique(y)) >= 2 and n >= 20:
        n_train = int(0.8 * n)
        idx = np.random.permutation(n)
        train_idx, val_idx = idx[:n_train], idx[n_train:]

        clf = LogisticRegression(max_iter=1000, C=0.1)  # lower C to reduce overfit
        clf.fit(X[train_idx], y[train_idx])

        train_probs = clf.predict_proba(X[train_idx])[:, 1]
        val_probs = clf.predict_proba(X[val_idx])[:, 1]

        train_auroc = roc_auc_score(y[train_idx], train_probs)
        val_auroc = roc_auc_score(y[val_idx], val_probs)

        # Correlation of classifier score with correctness on val
        val_corr = np.corrcoef(val_probs, y[val_idx])[0, 1]

        logger.info(f"Train AUROC: {train_auroc:.3f}")
        logger.info(f"Val AUROC: {val_auroc:.3f}")
        logger.info(f"Val classifier ↔ correctness: r={val_corr:.3f}")

        # Save classifier
        with open(out / "spatial_classifier.pkl", "wb") as f:
            pickle.dump(clf, f)
        logger.info(f"Saved spatial classifier to {out / 'spatial_classifier.pkl'}")

    np.savez(out / "spatial_features.npz",
             spatial_ratios=X,
             lookback_feats=np.array(all_lookback_feats),
             correct_labels=y)
    logger.info(f"Saved features to {out}")

    # Final summary
    logger.info(f"  Focused + Correct:   {focused_correct}")
    logger.info(f"  Focused + Wrong:     {focused_wrong}")
    logger.info(f"  Unfocused + Correct: {unfocused_correct}")
    logger.info(f"  Unfocused + Wrong:   {unfocused_wrong}")


if __name__ == "__main__":
    main()
