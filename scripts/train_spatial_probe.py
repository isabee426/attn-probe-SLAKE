"""Train spatial grounding probe on SLAKE organ bounding boxes.

Step 1 of the pipeline: must run BEFORE GRPO training.
Generates the spatial_classifier.pkl used as the faithfulness reward signal.

Two label modes:
  --labels correctness   (default) Probe predicts correctness from spatial attention.
                         Circular but worked well in original April 4 experiment.
  --labels bbox_overlap  Probe predicts whether attention overlaps with ground-truth
                         bbox (non-circular). Use for publishable version.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../final_version/src python scripts/train_spatial_probe.py \
        --slake-dir /data3/ishaplan/slake_full/Slake1.0 \
        --output /data3/ishaplan/slake_reproduction/checkpoints/spatial_probe/ \
        --organ-only --labels bbox_overlap
"""

import argparse
import json
import math
import pickle
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from faithscan.models.lookback_lens import extract_lookback_from_generate
from faithscan.reward import compute_correctness, extract_answer_tag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    slake_path = Path(slake_dir)
    split_file = slake_path / f"{split}.json"
    if not split_file.exists():
        alt_names = {"train": "train.json", "val": "validate.json", "test": "test.json"}
        split_file = slake_path / alt_names.get(split, f"{split}.json")
    if not split_file.exists():
        logger.warning(f"Split file not found: {split_file}")
        return []

    with open(split_file) as f:
        questions = json.load(f)
    questions = [q for q in questions if q.get("q_lang") == "en"]

    examples = []
    for q in questions:
        img_name = q["img_name"]
        img_dir = img_name.split("/")[0]
        det_path = slake_path / "imgs" / img_dir / "detection.json"
        if not det_path.exists():
            continue

        with open(det_path) as f:
            detections = json.load(f)
        organ_bboxes = {}
        for det in detections:
            for organ_name, bbox in det.items():
                organ_bboxes[organ_name] = bbox
        if not organ_bboxes:
            continue

        organs = list(organ_bboxes.keys())
        target_organ = find_organ_in_text(q["question"], organs)
        bbox_source = "question"
        if target_organ is None:
            target_organ = find_organ_in_text(q.get("answer", ""), organs)
            bbox_source = "answer"
        if target_organ is not None:
            target_bbox = organ_bboxes[target_organ]
        else:
            target_organ = "union"
            bbox_source = "union"
            all_bboxes = list(organ_bboxes.values())
            x_min = min(b[0] for b in all_bboxes)
            y_min = min(b[1] for b in all_bboxes)
            x_max = max(b[0] + b[2] for b in all_bboxes)
            y_max = max(b[1] + b[3] for b in all_bboxes)
            target_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

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

    n_q = sum(1 for e in examples if e["bbox_source"] == "question")
    n_a = sum(1 for e in examples if e["bbox_source"] == "answer")
    n_u = sum(1 for e in examples if e["bbox_source"] == "union")
    logger.info(f"Loaded {len(examples)}: {n_q} by question, {n_a} by answer, {n_u} union")
    return examples


def compute_spatial_focus(spatial_attn, bbox, image_size, patch_size=14):
    n_layers, n_heads, n_vis_tokens = spatial_attn.shape
    w, h = image_size
    grid_w = max(w // patch_size, 1)
    grid_h = max(h // patch_size, 1)

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
    other_idx = [i for i in range(min(n_vis_tokens, grid_w * grid_h))
                 if i not in bbox_patch_indices]
    if not other_idx:
        return None

    in_bbox = spatial_attn[:, :, bbox_idx].mean(dim=2)
    out_bbox = spatial_attn[:, :, other_idx].mean(dim=2)
    ratio = in_bbox / (in_bbox + out_bbox + 1e-10)
    return ratio.flatten().numpy()


def compute_bbox_overlap_label(spatial_attn, bbox, image_size, patch_size, threshold=0.5):
    """Binary label: does mean attention to bbox exceed threshold?

    Non-circular label — doesn't depend on correctness at all.
    """
    focus = compute_spatial_focus(spatial_attn, bbox, image_size, patch_size)
    if focus is None:
        return None, None
    mean_focus = float(focus.mean())
    label = 1.0 if mean_focus > threshold else 0.0
    return label, focus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Thinking")
    parser.add_argument("--slake-dir", default="/data3/ishaplan/slake_full/Slake1.0")
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="/data3/ishaplan/slake_reproduction/checkpoints/spatial_probe/")
    parser.add_argument("--organ-only", action="store_true",
                        help="Skip union bbox (default for reproduction)")
    parser.add_argument("--labels", choices=["correctness", "bbox_overlap"], default="bbox_overlap",
                        help="Label mode: correctness (circular) or bbox_overlap (non-circular)")
    args = parser.parse_args()

    device = "cuda"
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    examples = []
    for split in ["train", "validate"]:
        examples.extend(load_slake_with_detections(args.slake_dir, split))

    if args.organ_only:
        examples = [e for e in examples
                    if e["target_bbox"] is not None and e["bbox_source"] != "union"]
    else:
        examples = [e for e in examples if e["target_bbox"] is not None]

    logger.info(f"Using {len(examples)} examples with spatial annotations")
    examples = examples[:args.max_examples]

    all_spatial_ratios = []
    all_labels = []
    all_correct = []

    for i, ex in enumerate(tqdm(examples, desc="Extracting spatial features")):
        try:
            image = Image.open(ex["image_path"]).convert("RGB")
        except Exception:
            continue
        try:
            result = extract_lookback_from_generate(
                model, processor, image, ex["question"], device,
                max_new_tokens=args.max_new_tokens, do_sample=False,
            )
        except Exception as e:
            logger.debug(f"Skip {i}: {e}")
            continue

        spatial_attn = result["spatial_vision_attn"]
        n_vis = spatial_attn.shape[2]
        img_w, img_h = image.size
        grid_side = int(math.sqrt(n_vis))
        eff_patch = max(img_w, img_h) / grid_side if grid_side > 0 else 14

        focus = compute_spatial_focus(
            spatial_attn, ex["target_bbox"], image.size, patch_size=int(eff_patch))
        if focus is None:
            continue

        raw = result["answer_text"]
        extracted = extract_answer_tag(raw)
        c = compute_correctness(extracted if extracted else raw, ex["answer"])
        correct = 1.0 if c > 0.5 else 0.0
        all_correct.append(correct)

        if args.labels == "correctness":
            all_labels.append(correct)
        else:
            mean_focus = float(focus.mean())
            all_labels.append(1.0 if mean_focus > 0.5 else 0.0)

        all_spatial_ratios.append(focus)

        if (i + 1) % 50 == 0:
            n = len(all_labels)
            logger.info(f"  {n} examples | accuracy={np.mean(all_correct):.3f} | "
                        f"mean focus={np.mean([s.mean() for s in all_spatial_ratios]):.4f}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    X = np.array(all_spatial_ratios)
    y = np.array(all_labels)
    y_correct = np.array(all_correct)

    logger.info(f"Final: {len(y)} examples, labels mode={args.labels}")
    logger.info(f"  Positive labels: {int(y.sum())}/{len(y)} ({100*y.mean():.1f}%)")
    logger.info(f"  Accuracy: {y_correct.mean():.3f}")

    if len(np.unique(y)) >= 2 and len(y) >= 20:
        n_train = int(0.8 * len(y))
        idx = np.random.RandomState(42).permutation(len(y))
        train_idx, val_idx = idx[:n_train], idx[n_train:]

        # Sweep C values
        best_auroc = 0
        best_clf = None
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(max_iter=1000, C=C)
            clf.fit(X[train_idx], y[train_idx])
            val_probs = clf.predict_proba(X[val_idx])[:, 1]
            auroc = roc_auc_score(y[val_idx], val_probs)
            logger.info(f"  C={C}: val AUROC={auroc:.3f}")
            if auroc > best_auroc:
                best_auroc = auroc
                best_clf = clf

        logger.info(f"Best val AUROC: {best_auroc:.3f}")

        # Also check correlation with correctness on val
        val_probs = best_clf.predict_proba(X[val_idx])[:, 1]
        corr = np.corrcoef(val_probs, y_correct[val_idx])[0, 1]
        logger.info(f"Probe score <-> correctness (val): r={corr:.3f}")

        with open(out / "spatial_classifier.pkl", "wb") as f:
            pickle.dump(best_clf, f)
        logger.info(f"Saved classifier to {out / 'spatial_classifier.pkl'}")

    np.savez(out / "spatial_features.npz",
             spatial_ratios=X, labels=y, correct_labels=y_correct)
    logger.info(f"Saved features to {out / 'spatial_features.npz'}")


if __name__ == "__main__":
    main()
