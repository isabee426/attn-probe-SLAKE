"""Qualitative rollout analysis: generate 8 rollouts per example, compare behavior.

Captures the key qualitative findings from the Notion notes:
- Response length (conciseness)
- Reasoning loops ("Wait, no, wait")
- Visual specificity ("axial view" vs "abdominal region")
- Answer decisiveness (commits vs loops to max tokens)

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../final_version/src python scripts/rollout_analysis.py \
        --corr-ckpt /data3/ishaplan/slake_reproduction/checkpoints/correctness_only/seed42/best \
        --spatial-ckpt /data3/ishaplan/slake_reproduction/checkpoints/spatial_grpo/a07_seed42/best \
        --n 30 --output results/rollout_analysis.json
"""

import argparse
import json
import re
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from faithscan.data.dataset import load_multi_dataset, get_image
from faithscan.reward import extract_answer_tag, compute_correctness
from qwen_vl_utils import process_vision_info


def load_model(checkpoint=None):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    name = "Qwen/Qwen3-VL-2B-Thinking"
    processor = AutoProcessor.from_pretrained(name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        name, torch_dtype=torch.bfloat16,
    ).cuda()
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint).cuda()
        model = model.merge_and_unload()
    model.eval()
    return model, processor


def generate_rollouts(model, processor, image, question, n=8, max_new_tokens=512):
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                "Answer this medical question. Think step by step, then give "
                "your final answer in <answer>...</answer> tags.\n"
                f"Question: {question}"
            )},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda")

    prefill_len = inputs["input_ids"].shape[1]

    # Greedy for comparison
    with torch.no_grad():
        gen_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    greedy_ids = gen_out[0, prefill_len:]
    greedy_text = processor.tokenizer.decode(greedy_ids, skip_special_tokens=False).strip()

    # Sampled rollouts
    batched = {}
    for k, v in inputs.items():
        if v.shape[0] == 1:
            batched[k] = v.expand(n, *v.shape[1:]).clone()
        else:
            batched[k] = v.repeat(n, *([1] * (v.dim() - 1)))

    with torch.no_grad():
        gen_out = model.generate(
            **batched, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=1.0, top_p=1.0)

    sampled = []
    for b in range(n):
        ids = gen_out[b, prefill_len:]
        sampled.append(processor.tokenizer.decode(ids, skip_special_tokens=False).strip())

    return greedy_text, sampled


def analyze_response(text):
    """Extract behavioral metrics from a response."""
    n_tokens = len(text.split())
    has_answer = extract_answer_tag(text) is not None
    n_wait = len(re.findall(r'\b[Ww]ait\b', text))
    n_no = len(re.findall(r'\b[Nn]o,?\s', text))
    has_loop = n_wait >= 2 or (n_wait >= 1 and n_no >= 1)
    hit_max = n_tokens >= 480  # near 512 token limit
    return {
        "n_tokens": n_tokens,
        "has_answer_tag": has_answer,
        "reasoning_loops": n_wait + n_no,
        "has_loop": has_loop,
        "hit_max_tokens": hit_max,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corr-ckpt", required=True)
    parser.add_argument("--spatial-ckpt", required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--slake-dir", default="/data3/ishaplan/slake_full/Slake1.0")
    parser.add_argument("--output", default="results/rollout_analysis.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load organ-filtered val set (same split as training)
    examples = load_multi_dataset(
        datasets=["slake"], splits=["train", "val"],
        image_cache_dir="data/images",
    )
    slake_path = Path(args.slake_dir)
    all_organs = set()
    for img_dir in (slake_path / "imgs").iterdir():
        det_path = img_dir / "detection.json"
        if det_path.exists():
            with open(det_path) as f:
                dets = json.load(f)
            for det in dets:
                for organ in det:
                    all_organs.add(organ.lower())

    examples = [ex for ex in examples
                if any(o in (ex.question.lower() + " " + ex.answer.lower()) for o in all_organs)]
    np.random.RandomState(args.seed).shuffle(examples)
    n_val = max(1, len(examples) // 11)
    val_examples = examples[len(examples) - n_val:]

    if args.n < len(val_examples):
        rng = np.random.RandomState(0)
        val_examples = list(rng.choice(val_examples, args.n, replace=False))

    print(f"Analyzing {len(val_examples)} examples x 3 models x ({args.rollouts} rollouts + 1 greedy)")

    all_results = []

    for model_name, ckpt in [("zero_shot", None), ("corr_only", args.corr_ckpt), ("spatial", args.spatial_ckpt)]:
        print(f"\n=== {model_name} ===")
        model, proc = load_model(ckpt)

        for i, ex in enumerate(tqdm(val_examples, desc=model_name)):
            try:
                image = get_image(ex)
            except Exception:
                continue

            greedy, sampled = generate_rollouts(
                model, proc, image, ex.question, n=args.rollouts)

            greedy_metrics = analyze_response(greedy)
            greedy_ans = extract_answer_tag(greedy)
            greedy_c = compute_correctness(greedy, ex.answer)

            sampled_metrics = [analyze_response(s) for s in sampled]
            sampled_correct = [compute_correctness(s, ex.answer) for s in sampled]

            if len(all_results) <= i:
                all_results.append({"question": ex.question, "gt": ex.answer})

            all_results[i][model_name] = {
                "greedy": greedy,
                "greedy_answer": greedy_ans,
                "greedy_correct": greedy_c,
                "greedy_metrics": greedy_metrics,
                "sampled_avg_tokens": np.mean([m["n_tokens"] for m in sampled_metrics]),
                "sampled_loop_rate": np.mean([m["has_loop"] for m in sampled_metrics]),
                "sampled_hit_max_rate": np.mean([m["hit_max_tokens"] for m in sampled_metrics]),
                "sampled_correct_mean": np.mean(sampled_correct),
                "sampled_correct_std": np.std(sampled_correct),
            }

        del model
        torch.cuda.empty_cache()

    # Aggregate summary
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)
    for model_name in ["zero_shot", "corr_only", "spatial"]:
        greedy_c = [r[model_name]["greedy_correct"] for r in all_results if model_name in r]
        avg_tok = [r[model_name]["greedy_metrics"]["n_tokens"] for r in all_results if model_name in r]
        loop_rate = [r[model_name]["greedy_metrics"]["has_loop"] for r in all_results if model_name in r]
        max_rate = [r[model_name]["greedy_metrics"]["hit_max_tokens"] for r in all_results if model_name in r]
        sam_c = [r[model_name]["sampled_correct_mean"] for r in all_results if model_name in r]

        print(f"\n{model_name}:")
        print(f"  Greedy F1: {np.mean(greedy_c):.3f}")
        print(f"  Greedy tokens: {np.mean(avg_tok):.0f} avg")
        print(f"  Reasoning loops: {100*np.mean(loop_rate):.0f}%")
        print(f"  Hit max tokens: {100*np.mean(max_rate):.0f}%")
        print(f"  Sampled F1: {np.mean(sam_c):.3f} +/- {np.std(sam_c):.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
