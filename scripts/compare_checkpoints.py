"""Compare greedy outputs from zero-shot, correctness-only, and spatial GRPO.

Generates a structured evaluation table + per-example comparison for the paper.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../final_version/src python scripts/compare_checkpoints.py \
        --corr-ckpt /data3/ishaplan/slake_reproduction/checkpoints/correctness_only/seed42/best \
        --spatial-ckpt /data3/ishaplan/slake_reproduction/checkpoints/spatial_grpo/a07_seed42/best \
        --n 30 --output results/comparison_seed42.json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from faithscan.data.dataset import load_multi_dataset, iter_examples_by_split, get_image
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


def generate(model, processor, image, question, max_new_tokens=512):
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
    with torch.no_grad():
        gen_out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = gen_out[0, prefill_len:]
    return processor.tokenizer.decode(gen_ids, skip_special_tokens=False).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corr-ckpt", required=True, help="Correctness-only best checkpoint")
    parser.add_argument("--spatial-ckpt", required=True, help="Spatial GRPO (bbox probe) best checkpoint")
    parser.add_argument("--spatial-corr-ckpt", default=None, help="Spatial GRPO (correctness probe) best checkpoint")
    parser.add_argument("--n", type=int, default=30, help="Number of val examples to compare")
    parser.add_argument("--slake-dir", default="/data3/ishaplan/slake_full/Slake1.0")
    parser.add_argument("--output", default="results/comparison.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load organ-filtered SLAKE val set (same as training).
    examples = load_multi_dataset(
        datasets=["slake"], splits=["train", "val"],
        image_cache_dir="data/images",
    )

    # Organ-only filter
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

    def has_organ(ex):
        text = ex.question.lower() + " " + ex.answer.lower()
        return any(o in text for o in all_organs)

    examples = [ex for ex in examples if has_organ(ex)]

    # Reproduce the 10/1 split
    np.random.RandomState(args.seed).shuffle(examples)
    n_val = max(1, len(examples) // 11)
    val_examples = examples[len(examples) - n_val:]

    # Subsample
    if args.n < len(val_examples):
        rng = np.random.RandomState(0)
        val_examples = list(rng.choice(val_examples, args.n, replace=False))

    print(f"Evaluating {len(val_examples)} val examples")

    # Load models sequentially (GPU memory)
    results = []

    print("\n=== Zero-shot ===")
    model_zs, proc = load_model(None)
    zs_correct = []
    for ex in tqdm(val_examples, desc="Zero-shot"):
        try:
            image = get_image(ex)
        except Exception:
            continue
        raw = generate(model_zs, proc, image, ex.question)
        extracted = extract_answer_tag(raw)
        c = compute_correctness(raw, ex.answer)
        zs_correct.append(c)
        results.append({"question": ex.question, "gt": ex.answer,
                        "zs_raw": raw, "zs_answer": extracted, "zs_correct": c})
    del model_zs
    torch.cuda.empty_cache()

    print(f"\n=== Correctness-only: {args.corr_ckpt} ===")
    model_co, proc = load_model(args.corr_ckpt)
    co_correct = []
    for i, ex in enumerate(tqdm(val_examples, desc="Corr-only")):
        try:
            image = get_image(ex)
        except Exception:
            continue
        raw = generate(model_co, proc, image, ex.question)
        extracted = extract_answer_tag(raw)
        c = compute_correctness(raw, ex.answer)
        co_correct.append(c)
        results[i]["co_raw"] = raw
        results[i]["co_answer"] = extracted
        results[i]["co_correct"] = c
    del model_co
    torch.cuda.empty_cache()

    print(f"\n=== Spatial GRPO (bbox probe): {args.spatial_ckpt} ===")
    model_sp, proc = load_model(args.spatial_ckpt)
    sp_correct = []
    for i, ex in enumerate(tqdm(val_examples, desc="Spatial-bbox")):
        try:
            image = get_image(ex)
        except Exception:
            continue
        raw = generate(model_sp, proc, image, ex.question)
        extracted = extract_answer_tag(raw)
        c = compute_correctness(raw, ex.answer)
        sp_correct.append(c)
        results[i]["sp_raw"] = raw
        results[i]["sp_answer"] = extracted
        results[i]["sp_correct"] = c
    del model_sp
    torch.cuda.empty_cache()

    # Optional: spatial with correctness-labeled probe
    spc_correct = []
    if args.spatial_corr_ckpt:
        print(f"\n=== Spatial GRPO (corr probe): {args.spatial_corr_ckpt} ===")
        model_spc, proc = load_model(args.spatial_corr_ckpt)
        for i, ex in enumerate(tqdm(val_examples, desc="Spatial-corr")):
            try:
                image = get_image(ex)
            except Exception:
                continue
            raw = generate(model_spc, proc, image, ex.question)
            extracted = extract_answer_tag(raw)
            c = compute_correctness(raw, ex.answer)
            spc_correct.append(c)
            results[i]["spc_raw"] = raw
            results[i]["spc_answer"] = extracted
            results[i]["spc_correct"] = c
        del model_spc
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    n = len(zs_correct)
    zs_acc = np.mean(zs_correct)
    co_acc = np.mean(co_correct)
    sp_acc = np.mean(sp_correct)

    print(f"{'Model':<30} {'Token F1':>10} {'Exact (>0.5)':>15}")
    print("-" * 57)
    print(f"{'Zero-shot':<30} {zs_acc:>10.3f} {sum(1 for c in zs_correct if c > 0.5):>10}/{n}")
    print(f"{'Correctness-only':<30} {co_acc:>10.3f} {sum(1 for c in co_correct if c > 0.5):>10}/{n}")
    print(f"{'Spatial (bbox probe)':<30} {sp_acc:>10.3f} {sum(1 for c in sp_correct if c > 0.5):>10}/{n}")
    if spc_correct:
        spc_acc = np.mean(spc_correct)
        print(f"{'Spatial (corr probe)':<30} {spc_acc:>10.3f} {sum(1 for c in spc_correct if c > 0.5):>10}/{n}")
    print()
    print(f"Spatial-bbox - Corr-only: {sp_acc - co_acc:+.3f} ({100*(sp_acc - co_acc)/max(co_acc,1e-6):+.1f}% relative)")
    if spc_correct:
        print(f"Spatial-corr - Corr-only: {spc_acc - co_acc:+.3f} ({100*(spc_acc - co_acc)/max(co_acc,1e-6):+.1f}% relative)")
        print(f"Spatial-corr - Spatial-bbox: {spc_acc - sp_acc:+.3f}")

    # Disagreements
    print("\n--- Disagreements (where models differ) ---")
    disagree = 0
    for r in results:
        zc = r.get("zs_correct", 0) > 0.5
        cc = r.get("co_correct", 0) > 0.5
        sc = r.get("sp_correct", 0) > 0.5
        scc = r.get("spc_correct", 0) > 0.5 if spc_correct else sc
        if zc != cc or cc != sc or sc != scc:
            disagree += 1
            print(f"\nQ: {r['question']}")
            print(f"GT: {r['gt']}")
            zs_e = r.get("zs_answer", "")
            co_e = r.get("co_answer", "")
            sp_e = r.get("sp_answer", "")
            print(f"  ZS: {'OK' if zc else 'WRONG'} — {zs_e}")
            print(f"  CO: {'OK' if cc else 'WRONG'} — {co_e}")
            print(f"  SP-bbox: {'OK' if sc else 'WRONG'} — {sp_e}")
            if spc_correct:
                spc_e = r.get("spc_answer", "")
                print(f"  SP-corr: {'OK' if scc else 'WRONG'} — {spc_e}")
    print(f"\n{disagree} disagreements out of {n} examples ({n - disagree} identical)")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n": n,
        "zs_f1": zs_acc, "co_f1": co_acc, "sp_bbox_f1": sp_acc,
        "delta_sp_bbox_co": sp_acc - co_acc,
    }
    if spc_correct:
        summary["sp_corr_f1"] = spc_acc
        summary["delta_sp_corr_co"] = spc_acc - co_acc
        summary["delta_sp_corr_bbox"] = spc_acc - sp_acc
    with open(out_path, "w") as f:
        json.dump({
            "summary": summary,
            "examples": results,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
