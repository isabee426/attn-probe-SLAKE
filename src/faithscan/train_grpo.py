"""GRPO training with DHCP cross-modal attention probe as faithfulness reward.

The DHCP probe measures how much the generated answer attends to image tokens.
Unlike HALP (which gives one score per question, not per rollout), the DHCP probe
scores each rollout individually since attention patterns change with each answer.

Usage:
    python -m faithscan.train_grpo --config configs/dhcp_grpo.yaml
"""

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from faithscan.data.dataset import (
    load_multi_dataset, load_vqarad, get_image, iter_examples_by_split,
)
from faithscan.models.dhcp_probe import (
    DHCPProbe, extract_cross_modal_attention,
)
from faithscan.reward import (
    compute_format_reward, compute_correctness, compute_composite_reward,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_policy_model(cfg: dict) -> tuple:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, get_peft_model

    model_name = cfg.get("name_or_path", "Qwen/Qwen2-VL-2B-Instruct")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
        cfg.get("dtype", "bfloat16"), torch.bfloat16,
    )
    device = cfg.get("device", "cuda")

    processor = AutoProcessor.from_pretrained(model_name)

    # Support Qwen3-VL (different model class) and Qwen2-VL.
    if "Qwen3" in model_name:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
        ).to(device)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
        ).to(device)

    use_lora = cfg.get("use_lora", True)
    if use_lora:
        lora_cfg = cfg.get("lora", {})
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
            ]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        for p in model.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Full fine-tuning: {trainable:,} parameters")

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    return model, processor, device, dtype


def load_dhcp_probe(cfg: dict, device: str) -> DHCPProbe | None:
    """Load pretrained DHCP probe (frozen)."""
    ckpt_path = cfg.get("dhcp_checkpoint")
    if not ckpt_path:
        return None

    dhcp_cfg = cfg.get("dhcp", {})
    target_layers = dhcp_cfg.get("target_layers", list(range(28)))
    n_heads = dhcp_cfg.get("n_heads", 12)

    probe = DHCPProbe(
        n_layers=len(target_layers),
        n_heads=n_heads,
        hidden_dim=dhcp_cfg.get("hidden_dim", 128),
        dropout=0.0,
        vision_pool=dhcp_cfg.get("vision_pool", 1),
    ).to(device)

    probe.load_state_dict(torch.load(ckpt_path, map_location=device))
    probe.eval()
    for p in probe.parameters():
        p.requires_grad = False

    logger.info(f"Loaded DHCP probe from {ckpt_path}")
    return probe


# ---------------------------------------------------------------------------
# Rollout with attention extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout_one(
    model, processor, image: Image.Image, question: str,
    num_responses: int, max_new_tokens: int, temperature: float,
    device: str,
    dhcp_probe: DHCPProbe | None = None,
    target_layers: list[int] | None = None, n_heads: int = 12,
    use_lookback_lens: bool = False,
    spatial_classifier=None, target_bbox: list | None = None,
    lookback_classifier=None,
    use_temporal_bbox: bool = False,
    use_answer_token_faith: bool = False,
    use_think_phase_faith: bool = False,
) -> list[dict]:
    """Sample responses and extract faithfulness features per rollout.

    Supports two faithfulness modes:
    - DHCP probe: teacher-forcing attention extraction (extra forward pass)
    - Lookback lens: vision attention ratios from generate() (no extra cost)
    """
    if target_layers is None:
        target_layers = list(range(28))
    from qwen_vl_utils import process_vision_info

    # Prompt must match across rollout, log-prob re-encoding, and validation.
    # Qwen3-VL-Thinking enters <think> mode natively via chat template.
    # "brief" prompt: shortest outputs (166 avg), 90% extraction at 512 tokens.
    prompt_text = f"Answer this medical question. Think step by step, then give your final answer in <answer>...</answer> tags.\nQuestion: {question}"
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    prefill_len = inputs["input_ids"].shape[1]

    # Find vision token boundaries for lookback lens.
    vision_start = vision_end = 0
    if use_lookback_lens:
        input_ids = inputs["input_ids"][0]
        tokenizer = processor.tokenizer
        vs_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        ve_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        ids = input_ids.tolist()
        for i, tid in enumerate(ids):
            if tid == vs_id and vision_start == 0:
                vision_start = i + 1
            if tid == ve_id:
                vision_end = i

        # Switch to eager attention for generate (needed for output_attentions).
        old_impl = getattr(model.config, '_attn_implementation', 'sdpa')
        model.config._attn_implementation = "eager"
        if hasattr(model.config, 'text_config'):
            old_text_impl = getattr(model.config.text_config, '_attn_implementation', 'sdpa')
            model.config.text_config._attn_implementation = "eager"

    # ------------------------------------------------------------------
    # Batched generate: all G responses in one forward pass.
    # Much faster than G sequential calls — prefill only happens once.
    # ------------------------------------------------------------------
    # Expand inputs to batch size G.
    # pixel_values is (n_patches, embed_dim) — not batch-indexed, repeat as-is.
    # input_ids / attention_mask are (1, seq_len) — expand along batch dim.
    batched_inputs = {}
    for k, v in inputs.items():
        if v.shape[0] == 1:
            batched_inputs[k] = v.expand(num_responses, *v.shape[1:]).clone()
        else:
            # pixel_values and similar: repeat G times along first dim.
            batched_inputs[k] = v.repeat(num_responses, *([1] * (v.dim() - 1)))

    generate_kwargs = dict(
        **batched_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        return_dict_in_generate=True,
        output_scores=True,
    )
    if use_lookback_lens:
        generate_kwargs["output_attentions"] = True

    gen_out = model.generate(**generate_kwargs)
    # gen_out.sequences: (G, prefill_len + max_new_tokens)
    # gen_out.scores: tuple of (G, vocab) per generated token
    # gen_out.attentions (if requested): tuple[token] of tuple[layer] of (G, heads, 1, seq)

    # Pre-compute faith geometry once (shared across all rollouts).
    if use_lookback_lens and target_bbox is not None:
        import math as _math
        n_vis_total = vision_end - vision_start
        grid_side = int(_math.sqrt(n_vis_total)) if n_vis_total > 0 else 1
        img_w, img_h = image.size
        eff_patch = max(img_w, img_h) / grid_side if grid_side > 0 else 14
        if use_temporal_bbox:
            from faithscan.models.lookback_lens import compute_temporal_bbox_faith
        else:
            from faithscan.models.lookback_lens import (
                compute_per_vision_token_attention, compute_spatial_focus,
            )
    elif use_lookback_lens:
        from faithscan.models.lookback_lens import (
            compute_vision_lookback_ratio, lookback_ratio_to_features,
        )
    if use_answer_token_faith or use_think_phase_faith:
        from faithscan.models.lookback_lens import (
            compute_answer_token_vision_faith,
            compute_think_vs_answer_vision_faith,
        )
        tokenizer = processor.tokenizer
        # <answer> → ['<', 'answer', '>'] — use 'answer' token ID (9217)
        _ids = tokenizer.encode("answer", add_special_tokens=False)
        _answer_tag_id = _ids[0] if _ids else 9217
        # </think> is a single special token (151668)
        _ids = tokenizer.encode("</think>", add_special_tokens=False)
        _think_end_id = _ids[0] if _ids else 151668

    rollouts = []
    for b in range(num_responses):
        gen_ids = gen_out.sequences[b, prefill_len:]
        # Strip padding (eos tokens after generation ends).
        eos_id = processor.tokenizer.eos_token_id
        if eos_id is not None:
            eos_positions = (gen_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                gen_ids = gen_ids[:eos_positions[0] + 1]

        answer_text = processor.tokenizer.decode(
            gen_ids, skip_special_tokens=False,
        ).strip()

        # Per-token log probs from batched scores.
        old_log_probs = []
        for step, logits in enumerate(gen_out.scores):
            if step >= len(gen_ids):
                break
            probs = torch.softmax(logits[b].float(), dim=-1)
            log_p = torch.log(probs + 1e-10)
            token_id = gen_ids[step].item()
            old_log_probs.append(log_p[token_id].item())

        # Faithfulness scoring — extract per-sequence attentions using index b.
        faith = None
        num_new = len(gen_ids)

        if use_lookback_lens and gen_out.attentions is not None and num_new > 0:
            # Slice attentions for sequence b: attentions[token][layer][b] → (heads,1,seq)
            attn_b = tuple(
                tuple(layer_attn[b:b+1] for layer_attn in step_attn)
                for step_attn in gen_out.attentions[:num_new]
            )

            if target_bbox is not None:
                if use_temporal_bbox:
                    faith = compute_temporal_bbox_faith(
                        attn_b, vision_start, vision_end,
                        target_bbox, (img_w, img_h), int(eff_patch), num_new,
                    )
                    if faith is None:
                        faith = 0.5
                else:
                    spatial_attn = compute_per_vision_token_attention(
                        attn_b, vision_start, vision_end, num_new,
                    )
                    spatial_feat = compute_spatial_focus(
                        spatial_attn, target_bbox, (img_w, img_h),
                        patch_size=int(eff_patch),
                    )
                    if spatial_feat is not None:
                        if spatial_classifier is not None:
                            faith = spatial_classifier.predict_proba(
                                spatial_feat.reshape(1, -1)
                            )[0, 1]
                        else:
                            faith = float(spatial_feat.mean())
                    else:
                        faith = 0.5
            elif lookback_classifier is not None:
                lb_ratio = compute_vision_lookback_ratio(
                    attn_b, vision_start, vision_end, num_new,
                )
                feat = lookback_ratio_to_features(lb_ratio, mode="mean")
                faith = lookback_classifier.predict_proba(
                    feat.numpy().reshape(1, -1)
                )[0, 1]
            else:
                lb_ratio = compute_vision_lookback_ratio(
                    attn_b, vision_start, vision_end, num_new,
                )
                feat = lookback_ratio_to_features(lb_ratio, mode="mean")
                faith = feat.mean().item()

        elif use_answer_token_faith and gen_out.attentions is not None and num_new > 0:
            attn_b = tuple(
                tuple(layer_attn[b:b+1] for layer_attn in step_attn)
                for step_attn in gen_out.attentions[:num_new]
            )
            faith = compute_answer_token_vision_faith(
                attn_b, vision_start, vision_end,
                gen_ids.tolist(), _answer_tag_id,
            )
        elif use_think_phase_faith and gen_out.attentions is not None and num_new > 0:
            attn_b = tuple(
                tuple(layer_attn[b:b+1] for layer_attn in step_attn)
                for step_attn in gen_out.attentions[:num_new]
            )
            faith = compute_think_vs_answer_vision_faith(
                attn_b, vision_start, vision_end,
                gen_ids.tolist(), _think_end_id,
            )
        elif dhcp_probe is not None:
            dhcp_feat = extract_cross_modal_attention(
                model, processor, image, question, answer_text,
                device, target_layers, n_heads,
            )
            halluc_prob = dhcp_probe(dhcp_feat.unsqueeze(0)).item()
            faith = 1.0 - halluc_prob

        rollouts.append({
            "gen_ids": gen_ids.cpu(),
            "answer_text": answer_text,
            "old_log_probs": old_log_probs,
            "dhcp_faith": faith,
        })

    # Restore attention implementation.
    if use_lookback_lens:
        model.config._attn_implementation = old_impl
        if hasattr(model.config, 'text_config'):
            model.config.text_config._attn_implementation = old_text_impl

    return rollouts


# ---------------------------------------------------------------------------
# Policy loss
# ---------------------------------------------------------------------------

def compute_new_log_probs(
    model, processor, image, question, gen_ids, device,
) -> torch.Tensor:
    from qwen_vl_utils import process_vision_info

    answer_text = processor.tokenizer.decode(
        gen_ids, skip_special_tokens=False,
    ).strip()

    # Must use the same prompt as rollout generation.
    prompt_text = f"Answer this medical question. Think step by step, then give your final answer in <answer>...</answer> tags.\nQuestion: {question}"
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]},
        {"role": "assistant", "content": answer_text},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    outputs = model(**dict(inputs))
    logits = outputs.logits

    tokenizer = processor.tokenizer
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    input_ids = inputs["input_ids"][0]
    ids_list = input_ids.tolist()

    asst_start = 0
    for i in range(len(ids_list) - 1, -1, -1):
        if ids_list[i] == im_start_id:
            asst_start = i
            break

    answer_logits = logits[0, asst_start - 1 : -1, :]
    answer_targets = input_ids[asst_start:]
    num_answer_tokens = len(answer_targets)

    log_probs_all = F.log_softmax(answer_logits.float(), dim=-1)
    token_log_probs = log_probs_all.gather(
        1, answer_targets.unsqueeze(1),
    ).squeeze(1)

    answer_prefix = tokenizer.encode("Answer: ", add_special_tokens=False)
    prefix_len = len(answer_prefix)
    offset = 0
    answer_ids_list = answer_targets.tolist()
    for i in range(len(answer_ids_list) - prefix_len + 1):
        if answer_ids_list[i : i + prefix_len] == answer_prefix:
            offset = i + prefix_len
            break
    if offset == 0:
        offset = min(5, num_answer_tokens - 1)

    gen_len = len(gen_ids)
    end = min(offset + gen_len, num_answer_tokens)
    aligned = token_log_probs[offset:end]

    if len(aligned) < gen_len:
        aligned = torch.cat([aligned, torch.zeros(gen_len - len(aligned), device=aligned.device)])
    elif len(aligned) > gen_len:
        aligned = aligned[:gen_len]

    return aligned


def dr_grpo_loss(
    new_log_probs, old_log_probs, advantage, clip_eps, kl_beta=0.0,
) -> torch.Tensor:
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    policy_loss = -torch.min(surr1, surr2).sum()

    if kl_beta > 0:
        kl = (ratio - 1) - (new_log_probs - old_log_probs)
        policy_loss = policy_loss + kl_beta * kl.sum()

    return policy_loss


# ---------------------------------------------------------------------------
# EBPO baseline
# ---------------------------------------------------------------------------

class EBPOBaseline:
    def __init__(self):
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, rewards):
        for r in rewards:
            self._n += 1
            delta = r - self._mean
            self._mean += delta / self._n
            self._m2 += delta * (r - self._mean)

    @property
    def global_mean(self):
        return self._mean

    @property
    def global_var(self):
        return self._m2 / max(self._n, 1)

    def shrink(self, rewards):
        if self._n < 2:
            return float(np.mean(rewards))
        group_mean = float(np.mean(rewards))
        group_var = float(np.var(rewards))
        S = group_var / (group_var + self.global_var + 1e-8)
        return (1 - S) * group_mean + S * self.global_mean


class FaithNormalizer:
    """Running z-score normalization for faith scores.

    Tracks global mean/std with Welford's algorithm.
    Normalizes via z-score then sigmoid to [0, 1].
    """
    def __init__(self):
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, scores):
        for s in scores:
            self._n += 1
            delta = s - self._mean
            self._mean += delta / self._n
            self._m2 += delta * (s - self._mean)

    @property
    def std(self):
        if self._n < 2:
            return 1.0
        return max((self._m2 / self._n) ** 0.5, 1e-8)

    def normalize(self, scores):
        """Z-score then sigmoid to [0, 1]."""
        if self._n < 10:
            # Not enough data yet — fall back to within-group min-max.
            f_min, f_max = min(scores), max(scores)
            f_range = f_max - f_min
            if f_range > 1e-8:
                return [(s - f_min) / f_range for s in scores]
            return [0.5] * len(scores)

        normed = []
        for s in scores:
            z = (s - self._mean) / self.std
            # Sigmoid: maps z-score to (0, 1). z=0 → 0.5, z=+2 → 0.88, z=-2 → 0.12
            normed.append(1.0 / (1.0 + math.exp(-z)))
        return normed


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model, processor, dhcp_probe,
    examples, device, alpha, max_new_tokens,
    target_layers=None, n_heads=12,
    spatial_classifier=None, slake_bboxes=None, example_bboxes=None,
    max_image_size: int = 0,
    val_max_examples: int = 0,
    use_lookback_lens: bool = False,
    use_temporal_bbox: bool = False,
    lookback_classifier=None,
    use_answer_token_faith: bool = False,
    use_think_phase_faith: bool = False,
    faith_normalizer: "FaithNormalizer | None" = None,
) -> tuple[float, float, float]:
    """Greedy-decode on val, return (mean_reward, mean_correct, mean_faith).

    Spatial faith is always measured when spatial_classifier is provided,
    regardless of alpha. This lets correctness-only runs track grounding.
    """
    if target_layers is None:
        target_layers = list(range(28))
    from qwen_vl_utils import process_vision_info

    if val_max_examples > 0 and len(examples) > val_max_examples:
        rng = np.random.RandomState(0)
        examples = list(rng.choice(examples, val_max_examples, replace=False))

    use_spatial = spatial_classifier is not None and (slake_bboxes or example_bboxes)
    needs_attentions = use_spatial or use_lookback_lens or use_answer_token_faith or use_think_phase_faith
    if needs_attentions:
        old_impl = getattr(model.config, '_attn_implementation', 'sdpa')
        model.config._attn_implementation = "eager"
        if hasattr(model.config, 'text_config'):
            old_text_impl = getattr(model.config.text_config, '_attn_implementation', 'sdpa')
            model.config.text_config._attn_implementation = "eager"
    if use_spatial:
        import math as _math
        from faithscan.models.lookback_lens import (
            compute_per_vision_token_attention, compute_spatial_focus,
        )
    if use_lookback_lens:
        import math as _math
        if use_temporal_bbox:
            from faithscan.models.lookback_lens import compute_temporal_bbox_faith
        else:
            from faithscan.models.lookback_lens import (
                compute_vision_lookback_ratio, lookback_ratio_to_features,
            )
    if use_answer_token_faith or use_think_phase_faith:
        from faithscan.models.lookback_lens import (
            compute_answer_token_vision_faith,
            compute_think_vs_answer_vision_faith,
        )
        _tokenizer = processor.tokenizer
        _ids = _tokenizer.encode("answer", add_special_tokens=False)
        _answer_tag_id = _ids[0] if _ids else 9217
        _ids = _tokenizer.encode("</think>", add_special_tokens=False)
        _think_end_id = _ids[0] if _ids else 151668

    model.eval()
    rewards, all_correct, all_faith = [], [], []

    for ex in tqdm(examples, desc="Validating", leave=False):
        try:
            image = get_image(ex)
        except Exception:
            continue

        # Resize large images.
        if max_image_size and max(image.size) > max_image_size:
            img_scale = max_image_size / max(image.size)
            image = image.resize((int(image.size[0]*img_scale), int(image.size[1]*img_scale)))
        else:
            img_scale = 1.0

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"Answer this medical question. Think step by step, then give your final answer in <answer>...</answer> tags.\n"
                    f"Question: {ex.question}"
                )},
            ]},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)

        prefill_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
        )
        if needs_attentions:
            gen_kwargs["output_attentions"] = True

        gen_out = model.generate(**gen_kwargs)
        gen_ids = gen_out.sequences[0, prefill_len:]
        answer = processor.tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

        correctness = compute_correctness(answer, ex.answer)

        # Measure faithfulness via spatial probe, DHCP, or fallback.
        faithfulness = 0.5
        if use_spatial and gen_out.attentions is not None:
            ex_bbox = None
            if example_bboxes:
                ex_bbox = example_bboxes.get(ex.id)
                if ex_bbox is not None and img_scale != 1.0:
                    bx, by, bw, bh = ex_bbox
                    ex_bbox = [bx*img_scale, by*img_scale, bw*img_scale, bh*img_scale]
            elif slake_bboxes:
                q_lower = ex.question.lower()
                a_lower = ex.answer.lower() if hasattr(ex, 'answer') else ''
                for organs in slake_bboxes.values():
                    for organ, bbox in organs.items():
                        if organ in q_lower or organ in a_lower:
                            ex_bbox = bbox
                            break
                    if ex_bbox:
                        break
            if ex_bbox is not None:
                tokenizer = processor.tokenizer
                vs_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
                ve_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
                ids_list = inputs["input_ids"][0].tolist()
                vision_start = vision_end = 0
                for i, tid in enumerate(ids_list):
                    if tid == vs_id and vision_start == 0:
                        vision_start = i + 1
                    if tid == ve_id:
                        vision_end = i
                num_new = len(gen_ids)
                spatial_attn = compute_per_vision_token_attention(
                    gen_out.attentions, vision_start, vision_end, num_new,
                )
                n_vis = spatial_attn.shape[2]
                grid_side = int(_math.sqrt(n_vis))
                img_w, img_h = image.size
                eff_patch = max(img_w, img_h) / grid_side if grid_side > 0 else 14
                spatial_feat = compute_spatial_focus(
                    spatial_attn, ex_bbox, (img_w, img_h),
                    patch_size=int(eff_patch),
                )
                if spatial_feat is not None:
                    if spatial_classifier is not None:
                        if hasattr(spatial_classifier, "predict_proba"):
                            faithfulness = spatial_classifier.predict_proba(
                                spatial_feat.reshape(1, -1)
                            )[0, 1]
                        else:
                            faithfulness = float(np.clip(
                                spatial_classifier.predict(spatial_feat.reshape(1, -1))[0], 0, 1
                            ))
                    else:
                        # Raw mean bbox attention ratio (ablation).
                        faithfulness = float(spatial_feat.mean())
        elif use_lookback_lens and gen_out.attentions is not None:
            tokenizer = processor.tokenizer
            vs_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            ve_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
            ids_list = inputs["input_ids"][0].tolist()
            vision_start = vision_end = 0
            for i, tid in enumerate(ids_list):
                if tid == vs_id and vision_start == 0:
                    vision_start = i + 1
                if tid == ve_id:
                    vision_end = i
            num_new = len(gen_ids)
            if use_temporal_bbox:
                ex_bbox = example_bboxes.get(ex.id) if example_bboxes else None
                if ex_bbox is not None:
                    if img_scale != 1.0:
                        bx, by, bw, bh = ex_bbox
                        ex_bbox = [bx*img_scale, by*img_scale, bw*img_scale, bh*img_scale]
                    img_w, img_h = image.size
                    n_vis_total = vision_end - vision_start
                    grid_side = int(_math.sqrt(n_vis_total)) if n_vis_total > 0 else 1
                    eff_patch = max(img_w, img_h) / grid_side if grid_side > 0 else 14
                    f = compute_temporal_bbox_faith(
                        gen_out.attentions, vision_start, vision_end,
                        ex_bbox, (img_w, img_h), int(eff_patch), num_new,
                    )
                    if f is not None:
                        faithfulness = f
            else:
                lb_ratio = compute_vision_lookback_ratio(
                    gen_out.attentions, vision_start, vision_end, num_new,
                )
                feat = lookback_ratio_to_features(lb_ratio, mode="mean")
                if lookback_classifier is not None:
                    faithfulness = lookback_classifier.predict_proba(
                        feat.numpy().reshape(1, -1)
                    )[0, 1]
                else:
                    faithfulness = feat.mean().item()
        elif use_answer_token_faith and gen_out.attentions is not None:
            tokenizer = processor.tokenizer
            vs_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            ve_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
            ids_list = inputs["input_ids"][0].tolist()
            vision_start = vision_end = 0
            for i, tid in enumerate(ids_list):
                if tid == vs_id and vision_start == 0:
                    vision_start = i + 1
                if tid == ve_id:
                    vision_end = i
            faithfulness = compute_answer_token_vision_faith(
                gen_out.attentions, vision_start, vision_end,
                gen_ids.tolist(), _answer_tag_id,
            )
        elif use_think_phase_faith and gen_out.attentions is not None:
            tokenizer = processor.tokenizer
            vs_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
            ve_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
            ids_list = inputs["input_ids"][0].tolist()
            vision_start = vision_end = 0
            for i, tid in enumerate(ids_list):
                if tid == vs_id and vision_start == 0:
                    vision_start = i + 1
                if tid == ve_id:
                    vision_end = i
            faithfulness = compute_think_vs_answer_vision_faith(
                gen_out.attentions, vision_start, vision_end,
                gen_ids.tolist(), _think_end_id,
            )
        elif dhcp_probe is not None:
            dhcp_feat = extract_cross_modal_attention(
                model, processor, image, ex.question, answer,
                device, target_layers, n_heads,
            )
            halluc_prob = dhcp_probe(dhcp_feat.unsqueeze(0)).item()
            faithfulness = 1.0 - halluc_prob

        # Normalize faith the same way as training (z-score + sigmoid).
        if faith_normalizer is not None:
            faithfulness = faith_normalizer.normalize([faithfulness])[0]

        reward = compute_composite_reward(correctness, faithfulness, alpha)
        rewards.append(reward)
        all_correct.append(correctness)
        all_faith.append(faithfulness)

    if needs_attentions:
        model.config._attn_implementation = old_impl
        if hasattr(model.config, 'text_config'):
            model.config.text_config._attn_implementation = old_text_impl

    mean_r = float(np.mean(rewards)) if rewards else 0.0
    mean_c = float(np.mean(all_correct)) if all_correct else 0.0
    mean_f = float(np.mean(all_faith)) if all_faith else 0.0

    logger.info(f"  Val correct: {mean_c:.4f}, faith: {mean_f:.4f}")
    return mean_r, mean_c, mean_f


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: dict):
    torch.manual_seed(cfg.get("seed", 42))

    model_cfg = cfg["model"]
    model, processor, device, dtype = load_policy_model(model_cfg)
    dhcp_probe = load_dhcp_probe(cfg, device)

    dhcp_cfg = cfg.get("dhcp", {})
    target_layers = dhcp_cfg.get("target_layers", list(range(28)))
    n_heads = dhcp_cfg.get("n_heads", 12)
    use_lookback_lens = cfg.get("use_lookback_lens", False)
    use_temporal_bbox = cfg.get("use_temporal_bbox", False)
    use_answer_token_faith = cfg.get("use_answer_token_faith", False)
    use_think_phase_faith = cfg.get("use_think_phase_faith", False)
    if use_answer_token_faith:
        logger.info("Using answer-token vision attention as faith signal")
    if use_think_phase_faith:
        logger.info("Using think-phase vs answer-phase vision attention as faith signal")
    lookback_classifier = None
    if use_lookback_lens:
        if use_temporal_bbox:
            logger.info("Using temporal bbox faith (per-token bbox ratio across trajectory)")
        else:
            logger.info("Using Lookback Lens for per-rollout faithfulness")
        lb_clf_path = cfg.get("lookback_classifier")
        if lb_clf_path:
            import pickle
            with open(lb_clf_path, "rb") as f:
                lookback_classifier = pickle.load(f)
            logger.info(f"Loaded lookback classifier from {lb_clf_path}")
        elif not use_temporal_bbox:
            logger.info("No lookback classifier — using raw mean ratio")

    # Spatial grounding classifier
    spatial_classifier = None
    spatial_clf_path = cfg.get("spatial_classifier")
    if spatial_clf_path:
        import pickle
        with open(spatial_clf_path, "rb") as f:
            spatial_classifier = pickle.load(f)
        logger.info(f"Loaded spatial classifier from {spatial_clf_path}")

    # Load SLAKE bounding boxes for spatial grounding
    slake_bboxes = {}
    slake_dir = cfg.get("slake_dir")
    data_cfg = cfg.get("data", {})
    if slake_dir and (spatial_classifier is not None or data_cfg.get("organ_only")):
        import json
        slake_path = Path(slake_dir)
        for img_dir in (slake_path / "imgs").iterdir():
            det_path = img_dir / "detection.json"
            if det_path.exists():
                with open(det_path) as f:
                    dets = json.load(f)
                organ_bboxes = {}
                for det in dets:
                    for organ, bbox in det.items():
                        organ_bboxes[organ.lower()] = bbox
                slake_bboxes[img_dir.name] = organ_bboxes
        logger.info(f"Loaded bboxes for {len(slake_bboxes)} SLAKE images")

    # Load data.
    # example_bboxes: per-example bbox dict (VinDr style, keyed by example.id).
    # Used instead of slake_bboxes keyword matching when available.
    example_bboxes = {}
    dataset_name = data_cfg.get("dataset")

    if dataset_name == "vindrcxr":
        from faithscan.data.dataset import load_vindrcxr
        examples, example_bboxes = load_vindrcxr(
            image_dir=data_cfg.get("image_dir", "data/vindr_png"),
            splits=["train", "val"],
            val_ratio=1.0 / (data_cfg.get("train_val_ratio", 10) + 1),
            seed=cfg.get("seed", 42),
        )
        logger.info(f"Loaded VinDr: {len(examples)} examples, "
                     f"{len(example_bboxes)} with bboxes")
    else:
        dataset_list = data_cfg.get("datasets")
        if dataset_list:
            from faithscan.data.dataset import load_multi_dataset
            examples = load_multi_dataset(
                datasets=dataset_list, splits=["train", "val"],
                image_cache_dir=data_cfg.get("image_cache_dir", "data/images"),
            )
        else:
            examples = load_vqarad(
                splits=["train", "val"],
                image_cache_dir=data_cfg.get("image_cache_dir", "data/images"),
            )

    # Filter to organ-specific questions only (ones that mention an organ
    # we have a bbox for). Collect all organ names across all SLAKE images.
    if data_cfg.get("organ_only") and slake_bboxes:
        all_organs = set()
        for organs in slake_bboxes.values():
            all_organs.update(organs.keys())
        logger.info(f"Organ vocabulary: {sorted(all_organs)}")

        def has_organ_mention(ex):
            q_lower = ex.question.lower()
            a_lower = ex.answer.lower() if hasattr(ex, 'answer') else ''
            text = q_lower + ' ' + a_lower
            return any(organ in text for organ in all_organs)

        before = len(examples)
        examples = [ex for ex in examples if has_organ_mention(ex)]
        logger.info(f"Organ-only filter: {before} → {len(examples)} examples "
                     f"({100*len(examples)/before:.0f}%)")

    # Custom train/val split ratio (e.g. 10/1). If set, pool all examples
    # and re-split instead of using the dataset's native splits.
    # Skip for VinDr — it handles splitting internally by image_id.
    split_ratio = data_cfg.get("train_val_ratio")
    if split_ratio and dataset_name != "vindrcxr":
        all_examples = list(examples)
        np.random.RandomState(cfg.get("seed", 42)).shuffle(all_examples)
        n_val = max(1, len(all_examples) // (split_ratio + 1))
        n_train = len(all_examples) - n_val
        train_examples = all_examples[:n_train]
        val_examples = all_examples[n_train:]
        # Fix split labels.
        for ex in train_examples:
            ex.split = "train"
        for ex in val_examples:
            ex.split = "val"
    else:
        train_examples = list(iter_examples_by_split(examples, "train"))
        val_examples = list(iter_examples_by_split(examples, "val"))
    logger.info(f"Train: {len(train_examples)}, Val: {len(val_examples)}")

    # Training config.
    max_image_size = data_cfg.get("max_image_size", 0)

    grpo_cfg = cfg.get("grpo", {})
    G = grpo_cfg.get("num_rollouts", 8)
    alpha = grpo_cfg.get("reward_alpha", 0.7)
    clip_eps = grpo_cfg.get("clip_eps", 0.2)
    kl_beta = grpo_cfg.get("kl_beta", 0.05)
    temperature = grpo_cfg.get("temperature", 1.0)
    max_new_tokens = grpo_cfg.get("max_new_tokens", 64)
    epochs = grpo_cfg.get("epochs", 1)
    grad_accum_steps = grpo_cfg.get("grad_accum_steps", 8)
    log_every = grpo_cfg.get("log_every", 10)
    use_ebpo = grpo_cfg.get("ebpo_shrinkage", True)
    drop_unformatted = grpo_cfg.get("drop_unformatted", False)
    val_max_examples = grpo_cfg.get("val_max_examples", 0)

    train_cfg = cfg.get("training", {})
    base_lr = train_cfg.get("learning_rate", 1e-5)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=base_lr, weight_decay=train_cfg.get("weight_decay", 0.01),
        betas=(0.9, 0.95),
    )

    total_steps = (len(train_examples) // grad_accum_steps) * epochs
    use_cosine_lr = train_cfg.get("cosine_lr", True)
    save_every = train_cfg.get("save_every_steps", 10)

    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints/dhcp_grpo/"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ebpo = EBPOBaseline() if use_ebpo else None
    faith_norm = FaithNormalizer()
    global_step = 0
    best_val_reward = -float("inf")
    best_val_correct = 0.0

    for epoch in range(epochs):
        np.random.shuffle(train_examples)
        epoch_rewards, epoch_correct, epoch_faithful = [], [], []

        pbar = tqdm(train_examples, desc=f"Epoch {epoch + 1}/{epochs}")
        optimizer.zero_grad()
        q_idx = None

        for q_idx, ex in enumerate(pbar):
            try:
                image = get_image(ex)
            except Exception:
                continue

            # Resize large images (VinDr CXRs are ~2500×3000, causes OOM).
            if max_image_size and max(image.size) > max_image_size:
                scale = max_image_size / max(image.size)
                image = image.resize((int(image.size[0]*scale), int(image.size[1]*scale)))
            else:
                scale = 1.0

            # 1. Rollout with per-rollout faithfulness scoring.
            # Find target bbox if spatial classifier or temporal bbox is active.
            ex_bbox = None
            if spatial_classifier is not None or use_temporal_bbox:
                # Per-example bbox dict (VinDr style) — simple ID lookup.
                if example_bboxes:
                    ex_bbox = example_bboxes.get(ex.id)
                    # Scale bbox if image was resized.
                    if ex_bbox is not None and scale != 1.0:
                        bx, by, bw, bh = ex_bbox
                        ex_bbox = [bx*scale, by*scale, bw*scale, bh*scale]
                # SLAKE fallback — keyword matching.
                elif slake_bboxes:
                    img_dir = ''
                    if hasattr(ex, 'image_path') and ex.image_path:
                        parts = ex.image_path.replace('\\', '/').split('/')
                        for j, p in enumerate(parts):
                            if p.startswith('xmlab'):
                                img_dir = p
                                break
                    if img_dir and img_dir in slake_bboxes:
                        organs = slake_bboxes[img_dir]
                        q_lower = ex.question.lower()
                        a_lower = ex.answer.lower() if hasattr(ex, 'answer') else ''
                        for organ, bbox in organs.items():
                            if organ in q_lower or organ in a_lower:
                                ex_bbox = bbox
                                break

            model.eval()
            rollouts = rollout_one(
                model, processor, image, ex.question,
                num_responses=G, max_new_tokens=max_new_tokens,
                temperature=temperature, device=device,
                dhcp_probe=dhcp_probe,
                target_layers=target_layers, n_heads=n_heads,
                use_lookback_lens=use_lookback_lens,
                spatial_classifier=spatial_classifier, target_bbox=ex_bbox,
                lookback_classifier=lookback_classifier,
                use_temporal_bbox=use_temporal_bbox,
                use_answer_token_faith=use_answer_token_faith,
                use_think_phase_faith=use_think_phase_faith,
            )

            # 2. Compute rewards.
            # First pass: collect raw faithfulness scores.
            # If no spatial bbox matched, use correctness-only (alpha=1.0).
            effective_alpha = alpha
            if (spatial_classifier is not None or use_temporal_bbox) and ex_bbox is None:
                effective_alpha = 1.0  # no spatial signal → correctness only

            raw_faiths = []
            for ro in rollouts:
                f = ro["dhcp_faith"] if ro["dhcp_faith"] is not None else 0.5
                raw_faiths.append(f)

            # Normalize faith using global running stats (z-score + sigmoid).
            faith_norm.update(raw_faiths)
            norm_faiths = faith_norm.normalize(raw_faiths)

            rewards = []
            for ro, norm_f in zip(rollouts, norm_faiths):
                fmt = compute_format_reward(ro["answer_text"])
                correctness = compute_correctness(ro["answer_text"], ex.answer)

                reward = compute_composite_reward(
                    correctness, norm_f, effective_alpha, format_reward=fmt,
                )
                ro["reward"] = reward
                ro["correctness"] = correctness
                ro["faithfulness"] = norm_f
                ro["formatted"] = fmt > 0.0
                rewards.append(reward)

            torch.cuda.empty_cache()

            # 3. Advantages.
            # When drop_unformatted=True, exclude unformatted rollouts from
            # baseline and gradient entirely. Formatted rollouts compete only
            # against each other — stronger format pressure than reward=0.
            if drop_unformatted:
                fmt_rewards = [r for r, ro in zip(rewards, rollouts) if ro["formatted"]]
                if not fmt_rewards:
                    continue  # all unformatted — skip example
                if ebpo is not None:
                    ebpo.update(fmt_rewards)
                    baseline = ebpo.shrink(fmt_rewards)
                else:
                    baseline = float(np.mean(fmt_rewards))
                advantages = [
                    (r - baseline) if ro["formatted"] else 0.0
                    for r, ro in zip(rewards, rollouts)
                ]
            else:
                if ebpo is not None:
                    ebpo.update(rewards)
                    baseline = ebpo.shrink(rewards)
                else:
                    baseline = float(np.mean(rewards))
                advantages = [r - baseline for r in rewards]

            if all(abs(a) < 1e-8 for a in advantages):
                continue

            # 4. Policy gradient.
            model.train()
            for ro, adv in zip(rollouts, advantages):
                if abs(adv) < 1e-8:
                    continue

                new_lp = compute_new_log_probs(
                    model, processor, image, ex.question,
                    ro["gen_ids"].to(device), device,
                )
                old_lp = torch.tensor(
                    ro["old_log_probs"], device=device, dtype=torch.float32,
                )
                min_len = min(len(new_lp), len(old_lp))
                loss = dr_grpo_loss(
                    new_lp[:min_len], old_lp[:min_len], adv, clip_eps, kl_beta,
                )
                (loss / (G * grad_accum_steps)).backward()
                torch.cuda.empty_cache()

            if (q_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0,
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if use_cosine_lr and total_steps > 0:
                    lr = base_lr * 0.5 * (1 + math.cos(math.pi * global_step / total_steps))
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                if global_step % save_every == 0:
                    torch.cuda.empty_cache()
                    val_r, val_c, val_f = evaluate(
                        model, processor, dhcp_probe,
                        val_examples, device, alpha, max_new_tokens,
                        target_layers, n_heads,
                        spatial_classifier=spatial_classifier,
                        slake_bboxes=slake_bboxes,
                        example_bboxes=example_bboxes,
                        max_image_size=max_image_size,
                        val_max_examples=val_max_examples,
                        use_lookback_lens=use_lookback_lens,
                        use_temporal_bbox=use_temporal_bbox,
                        lookback_classifier=lookback_classifier,
                        use_answer_token_faith=use_answer_token_faith,
                        use_think_phase_faith=use_think_phase_faith,
                        faith_normalizer=faith_norm,
                    )
                    logger.info(
                        f"Step {global_step} val reward: {val_r:.4f} "
                        f"(correct: {val_c:.4f}, faith: {val_f:.4f}, "
                        f"best: {best_val_reward:.4f})"
                    )
                    if val_r > best_val_reward:
                        best_val_reward = val_r
                        best_val_correct = val_c
                        best_dir = ckpt_dir / "best"
                        if best_dir.exists():
                            import shutil
                            shutil.rmtree(best_dir)
                        model.save_pretrained(best_dir)
                        logger.info(f"New best at step {global_step}: {val_r:.4f}")

            # Track.
            epoch_rewards.extend(rewards)
            epoch_correct.extend([ro["correctness"] for ro in rollouts])
            epoch_faithful.extend([ro["faithfulness"] for ro in rollouts])

            if (q_idx + 1) % log_every == 0:
                recent_r = epoch_rewards[-log_every * G:]
                recent_c = epoch_correct[-log_every * G:]
                recent_f = epoch_faithful[-log_every * G:]
                pbar.set_postfix({
                    "reward": f"{np.mean(recent_r):.3f}",
                    "correct": f"{np.mean(recent_c):.3f}",
                    "faith": f"{np.mean(recent_f):.3f}",
                    "step": global_step,
                })

        # Flush remaining.
        if q_idx is not None and (q_idx + 1) % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # Epoch summary.
        logger.info(
            f"Epoch {epoch + 1} | Reward: {np.mean(epoch_rewards):.4f} | "
            f"Correct: {np.mean(epoch_correct):.4f} | "
            f"Faith: {np.mean(epoch_faithful):.4f} | Steps: {global_step}"
        )

        val_r, val_c, val_f = evaluate(
            model, processor, dhcp_probe,
            val_examples, device, alpha, max_new_tokens,
            target_layers, n_heads,
            spatial_classifier=spatial_classifier,
            slake_bboxes=slake_bboxes,
            example_bboxes=example_bboxes,
            max_image_size=max_image_size,
            use_lookback_lens=use_lookback_lens,
            use_temporal_bbox=use_temporal_bbox,
            lookback_classifier=lookback_classifier,
            use_answer_token_faith=use_answer_token_faith,
            use_think_phase_faith=use_think_phase_faith,
            faith_normalizer=faith_norm,
        )
        logger.info(f"Val reward: {val_r:.4f} (correct: {val_c:.4f}, faith: {val_f:.4f})")

        model.save_pretrained(ckpt_dir / f"epoch_{epoch + 1}")
        if val_r > best_val_reward:
            best_val_reward = val_r
            best_val_correct = val_c
            model.save_pretrained(ckpt_dir / "best")
            logger.info(f"New best: {val_r:.4f}")

        # Log.
        log_entry = {
            "epoch": epoch + 1,
            "train_reward": float(np.mean(epoch_rewards)),
            "train_correct": float(np.mean(epoch_correct)),
            "train_faithful": float(np.mean(epoch_faithful)),
            "val_reward": val_r,
            "val_correct": val_c,
            "val_faith": val_f,
            "global_step": global_step,
        }
        with open(ckpt_dir / "training_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    logger.info(
        f"Training complete. Best val reward: {best_val_reward:.4f} "
        f"(correct: {best_val_correct:.4f})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    from faithscan_vqarad.utils.config import load_config
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
