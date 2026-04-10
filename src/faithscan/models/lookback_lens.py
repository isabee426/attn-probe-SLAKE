"""Lookback Lens adapted for VLM hallucination detection.

Based on: "Detecting and Mitigating Contextual Hallucinations in Large Language
Models Using Only Attention Maps" (Chuang et al., EMNLP 2024)

Adaptation for VLMs: "context" = image/vision tokens, "new tokens" = generated
answer tokens. The lookback ratio measures how much each generated token attends
to the image vs previously generated text — a per-token, per-rollout signal.

Key property: different rollouts produce different tokens with different attention
patterns → different lookback ratios → different faithfulness scores. This solves
the constant-score problem we had with DHCP.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle
from pathlib import Path


class HeadSelector:
    """Sklearn-compatible transformer that selects specific head indices from a feature vector."""
    def __init__(self, indices):
        self.indices = np.array(indices)
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[:, self.indices]
    def get_params(self, deep=True):
        return {"indices": self.indices}
    def set_params(self, **params):
        self.indices = params.get("indices", self.indices)
        return self


class LookbackLensClassifier:
    """Simple logistic regression on lookback ratios, following the paper."""

    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, C=1.0)
        self.fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray):
        """Fit on (n_samples, n_layers * n_heads) features."""
        self.clf.fit(features, labels)
        self.fitted = True

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return hallucination probability for each sample."""
        return self.clf.predict_proba(features)[:, 1]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.clf, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.clf = pickle.load(f)
        self.fitted = True


def compute_lookback_ratio_from_attentions(
    attentions: tuple,
    context_length: int,
    num_new_tokens: int,
) -> torch.Tensor:
    """Compute lookback ratios from model.generate() attentions output.

    For each new token, each layer, each head: ratio of attention to context
    (image tokens) vs all tokens.

    Args:
        attentions: tuple of length num_new_tokens, each element is a tuple of
            length num_layers, each element is (batch, n_heads, 1, seq_len_so_far)
        context_length: number of context (image + question) tokens
        num_new_tokens: number of generated tokens

    Returns:
        (num_layers, num_heads, num_new_tokens) lookback ratio tensor
    """
    num_layers = len(attentions[0])
    num_heads = attentions[0][0].shape[1]

    lookback_ratio = torch.zeros(num_layers, num_heads, num_new_tokens)

    for i in range(num_new_tokens):
        for l in range(num_layers):
            # attentions[token_idx][layer_idx] shape: (1, n_heads, 1, seq_so_far)
            attn = attentions[i][l][0, :, -1, :]  # (n_heads, seq_so_far)

            attn_on_context = attn[:, :context_length].mean(-1)  # (n_heads,)
            attn_on_new = attn[:, context_length:].mean(-1)  # (n_heads,)

            # Avoid division by zero
            total = attn_on_context + attn_on_new + 1e-10
            lookback_ratio[l, :, i] = attn_on_context / total

    return lookback_ratio


def compute_vision_lookback_ratio(
    attentions: tuple,
    vision_start: int,
    vision_end: int,
    num_new_tokens: int,
) -> torch.Tensor:
    """Compute lookback ratio specifically for vision tokens (not all context).

    More precise than treating all context as "image" — only measures attention
    to the actual image token region.

    Args:
        attentions: from model.generate() with output_attentions=True
        vision_start: index of first vision token
        vision_end: index after last vision token
        num_new_tokens: number of generated tokens

    Returns:
        (num_layers, num_heads, num_new_tokens) vision lookback ratio
    """
    num_layers = len(attentions[0])
    num_heads = attentions[0][0].shape[1]

    lookback_ratio = torch.zeros(num_layers, num_heads, num_new_tokens)

    for i in range(num_new_tokens):
        for l in range(num_layers):
            attn = attentions[i][l][0, :, -1, :]  # (n_heads, seq_so_far)

            attn_on_vision = attn[:, vision_start:vision_end].sum(-1)  # (n_heads,)
            attn_total = attn.sum(-1)  # (n_heads,)

            lookback_ratio[l, :, i] = attn_on_vision / (attn_total + 1e-10)

    return lookback_ratio


def compute_per_vision_token_attention(
    attentions: tuple,
    vision_start: int,
    vision_end: int,
    num_new_tokens: int,
) -> torch.Tensor:
    """Compute per-vision-token attention from generated tokens.

    For each layer and head, averages the attention to each vision token
    across all generated tokens. Returns spatial attention map over the image.

    Returns:
        (num_layers, num_heads, n_vision_tokens) attention map
    """
    num_layers = len(attentions[0])
    num_heads = attentions[0][0].shape[1]
    n_vision = vision_end - vision_start

    if n_vision <= 0:
        return torch.zeros(num_layers, num_heads, 1)

    spatial_attn = torch.zeros(num_layers, num_heads, n_vision)

    for i in range(num_new_tokens):
        for l in range(num_layers):
            attn = attentions[i][l][0, :, -1, :]  # (n_heads, seq_so_far)
            # Extract attention to each vision token
            if attn.shape[1] > vision_end:
                vis_attn = attn[:, vision_start:vision_end]  # (n_heads, n_vision)
                spatial_attn[l] += vis_attn.cpu()

    # Average over generated tokens
    spatial_attn /= max(num_new_tokens, 1)
    return spatial_attn


def compute_spatial_focus(
    spatial_attn: torch.Tensor,
    bbox: list,
    image_size: tuple,
    patch_size: int = 32,
) -> np.ndarray | None:
    """Compute per-head ratio of attention inside vs outside bounding box.

    Args:
        spatial_attn: (n_layers, n_heads, n_vision_tokens) per-vision-token attention
        bbox: [x, y, w, h] in absolute pixel coordinates
        image_size: (width, height)
        patch_size: effective pixels per vision token

    Returns:
        (n_layers * n_heads,) spatial ratio vector, or None if bbox doesn't map.
        > 0.5 means more attention to bbox region.
    """
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


def lookback_ratio_to_features(
    lookback_ratio: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """Convert (num_layers, num_heads, num_tokens) to a fixed-size feature vector.

    Args:
        lookback_ratio: (num_layers, num_heads, num_new_tokens)
        mode: "mean" averages over tokens, "last" takes last token,
              "mean_last" concatenates both

    Returns:
        (num_layers * num_heads,) or (num_layers * num_heads * 2,) feature vector
    """
    n_layers, n_heads, n_tokens = lookback_ratio.shape

    if mode == "mean":
        return lookback_ratio.mean(dim=-1).flatten()  # (n_layers * n_heads,)
    elif mode == "last":
        return lookback_ratio[:, :, -1].flatten()
    elif mode == "mean_last":
        mean_feat = lookback_ratio.mean(dim=-1).flatten()
        last_feat = lookback_ratio[:, :, -1].flatten()
        return torch.cat([mean_feat, last_feat])
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_temporal_bbox_faith(
    attentions: tuple,
    vision_start: int,
    vision_end: int,
    bbox: list,
    image_size: tuple,
    patch_size: int,
    num_new_tokens: int,
) -> float | None:
    """Per-token bbox attention ratio averaged across generation trajectory.

    For each generated token t, computes:
        bbox_ratio[t] = sum(attn to bbox vision tokens) / sum(attn to all tokens)

    Then averages across all t and all heads/layers.

    A model that spirals in text ("Wait, actually...") has declining bbox_ratio[t]
    over time. This captures both spatial grounding AND temporal consistency.

    Args:
        attentions: from model.generate() with output_attentions=True
        vision_start: index of first vision token
        vision_end: index after last vision token
        bbox: [x, y, w, h] pixel coordinates
        image_size: (width, height)
        patch_size: effective pixels per vision token
        num_new_tokens: number of generated tokens

    Returns:
        Scalar faith in [0, 1], or None if bbox doesn't map to any vision tokens.
    """
    import math

    n_vision = vision_end - vision_start
    if n_vision <= 0:
        return None

    w, h = image_size
    grid_w = max(w // patch_size, 1)
    grid_h = max(h // patch_size, 1)

    bx, by, bw, bh = bbox
    px_start = max(int(bx / patch_size), 0)
    py_start = max(int(by / patch_size), 0)
    px_end = min(int((bx + bw) / patch_size) + 1, grid_w)
    py_end = min(int((by + bh) / patch_size) + 1, grid_h)

    bbox_indices = set()
    for py in range(py_start, py_end):
        for px in range(px_start, px_end):
            idx = py * grid_w + px
            if idx < n_vision:
                bbox_indices.add(idx)

    if not bbox_indices:
        return None

    # Absolute indices into the full sequence.
    abs_bbox_indices = [vision_start + i for i in bbox_indices]

    num_layers = len(attentions[0])
    num_heads = attentions[0][0].shape[1]

    total_ratio = 0.0
    count = 0

    for t in range(num_new_tokens):
        for l in range(num_layers):
            # (n_heads, seq_so_far)
            attn = attentions[t][l][0, :, -1, :]
            seq_len = attn.shape[1]

            valid_bbox = [i for i in abs_bbox_indices if i < seq_len]
            if not valid_bbox:
                continue

            bbox_attn = attn[:, valid_bbox].sum(-1)   # (n_heads,)
            total_attn = attn.sum(-1) + 1e-10          # (n_heads,)
            ratio = (bbox_attn / total_attn).mean().item()
            total_ratio += ratio
            count += 1

    if count == 0:
        return None

    return total_ratio / count


def compute_answer_token_vision_faith(
    attentions: tuple,
    vision_start: int,
    vision_end: int,
    gen_ids,
    answer_tag_id: int,
) -> float:
    """Ratio of vision attention at the answer tag token.

    Finds the first <answer> token in gen_ids and measures how much that
    token's attention goes to vision tokens vs everything else.
    High value = model looked at the image when committing to its answer.

    Returns scalar in [0, 1], or 0.5 if answer tag not found.
    """
    # Find position of first answer tag token.
    # <answer> tokenizes as ['<', 'answer', '>'] so we look for the 'answer'
    # token (answer_tag_id) preceded by '<' to avoid false matches.
    answer_pos = None
    for i, tid in enumerate(gen_ids):
        if int(tid) == answer_tag_id:
            # Check preceding token is '<' if possible
            if i == 0 or int(gen_ids[i-1]) == 27:  # 27 = '<'
                answer_pos = i
                break

    if answer_pos is None or answer_pos >= len(attentions):
        return 0.5

    num_layers = len(attentions[answer_pos])
    total_vision = 0.0
    total_all = 0.0

    for l in range(num_layers):
        attn = attentions[answer_pos][l][0, :, -1, :]  # (n_heads, seq_so_far)
        vision_attn = attn[:, vision_start:vision_end].sum(-1)  # (n_heads,)
        all_attn = attn.sum(-1)  # (n_heads,)
        total_vision += vision_attn.sum().item()
        total_all += all_attn.sum().item()

    if total_all < 1e-10:
        return 0.5
    return total_vision / total_all


def compute_think_vs_answer_vision_faith(
    attentions: tuple,
    vision_start: int,
    vision_end: int,
    gen_ids,
    think_end_id: int,
) -> float:
    """Ratio of (answer-phase vision attn) / (think-phase vision attn).

    Splits the generation at </think> token. Compares mean vision attention
    ratio during the think block vs the answer block.

    Returns scalar: >1 means model looked at image MORE during answering,
    <1 means model drifted from image by answer time.
    Normalized to [0, 1] via sigmoid for use as faith signal.
    """
    import math

    # Find </think> token position
    think_end_pos = None
    for i, tid in enumerate(gen_ids):
        if int(tid) == think_end_id:
            think_end_pos = i
            break

    num_tokens = len(attentions)
    num_layers = len(attentions[0])

    def mean_vision_ratio(token_range):
        total = 0.0
        count = 0
        for t in token_range:
            if t >= num_tokens:
                break
            for l in range(num_layers):
                attn = attentions[t][l][0, :, -1, :]
                v = attn[:, vision_start:vision_end].sum(-1).sum().item()
                a = attn.sum(-1).sum().item()
                if a > 1e-10:
                    total += v / a
                    count += 1
        return total / count if count > 0 else 0.5

    if think_end_pos is None:
        # No think block found — use full sequence
        return mean_vision_ratio(range(num_tokens))

    think_ratio = mean_vision_ratio(range(0, think_end_pos))
    answer_ratio = mean_vision_ratio(range(think_end_pos, num_tokens))

    if think_ratio < 1e-10:
        return 0.5

    # Raw ratio: how much did image attention persist into answer phase
    raw = answer_ratio / (think_ratio + 1e-10)
    # Sigmoid to map to [0, 1]: raw=1.0 → 0.73, raw=0.5 → 0.62, raw=0.0 → 0.5
    return 1.0 / (1.0 + math.exp(-raw))


def extract_lookback_from_generate(
    model,
    processor,
    image,
    question: str,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    do_sample: bool = True,
) -> dict:
    """Generate an answer and extract lookback ratios in one pass.

    Returns a dict with:
        - answer_text: str
        - lookback_ratio: (n_layers, n_heads, n_new_tokens) tensor
        - vision_lookback_ratio: (n_layers, n_heads, n_new_tokens) tensor
        - gen_ids: generated token ids
        - old_log_probs: per-token log probabilities
    """
    from qwen_vl_utils import process_vision_info

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
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    prefill_len = inputs["input_ids"].shape[1]

    # Find vision token boundaries.
    input_ids = inputs["input_ids"][0]
    tokenizer = processor.tokenizer
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    ids = input_ids.tolist()
    vs_pos = ve_pos = None
    for i, tid in enumerate(ids):
        if tid == vision_start_id and vs_pos is None:
            vs_pos = i
        if tid == vision_end_id:
            ve_pos = i

    vision_start = (vs_pos + 1) if vs_pos is not None else 0
    vision_end = ve_pos if ve_pos is not None else 0

    # Generate with attention output.
    # Must use eager attention — SDPA doesn't return attention weights.
    # During autoregressive generation, attention is (n_heads, 1, seq_so_far)
    # per step — tiny memory footprint, no OOM risk unlike teacher forcing.
    old_impl = getattr(model.config, '_attn_implementation', 'sdpa')
    model.config._attn_implementation = "eager"
    if hasattr(model.config, 'text_config'):
        old_text_impl = getattr(model.config.text_config, '_attn_implementation', 'sdpa')
        model.config.text_config._attn_implementation = "eager"

    # Two-phase generation:
    # Phase 1: thinking (up to think_budget tokens, stop if </think> produced)
    # Phase 2: force </think> if not produced, then generate answer
    think_budget = max(max_new_tokens - 64, max_new_tokens * 3 // 4)
    answer_budget = max_new_tokens - think_budget

    tokenizer = processor.tokenizer
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")

    with torch.no_grad():
        # Phase 1: thinking
        gen_out = model.generate(
            **inputs,
            max_new_tokens=think_budget,
            do_sample=do_sample,
            temperature=temperature,
            top_p=1.0,
            return_dict_in_generate=True,
            output_scores=True,
            output_attentions=True,
        )

    phase1_ids = gen_out.sequences[0, prefill_len:]
    phase1_attentions = gen_out.attentions
    phase1_scores = gen_out.scores

    # Check if model produced </think> and has answer content after it.
    phase1_list = phase1_ids.tolist()
    produced_think_end = think_end_id in phase1_list

    # Extract answer after </think> if present.
    has_answer_content = False
    if produced_think_end:
        think_pos = phase1_list.index(think_end_id)
        after_think = phase1_ids[think_pos + 1:]
        # Filter out special tokens to check for real content.
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        content_ids = [t for t in after_think.tolist()
                       if t not in (im_end_id,) and t is not None]
        has_answer_content = len(content_ids) > 2

    if has_answer_content:
        # Model produced </think> + actual answer — use as-is.
        all_gen_ids = phase1_ids
        all_attentions = phase1_attentions
        all_scores = phase1_scores
    else:
        # Force </think> (if not present) then generate answer.
        if produced_think_end:
            # Already has </think> but no answer — trim to </think> and regenerate.
            think_pos = phase1_list.index(think_end_id)
            trimmed = gen_out.sequences[0, :prefill_len + think_pos + 1].unsqueeze(0)
            forced_input = trimmed
            prefix_ids = phase1_ids[:think_pos + 1]
        else:
            think_end_tensor = torch.tensor([[think_end_id]], device=device)
            forced_input = torch.cat([gen_out.sequences, think_end_tensor], dim=1)
            prefix_ids = torch.cat([phase1_ids, think_end_tensor[0]])

        # Ban <think> token so model can't re-enter thinking mode.
        think_start_id = tokenizer.convert_tokens_to_ids("<think>")
        bad_words = [[think_start_id]] if think_start_id is not None else None

        with torch.no_grad():
            answer_out = model.generate(
                input_ids=forced_input,
                max_new_tokens=answer_budget,
                do_sample=False,  # greedy for answer
                return_dict_in_generate=True,
                output_scores=True,
                output_attentions=True,
                bad_words_ids=bad_words,
            )
        answer_ids = answer_out.sequences[0, forced_input.shape[1]:]
        all_gen_ids = torch.cat([prefix_ids, answer_ids])
        all_attentions = phase1_attentions
        all_scores = phase1_scores

    # Restore original attention implementation.
    model.config._attn_implementation = old_impl
    if hasattr(model.config, 'text_config'):
        model.config.text_config._attn_implementation = old_text_impl

    gen_ids = all_gen_ids
    answer_text = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
    num_new_tokens = len(phase1_ids)  # lookback only from thinking phase

    # Per-token log probs (from thinking phase only — scores match phase1_ids).
    old_log_probs = []
    for step, logits in enumerate(phase1_scores):
        probs = torch.softmax(logits[0].float(), dim=-1)
        log_p = torch.log(probs + 1e-10)
        token_id = phase1_ids[step].item()
        old_log_probs.append(log_p[token_id].item())

    # Compute lookback ratios.
    attentions = gen_out.attentions
    if attentions is not None and len(attentions) > 0:
        vision_lb = compute_vision_lookback_ratio(
            attentions, vision_start, vision_end, num_new_tokens,
        )
        context_lb = compute_lookback_ratio_from_attentions(
            attentions, prefill_len, num_new_tokens,
        )
        # Per-vision-token spatial attention map.
        spatial_attn = compute_per_vision_token_attention(
            attentions, vision_start, vision_end, num_new_tokens,
        )
    else:
        n_layers = 28  # fallback
        n_heads = 12
        vision_lb = torch.zeros(n_layers, n_heads, max(num_new_tokens, 1))
        context_lb = torch.zeros(n_layers, n_heads, max(num_new_tokens, 1))
        spatial_attn = torch.zeros(n_layers, n_heads, 1)

    return {
        "answer_text": answer_text,
        "gen_ids": gen_ids.cpu(),
        "old_log_probs": old_log_probs,
        "vision_lookback_ratio": vision_lb,
        "context_lookback_ratio": context_lb,
        "spatial_vision_attn": spatial_attn,
        # Raw attentions + boundaries for temporal bbox faith computation.
        "attentions": attentions,
        "vision_start": vision_start,
        "vision_end": vision_end,
    }
