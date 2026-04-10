"""Reward computation for GRPO training with DHCP probe."""

import re

import numpy as np
import torch


def extract_answer_tag(text: str) -> str | None:
    # Try <answer> tags first.
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Qwen3-VL-Thinking format: thinking is before </think>, answer is after.
    # Take text between first </think> and next <think> or <|im_end|> or end.
    think_match = re.search(r'</think>\s*(.*?)(?:<think>|<\|im_end\|>|<\|im_start\|>|$)', text, re.DOTALL)
    if think_match:
        answer = think_match.group(1).strip()
        # Remove special tokens.
        answer = re.sub(r'<\|[^>]+\|>', '', answer).strip()
        return answer if answer else None
    return None


def compute_format_reward(generated: str) -> float:
    has_answer = extract_answer_tag(generated) is not None
    return 1.0 if has_answer else 0.0


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into tokens."""
    return re.sub(r'[^\w\s]', '', text.strip().lower()).split()


def compute_correctness(generated: str, ground_truth: str) -> float:
    """Token F1 between extracted answer and ground truth.

    Returns F1 score in [0, 1]. For yes/no questions, returns 1.0 or 0.0.
    This is the standard metric used in medical VQA (Med-R1, VQA-RAD evals).
    """
    tagged = extract_answer_tag(generated)
    if tagged is not None:
        generated = tagged
    else:
        # No tags found — use last sentence only.
        sentences = [s.strip() for s in re.split(r'[.!?\n]', generated) if s.strip()]
        if sentences:
            generated = sentences[-1]

    gen_tokens = _tokenize(generated)
    gt_tokens = _tokenize(ground_truth)

    if not gen_tokens or not gt_tokens:
        return 0.0

    # Closed-ended: exact match for yes/no and numbers.
    gt_clean = ' '.join(gt_tokens)
    if gt_clean in ("yes", "no"):
        return 1.0 if gen_tokens[0] == gt_clean else 0.0
    if re.fullmatch(r'\d+', gt_clean):
        return 1.0 if gen_tokens[0] == gt_clean else 0.0

    # Token F1 for open-ended (What, Where, Which).
    gen_set = set(gen_tokens)
    gt_set = set(gt_tokens)
    overlap = gen_set & gt_set

    if not overlap:
        return 0.0

    precision = len(overlap) / len(gen_set)
    recall = len(overlap) / len(gt_set)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_composite_reward(
    correctness: float,
    faithfulness: float,
    alpha: float,
    format_reward: float = 1.0,
    format_weight: float = 0.1,
) -> float:
    # Hard gate: unformatted responses get 0 reward.
    if format_reward == 0.0:
        return 0.0
    content = alpha * correctness + (1.0 - alpha) * faithfulness
    return content
