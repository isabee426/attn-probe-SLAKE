"""DHCP-style cross-modal attention probe for hallucination detection.

Based on: "Detecting Hallucinations by Cross-modal Attention Pattern in VLMs"
(Zhang et al., ACM MM 2025, arXiv:2411.18659)

Architecture:
- Extract cross-modal attention: answer tokens attending to image tokens
  across ALL layers and ALL attention heads.
- Aggregate into a fixed-size feature tensor per rollout.
- Train a lightweight MLP to predict hallucination probability.

Key insight: the MLP learns which heads/layers matter automatically,
no manual head selection needed.

Qwen2-VL-2B architecture: 28 layers, 12 attention heads per layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DHCPProbe(nn.Module):
    """Cross-modal attention hallucination probe.

    Input: cross-modal attention features, flattened.
      - Legacy mode (vision_pool=1): shape (n_layers * n_heads,)
      - Paper-faithful mode: shape (n_layers * n_heads * vision_pool,)
    Output: hallucination probability in [0, 1].
    """

    def __init__(
        self,
        n_layers: int = 28,
        n_heads: int = 12,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        vision_pool: int = 1,
    ):
        super().__init__()
        input_dim = n_layers * n_heads * vision_pool

        # Paper-faithful: 2-layer MLP with ReLU, no dropout.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *(nn.Dropout(dropout),) if dropout > 0 else (),
            nn.Linear(hidden_dim, 1),
        )

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vision_pool = vision_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict hallucination probability.

        Args:
            x: (B, n_layers * n_heads) cross-modal attention features
        Returns:
            (B,) probabilities in [0, 1]
        """
        return torch.sigmoid(self.net(x).squeeze(-1))

    def forward_logit(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits before sigmoid."""
        return self.net(x).squeeze(-1)

    def compute_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.net(x).squeeze(-1)
        return F.binary_cross_entropy_with_logits(
            logits, labels.float(), pos_weight=pos_weight,
        )


def extract_cross_modal_attention(
    model,
    processor,
    image,
    question: str,
    answer: str,
    device: str,
    target_layers: list[int] | None = None,
    n_heads: int = 12,
) -> torch.Tensor:
    """Extract cross-modal attention features for a single (image, question, answer).

    Uses forward hooks on specific layers to capture attention weights one layer
    at a time, avoiding the OOM from materializing all 28 layers simultaneously.

    For each target layer and head, compute the mean attention weight from answer
    tokens to image tokens.

    Args:
        target_layers: Which layers to extract from. Defaults to [4, 8, 12, 16, 20, 24]
            (middle-to-late layers where visual processing happens per "Devils in
            Middle Layers" paper).

    Returns:
        (len(target_layers) * n_heads,) flattened attention feature vector.
    """
    from qwen_vl_utils import process_vision_info

    if target_layers is None:
        target_layers = [4, 8, 12, 16, 20, 24]

    # Build input with answer included (teacher forcing).
    # The answer may include <think>...</think> reasoning + <answer>...</answer>.
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": (
                "Answer this medical question. First explain your reasoning "
                "about what you see in the image in <think></think> tags, "
                "then put your final answer inside <answer></answer> tags.\n"
                f"Question: {question}"
            )},
        ]},
        {"role": "assistant", "content": answer},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(device)

    # Build vision and answer masks BEFORE forward pass.
    input_ids = inputs["input_ids"][0]
    tokenizer = processor.tokenizer
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

    ids = input_ids.tolist()
    seq_len = len(ids)

    # Find vision token range.
    vs_pos = ve_pos = None
    for i, tid in enumerate(ids):
        if tid == vision_start_id and vs_pos is None:
            vs_pos = i
        if tid == vision_end_id:
            ve_pos = i

    # Find assistant answer start (last <|im_start|>).
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    asst_start = None
    for i in range(len(ids) - 1, -1, -1):
        if ids[i] == im_start_id:
            asst_start = i
            break

    n_target = len(target_layers)
    if vs_pos is None or ve_pos is None or asst_start is None:
        return torch.zeros(n_target * n_heads, device=device)

    vision_indices = list(range(vs_pos + 1, ve_pos))
    answer_indices = list(range(asst_start, seq_len))

    if not vision_indices or not answer_indices:
        return torch.zeros(n_target * n_heads, device=device)

    # Get the language model's layers.
    if hasattr(model, 'model') and hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        layers_module = model.model.language_model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers_module = model.model.layers
    else:
        return torch.zeros(n_target * n_heads, device=device)

    # Hook into target decoder layers to compute Q@K attention scores ONLY for
    # answer→vision token pairs. This avoids materializing the full
    # (seq_len × seq_len) attention matrix that causes OOM with eager attention.
    # Memory cost per layer: (n_heads, n_answer, n_vision) ≈ tiny.
    captured_attn: dict[int, torch.Tensor] = {}
    answer_idx_t = torch.tensor(answer_indices, device=device)
    vision_idx_t = torch.tensor(vision_indices, device=device)

    def make_layer_hook(layer_idx):
        def hook_fn(module, args, output):
            hidden = args[0] if len(args) > 0 else None
            if hidden is None:
                return

            # Decoder layer applies input_layernorm before self_attn.
            hidden = module.input_layernorm(hidden)

            attn_mod = module.self_attn
            q = attn_mod.q_proj(hidden)
            k = attn_mod.k_proj(hidden)

            head_dim = attn_mod.head_dim
            n_q_heads = q.shape[-1] // head_dim
            n_kv_heads = k.shape[-1] // head_dim

            q = q.view(1, -1, n_q_heads, head_dim).transpose(1, 2)
            k = k.view(1, -1, n_kv_heads, head_dim).transpose(1, 2)

            if n_kv_heads < n_q_heads:
                k = k.repeat_interleave(n_q_heads // n_kv_heads, dim=1)

            # Compute attention from answer tokens to ALL keys, softmax over
            # the full sequence, then extract vision columns. This matches the
            # original DHCP paper which uses raw model attention weights
            # (softmax'd over all tokens). The vision-token attention values
            # reflect how much the model attends to the image vs text — a
            # hallucinating model attends less to image tokens overall.
            q_ans = q[0, :, answer_idx_t, :]   # (n_heads, n_ans, head_dim)
            k_all = k[0]                        # (n_heads, seq_len, head_dim)

            # (n_heads, n_ans, seq_len) — softmax over all keys
            scores = torch.bmm(q_ans, k_all.transpose(1, 2)) / (head_dim ** 0.5)
            attn_weights = torch.softmax(scores.float(), dim=-1)

            # Extract attention to vision tokens only
            attn_to_vision = attn_weights[:, :, vision_idx_t]  # (n_heads, n_ans, n_vis)
            # Mean over answer and vision tokens → per-head mean attention to image
            captured_attn[layer_idx] = attn_to_vision.mean(dim=(1, 2)).cpu()

        return hook_fn

    hooks = []
    for li in target_layers:
        if li < len(layers_module):
            h = layers_module[li].register_forward_hook(make_layer_hook(li))
            hooks.append(h)

    with torch.no_grad():
        model(**dict(inputs), output_attentions=False)

    for h in hooks:
        h.remove()

    # Build feature vector.
    features = torch.zeros(n_target, n_heads, device=device)
    for feat_idx, li in enumerate(target_layers):
        if li in captured_attn:
            per_head = captured_attn[li].to(device)
            actual_heads = per_head.shape[0]
            if actual_heads == n_heads:
                features[feat_idx] = per_head
            elif actual_heads < n_heads:
                repeat_factor = n_heads // actual_heads
                features[feat_idx] = per_head.repeat(repeat_factor)[:n_heads]
            else:
                features[feat_idx] = per_head[:n_heads]

    return features.flatten()


def extract_cross_modal_attention_from_generate(
    gen_out,
    input_ids: torch.Tensor,
    processor,
    n_layers: int = 28,
    n_heads: int = 12,
    device: str = "cuda",
) -> torch.Tensor:
    """Extract cross-modal attention from a generate() call with output_attentions=True.

    This is used during rollout to get per-rollout attention features
    without an extra forward pass.

    Args:
        gen_out: output from model.generate(output_attentions=True)
        input_ids: the prefill input_ids (before generation)
        processor: the tokenizer/processor
        n_layers: number of transformer layers
        n_heads: number of attention heads

    Returns:
        (n_layers * n_heads,) flattened attention feature vector.
    """
    tokenizer = processor.tokenizer
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    ids = input_ids.tolist()

    # Find vision token range in the prefill.
    vs_pos = ve_pos = None
    for i, tid in enumerate(ids):
        if tid == vision_start_id and vs_pos is None:
            vs_pos = i
        if tid == vision_end_id:
            ve_pos = i

    if vs_pos is None or ve_pos is None:
        return torch.zeros(n_layers * n_heads, device=device)

    vision_indices = list(range(vs_pos + 1, ve_pos))
    if not vision_indices:
        return torch.zeros(n_layers * n_heads, device=device)

    prefill_len = len(ids)

    # gen_out.attentions is a tuple of (n_gen_steps,) each containing
    # a tuple of (n_layers,) each with shape (B, n_heads, curr_seq_len, curr_seq_len).
    # For each generated token step, we extract attention to vision tokens.
    if not hasattr(gen_out, "attentions") or gen_out.attentions is None:
        return torch.zeros(n_layers * n_heads, device=device)

    features = torch.zeros(n_layers, n_heads, device=device)
    n_steps = len(gen_out.attentions)

    for step_idx in range(n_steps):
        step_attns = gen_out.attentions[step_idx]  # tuple of (n_layers,)
        for layer_idx in range(min(n_layers, len(step_attns))):
            attn = step_attns[layer_idx][0]  # (n_heads, 1_or_seq, full_seq)
            actual_heads = attn.shape[0]

            # Last row = current generated token attending to all previous.
            # Extract attention to vision tokens.
            last_row = attn[:, -1, :]  # (n_heads, full_seq)

            # Only take vision indices that exist in the attention matrix.
            valid_vision = [v for v in vision_indices if v < last_row.shape[-1]]
            if not valid_vision:
                continue

            vis_attn = last_row[:, valid_vision].mean(dim=-1)  # (n_heads,)

            if actual_heads == n_heads:
                features[layer_idx] += vis_attn.float()
            elif actual_heads < n_heads:
                repeat_factor = n_heads // actual_heads
                features[layer_idx] += vis_attn.repeat(repeat_factor)[:n_heads].float()

    # Average over generation steps.
    if n_steps > 0:
        features /= n_steps

    return features.flatten()
