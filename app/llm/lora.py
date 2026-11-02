"""
LoRA fine-tuning adapter layer (PEFT + structure-sensitive LoRA).

This is the core differentiator — "structure-sensitive LoRA". A vanilla LoRA
only optimizes semantics, but industrial XML generation demands structural
correctness. I re-weight the loss: structure tokens such as <tag>, </tag> and
attribute names get higher attention weight, so the model cares more about
structural symbols than content tokens. This is a targeted improvement over
standard LoRA — what separates this project from "just calling an API".

Note: this file is a configuration + training-entry skeleton; real training
runs on a GPU. The reproducible script structure and key hyperparameters are
provided for interview demonstration.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# LoRA config (PEFT)
# ---------------------------------------------------------------------------
LORA_CONFIG = {
    "r": 16,  # rank of low-rank matrices: balances trainable params vs expressiveness
    "lora_alpha": 32,  # scaling factor; alpha/r determines the update magnitude
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",  # the four attention projections
        "gate_proj",
        "up_proj",
        "down_proj",  # MLP (Qwen uses SwiGLU)
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# Structure-sensitive tokens: XML structural symbols, up-weighted in the loss.
STRUCTURE_TOKENS = ["<", ">", "</", "/>", "=", '"', "xml", "binding", "control"]


def build_structure_sensitive_loss(logits, labels, tokenizer, structure_tokens, boost: float = 2.0):
    """Structure-sensitive loss (training only, not the inference path).

    Standard CausalLM cross-entropy treats every token equally. In industrial
    XML, a single wrong structural token (`<`, `>`, attribute names) makes the
    whole fragment unparseable. Multiplying the loss at those positions by a
    boost weight is equivalent to telling the model "structural symbols are
    even less forgivable".
    """
    import torch  # lazy import — only needed in a training environment
    import torch.nn.functional as F  # noqa: N812  # PyTorch canonical alias

    # Find ids of structure tokens and build a position mask.
    structure_ids = set()
    for t in structure_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        structure_ids.update(ids)

    # Only positions where labels != -100 contribute to the loss.
    weight = torch.ones_like(logits, dtype=logits.dtype)
    for sid in structure_ids:
        mask = labels == sid
        weight = weight + (boost - 1.0) * mask.to(logits.dtype).unsqueeze(-1)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weight = weight[..., 1:, :].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    loss = loss.view(shift_labels.size())
    # Weight the loss and average over tokens.
    token_weight = shift_weight[..., 0]
    loss = (loss * token_weight).sum() / token_weight.sum()
    return loss


if __name__ == "__main__":
    # Training-entry reference (real training needs GPU + datasets + peft + transformers).
    print("LoRA training skeleton:")
    print(f"  LoRA config: {LORA_CONFIG}")
    print(f"  Structure tokens: {STRUCTURE_TOKENS}")
    print("  Training command (illustrative):")
    print(
        "  python scripts/train_lora.py --base qwen2.5-7b --data data/xml_pairs.jsonl "
        "--output checkpoints/nc-xml-lora"
    )
