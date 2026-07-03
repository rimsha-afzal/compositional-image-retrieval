"""Neutral-subtracted prompt ensemble text embeddings."""

from typing import Any

import torch

from embedding_extraction import _as_tensor
from embedding_io import l2_normalize


PROMPT_DELTA_TEMPLATES = [
    ("a photo of a face", "a photo of a face that is {modification_text}"),
    ("a portrait photo of a person", "a portrait photo of a person who is {modification_text}"),
    ("a close-up face image", "a close-up face image of a person who is {modification_text}"),
    ("a celebrity face photo", "a celebrity face photo with {modification_text}"),
    ("a human face", "a human face that is {modification_text}"),
]


def build_prompt_pairs(modification_text: str) -> list[tuple[str, str]]:
    """Build paired neutral and modified prompts for one text modification."""
    return [
        (neutral, modified.format(modification_text=modification_text))
        for neutral, modified in PROMPT_DELTA_TEMPLATES
    ]


def encode_texts(
    texts: list[str],
    clip_model: Any,
    processor: Any,
    device: str | torch.device,
) -> torch.Tensor:
    """Encode text prompts with CLIP."""
    resolved_device = torch.device(device)
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(resolved_device)
    with torch.no_grad():
        features = _as_tensor(clip_model.get_text_features(**inputs))
    return features.cpu().float()


def encode_prompt_delta(
    modification_text: str,
    clip_model: Any,
    processor: Any,
    device: str | torch.device,
    normalize_each_embedding: bool = True,
    normalize_final: bool = True,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Encode a paired neutral-subtracted prompt ensemble delta."""
    prompt_pairs = build_prompt_pairs(modification_text)
    flat_prompts = [prompt for pair in prompt_pairs for prompt in pair]
    embeddings = encode_texts(flat_prompts, clip_model, processor, device)
    neutral_embeddings = embeddings[0::2]
    modified_embeddings = embeddings[1::2]

    if normalize_each_embedding:
        neutral_embeddings = l2_normalize(neutral_embeddings)
        modified_embeddings = l2_normalize(modified_embeddings)

    deltas = modified_embeddings - neutral_embeddings
    mean_delta = deltas.mean(dim=0)
    mean_delta_norm = float(torch.linalg.norm(mean_delta).item())
    if normalize_final:
        mean_delta = l2_normalize(mean_delta)

    if torch.isnan(mean_delta).any():
        raise ValueError(f"Prompt-delta embedding for {modification_text!r} contains NaN values.")

    final_norm = float(torch.linalg.norm(mean_delta).item())
    if normalize_final and abs(final_norm - 1.0) > 1e-4:
        raise ValueError(f"Prompt-delta embedding for {modification_text!r} is not normalized.")

    diagnostics = {
        "num_prompt_pairs": len(prompt_pairs),
        "mean_delta_norm_before_final_normalization": mean_delta_norm,
        "final_norm": final_norm,
    }
    return mean_delta, diagnostics
