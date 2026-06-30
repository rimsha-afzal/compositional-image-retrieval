"""Compute reliability-weighted CelebA visual attribute directions."""

import json
from pathlib import Path
from typing import Any

import torch

from embedding_io import l2_normalize, load_embeddings, save_embeddings
from retrieval.visual_directions import load_split_metadata


def min_max_normalize(values: torch.Tensor) -> torch.Tensor:
    """Scale a 1D tensor to [0, 1], returning zeros when all values match."""
    min_value = values.min()
    max_value = values.max()
    denominator = max_value - min_value
    if float(denominator.item()) == 0.0:
        return torch.zeros_like(values)
    return (values - min_value) / denominator


def compute_reliability_visual_directions(
    image_embeddings: torch.Tensor,
    image_ids: list[int],
    attribute_names: list[str],
    label_rows: list[dict[str, int]],
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[dict[str, Any]]]:
    """Compute normalized visual directions and train-only attribute reliability."""
    if len(image_ids) != len(label_rows):
        raise ValueError("Image embedding IDs and metadata rows have different lengths.")

    expected_ids = list(range(len(label_rows)))
    if [int(image_id) for image_id in image_ids] != expected_ids:
        raise ValueError("Image embedding IDs must be split-local indices 0..N-1.")

    embeddings = image_embeddings.float()
    labels = torch.tensor(
        [[row[attribute] for attribute in attribute_names] for row in label_rows],
        dtype=torch.int64,
    )

    normalized_directions = []
    raw_directions = []
    raw_reliability_values = []
    metrics = []

    for col, attribute in enumerate(attribute_names):
        values = labels[:, col]
        pos_mask = values == 1
        neg_mask = values == -1
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())
        if n_pos == 0 or n_neg == 0:
            raise ValueError(f"Attribute {attribute!r} needs both positive and negative examples.")

        pos_embeddings = embeddings[pos_mask]
        neg_embeddings = embeddings[neg_mask]
        pos_mean = pos_embeddings.mean(dim=0)
        neg_mean = neg_embeddings.mean(dim=0)
        raw_direction = pos_mean - neg_mean
        direction_norm = torch.linalg.norm(raw_direction)
        spread_pos = torch.linalg.norm(pos_embeddings - pos_mean, dim=1).mean()
        spread_neg = torch.linalg.norm(neg_embeddings - neg_mean, dim=1).mean()
        raw_reliability = direction_norm / (spread_pos + spread_neg + epsilon)

        normalized_directions.append(l2_normalize(raw_direction))
        raw_directions.append(raw_direction)
        raw_reliability_values.append(raw_reliability)
        metrics.append(
            {
                "attribute": attribute,
                "support_positive": n_pos,
                "support_negative": n_neg,
                "direction_norm": float(direction_norm.item()),
                "spread_positive": float(spread_pos.item()),
                "spread_negative": float(spread_neg.item()),
                "raw_reliability": float(raw_reliability.item()),
            }
        )

    raw_reliability_tensor = torch.stack(raw_reliability_values).float()
    reliability_tensor = min_max_normalize(raw_reliability_tensor)
    for metric, reliability in zip(metrics, reliability_tensor.tolist()):
        metric["reliability"] = float(reliability)

    return (
        torch.stack(normalized_directions),
        reliability_tensor,
        torch.stack(raw_directions),
        attribute_names,
        metrics,
    )


def compute_and_save_reliability_visual_directions(
    metadata_path: str | Path,
    image_embedding_dir: str | Path,
    output_dir: str | Path,
    metrics_path: str | Path,
    split_label: str = "0",
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Compute reliability artifacts from train embeddings and save tensors plus metrics."""
    attribute_names, label_rows = load_split_metadata(metadata_path, split_label)
    image_embeddings, image_ids = load_embeddings(image_embedding_dir, "image_embeddings", "cpu")
    image_ids = [int(image_id) for image_id in image_ids]

    directions, reliability, raw_directions, ids, metrics = compute_reliability_visual_directions(
        image_embeddings=image_embeddings,
        image_ids=image_ids,
        attribute_names=attribute_names,
        label_rows=label_rows,
        epsilon=epsilon,
    )

    save_embeddings(directions, ids, output_dir, "visual_directions_reliable")
    save_embeddings(raw_directions, ids, output_dir, "visual_directions_raw")
    save_embeddings(reliability.unsqueeze(1), ids, output_dir, "visual_direction_reliability")

    payload = {
        "status": "pass",
        "source_split_label": split_label,
        "source_image_embedding_dir": str(image_embedding_dir),
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "num_source_images": len(image_ids),
        "num_attributes": len(attribute_names),
        "embedding_shape": list(directions.shape),
        "epsilon": epsilon,
        "formula": (
            "reliability = ||mu_pos - mu_neg|| / "
            "(avg||x_pos - mu_pos|| + avg||x_neg - mu_neg|| + epsilon)"
        ),
        "label_convention": {"present": 1, "absent": -1},
        "artifacts": {
            "directions": "visual_directions_reliable",
            "raw_directions": "visual_directions_raw",
            "reliability": "visual_direction_reliability",
        },
        "attributes": metrics,
    }
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
