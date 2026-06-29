"""Compute visual attribute directions from labeled image embeddings."""

import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from embedding_io import l2_normalize, load_embeddings, save_embeddings


ID_COLUMNS = {"image_id", "split"}


def load_split_metadata(metadata_path: str | Path, split_label: str = "0") -> tuple[list[str], list[dict[str, int]]]:
    """Load metadata rows for one subset split and verify labels are -1/+1."""
    with Path(metadata_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        attribute_names = [name for name in reader.fieldnames if name not in ID_COLUMNS]
        rows = [row for row in reader if row["split"] == split_label]

    if not rows:
        raise ValueError(f"No metadata rows found for split label {split_label!r}.")

    label_rows: list[dict[str, int]] = []
    bad_values: dict[str, set[int]] = {name: set() for name in attribute_names}
    for row in rows:
        labels = {}
        for attribute in attribute_names:
            value = int(row[attribute])
            labels[attribute] = value
            if value not in {-1, 1}:
                bad_values[attribute].add(value)
        label_rows.append(labels)

    bad_values = {name: values for name, values in bad_values.items() if values}
    if bad_values:
        examples = {name: sorted(values) for name, values in list(bad_values.items())[:5]}
        raise ValueError(f"Attribute labels must be -1/+1, found bad values: {examples}")

    return attribute_names, label_rows


def compute_visual_directions(
    image_embeddings: torch.Tensor,
    image_ids: list[int],
    attribute_names: list[str],
    label_rows: list[dict[str, int]],
) -> tuple[torch.Tensor, list[str], list[dict[str, Any]]]:
    """Compute normalize(mean positive - mean negative) for each attribute."""
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

    directions = []
    metrics = []
    for col, attribute in enumerate(attribute_names):
        values = labels[:, col]
        pos_mask = values == 1
        neg_mask = values == -1
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())
        if n_pos == 0 or n_neg == 0:
            raise ValueError(f"Attribute {attribute!r} needs both positive and negative examples.")

        pos_mean = embeddings[pos_mask].mean(dim=0)
        neg_mean = embeddings[neg_mask].mean(dim=0)
        raw_direction = pos_mean - neg_mean
        raw_norm = float(torch.linalg.norm(raw_direction).item())
        separation = float(1.0 - F.cosine_similarity(pos_mean.unsqueeze(0), neg_mean.unsqueeze(0)).item())

        directions.append(l2_normalize(raw_direction))
        metrics.append(
            {
                "attribute": attribute,
                "support_positive": n_pos,
                "support_negative": n_neg,
                "balance_score": min(n_pos, n_neg) / max(n_pos, n_neg),
                "direction_norm_before_normalization": raw_norm,
                "separation_score": separation,
            }
        )

    return torch.stack(directions), attribute_names, metrics


def compute_and_save_visual_directions(
    metadata_path: str | Path,
    image_embedding_dir: str | Path,
    output_dir: str | Path,
    metrics_path: str | Path,
    split_label: str = "0",
) -> dict[str, Any]:
    """Compute visual directions from train embeddings and save tensors plus metrics."""
    attribute_names, label_rows = load_split_metadata(metadata_path, split_label)
    image_embeddings, image_ids = load_embeddings(image_embedding_dir, "image_embeddings", "cpu")
    image_ids = [int(image_id) for image_id in image_ids]

    directions, ids, metrics = compute_visual_directions(
        image_embeddings=image_embeddings,
        image_ids=image_ids,
        attribute_names=attribute_names,
        label_rows=label_rows,
    )
    save_embeddings(directions, ids, output_dir, "visual_directions")

    payload = {
        "status": "pass",
        "source_split_label": split_label,
        "source_image_embedding_dir": str(image_embedding_dir),
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "num_source_images": len(image_ids),
        "num_attributes": len(attribute_names),
        "embedding_shape": list(directions.shape),
        "formula": "normalize(mean(image embeddings where attribute=+1) - mean(image embeddings where attribute=-1))",
        "label_convention": {"present": 1, "absent": -1},
        "attributes": metrics,
    }
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
