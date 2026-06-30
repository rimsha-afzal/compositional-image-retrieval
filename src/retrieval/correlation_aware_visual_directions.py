"""Correlation analysis and conservative visual-direction decorrelation."""

import csv
import json
from pathlib import Path
from typing import Any

import torch

from embedding_io import l2_normalize, load_embeddings, save_embeddings
from retrieval.visual_directions import load_split_metadata


def labels_to_tensor(
    attribute_names: list[str],
    label_rows: list[dict[str, int]],
) -> torch.Tensor:
    """Convert metadata label rows to an N x A float tensor."""
    return torch.tensor(
        [[row[attribute] for attribute in attribute_names] for row in label_rows],
        dtype=torch.float32,
    )


def compute_pearson_correlation(labels: torch.Tensor) -> torch.Tensor:
    """Compute Pearson correlation between attribute columns."""
    centered = labels - labels.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-12)
    normalized = centered / std
    return (normalized.T @ normalized) / labels.shape[0]


def compute_direction_cosine_matrix(directions: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarities between direction rows."""
    normalized = l2_normalize(directions.float())
    return normalized @ normalized.T


def save_square_matrix_csv(
    matrix: torch.Tensor,
    attribute_names: list[str],
    output_path: str | Path,
) -> None:
    """Save an attribute-by-attribute matrix as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["attribute", *attribute_names])
        for attribute, row in zip(attribute_names, matrix.tolist()):
            writer.writerow([attribute, *row])


def save_top_pairs_csv(
    matrix: torch.Tensor,
    attribute_names: list[str],
    output_path: str | Path,
) -> list[dict[str, Any]]:
    """Save upper-triangle attribute pairs sorted by absolute value."""
    rows = []
    for i, attribute_a in enumerate(attribute_names):
        for j in range(i + 1, len(attribute_names)):
            rows.append(
                {
                    "attribute_a": attribute_a,
                    "attribute_b": attribute_names[j],
                    "value": float(matrix[i, j].item()),
                }
            )
    rows = sorted(rows, key=lambda row: abs(row["value"]), reverse=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["attribute_a", "attribute_b", "value"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def compute_and_save_correlation_analysis(
    metadata_path: str | Path,
    visual_direction_dir: str | Path,
    output_dir: str | Path,
    split_label: str = "0",
    visual_direction_name: str = "visual_directions_reliable",
) -> dict[str, Any]:
    """Compute train-label correlations and visual-direction cosine similarities."""
    attribute_names, label_rows = load_split_metadata(metadata_path, split_label)
    directions, direction_ids = load_embeddings(visual_direction_dir, visual_direction_name, "cpu")
    direction_ids = [str(attribute) for attribute in direction_ids]
    if attribute_names != direction_ids:
        raise ValueError("Metadata attributes and visual direction IDs must have the same order.")

    labels = labels_to_tensor(attribute_names, label_rows)
    label_correlation = compute_pearson_correlation(labels)
    direction_cosine = compute_direction_cosine_matrix(directions)

    if torch.isnan(label_correlation).any() or torch.isnan(direction_cosine).any():
        raise ValueError("Correlation matrices contain NaN values.")

    output_dir = Path(output_dir)
    save_square_matrix_csv(label_correlation, attribute_names, output_dir / "label_correlation_matrix.csv")
    save_square_matrix_csv(direction_cosine, attribute_names, output_dir / "direction_cosine_matrix.csv")
    top_label = save_top_pairs_csv(label_correlation, attribute_names, output_dir / "top_label_correlations.csv")
    top_direction = save_top_pairs_csv(direction_cosine, attribute_names, output_dir / "top_direction_similarities.csv")

    payload = {
        "status": "pass",
        "source_split_label": split_label,
        "metadata_path": str(metadata_path),
        "visual_direction_dir": str(visual_direction_dir),
        "visual_direction_name": visual_direction_name,
        "output_dir": str(output_dir),
        "num_attributes": len(attribute_names),
        "top_label_correlation": top_label[0] if top_label else None,
        "top_direction_similarity": top_direction[0] if top_direction else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_square_matrix_csv(matrix_path: str | Path) -> tuple[list[str], torch.Tensor]:
    """Load a square matrix saved by save_square_matrix_csv."""
    with Path(matrix_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        attribute_names = header[1:]
        rows = []
        row_names = []
        for row in reader:
            row_names.append(row[0])
            rows.append([float(value) for value in row[1:]])
    if attribute_names != row_names:
        raise ValueError("Matrix CSV row and column attributes do not align.")
    return attribute_names, torch.tensor(rows, dtype=torch.float32)


def decorrelate_direction(
    direction: torch.Tensor,
    confounder_directions: torch.Tensor,
    lambda_reg: float = 1e-4,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, float]:
    """Remove projection of direction onto selected confounder directions."""
    if confounder_directions.numel() == 0:
        cleaned = l2_normalize(direction)
        return cleaned, float(torch.linalg.norm(direction.float()).item())

    c_matrix = l2_normalize(confounder_directions).T
    gram = c_matrix.T @ c_matrix
    regularized = gram + lambda_reg * torch.eye(gram.shape[0], dtype=gram.dtype)
    coefficients = torch.linalg.solve(regularized, c_matrix.T @ direction.float())
    projection = c_matrix @ coefficients
    cleaned = direction.float() - projection
    cleaned_norm = float(torch.linalg.norm(cleaned).item())
    return cleaned / (cleaned_norm + epsilon), cleaned_norm


def compute_decorrelated_directions(
    directions: torch.Tensor,
    attribute_names: list[str],
    cosine_matrix: torch.Tensor,
    direction_cosine_threshold: float = 0.30,
    max_confounders_per_attribute: int = 3,
    lambda_reg: float = 1e-4,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Remove projections onto the top correlated directions for each attribute."""
    directions = l2_normalize(directions.float())
    cleaned_rows = []
    diagnostics = []

    for row, attribute in enumerate(attribute_names):
        scores = cosine_matrix[row].clone()
        scores[row] = 0.0
        candidates = [
            index
            for index in torch.argsort(scores.abs(), descending=True).tolist()
            if abs(float(scores[index].item())) >= direction_cosine_threshold
        ][:max_confounders_per_attribute]

        cleaned, cleaned_norm_before_renormalization = decorrelate_direction(
            direction=directions[row],
            confounder_directions=directions[candidates] if candidates else directions.new_empty((0, directions.shape[1])),
            lambda_reg=lambda_reg,
            epsilon=epsilon,
        )
        if torch.isnan(cleaned).any():
            raise ValueError(f"Cleaned direction for {attribute!r} contains NaN values.")

        cleaned_rows.append(cleaned)
        cleaned_norm = torch.linalg.norm(cleaned).item()
        diagnostics.append(
            {
                "attribute": attribute,
                "original_norm": float(torch.linalg.norm(directions[row]).item()),
                "cleaned_norm_before_renormalization": cleaned_norm_before_renormalization,
                "cleaned_norm": float(cleaned_norm),
                "cosine_original_cleaned": float(torch.dot(directions[row], cleaned).item()),
                "num_removed_correlated_components": len(candidates),
                "removed_correlated_attributes": " ".join(attribute_names[index] for index in candidates),
            }
        )

    cleaned_directions = torch.stack(cleaned_rows)
    norms = torch.linalg.norm(cleaned_directions, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
        raise ValueError("Cleaned directions are not normalized.")
    return cleaned_directions, diagnostics


def compute_and_save_decorrelated_directions(
    visual_direction_dir: str | Path,
    reliability_dir: str | Path,
    direction_cosine_matrix_path: str | Path,
    output_dir: str | Path,
    visual_direction_name: str = "visual_directions_reliable",
    reliability_name: str = "visual_direction_reliability",
    output_direction_name: str = "decorrelated_visual_directions",
    direction_cosine_threshold: float = 0.30,
    max_confounders_per_attribute: int = 3,
    lambda_reg: float = 1e-4,
    epsilon: float = 1e-8,
) -> dict[str, Any]:
    """Create and save conservative correlation-aware visual directions."""
    directions, direction_ids = load_embeddings(visual_direction_dir, visual_direction_name, "cpu")
    reliability, reliability_ids = load_embeddings(reliability_dir, reliability_name, "cpu")
    matrix_ids, cosine_matrix = load_square_matrix_csv(direction_cosine_matrix_path)

    direction_ids = [str(attribute) for attribute in direction_ids]
    reliability_ids = [str(attribute) for attribute in reliability_ids]
    if direction_ids != matrix_ids:
        raise ValueError("Direction IDs and cosine matrix attributes must have the same order.")
    if direction_ids != reliability_ids:
        raise ValueError("Direction IDs and reliability IDs must have the same order.")
    if torch.isnan(directions).any():
        raise ValueError("Input directions contain NaN values.")

    cleaned_directions, diagnostics = compute_decorrelated_directions(
        directions=directions,
        attribute_names=direction_ids,
        cosine_matrix=cosine_matrix,
        direction_cosine_threshold=direction_cosine_threshold,
        max_confounders_per_attribute=max_confounders_per_attribute,
        lambda_reg=lambda_reg,
        epsilon=epsilon,
    )

    output_dir = Path(output_dir)
    save_embeddings(cleaned_directions, direction_ids, output_dir, output_direction_name)
    save_embeddings(reliability, reliability_ids, output_dir, reliability_name)
    save_embeddings(directions, direction_ids, output_dir, "original_visual_directions")

    diagnostics_path = output_dir / "decorrelation_diagnostics.csv"
    with diagnostics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostics)

    config = {
        "status": "pass",
        "visual_direction_dir": str(visual_direction_dir),
        "visual_direction_name": visual_direction_name,
        "reliability_dir": str(reliability_dir),
        "reliability_name": reliability_name,
        "direction_cosine_matrix_path": str(direction_cosine_matrix_path),
        "output_dir": str(output_dir),
        "output_direction_name": output_direction_name,
        "direction_cosine_threshold": direction_cosine_threshold,
        "max_confounders_per_attribute": max_confounders_per_attribute,
        "lambda_reg": lambda_reg,
        "epsilon": epsilon,
        "num_attributes": len(direction_ids),
        "num_attributes_with_removed_components": sum(
            row["num_removed_correlated_components"] > 0 for row in diagnostics
        ),
    }
    (output_dir / "decorrelation_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config
