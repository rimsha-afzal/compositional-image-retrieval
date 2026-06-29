"""Simple CLIP arithmetic baseline for compositional retrieval."""

import csv
import json
from pathlib import Path
from typing import Any

import torch

from embedding_io import l2_normalize, load_embeddings
from evaluation_metrics import average_metrics, evaluate_single_ranking


def parse_query_string(query: str) -> tuple[list[str], list[str]]:
    """Split a signed query string into positive and negative attributes."""
    positive = []
    negative = []
    for part in query.split(","):
        part = part.strip()
        if part.startswith("+"):
            positive.append(part[1:].strip())
        elif part.startswith("-"):
            negative.append(part[1:].strip())
    return positive, negative


def attribute_to_prompt(attribute: str) -> str:
    """Convert a CelebA attribute name into the saved CLIP text prompt."""
    return f"a photo of a person who is {attribute.replace('_', ' ').lower()}"


def build_query_embeddings(
    reference_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor | None,
    negative_embeddings: torch.Tensor | None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Apply image + positive text - negative text arithmetic fusion."""
    queries = reference_embeddings.float()
    if positive_embeddings is not None and positive_embeddings.numel() > 0:
        queries = queries + alpha * positive_embeddings.float().sum(dim=0, keepdim=True)
    if negative_embeddings is not None and negative_embeddings.numel() > 0:
        queries = queries - beta * negative_embeddings.float().sum(dim=0, keepdim=True)
    return l2_normalize(queries)


def retrieve_top_k_batch(
    query_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    image_ids: list[int],
    k: int,
    exclude_ids: list[int],
    device: str = "cpu",
) -> list[list[tuple[int, float]]]:
    """Retrieve top K image IDs for each query embedding."""
    query_embeddings = query_embeddings.to(device).float()
    image_embeddings = image_embeddings.to(device).float()
    similarities = query_embeddings @ image_embeddings.T

    id_to_row = {int(image_id): row for row, image_id in enumerate(image_ids)}
    for query_row, image_id in enumerate(exclude_ids):
        row = id_to_row.get(int(image_id))
        if row is not None:
            similarities[query_row, row] = -torch.inf

    values, indices = torch.topk(similarities, k=k, dim=1)
    results = []
    for row in range(query_embeddings.shape[0]):
        results.append(
            [
                (int(image_ids[index]), float(score))
                for index, score in zip(indices[row].cpu().tolist(), values[row].cpu())
            ]
        )
    return results


def load_ground_truth(path: str | Path) -> list[dict[str, Any]]:
    """Load a non-empty query/ground-truth JSON file."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty ground-truth file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Ground-truth JSON must contain a list of query records.")
    return data


def run_baseline(
    query_json: str | Path,
    image_embedding_dir: str | Path,
    text_embedding_dir: str | Path,
    output_dir: str | Path,
    alpha: float = 1.0,
    beta: float = 1.0,
    top_k: tuple[int, ...] = (1, 5, 10),
    batch_size: int = 256,
    device: str = "cpu",
    save_predictions: bool = False,
    save_run_config: bool = False,
) -> dict[str, Any]:
    """Run the arithmetic baseline and save metrics, optionally predictions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_embeddings, image_ids = load_embeddings(image_embedding_dir, "image_embeddings", "cpu")
    text_embeddings, text_ids = load_embeddings(text_embedding_dir, "text_embeddings", "cpu")
    image_ids = [int(image_id) for image_id in image_ids]
    image_row = {image_id: row for row, image_id in enumerate(image_ids)}
    text_lookup = {str(prompt): text_embeddings[row] for row, prompt in enumerate(text_ids)}
    entries = load_ground_truth(query_json)

    max_k = max(top_k)
    metric_rows = []
    prediction_rows = [] if save_predictions else None

    for entry in entries:
        positive, negative = parse_query_string(entry["query"])
        positive_embeddings = torch.stack(
            [text_lookup[attribute_to_prompt(attribute)] for attribute in positive]
        ) if positive else None
        negative_embeddings = torch.stack(
            [text_lookup[attribute_to_prompt(attribute)] for attribute in negative]
        ) if negative else None

        reference_items = [(int(reference), targets) for reference, targets in entry["ground_truth"].items()]
        for start in range(0, len(reference_items), batch_size):
            batch = reference_items[start:start + batch_size]
            references = [reference for reference, _ in batch]
            reference_rows = [image_row[reference] for reference in references]
            reference_embeddings = image_embeddings[reference_rows]
            query_embeddings = build_query_embeddings(
                reference_embeddings,
                positive_embeddings,
                negative_embeddings,
                alpha=alpha,
                beta=beta,
            )
            rankings = retrieve_top_k_batch(
                query_embeddings,
                image_embeddings,
                image_ids,
                k=max_k,
                exclude_ids=references,
                device=device,
            )

            for (reference, targets), ranking in zip(batch, rankings):
                retrieved = [image_id for image_id, _ in ranking]
                metrics = evaluate_single_ranking(retrieved, targets, ks=top_k)
                metric_rows.append(metrics)

                if prediction_rows is not None:
                    scores = [score for _, score in ranking]
                    prediction_rows.append(
                        {
                            "query": entry["query"],
                            "reference_index": reference,
                            "target_indices": " ".join(str(int(target)) for target in targets),
                            "retrieved_indices": " ".join(str(index) for index in retrieved),
                            "similarity_scores": " ".join(f"{score:.6f}" for score in scores),
                            **metrics,
                        }
                    )

    metrics = average_metrics(metric_rows)
    result = {
        "method": "clip_arithmetic_baseline",
        "query_json": str(query_json),
        "image_embedding_dir": str(image_embedding_dir),
        "text_embedding_dir": str(text_embedding_dir),
        "alpha": alpha,
        "beta": beta,
        "top_k": list(top_k),
        "query_instances": len(metric_rows),
        "gallery_size": len(image_ids),
        "metrics": metrics,
    }

    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if prediction_rows is not None:
        with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0].keys()))
            writer.writeheader()
            writer.writerows(prediction_rows)
    if save_run_config:
        (output_dir / "run_config.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
