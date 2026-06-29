"""Visual-direction retrieval for signed CelebA attribute queries."""

import csv
import json
from pathlib import Path
from typing import Any

import torch

from baseline_retrieval import load_ground_truth, parse_query_string, retrieve_top_k_batch
from embedding_io import l2_normalize, load_embeddings
from evaluation_metrics import average_metrics, evaluate_single_ranking


def build_visual_direction_queries(
    reference_embeddings: torch.Tensor,
    positive_directions: torch.Tensor | None,
    negative_directions: torch.Tensor | None,
    alpha: float = 1.0,
    beta_pos: float = 1.0,
    beta_neg: float = 1.0,
) -> torch.Tensor:
    """Build normalized image + positive directions - negative directions queries."""
    queries = alpha * reference_embeddings.float()
    if positive_directions is not None and positive_directions.numel() > 0:
        queries = queries + beta_pos * positive_directions.float().sum(dim=0, keepdim=True)
    if negative_directions is not None and negative_directions.numel() > 0:
        queries = queries - beta_neg * negative_directions.float().sum(dim=0, keepdim=True)
    return l2_normalize(queries)


def check_visual_direction_inputs(
    entries: list[dict[str, Any]],
    image_embeddings: torch.Tensor,
    image_ids: list[int],
    direction_embeddings: torch.Tensor,
    direction_ids: list[str],
) -> dict[str, Any]:
    """Verify query attributes, targets, and visual directions are compatible."""
    if image_embeddings.shape[1] != direction_embeddings.shape[1]:
        raise ValueError(
            f"Image dimension {image_embeddings.shape[1]} does not match direction dimension "
            f"{direction_embeddings.shape[1]}."
        )

    direction_set = {str(attribute) for attribute in direction_ids}
    image_id_set = {int(image_id) for image_id in image_ids}
    missing_attributes = set()
    missing_references = set()
    missing_targets = set()
    query_instances = 0

    for entry in entries:
        positive, negative = parse_query_string(entry["query"])
        missing_attributes.update(attribute for attribute in positive + negative if attribute not in direction_set)
        for reference, targets in entry["ground_truth"].items():
            query_instances += 1
            reference_id = int(reference)
            if reference_id not in image_id_set:
                missing_references.add(reference_id)
            missing_targets.update(int(target) for target in targets if int(target) not in image_id_set)

    payload = {
        "status": "pass",
        "image_embedding_shape": list(image_embeddings.shape),
        "visual_direction_shape": list(direction_embeddings.shape),
        "num_direction_ids": len(direction_ids),
        "query_count": len(entries),
        "query_instances": query_instances,
        "missing_attributes": sorted(missing_attributes),
        "missing_references": sorted(missing_references)[:20],
        "missing_targets": sorted(missing_targets)[:20],
        "num_missing_targets": len(missing_targets),
    }
    if missing_attributes or missing_references or missing_targets:
        payload["status"] = "fail"
        raise ValueError(json.dumps(payload, indent=2))
    return payload


def run_visual_direction_retrieval(
    query_json: str | Path,
    image_embedding_dir: str | Path,
    visual_direction_dir: str | Path,
    output_dir: str | Path,
    alpha: float = 1.0,
    beta_pos: float = 1.0,
    beta_neg: float = 1.0,
    top_k: tuple[int, ...] = (1, 5, 10),
    batch_size: int = 256,
    device: str = "cpu",
    save_predictions: bool = False,
    save_run_config: bool = False,
) -> dict[str, Any]:
    """Run retrieval with precomputed visual attribute directions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_embeddings, image_ids = load_embeddings(image_embedding_dir, "image_embeddings", "cpu")
    direction_embeddings, direction_ids = load_embeddings(visual_direction_dir, "visual_directions", "cpu")
    image_ids = [int(image_id) for image_id in image_ids]
    direction_ids = [str(attribute) for attribute in direction_ids]
    image_row = {image_id: row for row, image_id in enumerate(image_ids)}
    direction_lookup = {attribute: direction_embeddings[row] for row, attribute in enumerate(direction_ids)}
    entries = load_ground_truth(query_json)
    compatibility = check_visual_direction_inputs(
        entries=entries,
        image_embeddings=image_embeddings,
        image_ids=image_ids,
        direction_embeddings=direction_embeddings,
        direction_ids=direction_ids,
    )

    max_k = max(top_k)
    metric_rows = []
    prediction_rows = [] if save_predictions else None

    for entry in entries:
        positive, negative = parse_query_string(entry["query"])
        positive_directions = torch.stack([direction_lookup[attribute] for attribute in positive]) if positive else None
        negative_directions = torch.stack([direction_lookup[attribute] for attribute in negative]) if negative else None

        reference_items = [(int(reference), targets) for reference, targets in entry["ground_truth"].items()]
        for start in range(0, len(reference_items), batch_size):
            batch = reference_items[start:start + batch_size]
            references = [reference for reference, _ in batch]
            reference_rows = [image_row[reference] for reference in references]
            reference_embeddings = image_embeddings[reference_rows]
            query_embeddings = build_visual_direction_queries(
                reference_embeddings=reference_embeddings,
                positive_directions=positive_directions,
                negative_directions=negative_directions,
                alpha=alpha,
                beta_pos=beta_pos,
                beta_neg=beta_neg,
            )
            rankings = retrieve_top_k_batch(
                query_embeddings=query_embeddings,
                image_embeddings=image_embeddings,
                image_ids=image_ids,
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
        "method": "visual_direction_retrieval",
        "formula": "normalize(alpha * image + beta_pos * sum(positive directions) - beta_neg * sum(negative directions))",
        "query_json": str(query_json),
        "image_embedding_dir": str(image_embedding_dir),
        "visual_direction_dir": str(visual_direction_dir),
        "alpha": alpha,
        "beta_pos": beta_pos,
        "beta_neg": beta_neg,
        "top_k": list(top_k),
        "query_instances": len(metric_rows),
        "gallery_size": len(image_ids),
        "compatibility": compatibility,
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
