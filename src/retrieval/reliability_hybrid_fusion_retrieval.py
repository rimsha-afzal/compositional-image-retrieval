"""Hybrid CLIP text + reliability-weighted CelebA visual-direction retrieval."""

import csv
import json
from pathlib import Path
from typing import Any

import torch

from baseline_retrieval import (
    attribute_to_prompt,
    load_ground_truth,
    parse_query_string,
    retrieve_top_k_batch,
)
from embedding_io import l2_normalize, load_embeddings
from evaluation_metrics import average_metrics, evaluate_single_ranking
from retrieval.hybrid_fusion_retrieval import check_hybrid_inputs


def build_reliability_hybrid_queries(
    reference_embeddings: torch.Tensor,
    positive_text_embeddings: torch.Tensor | None,
    negative_text_embeddings: torch.Tensor | None,
    positive_visual_directions: torch.Tensor | None,
    negative_visual_directions: torch.Tensor | None,
    positive_reliability: torch.Tensor | None,
    negative_reliability: torch.Tensor | None,
    alpha: float = 1.0,
    beta_text: float = 2.0,
    beta_visual_pos: float = 0.5,
    beta_visual_neg: float = 0.25,
) -> torch.Tensor:
    """Build normalized image + text delta + reliability-weighted visual correction queries."""
    queries = alpha * reference_embeddings.float()

    text_delta = torch.zeros_like(queries[:1])
    if positive_text_embeddings is not None and positive_text_embeddings.numel() > 0:
        text_delta = text_delta + positive_text_embeddings.float().sum(dim=0, keepdim=True)
    if negative_text_embeddings is not None and negative_text_embeddings.numel() > 0:
        text_delta = text_delta - negative_text_embeddings.float().sum(dim=0, keepdim=True)

    if positive_visual_directions is not None and positive_visual_directions.numel() > 0:
        weighted_positive = positive_visual_directions.float() * positive_reliability.float().view(-1, 1)
        queries = queries + beta_visual_pos * weighted_positive.sum(dim=0, keepdim=True)
    if negative_visual_directions is not None and negative_visual_directions.numel() > 0:
        weighted_negative = negative_visual_directions.float() * negative_reliability.float().view(-1, 1)
        queries = queries - beta_visual_neg * weighted_negative.sum(dim=0, keepdim=True)

    queries = queries + beta_text * text_delta
    return l2_normalize(queries)


def check_reliability_inputs(
    entries: list[dict[str, Any]],
    image_embeddings: torch.Tensor,
    image_ids: list[int],
    text_embeddings: torch.Tensor,
    text_ids: list[str],
    visual_directions: torch.Tensor,
    visual_ids: list[str],
    reliability_scores: torch.Tensor,
    reliability_ids: list[str],
) -> dict[str, Any]:
    """Verify hybrid inputs and matching reliability rows."""
    payload = check_hybrid_inputs(
        entries=entries,
        image_embeddings=image_embeddings,
        image_ids=image_ids,
        text_embeddings=text_embeddings,
        text_ids=text_ids,
        visual_directions=visual_directions,
        visual_ids=visual_ids,
    )
    if reliability_scores.shape[0] != len(reliability_ids):
        raise ValueError("Reliability score rows and IDs have different lengths.")
    if reliability_scores.dim() not in {1, 2}:
        raise ValueError("Reliability scores must be a 1D tensor or a single-column 2D tensor.")
    if set(visual_ids) != set(reliability_ids):
        missing = sorted(set(visual_ids) - set(reliability_ids))
        extra = sorted(set(reliability_ids) - set(visual_ids))
        raise ValueError(
            "Visual direction IDs and reliability IDs must match. "
            f"Missing reliability IDs: {missing[:20]}; extra reliability IDs: {extra[:20]}"
        )
    payload["reliability_shape"] = list(reliability_scores.shape)
    return payload


def run_reliability_hybrid_fusion_retrieval(
    query_json: str | Path,
    image_embedding_dir: str | Path,
    text_embedding_dir: str | Path,
    visual_direction_dir: str | Path,
    output_dir: str | Path,
    text_embedding_name: str = "text_embeddings",
    visual_direction_name: str = "visual_directions_reliable",
    reliability_name: str = "visual_direction_reliability",
    alpha: float = 1.0,
    beta_text: float = 2.0,
    beta_visual_pos: float = 0.5,
    beta_visual_neg: float = 0.25,
    top_k: tuple[int, ...] = (1, 5, 10),
    batch_size: int = 256,
    device: str = "cpu",
    save_predictions: bool = False,
    save_run_config: bool = False,
) -> dict[str, Any]:
    """Run hybrid retrieval with visual directions scaled by per-attribute reliability."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_embeddings, image_ids = load_embeddings(image_embedding_dir, "image_embeddings", "cpu")
    text_embeddings, text_ids = load_embeddings(text_embedding_dir, text_embedding_name, "cpu")
    visual_directions, visual_ids = load_embeddings(visual_direction_dir, visual_direction_name, "cpu")
    reliability_scores, reliability_ids = load_embeddings(visual_direction_dir, reliability_name, "cpu")

    if reliability_scores.dim() == 2:
        if reliability_scores.shape[1] != 1:
            raise ValueError("Reliability score tensor must have exactly one column.")
        reliability_scores = reliability_scores.squeeze(1)

    image_ids = [int(image_id) for image_id in image_ids]
    text_ids = [str(prompt) for prompt in text_ids]
    visual_ids = [str(attribute) for attribute in visual_ids]
    reliability_ids = [str(attribute) for attribute in reliability_ids]
    image_row = {image_id: row for row, image_id in enumerate(image_ids)}
    text_lookup = {prompt: text_embeddings[row] for row, prompt in enumerate(text_ids)}
    visual_lookup = {attribute: visual_directions[row] for row, attribute in enumerate(visual_ids)}
    reliability_lookup = {
        attribute: reliability_scores[row] for row, attribute in enumerate(reliability_ids)
    }
    entries = load_ground_truth(query_json)
    compatibility = check_reliability_inputs(
        entries=entries,
        image_embeddings=image_embeddings,
        image_ids=image_ids,
        text_embeddings=text_embeddings,
        text_ids=text_ids,
        visual_directions=visual_directions,
        visual_ids=visual_ids,
        reliability_scores=reliability_scores,
        reliability_ids=reliability_ids,
    )

    max_k = max(top_k)
    metric_rows = []
    prediction_rows = [] if save_predictions else None

    for entry in entries:
        positive, negative = parse_query_string(entry["query"])
        positive_text = torch.stack(
            [text_lookup[attribute_to_prompt(attribute)] for attribute in positive]
        ) if positive else None
        negative_text = torch.stack(
            [text_lookup[attribute_to_prompt(attribute)] for attribute in negative]
        ) if negative else None
        positive_visual = torch.stack([visual_lookup[attribute] for attribute in positive]) if positive else None
        negative_visual = torch.stack([visual_lookup[attribute] for attribute in negative]) if negative else None
        positive_reliability = torch.stack([reliability_lookup[attribute] for attribute in positive]) if positive else None
        negative_reliability = torch.stack([reliability_lookup[attribute] for attribute in negative]) if negative else None

        reference_items = [(int(reference), targets) for reference, targets in entry["ground_truth"].items()]
        for start in range(0, len(reference_items), batch_size):
            batch = reference_items[start:start + batch_size]
            references = [reference for reference, _ in batch]
            reference_rows = [image_row[reference] for reference in references]
            reference_embeddings = image_embeddings[reference_rows]
            query_embeddings = build_reliability_hybrid_queries(
                reference_embeddings=reference_embeddings,
                positive_text_embeddings=positive_text,
                negative_text_embeddings=negative_text,
                positive_visual_directions=positive_visual,
                negative_visual_directions=negative_visual,
                positive_reliability=positive_reliability,
                negative_reliability=negative_reliability,
                alpha=alpha,
                beta_text=beta_text,
                beta_visual_pos=beta_visual_pos,
                beta_visual_neg=beta_visual_neg,
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
        "method": "reliability_hybrid_text_visual_fusion",
        "formula": (
            "normalize(alpha * image + beta_text * (sum(text_pos) - sum(text_neg)) "
            "+ beta_visual_pos * sum(r_a * visual_pos_a) "
            "- beta_visual_neg * sum(r_a * visual_neg_a))"
        ),
        "query_json": str(query_json),
        "image_embedding_dir": str(image_embedding_dir),
        "text_embedding_dir": str(text_embedding_dir),
        "text_embedding_name": text_embedding_name,
        "visual_direction_dir": str(visual_direction_dir),
        "visual_direction_name": visual_direction_name,
        "reliability_name": reliability_name,
        "alpha": alpha,
        "beta_text": beta_text,
        "beta_visual_pos": beta_visual_pos,
        "beta_visual_neg": beta_visual_neg,
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
