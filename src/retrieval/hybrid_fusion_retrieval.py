"""Hybrid CLIP text + CelebA visual-direction retrieval."""

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


def build_hybrid_queries(
    reference_embeddings: torch.Tensor,
    positive_text_embeddings: torch.Tensor | None,
    negative_text_embeddings: torch.Tensor | None,
    positive_visual_directions: torch.Tensor | None,
    negative_visual_directions: torch.Tensor | None,
    alpha: float = 1.0,
    beta_text: float = 1.0,
    beta_visual_pos: float = 0.5,
    beta_visual_neg: float = 0.25,
) -> torch.Tensor:
    """Build normalized image + text delta + signed visual correction queries."""
    queries = alpha * reference_embeddings.float()

    text_delta = torch.zeros_like(queries[:1])
    if positive_text_embeddings is not None and positive_text_embeddings.numel() > 0:
        text_delta = text_delta + positive_text_embeddings.float().sum(dim=0, keepdim=True)
    if negative_text_embeddings is not None and negative_text_embeddings.numel() > 0:
        text_delta = text_delta - negative_text_embeddings.float().sum(dim=0, keepdim=True)

    if positive_visual_directions is not None and positive_visual_directions.numel() > 0:
        queries = queries + beta_visual_pos * positive_visual_directions.float().sum(dim=0, keepdim=True)
    if negative_visual_directions is not None and negative_visual_directions.numel() > 0:
        queries = queries - beta_visual_neg * negative_visual_directions.float().sum(dim=0, keepdim=True)

    queries = queries + beta_text * text_delta
    return l2_normalize(queries)


def check_hybrid_inputs(
    entries: list[dict[str, Any]],
    image_embeddings: torch.Tensor,
    image_ids: list[int],
    text_embeddings: torch.Tensor,
    text_ids: list[str],
    visual_directions: torch.Tensor,
    visual_ids: list[str],
) -> dict[str, Any]:
    """Verify query attributes, references, targets, and both direction sources."""
    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            f"Image dimension {image_embeddings.shape[1]} does not match text dimension "
            f"{text_embeddings.shape[1]}."
        )
    if image_embeddings.shape[1] != visual_directions.shape[1]:
        raise ValueError(
            f"Image dimension {image_embeddings.shape[1]} does not match visual direction "
            f"dimension {visual_directions.shape[1]}."
        )

    text_prompt_set = {str(prompt) for prompt in text_ids}
    visual_attribute_set = {str(attribute) for attribute in visual_ids}
    image_id_set = {int(image_id) for image_id in image_ids}
    missing_text_prompts = set()
    missing_visual_attributes = set()
    missing_references = set()
    missing_targets = set()
    query_instances = 0

    for entry in entries:
        positive, negative = parse_query_string(entry["query"])
        for attribute in positive + negative:
            prompt = attribute_to_prompt(attribute)
            if prompt not in text_prompt_set:
                missing_text_prompts.add(prompt)
            if attribute not in visual_attribute_set:
                missing_visual_attributes.add(attribute)
        for reference, targets in entry["ground_truth"].items():
            query_instances += 1
            reference_id = int(reference)
            if reference_id not in image_id_set:
                missing_references.add(reference_id)
            missing_targets.update(int(target) for target in targets if int(target) not in image_id_set)

    payload = {
        "status": "pass",
        "image_embedding_shape": list(image_embeddings.shape),
        "text_embedding_shape": list(text_embeddings.shape),
        "visual_direction_shape": list(visual_directions.shape),
        "query_count": len(entries),
        "query_instances": query_instances,
        "missing_text_prompts": sorted(missing_text_prompts),
        "missing_visual_attributes": sorted(missing_visual_attributes),
        "missing_references": sorted(missing_references)[:20],
        "missing_targets": sorted(missing_targets)[:20],
        "num_missing_targets": len(missing_targets),
    }
    if missing_text_prompts or missing_visual_attributes or missing_references or missing_targets:
        payload["status"] = "fail"
        raise ValueError(json.dumps(payload, indent=2))
    return payload


def run_hybrid_fusion_retrieval(
    query_json: str | Path,
    image_embedding_dir: str | Path,
    text_embedding_dir: str | Path,
    visual_direction_dir: str | Path,
    output_dir: str | Path,
    text_embedding_name: str = "text_embeddings",
    alpha: float = 1.0,
    beta_text: float = 1.0,
    beta_visual_pos: float = 0.5,
    beta_visual_neg: float = 0.25,
    top_k: tuple[int, ...] = (1, 5, 10),
    batch_size: int = 256,
    device: str = "cpu",
    save_predictions: bool = False,
    save_run_config: bool = False,
) -> dict[str, Any]:
    """Run query-level hybrid retrieval with CLIP text and visual directions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_embeddings, image_ids = load_embeddings(image_embedding_dir, "image_embeddings", "cpu")
    text_embeddings, text_ids = load_embeddings(text_embedding_dir, text_embedding_name, "cpu")
    visual_directions, visual_ids = load_embeddings(visual_direction_dir, "visual_directions", "cpu")

    image_ids = [int(image_id) for image_id in image_ids]
    text_ids = [str(prompt) for prompt in text_ids]
    visual_ids = [str(attribute) for attribute in visual_ids]
    image_row = {image_id: row for row, image_id in enumerate(image_ids)}
    text_lookup = {prompt: text_embeddings[row] for row, prompt in enumerate(text_ids)}
    visual_lookup = {attribute: visual_directions[row] for row, attribute in enumerate(visual_ids)}
    entries = load_ground_truth(query_json)
    compatibility = check_hybrid_inputs(
        entries=entries,
        image_embeddings=image_embeddings,
        image_ids=image_ids,
        text_embeddings=text_embeddings,
        text_ids=text_ids,
        visual_directions=visual_directions,
        visual_ids=visual_ids,
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

        reference_items = [(int(reference), targets) for reference, targets in entry["ground_truth"].items()]
        for start in range(0, len(reference_items), batch_size):
            batch = reference_items[start:start + batch_size]
            references = [reference for reference, _ in batch]
            reference_rows = [image_row[reference] for reference in references]
            reference_embeddings = image_embeddings[reference_rows]
            query_embeddings = build_hybrid_queries(
                reference_embeddings=reference_embeddings,
                positive_text_embeddings=positive_text,
                negative_text_embeddings=negative_text,
                positive_visual_directions=positive_visual,
                negative_visual_directions=negative_visual,
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
        "method": "hybrid_text_visual_fusion",
        "formula": (
            "normalize(alpha * image + beta_text * (sum(text_pos) - sum(text_neg)) "
            "+ beta_visual_pos * sum(visual_pos) - beta_visual_neg * sum(visual_neg))"
        ),
        "query_json": str(query_json),
        "image_embedding_dir": str(image_embedding_dir),
        "text_embedding_dir": str(text_embedding_dir),
        "text_embedding_name": text_embedding_name,
        "visual_direction_dir": str(visual_direction_dir),
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
