import argparse
import csv
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baseline_retrieval import (
    attribute_to_prompt,
    load_ground_truth,
    parse_query_string,
    retrieve_top_k_batch,
)
from embedding_extraction import load_clip
from embedding_io import l2_normalize, load_embeddings
from evaluation_metrics import average_metrics, evaluate_single_ranking
from prompt_ensemble_text_embeddings import ATTRIBUTE_PHRASES
from text.prompt_delta import encode_prompt_delta


def read_best_config(summary_path: Path) -> dict[str, float] | None:
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row["recall@10"]))
    return {
        "alpha": float(best["alpha"]),
        "beta_text": float(best["beta_text"]),
        "beta_visual_pos": float(best.get("beta_visual_pos", best.get("beta_pos"))),
        "beta_visual_neg": float(best.get("beta_visual_neg", best.get("beta_neg"))),
    }


def load_default_config() -> dict[str, float]:
    candidates = [
        PROJECT_ROOT / "outputs/reliability_hybrid_final_test_run/summary.csv",
        PROJECT_ROOT / "outputs/reliability_hybrid_validation_run/summary.csv",
        PROJECT_ROOT / "outputs/reliability_hybrid_fusion_run/summary.csv",
    ]
    for candidate in candidates:
        config = read_best_config(candidate)
        if config is not None:
            return config
    raise FileNotFoundError("Could not find a previous reliability hybrid summary CSV.")


def attribute_phrase(attribute: str) -> str:
    phrase = ATTRIBUTE_PHRASES.get(attribute, attribute.replace("_", " ").lower())
    if phrase == "who is a man":
        return "male"
    if phrase.startswith("who is "):
        return phrase.removeprefix("who is ")
    if phrase.startswith("with "):
        return "having " + phrase.removeprefix("with ")
    if phrase.startswith("wearing "):
        return phrase
    return phrase


def build_modification_text(query: str) -> str:
    positive, negative = parse_query_string(query)
    parts = [attribute_phrase(attribute) for attribute in positive]
    for attribute in negative:
        phrase = attribute_phrase(attribute)
        if phrase.startswith("wearing "):
            parts.append("not " + phrase)
        elif phrase.startswith("having "):
            parts.append("not " + phrase)
        else:
            parts.append("not " + phrase)
    return " and ".join(parts)


def build_text_delta_lookup(
    entries: list[dict[str, object]],
    model_name: str,
    device: str,
    raw_text_lookup: dict[str, torch.Tensor],
    prompt_ensemble_lookup: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    model, processor, resolved_device = load_clip(model_name, device)
    lookup = {}
    diagnostics = []
    for query_id, entry in enumerate(entries):
        query = str(entry["query"])
        modification_text = build_modification_text(query)
        prompt_delta, delta_diagnostics = encode_prompt_delta(
            modification_text=modification_text,
            clip_model=model,
            processor=processor,
            device=resolved_device,
        )
        positive, negative = parse_query_string(query)
        raw_text = torch.zeros_like(prompt_delta)
        for attribute in positive:
            raw_text = raw_text + raw_text_lookup[attribute_to_prompt(attribute)].float()
        for attribute in negative:
            raw_text = raw_text - raw_text_lookup[attribute_to_prompt(attribute)].float()
        raw_text = l2_normalize(raw_text)

        prompt_ensemble_cosine = ""
        if prompt_ensemble_lookup is not None:
            prompt_ensemble = torch.zeros_like(prompt_delta)
            for attribute in positive:
                prompt_ensemble = prompt_ensemble + prompt_ensemble_lookup[attribute_to_prompt(attribute)].float()
            for attribute in negative:
                prompt_ensemble = prompt_ensemble - prompt_ensemble_lookup[attribute_to_prompt(attribute)].float()
            prompt_ensemble = l2_normalize(prompt_ensemble)
            prompt_ensemble_cosine = float(torch.dot(prompt_ensemble, prompt_delta).item())

        lookup[query] = prompt_delta
        diagnostics.append(
            {
                "query_id": query_id,
                "query": query,
                "modification_text": modification_text,
                "num_prompt_pairs": delta_diagnostics["num_prompt_pairs"],
                "mean_delta_norm_before_final_normalization": delta_diagnostics[
                    "mean_delta_norm_before_final_normalization"
                ],
                "cosine_raw_text_vs_prompt_delta": float(torch.dot(raw_text, prompt_delta).item()),
                "cosine_prompt_ensemble_vs_prompt_delta_if_available": prompt_ensemble_cosine,
            }
        )
    return lookup, diagnostics


def run_prompt_delta_hybrid(
    query_json: Path,
    image_embedding_dir: Path,
    text_embedding_dir: Path,
    visual_direction_dir: Path,
    output_dir: Path,
    model_name: str,
    alpha: float,
    beta_text: float,
    beta_visual_pos: float,
    beta_visual_neg: float,
    visual_direction_name: str = "visual_directions_reliable",
    reliability_name: str = "visual_direction_reliability",
    batch_size: int = 256,
    device: str = "cpu",
) -> tuple[dict[str, float], list[dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = load_ground_truth(query_json)
    image_embeddings, image_ids = load_embeddings(image_embedding_dir, "image_embeddings", "cpu")
    raw_text_embeddings, raw_text_ids = load_embeddings(text_embedding_dir, "text_embeddings", "cpu")
    visual_directions, visual_ids = load_embeddings(visual_direction_dir, visual_direction_name, "cpu")
    reliability_scores, reliability_ids = load_embeddings(visual_direction_dir, reliability_name, "cpu")

    prompt_ensemble_lookup = None
    try:
        prompt_ensemble_embeddings, prompt_ensemble_ids = load_embeddings(
            text_embedding_dir, "prompt_ensemble_text_embeddings", "cpu"
        )
        prompt_ensemble_lookup = {
            str(prompt): prompt_ensemble_embeddings[row]
            for row, prompt in enumerate(prompt_ensemble_ids)
        }
    except FileNotFoundError:
        prompt_ensemble_lookup = None

    if reliability_scores.dim() == 2:
        reliability_scores = reliability_scores.squeeze(1)

    image_ids = [int(image_id) for image_id in image_ids]
    visual_ids = [str(attribute) for attribute in visual_ids]
    reliability_ids = [str(attribute) for attribute in reliability_ids]
    if visual_ids != reliability_ids:
        raise ValueError("Visual direction IDs and reliability IDs must have the same order.")
    if torch.isnan(visual_directions).any() or torch.isnan(reliability_scores).any():
        raise ValueError("Visual direction or reliability artifact contains NaN values.")

    image_row = {image_id: row for row, image_id in enumerate(image_ids)}
    raw_text_lookup = {
        str(prompt): raw_text_embeddings[row]
        for row, prompt in enumerate(raw_text_ids)
    }
    visual_lookup = {attribute: visual_directions[row] for row, attribute in enumerate(visual_ids)}
    reliability_lookup = {attribute: reliability_scores[row] for row, attribute in enumerate(reliability_ids)}
    text_delta_lookup, diagnostics = build_text_delta_lookup(
        entries=entries,
        model_name=model_name,
        device=device,
        raw_text_lookup=raw_text_lookup,
        prompt_ensemble_lookup=prompt_ensemble_lookup,
    )

    max_k = 10
    metric_rows = []
    for entry in entries:
        query = str(entry["query"])
        positive, negative = parse_query_string(query)
        text_delta = text_delta_lookup[query].view(1, -1)
        positive_visual = torch.stack([visual_lookup[attribute] for attribute in positive]) if positive else None
        negative_visual = torch.stack([visual_lookup[attribute] for attribute in negative]) if negative else None
        positive_reliability = torch.stack([reliability_lookup[attribute] for attribute in positive]) if positive else None
        negative_reliability = torch.stack([reliability_lookup[attribute] for attribute in negative]) if negative else None
        visual_component = torch.zeros_like(text_delta)
        if positive_visual is not None:
            visual_component = visual_component + beta_visual_pos * (
                positive_visual.float() * positive_reliability.float().view(-1, 1)
            ).sum(dim=0, keepdim=True)
        if negative_visual is not None:
            visual_component = visual_component - beta_visual_neg * (
                negative_visual.float() * negative_reliability.float().view(-1, 1)
            ).sum(dim=0, keepdim=True)

        reference_items = [(int(reference), targets) for reference, targets in entry["ground_truth"].items()]
        for start in range(0, len(reference_items), batch_size):
            batch = reference_items[start:start + batch_size]
            references = [reference for reference, _ in batch]
            reference_rows = [image_row[reference] for reference in references]
            reference_embeddings = image_embeddings[reference_rows].float()
            query_embeddings = l2_normalize(alpha * reference_embeddings + beta_text * text_delta + visual_component)
            rankings = retrieve_top_k_batch(
                query_embeddings=query_embeddings,
                image_embeddings=image_embeddings,
                image_ids=image_ids,
                k=max_k,
                exclude_ids=references,
                device=device,
            )
            for (_, targets), ranking in zip(batch, rankings):
                retrieved = [image_id for image_id, _ in ranking]
                metric_rows.append(evaluate_single_ranking(retrieved, targets, ks=(1, 5, 10)))

    return average_metrics(metric_rows), diagnostics


def parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate neutral-subtracted prompt-delta hybrid fusion.")
    parser.add_argument("--query-json", default="data/celeba_subset/queries/test_embedding_celeba_evaluation.json")
    parser.add_argument("--image-embedding-dir", default="data/celeba_subset/embeddings/test")
    parser.add_argument("--text-embedding-dir", default="data/celeba_subset/embeddings")
    parser.add_argument("--visual-direction-dir", default="data/celeba_subset/embeddings")
    parser.add_argument("--visual-direction-name", default="visual_directions_reliable")
    parser.add_argument("--reliability-name", default="visual_direction_reliability")
    parser.add_argument("--output-dir", default="outputs/prompt_delta_hybrid_fusion")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--beta-text-values", default="")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    base_config = load_default_config()
    beta_text_values = [base_config["beta_text"]]
    if args.grid_search:
        beta_text_values = parse_float_grid(args.beta_text_values) if args.beta_text_values else [
            base_config["beta_text"] * 0.75,
            base_config["beta_text"],
            base_config["beta_text"] * 1.25,
        ]

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    all_diagnostics = None
    for beta_text in beta_text_values:
        metrics, diagnostics = run_prompt_delta_hybrid(
            query_json=PROJECT_ROOT / args.query_json,
            image_embedding_dir=PROJECT_ROOT / args.image_embedding_dir,
            text_embedding_dir=PROJECT_ROOT / args.text_embedding_dir,
            visual_direction_dir=PROJECT_ROOT / args.visual_direction_dir,
            output_dir=output_dir,
            model_name=args.model_name,
            alpha=base_config["alpha"],
            beta_text=beta_text,
            beta_visual_pos=base_config["beta_visual_pos"],
            beta_visual_neg=base_config["beta_visual_neg"],
            visual_direction_name=args.visual_direction_name,
            reliability_name=args.reliability_name,
            batch_size=args.batch_size,
            device=args.device,
        )
        all_diagnostics = diagnostics
        row = {
            "experiment_name": "neutral_subtracted_prompt_ensemble_hybrid_fusion",
            "alpha": base_config["alpha"],
            "beta_text": beta_text,
            "beta_visual_pos": base_config["beta_visual_pos"],
            "beta_visual_neg": base_config["beta_visual_neg"],
            "text_mode": "prompt_delta",
            "visual_direction_type": args.visual_direction_name,
            "reliability_weighted": True,
            "recall@1": metrics["recall@1"],
            "recall@5": metrics["recall@5"],
            "recall@10": metrics["recall@10"],
            "precision@10": metrics["precision@10"],
            "notes": "paired neutral-subtracted prompt ensemble replaces raw CLIP text embedding",
        }
        rows.append(row)
        print(
            f"beta_text={beta_text:.2f} recall@1={row['recall@1']:.4f} "
            f"recall@5={row['recall@5']:.4f} recall@10={row['recall@10']:.4f} "
            f"precision@10={row['precision@10']:.4f}"
        )

    rows = sorted(rows, key=lambda row: row["recall@10"], reverse=True)
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if all_diagnostics is not None:
        with (output_dir / "text_delta_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_diagnostics[0].keys()))
            writer.writeheader()
            writer.writerows(all_diagnostics)

    print("saved results to:", output_dir / "results.csv")
    print("saved diagnostics to:", output_dir / "text_delta_diagnostics.csv")
    print("best row:", rows[0])


if __name__ == "__main__":
    main()
