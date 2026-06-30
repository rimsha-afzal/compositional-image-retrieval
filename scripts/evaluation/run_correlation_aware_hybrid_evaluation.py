import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.reliability_hybrid_fusion_retrieval import run_reliability_hybrid_fusion_retrieval


def parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


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
        PROJECT_ROOT / "outputs/hybrid_fusion_run/summary.csv",
    ]
    for candidate in candidates:
        config = read_best_config(candidate)
        if config is not None:
            return config
    raise FileNotFoundError("Could not find a previous hybrid/reliability summary CSV.")


def default_output_dir(output_root: str, run_label: str, config: dict[str, float]) -> Path:
    return Path(output_root) / (
        f"{run_label}_alpha{config['alpha']:.2f}_"
        f"btext{config['beta_text']:.2f}_"
        f"bvpos{config['beta_visual_pos']:.2f}_"
        f"bvneg{config['beta_visual_neg']:.2f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate correlation-aware reliability hybrid retrieval.")
    parser.add_argument("--query-json", default="data/celeba_subset/queries/test_embedding_celeba_evaluation.json")
    parser.add_argument("--image-embedding-dir", default="data/celeba_subset/embeddings/test")
    parser.add_argument("--text-embedding-dir", default="data/celeba_subset/embeddings")
    parser.add_argument("--text-embedding-name", default="text_embeddings")
    parser.add_argument("--decorrelated-direction-dir", default="outputs/decorrelated_directions")
    parser.add_argument("--decorrelated-direction-name", default="decorrelated_visual_directions")
    parser.add_argument("--reliability-name", default="visual_direction_reliability")
    parser.add_argument("--output-root", default="outputs/correlation_aware_hybrid_fusion")
    parser.add_argument("--run-label", default="test_subset")
    parser.add_argument("--direction-cosine-threshold", type=float, default=0.30)
    parser.add_argument("--max-confounders-per-attribute", type=int, default=3)
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--beta-visual-pos-values", default="")
    parser.add_argument("--beta-visual-neg-values", default="")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    base_config = load_default_config()
    beta_pos_values = [base_config["beta_visual_pos"]]
    beta_neg_values = [base_config["beta_visual_neg"]]
    if args.grid_search:
        beta_pos_values = parse_float_grid(args.beta_visual_pos_values) if args.beta_visual_pos_values else [
            base_config["beta_visual_pos"] * 0.75,
            base_config["beta_visual_pos"],
            base_config["beta_visual_pos"] * 1.25,
        ]
        beta_neg_values = parse_float_grid(args.beta_visual_neg_values) if args.beta_visual_neg_values else [
            base_config["beta_visual_neg"] * 0.5,
            base_config["beta_visual_neg"],
            base_config["beta_visual_neg"] * 2.0,
        ]

    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for beta_pos in beta_pos_values:
        for beta_neg in beta_neg_values:
            config = {
                **base_config,
                "beta_visual_pos": beta_pos,
                "beta_visual_neg": beta_neg,
            }
            result = run_reliability_hybrid_fusion_retrieval(
                query_json=PROJECT_ROOT / args.query_json,
                image_embedding_dir=PROJECT_ROOT / args.image_embedding_dir,
                text_embedding_dir=PROJECT_ROOT / args.text_embedding_dir,
                visual_direction_dir=PROJECT_ROOT / args.decorrelated_direction_dir,
                output_dir=PROJECT_ROOT / default_output_dir(args.output_root, args.run_label, config),
                text_embedding_name=args.text_embedding_name,
                visual_direction_name=args.decorrelated_direction_name,
                reliability_name=args.reliability_name,
                alpha=config["alpha"],
                beta_text=config["beta_text"],
                beta_visual_pos=config["beta_visual_pos"],
                beta_visual_neg=config["beta_visual_neg"],
                batch_size=args.batch_size,
                device=args.device,
            )
            row = {
                "experiment_name": "correlation_aware_reliability_hybrid_text_visual_fusion",
                "alpha": config["alpha"],
                "beta_text": config["beta_text"],
                "beta_visual_pos": config["beta_visual_pos"],
                "beta_visual_neg": config["beta_visual_neg"],
                "direction_type": "decorrelated_train_split_directions",
                "reliability_weighted": True,
                "direction_cosine_threshold": args.direction_cosine_threshold,
                "max_confounders_per_attribute": args.max_confounders_per_attribute,
                "recall@1": result["metrics"]["recall@1"],
                "recall@5": result["metrics"]["recall@5"],
                "recall@10": result["metrics"]["recall@10"],
                "precision@10": result["metrics"]["precision@10"],
                "notes": "train-split reliability plus conservative correlation-aware projection removal",
            }
            rows.append(row)
            print(
                f"alpha={row['alpha']:.2f} beta_text={row['beta_text']:.2f} "
                f"beta_pos={row['beta_visual_pos']:.2f} beta_neg={row['beta_visual_neg']:.2f} "
                f"recall@10={row['recall@10']:.4f} precision@10={row['precision@10']:.4f}"
            )

    rows = sorted(rows, key=lambda row: row["recall@10"], reverse=True)
    results_path = output_root / "results.csv"
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("saved results to:", results_path)
    print("best row:", rows[0])


if __name__ == "__main__":
    main()
