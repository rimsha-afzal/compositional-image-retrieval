import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.reliability_hybrid_fusion_retrieval import run_reliability_hybrid_fusion_retrieval


def default_output_dir(
    output_root: str,
    run_label: str,
    alpha: float,
    beta_text: float,
    beta_visual_pos: float,
    beta_visual_neg: float,
) -> Path:
    """Build a compact output folder name for one reliability-weighted hybrid run."""
    return (
        Path(output_root)
        / (
            f"{run_label}_alpha{alpha:.2f}_"
            f"btext{beta_text:.2f}_"
            f"bvpos{beta_visual_pos:.2f}_"
            f"bvneg{beta_visual_neg:.2f}"
        )
    )


def parse_float_grid(value: str) -> list[float]:
    """Parse comma-separated float values from the CLI."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run reliability-weighted hybrid CLIP text + CelebA visual-direction retrieval."
    )
    parser.add_argument("--query-json", default="data/celeba_subset/queries/test_embedding_celeba_evaluation.json")
    parser.add_argument("--image-embedding-dir", default="data/celeba_subset/embeddings/test")
    parser.add_argument("--text-embedding-dir", default="data/celeba_subset/embeddings")
    parser.add_argument("--text-embedding-name", default="text_embeddings")
    parser.add_argument("--visual-direction-dir", default="data/celeba_subset/embeddings")
    parser.add_argument("--visual-direction-name", default="visual_directions_reliable")
    parser.add_argument("--reliability-name", default="visual_direction_reliability")
    parser.add_argument("--output-root", default="outputs/reliability_hybrid_fusion_run")
    parser.add_argument("--run-label", default="test_subset")
    parser.add_argument("--alpha-values", default="1.0")
    parser.add_argument("--beta-text-values", default="2.0")
    parser.add_argument("--beta-visual-pos-values", default="0.5")
    parser.add_argument("--beta-visual-neg-values", default="0.25")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-run-config", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    alpha_values = parse_float_grid(args.alpha_values)
    beta_text_values = parse_float_grid(args.beta_text_values)
    beta_visual_pos_values = parse_float_grid(args.beta_visual_pos_values)
    beta_visual_neg_values = parse_float_grid(args.beta_visual_neg_values)
    rows = []

    total = (
        len(alpha_values)
        * len(beta_text_values)
        * len(beta_visual_pos_values)
        * len(beta_visual_neg_values)
    )
    run_index = 0
    for alpha in alpha_values:
        for beta_text in beta_text_values:
            for beta_visual_pos in beta_visual_pos_values:
                for beta_visual_neg in beta_visual_neg_values:
                    run_index += 1
                    relative_output = default_output_dir(
                        args.output_root,
                        args.run_label,
                        alpha,
                        beta_text,
                        beta_visual_pos,
                        beta_visual_neg,
                    )
                    result = run_reliability_hybrid_fusion_retrieval(
                        query_json=PROJECT_ROOT / args.query_json,
                        image_embedding_dir=PROJECT_ROOT / args.image_embedding_dir,
                        text_embedding_dir=PROJECT_ROOT / args.text_embedding_dir,
                        visual_direction_dir=PROJECT_ROOT / args.visual_direction_dir,
                        output_dir=PROJECT_ROOT / relative_output,
                        text_embedding_name=args.text_embedding_name,
                        visual_direction_name=args.visual_direction_name,
                        reliability_name=args.reliability_name,
                        alpha=alpha,
                        beta_text=beta_text,
                        beta_visual_pos=beta_visual_pos,
                        beta_visual_neg=beta_visual_neg,
                        batch_size=args.batch_size,
                        device=args.device,
                        save_predictions=args.save_predictions,
                        save_run_config=args.save_run_config,
                    )
                    row = {
                        "experiment_name": result["method"],
                        "alpha": alpha,
                        "beta_text": beta_text,
                        "beta_pos": beta_visual_pos,
                        "beta_neg": beta_visual_neg,
                        "output_dir": str(relative_output),
                        **result["metrics"],
                        "notes": "train-split reliability scales each visual attribute direction",
                    }
                    rows.append(row)
                    print(
                        f"[{run_index:02d}/{total:02d}] "
                        f"alpha={alpha:.2f} beta_text={beta_text:.2f} "
                        f"beta_pos={beta_visual_pos:.2f} "
                        f"beta_neg={beta_visual_neg:.2f} "
                        f"recall@1={result['metrics']['recall@1']:.4f} "
                        f"recall@5={result['metrics']['recall@5']:.4f} "
                        f"recall@10={result['metrics']['recall@10']:.4f} "
                        f"precision@10={result['metrics']['precision@10']:.4f}"
                    )

    rows = sorted(rows, key=lambda row: row["recall@10"], reverse=True)
    requested_columns = [
        "experiment_name",
        "alpha",
        "beta_text",
        "beta_pos",
        "beta_neg",
        "recall@1",
        "recall@5",
        "recall@10",
        "precision@10",
        "notes",
    ]
    extra_columns = [column for column in rows[0].keys() if column not in requested_columns]
    summary_path = output_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=requested_columns + extra_columns)
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    print(f"\nSaved summary to {summary_path}")
    print("\nBest reliability-weighted hybrid by recall@10:")
    print(
        f"  alpha={best['alpha']:.2f} "
        f"beta_text={best['beta_text']:.2f} "
        f"beta_pos={best['beta_pos']:.2f} "
        f"beta_neg={best['beta_neg']:.2f} "
        f"recall@1={best['recall@1']:.4f} "
        f"recall@5={best['recall@5']:.4f} "
        f"recall@10={best['recall@10']:.4f} "
        f"precision@10={best['precision@10']:.4f}"
    )


if __name__ == "__main__":
    main()
