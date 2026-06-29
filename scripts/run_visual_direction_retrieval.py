import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.visual_direction_retrieval import run_visual_direction_retrieval


def default_output_dir(alpha: float, beta_pos: float, beta_neg: float) -> Path:
    """Build a compact output folder name for one visual-direction run."""
    return (
        Path("outputs")
        / "visual_direction_run"
        / f"test_subset_alpha{alpha:.2f}_bpos{beta_pos:.2f}_bneg{beta_neg:.2f}"
    )


def parse_float_grid(value: str) -> list[float]:
    """Parse comma-separated float values from the CLI."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Run visual-direction retrieval on the CelebA test subset.")
    parser.add_argument("--query-json", default="data/celeba_subset/queries/test_embedding_celeba_evaluation.json")
    parser.add_argument("--image-embedding-dir", default="data/celeba_subset/embeddings/test")
    parser.add_argument("--visual-direction-dir", default="data/celeba_subset/embeddings")
    parser.add_argument("--output-root", default="outputs/visual_direction_run")
    parser.add_argument("--alpha-values", default="0.5,1.0,2.0")
    parser.add_argument("--beta-pos-values", default="0.5,1.0,2.0")
    parser.add_argument("--beta-neg-values", default="0.5,1.0,2.0")
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
    beta_pos_values = parse_float_grid(args.beta_pos_values)
    beta_neg_values = parse_float_grid(args.beta_neg_values)
    rows = []

    for alpha in alpha_values:
        for beta_pos in beta_pos_values:
            for beta_neg in beta_neg_values:
                relative_output = default_output_dir(alpha, beta_pos, beta_neg)
                result = run_visual_direction_retrieval(
                    query_json=PROJECT_ROOT / args.query_json,
                    image_embedding_dir=PROJECT_ROOT / args.image_embedding_dir,
                    visual_direction_dir=PROJECT_ROOT / args.visual_direction_dir,
                    output_dir=PROJECT_ROOT / relative_output,
                    alpha=alpha,
                    beta_pos=beta_pos,
                    beta_neg=beta_neg,
                    batch_size=args.batch_size,
                    device=args.device,
                    save_predictions=args.save_predictions,
                    save_run_config=args.save_run_config,
                )
                row = {
                    "method": result["method"],
                    "alpha": alpha,
                    "beta_pos": beta_pos,
                    "beta_neg": beta_neg,
                    "output_dir": str(relative_output),
                    **result["metrics"],
                }
                rows.append(row)
                print(
                    f"alpha={alpha:.2f} beta_pos={beta_pos:.2f} beta_neg={beta_neg:.2f} "
                    f"recall@1={result['metrics']['recall@1']:.4f} "
                    f"recall@5={result['metrics']['recall@5']:.4f} "
                    f"recall@10={result['metrics']['recall@10']:.4f}"
                )

    summary_path = output_root / "summary.csv"
    rows = sorted(rows, key=lambda row: row["recall@10"], reverse=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
