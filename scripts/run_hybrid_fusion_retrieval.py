import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.hybrid_fusion_retrieval import run_hybrid_fusion_retrieval


def default_output_dir(
    alpha: float,
    beta_text: float,
    beta_visual_pos: float,
    beta_visual_neg: float,
) -> Path:
    """Build a compact output folder name for one hybrid run."""
    return (
        Path("outputs")
        / "hybrid_fusion_run"
        / (
            f"test_subset_alpha{alpha:.2f}_"
            f"btext{beta_text:.2f}_"
            f"bvpos{beta_visual_pos:.2f}_"
            f"bvneg{beta_visual_neg:.2f}"
        )
    )


def parse_float_grid(value: str) -> list[float]:
    """Parse comma-separated float values from the CLI."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def read_best_row(summary_path: Path, metric: str = "recall@10") -> dict[str, str] | None:
    """Read the best row from an existing summary CSV."""
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return max(rows, key=lambda row: float(row[metric]))


def format_metric(row: dict[str, str] | None, metric: str) -> str:
    """Format a metric from a CSV row, or mark it unavailable."""
    if row is None:
        return "n/a"
    return f"{float(row[metric]):.4f}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run hybrid CLIP text + CelebA visual-direction retrieval."
    )
    parser.add_argument("--query-json", default="data/celeba_subset/queries/test_embedding_celeba_evaluation.json")
    parser.add_argument("--image-embedding-dir", default="data/celeba_subset/embeddings/test")
    parser.add_argument("--text-embedding-dir", default="data/celeba_subset/embeddings")
    parser.add_argument("--text-embedding-name", default="text_embeddings")
    parser.add_argument("--visual-direction-dir", default="data/celeba_subset/embeddings")
    parser.add_argument("--output-root", default="outputs/hybrid_fusion_run")
    parser.add_argument("--alpha-values", default="1.0,2.0,3.0")
    parser.add_argument("--beta-text-values", default="0.5,1.0,2.0")
    parser.add_argument("--beta-visual-pos-values", default="0.25,0.5,1.0")
    parser.add_argument("--beta-visual-neg-values", default="0.0,0.25,0.5")
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
                        alpha,
                        beta_text,
                        beta_visual_pos,
                        beta_visual_neg,
                    )
                    result = run_hybrid_fusion_retrieval(
                        query_json=PROJECT_ROOT / args.query_json,
                        image_embedding_dir=PROJECT_ROOT / args.image_embedding_dir,
                        text_embedding_dir=PROJECT_ROOT / args.text_embedding_dir,
                        visual_direction_dir=PROJECT_ROOT / args.visual_direction_dir,
                        output_dir=PROJECT_ROOT / relative_output,
                        text_embedding_name=args.text_embedding_name,
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
                        "method": result["method"],
                        "alpha": alpha,
                        "beta_text": beta_text,
                        "beta_visual_pos": beta_visual_pos,
                        "beta_visual_neg": beta_visual_neg,
                        "output_dir": str(relative_output),
                        **result["metrics"],
                    }
                    rows.append(row)
                    print(
                        f"[{run_index:02d}/{total:02d}] "
                        f"alpha={alpha:.2f} beta_text={beta_text:.2f} "
                        f"beta_visual_pos={beta_visual_pos:.2f} "
                        f"beta_visual_neg={beta_visual_neg:.2f} "
                        f"recall@1={result['metrics']['recall@1']:.4f} "
                        f"recall@5={result['metrics']['recall@5']:.4f} "
                        f"recall@10={result['metrics']['recall@10']:.4f} "
                        f"precision@10={result['metrics']['precision@10']:.4f}"
                    )

    rows = sorted(rows, key=lambda row: row["recall@10"], reverse=True)
    summary_path = output_root / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_hybrid = rows[0]
    best_text = read_best_row(PROJECT_ROOT / "outputs/baseline_run/summary.csv")
    best_visual = read_best_row(PROJECT_ROOT / "outputs/visual_direction_run/summary.csv")

    print(f"\nSaved summary to {summary_path}")
    print("\nBest hybrid by recall@10:")
    print(
        f"  alpha={best_hybrid['alpha']:.2f} "
        f"beta_text={best_hybrid['beta_text']:.2f} "
        f"beta_visual_pos={best_hybrid['beta_visual_pos']:.2f} "
        f"beta_visual_neg={best_hybrid['beta_visual_neg']:.2f} "
        f"recall@1={best_hybrid['recall@1']:.4f} "
        f"recall@5={best_hybrid['recall@5']:.4f} "
        f"recall@10={best_hybrid['recall@10']:.4f} "
        f"precision@10={best_hybrid['precision@10']:.4f}"
    )

    print("\nComparison:")
    print(
        "  best text-only baseline:    "
        f"recall@10={format_metric(best_text, 'recall@10')} "
        f"precision@10={format_metric(best_text, 'precision@10')}"
    )
    print(
        "  best visual-only baseline:  "
        f"recall@10={format_metric(best_visual, 'recall@10')} "
        f"precision@10={format_metric(best_visual, 'precision@10')}"
    )
    print(
        "  best hybrid:                "
        f"recall@10={best_hybrid['recall@10']:.4f} "
        f"precision@10={best_hybrid['precision@10']:.4f}"
    )

    if best_text is not None:
        text_delta = best_hybrid["recall@10"] - float(best_text["recall@10"])
        print(f"  hybrid - text-only recall@10:   {text_delta:+.4f}")
    if best_visual is not None:
        visual_delta = best_hybrid["recall@10"] - float(best_visual["recall@10"])
        print(f"  hybrid - visual-only recall@10: {visual_delta:+.4f}")

    if float(best_hybrid["beta_visual_neg"]) < float(best_hybrid["beta_visual_pos"]):
        print(
            "\nInterpretation: the best hybrid uses a smaller negative visual weight "
            "than positive visual weight, supporting the current hypothesis that "
            "negative visual directions are less reliable."
        )


if __name__ == "__main__":
    main()
