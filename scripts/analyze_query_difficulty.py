import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from query_difficulty_analysis import (
    aggregate_predictions_by_bucket,
    assign_quantile_buckets,
    build_markdown_table,
    compute_query_availability_scores,
    load_ground_truth,
    load_prediction_rows,
    load_split_index_set,
    write_csv,
)


DEFAULT_METHOD_PREDICTIONS = {
    "CLIP text-only baseline": (
        "outputs/query_difficulty_analysis/prediction_runs/"
        "text_baseline/test_subset_alpha2.00_beta1.00/predictions.csv"
    ),
    "CelebA visual-direction only": (
        "outputs/visual_direction_run/test_subset_alpha2.00_bpos1.00_bneg0.50/predictions.csv"
    ),
    "Hybrid text + visual fusion": (
        "outputs/hybrid_fusion_run/test_subset_alpha1.00_btext2.00_bvpos0.50_bvneg0.25/predictions.csv"
    ),
    "Hybrid prompt-ensemble text + visual fusion": (
        "outputs/hybrid_fusion_run/test_subset_alpha1.00_btext2.00_bvpos0.25_bvneg0.25/predictions.csv"
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze saved retrieval predictions by query difficulty."
    )
    parser.add_argument("--difficulty-query-json", default="data/celeba_subset/queries/celeba_evaluation.json")
    parser.add_argument("--metadata", default="data/celeba_subset/image_metadata.csv")
    parser.add_argument("--difficulty-split-label", default="0", help="Split label used for difficulty; 0=train.")
    parser.add_argument("--output-dir", default="outputs/query_difficulty_analysis")
    parser.add_argument("--num-images", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_ground_truth(PROJECT_ROOT / args.difficulty_query_json)
    train_indices = load_split_index_set(PROJECT_ROOT / args.metadata, args.difficulty_split_label)
    scores = compute_query_availability_scores(
        entries,
        num_images=args.num_images,
        valid_index_set=train_indices,
    )
    buckets = assign_quantile_buckets(scores)

    rows = []
    warnings = []
    for method, relative_path in DEFAULT_METHOD_PREDICTIONS.items():
        prediction_path = PROJECT_ROOT / relative_path
        if not prediction_path.exists():
            warnings.append(
                f"Skipped `{method}` because no saved per-instance predictions were found at `{relative_path}`. "
                "The aggregate `metrics.json` files are not enough for query-level stratification."
            )
            continue
        rows.extend(
            aggregate_predictions_by_bucket(
                load_prediction_rows(prediction_path),
                query_to_bucket=buckets,
                method=method,
            )
        )

    query_difficulty_path = output_dir / "query_difficulty_scores.json"
    query_difficulty_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "formula": "0.5 * normalized_log_num_sources + 0.5 * normalized_log_avg_targets",
                    "bucketing": "hard/medium/easy quantiles over query_availability_score",
                    "difficulty_source": args.difficulty_query_json,
                    "difficulty_split_label": args.difficulty_split_label,
                    "num_split_indices": len(train_indices),
                    "num_queries": len(scores),
                },
                "queries": {
                    query: {**score, "difficulty_bucket": buckets[query]}
                    for query, score in scores.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = output_dir / "difficulty_bucket_metrics.csv"
    md_path = output_dir / "difficulty_bucket_metrics.md"
    write_csv(rows, csv_path)
    markdown = build_markdown_table(rows, warnings=warnings)
    md_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"Saved query difficulty scores: {query_difficulty_path}")
    print(f"Saved CSV summary: {csv_path}")
    print(f"Saved markdown summary: {md_path}")


if __name__ == "__main__":
    main()
