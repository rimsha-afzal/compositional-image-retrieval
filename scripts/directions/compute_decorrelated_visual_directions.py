import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.correlation_aware_visual_directions import compute_and_save_decorrelated_directions


def parse_args():
    parser = argparse.ArgumentParser(description="Compute conservative decorrelated visual directions.")
    parser.add_argument("--visual-direction-dir", default="data/celeba_subset/embeddings")
    parser.add_argument("--visual-direction-name", default="visual_directions_reliable")
    parser.add_argument("--reliability-dir", default="data/celeba_subset/embeddings")
    parser.add_argument("--reliability-name", default="visual_direction_reliability")
    parser.add_argument("--direction-cosine-matrix", default="outputs/correlation_analysis/direction_cosine_matrix.csv")
    parser.add_argument("--output-dir", default="outputs/decorrelated_directions")
    parser.add_argument("--output-direction-name", default="decorrelated_visual_directions")
    parser.add_argument("--direction-cosine-threshold", type=float, default=0.30)
    parser.add_argument("--max-confounders-per-attribute", type=int, default=3)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    return parser.parse_args()


def main():
    args = parse_args()
    result = compute_and_save_decorrelated_directions(
        visual_direction_dir=PROJECT_ROOT / args.visual_direction_dir,
        visual_direction_name=args.visual_direction_name,
        reliability_dir=PROJECT_ROOT / args.reliability_dir,
        reliability_name=args.reliability_name,
        direction_cosine_matrix_path=PROJECT_ROOT / args.direction_cosine_matrix,
        output_dir=PROJECT_ROOT / args.output_dir,
        output_direction_name=args.output_direction_name,
        direction_cosine_threshold=args.direction_cosine_threshold,
        max_confounders_per_attribute=args.max_confounders_per_attribute,
        lambda_reg=args.lambda_reg,
        epsilon=args.epsilon,
    )
    print("decorrelated direction computation:", result["status"])
    print("attributes:", result["num_attributes"])
    print("attributes with removed components:", result["num_attributes_with_removed_components"])
    print("saved to:", PROJECT_ROOT / args.output_dir)


if __name__ == "__main__":
    main()
