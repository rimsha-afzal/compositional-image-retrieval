import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.correlation_aware_visual_directions import compute_and_save_correlation_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze CelebA train attribute and direction correlations.")
    parser.add_argument("--metadata", default="data/celeba_subset/image_metadata.csv")
    parser.add_argument("--visual-direction-dir", default="data/celeba_subset/embeddings")
    parser.add_argument("--visual-direction-name", default="visual_directions_reliable")
    parser.add_argument("--output-dir", default="outputs/correlation_analysis")
    parser.add_argument("--split-label", default="0", help="0=train, 1=val, 2=test in subset metadata.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = compute_and_save_correlation_analysis(
        metadata_path=PROJECT_ROOT / args.metadata,
        visual_direction_dir=PROJECT_ROOT / args.visual_direction_dir,
        visual_direction_name=args.visual_direction_name,
        output_dir=PROJECT_ROOT / args.output_dir,
        split_label=args.split_label,
    )
    print("correlation analysis:", result["status"])
    print("attributes:", result["num_attributes"])
    print("top label correlation:", result["top_label_correlation"])
    print("top direction similarity:", result["top_direction_similarity"])
    print("saved to:", PROJECT_ROOT / args.output_dir)


if __name__ == "__main__":
    main()
