import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.visual_directions import compute_and_save_visual_directions


def parse_args():
    parser = argparse.ArgumentParser(description="Compute visual attribute directions.")
    parser.add_argument("--metadata", default="data/celeba_subset/image_metadata.csv")
    parser.add_argument("--image-embedding-dir", default="data/celeba_subset/embeddings/train")
    parser.add_argument("--output-dir", default="data/celeba_subset/embeddings")
    parser.add_argument("--metrics", default="data/celeba_subset/checks/visual_direction_metrics.json")
    parser.add_argument("--split-label", default="0", help="0=train, 1=val, 2=test in subset metadata.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = compute_and_save_visual_directions(
        metadata_path=PROJECT_ROOT / args.metadata,
        image_embedding_dir=PROJECT_ROOT / args.image_embedding_dir,
        output_dir=PROJECT_ROOT / args.output_dir,
        metrics_path=PROJECT_ROOT / args.metrics,
        split_label=args.split_label,
    )
    print("visual direction computation:", result["status"])
    print("source images:", result["num_source_images"])
    print("attributes:", result["num_attributes"])
    print("shape:", result["embedding_shape"])
    print("saved to:", PROJECT_ROOT / args.output_dir)


if __name__ == "__main__":
    main()
