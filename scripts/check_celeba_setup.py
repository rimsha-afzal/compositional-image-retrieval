import argparse
import json
from pathlib import Path
import sys

# Allow importing from src/ without installing the project as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loading import (
    load_celeba_split,
    load_ground_truth_json,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Check that CelebA and ground truth load correctly."
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="Folder containing the celeba/ directory."
    )

    parser.add_argument(
        "--ground-truth",
        type=str,
        default="data/raw/celeba_evaluation.json",
        help="Path to the provided celeba_evaluation.json."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test", "all"],
        help="CelebA split to load. Official evaluation uses test.",
    )

    parser.add_argument(
        "--check-all-indices",
        action="store_true",
        help="Validate that all source and target indices in the JSON are inside the dataset range.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_root = Path(args.data_root)
    ground_truth_path = Path(args.ground_truth)

    print("Checking CelebA setup...")
    print(f"Data root: {data_root}")
    print(f"Ground truth: {ground_truth_path}")
    print()

    # 1. Basic file checks
    required_paths = [
        data_root / "celeba",
        data_root / "celeba" / "list_attr_celeba.txt",
        data_root / "celeba" / "list_eval_partition.txt",
        data_root / "celeba" / "img_align_celeba",
        ground_truth_path,
    ]

    missing = [str(path) for path in required_paths if not path.exists()]

    if missing:
        print("Missing required paths:")
        for path in missing:
            print(f"  - {path}")
        raise FileNotFoundError("Some required CelebA files are missing.")

    #2. Load official CelebA split
    celeba = load_celeba_split(
        data_root=data_root,
        split=args.split,
        target_type="attr",
    )

    print(f"CelebA split loaded: {args.split}")
    print(f"Number of images: {len(celeba)}")

    #3. Load ground truth
    gt = load_ground_truth_json(ground_truth_path)

    print(f"Ground-truth queries loaded: {len(gt)}")

    if len(gt) == 0:
        raise ValueError("Ground-truth JSON is empty.")
    
    print()
    print("Done.")


if __name__ == "__main__":
    main()
   
