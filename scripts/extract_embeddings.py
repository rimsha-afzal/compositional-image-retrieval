import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embedding_extraction import extract_image_embeddings, extract_text_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Extract CLIP embeddings.")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--output-dir", default="data/celeba_subset/embeddings/clip_vit_b32")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--image-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    if not args.image_only:
        extract_text_embeddings(
            model_name=args.model_name,
            output_dir=output_dir,
            batch_size=args.batch_size,
            device=args.device,
        )

    if not args.text_only:
        extract_image_embeddings(
            data_root=args.data_root,
            split=args.split,
            model_name=args.model_name,
            output_dir=output_dir,
            batch_size=args.batch_size,
            device=args.device,
            max_images=args.max_images,
        )


if __name__ == "__main__":
    main()
