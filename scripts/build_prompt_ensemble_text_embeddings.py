import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prompt_ensemble_text_embeddings import build_and_save_prompt_ensemble_text_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Build prompt-ensemble CLIP text embeddings.")
    parser.add_argument("--output-dir", default="data/celeba_subset/embeddings")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--filename", default="prompt_ensemble_text_embeddings")
    return parser.parse_args()


def main():
    args = parse_args()
    embeddings, ids = build_and_save_prompt_ensemble_text_embeddings(
        output_dir=PROJECT_ROOT / args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        filename=args.filename,
    )
    print(f"Saved {len(ids)} prompt-ensemble text embeddings")
    print(f"shape: {tuple(embeddings.shape)}")
    print(f"output: {PROJECT_ROOT / args.output_dir}")
    print(f"filename: {args.filename}")


if __name__ == "__main__":
    main()
