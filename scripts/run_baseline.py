import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baseline_retrieval import run_baseline


def default_output_dir(alpha: float, beta: float) -> Path:
    """Build a compact output folder name for one alpha/beta run."""
    return Path("outputs") / "baseline_run" / f"test_subset_alpha{alpha:.2f}_beta{beta:.2f}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CLIP arithmetic baseline.")
    parser.add_argument("--query-json", default="data/celeba_subset/queries/test_embedding_celeba_evaluation.json")
    parser.add_argument("--image-embedding-dir", default="data/celeba_subset/embeddings/test")
    parser.add_argument("--text-embedding-dir", default="data/celeba_subset/embeddings")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-run-config", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.alpha, args.beta)
    result = run_baseline(
        query_json=PROJECT_ROOT / args.query_json,
        image_embedding_dir=PROJECT_ROOT / args.image_embedding_dir,
        text_embedding_dir=PROJECT_ROOT / args.text_embedding_dir,
        output_dir=PROJECT_ROOT / output_dir,
        alpha=args.alpha,
        beta=args.beta,
        batch_size=args.batch_size,
        device=args.device,
        save_predictions=args.save_predictions,
        save_run_config=args.save_run_config,
    )
    print(f"Saved metrics to {PROJECT_ROOT / output_dir / 'metrics.json'}")
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
