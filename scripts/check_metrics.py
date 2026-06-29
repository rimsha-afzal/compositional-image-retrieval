from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation_metrics import evaluate_single_ranking, average_metrics


def main():
    # Fake retrieved ranking produced by a model
    retrieved = [12, 90, 7, 3, 22, 100]

    # Fake professor ground-truth valid targets
    ground_truth = {7, 22, 100}

    metrics = evaluate_single_ranking(
        retrieved_indices=retrieved,
        ground_truth_indices=ground_truth,
        ks=(1, 5, 10),
    )

    print("Single ranking metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Fake multiple source images
    all_rows = [
        evaluate_single_ranking([1, 2, 3, 4, 5], {3, 9}, ks=(1, 5, 10)),
        evaluate_single_ranking([8, 9, 10, 11], {20, 21}, ks=(1, 5, 10)),
        evaluate_single_ranking([30, 31, 32], {30, 40}, ks=(1, 5, 10)),
    ]

    averaged = average_metrics(all_rows)

    print()
    print("Averaged metrics:")
    for key, value in averaged.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
