"""Official retrieval metrics for the CelebA CIR benchmark."""

from typing import Iterable


def recall_at_k(
    retrieved_indices: list[int],
    ground_truth_indices: set[int],
    k: int,
) -> float:
    """Return 1.0 if a valid target appears in the top K results."""
    if k <= 0:
        raise ValueError("k must be positive.")

    top_k = set(retrieved_indices[:k])
    return 1.0 if top_k & ground_truth_indices else 0.0


def precision_at_k(
    retrieved_indices: list[int],
    ground_truth_indices: set[int],
    k: int,
) -> float:
    """Return the fraction of top K results that are valid targets."""
    if k <= 0:
        raise ValueError("k must be positive.")

    top_k = retrieved_indices[:k]
    if len(top_k) == 0:
        return 0.0

    hits = sum(1 for idx in top_k if idx in ground_truth_indices)
    return hits / k


def evaluate_single_ranking(
    retrieved_indices: list[int],
    ground_truth_indices: Iterable[int],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Recall@K and Precision@K for one source image."""
    ground_truth_set = set(int(idx) for idx in ground_truth_indices)
    metrics = {}

    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(
            retrieved_indices=retrieved_indices,
            ground_truth_indices=ground_truth_set,
            k=k,
        )
        metrics[f"precision@{k}"] = precision_at_k(
            retrieved_indices=retrieved_indices,
            ground_truth_indices=ground_truth_set,
            k=k,
        )

    return metrics


def average_metrics(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    """Average metric dictionaries across source images."""
    if len(metric_rows) == 0:
        raise ValueError("Cannot average an empty list of metric rows.")

    averaged = {}
    for name in metric_rows[0].keys():
        averaged[name] = sum(row[name] for row in metric_rows) / len(metric_rows)

    return averaged
