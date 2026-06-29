"""Query-difficulty stratification for saved retrieval predictions."""

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


METRIC_COLUMNS = ("recall@1", "recall@5", "recall@10", "precision@10")


def load_ground_truth(path: str | Path) -> list[dict[str, Any]]:
    """Load query ground truth entries."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Expected ground-truth JSON to contain a list of query records.")
    return payload


def load_split_index_set(metadata_path: str | Path, split_label: str = "0") -> set[int]:
    """Return row indices whose metadata split matches split_label."""
    with Path(metadata_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {index for index, row in enumerate(reader) if str(row["split"]) == str(split_label)}


def compute_query_availability_scores(
    ground_truth_entries: list[dict[str, Any]],
    num_images: int | None = None,
    valid_index_set: set[int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute archive-compatible query availability scores from ground truth.

    Availability is higher when a query has more valid reference images and more
    valid targets per reference. Low availability is treated as high difficulty.
    """
    query_target_counts: dict[str, list[int]] = {}

    for entry_index, entry in enumerate(ground_truth_entries):
        query = str(entry.get("query", f"query_{entry_index}"))
        query_target_counts.setdefault(query, [])
        ground_truth = entry.get("ground_truth", {})
        if not isinstance(ground_truth, dict):
            raise ValueError(f"Entry for query {query!r} has no ground_truth mapping.")

        for source, targets in ground_truth.items():
            try:
                source_index = int(source)
            except (TypeError, ValueError):
                continue
            if valid_index_set is not None and source_index not in valid_index_set:
                continue

            if isinstance(targets, (str, bytes)) or not hasattr(targets, "__iter__"):
                targets = [targets]
            elif not isinstance(targets, list):
                targets = list(targets)

            valid_targets = 0
            for target in targets:
                try:
                    target_index = int(target)
                except (TypeError, ValueError):
                    continue
                indexable = num_images is None or 0 <= target_index < num_images
                in_split = valid_index_set is None or target_index in valid_index_set
                if indexable and in_split:
                    valid_targets += 1
            if valid_targets > 0:
                query_target_counts[query].append(valid_targets)

    raw_stats = {}
    for query, targets_per_source in query_target_counts.items():
        num_sources = len(targets_per_source)
        total_targets = int(sum(targets_per_source))
        avg_targets = total_targets / num_sources if num_sources else 0.0
        raw_stats[query] = {
            "query": query,
            "num_sources": int(num_sources),
            "total_valid_targets": total_targets,
            "avg_targets_per_source": avg_targets,
            "median_targets_per_source": median(targets_per_source) if targets_per_source else 0.0,
            "min_targets_per_source": min(targets_per_source) if targets_per_source else 0,
            "max_targets_per_source": max(targets_per_source) if targets_per_source else 0,
        }

    max_log_sources = max((math.log1p(row["num_sources"]) for row in raw_stats.values()), default=0.0)
    max_log_targets = max((math.log1p(row["avg_targets_per_source"]) for row in raw_stats.values()), default=0.0)

    scores = {}
    for query, row in raw_stats.items():
        source_component = math.log1p(row["num_sources"]) / max_log_sources if max_log_sources else 0.0
        target_component = (
            math.log1p(row["avg_targets_per_source"]) / max_log_targets
            if max_log_targets
            else 0.0
        )
        availability = max(0.0, min(1.0, 0.5 * source_component + 0.5 * target_component))
        scores[query] = {
            **row,
            "source_component": source_component,
            "target_component": target_component,
            "query_availability_score": availability,
        }
    return scores


def assign_quantile_buckets(scores: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Assign hard/medium/easy buckets from low/mid/high availability quantiles."""
    ordered = sorted(
        scores.items(),
        key=lambda item: (item[1]["query_availability_score"], item[0]),
    )
    n = len(ordered)
    buckets = {}
    for index, (query, _) in enumerate(ordered):
        if index < n / 3:
            bucket = "hard"
        elif index < 2 * n / 3:
            bucket = "medium"
        else:
            bucket = "easy"
        buckets[query] = bucket
    return buckets


def load_prediction_rows(path: str | Path) -> list[dict[str, str]]:
    """Load saved per-instance prediction rows."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_predictions_by_bucket(
    rows: list[dict[str, str]],
    query_to_bucket: dict[str, str],
    method: str,
) -> list[dict[str, Any]]:
    """Average saved per-instance metrics within each difficulty bucket."""
    totals = {
        bucket: {"method": method, "num_queries": 0, **{metric: 0.0 for metric in METRIC_COLUMNS}}
        for bucket in ("hard", "medium", "easy")
    }
    seen_queries = {bucket: set() for bucket in totals}
    counts = {bucket: 0 for bucket in totals}

    for row in rows:
        query = row["query"]
        bucket = query_to_bucket[query]
        seen_queries[bucket].add(query)
        counts[bucket] += 1
        for metric in METRIC_COLUMNS:
            totals[bucket][metric] += float(row[metric])

    out = []
    for bucket in ("hard", "medium", "easy"):
        count = counts[bucket]
        item = {
            "difficulty_bucket": bucket,
            "method": method,
            "num_queries": len(seen_queries[bucket]),
        }
        for metric in METRIC_COLUMNS:
            item[metric] = totals[bucket][metric] / count if count else 0.0
        out.append(item)
    return out


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write stratified metric rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["difficulty_bucket", "method", "num_queries", *METRIC_COLUMNS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown_table(rows: list[dict[str, Any]], warnings: list[str] | None = None) -> str:
    """Render stratified metrics as markdown."""
    lines = [
        "# Query Difficulty Stratification",
        "",
        "Buckets are quantiles of train-split query availability: low availability is `hard`, middle is `medium`, high is `easy`.",
        "",
        "| difficulty_bucket | method | num_queries | recall@1 | recall@5 | recall@10 | precision@10 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['difficulty_bucket']} | {row['method']} | {row['num_queries']} "
            f"| {row['recall@1']:.4f} | {row['recall@5']:.4f} "
            f"| {row['recall@10']:.4f} | {row['precision@10']:.4f} |"
        )
    if warnings:
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"
