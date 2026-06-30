import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baseline_retrieval import parse_query_string

ID_COLUMNS = {"image_id", "split"}


def load_split_rows(metadata_path: Path, split_label: str) -> tuple[list[str], np.ndarray]:
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        attribute_names = [name for name in reader.fieldnames if name not in ID_COLUMNS]
        rows = [
            [int(row[attribute]) for attribute in attribute_names]
            for row in reader
            if row["split"] == split_label
        ]
    if not rows:
        raise ValueError(f"No rows found for split label {split_label!r}.")
    return attribute_names, np.array(rows, dtype=np.int8)


def build_query_entries(
    query_strings: list[str],
    attribute_names: list[str],
    labels: np.ndarray,
    max_extra_differences: int = 2,
) -> list[dict[str, object]]:
    attribute_index = {attribute: index for index, attribute in enumerate(attribute_names)}
    entries = []
    for query in query_strings:
        positive, negative = parse_query_string(query)
        changed_attributes = positive + negative
        positive_cols = [attribute_index[attribute] for attribute in positive]
        negative_cols = [attribute_index[attribute] for attribute in negative]
        max_differences = len(changed_attributes) + max_extra_differences
        ground_truth = {}

        reference_mask = np.ones(labels.shape[0], dtype=bool)
        target_mask = np.ones(labels.shape[0], dtype=bool)
        if positive_cols:
            reference_mask &= (labels[:, positive_cols] == -1).all(axis=1)
            target_mask &= (labels[:, positive_cols] == 1).all(axis=1)
        if negative_cols:
            reference_mask &= (labels[:, negative_cols] == 1).all(axis=1)
            target_mask &= (labels[:, negative_cols] == -1).all(axis=1)

        target_indices = np.flatnonzero(target_mask)
        target_labels = labels[target_indices]
        for reference_index in np.flatnonzero(reference_mask):
            differences = (target_labels != labels[reference_index]).sum(axis=1)
            targets = target_indices[differences <= max_differences]
            targets = targets[targets != reference_index].astype(int).tolist()

            if targets:
                ground_truth[str(reference_index)] = targets

        if ground_truth:
            entries.append({"query": query, "ground_truth": ground_truth})

    return entries


def parse_args():
    parser = argparse.ArgumentParser(description="Build split-local CelebA CIR query JSON from metadata.")
    parser.add_argument("--metadata", default="data/celeba_subset/image_metadata.csv")
    parser.add_argument("--query-template", default="data/celeba_subset/queries/test_embedding_celeba_evaluation.json")
    parser.add_argument("--output", default="data/celeba_subset/queries/val_embedding_celeba_evaluation.json")
    parser.add_argument("--split-label", default="1", help="0=train, 1=val, 2=test in subset metadata.")
    parser.add_argument("--max-extra-differences", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    template_entries = json.loads((PROJECT_ROOT / args.query_template).read_text(encoding="utf-8"))
    query_strings = [entry["query"] for entry in template_entries]
    attribute_names, labels = load_split_rows(PROJECT_ROOT / args.metadata, args.split_label)
    entries = build_query_entries(
        query_strings=query_strings,
        attribute_names=attribute_names,
        labels=labels,
        max_extra_differences=args.max_extra_differences,
    )

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    query_instances = sum(len(entry["ground_truth"]) for entry in entries)
    targets = sum(
        len(targets)
        for entry in entries
        for targets in entry["ground_truth"].values()
    )
    print(f"wrote {output_path}")
    print(f"queries: {len(entries)}")
    print(f"query instances: {query_instances}")
    print(f"targets: {targets}")


if __name__ == "__main__":
    main()
