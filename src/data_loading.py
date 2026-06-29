"""Load CelebA data and the provided ground-truth JSON."""

import json
from pathlib import Path
from typing import Any

from torchvision.datasets import CelebA


def load_celeba_split(
    data_root: str | Path,
    split: str = "test",
    target_type: str = "attr",
) -> CelebA:
    """Load a torchvision CelebA split without downloading data."""
    return CelebA(
        root=str(Path(data_root)),
        split=split,
        target_type=target_type,
        download=False,
    )


def load_ground_truth_json(path: str | Path) -> list[dict[str, Any]]:
    """Load the official CelebA evaluation JSON as a list of query records."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Ground-truth JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    if not isinstance(ground_truth, list):
        raise ValueError("Expected ground-truth JSON to be a list of dictionaries.")

    return ground_truth
