"""Save, load, and normalize embedding tensors.

The name embedding_io uses "I/O" to mean input/output: this file writes
embedding tensors and row IDs to disk, then reads them back later.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def l2_normalize(embeddings: torch.Tensor) -> torch.Tensor:
    """Return embeddings scaled to unit length, row by row."""
    single = embeddings.dim() == 1
    if single:
        embeddings = embeddings.unsqueeze(0)
    normalized = F.normalize(embeddings.float(), p=2, dim=1, eps=1e-12)
    return normalized.squeeze(0) if single else normalized


def save_embeddings(
    embeddings: torch.Tensor,
    ids: list[Any],
    output_dir: str | Path,
    name: str,
) -> None:
    """Save embeddings as a .pt tensor and matching row IDs as a .npy file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings.cpu().float(), output_dir / f"{name}.pt")
    np.save(output_dir / f"{name}_ids.npy", np.array(ids))


def load_embeddings(
    output_dir: str | Path,
    name: str,
    device: str = "cpu",
) -> tuple[torch.Tensor, list[Any]]:
    """Load a saved embedding tensor and its matching row IDs."""
    output_dir = Path(output_dir)
    embeddings_path = output_dir / f"{name}.pt"
    ids_path = output_dir / f"{name}_ids.npy"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {embeddings_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing embedding IDs file: {ids_path}")

    embeddings = torch.load(embeddings_path, map_location=device, weights_only=True)
    ids = np.load(ids_path, allow_pickle=True).tolist()
    return embeddings, ids
