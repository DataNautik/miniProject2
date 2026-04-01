"""Visualization utilities for Part A of Mini-project 2.

This module provides lightweight helpers for:
- extracting 2D latent means from the trained VAE,
- plotting the latent space colored by class,
- drawing piecewise-linear geodesic candidates on top of the latent scatter.

The functions are intentionally modular so they can be reused later in
`run_partA.py` and in qualitative comparisons across training runs.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from partA_data import PartADataBundle, PARTA_CLASSES, get_parta_data
from partA_model import PartAGaussianVAE, PartAVAEConfig, build_parta_vae


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

PARTA_SCATTER_PATH = Path("partA_outputs") / "latent_scatter.png"
PARTA_SCATTER_WITH_CURVES_PATH = Path("partA_outputs") / "latent_scatter_with_curves.png"


# -----------------------------------------------------------------------------
# Latent extraction
# -----------------------------------------------------------------------------

@torch.no_grad()
def extract_latent_means(
    model: PartAGaussianVAE,
    data_bundle: PartADataBundle,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract posterior mean latents for the full Part A subset.

    Args:
        model: Trained Part A VAE.
        data_bundle: Bundle returned by `get_parta_data`.
        device: Torch device.

    Returns:
        latents: NumPy array of shape (N, 2).
        labels: NumPy array of shape (N,).
    """
    model.eval()

    all_latents: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for x, y in data_bundle.full_loader:
        x = x.to(device)
        z_mu = model.encode_mean(x)
        all_latents.append(z_mu.cpu())
        all_labels.append(y.cpu())

    latents = torch.cat(all_latents, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return latents, labels


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------


def _prepare_output_path(path: str | Path) -> Path:
    """Convert a path-like input to a Path and ensure the parent exists."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path



def plot_latent_scatter(
    latents: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path = PARTA_SCATTER_PATH,
    title: str = "Part A latent space",
) -> None:
    """Plot the latent means colored by class.

    Args:
        latents: Array of shape (N, 2).
        labels: Array of shape (N,).
        save_path: Output image path.
        title: Figure title.
    """
    if latents.ndim != 2 or latents.shape[1] != 2:
        raise ValueError(f"Expected latents of shape (N, 2), got {latents.shape}")

    out_path = _prepare_output_path(save_path)

    plt.figure(figsize=(8, 6))

    for class_id in sorted(np.unique(labels)):
        mask = labels == class_id
        plt.scatter(
            latents[mask, 0],
            latents[mask, 1],
            s=12,
            alpha=0.75,
            label=f"Class {class_id}",
        )

    plt.title(title)
    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def plot_latent_scatter_with_curves(
    latents: np.ndarray,
    labels: np.ndarray,
    curves: Iterable[np.ndarray],
    save_path: str | Path = PARTA_SCATTER_WITH_CURVES_PATH,
    title: str = "Part A latent space with geodesic candidates",
) -> None:
    """Plot the latent space together with piecewise-linear curves.

    Args:
        latents: Array of shape (N, 2).
        labels: Array of shape (N,).
        curves: Iterable of arrays, each of shape (K, 2).
        save_path: Output image path.
        title: Figure title.
    """
    if latents.ndim != 2 or latents.shape[1] != 2:
        raise ValueError(f"Expected latents of shape (N, 2), got {latents.shape}")

    out_path = _prepare_output_path(save_path)

    plt.figure(figsize=(8, 6))

    for class_id in sorted(np.unique(labels)):
        mask = labels == class_id
        plt.scatter(
            latents[mask, 0],
            latents[mask, 1],
            s=10,
            alpha=0.60,
            label=f"Class {class_id}",
        )

    for curve in curves:
        if curve.ndim != 2 or curve.shape[1] != 2:
            raise ValueError(f"Each curve must have shape (K, 2), got {curve.shape}")

        plt.plot(curve[:, 0], curve[:, 1], linewidth=1.5, alpha=0.9)
        plt.scatter(curve[0, 0], curve[0, 1], s=20)
        plt.scatter(curve[-1, 0], curve[-1, 1], s=20)

    plt.title(title)
    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Model loading helper
# -----------------------------------------------------------------------------


def load_trained_parta_vae(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    model_config: PartAVAEConfig | None = None,
) -> PartAGaussianVAE:
    """Load a trained Part A VAE checkpoint."""
    device = torch.device(device)
    model = build_parta_vae(config=model_config).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Quick manual check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    checkpoint_path = Path("partA_outputs") / "partA_vae.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}. "
            "Train the model first using partA_train.py."
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data_bundle = get_parta_data(seed=0)
    model = load_trained_parta_vae(checkpoint_path=checkpoint_path, device=device)

    latents, labels = extract_latent_means(model=model, data_bundle=data_bundle, device=device)
    plot_latent_scatter(latents=latents, labels=labels)

    print("Part A visualization check")
    print(f"Classes: {PARTA_CLASSES}")
    print(f"Latents shape: {latents.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Saved scatter plot to: {PARTA_SCATTER_PATH}")