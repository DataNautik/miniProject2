"""
Part B visualization helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_latent_scatter_with_curve_paths(
    latents: np.ndarray,
    labels: Optional[np.ndarray],
    curve_paths: list[np.ndarray],
    endpoints: Optional[np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    if labels is None:
        ax.scatter(latents[:, 0], latents[:, 1], s=12, alpha=0.7)
    else:
        scatter = ax.scatter(latents[:, 0], latents[:, 1], c=labels, s=12, alpha=0.75)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Class")

    for path_id, nodes in enumerate(curve_paths):
        ax.plot(nodes[:, 0], nodes[:, 1], linewidth=2)
        ax.scatter(nodes[[0, -1], 0], nodes[[0, -1], 1], s=50)

    if endpoints is not None and len(endpoints) > 0:
        ax.scatter(endpoints[:, :, 0].reshape(-1), endpoints[:, :, 1].reshape(-1), s=36, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.grid(True, alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cov_vs_num_decoders(
    cov_by_k: Dict[int, Dict[str, float]],
    out_path: Path,
    title: str = "Distance reliability (CoV) vs number of decoders",
) -> None:
    ks = sorted(cov_by_k.keys())
    euc = [cov_by_k[k]["euclidean_cov_global_mean"] for k in ks]
    geo = [cov_by_k[k]["geodesic_cov_global_mean"] for k in ks]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, euc, marker="o", label="Euclidean")
    ax.plot(ks, geo, marker="o", label="Geodesic")
    ax.set_xlabel("Number of decoders")
    ax.set_ylabel("Mean CoV across fixed pairs")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_plot_manifest(out_dir: Path, **entries: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "plot_manifest.json", "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
