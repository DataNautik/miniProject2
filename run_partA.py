"""End-to-end runner for Part A of Mini-project 2.

This script orchestrates the full Part A pipeline:
1. train or load a 2D Gaussian VAE,
2. extract latent means for the full subset,
3. select 25 latent pairs,
4. compute pull-back geodesic candidates,
5. save the final visualization and auxiliary artifacts.

The goal is to keep the experiment reproducible and easy to rerun.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from partA_data import PARTA_CLASSES, get_parta_data
from partA_geometry import compute_geodesic_curves, select_latent_pairs
from partA_model import PartAVAEConfig
from partA_train import PartATrainConfig, train_parta_vae
from partA_visualization import (
    PARTA_SCATTER_PATH,
    PARTA_SCATTER_WITH_CURVES_PATH,
    extract_latent_means,
    load_trained_parta_vae,
    plot_latent_scatter,
    plot_latent_scatter_with_curves,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RUN_PARTA_OUTPUT_DIR = Path("partA_outputs")
RUN_PARTA_LATENTS_PATH = RUN_PARTA_OUTPUT_DIR / "latents.npz"
RUN_PARTA_PAIRS_PATH = RUN_PARTA_OUTPUT_DIR / "selected_pairs.json"
RUN_PARTA_RUNINFO_PATH = RUN_PARTA_OUTPUT_DIR / "run_partA_config.json"

TRAIN_NEW_MODEL = True
LOAD_EXISTING_CHECKPOINT = False
CHECKPOINT_PATH = RUN_PARTA_OUTPUT_DIR / "partA_vae.pt"

SEED = 0
DEVICE = "mps"
N_GEODESIC_PAIRS = 25
GEODESIC_NODES = 20
GEODESIC_LR = 0.5
GEODESIC_MAX_ITER = 200


# -----------------------------------------------------------------------------
# Helper I/O
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)



def save_json(data: dict[str, Any], path: Path) -> None:
    """Save a dictionary to JSON."""
    path.write_text(json.dumps(data, indent=2))



def latent_pairs_to_serializable(pairs) -> list[dict[str, Any]]:
    """Convert LatentPair objects into JSON-serializable dictionaries."""
    serializable: list[dict[str, Any]] = []

    for pair in pairs:
        serializable.append(
            {
                "index_a": pair.index_a,
                "index_b": pair.index_b,
                "label_a": pair.label_a,
                "label_b": pair.label_b,
                "z_a": pair.z_a.tolist(),
                "z_b": pair.z_b.tolist(),
                "pair_type": pair.pair_type,
            }
        )

    return serializable


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    ensure_dir(RUN_PARTA_OUTPUT_DIR)

    device = torch.device(DEVICE)
    model_config = PartAVAEConfig()
    train_config = PartATrainConfig(seed=SEED, device=DEVICE)

    # -------------------------------------------------------------------------
    # Step 1: train or load model
    # -------------------------------------------------------------------------
    if TRAIN_NEW_MODEL:
        print("[Part A] Training a new Gaussian VAE...")
        model, history = train_parta_vae(
            model_config=model_config,
            train_config=train_config,
        )
        print(f"[Part A] Training complete. History length: {len(history)} epochs")

    elif LOAD_EXISTING_CHECKPOINT:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {CHECKPOINT_PATH}. "
                "Either train a new model or provide a valid checkpoint."
            )

        print(f"[Part A] Loading existing checkpoint from {CHECKPOINT_PATH}...")
        model = load_trained_parta_vae(
            checkpoint_path=CHECKPOINT_PATH,
            device=device,
            model_config=model_config,
        )

    else:
        raise ValueError("Either TRAIN_NEW_MODEL or LOAD_EXISTING_CHECKPOINT must be True.")

    # -------------------------------------------------------------------------
    # Step 2: load data and extract latent means
    # -------------------------------------------------------------------------
    print("[Part A] Loading data and extracting latent means...")
    data_bundle = get_parta_data(seed=SEED)
    latents, labels = extract_latent_means(model=model, data_bundle=data_bundle, device=device)

    np.savez(RUN_PARTA_LATENTS_PATH, latents=latents, labels=labels)
    plot_latent_scatter(latents=latents, labels=labels, save_path=PARTA_SCATTER_PATH)

    print(f"[Part A] Saved latent arrays to: {RUN_PARTA_LATENTS_PATH}")
    print(f"[Part A] Saved latent scatter plot to: {PARTA_SCATTER_PATH}")

    # -------------------------------------------------------------------------
    # Step 3: select pairs for geodesics
    # -------------------------------------------------------------------------
    print("[Part A] Selecting latent pairs...")
    pairs = select_latent_pairs(
        latents=latents,
        labels=labels,
        seed=SEED,
        n_same_class=10,
        n_cross_class=10,
        n_long_distance=5,
    )

    if len(pairs) != N_GEODESIC_PAIRS:
        raise RuntimeError(
            f"Expected {N_GEODESIC_PAIRS} pairs but got {len(pairs)}. "
            "Check pair-selection settings."
        )

    save_json({"pairs": latent_pairs_to_serializable(pairs)}, RUN_PARTA_PAIRS_PATH)
    print(f"[Part A] Saved selected pairs to: {RUN_PARTA_PAIRS_PATH}")

    # -------------------------------------------------------------------------
    # Step 4: compute geodesic candidates
    # -------------------------------------------------------------------------
    print("[Part A] Computing pull-back geodesic candidates...")
    curves = compute_geodesic_curves(
        model=model,
        pairs=pairs,
        device=device,
        n_nodes=GEODESIC_NODES,
        lr=GEODESIC_LR,
        max_iter=GEODESIC_MAX_ITER,
    )

    # -------------------------------------------------------------------------
    # Step 5: visualize final result
    # -------------------------------------------------------------------------
    print("[Part A] Saving final visualization...")
    plot_latent_scatter_with_curves(
        latents=latents,
        labels=labels,
        curves=curves,
        save_path=PARTA_SCATTER_WITH_CURVES_PATH,
        title="Part A: latent space with pull-back geodesic candidates",
    )

    print(f"[Part A] Saved final figure to: {PARTA_SCATTER_WITH_CURVES_PATH}")

    # -------------------------------------------------------------------------
    # Step 6: save run metadata
    # -------------------------------------------------------------------------
    save_json(
        {
            "seed": SEED,
            "device": DEVICE,
            "classes": PARTA_CLASSES,
            "n_pairs": N_GEODESIC_PAIRS,
            "geodesic_nodes": GEODESIC_NODES,
            "geodesic_lr": GEODESIC_LR,
            "geodesic_max_iter": GEODESIC_MAX_ITER,
            "train_new_model": TRAIN_NEW_MODEL,
            "load_existing_checkpoint": LOAD_EXISTING_CHECKPOINT,
            "checkpoint_path": str(CHECKPOINT_PATH),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
        },
        RUN_PARTA_RUNINFO_PATH,
    )

    print(f"[Part A] Saved run metadata to: {RUN_PARTA_RUNINFO_PATH}")
    print("[Part A] Pipeline finished successfully.")


if __name__ == "__main__":
    main()