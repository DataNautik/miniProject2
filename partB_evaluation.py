"""
Part B evaluation: fixed pair handling, Euclidean/geodesic distances, CoV aggregation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from partB_geometry import GeodesicConfig, optimize_ensemble_geodesic
from partB_model import EnsembleVAE


def _flatten_batch(batch) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        return batch[0].float().view(batch[0].shape[0], -1)
    return batch.float().view(batch.shape[0], -1)


def extract_latent_means(
    model: EnsembleVAE,
    loader: DataLoader,
    device: str,
) -> np.ndarray:
    """
    Posterior mean coordinates for all observations in the subset.
    Returns an array of shape [N, 2].
    """
    model.eval()
    zs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = _flatten_batch(batch).to(device)
            mu, _ = model.encode(x)
            zs.append(mu.cpu().numpy())
    latents = np.concatenate(zs, axis=0)
    if latents.shape[1] != 2:
        raise ValueError("Part B requires 2D latent coordinates.")
    return latents


def _normalize_pairs_payload(payload) -> list[list[int]]:
    if isinstance(payload, dict):
        if "pairs" in payload:
            payload = payload["pairs"]
        elif "selected_pairs" in payload:
            payload = payload["selected_pairs"]
        elif "pair_indices" in payload:
            payload = payload["pair_indices"]
        else:
            raise ValueError("Could not find a supported pair list key in the JSON file.")

    if not isinstance(payload, list):
        raise ValueError("Pair payload must be a list.")

    pairs = []
    for item in payload:
        # Case 1: plain [i, j]
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append([int(item[0]), int(item[1])])
            continue

        # Case 2: Part A serialized dict
        if isinstance(item, dict) and "index_a" in item and "index_b" in item:
            pairs.append([int(item["index_a"]), int(item["index_b"])])
            continue

        raise ValueError(
            "Each pair must be either [i, j] or a dict containing `index_a` and `index_b`."
        )

    return pairs


def load_or_create_pair_indices(
    path: Path,
    num_points: int,
    num_pairs: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Loads fixed pair indices if the JSON exists, otherwise creates and saves them once.
    """
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            pairs = _normalize_pairs_payload(json.load(f))
        pairs_np = np.asarray(pairs, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        pairs_np = np.empty((num_pairs, 2), dtype=np.int64)
        for i in range(num_pairs):
            a, b = rng.choice(num_points, size=2, replace=False)
            pairs_np[i] = [a, b]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"pairs": pairs_np.tolist()}, f, indent=2)

    if pairs_np.ndim != 2 or pairs_np.shape[1] != 2:
        raise ValueError("Pair array must have shape [M, 2].")
    if (pairs_np < 0).any() or (pairs_np >= num_points).any():
        raise IndexError("Pair indices are out of dataset range.")
    return pairs_np


def compute_euclidean_distances(
    latents: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    diffs = latents[pairs[:, 0]] - latents[pairs[:, 1]]
    return np.linalg.norm(diffs, axis=1)


def compute_geodesic_distances(
    model: EnsembleVAE,
    latents: np.ndarray,
    pairs: np.ndarray,
    geodesic_config: GeodesicConfig,
    device: str,
) -> tuple[np.ndarray, list[np.ndarray], list[list[float]]]:
    geo = np.zeros(len(pairs), dtype=np.float64)
    node_paths: list[np.ndarray] = []
    energy_histories: list[list[float]] = []

    for idx, (i, j) in enumerate(pairs):
        z0 = torch.tensor(latents[i], dtype=torch.float32, device=device)
        z1 = torch.tensor(latents[j], dtype=torch.float32, device=device)
        result = optimize_ensemble_geodesic(
            model=model,
            z_start=z0,
            z_end=z1,
            config=geodesic_config,
            device=device,
        )
        geo[idx] = float(result["distance"])
        node_paths.append(result["nodes"].detach().cpu().numpy())
        energy_histories.append([float(v) for v in result["energy_history"]])

    return geo, node_paths, energy_histories


def evaluate_model_distances(
    model: EnsembleVAE,
    latent_loader: DataLoader,
    pairs: np.ndarray,
    geodesic_config: GeodesicConfig,
    device: str,
    out_dir: Optional[Path] = None,
    save_paths: bool = True,
) -> Dict[str, np.ndarray | list[np.ndarray] | list[list[float]]]:
    latents = extract_latent_means(model, latent_loader, device=device)
    euclidean = compute_euclidean_distances(latents, pairs)
    geodesic, node_paths, energy_histories = compute_geodesic_distances(
        model=model,
        latents=latents,
        pairs=pairs,
        geodesic_config=geodesic_config,
        device=device,
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "latent_means.npy", latents)
        np.savez(
            out_dir / "distance_results.npz",
            pair_indices=pairs,
            euclidean=euclidean,
            geodesic=geodesic,
        )
        with open(out_dir / "distance_results.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pair_indices": pairs.tolist(),
                    "euclidean": euclidean.tolist(),
                    "geodesic": geodesic.tolist(),
                },
                f,
                indent=2,
            )
        with open(out_dir / "distance_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_id", "idx_a", "idx_b", "euclidean", "geodesic"])
            for pair_id, ((a, b), euc, geo) in enumerate(zip(pairs.tolist(), euclidean, geodesic)):
                writer.writerow([pair_id, a, b, float(euc), float(geo)])

        if save_paths:
            paths_dir = out_dir / "geodesic_paths"
            paths_dir.mkdir(parents=True, exist_ok=True)
            for pair_id, (nodes, hist) in enumerate(zip(node_paths, energy_histories)):
                np.save(paths_dir / f"pair_{pair_id:03d}_nodes.npy", nodes)
                with open(paths_dir / f"pair_{pair_id:03d}_energy_history.json", "w", encoding="utf-8") as f:
                    json.dump(hist, f, indent=2)

    return {
        "latents": latents,
        "euclidean": euclidean,
        "geodesic": geodesic,
        "node_paths": node_paths,
        "energy_histories": energy_histories,
    }


def aggregate_cov(
    per_seed_results: Dict[int, Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray | float]:
    """
    Aggregates a fixed-pair distance matrix across retrainings and computes CoV.

    Input:
        per_seed_results[seed]["euclidean"] -> [P]
        per_seed_results[seed]["geodesic"]  -> [P]

    Output:
        mean/std/cov arrays per pair, plus global means over pairs.
    """
    seeds = sorted(per_seed_results.keys())
    if not seeds:
        raise ValueError("No per-seed results provided.")

    euclidean_stack = np.stack([per_seed_results[s]["euclidean"] for s in seeds], axis=0)  # [S, P]
    geodesic_stack = np.stack([per_seed_results[s]["geodesic"] for s in seeds], axis=0)    # [S, P]

    euclidean_mean = euclidean_stack.mean(axis=0)
    geodesic_mean = geodesic_stack.mean(axis=0)
    euclidean_std = euclidean_stack.std(axis=0)
    geodesic_std = geodesic_stack.std(axis=0)

    eps = 1e-12
    euclidean_cov = euclidean_std / np.maximum(euclidean_mean, eps)
    geodesic_cov = geodesic_std / np.maximum(geodesic_mean, eps)

    return {
        "seeds": np.asarray(seeds, dtype=np.int64),
        "euclidean_mean": euclidean_mean,
        "euclidean_std": euclidean_std,
        "euclidean_cov": euclidean_cov,
        "geodesic_mean": geodesic_mean,
        "geodesic_std": geodesic_std,
        "geodesic_cov": geodesic_cov,
        "euclidean_cov_global_mean": float(euclidean_cov.mean()),
        "geodesic_cov_global_mean": float(geodesic_cov.mean()),
    }


def save_cov_summary(
    summary: Dict[str, np.ndarray | float],
    out_dir: Path,
    num_decoders: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / f"cov_summary_k{num_decoders}.npz", **summary)

    serializable = {}
    for key, value in summary.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value

    with open(out_dir / f"cov_summary_k{num_decoders}.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    with open(out_dir / f"cov_summary_k{num_decoders}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pair_id",
                "euclidean_mean",
                "euclidean_std",
                "euclidean_cov",
                "geodesic_mean",
                "geodesic_std",
                "geodesic_cov",
            ]
        )
        n_pairs = len(summary["euclidean_mean"])
        for i in range(n_pairs):
            writer.writerow(
                [
                    i,
                    float(summary["euclidean_mean"][i]),
                    float(summary["euclidean_std"][i]),
                    float(summary["euclidean_cov"][i]),
                    float(summary["geodesic_mean"][i]),
                    float(summary["geodesic_std"][i]),
                    float(summary["geodesic_cov"][i]),
                ]
            )
