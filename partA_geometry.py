"""Geometry utilities for Part A of Mini-project 2.

This module implements the core geometric tools needed for pull-back geodesics
under the decoder mean of the Gaussian VAE:

- a piecewise-linear latent curve representation,
- the energy of a decoded curve,
- optimization of connecting geodesics,
- reproducible pair selection in latent space.

The implementation follows a practical discrete formulation: instead of building
Jacobian-based energies explicitly, we optimize the energy of the decoded
piecewise-linear curve. This is simple, differentiable, and well suited for the
project report.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.optim as optim

from partA_model import PartAGaussianVAE


# -----------------------------------------------------------------------------
# Pair container
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class LatentPair:
    """A pair of latent endpoints together with metadata for plotting/reporting."""

    index_a: int
    index_b: int
    label_a: int
    label_b: int
    z_a: np.ndarray
    z_b: np.ndarray
    pair_type: str


# -----------------------------------------------------------------------------
# Piecewise-linear latent curve
# -----------------------------------------------------------------------------

class PiecewiseLinearCurve:
    """Piecewise-linear latent curve with fixed endpoints and learnable interior points."""

    def __init__(self, z_start: torch.Tensor, z_end: torch.Tensor, n_nodes: int = 20) -> None:
        if z_start.ndim != 1 or z_end.ndim != 1:
            raise ValueError("z_start and z_end must be 1D tensors")
        if z_start.shape != z_end.shape:
            raise ValueError("z_start and z_end must have the same shape")
        if n_nodes < 2:
            raise ValueError("n_nodes must be at least 2")

        self.z_start = z_start.detach().clone().view(1, -1)
        self.z_end = z_end.detach().clone().view(1, -1)
        self.n_nodes = n_nodes

        t = torch.linspace(0.0, 1.0, n_nodes, device=z_start.device).view(n_nodes, 1)
        line = (1.0 - t) * self.z_start + t * self.z_end

        if n_nodes > 2:
            interior = line[1:-1].clone().detach().requires_grad_(True)
        else:
            interior = torch.empty(0, z_start.numel(), device=z_start.device, requires_grad=True)

        self.interior = interior

    def points(self) -> torch.Tensor:
        """Return all curve points including endpoints, shape (K, 2)."""
        if self.interior.numel() == 0:
            return torch.cat((self.z_start, self.z_end), dim=0)
        return torch.cat((self.z_start, self.interior, self.z_end), dim=0)

    @torch.no_grad()
    def as_numpy(self) -> np.ndarray:
        """Return all curve points as a NumPy array."""
        return self.points().detach().cpu().numpy()


# -----------------------------------------------------------------------------
# Energy of decoded curves
# -----------------------------------------------------------------------------


def decode_curve_mean(model: PartAGaussianVAE, curve_points: torch.Tensor) -> torch.Tensor:
    """Decode latent curve points using the decoder mean map.

    Args:
        model: Trained Part A Gaussian VAE.
        curve_points: Tensor of shape (K, 2).

    Returns:
        Tensor of shape (K, 784).
    """
    return model.decode_mean(curve_points)



def curve_energy(model: PartAGaussianVAE, curve_points: torch.Tensor) -> torch.Tensor:
    """Compute the discrete pull-back energy of a curve.

    We use the decoded piecewise-linear curve and define the energy as

        E(c) = sum_i ||mu(z_{i+1}) - mu(z_i)||^2,

    where mu(.) is the decoder mean. This corresponds to a practical discrete
    approximation of the pull-back curve energy and is easy to optimize with
    autodiff.

    Args:
        model: Trained Part A Gaussian VAE.
        curve_points: Tensor of shape (K, 2).

    Returns:
        Scalar tensor with the curve energy.
    """
    decoded = decode_curve_mean(model, curve_points)  # (K, 784)
    delta = decoded[1:] - decoded[:-1]  # (K-1, 784)
    return torch.sum(delta * delta)



def curve_length(model: PartAGaussianVAE, curve_points: torch.Tensor) -> torch.Tensor:
    """Compute the discrete decoded curve length.

    This is not the quantity we optimize, but it is useful for reporting.
    """
    decoded = decode_curve_mean(model, curve_points)
    delta = decoded[1:] - decoded[:-1]
    segment_lengths = torch.linalg.norm(delta, dim=1)
    return torch.sum(segment_lengths)


# -----------------------------------------------------------------------------
# Geodesic optimization
# -----------------------------------------------------------------------------


def optimize_geodesic(
    model: PartAGaussianVAE,
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    n_nodes: int = 20,
    lr: float = 0.5,
    max_iter: int = 200,
) -> PiecewiseLinearCurve:
    """Optimize a connecting geodesic between two latent points.

    The endpoints are kept fixed. Only the interior nodes are optimized.

    Args:
        model: Trained Part A Gaussian VAE.
        z_start: Tensor of shape (2,).
        z_end: Tensor of shape (2,).
        n_nodes: Total number of nodes in the piecewise-linear curve.
        lr: LBFGS learning rate.
        max_iter: Maximum number of LBFGS steps.

    Returns:
        Optimized PiecewiseLinearCurve.
    """
    model.eval()
    curve = PiecewiseLinearCurve(z_start=z_start, z_end=z_end, n_nodes=n_nodes)

    if curve.interior.numel() == 0:
        return curve

    optimizer = optim.LBFGS([curve.interior], lr=lr, max_iter=20, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        energy = curve_energy(model, curve.points())
        energy.backward()
        return energy

    for _ in range(max_iter):
        optimizer.step(closure)

    return curve


# -----------------------------------------------------------------------------
# Pair selection
# -----------------------------------------------------------------------------


def _sample_unique_pairs(indices_a: np.ndarray, indices_b: np.ndarray, n_pairs: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Sample unique index pairs from two index sets."""
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    max_trials = 10000
    trials = 0

    while len(pairs) < n_pairs and trials < max_trials:
        ia = int(rng.choice(indices_a))
        ib = int(rng.choice(indices_b))
        trials += 1

        if ia == ib:
            continue

        key = tuple(sorted((ia, ib)))
        if key in seen:
            continue

        seen.add(key)
        pairs.append((ia, ib))

    if len(pairs) < n_pairs:
        raise RuntimeError("Could not sample the requested number of unique pairs")

    return pairs



def select_latent_pairs(
    latents: np.ndarray,
    labels: np.ndarray,
    seed: int = 0,
    n_same_class: int = 10,
    n_cross_class: int = 10,
    n_long_distance: int = 5,
) -> list[LatentPair]:
    """Select a reproducible and diverse set of latent pairs.

    Strategy:
    - same-class pairs for local within-class behavior,
    - cross-class pairs for inter-class behavior,
    - long-distance pairs for visually interesting geodesics.
    """
    rng = np.random.default_rng(seed)
    classes = sorted(np.unique(labels).tolist())
    pairs: list[LatentPair] = []
    used: set[tuple[int, int]] = set()

    # -------------------------------------------------------------------------
    # Same-class pairs
    # -------------------------------------------------------------------------
    same_pairs: list[tuple[int, int]] = []

    base_pairs_per_class = n_same_class // max(len(classes), 1)
    remainder_pairs = n_same_class % max(len(classes), 1)

    for class_offset, class_id in enumerate(classes):
        idx = np.where(labels == class_id)[0]
        n_pairs_for_class = base_pairs_per_class + (1 if class_offset < remainder_pairs else 0)

        if n_pairs_for_class == 0:
            continue

        sampled = _sample_unique_pairs(idx, idx, n_pairs_for_class, rng)
        same_pairs.extend(sampled)

    for ia, ib in same_pairs:
        key = tuple(sorted((ia, ib)))
        used.add(key)
        pairs.append(
            LatentPair(
                index_a=ia,
                index_b=ib,
                label_a=int(labels[ia]),
                label_b=int(labels[ib]),
                z_a=latents[ia].copy(),
                z_b=latents[ib].copy(),
                pair_type="same_class",
            )
        )

    # -------------------------------------------------------------------------
    # Cross-class pairs
    # -------------------------------------------------------------------------
    cross_pairs: list[tuple[int, int]] = []
    max_trials = 10000
    trials = 0

    while len(cross_pairs) < n_cross_class and trials < max_trials:
        ia = int(rng.integers(0, len(latents)))
        ib = int(rng.integers(0, len(latents)))
        trials += 1

        if ia == ib:
            continue
        if labels[ia] == labels[ib]:
            continue

        key = tuple(sorted((ia, ib)))
        if key in used:
            continue

        used.add(key)
        cross_pairs.append((ia, ib))

    if len(cross_pairs) < n_cross_class:
        raise RuntimeError("Could not sample enough cross-class pairs")

    for ia, ib in cross_pairs:
        pairs.append(
            LatentPair(
                index_a=ia,
                index_b=ib,
                label_a=int(labels[ia]),
                label_b=int(labels[ib]),
                z_a=latents[ia].copy(),
                z_b=latents[ib].copy(),
                pair_type="cross_class",
            )
        )

    # -------------------------------------------------------------------------
    # Long-distance pairs
    # -------------------------------------------------------------------------
    dists = np.linalg.norm(latents[:, None, :] - latents[None, :, :], axis=-1)
    triu_i, triu_j = np.triu_indices(len(latents), k=1)
    flat_pairs = list(zip(triu_i.tolist(), triu_j.tolist(), dists[triu_i, triu_j].tolist()))
    flat_pairs.sort(key=lambda item: item[2], reverse=True)

    added_long = 0
    for ia, ib, _ in flat_pairs:
        key = tuple(sorted((ia, ib)))
        if key in used:
            continue

        used.add(key)
        pairs.append(
            LatentPair(
                index_a=ia,
                index_b=ib,
                label_a=int(labels[ia]),
                label_b=int(labels[ib]),
                z_a=latents[ia].copy(),
                z_b=latents[ib].copy(),
                pair_type="long_distance",
            )
        )
        added_long += 1
        if added_long >= n_long_distance:
            break

    if added_long < n_long_distance:
        raise RuntimeError("Could not add enough long-distance pairs")

    return pairs


# -----------------------------------------------------------------------------
# Batch geodesic computation helper
# -----------------------------------------------------------------------------


def compute_geodesic_curves(
    model: PartAGaussianVAE,
    pairs: Iterable[LatentPair],
    device: torch.device,
    n_nodes: int = 20,
    lr: float = 0.5,
    max_iter: int = 200,
) -> list[np.ndarray]:
    """Compute one optimized piecewise-linear curve for each latent pair."""
    curves: list[np.ndarray] = []

    for pair in pairs:
        z_start = torch.tensor(pair.z_a, dtype=torch.float32, device=device)
        z_end = torch.tensor(pair.z_b, dtype=torch.float32, device=device)

        curve = optimize_geodesic(
            model=model,
            z_start=z_start,
            z_end=z_end,
            n_nodes=n_nodes,
            lr=lr,
            max_iter=max_iter,
        )
        curves.append(curve.as_numpy())

    return curves