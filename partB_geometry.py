"""
Part B geometry: exact ensemble decoder-pair energy and geodesic optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from partB_model import EnsembleVAE


@dataclass
class GeodesicConfig:
    num_nodes: int = 16
    max_steps: int = 60
    lbfgs_lr: float = 0.5
    tolerance: float = 1e-7
    line_search_fn: str = "strong_wolfe"

    def __post_init__(self) -> None:
        if self.num_nodes < 2:
            raise ValueError("num_nodes must be at least 2.")


def initialize_piecewise_linear_curve(
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """
    Returns linearly interpolated nodes with shape [num_nodes, latent_dim].
    """
    if z_start.ndim != 1 or z_end.ndim != 1:
        raise ValueError("z_start and z_end must be 1D latent vectors.")
    ts = torch.linspace(0.0, 1.0, steps=num_nodes, device=z_start.device, dtype=z_start.dtype)
    return (1.0 - ts[:, None]) * z_start[None, :] + ts[:, None] * z_end[None, :]


def _pairwise_decoder_segment_energy(
    model: EnsembleVAE,
    z0: torch.Tensor,
    z1: torch.Tensor,
) -> torch.Tensor:
    """
    Exact average over all decoder pairs:
        (1 / K^2) sum_{l,k} || f_l(z0) - f_k(z1) ||^2
    """
    with torch.set_grad_enabled(torch.is_grad_enabled()):
        x0 = model.decode_all_means(z0.unsqueeze(0)).squeeze(1)  # [K, D]
        x1 = model.decode_all_means(z1.unsqueeze(0)).squeeze(1)  # [K, D]
        diff = x0[:, None, :] - x1[None, :, :]                   # [K, K, D]
        return diff.pow(2).sum(dim=-1).mean()


def ensemble_curve_energy(
    model: EnsembleVAE,
    nodes: torch.Tensor,
) -> torch.Tensor:
    """
    Sum of exact ensemble segment energies along a piecewise-linear curve.
    """
    if nodes.ndim != 2:
        raise ValueError("nodes must have shape [num_nodes, latent_dim].")
    total = torch.zeros((), device=nodes.device, dtype=nodes.dtype)
    for i in range(nodes.shape[0] - 1):
        total = total + _pairwise_decoder_segment_energy(model, nodes[i], nodes[i + 1])
    return total


def optimize_ensemble_geodesic(
    model: EnsembleVAE,
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    config: GeodesicConfig,
    device: Optional[str] = None,
) -> Dict[str, torch.Tensor | list[float] | float]:
    """
    Optimizes only the interior nodes of a piecewise-linear latent curve with LBFGS.

    Returns:
        {
            "nodes": tensor [N, 2],
            "energy_history": list[float],
            "initial_energy": float,
            "final_energy": float,
            "distance": float,
        }

    Distance is reported as sqrt(energy), which is the natural scale-compatible
    quantity for comparing against Euclidean latent distances.
    """
    model.eval()
    dev = device or next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    z_start = z_start.detach().to(dev, dtype=dtype)
    z_end = z_end.detach().to(dev, dtype=dtype)

    if torch.allclose(z_start, z_end):
        nodes = torch.stack([z_start, z_end], dim=0)
        return {
            "nodes": nodes,
            "energy_history": [0.0],
            "initial_energy": 0.0,
            "final_energy": 0.0,
            "distance": 0.0,
        }

    with torch.no_grad():
        init_nodes = initialize_piecewise_linear_curve(z_start, z_end, config.num_nodes)
        initial_energy = float(ensemble_curve_energy(model, init_nodes).detach().cpu())

    if config.num_nodes == 2:
        final_energy = initial_energy
        return {
            "nodes": init_nodes,
            "energy_history": [initial_energy],
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "distance": float(max(final_energy, 0.0) ** 0.5),
        }

    interior = init_nodes[1:-1].clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [interior],
        lr=config.lbfgs_lr,
        max_iter=1,
        line_search_fn=config.line_search_fn,
    )

    energy_history = [initial_energy]
    best_energy = initial_energy
    best_nodes = init_nodes.detach().clone()

    def assemble_nodes() -> torch.Tensor:
        return torch.cat([z_start[None, :], interior, z_end[None, :]], dim=0)

    for _ in range(config.max_steps):
        def closure():
            optimizer.zero_grad(set_to_none=True)
            nodes = assemble_nodes()
            energy = ensemble_curve_energy(model, nodes)
            if not torch.isfinite(energy):
                raise FloatingPointError("Non-finite geodesic energy encountered.")
            energy.backward()
            return energy

        optimizer.step(closure)

        with torch.no_grad():
            nodes_now = assemble_nodes()
            energy_now = float(ensemble_curve_energy(model, nodes_now).detach().cpu())
            if not (energy_now == energy_now):  # NaN check
                raise FloatingPointError("NaN geodesic energy encountered.")
            energy_history.append(energy_now)

            if energy_now < best_energy:
                best_energy = energy_now
                best_nodes = nodes_now.detach().clone()

            if abs(energy_history[-2] - energy_history[-1]) <= config.tolerance:
                break

    if best_energy > initial_energy + 1e-8:
        raise AssertionError(
            f"Geodesic optimization failed to decrease energy: "
            f"initial={initial_energy:.6f}, best={best_energy:.6f}"
        )

    return {
        "nodes": best_nodes,
        "energy_history": energy_history,
        "initial_energy": initial_energy,
        "final_energy": best_energy,
        "distance": float(max(best_energy, 0.0) ** 0.5),
    }
