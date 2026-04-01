"""Part A model definition for Mini-project 2.

This module implements a compact Gaussian VAE for the Part A experiments:
- 2D latent space,
- standard Gaussian prior,
- Gaussian likelihood p(x|z),
- configurable decoder variance mode: fixed or global.

The implementation is intentionally simple, stable, and easy to reuse in the
training and geometry modules.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Literal, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

LATENT_DIM: int = 2
INPUT_DIM: int = 28 * 28
ENCODER_HIDDEN_DIMS: Tuple[int, int] = (256, 128)
DECODER_HIDDEN_DIMS: Tuple[int, int] = (128, 256)

DECODER_SIGMA_MODE: Literal["fixed", "global"] = "fixed"
FIXED_SIGMA_VALUE: float = 0.10
MIN_SIGMA_VALUE: float = 1e-4


# -----------------------------------------------------------------------------
# Config container
# -----------------------------------------------------------------------------

@dataclass
class PartAVAEConfig:
    """Configuration for the Part A Gaussian VAE."""

    input_dim: int = INPUT_DIM
    latent_dim: int = LATENT_DIM
    encoder_hidden_dims: Tuple[int, int] = ENCODER_HIDDEN_DIMS
    decoder_hidden_dims: Tuple[int, int] = DECODER_HIDDEN_DIMS
    decoder_sigma_mode: Literal["fixed", "global"] = DECODER_SIGMA_MODE
    fixed_sigma_value: float = FIXED_SIGMA_VALUE
    min_sigma_value: float = MIN_SIGMA_VALUE


# -----------------------------------------------------------------------------
# Network builders
# -----------------------------------------------------------------------------


def _build_mlp(input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int) -> nn.Sequential:
    """Build a simple MLP with ReLU activations."""
    layers: list[nn.Module] = []
    prev_dim = input_dim

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# Gaussian VAE
# -----------------------------------------------------------------------------

class PartAGaussianVAE(nn.Module):
    """Compact Gaussian VAE with 2D latent space for Part A.

    Encoder:
        x -> q(z|x) = N(mu_z(x), diag(sigma_z(x)^2))

    Decoder:
        z -> p(x|z) = N(mu_x(z), sigma_x^2 I)

    The decoder variance can be handled in two ways:
        - fixed: sigma_x is a user-defined constant
        - global: sigma_x is a single learned scalar shared across all pixels
    """

    def __init__(self, config: PartAVAEConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else PartAVAEConfig()

        # ---------------------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------------------
        self.encoder = _build_mlp(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.encoder_hidden_dims,
            output_dim=2 * self.config.latent_dim,
        )

        # ---------------------------------------------------------------------
        # Decoder mean
        # ---------------------------------------------------------------------
        self.decoder_mean = _build_mlp(
            input_dim=self.config.latent_dim,
            hidden_dims=self.config.decoder_hidden_dims,
            output_dim=self.config.input_dim,
        )

        # ---------------------------------------------------------------------
        # Decoder variance mode
        # ---------------------------------------------------------------------
        if self.config.decoder_sigma_mode == "fixed":
            self.register_buffer(
                "fixed_sigma",
                torch.tensor(float(self.config.fixed_sigma_value), dtype=torch.float32),
            )
            self.global_log_sigma = None

        elif self.config.decoder_sigma_mode == "global":
            initial_log_sigma = math.log(float(self.config.fixed_sigma_value))
            self.global_log_sigma = nn.Parameter(torch.tensor(initial_log_sigma, dtype=torch.float32))
            self.fixed_sigma = None

        else:
            raise ValueError(
                "decoder_sigma_mode must be either 'fixed' or 'global', "
                f"got {self.config.decoder_sigma_mode!r}."
            )

    # -------------------------------------------------------------------------
    # Encoder helpers
    # -------------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return encoder mean and log-variance for q(z|x)."""
        h = self.encoder(x)
        mu_z, log_var_z = torch.chunk(h, 2, dim=-1)
        return mu_z, log_var_z

    def posterior(self, x: torch.Tensor) -> td.Independent:
        """Return the approximate posterior q(z|x)."""
        mu_z, log_var_z = self.encode(x)
        std_z = torch.exp(0.5 * log_var_z)
        return td.Independent(td.Normal(loc=mu_z, scale=std_z), 1)

    # -------------------------------------------------------------------------
    # Decoder helpers
    # -------------------------------------------------------------------------

    def decode_mean(self, z: torch.Tensor) -> torch.Tensor:
        """Return the decoder mean mu_x(z)."""
        return self.decoder_mean(z)

    def decoder_sigma(self) -> torch.Tensor:
        """Return the decoder standard deviation sigma_x.

        Returns:
            A scalar tensor. It is clamped from below for numerical stability.
        """
        if self.config.decoder_sigma_mode == "fixed":
            sigma = self.fixed_sigma
        else:
            sigma = torch.exp(self.global_log_sigma)

        return torch.clamp(sigma, min=self.config.min_sigma_value)

    def likelihood(self, z: torch.Tensor) -> td.Independent:
        """Return p(x|z) as a diagonal Gaussian distribution."""
        mu_x = self.decode_mean(z)
        sigma_x = self.decoder_sigma()
        scale = torch.ones_like(mu_x) * sigma_x
        return td.Independent(td.Normal(loc=mu_x, scale=scale), 1)

    # -------------------------------------------------------------------------
    # Prior
    # -------------------------------------------------------------------------

    def prior(self, batch_size: int, device: torch.device) -> td.Independent:
        """Return the standard Gaussian prior p(z)."""
        loc = torch.zeros(batch_size, self.config.latent_dim, device=device)
        scale = torch.ones(batch_size, self.config.latent_dim, device=device)
        return td.Independent(td.Normal(loc=loc, scale=scale), 1)

    # -------------------------------------------------------------------------
    # Forward / loss
    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run a forward pass and return the main VAE quantities."""
        q_z = self.posterior(x)
        z = q_z.rsample()
        p_x_given_z = self.likelihood(z)
        p_z = self.prior(batch_size=x.shape[0], device=x.device)

        recon_log_prob = p_x_given_z.log_prob(x)
        kl = td.kl_divergence(q_z, p_z)
        elbo = recon_log_prob - kl
        loss = -elbo.mean()

        return {
            "z": z,
            "recon_log_prob": recon_log_prob,
            "kl": kl,
            "elbo": elbo,
            "loss": loss,
        }

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Return the optimization loss and logging metrics."""
        outputs = self.forward(x)

        loss = outputs["loss"]
        metrics = {
            "loss": float(loss.detach().item()),
            "elbo": float(outputs["elbo"].mean().detach().item()),
            "recon_log_prob": float(outputs["recon_log_prob"].mean().detach().item()),
            "kl": float(outputs["kl"].mean().detach().item()),
            "decoder_sigma": float(self.decoder_sigma().detach().item()),
        }
        return loss, metrics

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def sample_prior(self, n_samples: int, device: torch.device | None = None) -> torch.Tensor:
        """Sample x from the generative model p(z)p(x|z)."""
        if device is None:
            device = next(self.parameters()).device

        p_z = self.prior(batch_size=n_samples, device=device)
        z = p_z.sample()
        p_x_given_z = self.likelihood(z)
        return p_x_given_z.sample()

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Return the decoder mean of the posterior mean latent code."""
        mu_z, _ = self.encode(x)
        return self.decode_mean(mu_z)

    @torch.no_grad()
    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Return the posterior mean embedding for plotting and geometry."""
        mu_z, _ = self.encode(x)
        return mu_z


# -----------------------------------------------------------------------------
# Public builder
# -----------------------------------------------------------------------------


def build_parta_vae(config: PartAVAEConfig | None = None) -> PartAGaussianVAE:
    """Build the Part A Gaussian VAE."""
    return PartAGaussianVAE(config=config)


# -----------------------------------------------------------------------------
# Quick manual check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_parta_vae()
    x = torch.rand(8, INPUT_DIM)

    outputs = model.forward(x)
    print("Part A Gaussian VAE check")
    print(f"z shape: {tuple(outputs['z'].shape)}")
    print(f"loss: {outputs['loss'].item():.4f}")
    print(f"decoder sigma: {model.decoder_sigma().item():.4f}")