"""
Part B model: ensemble-decoder VAE for 2D MNIST latent geometry.

Design choices:
- Shared encoder.
- `num_decoders` decoder heads collected in an nn.ModuleList.
- Independent Gaussian likelihood for each decoder head.
- Ensemble ELBO uses the average reconstruction log-probability across decoders
  and a single KL term.

The code is intentionally compact and readable rather than overly abstract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EnsembleVAEConfig:
    input_dim: int = 28 * 28
    latent_dim: int = 2
    hidden_dim: int = 256
    num_decoders: int = 1
    beta: float = 1.0
    min_decoder_sigma: float = 1e-3
    init_decoder_log_sigma: float = -0.5

    def __post_init__(self) -> None:
        if self.latent_dim != 2:
            raise ValueError("Part B requires latent_dim=2.")
        if self.num_decoders < 1:
            raise ValueError("num_decoders must be >= 1.")


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class EnsembleVAE(nn.Module):
    """
    VAE with a single encoder and multiple decoder heads.

    Reconstruction term:
        avg_k log p_k(x | z)

    Total objective:
        E_q(z|x)[ avg_k log p_k(x | z) ] - beta * KL(q(z|x) || p(z))
    """

    def __init__(self, config: EnsembleVAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = MLPEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
        )
        self.decoders = nn.ModuleList(
            [
                MLPDecoder(
                    latent_dim=config.latent_dim,
                    hidden_dim=config.hidden_dim,
                    output_dim=config.input_dim,
                )
                for _ in range(config.num_decoders)
            ]
        )
        self._decoder_log_sigma = nn.Parameter(
            torch.full((config.num_decoders,), float(config.init_decoder_log_sigma))
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def posterior(self, x: torch.Tensor) -> torch.distributions.Normal:
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        return torch.distributions.Normal(mu, std)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode_mean(self, z: torch.Tensor, decoder_idx: int) -> torch.Tensor:
        return self.decoders[decoder_idx](z)

    def decode_all_means(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor of shape [K, B, D] where:
            K = num_decoders
            B = batch size
            D = input dimension
        """
        return torch.stack([decoder(z) for decoder in self.decoders], dim=0)

    def decoder_sigma(self, decoder_idx: Optional[int] = None) -> torch.Tensor:
        """
        Returns the decoder standard deviation(s).

        - If decoder_idx is None: shape [K]
        - Else: scalar tensor for one decoder
        """
        sigma = torch.exp(self._decoder_log_sigma).clamp_min(self.config.min_decoder_sigma)
        if decoder_idx is None:
            return sigma
        return sigma[decoder_idx]

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Per-example KL(q(z|x) || p(z)) for diagonal Gaussian posterior and standard normal prior.
        """
        return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=-1)

    def reconstruction_log_prob(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Returns per-example reconstruction log-probability averaged across decoders.
        Shape: [B]
        """
        x = x.view(x.shape[0], -1)
        means = self.decode_all_means(z)                      # [K, B, D]
        x_expanded = x.unsqueeze(0)                          # [1, B, D]
        sigma = self.decoder_sigma()[:, None, None]          # [K, 1, 1]
        dist = torch.distributions.Normal(means, sigma)
        log_probs = dist.log_prob(x_expanded).sum(dim=-1)    # [K, B]
        return log_probs.mean(dim=0)                         # [B]

    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        x = x.view(x.shape[0], -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, sample=sample)
        means = self.decode_all_means(z)
        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "decoder_means": means,
        }

    def loss(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns scalar loss plus per-batch logging quantities.
        """
        x = x.view(x.shape[0], -1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, sample=sample)

        recon_log_prob = self.reconstruction_log_prob(x, z)  # [B]
        kl = self.kl_divergence(mu, logvar)                  # [B]
        elbo = recon_log_prob - self.config.beta * kl        # [B]

        loss = -elbo.mean()
        return {
            "loss": loss,
            "elbo": elbo.mean().detach(),
            "recon_log_prob": recon_log_prob.mean().detach(),
            "kl": kl.mean().detach(),
        }
