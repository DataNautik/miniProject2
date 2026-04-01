"""Training utilities for Part A of Mini-project 2.

This module trains the Gaussian VAE defined in `partA_model.py` on the 3-class,
2048-sample MNIST subset prepared in `partA_data.py`.

The implementation focuses on:
- reproducibility,
- clean experiment logging,
- a light training loop with optional input noise,
- saving checkpoints and training artifacts for later geometry experiments.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from partA_data import PartADataBundle, PARTA_CLASSES, get_parta_data
from partA_model import PartAVAEConfig, PartAGaussianVAE, build_parta_vae


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PARTA_OUTPUT_DIR = Path("partA_outputs")
PARTA_MODEL_PATH = PARTA_OUTPUT_DIR / "partA_vae.pt"
PARTA_HISTORY_PATH = PARTA_OUTPUT_DIR / "train_history.json"
PARTA_CONFIG_PATH = PARTA_OUTPUT_DIR / "train_config.json"

PARTA_SEED = 0
PARTA_DEVICE = "mps"  # change to "cpu" or "cuda" if needed
PARTA_EPOCHS = 100
PARTA_LEARNING_RATE = 1e-3
PARTA_WEIGHT_DECAY = 0.0
PARTA_INPUT_NOISE_STD = 0.01
PARTA_USE_INPUT_NOISE = True
PARTA_CLIP_INPUTS_TO_UNIT_INTERVAL = True


# -----------------------------------------------------------------------------
# Config container
# -----------------------------------------------------------------------------

@dataclass
class PartATrainConfig:
    """Training configuration for Part A."""

    seed: int = PARTA_SEED
    device: str = PARTA_DEVICE
    epochs: int = PARTA_EPOCHS
    learning_rate: float = PARTA_LEARNING_RATE
    weight_decay: float = PARTA_WEIGHT_DECAY
    input_noise_std: float = PARTA_INPUT_NOISE_STD
    use_input_noise: bool = PARTA_USE_INPUT_NOISE
    clip_inputs_to_unit_interval: bool = PARTA_CLIP_INPUTS_TO_UNIT_INTERVAL
    output_dir: str = str(PARTA_OUTPUT_DIR)
    model_path: str = str(PARTA_MODEL_PATH)
    history_path: str = str(PARTA_HISTORY_PATH)
    config_path: str = str(PARTA_CONFIG_PATH)


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set all relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------


def ensure_output_dir(path: Path) -> None:
    """Create the output directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)



def save_json(data: dict[str, Any], path: Path) -> None:
    """Write a JSON dictionary to disk."""
    path.write_text(json.dumps(data, indent=2))


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------


def maybe_add_input_noise(
    x: torch.Tensor,
    noise_std: float,
    use_input_noise: bool,
    clip_to_unit_interval: bool,
) -> torch.Tensor:
    """Optionally add a small amount of Gaussian noise to the inputs.

    The handout notes that adding a bit of noise can improve the ensemble model
    used later in Part B. We keep the option here so the same training code can
    be re-used consistently.
    """
    if not use_input_noise or noise_std <= 0.0:
        return x

    x_noisy = x + noise_std * torch.randn_like(x)

    if clip_to_unit_interval:
        x_noisy = torch.clamp(x_noisy, 0.0, 1.0)

    return x_noisy


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------


def evaluate_epoch(model: PartAGaussianVAE, data_bundle: PartADataBundle, device: torch.device) -> dict[str, float]:
    """Evaluate the model on the validation split."""
    model.eval()

    totals = {
        "loss": 0.0,
        "elbo": 0.0,
        "recon_log_prob": 0.0,
        "kl": 0.0,
    }
    n_examples = 0

    with torch.no_grad():
        for x, _ in data_bundle.val_loader:
            x = x.to(device)
            _, metrics = model.loss(x)

            batch_size = x.shape[0]
            n_examples += batch_size
            for key in totals:
                totals[key] += metrics[key] * batch_size

    return {f"val_{key}": value / max(n_examples, 1) for key, value in totals.items()}


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: PartAGaussianVAE,
    data_bundle: PartADataBundle,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: PartATrainConfig,
) -> dict[str, float]:
    """Train the model for one epoch and return averaged metrics."""
    model.train()

    totals = {
        "loss": 0.0,
        "elbo": 0.0,
        "recon_log_prob": 0.0,
        "kl": 0.0,
    }
    n_examples = 0

    progress = tqdm(data_bundle.train_loader, desc="Training", leave=False)

    for x, _ in progress:
        x = x.to(device)
        x_train = maybe_add_input_noise(
            x=x,
            noise_std=config.input_noise_std,
            use_input_noise=config.use_input_noise,
            clip_to_unit_interval=config.clip_inputs_to_unit_interval,
        )

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.loss(x_train)
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        n_examples += batch_size
        for key in totals:
            totals[key] += metrics[key] * batch_size

        progress.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            elbo=f"{metrics['elbo']:.4f}",
            sigma=f"{metrics['decoder_sigma']:.4f}",
        )

    return {f"train_{key}": value / max(n_examples, 1) for key, value in totals.items()}



def train_parta_vae(
    model_config: PartAVAEConfig | None = None,
    train_config: PartATrainConfig | None = None,
) -> tuple[PartAGaussianVAE, list[dict[str, float]]]:
    """Train the Part A VAE and save artifacts to disk."""
    model_config = model_config if model_config is not None else PartAVAEConfig()
    train_config = train_config if train_config is not None else PartATrainConfig()

    set_seed(train_config.seed)
    device = torch.device(train_config.device)

    output_dir = Path(train_config.output_dir)
    model_path = Path(train_config.model_path)
    history_path = Path(train_config.history_path)
    config_path = Path(train_config.config_path)
    ensure_output_dir(output_dir)

    data_bundle = get_parta_data(seed=train_config.seed)
    model = build_parta_vae(config=model_config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")

    print("Part A training setup")
    print(f"Classes: {PARTA_CLASSES}")
    print(f"Device: {train_config.device}")
    print(f"Epochs: {train_config.epochs}")
    print(f"Decoder sigma mode: {model_config.decoder_sigma_mode}")
    print(f"Train size: {len(data_bundle.train_dataset)}")
    print(f"Validation size: {len(data_bundle.val_dataset)}")

    for epoch in range(train_config.epochs):
        train_metrics = train_one_epoch(
            model=model,
            data_bundle=data_bundle,
            optimizer=optimizer,
            device=device,
            config=train_config,
        )
        val_metrics = evaluate_epoch(model=model, data_bundle=data_bundle, device=device)

        epoch_metrics = {"epoch": epoch + 1, **train_metrics, **val_metrics}
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch + 1:03d}/{train_config.epochs:03d} | "
            f"train_loss={epoch_metrics['train_loss']:.4f} | "
            f"val_loss={epoch_metrics['val_loss']:.4f} | "
            f"train_elbo={epoch_metrics['train_elbo']:.4f} | "
            f"val_elbo={epoch_metrics['val_elbo']:.4f}"
        )

        if epoch_metrics["val_loss"] < best_val_loss:
            best_val_loss = epoch_metrics["val_loss"]
            torch.save(model.state_dict(), model_path)

    save_json({"history": history}, history_path)
    save_json(
        {
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
        },
        config_path,
    )

    print(f"Best model saved to: {model_path}")
    print(f"Training history saved to: {history_path}")
    print(f"Configuration saved to: {config_path}")

    # Reload best checkpoint before returning.
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, history


# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    train_parta_vae()
