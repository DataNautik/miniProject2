"""
Part B training utilities.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from partB_model import EnsembleVAE, EnsembleVAEConfig


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = None
    log_every: int = 50
    device: str = "cpu"


def _flatten_batch(batch) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        x = batch[0]
    else:
        x = batch
    return x.float().view(x.shape[0], -1)


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite values detected in {name}.")


def run_epoch(
    model: EnsembleVAE,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    train: bool,
    log_every: int = 50,
    grad_clip_norm: Optional[float] = None,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    totals = {"loss": 0.0, "elbo": 0.0, "recon_log_prob": 0.0, "kl": 0.0}
    n_examples = 0

    for step, batch in enumerate(loader):
        x = _flatten_batch(batch).to(device)
        batch_size = x.shape[0]

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            outputs = model.loss(x, sample=True)
            _check_finite("training loss", outputs["loss"])
            outputs["loss"].backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model.loss(x, sample=False)

        for key in totals:
            totals[key] += float(outputs[key].detach().cpu()) * batch_size
        n_examples += batch_size

        if train and log_every > 0 and (step + 1) % log_every == 0:
            print(
                f"[train] step={step + 1:04d} "
                f"loss={float(outputs['loss']):.4f} "
                f"elbo={float(outputs['elbo']):.4f} "
                f"recon={float(outputs['recon_log_prob']):.4f} "
                f"kl={float(outputs['kl']):.4f}"
            )

    if n_examples == 0:
        raise RuntimeError("Empty dataloader.")

    return {key: val / n_examples for key, val in totals.items()}


def fit_ensemble_vae(
    model: EnsembleVAE,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    train_config: TrainConfig,
    out_dir: Optional[Path] = None,
    resume_checkpoint: Optional[Path] = None,
) -> Dict[str, list[float]]:
    """
    Trains the model and optionally writes checkpoints/history under `out_dir`.
    """
    device = train_config.device
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    start_epoch = 0
    history = {
        "train_loss": [],
        "train_elbo": [],
        "train_recon_log_prob": [],
        "train_kl": [],
        "eval_loss": [],
        "eval_elbo": [],
        "eval_recon_log_prob": [],
        "eval_kl": [],
    }

    if resume_checkpoint is not None and resume_checkpoint.exists():
     start_epoch = 0
     best_eval_loss = float("inf")
     history = []

     if resume_checkpoint is not None and Path(resume_checkpoint).exists():
        try:
            payload = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            start_epoch = int(payload.get("epoch", 0))
            best_eval_loss = float(payload.get("best_eval_loss", float("inf")))
            history = payload.get("history", [])
            print(f"Resumed from checkpoint: {resume_checkpoint} at epoch {start_epoch}")
        except Exception as e:
            print(f"Warning: failed to load checkpoint {resume_checkpoint}: {e}")
            print("Ignoring corrupted checkpoint and restarting this run from scratch.")
            start_epoch = 0
            best_eval_loss = float("inf")
            history = []

    for epoch in range(start_epoch, train_config.epochs):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
            log_every=train_config.log_every,
            grad_clip_norm=train_config.grad_clip_norm,
        )
        eval_metrics = run_epoch(
            model=model,
            loader=eval_loader,
            optimizer=None,
            device=device,
            train=False,
            log_every=0,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_elbo"].append(train_metrics["elbo"])
        history["train_recon_log_prob"].append(train_metrics["recon_log_prob"])
        history["train_kl"].append(train_metrics["kl"])
        history["eval_loss"].append(eval_metrics["loss"])
        history["eval_elbo"].append(eval_metrics["elbo"])
        history["eval_recon_log_prob"].append(eval_metrics["recon_log_prob"])
        history["eval_kl"].append(eval_metrics["kl"])

        print(
            f"[epoch {epoch + 1:03d}/{train_config.epochs:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"eval_loss={eval_metrics['loss']:.4f} "
            f"eval_elbo={eval_metrics['elbo']:.4f}"
        )

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = out_dir / "checkpoint.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                    "model_config": asdict(model.config),
                    "train_config": asdict(train_config),
                },
                ckpt_path,
            )
            with open(out_dir / "training_history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    return history


def build_model(model_config: EnsembleVAEConfig, device: str) -> EnsembleVAE:
    model = EnsembleVAE(model_config)
    model.to(device)
    return model


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: str,
) -> EnsembleVAE:
    payload = torch.load(checkpoint_path, map_location=device)
    model_config = EnsembleVAEConfig(**payload["model_config"])
    model = EnsembleVAE(model_config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model
