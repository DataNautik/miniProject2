"""
Experiment runner for DTU 02460 mini-project Part B: ensemble VAE geometry.

This runner is designed to:
- reuse the Part A dataset subset,
- train ensemble VAE models for K in {1,2,3},
- evaluate Euclidean vs geodesic distance stability on fixed point pairs,
- cache all intermediate artifacts for resume/reproducibility.

Notes on Part A integration:
- If your existing `partA_data.py` already exposes a stable dataloader helper,
  the adapter below should pick it up automatically.
- If it does not, add the tiny compatibility shim shown in the integration notes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import partA_data
from partB_evaluation import (
    aggregate_cov,
    evaluate_model_distances,
    load_or_create_pair_indices,
    save_cov_summary,
)
from partB_geometry import GeodesicConfig, optimize_ensemble_geodesic
from partB_model import EnsembleVAEConfig
from partB_train import TrainConfig, build_model, fit_ensemble_vae, load_model_from_checkpoint
from partB_visualization import (
    plot_cov_vs_num_decoders,
    plot_latent_scatter_with_curve_paths,
    save_plot_manifest,
)


def _call_with_supported_kwargs(fn, **kwargs):
    import inspect
    sig = inspect.signature(fn)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**supported)


def _extract_xy_from_dataset(dataset):
    xs = []
    ys = []
    for item in dataset:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            x, y = item, -1
        xs.append(torch.as_tensor(x).float())
        ys.append(int(y))
    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def _make_default_dataloaders(
    batch_size: int,
    classes=(0, 1, 2),
    num_observations: int = 2048,
    seed: int = 0,
):
    """
    Fallback path if the existing Part A module only exposes a raw dataset helper.
    """
    candidates = [
        "get_mnist_subset",
        "load_mnist_subset",
        "make_mnist_subset",
        "build_mnist_subset",
    ]
    for name in candidates:
        if hasattr(partA_data, name):
            subset = _call_with_supported_kwargs(
                getattr(partA_data, name),
                classes=classes,
                selected_classes=classes,
                digits=classes,
                num_observations=num_observations,
                n_observations=num_observations,
                subset_size=num_observations,
                seed=seed,
            )
            if isinstance(subset, dict) and "dataset" in subset:
                subset = subset["dataset"]
            x, y = _extract_xy_from_dataset(subset)
            dataset = TensorDataset(x, y)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            return {
                "train_loader": loader,
                "eval_loader": eval_loader,
                "latent_loader": eval_loader,
                "labels": y.numpy(),
                "num_points": len(dataset),
            }
    raise RuntimeError(
        "Could not infer a supported Part A data API. "
        "Use the compatibility shim shown in the integration notes."
    )


def load_partA_subset_dataloaders(
    batch_size: int,
    classes=(0, 1, 2),
    num_observations: int = 2048,
    seed: int = 0,
):
    """
    Tries several likely Part A APIs before falling back to a raw subset builder.
    """
    candidates = [
        "get_partA_dataloaders",
        "get_dataloaders",
        "make_dataloaders",
        "build_dataloaders",
        "load_dataloaders",
    ]
    for name in candidates:
        if hasattr(partA_data, name):
            result = _call_with_supported_kwargs(
                getattr(partA_data, name),
                batch_size=batch_size,
                classes=classes,
                selected_classes=classes,
                digits=classes,
                num_observations=num_observations,
                n_observations=num_observations,
                subset_size=num_observations,
                seed=seed,
                shuffle=True,
            )
            if isinstance(result, dict):
                train_loader = result.get("train_loader") or result.get("loader") or result.get("train")
                eval_loader = result.get("eval_loader") or result.get("test_loader") or result.get("eval") or train_loader
                latent_loader = result.get("latent_loader") or eval_loader
                labels = result.get("labels", None)
                if train_loader is None:
                    continue
                if labels is None:
                    try:
                        dataset = latent_loader.dataset
                        _, y = _extract_xy_from_dataset(dataset)
                        labels = y.numpy()
                    except Exception:
                        labels = None
                num_points = len(latent_loader.dataset)
                return {
                    "train_loader": train_loader,
                    "eval_loader": eval_loader,
                    "latent_loader": latent_loader,
                    "labels": labels,
                    "num_points": num_points,
                }
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                train_loader = result[0]
                eval_loader = result[1]
                latent_loader = result[2] if len(result) >= 3 else eval_loader
                labels = None
                try:
                    _, y = _extract_xy_from_dataset(latent_loader.dataset)
                    labels = y.numpy()
                except Exception:
                    pass
                return {
                    "train_loader": train_loader,
                    "eval_loader": eval_loader,
                    "latent_loader": latent_loader,
                    "labels": labels,
                    "num_points": len(latent_loader.dataset),
                }

    return _make_default_dataloaders(
        batch_size=batch_size,
        classes=classes,
        num_observations=num_observations,
        seed=seed,
    )


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_or_create_qualitative_latent_pairs(
    path: Path,
    latents: np.ndarray,
    num_pairs: int = 4,
    seed: int = 0,
) -> np.ndarray:
    """
    Loads latent-space endpoint pairs used for the qualitative geodesic figure.
    If a Part A file exists at `path`, it is reused exactly. Otherwise a fixed file
    is created once and then reused for all Part B models.
    """
    if path.exists():
        data = np.load(path)
        if "pairs" in data:
            return np.asarray(data["pairs"], dtype=np.float32)
        if "latent_pairs" in data:
            return np.asarray(data["latent_pairs"], dtype=np.float32)
        raise ValueError("Qualitative latent pair file must contain `pairs` or `latent_pairs`.")

    rng = np.random.default_rng(seed)
    mins = latents.min(axis=0)
    maxs = latents.max(axis=0)
    pairs = []
    for _ in range(num_pairs):
        a = rng.uniform(mins, maxs)
        b = rng.uniform(mins, maxs)
        pairs.append([a, b])
    pairs = np.asarray(pairs, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pairs=pairs)
    return pairs


def load_cached_geodesic_paths(exp_dir: Path, max_pairs: int) -> list[np.ndarray]:
    paths_dir = exp_dir / "geodesic_paths"
    paths = []
    for pair_id in range(max_pairs):
        path = paths_dir / f"pair_{pair_id:03d}_nodes.npy"
        if path.exists():
            paths.append(np.load(path))
    return paths


def run_single_experiment(
    out_dir: Path,
    num_decoders: int,
    seed: int,
    data_bundle: dict,
    model_config: EnsembleVAEConfig,
    train_config: TrainConfig,
    geodesic_config: GeodesicConfig,
    pair_indices: np.ndarray,
    resume: bool,
    force_eval: bool,
) -> dict:
    exp_dir = out_dir / f"k_{num_decoders}" / f"seed_{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = exp_dir / "checkpoint.pt"
    distance_npz = exp_dir / "distance_results.npz"

    if checkpoint_path.exists() and not resume:
        print(f"Using existing checkpoint: {checkpoint_path}")
        model = load_model_from_checkpoint(checkpoint_path, device=train_config.device)
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = build_model(model_config, device=train_config.device)
        fit_ensemble_vae(
            model=model,
            train_loader=data_bundle["train_loader"],
            eval_loader=data_bundle["eval_loader"],
            train_config=train_config,
            out_dir=exp_dir,
            resume_checkpoint=checkpoint_path if (resume and checkpoint_path.exists()) else None,
        )
        model.eval()

    # Validation: K=1 must behave as the single-decoder baseline structurally.
    if num_decoders == 1 and len(model.decoders) != 1:
        raise AssertionError("num_decoders=1 did not create a single-decoder baseline model.")

    # Evaluate distances unless already cached.
    if distance_npz.exists() and not force_eval:
        cached = np.load(distance_npz)
        euclidean = cached["euclidean"]
        geodesic = cached["geodesic"]
        latents = np.load(exp_dir / "latent_means.npy")
        node_paths = load_cached_geodesic_paths(exp_dir, max_pairs=len(pair_indices))
    else:
        results = evaluate_model_distances(
            model=model,
            latent_loader=data_bundle["latent_loader"],
            pairs=pair_indices,
            geodesic_config=geodesic_config,
            device=train_config.device,
            out_dir=exp_dir,
            save_paths=True,
        )
        latents = results["latents"]
        euclidean = results["euclidean"]
        geodesic = results["geodesic"]
        node_paths = results["node_paths"]

        # Validation: every geodesic optimization should reduce or preserve energy.
        for pair_id, hist in enumerate(results["energy_histories"]):
            if len(hist) >= 2 and hist[-1] > hist[0] + 1e-8:
                raise AssertionError(
                    f"Geodesic energy increased for pair {pair_id}: initial={hist[0]}, final={hist[-1]}"
                )

    return {
        "model": model,
        "latents": latents,
        "euclidean": euclidean,
        "geodesic": geodesic,
        "node_paths": node_paths,
        "exp_dir": exp_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-root", type=str, default="out_partB")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num-decoders-list", type=str, default="1,2,3")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-observations", type=int, default=2048)
    parser.add_argument(    "--selected-pairs-path",    type=str,    default="partA_outputs/selected_pairs.json",)
    parser.add_argument(    "--qualitative-pairs-path",    type=str,    default="partA_outputs/partA_qualitative_latent_pairs.npz",)
    parser.add_argument("--num-pairs", type=int, default=12)
    parser.add_argument("--pair-seed", type=int, default=0)
    parser.add_argument("--num-qualitative-pairs", type=int, default=4)
    parser.add_argument("--qualitative-pair-seed", type=int, default=0)
    parser.add_argument("--num-geodesic-nodes", type=int, default=16)
    parser.add_argument("--geodesic-steps", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    mode = args.mode
    out_root = Path(args.output_root)

    if mode == "smoke":
        num_decoders_list = [1, 2]
        seeds = [0, 1]
        epochs = args.epochs if args.epochs is not None else 2
        num_pairs = min(args.num_pairs, 4)
        geodesic_steps = min(args.geodesic_steps, 12)
    else:
        num_decoders_list = parse_int_list(args.num_decoders_list)
        seeds = parse_int_list(args.seeds)
        epochs = args.epochs if args.epochs is not None else 50
        num_pairs = args.num_pairs
        geodesic_steps = args.geodesic_steps

    data_bundle = load_partA_subset_dataloaders(
        batch_size=args.batch_size,
        classes=(0, 1, 2),
        num_observations=args.num_observations,
        seed=0,
    )

    pair_indices = load_or_create_pair_indices(
        path=Path(args.selected_pairs_path),
        num_points=data_bundle["num_points"],
        num_pairs=num_pairs,
        seed=args.pair_seed,
    )

    manifest = {
        "mode": mode,
        "num_decoders_list": num_decoders_list,
        "seeds": seeds,
        "batch_size": args.batch_size,
        "num_observations": args.num_observations,
        "selected_pairs_path": str(args.selected_pairs_path),
        "qualitative_pairs_path": str(args.qualitative_pairs_path),
        "num_pairs": int(len(pair_indices)),
        "device": args.device,
    }
    save_json(out_root / "run_manifest.json", manifest)

    cov_by_k = {}
    representative_visuals = {}

    for num_decoders in num_decoders_list:
        model_config = EnsembleVAEConfig(
            input_dim=28 * 28,
            latent_dim=2,
            hidden_dim=args.hidden_dim,
            num_decoders=num_decoders,
            beta=args.beta,
        )
        train_config = TrainConfig(
            epochs=epochs,
            lr=args.lr,
            device=args.device,
        )
        geodesic_config = GeodesicConfig(
            num_nodes=args.num_geodesic_nodes,
            max_steps=geodesic_steps,
        )

        print(f"=== Running K={num_decoders} ===")
        per_seed_results = {}
        representative = None

        for seed in seeds:
            print(f"--- seed={seed} ---")
            result = run_single_experiment(
                out_dir=out_root,
                num_decoders=num_decoders,
                seed=seed,
                data_bundle=data_bundle,
                model_config=model_config,
                train_config=train_config,
                geodesic_config=geodesic_config,
                pair_indices=pair_indices,
                resume=args.resume,
                force_eval=args.force_eval,
            )
            per_seed_results[seed] = {
                "euclidean": result["euclidean"],
                "geodesic": result["geodesic"],
            }
            if representative is None:
                representative = result

        summary = aggregate_cov(per_seed_results)
        save_cov_summary(summary, out_root / "summaries", num_decoders=num_decoders)
        cov_by_k[num_decoders] = {
            "euclidean_cov_global_mean": summary["euclidean_cov_global_mean"],
            "geodesic_cov_global_mean": summary["geodesic_cov_global_mean"],
        }

        if representative is not None:
            qual_pairs = load_or_create_qualitative_latent_pairs(
                path=Path(args.qualitative_pairs_path),
                latents=representative["latents"],
                num_pairs=args.num_qualitative_pairs,
                seed=args.qualitative_pair_seed,
            )

            curve_paths = []
            for pair in qual_pairs:
                z0 = torch.tensor(pair[0], dtype=torch.float32, device=args.device)
                z1 = torch.tensor(pair[1], dtype=torch.float32, device=args.device)
                geo = optimize_ensemble_geodesic(
                    model=representative["model"],
                    z_start=z0,
                    z_end=z1,
                    config=geodesic_config,
                    device=args.device,
                )
                curve_paths.append(geo["nodes"].detach().cpu().numpy())

            rep_path = out_root / "plots" / f"latent_scatter_geodesics_k{num_decoders}.png"
            plot_latent_scatter_with_curve_paths(
                latents=representative["latents"],
                labels=data_bundle["labels"],
                curve_paths=curve_paths,
                endpoints=qual_pairs,
                out_path=rep_path,
                title=f"Part B latent geometry, K={num_decoders}",
            )
            representative_visuals[f"k_{num_decoders}"] = str(rep_path)

    cov_plot_path = out_root / "plots" / "cov_vs_num_decoders.png"
    plot_cov_vs_num_decoders(cov_by_k=cov_by_k, out_path=cov_plot_path)
    save_plot_manifest(out_root / "plots", cov_plot=str(cov_plot_path), **representative_visuals)

    save_json(out_root / "cov_by_num_decoders.json", cov_by_k)
    print("Done.")


if __name__ == "__main__":
    main()
