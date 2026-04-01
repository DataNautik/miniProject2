"""Part A data utilities for Mini-project 2.

This module prepares the MNIST subset used in Part A:
- only 3 classes are kept,
- a total of 2048 observations are selected,
- data is loaded as continuous pixels (not binarized),
- a tiny amount of noise can optionally be added later in the training loop.

The code is intentionally simple and reproducible.
"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PARTA_CLASSES: List[int] = [0, 1, 2]
PARTA_TOTAL_OBSERVATIONS: int = 2048
PARTA_BATCH_SIZE: int = 128
PARTA_TRAIN_RATIO: float = 0.8
PARTA_NUM_WORKERS: int = 0
PARTA_PIN_MEMORY: bool = False


# -----------------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------------

@dataclass
class PartADataBundle:
    """Container holding the datasets and loaders used in Part A."""

    train_dataset: Subset
    val_dataset: Subset
    full_subset: Subset
    train_loader: DataLoader
    val_loader: DataLoader
    full_loader: DataLoader


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------


def get_mnist_transform() -> transforms.Compose:
    """Return the transform used for Part A.

    We keep MNIST continuous in [0, 1] and flatten each image to shape (784,).
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )



def _filter_indices_by_class_and_limit(
    dataset: datasets.MNIST,
    classes: List[int],
    total_observations: int,
) -> List[int]:
    """Return a balanced list of indices for the requested classes.

    We try to distribute the 2048 observations as evenly as possible across the
    chosen classes. Any remainder is assigned to the first classes in the list.
    """
    if total_observations < len(classes):
        raise ValueError("total_observations must be at least the number of classes")

    targets = dataset.targets
    per_class = total_observations // len(classes)
    remainder = total_observations % len(classes)

    selected_indices: List[int] = []

    for class_offset, class_id in enumerate(classes):
        n_keep = per_class + (1 if class_offset < remainder else 0)
        class_indices = torch.where(targets == class_id)[0].tolist()

        if len(class_indices) < n_keep:
            raise ValueError(
                f"Class {class_id} only has {len(class_indices)} samples, "
                f"but {n_keep} were requested."
            )

        selected_indices.extend(class_indices[:n_keep])

    return selected_indices



def build_parta_subset(
    root: str = "data/",
    classes: List[int] | None = None,
    total_observations: int = PARTA_TOTAL_OBSERVATIONS,
    train: bool = True,
) -> Subset:
    """Build the 3-class, 2048-observation MNIST subset used in Part A."""
    if classes is None:
        classes = PARTA_CLASSES

    dataset = datasets.MNIST(
        root=root,
        train=train,
        download=True,
        transform=get_mnist_transform(),
    )

    selected_indices = _filter_indices_by_class_and_limit(
        dataset=dataset,
        classes=classes,
        total_observations=total_observations,
    )

    return Subset(dataset, selected_indices)



def split_parta_subset(
    subset: Subset,
    train_ratio: float = PARTA_TRAIN_RATIO,
    seed: int = 0,
) -> Tuple[Subset, Subset]:
    """Split the subset into train/validation subsets reproducibly."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    n_total = len(subset)
    n_train = int(train_ratio * n_total)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(subset, [n_train, n_val], generator=generator)
    return train_subset, val_subset



def build_dataloader(
    dataset: Subset,
    batch_size: int = PARTA_BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = PARTA_NUM_WORKERS,
    pin_memory: bool = PARTA_PIN_MEMORY,
) -> DataLoader:
    """Create a DataLoader with the project defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def get_parta_data(
    root: str = "data/",
    classes: List[int] | None = None,
    total_observations: int = PARTA_TOTAL_OBSERVATIONS,
    batch_size: int = PARTA_BATCH_SIZE,
    train_ratio: float = PARTA_TRAIN_RATIO,
    seed: int = 0,
) -> PartADataBundle:
    """Prepare datasets and dataloaders for Part A.

    Returns:
        PartADataBundle with:
        - full 2048-sample subset,
        - train split,
        - validation split,
        - corresponding dataloaders.
    """
    full_subset = build_parta_subset(
        root=root,
        classes=classes,
        total_observations=total_observations,
        train=True,
    )

    train_dataset, val_dataset = split_parta_subset(
        subset=full_subset,
        train_ratio=train_ratio,
        seed=seed,
    )

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    full_loader = build_dataloader(
        dataset=full_subset,
        batch_size=batch_size,
        shuffle=False,
    )

    return PartADataBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        full_subset=full_subset,
        train_loader=train_loader,
        val_loader=val_loader,
        full_loader=full_loader,
    )


# -----------------------------------------------------------------------------
# Quick manual check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    bundle = get_parta_data(seed=0)

    print("Part A data summary")
    print(f"Classes: {PARTA_CLASSES}")
    print(f"Total subset size: {len(bundle.full_subset)}")
    print(f"Train size: {len(bundle.train_dataset)}")
    print(f"Validation size: {len(bundle.val_dataset)}")

    x_batch, y_batch = next(iter(bundle.train_loader))
    print(f"Batch x shape: {tuple(x_batch.shape)}")
    print(f"Batch y shape: {tuple(y_batch.shape)}")
    print(f"Unique labels in first batch: {sorted(torch.unique(y_batch).tolist())}")
