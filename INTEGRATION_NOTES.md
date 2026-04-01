# Part B integration notes

These files are written to match the stated Part A architecture and keep Part A untouched unless your
existing `partA_data.py` does not expose a stable dataloader helper.

## Mandatory existing dependency
`run_partB.py` imports `partA_data.py` and tries these APIs in order:

- `get_partA_dataloaders(...)`
- `get_dataloaders(...)`
- `make_dataloaders(...)`
- `build_dataloaders(...)`
- `load_dataloaders(...)`

Accepted return shapes:
- dict with `train_loader`, `eval_loader`, optional `latent_loader`, optional `labels`
- tuple/list of `(train_loader, eval_loader[, latent_loader])`

It also has a fallback path for a raw subset helper:
- `get_mnist_subset(...)`
- `load_mnist_subset(...)`
- `make_mnist_subset(...)`
- `build_mnist_subset(...)`

## Minimal edit only if your Part A loader API is missing
Add the following adapter to `partA_data.py`:

```python
def get_partA_dataloaders(batch_size=128, classes=(0, 1, 2), num_observations=2048, seed=0, **kwargs):
    subset = get_mnist_subset(classes=classes, num_observations=num_observations, seed=seed)
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    return {
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "latent_loader": eval_loader,
    }
```

Replace `get_mnist_subset(...)` with your actual Part A subset builder if the name differs.

## Optional strict Part A qualitative-pair reuse
The assignment says to reuse the same random latent pairs as Part A for the qualitative figure.
`run_partB.py` looks for:

- `partA_qualitative_latent_pairs.npz`

If that file exists, it is reused exactly.
If it does not, Part B creates it once and then reuses it for all later runs.

Expected format:
- key `pairs` or `latent_pairs`
- array shape `[num_pairs, 2, 2]`
