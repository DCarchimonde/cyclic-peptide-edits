"""Train the fusion prototype from real, provenance-recorded feature caches.

Feature extraction is a separate step. This script never synthesizes missing
protein, chemical, edit, or geometry embeddings.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cyclic_peptide_ai.model import EditAwareFusion, FusionConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True,
                        help="CSV with molecule_id,target,split; train/test preassigned")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/fusion"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata.get("features_are_real") is not True:
        raise ValueError("Training requires real feature caches and their encoder provenance")
    for field in ("encoder_names", "checkpoint_revisions", "molecule_mapping_method", "edit_feature_definition"):
        if not metadata.get(field):
            raise ValueError(f"Feature provenance is missing {field}")
    table = pd.read_csv(args.split, dtype={"molecule_id": str})
    table = table.loc[table["split"].isin(["train", "test"])].copy()
    if table.groupby("molecule_id")["split"].nunique().gt(1).any():
        raise ValueError("A molecule_id occurs in both training and test sets")
    with np.load(args.features, allow_pickle=False) as cache:
        ids = cache["molecule_id"].astype(str).tolist()
        if len(ids) != len(set(ids)):
            raise ValueError("Feature cache molecule IDs must be unique")
        lookup = {identifier: i for i, identifier in enumerate(ids)}
        absent = [identifier for identifier in table["molecule_id"] if identifier not in lookup]
        if absent:
            raise ValueError(f"Feature cache missing {len(absent)} selected molecules; define a shared subset before comparison")
        positions = np.asarray([lookup[i] for i in table["molecule_id"]])
        names = ["protein", "chemical", "edit"]
        if "geometry" in cache.files:
            names.append("geometry")
        arrays = {}
        for name in names:
            raw = cache[name]
            if raw.ndim != 2 or raw.shape[0] != len(ids):
                raise ValueError(f"{name} is not aligned to molecule_id")
            arrays[name] = raw[positions].astype(np.float32)
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"{name} contains nonfinite features")

    train_mask = table["split"].eq("train").to_numpy()
    test_mask = table["split"].eq("test").to_numpy()
    if train_mask.sum() < 2 or test_mask.sum() < 2:
        raise ValueError("Both train and test sets need at least two selected molecules")
    y = pd.to_numeric(table["target"], errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite target")
    mean, scale = float(y[train_mask].mean()), float(y[train_mask].std())
    if scale == 0:
        raise ValueError("Training targets have zero variance")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = FusionConfig(
        protein_dim=arrays["protein"].shape[1], chemical_dim=arrays["chemical"].shape[1],
        edit_dim=arrays["edit"].shape[1],
        geometry_dim=arrays["geometry"].shape[1] if "geometry" in arrays else 0,
        hidden_dim=args.hidden,
    )
    device = torch.device(args.device)
    model = EditAwareFusion(config).to(device)
    train_tensors = [torch.from_numpy(arrays[name][train_mask]) for name in names]
    train_tensors.append(torch.from_numpy((y[train_mask] - mean) / scale))
    loader = DataLoader(TensorDataset(*train_tensors), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_function = nn.MSELoss()
    history = []
    # Epoch count is fixed in advance; the held-out test set is not used to
    # select checkpoints or tune hyperparameters.
    for epoch in range(args.epochs):
        model.train()
        squared_sum, count = 0.0, 0
        for batch in loader:
            features = {name: value.to(device) for name, value in zip(names, batch[:-1])}
            target = batch[-1].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(**features)["permeability"], target)
            loss.backward()
            optimizer.step()
            squared_sum += float(loss.detach()) * len(target)
            count += len(target)
        history.append({"epoch": epoch + 1, "training_mse_standardized": squared_sum / count})

    model.eval()
    outputs = []
    test_positions = np.flatnonzero(test_mask)
    with torch.no_grad():
        for offset in range(0, len(test_positions), args.batch_size):
            selected = test_positions[offset:offset + args.batch_size]
            features = {name: torch.from_numpy(arrays[name][selected]).to(device) for name in names}
            outputs.append(model(**features)["permeability"].cpu().numpy())
    prediction = np.concatenate(outputs) * scale + mean
    rank = float(spearmanr(y[test_mask], prediction).statistic)
    metrics = {
        "status": "trained_on_supplied_real_feature_cache",
        "train_records": int(train_mask.sum()), "test_records": int(test_mask.sum()),
        "mae": float(mean_absolute_error(y[test_mask], prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y[test_mask], prediction))),
        "spearman": rank if np.isfinite(rank) else None,
        "epochs": args.epochs, "seed": args.seed, "config": asdict(config),
        "feature_metadata": metadata,
        "checkpoint_selection": "fixed_epoch_count_no_test_set_selection",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(config),
                "target_mean": mean, "target_scale": scale, "feature_metadata": metadata},
               args.output / "model.pt")
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    prediction_table = table.loc[test_mask, ["molecule_id", "target", "split"]].copy()
    prediction_table["prediction"] = prediction
    prediction_table.to_csv(args.output / "predictions.csv", index=False)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

