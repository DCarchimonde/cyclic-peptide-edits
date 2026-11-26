"""Run an exploratory PAMPA baseline with an explicit source holdout.

This is a first data/model integration experiment, not the final benchmark.
Reported performance belongs to a chemical baseline, not the fusion network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


def molecular_features(smiles: str, generator):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    fingerprint = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), fingerprint)
    values = np.asarray([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol), Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol), Lipinski.NumRotatableBonds(mol),
        mol.GetNumHeavyAtoms(), rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol), Chem.GetFormalCharge(mol),
        rdMolDescriptors.CalcNumAmideBonds(mol),
        len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
    ], dtype=np.float32)
    if not np.isfinite(values).all():
        return None
    return canonical, np.concatenate([fingerprint, values])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()

    original = pd.read_csv(args.data, dtype={"molecule_id": str, "smiles": str})
    required = {"molecule_id", "smiles", "target", "source", "endpoint", "label_relation"}
    missing = required - set(original.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    data = original.copy()
    data["target"] = pd.to_numeric(data["target"], errors="coerce")
    finite = np.isfinite(data["target"])
    exact = data["label_relation"].fillna("").astype(str).str.lower().isin(["=", "eq", "exact"])
    pampa = data["endpoint"].fillna("").str.upper().isin(["PAMPA", "PAMPA_LOG10_CM_PER_S"])
    selected = data.loc[finite & exact & pampa].copy()
    if selected["source"].isna().any():
        raise ValueError("Source is required for the holdout experiment")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    rows, features, failures = [], [], []
    for row in selected.to_dict("records"):
        result = molecular_features(str(row["smiles"]), generator)
        if result is None:
            failures.append({"molecule_id": row["molecule_id"], "reason": "unparseable_or_nonfinite_features"})
            continue
        canonical, vector = result
        row["canonical_smiles"] = canonical
        rows.append(row)
        features.append(vector)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No usable quantified PAMPA records")
    x = np.stack(features)

    # Keep one deterministic observation per exact structure within a source.
    # Conflicting values are removed rather than silently averaged across assays.
    key = ["source", "canonical_smiles"]
    distinct_values = frame.groupby(key)["target"].transform("nunique")
    conflicts = distinct_values.gt(1)
    duplicate = frame.duplicated(key, keep="first")
    keep = ~(conflicts | duplicate)
    counts = {"conflicting_structure_source_rows_removed": int(conflicts.sum()),
              "same_structure_source_duplicate_rows": int((duplicate & ~conflicts).sum())}
    frame, x = frame.loc[keep].reset_index(drop=True), x[keep.to_numpy()]
    if frame["source"].nunique() < 2:
        raise ValueError("At least two independent source groups are needed")

    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, test_idx = next(split.split(x, groups=frame["source"]))
    train_smiles = set(frame.iloc[train_idx]["canonical_smiles"])
    overlap = frame.iloc[test_idx]["canonical_smiles"].isin(train_smiles).to_numpy()
    removed_overlap = int(overlap.sum())
    test_idx = test_idx[~overlap]
    if len(test_idx) < 2:
        raise ValueError("Holdout has fewer than two unique unseen molecules")

    y = frame["target"].to_numpy(dtype=float)
    estimator = ExtraTreesRegressor(
        n_estimators=args.trees, max_depth=14, min_samples_leaf=2,
        random_state=args.seed, n_jobs=args.jobs,
    )
    estimator.fit(x[train_idx], y[train_idx])
    prediction = estimator.predict(x[test_idx])
    mean_prediction = np.repeat(float(y[train_idx].mean()), len(test_idx))
    rank = float(spearmanr(y[test_idx], prediction).statistic)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "status": "exploratory_real_data_baseline_complete",
        "model": "Morgan2048_chiral_plus_12_descriptors_ExtraTrees",
        "endpoint": "PAMPA", "target": "source_provided_log_permeability",
        "input_rows": len(original), "quantified_pampa_rows": len(selected),
        "chemical_parse_failures": len(failures), **counts,
        "test_overlap_rows_removed": removed_overlap,
        "train_records": len(train_idx), "test_records": len(test_idx),
        "train_sources": sorted(frame.iloc[train_idx]["source"].unique().tolist()),
        "test_sources": sorted(frame.iloc[test_idx]["source"].unique().tolist()),
        "mae": float(mean_absolute_error(y[test_idx], prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y[test_idx], prediction))),
        "spearman": rank if np.isfinite(rank) else None,
        "training_mean_baseline_mae": float(mean_absolute_error(y[test_idx], mean_prediction)),
        "split": "one_source_group_holdout_with_exact_molecule_overlap_removed",
        "seed": args.seed, "trees": args.trees,
        "rdkit_version": rdkit.__version__, "python_version": platform.python_version(),
        "input_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "interpretation": [
            "One exploratory source split, not cross-validation or a final benchmark.",
            "Only quantified PAMPA rows are regressed; censored observations are excluded here.",
            "Source grouping controls this split; full cyclic-scaffold family grouping is not implemented.",
            "The fusion network and pretrained encoders were not used in this baseline.",
        ],
    }
    assignments = frame.copy()
    assignments["split"] = "excluded_overlap"
    assignments.loc[train_idx, "split"] = "train"
    assignments.loc[test_idx, "split"] = "test"
    assignments.to_csv(args.output / "split_assignments.csv", index=False)
    test_output = frame.iloc[test_idx].copy()
    test_output["prediction"] = prediction
    test_output.to_csv(args.output / "predictions.csv", index=False)
    pd.DataFrame(failures, columns=["molecule_id", "reason"]).to_csv(args.output / "parse_failures.csv", index=False)
    joblib.dump(estimator, args.output / "baseline.joblib", compress=3)
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

