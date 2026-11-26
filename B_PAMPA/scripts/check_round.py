"""Recompute the frozen test decisions and verify the identities used by every method."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TIE = 1e-7
EDIT_HASH = "c35bc14c0ae71ac2f73a2f4cc51a39d776e87b53546e19bea706647fdcd6f735"


def check_molecule_splits(edits):
    endpoints = pd.concat(
        [
            edits[["split", column]].rename(columns={column: "molecule"})
            for column in ("parent_id", "variant_id")
        ]
    ).drop_duplicates()
    if endpoints["molecule"].duplicated().any():
        raise ValueError("A molecule appears in more than one split")
    for key in ("backbone_id", "group_id", "series_id"):
        if edits.groupby(key)["split"].nunique().max() != 1:
            raise ValueError(f"{key} spans multiple splits")


def check_atoms(edits):
    from rdkit import Chem

    for row in edits.itertuples(index=False):
        parent = Chem.MolFromSmiles(row.parent_smiles)
        variant = Chem.MolFromSmiles(row.variant_smiles)
        atom_map = json.loads(row.parent_to_variant_atom_map)
        nitrogen = variant.GetAtomWithIdx(row.variant_n_idx)
        methyl = variant.GetAtomWithIdx(row.variant_methyl_idx)
        bonds = [a.GetIdx() for a in methyl.GetNeighbors()]
        if sorted(atom_map + [methyl.GetIdx()]) != list(range(variant.GetNumAtoms())):
            raise ValueError(f"Atom map does not cover {row.pair_id}")
        if (
            len(atom_map) != parent.GetNumAtoms()
            or atom_map[row.parent_n_idx] != nitrogen.GetIdx()
            or nitrogen.GetSymbol() != "N"
            or methyl.GetSymbol() != "C"
            or methyl.GetTotalNumHs() != 3
            or bonds != [nitrogen.GetIdx()]
        ):
            raise ValueError(f"Invalid N–methyl alignment for {row.pair_id}")
        product = Chem.RWMol(variant)
        product.RemoveAtom(methyl.GetIdx())
        Chem.SanitizeMol(product)
        if Chem.MolToSmiles(product) != Chem.MolToSmiles(parent):
            raise ValueError(f"Demethylation failed to reconstruct {row.parent_id}")


def score(rows):
    # The mask includes every tied first choice. Each observed site is then equally likely.
    ranked = rows.groupby("group_id").filter(lambda x: len(x) > 1).copy()
    by_group = ranked.groupby("group_id")
    predicted_best = (
        ranked.pred_delta - by_group.pred_delta.transform("max")
    ).abs() <= TIE
    ranked["measured_best"] = by_group.variant_y.transform("max")
    selected = ranked.loc[predicted_best].copy()
    selected["loss"] = selected.measured_best - selected.variant_y
    selected["hit"] = selected.loss.abs() <= TIE
    groups = selected.groupby(["parent_id", "group_id"])[["loss", "hit"]].mean()
    parents = groups.groupby(level="parent_id").mean()
    paired = rows.dropna(subset=["delta"])
    return dict(
        rank_parents=len(parents),
        deltas=len(paired),
        regret=float(parents.loss.mean()),
        top1=float(parents.hit.mean()),
        delta_mae=float((paired.pred_delta - paired.delta).abs().mean()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-atoms",
        action="store_true",
        help="Check the molecular graph for all 4,322 edits (requires RDKit)",
    )
    parser.add_argument(
        "--out", type=Path, help="Write JSON containing the verified metrics"
    )
    args = parser.parse_args()
    pair_path = ROOT / "results/pairs/edits.csv"
    if sha256(pair_path.read_bytes()).hexdigest() != EDIT_HASH:
        raise ValueError("Frozen edit table checksum changed")
    edits = pd.read_csv(pair_path)
    check_molecule_splits(edits)
    if args.check_atoms:
        check_atoms(edits)
    targets = edits.loc[edits.split.eq("test")].set_index(
        "pair_id", verify_integrity=True
    )
    reference_dir = ROOT / "results/references"
    sources = [ROOT / "results/model/predictions.csv"] + [
        reference_dir / filename
        for filename in (
            "cycdelta_refit_predictions.csv",
            "cycdelta_released_predictions.csv",
            "ncaa_predictions.csv",
            "dmpnn_predictions.csv",
        )
    ]
    forecasts = pd.concat(
        [pd.read_csv(path)[["pair_id", "model", "pred_delta"]] for path in sources]
    )
    matched = forecasts.loc[forecasts.pair_id.isin(targets.index)]
    comparison = pd.read_csv(reference_dir / "comparison.csv")
    reported = comparison.loc[comparison.cohort.eq("full_common_test")].set_index(
        "model", verify_integrity=True
    )
    summaries = {}
    for model, pred in matched.groupby("model"):
        pred = pred.set_index("pair_id", verify_integrity=True)
        if set(pred.index) != set(targets.index):
            raise ValueError(f"Incomplete test coverage for {model}")
        joined = targets.join(pred.pred_delta)
        result = score(joined)
        for name, value in result.items():
            np.testing.assert_allclose(
                value, reported.loc[model, name], rtol=0, atol=1e-7
            )
        summaries[model] = result
    if set(summaries) != set(reported.index):
        raise ValueError("Reported methods and prediction files differ")
    output = dict(
        edit_table_sha256=EDIT_HASH,
        molecular_edits_checked=len(edits) if args.check_atoms else None,
        shared_test_options=len(targets),
        models=summaries,
    )
    contents = json.dumps(output, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(contents + "\n", encoding="utf-8")
    print(contents)


if __name__ == "__main__":
    main()
