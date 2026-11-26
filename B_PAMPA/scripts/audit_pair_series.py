"""Audit graph, backbone and high-similarity leakage across the frozen chemical series."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from pair_chemistry import ROOT, graph_index, read_csv, write_csv

FP = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, includeChirality=True
)


def neighbors(fa, fb):
    """Find maximum exact Tanimoto by comparing every query to the smaller target set."""
    if len(fa) > len(fb):
        fa, fb = fb, fa
    return max(
        (max(DataStructs.BulkTanimotoSimilarity(a, fb)) for a in fa), default=0.0
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "results/pairs")
    p.add_argument("--index", type=Path, default=ROOT / ".cache/pair_graph_index.json")
    a = p.parse_args()
    index = graph_index(ROOT / "data/raw/cycpeptmpdb_all.csv", a.index)
    series = defaultdict(lambda: {"molecules": set(), "backbones": set()})
    folds = {}
    for r in read_csv(a.output / "edits.csv"):
        s = r["source"]
        folds[s] = r["split"]
        series[s]["molecules"].update([r["parent_smiles"], r["variant_smiles"]])
        series[s]["backbones"].add(index["backbones"][r["variant_smiles"]])
    cache = {
        m: FP.GetFingerprint(Chem.MolFromSmiles(m))
        for ss in series.values()
        for mm in ss.values()
        for m in mm
    }
    pairs = []
    worst = 0.0
    cutoff = 0.80
    for left, right in combinations(sorted(series), 2):
        row = dict(
            left=left, right=right, left_split=folds[left], right_split=folds[right]
        )
        for kind in ["molecules", "backbones"]:
            a1, b1 = series[left][kind], series[right][kind]
            row["shared_" + kind] = len(a1 & b1)
            row["max_similarity_" + kind] = neighbors(
                [cache[k] for k in sorted(a1)], [cache[k] for k in sorted(b1)]
            )
            if folds[left] != folds[right]:
                worst = max(worst, row["max_similarity_" + kind])
            if row["shared_" + kind] or row["max_similarity_" + kind] >= cutoff:
                raise ValueError(
                    f"These source series must be linked before splitting: {left}, {right}"
                )
        pairs.append(row)
    write_csv(a.output / "cross_family_similarity.csv", pairs)
    out = dict(
        similarity="chiral Morgan radius 2, 2048 bits, Tanimoto; exhaustive",
        merge_threshold=cutoff,
        independent_series=len(series),
        max_across_split=worst,
        shared_molecules_across_splits=0,
        shared_deNMe_backbones_across_splits=0,
        aspects="Both parents, all variants, and fully de-NMe backbones; frames inherit one graph split.",
    )
    (a.output / "split_audit.json").write_text(json.dumps(out, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
