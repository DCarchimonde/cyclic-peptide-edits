"""Regression tests for stereospecific reversible edits and the frozen sample IDs."""

import csv
import hashlib
import sys
import unittest
from pathlib import Path
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pair_chemistry import canonical, edits_for, eligible, read_csv


class PairDataTests(unittest.TestCase):
    def test_reverse_uses_exact_product_and_atom_mapping(self):
        row = read_csv(ROOT / "results/pairs/edits.csv")[0]
        parent = row["parent_smiles"]
        variant = row["variant_smiles"]
        edit = edits_for(variant)[0][parent]
        self.assertEqual(edit[3][edit[0]], edit[1])
        self.assertEqual(
            set(edit[3]) | {edit[2]},
            set(range(Chem.MolFromSmiles(variant).GetNumAtoms())),
        )
        randomized = Chem.MolToSmiles(Chem.MolFromSmiles(variant), doRandom=True)
        self.assertEqual(
            edits_for(canonical(Chem.MolFromSmiles(randomized)))[0][parent], edit
        )

    def test_stereochemical_inversion_is_not_same_parent(self):
        row = read_csv(ROOT / "results/pairs/edits.csv")[0]
        wrong = Chem.MolFromSmiles(row["parent_smiles"])
        next(
            a
            for a in wrong.GetAtoms()
            if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        ).InvertChirality()
        self.assertNotIn(canonical(wrong), edits_for(row["variant_smiles"])[0])

    def test_equivalent_sites_collapse(self):
        variant = canonical(Chem.MolFromSmiles("O=C1CN(C)C(=O)CN(C)C(=O)CN(C)1"))
        self.assertEqual(len(edits_for(variant)[0]), 1)

    def test_frozen_sample_matrix(self):
        p = ROOT / "results/pairs/edits.csv"
        self.assertEqual(
            hashlib.sha256(p.read_bytes()).hexdigest(),
            "c35bc14c0ae71ac2f73a2f4cc51a39d776e87b53546e19bea706647fdcd6f735",
        )
        splits = {}
        for r in read_csv(p):
            for role in ["variant_id", "parent_id", "backbone_id"]:
                self.assertEqual(splits.setdefault(r[role], r["split"]), r["split"])


if __name__ == "__main__":
    unittest.main()
