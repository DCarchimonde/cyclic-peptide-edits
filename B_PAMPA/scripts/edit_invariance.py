"""Check end-to-end edit scoring from every possible written head-to-tail ring origin."""

import argparse, ast, json
from pathlib import Path
import torch, pandas as pd, numpy as np
from rdkit import Chem
from fit_edits import normalize_prediction_pairs, pair_features, scaling, make_head
from encoder_cache import read_bank
from encoder_graph import rotation_indices, protein_residue_reindex


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--edits", type=Path, default=Path("results/pairs/edits.csv"))
    a.add_argument("--model-dir", type=Path, default=Path("results/model"))
    a.add_argument("--cache", type=Path, default=Path(".cache/encoders"))
    o = a.parse_args()
    e = pd.read_csv(o.edits)
    r = e[(e.topology == "Circle") & e.delta.notna()].sort_values("pair_id").iloc[0]
    bank = {s: read_bank(o.cache, s) for s in [r.parent_smiles, r.variant_smiles]}
    torch.set_num_threads(1)
    w = torch.load(
        o.model_dir / "fitted/explicit_edit.pt", map_location="cpu", weights_only=True
    )
    net = make_head(w["weight_dims"])
    net.load_state_dict(w["state"])
    net.eval()
    ns = bank[r.parent_smiles]["ns"]
    effects = []
    for k in range(len(ns)):
        row = r.copy()
        for name in ["parent", "variant"]:
            s = r[name + "_smiles"]
            idx = r[name + "_n_idx"]
            m = Chem.MolFromSmiles(s)
            n = int(bank[s]["ns"][k % len(bank[s]["ns"])])
            s2 = Chem.MolToSmiles(
                m, canonical=False, rootedAtAtom=n, isomericSmiles=True
            )
            ix = ast.literal_eval(m.GetProp("_smilesAtomOutputOrder"))
            row[name + "_smiles"] = s2
            row[name + "_n_idx"] = ix.index(int(idx))
        fixed = normalize_prediction_pairs(pd.DataFrame([row])).iloc[0]
        x = pair_features(fixed, bank, "explicit_edit")[None, :]
        with torch.inference_mode():
            effects.append(
                float(
                    net(
                        scaling(x, {k: v.numpy() for k, v in w["scale"].items()})
                    ).item()
                )
            )
    x = bank[r.parent_smiles]
    seq = str(x["source_order_seq"])
    c, starts = rotation_indices(seq, True)
    h = np.roll(x["prot"], -starts[0], axis=0)
    err = 0.0
    for k in range(len(seq)):
        seq2, j = rotation_indices(seq[k:] + seq[:k], True)
        assert c == seq2
        err = max(
            err,
            float(
                np.abs(
                    np.roll(protein_residue_reindex(h, j), k, axis=0) - x["prot"]
                ).max()
            ),
        )
    out = dict(
        pair_id=r.pair_id,
        verified_closed_peptide=bool(x["cyclic"]),
        actual_parent_y=r.parent_y,
        actual_variant_y=r.variant_y,
        parent_input_root_n_indices=ns.tolist(),
        fitted_score_per_root=effects,
        max_abs_score_change=float(np.ptp(effects)),
        max_rotated_aligned_esm_absolute_error=err,
    )
    assert np.ptp(effects) < 1e-6 and err < 1e-6 and out["verified_closed_peptide"]
    (o.model_dir / "rotation_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
