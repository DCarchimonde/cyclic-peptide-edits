"""Apply the frozen B_12 site rankers to the five alternatives to CP1 in Hosono et al., Fig. 2."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from edit_metrics import TOL
from external_ly import audit_overlap
from fetch_edit_encoders import verify
from fit_edits import eligible as encoder_eligible, molecule_bank, predict_frozen_head
from pair_chemistry import ROOT, canonical, edits_for, eligible


HEADS = ("chem_site", "explicit_edit")
DOI = "10.1038/s41467-023-36978-z"


def atom_index(mol, pos, atom_name):
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info and info.GetResidueNumber() == pos and info.GetName().strip() == atom_name:
            return atom.GetIdx()
    raise ValueError(f"Missing {pos} {atom_name}")


def structures_and_edges():
    # Fig. 2b reads Tyr, Leu, D-Leu, Leu, Leu, D-Pro around a head-to-tail ring.
    ring = Chem.RWMol(Chem.MolFromSequence("YLLLLP"))
    ring.RemoveAtom(atom_index(ring, 6, "OXT"))
    ring.AddBond(atom_index(ring, 6, "C"), atom_index(ring, 1, "N"), Chem.BondType.SINGLE)
    for pos in (3, 6):
        ring.GetAtomWithIdx(atom_index(ring, pos, "CA")).InvertChirality()
    Chem.SanitizeMol(ring)
    Chem.AssignStereochemistry(ring, cleanIt=True, force=True)
    stereo = [ring.GetAtomWithIdx(atom_index(ring, pos, "CA")).GetProp("_CIPCode") for pos in range(1, 7)]
    if stereo != ["S", "S", "R", "S", "S", "R"]:
        raise ValueError(f"CP1 alpha stereochemistry differs from Fig. 2: {stereo}")
    parent = canonical(ring)
    graphs = {"CP1": parent}
    rows = []
    for pos in range(1, 6):
        name = f"MP{pos}"
        child = Chem.RWMol(ring)
        idx = atom_index(child, pos, "N")
        donor = child.GetAtomWithIdx(idx)
        if donor.GetTotalNumHs() != 1:
            raise ValueError(f"Expected one amide H at CP1 position {pos}")
        if donor.GetNumExplicitHs():
            donor.SetNumExplicitHs(0)
        carbon = child.AddAtom(Chem.Atom(6))
        child.AddBond(idx, carbon, Chem.BondType.SINGLE)
        Chem.SanitizeMol(child)
        smiles = canonical(child)
        if smiles in graphs.values() or eligible(Chem.MolFromSmiles(smiles)):
            raise ValueError(f"Invalid or duplicate site option {name}")
        mapping = edits_for(smiles)[0].get(parent)
        if mapping is None:
            raise ValueError(f"{name} does not reconstruct CP1 by removing one backbone N-methyl group")
        graphs[name] = smiles
        rows.append(dict(pair_id=name, group_id="CP1", parent_smiles=parent, variant_smiles=smiles,
                         parent_n_idx=mapping[0], variant_n_idx=mapping[1], site=pos))
    return graphs, pd.DataFrame(rows)


def b12_source(explicit):
    options = [Path(explicit)] if explicit else sorted((ROOT / "runs").glob("pairwise-rank_*"), reverse=True)
    for path in options:
        if (path / "summary.json").is_file():
            meta = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            if meta.get("folds") == 8 and meta.get("fitted") == 16:
                return path
    raise SystemExit("Cannot find the completed B_12 run. Retain B_PAMPA/runs or pass --b12-run with its full path.")


def panel_ranks(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise ValueError("Each frozen checkpoint must score all five sites")
    ahead = (values[None, :] - values[:, None]) > TOL
    tied = np.isclose(values[None, :], values[:, None], atol=TOL, rtol=0)
    return 1 + ahead.sum(axis=1) + (tied.sum(axis=1) - 1) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b12-run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    out = args.out
    out.mkdir(exist_ok=True, parents=True)
    b12 = b12_source(args.b12_run)
    roles = pd.read_csv(ROOT / "results/model/series_sensitivity/folds.csv")
    training = pd.read_csv(b12 / "training.csv").set_index(["head", "outer"], verify_integrity=True)
    outer_ids = sorted(roles.outer.unique())
    if set(training.index) != {(h, s) for h in HEADS for s in outer_ids} or len(outer_ids) != 8:
        raise ValueError("B_12 checkpoints do not match the eight archived folds")

    graphs, frame = structures_and_edges()
    frame.to_csv(out / "site_inputs.csv", index=False)
    linked, seen = audit_overlap(graphs, set(graphs.values()), out)
    registry = pd.read_csv(ROOT / "results/pairs/source_counts.csv")
    related = set(registry.loc[registry.source.isin(linked), "series_id"])
    if set(linked) - set(registry.source):
        raise ValueError("Unrecognized source in the frozen overlap audit")
    weights = ROOT / "cache/weights"
    verify(weights, fetch=True)
    torch.set_num_threads(args.threads)
    bank = molecule_bank(frame, weights, out / "features", args.threads)
    full, excluded = encoder_eligible(frame, bank)
    if len(full) != 5 or not excluded.empty:
        excluded.to_csv(out / "encoder_exclusions.csv", index=False)
        raise ValueError("A planned site is missing from the encoded panel")
    rows, hashes = [], {}
    for head in HEADS:
        for outer in outer_ids:
            path = b12 / "fitted" / outer / (head + ".pt")
            with path.open("rb") as stream:
                hashes[f"{outer}/{head}"] = hashlib.file_digest(stream, "sha256").hexdigest()
            scores = predict_frozen_head(full, bank, path)
            pool = roles.loc[roles.outer.eq(outer) & roles.role.isin(["train", "valid"]), "series_id"]
            for site, score, rank in zip(full.pair_id, scores, panel_ranks(scores), strict=True):
                rows.append(dict(head=head, outer=outer, compound=site, score=float(score), rank=float(rank),
                                 no_linked_fit_series=not related.intersection(pool)))
    by_fold = pd.DataFrame(rows)
    by_fold.to_csv(out / "frozen_site_scores.csv", index=False)
    scopes = {"all_eight": by_fold}
    clean = by_fold[by_fold.no_linked_fit_series]
    if linked and not clean.empty:
        scopes["linked_series_held_out"] = clean
    choices = []
    for scope, forecasts in scopes.items():
        for head, group in forecasts.groupby("head"):
            ranks = group.groupby("compound")["rank"].mean()
            chosen = ranks.index[np.isclose(ranks, ranks.min(), atol=TOL, rtol=0)].tolist()
            choices.append(dict(scope=scope, head=head, folds=group.outer.nunique(), selected="|".join(chosen),
                                top_ties=len(chosen), selected_options=chosen))
    # Only the top observed PAMPA mean is read from the published bars; raw values are not imputed.
    outcome = json.loads((ROOT / "configs/hosono_2023_top1.json").read_text(encoding="utf-8"))
    winner = outcome["highest_mean_PAMPA"]
    if outcome["doi"] != DOI or winner not in set(frame.pair_id):
        raise ValueError("Figure outcome does not correspond to this panel")
    for row in choices:
        selected = row.pop("selected_options")
        row.update(observed_top=winner, top1_credit=float(winner in selected)/len(selected),
                   random_top1=1/len(frame))
    pd.DataFrame(choices).to_csv(out / "panel_choices.csv", index=False)
    status = "linked_case" if linked or seen else "no_known_frozen_overlap"
    summary = dict(doi=DOI, figure="2a-c", parent="CP1", molecular_options=len(frame), observed_top=winner,
                   inference_source=b12.name, checkpoint_sha256=hashes, linked_frozen_sources=linked,
                   snapshot_matches=seen, source_status=status, no_linked_fit_folds=clean.outer.nunique(),
                   counts_as_published_parent_panels=1, endpoint="highest plotted mean PAMPA only")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"B_HOSONO_PANEL_OK sites={len(frame)} fitted_heads_read={len(hashes)} "
          f"linked_series={len(related)} snapshot_matches={len(seen)} status={status}")


if __name__ == "__main__":
    main()
