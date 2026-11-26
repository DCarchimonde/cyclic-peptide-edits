"""Evaluate symmetric interchain PAE, upstream score, and two different fits."""

import argparse
import json
from pathlib import Path
import re
import numpy as np


def coordinates(path):
    result = {}
    for l in path.read_text().splitlines():
        if l.startswith("ATOM  ") and l[16] in (" ", "A"):
            result.setdefault(l[21], {}).setdefault((int(l[22:26]), l[26]), {})[
                l[12:16].strip()
            ] = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    return result


def atoms(chain, names):
    return np.array([v[n] for v in chain.values() for n in names])


def fit(mobile, fixed):
    u, _, v = np.linalg.svd((mobile - mobile.mean(0)).T @ (fixed - fixed.mean(0)))
    sign = np.sign(np.linalg.det(u @ v))
    rot = u @ np.diag([1.0, 1.0, sign]) @ v
    return rot, fixed.mean(0) - mobile.mean(0) @ rot


def rmsd(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def terminal(chain):
    rs = list(chain.values())
    return float(np.linalg.norm(rs[0]["N"] - rs[-1]["C"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("fold", type=Path)
    p.add_argument(
        "--known-receptor",
        type=Path,
        help="A fixed-sequence refold without the author design backbone.",
    )
    a = p.parse_args()
    f = coordinates(next(a.fold.glob("*_unrelaxed_rank_*.pdb")))
    assert set(f) == {"A", "B"} and len(f["B"]) == 118
    n = len(f["A"])
    out = json.loads(next(a.fold.glob("*_scores_rank_*.json")).read_text())
    plddt = np.array(out["plddt"])
    pae = np.array(out["pae"])
    ab = float(pae[:n, n:].mean())
    ba = float(pae[n:, :n].mean())
    ipae = (ab + ba) / 2
    log = (a.fold / "log.txt").read_text()
    m = re.search(r"rank_001_\S+\s+pLDDT=([\d.]+)\s+pTM=([\d.]+)\s+ipTM=([\d.]+)", log)
    assert m
    ptm = float(m.group(2))
    iptm = float(m.group(3))
    if a.known_receptor:
        rec = coordinates(a.known_receptor)["A"]
        reference = atoms(rec, ("CA",))
        pred = atoms(f["B"], ("CA",))
        rot, trans = fit(pred, reference)
        chem = json.loads((a.fold / "cyclic_feature_check.json").read_text())
        metrics = dict(
            sequence=chem["peptide"],
            prediction_source="published-sequence public HighFold refold",
            ipTM=iptm,
            pTM=ptm,
            global_pLDDT=float(plddt.mean()),
            peptide_pLDDT=float(plddt[:n].mean()),
            pae_peptide_to_receptor_A=ab,
            pae_receptor_to_peptide_A=ba,
            mean_interchain_PAE_A=ipae,
            normalized_ipAE=round(ipae, 2) / 31,
            fold_terminal_C_N_A=terminal(f["A"]),
            TREM2_CA_rmsd_to_6Y6C_A_A=rmsd(pred @ rot + trans, reference),
            peptide_RMSD_to_paper_backbone_A=None,
            paper_Rosetta_energy_recalculated=None,
            passes_notebook_gate=iptm > 0.8 and round(ipae, 2) / 31 < 0.3,
        )
        (a.run / "verified_scores.json").write_text(
            json.dumps(metrics, indent=2) + "\n"
        )
        return
    r = coordinates(a.run / "6y6c_0.pdb")
    assert set(r) == {"A", "B"} and n == len(r["A"]) and len(r["B"]) == 118
    rbb = atoms(r["A"], ("N", "CA", "C"))
    fbb = atoms(f["A"], ("N", "CA", "C"))
    rot, trans = fit(fbb, rbb)
    rb = atoms(r["B"], ("CA",))
    fb = atoms(f["B"], ("CA",))
    rot2, trans2 = fit(fb, rb)
    chem = json.loads((a.fold / "cyclic_feature_check.json").read_text())
    metrics = dict(
        sequence=chem["peptide"],
        prediction_source="public HighFold CPU substitution",
        ipTM=iptm,
        pTM=ptm,
        global_pLDDT=float(plddt.mean()),
        peptide_pLDDT=float(plddt[:n].mean()),
        pae_peptide_to_receptor_A=ab,
        pae_receptor_to_peptide_A=ba,
        mean_interchain_PAE_A=ipae,
        normalized_ipAE=round(ipae, 2) / 31,
        peptide_self_fit_N_CA_C_rmsd_A=rmsd(fbb @ rot + trans, rbb),
        TREM2_fit_CA_rmsd_A=rmsd(fb @ rot2 + trans2, rb),
        peptide_CA_rmsd_after_TREM2_fit_A=rmsd(
            atoms(f["A"], ("CA",)) @ rot2 + trans2, atoms(r["A"], ("CA",))
        ),
        rf_terminal_C_N_A=terminal(r["A"]),
        fold_terminal_C_N_A=terminal(f["A"]),
        passes_notebook_gate=iptm > 0.8 and round(ipae, 2) / 31 < 0.3,
    )
    (a.run / "verified_scores.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
