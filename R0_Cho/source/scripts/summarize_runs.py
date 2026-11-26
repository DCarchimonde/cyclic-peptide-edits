"""Tabulate measured model outputs without filling missing runs or unobserved references."""

import argparse
import csv
import json
from pathlib import Path


def write_csv(path, fields, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path, help="directory containing run subdirectories")
    p.add_argument("paper", type=Path, help="published figure and ESI transcription")
    a = p.parse_args()
    with a.paper.open() as f:
        reported = {r["paper_id"]: r for r in csv.DictReader(f)}
    fields = [
        "setting",
        "peptide",
        "pLDDT",
        "ipTM",
        "mean_interchain_PAE_A",
        "normalized_ipAE",
        "terminal_C_N_A",
        "passes_gate",
    ]
    rows = []
    comparisons = []
    for d in sorted(a.out.iterdir()):
        flag = d / "verified_scores.json"
        if not flag.is_file():
            continue
        m = json.loads(flag.read_text())
        rows.append(
            dict(
                setting=d.name,
                peptide=m["sequence"],
                pLDDT=m["global_pLDDT"],
                ipTM=m["ipTM"],
                mean_interchain_PAE_A=m["mean_interchain_PAE_A"],
                normalized_ipAE=m["normalized_ipAE"],
                terminal_C_N_A=m["fold_terminal_C_N_A"],
                passes_gate=m["passes_notebook_gate"],
            )
        )
        if d.name.startswith("known_"):
            name = d.name.replace("known_", "", 1)
            ref = reported[name]
            assert ref["sequence_cyclic_notation"] == f"c({m['sequence']})"
            comparisons.append(
                dict(
                    paper_id=name,
                    sequence=m["sequence"],
                    reported_pLDDT=ref["paper_pLDDT"],
                    offline_pLDDT=m["global_pLDDT"],
                    reported_ipTM=ref["paper_ipTM"],
                    offline_ipTM=m["ipTM"],
                    reported_ipAE=ref["paper_ipAE"],
                    offline_normalized_ipAE=m["normalized_ipAE"],
                    reported_RMSD_A=ref["paper_RMSD_A"],
                    offline_paper_backbone_RMSD_A="",
                    reported_dG_separated_per_dSASA_x100=ref[
                        "paper_dG_separated_per_dSASA_x100"
                    ],
                    offline_reported_Rosetta_metric="",
                    measured_affinity="",
                    reason_unavailable="author design coordinates and historical Rosetta protocol unavailable; no new binding measurement",
                )
            )
    write_csv(a.out / "current_inference_summary.csv", fields, rows)
    if comparisons:
        write_csv(
            a.out / "paper_vs_known_refold.csv", list(comparisons[0]), comparisons
        )


if __name__ == "__main__":
    main()
