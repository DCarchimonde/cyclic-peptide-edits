"""Count the published cache and apply notebook criteria to the 6Y6C rows."""

from collections import Counter
from pathlib import Path
import hashlib, ast, csv, json, sys, urllib.request

nb = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
if not nb.is_file():
    u = "https://raw.githubusercontent.com/hongliangduan/AI-Guided-Design-of-Cyclic-Peptide-Binders-Targeting-TREM2-Pipeline/7acdbbbb707f256cacd5f6f3e838bbe63a62dd40/scripts/rf_mpnn_highfold_code.ipynb"
    nb.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(u, timeout=60) as f:
        nb.write_bytes(f.read())
b = nb.read_bytes()
assert (
    hashlib.sha256(b).hexdigest()
    == "0e6b399a04c9dd0655bbfd70ae061197e1290e1898f4aa76bc5e20df8178be59"
)
lines = "".join(json.loads(b)["cells"][6]["outputs"][0]["text"]).splitlines()
tbl = ast.literal_eval(next(x for x in lines if x.startswith("[[")))
rows = sorted(
    [
        int(r[1]),
        int(r[2]),
        int(r[3]),
        r[5],
        float(r[6]),
        float(r[7]),
        float(r[8]),
        float(r[9]),
        float(r[11]),
        float(r[8]) > 0.8 and float(r[9]) < 0.3,
    ]
    for r in tbl
    if r[0].lower() == "6y6c"
)
with (out / "cached_model_rows.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "backbone",
            "sequence_index",
            "rank",
            "model_id",
            "pLDDT",
            "pTM",
            "ipTM",
            "normalized_ipAE",
            "shorter_chain_backbone_RMSD_A",
            "passed",
        ]
    )
    w.writerows(rows)
(out / "cached_summary.json").write_text(
    json.dumps(
        dict(
            all_targets_model_rows=len(tbl),
            rows_by_pdb=dict(Counter(r[0] for r in tbl)),
            rank1_ipTM_range=[
                min(r[6] for r in rows if r[2] == 1),
                max(r[6] for r in rows if r[2] == 1),
            ],
            rank1_normalized_ipAE_range=[
                min(r[7] for r in rows if r[2] == 1),
                max(r[7] for r in rows if r[2] == 1),
            ],
            present_backbone_indices=sorted({r[0] for r in rows}),
            target_rows=len(rows),
            backbones=len({r[0] for r in rows}),
            sequence_ids=len({tuple(r[:2]) for r in rows}),
            passed_model_rows=sum(r[-1] for r in rows),
            passed_sequence_ids=len({tuple(r[:2]) for r in rows if r[-1]}),
        ),
        indent=2,
    )
    + "\n"
)
