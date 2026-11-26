"""Check archived file integrity, entry identities, scores and masks without model weights."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def verify_hashes(bundle):
    for line in (bundle / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"File digest mismatch: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("bundle", type=Path, nargs="?", default=Path("."))
    a = p.parse_args()
    base = a.bundle.resolve()
    out = base / "outputs"
    verify_hashes(base)
    expected = {
        "known_" + r["paper_id"]: r["sequence_cyclic_notation"][2:-1]
        for r in csv.DictReader(
            (out / "published/figure2_and_esi_transcription.csv").open()
        )
    }
    for name in ("short15", "default50"):
        fa = (out / name / "mpnn/seqs/6y6c_0.fa").read_text().splitlines()
        assert len(fa) == 12
        expected[name] = fa[3]
        assert json.loads(
            (out / name / "repeat_fixed_backbone/repeat_check.json").read_text()
        )["byte_identical"]
    for name, seq in expected.items():
        d = out / name
        metrics = json.loads((d / "verified_scores.json").read_text())
        mask = json.loads((d / "fold/cyclic_feature_check.json").read_text())
        raw = json.loads(next((d / "fold").glob("*_scores_rank_*.json")).read_text())
        assert seq == metrics["sequence"] == mask["peptide"]
        assert (
            mask["cyclic_residue_indices"] == list(range(len(seq)))
            and abs(mask["terminal_relative_index"]) == 1
        )
        assert len(raw["pae"]) == len(raw["plddt"]) == len(seq) + 118
        assert len(list((d / "fold").glob("*_unrelaxed_rank_*.pdb"))) == 1
        assert metrics["passes_notebook_gate"] == (
            metrics["ipTM"] > 0.8 and metrics["normalized_ipAE"] < 0.3
        )
    count = json.loads((out / "cache/cached_summary.json").read_text())
    assert (
        count["target_rows"],
        count["backbones"],
        count["sequence_ids"],
        count["passed_sequence_ids"],
    ) == (170, 8, 34, 0)
    print(
        f"All archived hashes, {len(expected)} folded complexes, both five-draw repeats, and 170 cached TREM2 model rows pass."
    )


if __name__ == "__main__":
    main()
