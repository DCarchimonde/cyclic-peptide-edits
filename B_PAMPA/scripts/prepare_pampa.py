"""Prepare a PAMPA-only starter table without combining different assays.

This script preserves measurement records and source-supplied structural IDs.
It does not create train/test splits, claim chemical deduplication, or infer
missing labels. Potential censoring and detection-limit notes remain explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def clean_id(value: object) -> str:
    value = clean_text(value)
    if value.endswith(".0") and value[:-2].isdigit():
        return value[:-2]
    return value


def classify_label(value: float, notes: str) -> str:
    # CycPeptMPDB normalizes/clips some out-of-range or unquantified values.
    # Notes may refer to a different assay, so do not invent a precise PAMPA
    # bound from prose. The exploratory regression can select only "eq" rows.
    if value <= -10.0:
        return "floor_or_left_censored"
    if value >= -4.0:
        return "ceiling_or_right_censored"
    if notes:
        return "requires_assay_review"
    return "eq"


def prepare(raw: pd.DataFrame, notes_lookup: dict[str, str] | None = None) -> pd.DataFrame:
    required = {"CycPeptMPDB_ID", "Structurally_Unique_ID", "SMILES", "Source", "PAMPA"}
    if missing := required - set(raw.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    rows: list[dict] = []
    for record in raw.to_dict("records"):
        target = pd.to_numeric(record.get("PAMPA"), errors="coerce")
        smiles = clean_text(record.get("SMILES"))
        if pd.isna(target) or not smiles or smiles.lower() in {"nan", "none", "null"}:
            continue
        record_id = clean_id(record["CycPeptMPDB_ID"])
        structure_id = clean_id(record["Structurally_Unique_ID"])
        if not structure_id:
            raise ValueError(f"Missing source structural identity for record {record_id}")
        notes = " | ".join(
            part for part in (
                clean_text(record.get("Detection_Limit_1")),
                clean_text(record.get("Detection_Limit_2")),
            ) if part
        )
        if not notes and notes_lookup:
            notes = notes_lookup.get(record_id, "")
        molecule_id = "CycPeptMPDB:" + structure_id
        rows.append({
            "molecule_id": molecule_id,
            "smiles": smiles,
            "target": float(target),
            "source": clean_text(record["Source"]),
            "group": molecule_id,
            "endpoint": "PAMPA_log10_cm_per_s",
            "label_relation": classify_label(float(target), notes),
            "record_id": "CycPeptMPDB_record:" + record_id,
            "upstream_structurally_unique_id": structure_id,
            "sequence": clean_text(record.get("Sequence")),
            "monomer_length": clean_id(record.get("Monomer_Length")),
            "topology": clean_text(record.get("Molecule_Shape")),
            "detection_limit_notes": notes,
        })
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> dict:
    return {
        "records": len(table),
        "source_supplied_structural_ids": int(table["molecule_id"].nunique()),
        "literal_smiles_strings": int(table["smiles"].nunique()),
        "sources": dict(sorted(Counter(table["source"]).items())),
        "label_relations": dict(Counter(table["label_relation"])),
        "monomer_lengths": dict(Counter(table["monomer_length"])),
        "chemical_canonicalization_performed": False,
        "train_test_split_created": False,
        "matched_edit_pairs_counted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/cycpeptmpdb_all.csv")
    parser.add_argument("--four-d", type=Path, default=ROOT / "data/raw/cycpeptmpdb_4d.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.raw, low_memory=False, encoding="utf-8-sig")
    full = prepare(raw)
    full.to_csv(args.output_dir / "pampa.csv", index=False)
    summary = {
        "input_file": args.raw.name,
        "input_sha256": hashlib.sha256(args.raw.read_bytes()).hexdigest(),
        "raw_records": len(raw),
        "pampa": summarize(full),
        "label_policy": {
            "eq": "Numeric database value with no detection-limit notes or boundary flag; not independently verified assay precision.",
            "floor_or_left_censored": "At or below -10; retain the value but do not treat it as an exact regression label.",
            "ceiling_or_right_censored": "At or above -4; retain the value but flag possible upper clipping.",
            "requires_assay_review": "Detection-limit prose is present; its applicability to PAMPA requires source review.",
        },
        "group_policy": "group equals the upstream structural identity. For source holdout, also exclude or move any identity present across both sets; recompute chemical identity from isomeric SMILES before formal training.",
    }
    if args.four_d.exists():
        four_d = pd.read_csv(args.four_d, low_memory=False)
        notes_lookup = {
            row.record_id.removeprefix("CycPeptMPDB_record:"): row.detection_limit_notes
            for row in full.itertuples()
        }
        prepared_4d = prepare(four_d, notes_lookup)
        prepared_4d.to_csv(args.output_dir / "pampa_4d.csv", index=False)
        summary["four_d"] = summarize(prepared_4d)
        summary["four_d"]["input_sha256"] = hashlib.sha256(args.four_d.read_bytes()).hexdigest()
        summary["four_d"]["independent_new_labels"] = False
    (args.output_dir / "data_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
