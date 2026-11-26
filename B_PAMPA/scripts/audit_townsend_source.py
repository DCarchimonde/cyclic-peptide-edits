"""Reconcile Townsend's original two workbooks with all source records, pools and frozen edits."""

import argparse
import ast
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import log10
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from openpyxl import load_workbook
from rdkit import Chem
from pair_data import ROOT, canonical, protocol, read_csv, write_csv

BASE = "https://www.cambridge.org/engage/api-gateway/coe/assets/orp/resource/item/"
BOOKS = {
    "hexa": (
        "60c752c3ee301c003cc7ac8a",
        "de64c36b8b91a2a05b80e321c118f1c2faca70777277a8f33fe4877fe1d4ee08",
    ),
    "hepta": (
        "60c752c3337d6c1032e28778",
        "a079512f914476c2b4062aaeaae3d802e29a626a953a6506ade977068bf7c874",
    ),
}
TOKENS = {
    "bHph": "BHF",
    "dL": "l",
    "dP": "p",
    "F": "F",
    "L": "L",
    "Me_dL": "Ml",
    "meL": "ML",
    "Bn_Gly": "Bnz",
    "P": "P",
    "Pr_Gly": "Prp",
    "A": "A",
    "dA": "a",
    "meA": "MA",
    "Me_dA": "Ma",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workbook-dir", type=Path)
    a = ap.parse_args()
    namecol = "Original_Name_in_Source_Literature"
    raw = {
        r[namecol]: r
        for r in read_csv(ROOT / "data/raw/cycpeptmpdb_all.csv")
        if r["Source"] == "2020_Townsend" and r["PAMPA"]
    }
    cfg = json.loads((ROOT / "configs/pair_protocols.json").read_text())["sources"]
    checked = {}
    info = {}
    states = Counter()
    errors = []
    forward = defaultdict(set)
    reverse = defaultdict(set)
    with TemporaryDirectory() as temp:
        directory = a.workbook_dir or Path(temp)
        directory.mkdir(exist_ok=True, parents=True)
        for prefix, (item, h) in BOOKS.items():
            filename = f"curated-{prefix}mer-permeability-data.xlsx"
            url = BASE + item + "/original/" + filename
            path = directory / filename
            if not path.exists():
                with urlopen(
                    Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120
                ) as response:
                    path.write_bytes(response.read())
            if sha256(path.read_bytes()).hexdigest() != h:
                raise ValueError("Source workbook version mismatch")
            book = load_workbook(path, read_only=True, data_only=True)
            values = iter(book.active.values)
            next(values)
            cols = next(values)
            count = 0
            for row in values:
                source = dict(zip(cols, row, strict=True))
                count += 1
                name = prefix + "_" + str(source["Column Name"])
                r = raw[name]
                seq = ",".join(TOKENS[t] for t in ast.literal_eval(r["Sequence"]))
                if name in checked or seq != source["Top Sequence"]:
                    raise ValueError("Duplicate or wrong residue sequence: " + name)
                if canonical(Chem.MolFromSmiles(r["SMILES"])) != canonical(
                    Chem.MolFromSmiles(source["SMILES"])
                ):
                    raise ValueError("Wrong graph: " + name)
                key = protocol(r, cfg)
                pool = prefix + ":" + source["Expt"]
                forward[key].add(pool)
                reverse[pool].add(key)
                val = source["PappE-6"]
                y = float(r["PAMPA"])
                log_value = None
                if val is None:
                    status = "blank"
                    if y != -4:
                        raise ValueError(name)
                elif val == 0:
                    status = "zero"
                    if y != -10:
                        raise ValueError(name)
                else:
                    status = "positive"
                    log_value = log10(val) - 6
                    errors.append(abs(log_value - y))
                    if errors[-1] > 0.0050001:
                        raise ValueError("Label discrepancy: " + name)
                states[status] += 1
                checked[name] = dict(
                    record_id="CycPeptMPDB_record:" + r["CycPeptMPDB_ID"],
                    name=name,
                    source_pool=pool,
                    reconstructed_protocol=key,
                    source_sequence=seq,
                    source_mass=source["Mass"],
                    source_pct_transmission=source["%T"],
                    source_Papp_E_minus_6=val,
                    source_log10_cm_s=log_value,
                    source_status=status,
                    db_log10_cm_s=y,
                )
            info[prefix] = dict(url=url, sha256=h, rows=count)
            book.close()
    if set(checked) != set(raw) or any(
        len(x) != 1 for x in [*forward.values(), *reverse.values()]
    ):
        raise ValueError("Unreconciled samples/pools")
    groups = defaultdict(list)
    pools = defaultdict(set)
    sub = Counter()
    mass_errors = []
    for ed in read_csv(ROOT / "results/pairs/edits.csv"):
        if ed["source"] != "2020_Townsend":
            continue
        v = checked[ed["variant_original_names"]]
        groups[ed["group_id"]].append(v["source_mass"])
        pools[ed["group_id"]].add(v["source_pool"])
        if ed["parent_y"]:
            p = checked[ed["parent_original_names"]]
            if p["source_pool"] != v["source_pool"]:
                raise ValueError(ed["pair_id"])
            diff = [
                (i + 1, x, y)
                for i, (x, y) in enumerate(
                    zip(
                        p["source_sequence"].split(","),
                        v["source_sequence"].split(","),
                        strict=True,
                    )
                )
                if x != y
            ]
            if (
                len(diff) != 1
                or str(diff[0][0]) != ed["parent_residue_1based"]
                or diff[0][1:] not in {("L", "ML"), ("l", "Ml")}
            ):
                raise ValueError(ed["pair_id"])
            sub[diff[0][1] + " to " + diff[0][2]] += 1
            mass_errors.append(abs(v["source_mass"] - p["source_mass"] - 14.015650064))
    if any(len(v) > 1 for v in pools.values()):
        raise ValueError("Ranking group spans experimental pools")
    out = ROOT / "results/pairs/townsend_source_audit"
    out.mkdir(exist_ok=True)
    write_csv(out / "record_matches.csv", checked.values())
    audit = dict(
        point_pairs=len(mass_errors),
        source_sequence_substitutions=dict(sub),
        point_pairs_max_abs_mass_delta_error_vs_CH2_Da=max(mass_errors),
        rank_groups=sum(len(m) > 1 for m in groups.values()),
        rank_max_source_mass_range_Da=max(max(m) - min(m) for m in groups.values()),
    )
    stats = dict(
        author_preprint="https://doi.org/10.26434/chemrxiv.13335941.v1",
        workbooks=info,
        isomeric_graph_and_token_sequence_matches=len(checked),
        author_pools_in_exact_bijection_with_reconstruction=len(forward),
        source_PappE_minus_6_field=dict(states),
        finite_max_roundoff_vs_snapshot=max(errors),
        zero_encoded_minus10=states["zero"],
        blank_encoded_minus4=states["blank"],
        frozen_assay_checks=audit,
        interpretation="Workbook Mass is an author-reported field. Blank source Papp does not support a numeric upper bound.",
    )
    (out / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
