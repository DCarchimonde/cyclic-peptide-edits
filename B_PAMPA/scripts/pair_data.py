"""Build matched backbone N-methylation groups and measured parent–child deltas."""

from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from statistics import median
from pair_chemistry import (
    ROOT,
    read_csv,
    write_csv,
    identity,
    canonical,
    graph_index,
    edits_for,
    eligible,
)


def protocol(row, cfg):
    source = row["Source"]
    if source not in cfg:
        return ""
    rule = cfg[source]["group_rule"]
    name = row["Original_Name_in_Source_Literature"]
    if rule == "townsend_sublibrary":
        seq = ast.literal_eval(row["Sequence"])
        panel = name.split("_")[0]
        parts = (
            ["hexa", seq[0], seq[3]]
            if panel == "hexa"
            else ["hepta", seq[0], seq[1], seq[4]]
        )
    elif rule == "kelly_sublibrary":
        lib, well = name.split("_", 1)
        parts = [lib, well.split(".")[0]] if lib == "L1" else [lib]
    elif rule == "furukawa_sublibrary":
        match = re.fullmatch(r"(\d+\.\d+)-\d+(?:Me|H)?", name)
        if not match:
            return ""
        parts = [match[1]]
    else:
        parts = ["PAMPA-panel"]
    return ":".join([source, *parts])


def record_id(r):
    return "CycPeptMPDB_record:" + r["CycPeptMPDB_ID"]


def quality(r, wang):
    if r["Source"] == "2015_Wang":
        return wang[r["Original_Name_in_Source_Literature"]]
    if r["Detection_Limit_1"]:
        return None, "reported_limit_or_nonquantitative"
    y = float(r["PAMPA"])
    if y <= -10 or y >= -4:
        return None, "database_boundary"
    if r["T_PAMPA"] and float(r["T_PAMPA"]) <= 0:
        return None, "zero_transmission_QC"
    return y, "point"


def tokens(p, v):
    if not p:
        return ["", "", "", ""]
    left = ast.literal_eval(p["Sequence"])
    right = ast.literal_eval(v["Sequence"])
    if len(left) != len(right):
        return ["", "", "", ""]
    diffs = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
    if len(diffs) != 1:
        return ["", "", "", ""]
    i = diffs[0]
    return [i + 1, i + 1, left[i], right[i]]


def cells_for(raw, key, quality_by_id, quant=True):
    cells = defaultdict(list)
    for row in raw:
        if not row["_graph"] or (
            quant and quality_by_id[row["CycPeptMPDB_ID"]][0] is None
        ):
            continue
        panel = key(row)
        if panel:
            cells[(panel, row["_graph"])].append(row)
    return cells


def grouped(cells, edits):
    groups = defaultdict(dict)
    for panel, s in cells:
        for parent, edit in edits[s].items():
            groups[(panel, parent)][s] = edit
    return [
        (panel, parent, vs)
        for (panel, parent), vs in groups.items()
        if (panel, parent) in cells or len(vs) > 1
    ]


def make_rows(cfg, allcells, points, available, index, labels):
    output = []
    evidence = []
    for panel, parent, variants in grouped(points, index["edits"]):
        source = panel.split(":")[0]
        p_id = identity(parent)
        group = identity(panel + "|" + p_id, "grp")
        pp = points.get((panel, parent), [])
        status = (
            "measured_same_protocol"
            if pp
            else (
                "same_protocol_nonquantitative"
                if (panel, parent) in allcells
                else (
                    "other_protocol_only"
                    if parent in available
                    else "virtual_unobserved"
                )
            )
        )
        evidence.append(
            dict(
                group_id=group,
                protocol_id=panel,
                parent_id=p_id,
                n_sites=len(variants),
                parent_status=status,
            )
        )
        for child, (pn, vn, me, mapping) in variants.items():
            v_id = identity(child)
            vv = points[(panel, child)]
            val = lambda rr: median(labels[x["CycPeptMPDB_ID"]][0] for x in rr)
            py = val(pp) if pp else None
            vy = val(vv)
            ppos, vpos, pt, vt = tokens(pp[0] if pp else None, vv[0])
            output.append(
                dict(
                    pair_id=identity(group + "|" + v_id, "pair"),
                    group_id=group,
                    series_id=identity(cfg[source]["family"], "series"),
                    split=cfg[source]["split"],
                    protocol_id=panel,
                    source=source,
                    parent_id=p_id,
                    variant_id=v_id,
                    parent_smiles=parent,
                    variant_smiles=child,
                    edit_site=pn,
                    parent_n_idx=pn,
                    variant_n_idx=vn,
                    variant_methyl_idx=me,
                    parent_to_variant_atom_map=json.dumps(
                        mapping, separators=(",", ":")
                    ),
                    parent_y="" if py is None else py,
                    variant_y=vy,
                    delta="" if py is None else round(vy - py, 8),
                    parent_record_ids="|".join(map(record_id, pp)),
                    variant_record_ids="|".join(map(record_id, vv)),
                    parent_sequence=pp[0]["Sequence"] if pp else "",
                    variant_sequence=vv[0]["Sequence"],
                    parent_residue_1based=ppos,
                    variant_residue_1based=vpos,
                    parent_token=pt,
                    variant_token=vt,
                    parent_original_names="|".join(
                        r["Original_Name_in_Source_Literature"] for r in pp
                    ),
                    variant_original_names="|".join(
                        r["Original_Name_in_Source_Literature"] for r in vv
                    ),
                    topology=vv[0]["Molecule_Shape"],
                    n_measured_sites=len(variants),
                    backbone_id=identity(index["backbones"][child], "base"),
                )
            )
    output.sort(key=lambda r: (r["group_id"], r["variant_id"]))
    return output, sorted(evidence, key=lambda r: r["group_id"])


def summary(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)
    ranked = [v for v in groups.values() if len(v) >= 2]
    obs = {r["variant_id"] for r in rows} | {
        r["parent_id"] for r in rows if r["parent_y"] != ""
    }
    return dict(
        options=len(rows),
        measured_delta=sum(r["delta"] != "" for r in rows),
        groups_including_singletons=len(groups),
        rank_groups=len(ranked),
        rankable_unique_chemical_parents=len({v[0]["parent_id"] for v in ranked}),
        rank_options=sum(map(len, ranked)),
        parent_without_point_rank_groups=sum(v[0]["parent_y"] == "" for v in ranked),
        unique_measured_molecules=len(obs),
        distinct_chemical_parents=len({r["parent_id"] for r in rows}),
        de_NMe_backbones=len({r["backbone_id"] for r in rows}),
        source_count=len({r["source"] for r in rows}),
        protocol_count=len({r["protocol_id"] for r in rows}),
        series=len({r["series_id"] for r in rows}),
    )


def load_wang(path):
    ww = {}
    for r in read_csv(path):
        original = (
            r.get("source_compound")
            or r.get("original_name")
            or r.get("Original_Name_in_Source_Literature")
            or r.get("name")
        )
        if not original:
            col = next(
                k
                for k in r
                if k.lower() in {"compound", "compound_number", "compound_no"}
            )
            original = "Compound." + r[col]
        number = int(re.search(r"\d+", original)[0])
        if number in {12, 16, 29, 30, 31, 46, 49, 50, 59, 60}:
            ww["Compound." + str(number)] = (
                None,
                "low_recovery" if number in {12, 16} else "source_nonquantitative",
            )
            continue
        col = next(
            k
            for k in [
                "si_log10_p_cm_s",
                "source_log10_cm_s",
                "log10p_cm_s",
                "si_log10_cm_s",
                "log10_pampa_cm_s",
                "pampa_log10_cm_s",
                "log10_cm_s",
                "log10P_cm_s",
            ]
            if k in r
        )
        ww["Compound." + str(number)] = (float(r[col]), "point")
    return ww


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw/cycpeptmpdb_all.csv")
    ap.add_argument("--index", type=Path, default=ROOT / ".cache/pair_graph_index.json")
    ap.add_argument("--output", type=Path, default=ROOT / "results/pairs")
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "configs/pair_protocols.json").read_text())["sources"]
    raw = [r for r in read_csv(args.raw) if r["PAMPA"] and r["SMILES"]]
    index = graph_index(args.raw, args.index)
    wang = load_wang(ROOT / "configs/wang2015_pampa.csv")
    labels = {r["CycPeptMPDB_ID"]: quality(r, wang) for r in raw}
    for r in raw:
        r["_graph"] = index["graphs"].get(r["SMILES"])
    graphs = {r["_graph"] for r in raw if r["_graph"]}
    protocol_key = lambda r: protocol(r, cfg)
    points = cells_for(raw, protocol_key, labels)
    allcells = cells_for(raw, protocol_key, labels, False)
    rows, evidence = make_rows(cfg, allcells, points, graphs, index, labels)
    write_csv(out / "edits.csv", rows)
    write_csv(out / "parent_evidence.csv", evidence)
    groups = {
        r["group_id"]: {
            k: r[k]
            for k in (
                "group_id",
                "series_id",
                "split",
                "protocol_id",
                "parent_id",
                "n_measured_sites",
                "parent_y",
            )
        }
        for r in rows
    }
    write_csv(out / "groups.csv", groups.values())
    mols = {}
    for r in rows:
        for role in ["parent", "variant"]:
            key = r[role + "_id"]
            if key in mols and mols[key]["split"] != r["split"]:
                raise ValueError("Shared graph crosses split")
            mols[key] = dict(
                molecule_id=key,
                split=r["split"],
                series_id=r["series_id"],
                smiles=r[role + "_smiles"],
                backbone_id=r["backbone_id"],
            )
    write_csv(
        out / "molecule_split.csv",
        sorted(mols.values(), key=lambda m: m["molecule_id"]),
    )
    write_csv(
        out / "records.csv",
        [
            dict(
                record_id=record_id(r),
                source=r["Source"],
                name=r["Original_Name_in_Source_Literature"],
                protocol_id=protocol_key(r),
                molecule_id=identity(r["_graph"]) if r["_graph"] else "",
                structure_status=index["states"][r["SMILES"]] or "eligible",
                raw_pampa=r["PAMPA"],
                pampa_point=labels[r["CycPeptMPDB_ID"]][0],
                point_status=labels[r["CycPeptMPDB_ID"]][1],
                limit1=r["Detection_Limit_1"],
                limit2=r["Detection_Limit_2"],
                T_PAMPA=r["T_PAMPA"],
            )
            for r in raw
        ],
    )
    stages = []
    for name, key, quant in [
        ("all_source", lambda r: r["Source"], False),
        ("all_quant", lambda r: r["Source"], True),
        ("verified_sources", lambda r: r["Source"] if r["Source"] in cfg else "", True),
        ("verified_panels", protocol_key, True),
    ]:
        cells = cells_for(raw, key, labels, quant)
        gs = grouped(cells, index["edits"])
        stages.append(
            dict(
                stage=name,
                options=sum(len(v) for _, _, v in gs),
                deltas=sum(len(v) for p, g, v in gs if (p, g) in cells),
                rank_groups=sum(len(v) >= 2 for _, _, v in gs),
            )
        )
    write_csv(out / "coverage_stages.csv", stages)
    source_summary = [
        dict(
            source=s,
            family=c["family"],
            series_id=identity(c["family"], "series"),
            split=c["split"],
            **summary([r for r in rows if r["source"] == s]),
        )
        for s, c in cfg.items()
    ]
    write_csv(out / "series.csv", source_summary)
    write_csv(out / "source_counts.csv", source_summary)
    stats = dict(
        primary=summary(rows),
        splits={
            s: summary([r for r in rows if r["split"] == s])
            for s in ["train", "valid", "test"]
        },
        raw_PAMPA_rows=len(raw),
        stereo_eligible_rows=sum(bool(r["_graph"]) for r in raw),
        stereo_graphs=len(graphs),
        edges_sha256=hashlib.sha256((out / "edits.csv").read_bytes()).hexdigest(),
    )
    (out / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
