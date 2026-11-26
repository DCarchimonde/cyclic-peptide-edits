"""Summarise replicates, source-only candidate pairs and interpretable assay bounds."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from pair_chemistry import ROOT, graph_index, identity, read_csv, write_csv
from pair_data import cells_for, grouped, load_wang, protocol, quality, record_id


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=ROOT / ".cache/pair_graph_index.json")
    a = ap.parse_args()
    out = ROOT / "results/pairs"
    raw = [
        r
        for r in read_csv(ROOT / "data/raw/cycpeptmpdb_all.csv")
        if r["PAMPA"] and r["SMILES"]
    ]
    idx = graph_index(ROOT / "data/raw/cycpeptmpdb_all.csv", a.index)
    cfg = json.loads((ROOT / "configs/pair_protocols.json").read_text())["sources"]
    wang = load_wang(ROOT / "configs/wang2015_pampa.csv")
    labs = {r["CycPeptMPDB_ID"]: quality(r, wang) for r in raw}
    for r in raw:
        r["_graph"] = idx["graphs"].get(r["SMILES"])
    all_source = cells_for(raw, lambda r: r["Source"], labs)
    frozen = cells_for(raw, lambda r: protocol(r, cfg), labs)
    candidates = []
    for source, par, variants in grouped(all_source, idx["edits"]):
        if source not in cfg:
            candidates.append(
                dict(
                    source=source,
                    parent_id=identity(par),
                    measured_parent=int((source, par) in all_source),
                    n_options=len(variants),
                    n_source_delta=len(variants) if (source, par) in all_source else 0,
                    variant_ids="|".join(sorted(map(identity, variants))),
                    status="source-only; protocol requires review",
                )
            )
    write_csv(out / "protocol_review_candidates.csv", candidates)
    bounds = {}
    for r in raw:
        key = protocol(r, cfg)
        if not key or not r["_graph"]:
            continue
        limit = re.fullmatch(r"\s*<\s*(-?\d+(?:\.\d+)?)\s*", r["Detection_Limit_1"])
        if r["Source"] == "2020_Le Roux" and limit:
            bounds[(key, r["_graph"])] = (float(limit[1]), record_id(r))
        if (
            r["Source"] == "2015_Wang"
            and r["Original_Name_in_Source_Literature"] == "Compound.50"
        ):
            bounds[(key, r["_graph"])] = (-8.0, record_id(r))
    quant = {
        k: sum(labs[r["CycPeptMPDB_ID"]][0] for r in rr) / len(rr)
        for k, rr in frozen.items()
    }
    interval = []
    for panel, child in set(bounds) | set(quant):
        for parent in idx["edits"].get(child, {}):
            ip = (panel, parent)
            iv = (panel, child)
            if ip in quant and iv in bounds:
                upper, rid = bounds[iv]
                interval.append(
                    dict(
                        protocol_id=panel,
                        parent_id=identity(parent),
                        variant_id=identity(child),
                        relation="<",
                        log_delta_bound=upper - quant[ip],
                        bound_record=rid,
                    )
                )
            if ip in bounds and iv in quant:
                upper, rid = bounds[ip]
                interval.append(
                    dict(
                        protocol_id=panel,
                        parent_id=identity(parent),
                        variant_id=identity(child),
                        relation=">",
                        log_delta_bound=quant[iv] - upper,
                        bound_record=rid,
                    )
                )
    write_csv(
        out / "interval_edges.csv",
        interval,
        [
            "protocol_id",
            "parent_id",
            "variant_id",
            "relation",
            "log_delta_bound",
            "bound_record",
        ],
    )
    molecule_counts = Counter(r["_graph"] for r in raw if r["_graph"])
    by_src = defaultdict(set)
    within = Counter()
    for r in raw:
        if r["_graph"]:
            by_src[r["_graph"]].add(r["Source"])
            within[(r["Source"], r["_graph"])] += 1
    ev = read_csv(out / "parent_evidence.csv")
    ranks = [r for r in ev if int(r["n_sites"]) >= 2]
    stats = dict(
        stereo_invalid_rows=sum(not r["_graph"] for r in raw),
        unique_isomeric_graphs=len(molecule_counts),
        duplicate_graphs=sum(n > 1 for n in molecule_counts.values()),
        duplicate_graphs_across_sources=sum(len(s) > 1 for s in by_src.values()),
        excess_rows_within_source=sum(n - 1 for n in within.values()),
        duplicate_frozen_point_cells=sum(len(v) > 1 for v in frozen.values()),
        quantitative_group_parent_status=dict(
            Counter(r["parent_status"] for r in ranks)
        ),
        ranking_group_size=dict(
            sorted(Counter(int(r["n_sites"]) for r in ranks).items())
        ),
        protocol_specific_explicit_bounds=len(bounds),
        measured_interval_edges=len(interval),
        quality_statuses=dict(Counter(y[1] for y in labs.values())),
    )
    (out / "qualification.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(stats)


if __name__ == "__main__":
    main()
