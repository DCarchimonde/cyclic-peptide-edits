"""Audit upstream exact isomeric structures, PAMPA label pools, and ncAA-Scan edit membership."""

import argparse
import ast
import functools
from collections import defaultdict
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

SCAN = set(
    "WLDBGBOLVFXQEQ FGJYCQZOVFHZSJ OMDDTMIDUMCLBS XYMKVDFFOBJTDE DTQBDLGMKJDGKX OEVOOQJTKQNVSU GTDAFIFPWXDGLB NXLJBVKUHJSEJT".split()
)


@functools.lru_cache(None)
def canon(s):
    m = Chem.MolFromSmiles(s)
    if not m:
        raise ValueError("invalid upstream SMILES")
    return Chem.MolToSmiles(m, isomericSmiles=True)


@functools.lru_cache(None)
def scaffold(s):
    return Chem.MolToInchiKey(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s))).split("-")[0]


def single(a, b):
    if len(a) != len(b) or not a:
        return False
    return (
        min(
            sum(x != y for x, y in zip(a, k[i:] + k[:i]))
            for k in (b, b[::-1])
            for i in range(len(a))
        )
        == 1
    )


def run(a):
    a.out.mkdir(parents=True, exist_ok=True)
    e = pd.read_csv(a.edits)
    n = a.external / "ncAA-RepDistill/data/raw"
    c = a.external / "CycDelta/data"
    sets = {}
    d = pd.read_csv(
        c / "CycPeptMPDB/cycpept_mixed_converted.csv", usecols=["Index", "SMILES", "Method"]
    )
    for split in ("train", "valid", "test"):
        ids = {int(x) for x in (c / f"data_split/holdout_split/OD_{split}.txt").read_text().split()}
        g = d[d.Index.isin(ids)]
        sets[f"cycdelta_{split}_any"] = set(g.SMILES.map(canon))
        sets[f"cycdelta_{split}_pampa"] = set(
            g.loc[g.Method.str.upper().eq("PAMPA"), "SMILES"].map(canon)
        )
    for split in ("train", "val", "test"):
        t = pd.read_csv(n / f"downstream/ncaa_cpp_{split}.csv", usecols=["SMILES"])
        sets[f"ncaa_p3_{split}"] = set(t.SMILES.map(canon))
    for split in ("train", "val"):
        t = pd.read_csv(n / f"phase2_split/adapt_{split}.csv", usecols=["SMILES"])
        sets[f"ncaa_p2_{split}"] = set(t.SMILES.map(canon))
    t = pd.read_csv(
        n / "downstream/ncaa_cpp_test.csv", usecols=["SMILES", "Sequence", "Permeability"]
    )
    scan = defaultdict(list)
    for v in t.itertuples():
        scan[canon(v.SMILES)].append((tuple(ast.literal_eval(v.Sequence)), v.Permeability))
    rows = []
    for v in e.itertuples():
        row = {
            k: getattr(v, k)
            for k in ("pair_id", "group_id", "series_id", "split", "parent_id", "variant_id")
        }
        for role in ("parent", "variant"):
            s = canon(getattr(v, role + "_smiles"))
            for name, pool in sets.items():
                row[role + "_" + name] = s in pool
            row[role + "_murcko14"] = scaffold(s)
        ok = (
            row["parent_ncaa_p3_test"]
            and row["variant_ncaa_p3_test"]
            and row["parent_murcko14"] == row["variant_murcko14"]
            and row["parent_murcko14"] in SCAN
        )
        m = (
            [
                (a[1], b[1])
                for a in scan[canon(v.parent_smiles)]
                for b in scan[canon(v.variant_smiles)]
                if single(a[0], b[0])
            ]
            if ok
            else []
        )
        row["both_published_ncaa_scan_structural_pool"] = ok
        row["published_ncaa_scan_edge_eligible"] = bool(m)
        row["scan_edge_matches_both_measured_endpoints"] = bool(
            pd.notna(v.parent_y)
            and any(abs(p - v.parent_y) < 1e-4 and abs(c - v.variant_y) < 1e-4 for p, c in m)
        )
        rows.append(row)
    records = pd.DataFrame(rows)
    records.to_csv(a.out / "overlap_by_edit.csv", index=False)
    tally = []
    for split, t in records.groupby("split"):
        for name, pool in sets.items():
            p = t["parent_" + name]
            v = t["variant_" + name]
            either = p | v
            clean = (~either).groupby(t.group_id).all()
            tally.append(
                dict(
                    split=split,
                    external_pool=name,
                    upstream_structures=len(pool),
                    edit_options=len(t),
                    parent_seen=int(p.sum()),
                    variant_seen=int(v.sum()),
                    either_seen=int(either.sum()),
                    both_seen=int((p & v).sum()),
                    distinct_seen_structures=pd.concat(
                        [t.loc[p, "parent_id"], t.loc[v, "variant_id"]]
                    ).nunique(),
                    clean_parents_all_options=int(clean.sum()),
                    clean_rank_groups=int((clean & t.groupby("group_id").size().ge(2)).sum()),
                    unseen_sites=int((~either).sum()),
                )
            )
    pd.DataFrame(tally).to_csv(a.out / "overlap_by_split.csv", index=False)
    records.groupby("split").agg(
        options=("pair_id", "size"),
        both_structures=("both_published_ncaa_scan_structural_pool", "sum"),
        eligible_reported_edits=("published_ncaa_scan_edge_eligible", "sum"),
        with_matching_values=("scan_edge_matches_both_measured_endpoints", "sum"),
    ).reset_index().to_csv(a.out / "ncaa_scan_overlap.csv", index=False)
    all_counts = {s.count(".") + 1 for role in ("parent", "variant") for s in e[role + "_smiles"]}
    audit = []
    for part in ("train", "val"):
        cands = set()
        segments = set()
        multi = nt = eligible = missing = 0
        for g in pd.read_csv(
            n / f"pretrain/uniref_lt30_c30/{part}_sequences.csv",
            usecols=["smiles"],
            chunksize=10000,
        ):
            raw = g.smiles.dropna()
            nt += len(g)
            missing += len(g) - len(raw)
            nc = raw.str.count(r"\.") + 1
            for s in raw:
                segments.update(s.split("."))
            multi += int((nc > 1).sum())
            keep = raw[nc.isin(all_counts)]
            eligible += len(keep)
            cands.update(map(canon, keep))
        if any(
            Chem.MolFromSmiles(s) is None or len(Chem.GetMolFrags(Chem.MolFromSmiles(s))) != 1
            for s in segments
        ):
            raise ValueError("invalid component identity")
        for split, g in e.groupby("split"):
            p = g.parent_smiles.map(canon).isin(cands)
            v = g.variant_smiles.map(canon).isin(cands)
            audit.append(
                dict(
                    upstream_phase1_partition=part,
                    split=split,
                    upstream_rows=nt,
                    upstream_missing_smiles=missing,
                    upstream_multicomponent_rows=multi,
                    distinct_independently_closed_fragments=len(segments),
                    component_count_eligible_upstream_rows=eligible,
                    options=len(g),
                    parent_matches=int(p.sum()),
                    variant_matches=int(v.sum()),
                    any_endpoint_matches=int((p | v).sum()),
                )
            )
    pd.DataFrame(audit).to_csv(a.out / "ncaa_phase1_overlap.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edits", type=Path, default=Path("results/pairs/edits.csv"))
    p.add_argument("--external", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/references"))
    run(p.parse_args())
