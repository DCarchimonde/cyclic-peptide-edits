"""Discover measured single-N-methyl site choices in publications beyond the frozen source list."""
import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, DataStructs

from external_ly import PAIRS, fingerprint, score_panels
from fetch_data import fetch
from pair_chemistry import ROOT, canonical, edits_for, eligible, identity, read_csv, write_csv

REVISION = "0a897997d32c66c1570490d77b15ffbc4abf3410"
FILENAME = "peptide_cycpeptmpdb_CycPeptMPDB_Peptide_Assay_PAMPA.csv.jsonl"
MIRROR = {
    "id": "litefold_pampa", "local_path": "data/raw/litefold_pampa.jsonl",
    "url": f"https://huggingface.co/datasets/LiteFold/CycPepMPDB/resolve/{REVISION}/tables/{FILENAME}",
    "sha256": "af478487cffd11d79d0554662fccff40d4bdba9377d70d9ea6087a46db33db20",
    "max_bytes": 60 * 1024 * 1024,
}
LY = "2024ly"
LINK_THRESHOLD = 0.80
PANEL_COLS = ["source", "group_id", "parent_smiles", "parent_names", "parent_record_ids", "parent_point", "parent_status",
              "options", "point_options", "fully_quantitative", "prior_ly_publication", "old_exact_structures",
              "related_frozen_series", "max_molecule_tanimoto", "max_backbone_tanimoto", "priority_source_check"]
OPTION_COLS = ["source", "group_id", "pair_id", "variant_names", "variant_record_ids", "parent_smiles",
               "variant_smiles", "parent_n_idx", "variant_n_idx", "pampa_point", "point_status", "raw_pampa",
               "limit_1", "limit_2", "transmission"]


def source_key(label):
    return re.sub(r"[^a-z0-9]", "", label.lower())


@lru_cache(maxsize=None)
def as_graph(text):
    mol = Chem.MolFromSmiles(text) if text else None
    return canonical(mol) if mol is not None else ""


@lru_cache(maxsize=None)
def reversible(smiles):
    return edits_for(smiles)


def numeric(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def cell_point(rows):
    if not rows:
        return None, "no_parent_record_in_source"
    pts = set()
    for r in rows:
        y = numeric(r["PAMPA"])
        transmission = str(r.get("T_PAMPA") or "").strip()
        if any(str(r.get(k) or "").strip() for k in ("Detection_Limit_1", "Detection_Limit_2")):
            return None, "reported_assay_limit_review"
        if y is None or not -10 < y < -4:
            return None, "censored_boundary_or_nonquantitative"
        fraction = numeric(transmission)
        if transmission and (fraction is None or fraction <= 0):
            return None, "invalid_transmission"
        pts.add(y)
    if len(pts) != 1:
        return None, "repeated_structure_different_values_review"
    return pts.pop(), "database_point_pending_protocol_check"


def identifiers(rows, field):
    return "|".join(sorted({str(r[field]) for r in rows if r.get(field) is not None and str(r[field]) != ""}))


def snapshot(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    names = {source_key(r["Source"]) for r in rows}
    graphs = {as_graph(r["SMILES"]) for r in rows if r["SMILES"]}
    graphs.discard("")
    return names, graphs


def discovered(path, old_sources):
    by_source = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(Counter)
    required = {"Source", "SMILES", "PAMPA", "Original_Name_in_Source_Literature"}
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            record = json.loads(line)["row"]
            if not required.issubset(record):
                raise ValueError(f"Unexpected CycPeptMPDB columns: {sorted(required - record.keys())}")
            source = (record["Source"] or "").strip()
            if not source:
                raise ValueError("Missing literature source in the pinned PAMPA table")
            if source_key(source) in old_sources:
                continue
            counts[source]["rows"] += 1
            if not str(record.get("PAMPA") or "").strip():
                counts[source]["no_pampa"] += 1
                continue
            graph = as_graph(record["SMILES"])
            reason = eligible(Chem.MolFromSmiles(graph)) if graph else "missing_or_invalid_smiles"
            if reason:
                counts[source][reason] += 1
                continue
            database_id = str(record.get("ID") or record.get("CycPeptMPDB_ID") or "").strip()
            if not database_id:
                raise ValueError("Missing original record ID")
            record["record_id"] = database_id
            by_source[source][graph].append(record)
    return by_source, counts


def frozen_series():
    molecules = defaultdict(set)
    skeletons = defaultdict(dict)
    for r in read_csv(PAIRS):
        source = r["source"]
        molecules[source].update([r["parent_smiles"], r["variant_smiles"]])
        skeletons[source].setdefault(r["backbone_id"], r["variant_smiles"])
    fps = {}
    for source, refs in molecules.items():
        backbones = set()
        for backbone_id, variant in skeletons[source].items():
            graph = reversible(variant)[1]
            if identity(graph, "base") != backbone_id:
                raise ValueError(f"Inconsistent frozen backbone: {backbone_id}")
            backbones.add(graph)
        fps[source] = ([fingerprint(s) for s in refs], [fingerprint(s) for s in backbones])
    return fps


def make_panels(by_source, old_graphs, out):
    fps = frozen_series()

    @lru_cache(maxsize=None)
    def nearest(graph, series, kind):
        return max(DataStructs.BulkTanimotoSimilarity(fingerprint(graph), fps[series][kind]))

    panels, options, model_edges = [], [], []
    for source, structures in sorted(by_source.items()):
        matches = defaultdict(dict)
        for child in structures:
            for parent, mapped in reversible(child)[0].items():
                matches[parent][child] = mapped
        for parent, children in sorted(matches.items()):
            if len(children) < 2:
                continue
            group = identity(source + "|" + parent, "ext_group")
            parent_rows = structures.get(parent, [])
            parent_y, parent_status = cell_point(parent_rows)
            graphs = {parent, *children}
            stripped = {reversible(s)[1] for s in graphs}
            related, max_mol, max_back = [], 0.0, 0.0
            for ser in fps:
                mol_sim = max(nearest(s, ser, 0) for s in graphs)
                back_sim = max(nearest(s, ser, 1) for s in stripped)
                if max(mol_sim, back_sim) >= LINK_THRESHOLD:
                    related.append(ser)
                max_mol, max_back = max(max_mol, mol_sim), max(max_back, back_sim)
            group_opts = []
            for child, (pn, vn, _, _) in sorted(children.items()):
                obs = structures[child]
                y, status = cell_point(obs)
                group_opts.append(dict(source=source, group_id=group, pair_id=identity(group+"|"+child, "ext_pair"),
                                       variant_names=identifiers(obs, "Original_Name_in_Source_Literature"),
                                       variant_record_ids=identifiers(obs, "record_id"), parent_smiles=parent,
                                       variant_smiles=child, parent_n_idx=pn, variant_n_idx=vn,
                                       pampa_point=y, point_status=status, raw_pampa=identifiers(obs, "PAMPA"),
                                       limit_1=identifiers(obs, "Detection_Limit_1"), limit_2=identifiers(obs, "Detection_Limit_2"),
                                       transmission=identifiers(obs, "T_PAMPA")))
            npoint = sum(r["pampa_point"] is not None for r in group_opts)
            seen = len(graphs & old_graphs)
            prior = source_key(source) == LY
            full = npoint == len(group_opts)
            priority = full and not prior and not seen and not related
            panels.append(dict(source=source, group_id=group, parent_smiles=parent,
                               parent_names=identifiers(parent_rows, "Original_Name_in_Source_Literature"),
                               parent_record_ids=identifiers(parent_rows, "record_id"), parent_point=parent_y, parent_status=parent_status,
                               options=len(group_opts), point_options=npoint, fully_quantitative=full,
                               prior_ly_publication=prior, old_exact_structures=seen, related_frozen_series="|".join(sorted(related)),
                               max_molecule_tanimoto=max_mol, max_backbone_tanimoto=max_back, priority_source_check=priority))
            options.extend(group_opts)
            if npoint >= 2:
                fields = ["pair_id", "group_id", "parent_smiles", "variant_smiles", "parent_n_idx", "variant_n_idx"]
                # Score all structural options in each included group. No PAMPA values reach the encoders.
                model_edges.extend({k: row[k] for k in fields} for row in group_opts)
    write_csv(out / "source_panels.csv", panels, PANEL_COLS)
    write_csv(out / "source_options.csv", options, OPTION_COLS)
    return panels, model_edges


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8"))
    frozen = next(r for r in manifest["files"] if r["id"] == "cycpeptmpdb_all")
    for entry in [frozen, MIRROR]:
        try:
            fetch(entry, 90)
        except (OSError, ValueError) as e:
            raise SystemExit(f"Download {entry['url']} to {ROOT / entry['local_path']} then rerun.\n{e}") from e
    old_sources, old_graphs = snapshot(ROOT / frozen["local_path"])
    structures, coverage = discovered(ROOT / MIRROR["local_path"], old_sources)
    panels, edges = make_panels(structures, old_graphs, a.out)
    print(f"Source scan: {len(coverage)} new publication labels, {len(panels)} structural panels", flush=True)
    scored, hashes = score_panels(edges, a.out, a.threads)
    scored = set(scored)
    sources = []
    for name, stat in sorted(coverage.items()):
        group = [r for r in panels if r["source"] == name]
        sources.append(dict(source=name, prior_ly_publication=source_key(name) == LY, rows=stat["rows"],
                            accepted_structures=len(structures.get(name, {})), structural_panels=len(group),
                            two_point_options=sum(r["point_options"] >= 2 for r in group),
                            complete_quantitative=sum(r["fully_quantitative"] for r in group),
                            priority_source_check=sum(r["priority_source_check"] for r in group),
                            scored=sum(r["group_id"] in scored for r in group),
                            excluded_rows={k: v for k, v in stat.items() if k != "rows"}))
    priority = [r for r in panels if r["priority_source_check"]]
    info = dict(frozen_sha256=frozen["sha256"], mirror_revision=REVISION, mirror_sha256=MIRROR["sha256"],
                source_labels_absent_from_frozen=len(sources), structural_panels=len(panels), scored=len(scored),
                priority_source_check=len(priority), by_source=sources, checkpoint_sha256=hashes,
                reviewed_B08_source="2024_Ly", source_protocol_status="pending verification against primary tables",
                overlap_rule="any identical structure in original raw snapshot, or whole/de-NMe Tanimoto >= 0.80 to any frozen paired source")
    (a.out / "summary.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"B_SOURCE_PANELS_OK new_sources={len(sources)} panels={len(panels)} "
          f"scored={len(scored)} priority_review={len(priority)}")


if __name__ == "__main__":
    main()
