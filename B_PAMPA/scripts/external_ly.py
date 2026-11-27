"""Evaluate frozen PAMPA heads on the positional panels of Ly et al. (2024).

Sources: doi:10.1021/acs.jmedchem.3c02162, Tables 3 and 4 and the authors'
SMILE data CSV. Read structures only. Experimental endpoints are kept
outside the input to inference.
"""

import argparse
import csv
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

import pandas as pd
from rdkit import Chem
import torch

from audit_pair_series import FP, neighbors
from fetch_edit_encoders import verify
from fit_edits import HEADS, eligible as encoder_eligible, molecule_bank, predict_frozen_head
from pair_chemistry import ROOT, canonical, edits_for, eligible, identity, write_csv


DOI = "10.1021/acs.jmedchem.3c02162"
PMC_VERSION = "PMC10946398.1"
PMC_ROOT = "https://pmc-oa-opendata.s3.amazonaws.com/" + PMC_VERSION + "/"
SOURCE = ROOT / "data" / "raw" / "ly2024_smiles.csv"
PAIRS = ROOT / "results" / "pairs" / "edits.csv"
PANELS = {
    "14a": ("14c", "14d", "14e"),
    "18a": ("18c", "18d", "18e"),
    "19a": ("19c", "19d", "19e"),
}
CHECKPOINT_BLOBS = {
    "chem_only": "39e090f2bd6a130fd33cd07e069a30e4a81bf7c3",
    "chem_site": "9f9ea37183588cd82725614b851ef9fd1d17d4eb",
    "explicit_edit": "d83dc09725d37f00255af0056d614e6d3b5ae68a",
    "plain_fusion": "f33cb67df3958a51e329e9581776f373e3e064d5"
}
IDS = set(PANELS).union(*PANELS.values())
ID_PATTERN = re.compile(r"(?<![a-z0-9])((?:14|18|19)\s*[acde])(?![a-z0-9])", re.I)


def download_supplement():
    """Resolve the original CSV through NLM's published article manifest."""
    with urlopen(PMC_ROOT + PMC_VERSION + ".json", timeout=30) as response:
        record = json.load(response)
    if str(record.get("doi", "")).lower() != DOI:
        raise ValueError("The publication DOI does not match.")
    links = [urlsplit(item if isinstance(item, str) else item["url"]) for item in record["media_urls"]]
    csv_files = [link for link in links if link.path.lower().endswith(".csv")]
    if len(csv_files) != 1:
        raise ValueError("The article manifest does not identify one supplementary CSV.")
    link = csv_files[0]
    source_url = PMC_ROOT + link.path.rsplit("/", 1)[1]
    with urlopen(source_url, timeout=60) as response:
        data = response.read()
    checksum = parse_qs(link.query).get("md5", [])
    if checksum and hashlib.md5(data).hexdigest() != checksum[0].lower():
        raise ValueError("The original CSV does not match NLM's published checksum.")
    return data, source_url


def structure_rows(raw):
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        content = raw.decode("utf-16")
    else:
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            content = raw.decode("cp1252")
    if content.lstrip().startswith("<"):
        raise ValueError("The input is a web page, not the authors' SMILE data CSV.")
    try:
        delimiter = csv.Sniffer().sniff(content[:8192], delimiters=",;\t").delimiter
    except csv.Error:
        delimiter = max(",;\t", key=content.splitlines()[0].count) if content.strip() else ","
    structures = {}
    smiles_columns = []
    for row in csv.reader(io.StringIO(content), delimiter=delimiter):
        headers = [i for i, cell in enumerate(row) if "smile" in cell.casefold()]
        if headers and all("[" not in row[i] for i in headers):
            smiles_columns = headers
            continue
        labels = set()
        for i, cell in enumerate(row):
            if i not in smiles_columns and len(cell) < 65:
                labels.update(match.replace(" ", "").lower() for match in ID_PATTERN.findall(cell.strip()))
        labels &= IDS
        if not labels:
            continue
        if len(labels) != 1:
            raise ValueError(f"Ambiguous compound ID: {sorted(labels)}")
        compound = labels.pop()
        values = [row[i] for i in smiles_columns if i < len(row)] if smiles_columns else row
        graphs = set()
        for cell in values:
            token = cell.strip()
            if len(token) > 20 and any(symbol in token for symbol in "[=("):
                mol = Chem.MolFromSmiles(token)
                if mol is not None:
                    graphs.add(canonical(mol))
        if len(graphs) != 1:
            raise ValueError(f"Expected one unambiguous structure for {compound} in the original CSV.")
        graph = graphs.pop()
        if compound in structures and structures[compound] != graph:
            raise ValueError(f"Conflicting structures for {compound}.")
        structures[compound] = graph
    missing = sorted(IDS - structures.keys())
    if missing:
        raise ValueError(f"Missing compound IDs {missing}. Check the original SMILE data CSV.")
    return structures


def read_source():
    if SOURCE.is_file():
        raw = SOURCE.read_bytes()
        return structure_rows(raw), hashlib.sha256(raw).hexdigest(), str(SOURCE)
    try:
        raw, url = download_supplement()
        structures = structure_rows(raw)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"{exc}\nDownload Supporting Information > SMILE data (CSV) from "
            f"https://doi.org/{DOI} and save the original as {SOURCE}, then run again."
        ) from exc
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SOURCE.with_suffix(".part")
    tmp.write_bytes(raw)
    tmp.replace(SOURCE)
    return structures, hashlib.sha256(raw).hexdigest(), url


@lru_cache(maxsize=None)
def fingerprint(smi):
    return FP.GetFingerprint(Chem.MolFromSmiles(smi))


def audit_overlap(structures, comparison_smiles, out):
    """Use the existing whole-molecule and de-NMe family-merge criteria."""
    raw = pd.read_csv(PAIRS, usecols=["source", "parent_smiles", "variant_smiles", "backbone_id"])
    stripped = {}
    for row in raw.drop_duplicates("backbone_id").itertuples():
        backbone = edits_for(row.variant_smiles)[1]
        if identity(backbone, "base") != row.backbone_id:
            raise ValueError(f"Frozen backbone mismatch: {row.backbone_id}")
        stripped[row.backbone_id] = backbone
    queries = {"molecule": comparison_smiles,
               "backbone": {edits_for(s)[1] for s in comparison_smiles}}
    rows = []
    for series, group in raw.groupby("source"):
        refs = {"molecule": set(group.parent_smiles) | set(group.variant_smiles),
                "backbone": {stripped[k] for k in group.backbone_id}}
        for kind, query in queries.items():
            target = refs[kind]
            maximum = neighbors([fingerprint(s) for s in query], [fingerprint(s) for s in target])
            shared = len(query & target)
            rows.append(dict(source=series, representation=kind, exact_shared=shared,
                             max_tanimoto=maximum, linked=bool(shared or maximum >= 0.80)))
    write_csv(out / "frozen_overlap.csv", rows)
    # The full PAMPA table also contains structures outside the eight paired series.
    snapshot = pd.read_csv(ROOT / "data/processed/pampa.csv", usecols=["smiles"])
    observed = {canonical(Chem.MolFromSmiles(s)) for s in snapshot.smiles.unique()}
    seen = sorted(k for k, s in structures.items() if s in observed)
    return sorted({r["source"] for r in rows if r["linked"]}), seen


def checkpoint_digests(folder):
    digests = {}
    for name in HEADS:
        content = (folder / (name + ".pt")).read_bytes()
        header = f"blob {len(content)}\0".encode("ascii")
        if hashlib.sha1(header + content).hexdigest() != CHECKPOINT_BLOBS[name]:
            raise ValueError(f"{name} differs from its frozen checkpoint in 19fe374.")
        digests[name] = hashlib.sha256(content).hexdigest()
    return digests


def score_panels(edges, out, threads):
    """Score complete structure-only candidate panels with the four frozen heads."""
    frozen = ROOT / "results/model/fitted"
    checkpoint_hashes = checkpoint_digests(frozen)
    preds = []
    exclusions = []
    if edges:
        weights = ROOT / "cache/weights"
        verify(weights, fetch=True)
        frame = pd.DataFrame(edges)
        bank = molecule_bank(frame, weights, out / "features", threads)
        supported, rejected = encoder_eligible(frame, bank, complete=True)
        exclusions = rejected.to_dict("records")
        if not supported.empty:
            torch.set_num_threads(threads)
            for name in HEADS:
                estimates = predict_frozen_head(supported, bank, frozen / (name + ".pt"))
                for row, estimate in zip(supported.itertuples(), estimates):
                    preds.append(dict(parent=row.group_id, candidate=row.pair_id, model=name,
                                      predicted_delta=float(estimate)))
    write_csv(out / "encoder_exclusions.csv", exclusions, ["pair_id", "group_id", "reason"])
    write_csv(out / "predictions.csv", preds, ["parent", "candidate", "model", "predicted_delta"])
    scored = sorted({r["parent"] for r in preds})
    return scored, checkpoint_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    structures, digest, source = read_source()
    audit = []
    edges = []
    valid_structures = {s for s in structures.values() if not eligible(Chem.MolFromSmiles(s))}
    for parent, children in PANELS.items():
        p = structures[parent]
        options = []
        distinct = set()
        for child in children:
            v = structures[child]
            status = eligible(Chem.MolFromSmiles(p)) or eligible(Chem.MolFromSmiles(v))
            if not status:
                mapping = edits_for(v)[0].get(p)
                if mapping is None:
                    status = "outside_single_backbone_n_methyl_edit"
                elif v in distinct:
                    status = "equivalent_option"
                else:
                    distinct.add(v)
                    options.append(dict(pair_id=child, group_id=parent, parent_smiles=p, variant_smiles=v,
                                        parent_n_idx=mapping[0], variant_n_idx=mapping[1]))
                    status = "eligible_edit"
            audit.append(dict(parent=parent, candidate=child, structure_check=status))
        incomplete = any(row["structure_check"] not in
                         {"eligible_edit", "equivalent_option", "outside_single_backbone_n_methyl_edit"}
                         for row in audit if row["parent"] == parent)
        if len(options) >= 2 and not incomplete:
            edges.extend(options)
    write_csv(out / "candidate_audit.csv", audit)
    linked, seen = audit_overlap(structures, valid_structures, out)
    eligible_groups = sorted({e["group_id"] for e in edges})
    scored, checkpoint_hashes = score_panels(edges, out, args.threads)
    info = dict(doi=DOI, structure_source=source, source_sha256=digest, requested_panels=PANELS,
                structurally_rankable=eligible_groups, scored_panels=scored,
                linked_frozen_series=linked, structures_in_pampa_snapshot=seen,
                unchecked_structures=sorted(k for k, s in structures.items() if s not in valid_structures),
                checkpoint_sha256=checkpoint_hashes)
    (out / "summary.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"B_EXTERNAL_OK  rankable={len(eligible_groups)}/{len(PANELS)}  scored={len(scored)}"
          f"  linked_series={len(linked)}  snapshot_matches={len(seen)}", flush=True)


if __name__ == "__main__":
    main()
