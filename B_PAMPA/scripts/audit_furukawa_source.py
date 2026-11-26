"""Reconcile all 688 Furukawa compounds to the numeric and missing cells of the primary SI."""

import argparse, hashlib, json, re, subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.request import Request, urlopen
from pair_chemistry import ROOT, read_csv, write_csv

HASH = "4f7f5364c5178e1aa80f02872c5a5117f857f1d80663997a94c0be6992da9e53"
URLS = [
    "https://ndownloader.figshare.com/files/6434334",
    "https://pstorage-acs-6854636.s3-eu-west-1.amazonaws.com/6434334/jm6b01246_si_001.pdf",
]
DOI = "https://doi.org/10.1021/acs.jmedchem.6b01246.s001"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", type=Path)
    args = ap.parse_args()
    with TemporaryDirectory() as temp:
        source = args.pdf
        if source is None:
            source = Path(temp) / "SI.pdf"
            for url in URLS:
                try:
                    with urlopen(
                        Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=100
                    ) as resp:
                        data = resp.read()
                    if hashlib.sha256(data).hexdigest() == HASH:
                        source.write_bytes(data)
                        break
                except (URLError, TimeoutError):
                    continue
        if (
            not source.exists()
            or hashlib.sha256(source.read_bytes()).hexdigest() != HASH
        ):
            raise ValueError("Unexpected publisher SI; provide --pdf")
        pages = subprocess.check_output(
            ["pdftotext", "-layout", str(source), "-"], text=True
        )
    source_cells = {}
    for line in pages.splitlines():
        m = re.match(r"^\s*([12]\.\d+-\d+(?:Me|H)?)\s+\d{3}\.\d{3}\s+", line)
        if m:
            if m[1] in source_cells:
                raise ValueError("Duplicate source name " + m[1])
            source_cells[m[1]] = line.split()[-1]
    raw = [
        r
        for r in read_csv(ROOT / "data/raw/cycpeptmpdb_all.csv")
        if r["Source"] == "2016_Furukawa" and r["PAMPA"]
    ]
    if {r["Original_Name_in_Source_Literature"] for r in raw} != set(source_cells):
        raise ValueError("Source ID mismatch")
    edge_ids = {
        i
        for r in read_csv(ROOT / "results/pairs/edits.csv")
        for col in ["variant_record_ids", "parent_record_ids"]
        for i in r[col].split("|")
        if i
    }
    checked = []
    for r in raw:
        name = r["Original_Name_in_Source_Literature"]
        token = source_cells[name]
        numeric = re.fullmatch(r"-\d+\.\d+", token) is not None
        if numeric and abs(float(token) - float(r["PAMPA"])) > 0.0051:
            raise ValueError(name)
        if not numeric and (token != "—" or float(r["PAMPA"]) != -10):
            raise ValueError(name)
        rid = "CycPeptMPDB_record:" + r["CycPeptMPDB_ID"]
        checked.append(
            dict(
                record_id=rid,
                name=name,
                db_pampa=r["PAMPA"],
                source_Ave_LogPapp=token if numeric else "",
                source_token=token,
                db_T_PAMPA=r["T_PAMPA"],
                in_frozen_edits=int(rid in edge_ids),
            )
        )
    out = ROOT / "results/pairs/furukawa_source_audit"
    out.mkdir(exist_ok=True)
    write_csv(out / "record_matches.csv", checked)
    report = dict(
        source_doi=DOI,
        source_sha256=HASH,
        record_names_matched=len(checked),
        source_numeric_agreement=sum(bool(r["source_Ave_LogPapp"]) for r in checked),
        source_missing_encoded_minus10=sum(
            not r["source_Ave_LogPapp"] for r in checked
        ),
        excluded_training_candidates=[
            {
                "id": r["record_id"],
                "name": r["name"],
                "SI_value": r["source_Ave_LogPapp"],
            }
            for r in checked
            if r["name"] in {"2.3-03H", "2.4-03H"}
        ],
        interpretation="All eight table dashes are nonquantitative. Two training candidates have zero displayed transmission and finite log Papp in the authors’ numerical column. They were omitted under the predeclared QC filter.",
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
