"""Rescore six archived TREM2 complexes from the Cho reproduction."""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "source"
SCRIPTS = SOURCE / "scripts"
REC = SOURCE / "outputs/prepared/6y6c.pdb"


def python(script, *args):
    subprocess.run([sys.executable, "-X", "utf8", "-u", str(SCRIPTS / script), *map(str, args)],
                   cwd=SOURCE, check=True)


def rescore():
    python("validate_results.py", SOURCE)
    parent = PROJECT / "runs"
    parent.mkdir(exist_ok=True)
    dst = Path(mkdtemp(prefix=datetime.now().strftime("rescore_%Y%m%d_%H%M%S_"), dir=parent))
    for label in ("default50", "short15", "known_TP1", "known_TP2", "known_TP3", "known_TP4"):
        saved = SOURCE / "outputs" / label
        result = dst / label
        result.mkdir()
        if label.startswith("known_"):
            python("score.py", result, saved / "fold", "--known-receptor", REC)
        else:
            shutil.copyfile(saved / "6y6c_0.pdb", result / "6y6c_0.pdb")
            python("score.py", result, saved / "fold")
        new = json.loads((result / "verified_scores.json").read_text(encoding="utf-8"))
        old = json.loads((saved / "verified_scores.json").read_text(encoding="utf-8"))
        for metric in ("ipTM", "normalized_ipAE", "fold_terminal_C_N_A"):
            if not math.isclose(new[metric], old[metric], rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"{label}: {metric} differs from the archived calculation")
        print(f"{label:10}  ipTM={new['ipTM']:.4f}  normalized ipAE={new['normalized_ipAE']:.5f}")
    python("summarize_runs.py", dst, SOURCE / "outputs/published/figure2_and_esi_transcription.csv")
    with (dst / "current_inference_summary.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    passed = sum(row["passes_gate"] == "True" for row in rows)
    print(dst, flush=True)
    print(f"R0_RESCORE_OK complexes={len(rows)} passed={passed}/{len(rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["rescore"], nargs="?", default="rescore")
    parser.parse_args()
    print(f"R0 | Cho reproduction | {sys.executable}", flush=True)
    rescore()
