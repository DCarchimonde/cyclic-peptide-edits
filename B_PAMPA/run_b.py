"""Run the PAMPA paper project. quick replays the archived head and independently rescores all methods."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT
PAIRS = SOURCE / "results/pairs/edits.csv"
PAIR_SHA = "c35bc14c0ae71ac2f73a2f4cc51a39d776e87b53546e19bea706647fdcd6f735"


def run(script, *args):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    subprocess.run([sys.executable, "-X", "utf8", "-u", str(script), *map(str, args)],
                   cwd=SOURCE, env=env, check=True)


@contextmanager
def cache_lock(cache):
    """Share weights and feature files across successive runs without simultaneous writes."""
    cache.mkdir(exist_ok=True)
    with (cache / "active.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SystemExit("Another B pairs/train/series job is using the same cache. Let it finish and run again.") from exc
        yield


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", default="quick",choices=["quick","pairs","baseline","train","series","conformation"])
    parser.add_argument("--device", choices=["cpu","cuda"], default="cpu",
                        help="Use cuda only for encoder inference in train/series")
    parser.add_argument("--threads",type=int,default=2)
    a=parser.parse_args()
    if a.device!="cpu" and a.task not in ("train","series"):
        parser.error("--device cuda applies to the two pretrained encoders in train or series")
    if a.threads<1:
        parser.error("--threads must be positive")
    guard = cache_lock(PROJECT / "cache") if a.task in ("pairs", "train", "series") else nullcontext()
    with guard:
        dispatch(a)


def dispatch(a):
    print(f"B | PAMPA site selection | {sys.executable}",flush=True)
    dest=PROJECT/"runs"
    dest.mkdir(exist_ok=True)
    out=Path(mkdtemp(prefix=datetime.now().strftime(a.task+"_%Y%m%d_%H%M%S_"),dir=dest))
    cache=PROJECT/"cache"
    scripts=SOURCE/"scripts"
    graph=cache/"pair_graph.json"
    raw=SOURCE/"data/raw/cycpeptmpdb_all.csv"
    pampa=SOURCE/"data/processed/pampa.csv"
    print(f"New output: {out}",flush=True)
    if a.task in ("pairs", "train", "series"):
        run(scripts / "fetch_data.py", "--only", "cycpeptmpdb_all")
    elif a.task == "conformation":
        run(scripts / "fetch_data.py", "--only", "cycpeptmpdb_4d")
    if a.task=="quick":
        run(scripts/"replay_site_choice.py","--out",out/"kelly.csv")
        run(scripts/"check_round.py","--check-atoms","--out",out/"rescore.json")
        values=json.loads((out/"rescore.json").read_text(encoding="utf-8"))
        print(f"B_QUICK_OK edits={values['molecular_edits_checked']} test_options={values['shared_test_options']} methods={len(values['models'])}")
    elif a.task=="pairs":
        run(scripts/"pair_data.py","--raw",raw,"--index",graph,"--output",out/"pairs")
        with (out/"pairs/edits.csv").open("rb") as stream:
            digest=hashlib.file_digest(stream,"sha256").hexdigest()
        if digest!=PAIR_SHA:
            raise ValueError(f"Compare newly generated edits with the frozen table. SHA256={digest}")
        print(f"B_PAIRS_OK  frozen SHA256={digest}")
    elif a.task=="baseline":
        run(scripts/"run_baseline.py","--data",pampa,"--output",out/"baseline","--seed","2026","--trees","200","--jobs",a.threads)
        run(scripts/"audit_legacy.py","--refit",out/"baseline","--output",out/"audit")
        v=json.loads((out/"audit/audit.json").read_text(encoding="utf-8"))["refit_rescore"]
        print(f"B_BASELINE_OK  ExtraTrees MAE={v['mae']:.4f}  train-mean MAE={v['training_mean_baseline_mae']:.4f}")
    elif a.task=="conformation":
        run(scripts/"conform_ablation.py","--output",out/"conformation")
        print("B_CONFORMATION_OK")
    else:
        run(PROJECT/"encode_for_training.py","--device",a.device,"--threads",a.threads,
            "--work",cache,"--out",out)
        meta=json.loads((out/"encoder_run.json").read_text(encoding="utf-8"))
        options=["--edits",PAIRS,"--cache",meta["feature_cache"],"--weights",cache/"weights","--threads",a.threads]
        if a.task=="train":
            run(scripts/"fit_edits.py",*options,"--out",out/"model")
            with (out/"model/metrics.csv").open(encoding="utf-8",newline="") as f:
                result=next(r for r in csv.DictReader(f) if r["split"]=="test" and r["model"]=="explicit_edit")
            print(f"B_TRAIN_OK  rank_parents={result['rank_parents']}  deltas={result['deltas']}  regret={result['regret']}")
        else:
            run(scripts/"series_sensitivity.py",*options,"--work",out/"fitted_folds","--out",out/"series")
            print("B_SERIES_OK")


if __name__=="__main__":
    main()
