"""Nested leave-one-series-out evaluation with a separate worker for each series."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--edits", type=Path, default=Path("results/pairs/edits.csv"))
    a.add_argument("--weights", type=Path, required=True)
    a.add_argument("--cache", type=Path, default=Path(".cache/encoders"))
    a.add_argument("--out", type=Path, default=Path("results/model/series_sensitivity"))
    a.add_argument("--work", type=Path, default=Path(".cache/series_fits"))
    a.add_argument("--threads", type=int, default=2)
    a.add_argument("--outer")
    o = a.parse_args()
    from edit_metrics import summarize

    data = pd.read_csv(o.edits)
    o.out.mkdir(exist_ok=True, parents=True)
    import hashlib

    digest = hashlib.sha256(
        o.edits.read_bytes() + (Path(__file__).parent / "fit_edits.py").read_bytes()
    ).hexdigest()[:16]
    work = o.work / digest
    work.mkdir(parents=True, exist_ok=True)
    if o.outer:
        import torch
        from fit_edits import (
            HEADS,
            fit_absolute,
            fit_head,
            molecule_bank,
            eligible,
            integrity,
        )

        torch.set_num_threads(o.threads)
        bank = molecule_bank(data, o.weights, o.cache, o.threads)
        f, _ = eligible(data, bank)
        terms = []
        for s, g in f.groupby("series_id"):
            if (g.groupby("group_id").size() >= 2).any() and g.delta.notna().any():
                terms.append((len(g), s))
        val = [series for _, series in sorted(terms) if series != o.outer][:2]
        f = f.copy()
        f["split"] = np.select(
            [f.series_id == o.outer, f.series_id.isin(val)],
            ["test", "valid"],
            default="train",
        )
        integrity(f)
        test = (f.split == "test").to_numpy()
        loc = work / o.outer
        loc.mkdir(exist_ok=True)
        f[["series_id", "split"]].drop_duplicates().assign(outer=o.outer).rename(
            columns={"split": "role"}
        ).to_csv(loc / "roles.csv", index=False)
        for stage in ["absolute", *HEADS]:
            dest = loc / (stage + ".csv")
            if dest.exists():
                continue
            with TemporaryDirectory() as temp:
                if stage == "absolute":
                    y, _, ep, _ = fit_absolute(f, bank, Path(temp))
                    epochs = {"v01_predict_pair": ep}
                    y["random_site_zero_delta"] = np.zeros(len(f))
                    y["train_median_delta"] = np.full(
                        len(f), float(f[f.split == "train"].delta.median())
                    )
                else:
                    yy, _, ep, _ = fit_head(stage, f, bank, Path(temp))
                    y = {stage: yy}
                    epochs = {stage: ep}
            rows = pd.concat(
                [
                    pd.DataFrame(
                        dict(pair_id=f.pair_id[test], model=k, pred_delta=x[test])
                    )
                    for k, x in y.items()
                ]
            )
            pd.DataFrame(
                [
                    dict(outer=o.outer, model=k, selected_epoch=v)
                    for k, v in epochs.items()
                ]
            ).to_csv(loc / (stage + "_epochs.csv"), index=False)
            tmp = dest.with_suffix(".tmp")
            rows.to_csv(tmp, index=False)
            tmp.replace(dest)
            del y, rows
            import gc

            gc.collect()
            if sys.platform.startswith("linux"):
                import ctypes

                ctypes.CDLL(None).malloc_trim(0)
        return
    from edit_metrics import summarize

    preds = []
    epochs = []
    folds = []
    for outer in sorted(data.series_id.unique()):
        cmd = [
            sys.executable,
            __file__,
            "--edits",
            str(o.edits),
            "--weights",
            str(o.weights),
            "--cache",
            str(o.cache),
            "--out",
            str(o.out),
            "--work",
            str(o.work),
            "--threads",
            str(o.threads),
            "--outer",
            outer,
        ]
        for attempt in range(3):
            r = subprocess.run(cmd, env=dict(os.environ, MALLOC_ARENA_MAX="2"))
            if r.returncode == 0:
                break
        r.check_returncode()
        d = work / outer
        folds.append(pd.read_csv(d / "roles.csv"))
        for stage in [
            "absolute",
            "chem_only",
            "plain_fusion",
            "chem_site",
            "explicit_edit",
        ]:
            preds.append(pd.read_csv(d / (stage + ".csv")))
            epochs.append(pd.read_csv(d / (stage + "_epochs.csv")))
        print("Completed outer series", outer, flush=True)
    p = pd.concat(preds)
    p.to_csv(o.out / "out_of_series_predictions.csv", index=False)
    pd.concat(epochs).to_csv(o.out / "selected_epochs.csv", index=False)
    pd.concat(folds).to_csv(o.out / "folds.csv", index=False)
    f = data[data.pair_id.isin(p.pair_id)].copy()
    f["split"] = "leave_one_series_out"
    summarize(f, p, o.out)


if __name__ == "__main__":
    main()
