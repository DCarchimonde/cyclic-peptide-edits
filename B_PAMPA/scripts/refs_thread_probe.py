"""Compare two one-thread CycDelta fits to the delivered two-thread fit on identical cases."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from refs_assets import sha
from refs_report import evaluate


def main(a):
    edits = pd.read_csv(a.edits)
    test = edits.loc[edits.split.eq("test")]
    names = {
        "delivered_two_threads": a.out,
        "one_thread_1": a.single_a,
        "one_thread_2": a.single_b,
    }
    report = {"cohort": "full_common_test", "runs": {}}
    for label, folder in names.items():
        pred = pd.read_csv(folder / "cycdelta_refit_predictions.csv")
        groups, edges = evaluate(test, pred)
        agg = groups.groupby("parent_id")[["regret", "top1"]].mean().mean()
        info = json.loads((folder / "cycdelta_refit_run.json").read_text())
        report["runs"][label] = {
            "rows": len(pred),
            "test_rank_parents": groups.parent_id.nunique(),
            "test_deltas": len(edges),
            "regret": float(agg.regret),
            "top1": float(agg.top1),
            "delta_mae": float(edges.delta_mae.mean()),
            "best_epoch": info["best_epoch"],
            "best_validation_mae": info["best_validation_mae"],
            "refit_prediction_sha256": sha(folder / "cycdelta_refit_predictions.csv"),
            "released_prediction_sha256": sha(
                folder / "cycdelta_released_predictions.csv"
            ),
            "validation_curve_sha256": sha(folder / "cycdelta_learning_curve.csv"),
        }
    r = report["runs"]
    report["one_thread_replication"] = {
        k: r["one_thread_1"][k] == r["one_thread_2"][k]
        for k in (
            "refit_prediction_sha256",
            "released_prediction_sha256",
            "validation_curve_sha256",
            "best_epoch",
        )
    }
    report["controlled_difference"] = (
        "Same host, pinned source, inputs, initial seed, cached monomer vectors and fixed validation criterion. "
        "Command-level changes were torch intra-op thread count and OMP_NUM_THREADS (2 to 1); "
        "OPENBLAS_NUM_THREADS remained 1. This does not identify the old-host drift cause."
    )
    p2 = pd.read_csv(a.out / "cycdelta_refit_predictions.csv")
    ps = pd.read_csv(a.single_a / "cycdelta_refit_predictions.csv")
    paired = p2.merge(
        ps,
        on=["pair_id", "model", "split"],
        validate="one_to_one",
        suffixes=("_two", "_one"),
    )
    report["max_abs_all_predictions_one_vs_two"] = float(
        np.abs(paired.pred_delta_one - paired.pred_delta_two).max()
    )
    ps["model"] = "CycDelta_matched_refit_single_thread_sensitivity"
    ps.to_csv(a.out / "sensitivity_cycdelta_single_thread.csv", index=False)
    pd.read_csv(a.single_a / "cycdelta_learning_curve.csv").to_csv(
        a.out / "sensitivity_cycdelta_single_thread_curve.csv", index=False
    )
    (a.out / "cycdelta_threads_probe.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edits", type=Path, default=Path("results/pairs/edits.csv"))
    p.add_argument("--out", type=Path, default=Path("results/references"))
    p.add_argument("--single-a", type=Path, required=True)
    p.add_argument("--single-b", type=Path, required=True)
    main(p.parse_args())
