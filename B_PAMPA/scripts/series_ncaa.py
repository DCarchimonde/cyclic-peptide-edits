"""Fit the ncAA-RepDistill native absolute readout on each locked inner fold and score its outer series."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from refs_assets import NCAA, NCAA_SHA, sha
from refs_ncaa import fit
from series_reference_common import tables, assign, paired_summary, baseline_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--external", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--pack", type=Path)
    p.add_argument("--edits", type=Path, default=Path("results/pairs/edits.csv"))
    p.add_argument(
        "--support", type=Path, default=Path("results/model/common_support.csv")
    )
    p.add_argument(
        "--fold-file",
        type=Path,
        default=Path("results/model/series_sensitivity/folds.csv"),
    )
    p.add_argument(
        "--base-predictions",
        type=Path,
        default=Path("results/model/series_sensitivity/out_of_series_predictions.csv"),
    )
    p.add_argument(
        "--overlap", type=Path, default=Path("results/references/overlap_by_edit.csv")
    )
    p.add_argument("--out", type=Path, default=Path("results/references_series/ncaa"))
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    e, folds, sup = tables(a)
    smiles = sorted(set(e.parent_smiles) | set(e.variant_smiles))
    digests = [hashlib.sha256(s.encode()).hexdigest() for s in smiles]
    if a.pack:
        with np.load(a.pack, allow_pickle=False) as z:
            if (
                str(z["checkpoint"]) != NCAA_SHA
                or str(z["support"]) != sup
                or z["smiles_sha256"].tolist() != [h.encode() for h in digests]
            ):
                raise ValueError(
                    "native feature pack has different identities, checkpoint or support"
                )
            mat = z["vectors"]
            if hashlib.sha256(mat.tobytes()).hexdigest() != str(z["vectors_sha256"]):
                raise ValueError("feature pack hash mismatch")
    else:
        if a.cache is None:
            p.error("--pack or --cache is required")
        mat = np.stack(
            [np.load(a.cache / (h + ".npy"), allow_pickle=False) for h in digests]
        )
    if (
        mat.shape != (len(smiles), 768)
        or mat.dtype != np.float32
        or not np.isfinite(mat).all()
    ):
        raise ValueError("invalid official CLS features")
    feat = dict(zip(smiles, mat))
    src = a.external / "ncAA-RepDistill"
    sys.path.insert(0, str(src.resolve()))
    (a.out / "folds").mkdir(parents=True, exist_ok=True)
    allp = []
    runs = []
    method = "ncAA_RepDistill_matched_readout"
    for outer in sorted(e.series_id.unique()):
        f = assign(e, folds, outer)
        dest = a.out / "folds" / outer
        dest.mkdir(exist_ok=True)
        fit(f, feat, src, dest, a.edits, sup)
        read = pd.read_csv(dest / "ncaa_predictions.csv")
        pred = read[read.split == "test"].copy()
        if set(pred.pair_id) != set(f.loc[f.series_id == outer, "pair_id"]):
            raise ValueError("incorrect outer predictions")
        info = json.loads((dest / "ncaa_run.json").read_text())
        runs.append(dict(outer=outer, **info))
        pred[["pair_id", "model", "pred_delta"]].to_csv(
            dest / "test_predictions.csv", index=False
        )
        # Retain only outer predictions in this report. All epoch histories, weights and scalers are retained.
        (dest / "ncaa_predictions.csv").unlink()
        allp.append(pred[["pair_id", "model", "pred_delta"]])
        print("finished ncAA outer", outer, info["best_epoch"], flush=True)
    joined = pd.concat([baseline_rows(a), *allp], ignore_index=True)
    paired_summary(e, joined, method, a.out, a.overlap)
    pd.DataFrame(runs).to_csv(a.out / "fitted_folds.csv", index=False)
    info = dict(
        source_commit=NCAA,
        checkpoint_sha256=NCAA_SHA,
        features_sha256=hashlib.sha256(mat.tobytes()).hexdigest(),
        edits_sha256=sha(a.edits),
        support_sha256=sup,
        split_file_sha256=sha(a.fold_file),
        structures=len(smiles),
        outer_series=len(allp),
        opts=len(e),
        threads=a.threads,
        supervised_upstream_head=False,
        readout="native 256-ReLU readout from build_mlp; validation RMSE, seed 42",
        exposure="Phase2 unlabeled structural adaptation checked at both endpoints, whole-group removal for the clean subset. Prior teachers have incompletely enumerated training chemistry.",
    )
    (a.out / "run.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
