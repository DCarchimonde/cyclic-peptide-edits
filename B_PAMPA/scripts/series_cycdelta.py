"""Retrain the native CycDelta pair network in each frozen series fold from cached Uni-Mol features."""

import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import rdBase
from refs_assets import CYC, UM_SHA, sha
from refs_cycdelta import obtain, make_graphs, normalizer, train, scores
from series_reference_common import tables, assign, baseline_rows, paired_summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--external", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
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
    p.add_argument(
        "--out", type=Path, default=Path("results/references_series/cycdelta")
    )
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    np.random.seed(0)
    src = (a.external / "CycDelta").resolve()
    sys.path[:0] = [str(src / "smiles_delta_prediction"), str(src)]
    os.environ.setdefault("UNIMOL_WEIGHT_DIR", str((a.external / "unimol").resolve()))
    import predict as up

    e, folds, sup = tables(a)
    if rdBase.rdkitVersion != "2026.03.4":
        raise ValueError("CycDelta parser/descriptor environment must match the cache")
    expected = "monomers_ee7d5a83731f0cb832c3.npz"
    if not (a.cache / expected).is_file():
        raise ValueError("existing fixed-feature native cache is required for this run")
    all_records, i, h, d, gly, fail = obtain(e, up, a.cache)
    if fail:
        raise ValueError("a common group lacks an external graph")
    allp = []
    runs = []
    method = "CycDelta_matched_refit"
    for outer in sorted(e.series_id.unique()):
        f = assign(e, folds, outer)
        dest = a.out / "folds" / outer
        dest.mkdir(parents=True, exist_ok=True)
        tr = f[(f.split == "train") & f.delta.notna()]
        measured = sorted(set(tr.parent_smiles) | set(tr.variant_smiles))
        norm = normalizer(torch.cat([d[i[s]] for s in measured]))
        np.savez_compressed(
            dest / "train_normalizer.npz", mean=norm[0].numpy(), std=norm[1].numpy()
        )
        graphs = make_graphs(all_records, i, h, d, gly, norm, up)
        model, info = train(f, graphs, up, dest)
        test = f[f.split == "test"]
        pred = pd.DataFrame(
            dict(
                pair_id=test.pair_id,
                model=method,
                pred_delta=scores(model, test, graphs, up),
            )
        )
        pred.to_csv(dest / "test_predictions.csv", index=False)
        allp.append(pred)
        runs.append(dict(outer=outer, **info))
        print("completed CycDelta outer", outer, info, flush=True)
    joined = pd.concat([baseline_rows(a), *allp], ignore_index=True)
    paired_summary(e, joined, method, a.out, a.overlap)
    pd.DataFrame(runs).to_csv(a.out / "fitted_folds.csv", index=False)
    info = dict(
        source_commit=CYC,
        unimol_weight_sha256=UM_SHA,
        cached_monomers_sha256=sha(a.cache / expected),
        shared_support_sha256=sup,
        edits_sha256=sha(a.edits),
        folds_sha256=sha(a.fold_file),
        compound_count=len(all_records),
        source_supervised_weights_used=False,
        child_parent="same fixed chemical parent, including unmeasured/virtual parents",
        normalization="training structures only, 31 graph descriptors, native zero correction",
        rdkit=rdBase.rdkitVersion,
        evaluation="rank on all common options; delta only when parent measured; chemically disjoint outer fold",
    )
    (a.out / "run.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
