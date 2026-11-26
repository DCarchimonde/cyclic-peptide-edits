"""Use the previously frozen chemical-series folds and pair all comparisons on the same groups."""

import numpy as np
import pandas as pd
from refs_assets import benchmark
from edit_metrics import summarize


def tables(args):
    edits, checksum = benchmark(args.edits, args.support)
    folds = pd.read_csv(args.fold_file)
    if folds.duplicated(["outer", "series_id"]).any():
        raise ValueError("duplicate outer assignment")
    for outer, t in folds.groupby("outer"):
        if set(t["series_id"]) != set(edits.series_id) or set(
            t.loc[t.role == "test", "series_id"]
        ) != {outer}:
            raise ValueError("outer fold failed disjoint series check")
    return edits, folds, checksum


def assign(edits, folds, outer):
    roles = folds[folds.outer == outer].set_index("series_id").role
    e = edits.copy()
    e["split"] = e.series_id.map(roles)
    if e.split.isna().any() or e.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("incomplete or split group")
    endpoints = pd.concat(
        [
            e[["parent_smiles", "split"]].rename(columns={"parent_smiles": "s"}),
            e[["variant_smiles", "split"]].rename(columns={"variant_smiles": "s"}),
        ]
    )
    if endpoints.groupby("s").split.nunique().max() != 1:
        raise ValueError("molecule crosses fold roles")
    return e


def phase2_unseen_ids(e, overlap, out):
    fields = [
        "parent_ncaa_p2_train",
        "parent_ncaa_p2_val",
        "variant_ncaa_p2_train",
        "variant_ncaa_p2_val",
    ]
    source = pd.read_csv(overlap)
    audit = e[["pair_id", "group_id", "series_id"]].merge(
        source[["pair_id", *fields]], on="pair_id", validate="one_to_one"
    )
    if len(audit) != len(e):
        raise ValueError("incomplete Phase2 membership audit")
    audit["either_endpoint_seen_phase2"] = audit[fields].any(axis=1)
    audit["entire_group_unseen_phase2"] = ~audit.groupby(
        "group_id"
    ).either_endpoint_seen_phase2.transform("any")
    audit.to_csv(out / "phase2_membership.csv", index=False)
    series = []
    for s, g in audit.groupby("series_id"):
        clean = set(g[g.entire_group_unseen_phase2].pair_id)
        ge = e[e.pair_id.isin(clean)]
        series.append(
            dict(
                series_id=s,
                all_options=len(g),
                either_endpoint_seen=int(g.either_endpoint_seen_phase2.sum()),
                fully_unseen_group_options=len(ge),
                fully_unseen_groups=int(ge.group_id.nunique()),
                fully_unseen_rank_groups=int(
                    (ge.groupby("group_id").size() >= 2).sum()
                ),
                fully_unseen_deltas=int(ge.delta.notna().sum()),
            )
        )
    pd.DataFrame(series).to_csv(out / "phase2_exposure_by_series.csv", index=False)
    return set(audit.loc[audit.entire_group_unseen_phase2, "pair_id"])


def paired_summary(e, forecasts, new_method, out, overlap):
    """Ref-minus-baseline differences with explicit sign and kept complete measurement groups."""
    out.mkdir(parents=True, exist_ok=True)
    if forecasts.duplicated(["pair_id", "model"]).any():
        raise ValueError("duplicated method prediction")
    observed = set(e.pair_id)
    if any(set(g.pair_id) != observed for _, g in forecasts.groupby("model")):
        raise ValueError("reference and baseline do not share the exact same options")
    e = e.copy()
    e["split"] = "leave_one_series_out"
    forecasts.to_csv(out / "matched_outer_predictions.csv", index=False)
    cuts = {"all": observed}
    if new_method.startswith("ncAA"):
        cuts["phase2_unseen_complete_groups"] = phase2_unseen_ids(e, overlap, out)
    for label, ids in cuts.items():
        if not ids:
            continue
        directory = out / label
        summarize(
            e[e.pair_id.isin(ids)], forecasts[forecasts.pair_id.isin(ids)], directory
        )
        by = pd.read_csv(directory / "per_series.csv")
        ref = by[by.model == new_method]
        rows = []
        for m in sorted(set(by.model) - {new_method}):
            joint = ref.merge(
                by[by.model == m], on="series_id", suffixes=("_ref", "_base")
            )
            for k in ["regret", "hit", "delta_mae", "direction_accuracy"]:
                a = joint[(joint[k + "_n_ref"] > 0) & (joint[k + "_n_base"] > 0)]
                if a.empty:
                    continue
                n = a[k + "_n_ref"].to_numpy()
                if not np.array_equal(n, a[k + "_n_base"].to_numpy()):
                    raise ValueError("unmatched denominator")
                diff = a[k + "_sum_ref"].to_numpy() - a[k + "_sum_base"].to_numpy()
                lo = hi = np.nan
                if len(a) >= 3:
                    idx = np.random.default_rng(2026).integers(
                        0, len(a), size=(2000, len(a))
                    )
                    lo, hi = np.quantile(
                        diff[idx].sum(axis=1) / n[idx].sum(axis=1), [0.025, 0.975]
                    )
                rows.append(
                    dict(
                        reference=new_method,
                        baseline=m,
                        metric=k,
                        series=len(a),
                        denominator=int(n.sum()),
                        reference_minus_baseline=diff.sum() / n.sum(),
                        lower_95=lo,
                        upper_95=hi,
                    )
                )
        pd.DataFrame(rows).to_csv(
            directory / "reference_minus_baselines.csv", index=False
        )


def baseline_rows(args):
    methods = {
        "explicit_edit",
        "random_site_zero_delta",
        "train_median_delta",
        "extratrees_abs_diff",
        "chem_only",
    }
    data = pd.read_csv(args.base_predictions)
    return data[data.model.isin(methods)][["pair_id", "model", "pred_delta"]].copy()
