"""Evaluate common complete candidate groups and inspect exact upstream-exposure sensitivities."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def evaluate(t, forecasts):
    g = []
    e = []
    for m, f in forecasts.groupby("model"):
        test = t.merge(
            f[["pair_id", "pred_delta"]], on="pair_id", validate="one_to_one"
        )
        for group, frame in test.groupby("group_id"):
            if len(frame) < 2:
                continue
            selected = np.isclose(
                frame.pred_delta, frame.pred_delta.max(), atol=1e-7, rtol=0
            )
            hit = np.isclose(frame.variant_y, frame.variant_y.max(), atol=1e-7, rtol=0)
            r = frame.iloc[0]
            g.append(
                dict(
                    model=m,
                    parent_id=r.parent_id,
                    group_id=group,
                    series_id=r.series_id,
                    source=r.source,
                    regret=float(
                        frame.variant_y.max() - frame.loc[selected, "variant_y"].mean()
                    ),
                    top1=float(hit[selected].mean()),
                    selected_variant_ids="|".join(frame.loc[selected, "variant_id"]),
                    best_variant_ids="|".join(frame.loc[hit, "variant_id"]),
                )
            )
        q = test.loc[
            test.delta.notna(),
            ["pair_id", "series_id", "source", "delta", "pred_delta"],
        ].copy()
        q["model"] = m
        q["delta_mae"] = (q.delta - q.pred_delta).abs()
        q["direction"] = np.where(
            q.delta.abs() > 1e-7,
            (np.sign(q.delta) == np.sign(q.pred_delta)).astype(float),
            np.nan,
        )
        e.append(q)
    return pd.DataFrame(g), pd.concat(e, ignore_index=True)


def main(a):
    a.out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(a.edits)
    t = pairs[pairs.split == "test"].copy()
    overlap = pd.read_csv(a.out / "overlap_by_edit.csv")
    t = t.merge(
        overlap.drop(
            columns=["split", "group_id", "series_id", "parent_id", "variant_id"]
        ),
        on="pair_id",
        validate="one_to_one",
    )
    p = pd.concat(
        [
            pd.read_csv(f, usecols=["pair_id", "model", "pred_delta"])
            for f in a.predictions
        ],
        ignore_index=True,
    )
    p = p[p.pair_id.isin(t.pair_id)]
    if p.duplicated(["model", "pair_id"]).any() or not np.isfinite(p.pred_delta).all():
        raise ValueError("invalid predictions")
    common = set.intersection(*(set(m.pair_id) for _, m in p.groupby("model")))
    t = t[~t.group_id.isin(t.loc[~t.pair_id.isin(common), "group_id"])].copy()
    c = t.parent_cycdelta_train_any | t.variant_cycdelta_train_any
    n = (
        t.parent_ncaa_p2_train
        | t.variant_ncaa_p2_train
        | t.parent_ncaa_p2_val
        | t.variant_ncaa_p2_val
    )
    schemes = {
        "full_common_test": pd.Series(False, index=t.index),
        "both_endpoints_unseen_by_CycDelta_OD_train": c,
        "both_endpoints_unseen_in_ncAA_Phase2_train_and_val": n,
        "both_endpoints_unseen_by_both_upstream_pools": n | c,
    }
    overall = []
    families = []
    parent = []
    groups = []
    for cohort, exposed in schemes.items():
        test = t[~t.group_id.isin(t.loc[exposed, "group_id"])]
        if test.empty:
            continue
        g, e = evaluate(test, p)
        if g.empty:
            continue
        groups.append(g.assign(cohort=cohort))
        pp = g.groupby(["model", "series_id", "parent_id"], as_index=False).agg(
            regret=("regret", "mean"),
            top1=("top1", "mean"),
            protocol_groups=("group_id", "size"),
        )
        parent.append(pp.assign(cohort=cohort))
        for model, f in p.groupby("model"):
            x = pp[pp.model == model]
            y = e[e.model == model]
            local = []
            for s, part in test.groupby("series_id"):
                z = x[x.series_id == s]
                d = y[y.series_id == s]
                local.append(
                    dict(
                        cohort=cohort,
                        model=model,
                        series_id=s,
                        source="|".join(sorted(set(part.source))),
                        rank_parents=len(z),
                        deltas=len(d),
                        regret=z.regret.mean(),
                        top1=z.top1.mean(),
                        delta_mae=d.delta_mae.mean(),
                        direction=d.direction.mean(),
                    )
                )
            families += local
            loc = pd.DataFrame(local)
            overall.append(
                dict(
                    cohort=cohort,
                    model=model,
                    options=len(test),
                    rank_parents=len(x),
                    deltas=len(y),
                    series=len(local),
                    regret=x.regret.mean(),
                    top1=x.top1.mean(),
                    delta_mae=y.delta_mae.mean(),
                    direction=y.direction.mean(),
                    series_equal_regret=loc.regret.mean(),
                    series_equal_top1=loc.top1.mean(),
                    series_equal_delta_mae=loc.delta_mae.mean(),
                )
            )
    pd.DataFrame(overall).to_csv(a.out / "comparison.csv", index=False)
    fam = pd.DataFrame(families)
    fam.to_csv(a.out / "comparison_by_series.csv", index=False)
    # With two independent holdout sources the two leave-one-source-out estimates
    # are exactly the two source results. The range is descriptive, not a 95% interval.
    spans = fam.groupby(["cohort", "model"], as_index=False).agg(
        series_count=("series_id", "nunique"),
        regret_min=("regret", "min"),
        regret_max=("regret", "max"),
        top1_min=("top1", "min"),
        top1_max=("top1", "max"),
        delta_MAE_min=("delta_mae", "min"),
        delta_MAE_max=("delta_mae", "max"),
    )
    spans.to_csv(a.out / "comparison_series_ranges.csv", index=False)
    pd.concat(parent).to_csv(a.out / "comparison_by_parent.csv", index=False)
    pd.concat(groups).to_csv(a.out / "comparison_by_group.csv", index=False)
    ex = fam[fam.model.eq("explicit_edit")]
    if not ex.empty:
        metrics = ["regret", "top1", "delta_mae", "direction"]
        m = fam[~fam.model.eq("explicit_edit")].merge(
            ex[["cohort", "series_id"] + metrics],
            on=["cohort", "series_id"],
            suffixes=("_other", "_explicit"),
            validate="many_to_one",
        )
        for k in metrics:
            m["explicit_minus_comparator_" + k] = m[k + "_explicit"] - m[k + "_other"]
        m[
            ["cohort", "model", "series_id", "source", "rank_parents", "deltas"]
            + ["explicit_minus_comparator_" + k for k in metrics]
        ].to_csv(a.out / "comparison_differences_by_series.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edits", type=Path, default=Path("results/pairs/edits.csv"))
    p.add_argument("--out", type=Path, default=Path("results/references"))
    p.add_argument("--predictions", type=Path, nargs="+", required=True)
    main(p.parse_args())
