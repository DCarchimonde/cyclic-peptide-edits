"""Rank ties by expected first-choice loss and use paired resampling of series."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

TOL = 1e-7
MEASURES = ["regret", "hit", "delta_mae", "direction_accuracy"]


def score_predictions(edits, forecasts):
    if forecasts.duplicated(["pair_id", "model"]).any():
        raise ValueError("duplicate forecast")
    m = forecasts.merge(edits, on="pair_id", validate="many_to_one")
    if len(m) != len(forecasts) or not np.isfinite(m.pred_delta).all():
        raise ValueError("invalid forecasts")
    rows = []
    for (model, gid), g in m.groupby(["model", "group_id"], sort=True):
        if len(g) < 2:
            continue
        y = g.variant_y.to_numpy()
        p = g.pred_delta.to_numpy()
        chosen = np.isclose(p, p.max(), rtol=0, atol=TOL)
        best = np.isclose(y, y.max(), rtol=0, atol=TOL)
        r = g.iloc[0]
        rows.append(
            dict(
                model=model,
                group_id=gid,
                parent_id=r.parent_id,
                source=r.source,
                protocol_id=r.protocol_id,
                series_id=r.series_id,
                split=r.split,
                sites=len(g),
                selected_variant_ids="|".join(g.variant_id[chosen]),
                best_variant_ids="|".join(g.variant_id[best]),
                regret=float(y.max() - y[chosen].mean()),
                hit=float(best[chosen].mean()),
                measured_best=y.max(),
                measured_selected=y[chosen].mean(),
            )
        )
    groups = pd.DataFrame(
        rows,
        columns=[
            "model",
            "group_id",
            "parent_id",
            "source",
            "protocol_id",
            "series_id",
            "split",
            "sites",
            "selected_variant_ids",
            "best_variant_ids",
            "regret",
            "hit",
            "measured_best",
            "measured_selected",
        ],
    )
    edges = m[m.delta.notna()].copy()
    edges["delta_mae"] = (edges.pred_delta - edges.delta).abs()
    edges["direction_accuracy"] = np.where(
        np.abs(edges.delta) > TOL,
        (np.sign(edges.pred_delta) == np.sign(edges.delta)).astype(float),
        np.nan,
    )
    return groups, edges


def summarize(edits, forecasts, out, draws=2000, seed=2026):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    groups, edges = score_predictions(edits, forecasts)
    groups.to_csv(out / "per_group.csv", index=False)
    par = groups.groupby(
        ["model", "split", "series_id", "parent_id"], as_index=False
    ).agg(
        regret=("regret", "mean"),
        hit=("hit", "mean"),
        protocol_groups=("group_id", "size"),
    )
    par.to_csv(out / "per_parent.csv", index=False)
    records = []
    diagnostics = []
    for (model, split), g in edges.groupby(["model", "split"]):
        diagnostics.append(
            dict(
                model=model,
                split=split,
                n=len(g),
                measured_positive=(g.delta > TOL).sum(),
                measured_negative=(g.delta < -TOL).sum(),
                measured_zero=(np.abs(g.delta) <= TOL).sum(),
                predicted_positive_fraction=(g.pred_delta > TOL).mean(),
                predicted_median=g.pred_delta.median(),
                measured_median=g.delta.median(),
                spearman_measured_delta_vs_prediction=g.delta.corr(
                    g.pred_delta, method="spearman"
                )
                if g.pred_delta.nunique() > 1 and g.delta.nunique() > 1
                else np.nan,
            )
        )
    pd.DataFrame(diagnostics).to_csv(out / "delta_diagnostics.csv", index=False)
    for (model, split), x in forecasts.merge(
        edits[["pair_id", "split", "series_id"]], on="pair_id"
    ).groupby(["model", "split"]):
        for s in sorted(set(x.series_id)):
            r = par[(par.model == model) & (par.split == split) & (par.series_id == s)]
            d = edges[
                (edges.model == model) & (edges.split == split) & (edges.series_id == s)
            ]
            z = dict(
                model=model,
                split=split,
                series_id=s,
                rank_parents=len(r),
                rank_groups=int(r.protocol_groups.sum()),
                deltas=len(d),
            )
            for k, frame in [
                ("regret", r),
                ("hit", r),
                ("delta_mae", d),
                ("direction_accuracy", d),
            ]:
                v = frame[k].dropna()
                z[k + "_sum"] = v.sum()
                z[k + "_n"] = len(v)
                z[k] = v.mean()
            records.append(z)
    by = pd.DataFrame(records)
    by.to_csv(out / "per_series.csv", index=False)
    sums = []
    stats = []
    paired = []

    def ci(numerator, counts):
        if len(counts) < 3:
            return np.nan, np.nan
        idx = np.random.default_rng(seed).integers(
            0, len(counts), size=(draws, len(counts))
        )
        return np.quantile(
            numerator[idx].sum(axis=1) / counts[idx].sum(axis=1), [0.025, 0.975]
        )

    for (name, split), t in by.groupby(["model", "split"]):
        z = dict(
            model=name,
            split=split,
            series_total=len(t),
            ranking_series=int((t.rank_parents > 0).sum()),
            delta_series=int((t.deltas > 0).sum()),
            rank_parents=int(t.rank_parents.sum()),
            rank_groups=int(t.rank_groups.sum()),
            deltas=int(t.deltas.sum()),
        )
        for k in MEASURES:
            a = t[t[k + "_n"] > 0].sort_values("series_id")
            n = a[k + "_n"].sum()
            z[k] = a[k + "_sum"].sum() / n if n else np.nan
            z[k + "_series_equal"] = a[k].mean()
            lo, hi = ci(a[k + "_sum"].to_numpy(), a[k + "_n"].to_numpy())
            stats.append(
                dict(
                    model=name,
                    split=split,
                    metric=k,
                    estimate=z[k],
                    eligible_series=len(a),
                    lower_95=lo,
                    upper_95=hi,
                    draws=draws if len(a) >= 3 else 0,
                )
            )
        sums.append(z)
    for split, t in by.groupby("split"):
        a = t[t.model == "explicit_edit"]
        for other in sorted(set(t.model) - {"explicit_edit"}):
            j = a.merge(t[t.model == other], on="series_id", suffixes=("_a", "_b"))
            for k in MEASURES:
                s = j[(j[k + "_n_a"] > 0) & (j[k + "_n_b"] > 0)]
                if s.empty or (s[k + "_n_a"] != s[k + "_n_b"]).any():
                    continue
                counts = s[k + "_n_a"].to_numpy()
                diff = s[k + "_sum_a"].to_numpy() - s[k + "_sum_b"].to_numpy()
                lo, hi = ci(diff, counts)
                paired.append(
                    dict(
                        comparison="explicit_edit minus " + other,
                        split=split,
                        metric=k,
                        eligible_series=len(s),
                        difference=diff.sum() / counts.sum(),
                        lower_95=lo,
                        upper_95=hi,
                    )
                )
    pd.DataFrame(sums).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(stats).to_csv(out / "uncertainty.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "paired_method_differences.csv", index=False)
    return pd.DataFrame(sums)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--edits", type=Path, required=True)
    a.add_argument("--predictions", type=Path, nargs="+", required=True)
    a.add_argument("--out", type=Path, required=True)
    o = a.parse_args()
    f = pd.concat(pd.read_csv(p) for p in o.predictions)[
        ["pair_id", "model", "pred_delta"]
    ]
    common = set.intersection(*(set(g.pair_id) for _, g in f.groupby("model")))
    e = pd.read_csv(o.edits)
    bad = set(e.loc[~e.pair_id.isin(common), "group_id"])
    e = e[e.pair_id.isin(common) & ~e.group_id.isin(bad)]
    print(summarize(e, f[f.pair_id.isin(e.pair_id)], o.out).to_string(index=False))
