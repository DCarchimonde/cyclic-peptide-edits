"""Fit within-parent site rankers on the published series folds and compare first choices."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from edit_metrics import TOL, score_predictions
from fit_edits import (
    EPOCHS, ROOT, SEED, eligible, fit_scaler, integrity, make_head, make_x, molecule_bank, scaling,
)

HEADS = ("chem_site", "explicit_edit")
REFERENCES = ("random_site_zero_delta", "chem_site", "explicit_edit", "CycDelta_matched_refit",
              "ncAA_RepDistill_matched_readout")
KEYS = ["series_id", "parent_id"]


def training_pairs(frame):
    frame = frame.reset_index(drop=True)
    groups = frame.groupby("group_id", sort=False)
    rankable = frame[groups.group_id.transform("size") >= 2]
    per_parent = rankable.groupby("parent_id").group_id.nunique()
    y = frame.variant_y.to_numpy(dtype=float)
    pairs, targets, weights = [], [], []
    for _, g in rankable.groupby("group_id", sort=False):
        i, j = np.triu_indices(len(g), 1)
        left, right = g.index.to_numpy()[i], g.index.to_numpy()[j]
        delta = y[left] - y[right]
        if not np.isfinite(delta).all():
            raise ValueError("A training site is missing its quantitative variant measurement")
        pairs.append(np.column_stack([left, right]))
        targets.append(np.where(np.abs(delta) <= TOL, 0.5, (delta > 0).astype(float)))
        weights.append(np.full(len(i), 1.0 / (len(i) * per_parent.at[g.parent_id.iat[0]])))
    if not pairs:
        raise ValueError("No measured within-parent choices in the training series")
    a, b = torch.from_numpy(np.concatenate(pairs).astype("int64")).T
    weight = torch.from_numpy(np.concatenate(weights).astype("float32"))
    return a, b, torch.from_numpy(np.concatenate(targets).astype("float32")), weight / weight.sum()


def parent_scores(frame, scores, model):
    # Reuse the benchmark's tie and protocol rules. The head outputs ordinal scores.
    pred = pd.DataFrame(dict(pair_id=frame.pair_id, model=model, pred_delta=scores))
    groups, _ = score_predictions(frame.assign(delta=np.nan), pred)
    return groups.groupby(["model", *KEYS], as_index=False)[["regret", "hit"]].mean()


def fit_fold(frame, x, model, dest):
    train = frame.split.eq("train").to_numpy()
    valid = frame.split.eq("valid").to_numpy()
    test = frame.split.eq("test").to_numpy()
    scale = fit_scaler(x[train])
    tr, va, te = [scaling(x[mask], scale) for mask in (train, valid, test)]
    i, j, target, weight = training_pairs(frame.loc[train])
    torch.manual_seed(SEED)
    head = make_head(x.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.001)
    best = None
    for epoch in range(1, max(EPOCHS) + 1):
        head.train()
        opt.zero_grad()
        scores = head(tr)
        loss = F.binary_cross_entropy_with_logits(scores[i] - scores[j], target, reduction="none")
        (weight * loss).sum().backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step()
        if epoch in EPOCHS:
            head.eval()
            with torch.inference_mode():
                par = parent_scores(frame.loc[valid], head(va).numpy(), model)
            if par.empty:
                raise ValueError("Frozen validation roles contain no measured site choice")
            key = (float(par.regret.mean()), epoch)
            if not np.isfinite(key[0]):
                raise ValueError("Nonfinite validation loss")
            if best is None or key < best[0]:
                best = (key, copy.deepcopy(head.state_dict()))
    (valid_loss, epoch), state = best
    torch.save(dict(feature_head=model, score_type="ordinal", weight_dims=x.shape[1], state=state,
                    selected_epoch=epoch, scale={k: torch.tensor(v) for k, v in scale.items()}), dest)
    head.load_state_dict(state)
    head.eval()
    with torch.inference_mode():
        scores = head(te).numpy()
    test_frame = frame.loc[test]
    preds = test_frame[["pair_id", "series_id", "group_id"]].assign(model="rank_" + model, score=scores)
    par = parent_scores(test_frame, scores, "rank_" + model)
    return preds, par, dict(head=model, epoch=epoch, validation_regret=valid_loss, training_pairs=len(i))


def compare(par, archived, out):
    base = archived[archived.model.isin(REFERENCES)][["model", *KEYS, "regret", "hit"]].copy()
    combined = pd.concat([par, base], ignore_index=True)
    main = par[par.model.eq("rank_explicit_edit")].set_index(KEYS).sort_index()
    expected = set(REFERENCES) | {"rank_chem_site", "rank_explicit_edit"}
    if set(combined.model) != expected or combined.duplicated(["model", *KEYS]).any():
        raise ValueError("Missing model or duplicate parent in the common comparison")
    totals, diffs = [], []
    for model, rows in combined.groupby("model"):
        scores = rows.set_index(KEYS).sort_index()[["hit", "regret"]]
        if not main.index.equals(scores.index) or not np.isfinite(scores.to_numpy()).all():
            raise ValueError(f"Different parent panels or invalid first choices for {model}")
        by_series = scores.groupby(level="series_id").mean()
        totals.append(dict(model=model, parents=len(scores), series=len(by_series),
                           regret=scores.regret.mean(), hit=scores.hit.mean(),
                           regret_series_equal=by_series.regret.mean(), hit_series_equal=by_series.hit.mean()))
        if model == "rank_explicit_edit":
            continue
        diff = main.regret - scores.regret
        blocks = diff.groupby(level="series_id").agg(["sum", "count"])
        draws = np.random.default_rng(SEED).integers(len(blocks), size=(2000, len(blocks)))
        sums, counts = blocks["sum"].to_numpy(), blocks["count"].to_numpy()
        low, high = np.quantile(sums[draws].sum(axis=1) / counts[draws].sum(axis=1), [.025, .975])
        diffs.append(dict(comparator=model, rank_explicit_minus_comparator=diff.mean(), low95=low,
                          high95=high, series_equal_difference=(sums/counts).mean(),
                          series_with_lower_loss=int((sums < 0).sum()), series=len(blocks)))
    combined.to_csv(out / "per_parent.csv", index=False)
    combined.groupby(["model", "series_id"], as_index=False).agg(
        parents=("parent_id", "size"), hit=("hit", "mean"), regret=("regret", "mean")
    ).to_csv(out / "per_series.csv", index=False)
    pd.DataFrame(totals).to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(diffs).to_csv(out / "paired_regret.csv", index=False)
    return len(main), main.index.get_level_values("series_id").nunique()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("edits", "weights", "cache", "out"):
        ap.add_argument("--" + arg, type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(a.edits)
    bank = molecule_bank(data, a.weights, a.cache, a.threads)
    frame, excluded = eligible(data, bank)
    excluded.to_csv(a.out / "excluded.csv", index=False)
    archived_dir = ROOT / "results/model/series_sensitivity"
    reference_ids = pd.read_csv(archived_dir / "out_of_series_predictions.csv", usecols=["pair_id"])
    if set(frame.pair_id) != set(reference_ids.pair_id):
        raise ValueError("Site options differ from the published leave-one-series-out comparison")
    roles = pd.read_csv(archived_dir / "folds.csv")
    if set(roles.outer) != set(frame.series_id):
        raise ValueError("Incomplete outer-series manifest")
    preds, parent_rows, fitted = [], [], []
    for head in HEADS:
        x = make_x(frame, bank, head)
        for outer, mapping in roles.groupby("outer"):
            labels = mapping.set_index("series_id", verify_integrity=True).role
            split = frame.assign(split=frame.series_id.map(labels))
            if set(split.split) != {"train", "valid", "test"} or not split.split.eq("test").equals(split.series_id.eq(outer)):
                raise ValueError(f"Invalid split for {outer}")
            integrity(split)
            dest = a.out / "fitted" / outer
            dest.mkdir(parents=True, exist_ok=True)
            pred, parents, meta = fit_fold(split, x, head, dest / (head + ".pt"))
            preds.append(pred)
            parent_rows.append(parents)
            fitted.append(dict(outer=outer, **meta))
            print(f"Finished {outer} {head}", flush=True)
        del x
    pd.concat(preds).to_csv(a.out / "site_scores.csv", index=False)
    pd.DataFrame(fitted).to_csv(a.out / "training.csv", index=False)
    archived = pd.read_csv(ROOT / "results/references_series/comparison_per_parent.csv")
    n, s = compare(pd.concat(parent_rows), archived, a.out)
    meta = dict(options=len(frame), parents=n, rankable_series=s, folds=roles.outer.nunique(), fitted=len(fitted))
    (a.out / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"B_PAIRWISE_RANK_OK parents={n} rankable_series={s} fitted={len(fitted)}")


if __name__ == "__main__":
    main()
