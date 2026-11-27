"""Use measured singles to rank double products without seeing their endpoint."""
import argparse
import csv
import json
from collections import defaultdict
from decimal import Decimal
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from statistics import mean, median

from measured_squares import PANEL, frozen_rows, measured_squares

D = Decimal
METHODS = ("additive_singles", "best_single", "uniform_choice")
TASK = (*PANEL, "parent")


def write(path, keys, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def products(square_group):
    grouped = defaultdict(list)
    for r in square_group:
        grouped[r["double"]].append(r)
    options = []
    for product, paths in sorted(grouped.items()):
        observed = {r["y_double"] for r in paths}
        if len(observed) != 1:
            raise ValueError(f"Conflicting double-product measurements: {product}")
        options.append(dict(product=product, target=observed.pop(), paths=len(paths),
                            additive_singles=mean(r["y_a"] + r["y_b"] - r["y_parent"] for r in paths),
                            best_single=mean(max(r["y_a"], r["y_b"]) for r in paths),
                            uniform_choice=D(0)))
    return options


def decision_rows(group, options):
    best_observed = max(o["target"] for o in options)
    for method in METHODS:
        max_score = max(o[method] for o in options)
        selected = [o for o in options if o[method] == max_score]
        yield dict(**group, method=method, options=len(options), top_tie=len(selected),
                   regret=float(best_observed - mean(o["target"] for o in selected)),
                   top_hit=float(D(sum(o["target"] == best_observed for o in selected)) / len(selected)))


def summarize(rows):
    # Equalize repeated protocols per parent, then equalize parents within series.
    by_parent = defaultdict(list)
    for r in rows:
        by_parent[r["parent"]].append(r)
    return dict(rankable_panels=len(rows), parents=len(by_parent),
                hit_parent_equal=mean(mean(v["top_hit"] for v in s) for s in by_parent.values()),
                regret_parent_equal=mean(mean(v["regret"] for v in s) for s in by_parent.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data, actual = frozen_rows(a.pairs, a.expected_sha)
    all_squares = []
    unmapped = 0
    for panel, records in groupby(data, key=itemgetter(*PANEL)):
        found, skipped = measured_squares(panel, records)
        all_squares.extend(found)
        unmapped += skipped
    decisions, option_rows = [], []
    single_option = 0
    for key, records in groupby(sorted(all_squares, key=itemgetter(*TASK)), key=itemgetter(*TASK)):
        group = dict(zip(TASK, key))
        options = products(records)
        if len(options) < 2:
            single_option += 1
            continue
        option_rows.extend(dict(**group, **o) for o in options)
        decisions.extend(decision_rows(group, options))
    source_keys = ("source", "series_id", "split")
    strat = []
    for key, records in groupby(sorted(all_squares, key=itemgetter(*source_keys)), key=itemgetter(*source_keys)):
        group = dict(zip(source_keys, key))
        values = [r["excess_vs_additivity"] for r in records]
        strat.append(dict(**group, squares=len(values), positive=sum(v > 0 for v in values),
                          zero=sum(v == 0 for v in values), negative=sum(v < 0 for v in values),
                          median_signed=float(median(values)), median_absolute=float(median(map(abs, values)))))
    summary = []
    keys = (*source_keys, "method")
    for key, rows in groupby(sorted(decisions, key=itemgetter(*keys)), key=itemgetter(*keys)):
        summary.append(dict(zip(keys, key), **summarize(list(rows))))
    macro = []
    for key, rows in groupby(sorted(summary, key=itemgetter("split", "method")), key=itemgetter("split", "method")):
        group = list(rows)
        macro.append(dict(split=key[0], method=key[1], contributing_series=len(group),
                          hit_series_equal=mean(r["hit_parent_equal"] for r in group),
                          regret_series_equal=mean(r["regret_parent_equal"] for r in group)))
    write(a.out / "options.csv", (*TASK, "product", "paths", "target", *METHODS), option_rows)
    write(a.out / "rank_panels.csv", (*TASK, "method", "options", "top_tie", "regret", "top_hit"), decisions)
    write(a.out / "by_series.csv", (*keys, "rankable_panels", "parents", "hit_parent_equal", "regret_parent_equal"), summary)
    write(a.out / "interactions.csv", (*source_keys, "squares", "positive", "zero", "negative", "median_signed", "median_absolute"), strat)
    count = len(decisions) // len(METHODS)
    info = dict(input_sha256=actual, frozen_sha256_crlf=a.expected_sha, squares=len(all_squares),
                rankable_panels=count, single_option_panels=single_option, rankable_product_options=len(option_rows),
                unresolved_atom_map_closures=unmapped, series_interactions=strat, by_series=summary, by_split=macro,
                estimand="existing measured single products used to choose a double within one parent and protocol",
                tie_rule="uniform expected hit and regret over top-scoring options",
                interaction="measured double minus additive estimate on log10 PAMPA scale",
                primary_B_single_site_benchmark="separate")
    (a.out / "summary.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"B_SQUARE_SELECTION_OK squares={len(all_squares)} rankable_panels={count} "
          f"options={len(option_rows)} single_option_panels={single_option}", flush=True)


if __name__ == "__main__":
    main()
