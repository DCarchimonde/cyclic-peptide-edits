"""Compare two individual backbone N-methyl additions with their measured combination."""

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations, groupby
from operator import itemgetter
from pathlib import Path
from statistics import median

from pair_chemistry import write_csv


PANEL = ("source", "series_id", "split", "protocol_id", "backbone_id")
FIELDS = (*PANEL, "parent", "single_a", "single_b", "double", "base_n_a_0based",
          "base_n_b_0based", "pair_parent_a", "pair_parent_b", "pair_a_double",
          "pair_b_double", "y_parent", "y_a", "y_b", "y_double", "gain_vs_parent",
          "gain_vs_best_single", "excess_vs_additivity", "better_than_all_three")


@dataclass(frozen=True)
class Edge:
    pair_id: str
    site: int
    atom_map: list[int]


def frozen_rows(path, expected):
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        # The frozen builder writes CRLF; Git checkouts can present the same CSV with LF.
        crlf = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        if hashlib.sha256(crlf).hexdigest() != expected:
            raise ValueError(f"Frozen pair table mismatch: {actual}")
    text = io.StringIO(raw.decode("utf-8-sig"), newline="")
    return sorted(csv.DictReader(text), key=itemgetter(*PANEL)), actual


def measure_network(records):
    links = defaultdict(dict)
    values = {}
    for row in records:
        if not row["parent_y"] or not row["variant_y"]:
            continue
        for role in ("parent", "variant"):
            key = row[f"{role}_id"]
            # Eight decimals matches the frozen pair builder's delta precision.
            value = Decimal(row[f"{role}_y"]).quantize(Decimal("0.00000001"))
            if values.setdefault(key, value) != value:
                raise ValueError(f"Inconsistent point measurement within one protocol: {key}")
        children = links[row["parent_id"]]
        child = row["variant_id"]
        if child in children:
            raise ValueError(f"Repeated edge: {row['pair_id']}")
        children[child] = Edge(row["pair_id"], int(row["parent_n_idx"]),
                               json.loads(row["parent_to_variant_atom_map"]))
    return links, values


def measured_squares(panel, records):
    links, values = measure_network(records)
    found = []
    unresolved = 0
    for parent, children in sorted(links.items()):
        for a, b in combinations(sorted(children), 2):
            next_a, next_b = links.get(a, {}), links.get(b, {})
            edge_a, edge_b = children[a], children[b]
            for end in sorted(next_a.keys() & next_b.keys()):
                a_end, b_end = next_a[end], next_b[end]
                if (edge_a.site == edge_b.site or a_end.site != edge_a.atom_map[edge_b.site]
                        or b_end.site != edge_b.atom_map[edge_a.site]):
                    unresolved += 1
                    continue
                y0, ya, yb, yab = (values[k] for k in (parent, a, b, end))
                found.append(dict(
                    zip(PANEL, panel), parent=parent, single_a=a, single_b=b, double=end,
                    base_n_a_0based=edge_a.site, base_n_b_0based=edge_b.site,
                    pair_parent_a=edge_a.pair_id, pair_parent_b=edge_b.pair_id,
                    pair_a_double=a_end.pair_id, pair_b_double=b_end.pair_id,
                    y_parent=y0, y_a=ya, y_b=yb, y_double=yab,
                    gain_vs_parent=yab-y0, gain_vs_best_single=yab-max(ya, yb),
                    excess_vs_additivity=yab-ya-yb+y0,
                    better_than_all_three=yab > max(y0, ya, yb)
                ))
    return found, unresolved


def describe(rows):
    effects = [r["gain_vs_best_single"] for r in rows]
    return dict(
        squares=len(rows), distinct_parents=len({r["parent"] for r in rows}),
        distinct_double_products=len({r["double"] for r in rows}),
        protocols=len({r["protocol_id"] for r in rows}), series=len({r["series_id"] for r in rows}),
        better_than_both_singles=sum(x > 0 for x in effects),
        tied_with_best_single=sum(x == 0 for x in effects),
        below_best_single=sum(x < 0 for x in effects),
        better_than_all_three=sum(r["better_than_all_three"] for r in rows),
        median_gain_vs_best_single=float(median(effects)) if effects else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    records, digest = frozen_rows(args.pairs, args.expected_sha)
    rows = []
    excluded = defaultdict(int)
    by_source = defaultdict(list)
    for panel, group in groupby(records, key=itemgetter(*PANEL)):
        found, unresolved = measured_squares(panel, group)
        key = panel[:3]
        by_source[key].extend(found)
        excluded[key] += unresolved
        rows.extend(found)
    totals = describe(rows)
    summary = {
        "input_sha256": digest, "frozen_sha256_crlf": args.expected_sha,
        "endpoint": "measured log10 PAMPA permeability", "totals": totals,
        "unresolved_atom_map_closures": sum(excluded.values()),
        "by_source": [dict(source=k[0], series_id=k[1], split=k[2],
                           unresolved_atom_map_closures=excluded[k], **describe(v))
                      for k, v in by_source.items()],
        "interpretation": "Each row is a matched within-protocol comparison of two additions. "
                          "Rows can share compounds; counts compare reported point estimates.",
    }
    write_csv(args.out / "squares.csv", rows, FIELDS)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"B_SQUARES_OK squares={totals['squares']} better_than_both={totals['better_than_both_singles']} "
          f"better_than_all={totals['better_than_all_three']} series={totals['series']}", flush=True)


if __name__ == "__main__":
    main()
