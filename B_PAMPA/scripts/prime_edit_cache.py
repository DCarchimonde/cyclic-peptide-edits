"""Replay the label-free molecule batching used for the locked round, using the raw graph indices."""

from collections import defaultdict
from pathlib import Path
import argparse, json
import pandas as pd
from encoder_cache import encode_bank


def prime(graph, records, weights, cache, threads):
    c = json.loads(graph.read_text())
    p = pd.read_csv(records)
    src = defaultdict(set)
    groups = defaultdict(set)
    for r in p.itertuples():
        if r.smiles not in c["graphs"]:
            continue
        cur = c["graphs"][r.smiles]
        src[r.source].add(cur)
        for parent in c["edits"][cur]:
            groups[r.source, parent].add(cur)
    quick = set()
    bulk = set()
    for (s, pa), variants in groups.items():
        if len(variants) > 1 or pa in src[s]:
            dest = (
                bulk
                if s in {"2020_Townsend", "2021_Kelly", "2013_CHUGAI", "2016_Furukawa"}
                else quick
            )
            dest.update(variants)
            dest.add(pa)
    print("label-free pools", len(quick), len(bulk), flush=True)
    for pool in [quick, bulk]:
        encode_bank(pool, weights, cache, threads=threads)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--graph", type=Path, required=True)
    a.add_argument("--records", type=Path, required=True)
    a.add_argument("--weights", type=Path, required=True)
    a.add_argument("--cache", type=Path, required=True)
    a.add_argument("--threads", type=int, default=2)
    o = a.parse_args()
    prime(o.graph, o.records, o.weights, o.cache, o.threads)
