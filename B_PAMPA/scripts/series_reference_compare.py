"""Combine the native reference and local series-out reports without changing model forecasts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from edit_metrics import MEASURES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', type=Path, default=Path('results/references_series'))
    parser.add_argument('--local-dir', type=Path, default=Path('results/model/series_sensitivity'))
    args = parser.parse_args()
    dest = args.reference_dir
    methods = {
        'ncAA_RepDistill_matched_readout': 'ncaa',
        'CycDelta_matched_refit': 'cycdelta',
    }
    local_predictions = pd.read_csv(args.local_dir / 'out_of_series_predictions.csv')
    local_ids = set(local_predictions.pair_id)
    for name, dirname in methods.items():
        ref = pd.read_csv(dest / dirname / 'matched_outer_predictions.csv')
        ref = ref.loc[ref.model == name]
        if ref.pair_id.duplicated().any() or set(ref.pair_id) != local_ids:
            raise ValueError(f'unmatched outer predictions for {name}')
    for table in ['metrics', 'per_parent', 'per_series', 'uncertainty']:
        frames = [pd.read_csv(args.local_dir / (table + '.csv'))]
        for name, dirname in methods.items():
            frame = pd.read_csv(dest / dirname / 'all' / (table + '.csv'))
            frames.append(frame.loc[frame.model == name])
        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(dest / ('comparison_' + table + '.csv'), index=False)
    series = pd.read_csv(dest / 'comparison_per_series.csv')
    rows = []
    for name in methods:
        left = series.loc[series.model == name]
        for other, right in series.groupby('model'):
            if name == other:
                continue
            both = left.merge(right, on='series_id', suffixes=('_reference', '_other'), validate='one_to_one')
            if len(both) != len(left):
                raise ValueError(f'unmatched series for {name} and {other}')
            for metric in MEASURES:
                count = metric + '_n'
                total = metric + '_sum'
                if not np.array_equal(both[count + '_reference'], both[count + '_other']):
                    raise ValueError('unmatched denominator')
                keep = both.loc[both[count + '_reference'] > 0].sort_values('series_id')
                n = keep[count + '_reference'].to_numpy()
                diff = keep[total + '_reference'].to_numpy() - keep[total + '_other'].to_numpy()
                low, high = np.nan, np.nan
                if len(n) >= 3:
                    draws = np.random.default_rng(2026).integers(len(n), size=(2000, len(n)))
                    low, high = np.quantile(diff[draws].sum(axis=1) / n[draws].sum(axis=1), [.025, .975])
                rows.append(dict(reference=name, other=other, metric=metric, contributing_series=len(n), denominator=n.sum(), reference_minus_other=diff.sum()/n.sum(), lower_95=low, upper_95=high))
    pd.DataFrame(rows).to_csv(dest / 'reference_differences.csv', index=False)


if __name__ == '__main__':
    main()
