"""Recalculate every outer score directly from saved weights/scalers and check the delivered CSVs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from refs_assets import NCAA_SHA, benchmark, sha


def ncaa_predictor(e, args, checksum):
    root = (args.external / 'ncAA-RepDistill').resolve()
    sys.path[:0] = [str(root / 'scripts'), str(root)]
    from evaluate_regression import build_mlp, standardize_apply

    names = sorted(set(e.parent_smiles) | set(e.variant_smiles))
    hashed = [hashlib.sha256(s.encode()).hexdigest().encode() for s in names]
    with np.load(args.pack, allow_pickle=False) as f:
        features = f['vectors']
        if str(f['checkpoint']) != NCAA_SHA or str(f['support']) != checksum or hashed != f['smiles_sha256'].tolist() or hashlib.sha256(features.tobytes()).hexdigest() != str(f['vectors_sha256']):
            raise ValueError('ncAA packed features or support mismatch')

    def predict(test, folder):
        with np.load(folder / 'ncaa_readout_scaler.npz', allow_pickle=False) as norm:
            x = torch.from_numpy(standardize_apply(features, norm['mean'], norm['scale']))
        m = build_mlp(x.shape[1], [256], .1)
        weights = folder / 'ncaa_readout.pt'
        m.load_state_dict(torch.load(weights, weights_only=True, map_location='cpu'))
        m.eval()
        lookup = dict(zip(names, torch.cat([m(b).flatten() for b in x.split(256)]).tolist()))
        return np.array([lookup[v]-lookup[p] for p, v in zip(test.parent_smiles, test.variant_smiles)]), weights

    return predict


def cyc_predictor(e, args, checksum):
    from rdkit import rdBase
    from refs_cycdelta import make_graphs, obtain, scores
    root = (args.external / 'CycDelta').resolve()
    sys.path[:0] = [str(root / 'smiles_delta_prediction'), str(root)]
    import predict as upstream

    saved = json.loads((args.result_root / args.model / 'run.json').read_text())
    monomers = args.cache / 'monomers_ee7d5a83731f0cb832c3.npz'
    if rdBase.rdkitVersion != saved['rdkit'] or checksum != saved['shared_support_sha256'] or sha(monomers) != saved['cached_monomers_sha256']:
        raise ValueError('CycDelta descriptor version or cache/support mismatch')
    rec, idx, h, d, gly, failures = obtain(e, upstream, args.cache)
    if failures:
        raise ValueError('CycDelta could not parse delivered support')

    def predict(test, folder):
        with np.load(folder / 'train_normalizer.npz', allow_pickle=False) as archived:
            scaler = torch.from_numpy(archived['mean']), torch.from_numpy(archived['std'])
        subset = sorted(set(test.parent_smiles) | set(test.variant_smiles))
        graphs = make_graphs({s: rec[s] for s in subset}, idx, h, d, gly, scaler, upstream)
        m = upstream.CycGNN(d_emb=128, n_heads=4, dropout=.25, num_gnn_layer=2)
        weights = folder / 'cycdelta_refit.pt'
        m.load_state_dict(torch.load(weights, weights_only=True, map_location='cpu'))
        m.eval()
        return scores(m, test, graphs, upstream), weights

    return predict


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['ncaa','cycdelta'], required=True)
    parser.add_argument('--external', type=Path, required=True)
    parser.add_argument('--result-root', type=Path, default=Path('results/references_series'))
    parser.add_argument('--edits', type=Path, default=Path('results/pairs/edits.csv'))
    parser.add_argument('--support', type=Path, default=Path('results/model/common_support.csv'))
    parser.add_argument('--pack', type=Path, default=Path('results/references/ncaa_native_features.npz'))
    parser.add_argument('--cache', type=Path, default=Path('results/references_series'))
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    e, checksum = benchmark(args.edits, args.support)
    predict = (ncaa_predictor if args.model == 'ncaa' else cyc_predictor)(e, args, checksum)
    rows = []
    out = args.result_root / args.model
    for series, test in e.groupby('series_id', sort=True):
        folder = out / 'folds' / series
        expected = pd.read_csv(folder / 'test_predictions.csv').set_index('pair_id')
        if expected.index.duplicated().any() or set(expected.index) != set(test.pair_id):
            raise ValueError('outer test pair identities changed')
        actual, weights = predict(test, folder)
        err = abs(expected.loc[test.pair_id,'pred_delta'].to_numpy() - actual)
        if not np.isfinite(err).all() or err.max() > 1e-5:
            raise ValueError(f'fixed-weight prediction exceeds 1e-5 in {series}, got {err.max()}')
        rows.append(dict(outer=series, weights_sha256=sha(weights), options=len(test), max_absolute_difference=err.max(), mean_absolute_difference=err.mean()))
    pd.DataFrame(rows).to_csv(out / 'verify_saved_weights.csv', index=False)
    print(f"{args.model}: re-scored {len(e)} options, maximum absolute difference {max(r['max_absolute_difference'] for r in rows):.3g}")


if __name__ == '__main__':
    main()
