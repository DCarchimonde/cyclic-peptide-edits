"""Rescore the frozen v0.1.0 predictions and compare an independent refit."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compare(folder):
    s=pd.read_csv(folder/'split_assignments.csv').set_index('record_id',verify_integrity=True).sort_index()
    d=pd.read_csv(folder/'predictions.csv').set_index('record_id',verify_integrity=True).sort_index()
    train=s[s.split=='train'];test=s[s.split=='test']
    assert set(train.source).isdisjoint(test.source) and set(train.canonical_smiles).isdisjoint(test.canonical_smiles)
    np.testing.assert_array_equal(test.index,d.index);np.testing.assert_array_equal(test.target,d.target)
    e=d.prediction-d.target
    m=dict(train_records=len(train),test_records=len(test),train_sources=int(train.source.nunique()),test_sources=int(test.source.nunique()),
           excluded_overlap=int(s.split.eq('excluded_overlap').sum()),training_mean=float(train.target.mean()),
           mae=float(e.abs().mean()),rmse=float(np.sqrt((e**2).mean())),spearman=float(spearmanr(d.target,d.prediction).statistic),
           training_mean_baseline_mae=float((d.target-train.target.mean()).abs().mean()))
    old=json.loads((folder/'metrics.json').read_text())
    for k in ['mae','rmse','spearman','training_mean_baseline_mae']:np.testing.assert_allclose(old[k],m[k],atol=1e-12,rtol=0)
    return d,s,m

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=Path('outputs/baseline'))
    p.add_argument('--refit',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('data/processed/pampa.csv'))
    p.add_argument('--output',type=Path,default=Path('results/legacy'))
    a=p.parse_args()
    old,sa,om=compare(a.archive);new,sb,nm=compare(a.refit)
    pd.testing.assert_series_equal(sa.split,sb.split)
    x=new[['source','target']].copy();x['archive_prediction']=old.prediction;x['refit_prediction']=new.prediction
    x['training_mean']=nm['training_mean'];x['refit_minus_archive']=new.prediction-old.prediction
    x['refit_absolute_error']=(new.prediction-new.target).abs();x['training_mean_absolute_error']=(new.target-nm['training_mean']).abs()
    r={'data_sha256':sha(a.data),'archived_predictions_sha256':sha(a.archive/'predictions.csv'),'refit_predictions_sha256':sha(a.refit/'predictions.csv'),
       'training_parameters':{'trees':200,'seed':2026,'max_depth':14,'min_samples_leaf':2},'archive_rescore':om,'refit_rescore':nm,
       'identical_split_assignments':True,'maximum_absolute_refit_minus_archive_prediction':float(x.refit_minus_archive.abs().max()),
       'refit_worse_than_training_mean_mae_by':nm['mae']-nm['training_mean_baseline_mae'],
       'versions':{k:version(k) for k in ['numpy','pandas','rdkit','scipy','scikit-learn','joblib']},'python':platform.python_version()}
    a.output.mkdir(parents=True,exist_ok=True)
    x.to_csv(a.output/'aligned_predictions.csv')
    x.groupby('source').agg(n=('target','size'),refit_mae=('refit_absolute_error','mean'),training_mean_mae=('training_mean_absolute_error','mean')).to_csv(a.output/'source_metrics.csv')
    (a.output/'audit.json').write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':main()
