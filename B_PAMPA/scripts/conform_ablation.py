"""Matched ablation of water and hexane descriptors in cyclic peptide N-methyl edits."""
import argparse
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
WATER = ['Water_avgRMSD_All','Water_avgRMSD_BackBone','Water_avgGR','Water_Desolvation_Free_Energy','Water_3D_SASA','Water_3D_NPSA','Water_3D_PSA']
HEXANE = ['Hexane_avgRMSD_All','Hexane_avgRMSD_BackBone','Hexane_avgGR','Hexane_3D_SASA','Hexane_3D_NPSA','Hexane_3D_PSA']
ARMS = {'2D':0, '2D_water':7, '2D_water_hexane':13}
SEED = 20261124
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=512,includeChirality=True)
SITE = rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=128,includeChirality=True)

def checksum(p): return sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

@lru_cache(None)
def canonical(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s),isomericSmiles=True,canonical=True)

def load_meta(path,measurements,out):
    m=pd.read_csv(path)
    d=pd.read_csv(measurements).set_index('record_id',verify_integrity=True)
    m['record_id']='CycPeptMPDB_record:'+m.CycPeptMPDB_ID.astype(str)
    assert m.record_id.is_unique
    rem=d.loc[m.record_id]
    np.testing.assert_array_equal(rem.smiles,m.SMILES)
    np.testing.assert_array_equal(rem.source,m.Source)
    np.testing.assert_allclose(rem.target,m.PAMPA,rtol=0,atol=1e-12)
    cols=WATER+HEXANE
    m['can']=m.SMILES.map(canonical)
    complete=m[np.isfinite(m[cols].to_numpy(dtype=float)).all(axis=1)]
    uniq=complete.drop_duplicates(['can',*cols])
    av=uniq.groupby('can',sort=False)[cols].mean()
    dump(out/'metadata.json',{'n_rows':len(m),'unique_exact_isomeric_structures':int(m.can.nunique()),
         'source_structural_ids':int(m.Structurally_Unique_ID.nunique()),'rows_complete_both_solvents':len(complete),
         'structures_with_multiple_distinct_vectors':int(uniq.groupby('can').size().gt(1).sum()),'rows_by_source':m.Source.value_counts().to_dict(),
         'features_water':WATER,'features_hexane':HEXANE,'input_sha256':checksum(path),'measurement_sha256':checksum(measurements),
         'aggregation':'Mean distinct vectors per stereochemical structure. Published PAMPA was used only to cross-check the metadata.'})
    return dict(zip(av.index,av.to_numpy(dtype=np.float32))), complete.groupby('can').record_id.agg('|'.join).to_dict()

@lru_cache(None)
def global2d(s):
    m=Chem.MolFromSmiles(s)
    extra=np.array([Descriptors.MolWt(m),Descriptors.MolLogP(m),rdMolDescriptors.CalcTPSA(m),Lipinski.NumHDonors(m),
                    Lipinski.NumHAcceptors(m),Lipinski.NumRotatableBonds(m),m.GetNumHeavyAtoms(),rdMolDescriptors.CalcNumRings(m),
                    rdMolDescriptors.CalcFractionCSP3(m),Chem.GetFormalCharge(m),rdMolDescriptors.CalcNumAmideBonds(m),
                    len(Chem.FindMolChiralCenters(m,includeUnassigned=True))],dtype=np.float32)
    return np.r_[MORGAN.GetCountFingerprintAsNumPy(m).astype(np.float32),extra]

@lru_cache(None)
def site2d(s,n):
    m=Chem.MolFromSmiles(s)
    assert m.GetAtomWithIdx(int(n)).GetAtomicNum()==7
    return SITE.GetCountFingerprintAsNumPy(m,fromAtoms=[int(n)]).astype(np.float32)

def design(rows,env,variant_only):
    chem=[];parent_env=[];change_env=[]
    for r in rows.itertuples(index=False):
        p=global2d(r.parent_smiles);v=global2d(r.variant_smiles)
        ps=site2d(r.parent_smiles,r.parent_n_idx);vs=site2d(r.variant_smiles,r.variant_n_idx)
        chem.append(np.concatenate([p,v-p,ps,vs-ps]))
        wv=env[r.variant_smiles]
        if variant_only:parent_env.append(wv)
        else:
            wp=env[r.parent_smiles];parent_env.append(wp);change_env.append(wv-wp)
    return np.stack(chem),np.stack(parent_env),np.stack(change_env) if change_env else np.empty((len(rows),0),dtype=np.float32)

def diagnose(df):
    out=[]
    for split,d in [('all',df),*df.groupby('split')]:
        g=d.groupby('group_id').agg(n=('pair_id','size'),both=('both_4d','sum'),nv=('variant_id','nunique'))
        assert g.n.eq(g.nv).all()
        both=d.both_4d
        out.append(dict(split=split,candidate_edits=len(d),measured_delta=int(d.delta.notna().sum()),
                   parent_4d=int(d.parent_4d.sum()),variant_4d=int(d.variant_4d.sum()),both_4d=int(both.sum()),
                   both_4d_measured_delta=int((both&d.delta.notna()).sum()),measured_delta_series_4d=int(d.loc[both&d.delta.notna(),'series_id'].nunique()),
                   rankable_groups=int(g.n.ge(2).sum()),rankable_groups_with_at_least_two_covered=int(g.both.ge(2).sum()),
                   rankable_groups_entire_choice_set_covered=int((g.n.ge(2)&g.n.eq(g.both)).sum()),
                   all_series=int(d.series_id.nunique()),both_4d_series=int(d.loc[both,'series_id'].nunique())))
    return pd.DataFrame(out)

def fit(rows,features,tr,ev,label,costs):
    use=(tr&rows.delta.notna()).to_numpy()
    if not use.any() or not ev.any():return None
    idx=np.where(ev.to_numpy())[0]
    held=rows.iloc[idx].copy()
    x,p,d=features
    for arm,n in ARMS.items():
        start=time.monotonic()
        block=x if n==0 else np.column_stack((x,p[:,:n],d[:,:n]))
        model=ExtraTreesRegressor(n_estimators=160,max_depth=8,min_samples_leaf=2,max_features=1.,random_state=SEED,n_jobs=2)
        model.fit(block[use],rows.loc[use,'delta'].to_numpy())
        held['pred_'+arm]=model.predict(block[idx])
        costs.append(dict(evaluation=label,method=arm,train_deltas=int(use.sum()),evaluated_options=len(idx),fit_and_predict_seconds=round(time.monotonic()-start,2)))
    return held

def score(all_pred,complete):
    groups=[];series=[];summary=[]
    for ev,q in all_pred.groupby('evaluation',sort=False):
        for arm in ['random',*ARMS]:
            rows=[]
            for gid,g in q.groupby('group_id'):
                if gid not in complete or len(g)<2:continue
                y=g.variant_y.to_numpy()
                if arm=='random':top=np.ones(len(g),dtype=bool)
                else:
                    yhat=g['pred_'+arm].to_numpy()
                    top=np.isclose(yhat,yhat.max(),atol=1e-12,rtol=0)
                rows.append(dict(evaluation=ev,method=arm,group_id=gid,parent_id=g.parent_id.iloc[0],series_id=g.series_id.iloc[0],
                            source=g.source.iloc[0],parent_measured=g.parent_y.notna().any(),n_sites=len(g),n_at_prediction_maximum=int(top.sum()),
                            regret=float(y.max()-y[top].mean()),hit=float(np.isclose(y[top],y.max(),atol=1e-12,rtol=0).mean())))
            gdf=pd.DataFrame(rows,columns=['evaluation','method','group_id','parent_id','series_id','source','parent_measured','n_sites','n_at_prediction_maximum','regret','hit'])
            groups.extend(rows)
            d=q[q.delta.notna()]
            e=None if arm=='random' else (d['pred_'+arm]-d.delta).abs()
            summary.append(dict(evaluation=ev,method=arm,eval_pairs_with_measured_delta=len(d),eval_delta_series=d.series_id.nunique(),
                           delta_mae=None if e is None else e.mean(),delta_mae_series_equal=None if e is None else e.groupby(d.series_id).mean().mean(),
                           rankable_complete_groups=len(gdf),rankable_series=gdf.series_id.nunique(),parent_equal_regret=gdf.regret.mean(),
                           parent_equal_hit=gdf.hit.mean(),series_equal_regret=gdf.groupby('series_id').regret.mean().mean(),
                           series_equal_hit=gdf.groupby('series_id').hit.mean().mean()))
            for sid,sub in q.groupby('series_id'):
                sub=sub[sub.delta.notna()];sg=gdf[gdf.series_id==sid]
                series.append(dict(evaluation=ev,method=arm,series_id=sid,n_quantifiable=len(sub),n_rankable=len(sg),
                       delta_mae=None if arm=='random' else (sub['pred_'+arm]-sub.delta).abs().mean(),regret=sg.regret.mean(),hit=sg.hit.mean()))
    return pd.DataFrame(summary),pd.DataFrame(groups),pd.DataFrame(series)

def intervals(series):
    out=[]
    for ev,g in series.groupby('evaluation'):
        for metric,n in [('regret','n_rankable'),('hit','n_rankable'),('delta_mae','n_quantifiable')]:
            v=g.pivot(index='series_id',columns='method',values=metric)
            for arm in ['2D_water','2D_water_hexane']:
                diff=(v[arm]-v['2D']).dropna()
                if diff.empty:continue
                e=diff.to_numpy()
                w=g[g.method=='2D'].set_index('series_id').loc[diff.index,n].to_numpy(dtype=float)
                ci=[None,None]
                if len(e)>=3:
                    draw=np.random.default_rng(SEED).integers(0,len(e),size=(2000,len(e)))
                    ci=np.quantile((e[draw]*w[draw]).sum(1)/w[draw].sum(1),[.025,.975])
                out.append(dict(evaluation=ev,metric=metric,arm=arm,reference='2D',independent_series=len(e),
                                paired_change=np.average(e,weights=w),bootstrap_series_n=2000 if len(e)>=3 else 0,ci_lower=ci[0],ci_upper=ci[1]))
    return pd.DataFrame(out)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--four-d',type=Path,default=ROOT/'data/raw/cycpeptmpdb_4d.csv')
    p.add_argument('--measurements',type=Path,default=ROOT/'data/processed/pampa.csv')
    p.add_argument('--pairs',type=Path,default=ROOT/'results/pairs/edits.csv')
    p.add_argument('--output',type=Path,default=ROOT/'results/conformation')
    p.add_argument('--index-only',action='store_true')
    p.add_argument('--variant-only',action='store_true')
    a=p.parse_args()
    out=a.output/'variant_only' if a.variant_only else a.output
    out.mkdir(parents=True,exist_ok=True)
    env,ids=load_meta(a.four_d,a.measurements,out)
    if a.index_only:return
    df=pd.read_csv(a.pairs)
    assert df.pair_id.is_unique and df.groupby('series_id').split.nunique().eq(1).all()
    mols=pd.concat([df[[side+'_id','split']].rename(columns={side+'_id':'molecule'}) for side in ['parent','variant']])
    assert mols.groupby('molecule').split.nunique().eq(1).all()
    for side in ['parent','variant']:
        col=side+'_smiles'
        assert df[col].map(canonical).eq(df[col]).all()
        df[side+'_4d']=df[col].isin(env);df[side+'_4d_record_ids']=df[col].map(ids)
    df['both_4d']=df.parent_4d&df.variant_4d
    diagnose(df).to_csv(out/'coverage.csv',index=False)
    keep=['pair_id','group_id','series_id','source','split','parent_id','variant_id','parent_4d','variant_4d','both_4d','parent_4d_record_ids','variant_4d_record_ids']
    df[keep].to_csv(out/'pair_coverage.csv',index=False)
    chosen='variant_4d' if a.variant_only else 'both_4d'
    g=df.groupby('group_id')[chosen].agg(['size','sum']);complete=set(g.index[g['size'].ge(2)&g['size'].eq(g['sum'])])
    rows=df[df[chosen]].reset_index(drop=True)
    if rows.empty:
        dump(out/'run.json',{'status':'no covered edits'});return
    X=design(rows,env,a.variant_only);costs=[];runs={}
    held=fit(rows,X,rows.split.eq('train'),rows.split.isin(['valid','test']),'fixed_train_to_valid_and_test',costs)
    if held is not None:
        for split,q in held.groupby('split'):runs['fixed_'+split]=q.copy()
    te=rows[rows.split=='test']
    if not a.variant_only and (te.loc[te.delta.notna(),'series_id'].nunique()<2 or te.loc[te.group_id.isin(complete),'series_id'].nunique()<2):
        others=[]
        for sid in sorted(rows.series_id.unique()):
            a_test=fit(rows,X,rows.series_id.ne(sid),rows.series_id.eq(sid),'holdout_'+sid,costs)
            if a_test is not None: others.append(a_test)
        if others:runs['supplementary_LOSO']=pd.concat(others)
    dump(out/'run.json',{'pairs_sha256':checksum(a.pairs),'seed':SEED,'variant_only':a.variant_only,
                         'estimator':'160 ExtraTrees, max_depth=8, min_samples_leaf=2, all features',
                         'fixed_training':'train only, measured same-assay deltas','evaluations':list(runs),'trajectories_recomputed':0})
    if not runs:return
    pred=pd.concat([q.assign(evaluation=name) for name,q in runs.items()])
    cols=['evaluation','split','series_id','source','group_id','pair_id','parent_id','variant_id','parent_y','variant_y','delta',*['pred_'+k for k in ARMS]]
    pred[cols].to_csv(out/'predictions.csv',index=False)
    metrics,groups,series=score(pred,complete)
    for name,table in [('metrics',metrics),('parent_metrics',groups),('series_metrics',series),('paired_uncertainty',intervals(series)),('timings',pd.DataFrame(costs))]:
        table.to_csv(out/(name+'.csv'),index=False)
    print(metrics.to_string(index=False))

if __name__=='__main__':main()
