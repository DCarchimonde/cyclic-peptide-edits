"""Kelly source check by stereochemical sequence and sublibrary, independently of labels."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen

import numpy as np
import pandas as pd
import pair_data as pair

ROOT=Path(__file__).resolve().parents[1]
SOURCES={2:(25919156,'99cc1d11c6cd65fd9da8db8bc37e8c9ee44c5fc3ceb8579f13ea11327dca98d9'),
         3:(25919159,'e0088382722c31a0cf5a039b1cc4c4e1778574ec2e63b299abb3fddc5861523c'),
         4:(25919162,'4853b20ff248595a495dab1ad9c9faf1ff28e740ea603c14d1b318febb8b0f1c')}
TOKENS={'Acetyl':'ac-','ThrCleaved':'T','P':'P','p':'dP','L':'L','l':'dL','A':'A','a':'dA',
        'ML':'meL','Ml':'Me_dL','MA':'meA','Ma':'Me_dA','f':'dF','F':'F','Abu':'Abu','abu':'dAbu',
        'ParaFluoroPhe':'Phe(4-F)','34diFluoroDPhe':'dPhe(3,4-diF)','OMeTyr':'Tyr(Me)','nva':'dNva',
        'paraFluoroPhe':'dPhe(4-F)','Nva':'Nva','cha':'dCha','nal':'d1-Nal','Hph':'Hph','ParaNitroPhe':'Phe(4-NO2)'}

def sheets(directory):
    rows=[]
    for n,(fid,digest) in SOURCES.items():
        f=directory/f'ja0c06115_si_00{n}.xlsx'
        if not f.is_file():
            f.parent.mkdir(parents=True,exist_ok=True)
            url='https://ndownloader.figshare.com/files/'+str(fid)
            with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=45) as r:raw=r.read()
            if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Unexpected Kelly workbook bytes')
            f.write_bytes(raw)
        if hashlib.sha256(f.read_bytes()).hexdigest()!=digest:raise ValueError('Recheck workbook version')
        d=pd.read_excel(f,sheet_name=0)
        d['source_workbook']=f.name;d['row_1based']=d.index+2
        if n==2:
            d=d[d.Library.isin([2,3,4])].copy();d['library']=d.Library.map(lambda x:f'L{int(x)}')
        else:d['library']='L5' if n==3 else 'L1'
        tokens=d['Top Sequence'].map(lambda q:q.split(','))
        d['seq']=tokens.map(lambda t:tuple(TOKENS[x.replace('*','')] for x in t))
        d['isotope_positions']=tokens.map(lambda t:','.join(str(i+1) for i,x in enumerate(t) if '***' in x))
        rows.append(d)
    return pd.concat(rows,ignore_index=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workbooks',type=Path,default=ROOT/'build/kelly_workbooks')
    p.add_argument('--original',type=Path,default=ROOT/'data/raw/cycpeptmpdb_all.csv')
    p.add_argument('--pairs',type=Path,default=ROOT/'results/pairs/edits.csv')
    p.add_argument('--index',type=Path,default=ROOT/'.cache/pair_graph_index.json')
    p.add_argument('--output',type=Path,default=ROOT/'results/pairs/kelly_source_audit')
    p.add_argument('--expanded-pairs',type=Path)
    a=p.parse_args()
    k=[r for r in pair.read_csv(a.original) if r['Source']=='2021_Kelly']
    s=sheets(a.workbooks)
    raw=pd.DataFrame(k);raw['library']=raw.Original_Name_in_Source_Literature.str.split('_').str[0]
    raw['seq']=raw.Sequence.map(lambda x:tuple(ast.literal_eval(x)))
    raw['record_id']='CycPeptMPDB_record:'+raw.CycPeptMPDB_ID
    matched=raw.merge(s,on=['library','seq'],how='left',validate='one_to_one')
    assert len(matched)==len(s)==1519 and matched.logPapp.notna().all()
    diff=(matched.PAMPA.astype(float)-matched.logPapp).abs()
    assert diff.max()<0.00501
    flag=pd.to_numeric(matched.T_PAMPA,errors='coerce').eq(0)
    assert (flag==matched['Avg%T'].eq(0)).all() and (matched.Papp>0).all()
    t=matched.loc[flag,['record_id','library','Original_Name_in_Source_Literature','source_workbook','row_1based','PAMPA','Avg%T','Papp','logPapp','isotope_positions']].copy()
    t.columns=['record_id','library','db_name','source_workbook','row_1based','db_log10_cm_s','source_average_percent_transmission','source_Papp_in_1e-6_cm_s','source_log10_cm_s','source_isotope_token_positions_1based']
    a.output.mkdir(parents=True,exist_ok=True)
    t.to_csv(a.output/'source_zero_transmission.csv',index=False)
    pairs=pd.read_csv(a.pairs);pk=pairs[pairs.source=='2021_Kelly']
    iso=matched.set_index('record_id').isotope_positions.to_dict()
    check=[len({iso[i] for i in (r.parent_record_ids+'|'+r.variant_record_ids).split('|')})==1 for r in pk.itertuples() if isinstance(r.parent_record_ids,str)]
    gcheck=[len({iso[i] for ids in g.variant_record_ids for i in ids.split('|')})==1 for _,g in pk.groupby('group_id') if len(g)>=2]
    meta={'n_matched':len(matched),'observed_max_abs_db_minus_source_log10':float(diff.max()),
          'zero_transmission_with_positive_papp':int(flag.sum()),'zero_counts_by_library':t.library.value_counts().to_dict(),
          'n_frozen_measured_delta_checked_for_matching_isotope_position_lists':len(check),
          'n_same_isotope_position_lists_measured_delta':sum(check),'n_frozen_rank_groups_checked_for_matching_isotope_positions':len(gcheck),
          'n_same_isotope_position_lists_ranking':sum(gcheck),'source_files':{str(n):{'sha256':v[1],'doi':f'https://doi.org/10.1021/jacs.0c06115.s00{n}'} for n,v in SOURCES.items()},
          'source_methods':'https://doi.org/10.1021/jacs.0c06115.s001, pp S14–S18 and Scheme S1',
          'interpretation':'PAMPA Papp reported after manual integration of automatic zero-peaks when possible. No compound-level limit inferred. Thirteen L5 zero-transmission records have identical positive final Papp 0.14e-6.'}
    if a.expanded_pairs:
        graph=pair.graph_index(a.original,a.index);cfg=json.loads((ROOT/'configs/pair_protocols.json').read_text())['sources']
        q={r['CycPeptMPDB_ID']:pair.quality(r,{}) for r in k}
        for r in k:
            r['_graph']=graph['graphs'].get(r['SMILES'])
            if r['T_PAMPA'] and float(r['T_PAMPA'])==0:q[r['CycPeptMPDB_ID']]=(float(r['PAMPA']),'source_numeric')
        protocol=lambda r:pair.protocol(r,cfg)
        points=pair.cells_for(k,protocol,q);allcells=pair.cells_for(k,protocol,q,False)
        expanded,_=pair.make_rows(cfg,allcells,points,{r['_graph'] for r in k if r['_graph']},graph,q)
        e=pd.DataFrame(expanded,columns=pairs.columns)
        for c in ['parent_y','variant_y','delta']:e[c]=pd.to_numeric(e[c],errors='coerce')
        assert set(pk.pair_id)<=set(e.pair_id)
        count=lambda d:{'options':len(d),'measured_delta':int(d.delta.notna().sum()),'rank_groups':int(d.groupby('group_id').size().ge(2).sum())}
        meta.update(frozen_Kelly=count(pk),restore_finite_source_values_Kelly=count(e))
        a.expanded_pairs.parent.mkdir(parents=True,exist_ok=True)
        pd.concat([pairs[pairs.source!='2021_Kelly'],e],ignore_index=True).to_csv(a.expanded_pairs,index=False)
        meta['expanded_pairs_sha256']=hashlib.sha256(a.expanded_pairs.read_bytes()).hexdigest()
    (a.output/'summary.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))

if __name__=='__main__': main()
