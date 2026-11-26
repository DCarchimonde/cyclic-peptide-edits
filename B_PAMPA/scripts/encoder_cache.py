"""Cache real pretrained global, residue, edited nitrogen and two-bond environment vectors."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
import rdkit
from transformers import AutoModelForMaskedLM,AutoTokenizer
from encoder_graph import molecule_data,rotation_indices,protein_residue_reindex,MORGAN,LOCAL
from fetch_edit_encoders import SPEC,verify
from run_baseline import molecular_features
ROOT=Path(__file__).resolve().parents[1]
VERSION=hashlib.sha256(Path(__file__).read_bytes()+(Path(__file__).parent/'encoder_graph.py').read_bytes()+rdkit.__version__.encode()).hexdigest()[:16]
def cache_path(cache,s):return Path(cache)/(hashlib.sha256((VERSION+s).encode()).hexdigest()+'.npz')
def read_bank(cache,s):
    with np.load(cache_path(cache,s),allow_pickle=False) as f:return {k:f[k] for k in f.files}
def states(model,e):return model(input_ids=e['input_ids'],attention_mask=e['attention_mask']).last_hidden_state.cpu().numpy()
def pool_atoms(hidden,spans,offsets):
    out=[]
    for a,b in spans:
        idx=[k for k,(lo,hi) in enumerate(offsets) if min(hi,b)>max(lo,a)]
        if not idx:raise ValueError('atom_has_no_chemical_token')
        out.append(hidden[idx].mean(axis=0))
    return np.stack(out)
@torch.inference_mode()
def encode_bank(smiles,directory,cache,batch=16,threads=2):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    missing=sorted(set(str(s) for s in smiles if not cache_path(cache,str(s)).exists()),key=lambda x:(len(x),x))
    if not missing:return
    torch.set_num_threads(threads);verify(directory)
    tok={k:AutoTokenizer.from_pretrained(directory/k,local_files_only=True) for k in SPEC}
    model={k:AutoModelForMaskedLM.from_pretrained(directory/k,local_files_only=True).base_model.eval() for k in SPEC}
    seq_cache={}
    def save(s,**fields):
        p=cache_path(cache,s);t=p.with_suffix('.tmp.npz');np.savez_compressed(t,**fields);t.replace(p)
    for off in range(0,len(missing),batch):
        items={}
        for s in missing[off:off+batch]:
            try:items[s]=molecule_data(s)
            except ValueError as exc:save(s,error=str(exc))
        seqs=sorted({a[4] for a in items.values()}-seq_cache.keys())
        if seqs:
            e=tok['esm']([' '.join(s) for s in seqs],return_tensors='pt',padding=True)
            h=states(model['esm'],e);seq_cache.update({s:h[i,1:len(s)+1] for i,s in enumerate(seqs)})
        if not items:continue
        e=tok['chem'](list(items),return_offsets_mapping=True,return_tensors='pt',padding=True,truncation=False)
        long=(e['attention_mask'].sum(dim=1)>512).tolist()
        if any(long):
            for s,bad in list(zip(items.keys(),long)):
                if bad:items.pop(s);save(s,error='chemical_token_length_above_512')
            if not items:continue
            e=tok['chem'](list(items),return_offsets_mapping=True,return_tensors='pt',padding=True)
        h=states(model['chem'],e)
        for i,(s,(m,spans,r,cycle,seq,starts)) in enumerate(items.items()):
            try:
                atom=pool_atoms(h[i],spans,e['offset_mapping'][i].tolist());prot=protein_residue_reindex(seq_cache[seq],starts)
                ns=np.asarray([x['n_idx'] for x in r],dtype='int32');dist=Chem.GetDistanceMatrix(m)
                around=np.stack([atom[np.flatnonzero(dist[n]<=2)].mean(axis=0) for n in ns])
                local=np.stack([LOCAL.GetCountFingerprintAsNumPy(m,fromAtoms=[int(n)]).astype('float32') for n in ns])
                graph=molecular_features(s,MORGAN)[1];compact=np.r_[np.max(graph[:2048].reshape(8,256),axis=0),graph[2048:]]
                save(s,error='',smi=s,graph=graph,compact=compact,chem=atom.mean(axis=0),protein=prot.mean(axis=0),ns=ns,atom=atom[ns],env=around,prot=prot,local=local,
                     residue_json=json.dumps(r),seq=seq,cyclic=cycle,source_order_seq=''.join(z['aa'] for z in r),canonical_starts=starts,n_tokens=int(e['attention_mask'][i].sum()),atom_count=m.GetNumAtoms(),unknown_esm=sum(z['aa']=='X' for z in r))
            except ValueError as exc:save(s,error=str(exc))
        if off==0 or (off//batch+1)%20==0 or off+batch>=len(missing):print('Encoded',min(off+batch,len(missing)),len(missing),flush=True)
def demo(data,weights,cache,out):
    z=pd.read_csv(data).query("record_id=='CycPeptMPDB_record:1'").iloc[0]
    encode_bank([z.smiles],weights,cache,batch=1);x=read_bank(cache,z.smiles);r=json.loads(str(x['residue_json']));seq=str(x['source_order_seq']);roll=3
    a,i=rotation_indices(seq,True);b,j=rotation_indices(seq[roll:]+seq[:roll],True);assert a==b
    canonical=np.roll(x['prot'],-i[0],axis=0);rotated=np.roll(protein_residue_reindex(canonical,j),roll,axis=0)
    info=dict(record_id=z.record_id,source=z.source,target=z.target,upstream_sequence=z.sequence,smiles=z.smiles,residues=r,atom_numbering='0-based RDKit indices',
        backbone_cyclic=bool(x['cyclic']),projected_original_order=seq,canonical_esm_sequence=str(x['seq']),aligned_esm_rotation_max_abs_error=float(np.abs(x['prot']-rotated).max()),
        tensor_shapes={k:list(x[k].shape) for k in ['graph','chem','protein','atom','env','prot','local']},checkpoint_revisions={k:v[1] for k,v in SPEC.items()})
    out.mkdir(exist_ok=True,parents=True);(out/'record1.json').write_text(json.dumps(info,indent=2));(out/'example').mkdir(exist_ok=True)
    np.savez_compressed(out/'example/record1_pretrained_and_graph_inputs.npz',**x)
    from rdkit.Chem.Draw import rdMolDraw2D
    draw=rdMolDraw2D.MolDraw2DCairo(1100,850)
    for pos,z1 in enumerate(r,1):draw.drawOptions().atomLabels[z1['n_idx']]=f"N{z1['n_idx']} ({pos})"
    draw.DrawMolecule(Chem.MolFromSmiles(z.smiles),highlightAtoms=x['ns'].tolist());draw.FinishDrawing();(out/'record1.png').write_bytes(draw.GetDrawingText())
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--weights',type=Path,required=True);a.add_argument('--fetch',action='store_true');a.add_argument('--cache',type=Path,default=Path('.cache/encoders'))
    a.add_argument('--edits',type=Path);a.add_argument('--demo-data',type=Path);a.add_argument('--demo-out',type=Path,default=Path('results/model'));a.add_argument('--threads',type=int,default=2)
    o=a.parse_args();verify(o.weights,o.fetch)
    if o.demo_data:demo(o.demo_data,o.weights,o.cache,o.demo_out)
    if o.edits:
        x=pd.read_csv(o.edits);encode_bank(set(x.parent_smiles)|set(x.variant_smiles),o.weights,o.cache,threads=o.threads)
