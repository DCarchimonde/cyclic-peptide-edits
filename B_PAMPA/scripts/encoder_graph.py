"""Derive ordered residue and exact nitrogen identities from the stereochemical graph."""
from functools import lru_cache
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
ALPHA=Chem.MolFromSmarts('[N;X3,X4+]-[C;X4]-[C;X3](=[O;X1])')
MORGAN=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
LOCAL=rdFingerprintGenerator.GetMorganGenerator(radius=3,fpSize=256,includeChirality=True)
ATOM=re.compile(r'\[[^\]]+\]|Cl|Br|[BCNOPSFIbcnops*]')
def order(m):
    ns={}
    for n,a,c,o in m.GetSubstructMatches(ALPHA):ns.setdefault(a,[]).append((n,a,c))
    motifs=[]
    for a,v in ns.items():
        if len(v)>1:
            v=[t for t in v if any(b.GetAtomicNum()==6 and any(x.GetAtomicNum()==8 and m.GetBondBetweenAtoms(b.GetIdx(),x.GetIdx()).GetBondTypeAsDouble()==2 for x in b.GetNeighbors()) for b in m.GetAtomWithIdx(t[0]).GetNeighbors())]
        if len(v)!=1:raise ValueError('ambiguous_alpha_nitrogen')
        motifs.append(v[0])
    nn={n:(n,a,c) for n,a,c in motifs}
    if not nn:raise ValueError('no_alpha_amino_acid_motifs')
    if len(nn)!=len(motifs):raise ValueError('alpha_nitrogen_maps_multiple_residues')
    follow={}
    for n,a,c in motifs:
        nxt=[b.GetIdx() for b in m.GetAtomWithIdx(c).GetNeighbors() if b.GetIdx() in nn and b.GetIdx()!=n]
        if len(nxt)>1:raise ValueError('branched_peptide_order')
        if nxt:follow[n]=nxt[0]
    first=set(nn)-set(follow.values())
    if len(first)>1:raise ValueError('disconnected_alpha_backbone')
    cycle=not first
    n=min(nn) if cycle else next(iter(first));inds=[]
    while n not in inds:
        inds.append(n)
        if n not in follow:break
        n=follow[n]
    if len(inds)!=len(nn) or (cycle and n!=inds[0]):raise ValueError('nonlinear_or_incomplete_backbone')
    return [nn[n] for n in inds],cycle,follow
def fragments(m,refs,follow):
    carbonyl={n:c for n,a,c in refs};cut=Chem.RWMol(m)
    for n,z in follow.items():cut.RemoveBond(carbonyl[n],z)
    member={i:part for part in Chem.GetMolFrags(cut) for i in part};out=[]
    for n,a,c in refs:
        ids=set(member[n]);me=[]
        if len(ids.intersection(carbonyl))!=1:out.append(('X','X',sorted(ids),[]));continue
        me=[b.GetIdx() for b in m.GetAtomWithIdx(n).GetNeighbors() if b.GetDegree()==1 and b.GetAtomicNum()==6 and b.GetTotalNumHs()==3 and b.GetIdx()!=a]
        ids.difference_update(me)
        for b in m.GetAtomWithIdx(c).GetNeighbors():
            if b.GetDegree()==1 and b.GetAtomicNum()==8 and m.GetBondBetweenAtoms(c,b.GetIdx()).GetBondTypeAsDouble()==1:ids.discard(b.GetIdx())
        ids=sorted(ids);f=Chem.MolFromSmiles(Chem.MolFragmentToSmiles(m,atomsToUse=ids,isomericSmiles=True))
        if f is None:raise ValueError('invalid_residue_fragment')
        out.append((Chem.MolToSmiles(f,isomericSmiles=False),Chem.MolToSmiles(f,isomericSmiles=True),ids,me))
    return out
@lru_cache(maxsize=1)
def alphabet():
    out={}
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        m=Chem.MolFromSequence('A'+aa+'A');o,_,f=order(m);identity,stereo,*_=fragments(m,o,f)[1];out[identity]=(aa,stereo)
    assert len(out)==20
    return out
def rotation_indices(seq,cyclic):
    if not cyclic:return seq,[0]
    rotations=[seq[i:]+seq[:i] for i in range(len(seq))];canonical=min(rotations)
    return canonical,[i for i,v in enumerate(rotations) if v==canonical]
def protein_residue_reindex(h,starts):return np.stack([np.roll(h,i,axis=0) for i in starts]).mean(axis=0)
def molecule_data(s):
    m=Chem.MolFromSmiles(s)
    if m is None:raise ValueError('invalid_smiles')
    spans=[v.span() for v in ATOM.finditer(s)]
    if len(spans)!=m.GetNumAtoms():raise ValueError('atom_index_mismatch')
    refs,cyclic,follow=order(m);r=[]
    for (n,a,c),(sig,stereo,ids,me) in zip(refs,fragments(m,refs,follow)):
        aa,ref=alphabet().get(sig,('X',None))
        r.append(dict(n_idx=n,ca_idx=a,carbonyl_idx=c,aa=aa,stereo_matches_standard_projection=(ref==stereo) if aa!='X' else None,n_methyl=bool(me),atom_idx=ids,n_methyl_atom_idx=me))
    if len(r)<2:raise ValueError('less_than_two_residues')
    seq,starts=rotation_indices(''.join(x['aa'] for x in r),cyclic)
    return m,spans,r,cyclic,seq,starts
