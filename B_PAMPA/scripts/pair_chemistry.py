"""Reversible atom-aligned edits on connected, stereospecified cyclic peptide graphs."""

import csv
import hashlib
import json
from pathlib import Path
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
EDIT = Chem.MolFromSmarts("[C;X3](=[O])-[N;H0;X3;+0](-[CH3;D1;!R])-[C;X4]-[C;X3](=[O])")


def identity(text, prefix="mol"):
    return prefix + "_" + hashlib.sha256(text.encode()).hexdigest()[:18]


def canonical(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames or (list(rows[0]) if rows else []),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def eligible(mol):
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return "invalid_or_disconnected"
    if not any(len(r) >= 9 for r in mol.GetRingInfo().AtomRings()):
        return "missing_macrocycle"
    if any(
        x.specified == Chem.StereoSpecified.Unspecified
        for x in Chem.FindPotentialStereo(mol)
    ):
        return "unspecified_stereochemistry"
    return ""


def edits_for(smiles):
    mol = Chem.MolFromSmiles(smiles)
    index = {}
    methyls = set()
    for _, _, nitrogen, methyl, _, _, _ in mol.GetSubstructMatches(EDIT):
        methyls.add(methyl)
        changed = Chem.RWMol(mol)
        changed.RemoveAtom(methyl)
        parent = changed.GetMol()
        Chem.SanitizeMol(parent)
        key = canonical(parent)
        output_indices = json.loads(parent.GetProp("_smilesAtomOutputOrder"))
        atom_map = [a + int(a >= methyl) for a in output_indices]
        parent_n = atom_map.index(nitrogen)
        rebuilt = Chem.RWMol(Chem.MolFromSmiles(key))
        carbon = rebuilt.AddAtom(Chem.Atom(6))
        rebuilt.AddBond(parent_n, carbon, Chem.BondType.SINGLE)
        Chem.SanitizeMol(rebuilt)
        if canonical(rebuilt) != smiles:
            raise ValueError("Methyl add-back failed")
        edit = [parent_n, nitrogen, methyl, atom_map]
        if key not in index or edit < index[key]:
            index[key] = edit
    stripped = Chem.RWMol(mol)
    for i in sorted(methyls, reverse=True):
        stripped.RemoveAtom(i)
    Chem.SanitizeMol(stripped)
    return index, canonical(stripped)


def graph_index(path, cache):
    tag = [
        hashlib.sha256(path.read_bytes()).hexdigest(),
        rdBase.rdkitVersion,
        "reversible-methyl-v2",
    ]
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["tag"] == tag:
            return saved
    rows = [r for r in read_csv(path) if r["PAMPA"] and r["SMILES"]]
    graphs = {}
    states = {}
    edits = {}
    backbones = {}
    for count, row in enumerate(rows, 1):
        token = row["SMILES"]
        if token not in states:
            mol = Chem.MolFromSmiles(token)
            states[token] = eligible(mol)
            if not states[token]:
                s = canonical(mol)
                graphs[token] = s
                if s not in edits:
                    edits[s], backbones[s] = edits_for(s)
        if count % 1500 == 0:
            print("Indexed", count, "of", len(rows), flush=True)
    result = dict(
        tag=tag, graphs=graphs, states=states, edits=edits, backbones=backbones
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".partial")
    tmp.write_text(json.dumps(result, separators=(",", ":")))
    tmp.replace(cache)
    return result


if __name__ == "__main__":
    import sys

    graph_index(Path(sys.argv[1]), Path(sys.argv[2]))
