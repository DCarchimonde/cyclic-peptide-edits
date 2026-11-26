"""Export an atom-to-residue graph for CsA, the locked Kelly example and a cross-protocol control."""

import ast
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from pair_data import ROOT, read_csv, write_csv, canonical


def record_one(raw, out):
    record = next(r for r in raw if r["CycPeptMPDB_ID"] == "1")
    mol = Chem.MolFromSmiles(record["SMILES"])
    seq = ast.literal_eval(record["Sequence"])
    # Raw SMILES visits each peptide N in the order of HELM starting at Abu.
    nitrogens = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N"]
    if len(nitrogens) != len(seq) or seq[0] != "Abu":
        raise ValueError("Unexpected record 1 topology")
    cuts = {}
    backbone = set(max(mol.GetRingInfo().AtomRings(), key=len))
    for i in nitrogens:
        for neighbor in mol.GetAtomWithIdx(i).GetNeighbors():
            j = neighbor.GetIdx()
            if j in backbone and any(
                b.GetBondType() == Chem.BondType.DOUBLE
                and b.GetOtherAtom(neighbor).GetSymbol() == "O"
                for b in neighbor.GetBonds()
            ):
                cuts[i] = j
    if set(cuts) != set(nitrogens):
        raise ValueError("Peptide closure unresolved")
    cutmol = Chem.RWMol(mol)
    for n, c in cuts.items():
        cutmol.RemoveBond(n, c)
    owner = {}
    res = []
    for fragment in Chem.GetMolFrags(cutmol, asMols=False):
        n = next(i for i in fragment if i in nitrogens)
        pos = nitrogens.index(n) + 1
        c = cuts[nitrogens[pos % len(seq)]]
        if c not in fragment:
            raise ValueError("Residue order or closure mismatch")
        for atom in fragment:
            owner[atom] = pos
        res.append(
            dict(
                residue=pos,
                token=seq[pos - 1],
                nitrogen_idx=n,
                carbonyl_idx=c,
                source_smiles_atom_indices=" ".join(map(str, fragment)),
            )
        )
    atoms = []
    arr = []
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        x = [
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetTotalDegree(),
            atom.GetTotalNumHs(),
            int(atom.GetIsAromatic()),
            int(atom.GetChiralTag()),
            int(atom.IsInRing()),
        ]
        arr.append(x)
        atoms.append(
            dict(
                atom_idx=i,
                element=atom.GetSymbol(),
                residue=owner[i],
                token=seq[owner[i] - 1],
                chirality=str(atom.GetChiralTag()),
            )
        )
    edges = []
    features = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = [bond.GetBondTypeAsDouble(), int(bond.GetStereo()), int(bond.IsInRing())]
        edges.extend([[a, b], [b, a]])
        features.extend([feat, feat])
    write_csv(out / "record1_atoms.csv", atoms)
    write_csv(out / "record1_residues.csv", sorted(res, key=lambda r: r["residue"]))
    np.savez_compressed(
        out / "record1_graph.npz",
        x=np.asarray(arr, dtype=np.int64),
        edge_index=np.asarray(edges, dtype=np.int64).T,
        edge_attr=np.asarray(features, dtype=np.float32),
        atom_to_residue=np.asarray(
            [owner[i] for i in range(len(atoms))], dtype=np.int64
        ),
        y=np.float32(record["PAMPA"]),
    )
    info = dict(
        record_id="CycPeptMPDB_record:1",
        source=record["Source"],
        compound=record["Original_Name_in_Source_Literature"],
        pampa=float(record["PAMPA"]),
        unit="log10(cm/s)",
        raw_smiles=record["SMILES"],
        tokens=seq,
        atom_features=[
            "atomic number",
            "formal charge",
            "total degree",
            "total H",
            "aromatic",
            "RDKit chiral tag enum",
            "in ring",
        ],
        edge_features=["bond order", "RDKit double-bond stereo enum", "in ring"],
        shapes={
            "x": [len(arr), 7],
            "edge_index": [2, len(edges)],
            "edge_attr": [len(edges), 3],
        },
        closure=[cuts[nitrogens[0]], nitrogens[0]],
        closure_description="Last source monomer carbonyl → first monomer Abu N; both atom indices use raw SMILES.",
    )
    (out / "record1_graph.json").write_text(json.dumps(info, indent=2) + "\n")
    for a in mol.GetAtoms():
        if a.GetIdx() in nitrogens:
            a.SetProp("atomNote", str(owner[a.GetIdx()]))
    Draw.MolToFile(
        mol,
        str(out / "record1_structure.png"),
        size=(1000, 620),
        legend="CycPeptMPDB record 1 | Cyclosporin A | measured log10 P = -6.60",
    )


def kelly_example(edits, out):
    rows = [r for r in edits if r["group_id"] == "grp_01762f2f247e7d8e53"]
    rows.sort(key=lambda r: int(r["parent_n_idx"]))
    if len(rows) != 2 or not all(
        r["parent_record_ids"] == "CycPeptMPDB_record:6618" for r in rows
    ):
        raise ValueError("Frozen Kelly example not found")
    mols = [Chem.MolFromSmiles(rows[0]["parent_smiles"])] + [
        Chem.MolFromSmiles(r["variant_smiles"]) for r in rows
    ]
    rdDepictor.Compute2DCoords(mols[0])
    colors = [(0.88, 0.60, 0.12), (0.12, 0.61, 0.64)]
    highlights = [[int(r["parent_n_idx"]) for r in rows]]
    color_maps = [dict(zip(highlights[0], colors))]
    table = []
    legends = ["Parent #6618 | measured -7.24"]
    for j, r in enumerate(rows, 1):
        mapping = json.loads(r["parent_to_variant_atom_map"])
        rdDepictor.GenerateDepictionMatching2DStructure(
            mols[j], mols[0], list(enumerate(mapping))
        )
        sites = [int(r["variant_n_idx"]), int(r["variant_methyl_idx"])]
        highlights.append(sites)
        color_maps.append({x: colors[j - 1] for x in sites})
        rid = r["variant_record_ids"].split(":")[1]
        label = "D-Leu4" if rid == "6645" else "L-Leu6"
        legends.append(
            f"{label} N-Me #{rid} | measured {r['variant_y']} | delta +{r['delta']}"
        )
        table.append(
            dict(
                site_label=label,
                parent_record_id=r["parent_record_ids"],
                variant_record_id=r["variant_record_ids"],
                parent_n_idx=r["parent_n_idx"],
                variant_n_idx=r["variant_n_idx"],
                methyl_idx=r["variant_methyl_idx"],
                measured_parent_log10P=r["parent_y"],
                measured_variant_log10P=r["variant_y"],
                measured_delta=r["delta"],
                protocol_id=r["protocol_id"],
                group_id=r["group_id"],
                pair_id=r["pair_id"],
            )
        )
    for i, mol in enumerate(mols):
        mol.SetProp("_Name", legends[i])
    writer = Chem.SDWriter(str(out / "kelly_site_choice.sdf"))
    for m in mols:
        writer.write(m)
    writer.close()
    draw = rdMolDraw2D.MolDraw2DCairo(1620, 485, 540, 485)
    draw.drawOptions().legendFontSize = 17
    draw.DrawMolecules(
        mols, legends=legends, highlightAtoms=highlights, highlightAtomColors=color_maps
    )
    draw.FinishDrawing()
    (out / "kelly_site_choice.png").write_bytes(draw.GetDrawingText())
    write_csv(out / "kelly_site_choice.csv", table)


def cross_protocol(raw, out):
    wang = {
        r["source_compound"]: r["pampa_log10_cm_s"]
        for r in read_csv(ROOT / "configs/wang2015_pampa.csv")
        if r["pampa_log10_cm_s"]
    }
    source = {
        canonical(Chem.MolFromSmiles(r["SMILES"])): r
        for r in raw
        if r["Source"] == "2015_Wang"
        and r["Original_Name_in_Source_Literature"] in wang
    }
    matched = []
    for r in raw:
        if r["Source"] != "2020_Hosono" or (
            r["Original_Name_in_Source_Literature"] != "CP1"
            and not r["Original_Name_in_Source_Literature"].startswith("MP")
        ):
            continue
        key = canonical(Chem.MolFromSmiles(r["SMILES"]))
        w = source.get(key)
        if w:
            matched.append(
                dict(
                    hosono_name=r["Original_Name_in_Source_Literature"],
                    hosono_id=r["CycPeptMPDB_ID"],
                    wang_name=w["Original_Name_in_Source_Literature"],
                    wang_id=w["CycPeptMPDB_ID"],
                    hosono_log10P=float(r["PAMPA"]),
                    wang_si_log10P=float(wang[w["Original_Name_in_Source_Literature"]]),
                    smiles=key,
                )
            )
    p = next(r for r in matched if r["hosono_name"] == "CP1")
    for r in matched:
        r["hosono_delta"] = round(r["hosono_log10P"] - p["hosono_log10P"], 3)
        r["wang_delta"] = round(r["wang_si_log10P"] - p["wang_si_log10P"], 3)
    write_csv(out / "hosono_wang_same_graphs.csv", matched)


if __name__ == "__main__":
    rows = read_csv(ROOT / "data/raw/cycpeptmpdb_all.csv")
    out = ROOT / "results/pairs/examples"
    out.mkdir(exist_ok=True)
    record_one(rows, out)
    kelly_example(read_csv(ROOT / "results/pairs/edits.csv"), out)
    cross_protocol(rows, out)
