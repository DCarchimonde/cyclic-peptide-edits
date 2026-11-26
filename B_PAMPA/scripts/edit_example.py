"""Render the prespecified observed parent/two-site result; trace the actual score difference."""

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from encoder_cache import read_bank
from fit_edits import make_head, pair_features, scaling

GROUP = "grp_01762f2f247e7d8e53"
COLORS = [(0.22, 0.66, 0.70), (0.94, 0.62, 0.24)]


def draw_structures(rows, out):
    parent = Chem.MolFromSmiles(rows.iloc[0].parent_smiles)
    rdDepictor.Compute2DCoords(parent)
    mols = [parent]
    colors = [{int(r.parent_n_idx): COLORS[i] for i, r in enumerate(rows.itertuples())}]
    for i, r in enumerate(rows.itertuples()):
        m = Chem.MolFromSmiles(r.variant_smiles)
        rdDepictor.GenerateDepictionMatching2DStructure(
            m, parent, list(enumerate(json.loads(r.parent_to_variant_atom_map)))
        )
        mols.append(m)
        colors.append(
            {int(r.variant_n_idx): COLORS[i], int(r.variant_methyl_idx): COLORS[i]}
        )
    canvas = Image.new("RGB", (2100, 480), "white")
    files = ["real_parent.png", "real_variant_1.png", "real_variant_2.png"]
    font = ImageFont.truetype("DejaVuSans.ttf", 26)
    captions = ["Parent " + rows.iloc[0].parent_record_ids.split(":")[-1]] + [
        r.variant_record_ids.split(":")[-1] + f"   editable N{r.parent_n_idx}"
        for r in rows.itertuples()
    ]
    for i, (m, cp, fn, caption) in enumerate(zip(mols, colors, files, captions)):
        d = rdMolDraw2D.MolDraw2DCairo(1000, 590)
        opts = d.drawOptions()
        opts.padding = 0.04
        opts.bondLineWidth = 2.8
        opts.highlightRadius = 0.6
        d.DrawMolecule(m, highlightAtoms=list(cp), highlightAtomColors=cp)
        d.FinishDrawing()
        (out / fn).write_bytes(d.GetDrawingText())
        img = Image.open(out / fn).convert("RGB")
        img.thumbnail((690, 425), Image.Resampling.LANCZOS)
        canvas.paste(img, (700 * i + (700 - img.width) // 2, 0))
        ImageDraw.Draw(canvas).text(
            (350 + 700 * i, 448), caption, anchor="mm", font=font, fill="#1b2735"
        )
    canvas.save(out / "parent_and_variants.png")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--edits", type=Path, default=Path("results/pairs/edits.csv"))
    a.add_argument("--model-dir", type=Path, default=Path("results/model"))
    a.add_argument("--cache", type=Path, default=Path(".cache/encoders"))
    o = a.parse_args()
    torch.set_num_threads(1)
    out = o.model_dir / "example"
    out.mkdir(exist_ok=True, parents=True)
    e = pd.read_csv(o.edits)
    pred = pd.read_csv(o.model_dir / "predictions.csv")
    rows = e[e.group_id == GROUP].sort_values("parent_n_idx")
    rows.merge(pred, on="pair_id").to_csv(
        out / "demonstration_measurements_and_predictions.csv", index=False
    )
    draw_structures(rows, out)
    bank = {
        s: read_bank(o.cache, s)
        for s in set(rows.parent_smiles) | set(rows.variant_smiles)
    }
    w = torch.load(
        o.model_dir / "fitted/explicit_edit.pt", map_location="cpu", weights_only=True
    )
    net = make_head(w["weight_dims"])
    net.load_state_dict(w["state"])
    net.eval()
    x = np.stack([pair_features(r, bank, "explicit_edit") for r in rows.itertuples()])
    xt = scaling(x, {k: v.numpy() for k, v in w["scale"].items()})
    np.savez_compressed(
        out / "two_site_model_inputs.npz",
        pair_ids=rows.pair_id.to_numpy(dtype=str),
        parent_n_idx=rows.parent_n_idx.to_numpy(),
        variant_n_idx=rows.variant_n_idx.to_numpy(),
        unscaled_explicit_input=x,
        scaled_explicit_input=xt.numpy(),
        measured_delta=rows.delta.to_numpy(),
        measured_variant=rows.variant_y.to_numpy(),
    )
    diff = xt[1] - xt[0]
    steps = 256
    interp = (
        xt[0].unsqueeze(0)
        + (torch.arange(steps).view(-1, 1) + 0.5) / steps * diff.unsqueeze(0)
    ).requires_grad_()
    grad = torch.autograd.grad(net(interp).sum(), interp)[0].mean(dim=0)
    attr = (grad * diff).detach().numpy()
    with torch.inference_mode():
        scores = net(xt).numpy()
    blocks = {}
    offset = 0
    for name, width in [
        ("molecule_chem", 384),
        ("complete_graph", 268),
        ("global_protein", 320),
        ("edit_atom", 384),
        ("edit_atom_neighbourhood", 384),
        ("edit_chiral_graph", 256),
        ("aligned_residue_context", 320),
    ]:
        for lab in ["parent", "variant_minus_parent"]:
            blocks[name + "_" + lab] = float(np.sum(attr[offset : offset + width]))
            offset += width
    assert offset == len(attr)
    info = dict(
        group_id=GROUP,
        structure_order=rows.variant_record_ids.tolist(),
        frozen_predicted_delta=scores.tolist(),
        observed_delta=rows.delta.tolist(),
        model_preference_score_gap_second_minus_first=float(scores[1] - scores[0]),
        measured_gap_second_minus_first=float(
            rows.variant_y.iloc[1] - rows.variant_y.iloc[0]
        ),
        integrated_gradient_steps=steps,
        local_score_decomposition_second_minus_first=blocks,
        score_decomposition_error=float(sum(blocks.values()) - (scores[1] - scores[0])),
    )
    (out / "score_trace.json").write_text(json.dumps(info, indent=2))
    tests = pd.read_csv(o.model_dir / "per_group.csv").query(
        "model=='explicit_edit' and split=='test'"
    )
    ok = tests[(tests.hit == 1) & (tests.sites == 2)]
    z = e[
        e.group_id.isin(ok.group_id)
        & e.parent_y.notna()
        & e.variant_residue_1based.notna()
    ]
    span = z.groupby("group_id").variant_y.agg(["count", "min", "max"])
    eligible = span[
        (span["count"] == 2) & (span["max"] - span["min"] > 0.5)
    ].sort_index()
    cases = {
        "locked_example": GROUP,
        "largest_regret_heldout": tests.sort_values(
            ["regret", "group_id"], ascending=[False, True]
        )
        .iloc[0]
        .group_id,
    }
    if len(eligible):
        cases["first_correct_with_two_located_sites_and_measured_gap_above_0.5"] = (
            eligible.index[0]
        )
    pd.concat(
        [
            e[e.group_id == gid]
            .merge(pred[pred.model == "explicit_edit"], on="pair_id")
            .assign(case=k)
            for k, gid in cases.items()
        ]
    ).to_csv(out / "contrasting_examples.csv", index=False)


if __name__ == "__main__":
    main()
