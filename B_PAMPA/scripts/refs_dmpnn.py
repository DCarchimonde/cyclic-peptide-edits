"""Train an absolute Chemprop directed-bond MPNN and difference the common-parent predictions."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from rdkit import Chem
from chemprop.data import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import BondMessagePassing, MeanAggregation
from refs_assets import benchmark, measured, sha


class WholeGraphRegressor(nn.Module):
    def __init__(self, atom_dim, bond_dim):
        super().__init__()
        self.messages = BondMessagePassing(
            d_v=atom_dim, d_e=bond_dim, d_h=128, depth=3, dropout=0.1
        )
        self.mean = MeanAggregation()
        self.readout = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

    def forward(self, graphs):
        g = BatchMolGraph(graphs)
        return self.readout(self.mean(self.messages(g), g.batch)).flatten()


def run(a):
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    e, cs = benchmark(a.edits, a.support)
    names = sorted(set(e.parent_smiles) | set(e.variant_smiles))
    f = SimpleMoleculeMolGraphFeaturizer()
    g = {s: f(Chem.MolFromSmiles(s)) for s in names}
    tr = measured(e[e.split.eq("train")])
    v = measured(e[e.split.eq("valid")])
    mu = float(tr.y.mean())
    sd = float(tr.y.std(ddof=0))
    y = torch.tensor((tr.y.to_numpy() - mu) / sd, dtype=torch.float32)
    yv = torch.tensor(v.y.to_numpy(), dtype=torch.float32)
    model = WholeGraphRegressor(*f.shape)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    def pred(names):
        with torch.no_grad():
            return torch.cat(
                [
                    model([g[s] for s in names[j : j + 64]]) * sd + mu
                    for j in range(0, len(names), 64)
                ]
            )

    best = float("inf")
    hist = []
    for ep in range(1, 81):
        model.train()
        for idx in torch.randperm(len(tr)).split(48):
            b = [g[s] for s in tr.iloc[idx.tolist()].smiles]
            opt.zero_grad()
            loss = (model(b) - y[idx]).square().mean()
            loss.backward()
            opt.step()
        model.eval()
        rmse = float((pred(v.smiles.tolist()) - yv).square().mean().sqrt())
        if rmse < best:
            best = rmse
            be = ep
            state = copy.deepcopy(model.state_dict())
        hist.append({"epoch": ep, "valid_rmse": rmse})
        if ep - be >= 10:
            break
    model.load_state_dict(state)
    model.eval()
    vals = dict(zip(names, pred(names).tolist()))
    t = e[["pair_id", "split"]].copy()
    t["model"] = "Chemprop_DMPNN_absolute_difference"
    t["pred_delta"] = [vals[v] - vals[p] for p, v in zip(e.parent_smiles, e.variant_smiles)]
    t.to_csv(a.out / "dmpnn_predictions.csv", index=False)
    pd.DataFrame(hist).to_csv(a.out / "dmpnn_learning_curve.csv", index=False)
    torch.save(state, a.out / "dmpnn_readout.pt")
    (a.out / "dmpnn_run.json").write_text(
        json.dumps(
            dict(
                chemprop="2.2.1",
                edits_sha256=sha(a.edits),
                common_support_sha256=cs,
                atom_features=f.shape[0],
                bond_features=f.shape[1],
                train_absolute_measurements=len(tr),
                valid_absolute_measurements=len(v),
                seed=42,
                train_y_mean=mu,
                train_y_sd=sd,
                best_epoch=be,
                best_valid_rmse=best,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edits", type=Path, default=Path("results/pairs/edits.csv"))
    p.add_argument("--support", type=Path)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", type=Path, default=Path("results/references"))
    run(p.parse_args())
