"""Run one deterministic first D-MPNN training minibatch for a CPU-kernel sensitivity check."""

import argparse
from pathlib import Path
import json, os
import numpy as np, pandas as pd, torch
from rdkit import Chem
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from refs_dmpnn import WholeGraphRegressor
from refs_assets import measured, benchmark
from refs_repeat_audit import weight_hash, digest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--edits", type=Path, default=Path("results/pairs/edits.csv"))
parser.add_argument("--support", type=Path, default=Path("results/model/common_support.csv"))
parser.add_argument("--out-prefix", type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
torch.manual_seed(42)
np.random.seed(42)
e, _ = benchmark(args.edits, args.support)
t = measured(e[e.split == "train"])
f = SimpleMoleculeMolGraphFeaturizer()
m = WholeGraphRegressor(*f.shape)
init = weight_hash(m)
opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.0001)
ids = torch.randperm(len(t))[:48].tolist()
g = [f(Chem.MolFromSmiles(s)) for s in t.iloc[ids].smiles]
y = torch.tensor(((t.y - t.y.mean()) / t.y.std(ddof=0)).to_numpy(), dtype=torch.float32)[ids]
opt.zero_grad()
pred = m(g)
loss = (pred - y).square().mean()
loss.backward()
grad = np.concatenate([p.grad.detach().numpy().ravel() for p in m.parameters()])
opt.step()
d = dict(
    initial_weight_hash=init,
    first_batch_hash=digest([t.iloc[i].smiles for i in ids]),
    updated_weight_hash=weight_hash(m),
    cpu_mkl_compat=os.environ.get("MKL_CBWR", "default"),
    loss=float(loss.item()),
)
p = args.out_prefix
p.parent.mkdir(parents=True, exist_ok=True)
p.with_suffix(".json").write_text(json.dumps(d))
np.savez_compressed(p.with_suffix(".npz"), pred=pred.detach().numpy(), grad=grad)
