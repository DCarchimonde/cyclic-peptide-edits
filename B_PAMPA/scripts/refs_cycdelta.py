"""Run native fixed-parent, zero-corrected CycDelta and a same-fold whole-network refit."""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import rdBase
from refs_assets import CYC, UM, UM_SHA, benchmark, sha


def obtain(e, up, cache):
    names = sorted(set(e.parent_smiles) | set(e.variant_smiles))
    records = {}
    failures = {}
    for i, s in enumerate(names):
        try:
            records[s] = up.parse_cyclic_peptide(s, i)
        except (ValueError, RuntimeError) as exc:
            failures[s] = str(exc)
    mons = sorted(
        {r.smiles for rec in records.values() for r in rec.residues} | {up.GLYCINE_SMILES}
    )
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(("\n".join(mons) + rdBase.rdkitVersion + UM_SHA).encode()).hexdigest()[:20]
    file = cache / f"monomers_{key}.npz"
    if file.is_file():
        with np.load(file, allow_pickle=False) as f:
            if mons != f["smiles"].tolist():
                raise ValueError("monomer cache identity mismatch")
            h = torch.tensor(f["unimol"])
            d = torch.tensor(f["descriptors"])
    else:
        from process_monomer_descriptors import calc_descriptors

        h = up.encode_unimol(mons, 64, False)
        d = torch.tensor([calc_descriptors(s) for s in mons], dtype=torch.float32)
        np.savez_compressed(file, smiles=np.array(mons), unimol=h.numpy(), descriptors=d.numpy())
    ix = {s: i for i, s in enumerate(mons)}
    rows = {s: torch.tensor([ix[r.smiles] for r in rec.residues]) for s, rec in records.items()}
    return records, rows, h, d, ix[up.GLYCINE_SMILES], failures


def make_graphs(records, idx, emb, props, gly, norm, up):
    return {
        s: up.make_graph(
            rec, "PAMPA", emb[idx[s]], emb[gly], up.normalize_descriptors(props[idx[s]], *norm)
        )
        for s, rec in records.items()
    }


@torch.inference_mode()
def scores(model, table, g, up):
    names = sorted(set(table.parent_smiles) | set(table.variant_smiles))
    enc = {}
    for i in range(0, len(names), 96):
        t = names[i : i + 96]
        arr = model._pool_graph(up.collate_graphs([g[s] for s in t]), len(t))
        enc.update(zip(t, arr))
    zero = model.out_mlp(torch.zeros_like(next(iter(enc.values())))).item()
    out = []
    for i in range(0, len(table), 128):
        v = table.iloc[i : i + 128]
        diff = torch.stack([enc[c] - enc[p] for p, c in zip(v.parent_smiles, v.variant_smiles)])
        out.extend((model.out_mlp(diff).reshape(-1) - zero).tolist())
    return np.array(out)


def normalizer(v):
    ok = torch.isfinite(v)
    mean = torch.nanmean(torch.where(ok, v, torch.nan), dim=0)
    if not torch.isfinite(mean).all():
        raise ValueError("missing training descriptor")
    var = torch.where(ok, (v - mean).square(), 0).sum(0) / ok.sum(0).clamp_min(1)
    return mean, torch.where(var.sqrt() >= 1e-6, var.sqrt(), 1.0)


def train(e, g, up, out):
    train = e[e.split.eq("train") & e.delta.notna()].reset_index(drop=True)
    valid = e[e.split.eq("valid") & e.delta.notna()].reset_index(drop=True)
    torch.manual_seed(0)
    model = up.CycGNN(d_emb=128, n_heads=4, dropout=0.25, num_gnn_layer=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    best = float("inf")
    hist = []
    for epoch in range(1, 81):
        model.train()
        for ids in torch.randperm(len(train)).split(32):
            b = train.iloc[ids.tolist()]
            y = torch.tensor(b.delta.to_numpy(), dtype=torch.float32)
            tensor = {
                "child": up.collate_graphs([g[s] for s in b.variant_smiles]),
                "parent": up.collate_graphs([g[s] for s in b.parent_smiles]),
                "y": y,
            }
            opt.zero_grad()
            loss = (model(tensor) - y).square().mean()
            loss.backward()
            opt.step()
        model.eval()
        mae = float(np.mean(np.abs(scores(model, valid, g, up) - valid.delta)))
        hist.append({"epoch": epoch, "valid_delta_mae": mae})
        if mae < best:
            best = mae
            be = epoch
            weights = copy.deepcopy(model.state_dict())
        if epoch - be >= 10:
            break
    model.load_state_dict(weights)
    model.eval()
    torch.save(weights, out / "cycdelta_refit.pt")
    pd.DataFrame(hist).to_csv(out / "cycdelta_learning_curve.csv", index=False)
    return model, dict(
        best_epoch=be, best_validation_mae=best, train_edges=len(train), valid_edges=len(valid)
    )


def run(a):
    torch.set_num_threads(a.threads)
    np.random.seed(0)
    torch.manual_seed(0)
    a.out.mkdir(parents=True, exist_ok=True)
    src = (a.external / "CycDelta").resolve()
    sys.path[:0] = [str(src / "smiles_delta_prediction"), str(src)]
    os.environ.setdefault("UNIMOL_WEIGHT_DIR", str((a.external / "unimol").resolve()))
    import predict as up

    e, checksum = benchmark(a.edits, a.support)
    r, i, h, d, gly, fail = obtain(e, up, a.cache)
    pd.DataFrame(list(fail.items()), columns=["smiles", "reason"]).to_csv(
        a.out / "cycdelta_unsupported.csv", index=False
    )
    bad = set(e.loc[~(e.parent_smiles.isin(r) & e.variant_smiles.isin(r)), "group_id"])
    e = e[~e.group_id.isin(bad)].reset_index(drop=True)
    info = dict(
        edits_sha256=sha(a.edits),
        common_support_sha256=checksum,
        accepted_options=len(e),
        rejected_groups=len(bad),
        rdkit=rdBase.rdkitVersion,
        cycdelta_commit=CYC,
        unimol_commit=UM,
        unimol_sha256=UM_SHA,
        fixed_reference="parent_smiles",
        zero_correction="f(hV-hP)-f(0)",
    )
    check = src / "checkpoints/best.ckpt"
    if sha(check) != "4f18ce6ec033b78ce9ae29279bbf95fd69cd5a950d90a1f2698a74fea908433a":
        raise ValueError("incorrect released weights")

    def save(tag, method, val, meta):
        pd.DataFrame(
            {"pair_id": e.pair_id, "split": e.split, "model": method, "pred_delta": val}
        ).to_csv(a.out / f"cycdelta_{tag}_predictions.csv", index=False)
        (a.out / f"cycdelta_{tag}_run.json").write_text(
            json.dumps({**info, **meta}, indent=2) + "\n"
        )

    native = up.load_model(check, torch.device("cpu"))
    g = make_graphs(
        r,
        i,
        h,
        d,
        gly,
        up.load_normalizer(src / "smiles_delta_prediction/descriptor_normalizer.json"),
        up,
    )
    result = scores(native, e, g, up)
    row = e.iloc[0]
    direct = up.predict_deltas(
        native, [g[row.variant_smiles]], g[row.parent_smiles], 1, torch.device("cpu")
    )[0]
    if abs(direct - result[0]) > 1e-4:
        raise ValueError("native prediction mismatch")
    save(
        "released",
        "CycDelta_released_reference_only",
        result,
        {"direct_api_difference": float(abs(direct - result[0]))},
    )
    del g, native
    t = e.loc[e.split.eq("train") & e.delta.notna()]
    ts = sorted(set(t.parent_smiles) | set(t.variant_smiles))
    norm = normalizer(torch.cat([d[i[s]] for s in ts]))
    np.savez_compressed(
        a.out / "cycdelta_refit_normalizer.npz", mean=norm[0].numpy(), std=norm[1].numpy()
    )
    g = make_graphs(r, i, h, d, gly, norm, up)
    m, meta = train(e, g, up, a.out)
    save("refit", "CycDelta_matched_refit", scores(m, e, g, up), meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--external", type=Path, required=True)
    p.add_argument("--edits", type=Path, default=Path("results/pairs/edits.csv"))
    p.add_argument("--support", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/references"))
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    run(p.parse_args())
