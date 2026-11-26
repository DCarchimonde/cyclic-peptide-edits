"""Run the native frozen Phase2 graph-biased encoder and matched validation-selected readout."""

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from refs_assets import PT, NCAA_SHA, benchmark, measured, sha


def features(smiles, source, cache, threads, batch):
    sys.path.insert(0, str(source.resolve()))
    from data.dataset import SingleInputDataset
    from data.dataloader import collate_and_trim_smiles_batch
    from data.vocab_manager import VocabManager
    from model.modules.smiles_encoder import EnhancedSMILESEncoder

    cfg = yaml.safe_load((source / "config/phase2/phase2_repro_sweep24_e0_b0.yaml").read_text())
    hp = cfg["student"]["smiles_encoder"]
    graph = cfg["student"]["graph_attention"]
    vocab = VocabManager(str(source / cfg["data"]["paths"]["vocab_smiles"]))
    if sha(source / PT) != NCAA_SHA:
        raise ValueError("wrong Phase2 checkpoint")
    cache.mkdir(parents=True, exist_ok=True)
    paths = {s: cache / (hashlib.sha256(s.encode()).hexdigest() + ".npy") for s in smiles}
    pending = sorted((s for s in smiles if not paths[s].is_file()), key=lambda s: (len(s), s))
    if any(len(s) > hp["max_length"] - 2 for s in smiles):
        raise ValueError("exceeds single-window payload")
    torch.set_num_threads(threads)
    if pending:
        state = torch.load(source / PT, mmap=True, weights_only=True, map_location="cpu")[
            "model_state_dict"
        ]
        model = EnhancedSMILESEncoder(
            vocab_size=hp["vocab_size"],
            d_model=hp["embed_dim"],
            nhead=hp["num_heads"],
            num_layers=hp["num_layers"],
            dim_feedforward=hp["embed_dim"] * 4,
            dropout=hp["dropout"],
            max_seq_len=hp["max_length"],
            graph_attention=graph,
        )
        prefix = "smiles_encoder."
        model.load_state_dict(
            {k[len(prefix) :]: w for k, w in state.items() if k.startswith(prefix)}, strict=True
        )
        del state
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "eval.csv"
            pd.DataFrame({"canonical_smiles": pending}).to_csv(table, index=False)
            ds = SingleInputDataset(
                str(table),
                vocab,
                smiles_max_length=hp["max_length"],
                apply_smiles_masking=False,
                graph_attention_config=graph,
                label_column=None,
                normalization_config=None,
                split_name="reference_inference",
            )
            ds.graph_cache_size = 8
            with torch.inference_mode():
                for start in range(0, len(ds), batch):
                    rows = [ds[j] for j in range(start, min(start + batch, len(ds)))]
                    b = collate_and_trim_smiles_batch(rows)
                    arr = (
                        model(
                            b["smiles_input_ids"],
                            b["smiles_attention_mask"],
                            graph_distances=b["graph_distances"],
                        )[:, 0, :]
                        .float()
                        .numpy()
                    )
                    for s, v in zip(pending[start : start + len(rows)], arr):
                        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as f:
                            np.save(f, v, allow_pickle=False)
                            written = Path(f.name)
                        written.replace(paths[s])
                    if start % 80 == 0:
                        print("ncAA-RepDistill", start + len(rows), "/", len(ds), flush=True)
    return {s: np.load(p, allow_pickle=False) for s, p in paths.items()}


def fit(e, feat, source, out, ed_path, sup_hash):
    sys.path.insert(0, str((source / "scripts").resolve()))
    from evaluate_regression import (
        build_mlp,
        standardize_fit,
        standardize_apply,
        train_eval_mlp_one_seed,
    )
    from torch.utils.data import DataLoader, TensorDataset

    tr = measured(e[e.split.eq("train")])
    va = measured(e[e.split.eq("valid")])

    def vec(ss):
        return np.stack([feat[s] for s in ss])

    _, mean, sd = standardize_fit(vec(tr.smiles))

    def inp(ss):
        return torch.tensor(standardize_apply(vec(ss), mean, sd), dtype=torch.float32)

    x, y = inp(tr.smiles), torch.tensor(tr.y.to_numpy(), dtype=torch.float32)
    xv, yv = inp(va.smiles), torch.tensor(va.y.to_numpy(), dtype=torch.float32)
    torch.manual_seed(42)
    m = build_mlp(x.shape[1], [256], 0.1)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.0001)
    loader = DataLoader(TensorDataset(x, y), batch_size=256, shuffle=True, drop_last=False)
    best = float("inf")
    hist = []
    for epoch in range(1, 401):
        m.train()
        for xx, yy in loader:
            opt.zero_grad()
            loss = (m(xx).flatten() - yy).square().mean()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            score = float((m(xv).flatten() - yv).square().mean().sqrt())
        if score < best - 1e-8:
            best = score
            be = epoch
            state = copy.deepcopy(m.state_dict())
        hist.append({"epoch": epoch, "valid_rmse": score})
        if epoch - be >= 40:
            break
    native = train_eval_mlp_one_seed(
        x,
        y,
        xv,
        yv,
        xv,
        yv.numpy(),
        hidden_dims=[256],
        dropout=0.1,
        lr=0.001,
        weight_decay=0.0001,
        epochs=400,
        batch_size=256,
        patience=40,
        seed=42,
    )
    if native.best_epoch + 1 != be or abs(native.test_metrics["rmse"] - best) > 1e-5:
        raise RuntimeError("validation mismatch")
    m.load_state_dict(state)
    m.eval()
    pred = {}
    names = sorted(feat)
    with torch.no_grad():
        for start in range(0, len(names), 256):
            n = names[start : start + 256]
            pred.update(zip(n, m(inp(n)).flatten().tolist()))
    result = e[["pair_id", "split"]].copy()
    result["model"] = "ncAA_RepDistill_matched_readout"
    result["pred_delta"] = [pred[v] - pred[p] for p, v in zip(e.parent_smiles, e.variant_smiles)]
    result.to_csv(out / "ncaa_predictions.csv", index=False)
    pd.DataFrame(hist).to_csv(out / "ncaa_learning_curve.csv", index=False)
    torch.save(state, out / "ncaa_readout.pt")
    np.savez_compressed(out / "ncaa_readout_scaler.npz", mean=mean, scale=sd)
    info = dict(
        edits_sha256=sha(ed_path),
        common_support_sha256=sup_hash,
        checkpoint=NCAA_SHA,
        feature="smiles_repr (CLS)",
        head_seed=42,
        train_measurements=len(tr),
        valid_measurements=len(va),
        best_epoch=be,
        best_valid_rmse=best,
        extracted_structures=len(feat),
        native_readout_validation_rmse=native.test_metrics["rmse"],
        upstream_supervised_head_reused=False,
    )
    (out / "ncaa_run.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--external", type=Path, required=True)
    p.add_argument("--edits", type=Path, default=Path("results/pairs/edits.csv"))
    p.add_argument("--support", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/references"))
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--features-only", action="store_true")
    p.add_argument("--shard")
    p.add_argument("--smiles-file", type=Path)
    a = p.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.smiles_file:
        if not a.features_only:
            p.error("--smiles-file requires --features-only")
        ss = sorted(set(a.smiles_file.read_text().splitlines()))
    else:
        e, h = benchmark(a.edits, a.support)
        ss = sorted(set(e.parent_smiles) | set(e.variant_smiles))
    if a.shard:
        if not a.features_only:
            p.error("--shard requires --features-only")
        i, n = map(int, a.shard.split("/"))
        ss = sorted(ss, key=lambda v: (len(v), v))[len(ss) * i // n : len(ss) * (i + 1) // n]
    source = a.external / "ncAA-RepDistill"
    f = features(ss, source, a.cache, a.threads, a.batch)
    if not a.features_only:
        fit(e, f, source, a.out, a.edits, h)
