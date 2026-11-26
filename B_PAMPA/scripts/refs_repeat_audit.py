"""Trace component hashes and check same-host repeated predictions without using test labels for selection."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from refs_assets import benchmark, measured, sha

NAMES = ("cycdelta_released", "cycdelta_refit", "dmpnn", "ncaa")


def digest(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def weight_hash(model):
    h = hashlib.sha256()
    for name, v in sorted(model.state_dict().items()):
        a = v.detach().cpu().contiguous().numpy()
        h.update(
            name.encode() + repr(a.shape).encode() + str(a.dtype).encode() + a.tobytes()
        )
    return h.hexdigest()


def compare_first(t, pred, truth):
    merged = t[t.split.eq("test")].merge(
        pred[["pair_id", "pred_delta"]], on="pair_id", validate="one_to_one"
    )
    parents = []
    for _, group in merged.groupby("group_id"):
        if len(group) < 2:
            continue
        sel = np.isclose(group.pred_delta, group.pred_delta.max(), rtol=0, atol=1e-7)
        best = np.isclose(group.variant_y, group.variant_y.max(), rtol=0, atol=1e-7)
        parents.append(
            (group.variant_y.max() - group.variant_y[sel].mean(), best[sel].mean())
        )
    av = np.mean(parents, axis=0)
    q = merged.dropna(subset=["delta"])
    score = [float(av[0]), float(av[1]), float((q.delta - q.pred_delta).abs().mean())]
    return {
        "rerun": score,
        "first_run_rounded": truth,
        "rerun_minus_first_rounded": (np.array(score) - truth).tolist(),
    }


def run(a):
    torch.set_num_threads(2)
    t, h = benchmark(a.edits, a.support)
    old = json.loads((a.out / "verified_first_run.json").read_text())
    tr = measured(t[t.split.eq("train")])
    va = measured(t[t.split.eq("valid")])
    ids = lambda e: (
        e.protocol.astype(str) + "\t" + e.smiles + "\t" + e.y.astype(str)
    ).tolist()
    dtr = t[t.split.eq("train") & t.delta.notna()]
    dva = t[t.split.eq("valid") & t.delta.notna()]
    info = {
        "inputs": {
            "frozen_edits": sha(a.edits),
            "support": h,
            "ordered_train_delta_pair_ids": digest(dtr.pair_id),
            "ordered_valid_delta_pair_ids": digest(dva.pair_id),
            "ordered_train_absolute_measurements": digest(ids(tr)),
            "ordered_valid_absolute_measurements": digest(ids(va)),
        },
        "environment": {
            "cpu": next(
                s.split(":", 1)[-1].strip()
                for s in Path("/proc/cpuinfo").read_text().splitlines()
                if s.startswith("model name")
            ),
            "earlier_cpu": "Intel Xeon Platinum 8573C",
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "threads": 2,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "component_sha256": {},
        "fixed_host_repeat": {},
        "against_first_run": {},
        "scope": "Original per-batch RNG states, first-epoch orders, and fitted tensors did not survive the old host. Loss of original machine prevents an isolated hardware-causality test. The fixed-host repeat verifies current reconstructed programs.",
    }
    for p in (
        "torch-geometric",
        "pytorch-lightning",
        "torchmetrics",
        "scikit-learn",
        "numpy",
        "pandas",
        "rdkit",
        "chemprop",
        "unimol-tools",
        "numba",
    ):
        try:
            info["environment"][p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pass
    for name in [
        "CycDelta/GNN.py",
        "CycDelta/smiles_delta_prediction/predict.py",
        "CycDelta/smiles_delta_prediction/descriptor_normalizer.json",
        "CycDelta/checkpoints/best.ckpt",
        "ncAA-RepDistill/config/phase2/phase2_repro_sweep24_e0_b0.yaml",
        "ncAA-RepDistill/scripts/evaluate_regression.py",
        "ncAA-RepDistill/data/raw/adaptation/vocab_smiles.txt",
        "unimol/mol_pre_all_h_220816.pt",
        "unimol/mol.dict.txt",
    ]:
        path = a.external / name
        info["component_sha256"][name] = sha(path)
    sys.path.insert(0, str((a.external / "CycDelta").resolve()))
    from GNN import CycGNN
    from refs_dmpnn import WholeGraphRegressor
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

    torch.manual_seed(0)
    m = CycGNN(d_emb=128, n_heads=4, dropout=0.25, num_gnn_layer=2)
    init_c = weight_hash(m)
    order_c = torch.randperm(len(dtr)).tolist()
    torch.manual_seed(42)
    f = SimpleMoleculeMolGraphFeaturizer()
    m = WholeGraphRegressor(*f.shape)
    init_d = weight_hash(m)
    order_d = torch.randperm(len(tr)).tolist()
    src = (a.external / "ncAA-RepDistill").resolve()
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(src / "scripts"))
    from evaluate_regression import build_mlp

    torch.manual_seed(42)
    head = build_mlp(768, [256], 0.1)
    init_n = weight_hash(head)
    head_batches = torch.utils.data.DataLoader(
        torch.arange(len(tr)), batch_size=256, shuffle=True
    )
    order_n = torch.cat(list(head_batches)).tolist()
    info["reconstructed_initialization_and_batch_trace"] = {
        "cycdelta_seed": 0,
        "cycdelta_initial_tensors": init_c,
        "cycdelta_epoch_one_shuffled_pair_ids": digest(dtr.iloc[order_c].pair_id),
        "cycdelta_first_batch_n": 32,
        "dmpnn_seed": 42,
        "dmpnn_initial_tensors": init_d,
        "dmpnn_epoch_one_shuffled_observations": digest(np.array(ids(tr))[order_d]),
        "dmpnn_first_batch_n": 48,
        "ncaa_head_seed": 42,
        "ncaa_head_first_batch_n": 256,
        "ncaa_head_initial_tensors": init_n,
        "ncaa_epoch_one_shuffled_observations": digest(np.array(ids(tr))[order_n]),
        "historical_comparison": "seed, input table, split, training/validation sample counts, model size, optimizer, stopping target and patience match retained records. The old initialization tensors and shuffled record IDs were not separately hashed.",
    }
    labels = {
        "cycdelta_released": "CycDelta_released_reference_only",
        "cycdelta_refit": "CycDelta_matched_refit",
        "dmpnn": "Chemprop_DMPNN_absolute_difference",
        "ncaa": "ncAA_RepDistill_matched_readout",
    }
    curve_rows = []
    for name in NAMES:
        out = a.out / (name + "_predictions.csv")
        rep = a.repeat / (name + "_predictions.csv")
        if not out.exists():
            continue
        p = pd.read_csv(out)
        info["against_first_run"][name] = compare_first(
            t, p, old["metrics"][labels[name]]
        )
        target = old["original_csv_sha256"][out.name]
        info["against_first_run"][name].update(
            first_csv_sha256=target,
            current_csv_sha256=sha(out),
            bytes_match_first=target == sha(out),
        )
        if rep.exists():
            q = pd.read_csv(rep)
            m = p.merge(
                q,
                on=["pair_id", "split", "model"],
                validate="one_to_one",
                suffixes=("_a", "_b"),
            )
            if len(m) != len(q) or len(m) != len(p):
                raise ValueError("repeat index mismatch")
            a1 = m.pred_delta_a.to_numpy()
            a2 = m.pred_delta_b.to_numpy()
            info["fixed_host_repeat"][name] = dict(
                rows=len(m),
                max_abs_delta_prediction_change=float(np.max(np.abs(a1 - a2))),
                csv_byte_identity=sha(out) == sha(rep),
                repeat_sha256=sha(rep),
            )
        curve = a.out / (name.split("_")[0] + "_learning_curve.csv")
        again = a.repeat / curve.name
        if name != "cycdelta_released" and curve.exists() and again.exists():
            ca, cb = pd.read_csv(curve), pd.read_csv(again)
            if list(ca) != list(cb) or not ca.epoch.equals(cb.epoch):
                raise ValueError("validation curve index mismatch")
            metric = next(k for k in ca.columns if k != "epoch")
            abs_diff = (ca[metric] - cb[metric]).abs()
            nonzero = ca.loc[abs_diff > 0, "epoch"]
            info["fixed_host_repeat"].setdefault(name, {}).update(
                validation_curve_same_bytes=sha(curve) == sha(again),
                first_validation_divergence_epoch=(
                    int(nonzero.iloc[0]) if len(nonzero) else None
                ),
                max_validation_metric_change=float(abs_diff.max()),
                best_epoch_same=int(ca.loc[ca[metric].idxmin(), "epoch"])
                == int(cb.loc[cb[metric].idxmin(), "epoch"]),
            )
            curve_rows.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "epoch": ca.epoch,
                        "metric": metric,
                        "first": ca[metric],
                        "second": cb[metric],
                        "absolute_difference": abs_diff,
                    }
                )
            )
    if curve_rows:
        pd.concat(curve_rows, ignore_index=True).to_csv(
            a.out / "repeat_validation_curves.csv", index=False
        )
    (a.out / "repeat_audit.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edits", type=Path, default=Path("results/pairs/edits.csv"))
    p.add_argument(
        "--support", type=Path, default=Path("results/model/common_support.csv")
    )
    p.add_argument("--out", type=Path, default=Path("results/references"))
    p.add_argument("--repeat", type=Path, required=True)
    p.add_argument("--external", type=Path, required=True)
    run(p.parse_args())
