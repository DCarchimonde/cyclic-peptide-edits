"""Train frozen-encoder fusion heads on independently partitioned single-site edits."""

import argparse, ast, copy, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from rdkit import Chem
from encoder_cache import encode_bank, read_bank, SPEC, VERSION
from edit_metrics import score_predictions, summarize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cyclic_peptide_ai.model import EditAwareFusion, FusionConfig

HEADS = ("chem_only", "plain_fusion", "chem_site", "explicit_edit")
EPOCHS = (5, 10, 20, 35, 50)
SEED = 2026


def site(bank, idx):
    matches = np.flatnonzero(bank["ns"] == int(idx))
    if len(matches) != 1:
        raise ValueError("edit_n_not_uniquely_aligned_to_residue")
    return int(matches[0])


def molecule_bank(df, weights, cache, threads):
    strings = set(df.parent_smiles) | set(df.variant_smiles)
    encode_bank(strings, weights, cache, threads=threads)
    return {s: read_bank(cache, s) for s in strings}


def eligible(df, bank, complete=True):
    removed = []
    keep = []
    for i, r in enumerate(df.itertuples(index=False)):
        try:
            for s, idx in [
                (r.parent_smiles, r.parent_n_idx),
                (r.variant_smiles, r.variant_n_idx),
            ]:
                b = bank[s]
                if str(b["error"]):
                    raise ValueError(str(b["error"]))
                site(b, idx)
            keep.append(i)
        except ValueError as e:
            removed.append(dict(pair_id=r.pair_id, group_id=r.group_id, reason=str(e)))
    result = df.iloc[keep].copy()
    if complete and removed:
        bad = {r["group_id"] for r in removed}
        for r in result[result.group_id.isin(bad)].itertuples():
            removed.append(
                dict(
                    pair_id=r.pair_id,
                    group_id=r.group_id,
                    reason="incomplete_candidate_set",
                )
            )
        result = result[~result.group_id.isin(bad)]
    return result.reset_index(drop=True), pd.DataFrame(
        removed, columns=["pair_id", "group_id", "reason"]
    )


def pair_features(row, bank, name):
    a, b = bank[row.parent_smiles], bank[row.variant_smiles]
    i, j = site(a, row.parent_n_idx), site(b, row.variant_n_idx)
    parts = []

    def add(x, y):
        parts.extend([x, y - x])

    for k in ["chem", "compact"]:
        add(a[k], b[k])
    if name in ("plain_fusion", "explicit_edit"):
        add(a["protein"], b["protein"])
    if name in ("chem_site", "explicit_edit"):
        for k in ["atom", "env", "local"]:
            add(a[k][i], b[k][j])
    if name == "explicit_edit":
        add(a["prot"][i], b["prot"][j])
    return np.concatenate(parts).astype("float32")


def make_x(df, bank, name):
    return np.stack([pair_features(r, bank, name) for r in df.itertuples(index=False)])


def fit_scaler(x):
    s = StandardScaler().fit(x)
    inv = np.divide(1, s.scale_, out=np.zeros_like(s.scale_), where=s.var_ > 1e-10)
    return dict(mean=s.mean_.astype("float32"), inverse_sd=inv.astype("float32"))


def scaling(x, scale):
    return torch.from_numpy(
        np.clip((x - scale["mean"]) * scale["inverse_sd"], -6.0, 6.0).astype("float32")
    )


def make_head(width):
    return nn.Sequential(
        nn.Linear(width, 32),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(32, 1),
        nn.Flatten(start_dim=0),
    )


def validation_key(df, pred, epoch):
    if df.empty:
        return (abs(epoch - 20), 0.0, 0.0)
    g, e = score_predictions(
        df, pd.DataFrame(dict(pair_id=df.pair_id, model="selection", pred_delta=pred))
    )
    r = g.groupby("parent_id").regret.mean().mean() if len(g) else np.nan
    m = e.delta_mae.mean() if len(e) else np.nan
    if not np.isfinite(r) and not np.isfinite(m):
        return (abs(epoch - 20), 0.0, 0.0)
    return (
        float(r) if np.isfinite(r) else float(m),
        float(m) if np.isfinite(m) else 0.0,
        epoch,
    )


class EditLoss:
    def __init__(self, df):
        self.target = torch.from_numpy(df.variant_y.to_numpy("float32"))
        self.delta = torch.from_numpy(df.delta.fillna(0).to_numpy("float32"))
        self.exact = torch.from_numpy(df.delta.notna().to_numpy())
        self.group = torch.from_numpy(pd.factorize(df.group_id)[0])
        self.count = torch.bincount(self.group).float()
        self.exact_count = torch.bincount(
            self.group, weights=self.exact.float(), minlength=len(self.count)
        )
        self.ranked = self.count[self.group] >= 2
        self.ranked_groups = int((self.count >= 2).sum())
        self.exact_groups = int((self.exact_count > 0).sum())

    def __call__(self, pred):
        terms = []
        if self.exact_groups:
            mask = self.exact
            terms.append(
                (
                    (pred[mask] - self.delta[mask]).square()
                    / self.exact_count[self.group[mask]]
                ).sum()
                / self.exact_groups
            )
        if self.ranked_groups:
            err = pred - self.target
            means = (
                torch.zeros_like(self.count).index_add(0, self.group, err) / self.count
            )
            terms.append(
                (
                    (err[self.ranked] - means[self.group[self.ranked]]).square()
                    / self.count[self.group[self.ranked]]
                ).sum()
                / self.ranked_groups
            )
        if not terms:
            raise ValueError("no training supervision")
        return sum(terms) / len(terms)


def fit_head(name, frame, bank, models):
    train = (frame.split == "train").to_numpy()
    val = (frame.split == "valid").to_numpy()
    x = make_x(frame, bank, name)
    scale = fit_scaler(x[train])
    xt = scaling(x, scale)
    tr = xt[train]
    va = xt[val]
    cost = EditLoss(frame[train])
    torch.manual_seed(SEED)
    head = make_head(x.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.001)
    best = None
    bestkey = None
    hist = []
    for epoch in range(1, max(EPOCHS) + 1):
        head.train()
        opt.zero_grad()
        loss = cost(head(tr))
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        opt.step()
        if epoch in EPOCHS:
            head.eval()
            with torch.inference_mode():
                key = validation_key(
                    frame[val], head(va).numpy() if val.any() else [], epoch
                )
            hist.append(
                dict(
                    model=name,
                    epoch=epoch,
                    train_loss=float(loss.detach()),
                    validation_primary=key[0],
                    validation_secondary=key[1],
                )
            )
            if bestkey is None or key < bestkey:
                bestkey = key
                best = (epoch, copy.deepcopy(head.state_dict()))
    head.load_state_dict(best[1])
    head.eval()
    with torch.inference_mode():
        pred = head(xt).numpy()
    torch.save(
        dict(
            weight_dims=int(x.shape[1]),
            selected_epoch=best[0],
            state=best[1],
            scale={k: torch.tensor(v) for k, v in scale.items()},
        ),
        models / (name + ".pt"),
    )
    return pred, hist, best[0], sum(p.numel() for p in head.parameters())


def absolute_observations(frame):
    cols = ["source", "protocol_id", "split"]
    p = frame[cols + ["parent_smiles", "parent_y"]].rename(
        columns={"parent_smiles": "smi", "parent_y": "y"}
    )
    v = frame[cols + ["variant_smiles", "variant_y"]].rename(
        columns={"variant_smiles": "smi", "variant_y": "y"}
    )
    obs = pd.concat([p, v]).dropna(subset=["y"]).drop_duplicates()
    if (obs.groupby(cols + ["smi"]).y.nunique() > 1).any():
        raise ValueError("conflicting absolute observations")
    return obs


def fit_absolute(frame, bank, models):
    obs = absolute_observations(frame)
    train = obs[obs.split == "train"]
    graph = np.stack([bank[s]["graph"] for s in train.smi])
    tree = ExtraTreesRegressor(
        n_estimators=80, max_depth=14, min_samples_leaf=2, random_state=SEED, n_jobs=2
    )
    tree.fit(graph, train.y)
    joblib.dump(tree, models / "extratrees_absolute.joblib", compress=3)
    smiles = sorted(set(frame.parent_smiles) | set(frame.variant_smiles))
    idx = {s: i for i, s in enumerate(smiles)}
    ia = np.array([idx[s] for s in frame.parent_smiles])
    ib = np.array([idx[s] for s in frame.variant_smiles])
    p = tree.predict(np.stack([bank[s]["graph"] for s in smiles]))
    preds = {"extratrees_abs_diff": p[ib] - p[ia]}
    f = {
        "protein": np.stack([bank[s]["protein"] for s in smiles]),
        "chemical": np.stack(
            [np.r_[bank[s]["chem"], bank[s]["compact"]] for s in smiles]
        ),
    }
    ti = np.array([idx[s] for s in train.smi])
    scale = {k: fit_scaler(v[ti]) for k, v in f.items()}
    xt = {k: scaling(v, scale[k]) for k, v in f.items()}
    xt["edit"] = torch.zeros((len(smiles), 1))
    yt = torch.tensor(train.y.to_numpy("float32"))
    mu = float(yt.mean())
    sd = max(float(yt.std(unbiased=False)), 1e-5)
    yt = (yt - mu) / sd
    cfg = FusionConfig(
        protein_dim=f["protein"].shape[1],
        chemical_dim=f["chemical"].shape[1],
        edit_dim=1,
        hidden_dim=32,
        dropout=0.1,
    )
    torch.manual_seed(SEED)
    net = EditAwareFusion(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.001)
    val = (frame.split == "valid").to_numpy()

    def features(ix):
        return {k: v[ix] for k, v in xt.items()}

    tr = features(ti)
    hist = []
    best = None
    bestkey = None
    for epoch in range(1, max(EPOCHS) + 1):
        net.train()
        opt.zero_grad()
        loss = (net(**tr)["permeability"] - yt).square().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if epoch in EPOCHS:
            net.eval()
            with torch.inference_mode():
                pv = (
                    net.predict_pair(features(ia[val]), features(ib[val]))[
                        "delta"
                    ].numpy()
                    * sd
                    if val.any()
                    else []
                )
            key = validation_key(frame[val], pv, epoch)
            hist.append(
                dict(
                    model="v01_predict_pair",
                    epoch=epoch,
                    train_loss=float(loss.detach()),
                    validation_primary=key[0],
                    validation_secondary=key[1],
                )
            )
            if bestkey is None or key < bestkey:
                bestkey = key
                best = (epoch, copy.deepcopy(net.state_dict()))
    net.load_state_dict(best[1])
    net.eval()
    with torch.inference_mode():
        preds["v01_predict_pair"] = (
            net.predict_pair(features(ia), features(ib))["delta"].numpy() * sd
        )
    torch.save(
        dict(
            state=best[1],
            config=cfg.__dict__,
            y_mean=mu,
            y_sd=sd,
            scale={
                k: {t: torch.tensor(v) for t, v in vals.items()}
                for k, vals in scale.items()
            },
        ),
        models / "v01_predict_pair.pt",
    )
    return preds, hist, best[0], len(train)


def integrity(frame):
    if (
        frame.pair_id.duplicated().any()
        or (frame.groupby("series_id").split.nunique() > 1).any()
        or (frame.groupby("group_id").split.nunique() > 1).any()
    ):
        raise ValueError("pair, parent, or series split overlap")
    s = pd.concat(
        [
            frame[["parent_smiles", "split"]].rename(columns={"parent_smiles": "s"}),
            frame[["variant_smiles", "split"]].rename(columns={"variant_smiles": "s"}),
        ]
    )
    if (s.groupby("s").split.nunique() > 1).any():
        raise ValueError("molecule split overlap")


def normalize_prediction_pairs(table):
    from pair_chemistry import edits_for, eligible as structure_eligibility

    df = table.copy()

    def canon(s, idx):
        m = Chem.MolFromSmiles(s)
        s = Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
        return s, ast.literal_eval(m.GetProp("_smilesAtomOutputOrder")).index(int(idx))

    for i, r in table.iterrows():
        p, pn = canon(r.parent_smiles, r.parent_n_idx)
        v, vn = canon(r.variant_smiles, r.variant_n_idx)
        child = Chem.MolFromSmiles(v)
        reason = structure_eligibility(child)
        if reason:
            raise ValueError(reason)
        rw = Chem.RWMol(Chem.MolFromSmiles(p))
        idx = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(pn, idx, Chem.BondType.SINGLE)
        Chem.SanitizeMol(rw)
        if Chem.MolToSmiles(rw, isomericSmiles=True) != v:
            raise ValueError("add-back failed")
        opts, _ = edits_for(v)
        if p not in opts:
            raise ValueError("variant is not a single backbone N-methyl edit")
        # Verify declared variant nitrogen explicitly even for equivalent symmetry sites.
        nbrs = [
            a.GetIdx()
            for a in child.GetAtomWithIdx(vn).GetNeighbors()
            if a.GetAtomicNum() == 6 and a.GetDegree() == 1 and a.GetTotalNumHs() == 3
        ]
        if not any(_demethyl(child, j) == p for j in nbrs):
            raise ValueError("supplied variant nitrogen fails methyl removal")
        df.loc[
            i, ["parent_smiles", "variant_smiles", "parent_n_idx", "variant_n_idx"]
        ] = [p, v, opts[p][0], opts[p][1]]
    return df


def _demethyl(m, i):
    rw = Chem.RWMol(m)
    rw.RemoveAtom(i)
    Chem.SanitizeMol(rw)
    return Chem.MolToSmiles(rw, isomericSmiles=True)


def predict_frozen_head(frame, bank, checkpoint):
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    net = make_head(weights["weight_dims"])
    net.load_state_dict(weights["state"])
    net.eval()
    scale = {key: value.numpy() for key, value in weights["scale"].items()}
    with torch.inference_mode():
        return net(scaling(make_x(frame, bank, checkpoint.stem), scale)).numpy()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--edits", type=Path, default=Path("results/pairs/edits.csv"))
    a.add_argument("--weights", type=Path, required=True)
    a.add_argument("--cache", type=Path, default=Path(".cache/encoders"))
    a.add_argument("--out", type=Path, default=Path("results/model"))
    a.add_argument("--threads", type=int, default=2)
    a.add_argument("--predict-head", type=Path)
    o = a.parse_args()
    tick = time.monotonic()
    torch.set_num_threads(o.threads)
    df = pd.read_csv(o.edits)
    if o.predict_head:
        df = normalize_prediction_pairs(df)
        bank = molecule_bank(df, o.weights, o.cache, o.threads)
        df, ex = eligible(df, bank)
        if len(ex):
            raise ValueError(str(ex.iloc[0].to_dict()))
        name = o.predict_head.stem
        yp = predict_frozen_head(df, bank, o.predict_head)
        pd.DataFrame(dict(pair_id=df.pair_id, model=name, pred_delta=yp)).to_csv(
            o.out, index=False
        )
        return
    integrity(df)
    out = o.out
    out.mkdir(exist_ok=True, parents=True)
    models = out / "fitted"
    models.mkdir(exist_ok=True)
    bank = molecule_bank(df, o.weights, o.cache, o.threads)
    f, ex = eligible(df, bank)
    ex.to_csv(out / "encoder_exclusions.csv", index=False)
    f[["pair_id", "group_id", "series_id", "split", "parent_id", "variant_id"]].to_csv(
        out / "common_support.csv", index=False
    )
    pred, hist, ep, nobs = fit_absolute(f, bank, models)
    epochs = {"v01_predict_pair": ep}
    nparams = {}
    pred["random_site_zero_delta"] = np.zeros(len(f))
    pred["train_median_delta"] = np.full(
        len(f), float(f[f.split == "train"].delta.median())
    )
    for name in HEADS:
        p, records, ep, n = fit_head(name, f, bank, models)
        pred[name] = p
        hist.extend(records)
        epochs[name] = ep
        nparams[name] = n
        print(name, ep, n, flush=True)
    forecasts = pd.concat(
        [
            pd.DataFrame(dict(pair_id=f.pair_id, model=k, pred_delta=v))
            for k, v in pred.items()
        ],
        ignore_index=True,
    )
    forecasts.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(hist).to_csv(out / "validation_history.csv", index=False)
    metrics = summarize(f, forecasts, out)
    mols = set(f.parent_smiles) | set(f.variant_smiles)
    info = dict(
        seed=SEED,
        torch_threads=o.threads,
        checkpoint_epochs=EPOCHS,
        selected_epochs=epochs,
        head_parameters=nparams,
        seconds=round(time.monotonic() - tick, 2),
        edits_sha256=hashlib.sha256(o.edits.read_bytes()).hexdigest(),
        encoder_spec=SPEC,
        encoder_cache_version=VERSION,
        training_observed_molecule_protocols=nobs,
        input_edges=len(df),
        common_edges=len(f),
        excluded_reasons=ex.reason.value_counts().to_dict(),
        common_unique_molecules=len(mols),
        residues_in_common_unique_molecules=sum(len(bank[s]["ns"]) for s in mols),
        unknown_esm_residues=sum(int(bank[s]["unknown_esm"]) for s in mols),
        fully_closed_alpha_backbone_molecules=sum(
            bool(bank[s]["cyclic"]) for s in mols
        ),
        sorted_splits=f.split.value_counts().to_dict(),
    )
    (out / "run.json").write_text(json.dumps(info, indent=2))
    print(metrics.query("split=='test'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
