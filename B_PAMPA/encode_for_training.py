"""Populate the original B feature cache. The optional CUDA path moves only frozen model forwards."""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT
sys.path.insert(0, str(SOURCE / "scripts"))

import numpy as np
import pandas as pd
import torch
import encoder_cache
from fetch_edit_encoders import verify
from pair_chemistry import graph_index
from prime_edit_cache import prime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    args.work.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    info = {"device": str(device), "torch": str(torch.__version__), "numpy": np.__version__,
            "cuda_build": torch.version.cuda, "local_heads": "CPU", "threads": args.threads}
    tag = "cpu"
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable to this interpreter. Select a cu124 environment on a 4090, or use --device cpu.")
        torch.ones(1, device=device).sum().item()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        info["gpu"] = torch.cuda.get_device_name(device)
        info["capability"] = list(torch.cuda.get_device_capability(device))
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", f"cuda_{info['gpu']}_{torch.__version__}")
    cache = args.work / ("features_" + tag)
    weights = args.work / "weights"
    pair_file = SOURCE / "results/pairs/edits.csv"
    graph_file = args.work / "pair_graph.json"
    records = SOURCE / "data/processed/pampa.csv"
    print("Checking the pinned ESM-2 and ChemBERTa weights:", weights, flush=True)
    info["files_sha256"] = verify(weights, fetch=True)
    if device.type == "cuda":
        # Keep graph/stereochemical features, tokenization, masking and pooling exactly as in the archive.
        native_states = encoder_cache.states
        def gpu_states(model, batch):
            if next(model.parameters()).device != device:
                model.to(device)
            return native_states(model, {key: batch[key].to(device)
                                         for key in ("input_ids", "attention_mask")})
        # Compare the two pretrained paths for one real molecule before filling the independent CUDA cache.
        example = pd.read_csv(pair_file, nrows=1).parent_smiles.iat[0]
        small = args.out / "cpu_check"
        encoder_cache.encode_bank([example], weights, small, batch=1, threads=args.threads)
        encoder_cache.states = gpu_states
        gpu_check = args.out / "cuda_check"
        encoder_cache.encode_bank([example], weights, gpu_check, batch=1, threads=args.threads)
        a = encoder_cache.read_bank(small, example)
        b = encoder_cache.read_bank(gpu_check, example)
        differences={}
        for key in ("protein", "chem", "env", "atom", "prot"):
            if a[key].shape!=b[key].shape or not np.isfinite(b[key]).all():
                raise ValueError(f"CUDA feature check failed: {key}")
            differences[key]=float(np.max(np.abs(a[key]-b[key])))
        info["single_molecule_max_abs_gpu_minus_cpu"] = differences
        print("CUDA encoder check", differences, flush=True)
    # Same demo and batch schedule as the archived CPU workflow.
    encoder_cache.demo(records, weights, cache, args.out/"encoder_demo")
    graph_index(SOURCE / "data/raw/cycpeptmpdb_all.csv", graph_file)
    prime(graph_file, records, weights, cache, args.threads)
    pairs = pd.read_csv(pair_file)
    all_structures=set(pairs.parent_smiles) | set(pairs.variant_smiles)
    encoder_cache.encode_bank(all_structures, weights, cache, threads=args.threads)
    info["feature_cache"]=str(cache.resolve())
    (args.out/"encoder_run.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
