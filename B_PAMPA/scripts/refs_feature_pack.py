"""Pack and restore native ncAA-RepDistill CLS outputs by canonical-structure SHA-256."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from refs_assets import NCAA_SHA, benchmark, sha


def run(a):
    rows, support_digest = benchmark(a.edits, a.support)
    smiles = sorted(set(rows.parent_smiles) | set(rows.variant_smiles))
    ids = np.asarray(
        [hashlib.sha256(s.encode()).hexdigest() for s in smiles], dtype="S64"
    )
    files = [a.cache / (x.decode() + ".npy") for x in ids]
    if a.restore:
        with np.load(a.pack, allow_pickle=False) as packed:
            if (
                str(packed["checkpoint"]) != NCAA_SHA
                or str(packed["support"]) != support_digest
            ):
                raise ValueError("different checkpoint or support set")
            if not np.array_equal(ids, packed["smiles_sha256"]):
                raise ValueError("different canonical structure IDs")
            matrix = packed["vectors"]
            if (
                str(packed["vectors_sha256"])
                != hashlib.sha256(matrix.tobytes()).hexdigest()
            ):
                raise ValueError("vector digest mismatch")
            if matrix.shape != (len(smiles), 768) or matrix.dtype != np.float32:
                raise ValueError("invalid native features")
            a.cache.mkdir(parents=True, exist_ok=True)
            for fn, vector in zip(files, matrix):
                if fn.is_file():
                    if not np.array_equal(np.load(fn, allow_pickle=False), vector):
                        raise ValueError(
                            f"different cached vector already exists: {fn}"
                        )
                else:
                    np.save(fn, vector, allow_pickle=False)
    else:
        matrix = np.stack([np.load(p, allow_pickle=False) for p in files])
        if (
            matrix.shape != (len(smiles), 768)
            or matrix.dtype != np.float32
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("invalid native features")
        np.savez_compressed(
            a.pack,
            smiles_sha256=ids,
            vectors=matrix,
            checkpoint=NCAA_SHA,
            support=support_digest,
            vectors_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
        )
    print(len(smiles), "structures", sha(a.pack))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edits", type=Path, default=Path("results/pairs/edits.csv"))
    p.add_argument(
        "--support", type=Path, default=Path("results/model/common_support.csv")
    )
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument(
        "--pack", type=Path, default=Path("results/references/ncaa_native_features.npz")
    )
    p.add_argument("--restore", action="store_true")
    run(p.parse_args())
