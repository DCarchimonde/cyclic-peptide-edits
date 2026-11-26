"""Acquire hash-pinned ncAA-RepDistill, CycDelta and Uni-Mol sources and weights."""

import argparse
import hashlib
import shutil
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NCAA = "c88857132c028ee463fafb9167a51bf54177c6f2"
CYC = "0a686c93f4b10fd3318e39f38893cdc9cbd0a848"
UM = "9f19c45c718192888a1c8a1c905f69f0755ea502"
PT = "artifacts/checkpoints/phase2_repro_sweep24_e0_b0/best.pt"
NCAA_SHA = "34bf3930189ddbb138e7eb6ca6bf4786f1baed2067c987030a6f76875679fbf4"
UM_SHA = "7f5f14bb28bf479a0b8f38ebab4bb9d7ae867383f14c4d694d056500fa3446a7"
ARCHIVE_SHA = {
    "CycDelta": "3c1fa749315bfb04df8f54ad4ed5bc8cebf699f19205f8b80daa558d37fbf187",
    "ncAA-RepDistill": "e38f79e90477a53274766b60be47b4a897defa860d44812f4d6a6781ef1b6776",
}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(url, path, checksum=None, size=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if (
        checksum
        and path.is_file()
        and (not size or path.stat().st_size == size)
        and sha(path) == checksum
    ):
        return
    tmp = path.with_name(path.name + ".part")
    if size:
        with tmp.open("wb") as f:
            f.truncate(size)

        def fetch(start):
            end = min(size, start + 16 * 1024**2) - 1
            q = urllib.request.Request(
                url + ("?" if "?" not in url else "&") + f"r={start}-{end}",
                headers={"Range": f"bytes={start}-{end}"},
            )
            for i in range(4):
                try:
                    with urllib.request.urlopen(q, timeout=65) as r:
                        if (
                            r.headers.get("Content-Range")
                            != f"bytes {start}-{end}/{size}"
                        ):
                            raise ValueError("mismatched range")
                        data = r.read()
                    if len(data) != end - start + 1:
                        raise ValueError("short range")
                    with tmp.open("r+b") as f:
                        f.seek(start)
                        f.write(data)
                    return
                except Exception:
                    if i == 3:
                        raise
                    time.sleep(2**i)

        with ThreadPoolExecutor(5) as pool:
            list(pool.map(fetch, range(0, size, 16 * 1024**2)))
    else:
        with urllib.request.urlopen(url, timeout=100) as r, tmp.open("wb") as o:
            shutil.copyfileobj(r, o)
    if checksum and sha(tmp) != checksum:
        raise ValueError("checksum mismatch for " + path.name)
    tmp.replace(path)


def acquire(out):
    out.mkdir(parents=True, exist_ok=True)
    for repo, rev in [("cihebiyql/ncAA-RepDistill", NCAA), ("cyclontx/CycDelta", CYC)]:
        root = out / repo.split("/")[-1]
        archive = out / (root.name + ".zip")
        download(
            f"https://codeload.github.com/{repo}/zip/{rev}",
            archive,
            ARCHIVE_SHA[root.name],
        )
        with zipfile.ZipFile(archive) as z:
            for f in z.infolist():
                p = Path(*Path(f.filename).parts[1:])
                if ".." in p.parts or p.is_absolute():
                    raise ValueError("unexpected archive path")
                dst = root / p
                if f.is_dir():
                    continue
                if dst.is_file():
                    with z.open(f) as src:
                        archived = src.read()
                    if archived.startswith(
                        b"version https://git-lfs.github.com/spec/v1\n"
                    ):
                        # The expanded data files used below have their own expected SHA-256.
                        continue
                    if hashlib.sha256(archived).hexdigest() != sha(dst):
                        raise ValueError(
                            f"local source differs from pinned archive: {dst}"
                        )
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                with z.open(f) as src, dst.open("wb") as dest:
                    shutil.copyfileobj(src, dest)
    n = out / "ncAA-RepDistill"
    media = (
        f"https://media.githubusercontent.com/media/cihebiyql/ncAA-RepDistill/{NCAA}/"
    )
    hf = f"https://huggingface.co/dptech/Uni-Mol-Models/resolve/{UM}/"
    files = [(media + PT, n / PT, NCAA_SHA, 1602726329)]
    for split, c, s in [
        (
            "train",
            "654b339837fb74c8e0907f89e659dbbb0d504c8445a0e68b11bb318cf95f4006",
            145669344,
        ),
        (
            "val",
            "bde1235e0f65a7332206aaf6b3474e620b04adccf629104118b56d8a9e1c88cd",
            16132607,
        ),
    ]:
        p = f"data/raw/pretrain/uniref_lt30_c30/{split}_sequences.csv"
        files.append((media + p, n / p, c, s))
    files += [
        (
            hf + "mol.dict.txt",
            out / "unimol/mol.dict.txt",
            "94135cb9a9198f988de684cb61e2c372882a3bd59b8320effbae704c38057127",
            None,
        ),
        (
            hf + "mol_pre_all_h_220816.pt",
            out / "unimol/mol_pre_all_h_220816.pt",
            UM_SHA,
            190540187,
        ),
    ]
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(lambda args: download(*args), files))


def benchmark(edits, support=None):
    import pandas as pd

    e = pd.read_csv(edits)
    if not support:
        return e, None
    ids = pd.read_csv(support, usecols=["pair_id"]).pair_id
    if ids.duplicated().any() or not set(ids) <= set(e.pair_id):
        raise ValueError("invalid common IDs")
    rows = e[e.pair_id.isin(ids)].reset_index(drop=True)
    if set(rows.group_id) & set(e.loc[~e.pair_id.isin(ids), "group_id"]):
        raise ValueError("truncated option group")
    return rows, sha(support)


def measured(frame):
    import pandas as pd

    parts = []
    for role in ("parent", "variant"):
        s = frame[["protocol_id", role + "_smiles", role + "_y"]].copy()
        s.columns = ["protocol", "smiles", "y"]
        parts.append(s.dropna(subset=["y"]))
    return (
        pd.concat(parts)
        .drop_duplicates(["protocol", "smiles"])
        .sort_values(["protocol", "smiles"])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    acquire(p.parse_args().out)
