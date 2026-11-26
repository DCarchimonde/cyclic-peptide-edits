"""Download the pinned public CSVs used by this starter project.

No model weights or trajectory archives are downloaded. Existing matching files
are reused. See data/sources.json for attribution, commits, and checksums.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(entry: dict, timeout: float) -> dict:
    destination = ROOT / entry["local_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = entry["sha256"]
    if destination.exists() and sha256(destination) == expected:
        return {"file": str(destination.relative_to(ROOT)), "status": "cached"}
    request = urllib.request.Request(
        entry["url"], headers={"User-Agent": "cyclic-peptide-ai-research/0.1"}
    )
    with TemporaryDirectory(dir=destination.parent) as folder:
        temporary = Path(folder) / destination.name
        total = 0
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > entry.get("max_bytes", 32 * 1024 * 1024):
                        raise ValueError("Response exceeds the expected small-file size.")
                    handle.write(block)
        actual = sha256(temporary)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {entry['id']}: {actual}")
        temporary.replace(destination)
        return {
            "file": str(destination.relative_to(ROOT)),
            "status": "downloaded",
            "bytes": total,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--only", nargs="+", help="Download only these source entry IDs.")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/sources.json").read_text(encoding="utf-8"))
    entries = manifest["files"]
    if args.only:
        known = {entry["id"] for entry in entries}
        if unknown := set(args.only) - known:
            parser.error(f"Unknown IDs: {sorted(unknown)}")
        entries = [entry for entry in entries if entry["id"] in args.only]
    for entry in entries:
        print(json.dumps(fetch(entry, args.timeout), ensure_ascii=False))


if __name__ == "__main__":
    main()
