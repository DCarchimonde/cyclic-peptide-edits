"""Fetch pin-by-commit ESM-2 and ChemBERTa encoders, validate SHA256 of their trained weights."""

from pathlib import Path
import argparse, json, hashlib, urllib.request
from concurrent.futures import ThreadPoolExecutor

SPEC = {
    "esm": (
        "biohub/esm2_t6_8M_UR50D",
        "c731040fcd8d73dceaa04b0a8e6329b345b0f5df",
        "model.safetensors",
        "24c5fa474c48f3b754b86efe752d5f189d2bcd88190fa2270fc92b2ef3034189",
        (
            "config.json",
            "vocab.txt",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ),
    ),
    "chem": (
        "DeepChem/ChemBERTa-77M-MLM",
        "ed8a5374f2024ec8da53760af91a33fb8f6a15ff",
        "pytorch_model.bin",
        "95571af26b61ba8d56b9d61ff108b5c9afe96fd69ea3cbd1901d9421e3d2dd6d",
        (
            "config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ),
    ),
}


def verify(directory, fetch=False):
    jobs = [(k, s, n) for k, s in SPEC.items() for n in [s[2], *s[4]]]

    def job(inp):
        k, s, n = inp
        p = directory / k / n
        if not p.exists() and fetch:
            p.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(
                f"https://huggingface.co/{s[0]}/resolve/{s[1]}/{n}", timeout=120
            ) as req:
                b = req.read()
            if n == s[2] and hashlib.sha256(b).hexdigest() != s[3]:
                raise ValueError("weight hash mismatch")
            tmp = p.with_suffix(p.suffix + ".part")
            tmp.write_bytes(b)
            tmp.replace(p)
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if n == s[2] and h != s[3]:
            raise ValueError("weight hash mismatch")
        return str(k) + "/" + n, h

    with ThreadPoolExecutor(max_workers=6) as pool:
        return dict(pool.map(job, jobs))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--weights", type=Path, required=True)
    o = a.parse_args()
    print(json.dumps(verify(o.weights, True), indent=2))
