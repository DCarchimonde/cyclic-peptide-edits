"""Set up a CPU environment for a single public project from Anaconda Prompt."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENVIRONMENTS = {"R0": "cyc_r0", "B": "cyc_b"}
REQUIREMENTS = {
    "R0": ROOT / "R0_Cho" / "requirements.txt",
    "B": ROOT / "B_PAMPA" / "configs" / "requirements_edit_cpu.txt",
}


def conda_executable():
    for candidate in (os.environ.get("CONDA_EXE"), shutil.which("conda.exe"),
                      Path(sys.prefix) / "Scripts" / "conda.exe", shutil.which("conda")):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise SystemExit("Open Anaconda Prompt and run SETUP_WINDOWS.cmd there.")


def check_python():
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"This project needs Python 3.12. The selected interpreter is {sys.executable}.")


def probe(track):
    check_python()
    import numpy as np

    if track == "B":
        import pandas
        import rdkit
        import sklearn
        import torch
        import transformers

        result = torch.from_numpy(np.array([1.25], dtype="float32")).numpy()
        if result[0] != 1.25 or torch.version.cuda is not None:
            raise SystemExit("Expected a working NumPy bridge and the CPU build of PyTorch.")
        print(f"torch={torch.__version__} rdkit={rdkit.__version__} "
              f"pandas={pandas.__version__} sklearn={sklearn.__version__} transformers={transformers.__version__}")
    print(f"numpy={np.__version__}")
    print(f"{track}_ENV_OK  {sys.executable}", flush=True)


def install(track):
    conda = conda_executable()
    env_name = ENVIRONMENTS[track]
    known = json.loads(subprocess.check_output([conda, "env", "list", "--json"]))
    if not any(Path(p).name.casefold() == env_name.casefold() for p in known["envs"]):
        subprocess.run([conda, "create", "-n", env_name, "-y", "--override-channels",
                        "-c", "conda-forge", "python=3.12", "pip"], check=True)
    python = [conda, "run", "--no-capture-output", "-n", env_name, "python", "-X", "utf8"]
    script = str(Path(__file__).resolve())
    subprocess.run(python + [script, track, "--check-python"], check=True)
    pip = python + ["-m", "pip", "install", "--disable-pip-version-check"]
    if track == "B":
        subprocess.run(pip + ["torch==2.5.1+cpu", "--index-url", "https://download.pytorch.org/whl/cpu"],
                       check=True)
    subprocess.run(pip + ["-r", str(REQUIREMENTS[track])], check=True)
    subprocess.run(python + [script, track, "--probe"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", type=str.upper, choices=ENVIRONMENTS)
    parser.add_argument("--check-python", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.check_python:
            check_python()
        elif args.probe:
            probe(args.track)
        else:
            install(args.track)
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
