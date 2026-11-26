"""Replay the fitted edit head on the two archived Kelly variants without a download."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fit_edits import make_head, scaling

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, help="Optional CSV path for the two scored edits"
    )
    args = parser.parse_args()
    model_dir = ROOT / "results/model"
    measured = pd.read_csv(ROOT / "results/pairs/examples/kelly_site_choice.csv")
    measured = measured.set_index("pair_id", verify_integrity=True)
    fitted = torch.load(
        model_dir / "fitted/explicit_edit.pt", map_location="cpu", weights_only=True
    )
    head = make_head(fitted["weight_dims"])
    head.load_state_dict(fitted["state"])
    head.eval()
    torch.set_num_threads(1)
    with np.load(model_dir / "example/two_site_model_inputs.npz") as saved:
        measured = measured.loc[saved["pair_ids"]].copy()
        np.testing.assert_allclose(measured.measured_delta, saved["measured_delta"])
        scale = {key: value.numpy() for key, value in fitted["scale"].items()}
        input_matrix = scaling(saved["unscaled_explicit_input"], scale)
        np.testing.assert_allclose(input_matrix.numpy(), saved["scaled_explicit_input"])
    with torch.inference_mode():
        measured["predicted_delta"] = head(input_matrix).numpy()
    best = measured.predicted_delta.idxmax()
    observed_best = measured.measured_variant_log10P.idxmax()
    measured["model_first_choice"] = measured.index == best
    measured["measured_best"] = measured.index == observed_best
    loss = (
        measured.loc[observed_best, "measured_variant_log10P"]
        - measured.loc[best, "measured_variant_log10P"]
    )
    print(
        measured[
            [
                "site_label",
                "measured_delta",
                "predicted_delta",
                "model_first_choice",
                "measured_best",
            ]
        ].to_string(index=False)
    )
    print(f"First-choice loss = {loss:.2f} log10 units")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        measured.to_csv(args.out, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
