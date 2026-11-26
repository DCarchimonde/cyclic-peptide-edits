"""Functional checks with synthetic tensors, never biological validation."""

from __future__ import annotations

import unittest

import torch

from cyclic_peptide_ai.model import EditAwareFusion, FusionConfig


class FusionModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.config = FusionConfig(7, 11, 5, geometry_dim=3, hidden_dim=16)
        self.model = EditAwareFusion(self.config)

    def features(self, batch_size: int = 4, with_geometry: bool = True) -> dict:
        values = {
            "protein": torch.randn(batch_size, 7),
            "chemical": torch.randn(batch_size, 11),
            "edit": torch.randn(batch_size, 5),
        }
        if with_geometry:
            values["geometry"] = torch.randn(batch_size, 3)
        return values

    def test_forward_and_backward_with_geometry(self) -> None:
        inputs = self.features()
        result = self.model(**inputs)
        self.assertEqual(result["permeability"].shape, (4,))
        self.assertEqual(result["embedding"].shape, (4, 16))
        loss = (result["permeability"] - torch.linspace(-7.0, -5.0, 4)).square().mean()
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

    def test_geometry_is_optional(self) -> None:
        inputs = self.features(with_geometry=False)
        self.assertEqual(self.model(**inputs)["permeability"].shape, (4,))
        model_2d = EditAwareFusion(FusionConfig(7, 11, 5, hidden_dim=16))
        self.assertEqual(model_2d(**inputs)["permeability"].shape, (4,))
        with self.assertRaisesRegex(ValueError, "geometry_dim is zero"):
            model_2d(**self.features())

    def test_pair_difference_is_antisymmetric_during_inference(self) -> None:
        self.model.eval()
        parent, candidate = self.features(), self.features()
        with torch.no_grad():
            ab = self.model.predict_pair(parent, candidate)
            ba = self.model.predict_pair(candidate, parent)
            aa = self.model.predict_pair(parent, parent)
        torch.testing.assert_close(ab["delta"], -ba["delta"])
        torch.testing.assert_close(aa["delta"], torch.zeros(4))
        torch.testing.assert_close(ab["delta"], ab["candidate"] - ab["parent"])

    def test_invalid_input_dimensions_fail_with_useful_message(self) -> None:
        inputs = self.features()
        inputs["chemical"] = torch.randn(4, 10)
        with self.assertRaisesRegex(ValueError, "chemical must have shape"):
            self.model(**inputs)
        inputs = self.features()
        inputs["edit"] = torch.randn(3, 5)
        with self.assertRaisesRegex(ValueError, "edit batch size"):
            self.model(**inputs)

    def test_pair_batch_sizes_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal batch sizes"):
            self.model.predict_pair(self.features(4), self.features(3))


if __name__ == "__main__":
    unittest.main()
