"""Trainable fusion prototype for cached cyclic-peptide features.

This module does not download encoders or generate molecular conformations.
The caller supplies real features from the selected encoders and records their
provenance. Random tensors are appropriate only for software checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FusionConfig:
    protein_dim: int
    chemical_dim: int
    edit_dim: int
    geometry_dim: int = 0
    hidden_dim: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("protein_dim", "chemical_dim", "edit_dim", "hidden_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.geometry_dim < 0:
            raise ValueError("geometry_dim must be nonnegative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class EditAwareFusion(nn.Module):
    """Fuse peptide context, full chemical features and explicit edit features.

    All input features have shape ``[batch, configured_dimension]``. Geometry
    is optional for an entire batch. The output is a scalar property in the
    training label's scale; it is not a probability, oral-availability score,
    binding affinity or experimentally established improvement.

    The architecture is an initial gated-fusion implementation. It does not
    establish a uniquely identifiable common/edit decomposition or novelty.
    """

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.protein_projection = nn.Sequential(
            nn.Linear(config.protein_dim, h), nn.GELU()
        )
        self.chemical_projection = nn.Sequential(
            nn.Linear(config.chemical_dim, h), nn.GELU()
        )
        self.context_fusion = nn.Sequential(nn.Linear(2 * h, h), nn.GELU())
        # A zero edit vector contributes zero through this branch. Its exact
        # chemical meaning must be defined by the external edit featurizer.
        self.edit_projection = nn.Sequential(
            nn.Linear(config.edit_dim, h, bias=False), nn.GELU()
        )
        self.edit_gate = nn.Sequential(nn.Linear(2 * h, h), nn.Sigmoid())
        if config.geometry_dim:
            self.geometry_projection: nn.Module | None = nn.Sequential(
                nn.Linear(config.geometry_dim, h, bias=False), nn.GELU()
            )
            self.geometry_gate: nn.Module | None = nn.Sequential(
                nn.Linear(2 * h, h), nn.Sigmoid()
            )
        else:
            self.geometry_projection = None
            self.geometry_gate = None
        self.normalization = nn.LayerNorm(h)
        self.dropout = nn.Dropout(config.dropout)
        self.permeability_head = nn.Linear(h, 1)

    @staticmethod
    def _check_feature(
        name: str, feature: Tensor, width: int, batch_size: int | None = None
    ) -> None:
        if feature.ndim != 2 or feature.shape[1] != width:
            raise ValueError(
                f"{name} must have shape [batch, {width}], got {tuple(feature.shape)}"
            )
        if batch_size is not None and feature.shape[0] != batch_size:
            raise ValueError(f"{name} batch size must be {batch_size}")

    def encode(
        self,
        protein: Tensor,
        chemical: Tensor,
        edit: Tensor,
        geometry: Tensor | None = None,
    ) -> Tensor:
        """Return a fused ``[batch, hidden_dim]`` feature tensor.

        Missing geometry skips that branch. It is not interpreted as an
        experimentally observed zero-valued conformation. A production model
        used without geometry must be trained/evaluated for that use case.
        """
        c = self.config
        self._check_feature("protein", protein, c.protein_dim)
        batch_size = protein.shape[0]
        self._check_feature("chemical", chemical, c.chemical_dim, batch_size)
        self._check_feature("edit", edit, c.edit_dim, batch_size)
        p = self.protein_projection(protein)
        m = self.chemical_projection(chemical)
        context = self.context_fusion(torch.cat((p, m), dim=-1))
        edit_hidden = self.edit_projection(edit)
        gate = self.edit_gate(torch.cat((context, edit_hidden), dim=-1))
        fused = context + gate * edit_hidden
        if geometry is not None:
            if self.geometry_projection is None or self.geometry_gate is None:
                raise ValueError("geometry was supplied but geometry_dim is zero")
            self._check_feature("geometry", geometry, c.geometry_dim, batch_size)
            geometry_hidden = self.geometry_projection(geometry)
            geometry_gate = self.geometry_gate(
                torch.cat((fused, geometry_hidden), dim=-1)
            )
            fused = fused + geometry_gate * geometry_hidden
        return self.dropout(self.normalization(fused))

    def forward(
        self,
        protein: Tensor,
        chemical: Tensor,
        edit: Tensor,
        geometry: Tensor | None = None,
    ) -> dict[str, Tensor]:
        embedding = self.encode(protein, chemical, edit, geometry)
        return {
            "permeability": self.permeability_head(embedding).squeeze(-1),
            "embedding": embedding,
        }

    def predict_pair(
        self,
        parent: Mapping[str, Tensor | None],
        candidate: Mapping[str, Tensor | None],
    ) -> dict[str, Tensor]:
        """Predict both molecules and return ``candidate - parent``.

        Each mapping contains protein, chemical and edit tensors, and may
        contain geometry. This is the ordinary difference-of-predictions
        baseline, not a separately trained chemical-edit response head.
        In ``eval()`` mode it is antisymmetric when the mappings are swapped.
        Dropout in training mode introduces the usual sampling variation.
        """
        parent_result = self(**parent)["permeability"]
        candidate_result = self(**candidate)["permeability"]
        if parent_result.shape != candidate_result.shape:
            raise ValueError("parent and candidate must contain equal batch sizes")
        return {
            "parent": parent_result,
            "candidate": candidate_result,
            "delta": candidate_result - parent_result,
        }
