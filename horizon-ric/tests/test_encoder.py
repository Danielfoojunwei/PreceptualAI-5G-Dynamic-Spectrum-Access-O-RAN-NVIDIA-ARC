"""SpatialPrior + EntityTokenizer tests."""

import numpy as np
import pytest
import torch

from horizon_ric.encoder import (
    AssetAttributes,
    EntityTokenizer,
    EntityTokenizerConfig,
    EntityType,
    SpatialPrior,
    SpatialPriorTensor,
)
from horizon_ric.planner.physics.geodesy import ECEF

# ─── SpatialPrior ────────────────────────────────────────────────────────


class TestSpatialPriorBuild:
    def test_empty_returns_empty_tensors(self):
        sp = SpatialPrior()
        t = sp.build([])
        assert isinstance(t, SpatialPriorTensor)
        assert t.node_xyz_m.shape == (0, 3)
        assert t.los_mask.shape == (0, 0)

    def test_two_ground_entities(self):
        sp = SpatialPrior(min_elevation_deg=5.0)
        ents = [
            {"placement": "ground", "lat_deg": 0.0, "lon_deg": 0.0},
            {"placement": "ground", "lat_deg": 30.0, "lon_deg": 60.0},
        ]
        t = sp.build(ents)
        assert t.node_xyz_m.shape == (2, 3)
        assert t.los_mask.shape == (2, 2)
        # Diagonal is False (no self-LOS).
        assert not t.los_mask[0, 0]
        assert not t.los_mask[1, 1]
        # Slant range entry is non-zero between the two
        assert t.slant_range_m[0, 1] > 0

    def test_overhead_satellite_visible(self):
        sp = SpatialPrior(min_elevation_deg=5.0)
        ents = [
            {"placement": "ground", "lat_deg": 0.0, "lon_deg": 0.0},
            {
                "placement": "ecef",
                "ecef": ECEF(6_378_137.0 + 1_000_000.0, 0.0, 0.0),
            },
        ]
        t = sp.build(ents)
        # ES sees the overhead sat — LOS true, near-zenith elevation.
        assert t.los_mask[0, 1]
        assert t.elevation_deg[0, 1] > 80.0

    def test_distant_below_horizon_not_visible(self):
        sp = SpatialPrior(min_elevation_deg=10.0)
        # ES at (0,0) and a sat at the antipode (180° lon at GSO altitude)
        from horizon_ric.planner.physics.geodesy import gso_satellite_ecef

        sat_ecef = gso_satellite_ecef(180.0)
        ents = [
            {"placement": "ground", "lat_deg": 0.0, "lon_deg": 0.0},
            {"placement": "ecef", "ecef": sat_ecef},
        ]
        t = sp.build(ents)
        assert not t.los_mask[0, 1]

    def test_invalid_min_elevation_rejected(self):
        with pytest.raises(ValueError):
            SpatialPrior(min_elevation_deg=-1.0)
        with pytest.raises(ValueError):
            SpatialPrior(min_elevation_deg=90.0)

    def test_unknown_placement_rejected(self):
        sp = SpatialPrior()
        with pytest.raises(ValueError):
            sp.build([{"placement": "moon"}])

    def test_to_torch_round_trip(self):
        sp = SpatialPrior()
        ents = [
            {"placement": "ground", "lat_deg": 0.0, "lon_deg": 0.0},
            {"placement": "ground", "lat_deg": 45.0, "lon_deg": 90.0},
        ]
        t = sp.build(ents)
        torch_dict = t.to_torch()
        for key in ("node_xyz_m", "los_mask", "slant_range_m",
                    "relative_bearing_deg", "elevation_deg"):
            assert key in torch_dict
            assert isinstance(torch_dict[key], torch.Tensor)


# ─── EntityTokenizer ─────────────────────────────────────────────────────


class TestEntityTokenizer:
    def test_single_asset_shape(self):
        tok = EntityTokenizer(EntityTokenizerConfig(d_model=32))
        a = AssetAttributes(
            entity_type=EntityType.UE,
            attributes=torch.randn(16),  # default UE attr_dim=16
            xyz_m=(1e6, 0.0, 0.0),
        )
        out = tok([a])
        assert out.shape == (1, 32)

    def test_multiple_heterogeneous_assets(self):
        tok = EntityTokenizer(EntityTokenizerConfig(d_model=64))
        assets = [
            AssetAttributes(EntityType.UE, torch.randn(16), (1e6, 0.0, 0.0)),
            # v3 audit fix: SAT_NGSO default attr_dim is now LINK_STATE_DIM=12
            # (was 8, mismatched with link_state.py producer width).
            AssetAttributes(EntityType.SAT_NGSO, torch.randn(12), (7e6, 0.0, 0.0)),
            AssetAttributes(EntityType.CELL, torch.randn(12), (6.4e6, 0.0, 0.0)),
            AssetAttributes(EntityType.BEAM, torch.randn(10), (6.4e6, 0.0, 1.0)),
        ]
        out = tok(assets)
        assert out.shape == (4, 64)
        # Different types should produce different tokens (with very high prob).
        assert not torch.allclose(out[0], out[1])

    def test_empty_returns_empty_tensor(self):
        tok = EntityTokenizer(EntityTokenizerConfig(d_model=16))
        out = tok([])
        assert out.shape == (0, 16)

    def test_wrong_attr_dim_rejected(self):
        tok = EntityTokenizer(EntityTokenizerConfig(d_model=32))
        bad = AssetAttributes(
            entity_type=EntityType.UE,
            attributes=torch.randn(99),
            xyz_m=(0.0, 0.0, 0.0),
        )
        with pytest.raises(ValueError):
            tok([bad])

    def test_grad_flows_through_tokens(self):
        tok = EntityTokenizer(EntityTokenizerConfig(d_model=16))
        attr = torch.randn(16, requires_grad=True)
        a = AssetAttributes(EntityType.UE, attr, (1.0, 2.0, 3.0))
        out = tok([a])
        out.sum().backward()
        assert attr.grad is not None
        assert attr.grad.abs().sum().item() > 0
