"""SpatialPrior — pure-geometry tensors that bias the world model.

Given the rApp's known WGS-84 geometry (ground stations, satellites in
known orbits, beam footprints), every later attention layer should
HONOUR these positions, not relearn them. SpatialPrior materialises:

    node_xyz_m       (N, 3)   ECEF position of each entity
    los_mask         (N, N)   1 if i can see j above min-elevation, else 0
    slant_range_m    (N, N)   slant distance i↔j (only meaningful when LOS)
    relative_bearing (N, N)   azimuth from i to j in degrees, [0, 360)

These four tensors are what the design briefs call the "spatial prior"
that downstream Perceiver/HGT layers add as attention bias (à la ALiBi /
RoPE-3D). Zero learnable parameters; updates each tick.

The class is deliberately framework-light: outputs are NumPy arrays. A
thin `to_torch()` helper converts when callers need PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from horizon_ric.planner.physics.geodesy import (
    ECEF,
    geodetic_to_ecef,
    look_angles_to_target,
)


@dataclass(frozen=True)
class SpatialPriorTensor:
    """Bundle of geometry tensors. All arrays share the leading axis N
    (number of entities). Entries [i, j] use convention "from i to j"."""

    node_xyz_m: np.ndarray         # (N, 3)
    los_mask: np.ndarray           # (N, N), bool
    slant_range_m: np.ndarray      # (N, N), float64
    relative_bearing_deg: np.ndarray  # (N, N), float64
    elevation_deg: np.ndarray      # (N, N), float64

    def to_torch(self):  # pragma: no cover — tested under torch path
        import torch

        return {
            "node_xyz_m": torch.from_numpy(self.node_xyz_m).to(torch.float32),
            "los_mask": torch.from_numpy(self.los_mask),
            "slant_range_m": torch.from_numpy(self.slant_range_m).to(torch.float32),
            "relative_bearing_deg": torch.from_numpy(self.relative_bearing_deg).to(
                torch.float32
            ),
            "elevation_deg": torch.from_numpy(self.elevation_deg).to(torch.float32),
        }


class SpatialPrior:
    """Build spatial-prior tensors from a list of (kind, lat, lon, h_m).

    Two entity placements are supported:

      * "ground": placed at WGS-84 (lat, lon, height) — gateways, ES, UEs.
      * "ecef":   ECEF coordinates supplied directly — satellites whose
                  position came from `orbital.state_ecef_m()`.
    """

    def __init__(self, min_elevation_deg: float = 5.0):
        if not 0.0 <= min_elevation_deg < 90.0:
            raise ValueError(
                f"min_elevation_deg must be in [0,90), got {min_elevation_deg}"
            )
        self.min_elevation_deg = min_elevation_deg

    def build(
        self,
        entities: list[dict],
    ) -> SpatialPriorTensor:
        """Compute the spatial-prior tensors.

        Each entity dict must have:
            placement: "ground" | "ecef"
            lat_deg, lon_deg, height_m   (when placement="ground")
            ecef                         (ECEF) (when placement="ecef")

        Returns:
            `SpatialPriorTensor` whose [i, j] entry describes the
            geometry from entity i to entity j.
        """
        if not entities:
            return SpatialPriorTensor(
                node_xyz_m=np.zeros((0, 3)),
                los_mask=np.zeros((0, 0), dtype=bool),
                slant_range_m=np.zeros((0, 0)),
                relative_bearing_deg=np.zeros((0, 0)),
                elevation_deg=np.zeros((0, 0)),
            )

        n = len(entities)
        ecefs: list[ECEF] = []
        ground_indices: list[int] = []
        latlon_for_ground: list[tuple[float, float, float]] = []
        for i, e in enumerate(entities):
            placement = e.get("placement", "ground")
            if placement == "ground":
                lat, lon, h = (
                    float(e["lat_deg"]),
                    float(e["lon_deg"]),
                    float(e.get("height_m", 0.0)),
                )
                ecefs.append(geodetic_to_ecef(lat, lon, h))
                ground_indices.append(i)
                latlon_for_ground.append((lat, lon, h))
            elif placement == "ecef":
                p = e["ecef"]
                if not isinstance(p, ECEF):
                    raise TypeError(
                        f"entity {i} placement=ecef requires `ecef: ECEF`"
                    )
                ecefs.append(p)
            else:
                raise ValueError(
                    f"entity {i} has unknown placement {placement!r}"
                )

        node_xyz = np.array([[p.x, p.y, p.z] for p in ecefs], dtype=np.float64)

        los_mask = np.zeros((n, n), dtype=bool)
        slant = np.zeros((n, n), dtype=np.float64)
        bearing = np.zeros((n, n), dtype=np.float64)
        elev = np.zeros((n, n), dtype=np.float64)

        # Pairwise look-angles. For "ground" sources we use the proper
        # WGS-84 ENU computation; for "ecef" sources (satellites looking
        # down) we fall back to a great-circle-style angle through the
        # geocentric origin — sufficient for visibility/bearing of the
        # downstream beam-pointing prior.
        # Build a reverse index from entity index to its ground (lat, lon, h)
        # to feed `look_angles_to_target`.
        ground_lookup: dict[int, tuple[float, float, float]] = {
            idx: ll for idx, ll in zip(ground_indices, latlon_for_ground)
        }

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                target = ecefs[j]
                if i in ground_lookup:
                    lat, lon, h = ground_lookup[i]
                    look = look_angles_to_target(lat, lon, target, h)
                    elev[i, j] = look.elevation_deg
                    slant[i, j] = look.range_m
                    bearing[i, j] = look.azimuth_deg
                    los_mask[i, j] = look.elevation_deg >= self.min_elevation_deg
                else:
                    # Source is a satellite (or generic ECEF). Compute
                    # geocentric vector to target and a coarse "elevation"
                    # = 90° − angle(source vector, source→target vector).
                    s = ecefs[i]
                    dx = target.x - s.x
                    dy = target.y - s.y
                    dz = target.z - s.z
                    dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                    slant[i, j] = dist
                    src_norm = float(np.sqrt(s.x * s.x + s.y * s.y + s.z * s.z))
                    if dist > 0 and src_norm > 0:
                        cos_a = (s.x * dx + s.y * dy + s.z * dz) / (src_norm * dist)
                        # Dot product with the ECEF "up" vector (sat-centric).
                        cos_a = max(-1.0, min(1.0, cos_a))
                        nadir_angle = float(np.degrees(np.arccos(cos_a)))
                        # Look angle below local horizon when nadir > 90°.
                        elev[i, j] = 90.0 - nadir_angle
                        los_mask[i, j] = elev[i, j] >= self.min_elevation_deg
                        # Bearing from sat to target in the local ENU
                        # is approximated by atan2 of (Δx, Δy); good enough
                        # for relative attention bias.
                        bearing[i, j] = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
                    else:
                        los_mask[i, j] = False

        return SpatialPriorTensor(
            node_xyz_m=node_xyz,
            los_mask=los_mask,
            slant_range_m=slant,
            relative_bearing_deg=bearing,
            elevation_deg=elev,
        )


__all__ = ["SpatialPrior", "SpatialPriorTensor"]
