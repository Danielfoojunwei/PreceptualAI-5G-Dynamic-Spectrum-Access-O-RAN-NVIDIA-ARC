"""
Satellite Orbital Dynamics for PreceptualAI.

GPU-accelerated computation of satellite positions, elevation angles,
Doppler shifts, and handover timing using SGP4 propagation.

All outputs are PyTorch tensors for direct use in vectorized environments.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from sgp4.api import Satrec, WGS72
from sgp4.api import jday


@dataclass
class GroundStation:
    """Ground station location."""
    name: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0


# Reference ground stations (major satellite ground station locations)
REFERENCE_STATIONS = [
    GroundStation("Svalbard_KSAT", 78.23, 15.39, 0.05),
    GroundStation("Tromso_KSAT", 69.66, 18.94, 0.10),
    GroundStation("Singapore", 1.35, 103.82, 0.01),
    GroundStation("Hawaii_AWS", 19.46, -155.59, 0.10),
    GroundStation("Oregon_AWS", 45.52, -122.68, 0.05),
    GroundStation("Sao_Paulo", -23.55, -46.63, 0.80),
    GroundStation("Dubai", 25.20, 55.27, 0.01),
    GroundStation("Sydney", -33.87, 151.21, 0.05),
    GroundStation("London", 51.51, -0.13, 0.05),
    GroundStation("Tokyo", 35.68, 139.69, 0.04),
]


class OrbitalDynamics:
    """
    Satellite orbital dynamics engine.

    Computes satellite positions, ground station visibility, elevation
    angles, Doppler shifts, and handover timing. Outputs are PyTorch
    tensors for GPU environment integration.
    """

    # Earth constants
    R_EARTH_KM = 6371.0
    MU_EARTH = 3.986004418e14  # m^3/s^2
    OMEGA_EARTH = 7.2921159e-5  # rad/s
    C_LIGHT = 299792458.0  # m/s

    def __init__(
        self,
        altitude_km: float = 550.0,
        inclination_deg: float = 53.0,
        num_satellites: int = 1,
        ground_stations: List[GroundStation] = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.altitude_km = altitude_km
        self.inclination_deg = inclination_deg
        self.num_satellites = num_satellites
        self.device = device
        self.stations = ground_stations or REFERENCE_STATIONS[:5]

        # Orbital parameters
        self.orbital_radius_km = self.R_EARTH_KM + altitude_km
        self.orbital_period_s = 2 * math.pi * math.sqrt(
            (self.orbital_radius_km * 1e3) ** 3 / self.MU_EARTH
        )
        self.orbital_velocity_ms = math.sqrt(
            self.MU_EARTH / (self.orbital_radius_km * 1e3)
        )

        # Pre-compute ground station positions in ECEF (km)
        gs_positions = []
        for gs in self.stations:
            x, y, z = self._geodetic_to_ecef(gs.lat_deg, gs.lon_deg, gs.alt_km)
            gs_positions.append([x, y, z])
        self._gs_ecef = torch.tensor(gs_positions, device=device, dtype=torch.float32)

        # Initialize satellite phases (evenly spaced in orbit)
        self._sat_phases = torch.linspace(
            0, 2 * math.pi, num_satellites + 1, device=device
        )[:-1]

        # Time tracking
        self._time_s = 0.0

    def step(self, dt_s: float = 1.0) -> None:
        """Advance simulation by dt seconds."""
        self._time_s += dt_s
        # Advance satellite phases
        omega = 2 * math.pi / self.orbital_period_s
        self._sat_phases = (self._sat_phases + omega * dt_s) % (2 * math.pi)

    def get_satellite_positions(self) -> torch.Tensor:
        """
        Compute current satellite positions in ECEF (km).

        Returns:
            positions: (num_satellites, 3) ECEF coordinates in km
        """
        incl = self.inclination_deg * math.pi / 180.0
        r = self.orbital_radius_km

        # Earth rotation
        theta_earth = self.OMEGA_EARTH * self._time_s

        # Satellite positions in orbital plane, then rotated
        x_orb = r * torch.cos(self._sat_phases)
        y_orb = r * torch.sin(self._sat_phases) * math.cos(incl)
        z_orb = r * torch.sin(self._sat_phases) * math.sin(incl)

        # Rotate by Earth's rotation (simplified — no RAAN precession)
        x_ecef = x_orb * math.cos(theta_earth) - y_orb * math.sin(theta_earth)
        y_ecef = x_orb * math.sin(theta_earth) + y_orb * math.cos(theta_earth)
        z_ecef = z_orb

        return torch.stack([x_ecef, y_ecef, z_ecef], dim=-1)

    def get_elevation_angles(self) -> torch.Tensor:
        """
        Compute elevation angles from each ground station to each satellite.

        Returns:
            elevation: (num_satellites, num_stations) elevation in degrees
        """
        sat_pos = self.get_satellite_positions()  # (S, 3)
        gs_pos = self._gs_ecef  # (G, 3)

        # Vector from GS to satellite
        # (S, 1, 3) - (1, G, 3) = (S, G, 3)
        delta = sat_pos.unsqueeze(1) - gs_pos.unsqueeze(0)
        dist = torch.norm(delta, dim=-1)  # (S, G)

        # Elevation angle = arcsin((|sat| - R_earth * cos(central_angle)) / dist)
        # Simplified: use dot product with local up vector
        gs_norm = gs_pos / torch.norm(gs_pos, dim=-1, keepdim=True)  # (G, 3) unit vectors
        cos_nadir = torch.sum(delta * gs_norm.unsqueeze(0), dim=-1) / (dist + 1e-10)  # (S, G)
        elevation_rad = torch.asin(cos_nadir.clamp(-1, 1)) - math.pi / 2
        # Correct: elevation = 90 - nadir angle
        nadir_angle = torch.acos(cos_nadir.clamp(-1, 1))
        elevation_deg = 90.0 - nadir_angle * (180.0 / math.pi)

        return elevation_deg

    def get_doppler_shift(self, freq_hz: float = 30e9) -> torch.Tensor:
        """
        Compute Doppler shift from each satellite to each ground station.

        Args:
            freq_hz: carrier frequency in Hz (default 30 GHz Ka-band)

        Returns:
            doppler_hz: (num_satellites, num_stations) Doppler shift in Hz
        """
        sat_pos = self.get_satellite_positions()  # (S, 3)
        gs_pos = self._gs_ecef  # (G, 3)

        # Satellite velocity (tangential to orbit)
        incl = self.inclination_deg * math.pi / 180.0
        v_mag = self.orbital_velocity_ms / 1e3  # km/s

        vx = -v_mag * torch.sin(self._sat_phases)
        vy = v_mag * torch.cos(self._sat_phases) * math.cos(incl)
        vz = v_mag * torch.cos(self._sat_phases) * math.sin(incl)

        theta_earth = self.OMEGA_EARTH * self._time_s
        vx_ecef = vx * math.cos(theta_earth) - vy * math.sin(theta_earth)
        vy_ecef = vx * math.sin(theta_earth) + vy * math.cos(theta_earth)
        vz_ecef = vz

        sat_vel = torch.stack([vx_ecef, vy_ecef, vz_ecef], dim=-1)  # (S, 3)

        # Line-of-sight unit vector from GS to satellite
        delta = sat_pos.unsqueeze(1) - gs_pos.unsqueeze(0)  # (S, G, 3)
        dist = torch.norm(delta, dim=-1, keepdim=True)  # (S, G, 1)
        los = delta / (dist + 1e-10)  # (S, G, 3)

        # Radial velocity = dot(sat_vel, los)
        v_radial = torch.sum(sat_vel.unsqueeze(1) * los, dim=-1)  # (S, G) in km/s
        v_radial_ms = v_radial * 1e3  # m/s

        # Doppler shift: f_d = -f * v_r / c
        doppler_hz = -freq_hz * v_radial_ms / self.C_LIGHT

        return doppler_hz

    def get_visibility_mask(self, min_elevation_deg: float = 5.0) -> torch.Tensor:
        """
        Compute which satellites are visible from which ground stations.

        Returns:
            visible: (num_satellites, num_stations) bool tensor
        """
        elevation = self.get_elevation_angles()
        return elevation > min_elevation_deg

    def get_time_to_set(self, min_elevation_deg: float = 5.0) -> torch.Tensor:
        """
        Estimate time until each visible satellite sets below min elevation.

        Returns:
            time_to_set_s: (num_satellites, num_stations) seconds until set
                           (inf for non-visible satellites)
        """
        elevation = self.get_elevation_angles()
        visible = elevation > min_elevation_deg

        # Approximate: at LEO 550km, max pass duration ~10 min
        # Linear approximation based on current elevation
        max_pass_s = 600.0  # 10 minutes
        # Higher elevation = more time remaining (roughly)
        fraction_remaining = (elevation - min_elevation_deg).clamp(min=0) / 85.0
        time_remaining = max_pass_s * fraction_remaining

        time_remaining[~visible] = float("inf")
        return time_remaining

    def get_best_satellite(self, min_elevation_deg: float = 5.0) -> torch.Tensor:
        """
        For each ground station, find the best satellite (highest elevation).

        Returns:
            best_sat_idx: (num_stations,) index of best satellite per station
        """
        elevation = self.get_elevation_angles()  # (S, G)
        elevation[elevation < min_elevation_deg] = -90.0
        return elevation.argmax(dim=0)

    @staticmethod
    def _geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float) -> Tuple[float, float, float]:
        """Convert geodetic coordinates to ECEF (km)."""
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        R = 6371.0 + alt_km
        x = R * math.cos(lat) * math.cos(lon)
        y = R * math.cos(lat) * math.sin(lon)
        z = R * math.sin(lat)
        return x, y, z

    @property
    def num_stations(self) -> int:
        return len(self.stations)

    @property
    def time_s(self) -> float:
        return self._time_s
