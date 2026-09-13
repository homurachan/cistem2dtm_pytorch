from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import numpy as np
import torch

from .constants import DEG_TO_RAD, RAD_TO_DEG

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class Orientation:
    phi_deg: float
    theta_deg: float
    psi_deg: float
    grid_index: int
    orientation_index: int


def cisTEM_rotation_matrix(
    phi_deg: float | Tensor,
    theta_deg: float | Tensor,
    psi_deg: float | Tensor,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Direct translation of cisTEM ``RotationMatrix::SetToEulerRotation``.

    No external Euler-angle library is used.  The matrix entries preserve
    cisTEM's active, intrinsic Z-Y'-Z'' convention at source commit 5bc5f8c.
    """
    phi = torch.as_tensor(phi_deg, device=device, dtype=dtype) * DEG_TO_RAD
    theta = torch.as_tensor(theta_deg, device=device, dtype=dtype) * DEG_TO_RAD
    psi = torch.as_tensor(psi_deg, device=device, dtype=dtype) * DEG_TO_RAD
    phi, theta, psi = torch.broadcast_tensors(phi, theta, psi)

    cphi, sphi = torch.cos(phi), torch.sin(phi)
    ctheta, stheta = torch.cos(theta), torch.sin(theta)
    cpsi, spsi = torch.cos(psi), torch.sin(psi)

    m00 = cphi * ctheta * cpsi - sphi * spsi
    m10 = sphi * ctheta * cpsi + cphi * spsi
    m20 = -stheta * cpsi
    m01 = -cphi * ctheta * spsi - sphi * cpsi
    m11 = -sphi * ctheta * spsi + cphi * cpsi
    m21 = stheta * spsi
    m02 = stheta * cphi
    m12 = stheta * sphi
    m22 = ctheta

    return torch.stack(
        (
            torch.stack((m00, m01, m02), dim=-1),
            torch.stack((m10, m11, m12), dim=-1),
            torch.stack((m20, m21, m22), dim=-1),
        ),
        dim=-2,
    )


def _parse_symmetry(symmetry: str) -> tuple[str, int]:
    text = symmetry.strip().upper()
    match = re.fullmatch(r"([CD])\s*(\d+)", text)
    if match:
        order = int(match.group(2))
        if order <= 0:
            raise ValueError("symmetry order must be positive")
        return match.group(1), order
    if text in {"T", "O", "I"}:
        return text, 1
    raise ValueError(f"unsupported cisTEM symmetry: {symmetry!r}")


def _f32(value: float | int | np.floating) -> np.float32:
    """Round one scalar operation to C++ ``float`` precision."""
    return np.float32(value)


def _cxx_positive_int(value: np.float32) -> int:
    """C++ cast-to-int semantics for the positive quantities used here."""
    return int(value)


@dataclass(slots=True)
class EulerSearch:
    """cisTEM EulerSearch grid with C++ float32 loop semantics.

    The apparently minor float32 details are essential.  In cisTEM,
    ``theta``, ``phi``, their steps, and the loop accumulators are all C++
    ``float`` values.  Python float64 accumulation changes which values land
    just below ``phi_max`` and therefore changes the number of views.  For
    C1/3 degrees with cyclic theta extended to 180 degrees, this implementation
    returns 4606 views (indices 0..4605), matching the target source.
    """

    symmetry: str
    angular_step_deg: float
    psi_step_deg: float
    preserve_psi_360_duplicate: bool = True
    extend_cyclic_theta_to_180: bool = True

    def _ranges_f32(self) -> tuple[np.float32, np.float32, bool]:
        family, order = _parse_symmetry(self.symmetry)
        if family == "C":
            phi_max = _f32(_f32(360.0) / _f32(order))
            theta_max = _f32(180.0 if self.extend_cyclic_theta_to_180 else 90.0)
            return phi_max, theta_max, True
        if family == "D":
            return _f32(_f32(360.0) / _f32(order)), _f32(90.0), False
        if family == "T":
            return _f32(180.0), _f32(54.7), False
        if family == "O":
            return _f32(90.0), _f32(54.7), False
        if family == "I":
            return _f32(180.0), _f32(31.7), False
        raise AssertionError(family)

    def _ranges(self) -> tuple[float, float, bool]:
        phi, theta, mirror = self._ranges_f32()
        return float(phi), float(theta), mirror

    def out_of_plane_positions(self) -> list[tuple[float, float]]:
        if self.angular_step_deg <= 0:
            raise ValueError("angular_step_deg must be positive")

        phi_max, theta_max, _ = self._ranges_f32()
        angular_step = _f32(self.angular_step_deg)
        intervals = _cxx_positive_int(_f32(_f32(theta_max / angular_step) + _f32(0.5)))
        if intervals <= 0:
            raise ValueError("angular_step_deg produced zero theta intervals")
        theta_step = _f32(theta_max / _f32(intervals))
        theta_limit = _f32(theta_max + _f32(theta_step / _f32(2.0)))

        positions: list[tuple[float, float]] = []
        theta = _f32(0.0)
        while bool(theta < theta_limit):
            if bool(theta == _f32(0.0)) or bool(theta == _f32(180.0)):
                phi_step = phi_max
            else:
                # deg_2_rad(float), sinf, fabsf and all divisions are float.
                theta_rad = _f32(theta * _f32(math.pi / 180.0))
                sine = _f32(np.sin(theta_rad))
                phi_step = _f32(abs(_f32(angular_step / sine)))
                if bool(phi_step > phi_max):
                    phi_step = phi_max
                phi_intervals = _cxx_positive_int(
                    _f32(_f32(phi_max / phi_step) + _f32(0.5))
                )
                if phi_intervals <= 0:
                    phi_intervals = 1
                phi_step = _f32(phi_max / _f32(phi_intervals))

            # Source uses ``for (phi = 0.0; phi < phi_max; phi += phi_step)``
            # with no epsilon.  Preserve that exact comparison.
            phi = _f32(0.0)
            while bool(phi < phi_max):
                positions.append((float(phi), float(theta)))
                phi = _f32(phi + phi_step)
            theta = _f32(theta + theta_step)
        return positions

    def psi_values(self) -> list[float]:
        if self.psi_step_deg <= 0:
            raise ValueError("psi_step_deg must be positive")
        step = _f32(self.psi_step_deg)
        current = _f32(0.0)
        maximum = _f32(360.0)
        values: list[float] = []
        # match_template.cpp uses a float accumulation loop, not integer
        # multiplication.  Explicit user steps are not re-quantized.
        while bool(current <= maximum):
            values.append(float(current))
            current = _f32(current + step)
        if not self.preserve_psi_360_duplicate and values:
            # Remove the geometrically duplicate endpoint only when the loop
            # actually reached 360 (within one float32 ulp-scale tolerance).
            if abs(values[-1] - 360.0) <= 1.0e-4:
                values.pop()
        return values

    def orientations(self) -> list[Orientation]:
        positions = self.out_of_plane_positions()
        psi_values = self.psi_values()
        out: list[Orientation] = []
        orientation_index = 0
        for grid_index, (phi, theta) in enumerate(positions):
            for psi in psi_values:
                out.append(
                    Orientation(
                        phi_deg=phi,
                        theta_deg=theta,
                        psi_deg=psi,
                        grid_index=grid_index,
                        orientation_index=orientation_index,
                    )
                )
                orientation_index += 1
        return out

    def iter_orientations(self) -> Iterator[Orientation]:
        yield from self.orientations()

    @property
    def number_of_out_of_plane_positions(self) -> int:
        return len(self.out_of_plane_positions())

    @property
    def last_out_of_plane_grid_index(self) -> int:
        return self.number_of_out_of_plane_positions - 1

    @property
    def number_of_psi_positions(self) -> int:
        return len(self.psi_values())

    @property
    def number_of_orientations(self) -> int:
        return self.number_of_out_of_plane_positions * self.number_of_psi_positions


def cisTEM_auto_angular_step(required_resolution_angstrom: float, mask_radius_angstrom: float) -> float:
    """Direct float32 translation of cisTEM ``CalculateAngularStep``."""
    if required_resolution_angstrom <= 0 or mask_radius_angstrom <= 0:
        raise ValueError("resolution and mask radius must be positive")
    value = _f32(
        _f32(RAD_TO_DEG)
        * _f32(_f32(2.0) * _f32(required_resolution_angstrom) / _f32(mask_radius_angstrom))
    )
    return float(value)


def cisTEM_auto_psi_step(search_pixel_size_angstrom: float, mask_radius_angstrom: float) -> float:
    """Float32 translation of match_template.cpp's automatic psi rule."""
    if search_pixel_size_angstrom <= 0 or mask_radius_angstrom <= 0:
        raise ValueError("pixel size and mask radius must be positive")
    raw = _f32(
        _f32(RAD_TO_DEG)
        * _f32(_f32(search_pixel_size_angstrom) / _f32(mask_radius_angstrom))
    )
    intervals = _cxx_positive_int(_f32(_f32(360.0) / raw + _f32(0.5)))
    intervals = max(intervals, 1)
    return float(_f32(_f32(360.0) / _f32(intervals)))


def angle_tensor(orientations: Iterable[Orientation], *, device: torch.device, dtype: torch.dtype) -> Tensor:
    values = [(o.phi_deg, o.theta_deg, o.psi_deg) for o in orientations]
    return torch.tensor(values, device=device, dtype=dtype)
