from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .geometry import Orientation, cisTEM_rotation_matrix

Tensor = torch.Tensor


@dataclass(slots=True)
class OrientationBatchTensors:
    """Tensor views for one orientation batch."""

    phi: Tensor
    theta: Tensor
    psi: Tensor
    orientation_index: Tensor
    rotation_matrices: Tensor


@dataclass(slots=True)
class WinnerMetadataTable:
    """Lookup table used to materialize winner maps once after the search."""

    phi: Tensor
    theta: Tensor
    psi: Tensor
    defocus_offsets: Tensor
    pixel_size_offsets: Tensor
    total_orientation_count: int

    def materialize(
        self,
        winner_task_index: Tensor,
        mip: Tensor,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if tuple(winner_task_index.shape) != tuple(mip.shape):
            raise ValueError("winner_task_index and mip must have the same shape")
        if self.total_orientation_count <= 0:
            raise ValueError("total_orientation_count must be positive")
        if self.defocus_offsets.numel() <= 0 or self.pixel_size_offsets.numel() <= 0:
            raise ValueError("defocus and pixel-size lookup tables cannot be empty")

        device = winner_task_index.device
        sentinel = torch.iinfo(torch.int64).max
        # A finite task index is the authoritative winner marker.  This also
        # supports the optional negative-infinity MIP initialization without
        # changing the default cisTEM-compatible MIP=0 behavior.
        valid = winner_task_index != sentinel
        safe = torch.where(valid, winner_task_index, torch.zeros_like(winner_task_index))

        orientation_index = torch.remainder(safe, int(self.total_orientation_count))
        condition_index = torch.div(
            safe,
            int(self.total_orientation_count),
            rounding_mode="floor",
        )
        defocus_count = int(self.defocus_offsets.numel())
        defocus_index = torch.remainder(condition_index, defocus_count)
        pixel_index = torch.div(condition_index, defocus_count, rounding_mode="floor")

        def lookup(values: Tensor, indices: Tensor) -> Tensor:
            source = values.to(device=device)
            selected = source[indices]
            return torch.where(valid, selected.to(dtype), torch.zeros((), device=device, dtype=dtype))

        return (
            lookup(self.phi, orientation_index),
            lookup(self.theta, orientation_index),
            lookup(self.psi, orientation_index),
            lookup(self.defocus_offsets, defocus_index),
            lookup(self.pixel_size_offsets, pixel_index),
        )


class OrientationTensorTable:
    """All Euler angles, indices, and optionally rotation matrices on one device."""

    def __init__(
        self,
        orientations: Sequence[Orientation],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        precompute_rotation_matrices: bool = True,
        precompute_chunk_size: int = 262_144,
    ) -> None:
        if not orientations:
            raise ValueError("orientation table cannot be empty")
        if precompute_chunk_size <= 0:
            raise ValueError("precompute_chunk_size must be positive")
        self.device = device
        self.dtype = dtype
        self.count = len(orientations)

        phi_np = np.fromiter((o.phi_deg for o in orientations), dtype=np.float32, count=self.count)
        theta_np = np.fromiter((o.theta_deg for o in orientations), dtype=np.float32, count=self.count)
        psi_np = np.fromiter((o.psi_deg for o in orientations), dtype=np.float32, count=self.count)
        index_np = np.fromiter((o.orientation_index for o in orientations), dtype=np.int64, count=self.count)

        self.phi = torch.from_numpy(phi_np).to(device=device, dtype=dtype)
        self.theta = torch.from_numpy(theta_np).to(device=device, dtype=dtype)
        self.psi = torch.from_numpy(psi_np).to(device=device, dtype=dtype)
        self.orientation_index = torch.from_numpy(index_np).to(device=device, dtype=torch.int64)

        self.rotation_matrices: Tensor | None = None
        if precompute_rotation_matrices:
            matrices = torch.empty((self.count, 3, 3), device=device, dtype=dtype)
            for start in range(0, self.count, int(precompute_chunk_size)):
                stop = min(start + int(precompute_chunk_size), self.count)
                matrices[start:stop] = cisTEM_rotation_matrix(
                    self.phi[start:stop],
                    self.theta[start:stop],
                    self.psi[start:stop],
                    device=device,
                    dtype=dtype,
                )
            self.rotation_matrices = matrices

    @staticmethod
    def _contiguous_range(indices: Sequence[int]) -> tuple[int, int] | None:
        if not indices:
            return None
        start = int(indices[0])
        stop = start + len(indices)
        if int(indices[-1]) != stop - 1:
            return None
        if any(int(value) != start + offset for offset, value in enumerate(indices)):
            return None
        return start, stop

    def take(self, global_indices: Sequence[int]) -> OrientationBatchTensors:
        contiguous = self._contiguous_range(global_indices)
        if contiguous is not None:
            start, stop = contiguous
            phi = self.phi[start:stop]
            theta = self.theta[start:stop]
            psi = self.psi[start:stop]
            orientation_index = self.orientation_index[start:stop]
            matrices = (
                self.rotation_matrices[start:stop]
                if self.rotation_matrices is not None
                else cisTEM_rotation_matrix(phi, theta, psi, device=self.device, dtype=self.dtype)
            )
        else:
            index = torch.as_tensor(global_indices, device=self.device, dtype=torch.int64)
            phi = self.phi.index_select(0, index)
            theta = self.theta.index_select(0, index)
            psi = self.psi.index_select(0, index)
            orientation_index = self.orientation_index.index_select(0, index)
            matrices = (
                self.rotation_matrices.index_select(0, index)
                if self.rotation_matrices is not None
                else cisTEM_rotation_matrix(phi, theta, psi, device=self.device, dtype=self.dtype)
            )
        return OrientationBatchTensors(
            phi=phi,
            theta=theta,
            psi=psi,
            orientation_index=orientation_index,
            rotation_matrices=matrices,
        )

    def winner_metadata(
        self,
        defocus_offsets: Sequence[float],
        pixel_size_offsets: Sequence[float],
    ) -> WinnerMetadataTable:
        return WinnerMetadataTable(
            phi=self.phi,
            theta=self.theta,
            psi=self.psi,
            defocus_offsets=torch.as_tensor(defocus_offsets, device=self.device, dtype=self.dtype),
            pixel_size_offsets=torch.as_tensor(pixel_size_offsets, device=self.device, dtype=self.dtype),
            total_orientation_count=self.count,
        )
