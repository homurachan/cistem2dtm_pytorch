from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from .config import WeightingConfig
from .ctf import CTF
from .fft import reciprocal_frequencies_rfft2
from .gisspa import (
    repository_image_frequency_weight,
    repository_projection_frequency_weight,
)
from .image_ops import RadialCurve, evaluate_radial_curve

Tensor = torch.Tensor


class ExtraProjectionFilter(Protocol):
    """Extension point for additional projection-side filters."""

    def evaluate(self, shape: tuple[int, int], *, device: torch.device, dtype: torch.dtype) -> Tensor: ...


@dataclass(slots=True)
class FilterPipeline:
    whitening_curve: RadialCurve
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    low_resolution_limit_angstrom: float = 300.0
    high_resolution_limit_angstrom: float = 8.0
    extra_filters: list[ExtraProjectionFilter] = field(default_factory=list)

    @property
    def per_projection_whitening(self) -> bool:
        return self.weighting.mode == "gisspa_repo"

    def _ctf_values(
        self,
        ctf: CTF,
        shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        normalized_frequency_scale: float,
    ) -> tuple[Tensor, Tensor]:
        h, w = shape
        fx, fy, radius = reciprocal_frequencies_rfft2(h, w, device=device, dtype=dtype)
        scale = float(normalized_frequency_scale)
        squared = (fx * scale).square() + (fy * scale).square()
        azimuth = torch.atan2(fy, fx)
        return ctf.evaluate(squared, azimuth), radius

    def build_projection(
        self,
        ctf: CTF,
        shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        normalized_frequency_scale: float = 1.0,
    ) -> Tensor:
        """Build the projection-side frequency filter.

        ``cistem`` is unchanged from v0.3.4.  ``gisspa_repo`` assumes the
        central slice has already been independently whitened and applies the
        signed repository CTF weight on the projection grid.
        """
        ctf_values, radius = self._ctf_values(
            ctf,
            shape,
            device=device,
            dtype=dtype,
            normalized_frequency_scale=normalized_frequency_scale,
        )
        scale = float(normalized_frequency_scale)
        if self.weighting.mode == "cistem":
            whitening = evaluate_radial_curve(
                self.whitening_curve.to(device, dtype), radius * scale
            )
            result = ctf_values * whitening
        elif self.weighting.mode == "gisspa_repo":
            result = repository_projection_frequency_weight(
                ctf_values,
                shape,
                pixel_size_angstrom=ctf.pixel_size_angstrom,
                high_resolution_angstrom=self.high_resolution_limit_angstrom,
                kk=self.weighting.kk,
                a=self.weighting.a,
                b=self.weighting.b,
                b2=self.weighting.b2,
                bfactor=self.weighting.bfactor,
                bfactor2=self.weighting.bfactor2,
                bfactor3=self.weighting.bfactor3,
                cosine_edge_width_pixels=self.weighting.cosine_edge_width_pixels,
                epsilon=self.weighting.projection_whitening_epsilon,
                normalized_frequency_scale=scale,
            )
        else:  # validated config, defensive for programmatic callers
            raise ValueError(f"unknown weighting mode: {self.weighting.mode}")

        for extra in self.extra_filters:
            result = result * extra.evaluate(shape, device=device, dtype=dtype)
        result = result.clone()
        result[0, 0] = 0.0
        return result

    # Backward-compatible name used by earlier releases and external callers.
    build = build_projection

    def build_image_weight(
        self,
        ctf: CTF,
        shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Build the full-search-grid image-side GisSPA factor."""
        if self.weighting.mode != "gisspa_repo":
            return torch.ones(
                (int(shape[0]), int(shape[1]) // 2 + 1),
                device=device,
                dtype=dtype,
            )
        ctf_values, _ = self._ctf_values(
            ctf,
            shape,
            device=device,
            dtype=dtype,
            normalized_frequency_scale=1.0,
        )
        return repository_image_frequency_weight(
            ctf_values,
            shape,
            pixel_size_angstrom=ctf.pixel_size_angstrom,
            low_resolution_angstrom=self.low_resolution_limit_angstrom,
            high_resolution_angstrom=self.high_resolution_limit_angstrom,
            kk=self.weighting.kk,
            a=self.weighting.a,
            b=self.weighting.b,
            b2=self.weighting.b2,
            bfactor=self.weighting.bfactor,
            bfactor2=self.weighting.bfactor2,
            bfactor3=self.weighting.bfactor3,
            cosine_edge_width_pixels=self.weighting.cosine_edge_width_pixels,
            image_high_frequency_damping=self.weighting.image_high_frequency_damping,
            epsilon=self.weighting.projection_whitening_epsilon,
        )
