from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .constants import DEG_TO_RAD
from .fft import reciprocal_frequencies_rfft2

Tensor = torch.Tensor


def electron_wavelength_angstrom(voltage_kv: float) -> float:
    """Relativistic electron wavelength in angstroms."""
    volts = float(voltage_kv) * 1000.0
    return 12.2639 / math.sqrt(volts * (1.0 + 0.97845e-6 * volts))


@dataclass(slots=True)
class CTF:
    voltage_kv: float
    spherical_aberration_mm: float
    amplitude_contrast: float
    pixel_size_angstrom: float
    defocus1_angstrom: float
    defocus2_angstrom: float
    astigmatism_angle_deg: float = 0.0
    additional_phase_shift_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.pixel_size_angstrom <= 0:
            raise ValueError("pixel_size_angstrom must be positive")
        if not 0 <= self.amplitude_contrast <= 1:
            raise ValueError("amplitude_contrast must be between 0 and 1")

    @property
    def wavelength_pixels(self) -> float:
        return electron_wavelength_angstrom(self.voltage_kv) / self.pixel_size_angstrom

    @property
    def spherical_aberration_pixels(self) -> float:
        return self.spherical_aberration_mm * 1.0e7 / self.pixel_size_angstrom

    @property
    def defocus1_pixels(self) -> float:
        return self.defocus1_angstrom / self.pixel_size_angstrom

    @property
    def defocus2_pixels(self) -> float:
        return self.defocus2_angstrom / self.pixel_size_angstrom

    @property
    def astigmatism_angle_rad(self) -> float:
        return self.astigmatism_angle_deg * DEG_TO_RAD

    @property
    def additional_phase_shift_rad(self) -> float:
        return self.additional_phase_shift_deg * DEG_TO_RAD

    @property
    def amplitude_contrast_phase_shift(self) -> float:
        if abs(self.amplitude_contrast - 1.0) < 1.0e-3:
            return math.pi / 2.0
        return math.atan(
            self.amplitude_contrast
            / math.sqrt(max(1.0 - self.amplitude_contrast * self.amplitude_contrast, 1.0e-12))
        )

    def with_defocus_offset(self, offset_angstrom: float) -> "CTF":
        return CTF(
            voltage_kv=self.voltage_kv,
            spherical_aberration_mm=self.spherical_aberration_mm,
            amplitude_contrast=self.amplitude_contrast,
            pixel_size_angstrom=self.pixel_size_angstrom,
            defocus1_angstrom=self.defocus1_angstrom + offset_angstrom,
            defocus2_angstrom=self.defocus2_angstrom + offset_angstrom,
            astigmatism_angle_deg=self.astigmatism_angle_deg,
            additional_phase_shift_deg=self.additional_phase_shift_deg,
        )

    def defocus_given_azimuth(self, azimuth_rad: Tensor) -> Tensor:
        d1 = self.defocus1_pixels
        d2 = self.defocus2_pixels
        return 0.5 * (
            d1
            + d2
            + torch.cos(2.0 * (azimuth_rad - self.astigmatism_angle_rad)) * (d1 - d2)
        )

    def phase_aberration(self, squared_spatial_frequency: Tensor, azimuth_rad: Tensor) -> Tensor:
        wavelength = self.wavelength_pixels
        defocus = self.defocus_given_azimuth(azimuth_rad)
        return (
            math.pi
            * wavelength
            * squared_spatial_frequency
            * (
                defocus
                - 0.5
                * wavelength
                * wavelength
                * squared_spatial_frequency
                * self.spherical_aberration_pixels
            )
            + self.additional_phase_shift_rad
            + self.amplitude_contrast_phase_shift
        )

    def evaluate(self, squared_spatial_frequency: Tensor, azimuth_rad: Tensor) -> Tensor:
        if self.defocus1_pixels == 0.0 and self.defocus2_pixels == 0.0:
            return torch.full_like(squared_spatial_frequency, -0.7)
        return -torch.sin(self.phase_aberration(squared_spatial_frequency, azimuth_rad))

    def image(
        self,
        shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        h, w = shape
        fx, fy, _ = reciprocal_frequencies_rfft2(h, w, device=device, dtype=dtype)
        squared = fx.square() + fy.square()
        azimuth = torch.atan2(fy, fx)
        return self.evaluate(squared, azimuth)
