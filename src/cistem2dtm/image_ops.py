from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .constants import TINY
from .fft import (
    full_fft_resize_real,
    reciprocal_frequencies_rfft2,
    rfft2_cistem,
    rfft_sum_of_squares,
)

Tensor = torch.Tensor


@dataclass(slots=True)
class RadialCurve:
    x: Tensor
    y: Tensor

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "RadialCurve":
        return RadialCurve(self.x.to(device=device, dtype=dtype), self.y.to(device=device, dtype=dtype))

    @property
    def number_of_points(self) -> int:
        return int(self.x.numel())


def replace_outliers_with_mean(image: Tensor, sigma_threshold: float = 5.0) -> tuple[Tensor, Tensor]:
    if image.ndim != 2:
        raise ValueError("replace_outliers_with_mean expects a 2-D image")
    mean = image.mean()
    sigma = image.std(unbiased=False).clamp_min(TINY)
    mask = torch.abs(image - mean) > sigma_threshold * sigma
    return torch.where(mask, mean, image), mask


def edge_mean(image: Tensor) -> Tensor:
    if image.ndim != 2:
        raise ValueError("edge_mean expects a 2-D tensor")
    h, w = image.shape
    if h == 1 or w == 1:
        return image.mean()
    if h == 2:
        return image.mean()
    return torch.cat((image[0], image[-1], image[1:-1, 0], image[1:-1, -1])).mean()


def center_pad_crop(
    image: Tensor,
    output_shape: tuple[int, int],
    *,
    fill_value: float | Tensor = 0.0,
) -> tuple[Tensor, tuple[slice, slice], tuple[slice, slice]]:
    """Center clip/pad a 2-D tensor, returning source and destination slices."""
    if image.ndim != 2:
        raise ValueError("center_pad_crop expects a 2-D tensor")
    in_h, in_w = (int(v) for v in image.shape)
    out_h, out_w = (int(v) for v in output_shape)
    if isinstance(fill_value, Tensor):
        out = torch.empty((out_h, out_w), device=image.device, dtype=image.dtype)
        out.fill_(float(fill_value.item()))
    else:
        out = torch.full((out_h, out_w), float(fill_value), device=image.device, dtype=image.dtype)

    copy_h, copy_w = min(in_h, out_h), min(in_w, out_w)
    src_y0 = (in_h - copy_h) // 2
    src_x0 = (in_w - copy_w) // 2
    dst_y0 = (out_h - copy_h) // 2
    dst_x0 = (out_w - copy_w) // 2
    src = (slice(src_y0, src_y0 + copy_h), slice(src_x0, src_x0 + copy_w))
    dst = (slice(dst_y0, dst_y0 + copy_h), slice(dst_x0, dst_x0 + copy_w))
    out[dst] = image[src]
    return out, src, dst


def _symmetric_index(coords: Tensor, size: int) -> Tensor:
    """Repeated-edge symmetric indices used by cisTEM replicative padding."""
    if size <= 1:
        return torch.zeros_like(coords, dtype=torch.long)
    period = 2 * int(size)
    mod = torch.remainder(coords, period)
    return torch.where(mod < size, mod, period - 1 - mod).to(torch.long)


def center_pad_crop_with_mode(
    image: Tensor,
    output_shape: tuple[int, int],
    *,
    mode: str,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor]:
    """Center clip/pad a 2-D image and return the directly copied-pixel mask.

    ``noise`` reproduces cisTEM's default real-space padding after first-pass
    whitening: every padding pixel is drawn independently from N(0, 1).
    ``replicate`` implements the repeated-edge symmetric extension used by
    ``ClipIntoWithReplicativePadding``.  ``edge`` is retained as the legacy
    edge-mean fill mode from cistem2dtm 0.2.0.
    """
    if image.ndim != 2:
        raise ValueError("center_pad_crop_with_mode expects a 2-D image")
    out_h, out_w = (int(v) for v in output_shape)
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output dimensions must be positive")

    in_h, in_w = (int(v) for v in image.shape)
    copy_h, copy_w = min(in_h, out_h), min(in_w, out_w)
    src_y0 = (in_h - copy_h) // 2
    src_x0 = (in_w - copy_w) // 2
    source = image[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
    dst_y0 = (out_h - copy_h) // 2
    dst_x0 = (out_w - copy_w) // 2

    valid = torch.zeros((out_h, out_w), device=image.device, dtype=torch.bool)
    valid[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = True

    if mode == "replicate":
        in_cy, in_cx = copy_h // 2, copy_w // 2
        out_cy, out_cx = out_h // 2, out_w // 2
        y = torch.arange(out_h, device=image.device, dtype=torch.long) - out_cy + in_cy
        x = torch.arange(out_w, device=image.device, dtype=torch.long) - out_cx + in_cx
        y = _symmetric_index(y, copy_h)
        x = _symmetric_index(x, copy_w)
        return source.index_select(0, y).index_select(1, x), valid

    if mode == "noise":
        # Generate in float32 for broad CPU/GPU dtype support, then cast to the
        # storage dtype.  The distribution, not a particular sample variance,
        # is N(0, 1), matching cisTEM's AddGaussianNoise(1.0f).
        out = torch.randn(
            (out_h, out_w),
            device=image.device,
            dtype=torch.float32,
            generator=generator,
        ).to(dtype=image.dtype)
    elif mode == "zero":
        out = torch.zeros((out_h, out_w), device=image.device, dtype=image.dtype)
    elif mode == "edge":
        mean = edge_mean(source)
        out = torch.full((out_h, out_w), float(mean.item()), device=image.device, dtype=image.dtype)
    else:
        raise ValueError(f"unknown padding mode: {mode}")

    out[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w] = source
    return out, valid


def pad_to_square(
    image: Tensor,
    size: int,
    *,
    mode: str,
    generator: Optional[torch.Generator] = None,
) -> tuple[Tensor, Tensor]:
    """cisTEM-style square clipping/padding plus a valid-pixel mask."""
    return center_pad_crop_with_mode(
        image,
        (int(size), int(size)),
        mode=mode,
        generator=generator,
    )


def _linear_bin_accumulate(values: Tensor, positions: Tensor, n_bins: int) -> tuple[Tensor, Tensor]:
    positions = positions.clamp(0.0, float(n_bins - 1))
    lo = torch.floor(positions).to(torch.long)
    hi = torch.clamp(lo + 1, max=n_bins - 1)
    frac = positions - lo.to(positions.dtype)
    w_lo = 1.0 - frac
    w_hi = frac
    sums = torch.zeros(n_bins, device=values.device, dtype=values.dtype)
    counts = torch.zeros(n_bins, device=values.device, dtype=values.dtype)
    sums.scatter_add_(0, lo, values * w_lo)
    counts.scatter_add_(0, lo, w_lo)
    different = hi != lo
    if bool(different.any()):
        sums.scatter_add_(0, hi[different], values[different] * w_hi[different])
        counts.scatter_add_(0, hi[different], w_hi[different])
    return sums, counts


def radial_power_curve_cistem(fourier: Tensor, real_shape: tuple[int, int]) -> RadialCurve:
    """Reproduce Image::Compute1DPowerSpectrumCurve for a 2-D rFFT image.

    The explicit Hermitian mates on x=0 in the negative-y half are excluded,
    matching Image::FourierComponentIsExplicitHermitianMate in the target source.
    """
    if fourier.ndim != 2 or not torch.is_complex(fourier):
        raise ValueError("fourier must be a 2-D complex rFFT image")
    h, w = (int(v) for v in real_shape)
    n_points = int((w / 2.0 + 1.0) * math.sqrt(2.0) + 1.0)
    x_axis = torch.linspace(0.0, 0.5 * math.sqrt(2.0), n_points, device=fourier.device, dtype=fourier.real.dtype)
    _, _, radius = reciprocal_frequencies_rfft2(h, w, device=fourier.device, dtype=fourier.real.dtype)
    power = fourier.real.square() + fourier.imag.square()

    mask = torch.ones_like(power, dtype=torch.bool)
    first_negative_y = h // 2 + 1 if h % 2 == 0 else (h + 1) // 2
    if first_negative_y < h:
        mask[first_negative_y:, 0] = False
    radius_values = radius[mask].reshape(-1)
    power_values = power[mask].reshape(-1)
    step = float(x_axis[-1].item()) / max(n_points - 1, 1)
    positions = radius_values / max(step, TINY)
    sums, counts = _linear_bin_accumulate(power_values, positions, n_points)
    average = torch.where(counts > 0, sums / counts.clamp_min(TINY), torch.zeros_like(sums))
    return RadialCurve(x=x_axis, y=average)


def make_whitening_curve(fourier: Tensor, real_shape: tuple[int, int]) -> RadialCurve:
    curve = radial_power_curve_cistem(fourier, real_shape)
    amplitude = torch.sqrt(curve.y.clamp_min(0.0))
    reciprocal = torch.where(amplitude > 0, amplitude.reciprocal(), torch.zeros_like(amplitude))
    maximum = reciprocal.max().clamp_min(TINY)
    return RadialCurve(curve.x, reciprocal / maximum)


def evaluate_radial_curve(curve: RadialCurve, radius: Tensor) -> Tensor:
    if curve.x.numel() < 2:
        return torch.full_like(radius, float(curve.y[0].item()))
    start = curve.x[0]
    end = curve.x[-1]
    step = (end - start) / (curve.x.numel() - 1)
    positions = ((radius - start) / step.clamp_min(TINY)).clamp(0, curve.x.numel() - 1)
    lo = torch.floor(positions).to(torch.long)
    hi = torch.clamp(lo + 1, max=curve.x.numel() - 1)
    frac = positions - lo.to(positions.dtype)
    return curve.y[lo] * (1.0 - frac) + curve.y[hi] * frac


def apply_radial_curve_rfft2(fourier: Tensor, real_shape: tuple[int, int], curve: RadialCurve) -> Tensor:
    h, w = real_shape
    _, _, radius = reciprocal_frequencies_rfft2(h, w, device=fourier.device, dtype=fourier.real.dtype)
    values = evaluate_radial_curve(curve.to(fourier.device, fourier.real.dtype), radius)
    return fourier * values


def preprocess_first_pass(image: Tensor) -> tuple[Tensor, RadialCurve, Tensor, Tensor]:
    """Outlier replacement, whitening and real-space unit-variance normalization."""
    cleaned, outlier_mask = replace_outliers_with_mean(image, 5.0)
    fourier = rfft2_cistem(cleaned, centered_real_space=False)
    fourier = fourier.clone()
    fourier[0, 0] = 0
    whitening = make_whitening_curve(fourier, tuple(cleaned.shape))
    fourier = apply_radial_curve_rfft2(fourier, tuple(cleaned.shape), whitening)
    fourier[0, 0] = 0
    energy = rfft_sum_of_squares(fourier, int(cleaned.shape[-1])).clamp_min(TINY)
    fourier = fourier / torch.sqrt(energy)
    whitened = torch.fft.irfft2(fourier, s=cleaned.shape, norm="forward")
    return whitened, whitening, cleaned, outlier_mask


def normalize_fourier_for_footprint(
    fourier: Tensor,
    *,
    last_real_size: int,
    normalization_pixels: int,
) -> Tensor:
    """Set DC to zero and normalize an rFFT image to the template footprint."""
    if not torch.is_complex(fourier):
        raise TypeError("fourier must be complex")
    result = fourier.clone()
    result[0, 0] = 0
    energy = rfft_sum_of_squares(result, int(last_real_size)).clamp_min(TINY)
    denominator = max(float(normalization_pixels), 1.0)
    return result / torch.sqrt(energy / denominator)


def apply_fourier_weight_and_normalize(
    fourier: Tensor,
    weight: Tensor,
    *,
    last_real_size: int,
    normalization_pixels: int,
) -> Tensor:
    """Apply a real/complex Fourier weight and restore footprint normalization."""
    if tuple(fourier.shape) != tuple(weight.shape):
        raise ValueError("fourier and weight shapes differ")
    weighted = fourier * weight.to(device=fourier.device, dtype=fourier.dtype)
    return normalize_fourier_for_footprint(
        weighted,
        last_real_size=last_real_size,
        normalization_pixels=normalization_pixels,
    )


def preprocess_second_pass(image: Tensor, normalization_pixels: int) -> Tensor:
    """Centered FFT and cisTEM template-footprint normalization; no second whitening."""
    fourier = rfft2_cistem(image, centered_real_space=True)
    return normalize_fourier_for_footprint(
        fourier,
        last_real_size=int(image.shape[-1]),
        normalization_pixels=normalization_pixels,
    )


def resize_real_fourier(image: Tensor, shape: tuple[int, int]) -> Tensor:
    return full_fft_resize_real(image, shape)


def local_mean_std(x: Tensor) -> tuple[Tensor, Tensor]:
    return x.mean(), x.std(unbiased=False)


def circular_mask(shape: tuple[int, int], radius: float, center: Optional[tuple[float, float]] = None) -> Tensor:
    h, w = shape
    cy, cx = center if center is not None else ((h - 1) / 2.0, (w - 1) / 2.0)
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    return (yy - cy).square() + (xx - cx).square() <= radius * radius


def nearest_resize(image: Tensor, shape: tuple[int, int]) -> Tensor:
    return F.interpolate(image[None, None], size=shape, mode="nearest")[0, 0]


def bilinear_resize(image: Tensor, shape: tuple[int, int]) -> Tensor:
    return F.interpolate(image[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]


def center_pad_crop_nd(
    value: Tensor,
    output_shape: tuple[int, ...],
    *,
    fill_value: float | Tensor = 0.0,
) -> Tensor:
    """Center clip/pad the final dimensions of a tensor.

    The integer placement follows cisTEM's ClipInto/Resize centre convention:
    the lower-side offset is floor((new-old)/2), leaving an odd extra pixel on
    the upper side for even source dimensions.
    """
    nd = len(output_shape)
    if nd < 1 or value.ndim < nd:
        raise ValueError("output_shape does not match tensor dimensionality")
    output_shape = tuple(int(v) for v in output_shape)
    if any(v <= 0 for v in output_shape):
        raise ValueError("output dimensions must be positive")
    scalar = float(fill_value.item()) if isinstance(fill_value, Tensor) else float(fill_value)
    out = torch.full((*value.shape[:-nd], *output_shape), scalar, device=value.device, dtype=value.dtype)
    src_slices: list[slice] = []
    dst_slices: list[slice] = []
    for old, new in zip(value.shape[-nd:], output_shape):
        amount = min(int(old), int(new))
        src_start = (int(old) - amount) // 2
        dst_start = (int(new) - amount) // 2
        src_slices.append(slice(src_start, src_start + amount))
        dst_slices.append(slice(dst_start, dst_start + amount))
    out[(..., *dst_slices)] = value[(..., *src_slices)]
    return out


def average_inside_centered_sphere(volume: Tensor, radius_pixels: float) -> Tensor:
    """Average real values inside a centred sphere, as used for cisTEM padding."""
    if volume.ndim != 3:
        raise ValueError("volume must be 3-D")
    z, y, x = volume.shape
    cz, cy, cx = (z - 1) / 2.0, (y - 1) / 2.0, (x - 1) / 2.0
    zz, yy, xx = torch.meshgrid(
        torch.arange(z, device=volume.device, dtype=torch.float32),
        torch.arange(y, device=volume.device, dtype=torch.float32),
        torch.arange(x, device=volume.device, dtype=torch.float32),
        indexing="ij",
    )
    mask = (zz - cz).square() + (yy - cy).square() + (xx - cx).square() <= float(radius_pixels) ** 2
    if not bool(mask.any()):
        return volume.mean()
    return volume[mask].mean()


def _match_parity(candidate: int, reference: int) -> int:
    candidate = max(int(candidate), 1)
    if candidate % 2 != reference % 2:
        candidate += 1
    return candidate


@dataclass(slots=True)
class PixelSizeChangeResult:
    volume: Tensor
    used_factor: float
    pad_shape: tuple[int, int, int]
    intermediate_shape: tuple[int, int, int]
    padding_value: float


def change_pixel_size_cistem(
    volume: Tensor,
    wanted_factor: float,
    *,
    tolerance: float = 0.001,
    max_intermediate_dimension: int = 1536,
) -> PixelSizeChangeResult:
    """PyTorch translation of Image::ChangePixelSize for a real 3-D template.

    The source pads in real space, Fourier-crops/pads to an intermediate shape,
    then clips back into the preallocated output volume.  The returned
    ``used_factor`` is the exact integer-ratio factor actually realized.
    """
    if volume.ndim != 3:
        raise ValueError("volume must be 3-D")
    if wanted_factor <= 0 or tolerance <= 0:
        raise ValueError("wanted_factor and tolerance must be positive")
    old = tuple(int(v) for v in volume.shape)
    if abs(float(wanted_factor) - 1.0) < float(tolerance):
        return PixelSizeChangeResult(volume.clone(), float(wanted_factor), old, old, 0.0)

    minimum = min(old)
    if tolerance < 1.0 / (minimum * wanted_factor):
        intermediate_factor = 1.0 / (tolerance * minimum)
        pad = tuple(_match_parity(int(v * intermediate_factor), v) for v in old)
        new = tuple(_match_parity(int(p / wanted_factor), o) for p, o in zip(pad, old))
    else:
        intermediate_factor = float(wanted_factor)
        pad = tuple(_match_parity(int(v * intermediate_factor), v) for v in old)
        new = old

    largest = max(*pad, *new)
    if largest > int(max_intermediate_dimension):
        raise MemoryError(
            "cisTEM-exact pixel-size resampling would create dimension "
            f"{largest}, exceeding max_pixel_resample_dimension={max_intermediate_dimension}; "
            "raise that limit or use runtime.pixel_size_backend='coordinate'"
        )

    padding_value_t = (
        average_inside_centered_sphere(volume, 0.45 * minimum)
        if any(p > o for p, o in zip(pad, old))
        else torch.zeros((), device=volume.device, dtype=volume.dtype)
    )
    padded = center_pad_crop_nd(volume, pad, fill_value=padding_value_t)
    resized = full_fft_resize_real(padded, new)
    output = center_pad_crop_nd(resized, old, fill_value=padding_value_t)
    return PixelSizeChangeResult(
        volume=output,
        used_factor=float(pad[-1]) / float(new[-1]),
        pad_shape=pad,
        intermediate_shape=new,
        padding_value=float(padding_value_t.item()),
    )
