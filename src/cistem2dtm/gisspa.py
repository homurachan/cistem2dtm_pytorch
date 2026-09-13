from __future__ import annotations

"""Repository-compatible GisSPA frequency weighting.

The implementation follows homurachan/GisSPA ``GPU_func.cu`` while retaining
the cisTEM/PyTorch 3-D projector and norm-type-1 MIP accumulator.  GisSPA's
real image-side factor is evaluated on the full search Fourier grid, and its
signed template-side CTF factor is evaluated on the projection grid.  Keeping
the two native grids separate is important for rectangular micrographs and for
the repository's fixed-width cosine shoulders.
"""

from dataclasses import dataclass
import math

import torch

from .constants import TINY
from .fft import reciprocal_frequencies_rfft2

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class ProjectionRadialBins:
    bins: Tensor
    valid_indices: Tensor
    valid_bins: Tensor
    counts: Tensor
    number_of_bins: int


def repository_radial_bins(
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> ProjectionRadialBins:
    """Return GisSPA's nearest-radius-minus-one bins for an rFFT half plane."""
    n = int(size)
    q = n // 2 + 1
    y = torch.arange(n, device=device, dtype=dtype)
    x = torch.arange(q, device=device, dtype=dtype)
    dy = torch.minimum(y, torch.tensor(float(n), device=device, dtype=dtype) - y)
    yy, xx = torch.meshgrid(dy, x, indexing="ij")
    bins = torch.floor(torch.sqrt(xx.square() + yy.square()) + 0.5).to(torch.long) - 1
    number_of_bins = max(n // 2, 1)
    valid = (bins >= 0) & (bins < number_of_bins)
    valid_indices = torch.nonzero(valid.reshape(-1), as_tuple=False).reshape(-1)
    valid_bins = bins.reshape(-1).index_select(0, valid_indices)
    counts = torch.bincount(valid_bins, minlength=number_of_bins).to(dtype)
    return ProjectionRadialBins(
        bins=bins,
        valid_indices=valid_indices,
        valid_bins=valid_bins,
        counts=counts,
        number_of_bins=number_of_bins,
    )


def whiten_projections_repository(
    slices: Tensor,
    radial: ProjectionRadialBins,
    *,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    """Whiten every Fourier projection by its own GisSPA radial power curve.

    ``slices`` is an rFFT half plane with shape ``(B, N, N//2+1)``.  GisSPA's
    source bins only the stored x>=0 half and therefore does not use Hermitian
    multiplicity weights here.
    """
    if slices.ndim != 3 or not torch.is_complex(slices):
        raise ValueError("slices must have shape (batch, N, N//2+1) and be complex")
    batch = int(slices.shape[0])
    flat_power = (slices.real.square() + slices.imag.square()).reshape(batch, -1)
    values = flat_power.index_select(1, radial.valid_indices)
    sums = torch.zeros(
        (batch, radial.number_of_bins),
        device=slices.device,
        dtype=torch.float32,
    )
    bin_index = radial.valid_bins.unsqueeze(0).expand(batch, -1)
    sums.scatter_add_(1, bin_index, values.to(torch.float32))
    counts = radial.counts.to(device=slices.device, dtype=torch.float32).clamp_min(1.0)
    mean_power = sums / counts.unsqueeze(0)
    inverse_amplitude = torch.rsqrt(mean_power.clamp_min(float(epsilon)))

    weights_flat = torch.zeros(
        (batch, int(slices.shape[-2] * slices.shape[-1])),
        device=slices.device,
        dtype=slices.real.dtype,
    )
    gathered = inverse_amplitude.gather(1, bin_index).to(weights_flat.dtype)
    weights_flat.index_copy_(1, radial.valid_indices, gathered)
    weights = weights_flat.reshape(batch, slices.shape[-2], slices.shape[-1])
    whitened = slices * weights
    # The source excludes r=-1 and the matching code subsequently removes DC.
    whitened[:, 0, 0] = 0
    return whitened, mean_power


def repository_cosine_bandpass(
    shape: tuple[int, int],
    *,
    pixel_size_angstrom: float,
    low_resolution_angstrom: float,
    high_resolution_angstrom: float,
    edge_width_pixels: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """GisSPA image-side bandpass, including its cosine shoulders."""
    h, w = (int(v) for v in shape)
    q = w // 2 + 1
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(q, device=device, dtype=dtype)
    dy = torch.minimum(y, torch.tensor(float(h), device=device, dtype=dtype) - y)
    yy, xx = torch.meshgrid(dy, x, indexing="ij")
    radius_pixels = torch.sqrt(xx.square() + yy.square())
    r_round = torch.floor(radius_pixels + 0.5) - 1.0
    l = float(max(h, w))
    high_cut = l * float(pixel_size_angstrom) / float(high_resolution_angstrom)
    low_cut = l * float(pixel_size_angstrom) / float(low_resolution_angstrom)
    edge = max(float(edge_width_pixels), 1.0e-6)

    passband = (r_round < high_cut) & (r_round >= low_cut)
    high_shoulder = (r_round >= high_cut) & (r_round < high_cut + edge)
    low_shoulder = (r_round >= low_cut - edge) & (r_round < low_cut) & (r_round >= 0)

    out = torch.zeros_like(radius_pixels)
    out = torch.where(passband, torch.ones_like(out), out)
    high_value = 0.5 * torch.cos(math.pi * (r_round - high_cut) / (2.0 * edge)) + 0.5
    low_value = 0.5 * torch.cos(math.pi * (low_cut - r_round) / (2.0 * edge)) + 0.5
    out = torch.where(high_shoulder, high_value, out)
    out = torch.where(low_shoulder, low_value, out)
    return out


def repository_template_bandpass(
    shape: tuple[int, int],
    *,
    pixel_size_angstrom: float,
    high_resolution_angstrom: float,
    edge_width_pixels: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Exact template-side repository cutoff.

    ``GPU_func.cu::apply_weighting_function`` accepts every non-negative
    radius below the high-resolution cutoff.  Its low-resolution shoulder is
    therefore not reached by the source branch order; only the image-side
    filter supplies the low-frequency cutoff.
    """
    h, w = (int(v) for v in shape)
    q = w // 2 + 1
    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(q, device=device, dtype=dtype)
    dy = torch.minimum(y, torch.tensor(float(h), device=device, dtype=dtype) - y)
    yy, xx = torch.meshgrid(dy, x, indexing="ij")
    radius_pixels = torch.sqrt(xx.square() + yy.square())
    r_round = torch.floor(radius_pixels + 0.5) - 1.0
    l = float(max(h, w))
    high_cut = l * float(pixel_size_angstrom) / float(high_resolution_angstrom)
    edge = max(float(edge_width_pixels), 1.0e-6)

    passband = (r_round < high_cut) & (r_round >= 0)
    high_shoulder = (r_round >= high_cut) & (r_round < high_cut + edge)
    out = torch.zeros_like(radius_pixels)
    out = torch.where(passband, torch.ones_like(out), out)
    high_value = 0.5 * torch.cos(
        math.pi * (r_round - high_cut) / (2.0 * edge)
    ) + 0.5
    out = torch.where(high_shoulder, high_value, out)
    return out


def _repository_noise_to_signal_ratio(
    spatial_frequency: Tensor,
    *,
    a: float,
    b: float,
    b2: float,
    bfactor: float,
    bfactor2: float,
    bfactor3: float,
) -> Tensor:
    """Return the repository model N(s)/S(s) without forming two exponentials."""
    s2 = spatial_frequency.square()
    log_ratio = (
        (float(a) - float(bfactor)) * s2
        + (float(b) - float(bfactor2)) * spatial_frequency
        + (float(b2) - float(bfactor3))
    )
    return torch.exp(log_ratio.clamp(-80.0, 80.0))


def _repository_spatial_frequency(
    shape: tuple[int, int],
    *,
    pixel_size_angstrom: float,
    device: torch.device,
    dtype: torch.dtype,
    normalized_frequency_scale: float,
) -> Tensor:
    h, w = (int(v) for v in shape)
    _, _, normalized_radius = reciprocal_frequencies_rfft2(
        h, w, device=device, dtype=dtype
    )
    return (
        normalized_radius
        * float(normalized_frequency_scale)
        / float(pixel_size_angstrom)
    )


def repository_image_frequency_weight(
    ctf_signed: Tensor,
    shape: tuple[int, int],
    *,
    pixel_size_angstrom: float,
    low_resolution_angstrom: float,
    high_resolution_angstrom: float,
    kk: float,
    a: float,
    b: float,
    b2: float,
    bfactor: float,
    bfactor2: float,
    bfactor3: float,
    cosine_edge_width_pixels: float,
    image_high_frequency_damping: bool,
    epsilon: float,
    normalized_frequency_scale: float = 1.0,
) -> Tensor:
    """Real, non-negative GisSPA image-side frequency weight.

    After image whitening the repository applies

        sqrt(|CTF|^2 + N/S) / sqrt((kk+1) N/S + kk |CTF|^2),

    followed by the image-side cosine bandpass and optional >1/6 A^-1
    damping.  No phase flip is applied here; the CTF sign is placed on the
    projection-side factor instead.
    """
    if kk < 0:
        raise ValueError("GisSPA kk must be non-negative")
    spatial_frequency = _repository_spatial_frequency(
        shape,
        pixel_size_angstrom=pixel_size_angstrom,
        device=ctf_signed.device,
        dtype=ctf_signed.dtype,
        normalized_frequency_scale=normalized_frequency_scale,
    )
    ratio = _repository_noise_to_signal_ratio(
        spatial_frequency,
        a=a, b=b, b2=b2,
        bfactor=bfactor, bfactor2=bfactor2, bfactor3=bfactor3,
    )
    ctf2 = ctf_signed.square()
    denominator = ((float(kk) + 1.0) * ratio + float(kk) * ctf2).clamp_min(
        float(epsilon)
    )
    weight = torch.sqrt((ctf2 + ratio).clamp_min(0.0) / denominator)

    scale = float(normalized_frequency_scale)
    effective_pixel_size = float(pixel_size_angstrom) / max(scale, 1.0e-12)
    weight = weight * repository_cosine_bandpass(
        shape,
        pixel_size_angstrom=effective_pixel_size,
        low_resolution_angstrom=low_resolution_angstrom,
        high_resolution_angstrom=high_resolution_angstrom,
        edge_width_pixels=cosine_edge_width_pixels,
        device=ctf_signed.device,
        dtype=ctf_signed.dtype,
    )
    if image_high_frequency_damping:
        s2 = spatial_frequency.square()
        weight = weight * torch.where(
            spatial_frequency > (1.0 / 6.0),
            torch.exp(-100.0 * s2),
            torch.ones_like(spatial_frequency),
        )
    weight = torch.where(torch.isfinite(weight), weight, torch.zeros_like(weight))
    weight = weight.clone()
    weight[0, 0] = 0.0
    return weight


def repository_projection_frequency_weight(
    ctf_signed: Tensor,
    shape: tuple[int, int],
    *,
    pixel_size_angstrom: float,
    high_resolution_angstrom: float,
    kk: float,
    a: float,
    b: float,
    b2: float,
    bfactor: float,
    bfactor2: float,
    bfactor3: float,
    cosine_edge_width_pixels: float,
    epsilon: float,
    normalized_frequency_scale: float = 1.0,
) -> Tensor:
    """Signed GisSPA projection-side CTF weight after per-projection whitening.

    The repository uses |CTF|/sqrt(D) on phase-flipped images.  Applying the
    signed CTF here and leaving the image unflipped is Fourier-correlation
    equivalent, with D=(kk+1)N/S+kk|CTF|^2.
    """
    if kk < 0:
        raise ValueError("GisSPA kk must be non-negative")
    spatial_frequency = _repository_spatial_frequency(
        shape,
        pixel_size_angstrom=pixel_size_angstrom,
        device=ctf_signed.device,
        dtype=ctf_signed.dtype,
        normalized_frequency_scale=normalized_frequency_scale,
    )
    ratio = _repository_noise_to_signal_ratio(
        spatial_frequency,
        a=a, b=b, b2=b2,
        bfactor=bfactor, bfactor2=bfactor2, bfactor3=bfactor3,
    )
    ctf2 = ctf_signed.square()
    denominator = ((float(kk) + 1.0) * ratio + float(kk) * ctf2).clamp_min(
        float(epsilon)
    )
    weight = ctf_signed / torch.sqrt(denominator)

    scale = float(normalized_frequency_scale)
    effective_pixel_size = float(pixel_size_angstrom) / max(scale, 1.0e-12)
    weight = weight * repository_template_bandpass(
        shape,
        pixel_size_angstrom=effective_pixel_size,
        high_resolution_angstrom=high_resolution_angstrom,
        edge_width_pixels=cosine_edge_width_pixels,
        device=ctf_signed.device,
        dtype=ctf_signed.dtype,
    )
    weight = torch.where(torch.isfinite(weight), weight, torch.zeros_like(weight))
    weight = weight.clone()
    weight[0, 0] = 0.0
    return weight


def repository_combined_projection_weight(
    ctf_signed: Tensor,
    shape: tuple[int, int],
    *,
    pixel_size_angstrom: float,
    low_resolution_angstrom: float,
    high_resolution_angstrom: float,
    kk: float,
    a: float,
    b: float,
    b2: float,
    bfactor: float,
    bfactor2: float,
    bfactor3: float,
    cosine_edge_width_pixels: float,
    image_high_frequency_damping: bool,
    epsilon: float,
    normalized_frequency_scale: float = 1.0,
) -> Tensor:
    """Product of the repository image and projection factors on one grid.

    This helper is useful for algebraic regression tests.  Production matching
    evaluates the two factors separately on their native Fourier grids.
    """
    image = repository_image_frequency_weight(
        ctf_signed, shape,
        pixel_size_angstrom=pixel_size_angstrom,
        low_resolution_angstrom=low_resolution_angstrom,
        high_resolution_angstrom=high_resolution_angstrom,
        kk=kk, a=a, b=b, b2=b2, bfactor=bfactor,
        bfactor2=bfactor2, bfactor3=bfactor3,
        cosine_edge_width_pixels=cosine_edge_width_pixels,
        image_high_frequency_damping=image_high_frequency_damping,
        epsilon=epsilon,
        normalized_frequency_scale=normalized_frequency_scale,
    )
    projection = repository_projection_frequency_weight(
        ctf_signed, shape,
        pixel_size_angstrom=pixel_size_angstrom,
        high_resolution_angstrom=high_resolution_angstrom,
        kk=kk, a=a, b=b, b2=b2, bfactor=bfactor,
        bfactor2=bfactor2, bfactor3=bfactor3,
        cosine_edge_width_pixels=cosine_edge_width_pixels,
        epsilon=epsilon,
        normalized_frequency_scale=normalized_frequency_scale,
    )
    return image * projection
