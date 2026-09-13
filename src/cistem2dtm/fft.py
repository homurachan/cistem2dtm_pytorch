from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .constants import TINY

Tensor = torch.Tensor


def real_dtype_from_name(name: str) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    if real_dtype == torch.float64:
        return torch.complex128
    return torch.complex64


def fft_work_dtype(x: Tensor, shape: Sequence[int] | None = None) -> torch.dtype:
    """Return a PyTorch FFT-safe real dtype.

    CUDA supports half FFT only for restricted power-of-two dimensions.  CPU FFT
    does not support half/bfloat16.  The package accepts half storage/compute but
    promotes the FFT itself to float32 whenever PyTorch cannot execute it safely.
    """
    if x.dtype not in (torch.float16, torch.bfloat16):
        return x.dtype
    if x.device.type != "cuda":
        return torch.float32
    dims = tuple(int(v) for v in (shape or x.shape))
    if x.dtype == torch.bfloat16:
        return torch.float32
    if any(v <= 0 or (v & (v - 1)) != 0 for v in dims):
        return torch.float32
    return x.dtype


def rfft2_cistem(x: Tensor, *, centered_real_space: bool = False) -> Tensor:
    """cisTEM-style forward real FFT: forward transform carries 1/N scaling."""
    if x.ndim < 2:
        raise ValueError("rfft2_cistem expects at least two dimensions")
    work = torch.fft.ifftshift(x, dim=(-2, -1)) if centered_real_space else x
    dtype = fft_work_dtype(work, work.shape[-2:])
    return torch.fft.rfft2(work.to(dtype), dim=(-2, -1), norm="forward")


def centered_embed_phase_rfft2_phase(
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
) -> Tensor:
    """Return the phase that converts high-side padding into centre embedding."""
    out_h, out_w = (int(v) for v in output_shape)
    in_h, in_w = (int(v) for v in input_shape)
    use_h, use_w = min(in_h, out_h), min(in_w, out_w)
    dst_y = (out_h - use_h) // 2
    dst_x = (out_w - use_w) // 2
    fy = torch.fft.fftfreq(out_h, d=1.0, device=device, dtype=torch.float64)
    fx = torch.fft.rfftfreq(out_w, d=1.0, device=device, dtype=torch.float64)
    angle = -2.0 * math.pi * (fy[:, None] * dst_y + fx[None, :] * dst_x)
    # Build in complex128 and cast once.  The common prephase-input path keeps
    # this cache in complex64 even when the per-batch FFT uses complex32.
    return torch.polar(torch.ones_like(angle), angle).to(dtype)


def centered_embed_phase_rfft2_cistem(
    x: Tensor,
    output_shape: tuple[int, int],
    *,
    phase: Tensor | None = None,
    apply_phase: bool = True,
) -> tuple[Tensor, Tensor]:
    """FFT a centered small real image directly into a larger FFT shape.

    ``torch.fft.rfft2(..., s=output_shape)`` pads on the high-index side, while
    cisTEM/PyTorch 2DTM center-embeds the projection.  ``phase`` represents that
    translation.  With ``apply_phase=False`` the unshifted spectrum is returned
    so the conjugate phase can instead be multiplied into the input Fourier image
    once outside the orientation loop.
    """
    if x.ndim < 2:
        raise ValueError("centered_embed_phase_rfft2_cistem expects at least two dimensions")
    out_h, out_w = (int(v) for v in output_shape)
    in_h, in_w = (int(v) for v in x.shape[-2:])
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output_shape must be positive")

    work = x
    use_h, use_w = min(in_h, out_h), min(in_w, out_w)
    if (use_h, use_w) != (in_h, in_w):
        src_y = (in_h - use_h) // 2
        src_x = (in_w - use_w) // 2
        work = x[..., src_y : src_y + use_h, src_x : src_x + use_w]

    dtype = fft_work_dtype(work, output_shape)
    spectrum = torch.fft.rfft2(
        work.to(dtype),
        s=output_shape,
        dim=(-2, -1),
        norm="forward",
    )
    wanted_phase_shape = (out_h, out_w // 2 + 1)
    if phase is None:
        phase = centered_embed_phase_rfft2_phase(
            (in_h, in_w),
            output_shape,
            device=x.device,
            dtype=torch.complex64,
        )
    elif tuple(phase.shape) != wanted_phase_shape:
        raise ValueError("cached phase has the wrong shape")
    if apply_phase:
        spectrum.mul_(phase.to(device=spectrum.device, dtype=spectrum.dtype))
    return spectrum, phase


def irfft2_cistem(f: Tensor, shape: tuple[int, int], *, centered_real_space: bool = False) -> Tensor:
    """cisTEM-style inverse FFT: inverse is unscaled relative to forward-normalized data."""
    out = torch.fft.irfft2(f, s=shape, dim=(-2, -1), norm="forward")
    return torch.fft.fftshift(out, dim=(-2, -1)) if centered_real_space else out


def rfftn_cistem(x: Tensor, *, centered_real_space: bool = False) -> Tensor:
    if x.ndim < 3:
        raise ValueError("rfftn_cistem expects at least three dimensions")
    dims = (-3, -2, -1)
    work = torch.fft.ifftshift(x, dim=dims) if centered_real_space else x
    dtype = fft_work_dtype(work, work.shape[-3:])
    return torch.fft.rfftn(work.to(dtype), dim=dims, norm="forward")


def irfftn_cistem(f: Tensor, shape: tuple[int, int, int], *, centered_real_space: bool = False) -> Tensor:
    dims = (-3, -2, -1)
    out = torch.fft.irfftn(f, s=shape, dim=dims, norm="forward")
    return torch.fft.fftshift(out, dim=dims) if centered_real_space else out


def hermitian_weights(last_real_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    n_half = last_real_size // 2 + 1
    weights = torch.full((n_half,), 2.0, device=device, dtype=dtype)
    weights[0] = 1.0
    if last_real_size % 2 == 0:
        weights[-1] = 1.0
    return weights


def rfft_sum_of_squares(f: Tensor, last_real_size: int) -> Tensor:
    """Full Fourier-space sum |F|^2 from a last-axis rFFT half spectrum."""
    if not torch.is_complex(f):
        raise TypeError("rfft_sum_of_squares expects a complex tensor")
    value_dtype = f.real.dtype
    w = hermitian_weights(last_real_size, device=f.device, dtype=value_dtype)
    view = [1] * f.ndim
    view[-1] = w.numel()
    return ((f.real.square() + f.imag.square()) * w.view(view)).sum()


def normalize_fourier_energy(f: Tensor, last_real_size: int, denominator: float = 1.0) -> Tensor:
    energy = rfft_sum_of_squares(f, last_real_size)
    scale = torch.sqrt((energy / float(denominator)).clamp_min(TINY))
    return f / scale


def full_fft_resize_real(x: Tensor, output_shape: tuple[int, ...]) -> Tensor:
    """Fourier crop/pad a real tensor with cisTEM's forward-normalized convention.

    This helper intentionally uses a full complex spectrum for the resize step;
    the matching/projector path remains half-Hermitian rFFT.  Full-spectrum
    center crop/pad avoids ambiguity at odd/even Nyquist boundaries and keeps all
    operations in PyTorch.
    """
    nd = len(output_shape)
    if nd not in (2, 3) or x.ndim < nd:
        raise ValueError("output_shape must describe the final 2 or 3 dimensions")
    input_shape = tuple(int(v) for v in x.shape[-nd:])
    if input_shape == tuple(output_shape):
        return x.clone()
    dims = tuple(range(x.ndim - nd, x.ndim))
    dtype = fft_work_dtype(x, input_shape)
    spectrum = torch.fft.fftn(x.to(dtype), dim=dims, norm="forward")
    spectrum = torch.fft.fftshift(spectrum, dim=dims)

    target = torch.zeros((*x.shape[:-nd], *output_shape), dtype=spectrum.dtype, device=x.device)
    src_slices: list[slice] = []
    dst_slices: list[slice] = []
    for old, new in zip(input_shape, output_shape):
        amount = min(old, new)
        old_start = (old - amount) // 2
        new_start = (new - amount) // 2
        src_slices.append(slice(old_start, old_start + amount))
        dst_slices.append(slice(new_start, new_start + amount))
    target[(..., *dst_slices)] = spectrum[(..., *src_slices)]
    target = torch.fft.ifftshift(target, dim=dims)
    out = torch.fft.ifftn(target, s=output_shape, dim=dims, norm="forward").real
    return out.to(x.dtype if x.dtype in (torch.float32, torch.float64) else out.dtype)


def signed_frequency_indices(n: int, *, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
    return torch.fft.fftfreq(n, d=1.0, device=device, dtype=dtype) * n


def reciprocal_frequencies_rfft2(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor]:
    fy = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype)
    fx = torch.fft.rfftfreq(width, d=1.0, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    return xx, yy, radius


def reciprocal_frequencies_rfftn(
    depth: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    fz = torch.fft.fftfreq(depth, d=1.0, device=device, dtype=dtype)
    fy = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype)
    fx = torch.fft.rfftfreq(width, d=1.0, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(fz, fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square() + zz.square())
    return xx, yy, zz, radius


def next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value - 1).bit_length())


def is_factorable(value: int, primes: Sequence[int]) -> bool:
    if value < 1:
        return False
    n = int(value)
    for p in sorted(set(int(v) for v in primes if v > 1)):
        while n % p == 0:
            n //= p
    return n == 1


def closest_factorized_upper(value: int, primes: Sequence[int], max_fraction: float = 0.1) -> int:
    value = max(1, int(value))
    upper = max(value, int(math.ceil(value * (1.0 + max_fraction))))
    for candidate in range(value, upper + 1):
        if is_factorable(candidate, primes):
            return candidate
    candidate = upper + 1
    while not is_factorable(candidate, primes):
        candidate += 1
    return candidate
