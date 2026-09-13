from __future__ import annotations

from dataclasses import dataclass
import warnings

import torch

Tensor = torch.Tensor

try:  # Optional; Linux CUDA PyTorch wheels commonly bundle a matching Triton.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only build/test hosts
    triton = None
    tl = None


def triton_correlation_is_available() -> bool:
    return triton is not None and tl is not None


def complex32_dtype() -> torch.dtype:
    dtype = getattr(torch, "complex32", None)
    if dtype is None:
        dtype = getattr(torch, "chalf", None)
    if dtype is None:  # pragma: no cover - modern supported PyTorch exposes one alias
        raise RuntimeError("this PyTorch build does not expose torch.complex32/torch.chalf")
    return dtype


def complex_real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    if dtype == complex32_dtype():
        return torch.float16
    raise TypeError(f"unsupported complex dtype: {dtype}")


def is_power_of_two(value: int) -> bool:
    value = int(value)
    return value > 0 and (value & (value - 1)) == 0


def is_power_of_two_shape(shape: tuple[int, int]) -> bool:
    return all(is_power_of_two(v) for v in shape)


_HALF_FFT_PROBE: dict[str, tuple[bool, str | None]] = {}


def cuda_half_fft_supported(device: torch.device) -> tuple[bool, str | None]:
    """Probe the exact half-RFFT/chalf-IRFFT path once per CUDA device."""
    if device.type != "cuda":
        return False, "half correlation FFTs require CUDA"
    key = str(device)
    cached = _HALF_FFT_PROBE.get(key)
    if cached is not None:
        return cached
    try:
        # Probe the actual direct-s use case: a non-power-of-two small input is
        # zero-padded by the FFT call to power-of-two transform lengths.
        x = torch.linspace(-0.25, 0.25, 36, device=device, dtype=torch.float32).reshape(6, 6)
        x = (x - x.mean()).to(torch.float16)
        f = torch.fft.rfft2(x, s=(8, 8), norm="forward")
        y = torch.fft.irfft2(f, s=(8, 8), norm="forward")
        expected_complex = complex32_dtype()
        ok = (
            f.dtype == expected_complex
            and y.dtype == torch.float16
            and bool(torch.isfinite(torch.view_as_real(f)).all().item())
            and bool(torch.isfinite(y).all().item())
            and float(y.abs().sum().item()) > 0.0
        )
        reason = None if ok else (
            f"half FFT probe returned {f.dtype}/{y.dtype}, expected "
            f"{expected_complex}/torch.float16"
        )
    except Exception as exc:  # pragma: no cover - requires a CUDA runtime mismatch
        ok = False
        reason = f"{type(exc).__name__}: {exc}"
    _HALF_FFT_PROBE[key] = (ok, reason)
    return ok, reason


@dataclass(frozen=True, slots=True)
class CorrelationPrecisionPlan:
    requested: str
    effective: str
    real_dtype: torch.dtype
    complex_dtype: torch.dtype
    fallback_reason: str | None = None


def resolve_correlation_precision(
    requested: str,
    *,
    device: torch.device,
    fft_shape: tuple[int, int],
    half_fft_shape_mode: str,
    fallback_to_float32: bool,
) -> CorrelationPrecisionPlan:
    if requested == "float32":
        return CorrelationPrecisionPlan(
            requested=requested,
            effective="float32",
            real_dtype=torch.float32,
            complex_dtype=torch.complex64,
        )
    if requested != "mixed_float16":
        raise ValueError(f"unknown correlation precision: {requested}")

    def fallback(reason: str) -> CorrelationPrecisionPlan:
        if not fallback_to_float32:
            raise RuntimeError(reason)
        warnings.warn(
            f"mixed_float16 correlation unavailable ({reason}); falling back to float32",
            RuntimeWarning,
            stacklevel=3,
        )
        return CorrelationPrecisionPlan(
            requested=requested,
            effective="float32",
            real_dtype=torch.float32,
            complex_dtype=torch.complex64,
            fallback_reason=reason,
        )

    if device.type != "cuda":
        return fallback("the mixed correlation FFT path requires CUDA")
    if not is_power_of_two_shape(fft_shape):
        reason = (
            f"half RFFT/IRFFT requires power-of-two transformed dimensions, got {fft_shape}"
        )
        if half_fft_shape_mode == "require_power2":
            raise ValueError(reason)
        return fallback(reason)
    supported, reason = cuda_half_fft_supported(device)
    if not supported:
        return fallback(reason or "the CUDA half FFT probe failed")
    return CorrelationPrecisionPlan(
        requested=requested,
        effective="mixed_float16",
        real_dtype=torch.float16,
        complex_dtype=complex32_dtype(),
    )


if triton is not None and tl is not None:  # pragma: no branch

    @triton.jit
    def _prephase_cast_kernel(
        input_ri_ptr,
        phase_ri_ptr,
        output_ri_ptr,
        total_complex,
        HAS_PHASE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_complex
        ir = tl.load(input_ri_ptr + 2 * offsets, mask=mask, other=0.0).to(tl.float32)
        ii = tl.load(input_ri_ptr + 2 * offsets + 1, mask=mask, other=0.0).to(tl.float32)
        if HAS_PHASE:
            pr = tl.load(phase_ri_ptr + 2 * offsets, mask=mask, other=1.0).to(tl.float32)
            pi = tl.load(phase_ri_ptr + 2 * offsets + 1, mask=mask, other=0.0).to(tl.float32)
            # input * conjugate(phase)
            out_r = ir * pr + ii * pi
            out_i = ii * pr - ir * pi
        else:
            out_r = ir
            out_i = ii
        tl.store(output_ri_ptr + 2 * offsets, out_r, mask=mask)
        tl.store(output_ri_ptr + 2 * offsets + 1, out_i, mask=mask)


    @triton.jit
    def _conjugate_multiply_inplace_kernel(
        image_ri_ptr,
        projection_ri_ptr,
        inverse_scale_ptr,
        total_complex,
        plane_complex,
        HAS_SCALE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_complex
        plane_offsets = offsets % plane_complex

        ir = tl.load(image_ri_ptr + 2 * plane_offsets, mask=mask, other=0.0).to(tl.float32)
        ii = tl.load(image_ri_ptr + 2 * plane_offsets + 1, mask=mask, other=0.0).to(tl.float32)
        pr = tl.load(projection_ri_ptr + 2 * offsets, mask=mask, other=0.0).to(tl.float32)
        pi = tl.load(projection_ri_ptr + 2 * offsets + 1, mask=mask, other=0.0).to(tl.float32)

        # image * conjugate(projection)
        out_r = ir * pr + ii * pi
        out_i = ii * pr - ir * pi
        if HAS_SCALE:
            batch_offsets = offsets // plane_complex
            inv_scale = tl.load(inverse_scale_ptr + batch_offsets, mask=mask, other=1.0).to(tl.float32)
            out_r *= inv_scale
            out_i *= inv_scale
        tl.store(projection_ri_ptr + 2 * offsets, out_r, mask=mask)
        tl.store(projection_ri_ptr + 2 * offsets + 1, out_i, mask=mask)


def _torch_prephase_cast(
    image_fourier: Tensor,
    phase: Tensor | None,
    output_dtype: torch.dtype,
) -> Tensor:
    src = torch.view_as_real(image_fourier.resolve_conj()).to(torch.float32)
    ir, ii = src[..., 0], src[..., 1]
    if phase is not None:
        phase_ri = torch.view_as_real(phase.resolve_conj()).to(
            device=image_fourier.device, dtype=torch.float32
        )
        pr, pi = phase_ri[..., 0], phase_ri[..., 1]
        out_r = ir * pr + ii * pi
        out_i = ii * pr - ir * pi
    else:
        out_r, out_i = ir, ii
    real_dtype = complex_real_dtype(output_dtype)
    packed = torch.stack((out_r, out_i), dim=-1).to(real_dtype).contiguous()
    return torch.view_as_complex(packed)


def prepare_image_fourier(
    image_fourier: Tensor,
    *,
    phase: Tensor | None,
    output_dtype: torch.dtype,
    backend: str = "auto",
    block_size: int = 256,
    num_warps: int = 4,
    fallback_to_torch: bool = True,
) -> tuple[Tensor, str]:
    """Prephase the input once and optionally cast complex64 to complex32."""
    if not torch.is_complex(image_fourier):
        raise TypeError("image_fourier must be complex")
    if phase is not None and tuple(phase.shape) != tuple(image_fourier.shape):
        raise ValueError("phase and image Fourier shapes differ")
    if output_dtype not in {torch.complex64, complex32_dtype()}:
        raise TypeError(f"unsupported correlation Fourier dtype: {output_dtype}")
    if phase is None and image_fourier.dtype == output_dtype:
        # The strict float32/explicit path can reuse the normalized input FFT
        # directly and remains bitwise identical to v0.3.3.
        return image_fourier, "identity"

    wants_triton = backend in {"auto", "triton"}
    use_triton = (
        wants_triton
        and image_fourier.device.type == "cuda"
        and triton_correlation_is_available()
    )
    if backend == "triton" and not use_triton and not fallback_to_torch:
        raise RuntimeError("Triton correlation prephase requested but unavailable")
    if use_triton:
        try:
            image_fourier = image_fourier.contiguous()
            phase_tensor = (
                phase.to(device=image_fourier.device, dtype=torch.complex64).contiguous()
                if phase is not None
                else torch.ones((1,), device=image_fourier.device, dtype=torch.complex64)
            )
            output_real_dtype = complex_real_dtype(output_dtype)
            output_ri = torch.empty(
                (*image_fourier.shape, 2),
                device=image_fourier.device,
                dtype=output_real_dtype,
            )
            total = int(image_fourier.numel())
            grid = (triton.cdiv(total, int(block_size)),)
            _prephase_cast_kernel[grid](
                torch.view_as_real(image_fourier),
                torch.view_as_real(phase_tensor),
                output_ri,
                total,
                HAS_PHASE=phase is not None,
                BLOCK_SIZE=int(block_size),
                num_warps=int(num_warps),
            )
            return torch.view_as_complex(output_ri), "triton"
        except Exception as exc:  # pragma: no cover - CUDA/Triton runtime dependent
            if not fallback_to_torch:
                raise
            warnings.warn(
                f"Triton input prephase failed ({type(exc).__name__}: {exc}); using PyTorch",
                RuntimeWarning,
                stacklevel=2,
            )
    return _torch_prephase_cast(image_fourier, phase, output_dtype), "torch"


def _torch_conjugate_multiply_inplace(
    image_fourier: Tensor, projections: Tensor, inverse_scale: Tensor | None = None
) -> None:
    # Preserve the exact v0.3.3 operation order for the strict complex64 path.
    if image_fourier.dtype == torch.complex64 and projections.dtype == torch.complex64:
        projections.conj_physical_()
        torch.mul(image_fourier.unsqueeze(0), projections, out=projections)
        if inverse_scale is not None:
            projections.mul_(inverse_scale.to(projections.real.dtype)[:, None, None])
        return

    image_ri = torch.view_as_real(image_fourier.resolve_conj())
    projection_ri = torch.view_as_real(projections.resolve_conj())
    # Compute before either output component is overwritten.  Float32 arithmetic
    # is intentional even when the destination is complex32/chalf.
    ir = image_ri[..., 0].to(torch.float32)
    ii = image_ri[..., 1].to(torch.float32)
    pr = projection_ri[..., 0].to(torch.float32)
    pi = projection_ri[..., 1].to(torch.float32)
    out_r = ir.unsqueeze(0) * pr + ii.unsqueeze(0) * pi
    out_i = ii.unsqueeze(0) * pr - ir.unsqueeze(0) * pi
    if inverse_scale is not None:
        scale = inverse_scale.to(device=projections.device, dtype=torch.float32)[:, None, None]
        out_r = out_r * scale
        out_i = out_i * scale
    projection_ri[..., 0].copy_(out_r.to(projection_ri.dtype))
    projection_ri[..., 1].copy_(out_i.to(projection_ri.dtype))


def conjugate_multiply_inplace(
    image_fourier: Tensor,
    projections: Tensor,
    *,
    backend: str = "auto",
    block_size: int = 256,
    num_warps: int = 4,
    fallback_to_torch: bool = True,
    inverse_scale: Tensor | None = None,
) -> str:
    """Replace ``projections`` with ``image * conjugate(projections)``."""
    if image_fourier.ndim != 2 or projections.ndim != 3:
        raise ValueError("expected image (H,Q) and projection batch (B,H,Q)")
    if tuple(projections.shape[-2:]) != tuple(image_fourier.shape):
        raise ValueError("image/projection Fourier shapes differ")
    if image_fourier.dtype != projections.dtype:
        raise TypeError(
            f"image/projection Fourier dtypes differ: {image_fourier.dtype} vs {projections.dtype}"
        )
    if image_fourier.dtype not in {torch.complex64, complex32_dtype()}:
        raise TypeError(f"unsupported correlation Fourier dtype: {image_fourier.dtype}")
    if inverse_scale is not None:
        if inverse_scale.ndim != 1 or int(inverse_scale.numel()) != int(projections.shape[0]):
            raise ValueError("inverse_scale must contain one value per projection")
        inverse_scale = inverse_scale.to(device=projections.device, dtype=torch.float32).contiguous()

    wants_triton = backend in {"auto", "triton"}
    use_triton = (
        wants_triton
        and image_fourier.device.type == "cuda"
        and triton_correlation_is_available()
    )
    if backend == "triton" and not use_triton and not fallback_to_torch:
        raise RuntimeError("Triton correlation multiply requested but unavailable")
    if use_triton:
        try:
            image_fourier = image_fourier.contiguous()
            if not projections.is_contiguous():
                raise ValueError("projection Fourier batch must be contiguous for Triton")
            total = int(projections.numel())
            plane = int(image_fourier.numel())
            grid = (triton.cdiv(total, int(block_size)),)
            _conjugate_multiply_inplace_kernel[grid](
                torch.view_as_real(image_fourier),
                torch.view_as_real(projections),
                inverse_scale if inverse_scale is not None else torch.ones((1,), device=projections.device, dtype=torch.float32),
                total,
                plane,
                HAS_SCALE=inverse_scale is not None,
                BLOCK_SIZE=int(block_size),
                num_warps=int(num_warps),
            )
            return "triton"
        except Exception as exc:  # pragma: no cover - CUDA/Triton runtime dependent
            if not fallback_to_torch:
                raise
            warnings.warn(
                f"Triton conjugate multiply failed ({type(exc).__name__}: {exc}); using PyTorch",
                RuntimeWarning,
                stacklevel=2,
            )
    _torch_conjugate_multiply_inplace(image_fourier, projections, inverse_scale)
    return "torch"


def tensor_health(value: Tensor, *, require_nonzero: bool = True) -> dict[str, float | int | bool]:
    """Synchronizing numerical-health summary intended only for initial mixed batches."""
    real = torch.view_as_real(value.resolve_conj()) if torch.is_complex(value) else value
    finite = torch.isfinite(real)
    finite_count = int(finite.sum().item())
    total = int(real.numel())
    safe = torch.where(finite, real.to(torch.float32), torch.zeros((), device=real.device))
    absolute_sum = float(safe.abs().sum().item())
    maximum = float(safe.abs().amax().item()) if total else 0.0
    healthy = finite_count == total and (absolute_sum > 0.0 or not require_nonzero)
    return {
        "healthy": healthy,
        "finite_count": finite_count,
        "total_count": total,
        "nonfinite_count": total - finite_count,
        "absolute_sum": absolute_sum,
        "maximum_absolute": maximum,
    }
