from __future__ import annotations

import warnings

import torch

Tensor = torch.Tensor

try:  # Optional: CUDA PyTorch wheels on Linux commonly install matching Triton.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only build/test hosts
    triton = None
    tl = None


def triton_is_available() -> bool:
    return triton is not None and tl is not None


if triton is not None and tl is not None:  # pragma: no branch

    @triton.jit
    def _half_hermitian_trilinear_kernel(
        volume_ri_ptr,
        coordinates_ptr,
        output_ri_ptr,
        total_elements,
        N: tl.constexpr,
        Q: tl.constexpr,
        PLANE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        batch_index = offsets // PLANE
        plane_offset = offsets - batch_index * PLANE
        coordinate_base = batch_index * (3 * PLANE) + plane_offset
        x = tl.load(coordinates_ptr + coordinate_base, mask=mask, other=0.0)
        y = tl.load(coordinates_ptr + coordinate_base + PLANE, mask=mask, other=0.0)
        z = tl.load(coordinates_ptr + coordinate_base + 2 * PLANE, mask=mask, other=0.0)

        negative_x = x < 0.0
        i0 = tl.floor(x).to(tl.int32)
        j0 = tl.floor(y).to(tl.int32)
        k0 = tl.floor(z).to(tl.int32)
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        lower = -(N // 2)
        upper = (N - 1) // 2
        upper_x = N // 2
        lower_x = -upper_x
        valid_positive_x = (~negative_x) & (i1 <= upper_x)
        valid_negative_x = negative_x & (i0 >= lower_x)
        valid = (
            (valid_positive_x | valid_negative_x)
            & (j0 >= lower)
            & (j1 <= upper)
            & (k0 >= lower)
            & (k1 <= upper)
        )
        load_mask = mask & valid

        result_real = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        result_imag = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for dz in range(2):
            kk = k0 + dz
            wz = 1.0 - tl.abs(z - kk.to(tl.float32))
            for dy in range(2):
                jj = j0 + dy
                wy = 1.0 - tl.abs(y - jj.to(tl.float32))
                for dx in range(2):
                    ii = i0 + dx
                    wx = 1.0 - tl.abs(x - ii.to(tl.float32))
                    weight = wx * wy * wz

                    x_index = tl.where(negative_x, -ii, ii)
                    x_index = tl.maximum(0, tl.minimum(Q - 1, x_index))
                    y_logical = tl.where(negative_x, -jj, jj)
                    z_logical = tl.where(negative_x, -kk, kk)
                    y_index = tl.where(y_logical < 0, y_logical + N, y_logical)
                    z_index = tl.where(z_logical < 0, z_logical + N, z_logical)
                    address = (z_index * N + y_index) * Q + x_index
                    real_value = tl.load(volume_ri_ptr + 2 * address, mask=load_mask, other=0.0)
                    imag_value = tl.load(volume_ri_ptr + 2 * address + 1, mask=load_mask, other=0.0)
                    imag_value = tl.where(negative_x, -imag_value, imag_value)
                    result_real += real_value * weight
                    result_imag += imag_value * weight

        tl.store(output_ri_ptr + 2 * offsets, result_real, mask=mask)
        tl.store(output_ri_ptr + 2 * offsets + 1, result_imag, mask=mask)


def triton_half_hermitian_trilinear(
    volume_half: Tensor,
    coordinates: Tensor,
    *,
    block_size: int = 256,
    num_warps: int = 4,
) -> Tensor:
    """Fused eight-neighbour interpolation of a complex rFFT half volume."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed")
    if volume_half.device.type != "cuda" or coordinates.device.type != "cuda":
        raise RuntimeError("the Triton projector requires CUDA tensors")
    if volume_half.dtype != torch.complex64:
        raise TypeError("the Triton projector currently requires complex64 volume data")
    if coordinates.dtype != torch.float32:
        coordinates = coordinates.to(torch.float32)
    if coordinates.ndim != 4 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (batch, 3, height, half_width)")
    coordinates = coordinates.contiguous()
    volume_half = volume_half.contiguous()

    n = int(volume_half.shape[0])
    q = int(volume_half.shape[-1])
    if tuple(volume_half.shape) != (n, n, n // 2 + 1):
        raise ValueError("volume_half must have cubic rFFT shape (N, N, N//2+1)")
    batch, _, height, half_width = (int(v) for v in coordinates.shape)
    plane = height * half_width
    total = batch * plane
    output = torch.empty((batch, height, half_width), device=coordinates.device, dtype=torch.complex64)
    grid = (triton.cdiv(total, int(block_size)),)
    _half_hermitian_trilinear_kernel[grid](
        torch.view_as_real(volume_half),
        coordinates,
        torch.view_as_real(output),
        total,
        N=n,
        Q=q,
        PLANE=plane,
        BLOCK_SIZE=int(block_size),
        num_warps=int(num_warps),
    )
    return output


def warn_triton_fallback(reason: str) -> None:
    warnings.warn(
        f"Triton projector unavailable ({reason}); falling back to the PyTorch gather backend",
        RuntimeWarning,
        stacklevel=2,
    )
