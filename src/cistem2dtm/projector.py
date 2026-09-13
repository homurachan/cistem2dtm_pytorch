from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F

from .constants import TINY
from .fft import (
    centered_embed_phase_rfft2_cistem,
    centered_embed_phase_rfft2_phase,
    irfft2_cistem,
    rfft2_cistem,
    rfftn_cistem,
    signed_frequency_indices,
)
from .gisspa import ProjectionRadialBins, repository_radial_bins, whiten_projections_repository
from .triton_gather import (
    triton_half_hermitian_trilinear,
    triton_is_available,
    warn_triton_fallback,
)

Tensor = torch.Tensor
ProjectorBackend = Literal["auto", "triton", "gather", "grid_sample"]
ProjectionFFTMode = Literal["auto", "direct_s", "explicit"]
CenterPhaseMode = Literal["input", "projection"]

if TYPE_CHECKING:
    from .timing import PerformanceProfiler


@dataclass(slots=True)
class ProjectionBatch:
    """Products for one orientation batch.

    Debug intermediates are optional in production mode.  Keeping only the final
    Fourier projections substantially lowers peak memory without changing the
    numerical path.
    """

    padded_fourier: Tensor
    fourier_slice: Tensor | None = None
    sampled_filter: Tensor | None = None
    filtered_slice: Tensor | None = None
    real_before_edge_subtraction: Tensor | None = None
    edge_average: Tensor | None = None
    real_after_edge_subtraction: Tensor | None = None
    variance: Tensor | None = None
    real_normalized: Tensor | None = None
    padded_real: Tensor | None = None
    correlation_inverse_scale: Tensor | None = None
    projection_radial_power: Tensor | None = None


def _logical_bounds(n: int) -> tuple[int, int]:
    return -(n // 2), (n - 1) // 2


def _batch_edge_mean(images: Tensor) -> Tensor:
    if images.ndim != 3:
        raise ValueError("images must have shape (batch, height, width)")
    h, w = images.shape[-2:]
    if h <= 2 or w <= 1:
        return images.mean(dim=(-2, -1))
    edge = torch.cat(
        (
            images[:, 0, :],
            images[:, -1, :],
            images[:, 1:-1, 0],
            images[:, 1:-1, -1],
        ),
        dim=-1,
    )
    return edge.mean(dim=-1)


def _center_pad_crop_batch(images: Tensor, output_shape: tuple[int, int]) -> Tensor:
    if images.ndim != 3:
        raise ValueError("images must have shape (batch, height, width)")
    batch, in_h, in_w = images.shape
    out_h, out_w = (int(v) for v in output_shape)
    out = torch.zeros((batch, out_h, out_w), device=images.device, dtype=images.dtype)
    copy_h, copy_w = min(in_h, out_h), min(in_w, out_w)
    src_y = (in_h - copy_h) // 2
    src_x = (in_w - copy_w) // 2
    dst_y = (out_h - copy_h) // 2
    dst_x = (out_w - copy_w) // 2
    out[:, dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = images[
        :, src_y : src_y + copy_h, src_x : src_x + copy_w
    ]
    return out


class FourierProjector:
    """cisTEM-compatible Fourier central-slice projector with selectable backends."""

    def __init__(
        self,
        template_volume: Tensor,
        search_shape: tuple[int, int],
        *,
        backend: ProjectorBackend = "auto",
        projection_size: int | None = None,
        triton_fallback_to_gather: bool = True,
        triton_block_size: int = 256,
        triton_num_warps: int = 4,
        cache_grid_sample_source: bool = True,
        cache_sampled_filter: bool = True,
        projection_fft_mode: ProjectionFFTMode = "auto",
        center_phase_mode: CenterPhaseMode = "projection",
        fft_real_dtype: torch.dtype = torch.float32,
        per_projection_whitening: bool = False,
        projection_whitening_epsilon: float = 1.0e-12,
        mixed_precision_range_scaling: bool = True,
        mixed_precision_l1_target: float = 16384.0,
    ) -> None:
        if template_volume.ndim != 3:
            raise ValueError("template_volume must be 3-D")
        if len(set(int(v) for v in template_volume.shape)) != 1:
            raise ValueError("template_volume must be cubic")
        if backend not in ("auto", "triton", "gather", "grid_sample"):
            raise ValueError(f"unknown projector backend: {backend}")
        if projection_fft_mode not in ("auto", "direct_s", "explicit"):
            raise ValueError(f"unknown projection FFT mode: {projection_fft_mode}")
        if center_phase_mode not in ("input", "projection"):
            raise ValueError(f"unknown center phase mode: {center_phase_mode}")
        if fft_real_dtype not in (torch.float32, torch.float16):
            raise TypeError("large projection FFT dtype must be float32 or float16")
        self.template_volume = template_volume.contiguous()
        self.volume_size = int(template_volume.shape[-1])
        self.projection_size = int(projection_size or self.volume_size)
        self.search_shape = tuple(int(v) for v in search_shape)
        self.requested_backend = backend
        self.triton_fallback_to_gather = bool(triton_fallback_to_gather)
        self.triton_block_size = int(triton_block_size)
        self.triton_num_warps = int(triton_num_warps)
        self.cache_grid_sample_source = bool(cache_grid_sample_source)
        self.cache_sampled_filter = bool(cache_sampled_filter)
        self.projection_fft_mode = projection_fft_mode
        self.center_phase_mode = center_phase_mode
        self.fft_real_dtype = fft_real_dtype
        self.per_projection_whitening = bool(per_projection_whitening)
        self.projection_whitening_epsilon = float(projection_whitening_epsilon)
        self.mixed_precision_range_scaling = bool(mixed_precision_range_scaling)
        self.mixed_precision_l1_target = float(mixed_precision_l1_target)
        self.backend = self._resolve_backend(backend)

        self.volume_fourier_half = rfftn_cistem(
            self.template_volume, centered_real_space=True
        ).contiguous()
        if self.backend == "triton" and self.volume_fourier_half.dtype != torch.complex64:
            if not self.triton_fallback_to_gather:
                raise TypeError("Triton gather currently requires complex64 Fourier volume data")
            warn_triton_fallback(
                f"Fourier volume dtype is {self.volume_fourier_half.dtype}, not complex64"
            )
            self.backend = "gather"
        self._volume_fourier_full_shifted: Tensor | None = None
        self._grid_source: Tensor | None = None
        self._base_coordinates: dict[tuple[int, torch.device, torch.dtype], Tensor] = {}
        self._filter_cache: dict[
            tuple[int, int, float, torch.dtype, str], tuple[Tensor, Tensor]
        ] = {}
        self._embedding_phase_cache: dict[tuple[int, int, int, int, torch.dtype, str], Tensor] = {}
        self._projection_radial_bins: ProjectionRadialBins | None = None

    def _resolve_backend(self, backend: ProjectorBackend) -> Literal["triton", "gather", "grid_sample"]:
        if backend == "grid_sample":
            return "grid_sample"
        wants_triton = backend in {"auto", "triton"}
        if wants_triton and self.template_volume.device.type == "cuda" and triton_is_available():
            return "triton"
        if backend == "triton" and not self.triton_fallback_to_gather:
            reason = "Triton is not installed" if not triton_is_available() else "template is not on CUDA"
            raise RuntimeError(reason)
        if backend == "triton":
            reason = "Triton is not installed" if not triton_is_available() else "template is not on CUDA"
            warn_triton_fallback(reason)
        return "gather"

    @property
    def effective_backend(self) -> str:
        return self.backend

    @property
    def effective_fft_mode(self) -> str:
        return "direct_s" if self.projection_fft_mode == "auto" else self.projection_fft_mode

    @property
    def effective_center_phase_mode(self) -> str:
        if self.effective_fft_mode == "explicit":
            return "not_required"
        return self.center_phase_mode

    @property
    def large_fft_real_dtype(self) -> torch.dtype:
        return self.fft_real_dtype

    def center_embedding_phase(self) -> Tensor | None:
        """Return the direct-s center-embedding phase for one-time input prephasing."""
        if self.effective_fft_mode == "explicit":
            return None
        phase_key = (
            self.projection_size,
            self.projection_size,
            int(self.search_shape[0]),
            int(self.search_shape[1]),
            torch.complex64,
            str(self.device),
        )
        phase = self._embedding_phase_cache.get(phase_key)
        if phase is None:
            phase = centered_embed_phase_rfft2_phase(
                (self.projection_size, self.projection_size),
                self.search_shape,
                device=self.device,
                dtype=torch.complex64,
            )
            self._embedding_phase_cache[phase_key] = phase
        return phase

    @property
    def device(self) -> torch.device:
        return self.template_volume.device

    @property
    def real_dtype(self) -> torch.dtype:
        return self.template_volume.dtype

    def _base_plane(self, size: int, dtype: torch.dtype) -> Tensor:
        key = (int(size), self.device, dtype)
        cached = self._base_coordinates.get(key)
        if cached is not None:
            return cached
        y = signed_frequency_indices(size, device=self.device, dtype=dtype)
        x = torch.arange(size // 2 + 1, device=self.device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        base = torch.stack((xx, yy, torch.zeros_like(xx)), dim=0)
        self._base_coordinates[key] = base
        return base

    def rotated_coordinates(
        self,
        rotation_matrices: Tensor,
        *,
        real_space_binning_factor: float = 1.0,
        pixel_size_factor: float = 1.0,
    ) -> tuple[Tensor, float]:
        if rotation_matrices.ndim == 2:
            rotation_matrices = rotation_matrices.unsqueeze(0)
        if rotation_matrices.ndim != 3 or rotation_matrices.shape[-2:] != (3, 3):
            raise ValueError("rotation_matrices must have shape (batch, 3, 3)")
        if real_space_binning_factor <= 0 or pixel_size_factor <= 0:
            raise ValueError("binning and pixel-size factors must be positive")
        base = self._base_plane(self.projection_size, rotation_matrices.dtype)
        coordinates = torch.einsum("bij,jhw->bihw", rotation_matrices, base)
        scale = (
            float(self.volume_size)
            / float(self.projection_size)
            / float(real_space_binning_factor)
            / float(pixel_size_factor)
        )
        return coordinates * scale, scale

    def _sample_half_gather(self, coordinates: Tensor) -> Tensor:
        if coordinates.ndim != 4 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape (batch, 3, height, half_width)")
        volume = self.volume_fourier_half
        n = self.volume_size
        q = n // 2 + 1
        lower, upper = _logical_bounds(n)
        upper_x = n // 2
        lower_x = -upper_x

        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        negative_x = x < 0
        i0 = torch.floor(x).to(torch.long)
        j0 = torch.floor(y).to(torch.long)
        k0 = torch.floor(z).to(torch.long)
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        valid = (
            (((~negative_x) & (i1 <= upper_x)) | (negative_x & (i0 >= lower_x)))
            & (j0 >= lower)
            & (j1 <= upper)
            & (k0 >= lower)
            & (k1 <= upper)
        )

        flat = volume.reshape(-1)
        result = torch.zeros_like(x, dtype=volume.dtype)
        for dz in (0, 1):
            kk = k0 + dz
            wz = 1.0 - torch.abs(z - kk.to(z.dtype))
            for dy in (0, 1):
                jj = j0 + dy
                wy = 1.0 - torch.abs(y - jj.to(y.dtype))
                for dx in (0, 1):
                    ii = i0 + dx
                    wx = 1.0 - torch.abs(x - ii.to(x.dtype))
                    x_index = torch.where(negative_x, -ii, ii).clamp(0, q - 1)
                    y_index = torch.where(
                        negative_x, torch.remainder(-jj, n), torch.remainder(jj, n)
                    )
                    z_index = torch.where(
                        negative_x, torch.remainder(-kk, n), torch.remainder(kk, n)
                    )
                    address = (z_index * n + y_index) * q + x_index
                    value = flat[address]
                    value = torch.where(negative_x, torch.conj(value), value)
                    result = result + value * (wx * wy * wz)
        result = torch.where(valid, result, torch.zeros((), device=result.device, dtype=result.dtype))
        self._enforce_output_x0_hermitian_line(result)
        result[:, 0, 0] = 0
        return result

    def _sample_half_triton(self, coordinates: Tensor) -> Tensor:
        try:
            result = triton_half_hermitian_trilinear(
                self.volume_fourier_half,
                coordinates,
                block_size=self.triton_block_size,
                num_warps=self.triton_num_warps,
            )
        except Exception as exc:
            if not self.triton_fallback_to_gather:
                raise
            # A matching Triton package can still fail its first runtime JIT on
            # an unsupported driver/compiler combination.  Fall back once and
            # keep the resolved backend on gather for all later batches.
            warn_triton_fallback(f"runtime JIT failed: {type(exc).__name__}: {exc}")
            self.backend = "gather"
            return self._sample_half_gather(coordinates)
        self._enforce_output_x0_hermitian_line(result)
        result[:, 0, 0] = 0
        return result

    @staticmethod
    def _enforce_output_x0_hermitian_line(slices: Tensor) -> None:
        p = int(slices.shape[-2])
        upper = (p - 1) // 2
        if upper <= 0:
            return
        positive = torch.arange(1, upper + 1, device=slices.device)
        negative_physical = p - positive
        slices[:, negative_physical, 0] = torch.conj(slices[:, positive, 0])

    def _full_shifted_spectrum(self) -> Tensor:
        if self._volume_fourier_full_shifted is None:
            dims = (-3, -2, -1)
            centered = torch.fft.ifftshift(self.template_volume, dim=dims)
            full = torch.fft.fftn(centered, dim=dims, norm="forward")
            self._volume_fourier_full_shifted = torch.fft.fftshift(full, dim=dims).contiguous()
        return self._volume_fourier_full_shifted

    def _grid_sample_source_tensor(self) -> Tensor:
        if self.cache_grid_sample_source and self._grid_source is not None:
            return self._grid_source
        full = self._full_shifted_spectrum()
        source = torch.stack((full.real, full.imag), dim=0).unsqueeze(0).contiguous()
        if self.cache_grid_sample_source:
            self._grid_source = source
        return source

    def _sample_full_grid(self, coordinates: Tensor) -> Tensor:
        if coordinates.ndim != 4 or coordinates.shape[1] != 3:
            raise ValueError("coordinates must have shape (batch, 3, height, half_width)")
        n = self.volume_size
        lower, upper = _logical_bounds(n)
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        x0, y0, z0 = torch.floor(x), torch.floor(y), torch.floor(z)
        valid = (
            (x0 >= lower)
            & (x0 + 1 <= upper)
            & (y0 >= lower)
            & (y0 + 1 <= upper)
            & (z0 >= lower)
            & (z0 + 1 <= upper)
        )
        denominator = max(n - 1, 1)
        gx = 2.0 * ((x - lower) / denominator) - 1.0
        gy = 2.0 * ((y - lower) / denominator) - 1.0
        gz = 2.0 * ((z - lower) / denominator) - 1.0
        grid = torch.stack((gx, gy, gz), dim=-1).unsqueeze(0)
        sampled = F.grid_sample(
            self._grid_sample_source_tensor(),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[0]
        result = torch.complex(sampled[0], sampled[1])
        result = torch.where(valid, result, torch.zeros((), device=result.device, dtype=result.dtype))
        self._enforce_output_x0_hermitian_line(result)
        result[:, 0, 0] = 0
        return result

    @staticmethod
    def _sample_half_filter(filter_half: Tensor, output_size: int, scale: float) -> Tensor:
        if filter_half.ndim != 2:
            raise ValueError("filter_half must be 2-D")
        n, q = filter_half.shape
        if q != n // 2 + 1:
            raise ValueError("filter_half must be an rFFT half spectrum of a square image")
        device, dtype = filter_half.device, filter_half.dtype
        v = signed_frequency_indices(output_size, device=device, dtype=dtype)
        u = torch.arange(output_size // 2 + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(v * float(scale), u * float(scale), indexing="ij")
        i0 = torch.floor(xx).to(torch.long)
        j0 = torch.floor(yy).to(torch.long)
        i1, j1 = i0 + 1, j0 + 1
        lower, upper = _logical_bounds(n)
        valid = (i0 >= 0) & (i1 <= n // 2) & (j0 >= lower) & (j1 <= upper)
        result = torch.zeros_like(xx)
        flat = filter_half.reshape(-1)
        for dy in (0, 1):
            jj = j0 + dy
            wy = 1.0 - torch.abs(yy - jj.to(dtype))
            y_index = torch.remainder(jj, n)
            for dx in (0, 1):
                ii = i0 + dx
                wx = 1.0 - torch.abs(xx - ii.to(dtype))
                index = y_index * q + ii.clamp(0, q - 1)
                result = result + flat[index] * wx * wy
        result = torch.where(valid, result, torch.zeros((), device=device, dtype=dtype))
        result[0, 0] = 0
        return result

    def _cached_sampled_filter(self, filter_half: Tensor, scale: float) -> Tensor:
        key = (
            id(filter_half),
            self.projection_size,
            round(float(scale), 12),
            filter_half.dtype,
            str(filter_half.device),
        )
        if self.cache_sampled_filter:
            cached = self._filter_cache.get(key)
            if cached is not None:
                source, sampled = cached
                # Retaining the source tensor prevents Python-id / CUDA-pointer
                # reuse from ever returning a filter from another defocus plane.
                if source is filter_half:
                    return sampled
        sampled = self._sample_half_filter(filter_half, self.projection_size, scale)
        if self.cache_sampled_filter:
            self._filter_cache[key] = (filter_half, sampled)
        return sampled

    def extract_slices(
        self,
        rotation_matrices: Tensor,
        *,
        real_space_binning_factor: float = 1.0,
        pixel_size_factor: float = 1.0,
        backend: ProjectorBackend | None = None,
        profiler: "PerformanceProfiler | None" = None,
    ) -> tuple[Tensor, float]:
        with (
            profiler.batch_section("projection.rotated_coordinates")
            if profiler is not None
            else nullcontext()
        ):
            coordinates, scale = self.rotated_coordinates(
                rotation_matrices,
                real_space_binning_factor=real_space_binning_factor,
                pixel_size_factor=pixel_size_factor,
            )
        selected = self._resolve_backend(backend) if backend is not None else self.backend
        with (
            profiler.batch_section(f"projection.sample_slice_{selected}")
            if profiler is not None
            else nullcontext()
        ):
            if selected == "gather":
                return self._sample_half_gather(coordinates), scale
            if selected == "triton":
                return self._sample_half_triton(coordinates), scale
            if selected == "grid_sample":
                return self._sample_full_grid(coordinates), scale
        raise ValueError(f"unknown projector backend: {selected}")

    def _projection_whitening_bins(self) -> ProjectionRadialBins:
        if self._projection_radial_bins is None:
            self._projection_radial_bins = repository_radial_bins(
                self.projection_size,
                device=self.device,
                dtype=torch.float32,
            )
        return self._projection_radial_bins

    def _prepare_half_fft_input(
        self, normalized: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Mean-center and power-of-two scale each projection for safe half FFT."""
        if self.fft_real_dtype != torch.float16:
            return normalized, None
        # Zeroing the Fourier DC after the transform is mathematically equivalent
        # to removing the real-space mean first, but the latter prevents the DC
        # butterfly from overflowing before cuFFT applies forward normalization.
        centered = normalized - normalized.mean(dim=(-2, -1), keepdim=True)
        if not self.mixed_precision_range_scaling:
            return centered, torch.ones(
                (centered.shape[0],), device=centered.device, dtype=torch.float32
            )
        l1 = centered.abs().sum(dim=(-2, -1), dtype=torch.float32).clamp_min(TINY)
        ratio = torch.full_like(l1, self.mixed_precision_l1_target) / l1
        exponent = torch.floor(torch.log2(ratio)).clamp(max=0.0, min=-60.0)
        scale = torch.exp2(exponent)
        inverse_scale = torch.exp2(-exponent)
        return centered * scale[:, None, None], inverse_scale

    def _projection_fft(
        self, normalized: Tensor, *, retain_intermediates: bool
    ) -> tuple[Tensor, Tensor | None]:
        mode = self.effective_fft_mode
        work = normalized.to(self.fft_real_dtype)
        if mode == "explicit":
            padded = _center_pad_crop_batch(work, self.search_shape)
            fourier = rfft2_cistem(padded, centered_real_space=False)
            return fourier, padded

        phase = self.center_embedding_phase()
        fourier, _ = centered_embed_phase_rfft2_cistem(
            work,
            self.search_shape,
            phase=phase,
            apply_phase=self.center_phase_mode == "projection",
        )
        padded = _center_pad_crop_batch(work, self.search_shape) if retain_intermediates else None
        return fourier, padded

    def project_and_normalize(
        self,
        rotation_matrices: Tensor,
        projection_filter_half: Tensor,
        *,
        real_space_binning_factor: float = 1.0,
        pixel_size_factor: float = 1.0,
        backend: ProjectorBackend | None = None,
        filter_scale_override: float | None = None,
        profiler: "PerformanceProfiler | None" = None,
        retain_intermediates: bool = True,
    ) -> ProjectionBatch:
        slices, scale = self.extract_slices(
            rotation_matrices,
            real_space_binning_factor=real_space_binning_factor,
            pixel_size_factor=pixel_size_factor,
            backend=backend,
            profiler=profiler,
        )
        projection_radial_power: Tensor | None = None
        if self.per_projection_whitening:
            with (
                profiler.batch_section("projection.per_projection_whitening")
                if profiler is not None
                else nullcontext()
            ):
                slices, projection_radial_power = whiten_projections_repository(
                    slices,
                    self._projection_whitening_bins(),
                    epsilon=self.projection_whitening_epsilon,
                )
        filter_scale = float(scale if filter_scale_override is None else filter_scale_override)
        with (
            profiler.batch_section("projection.sample_filter")
            if profiler is not None
            else nullcontext()
        ):
            sampled_filter = self._cached_sampled_filter(projection_filter_half, filter_scale)
        with (
            profiler.batch_section("projection.filter_multiply")
            if profiler is not None
            else nullcontext()
        ):
            filtered = slices * sampled_filter.unsqueeze(0)
            filtered[:, 0, 0] = 0

        projection_shape = (self.projection_size, self.projection_size)
        with (
            profiler.batch_section("projection.irfft_small")
            if profiler is not None
            else nullcontext()
        ):
            real = irfft2_cistem(filtered, projection_shape, centered_real_space=True)
        with (
            profiler.batch_section("projection.edge_mean")
            if profiler is not None
            else nullcontext()
        ):
            edge = _batch_edge_mean(real)
        with (
            profiler.batch_section("projection.edge_subtract")
            if profiler is not None
            else nullcontext()
        ):
            after_edge = real - edge[:, None, None]

        ratio = float(self.projection_size * self.projection_size) / float(
            self.search_shape[0] * self.search_shape[1]
        )
        with (
            profiler.batch_section("projection.variance_normalize")
            if profiler is not None
            else nullcontext()
        ):
            mean_square = after_edge.square().mean(dim=(-2, -1))
            mean = after_edge.mean(dim=(-2, -1))
            variance = (mean_square * ratio - (mean * ratio).square()).clamp_min(TINY)
            normalized = after_edge / torch.sqrt(variance)[:, None, None]

        # direct_s asks cuFFT to enlarge the transform directly.  In the mixed
        # path an exact power-of-two scale protects half cuFFT's dynamic range;
        # the inverse scale is restored in the fused Fourier multiplication.
        fft_input, correlation_inverse_scale = self._prepare_half_fft_input(normalized)
        fft_mode = self.effective_fft_mode
        if fft_mode == "explicit":
            with (
                profiler.batch_section("projection.pad_to_search")
                if profiler is not None
                else nullcontext()
            ):
                padded = _center_pad_crop_batch(
                    fft_input.to(self.fft_real_dtype), self.search_shape
                )
            with (
                profiler.batch_section("projection.rfft_large")
                if profiler is not None
                else nullcontext()
            ):
                padded_fourier = rfft2_cistem(padded, centered_real_space=False)
        else:
            with (
                profiler.batch_section("projection.rfft_large_direct_s")
                if profiler is not None
                else nullcontext()
            ):
                padded_fourier, padded = self._projection_fft(
                    fft_input,
                    retain_intermediates=retain_intermediates,
                )
        # Avoid complex32 scalar-operation gaps by zeroing the real view.
        torch.view_as_real(padded_fourier)[:, 0, 0, :].zero_()

        return ProjectionBatch(
            padded_fourier=padded_fourier,
            fourier_slice=slices if retain_intermediates else None,
            sampled_filter=sampled_filter if retain_intermediates else None,
            filtered_slice=filtered if retain_intermediates else None,
            real_before_edge_subtraction=real if retain_intermediates else None,
            edge_average=edge if retain_intermediates else None,
            real_after_edge_subtraction=after_edge if retain_intermediates else None,
            variance=variance if retain_intermediates else None,
            real_normalized=normalized if retain_intermediates else None,
            padded_real=padded if retain_intermediates else None,
            correlation_inverse_scale=correlation_inverse_scale,
            projection_radial_power=(
                projection_radial_power if retain_intermediates else None
            ),
        )

    def compare_backends(
        self,
        rotation_matrices: Tensor,
        *,
        real_space_binning_factor: float = 1.0,
        pixel_size_factor: float = 1.0,
    ) -> dict[str, Tensor]:
        gather, _ = self.extract_slices(
            rotation_matrices,
            real_space_binning_factor=real_space_binning_factor,
            pixel_size_factor=pixel_size_factor,
            backend="gather",
        )
        grid, _ = self.extract_slices(
            rotation_matrices,
            real_space_binning_factor=real_space_binning_factor,
            pixel_size_factor=pixel_size_factor,
            backend="grid_sample",
        )
        difference = gather - grid
        denominator = torch.linalg.vector_norm(
            gather.reshape(gather.shape[0], -1), dim=1
        ).clamp_min(TINY)
        relative_l2 = torch.linalg.vector_norm(
            difference.reshape(difference.shape[0], -1), dim=1
        ) / denominator
        result = {
            "gather": gather,
            "grid_sample": grid,
            "difference": difference,
            "relative_l2": relative_l2,
            "maximum_absolute_difference": difference.abs().amax(dim=(-2, -1)),
        }
        if self.device.type == "cuda" and triton_is_available():
            triton_slice, _ = self.extract_slices(
                rotation_matrices,
                real_space_binning_factor=real_space_binning_factor,
                pixel_size_factor=pixel_size_factor,
                backend="triton",
            )
            triton_difference = gather - triton_slice
            result.update(
                {
                    "triton": triton_slice,
                    "triton_difference": triton_difference,
                    "triton_relative_l2": torch.linalg.vector_norm(
                        triton_difference.reshape(triton_difference.shape[0], -1), dim=1
                    ) / denominator,
                    "triton_maximum_absolute_difference": triton_difference.abs().amax(dim=(-2, -1)),
                }
            )
        return result
