from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import torch
import torch.nn.functional as F

from .orientation_cache import WinnerMetadataTable

from .constants import (
    HISTOGRAM_BINS,
    HISTOGRAM_FIRST_MIDPOINT,
    HISTOGRAM_MAX,
    HISTOGRAM_MIN,
    HISTOGRAM_STEP,
    STAT_EPSILON,
    TINY,
)

Tensor = torch.Tensor

if TYPE_CHECKING:
    from .timing import PerformanceProfiler


@dataclass(slots=True)
class RawSearchResult:
    mip: Tensor
    phi: Tensor
    theta: Tensor
    psi: Tensor
    defocus: Tensor
    pixel_size: Tensor
    correlation_sum: Tensor
    correlation_sum_squares: Tensor
    histogram: Tensor
    number_searched: int
    winner_task_index: Tensor

    def to(self, device: torch.device) -> "RawSearchResult":
        return RawSearchResult(
            mip=self.mip.to(device),
            phi=self.phi.to(device),
            theta=self.theta.to(device),
            psi=self.psi.to(device),
            defocus=self.defocus.to(device),
            pixel_size=self.pixel_size.to(device),
            correlation_sum=self.correlation_sum.to(device),
            correlation_sum_squares=self.correlation_sum_squares.to(device),
            histogram=self.histogram.to(device),
            number_searched=int(self.number_searched),
            winner_task_index=self.winner_task_index.to(device),
        )

    def cpu(self) -> "RawSearchResult":
        return self.to(torch.device("cpu"))

    def maps(self) -> dict[str, Tensor]:
        return {
            "mip": self.mip,
            "phi": self.phi,
            "theta": self.theta,
            "psi": self.psi,
            "defocus": self.defocus,
            "pixel_size": self.pixel_size,
            "correlation_sum": self.correlation_sum,
            "correlation_sum_squares": self.correlation_sum_squares,
        }


@dataclass(slots=True)
class ScaledSearchResult:
    mip_zscore: Tensor
    scaled_mip: Tensor
    local_mean: Tensor
    local_std: Tensor
    normalized_sum: Tensor
    normalized_sum_squares: Tensor
    global_mean: float
    global_std: float
    counted_pixels: int
    raw_histogram: Tensor
    smoothed_histogram: Tensor
    histogram_x: Tensor
    source_curve_x_scaled: Tensor


class SearchAccumulators:
    """Streaming MIP, moments and empirical histogram.

    The MIP and correlation moments follow the source-valid search ROI.  The
    empirical histogram may use a separate compatibility ROI, but that ROI must
    never punch a hole in ``sum``/``sumSq`` because cisTEM only invalidates
    moments with the final post-search valid-area mask.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        mip_initial_value: float = 0.0,
        valid_mask: Tensor | None = None,
        statistics_mask: Tensor | None = None,
        histogram_mode: str = "exact",
        histogram_backend: str = "histc",
        histogram_sample_orientation_stride: int = 16,
        histogram_sample_pixel_stride: int = 4,
        winner_metadata: WinnerMetadataTable | None = None,
    ) -> None:
        self.shape = tuple(int(v) for v in shape)
        self.device = device
        self.dtype = dtype
        self.mip = torch.full(self.shape, float(mip_initial_value), device=device, dtype=dtype)
        self.correlation_sum = torch.zeros(self.shape, device=device, dtype=dtype)
        self.correlation_sum_squares = torch.zeros(self.shape, device=device, dtype=dtype)
        # float64 preserves integer exactness for the search sizes used here and
        # also permits deterministic reweighting in sampled histogram mode.
        self.histogram = torch.zeros(HISTOGRAM_BINS, device=device, dtype=torch.float64)
        self.number_searched = 0
        self.winner_task_index = torch.full(
            self.shape,
            torch.iinfo(torch.int64).max,
            device=device,
            dtype=torch.int64,
        )
        if valid_mask is None:
            self.valid_mask = torch.ones(self.shape, device=device, dtype=torch.bool)
        else:
            if tuple(valid_mask.shape) != self.shape:
                raise ValueError("valid_mask shape differs from accumulator shape")
            self.valid_mask = valid_mask.to(device=device, dtype=torch.bool)
        if statistics_mask is None:
            self.statistics_mask = self.valid_mask
        else:
            if tuple(statistics_mask.shape) != self.shape:
                raise ValueError("statistics_mask shape differs from accumulator shape")
            # Historical name retained for config compatibility: this mask is
            # histogram-only.  Moments always use valid_mask.
            self.statistics_mask = statistics_mask.to(device=device, dtype=torch.bool)

        if histogram_mode not in {"exact", "sampled", "off"}:
            raise ValueError(f"unknown histogram mode: {histogram_mode}")
        if histogram_backend not in {"histc", "bincount"}:
            raise ValueError(f"unknown histogram backend: {histogram_backend}")
        if histogram_sample_orientation_stride <= 0 or histogram_sample_pixel_stride <= 0:
            raise ValueError("histogram sample strides must be positive")
        self.histogram_mode = histogram_mode
        self.histogram_backend = histogram_backend
        self.histogram_sample_orientation_stride = int(histogram_sample_orientation_stride)
        self.histogram_sample_pixel_stride = int(histogram_sample_pixel_stride)
        self.winner_metadata = winner_metadata
        # Backward-compatible low-level API: callers that do not provide a
        # winner lookup table may still pass per-batch metadata.  The production
        # matcher always supplies winner_metadata, so no five full-size maps are
        # allocated or updated inside its search loop.
        self._legacy_winner_maps = None
        if winner_metadata is None:
            self._legacy_winner_maps = [
                torch.zeros(self.shape, device=device, dtype=dtype) for _ in range(5)
            ]
            # Preserve the historical low-level attributes for external tests and
            # callers.  The production matcher supplies winner_metadata, so these
            # aliases and buffers do not exist in the performance path.
            self.phi, self.theta, self.psi, self.defocus, self.pixel_size = self._legacy_winner_maps
        self._moment_slice = self._rectangular_mask_slice(self.valid_mask)
        self._histogram_slice = self._rectangular_mask_slice(self.statistics_mask)
        self._moment_mask_float = self.valid_mask.to(dtype=self.dtype)

    @staticmethod
    def _rectangular_mask_slice(mask: Tensor) -> tuple[slice, slice] | None:
        """Return a zero-copy rectangular ROI when a boolean mask is a box.

        cisTEM's search and histogram ROIs are rectangles.  Detecting that once
        avoids a large boolean-indexed copy for every orientation batch.  The
        one-time scalar reads occur during accumulator construction, not in the
        search loop.
        """
        if mask.ndim != 2 or mask.numel() == 0:
            return None
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        row_ids = torch.nonzero(rows, as_tuple=False).flatten()
        col_ids = torch.nonzero(cols, as_tuple=False).flatten()
        if row_ids.numel() == 0 or col_ids.numel() == 0:
            return None
        y0 = int(row_ids[0].item())
        y1 = int(row_ids[-1].item()) + 1
        x0 = int(col_ids[0].item())
        x1 = int(col_ids[-1].item()) + 1
        expected = torch.zeros_like(mask)
        expected[y0:y1, x0:x1] = True
        if not bool(torch.equal(mask, expected)):
            return None
        return slice(y0, y1), slice(x0, x1)

    def _masked_values(self, values: Tensor, *, histogram: bool) -> Tensor:
        mask = self.statistics_mask if histogram else self.valid_mask
        rectangular = self._histogram_slice if histogram else self._moment_slice
        if rectangular is not None:
            ys, xs = rectangular
            return values[:, ys, xs]
        return values[:, mask]

    @staticmethod
    def _legacy_bincount_histogram(values: Tensor) -> Tensor:
        flat = values.reshape(-1)
        bins = torch.floor((flat - HISTOGRAM_MIN) / HISTOGRAM_STEP).to(torch.int64)
        in_range = (flat >= HISTOGRAM_MIN) & (flat < HISTOGRAM_MAX)
        # Send invalid samples to a disposable overflow bin.  This keeps the
        # legacy exact backend free of the former boolean-indexed CCC copy and
        # avoids a host-side ``if bool(in_range.any())`` synchronization.
        bins = torch.where(
            in_range,
            bins.clamp(0, HISTOGRAM_BINS - 1),
            torch.full_like(bins, HISTOGRAM_BINS),
        )
        return torch.bincount(
            bins,
            minlength=HISTOGRAM_BINS + 1,
        )[:HISTOGRAM_BINS].to(torch.float64)

    @staticmethod
    def _histc_exact_histogram(values: Tensor) -> Tensor:
        """Fast exact-bin histogram using bounded float32 partial counts.

        CUDA ``torch.histc`` avoids the very large int64 bin-index tensor used
        by ``bincount``.  Its result dtype is float32, so every partial call is
        limited to at most ``2**24`` input samples, where integer counts are
        exactly representable.  Partial counts are rounded, converted to int64,
        and accumulated in float64.  ``torch.histc`` includes its upper endpoint
        in the final bin, whereas cisTEM uses a half-open interval, so values
        exactly equal to ``HISTOGRAM_MAX`` are explicitly removed.
        """
        output = torch.zeros(HISTOGRAM_BINS, device=values.device, dtype=torch.float64)
        if values.numel() == 0:
            return output

        if values.ndim == 1:
            values = values.unsqueeze(0)
        max_exact_samples = 1 << 24
        batch = int(values.shape[0])
        samples_per_orientation = int(values[0].numel())

        if samples_per_orientation <= max_exact_samples:
            orientations_per_chunk = max(1, max_exact_samples // samples_per_orientation)
            chunks = (
                values[start : start + orientations_per_chunk]
                for start in range(0, batch, orientations_per_chunk)
            )
        else:
            # This is only expected above a 4096^2 ROI.  Split each orientation
            # into bounded flat chunks; ordinary <=4k searches remain zero-copy
            # rectangular views throughout the hot path.
            chunks = (
                flat[start : start + max_exact_samples]
                for orientation in values
                for flat in (orientation.reshape(-1),)
                for start in range(0, int(flat.numel()), max_exact_samples)
            )

        for chunk in chunks:
            partial = torch.histc(
                chunk.to(torch.float32),
                bins=HISTOGRAM_BINS,
                min=float(HISTOGRAM_MIN),
                max=float(HISTOGRAM_MAX),
            )
            partial_int = torch.round(partial).to(torch.int64)
            partial_int[-1].sub_(torch.count_nonzero(chunk == float(HISTOGRAM_MAX)))
            output.add_(partial_int.to(torch.float64))
        return output

    def _histogram_counts(self, values: Tensor) -> Tensor:
        if self.histogram_backend == "histc":
            return self._histc_exact_histogram(values)
        return self._legacy_bincount_histogram(values)

    def _update_histogram(self, values: Tensor, task_indices: Tensor) -> None:
        if self.histogram_mode == "off":
            return
        roi_values = self._masked_values(values, histogram=True)
        if self.histogram_mode == "exact":
            self.histogram.add_(self._histogram_counts(roi_values))
            return

        # Deterministic low-cost sampling in global task-index space.  This is
        # invariant to batch boundaries and one-server multi-GPU sharding.
        orientation_stride = self.histogram_sample_orientation_stride
        pixel_stride = self.histogram_sample_pixel_stride
        task_indices_device = task_indices.to(device=self.device, dtype=torch.int64)
        sampled_orientations = torch.remainder(task_indices_device, orientation_stride) == 0

        if roi_values.ndim == 3:
            sampled = roi_values[
                sampled_orientations,
                ::pixel_stride,
                ::pixel_stride,
            ]
        else:
            # Non-rectangular fallback is flattened per orientation.
            sampled = roi_values[sampled_orientations, ::pixel_stride]
        sampled_count = int(sampled.numel())
        full_count = int(roi_values.numel())
        if sampled_count > 0:
            weight = float(full_count) / float(sampled_count)
            self.histogram.add_(self._histogram_counts(sampled) * weight)

    def update(
        self,
        ccf: Tensor,
        *,
        task_indices: Tensor,
        phi: Tensor | None = None,
        theta: Tensor | None = None,
        psi: Tensor | None = None,
        defocus: Tensor | None = None,
        pixel_size: Tensor | None = None,
        profiler: "PerformanceProfiler | None" = None,
    ) -> None:
        if ccf.ndim != 3 or tuple(ccf.shape[-2:]) != self.shape:
            raise ValueError("ccf must have shape (batch, height, width)")
        batch = int(ccf.shape[0])
        if int(task_indices.numel()) != batch:
            raise ValueError("task_indices must have one value per CCF")
        optional_metadata = (phi, theta, psi, defocus, pixel_size)
        if any(value is not None for value in optional_metadata):
            if not all(value is not None for value in optional_metadata):
                raise ValueError("legacy winner metadata must supply phi/theta/psi/defocus/pixel_size together")
            if any(int(value.numel()) != batch for value in optional_metadata if value is not None):
                raise ValueError("all legacy metadata arrays must have one value per CCF")
        values = ccf.to(self.dtype)

        with (profiler.batch_section("accumulator.moments") if profiler is not None else nullcontext()):
            moment_values = self._masked_values(values, histogram=False)
            if self._moment_slice is not None:
                ys, xs = self._moment_slice
                self.correlation_sum[ys, xs].add_(moment_values.sum(dim=0))
                self.correlation_sum_squares[ys, xs].add_(moment_values.square().sum(dim=0))
            else:
                self.correlation_sum.add_(values.sum(dim=0) * self._moment_mask_float)
                self.correlation_sum_squares.add_(values.square().sum(dim=0) * self._moment_mask_float)
            self.number_searched += batch

        if self.histogram_mode != "off":
            histogram_section = (
                "accumulator.histogram_exact"
                if self.histogram_mode == "exact"
                else "accumulator.histogram_sampled"
            )
            with (
                profiler.batch_section(histogram_section)
                if profiler is not None
                else nullcontext()
            ):
                self._update_histogram(values, task_indices)

        with (profiler.batch_section("accumulator.mip_reduce") if profiler is not None else nullcontext()):
            # Invalid pixels are excluded only from the update predicate.  This
            # avoids materialising a second full CCF batch with masked_fill.
            batch_max, local_index = torch.max(values, dim=0)  # first occurrence wins within batch
            improved = self.valid_mask & (batch_max > self.mip)

        # Deliberately no bool(improved.any()) here: converting a CUDA tensor to
        # a Python bool forced one device synchronization per batch.  Empty
        # advanced-index assignments are valid and preserve strict cisTEM >
        # semantics without a host round trip.
        with (profiler.batch_section("accumulator.winner_task") if profiler is not None else nullcontext()):
            self.mip[improved] = batch_max[improved]
            task_indices_device = task_indices.to(device=self.device, dtype=torch.int64)
            selected_task = task_indices_device[local_index]
            self.winner_task_index[improved] = selected_task[improved]
            if self._legacy_winner_maps is not None and phi is not None:
                for destination, source in zip(
                    self._legacy_winner_maps,
                    (phi, theta, psi, defocus, pixel_size),
                ):
                    assert source is not None
                    chosen = source.to(device=self.device, dtype=self.dtype)[local_index]
                    destination[improved] = chosen[improved]

    def result(self) -> RawSearchResult:
        if self.winner_metadata is None:
            assert self._legacy_winner_maps is not None
            phi, theta, psi, defocus, pixel_size = self._legacy_winner_maps
        else:
            phi, theta, psi, defocus, pixel_size = self.winner_metadata.materialize(
                self.winner_task_index,
                self.mip,
                dtype=self.dtype,
            )
        return RawSearchResult(
            mip=self.mip,
            phi=phi,
            theta=theta,
            psi=psi,
            defocus=defocus,
            pixel_size=pixel_size,
            correlation_sum=self.correlation_sum,
            correlation_sum_squares=self.correlation_sum_squares,
            histogram=self.histogram,
            number_searched=self.number_searched,
            winner_task_index=self.winner_task_index,
        )

def merge_raw_results(results: Sequence[RawSearchResult], *, device: torch.device | None = None) -> RawSearchResult:
    if not results:
        raise ValueError("at least one raw result is required")
    device = device or results[0].mip.device
    first = results[0].to(device)
    merged = RawSearchResult(
        mip=first.mip.clone(),
        phi=first.phi.clone(),
        theta=first.theta.clone(),
        psi=first.psi.clone(),
        defocus=first.defocus.clone(),
        pixel_size=first.pixel_size.clone(),
        correlation_sum=first.correlation_sum.clone(),
        correlation_sum_squares=first.correlation_sum_squares.clone(),
        histogram=first.histogram.clone(),
        number_searched=int(first.number_searched),
        winner_task_index=first.winner_task_index.clone(),
    )
    for partial_original in results[1:]:
        partial = partial_original.to(device)
        if tuple(partial.mip.shape) != tuple(merged.mip.shape):
            raise ValueError("cannot merge results with different image shapes")
        merged.correlation_sum.add_(partial.correlation_sum)
        merged.correlation_sum_squares.add_(partial.correlation_sum_squares)
        merged.histogram.add_(partial.histogram)
        merged.number_searched += int(partial.number_searched)

        greater = partial.mip > merged.mip
        equal_better_order = (
            (partial.mip == merged.mip)
            & (partial.mip > 0)
            & (partial.winner_task_index < merged.winner_task_index)
        )
        choose = greater | equal_better_order
        if bool(choose.any()):
            for name in ("mip", "phi", "theta", "psi", "defocus", "pixel_size", "winner_task_index"):
                destination = getattr(merged, name)
                source = getattr(partial, name)
                destination[choose] = source[choose]
    return merged


def _savgol_window5_order3(values: Tensor) -> Tensor:
    """Savitzky-Golay smoothing for the exact 5-point/order-3 request.

    Interior coefficients are [-3, 12, 17, 12, -3] / 35.  Replicated edge
    padding keeps the operation dependency-free and deterministic; the interior,
    which dominates a 512-bin histogram, is identical to the standard fit.
    """
    if values.ndim != 1:
        raise ValueError("histogram must be one-dimensional")
    kernel = torch.tensor([-3.0, 12.0, 17.0, 12.0, -3.0], device=values.device, dtype=torch.float64) / 35.0
    padded = F.pad(values.to(torch.float64)[None, None], (2, 2), mode="replicate")
    return F.conv1d(padded, kernel[None, None])[0, 0]


def scale_search_statistics(
    raw: RawSearchResult,
    *,
    disable_flat_fielding: bool,
    valid_mask: Tensor | None = None,
    epsilon: float = STAT_EPSILON,
) -> ScaledSearchResult:
    if raw.number_searched <= 0:
        raise ValueError("number_searched must be positive")
    sum_map = raw.correlation_sum
    sum_squares = raw.correlation_sum_squares
    counted = sum_squares > float(epsilon)
    if valid_mask is not None:
        if tuple(valid_mask.shape) != tuple(sum_squares.shape):
            raise ValueError("valid_mask shape differs from scaled statistic maps")
        valid_mask = valid_mask.to(device=sum_squares.device, dtype=torch.bool)
        counted = counted & valid_mask
    counted_pixels = int(counted.sum().item())
    if counted_pixels == 0:
        raise RuntimeError("no valid CCC pixels were accumulated")

    global_sum = sum_map[counted].to(torch.float64).sum()
    global_sum_squares = sum_squares[counted].to(torch.float64).sum()
    total = float(raw.number_searched * counted_pixels)
    global_mean_t = global_sum / total
    variance_t = global_sum_squares / total - global_mean_t.square()
    global_std_t = torch.sqrt(variance_t.clamp_min(TINY))
    global_mean = float(global_mean_t.item())
    global_std = float(global_std_t.item())

    # ResizeImage_postSearch has already mean-filled the raw MIP outside its
    # binarized cosine mask.  cisTEM then applies the same global affine scaling
    # to every pixel, so no second border rewrite belongs here.
    mip_zscore = (raw.mip - global_mean) / global_std
    n = float(raw.number_searched)
    normalized_sum_squares = (
        sum_squares
        - 2.0 * global_mean * sum_map
        + n * global_mean * global_mean
    ) / (global_std * global_std)
    normalized_sum = (sum_map - n * global_mean) / global_std
    local_mean = normalized_sum / n
    local_variance = normalized_sum_squares / n - local_mean.square()
    local_std = torch.sqrt(local_variance.clamp_min(0.0))

    # cisTEM leaves zeroed accumulators unchanged outside the counted ROI.
    # Without this guard, subtracting the global mean would create a non-zero
    # corr_average border even though no CCF values were accumulated there.
    normalized_sum = torch.where(counted, normalized_sum, torch.zeros_like(normalized_sum))
    normalized_sum_squares = torch.where(
        counted, normalized_sum_squares, torch.zeros_like(normalized_sum_squares)
    )
    local_mean = torch.where(counted, local_mean, torch.zeros_like(local_mean))
    local_std = torch.where(counted, local_std, torch.zeros_like(local_std))

    scaled_mip = mip_zscore.clone()
    usable = counted & (local_std > float(epsilon))
    if not disable_flat_fielding:
        scaled_mip[usable] = (mip_zscore[usable] - local_mean[usable]) / local_std[usable]

    raw_hist = raw.histogram.to(torch.float64)
    smooth = _savgol_window5_order3(raw_hist).clamp_min(0.0)
    raw_midpoints = (
        torch.arange(HISTOGRAM_BINS, device=raw_hist.device, dtype=torch.float64) * HISTOGRAM_STEP
        + HISTOGRAM_FIRST_MIDPOINT
    )
    scaled_x = (raw_midpoints - global_mean) / global_std
    return ScaledSearchResult(
        mip_zscore=mip_zscore,
        scaled_mip=scaled_mip,
        local_mean=local_mean,
        local_std=local_std,
        normalized_sum=normalized_sum,
        normalized_sum_squares=normalized_sum_squares,
        global_mean=global_mean,
        global_std=global_std,
        counted_pixels=counted_pixels,
        raw_histogram=raw.histogram,
        smoothed_histogram=torch.round(smooth).to(torch.int64),
        histogram_x=raw_midpoints,
        source_curve_x_scaled=scaled_x,
    )


def _normal_isf(tail_probability: float) -> float:
    """Inverse standard-normal survival function without SciPy.

    Acklam's rational approximation is evaluated from the tail probability
    directly, avoiding ``1-p`` cancellation for very large searches.  Two
    scalar Newton refinements use :func:`math.erfc`, not SciPy and not a
    CPU/GPU tensor round trip.
    """
    tiny = float.fromhex("0x1.0p-1022")
    eps = 2.220446049250313e-16
    qtail = min(max(float(tail_probability), tiny), 1.0 - eps)
    # Coefficients from Peter J. Acklam's inverse-normal approximation.
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e00, 3.754408661907416e00)
    plow = 0.02425

    def ppf(probability: float) -> float:
        if probability < plow:
            r = math.sqrt(-2.0 * math.log(probability))
            return (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / (
                ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0)
            )
        if probability > 1.0 - plow:
            r = math.sqrt(-2.0 * math.log1p(-probability))
            return -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / (
                ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0)
            )
        r = probability - 0.5
        s = r * r
        return (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / (
            (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1.0)
        )

    if qtail <= 0.5:
        value = -ppf(qtail)
    else:
        value = ppf(1.0 - qtail)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    for _ in range(2):
        survival = 0.5 * math.erfc(value / math.sqrt(2.0))
        density = math.exp(-0.5 * value * value) * inv_sqrt_2pi
        if density == 0.0:
            break
        value += (survival - qtail) / density
    return value


def theoretical_threshold(
    *,
    expected_false_positives: float,
    number_of_valid_search_pixels: int,
    number_searched: int,
    independent_fraction: float,
    defocus_positions: int = 1,
    ignore_defocus_for_threshold: bool = False,
) -> float:
    if expected_false_positives <= 0:
        raise ValueError("expected_false_positives must be positive")
    if number_of_valid_search_pixels <= 0 or number_searched <= 0:
        raise ValueError("search counts must be positive")
    effective = float(number_of_valid_search_pixels) * float(number_searched) * float(independent_fraction)
    if ignore_defocus_for_threshold:
        effective /= max(int(defocus_positions), 1)
    return _normal_isf(float(expected_false_positives) / effective)


def expected_gaussian_survival(z: Tensor, number_independent: float) -> Tensor:
    z64 = z.to(torch.float64)
    return 0.5 * float(number_independent) * torch.erfc(z64 / math.sqrt(2.0))


@dataclass(slots=True)
class Peak:
    rank: int
    x_pixel: int
    y_pixel: int
    x_angstrom: float
    y_angstrom: float
    score: float
    mip_zscore: float
    phi_deg: float
    theta_deg: float
    psi_deg: float
    defocus_offset_angstrom: float
    pixel_size_offset_angstrom: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "rank": self.rank,
            "x_pixel": self.x_pixel,
            "y_pixel": self.y_pixel,
            "x_angstrom": self.x_angstrom,
            "y_angstrom": self.y_angstrom,
            "score": self.score,
            "mip_zscore": self.mip_zscore,
            "phi_deg": self.phi_deg,
            "theta_deg": self.theta_deg,
            "psi_deg": self.psi_deg,
            "defocus_offset_angstrom": self.defocus_offset_angstrom,
            "pixel_size_offset_angstrom": self.pixel_size_offset_angstrom,
        }


def peak_pick(
    scaled_mip: Tensor,
    mip_zscore: Tensor,
    parameter_maps: Mapping[str, Tensor],
    *,
    threshold: float,
    minimum_radius_pixels: float,
    pixel_size_angstrom: float,
    maximum_peaks: int = 1000,
) -> list[Peak]:
    if scaled_mip.ndim != 2 or tuple(mip_zscore.shape) != tuple(scaled_mip.shape):
        raise ValueError("MIP maps must be two-dimensional and have equal shapes")
    required = {"phi", "theta", "psi", "defocus", "pixel_size"}
    missing = required - set(parameter_maps)
    if missing:
        raise KeyError(f"missing parameter maps: {sorted(missing)}")
    work = scaled_mip.detach().clone()
    h, w = work.shape
    radius_sq = max(float(minimum_radius_pixels), 0.0) ** 2
    peaks: list[Peak] = []
    yy_all = torch.arange(h, device=work.device)
    xx_all = torch.arange(w, device=work.device)

    for rank in range(1, max(int(maximum_peaks), 0) + 1):
        flat_index = int(torch.argmax(work).item())
        score = float(work.reshape(-1)[flat_index].item())
        if not math.isfinite(score) or score < float(threshold):
            break
        y, x = divmod(flat_index, w)
        peaks.append(
            Peak(
                rank=rank,
                x_pixel=x,
                y_pixel=y,
                x_angstrom=float(x * pixel_size_angstrom),
                y_angstrom=float(y * pixel_size_angstrom),
                score=score,
                mip_zscore=float(mip_zscore[y, x].item()),
                phi_deg=float(parameter_maps["phi"][y, x].item()),
                theta_deg=float(parameter_maps["theta"][y, x].item()),
                psi_deg=float(parameter_maps["psi"][y, x].item()),
                defocus_offset_angstrom=float(parameter_maps["defocus"][y, x].item()),
                pixel_size_offset_angstrom=float(parameter_maps["pixel_size"][y, x].item()),
            )
        )
        if radius_sq <= 0:
            work[y, x] = -torch.inf
            continue
        y_mask = (yy_all - y).square()[:, None]
        x_mask = (xx_all - x).square()[None, :]
        work[(y_mask + x_mask) <= radius_sq] = -torch.inf
    return peaks
