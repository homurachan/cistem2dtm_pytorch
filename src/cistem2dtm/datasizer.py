from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import torch

from .config import SearchConfig
from .fft import full_fft_resize_real, is_factorable, next_power_of_two, rfft2_cistem, rfft_sum_of_squares
from .image_ops import center_pad_crop_nd, center_pad_crop_with_mode, pad_to_square

Tensor = torch.Tensor


def _f32(value: float | int | np.floating) -> np.float32:
    """Round a scalar operation to the C++ ``float`` precision used by cisTEM."""
    return np.float32(value)


@dataclass(slots=True)
class SizingPlan:
    original_shape: tuple[int, int]
    template_shape: tuple[int, int, int]
    requested_high_resolution_limit_angstrom: float
    high_resolution_limit_angstrom: float
    target_binning_factor: float
    realized_binning_factor: float
    binning_error_angstrom: float
    bin_offset_2d: int
    max_search_size_applied: bool
    search_pixel_size_angstrom: float
    pre_scaling_shape: tuple[int, int]
    cropped_shape: tuple[int, int]
    search_shape: tuple[int, int]
    output_shape_without_rescaling: tuple[int, int]
    output_shape_with_rescaling: tuple[int, int]
    template_pre_scaling_shape: tuple[int, int, int]
    template_cropped_shape: tuple[int, int, int]
    template_search_shape: tuple[int, int, int]
    resampling_is_needed: bool
    resizing_is_needed: bool
    rotated_by_90: bool
    first_pre_padding: tuple[int, int]
    first_post_padding: tuple[int, int]
    fft_pre_padding: tuple[int, int]
    fft_post_padding: tuple[int, int]
    pre_padding: tuple[int, int]
    post_padding: tuple[int, int]
    valid_lower_xy: tuple[int, int]
    valid_upper_xy: tuple[int, int]
    roi_shape: tuple[int, int]
    number_of_valid_search_pixels: int
    statistics_roi_mode: str
    statistics_valid_lower_xy: tuple[int, int]
    statistics_valid_upper_xy: tuple[int, int]
    statistics_roi_shape: tuple[int, int]
    number_of_statistics_pixels: int
    threshold_pixel_mode: str
    number_of_threshold_pixels: int
    number_of_pixels_for_normalization: int

    @property
    def full_binning_factor(self) -> float:
        return self.realized_binning_factor

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PostSearchResult:
    maps: dict[str, Tensor]
    valid_mask: Tensor


class DataSizer:
    """cisTEM ``TemplateMatchingDataSizer`` translated to PyTorch.

    The sizing arithmetic intentionally uses float32 and C++-style integer
    conversion.  That matters for both the realized pixel size and the exact
    dimensions selected by the target source commit.
    """

    STANDARD_FACTOR_LIMITS = (2, 3, 5, 7, 9, 13)
    MAX_2D_SEARCH_DIMENSION = 4096
    ACCEPTABLE_PIXEL_SIZE_ERROR_ANGSTROM = 0.00005
    POST_SEARCH_MASK_EDGE_PIXELS = 7.0
    NN_SENTINEL = -torch.finfo(torch.float32).max

    def __init__(
        self,
        image_shape: tuple[int, int],
        template_shape: tuple[int, int, int],
        search: SearchConfig,
        *,
        image_fft_size_mode: str | None = None,
    ) -> None:
        self.search = search
        self.image_fft_size_mode = str(image_fft_size_mode or search.fft_size_mode)
        if self.image_fft_size_mode not in {"cistem-standard", "power2", "exact"}:
            raise ValueError(f"unknown image FFT size mode: {self.image_fft_size_mode}")
        self.image_shape = tuple(int(v) for v in image_shape)
        self.template_shape = tuple(int(v) for v in template_shape)
        if len(self.image_shape) != 2:
            raise ValueError("image_shape must be (height, width)")
        if len(self.template_shape) != 3:
            raise ValueError("template_shape must be (depth, height, width)")
        if len(set(self.template_shape)) != 1:
            raise ValueError("cisTEM match_template expects a cubic template volume")
        self.plan = self._build_plan()

    @staticmethod
    def _get_binned_size(input_size: float | int, wanted_binning_factor: float) -> int:
        # C++: int(input_size / wanted_binning_factor + 0.5f)
        value = _f32(_f32(input_size) / _f32(wanted_binning_factor) + _f32(0.5))
        return int(value)

    @classmethod
    def _realized_binning_factor(cls, wanted: float, input_size: int) -> float:
        wanted_binned_size = cls._get_binned_size(input_size, wanted)
        wanted_binned_size = max(2, wanted_binned_size)
        if wanted_binned_size % 2 == 1:
            wanted_binned_size += 1
        return float(_f32(_f32(input_size) / _f32(wanted_binned_size)))

    @staticmethod
    def _realized_high_resolution_limit_for_size(
        pixel_size: float,
        input_size: int,
        wanted_size: int,
    ) -> float:
        wanted = int(wanted_size)
        if wanted % 2 == 1:
            wanted += 1
        factor = _f32(_f32(input_size) / _f32(wanted))
        return float(_f32(_f32(2.0) * _f32(pixel_size) * factor))

    @staticmethod
    def _split_padding(total: int, original_size: int) -> tuple[int, int]:
        """Return (pre, post), preserving cisTEM's odd/even centre rule."""
        pre = int(total) // 2
        post = int(total) // 2
        if original_size % 2 == 0:
            post += int(total) % 2
        else:
            pre += int(total) % 2
        return pre, post

    @staticmethod
    def _allowed_factors_for_limit(limit: int) -> tuple[int, ...]:
        # ReturnClosestFactorizedUpper is called with progressively less strict
        # factor limits.  This reproduces the intended standard-FFT sequence.
        primes = (2, 3, 5, 7, 11, 13)
        return tuple(p for p in primes if p <= int(limit))

    @classmethod
    def _closest_factorized_upper_for_limit(cls, value: int, factor_limit: int) -> int:
        value = max(2, int(value))
        if factor_limit == 2:
            candidate = next_power_of_two(value)
        else:
            factors = cls._allowed_factors_for_limit(factor_limit)
            candidate = value
            # Source requests an even FFT dimension.
            if candidate % 2:
                candidate += 1
            while not is_factorable(candidate, factors):
                candidate += 2
        return int(candidate)

    def _nice_size(self, size: int) -> int:
        mode = self.image_fft_size_mode
        if mode == "exact":
            return int(size)
        if mode == "power2":
            return next_power_of_two(size)
        for factor_limit in self.STANDARD_FACTOR_LIMITS:
            candidate = self._closest_factorized_upper_for_limit(size, factor_limit)
            if float(candidate - size) < float(size) * 0.1:
                return candidate
        # Defensive fallback; the source's final factor limit normally succeeds.
        return self._closest_factorized_upper_for_limit(size, 13)

    def _template_nice_size(self, size: int) -> int:
        if self.search.fft_size_mode == "power2":
            return next_power_of_two(size)
        if self.search.fft_size_mode == "exact":
            return int(size)
        return self._closest_factorized_upper_for_limit(size, 5)

    def _scan_realized_binning(
        self,
        target_binning: float,
        max_square_size: int,
    ) -> tuple[float, int, int, float]:
        """Translate ``TemplateMatchingDataSizer::GetFFTSize`` bin scan.

        Returns (closest factor, cropped size, final pre-scaling offset,
        smallest factor error).  A high safety bound is retained only to turn a
        pathological source-style infinite scan into a useful Python error.
        """
        closest_factor = 1.0
        cropped_size = max_square_size
        final_offset = 0
        smallest_error = abs(float(target_binning) - 1.0)
        n_tries = 0
        bin_scan = 1
        safety = 0
        while n_tries < 2:
            current_factor = self._realized_binning_factor(target_binning, max_square_size)
            current_size = self._get_binned_size(max_square_size, current_factor)
            closest_size = current_size
            closest_factor = current_factor
            smallest_error = abs(float(_f32(target_binning)) - float(_f32(closest_factor)))
            offset = 0
            while current_size <= self.MAX_2D_SEARCH_DIMENSION:
                if smallest_error * float(_f32(self.search.pixel_size_angstrom)) < self.ACCEPTABLE_PIXEL_SIZE_ERROR_ANGSTROM:
                    break
                offset += bin_scan
                if max_square_size + offset < 2:
                    raise ValueError("cisTEM binning scan produced a non-positive pre-scaling size")
                current_factor = self._realized_binning_factor(
                    target_binning,
                    max_square_size + offset,
                )
                current_size = self._get_binned_size(max_square_size + offset, current_factor)
                current_error = abs(float(_f32(current_factor)) - float(_f32(target_binning)))
                if current_error < smallest_error:
                    closest_size = current_size
                    closest_factor = current_factor
                    smallest_error = current_error
                safety += 1
                if safety > 200_000:
                    raise RuntimeError(
                        "cisTEM-compatible binning scan exceeded its safety bound; "
                        "please report the input shape, pixel size, and resolution limit"
                    )
            final_offset = offset
            cropped_size = closest_size
            n_tries += 1
            if cropped_size <= self.image_shape[1] and cropped_size <= self.image_shape[0]:
                n_tries += 1
            else:
                bin_scan = -1
        return float(_f32(closest_factor)), int(cropped_size), int(final_offset), float(smallest_error)

    def _build_plan(self) -> SizingPlan:
        h, w = self.image_shape
        pixel_size = float(_f32(self.search.pixel_size_angstrom))
        requested_high_res = float(_f32(self.search.high_resolution_limit_angstrom))
        high_res = max(requested_high_res, float(_f32(_f32(2.0) * _f32(pixel_size))))
        max_search_applied = False
        if h > self.search.max_search_size or w > self.search.max_search_size:
            high_limit_x = self._realized_high_resolution_limit_for_size(
                pixel_size, w, self.search.max_search_size
            )
            high_limit_y = self._realized_high_resolution_limit_for_size(
                pixel_size, h, self.search.max_search_size
            )
            high_res = max(high_limit_x, high_limit_y)
            max_search_applied = True

        high_res_f32 = _f32(high_res)
        nyquist_f32 = _f32(_f32(2.0) * _f32(pixel_size))
        # FloatsAreAlmostTheSame is only intended to catch the Nyquist equality.
        resample = not math.isclose(
            float(high_res_f32), float(nyquist_f32), rel_tol=1.0e-6, abs_tol=1.0e-6
        )
        max_square = max(h, w)
        target_binning = float(
            _f32(_f32(high_res_f32 / _f32(pixel_size)) / _f32(2.0))
        )

        if resample:
            realized, cropped_n, bin_offset, factor_error = self._scan_realized_binning(
                target_binning, max_square
            )
            pre_n = max_square + bin_offset
            pre_scaling_shape = (pre_n, pre_n)
            cropped_shape = (cropped_n, cropped_n)
        else:
            realized = 1.0
            factor_error = abs(target_binning - 1.0)
            bin_offset = 0
            pre_scaling_shape = (h, w)
            cropped_shape = (h, w)

        search_shape = (
            self._nice_size(cropped_shape[0]),
            self._nice_size(cropped_shape[1]),
        )
        resizing = (not resample) and search_shape != self.image_shape

        # First padding: original -> square pre-scaling box, or original -> final FFT box.
        if resample:
            total_x_first = pre_scaling_shape[1] - w
            total_y_first = pre_scaling_shape[0] - h
        else:
            total_x_first = search_shape[1] - cropped_shape[1]
            total_y_first = search_shape[0] - cropped_shape[0]
        first_pre_x, first_post_x = self._split_padding(total_x_first, w)
        first_pre_y, first_post_y = self._split_padding(total_y_first, h)

        if resample:
            final_pre_x = int(math.ceil(float(first_pre_x) / realized))
            final_post_x = int(math.ceil(float(first_post_x) / realized))
            final_pre_y = int(math.ceil(float(first_pre_y) / realized))
            final_post_y = int(math.ceil(float(first_post_y) / realized))
            fft_pre_x = (search_shape[1] - cropped_shape[1]) // 2
            fft_post_x = (search_shape[1] - cropped_shape[1]) // 2
            fft_pre_y = (search_shape[0] - cropped_shape[0]) // 2
            fft_post_y = (search_shape[0] - cropped_shape[0]) // 2
            final_pre_x += fft_pre_x
            final_post_x += fft_post_x
            final_pre_y += fft_pre_y
            final_post_y += fft_post_y
        else:
            fft_pre_x = first_pre_x
            fft_post_x = first_post_x
            fft_pre_y = first_pre_y
            fft_post_y = first_post_y
            final_pre_x, final_post_x = first_pre_x, first_post_x
            final_pre_y, final_post_y = first_pre_y, first_post_y

        lower_x, lower_y = final_pre_x, final_pre_y
        upper_x = search_shape[1] - 1 - final_post_x
        upper_y = search_shape[0] - 1 - final_post_y
        roi_w = upper_x - lower_x + 1
        roi_h = upper_y - lower_y + 1
        if roi_h <= 0 or roi_w <= 0:
            raise ValueError("sizing produced an empty valid search area")

        statistics_mode = self.search.statistics_roi_mode
        if statistics_mode == "full":
            stats_lower_x, stats_lower_y = 0, 0
            stats_upper_x = search_shape[1] - 1
            stats_upper_y = search_shape[0] - 1
        elif statistics_mode == "source":
            stats_lower_x, stats_lower_y = lower_x, lower_y
            stats_upper_x, stats_upper_y = upper_x, upper_y
        elif statistics_mode == "precompiled":
            if self.search.statistics_roi_border_pixels is None:
                # The reference precompiled full-Nyquist binary behaves as if
                # the source-valid ROI is eroded once more by the FFT padding:
                # 320 -> 324 has source bounds [2,321], then statistics bounds
                # [4,319], i.e. 316 x 316.  Keep the lower/upper padding
                # asymmetric when the FFT box has odd extra pixels.
                stats_lower_x = lower_x + fft_pre_x
                stats_lower_y = lower_y + fft_pre_y
                stats_upper_x = upper_x - fft_post_x
                stats_upper_y = upper_y - fft_post_y
            else:
                border = int(self.search.statistics_roi_border_pixels)
                stats_lower_x = lower_x + border
                stats_lower_y = lower_y + border
                stats_upper_x = upper_x - border
                stats_upper_y = upper_y - border
        else:  # protected by SearchConfig.validate, retained for direct construction
            raise ValueError(f"unknown statistics_roi_mode: {statistics_mode}")

        stats_roi_w = stats_upper_x - stats_lower_x + 1
        stats_roi_h = stats_upper_y - stats_lower_y + 1
        if stats_roi_h <= 0 or stats_roi_w <= 0:
            raise ValueError(
                "statistics ROI is empty; reduce statistics_roi_border_pixels "
                "or choose statistics_roi_mode='source'"
            )

        threshold_mode = self.search.threshold_pixel_mode
        if threshold_mode == "source":
            threshold_pixels = roi_h * roi_w
        elif threshold_mode == "statistics":
            threshold_pixels = stats_roi_h * stats_roi_w
        elif threshold_mode == "full":
            threshold_pixels = search_shape[0] * search_shape[1]
        else:
            raise ValueError(f"unknown threshold_pixel_mode: {threshold_mode}")

        normalization_pixels = (
            search_shape[1] - final_post_x - final_pre_x
        ) * (
            search_shape[0] - final_post_y - final_pre_y
        )

        template_n = int(self.template_shape[-1])
        template_cropped_n = self._get_binned_size(template_n, realized)
        template_search_n = self._template_nice_size(template_cropped_n)
        search_pixel = float(_f32(_f32(pixel_size) * _f32(realized)))
        output_without = (roi_h, roi_w) if (resample or resizing) else search_shape
        rotated = bool(
            self.search.allow_rotation_for_speed
            and self.image_fft_size_mode != "power2"
            and (search_shape[1] & (search_shape[1] - 1)) != 0
            and (search_shape[0] & (search_shape[0] - 1)) == 0
        )
        if rotated:
            # The target source gates this behind ROTATEFORSPEED.  Keep the flag
            # and coordinate support, but it is inactive for square searches.
            search_shape = (search_shape[1], search_shape[0])
            lower_x, lower_y = lower_y, lower_x
            upper_x, upper_y = upper_y, upper_x
            roi_h, roi_w = roi_w, roi_h
            stats_lower_x, stats_lower_y = stats_lower_y, stats_lower_x
            stats_upper_x, stats_upper_y = stats_upper_y, stats_upper_x
            stats_roi_h, stats_roi_w = stats_roi_w, stats_roi_h

        return SizingPlan(
            original_shape=(h, w),
            template_shape=self.template_shape,
            requested_high_resolution_limit_angstrom=requested_high_res,
            high_resolution_limit_angstrom=float(high_res_f32),
            target_binning_factor=float(_f32(target_binning)),
            realized_binning_factor=float(_f32(realized)),
            binning_error_angstrom=float(_f32(factor_error * pixel_size)),
            bin_offset_2d=bin_offset,
            max_search_size_applied=max_search_applied,
            search_pixel_size_angstrom=search_pixel,
            pre_scaling_shape=pre_scaling_shape,
            cropped_shape=cropped_shape,
            search_shape=search_shape,
            output_shape_without_rescaling=output_without,
            output_shape_with_rescaling=(h, w),
            template_pre_scaling_shape=self.template_shape,
            template_cropped_shape=(template_cropped_n,) * 3,
            template_search_shape=(template_search_n,) * 3,
            resampling_is_needed=resample,
            resizing_is_needed=resizing,
            rotated_by_90=rotated,
            first_pre_padding=(first_pre_y, first_pre_x),
            first_post_padding=(first_post_y, first_post_x),
            fft_pre_padding=(fft_pre_y, fft_pre_x),
            fft_post_padding=(fft_post_y, fft_post_x),
            pre_padding=(final_pre_y, final_pre_x),
            post_padding=(final_post_y, final_post_x),
            valid_lower_xy=(lower_x, lower_y),
            valid_upper_xy=(upper_x, upper_y),
            roi_shape=(roi_h, roi_w),
            number_of_valid_search_pixels=int(roi_h * roi_w),
            statistics_roi_mode=statistics_mode,
            statistics_valid_lower_xy=(stats_lower_x, stats_lower_y),
            statistics_valid_upper_xy=(stats_upper_x, stats_upper_y),
            statistics_roi_shape=(stats_roi_h, stats_roi_w),
            number_of_statistics_pixels=int(stats_roi_h * stats_roi_w),
            threshold_pixel_mode=threshold_mode,
            number_of_threshold_pixels=int(threshold_pixels),
            number_of_pixels_for_normalization=int(normalization_pixels),
        )

    @staticmethod
    def _physical_center(shape: tuple[int, int]) -> tuple[int, int]:
        return int(shape[0]) // 2, int(shape[1]) // 2

    def valid_mask(self, *, device: torch.device) -> Tensor:
        """Pixels eligible to win the MIP and orientation parameter maps."""
        h, w = self.plan.search_shape
        lower_x, lower_y = self.plan.valid_lower_xy
        upper_x, upper_y = self.plan.valid_upper_xy
        mask = torch.zeros((h, w), device=device, dtype=torch.bool)
        mask[lower_y : upper_y + 1, lower_x : upper_x + 1] = True
        return mask

    def statistics_mask(self, *, device: torch.device) -> Tensor:
        """Histogram sampling ROI retained for precompiled-binary diagnostics.

        Correlation moments intentionally do *not* use this mask in v0.3.3;
        they follow :meth:`valid_mask` and the final post-search cosine mask,
        matching cisTEM's ResizeImage_postSearch/CalcGlobalCCCScalingFactor
        path and avoiding a locally unscaled border in scaled-MIP.
        """
        h, w = self.plan.search_shape
        lower_x, lower_y = self.plan.statistics_valid_lower_xy
        upper_x, upper_y = self.plan.statistics_valid_upper_xy
        mask = torch.zeros((h, w), device=device, dtype=torch.bool)
        mask[lower_y : upper_y + 1, lower_x : upper_x + 1] = True
        return mask

    def resize_image_pre_search(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Apply the target source's real/Fourier sizing sequence."""
        if tuple(image.shape) != self.image_shape:
            raise ValueError(f"input image has shape {tuple(image.shape)}; expected {self.image_shape}")
        generator = torch.Generator(device=image.device)
        generator.manual_seed(int(self.search.random_seed))
        plan = self.plan

        if plan.resampling_is_needed:
            square, _ = pad_to_square(
                image,
                plan.pre_scaling_shape[0],
                mode=self.search.padding_mode,
                generator=generator,
            )
            resized = full_fft_resize_real(square, plan.cropped_shape)
            # Source zeroes DC and normalizes the Fourier-resized intermediate
            # before transforming it back to real space.
            f = rfft2_cistem(resized)
            f = f.clone()
            f[0, 0] = 0
            energy = rfft_sum_of_squares(f, resized.shape[-1]).clamp_min(1.0e-12)
            resized = torch.fft.irfft2(f / torch.sqrt(energy), s=resized.shape, norm="forward")
            final, _ = center_pad_crop_with_mode(
                resized,
                plan.search_shape,
                mode=self.search.padding_mode,
                generator=generator,
            )
        elif plan.resizing_is_needed:
            # This is the important full-Nyquist 320 -> 324 path.  v0.2.0
            # hard-coded zero padding here and therefore ignored padding_mode.
            final, _ = center_pad_crop_with_mode(
                image,
                plan.search_shape,
                mode=self.search.padding_mode,
                generator=generator,
            )
        else:
            final = image.clone()

        if plan.rotated_by_90:
            final = torch.rot90(final, k=1, dims=(-2, -1))
        return final, self.valid_mask(device=image.device)

    def _crop_valid_roi(self, image: Tensor) -> Tensor:
        lower_x, lower_y = self.plan.valid_lower_xy
        upper_x, upper_y = self.plan.valid_upper_xy
        return image[lower_y : upper_y + 1, lower_x : upper_x + 1]

    @staticmethod
    def _symmetric_index(coords: Tensor, size: int) -> Tensor:
        """Repeated-edge symmetric extension used by ClipIntoWithReplicativePadding."""
        if size <= 1:
            return torch.zeros_like(coords, dtype=torch.long)
        period = 2 * int(size)
        mod = torch.remainder(coords, period)
        return torch.where(mod < size, mod, period - 1 - mod).to(torch.long)

    @classmethod
    def _center_symmetric_resize(cls, image: Tensor, output_shape: tuple[int, int]) -> Tensor:
        """Center an image in a larger box with cisTEM-style symmetric padding."""
        out_h, out_w = (int(v) for v in output_shape)
        in_h, in_w = (int(v) for v in image.shape[-2:])
        in_cy, in_cx = in_h // 2, in_w // 2
        out_cy, out_cx = out_h // 2, out_w // 2
        y = torch.arange(out_h, device=image.device, dtype=torch.long) - out_cy + in_cy
        x = torch.arange(out_w, device=image.device, dtype=torch.long) - out_cx + in_cx
        y = cls._symmetric_index(y, in_h)
        x = cls._symmetric_index(x, in_w)
        return image.index_select(-2, y).index_select(-1, x)

    def _roi_replicative_search_box(self, image: Tensor) -> Tensor:
        roi = self._crop_valid_roi(image)
        return self._center_symmetric_resize(roi, self.plan.search_shape)

    def _post_search_valid_mask(
        self,
        *,
        output_shape: tuple[int, int],
        x_radius: float,
        y_radius: float,
        device: torch.device,
    ) -> Tensor:
        """Translate CosineRectangularMask(..., edge=7) then Binarise(0.9)."""
        h, w = output_shape
        cy, cx = self._physical_center(output_shape)
        yy = torch.abs(torch.arange(h, device=device, dtype=torch.float32) - float(cy))
        xx = torch.abs(torch.arange(w, device=device, dtype=torch.float32) - float(cx))
        edge_width = float(self.POST_SEARCH_MASK_EDGE_PIXELS)
        inner_y = max(float(y_radius) - 0.5 * edge_width, 0.0)
        inner_x = max(float(x_radius) - 0.5 * edge_width, 0.0)
        outer_y = inner_y + edge_width
        outer_x = inner_x + edge_width

        wy = torch.ones_like(yy)
        wx = torch.ones_like(xx)
        y_edge = (yy > inner_y) & (yy < outer_y)
        x_edge = (xx > inner_x) & (xx < outer_x)
        wy[y_edge] = (1.0 + torch.cos(math.pi * (yy[y_edge] - inner_y) / edge_width)) * 0.5
        wx[x_edge] = (1.0 + torch.cos(math.pi * (xx[x_edge] - inner_x) / edge_width)) * 0.5
        wy[yy >= outer_y] = 0.0
        wx[xx >= outer_x] = 0.0
        weights = wy[:, None] * wx[None, :]
        return weights >= 0.9

    @staticmethod
    def _shift_without_wrap(image: Tensor, dy: int, dx: int, fill_value: float) -> Tensor:
        out = torch.full_like(image, fill_value)
        h, w = image.shape[-2:]
        src_y0 = max(0, -dy)
        src_y1 = min(h, h - dy)
        src_x0 = max(0, -dx)
        src_x1 = min(w, w - dx)
        dst_y0 = max(0, dy)
        dst_y1 = dst_y0 + max(0, src_y1 - src_y0)
        dst_x0 = max(0, dx)
        dst_x1 = dst_x0 + max(0, src_x1 - src_x0)
        if src_y1 > src_y0 and src_x1 > src_x0:
            out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
        return out

    def _fill_nearest_neighbors(self, sparse: Tensor, valid_mask: Tensor) -> Tensor:
        """Fill sparse label maps from the original samples, without propagation."""
        sentinel = float(self.NN_SENTINEL)
        source = sparse
        output = torch.where(valid_mask, source, torch.zeros_like(source))
        missing = valid_mask & (source == sentinel)
        if not bool(missing.any()):
            return output

        # Source chooses an odd neighborhood large enough for twice the binning.
        radius = max(1, int(math.ceil(float(self.plan.full_binning_factor))))
        offsets: list[tuple[int, int, int, int]] = []
        order = 0
        # Same row is checked first from left to right.
        for dx in range(-radius, radius + 1):
            if dx != 0:
                offsets.append((dx * dx, order, 0, dx))
                order += 1
        for yoff in range(1, radius + 1):
            for dy in (-yoff, yoff):
                for dx in range(-radius, radius + 1):
                    offsets.append((dy * dy + dx * dx, order, dy, dx))
                    order += 1
        offsets.sort(key=lambda item: (item[0], item[1]))

        for _, _, dy, dx in offsets:
            candidate = self._shift_without_wrap(source, -dy, -dx, sentinel)
            take = missing & (candidate != sentinel)
            if bool(take.any()):
                output[take] = candidate[take]
                missing[take] = False
                if not bool(missing.any()):
                    break
        if bool(missing.any()):
            # This should not occur for supported binning factors.  Iterative
            # expansion is a deterministic safety net rather than returning a
            # sentinel in an otherwise valid output pixel.
            frontier = output.clone()
            known = valid_mask & (source != sentinel)
            for _ in range(max(output.shape)):
                if not bool(missing.any()):
                    break
                for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    cand = self._shift_without_wrap(frontier, -dy, -dx, 0.0)
                    cand_known = self._shift_without_wrap(known.to(torch.float32), -dy, -dx, 0.0) > 0.5
                    take = missing & cand_known
                    output[take] = cand[take]
                    known[take] = True
                    missing[take] = False
                frontier = output
        if bool(missing.any()):
            raise RuntimeError("nearest-neighbor post-search interpolation left unfilled valid pixels")
        return output

    def _scatter_label_to_original(self, image: Tensor) -> Tensor:
        plan = self.plan
        sentinel = float(self.NN_SENTINEL)
        out = torch.full(plan.original_shape, sentinel, device=image.device, dtype=image.dtype)
        lower_x, lower_y = plan.valid_lower_xy
        upper_x, upper_y = plan.valid_upper_xy
        search_cy, search_cx = self._physical_center(plan.search_shape)
        out_cy, out_cx = self._physical_center(plan.original_shape)
        ys = torch.arange(lower_y, upper_y + 1, device=image.device, dtype=torch.float32)
        xs = torch.arange(lower_x, upper_x + 1, device=image.device, dtype=torch.float32)
        factor = torch.tensor(float(plan.full_binning_factor), device=image.device, dtype=torch.float32)
        target_y = torch.trunc((ys - float(search_cy)) * factor).to(torch.long) + out_cy
        target_x = torch.trunc((xs - float(search_cx)) * factor).to(torch.long) + out_cx
        valid_y = (target_y >= 0) & (target_y < plan.original_shape[0])
        valid_x = (target_x >= 0) & (target_x < plan.original_shape[1])
        if not bool(valid_y.all()) or not bool(valid_x.all()):
            raise RuntimeError("cisTEM nearest-neighbor coordinate mapping produced an out-of-bounds pixel")
        source = image[lower_y : upper_y + 1, lower_x : upper_x + 1]
        yy, xx = torch.meshgrid(target_y, target_x, indexing="ij")
        out[yy.reshape(-1), xx.reshape(-1)] = source.reshape(-1)
        return out

    @staticmethod
    def _mean_fill_outside(image: Tensor, valid_mask: Tensor) -> Tensor:
        if not bool(valid_mask.any()):
            raise RuntimeError("post-search valid-area mask is empty")
        out = image.clone()
        mean = out[valid_mask].to(torch.float64).mean().to(out.dtype)
        out[~valid_mask] = mean
        return out

    def resize_all_results(
        self,
        maps: Mapping[str, Tensor],
        *,
        apply_result_rescaling: bool,
    ) -> PostSearchResult:
        """Translate ``ResizeImage_postSearch`` for all eight result images."""
        plan = self.plan
        label_names = {"phi", "theta", "psi", "defocus", "pixel_size", "winner_task_index"}
        continuous_names = {"mip", "correlation_sum", "correlation_sum_squares"}
        expected_names = label_names | continuous_names
        unknown = set(maps) - expected_names
        if unknown:
            raise KeyError(f"unknown post-search map(s): {sorted(unknown)}")
        working: dict[str, Tensor] = {}
        for name, value in maps.items():
            if tuple(value.shape) != plan.search_shape:
                raise ValueError(f"{name} has shape {tuple(value.shape)}; expected {plan.search_shape}")
            working[name] = value.clone()

        if plan.rotated_by_90:
            for name in working:
                working[name] = torch.rot90(working[name], k=-1, dims=(-2, -1))
            if "psi" in working:
                working["psi"] = torch.remainder(working["psi"] + 90.0, 360.0)
                working["psi"] = torch.where(working["psi"] == 0.0, torch.full_like(working["psi"], 360.0), working["psi"])

        if (plan.resampling_is_needed or plan.resizing_is_needed) and not apply_result_rescaling:
            cropped = {name: self._crop_valid_roi(value).clone() for name, value in working.items()}
            mask = torch.ones(plan.roi_shape, device=next(iter(cropped.values())).device, dtype=torch.bool)
            return PostSearchResult(cropped, mask)

        # Before Fourier upsampling, cisTEM replaces every padding region by a
        # symmetric extension of the valid ROI.
        if plan.resampling_is_needed or plan.resizing_is_needed:
            for name in working:
                working[name] = self._roi_replicative_search_box(working[name])

        if plan.resizing_is_needed:
            # Resize is a centered real-space clip/pad, not interpolation.
            for name in working:
                working[name] = center_pad_crop_nd(working[name], plan.original_shape, fill_value=0.0)
        elif plan.resampling_is_needed:
            source_for_labels = {name: working[name] for name in label_names if name in working}
            for name in continuous_names:
                if name not in working:
                    continue
                cropped = center_pad_crop_nd(working[name], plan.cropped_shape, fill_value=0.0)
                pre_scaled = full_fft_resize_real(cropped, plan.pre_scaling_shape)
                working[name] = center_pad_crop_nd(pre_scaled, plan.original_shape, fill_value=0.0)
            for name, value in source_for_labels.items():
                working[name] = self._scatter_label_to_original(value)

        output_shape = tuple(next(iter(working.values())).shape)
        search_cy, search_cx = self._physical_center(plan.search_shape)
        x_radius = float(search_cx - plan.pre_padding[1])
        y_radius = float(search_cy - plan.pre_padding[0])
        if plan.rotated_by_90:
            x_radius, y_radius = y_radius, x_radius
        if plan.resampling_is_needed:
            x_radius *= float(plan.full_binning_factor)
            y_radius *= float(plan.full_binning_factor)
        valid_mask = self._post_search_valid_mask(
            output_shape=output_shape,
            x_radius=x_radius,
            y_radius=y_radius,
            device=next(iter(working.values())).device,
        )

        if plan.resampling_is_needed:
            for name in label_names:
                if name in working:
                    working[name] = self._fill_nearest_neighbors(working[name], valid_mask)

        # cisTEM fills display maps with their mean outside the mask and zeros
        # the two accumulators so the later global statistics ignore that area.
        for name in ("mip", "phi", "theta", "psi", "defocus", "pixel_size"):
            if name in working:
                working[name] = self._mean_fill_outside(working[name], valid_mask)
        for name in ("correlation_sum", "correlation_sum_squares"):
            if name in working:
                working[name] = working[name].clone()
                working[name][~valid_mask] = 0.0
        if "winner_task_index" in working:
            winner = working["winner_task_index"].clone()
            winner[~valid_mask] = -1.0
            working["winner_task_index"] = winner

        return PostSearchResult(working, valid_mask)
