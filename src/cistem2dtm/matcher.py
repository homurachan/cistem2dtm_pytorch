from __future__ import annotations

import math
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .config import MatchConfig, TimingConfig, effective_image_fft_size_mode
from .constants import CISTEM_SOURCE_COMMIT, HISTOGRAM_BINS
from .correlation import (
    CorrelationPrecisionPlan,
    conjugate_multiply_inplace,
    prepare_image_fourier,
    resolve_correlation_precision,
    tensor_health,
)
from .ctf import CTF
from .datasizer import DataSizer
from .debug import DebugRecorder, DebugStop
from .fft import irfft2_cistem, real_dtype_from_name
from .filters import FilterPipeline
from .geometry import (
    EulerSearch,
    Orientation,
    cisTEM_auto_angular_step,
    cisTEM_auto_psi_step,
    cisTEM_rotation_matrix,
)
from .image_ops import (
    PixelSizeChangeResult,
    apply_fourier_weight_and_normalize,
    change_pixel_size_cistem,
    preprocess_first_pass,
    preprocess_second_pass,
)
from .io import (
    json_ready,
    read_search_image,
    read_template_volume,
    write_json,
    write_mrc,
    write_rows_tsv,
)
from .orientation_cache import OrientationTensorTable
from .projector import FourierProjector, ProjectionBatch
from .stack import CCFStackResult, CCFStackWriter
from .timing import PerformanceProfiler
from .statistics import (
    Peak,
    RawSearchResult,
    ScaledSearchResult,
    SearchAccumulators,
    expected_gaussian_survival,
    peak_pick,
    scale_search_statistics,
    theoretical_threshold,
)

Tensor = torch.Tensor


@dataclass(slots=True)
class PreparedData:
    image_original: Tensor
    template_original: Tensor
    image_first_whitened: Tensor
    image_search_real: Tensor
    image_search_fourier: Tensor
    valid_mask: Tensor
    histogram_mask: Tensor
    whitening_curve: Any
    data_sizer: DataSizer
    input_header_pixel_size_angstrom: float
    template_header_pixel_size_angstrom: float
    device: torch.device
    storage_dtype: torch.dtype
    geometry_dtype: torch.dtype
    correlation_precision: CorrelationPrecisionPlan
    center_phase_mode_effective: str
    correlation_multiply_backend_resolved: str
    requested_fft_size_mode: str
    effective_fft_size_mode: str
    debug: DebugRecorder | None = None
    profiler: PerformanceProfiler | None = None


@dataclass(slots=True)
class RawRunOutput:
    raw: RawSearchResult
    prepared: PreparedData
    orientations: list[Orientation]
    all_orientation_count: int
    selected_orientation_indices: list[int]
    defocus_offsets: list[float]
    pixel_size_offsets: list[float]
    ccf_stack: CCFStackResult
    elapsed_seconds: float
    pixel_size_resampling: list[dict[str, Any]] = field(default_factory=list)
    execution_devices: list[str] = field(default_factory=list)
    search_spacing: dict[str, Any] = field(default_factory=dict)
    profiler: PerformanceProfiler | None = None
    timing_report: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MatchResult:
    raw: RawSearchResult
    scaled: ScaledSearchResult
    peaks: list[Peak]
    threshold: float
    output_pixel_size_angstrom: float
    files: dict[str, Path]
    metadata: dict[str, Any]
    ccf_stack: CCFStackResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "output_pixel_size_angstrom": self.output_pixel_size_angstrom,
            "number_of_peaks": len(self.peaks),
            "files": {name: str(path) for name, path in self.files.items()},
            "metadata": json_ready(self.metadata),
            "ccf_stack": {
                "mode": self.ccf_stack.mode,
                "mrc_path": str(self.ccf_stack.mrc_path) if self.ccf_stack.mrc_path else None,
                "tensor_path": str(self.ccf_stack.tensor_path) if self.ccf_stack.tensor_path else None,
                "metadata_path": str(self.ccf_stack.metadata_path) if self.ccf_stack.metadata_path else None,
                "shard_paths": [str(path) for path in self.ccf_stack.shard_paths],
                "number_written": self.ccf_stack.number_written,
            },
        }


def resolve_device(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({requested}) but torch.cuda.is_available() is false")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; visible device count is {torch.cuda.device_count()}"
            )
    return device



def _resolved_timing_config(cfg: MatchConfig) -> TimingConfig:
    timing = TimingConfig(**asdict(cfg.timing))
    # Backward compatibility: the old --record-timing switch now activates the
    # low-overhead profiler instead of synchronizing and rewriting the debug
    # manifest for every orientation batch.
    if cfg.debug.record_timing and not timing.enabled:
        timing.enabled = True
        timing.mode = "sampled"
        if timing.output_dir is None and cfg.debug.debug_dir:
            timing.output_dir = cfg.debug.debug_dir
    return timing

def _symmetric_offsets(search_range: float, step: float) -> list[float]:
    if search_range <= 0 or step <= 0:
        return [0.0]
    half = int(float(search_range) / float(step) + 0.5)
    return [float(i) * float(step) for i in range(-half, half + 1)]


def _batch_slices(values: Sequence[Any], batch_size: int) -> Iterable[tuple[int, Sequence[Any]]]:
    for start in range(0, len(values), int(batch_size)):
        yield start, values[start : start + int(batch_size)]


def _orientation_rows(orientations: Sequence[Orientation]) -> list[dict[str, Any]]:
    return [
        {
            "orientation_index": o.orientation_index,
            "grid_index": o.grid_index,
            "phi_deg": o.phi_deg,
            "theta_deg": o.theta_deg,
            "psi_deg": o.psi_deg,
        }
        for o in orientations
    ]


def _clone_raw_with_resized_maps(
    raw: RawSearchResult,
    data_sizer: DataSizer,
    apply: bool,
) -> tuple[RawSearchResult, Tensor]:
    post = data_sizer.resize_all_results(
        {
            "mip": raw.mip,
            "phi": raw.phi,
            "theta": raw.theta,
            "psi": raw.psi,
            "defocus": raw.defocus,
            "pixel_size": raw.pixel_size,
            "correlation_sum": raw.correlation_sum,
            "correlation_sum_squares": raw.correlation_sum_squares,
            "winner_task_index": raw.winner_task_index.to(torch.float32),
        },
        apply_result_rescaling=apply,
    )
    resized = post.maps
    winner = torch.round(resized["winner_task_index"]).to(torch.int64)
    result = RawSearchResult(
        mip=resized["mip"],
        phi=resized["phi"],
        theta=resized["theta"],
        psi=resized["psi"],
        defocus=resized["defocus"],
        pixel_size=resized["pixel_size"],
        correlation_sum=resized["correlation_sum"],
        correlation_sum_squares=resized["correlation_sum_squares"],
        histogram=raw.histogram,
        number_searched=raw.number_searched,
        winner_task_index=winner,
    )
    return result, post.valid_mask


class TemplateMatcher:
    """Executable PyTorch implementation of cisTEM 2-D template matching.

    The scientific target is ``match_template.cpp`` at
    ``5bc5f8cd5f804d8b12771aed156dc928510b1e47``.  The class exposes preparation,
    raw search and finalization separately so that one-server multi-GPU workers
    can search disjoint orientation ranges and merge without any cisTEM socket or
    wxWidgets infrastructure.
    """

    def __init__(self, config: MatchConfig) -> None:
        self.config = config
        self.config.validate()

    def prepare(self, *, device_override: str | torch.device | None = None) -> PreparedData:
        cfg = self.config
        torch.set_num_threads(int(cfg.runtime.cpu_threads))
        if cfg.runtime.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        device = (
            device_override
            if isinstance(device_override, torch.device)
            else resolve_device(str(device_override or cfg.runtime.device))
        )
        requested_storage_dtype = real_dtype_from_name(cfg.runtime.dtype)
        # Several CPU kernels used by the compatibility path do not implement
        # float16/bfloat16. Keep the public dtype request, but promote CPU
        # execution to float32. CUDA still uses the requested storage dtype,
        # while FFT wrappers promote unsupported half transforms internally.
        storage_dtype = (
            torch.float32
            if device.type == "cpu" and requested_storage_dtype != torch.float32
            else requested_storage_dtype
        )
        geometry_dtype = torch.float32

        timing_cfg = _resolved_timing_config(cfg)
        profiler = PerformanceProfiler(
            timing_cfg,
            device=device,
            output_dir=cfg.output_dir,
            output_prefix=cfg.output_prefix,
            package_version="0.3.6",
        )
        profiler.set_metadata(
            input_mrc=str(cfg.input_mrc),
            template_mrc=str(cfg.template_mrc),
            requested_dtype=cfg.runtime.dtype,
            effective_storage_dtype=str(storage_dtype).removeprefix("torch."),
            correlation_precision_requested=cfg.runtime.correlation_precision,
            center_phase_mode=cfg.runtime.center_phase_mode,
            correlation_multiply_backend=cfg.runtime.correlation_multiply_backend,
            correlation_triton_block_size=int(cfg.runtime.correlation_triton_block_size),
            correlation_triton_num_warps=int(cfg.runtime.correlation_triton_num_warps),
            half_fft_shape_mode=cfg.runtime.half_fft_shape_mode,
            orientation_batch_size=int(cfg.runtime.orientation_batch_size),
            projector_backend=cfg.runtime.projector_backend,
            pixel_size_backend=cfg.runtime.pixel_size_backend,
            weighting_mode=cfg.weighting.mode,
        )
        prepare_start = time.perf_counter()

        with profiler.wall_section("prepare.read_input_mrc"):
            image_np, image_header_pixel = read_search_image(
                cfg.input_mrc,
                cfg.search.image_slice,
                fallback_pixel_size=cfg.search.pixel_size_angstrom,
            )
        with profiler.wall_section("prepare.read_template_mrc"):
            template_np, template_header_pixel = read_template_volume(
                cfg.template_mrc,
                fallback_pixel_size=cfg.search.pixel_size_angstrom,
            )
        with profiler.wall_section("prepare.host_to_device", synchronize_cuda=True):
            image = torch.from_numpy(image_np).to(device=device, dtype=storage_dtype)
            template = torch.from_numpy(template_np).to(device=device, dtype=storage_dtype)
        with profiler.wall_section("prepare.data_sizer"):
            # Float32 preserves the v0.3.3/cisTEM-standard sizing exactly.  An
            # explicitly requested mixed half correlation may ask DataSizer for
            # per-axis powers of two (e.g. 4096x3888 -> 4096x4096).
            if (
                cfg.runtime.correlation_precision == "mixed_float16"
                and device.type != "cuda"
                and cfg.runtime.mixed_precision_fallback_to_float32
            ):
                image_fft_size_mode = cfg.search.fft_size_mode
            else:
                image_fft_size_mode = effective_image_fft_size_mode(
                    cfg.search, cfg.runtime
                )
            data_sizer = DataSizer(
                tuple(image.shape),
                tuple(template.shape),
                cfg.search,
                image_fft_size_mode=image_fft_size_mode,
            )
            correlation_precision = resolve_correlation_precision(
                cfg.runtime.correlation_precision,
                device=device,
                fft_shape=data_sizer.plan.search_shape,
                half_fft_shape_mode=cfg.runtime.half_fft_shape_mode,
                fallback_to_float32=cfg.runtime.mixed_precision_fallback_to_float32,
            )
            # If a CUDA/runtime fallback occurred after a power-of-two sizing
            # request, rebuild the float32 plan so fallback truly matches v0.3.3.
            if (
                correlation_precision.effective == "float32"
                and image_fft_size_mode != cfg.search.fft_size_mode
            ):
                data_sizer = DataSizer(
                    tuple(image.shape),
                    tuple(template.shape),
                    cfg.search,
                    image_fft_size_mode=cfg.search.fft_size_mode,
                )

            if cfg.runtime.center_phase_mode == "auto":
                center_phase_mode_effective = (
                    "input"
                    if correlation_precision.effective == "mixed_float16"
                    else "projection"
                )
            else:
                center_phase_mode_effective = cfg.runtime.center_phase_mode
            # ``auto`` deliberately keeps the float32 strict path bitwise equal
            # to v0.3.3.  The accelerated Triton multiply is selected by auto
            # whenever input prephasing or mixed precision is active.
            if cfg.runtime.correlation_multiply_backend == "auto":
                correlation_multiply_backend_resolved = (
                    "torch"
                    if (
                        correlation_precision.effective == "float32"
                        and center_phase_mode_effective == "projection"
                    )
                    else "auto"
                )
            else:
                correlation_multiply_backend_resolved = (
                    cfg.runtime.correlation_multiply_backend
                )

        profiler.set_metadata(
            correlation_precision_effective=correlation_precision.effective,
            correlation_real_dtype=str(correlation_precision.real_dtype).removeprefix("torch."),
            correlation_complex_dtype=str(correlation_precision.complex_dtype).removeprefix("torch."),
            correlation_precision_fallback_reason=correlation_precision.fallback_reason,
            fft_size_mode_requested=cfg.search.fft_size_mode,
            fft_size_mode_effective=data_sizer.image_fft_size_mode,
            center_phase_mode_effective=center_phase_mode_effective,
            correlation_multiply_backend_resolved=(
                correlation_multiply_backend_resolved
            ),
            input_shape=list(image.shape),
            template_shape=list(template.shape),
            search_shape=list(data_sizer.plan.search_shape),
            template_search_shape=list(data_sizer.plan.template_search_shape),
            ccf_public_dtype="float32",
            ccf_pixels_per_orientation=int(data_sizer.plan.search_shape[0] * data_sizer.plan.search_shape[1]),
            estimated_ccf_batch_work_bytes=int(
                cfg.runtime.orientation_batch_size
                * data_sizer.plan.search_shape[0]
                * data_sizer.plan.search_shape[1]
                * (2 if correlation_precision.real_dtype == torch.float16 else 4)
            ),
            estimated_ccf_batch_float32_bytes=int(
                cfg.runtime.orientation_batch_size
                * data_sizer.plan.search_shape[0]
                * data_sizer.plan.search_shape[1]
                * 4
            ),
            estimated_rfft_batch_work_bytes=int(
                cfg.runtime.orientation_batch_size
                * data_sizer.plan.search_shape[0]
                * (data_sizer.plan.search_shape[1] // 2 + 1)
                * (4 if correlation_precision.real_dtype == torch.float16 else 8)
            ),
            estimated_rfft_batch_complex64_bytes=int(
                cfg.runtime.orientation_batch_size
                * data_sizer.plan.search_shape[0]
                * (data_sizer.plan.search_shape[1] // 2 + 1)
                * 8
            ),
            search_pixel_size_angstrom=float(data_sizer.plan.search_pixel_size_angstrom),
            high_resolution_limit_angstrom=float(data_sizer.plan.high_resolution_limit_angstrom),
            max_search_size_applied=bool(data_sizer.plan.max_search_size_applied),
        )

        debug = DebugRecorder(cfg.debug, cfg.search.pixel_size_angstrom) if cfg.debug.enabled else None
        if debug:
            debug.add_timing("stage_01_input_and_sizing", time.perf_counter() - prepare_start)
            debug.save(1, "input_image", image)
            debug.save(1, "input_template", template)
            debug.save(1, "sizing_plan", data_sizer.plan.to_dict())
            debug.save(
                1,
                "input_headers",
                {
                    "image_header_pixel_size_angstrom": image_header_pixel,
                    "template_header_pixel_size_angstrom": template_header_pixel,
                    "configured_pixel_size_angstrom": cfg.search.pixel_size_angstrom,
                },
            )
            debug.finish_stage(1)

        stage_start = time.perf_counter()
        with profiler.wall_section("prepare.first_whitening", synchronize_cuda=True):
            first_whitened, whitening, cleaned, outlier_mask = preprocess_first_pass(image)
        if debug:
            debug.add_timing("stage_02_first_whitening", time.perf_counter() - stage_start)
            debug.save(2, "outlier_replaced_image", cleaned)
            debug.save(2, "outlier_mask", outlier_mask.to(torch.float32))
            debug.save(2, "whitening_curve_x", whitening.x)
            debug.save(2, "whitening_curve_y", whitening.y)
            debug.save(2, "first_whitened_image", first_whitened)
            debug.save(
                2,
                "first_whitened_statistics",
                {
                    "mean": float(first_whitened.mean().item()),
                    "std_unbiased_false": float(first_whitened.std(unbiased=False).item()),
                    "outlier_count": int(outlier_mask.sum().item()),
                },
            )
            debug.finish_stage(2)

        stage_start = time.perf_counter()
        with profiler.wall_section("prepare.resize_pre_search", synchronize_cuda=True):
            search_real, valid_mask_from_resize = data_sizer.resize_image_pre_search(first_whitened)
        with profiler.wall_section("prepare.build_masks", synchronize_cuda=True):
            valid_mask = data_sizer.valid_mask(device=device)
            histogram_mask = data_sizer.statistics_mask(device=device)
        with profiler.wall_section("prepare.second_normalization_and_input_fft", synchronize_cuda=True):
            input_fourier = preprocess_second_pass(
                search_real,
                data_sizer.plan.number_of_pixels_for_normalization,
            )
        if debug:
            debug.add_timing("stage_03_resize_and_second_normalization", time.perf_counter() - stage_start)
            debug.save(3, "search_real", search_real)
            debug.save(3, "valid_mask_analytic", valid_mask.to(torch.float32))
            debug.save(3, "histogram_mask", histogram_mask.to(torch.float32))
            # Backward-compatible debug checkpoint name from v0.3.x.
            debug.save(3, "statistics_mask", histogram_mask.to(torch.float32))
            debug.save(3, "valid_mask_resampled", valid_mask_from_resize.to(torch.float32))
            debug.save(3, "search_fourier_real", input_fourier.real)
            debug.save(3, "search_fourier_imag", input_fourier.imag)
            debug.save(3, "search_fourier_abs", input_fourier.abs())
            debug.finish_stage(3)

        if profiler.is_cuda:
            torch.cuda.synchronize(device)
        profiler.record_wall("prepare.total", time.perf_counter() - prepare_start)

        return PreparedData(
            image_original=image,
            template_original=template,
            image_first_whitened=first_whitened,
            image_search_real=search_real,
            image_search_fourier=input_fourier,
            valid_mask=valid_mask,
            histogram_mask=histogram_mask,
            whitening_curve=whitening,
            data_sizer=data_sizer,
            input_header_pixel_size_angstrom=image_header_pixel,
            template_header_pixel_size_angstrom=template_header_pixel,
            device=device,
            storage_dtype=storage_dtype,
            geometry_dtype=geometry_dtype,
            correlation_precision=correlation_precision,
            center_phase_mode_effective=center_phase_mode_effective,
            correlation_multiply_backend_resolved=(
                correlation_multiply_backend_resolved
            ),
            requested_fft_size_mode=cfg.search.fft_size_mode,
            effective_fft_size_mode=data_sizer.image_fft_size_mode,
            debug=debug,
            profiler=profiler,
        )

    def build_orientations(self, prepared: PreparedData) -> tuple[list[Orientation], dict[str, Any]]:
        cfg = self.config
        single = cfg.debug.single_orientation
        if single is not None:
            orientation = Orientation(
                phi_deg=float(single[0]),
                theta_deg=float(single[1]),
                psi_deg=float(single[2]),
                grid_index=0,
                orientation_index=0,
            )
            return [orientation], {
                "mask_radius_angstrom": cfg.search.particle_radius_angstrom or 200.0,
                "angular_step_deg": 0.0,
                "psi_step_deg": 0.0,
                "number_of_out_of_plane_positions": 1,
                "last_out_of_plane_grid_index": 0,
                "number_of_psi_positions": 1,
                "number_of_orientations": 1,
                "single_orientation_debug_mode": True,
            }

        radius = cfg.search.particle_radius_angstrom if cfg.search.particle_radius_angstrom >= 1.0 else 200.0
        angular = cfg.search.angular_step_deg
        if angular <= 0:
            angular = cisTEM_auto_angular_step(
                prepared.data_sizer.plan.high_resolution_limit_angstrom,
                radius,
            )
        psi_step = cfg.search.in_plane_step_deg
        if psi_step is None or psi_step <= 0:
            psi_step = cisTEM_auto_psi_step(
                prepared.data_sizer.plan.search_pixel_size_angstrom,
                radius,
            )
        search = EulerSearch(
            symmetry=cfg.search.symmetry,
            angular_step_deg=float(angular),
            psi_step_deg=float(psi_step),
            preserve_psi_360_duplicate=cfg.search.preserve_psi_360_duplicate,
            extend_cyclic_theta_to_180=True,
        )
        out_of_plane_count = search.number_of_out_of_plane_positions
        psi_count = search.number_of_psi_positions
        orientations = search.orientations()
        return orientations, {
            "mask_radius_angstrom": float(radius),
            "angular_step_deg": float(angular),
            "psi_step_deg": float(psi_step),
            "number_of_out_of_plane_positions": int(out_of_plane_count),
            "last_out_of_plane_grid_index": int(out_of_plane_count - 1),
            "number_of_psi_positions": int(psi_count),
            "number_of_orientations": int(len(orientations)),
            "single_orientation_debug_mode": False,
        }

    def run_raw(
        self,
        *,
        device_override: str | torch.device | None = None,
        orientation_indices: Sequence[int] | None = None,
        stack_path_override: str | Path | None = None,
    ) -> RawRunOutput:
        start_time = time.perf_counter()
        cfg = self.config
        prepared = self.prepare(device_override=device_override)
        device = prepared.device
        profiler = prepared.profiler
        assert profiler is not None

        orientation_start = time.perf_counter()
        with profiler.wall_section("search.euler_generation"):
            orientations_all, spacing = self.build_orientations(prepared)
        if prepared.debug:
            prepared.debug.add_timing("stage_04_euler_grid_and_rotations", time.perf_counter() - orientation_start)
        total_orientation_count = len(orientations_all)
        if total_orientation_count == 0:
            raise RuntimeError("Euler search generated no orientations")

        if orientation_indices is None:
            first = min(cfg.runtime.first_orientation, total_orientation_count)
            last_exclusive = (
                total_orientation_count
                if cfg.runtime.last_orientation is None
                else min(cfg.runtime.last_orientation + 1, total_orientation_count)
            )
            selected_indices = list(range(first, last_exclusive))
        else:
            selected_indices = [int(v) for v in orientation_indices]
            if any(v < 0 or v >= total_orientation_count for v in selected_indices):
                raise IndexError("an orientation index is outside the generated Euler grid")
        selected = [orientations_all[i] for i in selected_indices]
        if not selected:
            raise ValueError("selected orientation range is empty")

        debug = prepared.debug
        if debug:
            debug.save(4, "search_spacing", spacing)
            debug.save(4, "all_orientation_count", total_orientation_count)
            debug.save(4, "selected_orientations", _orientation_rows(selected))
            first_angles = selected[: min(len(selected), 32)]
            matrices = cisTEM_rotation_matrix(
                torch.tensor([o.phi_deg for o in first_angles], device=device),
                torch.tensor([o.theta_deg for o in first_angles], device=device),
                torch.tensor([o.psi_deg for o in first_angles], device=device),
                device=device,
                dtype=prepared.geometry_dtype,
            )
            debug.save(4, "first_rotation_matrices", matrices)
            debug.finish_stage(4)

        defocus_offsets = _symmetric_offsets(
            cfg.search.defocus_search_range_angstrom,
            cfg.search.defocus_step_angstrom,
        )
        pixel_offsets = _symmetric_offsets(
            cfg.search.pixel_size_search_range_angstrom,
            cfg.search.pixel_size_step_angstrom,
        )
        total_stack_images = len(selected) * len(defocus_offsets) * len(pixel_offsets)
        batches_per_condition = math.ceil(len(selected) / int(cfg.runtime.orientation_batch_size))
        total_batches = batches_per_condition * len(defocus_offsets) * len(pixel_offsets)
        profiler.set_metadata(
            all_orientation_count=int(total_orientation_count),
            selected_orientation_count=int(len(selected)),
            defocus_position_count=int(len(defocus_offsets)),
            pixel_size_position_count=int(len(pixel_offsets)),
            total_search_tasks=int(total_stack_images),
            total_orientation_batches=int(total_batches),
        )

        # Build angle/index tensors once.  Rotation matrices are also cached by
        # default (~57 MiB for 1.6 M orientations), removing all per-batch angle
        # tensor construction and trigonometric kernels.
        with profiler.wall_section("search.orientation_tensor_cache", synchronize_cuda=True):
            orientation_table = OrientationTensorTable(
                orientations_all,
                device=device,
                dtype=prepared.geometry_dtype,
                precompute_rotation_matrices=(
                    cfg.runtime.cache_orientation_tensors
                    and cfg.runtime.precompute_rotation_matrices
                ),
                precompute_chunk_size=cfg.runtime.orientation_precompute_chunk_size,
            )
            winner_metadata = orientation_table.winner_metadata(
                defocus_offsets,
                pixel_offsets,
            )

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if stack_path_override is not None:
            stack_path = Path(stack_path_override)
        elif cfg.runtime.ccf_stack_path:
            stack_path = Path(cfg.runtime.ccf_stack_path)
        else:
            suffix = ".mrc" if cfg.runtime.ccf_stack_mode == "mrc" else ".pt"
            stack_path = output_dir / f"{cfg.output_prefix}_ccf_stack{suffix}"

        with profiler.wall_section("search.stack_writer_init"):
            writer = CCFStackWriter(
                cfg.runtime.ccf_stack_mode,
                total_images=total_stack_images,
                image_shape=prepared.data_sizer.plan.search_shape,
                pixel_size_angstrom=prepared.data_sizer.plan.search_pixel_size_angstrom,
                path=stack_path if cfg.runtime.ccf_stack_mode != "none" else None,
            )
        with profiler.wall_section("search.accumulator_init", synchronize_cuda=True):
            accum = SearchAccumulators(
                prepared.data_sizer.plan.search_shape,
                device=device,
                dtype=torch.float32,
                mip_initial_value=cfg.search.mip_initial_value,
                valid_mask=prepared.valid_mask,
                statistics_mask=prepared.histogram_mask,
                histogram_mode=cfg.runtime.histogram_mode,
                histogram_backend=cfg.runtime.histogram_backend,
                histogram_sample_orientation_stride=cfg.runtime.histogram_sample_orientation_stride,
                histogram_sample_pixel_stride=cfg.runtime.histogram_sample_pixel_stride,
                winner_metadata=winner_metadata,
            )
        pipeline = FilterPipeline(
            prepared.whitening_curve,
            weighting=cfg.weighting,
            low_resolution_limit_angstrom=cfg.search.low_resolution_limit_angstrom,
            high_resolution_limit_angstrom=cfg.search.high_resolution_limit_angstrom,
        )
        first_debug_batch_done = False
        stack_index = 0
        batch_counter = 0
        completed_orientations = 0
        pixel_size_resampling: list[dict[str, Any]] = []
        raw_loop_start = time.perf_counter()
        profiler.start_search(
            total_batches=total_batches,
            total_orientations=total_stack_images,
        )

        try:
            for pixel_index, pixel_offset in enumerate(pixel_offsets):
                wanted_factor = (
                    prepared.data_sizer.plan.search_pixel_size_angstrom + float(pixel_offset)
                ) / prepared.data_sizer.plan.search_pixel_size_angstrom
                if wanted_factor <= 0:
                    raise ValueError(
                        f"pixel-size search offset {pixel_offset} produces a non-positive sampling factor"
                    )

                used_factor = float(wanted_factor)
                resample_info: PixelSizeChangeResult | None = None
                with profiler.wall_section("search.template_pixel_resample", synchronize_cuda=True):
                    if cfg.runtime.pixel_size_backend == "cistem_exact":
                        resample_info = change_pixel_size_cistem(
                            prepared.template_original,
                            wanted_factor,
                            tolerance=cfg.runtime.pixel_size_tolerance,
                            max_intermediate_dimension=cfg.runtime.max_pixel_resample_dimension,
                        )
                        template_for_search = resample_info.volume
                        used_factor = resample_info.used_factor
                        projector_pixel_factor = 1.0
                        pixel_size_resampling.append(
                            {
                                "offset_angstrom": pixel_offset,
                                "wanted_factor": wanted_factor,
                                "used_factor": used_factor,
                                "pad_shape": resample_info.pad_shape,
                                "intermediate_shape": resample_info.intermediate_shape,
                                "padding_value": resample_info.padding_value,
                            }
                        )
                    else:
                        template_for_search = prepared.template_original
                        projector_pixel_factor = wanted_factor
                        pixel_size_resampling.append(
                            {
                                "offset_angstrom": pixel_offset,
                                "wanted_factor": wanted_factor,
                                "used_factor": wanted_factor,
                                "backend": "coordinate",
                            }
                        )

                # FourierProjector construction performs the 3-D template FFT and
                # prepares the half-Hermitian volume.  It is a one-time cost per
                # pixel-size search position, not a per-orientation cost.
                with profiler.wall_section("search.projector_init_3d_fft", synchronize_cuda=True):
                    projector = FourierProjector(
                        template_for_search,
                        prepared.data_sizer.plan.search_shape,
                        backend=cfg.runtime.projector_backend,
                        projection_size=int(template_for_search.shape[-1]),
                        triton_fallback_to_gather=cfg.runtime.triton_fallback_to_gather,
                        triton_block_size=cfg.runtime.triton_block_size,
                        triton_num_warps=cfg.runtime.triton_num_warps,
                        cache_grid_sample_source=cfg.runtime.cache_grid_sample_source,
                        cache_sampled_filter=cfg.runtime.cache_sampled_projection_filter,
                        projection_fft_mode=cfg.runtime.projection_fft_mode,
                        center_phase_mode=prepared.center_phase_mode_effective,
                        fft_real_dtype=prepared.correlation_precision.real_dtype,
                        per_projection_whitening=pipeline.per_projection_whitening,
                        projection_whitening_epsilon=(
                            cfg.weighting.projection_whitening_epsilon
                        ),
                        mixed_precision_range_scaling=(
                            cfg.runtime.mixed_precision_range_scaling
                        ),
                        mixed_precision_l1_target=cfg.runtime.mixed_precision_l1_target,
                    )
                pixel_size_resampling[-1]["projector_backend_effective"] = projector.effective_backend
                pixel_size_resampling[-1]["projection_fft_mode_effective"] = projector.effective_fft_mode
                filter_scale = (
                    float(projector.volume_size)
                    / float(projector.projection_size)
                    / float(prepared.data_sizer.plan.full_binning_factor)
                )
                input_phase = (
                    projector.center_embedding_phase()
                    if (
                        prepared.center_phase_mode_effective == "input"
                        and projector.effective_fft_mode == "direct_s"
                    )
                    else None
                )
                base_image_correlation_fourier: Tensor | None = None
                base_prephase_backend_effective: str | None = None
                if cfg.weighting.mode == "cistem":
                    with profiler.wall_section(
                        "search.input_fourier_prephase", synchronize_cuda=True
                    ):
                        (
                            base_image_correlation_fourier,
                            base_prephase_backend_effective,
                        ) = prepare_image_fourier(
                            prepared.image_search_fourier,
                            phase=input_phase,
                            output_dtype=prepared.correlation_precision.complex_dtype,
                            backend=prepared.correlation_multiply_backend_resolved,
                            block_size=cfg.runtime.correlation_triton_block_size,
                            num_warps=cfg.runtime.correlation_triton_num_warps,
                            fallback_to_torch=(
                                cfg.runtime.correlation_triton_fallback_to_torch
                            ),
                        )
                pixel_size_resampling[-1]["center_phase_mode_effective"] = (
                    projector.effective_center_phase_mode
                )
                pixel_size_resampling[-1]["input_prephase_backend_effective"] = (
                    base_prephase_backend_effective
                    if base_prephase_backend_effective is not None
                    else "per_defocus_gisspa"
                )
                pixel_size_resampling[-1]["correlation_precision_effective"] = (
                    prepared.correlation_precision.effective
                )
                pixel_size_resampling[-1]["correlation_fft_shape"] = list(
                    prepared.data_sizer.plan.search_shape
                )
                # Resolve an auto/failed Triton correlation backend once.  A
                # first-JIT failure must not be retried and warned for every
                # orientation batch.
                multiply_backend_runtime = (
                    prepared.correlation_multiply_backend_resolved
                )
                correlation_complex_dtype_runtime = (
                    prepared.correlation_precision.complex_dtype
                )

                for defocus_index, defocus_offset in enumerate(defocus_offsets):
                    microscope = cfg.microscope
                    projection_ctf = CTF(
                        voltage_kv=microscope.voltage_kv,
                        spherical_aberration_mm=microscope.spherical_aberration_mm,
                        amplitude_contrast=microscope.amplitude_contrast,
                        pixel_size_angstrom=cfg.search.pixel_size_angstrom,
                        defocus1_angstrom=microscope.defocus1_angstrom + float(defocus_offset),
                        defocus2_angstrom=microscope.defocus2_angstrom + float(defocus_offset),
                        astigmatism_angle_deg=microscope.defocus_angle_deg,
                        additional_phase_shift_deg=microscope.phase_shift_deg,
                    )
                    filter_start = time.perf_counter()
                    with profiler.wall_section(
                        "search.ctf_and_projection_filter", synchronize_cuda=True
                    ):
                        projection_filter = pipeline.build_projection(
                            projection_ctf,
                            (projector.volume_size, projector.volume_size),
                            device=device,
                            dtype=prepared.geometry_dtype,
                            normalized_frequency_scale=1.0,
                        )

                    image_weight: Tensor | None = None
                    image_source_fourier = prepared.image_search_fourier
                    if cfg.weighting.mode == "gisspa_repo":
                        image_ctf = CTF(
                            voltage_kv=microscope.voltage_kv,
                            spherical_aberration_mm=microscope.spherical_aberration_mm,
                            amplitude_contrast=microscope.amplitude_contrast,
                            pixel_size_angstrom=(
                                prepared.data_sizer.plan.search_pixel_size_angstrom
                            ),
                            defocus1_angstrom=(
                                microscope.defocus1_angstrom + float(defocus_offset)
                            ),
                            defocus2_angstrom=(
                                microscope.defocus2_angstrom + float(defocus_offset)
                            ),
                            astigmatism_angle_deg=microscope.defocus_angle_deg,
                            additional_phase_shift_deg=microscope.phase_shift_deg,
                        )
                        with profiler.wall_section(
                            "search.gisspa_image_weight", synchronize_cuda=True
                        ):
                            image_weight = pipeline.build_image_weight(
                                image_ctf,
                                prepared.data_sizer.plan.search_shape,
                                device=device,
                                dtype=prepared.geometry_dtype,
                            )
                            image_source_fourier = apply_fourier_weight_and_normalize(
                                prepared.image_search_fourier,
                                image_weight,
                                last_real_size=int(
                                    prepared.data_sizer.plan.search_shape[-1]
                                ),
                                normalization_pixels=(
                                    prepared.data_sizer.plan.number_of_pixels_for_normalization
                                ),
                            )
                        with profiler.wall_section(
                            "search.input_fourier_prephase", synchronize_cuda=True
                        ):
                            (
                                image_correlation_fourier,
                                prephase_backend_effective,
                            ) = prepare_image_fourier(
                                image_source_fourier,
                                phase=input_phase,
                                output_dtype=correlation_complex_dtype_runtime,
                                backend=(
                                    prepared.correlation_multiply_backend_resolved
                                ),
                                block_size=(
                                    cfg.runtime.correlation_triton_block_size
                                ),
                                num_warps=(
                                    cfg.runtime.correlation_triton_num_warps
                                ),
                                fallback_to_torch=(
                                    cfg.runtime.correlation_triton_fallback_to_torch
                                ),
                            )
                    else:
                        if base_image_correlation_fourier is None:
                            raise RuntimeError(
                                "internal error: missing cisTEM image correlation Fourier"
                            )
                        image_correlation_fourier = base_image_correlation_fourier
                        prephase_backend_effective = (
                            base_prephase_backend_effective or "identity"
                        )

                    pixel_size_resampling[-1].setdefault(
                        "input_prephase_backends_by_defocus", []
                    ).append(prephase_backend_effective)
                    if debug:
                        debug.add_timing(
                            "stage_05_ctf_and_projection_filter",
                            time.perf_counter() - filter_start,
                        )
                    if debug and not first_debug_batch_done:
                        debug.save(
                            5,
                            "ctf_image",
                            projection_ctf.image(
                                (projector.volume_size,) * 2, device=device
                            ),
                        )
                        debug.save(5, "projection_filter", projection_filter)
                        if image_weight is not None:
                            debug.save(5, "gisspa_image_weight", image_weight)
                            debug.save(
                                5,
                                "gisspa_weighted_image_fourier_abs",
                                image_source_fourier.abs(),
                            )
                        debug.save(
                            5,
                            "filter_parameters",
                            {
                                "defocus_offset_angstrom": defocus_offset,
                                "pixel_size_offset_angstrom": pixel_offset,
                                "filter_scale_for_search": filter_scale,
                                "weighting_mode": cfg.weighting.mode,
                                "weighting_parameters": asdict(cfg.weighting),
                                "image_weight_grid": (
                                    list(prepared.data_sizer.plan.search_shape)
                                    if image_weight is not None
                                    else None
                                ),
                                "projection_weight_grid": [
                                    projector.volume_size,
                                    projector.volume_size,
                                ],
                            },
                        )
                        debug.finish_stage(5)

                    # Use a real first batch as a numerical probe.  The old
                    # zero-valued API probe could not detect half-FFT overflow
                    # and allowed a million-orientation run to fail only during
                    # final statistics.  Range scaling should make this pass;
                    # a configured fallback switches to float32 before any CCC
                    # is accumulated.
                    if (
                        projector.fft_real_dtype == torch.float16
                        and cfg.runtime.mixed_precision_health_check_batches > 0
                        and len(selected_indices) > 0
                    ):
                        with profiler.wall_section(
                            "search.mixed_precision_preflight", synchronize_cuda=True
                        ):
                            probe_count = min(
                                cfg.runtime.orientation_batch_size,
                                len(selected_indices),
                            )
                            probe_indices = selected_indices[:probe_count]
                            probe_cached = orientation_table.take(probe_indices)
                            probe_health: dict[str, Any] = {
                                "image_fourier": tensor_health(image_correlation_fourier)
                            }
                            probe_projection = projector.project_and_normalize(
                                probe_cached.rotation_matrices,
                                projection_filter,
                                real_space_binning_factor=(
                                    prepared.data_sizer.plan.full_binning_factor
                                ),
                                pixel_size_factor=projector_pixel_factor,
                                backend=None,
                                filter_scale_override=filter_scale,
                                profiler=None,
                                retain_intermediates=False,
                            )
                            probe_health["projection_fourier"] = tensor_health(
                                probe_projection.padded_fourier
                            )
                            if probe_projection.correlation_inverse_scale is not None:
                                inv = probe_projection.correlation_inverse_scale.to(torch.float32)
                                probe_health["projection_range_scale"] = {
                                    "healthy": bool(torch.isfinite(inv).all().item()),
                                    "minimum_inverse_scale": float(inv.min().item()),
                                    "maximum_inverse_scale": float(inv.max().item()),
                                    "scaled_projection_count": int((inv > 1.0).sum().item()),
                                }
                            probe_product = probe_projection.padded_fourier
                            probe_backend = multiply_backend_runtime
                            if all(
                                bool(item["healthy"])
                                for item in probe_health.values()
                            ):
                                probe_backend = conjugate_multiply_inplace(
                                    image_correlation_fourier,
                                    probe_product,
                                    backend=multiply_backend_runtime,
                                    block_size=cfg.runtime.correlation_triton_block_size,
                                    num_warps=cfg.runtime.correlation_triton_num_warps,
                                    fallback_to_torch=(
                                        cfg.runtime.correlation_triton_fallback_to_torch
                                    ),
                                    inverse_scale=(
                                        probe_projection.correlation_inverse_scale
                                    ),
                                )
                                probe_health["correlation_product"] = tensor_health(
                                    probe_product
                                )
                            if all(
                                bool(item["healthy"])
                                for item in probe_health.values()
                            ):
                                probe_ccf = irfft2_cistem(
                                    probe_product,
                                    prepared.data_sizer.plan.search_shape,
                                )
                                probe_health["ccf"] = tensor_health(probe_ccf)

                            healthy = all(
                                bool(item["healthy"])
                                for item in probe_health.values()
                            )
                            pixel_size_resampling[-1].setdefault(
                                "mixed_precision_health", []
                            ).append(
                                {
                                    "defocus_offset_angstrom": float(defocus_offset),
                                    "healthy": healthy,
                                    "stages": probe_health,
                                }
                            )
                            if not healthy:
                                failed = [
                                    name
                                    for name, item in probe_health.items()
                                    if not bool(item["healthy"])
                                ]
                                reason = (
                                    "mixed_float16 numerical preflight failed at "
                                    + ", ".join(failed)
                                )
                                if not cfg.runtime.mixed_precision_fallback_to_float32:
                                    raise RuntimeError(
                                        reason
                                        + "; enable mixed_precision_fallback_to_float32 "
                                        "or use correlation_precision='float32'"
                                    )
                                warnings.warn(
                                    reason
                                    + "; falling back to float32 on the existing FFT grid",
                                    RuntimeWarning,
                                    stacklevel=2,
                                )
                                projector.fft_real_dtype = torch.float32
                                correlation_complex_dtype_runtime = torch.complex64
                                image_correlation_fourier, prephase_backend_effective = (
                                    prepare_image_fourier(
                                        image_source_fourier,
                                        phase=input_phase,
                                        output_dtype=torch.complex64,
                                        backend=(
                                            prepared.correlation_multiply_backend_resolved
                                        ),
                                        block_size=(
                                            cfg.runtime.correlation_triton_block_size
                                        ),
                                        num_warps=(
                                            cfg.runtime.correlation_triton_num_warps
                                        ),
                                        fallback_to_torch=(
                                            cfg.runtime.correlation_triton_fallback_to_torch
                                        ),
                                    )
                                )
                                if cfg.weighting.mode == "cistem":
                                    base_image_correlation_fourier = image_correlation_fourier
                                    base_prephase_backend_effective = (
                                        prephase_backend_effective
                                    )
                                pixel_size_resampling[-1][
                                    "correlation_precision_effective"
                                ] = "float32_runtime_fallback"
                                pixel_size_resampling[-1][
                                    "correlation_precision_runtime_fallback_reason"
                                ] = reason
                            else:
                                multiply_backend_runtime = probe_backend

                    for _local_start, orientation_batch in _batch_slices(
                        selected,
                        cfg.runtime.orientation_batch_size,
                    ):
                        batch_size = len(orientation_batch)
                        profiler.begin_batch(
                            batch_counter,
                            batch_size,
                            metadata={
                                "orientation_first": int(orientation_batch[0].orientation_index),
                                "orientation_last": int(orientation_batch[-1].orientation_index),
                                "pixel_size_index": int(pixel_index),
                                "defocus_index": int(defocus_index),
                            },
                        )
                        rows: list[dict[str, Any]] = []
                        try:
                            batch_global_indices = selected_indices[
                                _local_start : _local_start + batch_size
                            ]
                            with profiler.batch_section("geometry.orientation_cache_slice"):
                                cached_orientation = orientation_table.take(batch_global_indices)
                                phi = cached_orientation.phi
                                theta = cached_orientation.theta
                                psi = cached_orientation.psi
                                global_orientation_indices = cached_orientation.orientation_index
                                matrices = cached_orientation.rotation_matrices
                            projection: ProjectionBatch = projector.project_and_normalize(
                                matrices,
                                projection_filter,
                                real_space_binning_factor=prepared.data_sizer.plan.full_binning_factor,
                                pixel_size_factor=projector_pixel_factor,
                                backend=None,
                                filter_scale_override=filter_scale,
                                profiler=profiler,
                                retain_intermediates=bool(debug and not first_debug_batch_done),
                            )
                            # A first-use Triton JIT failure can switch the
                            # projector to gather.  Refresh metadata after the call
                            # so the final report records the backend actually used.
                            pixel_size_resampling[-1]["projector_backend_effective"] = projector.effective_backend
                            with profiler.batch_section("ccf.conjugate_multiply_fused"):
                                # Reuse the projection Fourier buffer.  The Triton
                                # backend fuses physical conjugation and complex
                                # multiplication into one full-spectrum traversal.
                                product = projection.padded_fourier
                                multiply_backend_effective = conjugate_multiply_inplace(
                                    image_correlation_fourier,
                                    product,
                                    backend=multiply_backend_runtime,
                                    block_size=cfg.runtime.correlation_triton_block_size,
                                    num_warps=cfg.runtime.correlation_triton_num_warps,
                                    fallback_to_torch=cfg.runtime.correlation_triton_fallback_to_torch,
                                    inverse_scale=projection.correlation_inverse_scale,
                                )
                                pixel_size_resampling[-1][
                                    "correlation_multiply_backend_effective"
                                ] = multiply_backend_effective
                                multiply_backend_runtime = multiply_backend_effective
                            with profiler.batch_section("ccf.irfft_large"):
                                ccf = irfft2_cistem(
                                    product, prepared.data_sizer.plan.search_shape
                                )
                            with profiler.batch_section("ccf.cast_float32"):
                                ccf = ccf.to(torch.float32)

                            with profiler.batch_section("metadata.task_tensors"):
                                condition_base = (
                                    (pixel_index * len(defocus_offsets) + defocus_index)
                                    * total_orientation_count
                                )
                                task_indices = global_orientation_indices + condition_base
                            accum.update(
                                ccf,
                                task_indices=task_indices,
                                profiler=profiler,
                            )

                            # CCF-stack metadata is unnecessary in the normal
                            # ``none`` mode.  Skipping it also avoids per-orientation
                            # CUDA ``.item()`` synchronizations after the accumulator.
                            rows: list[dict[str, object]] = []
                            if writer.mode != "none":
                                with profiler.batch_section("metadata.rows"):
                                    condition_base_cpu = int(
                                        (pixel_index * len(defocus_offsets) + defocus_index)
                                        * total_orientation_count
                                    )
                                    for local, orientation in enumerate(orientation_batch):
                                        rows.append(
                                            {
                                                "stack_index": stack_index + local,
                                                "task_index": condition_base_cpu + int(orientation.orientation_index),
                                                "orientation_index": orientation.orientation_index,
                                                "phi_deg": orientation.phi_deg,
                                                "theta_deg": orientation.theta_deg,
                                                "psi_deg": orientation.psi_deg,
                                                "defocus_offset_angstrom": defocus_offset,
                                                "pixel_size_offset_angstrom": pixel_offset,
                                            }
                                        )
                                with profiler.batch_section("ccf_stack.write"):
                                    writer.write(ccf, rows)
                            stack_index += batch_size
                        finally:
                            profiler.end_batch()

                        batch_counter += 1
                        completed_orientations += batch_size
                        profiler.maybe_print_progress(
                            completed_batches=batch_counter,
                            completed_orientations=completed_orientations,
                            total_batches=total_batches,
                            total_orientations=total_stack_images,
                        )

                        # Keep numerical debug I/O outside sampled batch timings so
                        # saving the first batch does not masquerade as GPU compute.
                        if debug and not first_debug_batch_done:
                            assert projection.fourier_slice is not None
                            assert projection.sampled_filter is not None
                            assert projection.filtered_slice is not None
                            assert projection.real_before_edge_subtraction is not None
                            assert projection.edge_average is not None
                            assert projection.real_after_edge_subtraction is not None
                            assert projection.variance is not None
                            assert projection.real_normalized is not None
                            assert projection.padded_real is not None
                            debug.save(6, "fourier_slice_real", projection.fourier_slice.real)
                            debug.save(6, "fourier_slice_imag", projection.fourier_slice.imag)
                            debug.save(6, "sampled_filter", projection.sampled_filter)
                            if projection.projection_radial_power is not None:
                                debug.save(6, "projection_radial_power", projection.projection_radial_power)
                            if projection.correlation_inverse_scale is not None:
                                debug.save(6, "correlation_inverse_scale", projection.correlation_inverse_scale)
                            debug.save(6, "filtered_slice_real", projection.filtered_slice.real)
                            debug.save(6, "filtered_slice_imag", projection.filtered_slice.imag)
                            if cfg.debug.compare_projectors:
                                comparison = projector.compare_backends(
                                    matrices,
                                    real_space_binning_factor=prepared.data_sizer.plan.full_binning_factor,
                                    pixel_size_factor=projector_pixel_factor,
                                )
                                for name, value in comparison.items():
                                    debug.save(6, f"projector_compare_{name}", value)
                            debug.finish_stage(6)

                            debug.save(7, "projection_real_before_edge", projection.real_before_edge_subtraction)
                            debug.save(7, "projection_edge_average", projection.edge_average)
                            debug.save(7, "projection_real_after_edge", projection.real_after_edge_subtraction)
                            debug.save(7, "projection_variance", projection.variance)
                            debug.save(7, "projection_normalized", projection.real_normalized)
                            debug.save(7, "projection_padded", projection.padded_real)
                            debug.finish_stage(7)

                            debug.save(8, "first_ccf_batch", ccf)
                            debug.save(8, "first_ccf_metadata", rows)
                            debug.finish_stage(8)
                            first_debug_batch_done = True

                        if (
                            debug
                            and cfg.debug.save_every_n_orientations > 0
                            and stack_index % cfg.debug.save_every_n_orientations == 0
                        ):
                            debug.save(9, f"partial_mip_after_{stack_index}", accum.mip)
        finally:
            with profiler.wall_section("search.ccf_stack_close"):
                ccf_stack = writer.close()
            profiler.finish_search(
                total_batches=batch_counter,
                total_orientations=completed_orientations,
            )

        if ccf_stack.mode == "cpu" and ccf_stack.tensor is not None:
            assert stack_path is not None
            with profiler.wall_section("search.ccf_stack_cpu_serialize"):
                torch.save(ccf_stack.tensor, stack_path)
                ccf_stack.tensor_path = Path(stack_path)

        with profiler.wall_section("search.accumulator_result", synchronize_cuda=True):
            raw = accum.result()
        if debug:
            debug.add_timing("stage_09_raw_search_total", time.perf_counter() - raw_loop_start)
            debug.save(9, "raw_mip_search_size", raw.mip)
            debug.save(9, "raw_phi_search_size", raw.phi)
            debug.save(9, "raw_theta_search_size", raw.theta)
            debug.save(9, "raw_psi_search_size", raw.psi)
            debug.save(9, "raw_defocus_search_size", raw.defocus)
            debug.save(9, "raw_pixel_size_search_size", raw.pixel_size)
            debug.save(9, "raw_correlation_sum_search_size", raw.correlation_sum)
            debug.save(9, "raw_correlation_sum_squares_search_size", raw.correlation_sum_squares)
            debug.save(9, "raw_histogram", raw.histogram)
            debug.save(9, "number_searched", raw.number_searched)
            debug.finish_stage(9)

        return RawRunOutput(
            raw=raw,
            prepared=prepared,
            orientations=selected,
            all_orientation_count=total_orientation_count,
            selected_orientation_indices=selected_indices,
            defocus_offsets=defocus_offsets,
            pixel_size_offsets=pixel_offsets,
            ccf_stack=ccf_stack,
            elapsed_seconds=time.perf_counter() - start_time,
            pixel_size_resampling=pixel_size_resampling,
            execution_devices=[str(device)],
            search_spacing=spacing,
            profiler=profiler,
            timing_report=profiler.report(),
        )

    def finalize(self, run: RawRunOutput, *, write_outputs: bool = True) -> MatchResult:
        finalize_start = time.perf_counter()
        cfg = self.config
        prepared = run.prepared
        profiler = run.profiler or PerformanceProfiler(
            TimingConfig(enabled=False),
            device=prepared.device,
            output_dir=cfg.output_dir,
            output_prefix=cfg.output_prefix,
            package_version="0.3.6",
        )

        with profiler.wall_section("finalize.resize_results", synchronize_cuda=True):
            raw_resized, output_valid_mask = _clone_raw_with_resized_maps(
                run.raw,
                prepared.data_sizer,
                cfg.search.apply_result_rescaling,
            )
        if prepared.debug:
            prepared.debug.save(10, "post_resize_valid_mask", output_valid_mask)
            prepared.debug.save(10, "post_resize_raw_mip", raw_resized.mip)
            prepared.debug.save(10, "post_resize_phi", raw_resized.phi)
            prepared.debug.save(10, "post_resize_theta", raw_resized.theta)
            prepared.debug.save(10, "post_resize_psi", raw_resized.psi)
            prepared.debug.save(10, "post_resize_correlation_sum", raw_resized.correlation_sum)
            prepared.debug.save(10, "post_resize_correlation_sum_squares", raw_resized.correlation_sum_squares)

        with profiler.wall_section("finalize.scale_statistics", synchronize_cuda=True):
            scaled = scale_search_statistics(
                raw_resized,
                disable_flat_fielding=cfg.search.disable_flat_fielding,
                valid_mask=output_valid_mask,
            )
        output_pixel_size = (
            cfg.search.pixel_size_angstrom
            if cfg.search.apply_result_rescaling
            else prepared.data_sizer.plan.search_pixel_size_angstrom
        )
        with profiler.wall_section("finalize.threshold"):
            if cfg.search.threshold_override is None:
                threshold = theoretical_threshold(
                    expected_false_positives=cfg.search.expected_false_positives,
                    number_of_valid_search_pixels=prepared.data_sizer.plan.number_of_threshold_pixels,
                    number_searched=raw_resized.number_searched,
                    independent_fraction=cfg.search.fraction_of_search_positions_independent,
                    defocus_positions=len(run.defocus_offsets),
                    ignore_defocus_for_threshold=cfg.search.ignore_defocus_for_threshold,
                )
                threshold_source = "cistem_theoretical"
            else:
                threshold = float(cfg.search.threshold_override)
                threshold_source = "json_override"
        minimum_radius_angstrom = (
            cfg.search.minimum_peak_radius_angstrom
            if cfg.search.minimum_peak_radius_angstrom is not None
            else 10.0
        )
        with profiler.wall_section("finalize.peak_pick", synchronize_cuda=True):
            peaks = peak_pick(
                scaled.scaled_mip,
                scaled.mip_zscore,
                {
                    "phi": raw_resized.phi,
                    "theta": raw_resized.theta,
                    "psi": raw_resized.psi,
                    "defocus": raw_resized.defocus,
                    "pixel_size": raw_resized.pixel_size,
                },
                threshold=threshold,
                minimum_radius_pixels=float(minimum_radius_angstrom) / output_pixel_size,
                pixel_size_angstrom=output_pixel_size,
                maximum_peaks=cfg.search.maximum_peaks,
            )

        with profiler.wall_section("finalize.histogram_curves", synchronize_cuda=True):
            independent_trials = (
                prepared.data_sizer.plan.number_of_threshold_pixels
                * raw_resized.number_searched
                * cfg.search.fraction_of_search_positions_independent
            )
            if cfg.search.ignore_defocus_for_threshold:
                independent_trials /= max(len(run.defocus_offsets), 1)
            histogram_rows: list[dict[str, Any]] = []
            if cfg.runtime.histogram_mode != "off":
                survival = torch.flip(
                    torch.cumsum(torch.flip(scaled.smoothed_histogram.to(torch.float64), dims=(0,)), dim=0),
                    dims=(0,),
                )
                expected_survival = expected_gaussian_survival(
                    scaled.histogram_x,
                    independent_trials,
                )
                histogram_rows = [
                    {
                        "bin": i,
                        "snr_midpoint": float(scaled.histogram_x[i].item()),
                        "source_curve_x_scaled": float(scaled.source_curve_x_scaled[i].item()),
                        "raw_count": int(round(float(scaled.raw_histogram[i].item()))),
                        "smoothed_count": int(round(float(scaled.smoothed_histogram[i].item()))),
                        "observed_survival": float(survival[i].item()),
                        "expected_gaussian_survival": float(expected_survival[i].item()),
                    }
                    for i in range(HISTOGRAM_BINS)
                ]

        with profiler.wall_section("finalize.metadata", synchronize_cuda=True):
            runtime_precision_values = sorted({
                str(item.get("correlation_precision_effective", prepared.correlation_precision.effective))
                for item in run.pixel_size_resampling
            })
            runtime_precision_effective = (
                runtime_precision_values[0]
                if len(runtime_precision_values) == 1
                else "+".join(runtime_precision_values)
            )
            runtime_real_dtype = (
                "float32"
                if runtime_precision_effective.startswith("float32")
                else str(prepared.correlation_precision.real_dtype).removeprefix("torch.")
            )
            runtime_complex_dtype = (
                "complex64"
                if runtime_precision_effective.startswith("float32")
                else str(prepared.correlation_precision.complex_dtype).removeprefix("torch.")
            )
            metadata: dict[str, Any] = {
                "package_version": "0.3.6",
                "cistem_source_commit": CISTEM_SOURCE_COMMIT,
                "input_mrc": str(Path(cfg.input_mrc).resolve()),
                "template_mrc": str(Path(cfg.template_mrc).resolve()),
                "device": run.execution_devices if len(run.execution_devices) > 1 else (run.execution_devices[0] if run.execution_devices else str(prepared.device)),
                "dtype_requested": cfg.runtime.dtype,
                "dtype_effective": str(prepared.storage_dtype).removeprefix("torch."),
                "dtype_promoted_on_cpu": bool(
                    prepared.device.type == "cpu" and cfg.runtime.dtype != "float32"
                ),
                "correlation_precision_requested": cfg.runtime.correlation_precision,
                "correlation_precision_effective": runtime_precision_effective,
                "correlation_precision_effective_initial": prepared.correlation_precision.effective,
                "correlation_precision_effective_per_condition": runtime_precision_values,
                "correlation_real_dtype": runtime_real_dtype,
                "correlation_complex_dtype": runtime_complex_dtype,
                "correlation_precision_fallback_reason": (
                    prepared.correlation_precision.fallback_reason
                ),
                "half_fft_shape_mode": cfg.runtime.half_fft_shape_mode,
                "mixed_precision_range_scaling": cfg.runtime.mixed_precision_range_scaling,
                "mixed_precision_l1_target": cfg.runtime.mixed_precision_l1_target,
                "mixed_precision_health_check_batches": cfg.runtime.mixed_precision_health_check_batches,
                "fft_size_mode_requested": prepared.requested_fft_size_mode,
                "fft_size_mode_effective": prepared.effective_fft_size_mode,
                "center_phase_mode": cfg.runtime.center_phase_mode,
                "center_phase_mode_resolved": prepared.center_phase_mode_effective,
                "correlation_multiply_backend": (
                    cfg.runtime.correlation_multiply_backend
                ),
                "correlation_triton_block_size": cfg.runtime.correlation_triton_block_size,
                "correlation_triton_num_warps": cfg.runtime.correlation_triton_num_warps,
                "ccf_public_dtype": "float32",
                "correlation_multiply_backend_resolved": (
                    prepared.correlation_multiply_backend_resolved
                ),
                "projector_backend": cfg.runtime.projector_backend,
                "projector_backend_effective": sorted({
                    str(item.get("projector_backend_effective", cfg.runtime.projector_backend))
                    for item in run.pixel_size_resampling
                }),
                "projection_fft_mode": cfg.runtime.projection_fft_mode,
                "projection_fft_mode_effective": sorted({
                    str(item.get("projection_fft_mode_effective", cfg.runtime.projection_fft_mode))
                    for item in run.pixel_size_resampling
                }),
                "center_phase_mode_effective": sorted({
                    str(item.get("center_phase_mode_effective", cfg.runtime.center_phase_mode))
                    for item in run.pixel_size_resampling
                }),
                "input_prephase_backend_effective": sorted({
                    str(item.get("input_prephase_backend_effective", "unknown"))
                    for item in run.pixel_size_resampling
                }),
                "correlation_multiply_backend_effective": sorted({
                    str(item.get("correlation_multiply_backend_effective", "unknown"))
                    for item in run.pixel_size_resampling
                }),
                "cache_grid_sample_source": cfg.runtime.cache_grid_sample_source,
                "cache_sampled_projection_filter": cfg.runtime.cache_sampled_projection_filter,
                "cache_orientation_tensors": cfg.runtime.cache_orientation_tensors,
                "precompute_rotation_matrices": cfg.runtime.precompute_rotation_matrices,
                "pixel_size_backend": cfg.runtime.pixel_size_backend,
                "sizing": prepared.data_sizer.plan.to_dict(),
                "angular_sampling": run.search_spacing,
                "search_grid_shape": list(prepared.data_sizer.plan.search_shape),
                "output_grid_shape": list(raw_resized.mip.shape),
                "output_valid_pixels_after_rescaling": int(output_valid_mask.sum().item()),
                "mip_valid_pixels_on_search_grid": int(prepared.valid_mask.sum().item()),
                "moment_valid_pixels_on_search_grid": int(prepared.valid_mask.sum().item()),
                "histogram_valid_pixels_on_search_grid": int(prepared.histogram_mask.sum().item()),
                # Backward-compatible alias; this field described the histogram
                # ROI in v0.3.x, not the moment ROI after the edge fix.
                "statistics_valid_pixels_on_search_grid": int(prepared.histogram_mask.sum().item()),
                "threshold_pixels_on_search_grid": int(prepared.data_sizer.plan.number_of_threshold_pixels),
                "number_of_all_orientations": run.all_orientation_count,
                "number_of_selected_orientations": len(run.orientations),
                "selected_orientation_first": min(run.selected_orientation_indices),
                "selected_orientation_last": max(run.selected_orientation_indices),
                "number_of_defocus_positions": len(run.defocus_offsets),
                "number_of_pixel_size_positions": len(run.pixel_size_offsets),
                "number_searched": raw_resized.number_searched,
                "global_ccc_mean": scaled.global_mean,
                "global_ccc_std": scaled.global_std,
                "counted_pixels": scaled.counted_pixels,
                "threshold": threshold,
                "threshold_source": threshold_source,
                "threshold_override": cfg.search.threshold_override,
                "weighting": asdict(cfg.weighting),
                "independent_trials": independent_trials,
                "output_pixel_size_angstrom": output_pixel_size,
                "elapsed_seconds_raw_search": run.elapsed_seconds,
                "histogram_mode": cfg.runtime.histogram_mode,
                "histogram_backend": cfg.runtime.histogram_backend,
                "histogram_sample_orientation_stride": cfg.runtime.histogram_sample_orientation_stride,
                "histogram_sample_pixel_stride": cfg.runtime.histogram_sample_pixel_stride,
                "pixel_size_resampling": run.pixel_size_resampling,
                "known_compatibility_choices": {
                    "mip_initial_value": cfg.search.mip_initial_value,
                    "psi_360_duplicate_preserved": cfg.search.preserve_psi_360_duplicate,
                    "histogram_off_by_one_bug_fixed": True,
                    "non_mkl_erfcinv_assignment_bug_fixed": True,
                    "padding_mode": cfg.search.padding_mode,
                    "histogram_roi_mode": cfg.search.statistics_roi_mode,
                    "moments_follow_source_and_post_search_mask": True,
                    "threshold_pixel_mode": cfg.search.threshold_pixel_mode,
                },
            }

        files: dict[str, Path] = {}
        if write_outputs:
            output_dir = Path(cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            prefix = output_dir / cfg.output_prefix
            mrc_maps = {
                "mip": scaled.mip_zscore,
                "scaled_mip": scaled.scaled_mip,
                "phi": raw_resized.phi,
                "theta": raw_resized.theta,
                "psi": raw_resized.psi,
                "defocus": raw_resized.defocus,
                "pixel_size": raw_resized.pixel_size,
                "corr_average": scaled.local_mean,
                "corr_stddev": scaled.local_std,
            }
            if cfg.runtime.save_raw_accumulators:
                mrc_maps.update(
                    {
                        "raw_mip": raw_resized.mip,
                        "raw_correlation_sum": raw_resized.correlation_sum,
                        "raw_correlation_sum_squares": raw_resized.correlation_sum_squares,
                    }
                )
            for name, value in mrc_maps.items():
                with profiler.wall_section(f"io.write_mrc.{name}", synchronize_cuda=True):
                    files[name] = write_mrc(f"{prefix}_{name}.mrc", value, output_pixel_size)
            if histogram_rows:
                with profiler.wall_section("io.write_histogram"):
                    files["histogram"] = write_rows_tsv(
                        f"{prefix}_histogram.tsv",
                        histogram_rows,
                        fieldnames=list(histogram_rows[0]),
                    )
            peak_rows = [peak.to_dict() for peak in peaks]
            peak_fields = list(Peak(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0).to_dict())
            with profiler.wall_section("io.write_peaks"):
                files["peaks"] = write_rows_tsv(
                    f"{prefix}_peaks.tsv",
                    peak_rows,
                    fieldnames=peak_fields,
                )
            with profiler.wall_section("io.write_config"):
                files["config"] = write_json(f"{prefix}_config.json", cfg.to_dict())
            with profiler.wall_section("io.write_orientations"):
                files["orientations"] = write_rows_tsv(
                    f"{prefix}_orientations.tsv",
                    _orientation_rows(run.orientations),
                    fieldnames=["orientation_index", "grid_index", "phi_deg", "theta_deg", "psi_deg"],
                )
            if run.ccf_stack.mode != "none":
                if run.ccf_stack.mrc_path is not None:
                    files["ccf_stack"] = Path(run.ccf_stack.mrc_path)
                elif run.ccf_stack.tensor_path is not None:
                    files["ccf_stack"] = Path(run.ccf_stack.tensor_path)
                if run.ccf_stack.metadata_path is not None:
                    files["ccf_stack_metadata"] = Path(run.ccf_stack.metadata_path)

        debug = prepared.debug
        if debug:
            debug.add_timing("stage_10_postresize_scaled_mip_and_peaks", time.perf_counter() - finalize_start)
            debug.save(10, "mip_zscore", scaled.mip_zscore)
            debug.save(10, "scaled_mip", scaled.scaled_mip)
            debug.save(10, "local_mean", scaled.local_mean)
            debug.save(10, "local_std", scaled.local_std)
            debug.save(10, "threshold", threshold)
            debug.save(10, "peaks", [p.to_dict() for p in peaks])

        profiler.finish_run()
        timing_report = profiler.report() if profiler.enabled else dict(run.timing_report)
        run.timing_report = timing_report
        if timing_report:
            # Keep the metadata compact; the full per-batch samples remain in the
            # standalone timing JSON/TSV files.
            compact_timing = dict(timing_report)
            compact_timing.pop("batch_samples", None)
            metadata["performance_timing"] = compact_timing

        if write_outputs and profiler.enabled:
            timing_files = profiler.write()
            files.update(timing_files)
            metadata["timing_files"] = {name: str(path) for name, path in timing_files.items()}
        elif write_outputs and timing_report:
            # Multi-GPU workers return independent timing reports. Preserve them
            # in one aggregate JSON even though there is no single CUDA profiler
            # object in the CPU finalization process.
            timing_path = Path(cfg.output_dir) / f"{cfg.output_prefix}_timing_multigpu.json"
            files["timing_json"] = write_json(timing_path, timing_report)
            metadata["timing_files"] = {"timing_json": str(files["timing_json"])}

        if write_outputs:
            output_dir = Path(cfg.output_dir)
            prefix = output_dir / cfg.output_prefix
            # Metadata is intentionally written last so it contains timing paths
            # and the final performance summary. The metadata write itself is not
            # included in the timing report.
            files["metadata"] = write_json(f"{prefix}_metadata.json", metadata)

        if debug:
            debug.save(10, "metadata", metadata)
            debug.finish_stage(10)

        return MatchResult(
            raw=raw_resized,
            scaled=scaled,
            peaks=peaks,
            threshold=threshold,
            output_pixel_size_angstrom=output_pixel_size,
            files=files,
            metadata=metadata,
            ccf_stack=run.ccf_stack,
        )

    def run(self) -> MatchResult:
        if len(self.config.runtime.devices) > 1:
            from .multigpu import run_multi_gpu

            return run_multi_gpu(self.config)
        if len(self.config.runtime.devices) == 1:
            device = torch.device(f"cuda:{self.config.runtime.devices[0]}")
            run = self.run_raw(device_override=device)
        else:
            run = self.run_raw()
        return self.finalize(run)
