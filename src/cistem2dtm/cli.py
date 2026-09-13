from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import MatchConfig, effective_image_fft_size_mode
from .batch import plan_batch, run_batch
from .pdb_template import generate_pdb_template, resolve_template_for_match
from .datasizer import DataSizer
from .debug import DebugStop, STAGE_NAMES
from .diagnostics import diagnose_histogram_against_plan, read_histogram_summary
from .geometry import EulerSearch, cisTEM_auto_angular_step, cisTEM_auto_psi_step
from .io import read_search_image, read_template_volume, write_json
from .matcher import TemplateMatcher
from .synthetic import generate_synthetic_example
from .validation import compare_debug_directories


def _parse_devices(text: str | None) -> list[int] | None:
    if text is None:
        return None
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        return []
    return [int(value) for value in values]


def _apply_run_overrides(cfg: MatchConfig, args: argparse.Namespace, *, require_paths: bool = True) -> MatchConfig:
    direct = {
        "input_mrc": args.input_mrc,
        "micrographs_star": args.micrographs_star,
        "template_mrc": args.template_mrc,
        "output_dir": args.output_dir,
        "output_prefix": args.output_prefix,
    }
    for key, value in direct.items():
        if value is not None:
            setattr(cfg, key, value)
    runtime = {
        "device": args.device,
        "dtype": args.dtype,
        "projector_backend": args.projector_backend,
        "triton_fallback_to_gather": args.triton_fallback_to_gather,
        "triton_block_size": args.triton_block_size,
        "triton_num_warps": args.triton_num_warps,
        "cache_grid_sample_source": args.cache_grid_sample_source,
        "cache_sampled_projection_filter": args.cache_sampled_projection_filter,
        "cache_orientation_tensors": args.cache_orientation_tensors,
        "precompute_rotation_matrices": args.precompute_rotation_matrices,
        "orientation_precompute_chunk_size": args.orientation_precompute_chunk_size,
        "projection_fft_mode": args.projection_fft_mode,
        "center_phase_mode": args.center_phase_mode,
        "correlation_multiply_backend": args.correlation_multiply_backend,
        "correlation_triton_fallback_to_torch": args.correlation_triton_fallback_to_torch,
        "correlation_triton_block_size": args.correlation_triton_block_size,
        "correlation_triton_num_warps": args.correlation_triton_num_warps,
        "correlation_precision": args.correlation_precision,
        "half_fft_shape_mode": args.half_fft_shape_mode,
        "mixed_precision_fallback_to_float32": args.mixed_precision_fallback_to_float32,
        "mixed_precision_range_scaling": args.mixed_precision_range_scaling,
        "mixed_precision_l1_target": args.mixed_precision_l1_target,
        "mixed_precision_health_check_batches": args.mixed_precision_health_check_batches,
        "pixel_size_backend": args.pixel_size_backend,
        "orientation_batch_size": args.orientation_batch_size,
        "ccf_stack_mode": args.ccf_stack_mode,
        "ccf_stack_path": args.ccf_stack_path,
        "first_orientation": args.first_orientation,
        "last_orientation": args.last_orientation,
        "cpu_threads": args.cpu_threads,
        "histogram_mode": args.histogram_mode,
        "histogram_backend": args.histogram_backend,
        "histogram_sample_orientation_stride": args.histogram_sample_orientation_stride,
        "histogram_sample_pixel_stride": args.histogram_sample_pixel_stride,
    }
    for key, value in runtime.items():
        if value is not None:
            setattr(cfg.runtime, key, value)
    devices = _parse_devices(args.devices)
    if devices is not None:
        cfg.runtime.devices = devices
    search = {
        "angular_step_deg": args.angular_step_deg,
        "in_plane_step_deg": args.in_plane_step_deg,
        "symmetry": args.symmetry,
        "apply_result_rescaling": args.apply_result_rescaling,
        "disable_flat_fielding": args.disable_flat_fielding,
        "padding_mode": args.padding_mode,
        "random_seed": args.random_seed,
        "statistics_roi_mode": args.statistics_roi_mode,
        "statistics_roi_border_pixels": args.statistics_roi_border_pixels,
        "threshold_pixel_mode": args.threshold_pixel_mode,
        "threshold_override": args.threshold_override,
    }
    for key, value in search.items():
        if value is not None:
            setattr(cfg.search, key, value)
    weighting = {
        "mode": args.weighting_mode,
        "kk": args.gisspa_kk,
        "a": args.gisspa_a,
        "b": args.gisspa_b,
        "b2": args.gisspa_b2,
        "bfactor": args.gisspa_bfactor,
        "bfactor2": args.gisspa_bfactor2,
        "bfactor3": args.gisspa_bfactor3,
        "cosine_edge_width_pixels": args.gisspa_cosine_edge_width_pixels,
    }
    for key, value in weighting.items():
        if value is not None:
            setattr(cfg.weighting, key, value)
    template = {
        "source": args.template_source,
        "pdb_path": args.template_pdb,
        "box_size": args.template_box_size,
        "pixel_size_angstrom": args.template_pixel_size,
        "resolution_angstrom": args.template_resolution,
        "center": args.template_center,
        "include_hetatm": args.template_include_hetatm,
        "center_reference_json": args.template_center_reference_json,
        "center_metadata_path": args.template_center_metadata_path,
        "generated_mrc_path": args.template_generated_mrc_path,
        "reuse_generated": args.template_reuse_generated,
        "y_flip": args.template_y_flip,
        "raster_backend": args.template_raster_backend,
        "atom_batch_size": args.template_atom_batch_size,
    }
    for key, value in template.items():
        if value is not None:
            setattr(cfg.template, key, value)
    if args.template_include_chains is not None:
        cfg.template.include_chains = [v for v in args.template_include_chains.split(",")]
    if args.template_exclude_chains is not None:
        cfg.template.exclude_chains = [v for v in args.template_exclude_chains.split(",")]
    batch = {
        "micrograph_root": args.micrograph_root,
        "output_subdirectories": args.batch_output_subdirectories,
        "continue_on_error": args.batch_continue_on_error,
        "write_resolved_configs": args.batch_write_resolved_configs,
        "manifest_prefix": args.batch_manifest_prefix,
        "first_micrograph": args.first_micrograph,
        "last_micrograph": args.last_micrograph,
    }
    for key, value in batch.items():
        if value is not None:
            setattr(cfg.batch, key, value)
    debug = {
        "debug_dir": args.debug_dir,
        "level": args.debug_level,
        "stop_after_stage": args.stop_after_stage,
        "compare_projectors": args.compare_projectors,
        "save_tensors": args.save_debug_tensors,
        "record_timing": args.record_timing,
    }
    for key, value in debug.items():
        if value is not None:
            setattr(cfg.debug, key, value)
    timing = {
        "enabled": args.timing,
        "mode": args.timing_mode,
        "output_dir": args.timing_output_dir,
        "warmup_batches": args.timing_warmup_batches,
        "sample_every_batches": args.timing_sample_every_batches,
        "max_samples": args.timing_max_samples,
        "progress_every_batches": args.timing_progress_every_batches,
        "print_summary": args.timing_print_summary,
        "save_batch_samples": args.timing_save_batch_samples,
        "record_cuda_memory": args.timing_record_cuda_memory,
        "emit_nvtx": args.timing_emit_nvtx,
    }
    for key, value in timing.items():
        if value is not None:
            setattr(cfg.timing, key, value)
    if args.single_orientation is not None:
        phi, theta, psi = args.single_orientation
        cfg.debug.single_orientation_phi_deg = phi
        cfg.debug.single_orientation_theta_deg = theta
        cfg.debug.single_orientation_psi_deg = psi
        if not cfg.debug.debug_dir:
            cfg.debug.debug_dir = str(Path(cfg.output_dir) / f"{cfg.output_prefix}_debug")
    cfg.validate(require_paths=require_paths)
    return cfg


def _symmetric_count(search_range: float, step: float) -> int:
    if search_range <= 0 or step <= 0:
        return 1
    return 2 * int(search_range / step + 0.5) + 1


def _plan_single(cfg: MatchConfig) -> dict[str, Any]:
    image, image_pixel = read_search_image(cfg.input_mrc, cfg.search.image_slice, cfg.search.pixel_size_angstrom)
    template, template_pixel = read_template_volume(cfg.template_mrc, cfg.search.pixel_size_angstrom)
    image_fft_size_mode = effective_image_fft_size_mode(cfg.search, cfg.runtime)
    sizer = DataSizer(
        tuple(image.shape),
        tuple(template.shape),
        cfg.search,
        image_fft_size_mode=image_fft_size_mode,
    )
    if cfg.debug.single_orientation is not None:
        number_orientations = 1
        number_out_of_plane = 1
        number_psi = 1
        angular = 0.0
        psi = 0.0
    else:
        radius = cfg.search.particle_radius_angstrom if cfg.search.particle_radius_angstrom >= 1.0 else 200.0
        angular = cfg.search.angular_step_deg or cisTEM_auto_angular_step(
            sizer.plan.high_resolution_limit_angstrom, radius
        )
        psi = cfg.search.in_plane_step_deg or cisTEM_auto_psi_step(
            sizer.plan.search_pixel_size_angstrom, radius
        )
        euler_search = EulerSearch(
            cfg.search.symmetry,
            angular,
            psi,
            preserve_psi_360_duplicate=cfg.search.preserve_psi_360_duplicate,
            extend_cyclic_theta_to_180=True,
        )
        number_out_of_plane = euler_search.number_of_out_of_plane_positions
        number_psi = euler_search.number_of_psi_positions
        number_orientations = euler_search.number_of_orientations
    number_defocus = _symmetric_count(cfg.search.defocus_search_range_angstrom, cfg.search.defocus_step_angstrom)
    number_pixel = _symmetric_count(cfg.search.pixel_size_search_range_angstrom, cfg.search.pixel_size_step_angstrom)
    total_tasks = number_orientations * number_defocus * number_pixel
    h, w = sizer.plan.search_shape
    return {
        "input_shape": list(image.shape),
        "template_shape": list(template.shape),
        "input_header_pixel_size_angstrom": image_pixel,
        "template_header_pixel_size_angstrom": template_pixel,
        "sizing": sizer.plan.to_dict(),
        "angular_step_deg": angular,
        "psi_step_deg": psi,
        "number_of_out_of_plane_positions": number_out_of_plane,
        "last_out_of_plane_grid_index": number_out_of_plane - 1,
        "number_of_psi_positions": number_psi,
        "number_of_orientations": number_orientations,
        "ccf_search_grid_shape": list(sizer.plan.search_shape),
        "output_grid_shape": list(
            sizer.plan.output_shape_with_rescaling
            if cfg.search.apply_result_rescaling
            else sizer.plan.output_shape_without_rescaling
        ),
        "search_pixel_size_angstrom": sizer.plan.search_pixel_size_angstrom,
        "output_pixel_size_angstrom": (
            cfg.search.pixel_size_angstrom
            if cfg.search.apply_result_rescaling
            else sizer.plan.search_pixel_size_angstrom
        ),
        "number_of_defocus_positions": number_defocus,
        "number_of_pixel_size_positions": number_pixel,
        "number_of_ccf_images": total_tasks,
        "orientation_batch_size": cfg.runtime.orientation_batch_size,
        "correlation_precision_requested": cfg.runtime.correlation_precision,
        "half_fft_shape_mode": cfg.runtime.half_fft_shape_mode,
        "mixed_precision_range_scaling": cfg.runtime.mixed_precision_range_scaling,
        "mixed_precision_l1_target": cfg.runtime.mixed_precision_l1_target,
        "mixed_precision_health_check_batches": cfg.runtime.mixed_precision_health_check_batches,
        "fft_size_mode_requested": cfg.search.fft_size_mode,
        "fft_size_mode_planned": image_fft_size_mode,
        "center_phase_mode": cfg.runtime.center_phase_mode,
        "correlation_multiply_backend": cfg.runtime.correlation_multiply_backend,
        "weighting": asdict(cfg.weighting),
        "threshold_override": cfg.search.threshold_override,
        "threshold_mode": (
            "cistem_theoretical" if cfg.search.threshold_override is None else "json_override"
        ),
        "number_of_orientation_batches": (
            (number_orientations + cfg.runtime.orientation_batch_size - 1)
            // cfg.runtime.orientation_batch_size
            * number_defocus
            * number_pixel
        ),
        "timing": asdict(cfg.timing),
        "number_of_valid_search_pixels": sizer.plan.number_of_valid_search_pixels,
        "number_of_mip_valid_search_pixels": sizer.plan.number_of_valid_search_pixels,
        "number_of_statistics_pixels": sizer.plan.number_of_statistics_pixels,
        "number_of_threshold_pixels": sizer.plan.number_of_threshold_pixels,
        "number_of_full_search_grid_pixels": h * w,
        "histogram_mode": cfg.runtime.histogram_mode,
        "histogram_backend": cfg.runtime.histogram_backend,
        "number_of_full_histogram_population_samples": sizer.plan.number_of_statistics_pixels * total_tasks,
        "number_of_observed_histogram_samples": (
            0
            if cfg.runtime.histogram_mode == "off"
            else sizer.plan.number_of_statistics_pixels * total_tasks
        ),
        "number_of_independent_trials": (
            sizer.plan.number_of_threshold_pixels
            * total_tasks
            * cfg.search.fraction_of_search_positions_independent
            / (number_defocus if cfg.search.ignore_defocus_for_threshold else 1)
        ),
        "number_of_full_search_grid_trials": (
            h
            * w
            * total_tasks
            * cfg.search.fraction_of_search_positions_independent
            / (number_defocus if cfg.search.ignore_defocus_for_threshold else 1)
        ),
        "ccf_stack_float32_bytes": total_tasks * h * w * 4,
        "ccf_stack_float32_gib": total_tasks * h * w * 4 / 1024**3,
    }


def _plan(cfg: MatchConfig) -> dict[str, Any]:
    if cfg.micrographs_star:
        return plan_batch(cfg)
    resolved, template_result = resolve_template_for_match(cfg)
    payload = _plan_single(resolved)
    if template_result is not None:
        payload["pdb_template"] = template_result.to_dict()
    return payload


def _add_common_run_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-mrc")
    parser.add_argument("--micrographs-star")
    parser.add_argument("--template-mrc")
    parser.add_argument("--template-source", choices=("mrc", "pdb"))
    parser.add_argument("--template-pdb")
    parser.add_argument("--template-box-size", type=int)
    parser.add_argument("--template-pixel-size", type=float)
    parser.add_argument("--template-resolution", type=float)
    parser.add_argument("--template-center", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--template-include-hetatm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--template-include-chains", help="comma-separated PDB chain IDs")
    parser.add_argument("--template-exclude-chains", help="comma-separated PDB chain IDs")
    parser.add_argument("--template-center-reference-json")
    parser.add_argument("--template-center-metadata-path")
    parser.add_argument("--template-generated-mrc-path")
    parser.add_argument("--template-reuse-generated", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--template-y-flip", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--template-raster-backend", choices=("vectorized", "reference"))
    parser.add_argument("--template-atom-batch-size", type=int)
    parser.add_argument("--micrograph-root")
    parser.add_argument("--batch-output-subdirectories", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch-continue-on-error", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch-write-resolved-configs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--batch-manifest-prefix")
    parser.add_argument("--first-micrograph", type=int)
    parser.add_argument("--last-micrograph", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--output-prefix")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--devices", help="comma-separated one-server GPU indices, for example 0,1,2,3")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--projector-backend", choices=("auto", "triton", "gather", "grid_sample"))
    parser.add_argument(
        "--triton-fallback-to-gather",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--triton-block-size", type=int)
    parser.add_argument("--triton-num-warps", type=int)
    parser.add_argument(
        "--cache-grid-sample-source",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--cache-sampled-projection-filter",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--cache-orientation-tensors",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--precompute-rotation-matrices",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--orientation-precompute-chunk-size", type=int)
    parser.add_argument(
        "--projection-fft-mode",
        choices=("auto", "direct_s", "explicit"),
        help="auto/direct_s uses rfft2(s=...) plus an exact center-shift phase; explicit retains the padded-real reference path",
    )
    parser.add_argument(
        "--center-phase-mode",
        choices=("auto", "input", "projection"),
        help="input prephases the micrograph once; projection retains the v0.3.3 per-batch order",
    )
    parser.add_argument(
        "--correlation-multiply-backend",
        choices=("auto", "triton", "torch"),
        help="fused image * conjugate(projection) implementation",
    )
    parser.add_argument(
        "--correlation-triton-fallback-to-torch",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--correlation-triton-block-size", type=int)
    parser.add_argument("--correlation-triton-num-warps", type=int)
    parser.add_argument(
        "--correlation-precision",
        choices=("float32", "mixed_float16"),
        help="mixed_float16 applies half/chalf only to the large correlation FFT path",
    )
    parser.add_argument(
        "--half-fft-shape-mode",
        choices=("pad_to_power2", "require_power2", "fallback_float32"),
        help="policy for non-power-of-two work grids in mixed_float16 mode",
    )
    parser.add_argument(
        "--mixed-precision-fallback-to-float32",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--mixed-precision-range-scaling",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--mixed-precision-l1-target", type=float)
    parser.add_argument("--mixed-precision-health-check-batches", type=int)
    parser.add_argument("--pixel-size-backend", choices=("cistem_exact", "coordinate"))
    parser.add_argument("--orientation-batch-size", type=int)
    parser.add_argument("--ccf-stack-mode", choices=("none", "cpu", "mrc"))
    parser.add_argument("--ccf-stack-path")
    parser.add_argument("--first-orientation", type=int)
    parser.add_argument("--last-orientation", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument(
        "--histogram-mode",
        choices=("exact", "sampled", "off"),
        help="exact=all CCC samples; sampled=deterministic weighted diagnostic; off=skip empirical histogram",
    )
    parser.add_argument(
        "--histogram-backend",
        choices=("histc", "bincount"),
        help="histc is the optimized exact backend; bincount retains the legacy implementation for validation",
    )
    parser.add_argument("--histogram-sample-orientation-stride", type=int)
    parser.add_argument("--histogram-sample-pixel-stride", type=int)
    parser.add_argument("--angular-step-deg", type=float)
    parser.add_argument("--in-plane-step-deg", type=float)
    parser.add_argument("--symmetry")
    parser.add_argument(
        "--apply-result-rescaling",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--disable-flat-fielding",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--padding-mode", choices=("noise", "zero", "edge", "replicate"))
    parser.add_argument("--random-seed", type=int)
    parser.add_argument(
        "--statistics-roi-mode", choices=("source", "precompiled", "full")
    )
    parser.add_argument("--statistics-roi-border-pixels", type=int)
    parser.add_argument(
        "--threshold-pixel-mode", choices=("source", "statistics", "full")
    )
    parser.add_argument(
        "--threshold-override",
        type=float,
        help="fixed score threshold; omit/null to use the cisTEM theoretical value",
    )
    parser.add_argument("--weighting-mode", choices=("cistem", "gisspa_repo"))
    parser.add_argument("--gisspa-kk", type=float)
    parser.add_argument("--gisspa-a", type=float)
    parser.add_argument("--gisspa-b", type=float)
    parser.add_argument("--gisspa-b2", type=float)
    parser.add_argument("--gisspa-bfactor", type=float)
    parser.add_argument("--gisspa-bfactor2", type=float)
    parser.add_argument("--gisspa-bfactor3", type=float)
    parser.add_argument("--gisspa-cosine-edge-width-pixels", type=float)
    parser.add_argument("--debug-dir")
    parser.add_argument("--debug-level", type=int)
    parser.add_argument("--stop-after-stage", type=int, choices=range(0, 11))
    parser.add_argument("--single-orientation", nargs=3, type=float, metavar=("PHI", "THETA", "PSI"))
    parser.add_argument("--compare-projectors", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-debug-tensors", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--record-timing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--timing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable the independent low-overhead performance profiler",
    )
    parser.add_argument(
        "--timing-mode",
        choices=("summary", "sampled", "synchronized"),
        help="summary=stage wall time; sampled=selected CUDA batches; synchronized=every batch",
    )
    parser.add_argument("--timing-output-dir")
    parser.add_argument("--timing-warmup-batches", type=int)
    parser.add_argument("--timing-sample-every-batches", type=int)
    parser.add_argument("--timing-max-samples", type=int)
    parser.add_argument("--timing-progress-every-batches", type=int)
    parser.add_argument("--timing-print-summary", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--timing-save-batch-samples", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--timing-record-cuda-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--timing-emit-nvtx", action=argparse.BooleanOptionalAction, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cistem2dtm",
        description="PyTorch cisTEM 2-D template matching pinned to source commit 5bc5f8c",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run template matching from a JSON configuration")
    run.add_argument("config", type=Path)
    _add_common_run_overrides(run)

    plan = sub.add_parser("plan", help="show sizing, angular count, task count, and optional CCF-stack size")
    plan.add_argument("config", type=Path)
    _add_common_run_overrides(plan)

    init = sub.add_parser("init-config", help="write a documented default JSON configuration")
    init.add_argument("path", type=Path)
    init.add_argument("--force", action="store_true")

    example = sub.add_parser("make-example", help="create a small synthetic MRC/template/config example")
    example.add_argument("output_dir", type=Path)
    example.add_argument("--template-size", type=int, default=12)
    example.add_argument("--search-size", type=int)
    example.add_argument("--pixel-size", type=float, default=1.5)
    example.add_argument("--noise-sigma", type=float, default=0.08)
    example.add_argument("--seed", type=int, default=7)

    compare = sub.add_parser("compare-debug", help="compare matching NPY checkpoints from two 10-stage debug directories")
    compare.add_argument("reference_dir", type=Path)
    compare.add_argument("candidate_dir", type=Path)
    compare.add_argument("--rtol", type=float, default=1.0e-5)
    compare.add_argument("--atol", type=float, default=1.0e-6)
    compare.add_argument("--pattern", default="*.npy")
    compare.add_argument("--report-prefix", type=Path)
    compare.add_argument("--fail-on-difference", action="store_true")

    stages = sub.add_parser("stages", help="print the ten numerical validation checkpoints")
    stages.add_argument("--json", action="store_true")

    diagnose = sub.add_parser(
        "diagnose-histogram",
        help="factor the effective trial count in a cisTEM/PyTorch histogram against a JSON config",
    )
    diagnose.add_argument("histogram", type=Path)
    diagnose.add_argument("config", type=Path)
    _add_common_run_overrides(diagnose)

    prepare_template = sub.add_parser(
        "prepare-template",
        help="generate/cache a 3-D MRC from template.source='pdb' without running 2DTM",
    )
    prepare_template.add_argument("config", type=Path)
    _add_common_run_overrides(prepare_template)

    self_test = sub.add_parser("self-test", help="generate and run a compact CPU end-to-end smoke test")
    self_test.add_argument("--output-dir", type=Path)
    self_test.add_argument("--keep", action="store_true", help="keep a temporary directory when --output-dir is omitted")
    return parser


def _default_config() -> MatchConfig:
    cfg = MatchConfig(input_mrc="search.mrc", template_mrc="template.mrc")
    cfg.debug.debug_dir = None
    return cfg


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init-config":
            if args.path.exists() and not args.force:
                raise FileExistsError(f"{args.path} already exists; pass --force to overwrite")
            args.path.parent.mkdir(parents=True, exist_ok=True)
            _default_config().save_json(args.path)
            print(args.path)
            return 0

        if args.command == "make-example":
            files = generate_synthetic_example(
                args.output_dir,
                template_size=args.template_size,
                search_size=args.search_size,
                pixel_size_angstrom=args.pixel_size,
                noise_sigma=args.noise_sigma,
                seed=args.seed,
            )
            print(json.dumps({key: str(value) for key, value in files.items()}, indent=2))
            return 0

        if args.command == "compare-debug":
            report = compare_debug_directories(
                args.reference_dir,
                args.candidate_dir,
                rtol=args.rtol,
                atol=args.atol,
                pattern=args.pattern,
                report_prefix=args.report_prefix,
            )
            print(json.dumps(report, indent=2))
            return 1 if args.fail_on_difference and not report["passed"] else 0

        if args.command == "stages":
            if args.json:
                print(json.dumps(STAGE_NAMES, indent=2))
            else:
                for number, name in STAGE_NAMES.items():
                    print(f"{number:2d}  {name}")
            return 0

        if args.command in {"run", "plan", "diagnose-histogram", "prepare-template"}:
            require_paths = args.command != "prepare-template"
            cfg = _apply_run_overrides(
                MatchConfig.load_json(args.config, require_paths=require_paths),
                args,
                require_paths=require_paths,
            )
            if args.command == "prepare-template":
                if cfg.template.source != "pdb":
                    raise ValueError("prepare-template requires template.source='pdb'")
                result = generate_pdb_template(cfg)
                print(json.dumps(result.to_dict(), indent=2))
                return 0
            plan_payload = _plan(cfg)
            if args.command == "plan":
                print(json.dumps(plan_payload, indent=2))
                return 0
            if args.command == "diagnose-histogram":
                if cfg.micrographs_star:
                    raise ValueError("diagnose-histogram requires a single input_mrc configuration")
                summary = read_histogram_summary(
                    args.histogram,
                    expected_false_positives=cfg.search.expected_false_positives,
                )
                report = diagnose_histogram_against_plan(
                    summary,
                    plan_payload,
                    independent_fraction=cfg.search.fraction_of_search_positions_independent,
                    ignore_defocus_for_threshold=cfg.search.ignore_defocus_for_threshold,
                )
                print(json.dumps(report, indent=2))
                return 0 if report["comparison"]["matches_within_0p1_percent"] else 1
            if cfg.micrographs_star:
                result = run_batch(cfg)
                print(json.dumps(result.to_dict(), indent=2))
                return 0
            resolved, template_result = resolve_template_for_match(cfg)
            result = TemplateMatcher(resolved).run()
            if template_result is not None:
                template_payload = template_result.to_dict()
                template_record = write_json(
                    Path(resolved.output_dir) / f"{resolved.output_prefix}_pdb_template.json",
                    template_payload,
                )
                result.files["pdb_template"] = template_record
                result.metadata["pdb_template"] = template_payload
                metadata_path = result.files.get("metadata")
                if metadata_path is not None:
                    write_json(metadata_path, result.metadata)
            print(json.dumps(result.to_dict(), indent=2))
            return 0

        if args.command == "self-test":
            temporary: tempfile.TemporaryDirectory[str] | None = None
            if args.output_dir is None:
                temporary = tempfile.TemporaryDirectory(prefix="cistem2dtm_selftest_")
                root = Path(temporary.name)
            else:
                root = args.output_dir
            files = generate_synthetic_example(root)
            cfg = MatchConfig.load_json(files["config"])
            cfg.runtime.device = "cpu"
            cfg.runtime.devices = []
            result = TemplateMatcher(cfg).run()
            payload = {"root": str(root), "result": result.to_dict()}
            print(json.dumps(payload, indent=2))
            if temporary is not None and args.keep:
                # TemporaryDirectory cannot be detached portably; copy it to a
                # stable sibling instead.
                kept = Path.cwd() / root.name
                import shutil

                shutil.copytree(root, kept, dirs_exist_ok=True)
                print(f"kept at {kept}", file=sys.stderr)
            if temporary is not None:
                temporary.cleanup()
            return 0

        parser.error(f"unhandled command {args.command}")
    except DebugStop as stop:
        print(json.dumps({"status": "debug_stop", "completed_stage": stop.stage}, indent=2))
        return 0
    except Exception as exc:
        print(f"cistem2dtm: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
