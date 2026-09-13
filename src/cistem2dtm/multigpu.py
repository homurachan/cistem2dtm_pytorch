from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .config import DebugConfig, MatchConfig
from .debug import DebugRecorder
from .matcher import RawRunOutput, TemplateMatcher
from .stack import CCFStackResult, merge_ccf_stack_shards
from .statistics import RawSearchResult, merge_raw_results


def _raw_payload(raw: RawSearchResult) -> dict[str, Any]:
    cpu = raw.cpu()
    return {
        "mip": cpu.mip,
        "phi": cpu.phi,
        "theta": cpu.theta,
        "psi": cpu.psi,
        "defocus": cpu.defocus,
        "pixel_size": cpu.pixel_size,
        "correlation_sum": cpu.correlation_sum,
        "correlation_sum_squares": cpu.correlation_sum_squares,
        "histogram": cpu.histogram,
        "number_searched": int(cpu.number_searched),
        "winner_task_index": cpu.winner_task_index,
    }


def _raw_from_payload(payload: dict[str, Any]) -> RawSearchResult:
    return RawSearchResult(
        mip=payload["mip"],
        phi=payload["phi"],
        theta=payload["theta"],
        psi=payload["psi"],
        defocus=payload["defocus"],
        pixel_size=payload["pixel_size"],
        correlation_sum=payload["correlation_sum"],
        correlation_sum_squares=payload["correlation_sum_squares"],
        histogram=payload["histogram"],
        number_searched=int(payload["number_searched"]),
        winner_task_index=payload["winner_task_index"],
    )


def _stack_payload(result: CCFStackResult) -> dict[str, Any]:
    return {
        "mode": result.mode,
        "mrc_path": str(result.mrc_path) if result.mrc_path else None,
        "tensor_path": str(result.tensor_path) if result.tensor_path else None,
        "metadata_path": str(result.metadata_path) if result.metadata_path else None,
        "number_written": int(result.number_written),
    }


def _stack_from_payload(payload: dict[str, Any]) -> CCFStackResult:
    return CCFStackResult(
        mode=payload["mode"],
        mrc_path=Path(payload["mrc_path"]) if payload.get("mrc_path") else None,
        tensor_path=Path(payload["tensor_path"]) if payload.get("tensor_path") else None,
        metadata_path=Path(payload["metadata_path"]) if payload.get("metadata_path") else None,
        number_written=int(payload.get("number_written", 0)),
    )


def _worker(
    rank: int,
    gpu_index: int,
    config_dict: dict[str, Any],
    orientation_indices: list[int],
    temp_dir: str,
    stack_path: str | None,
) -> None:
    temp = Path(temp_dir)
    status_path = temp / f"status_rank{rank}.json"
    partial_path = temp / f"partial_rank{rank}.pt"
    try:
        torch.cuda.set_device(int(gpu_index))
        cfg = MatchConfig.from_dict(config_dict)
        cfg.runtime.devices = []
        cfg.runtime.device = f"cuda:{gpu_index}"
        cfg.runtime.first_orientation = 0
        cfg.runtime.last_orientation = None
        # Each worker has a private debug directory; only rank 0 performs the
        # expensive independent gather/grid_sample comparison.
        if cfg.debug.enabled:
            root = Path(cfg.debug.debug_dir or "2dtm_debug")
            cfg.debug.debug_dir = str(root / f"gpu_{rank}")
            if rank != 0:
                cfg.debug.compare_projectors = False
        matcher = TemplateMatcher(cfg)
        run = matcher.run_raw(
            device_override=f"cuda:{gpu_index}",
            orientation_indices=orientation_indices,
            stack_path_override=stack_path,
        )
        torch.cuda.synchronize(gpu_index)
        if run.profiler is not None:
            run.profiler.finish_run()
            worker_timing = run.profiler.report()
        else:
            worker_timing = dict(run.timing_report)
        payload = {
            "raw": _raw_payload(run.raw),
            "selected_orientation_indices": list(run.selected_orientation_indices),
            "defocus_offsets": list(run.defocus_offsets),
            "pixel_size_offsets": list(run.pixel_size_offsets),
            "pixel_size_resampling": run.pixel_size_resampling,
            "search_spacing": run.search_spacing,
            "elapsed_seconds": float(run.elapsed_seconds),
            "ccf_stack": _stack_payload(run.ccf_stack),
            "timing_report": worker_timing,
        }
        torch.save(payload, partial_path)
        status_path.write_text(
            json.dumps({"ok": True, "partial_path": str(partial_path), "gpu": gpu_index}, indent=2),
            encoding="utf-8",
        )
    except BaseException as exc:  # propagate a useful worker traceback to the parent
        status_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "gpu": gpu_index,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise


def _split_contiguous(indices: Sequence[int], pieces: int) -> list[list[int]]:
    if pieces <= 0:
        raise ValueError("pieces must be positive")
    return [part.astype(np.int64).tolist() for part in np.array_split(np.asarray(indices, dtype=np.int64), pieces) if len(part)]


def run_multi_gpu(config: MatchConfig):
    """Run orientation shards on multiple GPUs in one server and merge exactly.

    Search tasks retain the source ordering through their global ``task_index``.
    MIP ties are resolved using that index, so the merged result matches a single
    process with strict ``>`` semantics even though workers finish independently.
    """
    config.validate()
    requested = [int(v) for v in config.runtime.devices]
    if len(requested) < 2:
        raise ValueError("run_multi_gpu requires at least two GPU indices")
    if not torch.cuda.is_available():
        raise RuntimeError("multi-GPU execution requested but torch.cuda.is_available() is false")
    visible = torch.cuda.device_count()
    if any(v < 0 or v >= visible for v in requested):
        raise RuntimeError(f"requested GPUs {requested}, but only {visible} CUDA devices are visible")
    if config.debug.stop_after_stage:
        raise ValueError("debug.stop_after_stage is not supported with multi-GPU execution; use one GPU for staged stops")

    # Build the exact global Euler list and DataSizer on CPU.  Workers independently
    # run the same deterministic preparation on their assigned devices.
    parent_cfg = MatchConfig.from_dict(config.to_dict())
    parent_cfg.runtime.devices = []
    parent_cfg.runtime.device = "cpu"
    original_debug = parent_cfg.debug
    parent_cfg.debug = DebugConfig()
    parent_matcher = TemplateMatcher(parent_cfg)
    prepared = parent_matcher.prepare(device_override="cpu")
    orientations_all, _ = parent_matcher.build_orientations(prepared)
    total = len(orientations_all)
    first = min(config.runtime.first_orientation, total)
    last_exclusive = total if config.runtime.last_orientation is None else min(config.runtime.last_orientation + 1, total)
    selected_indices = list(range(first, last_exclusive))
    if not selected_indices:
        raise ValueError("selected orientation range is empty")

    active_devices = requested[: min(len(requested), len(selected_indices))]
    chunks = _split_contiguous(selected_indices, len(active_devices))
    active_devices = active_devices[: len(chunks)]
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / f".cistem2dtm_multigpu_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)

    shard_paths: list[str | None] = []
    for rank in range(len(chunks)):
        if config.runtime.ccf_stack_mode == "none":
            shard_paths.append(None)
        else:
            suffix = ".mrc" if config.runtime.ccf_stack_mode == "mrc" else ".pt"
            shard_paths.append(str(temp_dir / f"ccf_rank{rank}{suffix}"))

    wall_start = time.perf_counter()
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for rank, (gpu_index, indices, stack_path) in enumerate(zip(active_devices, chunks, shard_paths)):
        process = context.Process(
            target=_worker,
            args=(rank, gpu_index, config.to_dict(), indices, str(temp_dir), stack_path),
            daemon=False,
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    failures: list[str] = []
    payloads: list[dict[str, Any]] = []
    for rank, process in enumerate(processes):
        status_path = temp_dir / f"status_rank{rank}.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {
            "ok": False,
            "error": f"worker exited with code {process.exitcode} without a status file",
        }
        if process.exitcode != 0 or not status.get("ok", False):
            failures.append(status.get("traceback") or status.get("error") or f"rank {rank} failed")
            continue
        payloads.append(torch.load(status["partial_path"], map_location="cpu", weights_only=True))
    if failures:
        diagnostic = temp_dir / "FAILED.txt"
        diagnostic.write_text("\n\n".join(failures), encoding="utf-8")
        raise RuntimeError(f"one or more multi-GPU workers failed; diagnostics: {diagnostic}\n{failures[0]}")

    partial_raw = [_raw_from_payload(payload["raw"]) for payload in payloads]
    merged_raw = merge_raw_results(partial_raw, device=torch.device("cpu"))
    ccf_shards = [_stack_from_payload(payload["ccf_stack"]) for payload in payloads]
    if config.runtime.ccf_stack_mode == "none":
        ccf_result = CCFStackResult(mode="none")
    else:
        if config.runtime.ccf_stack_path:
            final_stack_path = Path(config.runtime.ccf_stack_path)
        else:
            suffix = ".mrc" if config.runtime.ccf_stack_mode == "mrc" else ".pt"
            final_stack_path = output_dir / f"{config.output_prefix}_ccf_stack{suffix}"
        ccf_result = merge_ccf_stack_shards(
            ccf_shards,
            output_path=final_stack_path,
            image_shape=prepared.data_sizer.plan.search_shape,
            pixel_size_angstrom=prepared.data_sizer.plan.search_pixel_size_angstrom,
            remove_shards=True,
        )
        # The tensor has already been serialized; avoid retaining an optional
        # multi-gigabyte CCF stack in the final MatchResult.
        if ccf_result.mode == "cpu":
            ccf_result.tensor = None

    first_payload = payloads[0]
    if original_debug.enabled:
        prepared.debug = DebugRecorder(original_debug, config.search.pixel_size_angstrom)
        prepared.debug.save(
            9,
            "multigpu_shards",
            {
                "devices": active_devices,
                "orientation_ranges": [[chunk[0], chunk[-1]] for chunk in chunks],
                "worker_elapsed_seconds": [float(payload["elapsed_seconds"]) for payload in payloads],
            },
        )

    combined = RawRunOutput(
        raw=merged_raw,
        prepared=prepared,
        orientations=[orientations_all[index] for index in selected_indices],
        all_orientation_count=total,
        selected_orientation_indices=selected_indices,
        defocus_offsets=list(first_payload["defocus_offsets"]),
        pixel_size_offsets=list(first_payload["pixel_size_offsets"]),
        ccf_stack=ccf_result,
        elapsed_seconds=time.perf_counter() - wall_start,
        pixel_size_resampling=list(first_payload["pixel_size_resampling"]),
        execution_devices=[f"cuda:{index}" for index in active_devices],
        search_spacing=dict(first_payload.get("search_spacing", {})),
        timing_report={
            "schema_version": 1,
            "mode": "multi_gpu",
            "wall_seconds": time.perf_counter() - wall_start,
            "devices": [f"cuda:{index}" for index in active_devices],
            "orientation_ranges": [[chunk[0], chunk[-1]] for chunk in chunks],
            "worker_reports": [payload.get("timing_report", {}) for payload in payloads],
        } if config.timing.enabled or config.debug.record_timing else {},
    )
    # Finalization is intentionally done once on CPU after deterministic merging.
    final_matcher = TemplateMatcher(config)
    result = final_matcher.finalize(combined)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return result
