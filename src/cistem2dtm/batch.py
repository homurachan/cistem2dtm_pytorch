from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .config import MatchConfig
from .io import write_json, write_rows_tsv
from .matcher import TemplateMatcher
from .pdb_template import PDBTemplateResult, resolve_template_for_match
from .relion_star import MicrographRecord, read_relion_micrographs


@dataclass(slots=True)
class BatchJob:
    index: int
    record: MicrographRecord
    config: MatchConfig
    template_info: dict[str, Any] | None

    def to_plan_dict(self) -> dict[str, Any]:
        return {
            **self.record.to_dict(),
            "output_dir": self.config.output_dir,
            "output_prefix": self.config.output_prefix,
            "template_mrc": self.config.template_mrc,
            "template_info": self.template_info,
        }


@dataclass(slots=True)
class BatchRunResult:
    star_path: str
    jobs: list[dict[str, Any]]
    manifest_json: str
    manifest_tsv: str
    successful: int
    failed: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "star_path": self.star_path,
            "successful": self.successful,
            "failed": self.failed,
            "elapsed_seconds": self.elapsed_seconds,
            "manifest_json": self.manifest_json,
            "manifest_tsv": self.manifest_tsv,
            "jobs": self.jobs,
        }


def _safe_relative_stem(name: str, index: int) -> Path:
    raw = Path(name)
    parts = [part for part in raw.parts if part not in {"", ".", "..", raw.anchor}]
    if not parts:
        return Path(f"micrograph_{index:06d}")
    clean = Path(*parts)
    return clean.with_suffix("")


def _per_job_output(base: MatchConfig, record: MicrographRecord) -> Path:
    root = Path(base.output_dir)
    if not base.batch.output_subdirectories:
        return root
    return root / _safe_relative_stem(record.micrograph_name, record.index)


def _redirect_optional_output_paths(cfg: MatchConfig) -> None:
    output_dir = Path(cfg.output_dir)
    if cfg.runtime.ccf_stack_path:
        cfg.runtime.ccf_stack_path = str(output_dir / Path(cfg.runtime.ccf_stack_path).name)
    if cfg.debug.enabled:
        name = Path(cfg.debug.debug_dir or f"{cfg.output_prefix}_debug").name
        cfg.debug.debug_dir = str(output_dir / name)
    if cfg.timing.output_dir:
        cfg.timing.output_dir = str(output_dir / Path(cfg.timing.output_dir).name)


def config_for_micrograph(base: MatchConfig, record: MicrographRecord) -> MatchConfig:
    cfg = MatchConfig.from_dict(base.to_dict())
    cfg.micrographs_star = ""
    cfg.input_mrc = str(record.micrograph_path)
    cfg.output_dir = str(_per_job_output(base, record))
    if not base.batch.output_subdirectories:
        flattened = "_".join(_safe_relative_stem(record.micrograph_name, record.index).parts)
        cfg.output_prefix = f"{base.output_prefix}_{flattened}"
    cfg.search.pixel_size_angstrom = float(record.pixel_size_angstrom)
    cfg.microscope.voltage_kv = float(record.voltage_kv)
    cfg.microscope.spherical_aberration_mm = float(record.spherical_aberration_mm)
    cfg.microscope.amplitude_contrast = float(record.amplitude_contrast)
    cfg.microscope.defocus1_angstrom = float(record.defocus_u_angstrom)
    cfg.microscope.defocus2_angstrom = float(record.defocus_v_angstrom)
    cfg.microscope.defocus_angle_deg = float(record.defocus_angle_deg)
    cfg.microscope.phase_shift_deg = float(record.phase_shift_deg)
    cfg.runtime.devices = []
    _redirect_optional_output_paths(cfg)
    return cfg


def _template_cache_key(cfg: MatchConfig) -> tuple[Any, ...]:
    t = cfg.template
    pixel = t.pixel_size_angstrom if t.pixel_size_angstrom is not None else cfg.search.pixel_size_angstrom
    resolution = t.resolution_angstrom if t.resolution_angstrom is not None else 2.0 * float(pixel)
    return (
        str(Path(t.pdb_path).expanduser()),
        int(t.box_size or 0),
        float(pixel),
        float(resolution),
        bool(t.center),
        bool(t.include_hetatm),
        tuple(t.include_chains),
        tuple(t.exclude_chains),
        str(t.center_reference_json or ""),
        bool(t.y_flip),
        str(t.raster_backend),
    )


def build_batch_jobs(base: MatchConfig, *, generate_templates: bool = True) -> list[BatchJob]:
    if not base.micrographs_star:
        raise ValueError("micrographs_star is required for batch execution")
    records = read_relion_micrographs(base.micrographs_star, base)
    jobs: list[BatchJob] = []
    template_cache: dict[tuple[Any, ...], tuple[str, dict[str, Any] | None]] = {}
    explicit_template_paths: dict[str, tuple[Any, ...]] = {}
    template_device = None
    if base.runtime.devices:
        template_device = f"cuda:{base.runtime.devices[0]}"
    elif base.runtime.device not in {"auto", "cuda"}:
        template_device = base.runtime.device

    for record in records:
        cfg = config_for_micrograph(base, record)
        template_info: dict[str, Any] | None = None
        if cfg.template.source == "pdb":
            key = _template_cache_key(cfg)
            if cfg.template.generated_mrc_path:
                explicit_path = str(Path(cfg.template.generated_mrc_path).expanduser().resolve())
                previous_key = explicit_template_paths.get(explicit_path)
                if previous_key is not None and previous_key != key:
                    raise ValueError(
                        "template.generated_mrc_path is shared by multiple optics/pixel-size "
                        "conditions; leave it null to use automatic per-pixel-size cache names"
                    )
                explicit_template_paths[explicit_path] = key
            cached = template_cache.get(key)
            if cached is None:
                if not generate_templates:
                    pixel = (
                        cfg.template.pixel_size_angstrom
                        if cfg.template.pixel_size_angstrom is not None
                        else cfg.search.pixel_size_angstrom
                    )
                    resolution = (
                        cfg.template.resolution_angstrom
                        if cfg.template.resolution_angstrom is not None
                        else 2.0 * float(pixel)
                    )
                    template_info = {
                        "source": "pdb",
                        "pdb_path": cfg.template.pdb_path,
                        "box_size": cfg.template.box_size,
                        "pixel_size_angstrom": float(pixel),
                        "resolution_angstrom": float(resolution),
                        "generated": False,
                    }
                    template_cache[key] = ("", template_info)
                else:
                    # Generate in the common batch output root, not in each
                    # per-micrograph directory, so all jobs reuse one template
                    # per unique pixel-size/resolution specification.
                    template_cfg = MatchConfig.from_dict(cfg.to_dict())
                    template_cfg.output_dir = base.output_dir
                    resolved, result = resolve_template_for_match(
                        template_cfg,
                        device_override=template_device,
                    )
                    assert result is not None
                    template_info = result.to_dict()
                    template_cache[key] = (resolved.template_mrc, template_info)
                    cfg.template_mrc = resolved.template_mrc
                    cfg.template.source = "mrc"
            else:
                cached_path, template_info = cached
                if generate_templates:
                    cfg.template_mrc = cached_path
                    cfg.template.source = "mrc"
        elif not cfg.template_mrc:
            raise ValueError("template_mrc is required for batch MRC-template mode")
        cfg.validate()
        jobs.append(BatchJob(index=record.index, record=record, config=cfg, template_info=template_info))
    return jobs


def plan_batch(base: MatchConfig) -> dict[str, Any]:
    jobs = build_batch_jobs(base, generate_templates=False)
    optics_groups = sorted({job.record.optics_group for job in jobs})
    pixel_sizes = sorted({job.record.pixel_size_angstrom for job in jobs})
    return {
        "mode": "micrographs_star",
        "star_path": str(Path(base.micrographs_star).expanduser().resolve()),
        "number_of_micrographs": len(jobs),
        "optics_groups": optics_groups,
        "pixel_sizes_angstrom": pixel_sizes,
        "devices": list(base.runtime.devices),
        "scheduling": (
            "one_micrograph_per_gpu_worker"
            if len(base.runtime.devices) > 1
            else "sequential"
        ),
        "jobs": [job.to_plan_dict() for job in jobs],
    }


def _resolved_single_job_config(cfg: MatchConfig, device: str) -> MatchConfig:
    out = MatchConfig.from_dict(cfg.to_dict())
    out.runtime.devices = []
    out.runtime.device = device
    out.validate()
    return out


def _record_manifest_fields(record: MicrographRecord) -> dict[str, Any]:
    return {
        "pixel_size_angstrom": record.pixel_size_angstrom,
        "voltage_kv": record.voltage_kv,
        "spherical_aberration_mm": record.spherical_aberration_mm,
        "amplitude_contrast": record.amplitude_contrast,
        "defocus_u_angstrom": record.defocus_u_angstrom,
        "defocus_v_angstrom": record.defocus_v_angstrom,
        "defocus_angle_deg": record.defocus_angle_deg,
        "phase_shift_deg": record.phase_shift_deg,
        "ctf_max_resolution_angstrom": record.ctf_max_resolution_angstrom,
        "ctf_figure_of_merit": record.ctf_figure_of_merit,
    }


def _run_one(job: BatchJob, device: str) -> dict[str, Any]:
    cfg = _resolved_single_job_config(job.config, device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.batch.write_resolved_configs:
        cfg.save_json(output_dir / f"{cfg.output_prefix}_resolved_config.json")
    started = time.perf_counter()
    result = TemplateMatcher(cfg).run()
    elapsed = time.perf_counter() - started
    return {
        "index": job.index,
        "micrograph_name": job.record.micrograph_name,
        "micrograph_path": str(job.record.micrograph_path),
        "optics_group": job.record.optics_group,
        **_record_manifest_fields(job.record),
        "device": device,
        "status": "ok",
        "elapsed_seconds": elapsed,
        "output_dir": cfg.output_dir,
        "output_prefix": cfg.output_prefix,
        "threshold": result.threshold,
        "number_of_peaks": len(result.peaks),
        "files": {key: str(value) for key, value in result.files.items()},
        "template_info": job.template_info,
        "error": "",
        "traceback": "",
    }


def _gpu_worker(
    gpu_index: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    torch.cuda.set_device(gpu_index)
    device = f"cuda:{gpu_index}"
    while True:
        payload = task_queue.get()
        if payload is None:
            break
        index = int(payload["index"])
        try:
            cfg = MatchConfig.from_dict(payload["config"])
            record = MicrographRecord(
                index=index,
                micrograph_name=payload["record"]["micrograph_name"],
                micrograph_path=Path(payload["record"]["micrograph_path"]),
                optics_group=payload["record"]["optics_group"],
                pixel_size_angstrom=float(payload["record"]["pixel_size_angstrom"]),
                voltage_kv=float(payload["record"]["voltage_kv"]),
                spherical_aberration_mm=float(payload["record"]["spherical_aberration_mm"]),
                amplitude_contrast=float(payload["record"]["amplitude_contrast"]),
                defocus_u_angstrom=float(payload["record"]["defocus_u_angstrom"]),
                defocus_v_angstrom=float(payload["record"]["defocus_v_angstrom"]),
                defocus_angle_deg=float(payload["record"]["defocus_angle_deg"]),
                phase_shift_deg=float(payload["record"]["phase_shift_deg"]),
                ctf_max_resolution_angstrom=payload["record"].get("ctf_max_resolution_angstrom"),
                ctf_figure_of_merit=payload["record"].get("ctf_figure_of_merit"),
                raw_micrograph_row={},
                raw_optics_row={},
            )
            job = BatchJob(index=index, record=record, config=cfg, template_info=payload.get("template_info"))
            result_queue.put(_run_one(job, device))
        except BaseException as exc:
            result_queue.put(
                {
                    "index": index,
                    "micrograph_name": payload["record"].get("micrograph_name", ""),
                    "micrograph_path": payload["record"].get("micrograph_path", ""),
                    "optics_group": payload["record"].get("optics_group", ""),
                    "pixel_size_angstrom": payload["record"].get("pixel_size_angstrom", ""),
                    "voltage_kv": payload["record"].get("voltage_kv", ""),
                    "spherical_aberration_mm": payload["record"].get("spherical_aberration_mm", ""),
                    "amplitude_contrast": payload["record"].get("amplitude_contrast", ""),
                    "defocus_u_angstrom": payload["record"].get("defocus_u_angstrom", ""),
                    "defocus_v_angstrom": payload["record"].get("defocus_v_angstrom", ""),
                    "defocus_angle_deg": payload["record"].get("defocus_angle_deg", ""),
                    "phase_shift_deg": payload["record"].get("phase_shift_deg", ""),
                    "ctf_max_resolution_angstrom": payload["record"].get("ctf_max_resolution_angstrom", ""),
                    "ctf_figure_of_merit": payload["record"].get("ctf_figure_of_merit", ""),
                    "device": device,
                    "status": "failed",
                    "elapsed_seconds": 0.0,
                    "output_dir": payload["config"].get("output_dir", ""),
                    "output_prefix": payload["config"].get("output_prefix", ""),
                    "threshold": "",
                    "number_of_peaks": "",
                    "files": {},
                    "template_info": payload.get("template_info"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )


def _job_payload(job: BatchJob) -> dict[str, Any]:
    return {
        "index": job.index,
        "config": job.config.to_dict(),
        "record": job.record.to_dict(),
        "template_info": job.template_info,
    }


def _write_manifest(base: MatchConfig, rows: list[dict[str, Any]], elapsed: float) -> tuple[Path, Path]:
    output_dir = Path(base.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / base.batch.manifest_prefix
    ordered = sorted(rows, key=lambda row: int(row["index"]))
    payload = {
        "schema_version": 1,
        "star_path": str(Path(base.micrographs_star).expanduser().resolve()),
        "elapsed_seconds": elapsed,
        "successful": sum(row["status"] == "ok" for row in ordered),
        "failed": sum(row["status"] != "ok" for row in ordered),
        "jobs": ordered,
    }
    json_path = write_json(f"{prefix}_manifest.json", payload)
    tsv_rows = []
    for row in ordered:
        flattened = dict(row)
        flattened["files"] = json.dumps(row.get("files", {}), sort_keys=True)
        flattened["template_info"] = json.dumps(row.get("template_info"), sort_keys=True)
        tsv_rows.append(flattened)
    fields = [
        "index",
        "status",
        "device",
        "micrograph_name",
        "micrograph_path",
        "optics_group",
        "pixel_size_angstrom",
        "voltage_kv",
        "spherical_aberration_mm",
        "amplitude_contrast",
        "defocus_u_angstrom",
        "defocus_v_angstrom",
        "defocus_angle_deg",
        "phase_shift_deg",
        "ctf_max_resolution_angstrom",
        "ctf_figure_of_merit",
        "elapsed_seconds",
        "threshold",
        "number_of_peaks",
        "output_dir",
        "output_prefix",
        "error",
        "files",
        "template_info",
        "traceback",
    ]
    tsv_path = write_rows_tsv(f"{prefix}_manifest.tsv", tsv_rows, fieldnames=fields)
    return json_path, tsv_path


def run_batch(base: MatchConfig) -> BatchRunResult:
    """Run one complete micrograph per GPU worker; single device is sequential."""

    base.validate()
    jobs = build_batch_jobs(base, generate_templates=True)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []

    devices = [int(value) for value in base.runtime.devices]
    if len(devices) > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("multiple GPUs requested but CUDA is unavailable")
        visible = torch.cuda.device_count()
        invalid = [value for value in devices if value < 0 or value >= visible]
        if invalid:
            raise RuntimeError(f"invalid GPU indices {invalid}; visible device count is {visible}")
        context = mp.get_context("spawn")
        task_queue: mp.Queue = context.Queue()
        result_queue: mp.Queue = context.Queue()
        workers = [
            context.Process(target=_gpu_worker, args=(gpu, task_queue, result_queue), daemon=False)
            for gpu in devices
        ]
        for worker in workers:
            worker.start()
        for job in jobs:
            task_queue.put(_job_payload(job))
        for _ in workers:
            task_queue.put(None)
        remaining = len(jobs)
        while remaining:
            try:
                rows.append(result_queue.get(timeout=1.0))
                remaining -= 1
            except queue.Empty:
                if not any(worker.is_alive() for worker in workers):
                    break
        for worker in workers:
            worker.join()
        completed_indices = {int(row["index"]) for row in rows if int(row["index"]) < 10**12}
        for job in jobs:
            if job.index not in completed_indices:
                rows.append(
                    {
                        "index": job.index,
                        "micrograph_name": job.record.micrograph_name,
                        "micrograph_path": str(job.record.micrograph_path),
                        "optics_group": job.record.optics_group,
                        **_record_manifest_fields(job.record),
                        "device": "unknown",
                        "status": "failed",
                        "elapsed_seconds": 0.0,
                        "output_dir": job.config.output_dir,
                        "output_prefix": job.config.output_prefix,
                        "threshold": "",
                        "number_of_peaks": "",
                        "files": {},
                        "template_info": job.template_info,
                        "error": "GPU worker exited without returning a job result",
                        "traceback": "",
                    }
                )
    else:
        if devices:
            device = f"cuda:{devices[0]}"
        else:
            device = base.runtime.device
        for job in jobs:
            try:
                rows.append(_run_one(job, device))
            except BaseException as exc:
                rows.append(
                    {
                        "index": job.index,
                        "micrograph_name": job.record.micrograph_name,
                        "micrograph_path": str(job.record.micrograph_path),
                        "optics_group": job.record.optics_group,
                        **_record_manifest_fields(job.record),
                        "device": device,
                        "status": "failed",
                        "elapsed_seconds": 0.0,
                        "output_dir": job.config.output_dir,
                        "output_prefix": job.config.output_prefix,
                        "threshold": "",
                        "number_of_peaks": "",
                        "files": {},
                        "template_info": job.template_info,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                if not base.batch.continue_on_error:
                    break

    elapsed = time.perf_counter() - started
    manifest_json, manifest_tsv = _write_manifest(base, rows, elapsed)
    failed = sum(row["status"] != "ok" for row in rows)
    result = BatchRunResult(
        star_path=str(Path(base.micrographs_star).expanduser().resolve()),
        jobs=sorted(rows, key=lambda row: int(row["index"])),
        manifest_json=str(manifest_json),
        manifest_tsv=str(manifest_tsv),
        successful=sum(row["status"] == "ok" for row in rows),
        failed=failed,
        elapsed_seconds=elapsed,
    )
    if failed and not base.batch.continue_on_error:
        first = next(row for row in rows if row["status"] != "ok")
        raise RuntimeError(
            f"batch stopped with {failed} failed micrograph(s); manifest: {manifest_json}; "
            f"first error: {first['error']}"
        )
    return result
