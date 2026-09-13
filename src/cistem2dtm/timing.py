from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import torch

from .config import TimingConfig
from .io import json_ready


class PerformanceProfiler:
    """Low-overhead wall-clock and sampled CUDA-event profiler.

    The full 2DTM search can contain tens or hundreds of thousands of orientation
    batches. Synchronizing every operation would change the runtime that is being
    measured. The default ``sampled`` mode therefore records CUDA events only for
    selected batches and performs one synchronization at the end of each selected
    batch. Major stage wall times are measured exactly with synchronization only at
    stage boundaries.
    """

    def __init__(
        self,
        config: TimingConfig,
        *,
        device: torch.device,
        output_dir: str | Path,
        output_prefix: str,
        package_version: str,
        worker_rank: int | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.output_dir = Path(config.output_dir or output_dir)
        self.output_prefix = str(output_prefix)
        self.package_version = str(package_version)
        self.worker_rank = worker_rank
        self.enabled = bool(config.enabled)
        self.is_cuda = self.enabled and device.type == "cuda" and torch.cuda.is_available()

        self._created_at = time.perf_counter()
        self._wall_values: dict[str, list[float]] = {}
        self._cuda_values_ms: dict[str, list[float]] = {}
        self._batch_wall_values_ms: dict[str, list[float]] = {}
        self._batch_samples: list[dict[str, Any]] = []
        self._counters: dict[str, float | int] = {}
        self._metadata: dict[str, Any] = {}

        self._current_batch_index: int | None = None
        self._current_batch_size: int = 0
        self._current_batch_metadata: dict[str, Any] = {}
        self._current_batch_sampled: bool = False
        self._current_batch_wall_start: float = 0.0
        self._current_batch_event_start: torch.cuda.Event | None = None
        self._pending_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event, float]] = []
        self._pending_cpu_sections_ms: dict[str, float] = {}
        self._sample_count = 0
        self._total_batches = 0
        self._total_orientations = 0
        self._planned_total_batches = 0
        self._planned_total_orientations = 0
        self._effective_sample_interval = int(config.sample_every_batches)
        self._search_wall_seconds: float | None = None
        self._search_start: float | None = None
        self._last_progress_time = self._created_at
        self._last_progress_orientations = 0

        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    @property
    def batch_is_sampled(self) -> bool:
        return self._current_batch_sampled

    def set_metadata(self, **values: Any) -> None:
        if self.enabled:
            self._metadata.update(values)

    def increment(self, name: str, value: int | float = 1) -> None:
        if not self.enabled:
            return
        self._counters[name] = self._counters.get(name, 0) + value

    def record_wall(self, name: str, elapsed_seconds: float) -> None:
        if not self.enabled:
            return
        self._wall_values.setdefault(name, []).append(float(elapsed_seconds))

    @contextmanager
    def wall_section(self, name: str, *, synchronize_cuda: bool = False) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if synchronize_cuda and self.is_cuda:
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            if synchronize_cuda and self.is_cuda:
                torch.cuda.synchronize(self.device)
            self.record_wall(name, time.perf_counter() - started)

    def start_search(
        self,
        *,
        total_batches: int | None = None,
        total_orientations: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._planned_total_batches = int(total_batches or 0)
        self._planned_total_orientations = int(total_orientations or 0)
        interval = int(self.config.sample_every_batches)
        maximum = int(self.config.max_samples)
        if self.config.mode == "sampled" and maximum > 0 and self._planned_total_batches > 0:
            available = max(self._planned_total_batches - int(self.config.warmup_batches), 1)
            spread_interval = int(math.ceil(available / maximum))
            interval = max(interval, spread_interval)
        self._effective_sample_interval = max(interval, 1)
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        self._search_start = time.perf_counter()

    def finish_search(self, *, total_batches: int, total_orientations: int) -> None:
        if not self.enabled:
            return
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        if self._search_start is not None:
            self._search_wall_seconds = time.perf_counter() - self._search_start
            self.record_wall("search.total", self._search_wall_seconds)
        self._total_batches = int(total_batches)
        self._total_orientations = int(total_orientations)

    def finish_run(self) -> None:
        if not self.enabled:
            return
        if self.is_cuda:
            torch.cuda.synchronize(self.device)
        if "run.total" not in self._wall_values:
            self.record_wall("run.total", time.perf_counter() - self._created_at)

    def _should_sample_batch(self, batch_index: int) -> bool:
        if not self.enabled or self.config.mode == "summary":
            return False
        if batch_index < int(self.config.warmup_batches):
            return False
        if self.config.mode == "synchronized":
            return True
        if (batch_index - int(self.config.warmup_batches)) % int(self._effective_sample_interval) != 0:
            return False
        maximum = int(self.config.max_samples)
        return maximum <= 0 or self._sample_count < maximum

    def begin_batch(
        self,
        batch_index: int,
        batch_size: int,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        if self._current_batch_index is not None:
            raise RuntimeError("a timing batch is already active")
        self._current_batch_index = int(batch_index)
        self._current_batch_size = int(batch_size)
        self._current_batch_metadata = dict(metadata or {})
        self._current_batch_sampled = self._should_sample_batch(batch_index)
        self._pending_events = []
        self._pending_cpu_sections_ms = {}
        if self._current_batch_sampled:
            self._current_batch_wall_start = time.perf_counter()
            if self.is_cuda:
                self._current_batch_event_start = torch.cuda.Event(enable_timing=True)
                self._current_batch_event_start.record()
            else:
                self._current_batch_event_start = None
            self._sample_count += 1
        if self.config.emit_nvtx and self.is_cuda:
            torch.cuda.nvtx.range_push(f"2dtm_batch_{batch_index}")

    @contextmanager
    def batch_section(self, name: str) -> Iterator[None]:
        """Time one operation inside the active orientation batch.

        CUDA event pairs are created only for sampled batches. Host wall time is
        also retained because CUDA tensor-to-bool conversions and I/O can block the
        Python thread even when their kernels are short.
        """
        nvtx = bool(self.enabled and self.config.emit_nvtx and self.is_cuda)
        if nvtx:
            torch.cuda.nvtx.range_push(name)
        if not self.enabled or not self._current_batch_sampled:
            try:
                yield
            finally:
                if nvtx:
                    torch.cuda.nvtx.range_pop()
            return

        host_start = time.perf_counter()
        start_event: torch.cuda.Event | None = None
        end_event: torch.cuda.Event | None = None
        if self.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        try:
            yield
        finally:
            if self.is_cuda:
                assert start_event is not None and end_event is not None
                end_event.record()
                self._pending_events.append((name, start_event, end_event, (time.perf_counter() - host_start) * 1000.0))
            else:
                elapsed_ms = (time.perf_counter() - host_start) * 1000.0
                self._pending_cpu_sections_ms[name] = self._pending_cpu_sections_ms.get(name, 0.0) + elapsed_ms
            if nvtx:
                torch.cuda.nvtx.range_pop()

    def end_batch(self) -> None:
        if not self.enabled or self._current_batch_index is None:
            return
        if self.config.emit_nvtx and self.is_cuda:
            torch.cuda.nvtx.range_pop()

        if self._current_batch_sampled:
            cuda_sections: dict[str, float] = {}
            host_sections: dict[str, float] = dict(self._pending_cpu_sections_ms)
            batch_cuda_ms: float | None = None
            if self.is_cuda:
                batch_end = torch.cuda.Event(enable_timing=True)
                batch_end.record()
                torch.cuda.synchronize(self.device)
                if self._current_batch_event_start is not None:
                    batch_cuda_ms = float(self._current_batch_event_start.elapsed_time(batch_end))
                for name, start_event, end_event, host_ms in self._pending_events:
                    cuda_sections[name] = cuda_sections.get(name, 0.0) + float(start_event.elapsed_time(end_event))
                    host_sections[name] = host_sections.get(name, 0.0) + float(host_ms)
            batch_host_ms = (time.perf_counter() - self._current_batch_wall_start) * 1000.0

            if batch_cuda_ms is not None:
                self._cuda_values_ms.setdefault("batch.total", []).append(batch_cuda_ms)
            self._batch_wall_values_ms.setdefault("batch.total", []).append(batch_host_ms)
            for name, value in cuda_sections.items():
                self._cuda_values_ms.setdefault(name, []).append(value)
            for name, value in host_sections.items():
                self._batch_wall_values_ms.setdefault(name, []).append(value)

            memory: dict[str, int] = {}
            if self.is_cuda and self.config.record_cuda_memory:
                memory = {
                    "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
                }
            if self.config.save_batch_samples:
                self._batch_samples.append(
                    {
                        "batch_index": int(self._current_batch_index),
                        "batch_size": int(self._current_batch_size),
                        "host_total_ms": batch_host_ms,
                        "cuda_total_ms": batch_cuda_ms,
                        "host_sections_ms": host_sections,
                        "cuda_sections_ms": cuda_sections,
                        **self._current_batch_metadata,
                        **memory,
                    }
                )

        self._current_batch_index = None
        self._current_batch_size = 0
        self._current_batch_metadata = {}
        self._current_batch_sampled = False
        self._current_batch_event_start = None
        self._pending_events = []
        self._pending_cpu_sections_ms = {}

    def maybe_print_progress(
        self,
        *,
        completed_batches: int,
        completed_orientations: int,
        total_batches: int,
        total_orientations: int,
    ) -> None:
        if not self.enabled or int(self.config.progress_every_batches) <= 0:
            return
        if completed_batches % int(self.config.progress_every_batches) != 0 and completed_batches != total_batches:
            return
        now = time.perf_counter()
        elapsed = max(now - (self._search_start or self._created_at), 1.0e-12)
        rate = completed_orientations / elapsed
        interval_dt = max(now - self._last_progress_time, 1.0e-12)
        interval_orientations = completed_orientations - self._last_progress_orientations
        interval_rate = interval_orientations / interval_dt
        memory_text = ""
        if self.is_cuda and self.config.record_cuda_memory:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            peak = torch.cuda.max_memory_allocated(self.device) / 1024**3
            memory_text = f", memory={allocated:.2f} GiB, peak={peak:.2f} GiB"
        print(
            f"[2dtm timing] batches {completed_batches}/{total_batches}, "
            f"orientations {completed_orientations}/{total_orientations}, "
            f"rate={rate:.2f} ori/s, recent={interval_rate:.2f} ori/s{memory_text}",
            file=sys.stderr,
            flush=True,
        )
        self._last_progress_time = now
        self._last_progress_orientations = completed_orientations

    @staticmethod
    def _distribution(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0}
        ordered = sorted(float(v) for v in values)

        def percentile(q: float) -> float:
            if len(ordered) == 1:
                return ordered[0]
            position = (len(ordered) - 1) * q
            lo = int(math.floor(position))
            hi = int(math.ceil(position))
            if lo == hi:
                return ordered[lo]
            weight = position - lo
            return ordered[lo] * (1.0 - weight) + ordered[hi] * weight

        return {
            "count": len(ordered),
            "total": float(sum(ordered)),
            "mean": float(statistics.fmean(ordered)),
            "median": float(percentile(0.50)),
            "p90": float(percentile(0.90)),
            "p95": float(percentile(0.95)),
            "p99": float(percentile(0.99)),
            "minimum": float(ordered[0]),
            "maximum": float(ordered[-1]),
        }

    def report(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        wall_sections = {
            name: self._distribution(values)
            for name, values in sorted(self._wall_values.items())
        }
        batch_total_cuda = self._distribution(self._cuda_values_ms.get("batch.total", []))
        batch_total_host = self._distribution(self._batch_wall_values_ms.get("batch.total", []))
        mean_batch_cuda = float(batch_total_cuda.get("mean", 0.0))
        mean_batch_host = float(batch_total_host.get("mean", 0.0))

        section_names = sorted(
            (set(self._cuda_values_ms) | set(self._batch_wall_values_ms)) - {"batch.total"}
        )
        batch_sections: dict[str, Any] = {}
        for name in section_names:
            cuda_stats = self._distribution(self._cuda_values_ms.get(name, []))
            host_stats = self._distribution(self._batch_wall_values_ms.get(name, []))
            mean_cuda = float(cuda_stats.get("mean", 0.0))
            mean_host = float(host_stats.get("mean", 0.0))
            batch_sections[name] = {
                "cuda_ms": cuda_stats,
                "host_ms": host_stats,
                "estimated_total_cuda_seconds": (
                    mean_cuda * self._total_batches / 1000.0 if mean_cuda and self._total_batches else None
                ),
                "estimated_total_host_seconds": (
                    mean_host * self._total_batches / 1000.0 if mean_host and self._total_batches else None
                ),
                "fraction_of_sampled_batch_cuda": (
                    mean_cuda / mean_batch_cuda if mean_cuda and mean_batch_cuda else None
                ),
                "fraction_of_sampled_batch_host": (
                    mean_host / mean_batch_host if mean_host and mean_batch_host else None
                ),
                "host_over_cuda_ratio": (
                    mean_host / mean_cuda if mean_host and mean_cuda else None
                ),
            }

        device_info: dict[str, Any] = {
            "device": str(self.device),
            "is_cuda": self.is_cuda,
            "torch_cuda_version": torch.version.cuda,
        }
        memory: dict[str, Any] = {}
        if self.is_cuda:
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            device_info.update(
                {
                    "index": int(index),
                    "name": props.name,
                    "total_memory_bytes": int(props.total_memory),
                    "compute_capability": [int(props.major), int(props.minor)],
                }
            )
            try:
                plan_cache = torch.backends.cuda.cufft_plan_cache[index]
                device_info["cufft_plan_cache_size"] = int(plan_cache.size)
                device_info["cufft_plan_cache_max_size"] = int(plan_cache.max_size)
            except Exception:
                pass
            if self.config.record_cuda_memory:
                memory = {
                    "allocated_bytes_at_report": int(torch.cuda.memory_allocated(self.device)),
                    "reserved_bytes_at_report": int(torch.cuda.memory_reserved(self.device)),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
                }

        search_wall = self._search_wall_seconds
        throughput: dict[str, Any] = {}
        if search_wall and search_wall > 0:
            throughput = {
                "orientations_per_second": self._total_orientations / search_wall,
                "batches_per_second": self._total_batches / search_wall if self._total_batches else 0.0,
            }
            pixels = int(self._metadata.get("ccf_pixels_per_orientation", 0) or 0)
            if pixels:
                throughput["ccf_pixels_per_second"] = self._total_orientations * pixels / search_wall

        return {
            "schema_version": 1,
            "package_version": self.package_version,
            "created_unix_time": time.time(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "worker_rank": self.worker_rank,
            "configuration": asdict(self.config),
            "device": device_info,
            "metadata": json_ready(self._metadata),
            "counters": json_ready(self._counters),
            "search": {
                "wall_seconds": search_wall,
                "total_batches": self._total_batches,
                "total_orientations": self._total_orientations,
                "planned_total_batches": self._planned_total_batches,
                "planned_total_orientations": self._planned_total_orientations,
                "sampled_batches": self._sample_count,
                "effective_sample_interval_batches": self._effective_sample_interval,
                "throughput": throughput,
            },
            "wall_sections_seconds": wall_sections,
            "sampled_batch_total": {
                "cuda_ms": batch_total_cuda,
                "host_ms": batch_total_host,
                "estimated_all_batches_cuda_seconds": (
                    mean_batch_cuda * self._total_batches / 1000.0
                    if mean_batch_cuda and self._total_batches
                    else None
                ),
            },
            "sampled_batch_sections": batch_sections,
            "cuda_memory": memory,
            "batch_samples": self._batch_samples if self.config.save_batch_samples else [],
            "measurement_notes": [
                "summary stage times synchronize only at major boundaries",
                "sampled CUDA sections use events and one synchronization per sampled batch",
                "host section time includes Python overhead and any implicit CUDA synchronization",
                "estimated totals extrapolate sampled means and are not a replacement for search wall time",
            ],
        }

    def _base_path(self) -> Path:
        suffix = f"_gpu{self.worker_rank}" if self.worker_rank is not None else ""
        return self.output_dir / f"{self.output_prefix}{suffix}_timing"

    def write(self) -> dict[str, Path]:
        if not self.enabled:
            return {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report = self.report()
        base = self._base_path()
        json_path = base.with_suffix(".json")
        json_path.write_text(json.dumps(json_ready(report), indent=2, sort_keys=True), encoding="utf-8")

        wall_path = base.with_name(base.name + "_wall_sections.tsv")
        with wall_path.open("w", newline="", encoding="utf-8") as stream:
            fieldnames = [
                "section",
                "count",
                "total_seconds",
                "mean_seconds",
                "median_seconds",
                "p90_seconds",
                "p95_seconds",
                "p99_seconds",
                "minimum_seconds",
                "maximum_seconds",
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for name, values in report.get("wall_sections_seconds", {}).items():
                writer.writerow(
                    {
                        "section": name,
                        "count": values.get("count", 0),
                        "total_seconds": values.get("total", ""),
                        "mean_seconds": values.get("mean", ""),
                        "median_seconds": values.get("median", ""),
                        "p90_seconds": values.get("p90", ""),
                        "p95_seconds": values.get("p95", ""),
                        "p99_seconds": values.get("p99", ""),
                        "minimum_seconds": values.get("minimum", ""),
                        "maximum_seconds": values.get("maximum", ""),
                    }
                )

        sections_path = base.with_name(base.name + "_sections.tsv")
        with sections_path.open("w", newline="", encoding="utf-8") as stream:
            fieldnames = [
                "section",
                "samples",
                "mean_cuda_ms",
                "median_cuda_ms",
                "p90_cuda_ms",
                "p95_cuda_ms",
                "p99_cuda_ms",
                "mean_host_ms",
                "estimated_total_cuda_seconds",
                "estimated_total_host_seconds",
                "fraction_of_sampled_batch_cuda",
                "fraction_of_sampled_batch_host",
                "host_over_cuda_ratio",
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for name, entry in report.get("sampled_batch_sections", {}).items():
                cuda = entry.get("cuda_ms", {})
                host = entry.get("host_ms", {})
                writer.writerow(
                    {
                        "section": name,
                        "samples": max(int(cuda.get("count", 0)), int(host.get("count", 0))),
                        "mean_cuda_ms": cuda.get("mean", ""),
                        "median_cuda_ms": cuda.get("median", ""),
                        "p90_cuda_ms": cuda.get("p90", ""),
                        "p95_cuda_ms": cuda.get("p95", ""),
                        "p99_cuda_ms": cuda.get("p99", ""),
                        "mean_host_ms": host.get("mean", ""),
                        "estimated_total_cuda_seconds": entry.get("estimated_total_cuda_seconds", ""),
                        "estimated_total_host_seconds": entry.get("estimated_total_host_seconds", ""),
                        "fraction_of_sampled_batch_cuda": entry.get("fraction_of_sampled_batch_cuda", ""),
                        "fraction_of_sampled_batch_host": entry.get("fraction_of_sampled_batch_host", ""),
                        "host_over_cuda_ratio": entry.get("host_over_cuda_ratio", ""),
                    }
                )

        paths = {
            "timing_json": json_path,
            "timing_wall_sections": wall_path,
            "timing_sections": sections_path,
        }
        if self.config.save_batch_samples:
            batches_path = base.with_name(base.name + "_batches.tsv")
            section_names = sorted(
                {
                    name
                    for row in self._batch_samples
                    for source in (row.get("cuda_sections_ms", {}), row.get("host_sections_ms", {}))
                    for name in source
                }
            )
            fields = [
                "batch_index",
                "batch_size",
                "orientation_first",
                "orientation_last",
                "pixel_size_index",
                "defocus_index",
                "host_total_ms",
                "cuda_total_ms",
                "allocated_bytes",
                "reserved_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            ]
            fields.extend(f"cuda_ms:{name}" for name in section_names)
            fields.extend(f"host_ms:{name}" for name in section_names)
            with batches_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                for row in self._batch_samples:
                    flat = {
                        key: row.get(key, "")
                        for key in fields
                        if not key.startswith("cuda_ms:") and not key.startswith("host_ms:")
                    }
                    for name in section_names:
                        flat[f"cuda_ms:{name}"] = row.get("cuda_sections_ms", {}).get(name, "")
                        flat[f"host_ms:{name}"] = row.get("host_sections_ms", {}).get(name, "")
                    writer.writerow(flat)
            paths["timing_batches"] = batches_path

        summary_path = base.with_name(base.name + "_summary.txt")
        summary_path.write_text(self.format_summary(report), encoding="utf-8")
        paths["timing_summary"] = summary_path
        if self.config.print_summary:
            print(self.format_summary(report), file=sys.stderr, flush=True)
        return paths

    @staticmethod
    def format_summary(report: dict[str, Any]) -> str:
        if not report:
            return "Timing disabled.\n"
        search = report.get("search", {})
        throughput = search.get("throughput", {})
        lines = [
            "cisTEM2DTM performance summary",
            "=" * 34,
            f"device: {report.get('device', {}).get('device')}",
            f"search wall: {search.get('wall_seconds')} s",
            f"batches: {search.get('total_batches')}, orientations: {search.get('total_orientations')}",
            f"sampled batches: {search.get('sampled_batches')}",
        ]
        if throughput:
            lines.append(f"throughput: {throughput.get('orientations_per_second', 0.0):.3f} orientations/s")
        total = report.get("sampled_batch_total", {}).get("cuda_ms", {})
        if total.get("count", 0):
            lines.append(f"sampled batch CUDA mean: {total.get('mean', 0.0):.3f} ms")
        wall_rows = [
            (float(values.get("total", 0.0)), name, values)
            for name, values in report.get("wall_sections_seconds", {}).items()
        ]
        if wall_rows:
            lines.append("\nMajor wall-clock sections (nested totals may overlap):")
            for total_seconds, name, values in sorted(wall_rows, reverse=True)[:20]:
                lines.append(
                    f"  {name:48s} total={total_seconds:10.3f} s "
                    f"calls={int(values.get('count', 0)):5d} "
                    f"mean={float(values.get('mean', 0.0)):9.4f} s"
                )
        cuda_rows: list[tuple[float, str, dict[str, Any]]] = []
        host_rows: list[tuple[float, str, dict[str, Any]]] = []
        for name, entry in report.get("sampled_batch_sections", {}).items():
            cuda_rows.append((float(entry.get("fraction_of_sampled_batch_cuda") or 0.0), name, entry))
            host_rows.append((float(entry.get("fraction_of_sampled_batch_host") or 0.0), name, entry))
        if any(value > 0 for value, _, _ in cuda_rows):
            lines.append("\nTop sampled CUDA sections:")
            for fraction, name, entry in sorted(cuda_rows, reverse=True)[:15]:
                cuda = entry.get("cuda_ms", {})
                host = entry.get("host_ms", {})
                lines.append(
                    f"  {name:42s} cuda={cuda.get('mean', 0.0):9.3f} ms "
                    f"host={host.get('mean', 0.0):9.3f} ms "
                    f"share={100.0 * fraction:6.2f}%"
                )
        if host_rows:
            lines.append("\nTop sampled host sections (includes implicit CUDA waits):")
            for fraction, name, entry in sorted(host_rows, reverse=True)[:15]:
                cuda = entry.get("cuda_ms", {})
                host = entry.get("host_ms", {})
                lines.append(
                    f"  {name:42s} host={host.get('mean', 0.0):9.3f} ms "
                    f"cuda={cuda.get('mean', 0.0):9.3f} ms "
                    f"share={100.0 * fraction:6.2f}%"
                )
        memory = report.get("cuda_memory", {})
        if memory:
            lines.append(
                f"\npeak CUDA allocated: {memory.get('peak_allocated_bytes', 0) / 1024**3:.3f} GiB; "
                f"reserved: {memory.get('peak_reserved_bytes', 0) / 1024**3:.3f} GiB"
            )
        lines.append("")
        return "\n".join(lines)
