from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import DebugConfig
from .io import json_ready, write_mrc

Tensor = torch.Tensor


STAGE_NAMES = {
    1: "input_and_sizing",
    2: "first_whitening",
    3: "resize_and_second_normalization",
    4: "euler_grid_and_rotations",
    5: "ctf_and_projection_filter",
    6: "fourier_central_slice",
    7: "projection_realspace_normalization",
    8: "single_orientation_ccf",
    9: "raw_search_accumulators",
    10: "postresize_scaled_mip_and_peaks",
}


class DebugStop(RuntimeError):
    def __init__(self, stage: int) -> None:
        super().__init__(f"debug stop requested after stage {stage}")
        self.stage = stage


class DebugRecorder:
    def __init__(self, config: DebugConfig, pixel_size_angstrom: float) -> None:
        self.config = config
        self.pixel_size_angstrom = float(pixel_size_angstrom)
        self.root = Path(config.debug_dir or "2dtm_debug")
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest: dict[str, Any] = {
            "stages": STAGE_NAMES,
            "records": [],
            "timings_seconds": {},
        }
        self._timers: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def stage_dir(self, stage: int) -> Path:
        name = STAGE_NAMES.get(stage, f"stage_{stage}")
        path = self.root / f"{stage:02d}_{name}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def start_timer(self, name: str) -> None:
        if self.config.record_timing:
            self._timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        if not self.config.record_timing or name not in self._timers:
            return 0.0
        elapsed = time.perf_counter() - self._timers.pop(name)
        self.manifest["timings_seconds"][name] = elapsed
        # Timing updates are flushed at stage boundaries. Rewriting the JSON
        # manifest for every interval can dominate short profiling runs.
        return elapsed

    def add_timing(self, name: str, elapsed_seconds: float) -> None:
        """Accumulate a measured interval in the debug manifest."""
        if not self.config.record_timing:
            return
        timings = self.manifest["timings_seconds"]
        timings[name] = float(timings.get(name, 0.0)) + float(elapsed_seconds)
        # Defer disk I/O until save()/finish_stage(); the performance profiler
        # writes its own report once at the end of the run.

    def save(self, stage: int, name: str, value: Any, *, metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        directory = self.stage_dir(stage)
        record: dict[str, Any] = {"stage": stage, "name": name, "metadata": metadata or {}}
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            record.update({"shape": list(tensor.shape), "dtype": str(tensor.dtype), "complex": bool(torch.is_complex(tensor))})
            if self.config.save_tensors:
                torch.save(tensor, directory / f"{name}.pt")
            if self.config.save_npy:
                numpy_tensor = tensor
                if numpy_tensor.dtype == torch.bfloat16:
                    numpy_tensor = numpy_tensor.to(torch.float32)
                if numpy_tensor.dtype == torch.complex32:
                    numpy_tensor = numpy_tensor.to(torch.complex64)
                np.save(directory / f"{name}.npy", numpy_tensor.numpy())
            if self.config.save_mrc and not torch.is_complex(tensor) and tensor.ndim in (2, 3):
                write_mrc(directory / f"{name}.mrc", tensor, self.pixel_size_angstrom)
        elif isinstance(value, np.ndarray):
            record.update({"shape": list(value.shape), "dtype": str(value.dtype), "complex": bool(np.iscomplexobj(value))})
            if self.config.save_npy:
                np.save(directory / f"{name}.npy", value)
            if self.config.save_mrc and not np.iscomplexobj(value) and value.ndim in (2, 3):
                write_mrc(directory / f"{name}.mrc", value, self.pixel_size_angstrom)
        else:
            (directory / f"{name}.json").write_text(json.dumps(json_ready(value), indent=2, sort_keys=True), encoding="utf-8")
        self.manifest["records"].append(record)
        self.flush_manifest()

    def finish_stage(self, stage: int) -> None:
        if self.enabled:
            self.manifest["last_completed_stage"] = stage
            self.flush_manifest()
        if self.config.stop_after_stage == stage:
            raise DebugStop(stage)

    def flush_manifest(self) -> None:
        if self.enabled:
            (self.root / "manifest.json").write_text(
                json.dumps(json_ready(self.manifest), indent=2, sort_keys=True),
                encoding="utf-8",
            )
