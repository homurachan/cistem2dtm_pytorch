from __future__ import annotations

import csv
import fnmatch
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class ArrayComparison:
    relative_path: str
    status: str
    shape_reference: list[int] | None = None
    shape_candidate: list[int] | None = None
    dtype_reference: str | None = None
    dtype_candidate: str | None = None
    maximum_absolute_error: float | None = None
    mean_absolute_error: float | None = None
    rmse: float | None = None
    relative_l2_error: float | None = None
    allclose: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metric(reference: np.ndarray, candidate: np.ndarray, *, rtol: float, atol: float) -> ArrayComparison:
    raise RuntimeError("_metric requires a path and is not called directly")


def compare_array_files(
    reference_path: str | Path,
    candidate_path: str | Path,
    *,
    relative_path: str,
    rtol: float,
    atol: float,
) -> ArrayComparison:
    reference = np.load(reference_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)
    record = ArrayComparison(
        relative_path=relative_path,
        status="compared",
        shape_reference=list(reference.shape),
        shape_candidate=list(candidate.shape),
        dtype_reference=str(reference.dtype),
        dtype_candidate=str(candidate.dtype),
    )
    if reference.shape != candidate.shape:
        record.status = "shape_mismatch"
        record.message = "array shapes differ"
        return record
    if not (np.issubdtype(reference.dtype, np.number) and np.issubdtype(candidate.dtype, np.number)):
        record.status = "unsupported_dtype"
        record.message = "only numeric NPY arrays are compared"
        return record
    ref = reference.astype(np.complex128 if np.iscomplexobj(reference) or np.iscomplexobj(candidate) else np.float64)
    cand = candidate.astype(ref.dtype)
    finite = np.isfinite(ref) & np.isfinite(cand)
    if not finite.all():
        same_nonfinite = np.array_equal(np.isnan(ref), np.isnan(cand)) and np.array_equal(np.isinf(ref), np.isinf(cand))
        if not same_nonfinite:
            record.status = "nonfinite_mismatch"
            record.message = "NaN/Inf locations differ"
            return record
    difference = np.abs(cand - ref)
    if difference.size == 0:
        record.maximum_absolute_error = 0.0
        record.mean_absolute_error = 0.0
        record.rmse = 0.0
        record.relative_l2_error = 0.0
        record.allclose = True
        return record
    record.maximum_absolute_error = float(np.nanmax(difference))
    record.mean_absolute_error = float(np.nanmean(difference))
    record.rmse = float(math.sqrt(float(np.nanmean(difference * difference))))
    denominator = float(np.linalg.norm(ref.ravel()))
    numerator = float(np.linalg.norm((cand - ref).ravel()))
    record.relative_l2_error = numerator / max(denominator, np.finfo(np.float64).tiny)
    record.allclose = bool(np.allclose(ref, cand, rtol=rtol, atol=atol, equal_nan=True))
    if not record.allclose:
        record.status = "different"
    return record


def compare_debug_directories(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    *,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-6,
    pattern: str = "*.npy",
    report_prefix: str | Path | None = None,
) -> dict[str, Any]:
    reference_root = Path(reference_dir)
    candidate_root = Path(candidate_dir)
    if not reference_root.is_dir():
        raise FileNotFoundError(reference_root)
    if not candidate_root.is_dir():
        raise FileNotFoundError(candidate_root)
    reference_files = {
        path.relative_to(reference_root).as_posix(): path
        for path in reference_root.rglob("*.npy")
        if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(path.relative_to(reference_root).as_posix(), pattern)
    }
    candidate_files = {
        path.relative_to(candidate_root).as_posix(): path
        for path in candidate_root.rglob("*.npy")
        if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(path.relative_to(candidate_root).as_posix(), pattern)
    }
    records: list[ArrayComparison] = []
    for relative in sorted(reference_files.keys() | candidate_files.keys()):
        if relative not in reference_files:
            records.append(ArrayComparison(relative, "missing_reference", message="present only in candidate"))
        elif relative not in candidate_files:
            records.append(ArrayComparison(relative, "missing_candidate", message="present only in reference"))
        else:
            records.append(
                compare_array_files(
                    reference_files[relative],
                    candidate_files[relative],
                    relative_path=relative,
                    rtol=rtol,
                    atol=atol,
                )
            )
    compared = [record for record in records if record.status in {"compared", "different"}]
    report = {
        "reference_dir": str(reference_root.resolve()),
        "candidate_dir": str(candidate_root.resolve()),
        "rtol": rtol,
        "atol": atol,
        "pattern": pattern,
        "number_of_records": len(records),
        "number_compared": len(compared),
        "number_allclose": sum(record.allclose for record in compared),
        "passed": bool(records) and all(record.allclose for record in compared) and len(compared) == len(records),
        "records": [record.to_dict() for record in records],
    }
    if report_prefix is not None:
        prefix = Path(report_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        json_path = prefix.with_suffix(".json")
        tsv_path = prefix.with_suffix(".tsv")
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        fieldnames = list(ArrayComparison(relative_path="", status="").to_dict())
        with tsv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_dict())
        report["json_report"] = str(json_path)
        report["tsv_report"] = str(tsv_path)
    return report
