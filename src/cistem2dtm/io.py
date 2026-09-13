from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import mrcfile
import numpy as np
import torch

Tensor = torch.Tensor


def _voxel_size_from_mrc(handle: mrcfile.mrcfile.MrcFile, fallback: float | None = None) -> float:
    try:
        value = float(handle.voxel_size.x)
    except Exception:
        value = 0.0
    if not np.isfinite(value) or value <= 0.0:
        if fallback is None or fallback <= 0.0:
            raise ValueError("MRC voxel size is absent or invalid; provide pixel_size_angstrom in the configuration")
        return float(fallback)
    return value


def read_search_image(path: str | Path, image_slice: int = 1, fallback_pixel_size: float | None = None) -> tuple[np.ndarray, float]:
    """Read a 2-D MRC image or one 1-based slice from an MRC stack."""
    path = Path(path)
    if image_slice < 1:
        raise ValueError("image_slice is 1-based")
    with mrcfile.open(path, permissive=True, mode="r") as mrc:
        array = np.asarray(mrc.data)
        pixel_size = _voxel_size_from_mrc(mrc, fallback_pixel_size)
        if array.ndim == 2:
            if image_slice != 1:
                raise IndexError(f"{path} is a single 2-D image; only image_slice=1 is valid")
            image = array
        elif array.ndim == 3:
            index = image_slice - 1
            if index >= array.shape[0]:
                raise IndexError(f"image_slice={image_slice} exceeds stack depth {array.shape[0]}")
            image = array[index]
        else:
            raise ValueError(f"search MRC must be 2-D or 3-D; got shape {array.shape}")
        image = np.array(image, dtype=np.float32, copy=True, order="C")
    if not np.isfinite(image).all():
        raise ValueError(f"search image contains non-finite values: {path}")
    return image, pixel_size


def read_template_volume(path: str | Path, fallback_pixel_size: float | None = None) -> tuple[np.ndarray, float]:
    path = Path(path)
    with mrcfile.open(path, permissive=True, mode="r") as mrc:
        array = np.asarray(mrc.data)
        pixel_size = _voxel_size_from_mrc(mrc, fallback_pixel_size)
        if array.ndim != 3:
            raise ValueError(f"template MRC must be 3-D; got shape {array.shape}")
        volume = np.array(array, dtype=np.float32, copy=True, order="C")
    if len(set(volume.shape)) != 1:
        raise ValueError(f"cisTEM match_template requires a cubic template; got {volume.shape}")
    if not np.isfinite(volume).all():
        raise ValueError(f"template volume contains non-finite values: {path}")
    return volume, pixel_size


def tensor_to_numpy(value: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return np.asarray(value, dtype=np.float32)


def write_mrc(path: str | Path, value: Tensor | np.ndarray, pixel_size_angstrom: float, *, overwrite: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor_to_numpy(value)
    if array.ndim not in (2, 3):
        raise ValueError(f"MRC output must be 2-D or 3-D; got shape {array.shape}")
    with mrcfile.new(path, overwrite=overwrite) as mrc:
        mrc.set_data(np.ascontiguousarray(array, dtype=np.float32))
        mrc.voxel_size = float(pixel_size_angstrom)
        mrc.update_header_stats()
    return path


def create_mrc_stack(path: str | Path, shape: tuple[int, int, int], pixel_size_angstrom: float) -> mrcfile.mrcmemmap.MrcMemmap:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = mrcfile.new_mmap(path, shape=shape, mrc_mode=2, overwrite=True)
    handle.voxel_size = float(pixel_size_angstrom)
    return handle


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_rows_tsv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow({name: json_ready(row.get(name, "")) for name in fieldnames})
    return path
