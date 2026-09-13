from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Sequence

import mrcfile
import numpy as np
import torch

Tensor = torch.Tensor
CCFStackMode = Literal["none", "cpu", "mrc"]


@dataclass(slots=True)
class CCFStackResult:
    """Location of an optional retained CCF stack.

    ``cpu`` mode is serialized as a PyTorch tensor after the search.  ``mrc``
    mode is written incrementally.  ``shard_paths`` is used transiently by the
    one-server multi-GPU implementation and is normally empty after its final
    deterministic merge by ``task_index``.
    """

    mode: CCFStackMode
    tensor: Tensor | None = None
    mrc_path: Path | None = None
    tensor_path: Path | None = None
    metadata_path: Path | None = None
    number_written: int = 0
    shard_paths: list[Path] = field(default_factory=list)

    @property
    def data_path(self) -> Path | None:
        return self.mrc_path if self.mode == "mrc" else self.tensor_path


class CCFStackWriter:
    """Optional raw CCF retention without keeping orientation batches on GPU."""

    FIELDNAMES = [
        "stack_index",
        "task_index",
        "orientation_index",
        "phi_deg",
        "theta_deg",
        "psi_deg",
        "defocus_offset_angstrom",
        "pixel_size_offset_angstrom",
    ]

    def __init__(
        self,
        mode: CCFStackMode,
        *,
        total_images: int,
        image_shape: tuple[int, int],
        pixel_size_angstrom: float,
        path: str | Path | None,
    ) -> None:
        if mode not in ("none", "cpu", "mrc"):
            raise ValueError(f"unknown CCF stack mode: {mode}")
        self.mode = mode
        self.total_images = int(total_images)
        self.image_shape = tuple(int(v) for v in image_shape)
        self.pixel_size_angstrom = float(pixel_size_angstrom)
        self.path = Path(path) if path is not None else None
        self.metadata_path: Path | None = None
        self._tensor: Tensor | None = None
        self._mrc: mrcfile.mrcmemmap.MrcMemmap | None = None
        self._metadata_stream = None
        self._metadata_writer: csv.DictWriter | None = None
        self._number_written = 0
        self._closed_result: CCFStackResult | None = None

        if mode == "none":
            return
        if self.total_images <= 0:
            raise ValueError("total_images must be positive when retaining a CCF stack")
        if self.path is None:
            raise ValueError("a stack path is required")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.path.with_suffix(self.path.suffix + ".tsv")
        self._metadata_stream = self.metadata_path.open("w", newline="", encoding="utf-8")
        self._metadata_writer = csv.DictWriter(
            self._metadata_stream,
            fieldnames=self.FIELDNAMES,
            delimiter="\t",
            extrasaction="ignore",
        )
        self._metadata_writer.writeheader()

        if mode == "cpu":
            self._tensor = torch.empty((self.total_images, *self.image_shape), dtype=torch.float32, device="cpu")
        else:
            self._mrc = mrcfile.new_mmap(
                self.path,
                shape=(self.total_images, *self.image_shape),
                mrc_mode=2,
                overwrite=True,
            )
            self._mrc.voxel_size = self.pixel_size_angstrom

    def write(self, ccf: Tensor, metadata_rows: Iterable[dict[str, object]]) -> None:
        if self.mode == "none":
            return
        if self._closed_result is not None:
            raise RuntimeError("cannot write to a closed CCF stack")
        if ccf.ndim != 3 or tuple(ccf.shape[-2:]) != self.image_shape:
            raise ValueError("CCF batch shape differs from configured stack shape")
        rows = list(metadata_rows)
        if len(rows) != int(ccf.shape[0]):
            raise ValueError("metadata row count differs from CCF batch size")
        indices = [int(row["stack_index"]) for row in rows]
        if any(index < 0 or index >= self.total_images for index in indices):
            raise IndexError("stack_index is outside the allocated CCF stack")
        array = ccf.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if self.mode == "cpu":
            assert self._tensor is not None
            self._tensor[torch.tensor(indices, dtype=torch.long)] = array
        else:
            assert self._mrc is not None
            np_array = array.numpy()
            for local, index in enumerate(indices):
                self._mrc.data[index] = np_array[local]
        assert self._metadata_writer is not None
        for row in rows:
            self._metadata_writer.writerow({name: row.get(name, "") for name in self.FIELDNAMES})
        self._number_written += len(rows)

    def close(self) -> CCFStackResult:
        if self._closed_result is not None:
            return self._closed_result
        if self.mode == "none":
            self._closed_result = CCFStackResult(mode="none")
            return self._closed_result
        if self._metadata_stream is not None:
            self._metadata_stream.flush()
            self._metadata_stream.close()
            self._metadata_stream = None
        if self.mode == "mrc" and self._mrc is not None:
            self._mrc.update_header_stats()
            self._mrc.flush()
            self._mrc.close()
            self._mrc = None
        self._closed_result = CCFStackResult(
            mode=self.mode,
            tensor=self._tensor,
            mrc_path=self.path if self.mode == "mrc" else None,
            metadata_path=self.metadata_path,
            number_written=self._number_written,
        )
        return self._closed_result

    def __enter__(self) -> "CCFStackWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def read_ccf_metadata(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _write_ccf_metadata(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CCFStackWriter.FIELDNAMES, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in CCFStackWriter.FIELDNAMES})
    return path


def merge_ccf_stack_shards(
    shards: Sequence[CCFStackResult],
    *,
    output_path: str | Path,
    image_shape: tuple[int, int],
    pixel_size_angstrom: float,
    remove_shards: bool = False,
) -> CCFStackResult:
    """Merge multi-GPU CCF shards into strict global ``task_index`` order."""
    nonempty = [item for item in shards if item.mode != "none"]
    if not nonempty:
        return CCFStackResult(mode="none")
    modes = {item.mode for item in nonempty}
    if len(modes) != 1:
        raise ValueError("CCF shards use different storage modes")
    mode = nonempty[0].mode
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for shard_index, shard in enumerate(nonempty):
        if shard.metadata_path is None:
            raise ValueError("a CCF shard has no metadata file")
        rows = read_ccf_metadata(shard.metadata_path)
        if len(rows) != shard.number_written:
            raise RuntimeError(
                f"metadata rows ({len(rows)}) differ from number_written ({shard.number_written}) "
                f"for shard {shard_index}"
            )
        for row in rows:
            converted: dict[str, object] = dict(row)
            converted["_shard_index"] = shard_index
            converted["_source_stack_index"] = int(row["stack_index"])
            converted["task_index"] = int(row["task_index"])
            records.append(converted)
    records.sort(key=lambda row: int(row["task_index"]))
    task_indices = [int(row["task_index"]) for row in records]
    if len(task_indices) != len(set(task_indices)):
        raise RuntimeError("duplicate task_index values found while merging CCF shards")
    for final_index, row in enumerate(records):
        row["stack_index"] = final_index

    metadata_path = output_path.with_suffix(output_path.suffix + ".tsv")
    _write_ccf_metadata(metadata_path, records)
    total = len(records)
    if mode == "mrc":
        handles: list[mrcfile.mrcfile.MrcFile] = []
        try:
            for shard in nonempty:
                if shard.mrc_path is None:
                    raise ValueError("MRC CCF shard has no data path")
                handles.append(mrcfile.open(shard.mrc_path, mode="r", permissive=True))
            out = mrcfile.new_mmap(
                output_path,
                shape=(total, *image_shape),
                mrc_mode=2,
                overwrite=True,
            )
            try:
                out.voxel_size = float(pixel_size_angstrom)
                for final_index, row in enumerate(records):
                    source = handles[int(row["_shard_index"])]
                    out.data[final_index] = source.data[int(row["_source_stack_index"])]
                out.update_header_stats()
                out.flush()
            finally:
                out.close()
        finally:
            for handle in handles:
                handle.close()
        result = CCFStackResult(
            mode="mrc",
            mrc_path=output_path,
            metadata_path=metadata_path,
            number_written=total,
            shard_paths=[p for item in nonempty for p in ([item.mrc_path] if item.mrc_path else [])],
        )
    else:
        tensors: list[Tensor] = []
        for shard in nonempty:
            if shard.tensor is not None:
                tensor = shard.tensor
            elif shard.tensor_path is not None:
                tensor = torch.load(shard.tensor_path, map_location="cpu", weights_only=True)
            else:
                raise ValueError("CPU CCF shard has neither a tensor nor tensor_path")
            tensors.append(tensor.to(device="cpu", dtype=torch.float32))
        merged = torch.empty((total, *image_shape), dtype=torch.float32)
        for final_index, row in enumerate(records):
            merged[final_index] = tensors[int(row["_shard_index"])][int(row["_source_stack_index"])]
        torch.save(merged, output_path)
        result = CCFStackResult(
            mode="cpu",
            tensor=merged,
            tensor_path=output_path,
            metadata_path=metadata_path,
            number_written=total,
            shard_paths=[p for item in nonempty for p in ([item.tensor_path] if item.tensor_path else [])],
        )

    if remove_shards:
        for shard in nonempty:
            for path in (shard.mrc_path, shard.tensor_path, shard.metadata_path):
                if path is not None and Path(path) != output_path and Path(path) != metadata_path:
                    Path(path).unlink(missing_ok=True)
    return result
