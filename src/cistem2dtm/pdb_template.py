from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .config import MatchConfig, TemplateInputConfig
from .io import write_json, write_mrc


# Preserve the elemental model used by the supplied long-running pdb2mrc script.
ELEMENT_DATA: dict[str, tuple[float, float]] = {
    "H": (1.0, 1.00794),
    "C": (6.0, 12.0107),
    "N": (7.0, 14.00674),
    "O": (8.0, 15.9994),
    "P": (15.0, 30.973761),
    "S": (16.0, 32.066),
}


@dataclass(slots=True)
class PDBAtomTable:
    coordinates: np.ndarray
    elements: np.ndarray
    chain_ids: np.ndarray
    record_types: np.ndarray
    line_numbers: np.ndarray

    @property
    def atom_count(self) -> int:
        return int(self.coordinates.shape[0])

    def subset(self, mask: np.ndarray) -> "PDBAtomTable":
        mask = np.asarray(mask, dtype=bool)
        return PDBAtomTable(
            coordinates=np.asarray(self.coordinates[mask], dtype=np.float32),
            elements=np.asarray(self.elements[mask]),
            chain_ids=np.asarray(self.chain_ids[mask]),
            record_types=np.asarray(self.record_types[mask]),
            line_numbers=np.asarray(self.line_numbers[mask], dtype=np.int64),
        )


@dataclass(slots=True)
class PDBTemplateResult:
    mrc_path: Path
    center_metadata_path: Path
    template_metadata_path: Path
    center_angstrom: tuple[float, float, float]
    pixel_size_angstrom: float
    resolution_angstrom: float
    box_size: int
    atom_count: int
    reused: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mrc_path": str(self.mrc_path),
            "center_metadata_path": str(self.center_metadata_path),
            "template_metadata_path": str(self.template_metadata_path),
            "center_angstrom": list(self.center_angstrom),
            "pixel_size_angstrom": self.pixel_size_angstrom,
            "resolution_angstrom": self.resolution_angstrom,
            "box_size": self.box_size,
            "atom_count": self.atom_count,
            "reused": self.reused,
        }


def _pdb_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _infer_element(line: str) -> str:
    explicit = line[76:78].strip().upper() if len(line) >= 78 else ""
    if explicit:
        return explicit
    # For standard PDB ATOM records, the first alphabetic character in the
    # four-column atom name is the element (" CA " is carbon alpha, not Ca).
    atom_name = line[12:16] if len(line) >= 16 else ""
    letters = "".join(ch for ch in atom_name if ch.isalpha())
    return letters[:1].upper()


def parse_pdb_text(path: str | Path) -> PDBAtomTable:
    """Parse ATOM/HETATM records directly from fixed-width PDB text.

    Every readable record is retained independently.  Atom serial numbers are
    deliberately ignored, so duplicate/non-unique serials cannot drop atoms.
    Chain IDs are kept exactly (blank chain becomes an empty string).
    """

    path = Path(path)
    coordinates: list[tuple[float, float, float]] = []
    elements: list[str] = []
    chains: list[str] = []
    records: list[str] = []
    line_numbers: list[int] = []
    malformed: list[int] = []

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = line[0:6].strip().upper()
            if record not in {"ATOM", "HETATM"}:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                malformed.append(line_number)
                continue
            coordinates.append((x, y, z))
            elements.append(_infer_element(line))
            chains.append(line[21:22] if len(line) >= 22 else "")
            records.append(record)
            line_numbers.append(line_number)

    if not coordinates:
        detail = f"; malformed ATOM/HETATM lines: {malformed[:10]}" if malformed else ""
        raise ValueError(f"no readable ATOM/HETATM coordinates found in {path}{detail}")

    return PDBAtomTable(
        coordinates=np.asarray(coordinates, dtype=np.float32),
        elements=np.asarray(elements, dtype="U4"),
        chain_ids=np.asarray(chains, dtype="U4"),
        record_types=np.asarray(records, dtype="U6"),
        line_numbers=np.asarray(line_numbers, dtype=np.int64),
    )


def _base_atom_mask(atoms: PDBAtomTable, include_hetatm: bool) -> np.ndarray:
    if include_hetatm:
        return np.ones(atoms.atom_count, dtype=bool)
    return atoms.record_types == "ATOM"


def _chain_mask(
    atoms: PDBAtomTable,
    include_chains: Sequence[str],
    exclude_chains: Sequence[str],
) -> np.ndarray:
    mask = np.ones(atoms.atom_count, dtype=bool)
    if include_chains:
        mask &= np.isin(atoms.chain_ids, np.asarray(include_chains, dtype=atoms.chain_ids.dtype))
    if exclude_chains:
        mask &= ~np.isin(atoms.chain_ids, np.asarray(exclude_chains, dtype=atoms.chain_ids.dtype))
    return mask


def _element_arrays(elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    atomic_numbers = np.zeros(len(elements), dtype=np.float32)
    atomic_masses = np.zeros(len(elements), dtype=np.float32)
    for i, symbol in enumerate(elements):
        data = ELEMENT_DATA.get(str(symbol).upper())
        if data is not None:
            atomic_numbers[i] = data[0]
            atomic_masses[i] = data[1]
    return atomic_numbers, atomic_masses


def mass_weighted_center(coordinates: np.ndarray, elements: np.ndarray) -> np.ndarray:
    _, masses = _element_arrays(elements)
    mass_sum = np.sum(masses, dtype=np.float32)
    if float(mass_sum) <= 0.0:
        # Match the supplied reference implementation.
        return np.zeros(3, dtype=np.float32)
    weighted = coordinates.astype(np.float32, copy=False) * masses[:, None]
    return (
        np.sum(weighted, axis=0, dtype=np.float32) / np.float32(mass_sum)
    ).astype(np.float32, copy=False)


def _center_metadata_matches(path: Path, pdb_sha256: str, include_hetatm: bool) -> bool:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        obj.get("pdb_sha256") == pdb_sha256
        and bool(obj.get("include_hetatm", False)) == bool(include_hetatm)
        and isinstance(obj.get("center_angstrom"), list)
        and len(obj["center_angstrom"]) == 3
    )


def _load_center_reference(path: Path) -> np.ndarray:
    obj = json.loads(path.read_text(encoding="utf-8"))
    value = obj.get("center_angstrom")
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"invalid center reference JSON: {path}")
    center = np.asarray(value, dtype=np.float32)
    if not np.isfinite(center).all():
        raise ValueError(f"non-finite center reference in {path}")
    return center


def _round_like_python(values: torch.Tensor) -> torch.Tensor:
    # torch.round uses round-to-nearest-even, matching Python round for the
    # finite float32 coordinates used here.
    return torch.round(values).to(torch.int64)


def _rasterize_reference(
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    box_size: int,
    pixel_size_angstrom: float,
    resolution_angstrom: float,
    device: torch.device,
) -> torch.Tensor:
    rp = float((math.pi / (resolution_angstrom / pixel_size_angstrom)) ** 2)
    kn = float((rp / math.pi) ** 1.5)
    w = int(round(resolution_angstrom * 3.0 / pixel_size_angstrom))
    if w < 3:
        raise ValueError(
            "insufficient sampling for the requested PDB resolution; "
            "decrease template pixel size or increase resolution_angstrom"
        )
    volume = torch.zeros((box_size, box_size, box_size), dtype=torch.float32, device=device)
    half_box = float(box_size / 2.0)
    rp_t = torch.tensor(rp, dtype=torch.float32, device=device)
    kn_t = torch.tensor(kn, dtype=torch.float32, device=device)
    for coord, z_number in zip(coordinates, atomic_numbers):
        if float(z_number) == 0.0:
            continue
        xx = float(np.float32(coord[0])) / pixel_size_angstrom + half_box
        yy = float(np.float32(coord[1])) / pixel_size_angstrom + half_box
        zz = float(np.float32(coord[2])) / pixel_size_angstrom + half_box
        x0, x1 = round(xx - w), round(xx + w)
        y0, y1 = round(yy - w), round(yy + w)
        z0, z1 = round(zz - w), round(zz + w)
        if x0 >= box_size or y0 >= box_size or z0 >= box_size or x1 < 0 or y1 < 0 or z1 < 0:
            continue
        x0, y0, z0 = max(x0, 0), max(y0, 0), max(z0, 0)
        # Preserve the supplied implementation's exclusive upper bound and its
        # box_size-1 clipping behavior.
        x1, y1, z1 = min(x1, box_size - 1), min(y1, box_size - 1), min(z1, box_size - 1)
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            continue
        xs = torch.arange(x0, x1, dtype=torch.float32, device=device)
        ys = torch.arange(y0, y1, dtype=torch.float32, device=device)
        zs = torch.arange(z0, z1, dtype=torch.float32, device=device)
        dx = xs[None, None, :] - xx
        dy = ys[None, :, None] - yy
        dz = zs[:, None, None] - zz
        radius = torch.sqrt(dx * dx + dy * dy + dz * dz)
        volume[z0:z1, y0:y1, x0:x1] += (
            kn_t * float(z_number) * torch.exp(-radius * rp_t)
        ).to(torch.float32)
    return volume


def _rasterize_vectorized(
    coordinates: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    box_size: int,
    pixel_size_angstrom: float,
    resolution_angstrom: float,
    device: torch.device,
    atom_batch_size: int,
) -> torch.Tensor:
    """Chunked scatter-add implementation of the supplied per-atom kernel."""

    rp = float((math.pi / (resolution_angstrom / pixel_size_angstrom)) ** 2)
    kn = float((rp / math.pi) ** 1.5)
    w = int(round(resolution_angstrom * 3.0 / pixel_size_angstrom))
    if w < 3:
        raise ValueError(
            "insufficient sampling for the requested PDB resolution; "
            "decrease template pixel size or increase resolution_angstrom"
        )
    coords = torch.as_tensor(coordinates, dtype=torch.float32, device=device)
    z_numbers = torch.as_tensor(atomic_numbers, dtype=torch.float32, device=device)
    coords_px = coords / float(pixel_size_angstrom) + float(box_size / 2.0)
    lower_raw = _round_like_python(coords_px - float(w))
    upper_raw = _round_like_python(coords_px + float(w))
    lower = torch.clamp(lower_raw, min=0)
    upper = torch.clamp(upper_raw, max=box_size - 1)
    valid = (
        (lower[:, 0] < upper[:, 0])
        & (lower[:, 1] < upper[:, 1])
        & (lower[:, 2] < upper[:, 2])
        & (z_numbers != 0)
        & (lower_raw[:, 0] < box_size)
        & (lower_raw[:, 1] < box_size)
        & (lower_raw[:, 2] < box_size)
        & (upper_raw[:, 0] >= 0)
        & (upper_raw[:, 1] >= 0)
        & (upper_raw[:, 2] >= 0)
    )
    coords_px = coords_px[valid]
    lower = lower[valid]
    upper = upper[valid]
    z_numbers = z_numbers[valid]

    volume_flat = torch.zeros(box_size**3, dtype=torch.float32, device=device)
    width = 2 * w
    offsets = torch.arange(width, device=device, dtype=torch.int64)
    rp_t = torch.tensor(rp, dtype=torch.float32, device=device)
    kn_t = torch.tensor(kn, dtype=torch.float32, device=device)

    # Keep the peak temporary size bounded for unusual resolution settings.
    patch_voxels = max(1, width**3)
    memory_limited_batch = max(1, int(24_000_000 // patch_voxels))
    chunk_size = max(1, min(int(atom_batch_size), memory_limited_batch))

    for start in range(0, coords_px.shape[0], chunk_size):
        stop = min(start + chunk_size, coords_px.shape[0])
        c = coords_px[start:stop]
        lo = lower[start:stop]
        hi = upper[start:stop]
        zn = z_numbers[start:stop]
        xi = lo[:, 0:1] + offsets[None, :]
        yi = lo[:, 1:2] + offsets[None, :]
        zi = lo[:, 2:3] + offsets[None, :]
        vx = xi < hi[:, 0:1]
        vy = yi < hi[:, 1:2]
        vz = zi < hi[:, 2:3]

        dx = xi.to(torch.float32)[:, None, None, :] - c[:, 0, None, None, None]
        dy = yi.to(torch.float32)[:, None, :, None] - c[:, 1, None, None, None]
        dz = zi.to(torch.float32)[:, :, None, None] - c[:, 2, None, None, None]
        radius = torch.sqrt(dx * dx + dy * dy + dz * dz)
        contribution = (
            kn_t
            * zn[:, None, None, None]
            * torch.exp(-radius * rp_t)
        )
        mask = vz[:, :, None, None] & vy[:, None, :, None] & vx[:, None, None, :]
        flat_index = (
            zi[:, :, None, None] * (box_size * box_size)
            + yi[:, None, :, None] * box_size
            + xi[:, None, None, :]
        )
        volume_flat.scatter_add_(
            0,
            flat_index[mask].reshape(-1),
            contribution[mask].to(torch.float32).reshape(-1),
        )
    return volume_flat.view(box_size, box_size, box_size)


def rasterize_pdb_atoms(
    coordinates: np.ndarray,
    elements: np.ndarray,
    *,
    box_size: int,
    pixel_size_angstrom: float,
    resolution_angstrom: float,
    device: torch.device,
    backend: str = "vectorized",
    atom_batch_size: int = 1024,
) -> torch.Tensor:
    atomic_numbers, _ = _element_arrays(elements)
    if backend == "reference":
        return _rasterize_reference(
            coordinates,
            atomic_numbers,
            box_size=box_size,
            pixel_size_angstrom=pixel_size_angstrom,
            resolution_angstrom=resolution_angstrom,
            device=device,
        )
    if backend != "vectorized":
        raise ValueError(f"unknown PDB raster backend: {backend}")
    return _rasterize_vectorized(
        coordinates,
        atomic_numbers,
        box_size=box_size,
        pixel_size_angstrom=pixel_size_angstrom,
        resolution_angstrom=resolution_angstrom,
        device=device,
        atom_batch_size=atom_batch_size,
    )


def _resolved_device(text: str, devices: Sequence[int]) -> torch.device:
    if text == "auto":
        if torch.cuda.is_available():
            index = int(devices[0]) if devices else 0
            return torch.device(f"cuda:{index}")
        return torch.device("cpu")
    if text == "cuda":
        index = int(devices[0]) if devices else 0
        return torch.device(f"cuda:{index}")
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _default_paths(cfg: MatchConfig, pixel_size: float, resolution: float) -> tuple[Path, Path, Path]:
    tcfg = cfg.template
    pdb_path = Path(tcfg.pdb_path)
    cache_root = Path(cfg.output_dir) / "_template_cache"
    chain_tag = "all"
    if tcfg.include_chains:
        chain_tag = "inc-" + "-".join(tcfg.include_chains)
    if tcfg.exclude_chains:
        chain_tag += "_exc-" + "-".join(tcfg.exclude_chains)
    key = (
        f"{pdb_path.stem}_box{int(tcfg.box_size or 0)}"
        f"_apix{pixel_size:.6f}_res{resolution:.6f}_{chain_tag}"
    ).replace(" ", "_")
    mrc_path = Path(tcfg.generated_mrc_path) if tcfg.generated_mrc_path else cache_root / f"{key}.mrc"
    center_path = (
        Path(tcfg.center_metadata_path)
        if tcfg.center_metadata_path
        else cache_root / f"{pdb_path.stem}_center.json"
    )
    metadata_path = mrc_path.with_suffix(mrc_path.suffix + ".json")
    return mrc_path, center_path, metadata_path


def _template_signature(
    cfg: MatchConfig,
    *,
    pdb_sha256: str,
    pixel_size: float,
    resolution: float,
    center: np.ndarray,
) -> dict[str, Any]:
    tcfg = cfg.template
    return {
        "schema_version": 1,
        "pdb_sha256": pdb_sha256,
        "box_size": int(tcfg.box_size or 0),
        "pixel_size_angstrom": float(pixel_size),
        "resolution_angstrom": float(resolution),
        "center_enabled": bool(tcfg.center),
        "center_angstrom": [float(v) for v in center],
        "include_hetatm": bool(tcfg.include_hetatm),
        "include_chains": list(tcfg.include_chains),
        "exclude_chains": list(tcfg.exclude_chains),
        "y_flip": bool(tcfg.y_flip),
        "raster_backend": str(tcfg.raster_backend),
        "atom_batch_size": int(tcfg.atom_batch_size),
    }


def _can_reuse(metadata_path: Path, mrc_path: Path, signature: dict[str, Any]) -> bool:
    if not metadata_path.exists() or not mrc_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return metadata.get("signature") == signature


def generate_pdb_template(
    cfg: MatchConfig,
    *,
    device_override: str | None = None,
) -> PDBTemplateResult:
    """Generate/cache a cubic MRC from the PDB template configuration."""

    cfg.template.validate()
    if cfg.template.source != "pdb":
        raise ValueError("generate_pdb_template requires template.source='pdb'")
    tcfg = cfg.template
    pdb_path = Path(tcfg.pdb_path).expanduser().resolve()
    if not pdb_path.exists():
        raise FileNotFoundError(pdb_path)
    pixel_size = float(
        tcfg.pixel_size_angstrom
        if tcfg.pixel_size_angstrom is not None
        else cfg.search.pixel_size_angstrom
    )
    resolution = float(
        tcfg.resolution_angstrom
        if tcfg.resolution_angstrom is not None
        else 2.0 * pixel_size
    )
    box_size = int(tcfg.box_size or 0)
    mrc_path, center_path, metadata_path = _default_paths(cfg, pixel_size, resolution)
    mrc_path.parent.mkdir(parents=True, exist_ok=True)
    center_path.parent.mkdir(parents=True, exist_ok=True)

    parsed = parse_pdb_text(pdb_path)
    base = parsed.subset(_base_atom_mask(parsed, tcfg.include_hetatm))
    if base.atom_count == 0:
        raise ValueError("PDB selection contains no atoms after HETATM filtering")

    pdb_hash = _pdb_sha256(pdb_path)
    center_reference = Path(tcfg.center_reference_json) if tcfg.center_reference_json else None
    center_result_path = center_reference if center_reference is not None else center_path
    if center_reference is not None:
        center = _load_center_reference(center_reference)
        center_source = str(center_reference)
    elif center_path.exists() and _center_metadata_matches(
        center_path, pdb_hash, tcfg.include_hetatm
    ):
        # The sidecar is deliberately independent of chain include/exclude
        # filters, so later cross-validation templates reuse the original full
        # structure center.
        center = _load_center_reference(center_path)
        center_source = str(center_path)
    else:
        center = mass_weighted_center(base.coordinates, base.elements)
        center_source = "computed_from_original_unfiltered_chains"
        _, base_masses = _element_arrays(base.elements)
        center_payload = {
            "schema_version": 1,
            "source_pdb": str(pdb_path),
            "pdb_sha256": pdb_hash,
            "center_method": "mass_weighted",
            "center_angstrom": [float(v) for v in center],
            "include_hetatm": bool(tcfg.include_hetatm),
            "atom_count": base.atom_count,
            "mass_supported_atom_count": int(np.count_nonzero(base_masses)),
            "chain_counts": dict(sorted(Counter(base.chain_ids.tolist()).items())),
            "element_counts": dict(sorted(Counter(base.elements.tolist()).items())),
        }
        write_json(center_path, center_payload)

    selected_mask = _chain_mask(base, tcfg.include_chains, tcfg.exclude_chains)
    selected = base.subset(selected_mask)
    if selected.atom_count == 0:
        raise ValueError("PDB selection contains no atoms after chain filtering")
    shifted_coordinates = selected.coordinates.copy()
    if tcfg.center:
        shifted_coordinates -= center[None, :]

    signature = _template_signature(
        cfg,
        pdb_sha256=pdb_hash,
        pixel_size=pixel_size,
        resolution=resolution,
        center=center,
    )
    if tcfg.reuse_generated and _can_reuse(metadata_path, mrc_path, signature):
        return PDBTemplateResult(
            mrc_path=mrc_path,
            center_metadata_path=center_result_path,
            template_metadata_path=metadata_path,
            center_angstrom=tuple(float(v) for v in center),
            pixel_size_angstrom=pixel_size,
            resolution_angstrom=resolution,
            box_size=box_size,
            atom_count=selected.atom_count,
            reused=True,
        )

    device = _resolved_device(device_override or cfg.runtime.device, cfg.runtime.devices)
    started = time.perf_counter()
    volume = rasterize_pdb_atoms(
        shifted_coordinates,
        selected.elements,
        box_size=box_size,
        pixel_size_angstrom=pixel_size,
        resolution_angstrom=resolution,
        device=device,
        backend=tcfg.raster_backend,
        atom_batch_size=tcfg.atom_batch_size,
    )
    if tcfg.y_flip:
        volume = torch.flip(volume, dims=(1,))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    write_mrc(mrc_path, volume, pixel_size)

    atomic_numbers, atomic_masses = _element_arrays(selected.elements)
    metadata = {
        "schema_version": 1,
        "signature": signature,
        "source_pdb": str(pdb_path),
        "generated_mrc": str(mrc_path),
        "center_metadata": str(center_result_path),
        "center_source": center_source,
        "center_angstrom": [float(v) for v in center],
        "coordinate_shift_angstrom": [float(-v) if tcfg.center else 0.0 for v in center],
        "original_atom_count_after_record_filter": base.atom_count,
        "selected_atom_count": selected.atom_count,
        "selected_chain_counts": dict(sorted(Counter(selected.chain_ids.tolist()).items())),
        "selected_element_counts": dict(sorted(Counter(selected.elements.tolist()).items())),
        "unknown_element_atom_count": int(np.count_nonzero(atomic_numbers == 0)),
        "mass_supported_atom_count": int(np.count_nonzero(atomic_masses)),
        "box_size": box_size,
        "pixel_size_angstrom": pixel_size,
        "resolution_angstrom": resolution,
        "raster_backend": tcfg.raster_backend,
        "atom_batch_size": tcfg.atom_batch_size,
        "device": str(device),
        "elapsed_seconds": elapsed,
    }
    write_json(metadata_path, metadata)
    return PDBTemplateResult(
        mrc_path=mrc_path,
        center_metadata_path=center_result_path,
        template_metadata_path=metadata_path,
        center_angstrom=tuple(float(v) for v in center),
        pixel_size_angstrom=pixel_size,
        resolution_angstrom=resolution,
        box_size=box_size,
        atom_count=selected.atom_count,
        reused=False,
    )


def resolve_template_for_match(
    cfg: MatchConfig,
    *,
    device_override: str | None = None,
) -> tuple[MatchConfig, PDBTemplateResult | None]:
    """Return a config whose numerical core always sees ``template_mrc``."""

    if cfg.template.source == "mrc":
        cfg.validate()
        return cfg, None
    result = generate_pdb_template(cfg, device_override=device_override)
    resolved = MatchConfig.from_dict(cfg.to_dict())
    resolved.template_mrc = str(result.mrc_path)
    # The core remains unaware of PDB input and therefore retains its validated
    # v0.3.5 MRC pathway.
    resolved.template.source = "mrc"
    resolved.validate()
    return resolved, result
