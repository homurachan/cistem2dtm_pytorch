from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import MatchConfig


@dataclass(slots=True)
class StarLoop:
    block: str
    columns: list[str]
    rows: list[dict[str, str]]


@dataclass(slots=True)
class MicrographRecord:
    index: int
    micrograph_name: str
    micrograph_path: Path
    optics_group: str
    pixel_size_angstrom: float
    voltage_kv: float
    spherical_aberration_mm: float
    amplitude_contrast: float
    defocus_u_angstrom: float
    defocus_v_angstrom: float
    defocus_angle_deg: float
    phase_shift_deg: float
    ctf_max_resolution_angstrom: float | None
    ctf_figure_of_merit: float | None
    raw_micrograph_row: dict[str, str]
    raw_optics_row: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "micrograph_name": self.micrograph_name,
            "micrograph_path": str(self.micrograph_path),
            "optics_group": self.optics_group,
            "pixel_size_angstrom": self.pixel_size_angstrom,
            "voltage_kv": self.voltage_kv,
            "spherical_aberration_mm": self.spherical_aberration_mm,
            "amplitude_contrast": self.amplitude_contrast,
            "defocus_u_angstrom": self.defocus_u_angstrom,
            "defocus_v_angstrom": self.defocus_v_angstrom,
            "defocus_angle_deg": self.defocus_angle_deg,
            "phase_shift_deg": self.phase_shift_deg,
            "ctf_max_resolution_angstrom": self.ctf_max_resolution_angstrom,
            "ctf_figure_of_merit": self.ctf_figure_of_merit,
        }


def _tokens(line: str) -> list[str]:
    if line.lstrip().startswith(";"):
        raise ValueError("semicolon-delimited multiline STAR values are not supported")
    return shlex.split(line, comments=False, posix=True)


def parse_star_loops(path: str | Path) -> list[StarLoop]:
    """Read RELION-style STAR loops without starfile/pandas dependencies."""

    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    loops: list[StarLoop] = []
    block = ""
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        lower = stripped.lower()
        if lower.startswith("data_"):
            block = stripped[5:].strip()
            i += 1
            continue
        if lower != "loop_":
            i += 1
            continue

        i += 1
        columns: list[str] = []
        while i < len(lines):
            text = lines[i].strip()
            if not text or text.startswith("#"):
                i += 1
                continue
            if not text.startswith("_"):
                break
            columns.append(text.split()[0])
            i += 1
        if not columns:
            raise ValueError(f"STAR loop without columns in block data_{block}: {path}")

        values: list[str] = []
        while i < len(lines):
            text = lines[i].strip()
            if not text or text.startswith("#"):
                i += 1
                continue
            low = text.lower()
            if low == "loop_" or low.startswith("data_") or text.startswith("_"):
                break
            values.extend(_tokens(text))
            i += 1
        if len(values) % len(columns) != 0:
            raise ValueError(
                f"STAR loop data count {len(values)} is not divisible by {len(columns)} "
                f"columns in data_{block}: {path}"
            )
        rows = [
            dict(zip(columns, values[start : start + len(columns)]))
            for start in range(0, len(values), len(columns))
        ]
        loops.append(StarLoop(block=block, columns=columns, rows=rows))
    return loops


def _loop_for(loops: Sequence[StarLoop], block: str) -> StarLoop:
    matches = [loop for loop in loops if loop.block.lower() == block.lower()]
    if not matches:
        raise ValueError(f"STAR file is missing data_{block}")
    if len(matches) > 1:
        # RELION normally emits one loop per block; merge compatible loops.
        columns = matches[0].columns
        if any(loop.columns != columns for loop in matches[1:]):
            raise ValueError(f"multiple incompatible loops in data_{block}")
        return StarLoop(block=block, columns=columns, rows=[row for loop in matches for row in loop.rows])
    return matches[0]


def _optional_value(*sources: Mapping[str, str], names: Sequence[str]) -> str | None:
    for source in sources:
        for name in names:
            value = source.get(name)
            if value not in (None, "", ".", "?"):
                return value
    return None


def _float_value(
    *sources: Mapping[str, str],
    names: Sequence[str],
    default: float | None = None,
    required: bool = False,
) -> float | None:
    value = _optional_value(*sources, names=names)
    if value is None:
        if required:
            raise ValueError(f"required STAR field is missing: {', '.join(names)}")
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid numeric STAR value {value!r} for {names[0]}") from exc


def _resolve_micrograph_path(raw: str, star_path: Path, root: str | None) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate
    candidates: list[Path] = []
    if root:
        candidates.append(Path(root).expanduser() / candidate)
    candidates.append(star_path.parent / candidate)
    candidates.append(Path.cwd() / candidate)
    for item in candidates:
        if item.exists():
            return item.resolve()
    # Preserve the intended path even when the caller is only planning jobs.
    return candidates[0].resolve()


def read_relion_micrographs(path: str | Path, cfg: MatchConfig) -> list[MicrographRecord]:
    star_path = Path(path).expanduser().resolve()
    loops = parse_star_loops(star_path)
    optics_loop = _loop_for(loops, "optics")
    micrograph_loop = _loop_for(loops, "micrographs")

    optics_by_group: dict[str, dict[str, str]] = {}
    for row in optics_loop.rows:
        group = row.get("_rlnOpticsGroup")
        if group is None:
            raise ValueError("data_optics is missing _rlnOpticsGroup")
        optics_by_group[str(group)] = row

    records: list[MicrographRecord] = []
    for index, row in enumerate(micrograph_loop.rows):
        name = row.get("_rlnMicrographName")
        if not name:
            raise ValueError(f"micrograph row {index} is missing _rlnMicrographName")
        group = str(row.get("_rlnOpticsGroup", "1"))
        optics = optics_by_group.get(group)
        if optics is None:
            raise ValueError(f"micrograph row {index} refers to unknown optics group {group}")

        pixel = _float_value(
            row,
            optics,
            names=(
                "_rlnMicrographPixelSize",
                "_rlnMicrographOriginalPixelSize",
                "_rlnImagePixelSize",
            ),
            default=cfg.search.pixel_size_angstrom,
        )
        voltage = _float_value(row, optics, names=("_rlnVoltage",), default=cfg.microscope.voltage_kv)
        cs = _float_value(
            row,
            optics,
            names=("_rlnSphericalAberration",),
            default=cfg.microscope.spherical_aberration_mm,
        )
        amp = _float_value(
            row,
            optics,
            names=("_rlnAmplitudeContrast",),
            default=cfg.microscope.amplitude_contrast,
        )
        defocus_u = _float_value(row, names=("_rlnDefocusU",), required=True)
        defocus_v = _float_value(row, names=("_rlnDefocusV",), required=True)
        defocus_angle = _float_value(
            row,
            names=("_rlnDefocusAngle",),
            default=cfg.microscope.defocus_angle_deg,
        )
        phase_shift = _float_value(
            row,
            optics,
            names=("_rlnPhaseShift",),
            default=cfg.microscope.phase_shift_deg,
        )
        ctf_max = _float_value(row, names=("_rlnCtfMaxResolution",), default=None)
        ctf_fom = _float_value(row, names=("_rlnCtfFigureOfMerit",), default=None)
        assert pixel is not None and voltage is not None and cs is not None and amp is not None
        assert defocus_u is not None and defocus_v is not None and defocus_angle is not None
        assert phase_shift is not None
        records.append(
            MicrographRecord(
                index=index,
                micrograph_name=str(name),
                micrograph_path=_resolve_micrograph_path(str(name), star_path, cfg.batch.micrograph_root),
                optics_group=group,
                pixel_size_angstrom=float(pixel),
                voltage_kv=float(voltage),
                spherical_aberration_mm=float(cs),
                amplitude_contrast=float(amp),
                defocus_u_angstrom=float(defocus_u),
                defocus_v_angstrom=float(defocus_v),
                defocus_angle_deg=float(defocus_angle),
                phase_shift_deg=float(phase_shift),
                ctf_max_resolution_angstrom=None if ctf_max is None else float(ctf_max),
                ctf_figure_of_merit=None if ctf_fom is None else float(ctf_fom),
                raw_micrograph_row=dict(row),
                raw_optics_row=dict(optics),
            )
        )

    first = min(cfg.batch.first_micrograph, len(records))
    stop = len(records) if cfg.batch.last_micrograph is None else min(cfg.batch.last_micrograph + 1, len(records))
    selected = records[first:stop]
    if not selected:
        raise ValueError("selected micrograph range is empty")
    return selected
