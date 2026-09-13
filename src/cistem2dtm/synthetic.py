from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np

from .config import DebugConfig, MatchConfig, MicroscopeConfig, RuntimeConfig, SearchConfig


def _write_mrc(path: Path, array: np.ndarray, pixel_size: float) -> None:
    with mrcfile.new(path, overwrite=True) as handle:
        handle.set_data(np.ascontiguousarray(array, dtype=np.float32))
        handle.voxel_size = float(pixel_size)
        handle.update_header_stats()


def generate_synthetic_example(
    output_dir: str | Path,
    *,
    template_size: int = 12,
    search_size: int | None = None,
    pixel_size_angstrom: float = 1.5,
    noise_sigma: float = 0.08,
    seed: int = 7,
) -> dict[str, Path]:
    """Create a fast asymmetric 2DTM example and a ready-to-run JSON config."""
    if template_size < 8:
        raise ValueError("template_size must be at least 8")
    if template_size % 2:
        template_size += 1
    search_size = int(search_size or 2 * template_size)
    if search_size < template_size:
        raise ValueError("search_size must be at least template_size")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n = template_size
    z, y, x = np.mgrid[:n, :n, :n].astype(np.float32)
    c = (n - 1.0) / 2.0
    main = np.exp(-(((x - c) / (0.17 * n)) ** 2 + ((y - c) / (0.24 * n)) ** 2 + ((z - c) / (0.20 * n)) ** 2))
    satellite = 0.65 * np.exp(
        -(
            ((x - (c + 0.18 * n)) / (0.10 * n)) ** 2
            + ((y - (c - 0.16 * n)) / (0.12 * n)) ** 2
            + ((z - (c + 0.10 * n)) / (0.11 * n)) ** 2
        )
    )
    notch = 0.25 * np.exp(
        -(
            ((x - (c - 0.20 * n)) / (0.09 * n)) ** 2
            + ((y - (c + 0.18 * n)) / (0.10 * n)) ** 2
            + ((z - c) / (0.14 * n)) ** 2
        )
    )
    volume = (main + satellite - notch).astype(np.float32)
    projection = volume.sum(axis=0)
    # The zero-defocus branch of the target cisTEM CTF is the constant -0.7.
    projection = -0.7 * projection
    projection -= projection.mean()
    projection /= max(float(projection.std()), 1.0e-6)

    image = rng.normal(0.0, noise_sigma, (search_size, search_size)).astype(np.float32)
    y0 = (search_size - n) // 2 - 2
    x0 = (search_size - n) // 2 + 3
    image[y0 : y0 + n, x0 : x0 + n] += projection

    template_path = output / "synthetic_template.mrc"
    image_path = output / "synthetic_search.mrc"
    _write_mrc(template_path, volume, pixel_size_angstrom)
    _write_mrc(image_path, image, pixel_size_angstrom)

    config = MatchConfig(
        input_mrc=str(image_path),
        template_mrc=str(template_path),
        output_dir=str(output / "output"),
        output_prefix="synthetic",
        microscope=MicroscopeConfig(
            voltage_kv=300.0,
            spherical_aberration_mm=2.7,
            amplitude_contrast=0.07,
            defocus1_angstrom=0.0,
            defocus2_angstrom=0.0,
            defocus_angle_deg=0.0,
            phase_shift_deg=0.0,
        ),
        search=SearchConfig(
            pixel_size_angstrom=pixel_size_angstrom,
            low_resolution_limit_angstrom=100.0,
            high_resolution_limit_angstrom=2.0 * pixel_size_angstrom,
            angular_step_deg=90.0,
            in_plane_step_deg=180.0,
            symmetry="C1",
            particle_radius_angstrom=0.45 * n * pixel_size_angstrom,
            fft_size_mode="exact",
            expected_false_positives=1.0,
            minimum_peak_radius_angstrom=0.35 * n * pixel_size_angstrom,
            maximum_peaks=20,
        ),
        runtime=RuntimeConfig(
            device="auto",
            dtype="float32",
            projector_backend="auto",
            projection_fft_mode="auto",
            orientation_batch_size=4,
            ccf_stack_mode="none",
            cpu_threads=4,
        ),
        debug=DebugConfig(),
    )
    config_path = output / "synthetic_config.json"
    config.save_json(config_path)
    truth_path = output / "synthetic_truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "template_top_left_x_pixel": x0,
                "template_top_left_y_pixel": y0,
                "template_center_x_pixel": x0 + (n - 1) / 2.0,
                "template_center_y_pixel": y0 + (n - 1) / 2.0,
                "note": "This is a smoke/diagnostic example, not a cisTEM numerical oracle.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "template": template_path,
        "search": image_path,
        "config": config_path,
        "truth": truth_path,
    }
