from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class HistogramSummary:
    path: str
    format: str
    first_snr: float
    first_expected_survival: float
    inferred_independent_trials: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normal_survival(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def read_histogram_summary(
    path: str | Path,
    *,
    expected_false_positives: float = 1.0,
) -> HistogramSummary:
    """Read either cisTEM's whitespace histogram or this package's TSV.

    The expected-survival value in the first, very negative-SNR bin is almost
    the effective number of independent trials.  We divide by the exact normal
    survival and multiply by ``expected_false_positives`` so the diagnostic
    remains valid when a non-default false-positive target was used.
    """
    path = Path(path)
    if expected_false_positives <= 0:
        raise ValueError("expected_false_positives must be positive")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    first_data = next((line for line in lines if line.strip() and not line.lstrip().startswith("#")), None)
    if first_data is None:
        raise ValueError(f"{path} has no histogram data rows")

    if "\t" in first_data and any("expected_gaussian_survival" in line for line in lines[:3]):
        reader = csv.DictReader(lines, delimiter="\t")
        row = next(reader, None)
        if row is None:
            raise ValueError(f"{path} has no TSV histogram rows")
        z = float(row["snr_midpoint"])
        expected = float(row["expected_gaussian_survival"])
        fmt = "cistem2dtm-tsv"
    else:
        fields = first_data.split()
        if len(fields) < 4:
            raise ValueError(
                f"{path} does not look like a cisTEM histogram: expected at least four columns"
            )
        z = float(fields[0])
        expected = float(fields[3])
        fmt = "cistem-whitespace"

    survival = _normal_survival(z)
    if not 0.0 < survival <= 1.0:
        raise ValueError(f"invalid normal survival at first SNR {z}")
    inferred = expected * float(expected_false_positives) / survival
    return HistogramSummary(
        path=str(path),
        format=fmt,
        first_snr=z,
        first_expected_survival=expected,
        inferred_independent_trials=inferred,
    )


def diagnose_histogram_against_plan(
    summary: HistogramSummary,
    plan: dict[str, Any],
    *,
    independent_fraction: float,
    ignore_defocus_for_threshold: bool,
) -> dict[str, Any]:
    n_orient = int(plan["number_of_orientations"])
    n_defocus = int(plan["number_of_defocus_positions"])
    n_pixel = int(plan["number_of_pixel_size_positions"])
    n_valid = int(plan["number_of_valid_search_pixels"])
    n_statistics = int(plan.get("number_of_statistics_pixels", n_valid))
    n_threshold = int(plan.get("number_of_threshold_pixels", n_valid))
    search_h, search_w = (int(v) for v in plan["ccf_search_grid_shape"])
    n_full_grid = search_h * search_w
    if (
        n_orient <= 0
        or n_defocus <= 0
        or n_pixel <= 0
        or n_valid <= 0
        or n_statistics <= 0
        or n_threshold <= 0
    ):
        raise ValueError("plan contains a non-positive search count")

    fraction = float(independent_fraction)
    if ignore_defocus_for_threshold and n_defocus > 0:
        fraction /= n_defocus
    ccf_images = n_orient * n_defocus * n_pixel
    predicted = n_threshold * ccf_images * fraction
    predicted_statistics = n_statistics * ccf_images * fraction
    predicted_source_roi = n_valid * ccf_images * fraction
    predicted_full_grid = n_full_grid * ccf_images * fraction
    observed = float(summary.inferred_independent_trials)
    ratio = observed / predicted if predicted else math.nan
    full_grid_ratio = observed / predicted_full_grid if predicted_full_grid else math.nan

    denominator = ccf_images * fraction
    inferred_valid = observed / denominator if denominator else math.nan
    nearest_square = int(round(math.sqrt(inferred_valid))) if inferred_valid >= 0 else -1
    square_error = inferred_valid - nearest_square * nearest_square if nearest_square >= 0 else math.nan

    return {
        "histogram": summary.to_dict(),
        "configured_search": {
            "number_of_out_of_plane_positions": int(plan["number_of_out_of_plane_positions"]),
            "last_out_of_plane_grid_index": int(plan["last_out_of_plane_grid_index"]),
            "number_of_psi_positions": int(plan["number_of_psi_positions"]),
            "number_of_orientations": n_orient,
            "number_of_defocus_positions": n_defocus,
            "number_of_pixel_size_positions": n_pixel,
            "number_of_ccf_images": ccf_images,
            "number_of_valid_search_pixels": n_valid,
            "number_of_statistics_pixels": n_statistics,
            "number_of_threshold_pixels": n_threshold,
            "number_of_full_search_grid_pixels": n_full_grid,
            "independent_fraction_effective": fraction,
            "predicted_independent_trials": predicted,
            "predicted_statistics_roi_trials": predicted_statistics,
            "predicted_source_roi_trials": predicted_source_roi,
            "predicted_full_search_grid_trials": predicted_full_grid,
            "ccf_search_grid_shape": plan["ccf_search_grid_shape"],
            "output_grid_shape": plan["output_grid_shape"],
            "search_pixel_size_angstrom": plan["search_pixel_size_angstrom"],
            "output_pixel_size_angstrom": plan["output_pixel_size_angstrom"],
        },
        "comparison": {
            "observed_to_predicted_ratio": ratio,
            "absolute_trial_difference": observed - predicted,
            "relative_trial_difference": ratio - 1.0,
            "matches_within_0p1_percent": abs(ratio - 1.0) <= 0.001,
            "observed_to_full_search_grid_ratio": full_grid_ratio,
            "matches_full_search_grid_within_0p1_percent": abs(full_grid_ratio - 1.0) <= 0.001,
        },
        "factorization_hint": {
            "inferred_valid_spatial_pixels_if_all_other_counts_match": inferred_valid,
            "nearest_square_dimension": nearest_square,
            "nearest_square_pixels": nearest_square * nearest_square if nearest_square >= 0 else None,
            "difference_from_nearest_square_pixels": square_error,
            "warning": (
                "This is only a factorization hint. A mismatch may instead come from defocus, "
                "pixel-size, orientation-subset, or independence-factor settings."
            ),
        },
    }
