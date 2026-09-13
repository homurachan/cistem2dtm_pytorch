from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Optional


DTypeName = Literal["float32", "float16", "bfloat16"]
ProjectorBackend = Literal["auto", "triton", "gather", "grid_sample"]
ProjectionFFTMode = Literal["auto", "direct_s", "explicit"]
CCFStackMode = Literal["none", "cpu", "mrc"]
PaddingMode = Literal["noise", "zero", "edge", "replicate"]
StatisticsROIMode = Literal["source", "precompiled", "full"]
ThresholdPixelMode = Literal["source", "statistics", "full"]
FFTSizeMode = Literal["cistem-standard", "power2", "exact"]
PixelSizeBackend = Literal["cistem_exact", "coordinate"]
TimingMode = Literal["summary", "sampled", "synchronized"]
HistogramMode = Literal["exact", "sampled", "off"]
HistogramBackend = Literal["histc", "bincount"]
CenterPhaseMode = Literal["auto", "input", "projection"]
CorrelationMultiplyBackend = Literal["auto", "triton", "torch"]
CorrelationPrecision = Literal["float32", "mixed_float16"]
HalfFFTShapeMode = Literal["pad_to_power2", "require_power2", "fallback_float32"]
WeightingMode = Literal["cistem", "gisspa_repo"]
TemplateSourceMode = Literal["mrc", "pdb"]
PDBRasterBackend = Literal["vectorized", "reference"]


@dataclass(slots=True)
class MicroscopeConfig:
    voltage_kv: float = 300.0
    spherical_aberration_mm: float = 2.7
    amplitude_contrast: float = 0.07
    defocus1_angstrom: float = 10000.0
    defocus2_angstrom: float = 10000.0
    defocus_angle_deg: float = 0.0
    phase_shift_deg: float = 0.0

    def validate(self) -> None:
        if self.voltage_kv <= 0:
            raise ValueError("voltage_kv must be positive")
        if self.spherical_aberration_mm < 0:
            raise ValueError("spherical_aberration_mm cannot be negative")
        if not 0.0 <= self.amplitude_contrast <= 1.0:
            raise ValueError("amplitude_contrast must be between 0 and 1")


@dataclass(slots=True)
class SearchConfig:
    pixel_size_angstrom: float = 1.0
    low_resolution_limit_angstrom: float = 300.0
    high_resolution_limit_angstrom: float = 8.0
    angular_step_deg: float = 5.0  # 0 selects the cisTEM automatic rule
    in_plane_step_deg: Optional[float] = None
    symmetry: str = "C1"
    particle_radius_angstrom: float = 100.0  # 0 reproduces cisTEM's 200 A fallback
    template_padding: float = 1.0

    defocus_search_range_angstrom: float = 0.0
    defocus_step_angstrom: float = 50.0
    pixel_size_search_range_angstrom: float = 0.0
    pixel_size_step_angstrom: float = 0.02

    image_slice: int = 1
    apply_result_rescaling: bool = False
    disable_flat_fielding: bool = False
    expected_false_positives: float = 1.0
    # Optional direct score threshold.  null preserves cisTEM's theoretical threshold.
    threshold_override: Optional[float] = None
    fraction_of_search_positions_independent: float = 1.0
    ignore_defocus_for_threshold: bool = False
    minimum_peak_radius_angstrom: Optional[float] = None
    maximum_peaks: int = 1000

    preserve_psi_360_duplicate: bool = True
    mip_initial_value: float = 0.0
    fft_size_mode: FFTSizeMode = "cistem-standard"
    max_search_size: int = 1024
    # cisTEM fills real-space FFT padding with N(0, 1) noise by default after
    # first-pass whitening has normalized the image variance to one.
    padding_mode: PaddingMode = "noise"
    random_seed: int = 0

    # Histogram-only ROI. ``source`` follows the 5bc5f8c source-valid ROI.
    # ``precompiled`` is an empirical compatibility profile inferred from a
    # reference binary histogram, and ``full`` uses the complete FFT work grid.
    # IMPORTANT: this ROI no longer controls correlation sum/sumSq moments;
    # moments follow the source-valid ROI and the final post-search mask, as in
    # cisTEM ResizeImage_postSearch.
    statistics_roi_mode: StatisticsROIMode = "precompiled"
    statistics_roi_border_pixels: Optional[int] = None
    # The reference precompiled binary forms the theoretical threshold from
    # the full padded FFT grid, while accumulating observed moments/histogram
    # over the smaller statistics ROI.
    threshold_pixel_mode: ThresholdPixelMode = "full"
    allow_rotation_for_speed: bool = False

    def validate(self) -> None:
        if self.pixel_size_angstrom <= 0:
            raise ValueError("pixel_size_angstrom must be positive")
        if self.high_resolution_limit_angstrom <= 0:
            raise ValueError("high_resolution_limit_angstrom must be positive")
        if self.low_resolution_limit_angstrom <= 0:
            raise ValueError("low_resolution_limit_angstrom must be positive")
        if self.angular_step_deg < 0:
            raise ValueError("angular_step_deg cannot be negative; 0 selects automatic spacing")
        if self.in_plane_step_deg is not None and self.in_plane_step_deg <= 0:
            raise ValueError("in_plane_step_deg must be positive")
        if self.particle_radius_angstrom < 0:
            raise ValueError("particle_radius_angstrom cannot be negative")
        if abs(self.template_padding - 1.0) > 1.0e-6:
            raise ValueError(
                "cisTEM commit 5bc5f8c asserts template_padding == 1.0; "
                "this implementation preserves that constraint"
            )
        if self.defocus_search_range_angstrom < 0:
            raise ValueError("defocus_search_range_angstrom cannot be negative")
        if self.defocus_search_range_angstrom > 0 and self.defocus_step_angstrom <= 0:
            raise ValueError("defocus_step_angstrom must be positive")
        if self.pixel_size_search_range_angstrom < 0:
            raise ValueError("pixel_size_search_range_angstrom cannot be negative")
        if self.pixel_size_search_range_angstrom > 0 and self.pixel_size_step_angstrom <= 0:
            raise ValueError("pixel_size_step_angstrom must be positive")
        if self.expected_false_positives <= 0:
            raise ValueError("expected_false_positives must be positive")
        if self.threshold_override is not None and not math.isfinite(float(self.threshold_override)):
            raise ValueError("threshold_override must be finite or null")
        if not 0 < self.fraction_of_search_positions_independent <= 1:
            raise ValueError("fraction_of_search_positions_independent must be in (0, 1]")
        if self.image_slice < 1:
            raise ValueError("image_slice is 1-based and must be at least 1")
        if self.maximum_peaks < 0:
            raise ValueError("maximum_peaks cannot be negative")
        if self.max_search_size < 8:
            raise ValueError("max_search_size is unreasonably small")
        if self.padding_mode not in {"noise", "zero", "edge", "replicate"}:
            raise ValueError(f"unknown padding_mode: {self.padding_mode}")
        if self.statistics_roi_mode not in {"source", "precompiled", "full"}:
            raise ValueError(f"unknown statistics_roi_mode: {self.statistics_roi_mode}")
        if self.statistics_roi_border_pixels is not None and self.statistics_roi_border_pixels < 0:
            raise ValueError("statistics_roi_border_pixels cannot be negative")
        if self.threshold_pixel_mode not in {"source", "statistics", "full"}:
            raise ValueError(f"unknown threshold_pixel_mode: {self.threshold_pixel_mode}")


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "auto"
    devices: list[int] = field(default_factory=list)
    dtype: DTypeName = "float32"
    projector_backend: ProjectorBackend = "auto"
    triton_fallback_to_gather: bool = True
    triton_block_size: int = 256
    triton_num_warps: int = 4
    cache_grid_sample_source: bool = True
    cache_sampled_projection_filter: bool = True
    cache_orientation_tensors: bool = True
    precompute_rotation_matrices: bool = True
    orientation_precompute_chunk_size: int = 262_144
    projection_fft_mode: ProjectionFFTMode = "auto"
    # Move the direct-s center-embedding phase to the input Fourier image once
    # per projection size. ``projection`` retains the v0.3.3 operation order.
    center_phase_mode: CenterPhaseMode = "auto"
    # Fused image * conjugate(projection) kernel. ``auto`` selects Triton on
    # CUDA and the strict PyTorch implementation elsewhere.
    correlation_multiply_backend: CorrelationMultiplyBackend = "auto"
    correlation_triton_fallback_to_torch: bool = True
    correlation_triton_block_size: int = 1024
    correlation_triton_num_warps: int = 4
    # Scientific preprocessing/projector normalization remains float32.  The
    # optional mixed path uses half/chalf only for the large correlation FFTs.
    correlation_precision: CorrelationPrecision = "float32"
    # Half cuFFT requires power-of-two transform lengths.  pad_to_power2 turns
    # a standard 4096x3888 work grid into 4096x4096 only in mixed mode.
    half_fft_shape_mode: HalfFFTShapeMode = "pad_to_power2"
    mixed_precision_fallback_to_float32: bool = True
    # Range-safe mixed-half FFT.  Each normalized projection is mean-centered,
    # scaled by an exact power of two before half RFFT, and restored inside the
    # fused correlation multiply.
    mixed_precision_range_scaling: bool = True
    mixed_precision_l1_target: float = 16384.0
    mixed_precision_health_check_batches: int = 1
    pixel_size_backend: PixelSizeBackend = "cistem_exact"
    pixel_size_tolerance: float = 0.001
    max_pixel_resample_dimension: int = 1536
    orientation_batch_size: int = 32
    ccf_stack_mode: CCFStackMode = "none"
    ccf_stack_path: Optional[str] = None
    pin_memory: bool = True
    deterministic: bool = False
    compile: bool = False
    cpu_threads: int = 4
    first_orientation: int = 0
    last_orientation: Optional[int] = None
    save_raw_accumulators: bool = True

    # Empirical CCC histogram controls. ``exact`` visits every CCC sample,
    # ``sampled`` deterministically decimates orientations and pixels then
    # reweights the counts, and ``off`` skips the diagnostic histogram.
    histogram_mode: HistogramMode = "exact"
    histogram_backend: HistogramBackend = "histc"
    histogram_sample_orientation_stride: int = 16
    histogram_sample_pixel_stride: int = 4

    def validate(self) -> None:
        if self.orientation_batch_size <= 0:
            raise ValueError("orientation_batch_size must be positive")
        if self.projector_backend not in {"auto", "triton", "gather", "grid_sample"}:
            raise ValueError(f"unknown projector_backend: {self.projector_backend}")
        if self.triton_block_size <= 0 or self.triton_block_size & (self.triton_block_size - 1):
            raise ValueError("triton_block_size must be a positive power of two")
        if self.triton_num_warps not in {1, 2, 4, 8}:
            raise ValueError("triton_num_warps must be one of 1, 2, 4, 8")
        if self.orientation_precompute_chunk_size <= 0:
            raise ValueError("orientation_precompute_chunk_size must be positive")
        if self.projection_fft_mode not in {"auto", "direct_s", "explicit"}:
            raise ValueError(f"unknown projection_fft_mode: {self.projection_fft_mode}")
        if self.center_phase_mode not in {"auto", "input", "projection"}:
            raise ValueError(f"unknown center_phase_mode: {self.center_phase_mode}")
        if self.correlation_multiply_backend not in {"auto", "triton", "torch"}:
            raise ValueError(
                f"unknown correlation_multiply_backend: {self.correlation_multiply_backend}"
            )
        if (
            self.correlation_triton_block_size <= 0
            or self.correlation_triton_block_size
            & (self.correlation_triton_block_size - 1)
        ):
            raise ValueError(
                "correlation_triton_block_size must be a positive power of two"
            )
        if self.correlation_triton_num_warps not in {1, 2, 4, 8}:
            raise ValueError(
                "correlation_triton_num_warps must be one of 1, 2, 4, 8"
            )
        if self.correlation_precision not in {"float32", "mixed_float16"}:
            raise ValueError(f"unknown correlation_precision: {self.correlation_precision}")
        if self.half_fft_shape_mode not in {
            "pad_to_power2", "require_power2", "fallback_float32"
        }:
            raise ValueError(f"unknown half_fft_shape_mode: {self.half_fft_shape_mode}")
        if self.correlation_precision == "mixed_float16" and self.center_phase_mode == "projection":
            raise ValueError(
                "mixed_float16 requires center_phase_mode='input' or 'auto'; half/chalf "
                "phase multiplication is deliberately kept out of the per-batch projection path"
            )
        if self.mixed_precision_l1_target <= 0 or not math.isfinite(float(self.mixed_precision_l1_target)):
            raise ValueError("mixed_precision_l1_target must be finite and positive")
        if self.mixed_precision_health_check_batches < 0:
            raise ValueError("mixed_precision_health_check_batches cannot be negative")
        if self.pixel_size_tolerance <= 0:
            raise ValueError("pixel_size_tolerance must be positive")
        if self.max_pixel_resample_dimension < 8:
            raise ValueError("max_pixel_resample_dimension is unreasonably small")
        if self.first_orientation < 0:
            raise ValueError("first_orientation cannot be negative")
        if self.last_orientation is not None and self.last_orientation < self.first_orientation:
            raise ValueError("last_orientation must be >= first_orientation")
        if self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive")
        if self.histogram_mode not in {"exact", "sampled", "off"}:
            raise ValueError(f"unknown histogram_mode: {self.histogram_mode}")
        if self.histogram_backend not in {"histc", "bincount"}:
            raise ValueError(f"unknown histogram_backend: {self.histogram_backend}")
        if self.histogram_sample_orientation_stride <= 0:
            raise ValueError("histogram_sample_orientation_stride must be positive")
        if self.histogram_sample_pixel_stride <= 0:
            raise ValueError("histogram_sample_pixel_stride must be positive")


@dataclass(slots=True)
class WeightingConfig:
    """Frequency-weighting backend.

    ``cistem`` preserves v0.3.4 exactly.  ``gisspa_repo`` implements the six-
    coefficient repository model and per-projection radial whitening while
    retaining the current 3-D projector, MIP/scaled-MIP and I/O paths.
    """

    mode: WeightingMode = "cistem"
    kk: float = 3.0
    a: float = -9.32
    b: float = 2.65
    b2: float = 0.01908
    bfactor: float = -78.7757
    bfactor2: float = -12.9121
    bfactor3: float = 1.28732
    cosine_edge_width_pixels: float = 8.0
    image_high_frequency_damping: bool = True
    projection_whitening_epsilon: float = 1.0e-12

    def validate(self) -> None:
        if self.mode not in {"cistem", "gisspa_repo"}:
            raise ValueError(f"unknown weighting mode: {self.mode}")
        if self.kk < 0 or not math.isfinite(float(self.kk)):
            raise ValueError("weighting.kk must be finite and non-negative")
        for name in ("a", "b", "b2", "bfactor", "bfactor2", "bfactor3"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"weighting.{name} must be finite")
        if self.cosine_edge_width_pixels <= 0:
            raise ValueError("weighting.cosine_edge_width_pixels must be positive")
        if self.projection_whitening_epsilon <= 0:
            raise ValueError("weighting.projection_whitening_epsilon must be positive")


@dataclass(slots=True)
class TemplateInputConfig:
    """Optional PDB-to-MRC template preprocessing.

    The default ``mrc`` mode preserves every existing v0.3.5 configuration.
    ``pdb`` mode is resolved by the CLI/batch input layer before TemplateMatcher
    is called, so the numerical matching core continues to receive a normal
    cubic MRC template.
    """

    source: TemplateSourceMode = "mrc"
    pdb_path: str = ""
    box_size: Optional[int] = None
    pixel_size_angstrom: Optional[float] = None
    resolution_angstrom: Optional[float] = None
    center: bool = True
    include_hetatm: bool = False
    include_chains: list[str] = field(default_factory=list)
    exclude_chains: list[str] = field(default_factory=list)
    center_reference_json: Optional[str] = None
    center_metadata_path: Optional[str] = None
    generated_mrc_path: Optional[str] = None
    reuse_generated: bool = True
    y_flip: bool = False
    raster_backend: PDBRasterBackend = "vectorized"
    atom_batch_size: int = 1024

    def validate(self) -> None:
        if self.source not in {"mrc", "pdb"}:
            raise ValueError(f"unknown template source: {self.source}")
        if self.source == "pdb":
            if not self.pdb_path:
                raise ValueError("template.pdb_path is required when template.source='pdb'")
            if self.box_size is None or self.box_size < 8:
                raise ValueError("template.box_size must be at least 8 for PDB templates")
            if self.pixel_size_angstrom is not None and self.pixel_size_angstrom <= 0:
                raise ValueError("template.pixel_size_angstrom must be positive or null")
            if self.resolution_angstrom is not None and self.resolution_angstrom <= 0:
                raise ValueError("template.resolution_angstrom must be positive or null")
        if self.raster_backend not in {"vectorized", "reference"}:
            raise ValueError(f"unknown template.raster_backend: {self.raster_backend}")
        if self.atom_batch_size <= 0:
            raise ValueError("template.atom_batch_size must be positive")
        include = set(self.include_chains)
        exclude = set(self.exclude_chains)
        overlap = include & exclude
        if overlap:
            raise ValueError(f"chains cannot be both included and excluded: {sorted(overlap)}")


@dataclass(slots=True)
class BatchConfig:
    """RELION micrograph-STAR batch execution controls."""

    micrograph_root: Optional[str] = None
    output_subdirectories: bool = True
    continue_on_error: bool = False
    write_resolved_configs: bool = True
    manifest_prefix: str = "batch"
    first_micrograph: int = 0
    last_micrograph: Optional[int] = None

    def validate(self) -> None:
        if not self.manifest_prefix:
            raise ValueError("batch.manifest_prefix cannot be empty")
        if self.first_micrograph < 0:
            raise ValueError("batch.first_micrograph cannot be negative")
        if self.last_micrograph is not None and self.last_micrograph < self.first_micrograph:
            raise ValueError("batch.last_micrograph must be >= batch.first_micrograph")


@dataclass(slots=True)
class TimingConfig:
    """Performance timing controls independent of numerical debug output.

    ``summary`` records synchronized wall time only at major stage boundaries.
    ``sampled`` additionally records detailed CUDA-event timings for selected
    orientation batches. ``synchronized`` profiles every batch and is intended
    only for short benchmark subsets because it adds one synchronization per
    batch.
    """

    enabled: bool = False
    mode: TimingMode = "sampled"
    output_dir: Optional[str] = None
    warmup_batches: int = 5
    sample_every_batches: int = 100
    max_samples: int = 200
    progress_every_batches: int = 0
    print_summary: bool = True
    save_batch_samples: bool = True
    record_cuda_memory: bool = True
    emit_nvtx: bool = False

    def validate(self) -> None:
        if self.mode not in {"summary", "sampled", "synchronized"}:
            raise ValueError(f"unknown timing mode: {self.mode}")
        if self.warmup_batches < 0:
            raise ValueError("timing.warmup_batches cannot be negative")
        if self.sample_every_batches <= 0:
            raise ValueError("timing.sample_every_batches must be positive")
        if self.max_samples < 0:
            raise ValueError("timing.max_samples cannot be negative; 0 means unlimited")
        if self.progress_every_batches < 0:
            raise ValueError("timing.progress_every_batches cannot be negative")


@dataclass(slots=True)
class DebugConfig:
    debug_dir: Optional[str] = None
    level: int = 0
    save_tensors: bool = False
    save_mrc: bool = True
    save_npy: bool = True
    stop_after_stage: int = 0
    save_every_n_orientations: int = 0
    single_orientation_phi_deg: Optional[float] = None
    single_orientation_theta_deg: Optional[float] = None
    single_orientation_psi_deg: Optional[float] = None
    compare_projectors: bool = False
    record_timing: bool = False

    def validate(self) -> None:
        values = (
            self.single_orientation_phi_deg,
            self.single_orientation_theta_deg,
            self.single_orientation_psi_deg,
        )
        if any(v is not None for v in values) and not all(v is not None for v in values):
            raise ValueError("all three single-orientation angles must be supplied together")
        if self.level < 0:
            raise ValueError("debug level cannot be negative")
        if self.stop_after_stage not in range(0, 11):
            raise ValueError("stop_after_stage must be 0 or 1..10")

    @property
    def enabled(self) -> bool:
        return bool(self.debug_dir) or self.level > 0

    @property
    def single_orientation(self) -> Optional[tuple[float, float, float]]:
        if self.single_orientation_phi_deg is None:
            return None
        return (
            float(self.single_orientation_phi_deg),
            float(self.single_orientation_theta_deg),
            float(self.single_orientation_psi_deg),
        )


@dataclass(slots=True)
class MatchConfig:
    input_mrc: str = ""
    micrographs_star: str = ""
    template_mrc: str = ""
    output_dir: str = "2dtm_output"
    output_prefix: str = "match"
    microscope: MicroscopeConfig = field(default_factory=MicroscopeConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    weighting: WeightingConfig = field(default_factory=WeightingConfig)
    template: TemplateInputConfig = field(default_factory=TemplateInputConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)

    def validate(self, require_paths: bool = True) -> None:
        if require_paths:
            if not self.input_mrc and not self.micrographs_star:
                raise ValueError("input_mrc or micrographs_star is required")
            if self.input_mrc and self.micrographs_star:
                raise ValueError("input_mrc and micrographs_star are mutually exclusive")
            if self.template.source == "mrc" and not self.template_mrc:
                raise ValueError("template_mrc is required when template.source='mrc'")
        if not self.output_prefix:
            raise ValueError("output_prefix cannot be empty")
        self.microscope.validate()
        self.search.validate()
        self.runtime.validate()
        self.weighting.validate()
        self.template.validate()
        self.batch.validate()
        self.debug.validate()
        self.timing.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
        return cls(
            input_mrc=str(data.get("input_mrc", "")),
            micrographs_star=str(data.get("micrographs_star", "")),
            template_mrc=str(data.get("template_mrc", "")),
            output_dir=str(data.get("output_dir", "2dtm_output")),
            output_prefix=str(data.get("output_prefix", "match")),
            microscope=_construct_dataclass(MicroscopeConfig, data.get("microscope", {})),
            search=_construct_dataclass(SearchConfig, data.get("search", {})),
            runtime=_construct_dataclass(RuntimeConfig, data.get("runtime", {})),
            weighting=_construct_dataclass(WeightingConfig, data.get("weighting", {})),
            template=_construct_dataclass(TemplateInputConfig, data.get("template", {})),
            batch=_construct_dataclass(BatchConfig, data.get("batch", {})),
            debug=_construct_dataclass(DebugConfig, data.get("debug", {})),
            timing=_construct_dataclass(TimingConfig, data.get("timing", {})),
        )

    @classmethod
    def load_json(cls, path: str | Path, *, require_paths: bool = True) -> "MatchConfig":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("configuration root must be a JSON object")
        cfg = cls.from_dict(obj)
        cfg.validate(require_paths=require_paths)
        return cfg


def effective_image_fft_size_mode(
    search: SearchConfig, runtime: RuntimeConfig
) -> FFTSizeMode:
    """Return the 2-D search/correlation FFT sizing mode.

    The optional half-correlation backend may pad only the large 2-D search
    grid to powers of two.  The 3-D template FFT keeps ``search.fft_size_mode``
    so a 448^3 template is not silently changed into a 512^3 projector.
    """
    if (
        runtime.correlation_precision == "mixed_float16"
        and runtime.half_fft_shape_mode == "pad_to_power2"
    ):
        return "power2"
    return search.fft_size_mode


def _construct_dataclass(dataclass_type: type[Any], values: Any) -> Any:
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise TypeError(f"{dataclass_type.__name__} configuration must be an object")
    allowed = {f.name for f in fields(dataclass_type)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {dataclass_type.__name__} keys: {sorted(unknown)}")
    return dataclass_type(**values)


def merge_non_none(target: Any, values: dict[str, Any]) -> None:
    """Assign non-None keys to a dataclass; used by the CLI override layer."""
    if not is_dataclass(target):
        raise TypeError("target must be a dataclass instance")
    allowed = {f.name for f in fields(target)}
    for key, value in values.items():
        if value is not None:
            if key not in allowed:
                raise KeyError(key)
            setattr(target, key, value)
