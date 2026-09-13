"""PyTorch cisTEM 2D template matching.

Scientific target: cisTEM match_template.cpp at commit
5bc5f8cd5f804d8b12771aed156dc928510b1e47.
"""

from .config import (
    MatchConfig, MicroscopeConfig, SearchConfig, RuntimeConfig, WeightingConfig,
    TemplateInputConfig, BatchConfig, DebugConfig, TimingConfig,
)
from .geometry import cisTEM_rotation_matrix, EulerSearch
from .ctf import CTF
from .matcher import TemplateMatcher, MatchResult
from .runner import run_config

__all__ = [
    "MatchConfig",
    "MicroscopeConfig",
    "SearchConfig",
    "RuntimeConfig",
    "WeightingConfig",
    "TemplateInputConfig",
    "BatchConfig",
    "DebugConfig",
    "TimingConfig",
    "cisTEM_rotation_matrix",
    "EulerSearch",
    "CTF",
    "TemplateMatcher",
    "MatchResult",
    "run_config",
]

__version__ = "0.3.6"
CISTEM_SOURCE_COMMIT = "5bc5f8cd5f804d8b12771aed156dc928510b1e47"
