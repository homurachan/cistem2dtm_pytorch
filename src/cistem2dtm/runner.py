from __future__ import annotations

from .batch import BatchRunResult, run_batch
from .config import MatchConfig
from .matcher import MatchResult, TemplateMatcher
from .pdb_template import PDBTemplateResult, resolve_template_for_match


def run_config(config: MatchConfig) -> MatchResult | BatchRunResult:
    """Dispatch a validated config without exposing preprocessing to the core."""

    config.validate()
    if config.micrographs_star:
        return run_batch(config)
    resolved, _ = resolve_template_for_match(config)
    return TemplateMatcher(resolved).run()
