"""Immutable named notebook cases and ownership-checked grouped exports.

The bundle retains its experiment and figure references when later notebook
cells reuse ordinary variable names. Package figure tags use solution identity;
arbitrary untagged figures remain the caller's responsibility.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from overscreening_breakdown.notebook_examples.discontinuous_galerkin import (
    _preflight_export, _validate_case_name, export_experiment,
)


@dataclass(frozen=True, init=False)
class CaseArtifacts:
    name: str
    experiment: object
    figures: object
    resolution: object

    def __init__(self, name, experiment, *, figures=None, resolution=None):
        _validate_case_name(name)
        if figures is not None and not isinstance(figures, Mapping):
            raise ValueError("figures must be a mapping of filenames to figures")
        for key, value in (("name", name), ("experiment", experiment),
                           ("figures", MappingProxyType({} if figures is None else dict(figures))),
                           ("resolution", resolution)):
            object.__setattr__(self, key, value)


def export_cases(cases, directory):
    """Export each named case into its own folder after registry-wide preflight.

    A comparison figure may contain several solutions when the case's solution
    is one of them. Its exported settings record comparison provenance. No
    executable callbacks or figure objects are serialized into the settings.
    """
    if not isinstance(cases, Mapping):
        raise ValueError("cases must be a mapping of names to CaseArtifacts")
    registry = dict(cases)
    for name, case in registry.items():
        _validate_case_name(name)
        if not isinstance(case, CaseArtifacts) or name != case.name:
            raise ValueError("each registry key must equal its CaseArtifacts name")
        _preflight_export(case.experiment, figures=case.figures, resolution=case.resolution, case_name=name)
    directory = Path(directory).expanduser()
    return MappingProxyType({name: MappingProxyType(export_experiment(
        case.experiment, directory/name, figures=case.figures, resolution=case.resolution, case_name=name,
    )) for name, case in registry.items()})
