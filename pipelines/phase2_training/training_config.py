"""Load the per-pair secondary-qualification proportion scales.

``params.yaml`` holds ``training.secondary_proportion_scales``. Each value
multiplies the historical proportion ``P(secondary | primary)`` computed from
``inputs/Supply_sample.csv`` by
:func:`pipelines.phase2_training.qualification_mapper.build_prop_table`.

A value of ``1.0`` keeps the historical rate, and ``0.0`` removes that
secondary qualification for that primary. Inputs are validated here so a
misspelt work center never silently disables a training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from pipelines.phase2_training.qualification_mapper import DEFAULT_PRIORITY_LIST

ScaleKey = Tuple[str, str]
ScaleTable = Dict[ScaleKey, float]
DEFAULT_SCALE = 1.0


def _allowed_pairs(priority_list: List[str]) -> List[ScaleKey]:
    return [
        (primary, secondary)
        for primary in priority_list
        if primary != "Domestic"
        for secondary in priority_list
        if secondary != "Domestic" and secondary != primary
    ]


def load_secondary_proportion_scales(
    params_path: str | Path = "params.yaml",
    priority_list: Optional[List[str]] = None,
) -> ScaleTable:
    """Read ``training.secondary_proportion_scales`` with defaults applied.

    Missing sections and missing pairs default to ``1.0``. Unknown work centers,
    the diagonal, Domestic, and values outside ``[0, 1]`` raise ``ValueError``.
    """
    priority_list = priority_list or DEFAULT_PRIORITY_LIST
    params = yaml.safe_load(Path(params_path).read_text()) or {}
    section = (params.get("training") or {}).get("secondary_proportion_scales") or {}

    scales: ScaleTable = {pair: DEFAULT_SCALE for pair in _allowed_pairs(priority_list)}
    if not section:
        return scales
    if not isinstance(section, dict):
        raise ValueError(
            "training.secondary_proportion_scales must be a mapping of "
            "primary work center to secondary work center"
        )

    allowed = set(scales)
    for primary, entries in section.items():
        if primary == "Domestic":
            raise ValueError(
                "training.secondary_proportion_scales: Domestic is not configurable. "
                "Every dispatcher already holds it, and the historical sample has no "
                "Domestic-primary dispatcher with a secondary qualification."
            )
        if primary not in priority_list:
            raise ValueError(
                f"training.secondary_proportion_scales: unknown primary {primary!r}; "
                f"choose from {[p for p in priority_list if p != 'Domestic']}"
            )
        if entries is None:
            continue
        if not isinstance(entries, dict):
            raise ValueError(
                f"training.secondary_proportion_scales[{primary!r}] must be a mapping"
            )
        for secondary, value in entries.items():
            key = (primary, secondary)
            if key not in allowed:
                raise ValueError(
                    f"training.secondary_proportion_scales: unknown secondary "
                    f"{secondary!r} for primary {primary!r}; a secondary must be a "
                    "different non-Domestic work center"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"training.secondary_proportion_scales[{primary!r}][{secondary!r}] "
                    f"must be a number in [0, 1], got {value!r}"
                )
            scale = float(value)
            if not 0.0 <= scale <= 1.0:
                raise ValueError(
                    f"training.secondary_proportion_scales[{primary!r}][{secondary!r}] "
                    f"must be in [0, 1], got {scale}"
                )
            scales[key] = scale
    return scales
