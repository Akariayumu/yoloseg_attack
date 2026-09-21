from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .match import Outcome


def outcome_rates(outcomes: Iterable[Outcome]) -> dict[str, float | int]:
    values = list(outcomes)
    if not values:
        raise ValueError("At least one reference outcome is required")
    counts = Counter(values)
    total = len(values)
    detection_preserved = (
        counts[Outcome.SELECTIVE_DEGRADATION] + counts[Outcome.PRESERVED_NOT_DEGRADED]
    )
    result: dict[str, float | int] = {
        "references": total,
        "sdr": counts[Outcome.SELECTIVE_DEGRADATION] / total,
        "dpr": detection_preserved / total,
    }
    result.update({outcome.value: counts[outcome] / total for outcome in Outcome})
    return result
