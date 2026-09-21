from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean
from typing import Any


def aggregate_attack_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate completed per-reference attack records by method."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures = 0
    references = set()
    for record in records:
        if record.get("status") != "completed":
            failures += 1
            continue
        references.add(int(record["reference_index"]))
        for method, metrics in record["metrics"]["methods"].items():
            grouped[method].append(metrics)

    methods = {}
    for method, values in sorted(grouped.items()):
        official = [value["official"] for value in values]
        selective = sum(value["outcome"] == "selective_degradation" for value in official)
        preserved = sum(
            value["outcome"] in {"selective_degradation", "preserved_not_degraded"}
            for value in official
        )
        methods[method] = {
            "references": len(values),
            "sdr": selective / len(values),
            "dpr": preserved / len(values),
            "mean_confidence": fmean(value.get("adversarial_confidence", 0.0) for value in official),
            "mean_box_iou": fmean(value.get("box_iou", 0.0) for value in official),
            "mean_mask_iou": fmean(value.get("mask_iou", 0.0) for value in official),
            "mean_elapsed_seconds": fmean(value["elapsed_seconds"] for value in values),
            "outcomes": {
                outcome: sum(value["outcome"] == outcome for value in official)
                for outcome in sorted({value["outcome"] for value in official})
            },
        }
    return {
        "completed_references": len(references),
        "failed_records": failures,
        "methods": methods,
    }
