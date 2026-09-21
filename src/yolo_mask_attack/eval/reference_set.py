from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReferenceInstance:
    image_id: int | str
    instance_id: int
    class_id: int
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    mask_rle: dict[str, Any]
    proxy_level: int
    proxy_yx: tuple[int, int]
    ground_truth_box_iou: float
    ground_truth_mask_iou: float


@dataclass(frozen=True)
class ReferenceSet:
    version: str
    model_sha256: str
    dataset_sha256: str
    protocol: dict[str, Any]
    instances: tuple[ReferenceInstance, ...]


def freeze_reference_set(reference_set: ReferenceSet, path: str | Path) -> None:
    """Atomically create a reference set and refuse accidental protocol rewrites."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(reference_set), ensure_ascii=False, indent=2, sort_keys=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    except FileExistsError as exc:
        raise FileExistsError(f"Frozen reference set already exists: {destination}") from exc
