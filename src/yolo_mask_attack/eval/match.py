from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor

from yolo_mask_attack.models.decode import pairwise_box_iou


class Outcome(str, Enum):
    SELECTIVE_DEGRADATION = "selective_degradation"
    PRESERVED_NOT_DEGRADED = "preserved_not_degraded"
    MISSED = "missed"
    CLASS_CHANGED = "class_changed"
    BOX_SHIFTED = "box_shifted"


def greedy_iou_match(
    reference_boxes: Tensor, prediction_boxes: Tensor, iou_threshold: float = 0.5
) -> tuple[Tensor, Tensor, Tensor]:
    """Greedily select globally highest IoUs with one-to-one assignment."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    if len(reference_boxes) == 0 or len(prediction_boxes) == 0:
        empty_index = torch.empty(0, dtype=torch.long, device=reference_boxes.device)
        empty_iou = torch.empty(0, dtype=reference_boxes.dtype, device=reference_boxes.device)
        return empty_index, empty_index.clone(), empty_iou

    overlaps = pairwise_box_iou(reference_boxes, prediction_boxes)
    work = overlaps.clone()
    reference_indices, prediction_indices, values = [], [], []
    while work.numel():
        flat_index = work.argmax()
        value = work.flatten()[flat_index]
        if value < iou_threshold:
            break
        reference_index = flat_index // work.shape[1]
        prediction_index = flat_index % work.shape[1]
        reference_indices.append(reference_index)
        prediction_indices.append(prediction_index)
        values.append(value.clone())
        work[reference_index, :] = -1
        work[:, prediction_index] = -1

    if not values:
        empty_index = torch.empty(0, dtype=torch.long, device=reference_boxes.device)
        empty_iou = torch.empty(0, dtype=reference_boxes.dtype, device=reference_boxes.device)
        return empty_index, empty_index.clone(), empty_iou
    return torch.stack(reference_indices), torch.stack(prediction_indices), torch.stack(values)


def classify_outcome(
    *,
    matched: bool,
    class_preserved: bool,
    confidence_preserved: bool,
    box_iou: float,
    mask_iou: float,
    box_iou_min: float,
    mask_iou_max: float,
) -> Outcome:
    """Assign exactly one outcome using the protocol's precedence order."""
    if not matched:
        return Outcome.MISSED
    if not class_preserved:
        return Outcome.CLASS_CHANGED
    if not confidence_preserved or box_iou < box_iou_min:
        return Outcome.BOX_SHIFTED
    if mask_iou <= mask_iou_max:
        return Outcome.SELECTIVE_DEGRADATION
    return Outcome.PRESERVED_NOT_DEGRADED
