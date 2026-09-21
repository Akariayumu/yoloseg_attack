from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from yolo_mask_attack.models.geometry import box_iou_aligned
from yolo_mask_attack.models.masks import soft_iou


@dataclass(frozen=True)
class ConstraintThresholds:
    confidence_ratio: float = 0.8
    class_margin: float = 0.0
    box_iou_min: float = 0.9
    mask_iou_max: float = 0.5


def class_margin_violation(logits: Tensor, target_classes: Tensor, margin: float) -> Tensor:
    target = logits.gather(1, target_classes[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, target_classes[:, None], -torch.inf)
    return masked.max(dim=1).values - target + margin


def constraint_values(
    *,
    clean_confidence: Tensor,
    adversarial_confidence: Tensor,
    clean_boxes: Tensor,
    adversarial_boxes: Tensor,
    clean_masks: Tensor,
    adversarial_masks: Tensor,
    adversarial_logits: Tensor,
    target_classes: Tensor,
    thresholds: ConstraintThresholds,
) -> dict[str, Tensor]:
    """Return constraints in canonical g(x) <= 0 form."""
    return {
        "confidence": thresholds.confidence_ratio * clean_confidence - adversarial_confidence,
        "class": class_margin_violation(
            adversarial_logits, target_classes, thresholds.class_margin
        ),
        "box": thresholds.box_iou_min - box_iou_aligned(clean_boxes, adversarial_boxes),
        "mask": soft_iou(clean_masks, adversarial_masks) - thresholds.mask_iou_max,
    }


def all_feasible(values: dict[str, Tensor], tolerance: float = 1e-4) -> bool:
    return all(bool((value <= tolerance).all().item()) for value in values.values())
