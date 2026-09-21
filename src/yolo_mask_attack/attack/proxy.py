from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from yolo_mask_attack.models.decode import select_proxy_neighborhood
from yolo_mask_attack.models.geometry import box_iou_aligned
from yolo_mask_attack.models.masks import instance_masks, soft_iou
from yolo_mask_attack.models.types import DecodedBatch, RawOutput


@dataclass(frozen=True)
class ProxyObservation:
    candidate_index: Tensor
    confidence: Tensor
    box: Tensor
    logits: Tensor
    mask: Tensor


def locate_frozen_candidate(decoded: DecodedBatch, level: int, grid_yx: tuple[int, int]) -> int:
    """Resolve the stable feature-level/grid key stored in a frozen reference."""
    target = torch.tensor(grid_yx, device=decoded.grid_yx.device)
    matches = (decoded.levels == level) & (decoded.grid_yx == target).all(dim=1)
    indices = matches.nonzero(as_tuple=False).squeeze(1)
    if len(indices) != 1:
        raise ValueError(f"Expected one proxy candidate for level={level}, grid_yx={grid_yx}")
    return int(indices.item())


def observe_proxy(
    raw: RawOutput,
    decoded: DecodedBatch,
    *,
    frozen_index: int,
    target_class: int,
    input_hw: tuple[int, int],
    radius: int = 1,
) -> ProxyObservation:
    """Select and materialize one differentiable proxy observation."""
    selected = select_proxy_neighborhood(
        decoded,
        torch.tensor([frozen_index], device=decoded.boxes_xyxy.device),
        torch.tensor([target_class], device=decoded.boxes_xyxy.device),
        radius=radius,
    )[0]
    if selected < 0:
        raise ValueError("Frozen proxy candidate could not be selected")
    index = selected.reshape(1)
    box = decoded.boxes_xyxy[0, index]
    coefficients = decoded.coefficients[0, index]
    mask = instance_masks(coefficients, raw.prototypes[0], box, input_hw, upsample=True)
    return ProxyObservation(
        candidate_index=selected,
        confidence=decoded.scores[0, selected, target_class].reshape(1),
        box=box,
        logits=decoded.class_logits[0, index],
        mask=mask,
    )


def proxy_metrics(clean: ProxyObservation, adversarial: ProxyObservation) -> dict[str, Tensor]:
    return {
        "confidence": adversarial.confidence,
        "box_iou": box_iou_aligned(clean.box, adversarial.box),
        "mask_iou": soft_iou(clean.mask, adversarial.mask),
    }
