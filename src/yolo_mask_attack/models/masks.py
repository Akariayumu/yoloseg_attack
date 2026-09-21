from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def crop_mask(masks: Tensor, boxes_xyxy: Tensor) -> Tensor:
    """Differentiably zero mask pixels outside each xyxy box."""
    if masks.ndim != 3 or boxes_xyxy.shape != (masks.shape[0], 4):
        raise ValueError("Expected masks [N,H,W] and boxes [N,4]")
    _, height, width = masks.shape
    rows = torch.arange(height, device=masks.device, dtype=boxes_xyxy.dtype)[None, :, None]
    cols = torch.arange(width, device=masks.device, dtype=boxes_xyxy.dtype)[None, None, :]
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
    inside = (
        (cols >= x1[:, None, None])
        & (cols < x2[:, None, None])
        & (rows >= y1[:, None, None])
        & (rows < y2[:, None, None])
    )
    return masks * inside


def instance_masks(
    coefficients: Tensor,
    prototypes: Tensor,
    boxes_xyxy: Tensor,
    input_hw: Sequence[int],
    *,
    upsample: bool = True,
) -> Tensor:
    """Build differentiable instance masks from prototype coefficients."""
    if prototypes.ndim != 3:
        raise ValueError("Expected one image's prototypes with shape [K,H,W]")
    if coefficients.ndim != 2 or coefficients.shape[1] != prototypes.shape[0]:
        raise ValueError("Coefficient count must match prototype channels")
    proto_h, proto_w = prototypes.shape[-2:]
    input_h, input_w = int(input_hw[0]), int(input_hw[1])
    logits = torch.einsum("nk,khw->nhw", coefficients, prototypes)
    if upsample:
        logits = F.interpolate(
            logits[:, None], size=(input_h, input_w), mode="bilinear", align_corners=False
        )[:, 0]
        crop_boxes = boxes_xyxy
    else:
        crop_boxes = boxes_xyxy * boxes_xyxy.new_tensor(
            [proto_w / input_w, proto_h / input_h] * 2
        )
    masks = crop_mask(logits.sigmoid(), crop_boxes)
    return masks


def soft_iou(a: Tensor, b: Tensor, eps: float = 1e-6) -> Tensor:
    """IoU over the last two dimensions, retaining leading dimensions."""
    if a.shape != b.shape or a.ndim < 2:
        raise ValueError("IoU inputs must have equal shape and at least two dimensions")
    dims = (-2, -1)
    intersection = (a * b).sum(dim=dims)
    union = (a + b - a * b).sum(dim=dims)
    return intersection / union.clamp_min(eps)


def hard_iou(a: Tensor, b: Tensor, threshold: float = 0.5, eps: float = 1e-6) -> Tensor:
    return soft_iou((a > threshold).to(a.dtype), (b > threshold).to(b.dtype), eps)
