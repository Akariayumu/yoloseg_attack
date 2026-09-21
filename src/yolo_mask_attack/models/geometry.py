from __future__ import annotations

from torch import Tensor


def box_iou_aligned(boxes_a: Tensor, boxes_b: Tensor, eps: float = 1e-6) -> Tensor:
    """Differentiable aligned IoU for two [N,4] xyxy tensors."""
    if boxes_a.shape != boxes_b.shape or boxes_a.ndim != 2 or boxes_a.shape[1] != 4:
        raise ValueError("Expected equally shaped [N,4] box tensors")
    top_left = boxes_a[:, :2].maximum(boxes_b[:, :2])
    bottom_right = boxes_a[:, 2:].minimum(boxes_b[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0).prod(dim=1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0).prod(dim=1)
    return intersection / (area_a + area_b - intersection).clamp_min(eps)
