"""Differentiable decoding for the pinned YOLOv8 segmentation head."""

from __future__ import annotations

import torch
from torch import Tensor

from .types import DecodedBatch, RawOutput


def _make_anchors(features: tuple[Tensor, ...], strides: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    anchors, stride_values, levels, grid_yx = [], [], [], []
    for level, (feature, stride) in enumerate(zip(features, strides)):
        height, width = feature.shape[-2:]
        y, x = torch.meshgrid(
            torch.arange(height, device=feature.device, dtype=feature.dtype),
            torch.arange(width, device=feature.device, dtype=feature.dtype),
            indexing="ij",
        )
        anchors.append(torch.stack((x + 0.5, y + 0.5), dim=-1).reshape(-1, 2))
        grid_yx.append(torch.stack((y, x), dim=-1).reshape(-1, 2).long())
        stride_values.append(feature.new_full((height * width, 1), stride))
        levels.append(torch.full((height * width,), level, device=feature.device, dtype=torch.long))
    return (
        torch.cat(anchors),
        torch.cat(stride_values),
        torch.cat(levels),
        torch.cat(grid_yx),
    )


def decode_head(raw: RawOutput, strides: Tensor, reg_max: int) -> DecodedBatch:
    """Decode DFL boxes, sigmoid scores and coefficients without NMS."""
    batch, channels, candidates = raw.box_distributions.shape
    if channels != 4 * reg_max:
        raise ValueError(f"Expected {4 * reg_max} box channels, got {channels}")
    anchors, stride_values, levels, grid_yx = _make_anchors(raw.features, strides)
    if anchors.shape[0] != candidates:
        raise ValueError("Feature-map anchor count does not match raw predictions")

    distributions = raw.box_distributions.view(batch, 4, reg_max, candidates)
    probabilities = distributions.transpose(1, 2).softmax(dim=1)
    projection = torch.arange(reg_max, device=probabilities.device, dtype=probabilities.dtype)
    distances = (probabilities * projection[None, :, None, None]).sum(dim=1)

    anchor_xy = anchors.t().unsqueeze(0)
    left_top, right_bottom = distances.chunk(2, dim=1)
    xy1 = anchor_xy - left_top
    xy2 = anchor_xy + right_bottom
    boxes_xyxy = torch.cat((xy1, xy2), dim=1) * stride_values.t().unsqueeze(0)
    centers = (xy1 + xy2) / 2
    sizes = xy2 - xy1
    boxes_xywh = torch.cat((centers, sizes), dim=1) * stride_values.t().unsqueeze(0)

    return DecodedBatch(
        boxes_xywh=boxes_xywh.transpose(1, 2),
        boxes_xyxy=boxes_xyxy.transpose(1, 2),
        class_logits=raw.class_logits.transpose(1, 2),
        scores=raw.class_logits.sigmoid().transpose(1, 2),
        coefficients=raw.mask_coefficients.transpose(1, 2),
        levels=levels,
        grid_yx=grid_yx,
    )


def pairwise_box_iou(boxes_a: Tensor, boxes_b: Tensor, eps: float = 1e-6) -> Tensor:
    """Pairwise IoU for [M,4] and [N,4] xyxy boxes."""
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    area_a = (boxes_a[:, 2:] - boxes_a[:, :2]).clamp_min(0).prod(dim=-1)
    area_b = (boxes_b[:, 2:] - boxes_b[:, :2]).clamp_min(0).prod(dim=-1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection).clamp_min(eps)


def match_references_to_candidates(
    decoded: DecodedBatch,
    reference_boxes: Tensor,
    reference_classes: Tensor,
    *,
    batch_index: int = 0,
    iou_threshold: float = 0.6,
) -> Tensor:
    """Freeze each clean reference to the best-scoring overlapping raw candidate."""
    overlaps = pairwise_box_iou(reference_boxes, decoded.boxes_xyxy[batch_index])
    target_scores = decoded.scores[batch_index, :, reference_classes].transpose(0, 1)
    eligible = overlaps >= iou_threshold
    ranked = target_scores.masked_fill(~eligible, -torch.inf)
    indices = ranked.argmax(dim=1)
    return torch.where(eligible.any(dim=1), indices, torch.full_like(indices, -1))


def select_proxy_neighborhood(
    decoded: DecodedBatch,
    frozen_indices: Tensor,
    target_classes: Tensor,
    *,
    batch_index: int = 0,
    radius: int = 1,
) -> Tensor:
    """Track references by class score within the frozen anchor's level/grid neighborhood."""
    selected = []
    for frozen_index, target_class in zip(frozen_indices.tolist(), target_classes.tolist()):
        if frozen_index < 0:
            selected.append(-1)
            continue
        level = decoded.levels[frozen_index]
        center = decoded.grid_yx[frozen_index]
        delta = (decoded.grid_yx - center).abs()
        neighborhood = (decoded.levels == level) & (delta <= radius).all(dim=1)
        candidates = neighborhood.nonzero(as_tuple=False).squeeze(1)
        scores = decoded.scores[batch_index, candidates, target_class]
        selected.append(int(candidates[scores.argmax()].item()))
    return torch.tensor(selected, device=frozen_indices.device, dtype=torch.long)
