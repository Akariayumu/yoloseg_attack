from __future__ import annotations

import torch
from torch import Tensor


def rectangular_patch_mask(
    box_xyxy: Tensor,
    image_hw: tuple[int, int],
    *,
    width_ratio: float = 0.4,
    height_ratio: float = 0.25,
    vertical_position: float = 0.4,
) -> Tensor:
    """Place a rectangular texture patch inside a target box."""
    if box_xyxy.shape != (1, 4):
        raise ValueError("Expected one box with shape [1,4]")
    if not 0 < width_ratio <= 1 or not 0 < height_ratio <= 1:
        raise ValueError("Patch ratios must be in (0,1]")
    if not 0 <= vertical_position <= 1:
        raise ValueError("vertical_position must be in [0,1]")
    image_h, image_w = image_hw
    x1, y1, x2, y2 = box_xyxy[0]
    box_w, box_h = (x2 - x1).clamp_min(1), (y2 - y1).clamp_min(1)
    patch_w = (box_w * width_ratio).round().clamp_min(1)
    patch_h = (box_h * height_ratio).round().clamp_min(1)
    center_x = (x1 + x2) / 2
    center_y = y1 + vertical_position * box_h
    left = int((center_x - patch_w / 2).round().clamp(0, image_w - 1).item())
    top = int((center_y - patch_h / 2).round().clamp(0, image_h - 1).item())
    right = min(image_w, left + int(patch_w.item()))
    bottom = min(image_h, top + int(patch_h.item()))
    mask = torch.zeros((1, 1, image_h, image_w), dtype=torch.bool, device=box_xyxy.device)
    mask[:, :, top:bottom, left:right] = True
    return mask
