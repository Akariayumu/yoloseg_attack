from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class RawOutput:
    box_distributions: Tensor
    class_logits: Tensor
    mask_coefficients: Tensor
    features: tuple[Tensor, ...]
    prototypes: Tensor
    official_decoded: Tensor | None = None


@dataclass(frozen=True)
class DecodedBatch:
    boxes_xywh: Tensor
    boxes_xyxy: Tensor
    class_logits: Tensor
    scores: Tensor
    coefficients: Tensor
    levels: Tensor
    grid_yx: Tensor

    def as_ultralytics_prediction(self) -> Tensor:
        """Return [B, 4 + nc + nm, N] for Ultralytics NMS."""
        return __import__("torch").cat(
            (
                self.boxes_xywh.transpose(1, 2),
                self.scores.transpose(1, 2),
                self.coefficients.transpose(1, 2),
            ),
            dim=1,
        )
