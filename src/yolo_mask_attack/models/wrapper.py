from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import Tensor, nn

from .types import RawOutput


class YoloSegWrapper(nn.Module):
    """Thin owner for a pinned Ultralytics segmentation model.

    Raw-head capture is intentionally isolated here because upstream return formats
    drift. The official-equivalence regression test must be completed before experiments.
    """

    def __init__(self, weights: str | Path, device: str = "cuda:0") -> None:
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install the project dependencies before loading YOLO") from exc
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)
        self.yolo = YOLO(str(self.weights))
        self.net = self.yolo.model.to(device).eval()
        self.head = self.net.model[-1]
        self.device = device

    def predict(self, source: Any, **kwargs: Any) -> Any:
        """Full post-processed path; use only for clean references and evaluation."""
        return self.yolo.predict(source=source, device=self.device, **kwargs)

    def forward(self, images: Tensor) -> RawOutput:
        """Return a stable view of the differentiable Ultralytics Segment output."""
        output = self.net(images)
        try:
            (official_decoded, prototypes), raw = output
            features = tuple(raw["feats"])
            return RawOutput(
                box_distributions=raw["boxes"],
                class_logits=raw["scores"],
                mask_coefficients=raw["mask_coefficient"],
                features=features,
                prototypes=prototypes,
                official_decoded=official_decoded,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Unsupported Ultralytics Segment output; run the version-equivalence tests"
            ) from exc

    @property
    def strides(self) -> Tensor:
        return self.head.stride

    @property
    def num_classes(self) -> int:
        return int(self.head.nc)

    @property
    def reg_max(self) -> int:
        return int(self.head.reg_max)
