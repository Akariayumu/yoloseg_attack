from __future__ import annotations

from enum import Enum

from torch import Tensor


class Method(str, Enum):
    MASK_ONLY = "mask_only"
    JOINT = "joint"
    FIXED_WEIGHT = "fixed_weight"
    DYNAMIC_WEIGHT = "dynamic_weight"
    CONSTRAINED = "constrained"


def degradation_objective(mask_iou: Tensor) -> Tensor:
    """Minimization objective: lower mask IoU is better."""
    return mask_iou.mean()


def joint_attack_objective(
    mask_iou: Tensor,
    target_confidence: Tensor,
    box_iou: Tensor,
    *,
    confidence_weight: float = 1.0,
    box_weight: float = 1.0,
) -> Tensor:
    """Minimize mask quality together with detection confidence and box stability."""
    return (
        mask_iou.mean()
        + confidence_weight * target_confidence.mean()
        + box_weight * box_iou.mean()
    )


def fixed_weight_objective(
    mask_iou: Tensor, constraints: dict[str, Tensor], weights: dict[str, float]
) -> Tensor:
    loss = degradation_objective(mask_iou)
    for name, value in constraints.items():
        loss = loss + weights.get(name, 1.0) * value.clamp_min(0).mean()
    return loss


class DynamicWeights:
    """Increase penalties for currently violated preservation constraints."""

    def __init__(
        self,
        names: list[str],
        *,
        initial: float = 1.0,
        growth: float = 2.0,
        maximum: float = 100.0,
    ) -> None:
        self.values = {name: initial for name in names}
        self.growth = growth
        self.maximum = maximum

    def update(self, constraints: dict[str, Tensor]) -> None:
        for name, value in constraints.items():
            violation = float(value.detach().clamp_min(0).mean().item())
            if violation > 0:
                self.values[name] = min(
                    self.maximum, self.values.get(name, 1.0) * (1.0 + self.growth * violation)
                )
