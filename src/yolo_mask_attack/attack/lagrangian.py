from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LagrangianConfig:
    rho_initial: float = 1.0
    rho_growth: float = 1.5
    rho_max: float = 1000.0
    tolerance: float = 1e-4


class AugmentedLagrangian:
    """Inequality-constrained AL state independent of the model forward pass."""

    def __init__(self, config: LagrangianConfig | None = None) -> None:
        self.config = config or LagrangianConfig()
        self.rho = self.config.rho_initial
        self.multipliers: dict[str, Tensor] = {}

    def loss(self, objective: Tensor, constraints: dict[str, Tensor]) -> Tensor:
        total = objective
        for name, violation in constraints.items():
            multiplier = self.multipliers.get(name)
            if multiplier is None or multiplier.shape != violation.shape:
                multiplier = torch.zeros_like(violation)
                self.multipliers[name] = multiplier
            shifted = (multiplier + self.rho * violation).clamp_min(0)
            penalty = (shifted.square() - multiplier.square()) / (2 * self.rho)
            total = total + penalty.mean()
        return total

    @torch.no_grad()
    def update(self, constraints: dict[str, Tensor]) -> None:
        for name, violation in constraints.items():
            current = self.multipliers.get(name, torch.zeros_like(violation))
            self.multipliers[name] = (current + self.rho * violation.detach()).clamp_min(0)
        self.rho = min(self.rho * self.config.rho_growth, self.config.rho_max)

    def feasible(self, constraints: dict[str, Tensor]) -> bool:
        return all(
            bool((value <= self.config.tolerance).all().item())
            for value in constraints.values()
        )
