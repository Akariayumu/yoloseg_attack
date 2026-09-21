from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from .base import AttackConfig, gradient_step, initialize_delta


def pgd(image: Tensor, objective: Callable[[Tensor], Tensor], config: AttackConfig) -> Tensor:
    """Generic projected-gradient minimization loop."""
    delta = initialize_delta(image, config.epsilon, config.random_start)
    for _ in range(config.steps):
        delta = gradient_step(delta, image, objective(image + delta), config)
    return (image + delta).detach()
