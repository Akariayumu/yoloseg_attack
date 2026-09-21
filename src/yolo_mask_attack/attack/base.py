from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AttackConfig:
    epsilon: float = 8 / 255
    step_size: float = 1 / 255
    steps: int = 100
    random_start: bool = True


def initialize_delta(image: Tensor, epsilon: float, random_start: bool) -> Tensor:
    delta = torch.empty_like(image).uniform_(-epsilon, epsilon) if random_start else torch.zeros_like(image)
    return project_linf(delta, image, epsilon).detach().requires_grad_(True)


def project_linf(delta: Tensor, image: Tensor, epsilon: float) -> Tensor:
    """Project onto both the L-infinity ball and legal image range."""
    bounded = delta.clamp(-epsilon, epsilon)
    return (image + bounded).clamp(0.0, 1.0) - image


def gradient_step(delta: Tensor, image: Tensor, loss: Tensor, config: AttackConfig) -> Tensor:
    gradient = torch.autograd.grad(loss, delta, only_inputs=True)[0]
    with torch.no_grad():
        updated = delta - config.step_size * gradient.sign()
        updated = project_linf(updated, image, config.epsilon)
    return updated.detach().requires_grad_(True)
