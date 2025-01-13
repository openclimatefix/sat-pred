from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F


class LossFunction(ABC):
    """Loss function"""

    @property
    @classmethod
    @abstractmethod
    def name(self) -> str:
        """Return name of the loss function"""

    @abstractmethod
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return loss"""


class MultiscaleMAE(LossFunction):
    """Multiscale Mean Absolute Error"""

    def __init__(self, scales: list[tuple[int]] | None = None):
        """Multiscale Mean Absolute Error"""
        if scales is None:
            scales = [(1, 1, 1), (2, 4, 4)]
        self.scales = scales

    @property
    def name(self) -> str:
        return "multiscale_mae"

    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return loss"""

        target = target.copy()
        target[target == -1] = float("nan")

        loss = 0

        for scale in self.scales:
            y_hat_coarse = F.avg_pool3d(input, kernel_size=list(scale))
            y_coarse = F.avg_pool3d(target, kernel_size=list(scale))
            loss += torch.nanmean(F.l1_loss(y_hat_coarse, y_coarse, reduction="none"))

        return loss / len(self.scales)
