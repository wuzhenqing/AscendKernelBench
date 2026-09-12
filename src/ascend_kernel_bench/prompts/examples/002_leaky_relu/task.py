import torch
import torch.nn as nn


class Model(nn.Module):
    """LeakyReLU: y = x if x >= 0 else negative_slope * x."""

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(
            x, negative_slope=self.negative_slope
        )


def get_inputs():
    return [torch.randn(16, 4096)]


def get_init_inputs():
    return [0.01]
