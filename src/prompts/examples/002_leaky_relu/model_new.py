import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """Thin wrapper around the evaluator-loaded torch.ops.custom_op operator."""

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.custom_op.run(x, float(self.negative_slope))
