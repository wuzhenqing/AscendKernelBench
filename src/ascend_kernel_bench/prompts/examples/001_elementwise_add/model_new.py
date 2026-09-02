import torch
import torch.nn as nn

import custom_op


class ModelNew(nn.Module):
    """Thin wrapper around the compiled Ascend C operator."""

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return custom_op.run(A, B)
