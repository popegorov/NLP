from torch import nn
from torch import Tensor
import math
import torch


class RMSNorm(nn.Module):
    def __init__(self, input_size: int, prob: float=1.0, eps: float=1e-8):
        super().__init__()
        assert prob > 0.0

        self.input_size = input_size
        self.gamma = nn.Parameter(torch.ones(input_size), requires_grad=True)
        self.prob = prob
        self.eps = eps

    def forward(self, input_: Tensor) -> Tensor:
        if self.prob >= 1.0:
            input_norm = input_.norm(2, dim=-1, keepdim=True)
            den = input_norm / math.sqrt(self.input_size) + self.eps
            return (input_ / den) * self.gamma

        est_size = int(self.prob * self.input_size)
        to_calc = input_[..., :est_size]
        est_norm = to_calc.norm(2, dim=-1, keepdim=True)
        den = est_norm / torch.sqrt(est_size) + self.eps
        return (input_ / den) * self.gamma
        


    
