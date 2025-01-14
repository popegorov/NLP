from torch import nn
from torch import Tensor
import torch


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_: Tensor) -> Tensor:
        return input_ * self.sigmoid(self.beta * input_)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.U = nn.Linear(input_size, hidden_size, bias=True)
        self.W = nn.Linear(input_size, hidden_size, bias=True)
        self.V = nn.Linear(hidden_size, input_size, bias=True)
        self.swish = Swish()

    def forward(self, input_: Tensor) -> Tensor:
        x1 = self.swish(self.U(input_)) # batch_size, seq_len, hidden_dim * c
        x2 = self.W(input_) # batch_size, seq_len, hidden_dim * c
        x = self.V(x1 * x2)
        return x # batch_size, seq_len, hidden_dim
