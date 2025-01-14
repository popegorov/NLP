import torch
import math
from torch import nn
from rope import RoPE

class MultiHeadAttention(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, n_head: int, rope: RoPE, device: torch.device):
        super().__init__()

        self.n_head = n_head

        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.shuffler = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rope = RoPE(seq_len, hidden_dim // n_head, device)

    def forward(self, input_: torch.Tensor, mask: bool=True):
        batch_size, seq_len, hidden_dim = input_.shape
        head_size = hidden_dim // self.n_head

        Q = self.w_q(input_).view(batch_size, seq_len, self.n_head, head_size)
        K = self.w_k(input_).view(batch_size, seq_len, self.n_head, head_size)
        V = self.w_v(input_).view(batch_size, seq_len, self.n_head, head_size)

        Q = self.rope.apply(Q).transpose(1, 2)
        K = self.rope.apply(K).transpose(1, 2)
        V = V.transpose(1, 2)

        coefs = torch.softmax(torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(head_size), dim=-1) # batch_size, nhead, seq_len, seq_len
        if mask:
            coefs = torch.tril(coefs)

        output = torch.matmul(coefs, V) # batch_size, nhead, seq_len, head_size
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        output = self.shuffler(output)

        return output # batch_size, seq_len, hidden_dim








