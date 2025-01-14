import torch
from torch import nn

from feed_forward import SwiGLUFeedForward
from rms_norm import RMSNorm
from attention import MultiHeadAttention
from rope import RoPE


class LLaMaBlock(nn.Module):
    def __init__(self, seq_len:int, hidden_size: int, n_head: int, rope: RoPE, device: torch.device):
        super().__init__()
        self.rms_attn = RMSNorm(hidden_size)
        self.attention = MultiHeadAttention(seq_len, hidden_size, n_head, rope, device)
        self.rms_ffn = RMSNorm(hidden_size)
        self.swiglu = SwiGLUFeedForward(hidden_size, 8 * hidden_size // 3)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        x = self.attention(self.rms_attn(input_)) # batch_size, seq_len, hidden_dim
        x = input_ + x 
        x = x + self.swiglu(self.rms_ffn(x)) 
        return x # batch_size, seq_len, hidden_dim


class LLaMa(nn.Module):
    def __init__(self, vocab_size:int, n_stacks:int, seq_len: int, hidden_size: int, n_head: int, device: torch.device='cpu'):
        super().__init__()
        rope = RoPE(seq_len, hidden_size // n_head, device)
        self.n_stacks = n_stacks
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rmsnorm = RMSNorm(hidden_size)
        self.blocks = nn.Sequential()
        for i in range(n_stacks):
            self.blocks.add_module(f"LLaMa Block {i}", LLaMaBlock(seq_len, hidden_size, n_head, rope, device))

        self.output_linear = nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(self, input_: torch.Tensor):
        x = self.embed(input_) # batch_size, seq_len, hidden_dim
        for block in self.blocks:
            x = block(x)

        x = self.rmsnorm(x) 
        x = self.output_linear(x) 
        return x # batch_size, seq_len, vocab_size

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

    

