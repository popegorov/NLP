import torch
from torch import nn


class RoPE:
    def __init__(self, seq_len: int, embed_size: int, device: torch.device):
        self.base = 10000
        self.seq_len = seq_len
        self.embed_size = embed_size
        arange = torch.arange(0, self.embed_size, 2, device=device)
        theta = 1.0 / (self.base ** (arange / embed_size))
        idxs = torch.arange(seq_len, device=device)

        outer_product = torch.outer(theta, idxs)
        self.storage = torch.stack((torch.cos(outer_product), torch.sin(outer_product)), dim=-1)

    def apply(self, input_: torch.Tensor):
        batch_size, seq_len, nheads, hidden_size = input_.shape
        # cur_storage = self.storage[:seq_len]
        x = input_.view(batch_size, seq_len, nheads, hidden_size // 2, 2)
        cur_storage = self.storage.view(1, seq_len, 1, hidden_size // 2, 2)
        
        cos_theta = cur_storage[..., 0]
        sin_theta = cur_storage[..., 1]
        output = torch.stack([
            x[..., 0] * cos_theta - x[..., 1] * sin_theta,
            x[..., 0] * sin_theta + x[..., 1] * cos_theta,
        ], dim=-1)

        return output.view(batch_size, seq_len, nheads, hidden_size)
    
#     import torch
# import torch.nn as nn 


# device = torch.device("cpu")

# class RoPE(nn.Module):
#     def __init__(self, emb_dim: int, max_seq_len: int, theta: float = 10000.0):
#         super().__init__()
#         self.emb_dim = emb_dim
#         self.max_seq_len = max_seq_len
#         self.theta = theta
#         freqs = 1.0 / (theta ** (torch.arange(0, emb_dim, 2).float() / emb_dim))
#         t = torch.arange(max_seq_len, device=device)
#         freqs = torch.outer(t, freqs)
#         self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

#     def forward(self, queries: torch.Tensor, keys: torch.Tensor):
#         batch_size, seq_len, n_head, head_dim = queries.shape
#         queries_complex = torch.view_as_complex(queries.reshape(batch_size, seq_len, n_head, -1, 2))
#         keys_complex = torch.view_as_complex(keys.reshape(batch_size, seq_len, n_head, -1, 2))
#         freqs_cis = self.freqs_cis.unsqueeze(0).unsqueeze(2)
#         queries_roated = queries_complex * freqs_cis
#         keys_roated = keys_complex * freqs_cis
#         result_queries = torch.view_as_real(queries_roated).reshape(batch_size, seq_len, n_head, head_dim)
#         result_keys = torch.view_as_real(keys_roated).reshape(batch_size, seq_len, n_head, head_dim)
#         return result_queries, result_keys