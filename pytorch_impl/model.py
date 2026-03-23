from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)


    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be divisible by 2, got {d_k}.")

        inv_freq = 1 / theta ** (torch.arange(0, d_k, 2) / d_k)
        positions = torch.arange(max_seq_len)
        angles = torch.outer(positions, inv_freq).to(device)
        self.register_buffer("cos_angles", torch.cos(angles), persistent=False)
        self.register_buffer("sin_angles", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos_angles = cast(Tensor, self.cos_angles)
        sin_angles = cast(Tensor, self.sin_angles)
        cos_selected = cos_angles[token_positions]
        sin_selected = sin_angles[token_positions]
        cos_2d = torch.repeat_interleave(cos_selected, 2, dim=-1).to(x.dtype)
        sin_2d = torch.repeat_interleave(sin_selected, 2, dim=-1).to(x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rotated = torch.stack([-x_odd, x_even], dim=-1).reshape(*x.shape[:-1], x.shape[-1])
        return x * cos_2d + x_rotated * sin_2d


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_theta: float = 1e4,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, got d_model={d_model}, num_heads={num_heads}.")

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Q_proj = nn.Linear(d_model, d_model, bias=False)
        self.K_proj = nn.Linear(d_model, d_model, bias=False)
        self.V_proj = nn.Linear(d_model, d_model, bias=False)
        self.O_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionalEmbedding(self.d_k, theta=rope_theta, max_seq_len=max_seq_len)

    def forward(self, x: Tensor, use_rope: bool = False) -> Tensor:
        batch_size, seq_len, _ = x.shape
        q_proj = self.Q_proj(x)
        k_proj = self.K_proj(x)
        v_proj = self.V_proj(x)

        if q_proj.shape[-1] % self.d_k != 0:
            raise ValueError(
                "Projected attention width must be divisible by per-head size; "
                f"got projected_width={q_proj.shape[-1]}, d_k={self.d_k}."
            )
        local_num_heads = q_proj.shape[-1] // self.d_k

        q = q_proj.reshape(batch_size, seq_len, local_num_heads, self.d_k).transpose(1, 2)
        k = k_proj.reshape(batch_size, seq_len, local_num_heads, self.d_k).transpose(1, 2)
        v = v_proj.reshape(batch_size, seq_len, local_num_heads, self.d_k).transpose(1, 2)

        if use_rope:
            positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        return self.O_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.MHA = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.RMSNorm1 = nn.RMSNorm(d_model)
        self.RMSNorm2 = nn.RMSNorm(d_model)
        self.SwiGLU = SwiGLU(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.MHA(self.RMSNorm1(x), use_rope=True)
        x = x + self.SwiGLU(self.RMSNorm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, context_length, theta) for _ in range(num_layers)]
        )
        self.ln1 = nn.RMSNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.linear(self.ln1(x))
