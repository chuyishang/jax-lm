import math
from collections.abc import Callable, Iterable

import numpy.typing as npt
import torch
from einops import einsum
from jaxtyping import Float, Int
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        weights = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(weights, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.weights = nn.Parameter(weights)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        weights = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        std = embedding_dim ** -0.5
        nn.init.normal_(weights, mean=0.0, std=std)
        self.weights = nn.Parameter(weights)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weights[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_denom = (torch.mean(x**2, dim=-1, keepdim=True) + self.eps) ** 0.5
        return ((x / rms_denom) * self.weights).to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    @staticmethod
    def SiLu(x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(SwiGLU.SiLu(self.w1(x)) * self.w3(x))


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
        cos_selected = self.cos_angles[token_positions]
        sin_selected = self.sin_angles[token_positions]
        cos_2d = torch.repeat_interleave(cos_selected, 2, dim=-1).to(x.dtype)
        sin_2d = torch.repeat_interleave(sin_selected, 2, dim=-1).to(x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rotated = torch.stack([-x_odd, x_even], dim=-1).reshape(*x.shape[:-1], x.shape[-1])
        return x * cos_2d + x_rotated * sin_2d


def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max_vals = torch.max(in_features, dim=dim, keepdim=True).values
    shifted = in_features - max_vals
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def sdpa(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = scores / (K.size(-1) ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0.0, float("-inf"))
    return einsum(softmax(scores, dim=-1), V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope_theta: float = 1e4,
        max_seq_len: int = 1024,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, got d_model={d_model}, num_heads={num_heads}.")

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.Q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.K_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.V_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.O_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, device=device)),
            persistent=False,
        )
        self.rope = RotaryPositionalEmbedding(self.d_k, theta=rope_theta, max_seq_len=max_seq_len, device=device)

    def forward(self, x: Tensor, use_rope: bool = False) -> Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.Q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.K_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.V_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if use_rope:
            positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, positions)
            k = self.rope(k, positions)

        attn_out = sdpa(q, k, v, self.causal_mask[:seq_len, :seq_len])
        attn_out = attn_out.transpose(1, 2).contiguous().reshape_as(x)
        return self.O_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.MHA = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
        self.RMSNorm1 = RMSNorm(d_model)
        self.RMSNorm2 = RMSNorm(d_model)
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
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, theta) 
                for _ in range(num_layers)
            ]
        )
        self.ln1 = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.linear(self.ln1(x))


def cross_entropy_loss(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> torch.Tensor:
    max_logits = torch.max(inputs, dim=-1, keepdim=True).values
    shifted_logits = inputs - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1, keepdim=True)) + max_logits
    log_probs = inputs - log_sum_exp
    target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return -target_log_probs.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        lr = float(lr)
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr})

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                p.data -= (lr / math.sqrt(t + 1)) * p.grad.data
                state["t"] = t + 1
        return loss


AdamW = torch.optim.AdamW


def get_lr_schedule(t, a_max, a_min, warmup_iters, cosine_annealing_iters):
    t = int(t)
    a_max = float(a_max)
    a_min = float(a_min)
    warmup_iters = int(warmup_iters)
    cosine_annealing_iters = int(cosine_annealing_iters)
    if t < warmup_iters:
        return t / warmup_iters * a_max
    if t <= cosine_annealing_iters:
        angle = (t - warmup_iters) / (cosine_annealing_iters - warmup_iters) * math.pi
        return a_min + (1 + math.cos(angle)) / 2 * (a_max - a_min)
    return a_min


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_annealing_iters: int,
    warmup_start_factor: float = 0.1,
) -> torch.optim.lr_scheduler.LRScheduler:
    assert warmup_iters < cosine_annealing_iters, "warmup iters must be less than total iters"

    cosine_decay_iters = cosine_annealing_iters - warmup_iters
    if warmup_iters > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_iters,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_decay_iters,
            eta_min=min_learning_rate,
        )
        scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_iters],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_annealing_iters,
            eta_min=min_learning_rate,
        )

    return scheduler

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    start_indices = torch.randint(0, len(dataset) - context_length, (batch_size,))
    batch = []
    for start in start_indices:
        batch_item = torch.tensor(dataset[start : start + context_length + 1], dtype=torch.long)
        batch.append(batch_item)
    batch_tensor = torch.stack(batch, dim=0).to(device)
    return batch_tensor[:, :-1], batch_tensor[:, 1:]


def nan_in_gradients(parameters: Iterable[torch.nn.Parameter]) -> bool:
    for parameter in parameters:
        if parameter.grad is not None and torch.isnan(parameter.grad).any():
            return True
    return False


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
):
    obj = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    if scheduler is not None:
        obj["scheduler"] = scheduler.state_dict()
    torch.save(obj, out)


def load_checkpoint(
    src,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
):
    obj = torch.load(src)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    if scheduler is not None and "scheduler" in obj:
        scheduler.load_state_dict(obj["scheduler"])
    return obj["iteration"]
