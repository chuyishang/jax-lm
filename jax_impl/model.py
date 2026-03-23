import math
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from flax import nnx
from flax.nnx import State
from jax import Array
from jax.sharding import Mesh
from jaxtyping import Float, Int

AxisName: TypeAlias = str | None
Sharding: TypeAlias = tuple[AxisName, ...]


@dataclass(frozen=True, slots=True)
class ShardingConfig:
    dense_in: Sharding
    dense_out: Sharding
    embedding: Sharding
    lm_head: Sharding
    norm: Sharding


def get_sharding_config_for_mode(mode: str) -> ShardingConfig:
    """Return per-parameter partition specs for the selected parallelism mode."""
    if mode == "dp":
        return ShardingConfig(
            dense_in=(None, None),
            dense_out=(None, None),
            embedding=(None, None),
            lm_head=(None, None),
            norm=(None,),
        )
    if mode == "fsdp":
        return ShardingConfig(
            dense_in=("data", None),
            dense_out=(None, "data"),
            embedding=(None, None),
            lm_head=(None, None),
            norm=(None,),
        )
    if mode == "tp":
        return ShardingConfig(
            dense_in=(None, "tensor"),
            dense_out=("tensor", None),
            embedding=(None, "tensor"),
            lm_head=("tensor", None),
            norm=(None,),
        )
    if mode == "fsdp_tp":
        return ShardingConfig(
            dense_in=("data", "tensor"),
            dense_out=("tensor", "data"),
            embedding=(None, "tensor"),
            lm_head=("tensor", None),
            norm=(None,),
        )
    raise ValueError(f"Unsupported sharding mode: {mode}")


def get_batch_sharding_for_mode(mode: str) -> Sharding:
    """Return the batch partition spec for inputs/targets for a sharding mode."""
    if mode == "tp":
        return (None, None)
    if mode in ("dp", "fsdp", "fsdp_tp"):
        return ("data", None)
    raise ValueError(f"Unsupported sharding mode: {mode}")


def _validate_axis_divisibility(shape: tuple[int, ...], sharding: Sharding, mesh: Mesh, name: str) -> None:
    if len(shape) != len(sharding):
        raise ValueError(f"{name} has shape rank {len(shape)} but sharding rank {len(sharding)}.")

    for dim, axis_name in zip(shape, sharding, strict=True):
        if axis_name is None:
            continue
        axis_size = mesh.shape[axis_name]
        if dim % axis_size != 0:
            raise ValueError(f"{name} dimension {dim} is not divisible by mesh axis '{axis_name}' size {axis_size}.")


def validate_mesh_for_mode(mesh: Mesh, mode: str) -> None:
    """Validate that mesh axes and sizes are compatible with the selected mode."""
    if mode not in ("dp", "fsdp", "tp", "fsdp_tp"):
        raise ValueError(f"Unsupported sharding mode: {mode}. Expected one of: dp, fsdp, tp, fsdp_tp.")

    try:
        data_size = mesh.shape["data"]
        tensor_size = mesh.shape["tensor"]
    except KeyError as exc:
        raise ValueError("Mesh must define both 'data' and 'tensor' axes.") from exc

    mode_is_valid = {
        "dp": data_size > 1 and tensor_size == 1,
        "fsdp": data_size > 1 and tensor_size == 1,
        "tp": data_size == 1 and tensor_size > 1,
        "fsdp_tp": data_size > 1 and tensor_size > 1,
    }[mode]
    if mode_is_valid:
        return

    requirements = {
        "dp": "data > 1 and tensor == 1",
        "fsdp": "data > 1 and tensor == 1",
        "tp": "data == 1 and tensor > 1",
        "fsdp_tp": "data > 1 and tensor > 1",
    }[mode]
    raise ValueError(
        f"Invalid mesh for mode '{mode}': expected {requirements}, got data={data_size}, tensor={tensor_size}."
    )


def validate_model_partitioning(model_config: dict, mesh: Mesh, sharding_config: ShardingConfig) -> None:
    """Validate shape divisibility for representative tensors under the chosen partition specs."""
    d_model = int(model_config["d_model"])
    d_ff = int(model_config["d_ff"])
    vocab_size = int(model_config["vocab_size"])
    num_heads = int(model_config["num_heads"])

    # In TP-like modes, enforce whole-head partitioning across the tensor axis.
    if sharding_config.dense_in[1] == "tensor":
        tensor_size = mesh.shape["tensor"]
        if num_heads % tensor_size != 0:
            raise ValueError(
                f"num_heads={num_heads} must be divisible by mesh 'tensor' axis size {tensor_size} "
                "for TP/FSDP+TP head-aligned sharding."
            )

    _validate_axis_divisibility((d_model, d_model), sharding_config.dense_in, mesh, "dense_in(d_model,d_model)")
    _validate_axis_divisibility((d_model, d_model), sharding_config.dense_out, mesh, "dense_out(d_model,d_model)")
    _validate_axis_divisibility((d_model, d_ff), sharding_config.dense_in, mesh, "dense_in(d_model,d_ff)")
    _validate_axis_divisibility((d_ff, d_model), sharding_config.dense_out, mesh, "dense_out(d_ff,d_model)")
    _validate_axis_divisibility((vocab_size, d_model), sharding_config.embedding, mesh, "embedding(vocab_size,d_model)")
    _validate_axis_divisibility((d_model, vocab_size), sharding_config.lm_head, mesh, "lm_head(d_model,vocab_size)")
    _validate_axis_divisibility((d_model,), sharding_config.norm, mesh, "norm(d_model)")


def with_partitioning(init_fn: Callable, sharding: Sharding | None = None):
    """Wrapper function that applies partitioning to an initializer function. This function actually does a lot of heavy lifting."""
    if sharding is None:
        return init_fn
    return nnx.with_partitioning(init_fn, sharding)


class Linear(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        in_features: int,
        out_features: int,
        sharding: Sharding | None = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initializes a linear layer. Uses truncated normal initialization."""
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5
        # Initialize weights using the initializer
        init_fn = with_partitioning(
            nnx.initializers.truncated_normal(stddev=std, lower=-3.0 * std, upper=3.0 * std), sharding
        )
        weights_data = init_fn(rngs.params(), (in_features, out_features), dtype)
        self.weights = nnx.Param(weights_data)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Performs forward pass for the linear layer."""
        return jnp.einsum("...i,io->...o", x, self.weights.get_value())


class Embedding(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_embeddings: int,
        embedding_dim: int,
        sharding: Sharding | None = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        std = embedding_dim ** -0.5
        init_fn = with_partitioning(nnx.initializers.normal(stddev=std), sharding)
        weights_data = init_fn(rngs.params(), (num_embeddings, embedding_dim), dtype)
        self.weights = nnx.Param(weights_data)

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(self.weights.get_value(), token_ids, axis=0)


class RMSNorm(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        eps: float = 1e-5,
        sharding: Sharding | None = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        init_fn = with_partitioning(nnx.initializers.ones_init(), sharding)
        weights_data = init_fn(rngs.params(), (d_model,), dtype)
        self.weights = nnx.Param(weights_data)
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        in_dtype = x.dtype
        # d_model = x.shape[-1]
        rms_denom = (jnp.mean(x**2, axis=-1, keepdims=True) + self.eps) ** 0.5
        result = (x / rms_denom) * self.weights.get_value()
        return result.astype(in_dtype)


class SwiGLU(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        d_ff: int,
        sharding_config: ShardingConfig | None = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        s_dense_in = sharding_config.dense_in if sharding_config else None
        s_dense_out = sharding_config.dense_out if sharding_config else None

        self.w1 = Linear(rngs=rngs, in_features=d_model, out_features=d_ff, sharding=s_dense_in, dtype=dtype)
        self.w2 = Linear(rngs=rngs, in_features=d_ff, out_features=d_model, sharding=s_dense_out, dtype=dtype)
        self.w3 = Linear(rngs=rngs, in_features=d_model, out_features=d_ff, sharding=s_dense_in, dtype=dtype)

    @staticmethod
    def SiLu(x: Array) -> Array:
        return x * jax.nn.sigmoid(x)

    def __call__(self, x: Array) -> Array:
        """Peroforms the forward pass for the SwiGLU layer.
        Notes:
            - With sharding this becomes (tensor, data) x (data, tensor) x (tensor, data) -> (tensor, data)
        """
        product = SwiGLU.SiLu(self.w1(x)) * self.w3(x)
        return self.w2(product)


class RotaryPositionalEmbedding(nnx.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int):
        """Initialize the RoPE matrix.

        Args:
            theta (float): rotation constant value
            d_k (int): embedding dimension
            max_seq_len (int): max sequence length of the model

        Returns:
            None
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be divisible by 2"
        inv_freq = 1 / theta ** (jnp.arange(0, d_k, 2) / d_k)
        pos = jnp.arange(max_seq_len)
        angles = jnp.outer(pos, inv_freq)  # [max_seq_len, d_k / 2]
        self.cos_angles = nnx.Cache(jnp.cos(angles))
        self.sin_angles = nnx.Cache(jnp.sin(angles))

    def __call__(self, x: Array, token_positions: Array) -> Array:
        """Apply rotary positional embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, d_model]
            token_positions (torch.Tensor): Tensor of token positions of shape [batch_size, sequence_length]

        Returns:
            torch.Tensor: Input tensor with rotary positional embeddings applied, shape [batch_size, sequence_length, d_model]
        """
        cos_sel = self.cos_angles[token_positions]  # [B, S, d_k / 2] # (seq, pair)
        sin_sel = self.sin_angles[token_positions]
        cos_2d = jnp.repeat(cos_sel, 2, axis=-1).astype(x.dtype)  # [B, S, d_k]
        sin_2d = jnp.repeat(sin_sel, 2, axis=-1).astype(x.dtype)  # [B, S, d_k]

        def rotate_half(x):
            # x ~ [B, S, d_k]
            x_even = x[..., 0::2]  # [B, S, d_k / 2]
            x_odd = x[..., 1::2]  # [B, S, d_k / 2]
            x_rotated = jnp.stack([-x_odd, x_even], axis=-1).reshape(*x.shape[:-1], x.shape[-1])
            return x_rotated

        return x * cos_2d + rotate_half(x) * sin_2d  # [B, S, d_k] * [B, S, d_k]


def softmax(in_features: Float[Array, " ..."], dim: int) -> Float[Array, " ..."]:
    """Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max_vals = jnp.max(in_features, axis=dim, keepdims=True)
    in_features = in_features - max_vals
    exp = jnp.exp(in_features)
    out = exp / jnp.sum(exp, axis=dim, keepdims=True)
    return out


def sdpa(
    Q: Float[Array, " ... queries d_k"],
    K: Float[Array, " ... keys d_k"],
    V: Float[Array, " ... values d_v"],
    mask: Float[Array, " ... queries keys"] | None = None,
) -> Float[Array, " ... queries d_v"]:
    S: Float[Array, " ... queries keys"] = jnp.einsum("... q d, ... k d -> ... q k", Q, K)
    d_k = K.shape[-1]
    S = S / (d_k) ** 0.5
    if mask is not None:
        S = jnp.where(mask == 0.0, -jnp.inf, S)
    scores = softmax(S, dim=-1)
    output = jnp.einsum("... q k, ... k d -> ... q d", scores, V)
    # output = scores @ V
    return output


class MultiHeadSelfAttention(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_heads: int,
        rope_theta: float = 1e4,
        max_seq_len: int = 1024,
        sharding_config: ShardingConfig | None = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initializes multi-head self-attention block.

        Args:
            d_model (int): input dimension
            num_heads (int): num_heads
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_seq_len = max_seq_len

        s_dense_in = sharding_config.dense_in if sharding_config else None
        s_dense_out = sharding_config.dense_out if sharding_config else None

        self.Q_proj = Linear(
            rngs=rngs, in_features=d_model, out_features=self.d_k * num_heads, sharding=s_dense_in, dtype=dtype
        )
        self.K_proj = Linear(
            rngs=rngs, in_features=d_model, out_features=self.d_k * num_heads, sharding=s_dense_in, dtype=dtype
        )
        self.V_proj = Linear(
            rngs=rngs, in_features=d_model, out_features=self.d_k * num_heads, sharding=s_dense_in, dtype=dtype
        )
        self.O_proj = Linear(rngs=rngs, in_features=d_model, out_features=d_model, sharding=s_dense_out, dtype=dtype)

        # Initialize causal mask
        self.causal_mask = jnp.tril(jnp.ones((max_seq_len, max_seq_len)))
        # Intialize RoPE Module
        self.rope = RotaryPositionalEmbedding(d_k=self.d_k, theta=rope_theta, max_seq_len=max_seq_len)

    def __call__(self, x: Array, use_rope: bool = False):
        """Forward method for Multi-Head Self Attention.

        Args:
            x (torch.Tensor[B, S, d_model]): Input tensor
            device (torch.device): Device
            dtype (torch.dtype): dtype

        Returns:
            Tensor after MHSA is applied.
        """
        B, S, d_model = x.shape
        Q = self.Q_proj(x)
        K = self.K_proj(x)
        V = self.V_proj(x)
        # NOTE: we need to first reshape, then transpose, since we want H to be the outer dim
        Q = Q.reshape(B, S, self.num_heads, self.d_k).swapaxes(1, 2)
        K = K.reshape(B, S, self.num_heads, self.d_k).swapaxes(1, 2)
        V = V.reshape(B, S, self.num_heads, self.d_k).swapaxes(1, 2)
        if use_rope:
            seq_pos = jnp.arange(0, S)
            Q = self.rope(Q, seq_pos)
            K = self.rope(K, seq_pos)
        # causal_mask = torch.tril(torch.ones(S, S, device=x.device))
        causal_mask = self.causal_mask[:S, :S]  # slice existing causal mask for efficiency
        sdpa_out = sdpa(Q, K, V, causal_mask)
        # NOTE: sdpa_out ~ [B, H, S, d_k]. need out to be [B, S, d_k]
        sdpa_out = sdpa_out.swapaxes(1, 2).reshape(x.shape)
        out = self.O_proj(sdpa_out)
        return out


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        sharding_config: ShardingConfig | None = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        s_norm = sharding_config.norm if sharding_config else None

        self.MHA = MultiHeadSelfAttention(
            rngs=rngs,
            d_model=d_model,
            num_heads=num_heads,
            rope_theta=theta,
            max_seq_len=max_seq_len,
            sharding_config=sharding_config,
            dtype=dtype,
        )
        self.RMSNorm1 = RMSNorm(rngs=rngs, d_model=d_model, sharding=s_norm, dtype=dtype)
        self.RMSNorm2 = RMSNorm(rngs=rngs, d_model=d_model, sharding=s_norm, dtype=dtype)
        self.SwiGLU = SwiGLU(rngs=rngs, d_model=d_model, d_ff=d_ff, sharding_config=sharding_config, dtype=dtype)

    def __call__(self, x: Array):
        x = x + self.MHA(self.RMSNorm1(x), use_rope=True)
        x = x + self.SwiGLU(self.RMSNorm2(x))
        return x


class TransformerLM(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        sharding_config: ShardingConfig | None = None,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()

        s_embedding = sharding_config.embedding if sharding_config else None
        s_lm_head = sharding_config.lm_head if sharding_config else None
        s_norm = sharding_config.norm if sharding_config else None

        self.token_embeddings = Embedding(rngs, vocab_size, d_model, sharding=s_embedding, dtype=dtype)
        self.layers = nnx.List(
            [
                TransformerBlock(rngs, d_model, num_heads, d_ff, context_length, theta, sharding_config, dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln1 = RMSNorm(rngs=rngs, d_model=d_model, sharding=s_norm, dtype=dtype)
        self.linear = Linear(rngs, d_model, vocab_size, sharding=s_lm_head, dtype=dtype)

    def __call__(self, x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.linear(self.ln1(x))
        return x


def cross_entropy_loss(inputs: Float[Array, " batch_size vocab_size"], targets: Int[Array, " batch_size"]) -> Array:
    """Computes Cross-Entropy Loss"""
    # Stabilize logits to avoid overflow when exponentiating
    max_logits = jnp.max(inputs, axis=-1, keepdims=True)
    shifted_logits = inputs - max_logits

    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True)) + max_logits
    log_probs = inputs - log_sum_exp

    target_log_probs = jnp.take_along_axis(log_probs, jnp.expand_dims(targets, axis=1), axis=1).squeeze(axis=1)
    loss = -target_log_probs.mean()
    return loss


def get_lr_schedule(t: int, a_max: float, a_min: float, warmup_iters: int, cosine_annealing_iters: int) -> float:
    assert cosine_annealing_iters >= warmup_iters
    if t < warmup_iters:
        return t / warmup_iters * a_max
    elif t <= cosine_annealing_iters:
        return a_min + (1 + math.cos((t - warmup_iters) / (cosine_annealing_iters - warmup_iters) * math.pi)) / 2 * (
            a_max - a_min
        )
    else:
        return a_min


def make_lr_schedule(a_max: float, a_min: float, warmup_iters: int, cosine_annealing_iters: int) -> optax.Schedule:
    """LR Schedule using warmup_cosine_decay_schedule. Note that decay_iters is inclusive of warmup_iters."""
    if cosine_annealing_iters <= warmup_iters:
        raise ValueError("cosine_annealing_iters must be > warmup_iters")
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=a_max,
        warmup_steps=warmup_iters,
        decay_steps=cosine_annealing_iters,
        end_value=a_min,
    )


def build_optimizer_transform(
    optimizer_config: dict,
    gradient_clip: float,
    lr_schedule: optax.Schedule | None = None,
) -> optax.GradientTransformation:
    optimizer_type = optimizer_config["type"].lower()
    if optimizer_type != "adamw":
        raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")

    lr = lr_schedule if lr_schedule is not None else float(optimizer_config["lr"])
    transforms = []
    if gradient_clip > 0:
        transforms.append(optax.clip_by_global_norm(float(gradient_clip)))
    transforms.append(
        optax.adamw(
            learning_rate=lr,
            b1=float(optimizer_config["betas"][0]),
            b2=float(optimizer_config["betas"][1]),
            weight_decay=float(optimizer_config["weight_decay"]),
            eps=float(optimizer_config["eps"]),
        )
    )
    return optax.apply_if_finite(optax.chain(*transforms), max_consecutive_errors=int(np.iinfo(np.int32).max))


def gradient_clipping(gradient_state: State, max_l2_norm: float, eps=1e-6) -> State:
    gradients = jax.tree.leaves(gradient_state)
    total_norm_sq = jnp.asarray(0.0, dtype=jnp.float32)
    for grad in gradients:
        total_norm_sq = total_norm_sq + jnp.sum(jnp.square(grad.astype(jnp.float32)))
    total_norm = jnp.sqrt(total_norm_sq)
    scale = jnp.minimum(1.0, jnp.asarray(max_l2_norm, dtype=jnp.float32) / (total_norm + eps))
    return jax.tree.map(lambda g: g * scale.astype(g.dtype), gradient_state)


def get_batch(rngs: nnx.Rngs, dataset: npt.NDArray, batch_size: int, context_length: int) -> tuple[Array, Array]:
    max_start = len(dataset) - context_length
    if max_start <= 0:
        raise ValueError(f"Dataset too small for context_length={context_length} (len={len(dataset)}).")

    # JAX randint defaults to int32, so large datasets can overflow maxval. Fall back to NumPy for >2^31-1.
    if max_start <= np.iinfo(np.int32).max:
        start_indices = jax.random.randint(rngs.params(), shape=(batch_size,), minval=0, maxval=max_start)
        start_indices = np.asarray(start_indices)
    else:
        seed = int(
            jax.random.randint(rngs.params(), shape=(), minval=0, maxval=np.iinfo(np.int32).max, dtype=jnp.int32)
        )
        rng = np.random.default_rng(seed)
        start_indices = rng.integers(0, max_start, size=(batch_size,), dtype=np.int64)
    offsets = np.arange(context_length + 1, dtype=np.int64)
    batch_indices = start_indices[:, None] + offsets[None, :]
    batch = jnp.asarray(dataset[batch_indices], dtype=jnp.int32)

    train_batch = batch[:, :-1]
    target_batch = batch[:, 1:]
    assert train_batch.shape == target_batch.shape
    return (train_batch, target_batch)


def save_checkpoint(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    iteration: int,
    out,
    grad_state: State | None = None,
    loss: Array | None = None,
):
    state = nnx.state(model)
    optimizer_state = nnx.state(optimizer)
    obj = {
        "model": state,
        "optimizer": optimizer_state,
        "iteration": iteration,
    }
    if grad_state is not None:
        obj["grad_state"] = grad_state
    if loss is not None:
        obj["loss"] = loss

    with open(out, "wb") as f:
        pickle.dump(obj, f)


def load_checkpoint(
    src,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
):
    # TODO: add docstring
    with open(src, "rb") as f:
        obj = pickle.load(f)
    nnx.update(model, obj["model"])
    nnx.update(optimizer, obj["optimizer"])

    return obj["iteration"]


def nonfinite_in_pytree(pytree) -> Array:
    is_finite_tree = jax.tree.map(lambda x: jnp.all(jnp.isfinite(x)), pytree)
    all_finite = jax.tree.reduce(lambda a, b: a & b, is_finite_tree, initializer=jnp.array(True, dtype=jnp.bool_))
    return jnp.logical_not(all_finite)


@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, inputs: Array, targets: Array
) -> tuple[Array, State, bool]:
    B, S = targets.shape
    loss, grad_state = nnx.value_and_grad(
        lambda model: cross_entropy_loss(model(inputs).reshape(B * S, -1), targets.reshape(B * S))
    )(model)
    loss_is_finite = jnp.isfinite(loss)
    has_nonfinite = jnp.logical_or(jnp.logical_not(loss_is_finite), nonfinite_in_pytree(grad_state))
    grad_state_for_update = jax.tree.map(
        lambda g: jnp.where(loss_is_finite, g, jnp.full_like(g, jnp.nan)),
        grad_state,
    )
    optimizer.update(model, grad_state_for_update)
    return loss, grad_state, has_nonfinite


def create_model_and_optimizer(
    rngs: nnx.Rngs,
    model_config: dict,
    optimizer_config: dict,
    sharding_config: ShardingConfig | None,
    mesh: Mesh | None = None,
    gradient_clip: float = 0.0,
    lr_schedule: optax.Schedule | None = None,
):
    """
    JIT-compiled initialization of Model and Optimizer.

    This ensures parameters are allocated directly on the device (sharded)
    rather than materializing fully on the host CPU first.

    Args:
        rngs: Random number generators
        model_config: Model configuration dict (static)
        optimizer_config: Optimizer configuration dict (static)
        sharding_config: ShardingConfig for 3D parallelism (static)
        mesh: Device mesh for sharding
        gradient_clip: Global-norm clipping threshold applied inside the Optax chain

    Returns:
        Tuple of (model, optimizer)
    """

    # Initialize model
    @nnx.jit
    def init_model_and_optimizer(rngs: nnx.Rngs):
        model = TransformerLM(
            rngs=rngs,
            d_model=model_config["d_model"],
            num_heads=model_config["num_heads"],
            d_ff=model_config["d_ff"],
            theta=model_config["theta"],
            vocab_size=model_config["vocab_size"],
            context_length=model_config["context_length"],
            num_layers=model_config["num_layers"],
            sharding_config=sharding_config,
        )

        # Initialize Optimizer
        if optimizer_config["type"].lower() == "adamw":
            optimizer = nnx.Optimizer(
                model,
                build_optimizer_transform(
                    optimizer_config,
                    gradient_clip=gradient_clip,
                    lr_schedule=lr_schedule,
                ),
                wrt=nnx.Param,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")

        # Apply sharding constraints if mesh is provided
        if mesh is not None:
            # Shard model parameters according to their partition specs
            model_state = nnx.state(model)
            model_shardings = nnx.get_named_sharding(model_state, mesh)
            model_sharded_state = jax.lax.with_sharding_constraint(model_state, model_shardings)
            nnx.update(model, model_sharded_state)

            # Shard Optimizer State (inherits sharding from params)
            optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
            optimizer_shardings = nnx.get_named_sharding(optimizer_state, mesh)
            optimizer_sharded_state = jax.lax.with_sharding_constraint(optimizer_state, optimizer_shardings)
            nnx.update(optimizer, optimizer_sharded_state)

        return model, optimizer

    if mesh is not None:
        with jax.set_mesh(mesh):
            return init_model_and_optimizer(rngs)
    else:
        return init_model_and_optimizer(rngs)
