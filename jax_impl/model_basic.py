import math
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optax
from flax import nnx
from flax.nnx import State
from jax import Array
from jaxtyping import Float, Int

class Linear(nnx.Module):
    def __init__( self,
        rngs: nnx.Rngs,
        in_features: int,
        out_features: int,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        std = (2 / (in_features + out_features)) ** 0.5
        init_fn = nnx.initializers.truncated_normal(stddev=std, lower=-3.0 * std, upper=3.0 * std)
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
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        std = embedding_dim ** -0.5
        init_fn = nnx.initializers.normal(stddev=std)
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
        dtype: jnp.dtype = jnp.float32,
    ):
        init_fn = nnx.initializers.ones_init()
        weights_data = init_fn(rngs.params(), (d_model,), dtype)
        self.weights = nnx.Param(weights_data)
        self.eps = eps

    def __call__(self, x: Array) -> Array:
        in_dtype = x.dtype
        rms_denom = (jnp.mean(x**2, axis=-1, keepdims=True) + self.eps) ** 0.5
        result = (x / rms_denom) * self.weights.get_value()
        return result.astype(in_dtype)


class SwiGLU(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        d_ff: int,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.w1 = Linear(rngs=rngs, in_features=d_model, out_features=d_ff, dtype=dtype)
        self.w2 = Linear(rngs=rngs, in_features=d_ff, out_features=d_model, dtype=dtype)
        self.w3 = Linear(rngs=rngs, in_features=d_model, out_features=d_ff, dtype=dtype)

    @staticmethod
    def SiLu(x: Array) -> Array:
        return x * jax.nn.sigmoid(x)

    def __call__(self, x: Array) -> Array:
        return self.w2(SwiGLU.SiLu(self.w1(x)) * self.w3(x))


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

        x_even = x[..., 0::2]  # [B, S, d_k / 2]
        x_odd = x[..., 1::2]  # [B, S, d_k / 2]
        x_rotated = jnp.stack([-x_odd, x_even], axis=-1).reshape(*x.shape[:-1], x.shape[-1])
        return x * cos_2d + x_rotated * sin_2d  # [B, S, d_k] * [B, S, d_k]


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
    shifted = in_features - max_vals
    exp = jnp.exp(shifted)
    return exp / jnp.sum(exp, axis=dim, keepdims=True)


def sdpa(
    Q: Float[Array, " ... queries d_k"],
    K: Float[Array, " ... keys d_k"],
    V: Float[Array, " ... values d_v"],
    mask: Float[Array, " ... queries keys"] | None = None,
) -> Float[Array, " ... queries d_v"]:
    scores: Float[Array, " ... queries keys"] = jnp.einsum("... q d, ... k d -> ... q k", Q, K)
    scores = scores / (K.shape[-1] ** 0.5)
    if mask is not None:
        scores = jnp.where(mask == 0.0, -jnp.inf, scores)
    return jnp.einsum("... q k, ... k d -> ... q d", softmax(scores, dim=-1), V)


class MultiHeadSelfAttention(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_heads: int,
        rope_theta: float = 1e4,
        max_seq_len: int = 1024,
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
        self.Q_proj = Linear(rngs=rngs, in_features=d_model, out_features=d_model, dtype=dtype)
        self.K_proj = Linear(rngs=rngs, in_features=d_model, out_features=d_model, dtype=dtype)
        self.V_proj = Linear(rngs=rngs, in_features=d_model, out_features=d_model, dtype=dtype)
        self.O_proj = Linear(rngs=rngs, in_features=d_model, out_features=d_model, dtype=dtype)

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
        batch_size, seq_len, _ = x.shape
        q = self.Q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        k = self.K_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        v = self.V_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).swapaxes(1, 2)
        
        if use_rope:
            positions = jnp.arange(seq_len)
            q = self.rope(q, positions)
            k = self.rope(k, positions)
            
        attn_out = sdpa(q, k, v, self.causal_mask[:seq_len, :seq_len])
        attn_out = attn_out.swapaxes(1, 2).reshape(x.shape)
        return self.O_proj(attn_out)


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.MHA = MultiHeadSelfAttention(
            rngs=rngs,
            d_model=d_model,
            num_heads=num_heads,
            rope_theta=theta,
            max_seq_len=max_seq_len,
            dtype=dtype,
        )
        self.RMSNorm1 = RMSNorm(rngs=rngs, d_model=d_model, dtype=dtype)
        self.RMSNorm2 = RMSNorm(rngs=rngs, d_model=d_model, dtype=dtype)
        self.SwiGLU = SwiGLU(rngs=rngs, d_model=d_model, d_ff=d_ff, dtype=dtype)

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
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.token_embeddings = Embedding(rngs, vocab_size, d_model, dtype=dtype)
        self.layers = nnx.List(
            [
                TransformerBlock(rngs, d_model, num_heads, d_ff, context_length, theta, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln1 = RMSNorm(rngs=rngs, d_model=d_model, dtype=dtype)
        self.linear = Linear(rngs, d_model, vocab_size, dtype=dtype)

    def __call__(self, x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.linear(self.ln1(x))


def cross_entropy_loss(inputs: Float[Array, " batch_size vocab_size"], targets: Int[Array, " batch_size"]) -> Array:
    """Computes Cross-Entropy Loss"""
    # Stabilize logits to avoid overflow when exponentiating
    max_logits = jnp.max(inputs, axis=-1, keepdims=True)
    shifted_logits = inputs - max_logits

    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True)) + max_logits
    log_probs = inputs - log_sum_exp

    target_log_probs = jnp.take_along_axis(log_probs, jnp.expand_dims(targets, axis=1), axis=1).squeeze(axis=1)
    return -target_log_probs.mean()


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


def make_lr_schedule(a_max: float, a_min: float, warmup_iters: int, decay_iters: int) -> optax.Schedule:
    """LR Schedule using warmup_cosine_decay_schedule. Note that decay_iters is inclusive of warmup_iters."""
    if decay_iters <= warmup_iters:  # num decay steps = cosine_annealing_iters - warmup_iters
        raise ValueError("cosine_annealing_iters must be > warmup_iters")
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=a_max,
        warmup_steps=warmup_iters,
        decay_steps=decay_iters,
        end_value=a_min,
    )


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
def train_step(model: nnx.Module, optimizer: nnx.Optimizer, inputs: Array, targets: Array) -> tuple[Array, State, bool]:
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
    gradient_clip: float,
    lr_schedule: optax.Schedule | None = None,
):
    """
    JIT-compiled initialization of Model and Optimizer.

    Args:
        rngs: Random number generators
        model_config: Model configuration dict (static)
        optimizer_config: Optimizer configuration dict (static)
        gradient_clip: Global-norm clipping threshold applied inside the Optax chain
        lr_schedule: Optional Optax learning-rate schedule

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
        )

        # Initialize Optimizer
        optimizer = nnx.Optimizer(
            model,
            build_optimizer_transform(
                optimizer_config,
                gradient_clip=gradient_clip,
                lr_schedule=lr_schedule,
            ),
            wrt=nnx.Param,
        )
        return model, optimizer

    return init_model_and_optimizer(rngs)
