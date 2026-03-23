from __future__ import annotations

import os
from collections.abc import Callable

# from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import jax
import jax.numpy as jnp
import numpy.typing as npt
import optax
from flax import nnx
from flax.nnx import State
from jax import Array
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Float, Int

from data import tokenizer, train_bpe

try:
    from jax_impl.data import DatasetLike
except ModuleNotFoundError:
    DatasetLike = Any

from .common import (
    create_embedding_state,
    create_linear_layer_state,
    create_mha_state,
    create_rmsnorm_state,
    create_swiglu_state,
)

# from .conftest import tensor_to_array


def _import_distributed_modules():
    from jax_impl.distributed import model as distributed_model
    from jax_impl.distributed import train as distributed_train

    return distributed_model, distributed_train


def perpetual_rngs(seed: int):
    rngs = nnx.Rngs(seed)

    def create_wrapper(function: Callable[..., Any]):
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs, rngs=rngs)

        return wrapper

    return create_wrapper


@perpetual_rngs(0)
def run_linear(
    d_in: int, d_out: int, weights: Float[Array, " d_out d_in"], in_features: Float[Array, " ... d_in"], rngs: nnx.Rngs
) -> Float[Array, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Array, "... d_out"]: The transformed output of your linear module.
    """
    raise NotImplementedError("not implemented!")


@perpetual_rngs(1)
def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Array, " vocab_size d_model"],
    token_ids: Int[Array, " ..."],
    rngs: nnx.Rngs,
) -> Float[Array, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Array, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Array, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Array, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    raise NotImplementedError("not implemented!")


@perpetual_rngs(2)
def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Array, " d_ff d_model"],
    w2_weight: Float[Array, " d_model d_ff"],
    w3_weight: Float[Array, " d_ff d_model"],
    in_features: Float[Array, " ... d_model"],
    rngs: nnx.Rngs,
) -> Float[Array, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Array, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Array, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Array, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Array, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Array, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    raise NotImplementedError("not implemented!")


def run_scaled_dot_product_attention(
    Q: Float[Array, " ... queries d_k"],
    K: Float[Array, " ... keys d_k"],
    V: Float[Array, " ... values d_v"],
    mask: Float[Array, " ... queries keys"] | None = None,
) -> Float[Array, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Array, " ... queries d_k"]): Query tensor
        K (Float[Array, " ... keys d_k"]): Key tensor
        V (Float[Array, " ... values d_v"]): Values tensor
        mask (Float[Array, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Array, " ... queries d_v"]: Output of SDPA
    """
    raise NotImplementedError("not implemented!")


@perpetual_rngs(3)
def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Array, " d_k d_in"],
    k_proj_weight: Float[Array, " d_k d_in"],
    v_proj_weight: Float[Array, " d_v d_in"],
    o_proj_weight: Float[Array, " d_model d_v"],
    in_features: Float[Array, " ... sequence_length d_in"],
    rngs: nnx.Rngs,
) -> Float[Array, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Array, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Array, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Array, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Array, "d_model d_v"]): Weights for the output projection
        in_features (Float[Array, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Array, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError("not implemented!")


@perpetual_rngs(4)
def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Array, " d_k d_in"],
    k_proj_weight: Float[Array, " d_k d_in"],
    v_proj_weight: Float[Array, " d_v d_in"],
    o_proj_weight: Float[Array, " d_model d_v"],
    in_features: Float[Array, " ... sequence_length d_in"],
    rngs: nnx.Rngs,
    token_positions: Int[Array, " ... sequence_length"] | None = None,
) -> Float[Array, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Array, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Array, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Array, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Array, "d_model d_v"]): Weights for the output projection
        in_features (Float[Array, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Array, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Array, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError("not implemented!")


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Array, " ... sequence_length d_k"],
    token_positions: Int[Array, " ... sequence_length"],
) -> Float[Array, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Array, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Array, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Array, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    raise NotImplementedError("not implemented!")


@perpetual_rngs(5)
def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Array],
    in_features: Float[Array, " batch sequence_length d_model"],
    rngs: nnx.Rngs,
) -> Float[Array, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Array]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Array, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Array, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError("not implemented!")


@perpetual_rngs(6)
def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Array],
    in_indices: Int[Array, " batch_size sequence_length"],
    rngs: nnx.Rngs,
) -> Float[Array, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Array]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Array, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Array, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError("not implemented!")


@perpetual_rngs(7)
def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Array, " d_model"],
    in_features: Float[Array, " ... d_model"],
    rngs: nnx.Rngs,
) -> Float[Array, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Array,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    raise NotImplementedError("not implemented!")


def run_silu(in_features: Float[Array, " ..."]) -> Float[Array, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Array,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError("not implemented!")


@perpetual_rngs(8)
def run_get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, rngs: nnx.Rngs) -> tuple[Array, Array]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.

    Returns:
        Tuple of jax.Arrays of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError("not implemented!")


def run_softmax(in_features: Float[Array, " ..."], dim: int) -> Float[Array, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    raise NotImplementedError("not implemented!")


def run_cross_entropy(
    inputs: Float[Array, " batch_size vocab_size"], targets: Int[Array, " batch_size"]
) -> Float[Array, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Array, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Array, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Array, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError("not implemented!")


def run_gradient_clipping(gradient_state: State, max_l2_norm: float) -> State:
    """Given a gradient state, clip the gradients to have l2 norm at most max_l2_norm.

    Args:
        gradient_state (State): collection of gradients.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    Returns:
        State: the clipped gradient state.
    """
    raise NotImplementedError("not implemented!")


# def get_adamw_cls() -> Any:
# """
# Returns a torch.optim.Optimizer that implements AdamW.
# """
# return model.AdamW


def get_adamw_cls():
    """
    Returns a function that creates an nnx.Optimizer with Optax AdamW.
    """
    raise NotImplementedError("not implemented!")


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError("not implemented!")


def run_save_checkpoint(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError("not implemented!")


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nnx.Module,
    optimizer: nnx.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError("not implemented!")


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return tokenizer.Tokenizer(vocab, merges, special_tokens)
    # return ref.Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    return train_bpe.train_bpe(input_path, vocab_size, special_tokens)


def get_mesh(mesh_shape: list[int], mesh_axis_names: list[str]) -> Mesh:
    _, train = _import_distributed_modules()
    return train.create_mesh(mesh_shape, mesh_axis_names)


def get_sharded_batch(
    dataset: DatasetLike | Any,
    batch_size: int,
    context_length: int,
    mesh: Mesh,
    batch_partition_spec: P | None = None,
):
    raise NotImplementedError("not implemented!")


def get_sharded_model_and_optimizer(
    model_config: dict,
    optimizer_config: dict,
    mesh: Mesh,
    sharding_mode: str,
):
    raise NotImplementedError("not implemented!")


def _normalize_spec(spec: P) -> tuple:
    """Normalize a PartitionSpec to a tuple padded with None to a fixed length.

    JAX may drop trailing Nones (e.g. P('data', None) becomes P('data',)),
    so we canonicalize both sides before comparing.
    """
    return tuple(spec)


def assert_array_sharded(array: Array, expected_spec: P, mesh: Mesh, name: str = ""):
    """Verify a JAX array is physically sharded across devices as expected.

    Checks:
        1. The array's PartitionSpec matches expected_spec (modulo trailing Nones).
        2. There is one shard per device in the mesh.
        3. Each shard lives on a unique device.
        4. Each shard has the correct shape (global dim / mesh axis size for
           each partitioned axis).

    Args:
        array: The sharded JAX array to verify.
        expected_spec: The PartitionSpec the array should have.
        mesh: The device mesh used for sharding.
        name: Optional label for error messages.
    """
    label = f" ({name})" if name else ""
    shards = array.addressable_shards
    num_devices = mesh.devices.size

    sharding_spec = getattr(array.sharding, "spec", None)
    assert sharding_spec is not None, (
        f"[ERROR]: array.sharding doesn't have a .spec attribute. type: {type(array.sharding)}"
    )

    # 1. Spec matches (pad shorter spec with None for fair comparison)
    actual_tuple = _normalize_spec(sharding_spec)
    expected_tuple = _normalize_spec(expected_spec)
    max_len = max(len(actual_tuple), len(expected_tuple))
    actual_padded = actual_tuple + (None,) * (max_len - len(actual_tuple))
    expected_padded = expected_tuple + (None,) * (max_len - len(expected_tuple))
    assert actual_padded == expected_padded, f"Spec mismatch{label}: got {sharding_spec}, expected {expected_spec}"

    # 2. One shard per device
    assert len(shards) == num_devices, f"Shard count{label}: got {len(shards)}, expected {num_devices}"

    # 3. Shards on distinct devices
    shard_devices = {s.device for s in shards}
    assert len(shard_devices) == num_devices, f"Unique devices{label}: {len(shard_devices)} of {num_devices}"

    # 4. Correct per-shard shape
    expected_shard_shape = list(array.shape)
    for i, axis_name in enumerate(expected_padded):
        if axis_name is not None:
            expected_shard_shape[i] //= mesh.shape[axis_name]
    expected_shard_shape = tuple(expected_shard_shape)

    for shard in shards:
        assert shard.data.shape == expected_shard_shape, (
            f"Shard shape{label}: got {shard.data.shape}, "
            f"expected {expected_shard_shape} "
            f"(global {array.shape}, spec {expected_spec})"
        )


def get_model_param_arrays(model_instance, num_layers: int) -> dict[str, Array]:
    """Return a dict of param name -> JAX array for all weight matrices and embeddings.

    Covers every learnable weight that should be sharded, skipping
    RoPE tables and causal masks.
    """
    params = {}

    def _arr(variable):
        """Get the underlying JAX array from an nnx.Variable."""
        return variable.get_value()

    # Token embeddings
    params["token_embeddings"] = _arr(model_instance.token_embeddings.weights)

    # Per-layer parameters
    for i in range(num_layers):
        layer = model_instance.layers[i]
        p = f"layers.{i}"
        # MHA projections
        params[f"{p}.MHA.Q_proj"] = _arr(layer.MHA.Q_proj.weights)
        params[f"{p}.MHA.K_proj"] = _arr(layer.MHA.K_proj.weights)
        params[f"{p}.MHA.V_proj"] = _arr(layer.MHA.V_proj.weights)
        params[f"{p}.MHA.O_proj"] = _arr(layer.MHA.O_proj.weights)
        # RMSNorm
        params[f"{p}.RMSNorm1"] = _arr(layer.RMSNorm1.weights)
        params[f"{p}.RMSNorm2"] = _arr(layer.RMSNorm2.weights)
        # SwiGLU
        params[f"{p}.SwiGLU.w1"] = _arr(layer.SwiGLU.w1.weights)
        params[f"{p}.SwiGLU.w2"] = _arr(layer.SwiGLU.w2.weights)
        params[f"{p}.SwiGLU.w3"] = _arr(layer.SwiGLU.w3.weights)

    # Final layer norm and output head
    params["ln_final"] = _arr(model_instance.ln1.weights)
    params["lm_head"] = _arr(model_instance.linear.weights)

    return params


def get_expected_model_sharding_specs(num_layers: int, sharding_mode: str) -> dict[str, P]:
    """Returns expected PartitionSpec for each weight parameter based on ShardingConfig defaults.

    Maps each parameter to the sharding strategy used in TransformerLM:
        - token_embeddings: embedding
        - lm_head: lm_head
        - Q/K/V projections, SwiGLU w1/w3: dense_in
        - O projection, SwiGLU w2: dense_out
        - RMSNorm layers: norm
    """
    model, _ = _import_distributed_modules()
    sc = model.get_sharding_config_for_mode(sharding_mode)
    expected = {}

    expected["token_embeddings"] = P(*sc.embedding)

    for i in range(num_layers):
        p = f"layers.{i}"
        expected[f"{p}.MHA.Q_proj"] = P(*sc.dense_in)
        expected[f"{p}.MHA.K_proj"] = P(*sc.dense_in)
        expected[f"{p}.MHA.V_proj"] = P(*sc.dense_in)
        expected[f"{p}.MHA.O_proj"] = P(*sc.dense_out)
        expected[f"{p}.RMSNorm1"] = P(*sc.norm)
        expected[f"{p}.RMSNorm2"] = P(*sc.norm)
        expected[f"{p}.SwiGLU.w1"] = P(*sc.dense_in)
        expected[f"{p}.SwiGLU.w2"] = P(*sc.dense_out)
        expected[f"{p}.SwiGLU.w3"] = P(*sc.dense_in)

    expected["ln_final"] = P(*sc.norm)
    expected["lm_head"] = P(*sc.lm_head)

    return expected


def get_expected_batch_sharding_spec(sharding_mode: str) -> P:
    model, _ = _import_distributed_modules()
    return P(*model.get_batch_sharding_for_mode(sharding_mode))


def get_optimizer_state_arrays(optimizer_instance) -> list[Array]:
    """Return all optimizer state arrays (mu/nu from Adam).

    Filters to multi-dimensional arrays only, skipping scalar counters
    like the Adam step count.
    """
    opt_state = nnx.state(optimizer_instance, nnx.optimizer.OptState)
    arrays = []
    for leaf in jax.tree.leaves(opt_state):
        if isinstance(leaf, jax.Array) and leaf.ndim > 0:
            arrays.append(leaf)
    return arrays
