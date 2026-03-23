from __future__ import annotations

import pathlib
from functools import lru_cache

import jax.numpy as jnp
import torch
from flax import nnx
from flax.nnx import State
from jax import Array
from jaxtyping import Float, Int
from torch import Tensor

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def create_linear_layer_state(weights: Float[Array, " d_out d_in"]) -> State:
    # Reference fixtures are PyTorch-style [out, in]; our JAX Linear now stores [in, out].
    linear_weights = jnp.swapaxes(weights, -1, -2)
    return State(
        {
            "weights": nnx.Param(linear_weights),
        }
    )


def create_embedding_state(weights: Float[Array, " vocab_size d_model"]) -> State:
    return State(
        {
            "weights": nnx.Param(weights),
        }
    )


def create_rmsnorm_state(weights: Float[Array, " d_model"]) -> State:
    return State(
        {
            "weights": nnx.Param(weights),
        }
    )


def create_swiglu_state(
    w1: Float[Array, " d_ff d_model"], w2: Float[Array, " d_model d_ff"], w3: Float[Array, " d_ff d_model"]
) -> State:
    return State(
        {
            "w1": create_linear_layer_state(w1),
            "w2": create_linear_layer_state(w2),
            "w3": create_linear_layer_state(w3),
        }
    )


def create_mha_state(
    q_proj_weight: Float[Array, " d_k d_in"],
    k_proj_weight: Float[Array, " d_k d_in"],
    v_proj_weight: Float[Array, " d_v d_in"],
    o_proj_weight: Float[Array, " d_model d_v"],
) -> State:
    return State(
        {
            "Q_proj": create_linear_layer_state(q_proj_weight),
            "K_proj": create_linear_layer_state(k_proj_weight),
            "V_proj": create_linear_layer_state(v_proj_weight),
            "O_proj": create_linear_layer_state(o_proj_weight),
        }
    )
