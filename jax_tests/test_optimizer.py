import numpy
import torch
import jax
import jax.numpy as jnp
import flax
from flax import nnx
from flax.nnx import State
import optax

from .adapters import get_adamw_cls, run_get_lr_cosine_schedule
from .conftest import tensor_to_array


def _optimize(opt_class) -> State:
    model = nnx.Linear(3, 2, rngs=nnx.Rngs(0), use_bias=False)
    try:
        opt = opt_class(
            model=model,
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    except TypeError:
        opt = opt_class(
            lr=1e-3,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    rngs = nnx.Rngs(0)
    # Use 1000 optimization steps for testing
    for _ in range(1000):
        x = jax.random.uniform(rngs.params(), shape=(model.in_features,))
        y = jnp.array([x[0] + x[1], -x[2]])
        loss, grad_state = nnx.value_and_grad(lambda model: ((model(x) - y) ** 2).mean())(model)
        opt.update(model, grad_state)
    return jnp.concatenate(jax.tree.leaves(nnx.state(model)))


# def test_adamw(numpy_snapshot):
    # """
    # Our reference implementation yields slightly different results than the
    # Optax AdamW, since there are a couple different ways that you can apply
    # weight decay that are equivalent in principle, but differ in practice due to
    # floating point behavior. So, we test that the provided implementation matches
    # _either_ our reference implementation's expected results or those from the Optax AdamW.
    # """
    # # expected_weights = torch.load(FIXTURES_PATH / "adamw_expected_params.pt")
    # optax_weights = _optimize(lambda model, lr, weight_decay, betas, eps: nnx.Optimizer(model, optax.adamw(learning_rate=lr, weight_decay=weight_decay, b1=betas[0], b2=betas[1], eps=eps), wrt=nnx.Param))
    # actual_weights = _optimize(get_adamw_cls())

    # # Might need to exit early if the weights match optax, since that should also be valid
    # matches_optax = jnp.allclose(actual_weights, optax_weights, atol=1e-4)
    # if matches_optax:
        # return

    # numpy_snapshot.assert_match(
        # actual_weights,
        # atol=1e-4,
    # )


def test_adamw(numpy_snapshot):
    """
    Test the new function-based AdamW optimizer that uses optax.
    This should also match either the reference implementation or optax.
    """
    optax_weights = _optimize(lambda model, lr, weight_decay, betas, eps: nnx.Optimizer(model, optax.adamw(learning_rate=lr, weight_decay=weight_decay, b1=betas[0], b2=betas[1], eps=eps), wrt=nnx.Param))
    actual_weights = _optimize(get_adamw_cls())

    # Might need to exit early if the weights match optax, since that should also be valid
    matches_optax = jnp.allclose(actual_weights, optax_weights, atol=1e-4)
    if matches_optax:
        return

    numpy_snapshot.assert_match(
        actual_weights,
        atol=1e-4,
    )


def test_get_lr_cosine_schedule():
    max_learning_rate = 1
    min_learning_rate = 1 * 0.1
    warmup_iters = 7
    cosine_cycle_iters = 21

    expected_lrs = [
        0,
        0.14285714285714285,
        0.2857142857142857,
        0.42857142857142855,
        0.5714285714285714,
        0.7142857142857143,
        0.8571428571428571,
        1.0,
        0.9887175604818206,
        0.9554359905560885,
        0.9018241671106134,
        0.8305704108364301,
        0.7452476826029011,
        0.6501344202803414,
        0.55,
        0.44986557971965857,
        0.3547523173970989,
        0.26942958916356996,
        0.19817583288938662,
        0.14456400944391146,
        0.11128243951817937,
        0.1,
        0.1,
        0.1,
        0.1,
    ]
    actual_lrs = [
        run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for it in range(25)
    ]
    numpy.testing.assert_allclose(numpy.array(actual_lrs), numpy.array(expected_lrs))
