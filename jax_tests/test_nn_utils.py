import numpy
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.nnx import State

from .adapters import run_cross_entropy, run_gradient_clipping, run_softmax

from .conftest import tensor_to_array, array_to_tensor


def test_softmax_matches_pytorch():
    x = torch.tensor(
        [
            [0.4655, 0.8303, 0.9608, 0.9656, 0.6840],
            [0.2583, 0.2198, 0.9334, 0.2995, 0.1722],
            [0.1573, 0.6860, 0.1327, 0.7284, 0.6811],
        ]
    )
    expected = F.softmax(x, dim=-1)
    numpy.testing.assert_allclose(run_softmax(tensor_to_array(x), dim=-1), expected.detach().numpy(), atol=1e-6)
    # Test that softmax handles numerical overflow issues
    numpy.testing.assert_allclose(
        run_softmax(tensor_to_array(x + 100), dim=-1),
        expected.detach().numpy(),
        atol=1e-6,
    )


def test_cross_entropy():
    inputs_tensor = torch.tensor(
        [
            [
                [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
            ],
            [
                [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
            ],
        ]
    )
    inputs_array = tensor_to_array(inputs_tensor)

    targets_tensor = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])
    targets_array = tensor_to_array(targets_tensor)

    expected_torch = F.cross_entropy(inputs_tensor.view(-1, inputs_tensor.size(-1)), targets_tensor.view(-1))
    expected_jax = optax.softmax_cross_entropy(inputs_array.reshape(-1, inputs_array.shape[-1]), jax.nn.one_hot(targets_array.reshape(-1), inputs_array.shape[-1])).mean()
    numpy.testing.assert_allclose(
        run_cross_entropy(inputs_array.reshape(-1, inputs_array.shape[-1]), targets_array.reshape(-1)),
        expected_torch.detach().numpy(),
        atol=1e-4,
    )
    numpy.testing.assert_allclose(
        run_cross_entropy(inputs_array.reshape(-1, inputs_array.shape[-1]), targets_array.reshape(-1)),
        expected_jax,
        atol=1e-4,
    )

    # Test that cross-entropy handles numerical overflow issues
    large_inputs_tensor = 1000.0 * inputs_tensor
    large_inputs_array = tensor_to_array(large_inputs_tensor)

    large_expected_cross_entropy_torch = F.cross_entropy(large_inputs_tensor.view(-1, large_inputs_tensor.size(-1)), targets_tensor.view(-1))
    large_expected_cross_entropy_jax = optax.softmax_cross_entropy(large_inputs_array.reshape(-1, large_inputs_array.shape[-1]), jax.nn.one_hot(targets_array.reshape(-1), large_inputs_array.shape[-1])).mean()
    numpy.testing.assert_allclose(
        run_cross_entropy(large_inputs_array.reshape(-1, large_inputs_array.shape[-1]), targets_array.reshape(-1)),
        large_expected_cross_entropy_torch.detach().numpy(),
        atol=1e-4,
    )
    numpy.testing.assert_allclose(
        run_cross_entropy(large_inputs_array.reshape(-1, large_inputs_array.shape[-1]), targets_array.reshape(-1)),
        large_expected_cross_entropy_jax,
        atol=1e-4,
    )


def test_gradient_clipping():
    model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
    params = nnx.state(model)
    max_norm = 1e-2
    lr = 1e-3

    optimizer = nnx.Optimizer(model, optax.chain(optax.clip_by_global_norm(max_norm), optax.sgd(lr)), wrt=nnx.Param)

    x = jnp.ones((2, 2))
    y = jnp.ones((2, 3))
    loss, grad_state = nnx.value_and_grad(lambda model: ((model(x) - y) ** 2).mean())(model)

    clipped_grad_state = run_gradient_clipping(grad_state, max_norm)
    clipped_params = jax.tree.leaves(jax.tree.map(lambda p, g: p - lr * g, params, clipped_grad_state))

    optimizer.update(model, grad_state)
    expected_params = jax.tree.leaves(nnx.state(model))

    for p, expected_p in zip(clipped_params, expected_params):
        numpy.testing.assert_allclose(p, expected_p, atol=1e-6)
