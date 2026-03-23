import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import flax
from flax import nnx
from flax.nnx import State

from .adapters import get_adamw_cls, run_load_checkpoint, run_save_checkpoint


class _TestNet(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_input: int = 100, d_output: int = 10):
        super().__init__()
        self.fc1 = nnx.Linear(d_input, 200, rngs=rngs)
        self.fc2 = nnx.Linear(200, 100, rngs=rngs)
        self.fc3 = nnx.Linear(100, d_output, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.fc1(x))
        x = nnx.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_keys_of_pytree(pytree: nnx.State) -> list[str]:
    flat_leaves_with_paths, _ = jax.tree_util.tree_flatten_with_path(pytree)
    all_keys = [key_path for key_path, _ in flat_leaves_with_paths]
    return all_keys

def are_optimizers_equal(optimizer1_state_dict, optimizer2_state_dict, atol=1e-8, rtol=1e-5):
    # Check if the keys of the main dictionaries are equal (e.g., 'state', 'param_groups')
    if set(get_keys_of_pytree(optimizer1_state_dict)) != set(get_keys_of_pytree(optimizer2_state_dict)):
        return False

    # Check states
    def check_equalish(item1, item2):
        if isinstance(item1, jnp.ndarray) and isinstance(item2, jnp.ndarray):
            numpy.testing.assert_allclose(item1, item2, atol=atol, rtol=rtol)
        else:
            assert item1 == item2
    jax.tree.map(
        check_equalish,
        optimizer1_state_dict, optimizer2_state_dict
    )
    return True


def test_checkpointing(tmp_path):
    data_rng = nnx.Rngs(1)
    d_input = 100
    d_output = 10
    num_iters = 10

    model = _TestNet(rngs=nnx.Rngs(0), d_input=d_input, d_output=d_output)
    optimizer = get_adamw_cls()(
        model=model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    # Use 1000 optimization steps for testing
    it = 0
    for _ in range(num_iters):
        x = jax.random.uniform(data_rng(), shape=(d_input,))
        y = jax.random.uniform(data_rng(), shape=(d_output,))
        loss, grad_state = nnx.value_and_grad(lambda model: ((model(x) - y) ** 2).mean())(model)
        optimizer.update(model, grad_state)
        it += 1

    serialization_path = tmp_path / "checkpoint.pt"
    # Save the model
    run_save_checkpoint(
        model,
        optimizer,
        iteration=it,
        out=serialization_path,
    )

    # Load the model back again
    new_model = _TestNet(rngs=nnx.Rngs(0), d_input=d_input, d_output=d_output)
    new_optimizer = get_adamw_cls()(
        model=new_model,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    loaded_iterations = run_load_checkpoint(src=serialization_path, model=new_model, optimizer=new_optimizer)
    assert it == loaded_iterations

    # Compare the loaded model state with the original model state
    original_model_state = nnx.state(model)
    original_optimizer_state = nnx.state(optimizer)
    new_model_state = nnx.state(new_model)
    new_optimizer_state = nnx.state(new_optimizer)

    # Check that state dict keys match
    assert set(get_keys_of_pytree(original_model_state)) == set(get_keys_of_pytree(new_model_state))
    assert set(get_keys_of_pytree(original_optimizer_state)) == set(get_keys_of_pytree(new_optimizer_state))

    # compare the model state dicts
    jax.tree.map(
        lambda item1, item2: numpy.testing.assert_allclose(item1, item2, atol=1e-8, rtol=1e-5),
        original_model_state, new_model_state
    )
    
    # compare the optimizer state dicts
    assert are_optimizers_equal(original_optimizer_state, new_optimizer_state)
