"""Tests that model, batch, and optimizer arrays are physically sharded across devices.

Requires 8 CPU virtual devices. Run with:
    USE_CPU_SHARDING=1 uv run pytest jax_tests/test_sharding.py -v
Or ensure XLA_FLAGS includes --xla_force_host_platform_device_count=8.
"""

import os

# Must be set before JAX initializes to create virtual CPU devices
if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"

from collections import Counter

import jax
import numpy as np
import pytest
from flax import nnx

from jax_impl import model as model_module

from . import adapters

# ---------------------------------------------------------------------------
# Small model / optimizer config for testing
# ---------------------------------------------------------------------------
NUM_LAYERS = 2

MODEL_CONFIG = {
    "vocab_size": 256,
    "context_length": 32,
    "d_model": 64,
    "num_heads": 4,
    "d_ff": 128,
    "num_layers": NUM_LAYERS,
    "theta": 10000.0,
}

OPTIMIZER_CONFIG: dict[str, object] = {
    "type": "adamw",
    "lr": 1e-3,
    "betas": [0.9, 0.999],
    "weight_decay": 0.01,
    "eps": 1e-8,
}

MESH_AXIS_NAMES = ["data", "tensor"]
MODE_CASES: list[tuple[str, list[int]]] = [
    ("dp", [8, 1]),
    ("fsdp", [8, 1]),
    ("tp", [1, 4]),
    ("fsdp_tp", [4, 2]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class MockDataset:
    """Minimal dataset wrapper that satisfies train.get_batch_from_memmap."""

    def __init__(self, size: int = 10_000):
        self.data = np.random.randint(0, 256, size=size).astype(np.uint16)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ensure_enough_devices():
    if len(jax.devices()) < 8:
        pytest.skip("Need >= 8 devices. Set XLA_FLAGS='--xla_force_host_platform_device_count=8'")


@pytest.fixture(params=MODE_CASES, scope="module")
def sharding_case(request, ensure_enough_devices):
    mode, mesh_shape = request.param
    mesh = adapters.get_mesh(mesh_shape, MESH_AXIS_NAMES)
    return {"mode": mode, "mesh": mesh, "mesh_shape": mesh_shape}


@pytest.fixture(scope="module")
def unsharded_model_and_optimizer():
    rngs = nnx.Rngs(1234)
    return model_module.create_model_and_optimizer(rngs, MODEL_CONFIG, OPTIMIZER_CONFIG, None, mesh=None)


@pytest.fixture
def sharded_model_and_optimizer(sharding_case):
    return adapters.get_sharded_model_and_optimizer(
        MODEL_CONFIG,
        OPTIMIZER_CONFIG,
        sharding_case["mesh"],
        sharding_mode=sharding_case["mode"],
    )


@pytest.fixture
def sharded_batch(sharding_case):
    dataset = MockDataset()
    batch_spec = adapters.get_expected_batch_sharding_spec(sharding_case["mode"])
    return adapters.get_sharded_batch(
        dataset,
        batch_size=8,
        context_length=32,
        mesh=sharding_case["mesh"],
        batch_partition_spec=batch_spec,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_batch_sharding(sharding_case, sharded_batch):
    """Inputs and targets follow mode-specific batch partition specs."""
    mesh = sharding_case["mesh"]
    mode = sharding_case["mode"]
    inputs, targets = sharded_batch
    expected_spec = adapters.get_expected_batch_sharding_spec(mode)

    adapters.assert_array_sharded(inputs, expected_spec, mesh, name="inputs")
    adapters.assert_array_sharded(targets, expected_spec, mesh, name="targets")


def test_model_sharding(sharding_case, sharded_model_and_optimizer):
    """Every weight matrix and embedding is physically distributed for each sharding mode."""
    mesh = sharding_case["mesh"]
    mode = sharding_case["mode"]
    model_instance, _ = sharded_model_and_optimizer

    params = adapters.get_model_param_arrays(model_instance, NUM_LAYERS)
    expected_specs = adapters.get_expected_model_sharding_specs(NUM_LAYERS, sharding_mode=mode)

    for name, spec in expected_specs.items():
        assert name in params, f"Missing parameter: {name}"
        adapters.assert_array_sharded(params[name], spec, mesh, name=name)


def test_optimizer_sharding(sharding_case, sharded_model_and_optimizer):
    """Adam mu/nu states are physically distributed consistently with model params."""
    mesh = sharding_case["mesh"]
    model_instance, optimizer = sharded_model_and_optimizer

    model_params = adapters.get_model_param_arrays(model_instance, NUM_LAYERS)
    opt_arrays = adapters.get_optimizer_state_arrays(optimizer)

    # Adam stores mu and nu per parameter → 2x model params
    assert len(opt_arrays) == 2 * len(model_params), (
        f"Expected {2 * len(model_params)} optimizer state arrays, got {len(opt_arrays)}"
    )

    # Each optimizer state array must be physically distributed across all devices
    # with shard shapes consistent with its own spec
    from typing import cast

    from jax.sharding import NamedSharding

    for i, arr in enumerate(opt_arrays):
        adapters.assert_array_sharded(arr, cast(NamedSharding, arr.sharding).spec, mesh, name=f"opt_state[{i}]")

    # The set of PartitionSpecs across optimizer states should mirror the
    # model params (each spec appearing exactly 2x as often for mu + nu)
    #
    model_spec_counts = Counter(cast(NamedSharding, a.sharding).spec for a in model_params.values())
    opt_spec_counts = Counter(cast(NamedSharding, a.sharding).spec for a in opt_arrays)

    for spec, count in model_spec_counts.items():
        assert opt_spec_counts.get(spec, 0) == 2 * count, (
            f"PartitionSpec {spec}: expected {2 * count} optimizer arrays, got {opt_spec_counts.get(spec, 0)}"
        )


@pytest.mark.parametrize(
    ("mode", "mesh_shape"),
    [
        ("dp", [4, 2]),
        ("fsdp", [1, 8]),
        ("tp", [4, 2]),
        ("fsdp_tp", [8, 1]),
    ],
)
def test_invalid_mode_mesh_combinations_raise(mode: str, mesh_shape: list[int], ensure_enough_devices):
    mesh = adapters.get_mesh(mesh_shape, MESH_AXIS_NAMES)
    with pytest.raises(ValueError):
        model_module.validate_mesh_for_mode(mesh, mode)


def test_invalid_mode_name_raises():
    with pytest.raises(ValueError):
        model_module.get_sharding_config_for_mode("bad_mode")


def test_no_sharding_still_supported(unsharded_model_and_optimizer):
    """No-sharding path (mesh=None, sharding_config=None) still initializes and keeps arrays local."""
    model_instance, optimizer = unsharded_model_and_optimizer
    params = adapters.get_model_param_arrays(model_instance, NUM_LAYERS)
    opt_arrays = adapters.get_optimizer_state_arrays(optimizer)

    for arr in params.values():
        assert len(arr.addressable_shards) == 1
    for arr in opt_arrays:
        assert len(arr.addressable_shards) == 1
