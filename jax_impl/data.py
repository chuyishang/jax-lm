import os
from typing import TypeAlias

import jax
import numpy as np
from flax import nnx
from jax import Array
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_impl import model as model_module


class MemMapDataset:
    """Memory-efficient dataset using np.memmap for large datasets."""

    def __init__(self, data_path: str) -> None:
        """Initialize dataset with memory-mapped file.

        Args:
            data_path (str): Path to binary file containing tokenized data
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        print(f"Loaded dataset from {data_path} with {len(self.data):,} tokens")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.data[idx]


class DummyDataset:
    """Random token data for benchmarking. Same interface as MemMapDataset."""

    def __init__(self, vocab_size: int, num_tokens: int = 1_000_000) -> None:
        self.data = np.random.randint(0, vocab_size, size=num_tokens, dtype=np.uint16)
        print(f"Created dummy dataset with {len(self.data):,} random tokens (vocab_size={vocab_size})")

    def __len__(self) -> int:
        return len(self.data)


DatasetLike: TypeAlias = MemMapDataset | DummyDataset


def get_batch_from_memmap(
    rngs: nnx.Rngs,
    dataset: DatasetLike,
    batch_size: int,
    context_length: int,
    mesh: Mesh | None = None,
    batch_partition_spec: P | None = None,
) -> tuple[Array, Array]:
    """Sample a batch from memory-mapped dataset with optional sharding.

    Args:
        rngs (nnx.Rngs): JAX random number generator
        dataset (DatasetLike): Dataset instance (e.g. MemMapDataset or DummyDataset)
        batch_size (int): Number of sequences in batch
        context_length (int): Length of each sequence
        mesh (Optional[Mesh]): Mesh for sharding the batch across devices
        batch_partition_spec (Optional[P]): PartitionSpec for input/target batches when mesh is provided.

    Returns:
        tuple[Array, Array]: Tuple of (inputs, targets) arrays, optionally sharded
    """
    # Use the model's get_batch function with the memmap data
    inputs, targets = model_module.get_batch(
        rngs,
        dataset.data,
        batch_size,
        context_length,
    )

    # Apply sharding if mesh is provided
    if mesh is not None:
        spec = batch_partition_spec if batch_partition_spec is not None else P("data", None)
        # # Shard batch dimension across 'data' axis
        # sharding = NamedSharding(mesh, P('data', None))
        # inputs = jax.device_put(inputs, sharding)
        # targets = jax.device_put(targets, sharding)
        with jax.set_mesh(mesh):
            inputs = jax.device_put(inputs, spec)
            targets = jax.device_put(targets, spec)
    # breakpoint()
    return inputs, targets
