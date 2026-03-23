import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from pytorch_impl.model import TransformerLM


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(use_wandb: bool, config: dict, run_name: Optional[str] = None):
    """Setup logging, optionally with Weights & Biases."""
    if use_wandb:
        try:
            import wandb

            wandb.init(project=config.get("wandb_project", "cs336-assignment-1"), name=run_name, config=config)
            return wandb
        except ImportError:
            print("Warning: wandb not installed. Install with 'pip install wandb'")
            return None
    return None


class MemMapDataset:
    """Memory-efficient dataset using np.memmap for large datasets."""

    def __init__(self, data_path: str, verbose: bool = True):
        """
        Initialize dataset with memory-mapped file.

        Args:
            data_path: Path to binary file containing tokenized data
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        if verbose:
            print(f"Loaded dataset from {data_path} with {len(self.data):,} tokens")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_batch_from_memmap(
    dataset: MemMapDataset, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch from memory-mapped dataset.

    Args:
        dataset: MemMapDataset instance
        batch_size: Number of sequences in batch
        context_length: Length of each sequence
        device: Device to place tensors on

    Returns:
        Tuple of (inputs, targets) tensors
    """
    start_indices = torch.randint(0, len(dataset.data) - context_length, (batch_size,))
    batch = []
    for start in start_indices:
        batch_item = torch.tensor(dataset.data[start : start + context_length + 1], dtype=torch.long)
        batch.append(batch_item)
    batch_tensor = torch.stack(batch, dim=0).to(device)
    return batch_tensor[:, :-1], batch_tensor[:, 1:]


def get_lr_schedule(t: int, a_max: float, a_min: float, warmup_iters: int, cosine_annealing_iters: int) -> float:
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


@dataclass
class DistributedRuntime:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    is_main: bool
    mode: str
    # mainly needed for FSDP_TP
    dp_size: int
    tp_size: int
    dp_rank: int
    tp_rank: int


def is_data_parallel_mode(mode: str) -> bool:
    return mode in {"dp", "fsdp", "fsdp_tp"}


# FSDP_TP stuff
def data_parallel_world_size(runtime: DistributedRuntime) -> int:
    if runtime.mode == "fsdp_tp":
        return runtime.dp_size
    if runtime.mode in {"dp", "fsdp"}:
        return runtime.world_size
    return 1


def data_parallel_rank(runtime: DistributedRuntime) -> int:
    if runtime.mode == "fsdp_tp":
        return runtime.dp_rank
    if runtime.mode in {"dp", "fsdp"}:
        return runtime.rank
    return 0


def validate_tensor_parallel_config(config: dict, runtime: DistributedRuntime) -> None:
    if runtime.mode not in {"tp", "fsdp_tp"}:
        return

    tp_size = runtime.tp_size
    d_model = int(config["model"]["d_model"])
    d_ff = int(config["model"]["d_ff"])
    num_heads = int(config["model"]["num_heads"])

    if d_model % tp_size != 0:
        raise ValueError(
            f"{runtime.mode} requires d_model divisible by tp_size (d_model={d_model}, tp_size={tp_size})."
        )
    if d_ff % tp_size != 0:
        raise ValueError(f"{runtime.mode} requires d_ff divisible by tp_size (d_ff={d_ff}, tp_size={tp_size}).")
    if num_heads % tp_size != 0:
        raise ValueError(
            f"{runtime.mode} requires num_heads divisible by tp_size (num_heads={num_heads}, tp_size={tp_size})."
        )


def setup_distributed_runtime(config: dict) -> DistributedRuntime:
    distributed_config = config.get("distributed", {})
    if not distributed_config.get("enabled", False):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            raise RuntimeError(
                "distributed.enabled=false is incompatible with WORLD_SIZE>1. "
                "Use NPROC_PER_NODE=1 for non-distributed runs."
            )
        return DistributedRuntime(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main=True,
            mode="none",
            dp_size=1,
            tp_size=1,
            dp_rank=0,
            tp_rank=0,
        )

    mode = distributed_config.get("mode", "dp")
    if mode not in {"dp", "fsdp", "tp", "fsdp_tp"}:
        raise ValueError(f"Unknown distributed mode: {mode}")
    if not torch.cuda.is_available():
        raise ValueError(f"distributed.enabled=true with mode={mode} requires CUDA.")

    required_env_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE"]
    missing_env_vars = [name for name in required_env_vars if name not in os.environ]
    if missing_env_vars:
        missing_vars = ", ".join(missing_env_vars)
        raise RuntimeError(f"mode={mode} requires torchrun environment variables ({missing_vars}).")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise ValueError(f"mode={mode} requires WORLD_SIZE >= 2.")

    tp_size = 1
    dp_size = world_size
    if mode == "tp":
        tp_size = world_size
        dp_size = 1
    elif mode == "fsdp_tp":
        tp_size = int(distributed_config.get("tp_size", 0))
        if tp_size < 2:
            raise ValueError(f"mode=fsdp_tp requires distributed.tp_size >= 2 (got {tp_size}).")
        if world_size % tp_size != 0:
            raise ValueError(
                "mode=fsdp_tp requires WORLD_SIZE divisible by distributed.tp_size "
                f"(WORLD_SIZE={world_size}, tp_size={tp_size})."
            )
        dp_size = world_size // tp_size

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    dp_rank = rank if mode in {"dp", "fsdp"} else (rank // tp_size if mode == "fsdp_tp" else 0)
    tp_rank = rank if mode == "tp" else (rank % tp_size if mode == "fsdp_tp" else 0)
    return DistributedRuntime(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_main=rank == 0,
        mode=mode,
        dp_size=dp_size,
        tp_size=tp_size,
        dp_rank=dp_rank,
        tp_rank=tp_rank,
    )


def cleanup_distributed_runtime(runtime: DistributedRuntime) -> None:
    if runtime.enabled and dist.is_initialized():
        dist.destroy_process_group()


def apply_tensor_parallelism_plan(model: torch.nn.Module, tp_mesh: Any) -> torch.nn.Module:
    if not isinstance(model, TransformerLM):
        raise TypeError(f"Tensor parallel mode currently expects TransformerLM, got {type(model).__name__}.")
    if len(model.layers) == 0:
        raise ValueError("Tensor parallel mode requires at least one Transformer layer.")

    try:
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
    except ImportError as exc:
        raise RuntimeError("Tensor parallel mode requires torch.distributed.tensor.parallel support.") from exc

    tp_plan = {}
    for layer_idx in range(len(model.layers)):
        layer_prefix = f"layers.{layer_idx}"
        tp_plan[f"{layer_prefix}.MHA.Q_proj"] = ColwiseParallel()
        tp_plan[f"{layer_prefix}.MHA.K_proj"] = ColwiseParallel()
        tp_plan[f"{layer_prefix}.MHA.V_proj"] = ColwiseParallel()
        tp_plan[f"{layer_prefix}.MHA.O_proj"] = RowwiseParallel()
        tp_plan[f"{layer_prefix}.SwiGLU.w1"] = ColwiseParallel()
        tp_plan[f"{layer_prefix}.SwiGLU.w3"] = ColwiseParallel()
        tp_plan[f"{layer_prefix}.SwiGLU.w2"] = RowwiseParallel()
    return parallelize_module(model, tp_mesh, tp_plan)


def apply_tensor_parallelism(model: torch.nn.Module, runtime: DistributedRuntime) -> torch.nn.Module:
    try:
        from torch.distributed.device_mesh import init_device_mesh
    except ImportError as exc:
        raise RuntimeError("Tensor parallel mode requires torch.distributed.device_mesh support.") from exc

    tp_mesh = init_device_mesh("cuda", (runtime.tp_size,), mesh_dim_names=("tp",))
    return apply_tensor_parallelism_plan(model, tp_mesh)


def apply_fully_sharded_data_parallelism(model: torch.nn.Module, dp_mesh: Any) -> torch.nn.Module:
    if not isinstance(model, TransformerLM):
        raise TypeError(f"FSDP mode currently expects TransformerLM, got {type(model).__name__}.")
    if len(model.layers) == 0:
        raise ValueError("FSDP mode requires at least one Transformer layer.")
    try:
        # from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed.fsdp import fully_shard
    except ImportError as exc:
        raise RuntimeError("FSDP2 mode requires torch.distributed._composable.fsdp support.") from exc

    for layer in model.layers:
        fully_shard(layer, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)
    return model


def apply_fsdp(model: torch.nn.Module, runtime: DistributedRuntime) -> torch.nn.Module:
    try:
        from torch.distributed.device_mesh import init_device_mesh
    except ImportError as exc:
        raise RuntimeError("FSDP mode requires torch.distributed.device_mesh support.") from exc

    dp_mesh = init_device_mesh("cuda", (runtime.dp_size,), mesh_dim_names=("dp",))
    return apply_fully_sharded_data_parallelism(model, dp_mesh)


def apply_fsdp_tensor_parallelism(model: torch.nn.Module, runtime: DistributedRuntime) -> torch.nn.Module:
    try:
        from torch.distributed.device_mesh import init_device_mesh
    except ImportError as exc:
        raise RuntimeError("FSDP+TP mode requires torch.distributed.device_mesh support.") from exc

    mesh = init_device_mesh("cuda", (runtime.dp_size, runtime.tp_size), mesh_dim_names=("dp", "tp"))
    model = apply_tensor_parallelism_plan(model, mesh["tp"])
    return apply_fully_sharded_data_parallelism(model, mesh["dp"])


def configure_parallelism(model: torch.nn.Module, runtime: DistributedRuntime) -> torch.nn.Module:
    if not runtime.enabled:
        return model
    if runtime.mode == "dp":
        return DDP(model, device_ids=[runtime.local_rank], output_device=runtime.local_rank)
    if runtime.mode == "fsdp":
        return apply_fsdp(model, runtime)
    if runtime.mode == "tp":
        return apply_tensor_parallelism(model, runtime)
    if runtime.mode == "fsdp_tp":
        return apply_fsdp_tensor_parallelism(model, runtime)
    raise ValueError(f"Unsupported distributed mode in configure_parallelism: {runtime.mode}")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DDP):
        return model.module
    return model


def rank_local_checkpoint_path(checkpoint_path: Path, runtime: DistributedRuntime) -> Path:
    if runtime.mode not in {"tp", "fsdp", "fsdp_tp"}:
        return checkpoint_path
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_rank{runtime.rank}{checkpoint_path.suffix}")


def save_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out_path: Path,
    runtime: DistributedRuntime,
) -> None:
    if runtime.mode in {"tp", "fsdp", "fsdp_tp"}:
        shard_out_path = rank_local_checkpoint_path(out_path, runtime)
        checkpoint_obj = {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "distributed": {
                "mode": runtime.mode,
                "world_size": runtime.world_size,
                "dp_size": runtime.dp_size,
                "tp_size": runtime.tp_size,
                "rank": runtime.rank,
            },
        }
        torch.save(checkpoint_obj, shard_out_path)
        if runtime.enabled:
            dist.barrier()
        return

    if runtime.is_main:
        torch.save(
            {
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
            },
            out_path,
        )


def load_training_checkpoint(
    src_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    runtime: DistributedRuntime,
) -> int:
    if runtime.mode in {"tp", "fsdp", "fsdp_tp"}:
        shard_src_path = rank_local_checkpoint_path(Path(src_path), runtime)
        checkpoint_obj = torch.load(shard_src_path, map_location="cpu")
        distributed_meta = checkpoint_obj.get("distributed", {})
        expected = {
            "mode": runtime.mode,
            "world_size": runtime.world_size,
            "dp_size": runtime.dp_size,
            "tp_size": runtime.tp_size,
        }
        found = {
            "mode": distributed_meta.get("mode"),
            "world_size": distributed_meta.get("world_size"),
            "dp_size": distributed_meta.get("dp_size"),
            "tp_size": distributed_meta.get("tp_size"),
        }
        if found != expected:
            raise ValueError(f"Checkpoint topology mismatch: expected={expected}, found={found}.")
        unwrap_model(model).load_state_dict(checkpoint_obj["model"])
        optimizer.load_state_dict(checkpoint_obj["optimizer"])
        if runtime.enabled:
            dist.barrier()
        return checkpoint_obj["iteration"]

    checkpoint_obj = torch.load(src_path, map_location="cpu")
    unwrap_model(model).load_state_dict(checkpoint_obj["model"])
    optimizer.load_state_dict(checkpoint_obj["optimizer"])
    return checkpoint_obj["iteration"]


def reduce_scalar_mean(value: float, device: str, runtime: DistributedRuntime) -> float:
    if not runtime.enabled:
        return value
    reduced = torch.tensor(value, dtype=torch.float32, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= runtime.world_size
    return reduced.item()


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_dataset: MemMapDataset,
    val_dataset: Optional[MemMapDataset],
    config: dict,
    device: str,
    eval_iters: int,
    runtime: DistributedRuntime,
) -> dict:
    """
    Estimate loss on training and validation sets.

    Args:
        model: The model to evaluate
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        config: Configuration dictionary
        device: Device to run evaluation on
        eval_iters: Number of iterations to evaluate

    Returns:
        Dictionary with 'train' and 'val' losses
    """
    model.eval()
    losses = {}

    batch_size = config["eval"].get("eval_batch_size", config["training"]["batch_size"])
    if runtime.enabled and is_data_parallel_mode(runtime.mode):
        dp_world_size = data_parallel_world_size(runtime)
        if batch_size % dp_world_size != 0:
            raise ValueError(
                "eval batch size must be divisible by data-parallel world size "
                f"(eval_batch_size={batch_size}, dp_world_size={dp_world_size}, mode={runtime.mode})."
            )
        batch_size = batch_size // dp_world_size
    context_length = config["model"]["context_length"]

    # Evaluate on training set
    train_losses = []
    for _ in range(eval_iters):
        inputs, targets = get_batch_from_memmap(train_dataset, batch_size, context_length, device)
        logits = model(inputs)
        # Reshape for cross-entropy: (B, S, V) -> (B*S, V) and (B, S) -> (B*S,)
        B, S, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * S, V), targets.reshape(B * S))
        train_losses.append(loss.item())
    losses["train"] = reduce_scalar_mean(float(np.mean(train_losses)), device, runtime)

    # Evaluate on validation set if available
    if val_dataset is not None:
        val_losses = []
        for _ in range(eval_iters):
            inputs, targets = get_batch_from_memmap(val_dataset, batch_size, context_length, device)
            logits = model(inputs)
            B, S, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * S, V), targets.reshape(B * S))
            val_losses.append(loss.item())
        losses["val"] = reduce_scalar_mean(float(np.mean(val_losses)), device, runtime)

    model.train()
    return losses


def train(config_path: str, use_wandb: bool = False, run_name: Optional[str] = None):
    """
    Main training loop.

    Args:
        config_path: Path to YAML configuration file
        use_wandb: Whether to use Weights & Biases for logging
        run_name: Optional name for this training run
    """
    # Load configuration and setup distributed runtime first (for rank-local device/seed selection)
    config = load_config(config_path)
    runtime = setup_distributed_runtime(config)
    validate_tensor_parallel_config(config, runtime)

    try:
        if runtime.is_main:
            print(f"Loaded configuration from {config_path}")
            if runtime.enabled:
                if runtime.mode == "dp":
                    mode_label = "dp (DistributedDataParallel)"
                elif runtime.mode == "fsdp":
                    mode_label = "fsdp (Fully Sharded Data Parallel, FSDP2)"
                elif runtime.mode == "tp":
                    mode_label = "tp (Tensor Parallel)"
                elif runtime.mode == "fsdp_tp":
                    mode_label = f"fsdp_tp (FSDP2 + Tensor Parallel, dp={runtime.dp_size}, tp={runtime.tp_size})"
                else:
                    mode_label = runtime.mode
                print(f"Distributed mode: {mode_label} over {runtime.world_size} ranks")

        # Setup logging
        wandb_logger = setup_logging(use_wandb and runtime.is_main, config, run_name)

        # Set random seed for reproducibility.
        # DP/FSDP/FSDP+TP use DP-rank-offset streams; TP keeps the same stream on each rank.
        seed = config["training"].get("seed", 42)
        rank_seed_offset = data_parallel_rank(runtime) if runtime.enabled and is_data_parallel_mode(runtime.mode) else 0
        torch.manual_seed(seed + rank_seed_offset)
        np.random.seed(seed + rank_seed_offset)

        # Setup device
        if runtime.enabled:
            device = f"cuda:{runtime.local_rank}"
        else:
            device = config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
        if runtime.is_main:
            print(f"Using device: {device}")

        # Load datasets with memory mapping
        if runtime.is_main:
            print("\nLoading datasets...")
        train_dataset = MemMapDataset(config["data"]["train_path"], verbose=runtime.is_main)

        val_dataset = None
        if "val_path" in config["data"] and config["data"]["val_path"]:
            val_dataset = MemMapDataset(config["data"]["val_path"], verbose=runtime.is_main)

        # Initialize model
        if runtime.is_main:
            print("\nInitializing model...")
        base_model = TransformerLM(
            d_model=config["model"]["d_model"],
            num_heads=config["model"]["num_heads"],
            d_ff=config["model"]["d_ff"],
            theta=config["model"]["theta"],
            vocab_size=config["model"]["vocab_size"],
            context_length=config["model"]["context_length"],
            num_layers=config["model"]["num_layers"],
        ).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        if runtime.is_main:
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")

        model = configure_parallelism(base_model, runtime)

        # Initialize optimizer
        optimizer_config = config["optimizer"]
        optimizer_type = optimizer_config["type"].lower()
        use_foreach_optimizer_kernels = runtime.mode not in {"tp", "fsdp_tp"}
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=optimizer_config["lr"],
                betas=tuple(optimizer_config["betas"]),
                weight_decay=optimizer_config["weight_decay"],
                eps=optimizer_config["eps"],
                foreach=use_foreach_optimizer_kernels,
            )
        elif optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=optimizer_config["lr"],
                foreach=use_foreach_optimizer_kernels,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")

        # Load checkpoint if resuming
        start_iter = 0
        checkpoint_config = config.get("checkpoint", {})
        if checkpoint_config.get("resume_from"):
            if runtime.is_main:
                print(f"\nResuming from checkpoint: {checkpoint_config['resume_from']}")
            start_iter = load_training_checkpoint(checkpoint_config["resume_from"], model, optimizer, runtime)
            if runtime.is_main:
                print(f"Resumed from iteration {start_iter}")

        # Create checkpoint directory
        checkpoint_dir = Path(checkpoint_config.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training configuration
        training_config = config["training"]
        global_batch_size = training_config["batch_size"]
        if runtime.enabled and is_data_parallel_mode(runtime.mode):
            dp_world_size = data_parallel_world_size(runtime)
            if global_batch_size % dp_world_size != 0:
                raise ValueError(
                    "training batch size must be divisible by data-parallel world size "
                    f"(batch_size={global_batch_size}, dp_world_size={dp_world_size}, mode={runtime.mode})."
                )
            local_batch_size = global_batch_size // dp_world_size
        else:
            local_batch_size = global_batch_size
        context_length = config["model"]["context_length"]
        max_iters = training_config["max_iters"]
        eval_interval = training_config.get("eval_interval", 500)
        log_interval = training_config.get("log_interval", 100)
        gradient_clip = training_config.get("gradient_clip", 1.0)

        # Learning rate schedule configuration
        lr_schedule_config = config.get("lr_schedule", {})
        use_lr_schedule = "warmup_iters" in lr_schedule_config

        # Training loop
        if runtime.is_main:
            print("\n" + "=" * 80)
            print("Starting training")
            print("=" * 80)

        model.train()
        start_time = time.time()

        for iter_num in range(start_iter, max_iters):
            # Update learning rate according to schedule
            if use_lr_schedule:
                lr = get_lr_schedule(
                    iter_num,
                    lr_schedule_config["max_learning_rate"],
                    lr_schedule_config["min_learning_rate"],
                    lr_schedule_config["warmup_iters"],
                    lr_schedule_config["cosine_cycle_iters"],
                )
                # Update learning rate in optimizer
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                lr = optimizer_config["lr"]

            # Sample batch
            inputs, targets = get_batch_from_memmap(train_dataset, local_batch_size, context_length, device)

            # Forward pass
            logits = model(inputs)

            # Compute loss
            B, S, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * S, V), targets.reshape(B * S))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                if runtime.mode == "fsdp_tp":
                    # Cross-mesh norm aggregation is not supported for mixed DP and DPxTP DTensors.
                    pass
                elif runtime.mode == "tp":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip), foreach=False)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))

            # Optimizer step
            optimizer.step()

            # Logging
            if iter_num % log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = (iter_num - start_iter + 1) * global_batch_size * context_length / elapsed
                mean_loss = reduce_scalar_mean(loss.item(), device, runtime)

                if runtime.is_main:
                    print(
                        f"Iter {iter_num:6d} | Loss: {mean_loss:.4f} | LR: {lr:.2e} | Tokens/sec: {tokens_per_sec:,.0f}"
                    )

                    if wandb_logger:
                        wandb_logger.log(
                            {
                                "train/loss": mean_loss,
                                "train/lr": lr,
                                "train/tokens_per_sec": tokens_per_sec,
                                "iteration": iter_num,
                            }
                        )

            # Evaluation
            if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
                if runtime.is_main:
                    print("\n" + "-" * 80)
                    print(f"Evaluating at iteration {iter_num}...")

                eval_iters = config["eval"].get("eval_iters", 100)
                losses = estimate_loss(model, train_dataset, val_dataset, config, device, eval_iters, runtime)

                if runtime.is_main:
                    print(f"Train Loss: {losses['train']:.4f}")
                    if "val" in losses:
                        print(f"Val Loss:   {losses['val']:.4f}")
                    print("-" * 80 + "\n")

                    if wandb_logger:
                        log_dict = {"eval/train_loss": losses["train"], "iteration": iter_num}
                        if "val" in losses:
                            log_dict["eval/val_loss"] = losses["val"]
                        wandb_logger.log(log_dict)

            # Save checkpoint
            save_interval = checkpoint_config.get("save_interval", 5000)
            should_save = iter_num > 0 and (iter_num % save_interval == 0 or iter_num == max_iters - 1)
            if should_save:
                checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iter_num}.pt"
                if runtime.is_main:
                    if runtime.mode in {"tp", "fsdp", "fsdp_tp"}:
                        rank0_path = rank_local_checkpoint_path(checkpoint_path, runtime)
                        print(
                            f"Saving {runtime.mode.upper()} checkpoint shards (rank-local files), rank0 path: {rank0_path}"
                        )
            if should_save:
                checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iter_num}.pt"
                if runtime.is_main:
                    if runtime.mode in {"tp", "fsdp", "fsdp_tp"}:
                        rank0_path = rank_local_checkpoint_path(checkpoint_path, runtime)
                        print(
                            f"Saving {runtime.mode.upper()} checkpoint shards (rank-local files), rank0 path: {rank0_path}"
                        )
                    else:
                        print(f"Saving checkpoint to {checkpoint_path}")
                save_training_checkpoint(model, optimizer, iter_num, checkpoint_path, runtime)

                # Also save a "latest" checkpoint
                latest_path = checkpoint_dir / "checkpoint_latest.pt"
                save_training_checkpoint(model, optimizer, iter_num, latest_path, runtime)

        if runtime.is_main:
            print("\n" + "=" * 80)
            print("Training complete!")
            print("=" * 80)

            # Final checkpoint
            final_path = checkpoint_dir / "checkpoint_final.pt"
            if runtime.mode in {"tp", "fsdp", "fsdp_tp"}:
                rank0_final_path = rank_local_checkpoint_path(final_path, runtime)
                print(
                    f"\nSaving final {runtime.mode.upper()} checkpoint shards (rank-local files), "
                    f"rank0 path: {rank0_final_path}"
                )
            else:
                print(f"\nSaving final checkpoint to {final_path}")
        final_path = checkpoint_dir / "checkpoint_final.pt"
        save_training_checkpoint(model, optimizer, max_iters - 1, final_path, runtime)

        if runtime.is_main:
            if wandb_logger:
                wandb_logger.finish()
    finally:
        cleanup_distributed_runtime(runtime)


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    parser.add_argument(
        "--config", type=str, default="pytorch_impl/config.yaml", help="Path to YAML configuration file"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this training run (for W&B)")

    args = parser.parse_args()

    train(args.config, use_wandb=args.wandb, run_name=args.run_name)


if __name__ == "__main__":
    main()
