import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# For CPU testing with multiple virtual devices (set USE_CPU_SHARDING=1 to enable)
if os.environ.get("USE_CPU_SHARDING"):
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import argparse
import time
from pathlib import Path

import jax
import numpy as np
import wandb
import yaml
from flax import nnx
from jax import Array

from jax_impl import model_basic as model_module
from jax_impl.data import DatasetLike, DummyDataset, MemMapDataset


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(use_wandb: bool, config: dict, run_name: str | None = None):
    """Setup logging with Weights & Biases."""
    if use_wandb:
        wandb.init(project=config.get("wandb_project", "cs336-assignment-1"), name=run_name, config=config)
        return wandb
    return None


def get_batch_from_memmap(
    rngs: nnx.Rngs,
    dataset: DatasetLike,
    batch_size: int,
    context_length: int,
) -> tuple[Array, Array]:
    return model_module.get_batch(rngs, dataset.data, batch_size, context_length)


def estimate_loss(
    rngs: nnx.Rngs,
    model: nnx.Module,
    train_dataset: DatasetLike,
    val_dataset: DatasetLike | None,
    config: dict,
    eval_iters: int,
) -> dict:
    """Estimate loss on training and validation sets.

    Args:
        model: The model to evaluate
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        config: Configuration dictionary
        eval_iters: Number of iterations to evaluate

    Returns:
        Dictionary with 'train' and 'val' losses
    """
    losses = {}

    batch_size = config["eval"].get("eval_batch_size", config["training"]["batch_size"])
    context_length = config["model"]["context_length"]

    # Evaluate on training set
    train_losses = []
    for _ in range(eval_iters):
        inputs, targets = get_batch_from_memmap(rngs, train_dataset, batch_size, context_length)

        logits = model(inputs)
        # Reshape for cross-entropy: (B, S, V) -> (B*S, V) and (B, S) -> (B*S,)
        B, S, V = logits.shape
        loss = model_module.cross_entropy_loss(logits.reshape(B * S, V), targets.reshape(B * S))
        train_losses.append(float(loss))
    losses["train"] = np.mean(train_losses)

    # Evaluate on validation set if available
    if val_dataset is not None:
        val_losses = []
        for _ in range(eval_iters):
            inputs, targets = get_batch_from_memmap(rngs, val_dataset, batch_size, context_length)
            logits = model(inputs)
            B, S, V = logits.shape
            loss = model_module.cross_entropy_loss(logits.reshape(B * S, V), targets.reshape(B * S))
            val_losses.append(float(loss))
        losses["val"] = np.mean(val_losses)

    return losses


def train(
    config_path: str,
    use_wandb: bool = False,
    run_name: str | None = None,
    peak_flops: float | None = None,
) -> None:
    """Main training loop.

    Args:
        config_path: Path to YAML configuration file
        use_wandb: Whether to use Weights & Biases for logging
        run_name: Optional name for this training run
        peak_flops: Peak FLOPS of the hardware (e.g. 312e12 for A100). If provided, MFU is logged.

    Returns:
        None
    """
    # Load configuration
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")

    # Setup logging
    wandb_logger = setup_logging(use_wandb, config, run_name)

    # Set random seed for reproducibility
    seed = config["training"].get("seed", 42)
    rngs = nnx.Rngs(seed)
    np.random.seed(seed)

    # Load datasets
    print("\nLoading datasets...")
    dummy_data = config.get("data", {}).get("dummy_data", False)
    if dummy_data:
        vocab_size = config["model"]["vocab_size"]
        train_dataset = DummyDataset(vocab_size)
        val_dataset = None
    else:
        train_dataset = MemMapDataset(config["data"]["train_path"])
        val_dataset = None
        if "val_path" in config["data"] and config["data"]["val_path"]:
            val_dataset = MemMapDataset(config["data"]["val_path"])

    # Training configuration
    training_config = config["training"]
    batch_size = training_config["batch_size"]
    context_length = config["model"]["context_length"]
    max_iters = training_config["max_iters"]
    eval_interval = training_config.get("eval_interval", 500)
    log_interval = training_config.get("log_interval", 100)
    gradient_clip = training_config.get("gradient_clip", 1.0)

    # Learning rate schedule configuration
    lr_schedule_config = config.get("lr_schedule", {})
    use_lr_schedule = "warmup_iters" in lr_schedule_config
    optimizer_lr_schedule = None
    if use_lr_schedule:
        max_learning_rate = float(lr_schedule_config["max_learning_rate"])
        min_learning_rate = float(lr_schedule_config["min_learning_rate"])
        warmup_iters = int(lr_schedule_config["warmup_iters"])
        cosine_cycle_iters = int(lr_schedule_config["cosine_cycle_iters"])

        optimizer_lr_schedule = model_module.make_lr_schedule(
            max_learning_rate,
            min_learning_rate,
            warmup_iters,
            cosine_cycle_iters,
        )

    # Initialize model and optimizer
    print("\nInitializing model and optimizer...")
    model, optimizer = model_module.create_model_and_optimizer(
        rngs,
        config["model"],
        config["optimizer"],
        gradient_clip=gradient_clip,
        lr_schedule=optimizer_lr_schedule,
    )

    # Count parameters
    total_params = sum(p.size for p in jax.tree.leaves(nnx.state(model)) if isinstance(p, Array))
    print(f"Total parameters: {total_params:,}")

    print("Model sharding is disabled.")

    start_iter = 0
    # Load checkpoint if resuming
    checkpoint_config = config.get("checkpoint", {})
    if checkpoint_config.get("resume_from"):
        print(f"\nResuming from checkpoint: {checkpoint_config['resume_from']}")
        start_iter = model_module.load_checkpoint(checkpoint_config["resume_from"], model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    perf_warmup_steps = max(int(training_config.get("perf_warmup_steps", 0)), 0)
    perf_measure_start_iter = start_iter + perf_warmup_steps

    # Training loop
    print("\n" + "=" * 80)
    print("Starting training")
    print("=" * 80)

    model.train()
    window_elapsed = 0.0
    window_steps = 0
    if perf_measure_start_iter > start_iter:
        print(
            f"Skipping throughput/MFU logging through iteration {perf_measure_start_iter - 1} "
            f"(perf_warmup_steps={perf_warmup_steps})."
        )

    for iter_num in range(start_iter, max_iters):
        # Timing block start
        if iter_num >= perf_measure_start_iter:
            step_start = time.perf_counter()

        if optimizer_lr_schedule is not None:
            lr = float(optimizer_lr_schedule(iter_num))
        else:
            lr = float(config["optimizer"]["lr"])

        # Sample batch
        inputs, targets = get_batch_from_memmap(rngs, train_dataset, batch_size, context_length)

        loss, grad_state, has_nonfinite_update = model_module.train_step(model, optimizer, inputs, targets)

        if has_nonfinite_update:
            model_module.save_checkpoint(
                model,
                optimizer,
                iter_num,
                checkpoint_dir / f"checkpoint_iter_{iter_num}_nonfinite.pt",
                grad_state,
                loss,
            )
            print(f"Iter {iter_num:6d} | Loss: {float(loss):.4f} | LR: {lr:.2e}")
            print(f"Iteration {iter_num} contains a non-finite gradient or loss. Checkpoint saved.")
            raise ValueError("Non-finite gradient or loss")

        # Timing block end
        if iter_num >= perf_measure_start_iter:
            loss.block_until_ready()  # block to prevent async differences
            window_elapsed += time.perf_counter() - step_start
            window_steps += 1

        # Logging
        if iter_num % log_interval == 0:
            log_dict = {"train/loss": float(loss), "train/lr": lr, "iteration": iter_num}

            if iter_num < perf_measure_start_iter:
                print(
                    f"Iter {iter_num:6d} | Loss: {float(loss):.4f} | LR: {lr:.2e} | "
                    f"Perf warmup ({iter_num + 1}/{perf_measure_start_iter})"
                )
            else:
                tokens_per_sec = window_steps * batch_size * context_length / window_elapsed

                # MFU estimate if peak_flops is passed in
                mfu = (tokens_per_sec * 6 * total_params / peak_flops) if peak_flops else None
                mfu_str = f" | MFU: {mfu:.2%}" if mfu is not None else ""
                print(
                    f"Iter {iter_num:6d} | Loss: {float(loss):.4f} | LR: {lr:.2e} | Tokens/sec: {tokens_per_sec:,.0f}{mfu_str}"
                )

                log_dict["train/tokens_per_sec"] = tokens_per_sec
                if mfu is not None:
                    log_dict["train/mfu"] = mfu

                window_elapsed = 0.0
                window_steps = 0

            if wandb_logger:
                wandb_logger.log(log_dict)

        # Evaluation
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            print("\n" + "-" * 80)
            print(f"Evaluating at iteration {iter_num}...")

            eval_iters = config["eval"].get("eval_iters", 100)
            model.eval()
            losses = estimate_loss(
                rngs,
                model,
                train_dataset,
                val_dataset,
                config,
                eval_iters,
            )
            model.train()

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
        if iter_num > 0 and (iter_num % save_interval == 0 or iter_num == max_iters - 1):
            checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iter_num}.pt"
            print(f"Saving checkpoint to {checkpoint_path}")
            model_module.save_checkpoint(model, optimizer, iter_num, checkpoint_path)

            # Also save a "latest" checkpoint
            latest_path = checkpoint_dir / "checkpoint_latest.pt"
            model_module.save_checkpoint(model, optimizer, iter_num, latest_path)

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    # Final checkpoint
    final_path = checkpoint_dir / "checkpoint_final.pt"
    print(f"\nSaving final checkpoint to {final_path}")
    model_module.save_checkpoint(model, optimizer, max_iters - 1, final_path)

    if wandb_logger:
        wandb_logger.finish()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train Transformer Language Model with JAX (basic, no sharding)")
    parser.add_argument(
        "--config", type=str, default="jax_impl/config_basic.yaml", help="Path to YAML configuration file"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--run-name", type=str, default=None, help="Name for this training run (for W&B)")
    parser.add_argument(
        "--peak-flops",
        type=float,
        default=None,
        help="Peak hardware FLOPS for MFU logging (e.g., 312e12 for A100 bf16)",
    )

    args = parser.parse_args()

    train(
        args.config,
        use_wandb=args.wandb,
        run_name=args.run_name,
        peak_flops=args.peak_flops,
    )


if __name__ == "__main__":
    main()
