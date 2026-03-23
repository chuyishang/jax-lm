import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from pytorch_impl import model_basic as model_module


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(use_wandb: bool, config: dict, run_name: str | None = None):
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

    def __init__(self, data_path: str):
        """
        Initialize dataset with memory-mapped file.

        Args:
            data_path: Path to binary file containing tokenized data
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        print(f"Loaded dataset from {data_path} with {len(self.data):,} tokens")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DummyDataset:
    """Random token data for benchmarking. Same interface as MemMapDataset."""

    def __init__(self, vocab_size: int, num_tokens: int = 1_000_000):
        self.data = np.random.randint(0, vocab_size, size=num_tokens, dtype=np.uint16)
        print(f"Created dummy dataset with {len(self.data):,} random tokens (vocab_size={vocab_size})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_batch_from_memmap(
    dataset: MemMapDataset | DummyDataset, batch_size: int, context_length: int, device: str
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
    # Use the model's get_batch function with the memmap data
    return model_module.get_batch(dataset.data, batch_size, context_length, device)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_dataset: MemMapDataset | DummyDataset,
    val_dataset: MemMapDataset | DummyDataset | None,
    config: dict,
    device: str,
    eval_iters: int,
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
    context_length = config["model"]["context_length"]

    # Evaluate on training set
    train_losses = []
    for _ in range(eval_iters):
        inputs, targets = get_batch_from_memmap(train_dataset, batch_size, context_length, device)
        logits = model(inputs)
        # Reshape for cross-entropy: (B, S, V) -> (B*S, V) and (B, S) -> (B*S,)
        B, S, V = logits.shape
        loss = model_module.cross_entropy_loss(logits.reshape(B * S, V), targets.reshape(B * S))
        train_losses.append(loss.item())
    losses["train"] = np.mean(train_losses)

    # Evaluate on validation set if available
    if val_dataset is not None:
        val_losses = []
        for _ in range(eval_iters):
            inputs, targets = get_batch_from_memmap(val_dataset, batch_size, context_length, device)
            logits = model(inputs)
            B, S, V = logits.shape
            loss = model_module.cross_entropy_loss(logits.reshape(B * S, V), targets.reshape(B * S))
            val_losses.append(loss.item())
        losses["val"] = np.mean(val_losses)

    model.train()
    return losses


def train(
    config_path: str,
    use_wandb: bool = False,
    run_name: str | None = None,
    peak_flops: float | None = None,
):
    """
    Main training loop.

    Args:
        config_path: Path to YAML configuration file
        use_wandb: Whether to use Weights & Biases for logging
        run_name: Optional name for this training run
    """
    # Load configuration
    config = load_config(config_path)
    print(f"Loaded configuration from {config_path}")

    # Setup logging
    wandb_logger = setup_logging(use_wandb, config, run_name)

    # Set random seed for reproducibility
    seed = config["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup device
    device = config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    # Initialize model
    print("\nInitializing model...")
    model = model_module.TransformerLM(
        d_model=config["model"]["d_model"],
        num_heads=config["model"]["num_heads"],
        d_ff=config["model"]["d_ff"],
        theta=config["model"]["theta"],
        vocab_size=config["model"]["vocab_size"],
        context_length=config["model"]["context_length"],
        num_layers=config["model"]["num_layers"],
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize optimizer
    optimizer_config = config["optimizer"]
    if optimizer_config["type"].lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config["lr"],
            betas=tuple(optimizer_config["betas"]),
            weight_decay=optimizer_config["weight_decay"],
            eps=optimizer_config["eps"],
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")

    # Learning rate schedule configuration
    lr_schedule_config = config.get("lr_schedule", {})
    use_lr_schedule = all(
        key in lr_schedule_config for key in ("max_learning_rate", "min_learning_rate", "warmup_iters", "cosine_cycle_iters")
    )
    scheduler = None
    if use_lr_schedule:
        scheduler = model_module.build_lr_scheduler(
            optimizer,
            min_learning_rate=lr_schedule_config["min_learning_rate"],
            warmup_iters=lr_schedule_config["warmup_iters"],
            cosine_annealing_iters=lr_schedule_config["cosine_cycle_iters"],
            warmup_start_factor=lr_schedule_config.get("warmup_start_factor", 0.1),
        )

    # Load checkpoint if resuming
    start_iter = 0
    checkpoint_config = config.get("checkpoint", {})
    if checkpoint_config.get("resume_from"):
        print(f"\nResuming from checkpoint: {checkpoint_config['resume_from']}")
        start_iter = model_module.load_checkpoint(checkpoint_config["resume_from"], model, optimizer, scheduler)
        print(f"Resumed from iteration {start_iter}")

    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training configuration
    training_config = config["training"]
    batch_size = training_config["batch_size"]
    context_length = config["model"]["context_length"]
    max_iters = training_config["max_iters"]
    eval_interval = training_config.get("eval_interval", 500)
    log_interval = training_config.get("log_interval", 100)
    gradient_clip = training_config.get("gradient_clip", 1.0)
    perf_warmup_steps = max(int(training_config.get("perf_warmup_steps", 0)), 0)
    perf_measure_start_iter = start_iter + perf_warmup_steps
    is_cuda_device = str(device).startswith("cuda") and torch.cuda.is_available()

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
        if iter_num >= perf_measure_start_iter:
            if is_cuda_device:
                torch.cuda.synchronize()
            step_start = time.perf_counter()

        lr = optimizer.param_groups[0]["lr"]

        # Sample batch
        inputs, targets = get_batch_from_memmap(train_dataset, batch_size, context_length, device)

        # Forward pass
        logits = model(inputs)

        # Compute loss
        B, S, V = logits.shape
        loss = model_module.cross_entropy_loss(logits.reshape(B * S, V), targets.reshape(B * S))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))

        nan_in_grad_state = model_module.nan_in_gradients(model.parameters())
        loss_value = float(loss.item())
        if nan_in_grad_state or math.isnan(loss_value):
            nan_checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iter_num}_nan.pt"
            model_module.save_checkpoint(model, optimizer, iter_num, nan_checkpoint_path, scheduler)
            print(f"Iter {iter_num:6d} | Loss: {loss_value:.4f} | LR: {lr:.2e}")
            print("Iteration contains NaN in gradients or loss. Checkpoint saved.")
            raise ValueError("NaN in gradients or loss")

        # Optimizer step
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if iter_num >= perf_measure_start_iter:
            if is_cuda_device:
                torch.cuda.synchronize()
            window_elapsed += time.perf_counter() - step_start
            window_steps += 1

        # Logging
        if iter_num % log_interval == 0:
            log_dict = {"train/loss": loss_value, "train/lr": lr, "iteration": iter_num}
            if iter_num < perf_measure_start_iter:
                print(
                    f"Iter {iter_num:6d} | Loss: {loss_value:.4f} | LR: {lr:.2e} | "
                    f"Perf warmup ({iter_num + 1}/{perf_measure_start_iter})"
                )
            else:
                tokens_per_sec = window_steps * batch_size * context_length / window_elapsed
                mfu = (tokens_per_sec * 6 * total_params / peak_flops) if peak_flops else None
                mfu_str = f" | MFU: {mfu:.2%}" if mfu is not None else ""
                print(
                    f"Iter {iter_num:6d} | Loss: {loss_value:.4f} | LR: {lr:.2e} | "
                    f"Tokens/sec: {tokens_per_sec:,.0f}{mfu_str}"
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
            losses = estimate_loss(model, train_dataset, val_dataset, config, device, eval_iters)

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
            model_module.save_checkpoint(model, optimizer, iter_num, checkpoint_path, scheduler)

            # Also save a "latest" checkpoint
            latest_path = checkpoint_dir / "checkpoint_latest.pt"
            model_module.save_checkpoint(model, optimizer, iter_num, latest_path, scheduler)

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    # Final checkpoint
    final_path = checkpoint_dir / "checkpoint_final.pt"
    print(f"\nSaving final checkpoint to {final_path}")
    model_module.save_checkpoint(model, optimizer, max_iters - 1, final_path, scheduler)

    if wandb_logger:
        wandb_logger.finish()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train Transformer Language Model")
    parser.add_argument(
        "--config", type=str, default="pytorch_impl/config_basic.yaml", help="Path to YAML configuration file"
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

    train(args.config, use_wandb=args.wandb, run_name=args.run_name, peak_flops=args.peak_flops)


if __name__ == "__main__":
    main()
