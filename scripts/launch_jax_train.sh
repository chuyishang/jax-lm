#!/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

set -e

PYTHON_CMD="python"
CONFIG="jax_impl/config.yaml"
USE_WANDB=false
RUN_NAME="test-run_2-14"
COMPUTE_MFU="3.644e14"  # TODO: add a "GPU_TYPE" flag instead of the raw FLOPs, so that we can compute the COMPUTE_MFU instead of hardcoding it

CMD=("$PYTHON_CMD" -m jax_impl.train --config "$CONFIG")

if [ "$USE_WANDB" = true ]; then
    CMD+=(--wandb)
fi
if [ -n "$RUN_NAME" ]; then
    CMD+=(--run-name "$RUN_NAME")
fi
if [ -n "$COMPUTE_MFU" ]; then
    CMD+=(--peak-flops "$COMPUTE_MFU")
fi

exec "${CMD[@]}"
