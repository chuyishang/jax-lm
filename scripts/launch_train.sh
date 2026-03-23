#!/bin/bash
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"

set -e

PYTHON_CMD="uv run python"
CONFIG="pytorch_impl/config.yaml"
USE_WANDB=false
RUN_NAME=""
NPROC_PER_NODE=""

read -r -a PYTHON_CMD_ARR <<< "$PYTHON_CMD"

export CONFIG_PATH="$CONFIG"
read -r DIST_ENABLED DIST_MODE DIST_TP_SIZE < <(
    "${PYTHON_CMD_ARR[@]}" -c 'import os, yaml; cfg = yaml.safe_load(open(os.environ["CONFIG_PATH"])); dist_cfg = cfg.get("distributed", {}); tp_size = dist_cfg.get("tp_size", 0); tp_size = 0 if tp_size is None else int(tp_size); print(int(dist_cfg.get("enabled", False)), dist_cfg.get("mode", "dp"), tp_size)'
)

if [ -z "$NPROC_PER_NODE" ]; then
    if [ "$DIST_ENABLED" = "1" ] && { [ "$DIST_MODE" = "dp" ] || [ "$DIST_MODE" = "fsdp" ] || [ "$DIST_MODE" = "tp" ] || [ "$DIST_MODE" = "fsdp_tp" ]; }; then
        IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
        NPROC_PER_NODE="${#CUDA_DEVICES[@]}"
    else
        NPROC_PER_NODE="1"
    fi
fi

if [ "$DIST_ENABLED" = "1" ]; then
    if [ "$DIST_MODE" != "dp" ] && [ "$DIST_MODE" != "fsdp" ] && [ "$DIST_MODE" != "tp" ] && [ "$DIST_MODE" != "fsdp_tp" ]; then
        echo "Distributed mode must be one of: dp, fsdp, tp, fsdp_tp."
        exit 1
    fi
    if [ "$NPROC_PER_NODE" -lt 2 ]; then
        echo "Distributed mode '$DIST_MODE' requires NPROC_PER_NODE >= 2."
        exit 1
    fi
    if [ "$DIST_MODE" = "fsdp_tp" ]; then
        if [ "$DIST_TP_SIZE" -lt 2 ]; then
            echo "Distributed mode 'fsdp_tp' requires distributed.tp_size >= 2 in config."
            exit 1
        fi
        if [ $((NPROC_PER_NODE % DIST_TP_SIZE)) -ne 0 ]; then
            echo "For fsdp_tp, NPROC_PER_NODE must be divisible by distributed.tp_size (nproc=${NPROC_PER_NODE}, tp_size=${DIST_TP_SIZE})."
            exit 1
        fi
    fi
else
    if [ "$NPROC_PER_NODE" -ne 1 ]; then
        echo "Non-distributed runs require NPROC_PER_NODE=1."
        exit 1
    fi
fi

CMD=("${PYTHON_CMD_ARR[@]}" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" -m pytorch_impl.train --config "$CONFIG")

if [ "$USE_WANDB" = true ]; then
    CMD+=(--wandb)
fi
if [ -n "$RUN_NAME" ]; then
    CMD+=(--run-name "$RUN_NAME")
fi

exec "${CMD[@]}"
