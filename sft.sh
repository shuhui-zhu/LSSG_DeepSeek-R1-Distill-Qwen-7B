#!/bin/bash
#SBATCH --job-name=lssg-sft
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=1-23:59:59
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Run training inside the Apptainer container built from
# container/setup_env.def (CUDA 11.8 + PyTorch 2.7.1 + DeepSpeed 0.18.6).
#
# Build the image once with:
#   apptainer build ./container/image.sif ./container/setup_env.def
# ---------------------------------------------------------------------------

CONTAINER_IMG="${CONTAINER_IMG:-./container/image.sif}"
if [[ ! -f "$CONTAINER_IMG" ]]; then
    echo "ERROR: container image not found at '$CONTAINER_IMG'." >&2
    echo "Build it first, e.g.:" >&2
    echo "  apptainer build $CONTAINER_IMG container/setup_env.def" >&2
    exit 1
fi

PROJECT_DIR="$(pwd)"
TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_cache"
mkdir -p "$TRITON_CACHE_DIR" logs ckpts

MODEL=./DeepSeek-R1-Distill-Qwen-7B

# Env forwarded into the container. The image already provides CUDA at
# /usr/local/cuda, so we don't need to set CUDA_HOME from the host.
APPTAINER_ENV=(
    --env DS_SKIP_CUDA_CHECK=1
    --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
    --env TORCH_CUDA_ARCH_LIST="9.0"
    --env MAX_JOBS=4
    --env OMP_NUM_THREADS=4
)

# Bind project tree (so relative paths still resolve) + SLURM scratch.
APPTAINER_BINDS=( --bind "$PROJECT_DIR:$PROJECT_DIR" )
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    APPTAINER_BINDS+=( --bind "$SLURM_TMPDIR:$SLURM_TMPDIR" )
fi

apptainer exec --nv \
    --pwd "$PROJECT_DIR" \
    "${APPTAINER_BINDS[@]}" \
    "${APPTAINER_ENV[@]}" \
    "$CONTAINER_IMG" \
    torchrun --nproc_per_node=2 --master_port=6001 train.py \
        --output_dir "./ckpts/imng-DeepSeek-R1-Distill-Qwen-7B" \
        --model_name_or_path $MODEL \
        --ref_model_name_or_path $MODEL \
        --lm_kl_coeff 0.1 \
        --entropy_coeff 0.01 \
        --train_method "SFTwithKL" \
        --train_data_path "./data/train_imitation_bargain_merged.json" \
        --remove_unused_columns False \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --padding_side "right" \
        --truncation_side "left" \
        --max_length 2048 \
        --save_strategy epoch \
        --learning_rate 5e-6 \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.03 \
        --logging_steps 1 \
        --weight_decay 0. \
        --deepspeed "./configs/default_offload_opt_param.json" \
        --gradient_checkpointing True \
        --tf32 True \
        --bf16 True