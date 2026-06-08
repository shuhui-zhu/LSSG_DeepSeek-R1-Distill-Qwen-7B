#!/bin/bash
#SBATCH --job-name=lssg_3-train
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module load apptainer

CONTAINER_IMG="${CONTAINER_IMG:-./container/image.sif}"
if [[ ! -f "$CONTAINER_IMG" ]]; then
    echo "ERROR: container image not found at '$CONTAINER_IMG'." >&2
    echo "Build it first, e.g.:" >&2
    echo "  apptainer build $CONTAINER_IMG container/setup_env.def" >&2
    exit 1
fi

MODEL=ckpts/lssg2-DeepSeek-R1-Distill-Qwen-7B
OUTPUT_DIR=ckpts/lssg3-DeepSeek-R1-Distill-Qwen-7B
PREFIX=train_lssg_data_im_deepseekQwen7B
TRAIN_DATA="./data/${PREFIX}.json"

if [[ ! -d "$MODEL" ]]; then
    echo "ERROR: SFT checkpoint '$MODEL' not found." >&2
    echo "Run Step 1 first (sbatch sft.sh) and wait for it to finish." >&2
    exit 1
fi
if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "ERROR: training data '$TRAIN_DATA' not found." >&2
    echo "Run Steps 2 and 3 first (play_game.sh, assign_rewards.sh)." >&2
    exit 1
fi

PROJECT_DIR="$(pwd)"
TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_cache"
mkdir -p "$TRITON_CACHE_DIR" logs ckpts

APPTAINER_ENV=(
    --env DS_SKIP_CUDA_CHECK=1
    --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
    --env TORCH_CUDA_ARCH_LIST="9.0"
    --env MAX_JOBS=4
    --env OMP_NUM_THREADS=16
    --env PYTHONPATH="$PROJECT_DIR"
)

APPTAINER_BINDS=( --bind "$PROJECT_DIR:$PROJECT_DIR" )
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    APPTAINER_BINDS+=( --bind "$SLURM_TMPDIR:$SLURM_TMPDIR" )
fi

apptainer exec --nv \
    --pwd "$PROJECT_DIR" \
    "${APPTAINER_BINDS[@]}" \
    "${APPTAINER_ENV[@]}" \
    "$CONTAINER_IMG" \
    torchrun --nproc_per_node=4 --master_port=6002 train.py \
        --output_dir "$OUTPUT_DIR" \
        --model_name_or_path "$MODEL" \
        --ref_model_name_or_path "$MODEL" \
        --lm_kl_coeff 0.2 \
        --lm_sft_coeff 0.5 \
        --train_method "OfflinePO" \
        --train_data_path "$TRAIN_DATA" \
        --remove_unused_columns False \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --padding_side "right" \
        --truncation_side "left" \
        --max_length 2048 \
        --save_strategy epoch \
        --learning_rate 2e-6 \
        --lr_scheduler_type "cosine" \
        --warmup_ratio 0.03 \
        --logging_steps 1 \
        --weight_decay 0. \
        --deepspeed "./configs/default_offload_opt_param.json" \
        --gradient_checkpointing True \
        --tf32 True \
        --bf16 True
