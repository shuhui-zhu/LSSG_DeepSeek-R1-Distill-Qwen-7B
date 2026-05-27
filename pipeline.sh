#!/bin/bash
#SBATCH --job-name=lssg-pipeline
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Runs README steps 2-4 sequentially in one SLURM job:
#   Step 2: bash play_game.sh        (self-play, torchrun on 4 GPUs)
#   Step 3: bash assign_rewards.sh   (data processing, CPU/python)
#   Step 4: bash lssg.sh             (training, torchrun on 4 GPUs)
#
# Submit with: sbatch pipeline.sh
# Prerequisite: Step 1 (sft.sh) has finished successfully, so the checkpoint
#               ./ckpts/imng-DeepSeek-R1-Distill-Qwen-7B exists.

set -euo pipefail

module load apptainer

CONTAINER_IMG="${CONTAINER_IMG:-./container/image.sif}"
if [[ ! -f "$CONTAINER_IMG" ]]; then
    echo "ERROR: container image not found at '$CONTAINER_IMG'." >&2
    echo "Build it first, e.g.:" >&2
    echo "  apptainer build $CONTAINER_IMG container/setup_env.def" >&2
    exit 1
fi

SFT_CKPT="./ckpts/imng-DeepSeek-R1-Distill-Qwen-7B"
if [[ ! -d "$SFT_CKPT" ]]; then
    echo "ERROR: SFT checkpoint '$SFT_CKPT' not found." >&2
    echo "Run Step 1 first (sbatch sft.sh) and wait for it to finish." >&2
    exit 1
fi

PROJECT_DIR="$(pwd)"
TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_cache"
mkdir -p "$TRITON_CACHE_DIR" logs ckpts data/self_play_results

APPTAINER_ENV=(
    --env DS_SKIP_CUDA_CHECK=1
    --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
    --env TORCH_CUDA_ARCH_LIST="9.0"
    --env MAX_JOBS=4
    --env OMP_NUM_THREADS=4
    --env PYTHONPATH="$PROJECT_DIR"
)

APPTAINER_BINDS=( --bind "$PROJECT_DIR:$PROJECT_DIR" )
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    APPTAINER_BINDS+=( --bind "$SLURM_TMPDIR:$SLURM_TMPDIR" )
fi

run_in_container() {
    local step_name="$1"
    shift
    echo "============================================================"
    echo "[pipeline] $(date '+%F %T')  starting: $step_name"
    echo "============================================================"
    apptainer exec --nv \
        --pwd "$PROJECT_DIR" \
        "${APPTAINER_BINDS[@]}" \
        "${APPTAINER_ENV[@]}" \
        "$CONTAINER_IMG" \
        "$@"
    echo "[pipeline] $(date '+%F %T')  finished: $step_name"
}

run_in_container "Step 2 - self-play (play_game.sh)" \
    bash play_game.sh

run_in_container "Step 3 - assign rewards (assign_rewards.sh)" \
    bash assign_rewards.sh

run_in_container "Step 4 - LSSG training (lssg.sh)" \
    bash lssg.sh

echo "[pipeline] $(date '+%F %T')  all steps completed successfully."
