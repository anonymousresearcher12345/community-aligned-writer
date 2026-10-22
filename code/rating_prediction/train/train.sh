#!/bin/bash
#SBATCH --job-name=rm_train
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#
# Train the reward model with the exact configuration used for the reported
# model (single GPU; ~17h on one H100 including evaluation).
#
#   RATER_PARAMS=/path/to/raterParams.tsv sbatch code/rating_prediction/train/train.sh
#
# Run from the repository root. DATA_DIR defaults to code/rating_prediction/data
# and OUTPUT_DIR to code/rating_prediction/checkpoints/reward_model.

set -euo pipefail

# Activate your Python environment before launching or submitting this job.
if [ -n "${CA_PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="$CA_PROJECT_ROOT"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root or set CA_PROJECT_ROOT}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)" || exit 1
SCRIPT_DIR="$PROJECT_ROOT/code/rating_prediction/train"
[ -d "$SCRIPT_DIR" ] || { echo "Set CA_PROJECT_ROOT to the repository root." >&2; exit 1; }
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_DIR=${DATA_DIR:-$PROJECT_ROOT/code/rating_prediction/data}  # build_dataset.py + split.py outputs
RATER_PARAMS=${RATER_PARAMS:-}
: "${RATER_PARAMS:?Set RATER_PARAMS to your rater-parameter TSV}"
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/code/rating_prediction/checkpoints/reward_model}

# The custom BT scoring path requires a single, unwrapped (non-DataParallel) model.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
fi

"$PYTHON_BIN" "${SCRIPT_DIR}/train.py" \
  --ratings_split "${DATA_DIR}/ratings_split.parquet" \
  --notes_parquet "${DATA_DIR}/notes.parquet" \
  --rater_params_tsv "${RATER_PARAMS}" \
  --pairwise_train_file "${DATA_DIR}/pairwise_train.parquet" \
  --pairwise_val_file "${DATA_DIR}/pairwise_eval_val.parquet" \
  --pairwise_test_file "${DATA_DIR}/pairwise_eval_test.parquet" \
  --model_name answerdotai/ModernBERT-base \
  --max_len 1024 \
  --backbone_lr 2e-5 \
  --head_lr 1e-4 \
  --rater_lr_mult 5.0 \
  --weight_decay 0.01 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 96 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 96 \
  --warmup_steps 2000 \
  --w_bt 0.2 \
  --bt_batch_size 4 \
  --bt_target_passes 0.75 \
  --eval_steps 6000 \
  --save_only_model \
  --metric_for_best_model ai_binary_logloss \
  --logging_steps 50 \
  --dataloader_num_workers 6 \
  --dataloader_prefetch_factor 2 \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}"
