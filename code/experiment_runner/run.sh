#!/bin/bash
#SBATCH --job-name=experiment_runner
#SBATCH --output=experiment_runner_%j.out
#SBATCH --error=experiment_runner_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00

# Experiment runner for baseline and CA writers with shared evidence retrieval.
# First cycle is fetch-only; all subsequent cycles run the full RCT pipeline.
#   cycle 1: fetch-only
#   cycle 2+: RCT

set -uo pipefail

# Activate your Python environment before launching or submitting this job.
if [ -n "${CA_PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="$CA_PROJECT_ROOT"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root or set CA_PROJECT_ROOT}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)" || exit 1
SCRIPT_DIR="$PROJECT_ROOT/code/experiment_runner"
[ -d "$SCRIPT_DIR" ] || { echo "Set CA_PROJECT_ROOT to the repository root." >&2; exit 1; }
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$SCRIPT_DIR" || exit 1

NUM_POSTS="${NUM_POSTS:-100}"
GAP="${GAP:-1800}"   # 30 minutes between (last finish -> next start)
EXTRA_ARGS="${EXTRA_ARGS:-}"  # e.g. "--max-process 50"

echo "=== RCT run started $(date) | NUM_POSTS=$NUM_POSTS | GAP=${GAP}s ==="

i=0
while true; do
    i=$((i + 1))
    if [ "$i" -eq 1 ]; then
        MODE="fetch-only"
        CMD=("$PYTHON_BIN" main.py --num-posts "$NUM_POSTS" --fetch-only)
    else
        MODE="RCT"
        CMD=("$PYTHON_BIN" main.py --num-posts "$NUM_POSTS" --n-research 3)
        if [ -n "$EXTRA_ARGS" ]; then
            # shellcheck disable=SC2206
            CMD+=($EXTRA_ARGS)
        fi
    fi

    echo ""
    echo "--- Cycle $i [$MODE] $(date) ---"
    "${CMD[@]}"
    status=$?
    if [ "$status" -eq 42 ]; then
        echo "Baseline record target reached; stopping runner."
        exit 0
    elif [ "$status" -ne 0 ]; then
        echo "Cycle $i ($MODE) failed (exit $status), continuing..."
    fi

    echo ""
    echo "Sleeping ${GAP}s before next cycle... ($(date))"
    sleep "$GAP"
done
