#!/bin/bash
# Run batch_quantify_lst.py as a Slurm array: one task per session, then one
# job that combines the per-session CSVs once every task has finished.
#
#   bash submit_lesionomics_batch.bash RAW_DIR SEG_DIR OUT_DIR [extra batch_quantify_lst.py options]
#
# e.g.
#   bash submit_lesionomics_batch.bash \
#       $brier/HENRY/LST-AI/data/TEST_RETEST \
#       $brier/HENRY/LST-AI/derivatives/TEST_RETEST \
#       $brier/HENRY/LesionOmics_results/TEST_RETEST \
#       --subjects sub-100000289
#
# Optional environment variables:
#   LO_ENV      venv from setup_lesionomics_env.bash
#   PARTITION   Slurm partition (default general-cpu)
#   MAX_JOBS    max array tasks running at once (default 10)
set -euo pipefail

RAW_DIR=$(realpath "${1:?usage: $0 RAW_DIR SEG_DIR OUT_DIR [options]}")
SEG_DIR=$(realpath "${2:?usage: $0 RAW_DIR SEG_DIR OUT_DIR [options]}")
OUT_DIR=$(realpath -m "${3:?usage: $0 RAW_DIR SEG_DIR OUT_DIR [options]}")
shift 3
EXTRA=("$@")

SCRATCH_BASE="${SCRATCH_BASE:-/scratch2/fs1/matthew.brier/$USER}"
LO_ENV="${LO_ENV:-$SCRATCH_BASE/envs/lesionomics_env}"
PARTITION="${PARTITION:-general-cpu}"
ACCOUNT="${ACCOUNT:-compute2-matthew.brier}"
MAX_JOBS="${MAX_JOBS:-10}"
BATCH="$(dirname "$(realpath "$0")")/batch_quantify_lst.py"
VPY="$LO_ENV/bin/python"

[ -x "$VPY" ] || { echo "ERROR: no venv at $LO_ENV -- run setup_lesionomics_env.bash first" >&2; exit 1; }
if ! sinfo -h -p "$PARTITION" -o '%P' 2>/dev/null | grep -q .; then
    echo "ERROR: Slurm partition '$PARTITION' does not exist. Available:" >&2
    sinfo -s >&2
    exit 1
fi

BASE_ARGS=(--raw-dir "$RAW_DIR" --seg-dir "$SEG_DIR" --out-dir "$OUT_DIR")
"$VPY" "$BATCH" "${BASE_ARGS[@]}" "${EXTRA[@]}" --list
N=$("$VPY" "$BATCH" "${BASE_ARGS[@]}" "${EXTRA[@]}" --list | tail -1 | cut -d' ' -f1)
[ "$N" -gt 0 ] 2>/dev/null || { echo "ERROR: no sessions to process" >&2; exit 1; }

LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Commands for the jobs, quoted so paths with odd characters survive --wrap
printf -v TASK_CMD '%q ' "$VPY" "$BATCH" "${BASE_ARGS[@]}" "${EXTRA[@]}"
printf -v COMBINE_CMD '%q ' "$VPY" "$BATCH" "${BASE_ARGS[@]}" --combine-only

COMMON=(--partition="$PARTITION" --account="$ACCOUNT" --cpus-per-task=4 --export=ALL)

array_id=$(sbatch --parsable "${COMMON[@]}" \
    --job-name=lesionomics --array="0-$((N - 1))%$MAX_JOBS" \
    --time=01:00:00 --mem=16G \
    --output="$LOG_DIR/lesionomics_%A_%a.out" \
    --wrap="$TASK_CMD")
combine_id=$(sbatch --parsable "${COMMON[@]}" \
    --job-name=lesionomics_combine --dependency="afterany:$array_id" \
    --time=00:15:00 --mem=4G \
    --output="$LOG_DIR/lesionomics_combine_%j.out" \
    --wrap="$COMBINE_CMD")

echo "submitted array job $array_id ($N session(s), up to $MAX_JOBS at once)"
echo "and combine job $combine_id (runs after the array finishes)"
echo "logs:    $LOG_DIR"
echo "results: $OUT_DIR/all_lesions.csv, $OUT_DIR/sessions_summary.csv"
