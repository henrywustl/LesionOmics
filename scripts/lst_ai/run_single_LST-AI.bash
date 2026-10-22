#!/bin/bash
# Defaults are for a CPU job. submit_all_BIDS_LST-AI.bash overrides them on the
# sbatch command line for GPU jobs (MODE=gpu); options given there win over
# these #SBATCH lines.
#SBATCH --job-name=LST_AI
#SBATCH --partition=general
#SBATCH --account=compute2-matthew.brier
#SBATCH --time=04:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=8

# Expects (exported by the submit script): T1FILE FLAIRFILE PID OUTPUT_PATH TEMP_PATH
# Optional:
#   LST_ENV         venv made by setup_lst_env.bash
#   LST_DEVICE      cpu | <gpu index> | auto (default: GPU 0 if one is usable, else cpu)
#   LST_EXTRA_ARGS  extra lst flags, e.g. "--segment_only --probability_map"

# ---------------------------------------------------------------------------
# Everything that would otherwise be written under $HOME is redirected to
# scratch. A full home directory does not just block installs: it kills the
# job before this script runs, because the container runtime cannot write its
# auth token ("No space left on device" under ~/.cache/enroot).
#
# NOTE: the enroot/pyxis variables below only take effect if they are already
# set in the SUBMITTING shell, since the container is imported before this
# script executes. Put them in ~/.bashrc as well -- they are repeated here so
# anything the job itself downloads also stays off home.
# ---------------------------------------------------------------------------
SCRATCH_BASE="${SCRATCH_BASE:-/scratch2/fs1/matthew.brier/$USER}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_BASE/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_BASE/.cache/pip}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH_BASE/.cache/torch}"
export HF_HOME="${HF_HOME:-$SCRATCH_BASE/.cache/huggingface}"
# ENROOT_CACHE_PATH holds downloaded layer tarballs: safe on scratch, and
# moving it is what fixed the original "No space left on device" on the token.
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$SCRATCH_BASE/.cache/enroot}"
# ENROOT_DATA_PATH is different: it holds the EXTRACTED container rootfs, which
# must be executed from. Shared scratch is frequently mounted noexec, and then
# the container dies with "enroot-switchroot: failed to execute: /bin/sh:
# Permission denied". Leave it unset (enroot's node-local default); setting it
# here would be too late anyway, as pyxis imports the container before this runs.
if [ -n "${ENROOT_DATA_PATH:-}" ]; then
    echo "note: ENROOT_DATA_PATH=$ENROOT_DATA_PATH (must be local, not NFS)"
fi
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$SCRATCH_BASE/.cache/apptainer}"
export TMPDIR="${TMPDIR:-$SCRATCH_BASE/tmp}"
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$HF_HOME" "$TMPDIR" 2>/dev/null

# No `module load cuda` and no XLA_FLAGS: LST-AI 2.x runs on PyTorch, whose
# wheel bundles its own CUDA runtime. Both were only needed for TensorFlow (1.x).
module purge
module load nil
module load ris

# ---------------------------------------------------------------------------
# Environment (built once by setup_lst_env.bash)
# ---------------------------------------------------------------------------
LST_ENV="${LST_ENV:-$SCRATCH_BASE/envs/lst_env}"
if [ ! -f "$LST_ENV/bin/activate" ]; then
    echo "ERROR: no venv at $LST_ENV -- run setup_lst_env.bash first" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$LST_ENV/bin/activate"
command -v lst >/dev/null 2>&1 || { echo "ERROR: 'lst' not on PATH after activation" >&2; exit 1; }

# Use only the cores Slurm gave us. Left alone, torch and greedy size their
# thread pools from the whole node and oversubscribe the allocation.
NCPU="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"

# lst defaults to GPU 0 and fails outright without one, so pick explicitly.
LST_DEVICE="${LST_DEVICE:-auto}"
if [ "$LST_DEVICE" = auto ]; then
    if python -c 'import sys, torch; sys.exit(not torch.cuda.is_available())' 2>/dev/null; then
        LST_DEVICE=0
    else
        LST_DEVICE=cpu
    fi
fi

echo "Running LST-AI for $PID"
echo "T1:     $T1FILE"
echo "FLAIR:  $FLAIRFILE"
echo "env:    $LST_ENV   ($(command -v lst))"
echo "device: $LST_DEVICE   threads: $NCPU   node: $(hostname)"
echo "Start:  $(date)"

# Fail early rather than after the allocation is spent
for f in "$T1FILE" "$FLAIRFILE"; do
    [ -s "$f" ] || { echo "ERROR: missing or empty input: $f" >&2; exit 1; }
    [[ $f == *.nii.gz ]] || { echo "ERROR: lst needs .nii.gz input: $f" >&2; exit 1; }
done
mkdir -p "$OUTPUT_PATH" "$TEMP_PATH"

LST_ARGS=(--t1 "$T1FILE" --flair "$FLAIRFILE" --output "$OUTPUT_PATH" --temp "$TEMP_PATH"
          --device "$LST_DEVICE" --threads "$NCPU")
# shellcheck disable=SC2206
[ -n "${LST_EXTRA_ARGS:-}" ] && LST_ARGS+=($LST_EXTRA_ARGS)

lst "${LST_ARGS[@]}"
rc=$?

# A zero-byte or absent segmentation should not look like success to the
# submit script's "already processed" check on the next pass.
seg="$OUTPUT_PATH/space-flair_seg-lst.nii.gz"
if [ "$rc" -ne 0 ]; then
    echo "ERROR: lst exited $rc" >&2
elif [ ! -s "$seg" ]; then
    echo "ERROR: lst exited 0 but did not write $seg" >&2
    rc=1
else
    echo "wrote: $seg"
fi

echo "Finished: $(date)  (exit $rc, ${SECONDS}s)"
exit $rc
