#!/bin/bash
# ---------------------------------------------------------------------------
# One-time setup of the LST-AI Python environment on scratch.
#
#   bash setup_lst_env.bash            # torch with CUDA 12.6 (runs on GPU *and* CPU)
#   TORCH_FLAVOR=cpu bash setup_lst_env.bash   # CPU-only torch, ~2.5 GB smaller
#
# Run it on a login node (it needs internet). Compute nodes may not have
# internet, so this also downloads every model weight LST-AI needs at run time
# into the environment, rather than leaving that to the first job.
#
# Safe to re-run: an existing environment is reused and only missing pieces
# are installed / downloaded.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRATCH_BASE="${SCRATCH_BASE:-/scratch2/fs1/matthew.brier/$USER}"
LST_ENV="${LST_ENV:-$SCRATCH_BASE/envs/lst_env}"
TORCH_FLAVOR="${TORCH_FLAVOR:-cu126}"
# Only release candidates of LST-AI 2.x exist on PyPI so far, so a plain
# `pip install lst-ai` finds nothing. A pre-release in the specifier lets pip
# accept it without --pre (which would also pull pre-releases of numpy, torch, ...).
LST_SPEC="${LST_SPEC:-lst-ai>=2.0.0rc1}"

# Keep every cache and temp file off $HOME (same values as ~/.bashrc).
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_BASE/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_BASE/.cache/pip}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH_BASE/.cache/torch}"
export TMPDIR="${TMPDIR:-$SCRATCH_BASE/tmp}"
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TMPDIR" "$(dirname "$LST_ENV")"

# ---------------------------------------------------------------------------
# 1. A Python >= 3.10 to build the venv from. LST-AI 2.x refuses anything older.
# ---------------------------------------------------------------------------
py_ok() { "$1" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; }

PYTHON="${PYTHON:-python3}"
if ! py_ok "$PYTHON"; then
    echo "$PYTHON is older than 3.10; trying the anaconda module's python"
    module load anaconda3/2023.09-0 2>/dev/null || true
    PYTHON=$(command -v python3 || true)
fi
if [ -z "$PYTHON" ] || ! py_ok "$PYTHON"; then
    echo "ERROR: no Python >= 3.10 found. Set PYTHON=/path/to/python3.10+ and re-run." >&2
    exit 1
fi
# A conda env must not be active while the venv is created or used: its
# packages would leak in through PYTHONPATH / site-packages.
if [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-base}" != "base" ]; then
    echo "ERROR: conda env '$CONDA_DEFAULT_ENV' is active; run 'conda deactivate' first." >&2
    exit 1
fi
echo "base python: $PYTHON ($("$PYTHON" --version 2>&1))"

# ---------------------------------------------------------------------------
# 2. Scratch must allow executing files, or every compiled module in the
#    venv fails to load with "failed to map segment from shared object".
# ---------------------------------------------------------------------------
probe="$TMPDIR/.exec_probe_$$"
printf '#!/bin/sh\nexit 0\n' > "$probe" && chmod +x "$probe"
if ! "$probe" 2>/dev/null; then
    rm -f "$probe"
    echo "ERROR: $TMPDIR is mounted noexec; put LST_ENV/TMPDIR somewhere executable." >&2
    exit 1
fi
rm -f "$probe"

# ---------------------------------------------------------------------------
# 3. venv + torch + LST-AI
# ---------------------------------------------------------------------------
if [ ! -f "$LST_ENV/bin/activate" ]; then
    echo "creating venv: $LST_ENV"
    "$PYTHON" -m venv "$LST_ENV"
fi
# shellcheck disable=SC1091
source "$LST_ENV/bin/activate"
python -m pip install --upgrade pip

# torch first, from the index matching the cluster: PyPI's default torch wheel
# targets CUDA 13, which the RIS GPU driver (CUDA 12.x) cannot run, and then
# every GPU job dies with "driver too old". The cu126 wheel bundles its own
# CUDA runtime, so no `module load cuda` is needed, and it runs on CPU too.
pip install --index-url "https://download.pytorch.org/whl/$TORCH_FLAVOR" torch
pip install "$LST_SPEC"

# ---------------------------------------------------------------------------
# 4. Pre-download all model weights into the venv (compute nodes may be offline)
# ---------------------------------------------------------------------------
echo "downloading LST-AI model bundle + atlas"
python - <<'EOF'
from lst_ai.utils import DATA_DIR, download_data
download_data(path=DATA_DIR)
print("LST-AI data:", DATA_DIR)
EOF

echo "downloading HD-BET skull-stripping weights"
python - <<'EOF'
from brainles_hd_bet.utils import maybe_download_parameters, folder_with_parameter_files
for fold in range(5):
    maybe_download_parameters(fold)
print("HD-BET weights:", folder_with_parameter_files)
EOF

echo "downloading FastSurfer checkpoints (lesion annotation)"
python -m lst_ai.fastsurfer --checkpoints

# ---------------------------------------------------------------------------
# 5. Sanity check
# ---------------------------------------------------------------------------
command -v lst >/dev/null || { echo "ERROR: lst not on PATH" >&2; exit 1; }
python - <<'EOF'
import torch
print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}, "
      f"GPU visible here: {torch.cuda.is_available()}")
EOF
du -sh "$LST_ENV"
echo
echo "Done. Environment: $LST_ENV"
echo "Activate with:     source $LST_ENV/bin/activate"
