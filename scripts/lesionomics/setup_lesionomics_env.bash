#!/bin/bash
# ---------------------------------------------------------------------------
# One-time setup of the LesionOmics Python environment on scratch.
#
#   bash setup_lesionomics_env.bash
#   INSTALL_RADIOMICS=1 bash setup_lesionomics_env.bash   # + PyRadiomics texture features
#
# Installs this repository in editable mode (a `git pull` is picked up without
# reinstalling) plus picsl-greedy for the T1 -> FLAIR registration. Reuses the
# self-contained Python that scripts/lst_ai/setup_lst_env.bash installed on
# scratch, so run that one first. Run on a login node (needs internet).
# ---------------------------------------------------------------------------
set -euo pipefail

SCRATCH_BASE="${SCRATCH_BASE:-/scratch2/fs1/matthew.brier/$USER}"
LO_ENV="${LO_ENV:-$SCRATCH_BASE/envs/lesionomics_env}"
REPO="${REPO:-$(cd "$(dirname "$(realpath "$0")")/../.." && pwd)}"
PY_VERSION="${PY_VERSION:-3.11}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_BASE/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_BASE/.cache/pip}"
export TMPDIR="${TMPDIR:-$SCRATCH_BASE/tmp}"
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$(dirname "$LO_ENV")"

[ -f "$REPO/pyproject.toml" ] || { echo "ERROR: $REPO is not the LesionOmics repo (set REPO=...)" >&2; exit 1; }

# The Python on scratch from setup_lst_env.bash (not a module Python: a venv
# built on one of those ends up with a dangling bin/python).
if [ -z "${PYTHON:-}" ]; then
    PYTHON=$(ls -d "$SCRATCH_BASE"/python/cpython-"$PY_VERSION".*-linux-*/bin/python"$PY_VERSION" \
             2>/dev/null | sort -V | tail -1)
fi
if [ -z "$PYTHON" ] || ! "$PYTHON" -c '' 2>/dev/null; then
    echo "ERROR: no Python $PY_VERSION under $SCRATCH_BASE/python." >&2
    echo "       Run scripts/lst_ai/setup_lst_env.bash first, or set PYTHON=/path/to/python3." >&2
    exit 1
fi
echo "base python: $PYTHON"
echo "repository:  $REPO"

# (Re)create the venv if missing, broken, or built from another Python
base_real=$(readlink -f "$PYTHON")
venv_real=$(readlink -f "$LO_ENV/bin/python" 2>/dev/null || true)
if [ -d "$LO_ENV" ] && { ! "$LO_ENV/bin/python" -c '' 2>/dev/null || [ "$venv_real" != "$base_real" ]; }; then
    echo "recreating broken venv: $LO_ENV"
    "$PYTHON" -m venv --clear "$LO_ENV"
elif [ ! -x "$LO_ENV/bin/python" ]; then
    echo "creating venv: $LO_ENV"
    "$PYTHON" -m venv "$LO_ENV"
fi
VPY="$LO_ENV/bin/python"
"$VPY" -m pip --version >/dev/null 2>&1 || "$VPY" -m ensurepip --upgrade
"$VPY" -m pip install --upgrade pip

EXTRAS="dev"
[ "${INSTALL_RADIOMICS:-0}" = 1 ] && EXTRAS="dev,radiomics"
"$VPY" -m pip install -e "$REPO[$EXTRAS]" picsl-greedy

"$VPY" -c 'import lesionomics, picsl_greedy; print("lesionomics", lesionomics.__version__, "from", lesionomics.__file__)'
echo
echo "Done. Environment: $LO_ENV"
echo "Run the tests with: (cd $REPO && $VPY -m pytest tests -q)"
