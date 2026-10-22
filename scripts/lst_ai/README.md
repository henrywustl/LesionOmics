# LST-AI on RIS (compute2)

[LST-AI](https://github.com/CompImg/LST-AI) segments MS lesions from a T1w + FLAIR
pair. Its binary mask (`space-flair_seg-lst.nii.gz`) is the `--seg` input for
`lesionomics quantify`.

| Script | Purpose |
|---|---|
| `setup_lst_env.bash` | One-time: builds the venv on scratch and pre-downloads all model weights. Run on a login node. |
| `submit_all_BIDS_LST-AI.bash` | Submits one Slurm job per T1w/FLAIR pair in a BIDS dataset. |
| `run_single_LST-AI.bash` | The per-subject job (called by the submit script). |

```bash
# 1. once
bash setup_lst_env.bash                      # or TORCH_FLAVOR=cpu for a CPU-only env

# 2. a test run on two subjects, CPU queue
SUBJECTS="sub-01 sub-02" bash submit_all_BIDS_LST-AI.bash /path/to/BIDS

# everything, on GPUs
MODE=gpu bash submit_all_BIDS_LST-AI.bash /path/to/BIDS
```

Outputs go to `/scratch2/fs1/matthew.brier/$USER/<BIDS name>/derivatives/LST_AI/`,
logs to `.../logs/LST_AI/`.

Options for the submit script: `MODE=cpu|gpu` (default cpu), `MAX_JOBS`,
`SUBJECTS`, and `LST_EXTRA_ARGS` (passed to `lst`, e.g. `--segment_only` to skip
the FastSurfer anatomical annotation, or `--probability_map` to keep the
probability map in the temp folder).

Notes:
- LST-AI 2.x is PyTorch-based: no TensorFlow, `XLA_FLAGS`, `module load cuda`,
  separate HD-BET or `greedy` install are needed any more.
- Only pre-releases of LST-AI 2.x are on PyPI so far, so plain `pip install lst-ai`
  fails; the setup script installs `lst-ai>=2.0.0rc1`.
- The venv is built on a self-contained Python 3.11 that the setup script installs
  on scratch (`$SCRATCH_BASE/python`, via `uv`), not on a module Python: a venv only
  links to its base interpreter, and module paths under `/opt/thpc` are not the same
  on every node, which left `bin/python3.11` as a dangling link.
- Inputs must be `.nii.gz` and not skull-stripped (otherwise pass `--stripped`).
