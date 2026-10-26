# LesionOmics on LST-AI output (RIS compute2)

Quantifies every lesion of every subject/session from LST-AI segmentations.

| Script | Purpose |
|---|---|
| `setup_lesionomics_env.bash` | One-time: venv on scratch with this repo (editable) + `picsl-greedy`. Run after `scripts/lst_ai/setup_lst_env.bash`, which installs the Python it reuses. |
| `batch_quantify_lst.py` | Does the work, for all sessions or one (`--task-index`). |
| `submit_lesionomics_batch.bash` | Slurm array: one job per session, then a job that combines the results. |

Expected layout (BIDS-style; sessions optional):

```
RAW_DIR/sub-X/ses-Y/anat/*_T1w.nii.gz, *_FLAIR.nii.gz
SEG_DIR/sub-X/ses-Y/anat/space-flair_seg-lst.nii.gz
```

## Run

```bash
cd $brier/HENRY/LesionOmics
bash scripts/lesionomics/setup_lesionomics_env.bash          # once

RAW=$brier/HENRY/LST-AI/data/TEST_RETEST
SEG=$brier/HENRY/LST-AI/derivatives/TEST_RETEST
OUT=$brier/HENRY/LesionOmics_results/TEST_RETEST

PY=/scratch2/fs1/matthew.brier/$USER/envs/lesionomics_env/bin/python
$PY scripts/lesionomics/batch_quantify_lst.py --raw-dir $RAW --seg-dir $SEG --out-dir $OUT --list

# all sessions as Slurm jobs (extra options are passed through)
bash scripts/lesionomics/submit_lesionomics_batch.bash $RAW $SEG $OUT
bash scripts/lesionomics/submit_lesionomics_batch.bash $RAW $SEG $OUT --subjects sub-100000289
```

## What happens per session

1. The raw FLAIR must be on the segmentation's voxel grid (LST-AI writes its mask onto
   the original FLAIR grid, so it is).
2. The T1 is rigidly registered to the FLAIR with greedy (as LST-AI does internally),
   so FLAIR, T1 and mask share one grid. The log reports how far it had to move the T1;
   a few mm / degrees is normal, much more deserves a look at the saved
   `*_space-flair_T1w.nii.gz` over the FLAIR.
3. Lesions are labeled (`L1`, `L2`, ... largest first) and quantified on FLAIR and T1.

Outputs in `OUT`:

```
sub-X/ses-Y/anat/sub-X_ses-Y_desc-lesions_measurements.csv     one row per lesion
sub-X/ses-Y/anat/sub-X_ses-Y_space-flair_desc-lesions_dseg.nii.gz   lesion label map
sub-X/ses-Y/anat/sub-X_ses-Y_space-flair_T1w.nii.gz            T1 on the FLAIR grid
all_lesions.csv          every lesion, with subject/session columns
sessions_summary.csv     per session: status, lesion count, total volume
logs/
```

Sessions already done are skipped; `--overwrite` redoes them. `--combine-only` rebuilds
the two combined CSVs.

## Options worth knowing

- `--split-confluent [--split-method erosion|watershed|random_walker]` splits merged
  lesions into sub-lesions (`L2A`, `L2B`, ...). Off by default; on test data `watershed`
  turned 19 lesions into 139 pieces, so compare methods (`lesionomics compare-splitting`)
  before relying on it.
- `--radiomics` adds PyRadiomics texture features (install with
  `INSTALL_RADIOMICS=1 bash setup_lesionomics_env.bash`; much slower).
- Longitudinal tracking (`lesionomics track`) needs the sessions registered to each
  other first; this batch keeps every session in its own FLAIR space.
