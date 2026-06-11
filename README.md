# LesionOmics

A modality-agnostic toolkit for quantifying brain lesions in 3D neuroimaging data.
Designed for Multiple Sclerosis research but applicable to any 3D segmented region.

## Features

**Shape** (computed once per lesion, modality-independent):
- Volume (mm³), convex hull volume and solidity ratio
- Surface area (mm²), sphericity, surface curvature statistics
- Inertia-derived elongation, flatness, isotropy
- Euler number (topology), centre of mass in mm space
- Inter-lesion nearest-neighbour distance and neighbour counts within 10/20/30 mm

**Intensity** (per modality — any 3D channel accepted):
- Mean, median, mode, min, max, std, skewness, kurtosis, IQR, entropy, energy
- Boundary gradient mean/std (rim-enhancing vs. diffuse pattern)
- Perilesional ring and concentric-shell radial profiles
- NAWM-normalised mean (when a NAWM mask is supplied)
- 4D inertia eigenvalues combining spatial coordinates with per-modality intensity

**Pairwise** (all modality pairs, automatic):
- Mean intensity ratio, Pearson correlation, mutual information

**Texture** (via [PyRadiomics](https://pyradiomics.readthedocs.io/), per modality):
- GLCM, GLRLM, GLSZM, NGTDM feature sets

**Probabilistic** (when a probability map is supplied):
- Fuzzy volume, mean/std confidence, probability-weighted centroid, entropy
- Core volume (prob > 0.8), rim volume (0.3 < prob ≤ 0.8)

---

## Installation

```bash
pip install -e .                  # runtime only
pip install -e ".[dev]"           # + pytest
pip install -e ".[radiomics]"     # + PyRadiomics texture features
pip install -e ".[ants]"          # + antspyx (Jacobian pipeline)
```

Installing the package adds a single command-line entry point, **`lesionomics`**,
with one sub-command per tool.

> `antspyx` may need to be installed separately — see the
> [ANTsPy installation guide](https://github.com/ANTsX/ANTsPy#installation).

---

## Command-Line Tools

Everything is reachable through `lesionomics <command>`. Run
`lesionomics <command> --help` for the full option list of any command.

```
lesionomics quantify           Quantify lesions from a segmentation → CSV
lesionomics track              Match lesions across two timepoints → CSV
lesionomics compare-splitting  Benchmark the three splitting algorithms → CSVs + plots
lesionomics jacobian           Jacobian-determinant volume-change map (needs antspyx)
```

### 1. `quantify` — features for every lesion

Label lesions in a segmentation and write a per-lesion feature table.

```bash
# Minimal: one segmentation, one modality
lesionomics quantify \
    --seg data/truenet.nii.gz \
    --modalities T1=data/T1.nii.gz \
    --output lesion_measurements.csv

# Multi-modal, split confluent lesions, skip the slow texture features
lesionomics quantify \
    --seg data/truenet.nii.gz \
    --modalities T1=data/T1.nii.gz T2=data/T2.nii.gz FLAIR=data/FLAIR.nii.gz \
    --split-confluent --split-method watershed \
    --no-radiomics \
    --output lesion_measurements.csv

# With optional maps (NAWM normalisation + probability + Jacobian change)
lesionomics quantify \
    --seg data/truenet.nii.gz \
    --modalities T1=data/T1.nii.gz T2=data/T2.nii.gz \
    --prob-map data/truenet.nii.gz \
    --nawm-mask data/nawm.nii.gz \
    --jacobian-map jacobian_determinant.nii.gz
```

### 2. `track` — longitudinal lesion matching

Match lesions between two **co-registered** timepoints (register the follow-up
to baseline first), classifying each as `stable` / `split` / `merge` / `new` /
`resolved`.

```bash
lesionomics track \
    --seg-tp1 data/seg_baseline.nii.gz \
    --seg-tp2 data/seg_followup_in_baseline_space.nii.gz \
    --threshold 0.3 \
    --split-confluent --split-method watershed \
    --min-iou 0.1 \
    --output longitudinal_matches.csv
```

Output columns: `tp1_display_id`, `tp2_display_id`, `tp1_confluent_lesion_id`,
`tp2_confluent_lesion_id`, `sub_label`s, `overlap_vox`, `iou`, `match_type`.

`match_type` values: `stable` (1-to-1), `split` (1 TP1 → ≥2 TP2), `merge` (≥2 TP1 → 1 TP2),
`complex` (≥2 TP1 → ≥2 TP2), `new` (TP2 only), `resolved` (TP1 only).

### 3. `compare-splitting` — benchmark the splitting algorithms

Run all three confluent-splitting methods on the same input and emit comparison
spreadsheets and diagnostic plots. With no real data present it generates a
synthetic scene so you can try it immediately:

```bash
# Try it right now (synthetic data, fast)
lesionomics compare-splitting --no-quantify --output-dir splitting_comparison/

# On real data
lesionomics compare-splitting \
    --seg data/truenet.nii.gz \
    --modalities T1=data/T1.nii.gz T2=data/T2.nii.gz \
    --output-dir splitting_comparison/
```

Produces `comparison_per_lesion.csv`, `comparison_aggregated.csv`,
`comparison_pairwise_diff.csv`, and five PNG plots.

### 4. `jacobian` — volumetric-change map (requires antspyx)

SyN-register a follow-up scan to baseline and write the Jacobian determinant
(>1 = local expansion, <1 = contraction). Feed the result back into
`quantify --jacobian-map`.

```bash
lesionomics jacobian \
    --t1-fixed data/T1_baseline.nii.gz \
    --t1-moving data/T1_followup.nii.gz \
    --t2-fixed data/T2_baseline.nii.gz \
    --t2-moving data/T2_followup.nii.gz \
    --output jacobian_determinant.nii.gz
```

---

## Python API

```python
import nibabel as nib
from lesionomics import label_lesions, quantify_lesions, match_lesions_longitudinal

seg_img = nib.load("data/truenet.nii.gz")

# 1 — label connected lesion components
result = label_lesions(seg_img, threshold=0.3)
print(f"{result.n_lesions} lesion(s) in {result.n_confluent_lesions} contiguous region(s)")
print(result.hierarchy)
#    label  confluent_lesion_id sub_label display_id
#        1                    1      None         L1
#        2                    2      None         L2

# 2 — quantify shape + intensity features
modalities = {"T1": nib.load("data/T1.nii.gz"), "T2": nib.load("data/T2.nii.gz")}
df = quantify_lesions(result, modalities=modalities, include_radiomics=False)
df.to_csv("lesion_measurements.csv", index=False)
```

### Confluent lesion splitting

When lesions merge into a single contiguous region, three algorithms can
separate them:

| `split_method` | Algorithm |
|----------------|-----------|
| `"erosion"` *(default)* | Iterative morphological erosion → Voronoi (EDT nearest-seed) expansion |
| `"watershed"` | Distance-transform watershed; seeds at EDT local maxima. Respects narrow necks. |
| `"random_walker"` | Erosion-derived seeds + random-walk diffusion. Smooth, probabilistic splits. |

```python
result = label_lesions(seg_img, threshold=0.3,
                       split_confluent=True, split_method="watershed")
#    label  confluent_lesion_id sub_label display_id
#        1                    1         A        L1A  ← sub-lesion A of contiguous region 1
#        2                    1         B        L1B  ← sub-lesion B of contiguous region 1
#        3                    2      None         L2  ← isolated lesion (unsplit)
```

Isolated regions are named `L1`, `L2`, …; split sub-lesions become `L2A`,
`L2B`, … (Excel-style, so after `Z` comes `AA`). The `confluent_lesion_img`
NIfTI always holds the pre-split contiguous component labels.

### Longitudinal tracking

```python
r1 = label_lesions(seg_tp1, threshold=0.3, split_confluent=True)
r2 = label_lesions(seg_tp2, threshold=0.3, split_confluent=True)  # same voxel space
matches = match_lesions_longitudinal(r1, r2, min_iou=0.1)
```

Classification is contiguous-lesion-aware: split sub-lesions from the same
region are treated as a unit when detecting split/merge events.

---

## Example Scripts

Runnable end-to-end examples live in `examples/`. All fall back to synthetic
data when no real files are present, so they always run:

| Script | Demonstrates |
|--------|--------------|
| `examples/01_basic_quantification.py` | label → quantify → CSV |
| `examples/02_confluent_lesion_splitting.py` | All three splitting algorithms; hierarchy |
| `examples/03_longitudinal_tracking.py` | stable/new/resolved/split/merge matching |
| `examples/04_full_pipeline.py` | split + quantify + track + volume change |

```bash
python examples/01_basic_quantification.py
```

---

## Pipeline Overview

```
Input: segmentation / probability map + co-registered modality images
           │
           ▼
    label_lesions()          → LabeledResult (label_img, confluent_lesion_img, hierarchy)
      threshold → contiguous connected components
      optional confluent splitting (erosion / watershed / random_walker)
           │
           ▼
    quantify_lesions()       → DataFrame (one row per lesion, includes display_id)
      shape + intensity + texture + probabilistic features
           │
           ▼
    match_lesions_longitudinal()   → DataFrame (match_type: stable/split/merge/new/resolved)
      IoU matching across two timepoints, contiguous-lesion-aware
```

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

CI runs the test suite on Python 3.10–3.12 for every push and pull request.

## License

MIT — see [LICENSE](LICENSE).
