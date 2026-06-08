# LesionOmics

A modality-agnostic toolkit for quantifying brain lesions in 3D neuroimaging data.
Designed for Multiple Sclerosis research but applicable to any 3D segmented region.

## Features

**Shape** (computed once per lesion, modality-independent):
- Volume (mm³), convex hull volume and convexness ratio
- Surface area (mm²), surface-to-volume ratio, sphericity
- Inertia tensor eigenvalues; elongation, flatness, isotropy
- Euler number, inter-lesion nearest-neighbour distance
- Hemisphere (L/R), centre of mass in mm space

**Intensity** (per modality — any 3D channel accepted):
- Mean, median, mode, min, max, std, skewness, kurtosis, IQR, entropy
- Boundary gradient mean and std (rim-enhancing vs. diffuse pattern)
- NAWM-normalised mean (when a NAWM mask is supplied)
- 4D inertia eigenvalues combining spatial coordinates with per-modality intensity
- Concentric shell profiles (radial intensity gradient)

**Pairwise** (all modality pairs, automatic):
- Mean intensity ratio (e.g. T1/T2, T1/FLAIR)
- Within-lesion cross-modality Pearson correlation

**Texture** (via [PyRadiomics](https://pyradiomics.readthedocs.io/), per modality):
- GLCM, GLRLM, GLSZM, NGTDM feature sets

**Probabilistic** (when a probability map is supplied):
- Fuzzy volume, mean/std confidence, probability entropy
- Core volume (prob > 0.9), rim volume (0.3 < prob < 0.7)

## Installation

```bash
pip install -e .
```

> `antspyx` may need to be installed separately — see the
> [ANTsPy installation guide](https://github.com/ANTsX/ANTsPy#installation).

---

## Quick Start

```python
import nibabel as nib
from src.label_lesions import label_lesions
from src.quantifying_lesions import quantify_lesions

seg_img = nib.load("data/truenet.nii.gz")

# 1 — label connected lesion components
result = label_lesions(seg_img, threshold=0.3)

print(f"{result.n_lesions} lesion(s) in "
      f"{result.n_confluent_lesions} contiguous region(s)")
print(result.hierarchy)
#    label  confluent_lesion_id sub_label display_id
#        1                    1      None         L1
#        2                    2      None         L2
#        3                    3      None         L3

# 2 — quantify shape + intensity features
modalities = {
    "T1": nib.load("data/T1.nii.gz"),
    "T2": nib.load("data/T2.nii.gz"),
}
df = quantify_lesions(result, modalities=modalities, include_radiomics=False)
df.to_csv("lesion_measurements.csv", index=False)
```

---

## Confluent Lesion Splitting

When two or more lesions merge into a single contiguous region LesionOmics can
attempt to separate them. Three algorithms are available:

| `split_method`   | Algorithm |
|------------------|-----------|
| `"erosion"` *(default)* | Iterative morphological erosion → Voronoi (EDT nearest-seed) expansion |
| `"watershed"` | Distance-transform watershed; seeds at EDT local maxima. Respects narrow necks between regions. |
| `"random_walker"` | Erosion-derived seeds + random-walk diffusion boundary placement. Smooth, probabilistic splits. |

```python
result = label_lesions(
    seg_img,
    threshold=0.3,
    split_confluent=True,
    split_method="watershed",   # "erosion" | "watershed" | "random_walker"
)

# hierarchy after splitting:
#    label  confluent_lesion_id sub_label display_id
#        1                    1         A        L1A  ← sub-lesion A of contiguous region 1
#        2                    1         B        L1B  ← sub-lesion B of contiguous region 1
#        3                    2      None         L2  ← isolated lesion (unsplit)
```

Sub-lesion naming: isolated contiguous regions become `L1`, `L2`, …; when a
region is split the sub-lesions are labelled `L2A`, `L2B`, `L2C`, … (Excel-style,
so after `Z` comes `AA`, `AB`, …).

The `confluent_lesion_img` NIfTI always stores the *pre-split* contiguous
component labels, useful for spatial grouping and visualisation.

---

## Longitudinal Lesion Tracking

Both segmentations must be in the same voxel space (register the follow-up to
baseline before calling `label_lesions`).

```python
from src.tracking import match_lesions_longitudinal

result_tp1 = label_lesions(seg_tp1, threshold=0.3, split_confluent=True)
result_tp2 = label_lesions(seg_tp2, threshold=0.3, split_confluent=True)

matches = match_lesions_longitudinal(result_tp1, result_tp2, min_iou=0.1)
matches.to_csv("longitudinal_matches.csv", index=False)
```

Output columns:

| Column | Description |
|--------|-------------|
| `tp1_label` / `tp2_label` | Integer label in each timepoint's `label_img` |
| `tp1_confluent_lesion_id` / `tp2_confluent_lesion_id` | Pre-split contiguous region ID |
| `tp1_sub_label` / `tp2_sub_label` | `None` for isolated lesions, `"A"/"B"/…` for split sub-lesions |
| `tp1_display_id` / `tp2_display_id` | Human-readable ID (e.g. `"L2B"`) |
| `overlap_vox` | Shared voxel count |
| `iou` | Intersection-over-Union |
| `match_type` | `stable` / `split` / `merge` / `new` / `resolved` |

Classification is contiguous-lesion-aware: split sub-lesions from the same
contiguous region are treated as a unit when detecting split/merge events.

---

## Jacobian Determinant (Longitudinal Volume Change)

```python
import ants
from src.jacobian_determinant import compute_lesion_jacobian_pipeline

jacobian = compute_lesion_jacobian_pipeline(
    t1_fixed=ants.image_read("T1_tp1.nii.gz"),
    t1_moving=ants.image_read("T1_tp2.nii.gz"),
)
jacobian.to_file("jacobian_determinant.nii.gz")
```

Pass the Jacobian map to `quantify_lesions()` via `jacobian_map=` to get
per-lesion volumetric change features.

---

## Example Scripts

Runnable examples are in the `examples/` directory. All fall back to synthetic
data when no real files are present, so they always run end-to-end.

| Script | Demonstrates |
|--------|--------------|
| `examples/01_basic_quantification.py` | Minimal label → quantify → CSV workflow |
| `examples/02_confluent_lesion_splitting.py` | All three splitting algorithms; hierarchy inspection |
| `examples/03_longitudinal_tracking.py` | Matching across two timepoints; stable/new/resolved/split/merge |
| `examples/04_full_pipeline.py` | Combined: split + quantify + track + volume change |

```bash
python examples/01_basic_quantification.py
python examples/02_confluent_lesion_splitting.py
python examples/03_longitudinal_tracking.py
python examples/04_full_pipeline.py
```

---

## Command-Line Interface

```bash
python run.py \
    --seg data/truenet.nii.gz \
    --modalities T1=data/T1.nii.gz T2=data/T2.nii.gz \
    --threshold 0.3 \
    --split-confluent \
    --split-method watershed \
    --n-shells 5 \
    --output results.csv
```

All options: `python run.py --help`

---

## Pipeline Overview

```
Input: segmentation / probability map + co-registered modality images
           │
           ▼
    label_lesions()
      threshold → contiguous connected components
      optional confluent splitting (erosion / watershed / random_walker)
      → LabeledResult  (label_img, confluent_lesion_img, hierarchy)
           │
           ▼
    quantify_lesions()
      shape + intensity + texture + probabilistic features
      → DataFrame (one row per lesion, columns include display_id)
           │
           ▼
    match_lesions_longitudinal()     ← two timepoints, same voxel space
      IoU-based matching, contiguous-lesion-aware split/merge detection
      → DataFrame  (match_type: stable/split/merge/new/resolved)
           │
           ▼
    lesion_measurements.csv  +  longitudinal_matches.csv
```
