# LesionOmics

A modality-agnostic toolkit for quantifying brain lesions in 3D neuroimaging data.
Designed for Multiple Sclerosis research but applicable to any 3D segmented region.

## Features

**Shape** (computed once per lesion, modality-independent):
- Volume (voxels and mm³), bounding box dimensions and fill ratio
- Convex hull volume and convexness ratio
- Surface area (mm²), surface-to-volume ratio, sphericity
- Inertia tensor eigenvalues; elongation, flatness, isotropy (eigenvalue ratios)
- Euler number (topological complexity), inter-lesion nearest-neighbour distance
- Hemisphere (L/R), centre of mass in voxel and mm space

**Intensity** (per modality — any 3D channel accepted):
- Mean, median, mode, min, max, variance, std, skewness, kurtosis, 5th moment
- Percentiles (10th, 25th, 75th, 90th), IQR, coefficient of variation
- Energy, RMS, histogram entropy
- Boundary gradient mean and std (rim-enhancing vs. diffuse pattern)
- NAWM-normalised mean (when a NAWM mask is supplied)
- 4D inertia eigenvalues combining spatial coordinates with per-modality intensity

**Pairwise** (all modality pairs, automatic):
- Mean intensity ratio (e.g. T1/T2, T1/FLAIR)
- Within-lesion cross-modality Pearson correlation

**Texture** (via [PyRadiomics](https://pyradiomics.readthedocs.io/), per modality):
- GLCM, GLRLM, GLSZM, NGTDM feature sets

**Probabilistic** (when a probability map is supplied):
- Fuzzy volume (sum of probabilities), mean/std confidence
- Probability-weighted centroid, probability entropy
- Core volume (prob > 0.9), rim volume (0.3 < prob < 0.7)

## Installation

```bash
pip install -e .
```

> `antspyx` may need to be installed separately — see the
> [ANTsPy installation guide](https://github.com/ANTsX/ANTsPy#installation).

## Quick Start

```python
import nibabel as nib
from src.label_lesions import label_lesions
from src.quantifying_lesions import quantify_lesions

# 1. Create a labeled lesion image from a probability map
prob_img  = nib.load("data/truenet.nii.gz")
labeled_img = label_lesions(prob_img, threshold=0.3)

# 2. Load any co-registered modality images
modalities = {
    "T1": nib.load("data/T1.nii.gz"),
    "T2": nib.load("data/T2.nii.gz"),
}

# 3. Quantify
df = quantify_lesions(
    labeled_img=labeled_img,
    modalities=modalities,
    prob_map=prob_img,      # optional: enables probabilistic features
    nawm_mask=None,         # optional: enables NAWM-normalised intensities
    include_radiomics=True,
)

df.to_csv("lesion_measurements.csv", index=False)
print(f"{len(df)} lesions x {len(df.columns)} features")
```

## Adding New Modalities

Pass any co-registered 3D image — all feature sets are computed automatically
for every channel and every pair of channels:

```python
df = quantify_lesions(
    labeled_img=labeled_img,
    modalities={
        "T1":   t1_img,
        "FLAIR": flair_img,
        "ADC":  adc_img,
        "MTR":  mtr_img,
    },
)
```

## Confluent Lesion Splitting

```python
labeled_img = label_lesions(
    prob_img,
    threshold=0.3,
    split_confluent=True,   # erosion-derived Voronoi splitting
)
```

## Longitudinal Analysis (Jacobian Determinant)

```python
import ants
from src.jacobian_determinant import compute_lesion_jacobian_pipeline

jacobian = compute_lesion_jacobian_pipeline(
    t1_fixed=ants.image_read("T1_tp1.nii.gz"),
    t1_moving=ants.image_read("T1_tp2.nii.gz"),
    t2_fixed=ants.image_read("T2_tp1.nii.gz"),    # optional
    t2_moving=ants.image_read("T2_tp2.nii.gz"),   # optional
)
jacobian.to_file("jacobian_determinant.nii.gz")
```

## Pipeline Overview

```
Input: labeled mask + any 3D modality images
    |
    v
label_lesions()      threshold -> 3D connected components -> optional confluent splitting
    |
    v
quantify_lesions()   shape + intensity + texture + probabilistic features
    |
    v
lesion_measurements.csv
```
