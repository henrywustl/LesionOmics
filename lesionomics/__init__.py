"""LesionOmics — modality-agnostic lesion quantification from NIfTI neuroimages.

Public API
----------
Labeling & splitting:
    label_lesions, LabeledResult, relabel_from_largest_to_smallest,
    erosion_derived_confluent_lesion_splitting,
    watershed_confluent_lesion_splitting,
    random_walker_confluent_lesion_splitting

Quantification:
    quantify_lesions, voxel_volume_from_header

Longitudinal tracking:
    match_lesions_longitudinal

Jacobian (requires antspyx) — import directly from
    lesionomics.jacobian_determinant
"""

from .label_lesions import (
    label_lesions,
    LabeledResult,
    relabel_from_largest_to_smallest,
    erosion_derived_confluent_lesion_splitting,
    watershed_confluent_lesion_splitting,
    random_walker_confluent_lesion_splitting,
)
from .quantifying_lesions import quantify_lesions, voxel_volume_from_header
from .tracking import match_lesions_longitudinal

__version__ = "0.1.0"

__all__ = [
    "label_lesions",
    "LabeledResult",
    "relabel_from_largest_to_smallest",
    "erosion_derived_confluent_lesion_splitting",
    "watershed_confluent_lesion_splitting",
    "random_walker_confluent_lesion_splitting",
    "quantify_lesions",
    "voxel_volume_from_header",
    "match_lesions_longitudinal",
    "__version__",
]
