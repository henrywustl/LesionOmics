#!/usr/bin/env python3
"""End-to-end lesion quantification pipeline.

Usage
-----
Basic (uses data/ directory defaults):
    python run.py

Custom modalities:
    python run.py --modalities T1=data/T1.nii.gz T2=data/T2.nii.gz

All options:
    python run.py \\
        --seg data/truenet.nii.gz \\
        --modalities T1=data/T1.nii.gz T2=data/T2.nii.gz \\
        --threshold 0.3 \\
        --split-confluent \\
        --nawm-mask nawm.nii.gz \\
        --jacobian-map jac.nii.gz \\
        --prob-map data/truenet.nii.gz \\
        --n-shells 5 \\
        --no-radiomics \\
        --wavelet \\
        --log \\
        --output results.csv
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description="Quantify lesions from NIfTI segmentation.")
    p.add_argument(
        "--seg",
        default="data/truenet.nii.gz",
        help="Lesion segmentation / probability map (default: data/truenet.nii.gz)",
    )
    p.add_argument(
        "--modalities",
        nargs="+",
        metavar="NAME=PATH",
        default=["T1=data/T1.nii.gz", "T2=data/T2.nii.gz"],
        help="Modality images as NAME=PATH pairs (default: T1 and T2 from data/)",
    )
    p.add_argument(
        "--output",
        default="lesion_measurements.csv",
        help="Output CSV path (default: lesion_measurements.csv)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Binarization threshold for lesion labeling (default: 0.3)",
    )
    p.add_argument(
        "--split-confluent",
        action="store_true",
        help="Attempt to split confluent lesions",
    )
    p.add_argument(
        "--split-method",
        choices=["erosion", "watershed", "random_walker"],
        default="erosion",
        help="Algorithm used when --split-confluent is set (default: erosion)",
    )
    p.add_argument("--nawm-mask", help="NAWM binary mask NIfTI path")
    p.add_argument("--jacobian-map", help="Jacobian determinant map NIfTI path")
    p.add_argument(
        "--prob-map",
        help="Probabilistic segmentation map (if different from --seg)",
    )
    p.add_argument(
        "--n-shells",
        type=int,
        default=5,
        help="Number of concentric shells for radial profiles (default: 5)",
    )
    p.add_argument(
        "--no-radiomics",
        action="store_true",
        help="Skip PyRadiomics texture features",
    )
    p.add_argument(
        "--wavelet",
        action="store_true",
        help="Include wavelet filter bank in PyRadiomics (slow)",
    )
    p.add_argument(
        "--log",
        action="store_true",
        help="Include LoG filter bank in PyRadiomics (slow)",
    )
    return p.parse_args()


def main():
    import nibabel as nib
    from src.label_lesions import label_lesions
    from src.quantifying_lesions import quantify_lesions

    args = parse_args()

    # --- Load segmentation and label lesions --------------------------------
    seg_path = Path(args.seg)
    if not seg_path.exists():
        logger.error("Segmentation file not found: %s", seg_path)
        sys.exit(1)

    logger.info("Loading segmentation: %s", seg_path)
    seg_img = nib.load(seg_path)

    logger.info(
        "Labeling lesions (threshold=%.2f, split_confluent=%s, split_method=%s)",
        args.threshold, args.split_confluent, args.split_method,
    )
    labeled_result = label_lesions(
        seg_img,
        threshold=args.threshold,
        split_confluent=args.split_confluent,
        split_method=args.split_method,
    )
    labeled_img = labeled_result.label_img

    import numpy as np
    n_lesions = labeled_result.n_lesions
    logger.info("Found %d lesion(s)", n_lesions)
    if n_lesions == 0:
        logger.warning("No lesions found. Check threshold or input file.")
        sys.exit(0)

    # --- Load modalities ----------------------------------------------------
    modalities = {}
    for item in args.modalities:
        if "=" not in item:
            logger.error("Modality must be NAME=PATH, got: %s", item)
            sys.exit(1)
        name, path = item.split("=", 1)
        p = Path(path)
        if not p.exists():
            logger.warning("Modality file not found, skipping: %s", p)
            continue
        logger.info("Loading modality %s: %s", name, p)
        modalities[name] = nib.load(p)

    if not modalities:
        logger.error("No valid modality images found.")
        sys.exit(1)

    # --- Optional maps ------------------------------------------------------
    nawm_mask = None
    if args.nawm_mask:
        p = Path(args.nawm_mask)
        if p.exists():
            logger.info("Loading NAWM mask: %s", p)
            nawm_mask = nib.load(p)
        else:
            logger.warning("NAWM mask not found: %s", p)

    jacobian_map = None
    if args.jacobian_map:
        p = Path(args.jacobian_map)
        if p.exists():
            logger.info("Loading Jacobian map: %s", p)
            jacobian_map = nib.load(p)
        else:
            logger.warning("Jacobian map not found: %s", p)

    prob_map = None
    if args.prob_map:
        p = Path(args.prob_map)
        if p.exists():
            logger.info("Loading probability map: %s", p)
            prob_map = nib.load(p)
        else:
            logger.warning("Probability map not found: %s", p)

    # --- Quantify -----------------------------------------------------------
    logger.info("Quantifying lesions (n_shells=%d, radiomics=%s) ...", args.n_shells, not args.no_radiomics)
    df = quantify_lesions(
        labeled_img=labeled_result,
        modalities=modalities,
        prob_map=prob_map,
        nawm_mask=nawm_mask,
        jacobian_map=jacobian_map,
        include_radiomics=not args.no_radiomics,
        include_wavelet=args.wavelet,
        include_log=args.log,
        n_shells=args.n_shells,
    )

    # --- Save ---------------------------------------------------------------
    out_path = Path(args.output)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d lesions × %d features to %s", len(df), len(df.columns), out_path)

    # Print summary to stdout
    print(df.to_string(max_cols=10, max_rows=20))


if __name__ == "__main__":
    main()
