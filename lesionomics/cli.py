#!/usr/bin/env python3
"""LesionOmics command-line interface.

A single ``lesionomics`` entry point with one sub-command per tool:

    lesionomics quantify           Quantify lesions from a segmentation
    lesionomics track              Match lesions across two timepoints
    lesionomics compare-splitting  Benchmark the three splitting algorithms
    lesionomics jacobian           Jacobian-determinant change map (needs antspyx)

Run ``lesionomics <command> --help`` for the options of each.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("lesionomics")

SPLIT_METHODS = ["erosion", "watershed", "random_walker"]


# ---------------------------------------------------------------------------
# quantify
# ---------------------------------------------------------------------------

def _add_quantify_parser(sub):
    p = sub.add_parser(
        "quantify",
        help="Quantify shape/intensity/texture features for each lesion.",
        description="Label lesions in a segmentation and write a per-lesion feature CSV.",
    )
    p.add_argument("--seg", default="data/truenet.nii.gz",
                   help="Lesion segmentation / probability map (default: data/truenet.nii.gz)")
    p.add_argument("--modalities", nargs="+", metavar="NAME=PATH",
                   default=["T1=data/T1.nii.gz", "T2=data/T2.nii.gz"],
                   help="Modality images as NAME=PATH pairs (default: T1 and T2 from data/)")
    p.add_argument("--output", default="lesion_measurements.csv",
                   help="Output CSV path (default: lesion_measurements.csv)")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Binarization threshold (default: 0.3)")
    p.add_argument("--split-confluent", action="store_true",
                   help="Attempt to split confluent lesions")
    p.add_argument("--split-method", choices=SPLIT_METHODS, default="erosion",
                   help="Algorithm used when --split-confluent is set (default: erosion)")
    p.add_argument("--nawm-mask", help="NAWM binary mask NIfTI path")
    p.add_argument("--jacobian-map", help="Jacobian determinant map NIfTI path")
    p.add_argument("--prob-map", help="Probabilistic segmentation map (if different from --seg)")
    p.add_argument("--n-shells", type=int, default=5,
                   help="Number of concentric shells for radial profiles (default: 5)")
    p.add_argument("--no-radiomics", action="store_true",
                   help="Skip PyRadiomics texture features")
    p.add_argument("--wavelet", action="store_true", help="Include wavelet filter bank (slow)")
    p.add_argument("--log-filter", action="store_true", help="Include LoG filter bank (slow)")
    p.set_defaults(func=_run_quantify)
    return p


def _run_quantify(args):
    from .io import load_nifti, parse_modalities
    from .label_lesions import label_lesions
    from .quantifying_lesions import quantify_lesions

    seg_img = load_nifti(args.seg, required=True)

    logger.info("Labeling lesions (threshold=%.2f, split_confluent=%s, split_method=%s)",
                args.threshold, args.split_confluent, args.split_method)
    result = label_lesions(
        seg_img,
        threshold=args.threshold,
        split_confluent=args.split_confluent,
        split_method=args.split_method,
    )
    logger.info("Found %d lesion(s) in %d contiguous region(s)",
                result.n_lesions, result.n_confluent_lesions)
    if result.n_lesions == 0:
        logger.warning("No lesions found. Check threshold or input file.")
        return 0

    modalities = parse_modalities(args.modalities)
    if not modalities:
        logger.error("No valid modality images found.")
        return 1

    nawm_mask = load_nifti(args.nawm_mask, required=False) if args.nawm_mask else None
    jacobian_map = load_nifti(args.jacobian_map, required=False) if args.jacobian_map else None
    prob_map = load_nifti(args.prob_map, required=False) if args.prob_map else None

    logger.info("Quantifying (n_shells=%d, radiomics=%s) ...", args.n_shells, not args.no_radiomics)
    df = quantify_lesions(
        labeled_img=result,
        modalities=modalities,
        prob_map=prob_map,
        nawm_mask=nawm_mask,
        jacobian_map=jacobian_map,
        include_radiomics=not args.no_radiomics,
        include_wavelet=args.wavelet,
        include_log=args.log_filter,
        n_shells=args.n_shells,
    )

    df.to_csv(args.output, index=False)
    logger.info("Saved %d lesions × %d features to %s", len(df), len(df.columns), args.output)
    print(df.to_string(max_cols=10, max_rows=20))
    return 0


# ---------------------------------------------------------------------------
# track
# ---------------------------------------------------------------------------

def _add_track_parser(sub):
    p = sub.add_parser(
        "track",
        help="Match lesions between two co-registered timepoints.",
        description=(
            "Label both segmentations and match lesions by IoU, classifying each "
            "as stable / split / merge / new / resolved. Both inputs must already "
            "be in the same voxel space (co-register TP2 to TP1 first)."
        ),
    )
    p.add_argument("--seg-tp1", required=True, help="Baseline segmentation NIfTI")
    p.add_argument("--seg-tp2", required=True,
                   help="Follow-up segmentation NIfTI (already in TP1 voxel space)")
    p.add_argument("--output", default="longitudinal_matches.csv",
                   help="Output CSV path (default: longitudinal_matches.csv)")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Binarization threshold for both timepoints (default: 0.3)")
    p.add_argument("--split-confluent", action="store_true",
                   help="Split confluent lesions before matching")
    p.add_argument("--split-method", choices=SPLIT_METHODS, default="erosion",
                   help="Algorithm used when --split-confluent is set (default: erosion)")
    p.add_argument("--min-iou", type=float, default=0.1,
                   help="Minimum IoU to consider two lesions overlapping (default: 0.1)")
    p.set_defaults(func=_run_track)
    return p


def _run_track(args):
    from .io import load_nifti
    from .label_lesions import label_lesions
    from .tracking import match_lesions_longitudinal

    seg_tp1 = load_nifti(args.seg_tp1, required=True)
    seg_tp2 = load_nifti(args.seg_tp2, required=True)

    kwargs = dict(threshold=args.threshold,
                  split_confluent=args.split_confluent,
                  split_method=args.split_method)
    logger.info("Labeling baseline ...")
    r1 = label_lesions(seg_tp1, **kwargs)
    logger.info("Labeling follow-up ...")
    r2 = label_lesions(seg_tp2, **kwargs)
    logger.info("TP1: %d lesion(s); TP2: %d lesion(s)", r1.n_lesions, r2.n_lesions)

    matches = match_lesions_longitudinal(r1, r2, min_iou=args.min_iou)
    matches.to_csv(args.output, index=False)
    logger.info("Saved %d match row(s) to %s", len(matches), args.output)

    if not matches.empty:
        print("\nMatch summary:")
        print(matches["match_type"].value_counts().to_string())
        print()
        cols = ["tp1_display_id", "tp2_display_id", "iou", "match_type"]
        print(matches[cols].to_string(index=False))
    return 0


# ---------------------------------------------------------------------------
# compare-splitting
# ---------------------------------------------------------------------------

def _add_compare_parser(sub):
    p = sub.add_parser(
        "compare-splitting",
        help="Benchmark all three splitting algorithms side-by-side.",
        description=("Run erosion, watershed, and random_walker on the same input, "
                     "then write comparison spreadsheets and plots."),
    )
    p.add_argument("--seg", default="data/truenet.nii.gz",
                   help="Segmentation / probability map (default: data/truenet.nii.gz)")
    p.add_argument("--modalities", nargs="+", metavar="NAME=PATH",
                   default=["T1=data/T1.nii.gz", "T2=data/T2.nii.gz"],
                   help="Modality images as NAME=PATH pairs")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Binarisation threshold (default: 0.3)")
    p.add_argument("--output-dir", default="splitting_comparison",
                   help="Directory for outputs (default: splitting_comparison/)")
    p.add_argument("--no-quantify", action="store_true",
                   help="Skip full quantification (faster; counts + projections only)")
    p.set_defaults(func=_run_compare)
    return p


def _run_compare(args):
    from .io import parse_modalities
    from .compare import run_comparison, make_synthetic_data

    seg_path = Path(args.seg)
    if seg_path.exists():
        from .io import load_nifti
        seg_img = load_nifti(seg_path, required=True)
        modalities = parse_modalities(args.modalities)
        if not modalities:
            logger.warning("No modalities loaded; disabling quantification.")
            args.no_quantify = True
    else:
        seg_img, modalities = make_synthetic_data()

    run_comparison(
        seg_img=seg_img,
        modalities=modalities,
        threshold=args.threshold,
        output_dir=Path(args.output_dir),
        do_quantify=not args.no_quantify,
    )
    return 0


# ---------------------------------------------------------------------------
# jacobian
# ---------------------------------------------------------------------------

def _add_jacobian_parser(sub):
    p = sub.add_parser(
        "jacobian",
        help="Compute a Jacobian-determinant volume-change map (requires antspyx).",
        description=("Register a follow-up scan to baseline with SyN and write the "
                     "Jacobian determinant (>1 = local expansion, <1 = contraction)."),
    )
    p.add_argument("--t1-fixed", required=True, help="Baseline T1 NIfTI")
    p.add_argument("--t1-moving", required=True, help="Follow-up T1 NIfTI")
    p.add_argument("--t2-fixed", help="Optional baseline T2 (multi-modal registration)")
    p.add_argument("--t2-moving", help="Optional follow-up T2")
    p.add_argument("--voxel-size", type=float, default=1.0,
                   help="Isotropic voxel size for resampling before registration (default: 1.0)")
    p.add_argument("--output", default="jacobian_determinant.nii.gz",
                   help="Output NIfTI path (default: jacobian_determinant.nii.gz)")
    p.set_defaults(func=_run_jacobian)
    return p


def _run_jacobian(args):
    try:
        import ants
    except ImportError:
        logger.error("antspyx is required for the jacobian command. "
                     "Install with: pip install lesionomics[ants]")
        return 1
    from .jacobian_determinant import compute_lesion_jacobian_pipeline

    jac = compute_lesion_jacobian_pipeline(
        t1_fixed=ants.image_read(args.t1_fixed),
        t1_moving=ants.image_read(args.t1_moving),
        t2_fixed=ants.image_read(args.t2_fixed) if args.t2_fixed else None,
        t2_moving=ants.image_read(args.t2_moving) if args.t2_moving else None,
        voxel_size=args.voxel_size,
    )
    jac.to_file(args.output)
    logger.info("Saved Jacobian determinant to %s", args.output)
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lesionomics",
        description="Modality-agnostic lesion quantification toolkit.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose (DEBUG) logging")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    _add_quantify_parser(sub)
    _add_track_parser(sub)
    _add_compare_parser(sub)
    _add_jacobian_parser(sub)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
