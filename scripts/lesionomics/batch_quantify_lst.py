#!/usr/bin/env python3
"""Run LesionOmics on LST-AI segmentations for every subject/session.

Inputs (BIDS-style):
    <raw-dir>/sub-X[/ses-Y]/anat/*_T1w.nii.gz, *_FLAIR.nii.gz
    <seg-dir>/sub-X[/ses-Y]/anat/space-flair_seg-lst.nii.gz   (LST-AI output)

For each session:
    1. check the raw FLAIR is on the segmentation's voxel grid (LST-AI writes
       its mask onto the original FLAIR grid, so it should be);
    2. rigidly register the T1 to the FLAIR with greedy (as LST-AI does
       internally) so every image shares that grid;
    3. label the lesions and quantify them (FLAIR + T1 features).

Outputs:
    <out-dir>/sub-X[/ses-Y]/anat/
        sub-X_ses-Y_space-flair_T1w.nii.gz            T1 registered to FLAIR (check it!)
        sub-X_ses-Y_space-flair_desc-lesions_dseg.nii.gz   one integer label per lesion
        sub-X_ses-Y_desc-lesions_measurements.csv     one row per lesion
    <out-dir>/all_lesions.csv        every lesion from every session
    <out-dir>/sessions_summary.csv   lesion count and total volume per session

Examples:
    python batch_quantify_lst.py --raw-dir RAW --seg-dir SEG --out-dir OUT
    python batch_quantify_lst.py ... --subjects sub-100000289      # one subject
    python batch_quantify_lst.py ... --list                         # just list sessions
    python batch_quantify_lst.py ... --task-index 3                 # one session (Slurm array)
    python batch_quantify_lst.py ... --combine-only                 # rebuild combined CSVs
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from lesionomics import label_lesions, quantify_lesions

logger = logging.getLogger("batch_quantify_lst")

SEG_NAME = "space-flair_seg-lst.nii.gz"


# ---------------------------------------------------------------------------
# Finding the sessions
# ---------------------------------------------------------------------------

def find_sessions(raw_dir: Path, seg_dir: Path, subjects: list[str] | None) -> list[dict]:
    """One entry per LST-AI segmentation, with its raw T1/FLAIR resolved."""
    sessions = []
    for seg in sorted(seg_dir.glob(f"sub-*/**/{SEG_NAME}")):
        rel = seg.parent.relative_to(seg_dir)                 # sub-X[/ses-Y]/anat
        sub = next(p for p in rel.parts if p.startswith("sub-"))
        ses = next((p for p in rel.parts if p.startswith("ses-")), None)
        if subjects and sub not in subjects:
            continue
        anat = raw_dir / rel
        flair = sorted(anat.glob("*_FLAIR.nii.gz"))
        t1 = sorted(anat.glob("*_T1w.nii.gz"))
        sessions.append({
            "subject": sub,
            "session": ses,
            "id": f"{sub}_{ses}" if ses else sub,
            "rel": rel,
            "seg": seg,
            # First match, the same choice the LST-AI submit script makes
            "flair": flair[0] if flair else None,
            "t1": t1[0] if t1 else None,
        })
    return sessions


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _harmonized_copy(src: Path, dst: Path) -> Path:
    """Copy with qform = sform = nibabel's affine.

    greedy reads the qform while nibabel (and the rest of this pipeline) reads
    the sform; LST-AI harmonizes them the same way before registering
    (CompImg/LST-AI#44).
    """
    img = nib.load(src)
    code = int(img.header["sform_code"]) or int(img.header["qform_code"]) or 1
    img.set_qform(img.affine, code=code)
    img.set_sform(img.affine, code=code)
    nib.save(img, dst)
    return dst


def register_t1_to_flair(t1: Path, flair: Path, out: Path, threads: int) -> np.ndarray:
    """Rigid (6 DOF) T1 -> FLAIR registration and reslicing with greedy.

    Returns the 4x4 transform greedy found (on top of the header alignment).

    Same metric and schedule as LST-AI's rigid_reg, but initialised from the
    scanner coordinates (-ia-identity) rather than the image centres: T1 and
    FLAIR from one session already share scanner space, so that is the
    better starting point when their fields of view differ.
    """
    from picsl_greedy import Greedy3D

    with tempfile.TemporaryDirectory(prefix="lesionomics_reg_") as tmp:
        tmp = Path(tmp)
        fixed = _harmonized_copy(flair, tmp / "flair.nii.gz")
        moving = _harmonized_copy(t1, tmp / "t1.nii.gz")
        mat = tmp / "t1_to_flair.mat"
        greedy = Greedy3D()
        log = io.StringIO()   # greedy's own output; shown only on failure
        try:
            # The optimizer library also writes straight to the C-level
            # stdout, past `out=`, so capture file descriptor 1 as well.
            with _capture_fd1(tmp / "greedy_stdout.txt") as raw:
                greedy.execute(f"-V 0 -d 3 -a -dof 6 -m NMI -ia-identity -n 100x50x10 "
                               f"-i {fixed} {moving} -o {mat} -threads {threads}",
                               out=log, err=log)
                greedy.execute(f"-V 0 -d 3 -rf {fixed} -rm {moving} {out} -r {mat} "
                               f"-threads {threads}", out=log, err=log)
        except Exception:
            logger.error("greedy output:\n%s\n%s", log.getvalue()[-5000:], raw.read_text()[-5000:])
            raise
        return np.loadtxt(mat)


@contextlib.contextmanager
def _capture_fd1(path: Path):
    """Redirect the process-level stdout (fd 1) into `path` for the block."""
    sys.stdout.flush()
    saved = os.dup(1)
    with open(path, "w") as f:
        os.dup2(f.fileno(), 1)
        try:
            yield path
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------

def _same_grid(a: nib.Nifti1Image, b: nib.Nifti1Image, atol_mm: float = 1e-3) -> bool:
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=atol_mm)


def session_csv(out_dir: Path, s: dict) -> Path:
    return out_dir / s["rel"] / f"{s['id']}_desc-lesions_measurements.csv"


def process_session(s: dict, out_dir: Path, args) -> None:
    """Register, label and quantify one session; writes its CSV."""
    missing = [k for k in ("flair", "t1") if s[k] is None]
    if missing:
        logger.warning("%s: no raw %s found in %s, skipping", s["id"], "/".join(missing),
                       args.raw_dir / s["rel"])
        return

    sess_out = out_dir / s["rel"]
    sess_out.mkdir(parents=True, exist_ok=True)
    csv_path = session_csv(out_dir, s)
    t1_flair = sess_out / f"{s['id']}_space-flair_T1w.nii.gz"
    label_path = sess_out / f"{s['id']}_space-flair_desc-lesions_dseg.nii.gz"

    if csv_path.exists() and not args.overwrite:
        logger.info("%s: already done (%s), skipping; --overwrite to redo", s["id"], csv_path.name)
        return

    t0 = time.time()
    seg_img = nib.load(s["seg"])
    flair_img = nib.load(s["flair"])
    if not _same_grid(seg_img, flair_img):
        raise RuntimeError(
            f"{s['id']}: FLAIR {s['flair'].name} {flair_img.shape} is not on the "
            f"segmentation's grid {seg_img.shape}. Was the segmentation made from this FLAIR?")

    if not t1_flair.exists() or args.overwrite:
        logger.info("%s: registering T1 -> FLAIR", s["id"])
        mat = register_t1_to_flair(s["t1"], s["flair"], t1_flair, args.threads)
        angle = np.degrees(np.arccos(np.clip((np.trace(mat[:3, :3]) - 1) / 2, -1, 1)))
        logger.info("%s: T1 moved %.1f mm / %.1f deg from its header position to fit the FLAIR",
                    s["id"], float(np.linalg.norm(mat[:3, 3])), angle)
    t1_img = nib.load(t1_flair)
    if t1_img.shape[:3] != seg_img.shape[:3]:
        raise RuntimeError(f"{s['id']}: registered T1 {t1_img.shape} does not match "
                           f"the segmentation grid {seg_img.shape}")

    # The LST-AI mask is binary 0/1; 0.5 just binarizes it
    result = label_lesions(seg_img, threshold=0.5,
                           split_confluent=args.split_confluent,
                           split_method=args.split_method)
    nib.save(result.label_img, label_path)
    logger.info("%s: %d lesion(s) in %d contiguous region(s)",
                s["id"], result.n_lesions, result.n_confluent_lesions)

    if result.n_lesions:
        df = quantify_lesions(result, modalities={"FLAIR": flair_img, "T1": t1_img},
                              include_radiomics=args.radiomics, n_shells=args.n_shells)
    else:
        df = pd.DataFrame(columns=["label", "display_id"])
    df.insert(0, "session", s["session"])
    df.insert(0, "subject", s["subject"])
    df.to_csv(csv_path, index=False)
    logger.info("%s: wrote %s (%d lesions x %d columns) in %.0f s",
                s["id"], csv_path.name, len(df), df.shape[1], time.time() - t0)


# ---------------------------------------------------------------------------
# Combining
# ---------------------------------------------------------------------------

def combine(out_dir: Path, sessions: list[dict]) -> None:
    """Stack every per-session CSV into all_lesions.csv + sessions_summary.csv."""
    frames, rows = [], []
    for s in sessions:
        row = {"subject": s["subject"], "session": s["session"], "status": "not processed",
               "n_lesions": np.nan, "n_confluent_lesions": np.nan,
               "total_lesion_volume_mm3": np.nan}
        missing = [k.upper() for k in ("t1", "flair") if s[k] is None]
        if missing:
            row["status"] = f"missing raw {'/'.join(missing)}"
        elif session_csv(out_dir, s).exists():
            df = pd.read_csv(session_csv(out_dir, s))
            frames.append(df)
            lesions = df.dropna(subset=["label"])
            row.update(status="ok", n_lesions=len(lesions),
                       n_confluent_lesions=int(lesions["confluent_lesion_id"].nunique())
                       if "confluent_lesion_id" in lesions else 0,
                       total_lesion_volume_mm3=float(lesions["volume_mm3"].sum())
                       if "volume_mm3" in lesions else 0.0)
        rows.append(row)
    frames = [f for f in frames if len(f)]
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(out_dir / "all_lesions.csv", index=False)
    summary = pd.DataFrame(rows).astype({"n_lesions": "Int64", "n_confluent_lesions": "Int64"})
    summary.to_csv(out_dir / "sessions_summary.csv", index=False)
    done = sum(r["status"] == "ok" for r in rows)
    logger.info("combined %d/%d session(s) -> %s, %s", done, len(rows),
                out_dir / "all_lesions.csv", out_dir / "sessions_summary.csv")


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__.split("Examples:")[1])
    p.add_argument("--raw-dir", type=Path, required=True, help="BIDS dir with the raw T1w/FLAIR")
    p.add_argument("--seg-dir", type=Path, required=True, help="LST-AI derivatives dir")
    p.add_argument("--out-dir", type=Path, required=True, help="where results go")
    p.add_argument("--subjects", nargs="+", help="only these subjects (e.g. sub-100000289)")
    p.add_argument("--split-confluent", action="store_true", help="split merged lesions")
    p.add_argument("--split-method", default="erosion",
                   choices=["erosion", "watershed", "random_walker"])
    p.add_argument("--radiomics", action="store_true",
                   help="add PyRadiomics texture features (needs pyradiomics; slow)")
    p.add_argument("--n-shells", type=int, default=5)
    p.add_argument("--threads", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
                   help="threads for registration (default: Slurm allocation or 4)")
    p.add_argument("--overwrite", action="store_true", help="redo sessions already done")
    p.add_argument("--list", action="store_true", help="list the sessions found and exit")
    p.add_argument("--task-index", type=int,
                   help="process only this session (0-based; for Slurm arrays). "
                        "Defaults to $SLURM_ARRAY_TASK_ID when set.")
    p.add_argument("--combine-only", action="store_true",
                   help="only rebuild all_lesions.csv / sessions_summary.csv")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    for d in ("raw_dir", "seg_dir"):
        if not getattr(args, d).is_dir():
            p.error(f"--{d.replace('_', '-')} {getattr(args, d)} is not a directory")

    sessions = find_sessions(args.raw_dir, args.seg_dir, args.subjects)
    if not sessions:
        logger.error("no %s found under %s", SEG_NAME, args.seg_dir)
        return 1

    if args.list:
        for i, s in enumerate(sessions):
            flag = "" if s["t1"] and s["flair"] else "   <-- missing raw T1/FLAIR"
            print(f"{i:4d}  {s['id']}{flag}")
        print(f"{len(sessions)} session(s)")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # The combined tables always cover every session (not just --subjects), so
    # processing a few subjects never drops the others from all_lesions.csv.
    all_sessions = find_sessions(args.raw_dir, args.seg_dir, None)
    if args.combine_only:
        combine(args.out_dir, all_sessions)
        return 0

    task = args.task_index
    if task is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if task is not None:
        if not 0 <= task < len(sessions):
            logger.error("task index %d out of range (0-%d)", task, len(sessions) - 1)
            return 1
        process_session(sessions[task], args.out_dir, args)
        return 0

    failed = 0
    for s in sessions:
        try:
            process_session(s, args.out_dir, args)
        except Exception:
            failed += 1
            logger.exception("%s failed", s["id"])
    combine(args.out_dir, all_sessions)
    if failed:
        logger.error("%d session(s) failed; see messages above", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
