"""Side-by-side comparison of confluent lesion splitting methods.

Runs all three splitting algorithms (erosion, watershed, random_walker) on the
same input segmentation, quantifies lesions for each, writes comparison
spreadsheets, and produces diagnostic plots.

Used by the ``lesionomics compare-splitting`` CLI sub-command but also callable
directly: :func:`run_comparison` and :func:`make_synthetic_data`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

from .label_lesions import label_lesions
from .quantifying_lesions import quantify_lesions

logger = logging.getLogger(__name__)

METHODS = ["erosion", "watershed", "random_walker"]
SHAPE_COLS = [
    "volume_mm3", "surface_area_mm2", "sphericity", "solidity",
    "elongation", "flatness", "isotropy", "euler_number",
    "nearest_lesion_dist_mm",
]


# ---------------------------------------------------------------------------
# Synthetic data factory (used when real data files are absent)
# ---------------------------------------------------------------------------

def make_synthetic_data():
    """Return ``(seg_img, modalities)`` built from synthetic merged spheres."""
    logger.info("No real data found — generating synthetic scene.")
    shape = (80, 60, 60)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    header = nib.Nifti1Header()
    header.set_data_shape(shape)
    header.set_zooms((1.0, 1.0, 1.0))

    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    prob = np.zeros(shape, dtype=np.float32)

    # Cluster A: two merged spheres (r=6, centres 10 voxels apart)
    prob += 0.9 * np.exp(-((z - 20) ** 2 + (y - 30) ** 2 + (x - 30) ** 2) / (2 * 6 ** 2))
    prob += 0.9 * np.exp(-((z - 30) ** 2 + (y - 30) ** 2 + (x - 30) ** 2) / (2 * 6 ** 2))

    # Cluster B: three merged spheres in an L-shape
    prob += 0.9 * np.exp(-((z - 55) ** 2 + (y - 20) ** 2 + (x - 20) ** 2) / (2 * 5 ** 2))
    prob += 0.9 * np.exp(-((z - 55) ** 2 + (y - 30) ** 2 + (x - 20) ** 2) / (2 * 5 ** 2))
    prob += 0.9 * np.exp(-((z - 55) ** 2 + (y - 20) ** 2 + (x - 30) ** 2) / (2 * 5 ** 2))

    # Isolated small lesion
    prob += 0.9 * np.exp(-((z - 60) ** 2 + (y - 50) ** 2 + (x - 50) ** 2) / (2 * 3 ** 2))

    prob = np.clip(prob, 0.0, 1.0)
    seg_img = nib.Nifti1Image(prob, affine, header)

    # Synthetic T1 and T2 (lesions darker on T1, brighter on T2)
    t1 = np.random.default_rng(42).normal(800, 30, shape).astype(np.float32)
    t2 = np.random.default_rng(43).normal(1200, 40, shape).astype(np.float32)
    lesion_mask = prob > 0.3
    t1[lesion_mask] = np.random.default_rng(44).normal(500, 40, int(lesion_mask.sum()))
    t2[lesion_mask] = np.random.default_rng(45).normal(1800, 60, int(lesion_mask.sum()))

    modalities = {
        "T1": nib.Nifti1Image(t1, affine, header),
        "T2": nib.Nifti1Image(t2, affine, header),
    }
    return seg_img, modalities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_projection(labeled_arr: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.max(labeled_arr, axis=axis)


def _discrete_cmap(n: int):
    base = matplotlib.colormaps.get_cmap("tab20").resampled(n)
    colors = [(0, 0, 0, 1)] + [base(i) for i in range(n)]
    return ListedColormap(colors)


# ---------------------------------------------------------------------------
# Core comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    seg_img: nib.Nifti1Image,
    modalities: dict,
    threshold: float,
    output_dir: Path,
    do_quantify: bool,
):
    """Run all splitting methods and write spreadsheets + plots to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_imgs = {}
    dfs = {}

    for method in METHODS:
        logger.info("Running split_method='%s' ...", method)
        labeled_result = label_lesions(
            seg_img,
            threshold=threshold,
            split_confluent=True,
            split_method=method,
        )
        labeled_imgs[method] = labeled_result.label_img
        n = labeled_result.n_lesions
        logger.info("  → %d lesion(s)", n)

        if do_quantify and n > 0:
            df = quantify_lesions(
                labeled_img=labeled_result,
                modalities=modalities,
                include_radiomics=False,
            )
            df.insert(0, "method", method)
            dfs[method] = df
        else:
            dfs[method] = pd.DataFrame({"method": [method], "n_lesions": [n]})

    _save_spreadsheets(dfs, output_dir, do_quantify)
    _plot_label_count(dfs, output_dir)
    _plot_volume_distributions(dfs, output_dir)
    _plot_shape_metrics(dfs, output_dir)
    _plot_projections(labeled_imgs, output_dir)
    _plot_feature_heatmap(dfs, output_dir, do_quantify)

    logger.info("All outputs written to: %s", output_dir)


# ---------------------------------------------------------------------------
# Spreadsheet writers
# ---------------------------------------------------------------------------

def _save_spreadsheets(dfs: dict, output_dir: Path, do_quantify: bool):
    per_lesion = pd.concat(list(dfs.values()), ignore_index=True)
    per_lesion_path = output_dir / "comparison_per_lesion.csv"
    per_lesion.to_csv(per_lesion_path, index=False)
    logger.info("Saved per-lesion CSV: %s", per_lesion_path)

    if not do_quantify:
        return

    rows = []
    for method, df in dfs.items():
        if "volume_mm3" not in df.columns:
            continue
        numeric = df.select_dtypes(include="number").drop(columns=["label"], errors="ignore")
        row = {"method": method, "n_lesions": len(df)}
        for col in numeric.columns:
            row[f"{col}_mean"] = float(numeric[col].mean())
            row[f"{col}_std"] = float(numeric[col].std())
            row[f"{col}_median"] = float(numeric[col].median())
        rows.append(row)

    if rows:
        agg_path = output_dir / "comparison_aggregated.csv"
        pd.DataFrame(rows).to_csv(agg_path, index=False)
        logger.info("Saved aggregated CSV: %s", agg_path)

    if len(rows) >= 2:
        agg_df = pd.DataFrame(rows).set_index("method")
        mean_cols = [c for c in agg_df.columns if c.endswith("_mean")]
        diff_rows = []
        methods = list(agg_df.index)
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                d = {"method_a": m1, "method_b": m2}
                for col in mean_cols:
                    feat = col[:-5]
                    v1, v2 = agg_df.loc[m1, col], agg_df.loc[m2, col]
                    d[f"{feat}_abs_diff"] = abs(v1 - v2)
                    if abs(v1) > 1e-9:
                        d[f"{feat}_pct_diff"] = 100 * abs(v1 - v2) / abs(v1)
                diff_rows.append(d)
        diff_path = output_dir / "comparison_pairwise_diff.csv"
        pd.DataFrame(diff_rows).to_csv(diff_path, index=False)
        logger.info("Saved pairwise differences CSV: %s", diff_path)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def _plot_label_count(dfs: dict, output_dir: Path):
    counts = {}
    for method, df in dfs.items():
        counts[method] = int(df["n_lesions"].iloc[0]) if "n_lesions" in df.columns else len(df)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(list(counts.keys()), list(counts.values()),
                  color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of lesions")
    ax.set_title("Lesion count by splitting method")
    ax.set_ylim(0, max(counts.values()) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_lesion_counts.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", output_dir / "plot_lesion_counts.png")


def _plot_volume_distributions(dfs: dict, output_dir: Path):
    vol_data = {m: df["volume_mm3"].dropna().values
                for m, df in dfs.items() if "volume_mm3" in df.columns}
    if not vol_data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    positions = range(1, len(vol_data) + 1)
    bp = ax.boxplot(list(vol_data.values()), positions=list(positions),
                    patch_artist=True, widths=0.4,
                    medianprops={"color": "white", "linewidth": 2})
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(list(vol_data.keys()))
    ax.set_ylabel("Volume (mm³)")
    ax.set_title("Lesion volume distribution")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for (method, vals), color in zip(vol_data.items(), colors):
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=20, alpha=0.6, label=method, color=color,
                edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Volume (mm³)")
    ax.set_ylabel("Count")
    ax.set_title("Volume histogram by method")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "plot_volume_distributions.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", output_dir / "plot_volume_distributions.png")


def _plot_shape_metrics(dfs: dict, output_dir: Path):
    available = [c for c in SHAPE_COLS if any(c in df.columns for df in dfs.values())]
    focus = [c for c in ["sphericity", "elongation", "flatness", "isotropy", "solidity"]
             if c in available]
    if not focus:
        return

    n_cols = min(3, len(focus))
    n_rows = (len(focus) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, col in zip(axes, focus):
        data = {m: df[col].dropna().values for m, df in dfs.items() if col in df.columns}
        if not data:
            ax.set_visible(False)
            continue
        positions = range(1, len(data) + 1)
        bp = ax.boxplot(list(data.values()), positions=list(positions),
                        patch_artist=True, widths=0.4,
                        medianprops={"color": "white", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(list(data.keys()), fontsize=9)
        ax.set_title(col.replace("_", " ").capitalize())
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[len(focus):]:
        ax.set_visible(False)

    fig.suptitle("Shape metrics by splitting method", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "plot_shape_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_dir / "plot_shape_metrics.png")


def _plot_projections(labeled_imgs: dict, output_dir: Path):
    fig = plt.figure(figsize=(5 * len(METHODS), 12))
    gs = gridspec.GridSpec(3, len(METHODS), figure=fig, hspace=0.05, wspace=0.05)

    view_labels = ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"]
    for col_idx, method in enumerate(METHODS):
        labeled_arr = labeled_imgs[method].get_fdata().astype(np.int32)
        n_labels = int(labeled_arr.max())
        cmap = _discrete_cmap(max(n_labels, 1))

        for row_idx, proj_axis in enumerate([0, 1, 2]):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            proj = _max_projection(labeled_arr, axis=proj_axis)
            ax.imshow(proj.T, origin="lower", cmap=cmap,
                      vmin=0, vmax=max(n_labels, 1), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"{method}\n(n={n_labels})", fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(view_labels[row_idx], fontsize=9)

    fig.suptitle("Lesion label projections (colour = lesion ID)", fontsize=12)
    out = output_dir / "plot_label_projections.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)


def _plot_feature_heatmap(dfs: dict, output_dir: Path, do_quantify: bool):
    if not do_quantify:
        return

    focus_cols = [c for c in SHAPE_COLS if c != "euler_number"]
    rows = []
    for method, df in dfs.items():
        if "volume_mm3" not in df.columns:
            continue
        present = [c for c in focus_cols if c in df.columns]
        row = {c: float(df[c].mean()) for c in present}
        rows.append((method, row))

    if not rows:
        return

    all_cols = [c for c in focus_cols if all(c in r for _, r in rows)]
    if not all_cols:
        return

    matrix = np.array([[r[c] for c in all_cols] for _, r in rows], dtype=float)
    col_std = matrix.std(axis=0)
    col_std[col_std < 1e-9] = 1.0
    matrix_norm = (matrix - matrix.mean(axis=0)) / col_std

    method_names = [m for m, _ in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(all_cols) * 0.9), max(3, len(method_names) * 1.2)))
    im = ax.imshow(matrix_norm, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)
    ax.set_xticks(range(len(all_cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in all_cols], fontsize=8)
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names, fontsize=10)

    for i in range(len(method_names)):
        for j in range(len(all_cols)):
            raw = rows[i][1].get(all_cols[j], float("nan"))
            ax.text(j, i, f"{raw:.2g}", ha="center", va="center", fontsize=7, color="black")

    plt.colorbar(im, ax=ax, label="z-score (across methods)")
    ax.set_title("Mean feature values by splitting method (z-scored per feature)", fontsize=11)
    fig.tight_layout()
    out = output_dir / "plot_feature_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out)
