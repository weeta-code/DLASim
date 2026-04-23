#!/usr/bin/env python3
"""
Compare ground truth DLA images vs generated DLA images.

Produces:
  1. Side-by-side visual comparison grids
  2. Fractal dimension comparison table and histogram
  3. Radial mass profile M(r) comparison

Usage:
    python compare.py \
        --ground_truth ../data/output/images \
        --generated ../results/generated/individual \
        --output_dir ../results/comparison
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from fractal_dim import fractal_dimension, analyze_directory


def make_comparison_grid(gt_dir, gen_dir, output_path, n=8):
    """Create a side-by-side grid: top row ground truth, bottom row generated."""
    gt_paths = sorted([p for p in Path(gt_dir).glob("*.png")])[:n]
    gen_paths = sorted([p for p in Path(gen_dir).glob("*.png")])[:n]

    n_show = min(len(gt_paths), len(gen_paths), n)
    if n_show == 0:
        print("No images to compare")
        return

    fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5.5))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for i in range(n_show):
        gt_img = np.array(Image.open(gt_paths[i]).convert("L"))
        gen_img = np.array(Image.open(gen_paths[i]).convert("L"))

        axes[0, i].imshow(gt_img, cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        if i == 0:
            axes[0, i].set_ylabel("Ground Truth\n(Random Walk)", fontsize=11)

        axes[1, i].imshow(gen_img, cmap="gray", vmin=0, vmax=255)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])
        if i == 0:
            axes[1, i].set_ylabel("Generated\n(Custom DDPM)", fontsize=11)

    fig.suptitle("DLA Cluster Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison grid saved to {output_path}")


def fractal_dimension_comparison(gt_dir, gen_dir, output_dir):
    """Compare fractal dimensions with histogram and summary table."""
    print("\n--- Ground Truth Analysis ---")
    gt_summary = analyze_directory(gt_dir, label="Ground Truth (Random Walk)")

    print("\n--- Generated Analysis ---")
    gen_summary = analyze_directory(gen_dir, label="Generated (Custom DDPM)")

    if gt_summary is None or gen_summary is None:
        print("Cannot compare: missing data")
        return

    gt_dims = [r["fractal_dimension"] for r in gt_summary["per_image"]]
    gen_dims = [r["fractal_dimension"] for r in gen_summary["per_image"]]

    # Histogram comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        min(min(gt_dims), min(gen_dims)) - 0.05,
        max(max(gt_dims), max(gen_dims)) + 0.05,
        30
    )
    ax.hist(gt_dims, bins=bins, alpha=0.6, label=f"Ground Truth (mean={np.mean(gt_dims):.3f})",
            color="#2196F3", edgecolor="white")
    ax.hist(gen_dims, bins=bins, alpha=0.6, label=f"Generated (mean={np.mean(gen_dims):.3f})",
            color="#FF5722", edgecolor="white")
    ax.axvline(1.71, color="black", linestyle="--", linewidth=1.5, label="Theoretical D_f = 1.71")
    ax.set_xlabel("Fractal Dimension (D_f)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Box-Counting Fractal Dimension Distribution", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fractal_dim_histogram.png"), dpi=150)
    plt.close()

    # Summary table
    comparison = {
        "ground_truth": {
            "mean": gt_summary["fractal_dimension_mean"],
            "std": gt_summary["fractal_dimension_std"],
            "n": gt_summary["n_images"],
        },
        "generated": {
            "mean": gen_summary["fractal_dimension_mean"],
            "std": gen_summary["fractal_dimension_std"],
            "n": gen_summary["n_images"],
        },
        "theoretical": 1.71,
        "prior_work_finetuned_sd": {
            "50_images_5_epochs": {"mean": 1.86, "std": 0.08},
            "500_images_10_epochs": {"mean": 1.77, "std": 0.08},
        },
        "gap_gt_vs_theoretical": abs(gt_summary["fractal_dimension_mean"] - 1.71),
        "gap_gen_vs_theoretical": abs(gen_summary["fractal_dimension_mean"] - 1.71),
    }

    with open(os.path.join(output_dir, "comparison_summary.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 65)
    print(f"{'FRACTAL DIMENSION COMPARISON':^65}")
    print("=" * 65)
    print(f"{'Source':<35} {'Mean D_f':>10} {'Std':>8} {'N':>5}")
    print("-" * 65)
    print(f"{'Theoretical':<35} {'1.71':>10}")
    print(f"{'Ground Truth (this work)':<35} {gt_summary['fractal_dimension_mean']:>10.4f} "
          f"{gt_summary['fractal_dimension_std']:>8.4f} {gt_summary['n_images']:>5d}")
    print(f"{'Generated (Custom DDPM)':<35} {gen_summary['fractal_dimension_mean']:>10.4f} "
          f"{gen_summary['fractal_dimension_std']:>8.4f} {gen_summary['n_images']:>5d}")
    print(f"{'Prior: SD Fine-tuned (50 img)':<35} {'1.86':>10} {'0.08':>8}")
    print(f"{'Prior: SD Fine-tuned (500 img)':<35} {'1.77':>10} {'0.08':>8}")
    print("=" * 65)


def radial_mass_profile(image_path, threshold=128):
    """Compute M(r) = number of occupied pixels within radius r from center."""
    img = np.array(Image.open(image_path).convert("L"))
    binary = (img >= threshold).astype(np.float32)
    h, w = binary.shape
    cy, cx = h / 2, w / 2

    # Distance of each pixel from center
    y, x = np.mgrid[0:h, 0:w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Occupied pixel distances
    occ_dists = dist[binary > 0]
    if len(occ_dists) == 0:
        return None, None

    max_r = np.max(occ_dists)
    radii = np.linspace(1, max_r, 50)
    mass = np.array([np.sum(occ_dists <= r) for r in radii])

    return radii, mass


def compare_radial_profiles(gt_dir, gen_dir, output_path, n=20):
    """Compare average radial mass profiles M(r)."""
    gt_paths = sorted([p for p in Path(gt_dir).glob("*.png")])[:n]
    gen_paths = sorted([p for p in Path(gen_dir).glob("*.png")])[:n]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot individual + average for ground truth
    gt_profiles = []
    for p in gt_paths:
        r, m = radial_mass_profile(str(p))
        if r is not None:
            # Normalize radius
            r_norm = r / r.max()
            m_norm = m / m.max()
            gt_profiles.append(np.interp(np.linspace(0, 1, 100), r_norm, m_norm))
            ax.plot(r_norm, m_norm, color="#2196F3", alpha=0.1, linewidth=0.5)

    gen_profiles = []
    for p in gen_paths:
        r, m = radial_mass_profile(str(p))
        if r is not None:
            r_norm = r / r.max()
            m_norm = m / m.max()
            gen_profiles.append(np.interp(np.linspace(0, 1, 100), r_norm, m_norm))
            ax.plot(r_norm, m_norm, color="#FF5722", alpha=0.1, linewidth=0.5)

    # Plot averages
    if gt_profiles:
        gt_avg = np.mean(gt_profiles, axis=0)
        ax.plot(np.linspace(0, 1, 100), gt_avg, color="#2196F3",
                linewidth=2, label="Ground Truth (avg)")
    if gen_profiles:
        gen_avg = np.mean(gen_profiles, axis=0)
        ax.plot(np.linspace(0, 1, 100), gen_avg, color="#FF5722",
                linewidth=2, label="Generated (avg)")

    # Reference line for D_f = 1.71
    r_ref = np.linspace(0.05, 1, 100)
    ax.plot(r_ref, r_ref ** 1.71, "k--", linewidth=1, alpha=0.5,
            label=r"$r^{1.71}$ (theoretical)")

    ax.set_xlabel("Normalized Radius (r / r_max)", fontsize=12)
    ax.set_ylabel("Normalized Mass M(r) / M_total", fontsize=12)
    ax.set_title("Radial Mass Profile Comparison", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Radial profile comparison saved to {output_path}")


def main():
    p = argparse.ArgumentParser(description="Compare ground truth vs generated DLA images")
    p.add_argument("--ground_truth", type=str, required=True,
                   help="Directory of ground truth DLA images")
    p.add_argument("--generated", type=str, required=True,
                   help="Directory of generated DLA images")
    p.add_argument("--output_dir", type=str, default="../results/comparison")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("DLA Ground Truth vs Generated Comparison")
    print("=" * 65)

    # 1. Visual comparison grid
    make_comparison_grid(args.ground_truth, args.generated,
                         out / "visual_comparison.png")

    # 2. Fractal dimension comparison
    fractal_dimension_comparison(args.ground_truth, args.generated, str(out))

    # 3. Radial mass profiles
    compare_radial_profiles(args.ground_truth, args.generated,
                            out / "radial_profiles.png")

    print(f"\nAll comparison outputs saved to {out}/")


if __name__ == "__main__":
    main()
