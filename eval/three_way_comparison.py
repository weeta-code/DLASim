#!/usr/bin/env python3
"""
Three-way comparison figure: v3 baseline (from azimuthal_eval_100v100),
v3-controt epoch 69, v4-multichannel epoch 4, all measured against the
same training set.

Produces a single side-by-side chart for the meeting:
  - Panel 1: dipole p_1 (gen/train ratio per model)
  - Panel 2: quadrupole p_2
  - Panel 3: hole count
  - Panel 4: per-sample dipole magnitude distribution (overlaid histograms)

Output capped at <2000px to comply with Anthropic image limits.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def main():
    base = Path("/home/vectors/Documents/Research/Machta/dla_project/results")
    v3 = load_summary(base / "azimuthal_eval_100v100" / "summary.json")
    v3cr = load_summary(base / "eval_v3cr_e84" / "azimuthal" / "summary.json")
    v4mc = load_summary(base / "eval_v4mc_e24" / "azimuthal" / "summary.json")

    runs = [
        ("v3 baseline e64", v3, "#5C6BC0"),
        ("v3-controt e84", v3cr, "#43A047"),
        ("v4-multi e24", v4mc, "#E53935"),
    ]
    train_color = "#9E9E9E"

    metrics = [
        ("p1_fraction", "Dipole  p₁  (×train ratio)"),
        ("p2_fraction", "Quadrupole  p₂  (×train ratio)"),
        ("n_holes",     "Hole count  (×train ratio)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Panels 1-3: bar charts of mean ± 95% CI
    for ax, (key, title) in zip(axes.flat[:3], metrics):
        train_mean = v3["training"][key]["mean"]
        labels, ratios, ci_lo, ci_hi, colors = [], [], [], [], []
        for name, s, c in runs:
            gen = s["generated"][key]
            ratio = gen["mean"] / train_mean
            cilo = gen["ci95"][0] / train_mean
            cihi = gen["ci95"][1] / train_mean
            labels.append(name)
            ratios.append(ratio)
            ci_lo.append(cilo)
            ci_hi.append(cihi)
            colors.append(c)

        x = np.arange(len(labels))
        bars = ax.bar(x, ratios, color=colors, alpha=0.75, edgecolor="black", linewidth=0.8)
        # error bars from CI
        yerr_lo = np.array(ratios) - np.array(ci_lo)
        yerr_hi = np.array(ci_hi) - np.array(ratios)
        ax.errorbar(x, ratios, yerr=[yerr_lo, yerr_hi], fmt="none",
                    ecolor="black", capsize=4, linewidth=1.5)
        ax.axhline(1.0, color=train_color, linestyle="--", linewidth=1, label="train")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("ratio (gen mean / train mean)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        # annotate
        for xi, r in zip(x, ratios):
            ax.text(xi, r * 1.05 if r > 0 else 0.05, f"×{r:.2f}",
                    ha="center", fontsize=9, fontweight="bold")

    # Panel 4: dipole magnitude histograms
    ax4 = axes.flat[3]
    do_dirs = [
        ("v3 baseline e64", base / "azimuthal_eval_100v100", "#5C6BC0"),
        ("v3-controt e84", base / "eval_v3cr_e84" / "dipole_orientation", "#43A047"),
        ("v4-multi e24", base / "eval_v4mc_e24" / "dipole_orientation", "#E53935"),
    ]

    # Need the per-sample magnitudes from the per_sample CSVs
    import csv
    bins = np.linspace(0, 0.4, 30)

    # train comes from any of the runs (same data sampled deterministically)
    train_mags_path = base / "eval_v3cr_e84" / "dipole_orientation" / "dipole_orientation_summary.json"
    if train_mags_path.exists():
        with open(train_mags_path) as f:
            t = json.load(f)
            train_mean_mag = t["training"]["magnitude_mean"]
            train_std_mag = t["training"]["magnitude_std"]
        ax4.axvline(train_mean_mag, color=train_color, linestyle="--",
                    linewidth=1.5, label=f"train mean ({train_mean_mag:.3f})")
        ax4.axvspan(train_mean_mag - train_std_mag,
                    train_mean_mag + train_std_mag,
                    color=train_color, alpha=0.18)

    # use the dipole orientation summary json for per-sample magnitudes (use means+std as proxy)
    # Better: parse per_sample csv from azimuthal_metrics
    for name, s, c in runs:
        gen = s["generated"]
        # n, dipole magnitude isn't in azimuthal csv directly. Use sqrt of p1?
        # Actually summary doesn't have raw magnitudes. Read from dipole_orientation_summary.json
        pass

    # Plot per-sample distributions from dipole_orientation summaries
    do_summaries = []
    for name, dpath, c in do_dirs:
        sj = dpath / "dipole_orientation_summary.json"
        if sj.exists():
            with open(sj) as f:
                summary = json.load(f)
            do_summaries.append((name, summary, c))

    # Use mean and std to plot Gaussian approximations
    x = np.linspace(0, 0.5, 200)
    for name, summary, c in do_summaries:
        mu = summary["generated"]["magnitude_mean"]
        sigma = summary["generated"]["magnitude_std"]
        n = summary["generated"]["n"]
        ax4.plot(x, np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) /
                 (sigma * np.sqrt(2 * np.pi)),
                 color=c, linewidth=2,
                 label=f"{name} (n={n}, μ={mu:.3f})")
        ax4.axvline(mu, color=c, linewidth=0.8, linestyle=":")

    ax4.set_xlabel("dipole magnitude  |c₁|/N", fontsize=9)
    ax4.set_ylabel("density (Gaussian approx)", fontsize=9)
    ax4.set_title("Per-sample dipole magnitude  (lower = more isotropic)",
                  fontsize=10)
    ax4.set_xlim(0, 0.4)
    ax4.legend(fontsize=7, loc="upper right")
    ax4.grid(True, alpha=0.3)

    fig.suptitle("DLA diffusion comparison: v3 e64 vs v3-controt e84 vs "
                 "v4-multichannel e24", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = base / "comparison_images" / "three_way_comparison_2026-05-06.png"
    # 11x8 at dpi=150 = 1650x1200, under 2000px
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
