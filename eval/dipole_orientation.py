#!/usr/bin/env python3
"""
Dipole orientation analysis: is the dipole excess in generated samples
preferentially oriented (suggesting positional encoding / data leakage),
or randomly oriented per sample (suggesting the model fails to balance growth)?

For each image:
  - Compute COM-relative pixel positions
  - Compute c_1 = sum_j exp(-i theta_j)        (complex dipole coefficient)
  - Magnitude |c_1| measures dipole strength
  - Phase arg(c_1) measures dipole *direction*

If the model has a preferred orientation, the histogram of arg(c_1) values
across many generated samples should be non-uniform (concentrated near
some preferred angle, or near multiples of 90 degrees if the D4 augmentation
is leaking).

If the model just produces lopsided clusters with random orientation,
arg(c_1) should be uniform on [-pi, pi].

Two tests:
  1. Rayleigh test for circular uniformity
  2. Histogram visualization of dipole phases for train vs gen

Output:
  - {output_dir}/dipole_orientation.png  (polar histogram)
  - {output_dir}/dipole_orientation_summary.json
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_dipole(path, threshold=128):
    """Returns complex c_1 (unit-vector dipole) or None for empty images."""
    img = np.array(Image.open(path).convert("L"))
    binary = (img >= threshold).astype(np.uint8)
    coords = np.argwhere(binary > 0)
    if len(coords) < 20:
        return None
    com = coords.mean(axis=0)
    dy = coords[:, 0] - com[0]
    dx = coords[:, 1] - com[1]
    theta = np.arctan2(dy, dx)
    c1 = np.sum(np.exp(-1j * theta))
    # Normalize by total mass so |c1| is in [0, 1]-ish
    return c1 / len(coords)


def collect(directory, label, limit=None):
    paths = sorted(glob.glob(os.path.join(directory, "*.png")))
    if limit and len(paths) > limit:
        idx = np.linspace(0, len(paths) - 1, limit, dtype=int)
        paths = [paths[i] for i in idx]
    out = []
    for p in paths:
        c1 = compute_dipole(p)
        if c1 is not None:
            out.append({"file": os.path.basename(p),
                        "c1_real": float(c1.real),
                        "c1_imag": float(c1.imag),
                        "magnitude": float(np.abs(c1)),
                        "phase": float(np.angle(c1))})
    print(f"  {label}: {len(out)}/{len(paths)} valid")
    return out


def rayleigh_test(phases):
    """
    Tests H0: phases uniformly distributed on circle.
    Returns: (R_bar, p_value)
    R_bar = mean resultant length. R_bar -> 0 for uniform, -> 1 for concentrated.
    p-value approximation valid for n >= 50.
    """
    n = len(phases)
    if n < 8:
        return 0.0, 1.0
    C = np.cos(phases).sum()
    S = np.sin(phases).sum()
    R = np.sqrt(C**2 + S**2)
    R_bar = R / n
    # Asymptotic chi-square approximation (n large)
    Z = n * R_bar**2
    p = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*n)
                       - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288*n**2))
    p = max(0.0, min(1.0, p))
    return R_bar, p


def plot_polar(train_data, gen_data, output_path):
    """Polar histogram of dipole phases, train vs gen."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                              subplot_kw={'projection': 'polar'})

    n_bins = 24
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)

    for ax, data, color, label in [
        (axes[0], train_data, "#2196F3", "train"),
        (axes[1], gen_data,   "#FF5722", "generated"),
    ]:
        phases = np.array([d["phase"] for d in data])
        if len(phases) == 0:
            continue
        counts, _ = np.histogram(phases, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(bin_centers, counts, width=2*np.pi/n_bins,
               color=color, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{label}  n={len(phases)}", fontsize=11)
        ax.set_theta_zero_location("E")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Dipole orientation distribution  (uniform = no preferred direction)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_magnitude_vs_phase(train_data, gen_data, output_path):
    """Scatter: each sample as a point in (phase, magnitude) plane."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for data, color, label in [
        (train_data, "#2196F3", "train"),
        (gen_data,   "#FF5722", "generated"),
    ]:
        if len(data) == 0:
            continue
        phases = np.array([d["phase"] for d in data])
        mags = np.array([d["magnitude"] for d in data])
        ax.scatter(phases, mags, c=color, alpha=0.5, s=15, label=f"{label}  n={len(data)}")

    ax.set_xlabel("dipole phase  arg(c1)  (rad)")
    ax.set_ylabel("dipole magnitude  |c1| / N")
    ax.set_title("Per-sample dipole: phase vs magnitude")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    # mark cardinal axes (D4 candidates)
    for a in [0, np.pi/2, np.pi, -np.pi/2]:
        ax.axvline(a, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training_dir", required=True)
    p.add_argument("--generated_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_limit", type=int, default=500)
    p.add_argument("--gen_limit", type=int, default=None)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("== Dipole orientation analysis ==")
    train = collect(args.training_dir, "train", limit=args.train_limit)
    gen = collect(args.generated_dir, "generated", limit=args.gen_limit)

    if not train or not gen:
        print("[!] empty inputs.")
        return

    train_phases = np.array([d["phase"] for d in train])
    gen_phases = np.array([d["phase"] for d in gen])
    train_mags = np.array([d["magnitude"] for d in train])
    gen_mags = np.array([d["magnitude"] for d in gen])

    train_rbar, train_p = rayleigh_test(train_phases)
    gen_rbar, gen_p = rayleigh_test(gen_phases)

    summary = {
        "training": {
            "n": len(train),
            "rayleigh_R_bar": float(train_rbar),
            "rayleigh_p_uniform": float(train_p),
            "magnitude_mean": float(train_mags.mean()),
            "magnitude_std": float(train_mags.std(ddof=1)),
            "phase_mean": float(np.angle(np.mean(np.exp(1j*train_phases)))),
        },
        "generated": {
            "n": len(gen),
            "rayleigh_R_bar": float(gen_rbar),
            "rayleigh_p_uniform": float(gen_p),
            "magnitude_mean": float(gen_mags.mean()),
            "magnitude_std": float(gen_mags.std(ddof=1)),
            "phase_mean": float(np.angle(np.mean(np.exp(1j*gen_phases)))),
        },
        "interpretation": {
            "training_orientation": (
                "isotropic (random per-sample direction)" if train_p > 0.05
                else "preferentially oriented"
            ),
            "generated_orientation": (
                "isotropic (random per-sample direction)" if gen_p > 0.05
                else "preferentially oriented"
            ),
        },
    }

    with (out / "dipole_orientation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    plot_polar(train, gen, out / "dipole_orientation.png")
    plot_magnitude_vs_phase(train, gen, out / "dipole_phase_vs_magnitude.png")

    print()
    print(f"Train n={len(train)}: |R|={train_rbar:.4f}, p_uniform={train_p:.3g}, "
          f"mag={train_mags.mean():.4f}±{train_mags.std(ddof=1):.4f}")
    print(f"Gen   n={len(gen)}: |R|={gen_rbar:.4f}, p_uniform={gen_p:.3g}, "
          f"mag={gen_mags.mean():.4f}±{gen_mags.std(ddof=1):.4f}")
    print()
    print(f"Training orientation: {summary['interpretation']['training_orientation']}")
    print(f"Generated orientation: {summary['interpretation']['generated_orientation']}")
    print()
    print(f"Mag ratio gen/train: {gen_mags.mean()/train_mags.mean():.3f}x")


if __name__ == "__main__":
    main()
