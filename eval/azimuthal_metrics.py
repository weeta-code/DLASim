#!/usr/bin/env python3
"""
Azimuthal-symmetry / isotropy metrics for DLA comparison.

Implements the three Tier-1 metrics for the 04/17 PI meeting:

  1. Normalized azimuthal variance of mass per angular wedge
       var_theta_mass = Var_wedge(mass) / Mean_wedge(mass)
     Lower = more isotropic. Training DLAs are expected to have small but
     non-zero values since each single realization has discrete branches;
     the *distribution* over many samples is what we compare.

  2. Azimuthal Fourier power spectrum of the wedge-mass signal m[theta].
     Report low-mode fractions:
         p1 = |c_1|^2 / sum(|c_k|^2),  k >= 1     (dipole asymmetry)
         p2 = |c_2|^2 / sum(|c_k|^2),  k >= 1     (quadrupole)
         p_low = sum_{k<=5} |c_k|^2 / sum(|c_k|^2)
     A generated set that is systematically elongated / rectangular /
     off-center will have inflated p1 or p2 relative to the training set.

  3. Tree-ness check: after post-processing (closing + largest component),
     count holes (Euler-number-based) and flag is_tree = (n_components == 1
     and n_holes == 0). Reports the fraction of "valid-tree" samples.
     DLA clusters are strictly trees, so this is a binary-quality metric.

Also reports:
  - n_white, r_g, n_components, n_holes (matching evaluate_generated.py
    conventions so results can be joined).

Usage:
  python eval/azimuthal_metrics.py \
      --training_dir  results/clean_training_samples \
      --generated_dir results/epoch24_samples \
      --output_dir    results/azimuthal_eval

Produces:
  {output_dir}/per_sample_metrics.csv
  {output_dir}/summary.json
  {output_dir}/azimuthal_comparison.png   (capped <= 1600px wide)
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, label
from scipy.stats import ks_2samp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Shared preprocessing (consistent with evaluate_generated.py)
# ---------------------------------------------------------------------------
def load_binary(path, threshold=128):
    img = np.array(Image.open(path).convert("L"))
    return (img >= threshold).astype(np.uint8), img.shape


def largest_component(binary, closing_iters=2):
    """Close small gaps then keep largest CC. Mirrors evaluate_generated.py."""
    if closing_iters > 0:
        struct = np.ones((3, 3), dtype=np.uint8)
        binary = binary_closing(binary, structure=struct,
                                iterations=closing_iters).astype(np.uint8)
    labeled, n = label(binary)
    if n <= 1:
        return binary
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = sizes.argmax()
    return (labeled == largest).astype(np.uint8)


# ---------------------------------------------------------------------------
# Azimuthal metrics
# ---------------------------------------------------------------------------
def azimuthal_mass_profile(binary, n_wedges=72):
    """Return (wedge_mass[n_wedges], com_yx). COM is in pixel coords."""
    coords = np.argwhere(binary > 0)
    if len(coords) < 20:
        return None, None
    com = coords.mean(axis=0)
    dy = coords[:, 0] - com[0]
    dx = coords[:, 1] - com[1]
    angles = np.arctan2(dy, dx)  # [-pi, pi]
    # histogram mass by angle
    hist, _ = np.histogram(angles, bins=n_wedges, range=(-np.pi, np.pi))
    return hist.astype(np.float64), com


def azimuthal_stats(wedge_mass):
    """
    Given m[theta] (length N), compute:
      var_theta_mass   = Var(m) / Mean(m)  (normalized angular variance)
      p1, p2           = |c_k|^2 / sum_{k>=1}|c_k|^2  (AC power fractions)
      p_low            = sum_{k=1..5} |c_k|^2 / sum_{k>=1}|c_k|^2
    """
    m = np.asarray(wedge_mass, dtype=np.float64)
    mu = m.mean()
    if mu <= 0:
        return None
    var_norm = m.var() / mu  # Var / Mean (Poisson-scaled)

    # Power spectrum (drop DC)
    F = np.fft.rfft(m)
    power = np.abs(F) ** 2
    ac = power[1:]  # k >= 1
    if ac.sum() <= 0:
        return None
    p1 = ac[0] / ac.sum()
    p2 = ac[1] / ac.sum() if len(ac) > 1 else 0.0
    p_low = ac[:5].sum() / ac.sum()

    return {
        "var_theta_mass": float(var_norm),
        "p1_fraction": float(p1),
        "p2_fraction": float(p2),
        "p_low5_fraction": float(p_low),
    }


# ---------------------------------------------------------------------------
# Tree-ness
# ---------------------------------------------------------------------------
def tree_metrics(binary, closing_iters=2):
    """
    Returns dict with n_components (on dilated binary), n_holes, is_tree flag,
    using the same conventions as evaluate_generated.py.
    """
    # Dilate-close for topological connectivity (bridges plus-sign gaps)
    struct = np.ones((3, 3), dtype=np.uint8)
    conn = (
        binary_closing(binary, structure=struct,
                       iterations=closing_iters).astype(np.uint8)
        if closing_iters > 0 else binary
    )
    _, n_components = label(conn)
    _, n_bg = label(conn == 0)
    n_holes = max(0, n_bg - 1)
    return {
        "n_components": int(n_components),
        "n_holes": int(n_holes),
        "is_tree": bool(n_components == 1 and n_holes == 0),
    }


# ---------------------------------------------------------------------------
# Per-image driver
# ---------------------------------------------------------------------------
def analyze_image(path, n_wedges=72, threshold=128, closing_iters=2):
    binary, shape = load_binary(path, threshold)
    coords = np.argwhere(binary > 0)
    if len(coords) < 20:
        return None

    com = coords.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((coords - com) ** 2, axis=1))))

    # Azimuthal on raw binary (no closing) — we want the distribution of
    # pixels as generated / simulated, not a "cleaned" version.
    wedge_mass, _ = azimuthal_mass_profile(binary, n_wedges=n_wedges)
    az = azimuthal_stats(wedge_mass) if wedge_mass is not None else None
    if az is None:
        return None

    # Tree-ness on largest component (post-processed)
    tm = tree_metrics(binary, closing_iters=closing_iters)

    return {
        "filename": os.path.basename(path),
        "height": shape[0],
        "width": shape[1],
        "n_white": int(binary.sum()),
        "rg": rg,
        **az,
        **tm,
    }


def analyze_directory(image_dir, label_name, n_wedges=72, limit=None,
                      threshold=128, closing_iters=2):
    paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not paths:
        print(f"  [!] no PNGs in {image_dir}")
        return []
    if limit and len(paths) > limit:
        idx = np.linspace(0, len(paths) - 1, limit, dtype=int)
        paths = [paths[i] for i in idx]
    print(f"  Analyzing {len(paths)} images from {image_dir}  [{label_name}]")

    rows = []
    for p in paths:
        try:
            r = analyze_image(p, n_wedges=n_wedges, threshold=threshold,
                              closing_iters=closing_iters)
        except Exception as e:
            print(f"    skip {p}: {e}")
            continue
        if r is None:
            continue
        r["set"] = label_name
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Aggregate stats & plots
# ---------------------------------------------------------------------------
METRIC_KEYS = [
    ("var_theta_mass", "Azimuthal variance  Var/Mean"),
    ("p1_fraction",    "Dipole power fraction  (k=1)"),
    ("p2_fraction",    "Quadrupole power fraction  (k=2)"),
    ("p_low5_fraction","Low-mode (k<=5) power fraction"),
    ("rg",             "Radius of gyration (px)"),
    ("n_holes",        "Holes  (loops)"),
]


def bootstrap_ci_mean(vals, n_boot=5000, alpha=0.05, seed=0):
    """Percentile bootstrap 95% CI on the mean of `vals`. Returns (lo, hi)."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) < 2:
        return (float(vals.mean()), float(vals.mean()))
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    boot_means = vals[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def summarize(rows, label_name):
    if not rows:
        return None
    n = len(rows)
    out = {"label": label_name, "n_images": n}
    for k, _ in METRIC_KEYS:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n >= 2 else 0.0
        sem = std / np.sqrt(n) if n >= 2 else 0.0  # standard error of mean
        ci_lo, ci_hi = bootstrap_ci_mean(vals)
        out[k] = {
            "mean":   mean,
            "std":    std,
            "sem":    float(sem),
            "ci95":   [ci_lo, ci_hi],
            "median": float(np.median(vals)),
            "min":    float(vals.min()),
            "max":    float(vals.max()),
        }
    out["is_tree_fraction"] = float(
        np.mean([1.0 if r["is_tree"] else 0.0 for r in rows])
    )
    out["single_component_fraction"] = float(
        np.mean([1.0 if r["n_components"] == 1 else 0.0 for r in rows])
    )
    return out


def ks_tests(train_rows, gen_rows):
    out = {}
    for k, _ in METRIC_KEYS:
        a = np.array([r[k] for r in train_rows], dtype=np.float64)
        b = np.array([r[k] for r in gen_rows], dtype=np.float64)
        if len(a) < 2 or len(b) < 2:
            continue
        stat, p = ks_2samp(a, b)
        out[k] = {"statistic": float(stat), "p_value": float(p)}
    return out


def plot_comparison(train_rows, gen_rows, output_path):
    """3x2 panel. figsize (10, 6) at dpi=150 -> 1500x900 px (<1600)."""
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    colors = {"train": "#2196F3", "gen": "#FF5722"}

    for ax, (key, title) in zip(axes.flat, METRIC_KEYS):
        t = np.array([r[key] for r in train_rows], dtype=np.float64)
        g = np.array([r[key] for r in gen_rows], dtype=np.float64)
        bp = ax.boxplot(
            [t, g],
            tick_labels=["train", "generated"],
            patch_artist=True,
            widths=0.6,
            medianprops={"color": "black"},
        )
        for patch, c in zip(bp["boxes"], [colors["train"], colors["gen"]]):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Azimuthal-symmetry & tree-ness: training vs generated",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # dpi=150 keeps 10x6 -> 1500x900 < 1600 px on either side
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_mean_spectrum(train_rows, gen_rows, output_path,
                       n_wedges=72, train_dir=None, gen_dir=None,
                       max_dirs=200):
    """
    Mean wedge-mass and mean Fourier power spectra with uncertainty bands.

    Layout (3 panels, 10x8 at dpi=150 -> 1500x1200 px):
      [0,0] mean m(θ) with ±1σ shaded band (train vs gen)
      [0,1] mean |c_k|² (log-y), shaded IQR per mode (train vs gen)
      [1,:] log-ratio log10(<P_k^gen>/<P_k^train>) with bootstrap 95% CI
            (makes the "order of magnitude at high k" claim visual)
    """
    if not train_dir or not gen_dir:
        return

    def _collect(directory, limit):
        paths = sorted(glob.glob(os.path.join(directory, "*.png")))
        if limit and len(paths) > limit:
            idx = np.linspace(0, len(paths) - 1, limit, dtype=int)
            paths = [paths[i] for i in idx]
        profs, spectra = [], []
        for p in paths:
            try:
                binary, _ = load_binary(p)
                wm, _ = azimuthal_mass_profile(binary, n_wedges=n_wedges)
                if wm is None:
                    continue
                s = wm.sum()
                if s <= 0:
                    continue
                profs.append(wm / s)                     # unit-sum
                F = np.fft.rfft(wm.astype(np.float64))
                P = np.abs(F) ** 2
                P_ac = P[1:].sum()
                if P_ac <= 0:
                    continue
                # Per-sample AC-normalized power spectrum (dimensionless)
                spectra.append(P / P_ac)
            except Exception:
                continue
        if not profs:
            return np.zeros((0, n_wedges)), np.zeros((0, n_wedges // 2 + 1))
        return np.array(profs), np.array(spectra)

    t, tP = _collect(train_dir, max_dirs)
    g, gP = _collect(gen_dir, max_dirs)
    if t.size == 0 or g.size == 0:
        return

    theta = np.linspace(-np.pi, np.pi, n_wedges, endpoint=False)
    t_mean = t.mean(axis=0); t_std = t.std(axis=0, ddof=1)
    g_mean = g.mean(axis=0); g_std = g.std(axis=0, ddof=1)

    # spectra stats (median + IQR to be robust to heavy tails)
    t_med = np.median(tP, axis=0); t_lo = np.quantile(tP, 0.25, axis=0)
    t_hi = np.quantile(tP, 0.75, axis=0)
    g_med = np.median(gP, axis=0); g_lo = np.quantile(gP, 0.25, axis=0)
    g_hi = np.quantile(gP, 0.75, axis=0)

    # bootstrap CI on log10 of gen/train mean ratio (mode-wise)
    rng = np.random.default_rng(0)
    n_boot = 2000
    k_modes = tP.shape[1]
    log_ratio_boot = np.zeros((n_boot, k_modes))
    for b in range(n_boot):
        it = rng.integers(0, len(tP), size=len(tP))
        ig = rng.integers(0, len(gP), size=len(gP))
        tm = tP[it].mean(axis=0)
        gm = gP[ig].mean(axis=0)
        # protect log of zero at the DC slot
        log_ratio_boot[b] = np.where(
            (tm > 0) & (gm > 0), np.log10(gm / tm), 0.0
        )
    lr_mean = log_ratio_boot.mean(axis=0)
    lr_lo = np.quantile(log_ratio_boot, 0.025, axis=0)
    lr_hi = np.quantile(log_ratio_boot, 0.975, axis=0)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    # --- Panel 1: mean m(θ) with ±1σ bands ---
    ax1.plot(theta, t_mean, color="#2196F3", linewidth=1.5, label="train")
    ax1.fill_between(theta, t_mean - t_std, t_mean + t_std,
                     color="#2196F3", alpha=0.2, linewidth=0,
                     label="train ±1σ (samples)")
    ax1.plot(theta, g_mean, color="#FF5722", linewidth=1.5, label="generated")
    ax1.fill_between(theta, g_mean - g_std, g_mean + g_std,
                     color="#FF5722", alpha=0.2, linewidth=0,
                     label="generated ±1σ")
    ax1.set_xlabel("angle  θ  (rad)")
    ax1.set_ylabel("fractional mass per wedge")
    ax1.set_title(f"Mean azimuthal mass profile  (N={n_wedges} wedges)",
                  fontsize=10)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: AC-normalized |c_k|² with IQR bands ---
    k = np.arange(k_modes)
    ax2.fill_between(k[1:], t_lo[1:], t_hi[1:], color="#2196F3",
                     alpha=0.25, linewidth=0, label="train IQR")
    ax2.semilogy(k[1:], t_med[1:], color="#2196F3", marker="o",
                 markersize=3.5, linewidth=1.2, label="train median")
    ax2.fill_between(k[1:], g_lo[1:], g_hi[1:], color="#FF5722",
                     alpha=0.25, linewidth=0, label="generated IQR")
    ax2.semilogy(k[1:], g_med[1:], color="#FF5722", marker="s",
                 markersize=3.5, linewidth=1.2, label="generated median")
    ax2.set_xlabel("mode  k")
    ax2.set_ylabel(r"$|c_k|^2 / \sum_{k'\geq 1} |c_{k'}|^2$  (log)")
    ax2.set_title("AC-normalized azimuthal power spectrum", fontsize=10)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3, which="both")

    # --- Panel 3: log10 ratio gen/train per mode with 95% CI band ---
    ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax3.fill_between(k[1:], lr_lo[1:], lr_hi[1:], color="purple",
                     alpha=0.25, linewidth=0,
                     label="bootstrap 95% CI")
    ax3.plot(k[1:], lr_mean[1:], color="purple", marker="D", markersize=3.5,
             linewidth=1.2, label=r"$\log_{10}(\langle P_k^{gen}\rangle "
                                  r"/ \langle P_k^{train}\rangle)$")
    ax3.set_xlabel("mode  k")
    ax3.set_ylabel(r"log$_{10}$ ratio  (gen / train)")
    ax3.set_title("Mean-power ratio per mode  "
                  "(positive = generated exceeds training)", fontsize=10)
    ax3.legend(fontsize=7, loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training_dir", required=True)
    ap.add_argument("--generated_dir", required=True)
    ap.add_argument("--output_dir", default="results/azimuthal_eval")
    ap.add_argument("--n_wedges", type=int, default=72)
    ap.add_argument("--threshold", type=int, default=128)
    ap.add_argument("--closing_iters", type=int, default=2)
    ap.add_argument("--train_limit", type=int, default=500,
                    help="Limit training samples for speed")
    ap.add_argument("--gen_limit", type=int, default=None,
                    help="Limit generated samples (default: all)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n== Azimuthal metrics ==")
    train_rows = analyze_directory(
        args.training_dir, "train",
        n_wedges=args.n_wedges, limit=args.train_limit,
        threshold=args.threshold, closing_iters=args.closing_iters,
    )
    gen_rows = analyze_directory(
        args.generated_dir, "generated",
        n_wedges=args.n_wedges, limit=args.gen_limit,
        threshold=args.threshold, closing_iters=args.closing_iters,
    )

    if not train_rows or not gen_rows:
        print("[!] not enough valid samples in one of the directories.")
        return

    # --- CSV ---
    csv_path = out / "per_sample_metrics.csv"
    fieldnames = [
        "set", "filename", "height", "width", "n_white", "rg",
        "var_theta_mass", "p1_fraction", "p2_fraction", "p_low5_fraction",
        "n_components", "n_holes", "is_tree",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in train_rows + gen_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    # --- Summaries + KS tests ---
    summary = {
        "training":  summarize(train_rows, "train"),
        "generated": summarize(gen_rows,  "generated"),
        "ks_tests":  ks_tests(train_rows, gen_rows),
        "config": {
            "n_wedges": args.n_wedges,
            "threshold": args.threshold,
            "closing_iters": args.closing_iters,
            "training_dir": args.training_dir,
            "generated_dir": args.generated_dir,
        },
    }
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- Plots ---
    plot_comparison(train_rows, gen_rows, out / "azimuthal_comparison.png")
    plot_mean_spectrum(
        train_rows, gen_rows,
        out / "azimuthal_mean_spectrum.png",
        n_wedges=args.n_wedges,
        train_dir=args.training_dir,
        gen_dir=args.generated_dir,
    )

    # --- Console report ---
    t, g = summary["training"], summary["generated"]
    print("\n== Summary ==")
    print(f"  train  : n={t['n_images']}  is_tree={t['is_tree_fraction']:.2%}")
    print(f"  gen    : n={g['n_images']}  is_tree={g['is_tree_fraction']:.2%}")
    print()
    for k, title in METRIC_KEYS:
        tm, gm = t[k], g[k]
        ks = summary["ks_tests"].get(k, {})
        print(f"  {title}")
        print(f"    train  : {tm['mean']:.4g} ± {tm['std']:.4g}  "
              f"(median {tm['median']:.4g})")
        print(f"    gen    : {gm['mean']:.4g} ± {gm['std']:.4g}  "
              f"(median {gm['median']:.4g})")
        if ks:
            print(f"    KS p   : {ks['p_value']:.3g}  "
                  f"(stat={ks['statistic']:.3g})")
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {out / 'summary.json'}")
    print(f"Wrote: {out / 'azimuthal_comparison.png'}")
    print(f"Wrote: {out / 'azimuthal_mean_spectrum.png'}")


if __name__ == "__main__":
    main()
