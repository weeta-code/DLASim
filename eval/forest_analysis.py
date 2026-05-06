#!/usr/bin/env python3
"""
Forest analysis: quantify disjoint-cluster ("forest") behavior across runs.

Real DLAs are single connected components. If the model generates multiple
disjoint pieces, that's a fundamental quality issue. This script measures:

  - Raw connected component count (no closing)
  - Component count after closing (1, 2, 3 iter)
  - Fraction of mass in the LARGEST component
  - Number of "stray" small components (<5 pixels)

For each input directory, produces:
  {output_dir}/forest_summary.json
  {output_dir}/forest_histograms.png
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, label

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze_image(path, threshold=128, closing_iters=3, stray_threshold=5):
    """Returns dict of forest metrics for a single image."""
    img = np.array(Image.open(path).convert("L"))
    binary = (img >= threshold).astype(np.uint8)

    # Raw components
    raw_labeled, raw_n = label(binary)
    raw_sizes = np.bincount(raw_labeled.ravel())[1:] if raw_n > 0 else np.array([0])

    # Closed components (bridges nearby fragments)
    struct = np.ones((3, 3), dtype=np.uint8)
    closed = binary_closing(binary, structure=struct,
                             iterations=closing_iters).astype(np.uint8)
    closed_labeled, closed_n = label(closed)
    closed_sizes = (np.bincount(closed_labeled.ravel())[1:]
                    if closed_n > 0 else np.array([0]))

    n_white = int(binary.sum())
    largest_raw = int(raw_sizes.max()) if len(raw_sizes) else 0
    largest_closed = int(closed_sizes.max()) if len(closed_sizes) else 0

    n_strays = int((raw_sizes < stray_threshold).sum()) if len(raw_sizes) else 0

    return {
        "filename": os.path.basename(path),
        "n_white": n_white,
        "raw_components": int(raw_n),
        "raw_largest_size": largest_raw,
        "raw_largest_frac": largest_raw / max(n_white, 1),
        "closed_components": int(closed_n),
        "closed_largest_size": largest_closed,
        "closed_largest_frac": largest_closed / max(n_white, 1),
        "n_stray_components": n_strays,  # raw, fewer than `stray_threshold` px
        "raw_size_p90": float(np.percentile(raw_sizes, 90)) if len(raw_sizes) else 0.0,
    }


def analyze_dir(image_dir, label_name, limit=None, **kwargs):
    paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if limit and len(paths) > limit:
        idx = np.linspace(0, len(paths) - 1, limit, dtype=int)
        paths = [paths[i] for i in idx]
    print(f"  {label_name}: analyzing {len(paths)}")
    rows = []
    for p in paths:
        try:
            rows.append(analyze_image(p, **kwargs))
        except Exception as e:
            print(f"    skip {p}: {e}")
    return rows


def summarize(rows):
    if not rows:
        return {}
    n = len(rows)
    metrics = ["raw_components", "closed_components",
               "raw_largest_frac", "closed_largest_frac",
               "n_stray_components"]
    out = {"n_images": n}
    for k in metrics:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if n >= 2 else 0.0,
            "median": float(np.median(vals)),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    # Fraction of samples that are SINGLE connected (raw + closed)
    out["single_raw_frac"] = float(
        np.mean([1.0 if r["raw_components"] == 1 else 0.0 for r in rows]))
    out["single_closed_frac"] = float(
        np.mean([1.0 if r["closed_components"] == 1 else 0.0 for r in rows]))
    return out


def plot_distributions(results_by_label, output_path):
    """Histograms of raw components, largest-frac, etc."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [
        ("raw_components", "Raw # connected components", axes[0, 0]),
        ("closed_components", "Components after closing(3)", axes[0, 1]),
        ("raw_largest_frac", "Mass fraction in largest component (raw)",
         axes[1, 0]),
        ("n_stray_components", "Stray small components (<5 px, raw)",
         axes[1, 1]),
    ]
    colors = ["#5C6BC0", "#43A047", "#E53935", "#FB8C00", "#8E24AA"]

    for key, title, ax in metrics:
        for (label_name, rows), c in zip(results_by_label.items(), colors):
            if not rows:
                continue
            vals = np.array([r[key] for r in rows])
            if "frac" in key:
                bins = np.linspace(0, 1, 25)
            else:
                vmax = max(vals.max(), 1)
                bins = np.arange(0, vmax + 2)
            ax.hist(vals, bins=bins, alpha=0.55, color=c,
                    label=f"{label_name}  μ={vals.mean():.2f}",
                    edgecolor="black", linewidth=0.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(key, fontsize=9)
        ax.set_ylabel("# samples", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Forest analysis: disjoint-component behavior across runs",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training_dir", required=True)
    ap.add_argument("--gen_dirs", nargs="+", required=True,
                    help="One or more generated-sample dirs to compare")
    ap.add_argument("--gen_labels", nargs="+", default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("== Forest analysis ==")
    results = {}
    train = analyze_dir(args.training_dir, "train", limit=args.limit)
    results["train"] = train

    labels = args.gen_labels or [
        os.path.basename(d.rstrip("/")) for d in args.gen_dirs]
    for d, l in zip(args.gen_dirs, labels):
        results[l] = analyze_dir(d, l, limit=args.limit)

    # Summaries
    summaries = {k: summarize(v) for k, v in results.items()}
    with (out / "forest_summary.json").open("w") as f:
        json.dump(summaries, f, indent=2)

    plot_distributions(results, out / "forest_histograms.png")

    # Console report
    print()
    print(f"{'Run':<22} {'n':>4} {'raw_comp':>10} {'closed_comp':>12} "
          f"{'largest%':>10} {'strays':>8} {'single?':>8}")
    for k in results:
        s = summaries[k]
        if not s:
            continue
        rc = s["raw_components"]["mean"]
        cc = s["closed_components"]["mean"]
        lf = s["raw_largest_frac"]["mean"] * 100
        ns = s["n_stray_components"]["mean"]
        sf = s["single_closed_frac"] * 100
        print(f"{k:<22} {s['n_images']:>4} {rc:>10.2f} {cc:>12.2f} "
              f"{lf:>9.1f}% {ns:>8.2f} {sf:>7.1f}%")

    print(f"\nWrote {out / 'forest_summary.json'}")
    print(f"Wrote {out / 'forest_histograms.png'}")


if __name__ == "__main__":
    main()
