#!/usr/bin/env python3
"""
Comprehensive evaluation of generated DLA images.

Reports Machta's key metrics:
  - White pixel count distribution (should match training set)
  - Radius of gyration distribution (should match training set)
  - Connected components (should be 1 = no islands)
  - Euler number / holes (should be 0 = no loops)
  - Arm count (statistically characterized for DLA)

Compares to ground truth distribution and to post-processed results.
"""

import argparse
import os
import glob
import json
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_closing
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze_image(img_path, threshold=128, connectivity_dilation=3):
    """Compute all metrics for one image.

    connectivity_dilation: dilation kernel size for connectivity analysis.
    When particles are rendered as non-overlapping discs (scale=2, r=1 plus signs),
    pixel-level components count every plus sign. Dilating bridges these gaps
    so components = true topological components of the underlying cluster.
    """
    img = np.array(Image.open(img_path).convert("L"))
    binary = (img >= threshold).astype(np.uint8)

    coords = np.argwhere(binary > 0)
    n_white = len(coords)

    if n_white < 5:
        return None

    # R_g on raw pixels (the honest pixel R_g for fractal dim)
    com = coords.mean(axis=0)
    rg = float(np.sqrt(np.mean(np.sum((coords - com)**2, axis=1))))

    # Dilate for meaningful connectivity (merges non-touching plus signs)
    # At scale=2, r=1 rendering, particles are 4 pixels apart with 2-pixel gaps.
    # Need iterations >= 2 to bridge those gaps.
    if connectivity_dilation > 0:
        struct = np.ones((3, 3), dtype=np.uint8)
        connectivity_binary = binary_closing(binary, structure=struct,
                                             iterations=connectivity_dilation).astype(np.uint8)
    else:
        connectivity_binary = binary

    # Connected components (on dilated version)
    _, n_components = label(connectivity_binary)

    # Holes via Euler number: holes = bg_components - 1
    _, n_bg = label(connectivity_binary == 0)
    n_holes = max(0, n_bg - 1)

    # Arm count (crude)
    dy = coords[:, 0] - com[0]
    dx = coords[:, 1] - com[1]
    dist = np.sqrt(dx*dx + dy*dy)
    median_r = np.median(dist)
    outer_mask = dist > median_r
    if outer_mask.sum() >= 10:
        angles = np.arctan2(dy[outer_mask], dx[outer_mask])
        hist, _ = np.histogram(angles, bins=16, range=(-np.pi, np.pi))
        expected = outer_mask.sum() / 16
        n_arms = int(np.sum(hist > 1.5 * expected))
    else:
        n_arms = 0

    return {
        "filename": os.path.basename(img_path),
        "n_white": n_white,
        "rg": rg,
        "n_components": int(n_components),
        "n_holes": int(n_holes),
        "n_arms": n_arms,
    }


def post_process_largest_component(img, threshold=128, closing_size=3):
    """Apply closing (iterations) + extract largest component."""
    binary = (img >= threshold).astype(np.uint8)
    if closing_size > 0:
        struct = np.ones((3, 3), dtype=np.uint8)
        binary = binary_closing(binary, structure=struct,
                                iterations=closing_size).astype(np.uint8)
    labeled, n = label(binary)
    if n > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        largest = sizes.argmax()
        binary = (labeled == largest).astype(np.uint8)
    return binary * 255


def analyze_directory(image_dir, label_name="", post_process=False,
                      closing_size=2, threshold=128):
    paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not paths:
        print(f"No PNGs in {image_dir}")
        return []

    print(f"Analyzing {len(paths)} images from {image_dir}"
          + (" (post-processed)" if post_process else ""))

    results = []
    for p in paths:
        if post_process:
            raw = np.array(Image.open(p).convert("L"))
            pp = post_process_largest_component(raw, threshold, closing_size)
            # Save temp and analyze
            tmp_path = "/tmp/pp_temp.png"
            Image.fromarray(pp).save(tmp_path)
            r = analyze_image(tmp_path, threshold)
        else:
            r = analyze_image(p, threshold)

        if r is not None:
            results.append(r)

    return results


def summarize(results, label_name=""):
    if not results:
        return None
    wpx = np.array([r["n_white"] for r in results])
    rg = np.array([r["rg"] for r in results])
    comp = np.array([r["n_components"] for r in results])
    holes = np.array([r["n_holes"] for r in results])
    arms = np.array([r["n_arms"] for r in results])

    print(f"\n=== {label_name} ({len(results)} images) ===")
    print(f"  White pixels:       {wpx.mean():.1f} +/- {wpx.std():.1f}  "
          f"[{wpx.min()}, {wpx.max()}]")
    print(f"  Radius of gyration: {rg.mean():.2f} +/- {rg.std():.2f}")
    print(f"  Components:         mean={comp.mean():.1f}, median={int(np.median(comp))}, "
          f"single-connected={np.sum(comp == 1)}/{len(results)}")
    print(f"  Holes (loops):      mean={holes.mean():.2f}, zero-loop={np.sum(holes == 0)}/{len(results)}")
    print(f"  Arms:               {arms.mean():.2f} +/- {arms.std():.2f}")

    return {
        "label": label_name,
        "n_images": len(results),
        "white_pixels": {"mean": float(wpx.mean()), "std": float(wpx.std())},
        "rg": {"mean": float(rg.mean()), "std": float(rg.std())},
        "components": {"mean": float(comp.mean()),
                       "single_connected_frac": float(np.sum(comp == 1) / len(results))},
        "holes": {"mean": float(holes.mean()),
                  "zero_loop_frac": float(np.sum(holes == 0) / len(results))},
        "arms": {"mean": float(arms.mean()), "std": float(arms.std())},
    }


def plot_comparison(results_dict, output_path):
    """Bar/box plots comparing ground truth vs generated vs post-processed."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics = [
        ("n_white", "White pixel count", axes[0, 0]),
        ("rg", "Radius of gyration (pixels)", axes[0, 1]),
        ("n_components", "Connected components", axes[0, 2]),
        ("n_holes", "Number of holes (loops)", axes[1, 0]),
        ("n_arms", "Number of arms", axes[1, 1]),
    ]
    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

    for key, title, ax in metrics:
        data = []
        labels = []
        for i, (name, results) in enumerate(results_dict.items()):
            if results:
                vals = [r[key] for r in results]
                data.append(vals)
                labels.append(name)
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_title(title, fontsize=11)
            ax.tick_params(axis='x', rotation=15, labelsize=8)
            ax.grid(True, alpha=0.3)

    # Summary text in last panel
    axes[1, 2].axis('off')
    summary_text = "KEY FINDINGS:\n\n"
    for name, results in results_dict.items():
        if results:
            rg_vals = [r["rg"] for r in results]
            comp_vals = [r["n_components"] for r in results]
            summary_text += f"{name}:\n"
            summary_text += f"  R_g: {np.mean(rg_vals):.1f} +/- {np.std(rg_vals):.1f}\n"
            summary_text += f"  Single-conn: {np.sum(np.array(comp_vals) == 1)}/{len(results)}\n\n"
    axes[1, 2].text(0.02, 0.98, summary_text, transform=axes[1, 2].transAxes,
                    verticalalignment='top', fontsize=10, family='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ground_truth", required=True,
                   help="Directory of ground truth images")
    p.add_argument("--generated", required=True,
                   help="Directory of generated images")
    p.add_argument("--output_dir", default="results/evaluation")
    p.add_argument("--gt_limit", type=int, default=500,
                   help="Limit ground truth sample size (for speed)")
    p.add_argument("--closing_size", type=int, default=2,
                   help="Morphological closing kernel for post-processing")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Subsample ground truth for speed
    gt_paths = sorted(glob.glob(os.path.join(args.ground_truth, "*.png")))
    if args.gt_limit and len(gt_paths) > args.gt_limit:
        # Deterministic subsample
        idx = np.linspace(0, len(gt_paths) - 1, args.gt_limit, dtype=int)
        gt_sub_dir = "/tmp/gt_eval_subset"
        os.makedirs(gt_sub_dir, exist_ok=True)
        import shutil
        for i in idx:
            shutil.copy(gt_paths[i], gt_sub_dir)
        gt_dir = gt_sub_dir
    else:
        gt_dir = args.ground_truth

    # Analyze each set
    gt_results = analyze_directory(gt_dir, "Ground Truth")
    gen_results = analyze_directory(args.generated, "Generated (raw)")
    pp_results = analyze_directory(args.generated, "Generated (post-processed)",
                                   post_process=True,
                                   closing_size=args.closing_size)

    # Summarize
    gt_summary = summarize(gt_results, "Ground Truth")
    gen_summary = summarize(gen_results, "Generated (raw)")
    pp_summary = summarize(pp_results, "Generated (post-processed)")

    all_summary = {
        "ground_truth": gt_summary,
        "generated_raw": gen_summary,
        "generated_postprocessed": pp_summary,
    }

    # Save
    with open(os.path.join(args.output_dir, "evaluation_summary.json"), "w") as f:
        json.dump(all_summary, f, indent=2)

    # Plot
    plot_comparison({
        "Ground Truth": gt_results,
        "Generated (raw)": gen_results,
        "Generated (post-proc)": pp_results,
    }, os.path.join(args.output_dir, "evaluation_comparison.png"))

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
