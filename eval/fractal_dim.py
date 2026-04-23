#!/usr/bin/env python3
"""
Box-counting fractal dimension estimation for DLA images.

Given a directory of images, binarizes them, computes the box-counting
fractal dimension D_f, and reports statistics.

Box-counting method:
  1. Binarize the image (threshold at 128)
  2. For a range of box sizes epsilon, count N(epsilon) = number of boxes
     that contain at least one white pixel
  3. D_f = -slope of log(N) vs log(epsilon), estimated by linear regression
     over the scaling region

Usage:
    python fractal_dim.py --image_dir ../data/output/images
    python fractal_dim.py --image_dir ../results/generated/individual --label "Generated"
"""

import os
import glob
import argparse
import json
import numpy as np
from PIL import Image
from scipy import stats


def box_count(binary_image, box_size):
    """Count non-empty boxes of given size covering the binary image."""
    h, w = binary_image.shape
    count = 0
    for y in range(0, h, box_size):
        for x in range(0, w, box_size):
            box = binary_image[y:y+box_size, x:x+box_size]
            if np.any(box):
                count += 1
    return count


def fractal_dimension(image_path, threshold=128, min_box=2, max_box=None):
    """
    Compute box-counting fractal dimension of a single image.

    Returns:
        D_f: estimated fractal dimension
        r_squared: R^2 of the linear fit
        box_sizes: array of box sizes used
        counts: array of box counts
    """
    img = np.array(Image.open(image_path).convert("L"))
    binary = (img >= threshold).astype(np.uint8)

    if not np.any(binary):
        return None, None, None, None

    h, w = binary.shape
    if max_box is None:
        max_box = min(h, w) // 4

    # Box sizes: powers of 2 from min_box to max_box
    box_sizes = []
    s = min_box
    while s <= max_box:
        box_sizes.append(s)
        s *= 2
    # Also add some intermediate sizes for better regression
    extra = []
    for i in range(len(box_sizes) - 1):
        mid = int((box_sizes[i] + box_sizes[i+1]) / 2)
        if mid not in box_sizes:
            extra.append(mid)
    box_sizes = sorted(set(box_sizes + extra))

    counts = [box_count(binary, s) for s in box_sizes]

    # Filter out zero counts
    valid = [(s, c) for s, c in zip(box_sizes, counts) if c > 0]
    if len(valid) < 3:
        return None, None, None, None

    sizes, cnts = zip(*valid)
    log_eps = np.log(np.array(sizes, dtype=float))
    log_n = np.log(np.array(cnts, dtype=float))

    # Linear regression: log(N) = -D_f * log(eps) + const
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_eps, log_n)
    D_f = -slope
    r_squared = r_value ** 2

    return D_f, r_squared, np.array(sizes), np.array(cnts)


def analyze_directory(image_dir, threshold=128, label=""):
    """Compute fractal dimension for all images in a directory."""
    paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not paths:
        print(f"No PNG images found in {image_dir}")
        return None

    print(f"Analyzing {len(paths)} images from {image_dir}")
    if label:
        print(f"Label: {label}")

    results = []
    for path in paths:
        D_f, r2, sizes, counts = fractal_dimension(path, threshold=threshold)
        if D_f is not None:
            name = os.path.basename(path)
            results.append({
                "filename": name,
                "fractal_dimension": D_f,
                "r_squared": r2,
            })

    if not results:
        print("No valid results")
        return None

    dims = [r["fractal_dimension"] for r in results]
    r2s = [r["r_squared"] for r in results]

    summary = {
        "label": label,
        "image_dir": image_dir,
        "n_images": len(results),
        "fractal_dimension_mean": float(np.mean(dims)),
        "fractal_dimension_std": float(np.std(dims)),
        "fractal_dimension_median": float(np.median(dims)),
        "fractal_dimension_min": float(np.min(dims)),
        "fractal_dimension_max": float(np.max(dims)),
        "r_squared_mean": float(np.mean(r2s)),
        "per_image": results,
    }

    print(f"\n{'='*50}")
    print(f"{'Label:':<20} {label or 'N/A'}")
    print(f"{'Images analyzed:':<20} {len(results)}")
    print(f"{'D_f mean +/- std:':<20} {np.mean(dims):.4f} +/- {np.std(dims):.4f}")
    print(f"{'D_f median:':<20} {np.median(dims):.4f}")
    print(f"{'D_f range:':<20} [{np.min(dims):.4f}, {np.max(dims):.4f}]")
    print(f"{'R^2 mean:':<20} {np.mean(r2s):.4f}")
    print(f"{'Theoretical D_f:':<20} 1.71")
    print(f"{'='*50}\n")

    return summary


def main():
    p = argparse.ArgumentParser(description="Box-counting fractal dimension analysis")
    p.add_argument("--image_dir", type=str, required=True,
                   help="Directory of PNG images to analyze")
    p.add_argument("--threshold", type=int, default=128,
                   help="Binarization threshold (0-255)")
    p.add_argument("--label", type=str, default="",
                   help="Label for this set of images")
    p.add_argument("--output", type=str, default=None,
                   help="Output JSON file for results")
    args = p.parse_args()

    summary = analyze_directory(args.image_dir, args.threshold, args.label)

    if summary and args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
