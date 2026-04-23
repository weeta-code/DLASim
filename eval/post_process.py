#!/usr/bin/env python3
"""
Post-processing tools for generated DLA images.

Addresses the connectivity limitation of pixel-space diffusion by:
  1. Thresholding to binary
  2. Morphological closing (fills small gaps)
  3. Extracting the largest connected component
  4. Optional: skeletonization for line thinning

Also checks for loops (cycles) in the resulting structure.
"""

import argparse
import os
import glob
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_closing, label


def post_process_image(img, threshold=128, closing_size=1, largest_only=True):
    """
    Post-process a generated DLA image.

    Args:
        img: input image as numpy array (HxW, uint8)
        threshold: binarization threshold
        closing_size: size of morphological closing kernel (0 to disable)
        largest_only: if True, keep only the largest connected component

    Returns:
        processed: uint8 binary image
        stats: dict with processing statistics
    """
    binary = (img >= threshold).astype(np.uint8)

    initial_white = int(binary.sum())
    initial_components, _ = label(binary)
    n_initial_components = initial_components.max()

    # Morphological closing to join nearby fragments
    if closing_size > 0:
        struct = np.ones((closing_size, closing_size), dtype=np.uint8)
        binary = binary_closing(binary, structure=struct).astype(np.uint8)

    # Find connected components
    labeled, n_components = label(binary)

    if largest_only and n_components > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # background
        largest_label = sizes.argmax()
        binary = (labeled == largest_label).astype(np.uint8)

    final_white = int(binary.sum())
    final_components, _ = label(binary)
    n_final_components = final_components.max()

    stats = {
        "initial_white": initial_white,
        "initial_components": int(n_initial_components),
        "final_white": final_white,
        "final_components": int(n_final_components),
        "white_retained_pct": 100.0 * final_white / max(initial_white, 1),
    }

    return binary * 255, stats


def count_arms(binary_img, n_sectors=16):
    """
    Estimate number of primary arms by counting sectors containing pixels.
    Sort of a crude angular density measure.
    """
    coords = np.argwhere(binary_img > 0)
    if len(coords) < 10:
        return 0

    com = coords.mean(axis=0)
    # Angles from center of mass
    dy = coords[:, 0] - com[0]
    dx = coords[:, 1] - com[1]
    # Distance from com
    dist = np.sqrt(dx*dx + dy*dy)
    # Only consider outer ~50% of cluster
    median_r = np.median(dist)
    mask = dist > median_r
    if mask.sum() < 10:
        return 0

    angles = np.arctan2(dy[mask], dx[mask])
    # Bin angles
    hist, _ = np.histogram(angles, bins=n_sectors,
                           range=(-np.pi, np.pi))
    # Count "arms": sectors with more than expected uniform count
    expected = mask.sum() / n_sectors
    n_arms = int(np.sum(hist > 1.5 * expected))
    return n_arms


def detect_loops(binary_img):
    """
    Detect potential loops by computing Euler characteristic.
    For a simply connected tree: Euler number = 1 (1 component, 0 holes)
    Loops create holes: Euler number < 1

    Returns number of 'holes' (potential loops).
    """
    from scipy.ndimage import label as ndlabel

    # Label foreground components
    _, n_fg = ndlabel(binary_img > 0)

    # Label background components. Holes = background components - 1 (outer)
    _, n_bg = ndlabel(binary_img == 0)
    n_holes = max(0, n_bg - 1)

    return int(n_holes), int(n_fg)


def process_directory(input_dir, output_dir, closing_size=1):
    """Process all PNG images in input_dir, save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    print(f"Processing {len(paths)} images (closing_size={closing_size})")

    all_stats = []
    for p in paths:
        img = np.array(Image.open(p).convert("L"))
        processed, stats = post_process_image(img, closing_size=closing_size)

        # Loop detection on final image
        n_holes, n_components = detect_loops(processed)
        n_arms = count_arms(processed)

        stats["n_holes"] = n_holes
        stats["n_arms"] = n_arms
        stats["filename"] = os.path.basename(p)
        all_stats.append(stats)

        basename = os.path.basename(p)
        Image.fromarray(processed).save(os.path.join(output_dir, basename))

    # Summary
    print(f"\n=== Post-Processing Summary ===")
    import statistics as stat
    init_comp = [s["initial_components"] for s in all_stats]
    final_comp = [s["final_components"] for s in all_stats]
    retain = [s["white_retained_pct"] for s in all_stats]
    holes = [s["n_holes"] for s in all_stats]
    arms = [s["n_arms"] for s in all_stats]

    print(f"Images: {len(all_stats)}")
    print(f"Initial components: mean={np.mean(init_comp):.1f}, median={np.median(init_comp):.0f}")
    print(f"After largest-only:  mean={np.mean(final_comp):.2f} (should be 1.0)")
    print(f"White pixels retained: {np.mean(retain):.1f}%")
    print(f"Holes (potential loops): mean={np.mean(holes):.2f}, range=[{min(holes)}, {max(holes)}]")
    print(f"Zero-loop images: {sum(1 for h in holes if h == 0)}/{len(holes)}")
    print(f"Estimated arms: mean={np.mean(arms):.1f}")

    return all_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--closing_size", type=int, default=1,
                   help="Morphological closing kernel size (0 to disable)")
    args = p.parse_args()

    process_directory(args.input_dir, args.output_dir,
                      closing_size=args.closing_size)


if __name__ == "__main__":
    main()
