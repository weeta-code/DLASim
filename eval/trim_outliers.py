#!/usr/bin/env python3
"""
Trim outlier/debris pixels from generated images while preserving the
original plus-sign aesthetic. Does NOT close gaps or skeletonize.

Approach:
  1. Threshold to binary
  2. Use morphological closing to identify the "cluster region" (used only for
     component labeling, not saved as output)
  3. In the dilated version, keep only the largest connected region
  4. Map back: keep original pixels that fall within that largest region

This preserves the visual plus-sign structure but removes debris particles
scattered at image corners/edges.
"""

import argparse
import os
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_closing, binary_dilation


def trim_outliers(img, threshold=128, closing_iter=3):
    """
    Keep only pixels that belong to the largest cluster (defined via dilation).
    Preserves original pixel positions (no closing/thickening applied to output).
    """
    binary = (img >= threshold).astype(np.uint8)

    if binary.sum() < 5:
        return (binary * 255).astype(np.uint8)

    # Dilate to find the cluster region (used only for labeling)
    struct = np.ones((3, 3), dtype=np.uint8)
    dilated = binary_closing(binary, structure=struct,
                             iterations=closing_iter).astype(np.uint8)

    # Also do a more aggressive dilation to catch nearby stragglers
    expanded = binary_dilation(dilated, structure=struct,
                               iterations=2).astype(np.uint8)

    # Label components in the expanded version
    labeled, n = label(expanded)
    if n > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        largest_label = sizes.argmax()
        cluster_mask = (labeled == largest_label).astype(np.uint8)
    else:
        cluster_mask = expanded

    # Return ORIGINAL pixels that fall within the cluster region
    trimmed = binary * cluster_mask
    return (trimmed * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--closing_iter", type=int, default=3,
                   help="Closing iterations for cluster region identification")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    if args.limit:
        paths = paths[:args.limit]

    wpx_raw = []
    wpx_trim = []
    rg_trim = []

    for p in paths:
        img = np.array(Image.open(p).convert("L"))
        raw_white = int((img >= 128).sum())
        wpx_raw.append(raw_white)

        trimmed = trim_outliers(img, closing_iter=args.closing_iter)
        trim_white = int((trimmed >= 128).sum())
        wpx_trim.append(trim_white)

        # R_g on trimmed
        coords = np.argwhere(trimmed > 0)
        if len(coords) > 0:
            com = coords.mean(axis=0)
            rg = float(np.sqrt(np.mean(np.sum((coords - com)**2, axis=1))))
            rg_trim.append(rg)

        basename = os.path.basename(p)
        Image.fromarray(trimmed).save(os.path.join(args.output_dir, basename))

    print(f"Processed {len(paths)} images")
    print(f"White pixels: raw {np.mean(wpx_raw):.0f} +/- {np.std(wpx_raw):.0f}")
    print(f"              trimmed {np.mean(wpx_trim):.0f} +/- {np.std(wpx_trim):.0f}")
    print(f"              removed {np.mean(np.array(wpx_raw) - np.array(wpx_trim)):.0f} debris pixels on avg")
    print(f"R_g (trimmed): {np.mean(rg_trim):.2f} +/- {np.std(rg_trim):.2f}")


if __name__ == "__main__":
    main()
