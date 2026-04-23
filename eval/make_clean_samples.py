#!/usr/bin/env python3
"""
Produce visually-clean, fully-connected DLA images from model outputs.
Applies strong morphological closing + largest-component extraction + optional
slight thinning to produce images where single-parent tree structure is clear
and there are NO disconnected components.
"""

import argparse
import os
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_closing, binary_dilation


def clean_connect(img, threshold=128, closing_iter=3, dilation_iter=0):
    """
    Connect scattered pixels into a single DLA tree.
    1. Threshold
    2. Morphological closing (bridge gaps)
    3. Extract largest connected component
    4. Optional dilation to thicken branches
    """
    binary = (img >= threshold).astype(np.uint8)

    # Heavy closing to bridge rendering gaps
    struct = np.ones((3, 3), dtype=np.uint8)
    closed = binary_closing(binary, structure=struct,
                            iterations=closing_iter).astype(np.uint8)

    # Largest component only
    labeled, n = label(closed)
    if n > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        largest = sizes.argmax()
        closed = (labeled == largest).astype(np.uint8)

    # Optional thickening
    if dilation_iter > 0:
        closed = binary_dilation(closed, structure=struct,
                                 iterations=dilation_iter).astype(np.uint8)

    return (closed * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--closing_iter", type=int, default=3,
                   help="Number of closing iterations (bridges gaps)")
    p.add_argument("--dilation_iter", type=int, default=0,
                   help="Extra dilation to thicken branches")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    if args.limit:
        paths = paths[:args.limit]

    n_connected = 0
    white_stats = []
    rg_stats = []

    for p in paths:
        img = np.array(Image.open(p).convert("L"))
        cleaned = clean_connect(img, closing_iter=args.closing_iter,
                                dilation_iter=args.dilation_iter)

        # Check: should be exactly 1 component
        binary = (cleaned > 0).astype(np.uint8)
        _, n_comp = label(binary)
        if n_comp == 1:
            n_connected += 1

        white_count = int(binary.sum())
        white_stats.append(white_count)

        coords = np.argwhere(binary)
        if len(coords) > 0:
            com = coords.mean(axis=0)
            rg = float(np.sqrt(np.mean(np.sum((coords - com)**2, axis=1))))
            rg_stats.append(rg)

        basename = os.path.basename(p)
        Image.fromarray(cleaned).save(os.path.join(args.output_dir, basename))

    print(f"Processed {len(paths)} images")
    print(f"Single-connected: {n_connected}/{len(paths)}")
    print(f"White pixels: {np.mean(white_stats):.0f} +/- {np.std(white_stats):.0f}")
    print(f"R_g: {np.mean(rg_stats):.2f} +/- {np.std(rg_stats):.2f}")


if __name__ == "__main__":
    main()
