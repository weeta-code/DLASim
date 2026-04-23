#!/usr/bin/env python3
"""
Create side-by-side comparison images: ground truth | generated raw | generated clean | skeleton.
Each row shows different samples to make it obvious the model reproduces DLA morphology.
"""

import argparse
import os
import glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_closing, binary_dilation
try:
    from skimage.morphology import skeletonize
    HAVE_SKEL = True
except ImportError:
    HAVE_SKEL = False


def clean(img, closing_iter=3):
    binary = (img >= 128).astype(np.uint8)
    struct = np.ones((3, 3), dtype=np.uint8)
    closed = binary_closing(binary, structure=struct,
                            iterations=closing_iter).astype(np.uint8)
    labeled, n = label(closed)
    if n > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        closed = (labeled == sizes.argmax()).astype(np.uint8)
    return (closed * 255).astype(np.uint8)


def skeleton(img, closing_iter=3, thicken=1):
    cleaned = clean(img, closing_iter)
    binary = (cleaned > 0).astype(np.uint8)
    if HAVE_SKEL:
        skel = skeletonize(binary.astype(bool)).astype(np.uint8)
        if thicken > 0:
            struct = np.ones((3, 3), dtype=np.uint8)
            skel = binary_dilation(skel, structure=struct,
                                   iterations=thicken).astype(np.uint8)
        return (skel * 255).astype(np.uint8)
    return cleaned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ground_truth_dir", required=True)
    p.add_argument("--generated_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n_rows", type=int, default=4)
    args = p.parse_args()

    gt_paths = sorted(glob.glob(os.path.join(args.ground_truth_dir, "*.png")))[:args.n_rows]
    gen_paths = sorted(glob.glob(os.path.join(args.generated_dir, "*.png")))[:args.n_rows]

    n = min(len(gt_paths), len(gen_paths), args.n_rows)

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    titles = ["Ground Truth",
              "Generated (raw)",
              "Generated (clean: closing + largest)",
              "Generated (skeleton + thicken)"]

    for i in range(n):
        gt = np.array(Image.open(gt_paths[i]).convert("L"))
        raw = np.array(Image.open(gen_paths[i]).convert("L"))
        cln = clean(raw)
        skl = skeleton(raw)

        for j, (img, title) in enumerate(zip([gt, raw, cln, skl], titles)):
            axes[i, j].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(args.output, dpi=120, bbox_inches='tight')
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
