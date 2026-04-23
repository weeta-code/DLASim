#!/usr/bin/env python3
"""
Produce skeleton (1-pixel-wide tree) versions of generated DLA images.
Close gaps, keep largest component, then skeletonize to thin lines
showing the tree structure with single-parent connectivity.
"""

import argparse
import os
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import label, binary_closing, binary_dilation
try:
    from skimage.morphology import skeletonize
    HAVE_SKELETONIZE = True
except ImportError:
    HAVE_SKELETONIZE = False


def process(img, threshold=128, closing_iter=3, do_skeleton=True, thicken=2):
    """
    Clean + optionally skeletonize + thicken for visibility.
    """
    binary = (img >= threshold).astype(np.uint8)

    # Bridge gaps
    struct = np.ones((3, 3), dtype=np.uint8)
    closed = binary_closing(binary, structure=struct,
                            iterations=closing_iter).astype(np.uint8)

    # Largest component
    labeled, n = label(closed)
    if n > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        closed = (labeled == sizes.argmax()).astype(np.uint8)

    if do_skeleton and HAVE_SKELETONIZE:
        skel = skeletonize(closed.astype(bool)).astype(np.uint8)
        if thicken > 0:
            skel = binary_dilation(skel, structure=struct,
                                   iterations=thicken).astype(np.uint8)
        return (skel * 255).astype(np.uint8)
    else:
        return (closed * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--closing_iter", type=int, default=3)
    p.add_argument("--thicken", type=int, default=1,
                   help="Dilation after skeletonization to make visible")
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))[:args.limit]

    if not HAVE_SKELETONIZE:
        print("WARNING: skimage not available, falling back to closed version only")

    n_ok = 0
    for p in paths:
        img = np.array(Image.open(p).convert("L"))
        result = process(img, closing_iter=args.closing_iter,
                         thicken=args.thicken)
        binary = (result > 0).astype(np.uint8)
        _, n = label(binary)
        if n == 1:
            n_ok += 1
        Image.fromarray(result).save(os.path.join(args.output_dir,
                                                   os.path.basename(p)))

    print(f"Processed {len(paths)} images, {n_ok}/{len(paths)} single-connected")


if __name__ == "__main__":
    main()
