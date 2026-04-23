#!/usr/bin/env python3
"""
Test different rendering settings on a single DLA cluster.
Shows a grid of the same cluster rendered with different (scale, disc_radius)
combos, so we can visually pick the best settings before running full dataset.
"""

import struct
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from render_disc import load_particles, render_disc, build_disc_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--particle_file", required=True)
    p.add_argument("--output", default="render_test.png")
    args = p.parse_args()

    particles = load_particles(args.particle_file)
    print(f"Loaded {len(particles)} particles")
    print(f"Max radius (sim units): {np.max(np.sqrt(np.sum(particles**2, axis=1))):.1f}")

    # Test settings: (scale, disc_radius, image_size, label)
    configs = [
        (2, 1, 512, "scale=2, r=1, img=512"),
        (2, 1, 384, "scale=2, r=1, img=384"),
        (3, 1, 512, "scale=3, r=1, img=512"),
        (3, 1, 768, "scale=3, r=1, img=768"),
        (3, 1, 640, "scale=3, r=1, img=640"),
        (2, 2, 512, "scale=2, r=2, img=512"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (scale, disc_r, img_size, label) in enumerate(configs):
        img = render_disc(particles, img_size, disc_radius=disc_r, scale=scale)
        n_white = int(np.sum(img > 0))

        # Compute pixel R_g
        coords = np.argwhere(img > 0)
        if len(coords) > 0:
            com = coords.mean(axis=0)
            rg = float(np.sqrt(np.mean(np.sum((coords - com)**2, axis=1))))
        else:
            rg = 0.0

        # Compute disc area per particle
        disc_px = int(np.sum(build_disc_mask(disc_r))) if disc_r > 0 else 1

        axes[i].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(f"{label}\nimg={img_size}x{img_size}, disc_area={disc_px}px\n"
                          f"white_px={n_white}, R_g={rg:.1f}px",
                          fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(args.output, dpi=120, bbox_inches='tight')
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
