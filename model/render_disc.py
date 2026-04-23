#!/usr/bin/env python3
"""
Render DLA particle data with fixed-scale disc particles.

Each particle is rendered as a solid disc of fixed pixel radius at the
particle's center. The pixel-to-particle ratio is fixed across all images
(every particle has the same disc area), satisfying Machta's bijection
requirement while providing sufficient pixel density for the diffusion
model to learn from.

With disc_radius=2, each particle becomes a ~13-pixel disc (not just 1 pixel).
This gives the model 5-10x more signal than single-pixel rendering while
keeping the underlying representation consistent.

Usage:
    python render_disc.py \
        --particle_dir ../data/fixed_n1000_raw/particles \
        --output_dir ../data/fixed_n1000_disc2/images \
        --image_size 256 \
        --disc_radius 2
"""

import struct
import os
import glob
import argparse
import json
import numpy as np
from PIL import Image


def load_particles(path):
    with open(path, 'rb') as f:
        n = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(n * 16), dtype=np.float64).reshape(n, 2)
    return data


def build_disc_mask(radius):
    """Pre-compute a disc mask of given radius."""
    r = int(radius)
    y, x = np.mgrid[-r:r+1, -r:r+1]
    return (x*x + y*y <= radius*radius).astype(np.uint8)


def render_disc(particles, image_size, disc_radius=2, scale=1.0, centered=True):
    """
    Render particles as discs at fixed scale.

    Args:
        particles: Nx2 array of (x, y) coordinates
        image_size: output image size in pixels
        disc_radius: radius of each particle's disc in pixels
        scale: pixels per simulation unit (1.0 = 1 particle center per pixel)
        centered: if True, center the cluster at image center

    Returns:
        img: uint8 image (image_size x image_size), 0 or 255
    """
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    disc = build_disc_mask(disc_radius) * 255
    r = int(disc_radius)
    disc_size = 2 * r + 1

    if len(particles) == 0:
        return img

    # Center in the image
    if centered:
        com = particles.mean(axis=0)
        offset_x = image_size / 2.0 - com[0] * scale
        offset_y = image_size / 2.0 - com[1] * scale
    else:
        offset_x = image_size / 2.0
        offset_y = image_size / 2.0

    for px, py in particles:
        ix = int(round(px * scale + offset_x))
        iy = int(round(py * scale + offset_y))

        # Compute clipping region
        y0 = max(0, iy - r)
        y1 = min(image_size, iy + r + 1)
        x0 = max(0, ix - r)
        x1 = min(image_size, ix + r + 1)

        # Corresponding region of disc mask
        dy0 = y0 - (iy - r)
        dy1 = dy0 + (y1 - y0)
        dx0 = x0 - (ix - r)
        dx1 = dx0 + (x1 - x0)

        if y1 > y0 and x1 > x0:
            np.maximum(img[y0:y1, x0:x1], disc[dy0:dy1, dx0:dx1],
                       out=img[y0:y1, x0:x1])

    return img


def compute_pixel_stats(img, threshold=128):
    """Compute white pixel stats for an image."""
    binary = (img >= threshold).astype(np.uint8)
    coords = np.argwhere(binary > 0)
    n_white = len(coords)
    if n_white == 0:
        return 0, 0.0
    com = coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - com)**2, axis=1)))
    return n_white, float(rg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--particle_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--metadata_dir", default=None)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--scale", type=float, default=1.0,
                   help="Pixels per simulation unit")
    p.add_argument("--disc_radius", type=int, default=2,
                   help="Particle disc radius in pixels (default: 2 = 5x5 disc)")
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of files to render")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.metadata_dir:
        os.makedirs(args.metadata_dir, exist_ok=True)

    bin_files = sorted(glob.glob(os.path.join(args.particle_dir, "*.bin")))
    if args.limit:
        bin_files = bin_files[:args.limit]

    print(f"Rendering {len(bin_files)} particle files")
    print(f"  image_size = {args.image_size}")
    print(f"  scale = {args.scale} px/unit")
    print(f"  disc_radius = {args.disc_radius} px")
    print(f"  disc area = {int(np.sum(build_disc_mask(args.disc_radius)))} px per particle")

    all_stats = []

    for i, bf in enumerate(bin_files):
        particles = load_particles(bf)
        img = render_disc(particles, args.image_size,
                          disc_radius=args.disc_radius,
                          scale=args.scale)

        basename = os.path.splitext(os.path.basename(bf))[0]
        Image.fromarray(img).save(os.path.join(args.output_dir, basename + ".png"))

        n_white, pixel_rg = compute_pixel_stats(img)
        particle_rg = float(np.sqrt(np.mean(np.sum(
            (particles - particles.mean(axis=0))**2, axis=1))))

        stat = {
            "filename": basename,
            "particle_count": len(particles),
            "white_pixel_count": n_white,
            "pixel_rg": pixel_rg,
            "particle_rg": particle_rg,
            "disc_radius": args.disc_radius,
        }
        all_stats.append(stat)

        if args.metadata_dir:
            with open(os.path.join(args.metadata_dir, basename + ".json"), "w") as f:
                json.dump(stat, f, indent=2)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(bin_files)}] N={len(particles)}, "
                  f"white_px={n_white}, pixel_Rg={pixel_rg:.1f}")

    # Summary
    white_counts = [s["white_pixel_count"] for s in all_stats]
    pixel_rgs = [s["pixel_rg"] for s in all_stats]
    part_counts = [s["particle_count"] for s in all_stats]

    print(f"\n=== Rendering Summary ===")
    print(f"Total images: {len(all_stats)}")
    print(f"Particle count: {np.mean(part_counts):.0f} +/- {np.std(part_counts):.1f}")
    print(f"White pixels:   {np.mean(white_counts):.0f} +/- {np.std(white_counts):.1f}")
    print(f"Pixel R_g:      {np.mean(pixel_rgs):.2f} +/- {np.std(pixel_rgs):.2f}")
    print(f"Pixels per particle: {np.mean(white_counts)/np.mean(part_counts):.2f}")
    print(f"Occupancy: {np.mean(white_counts)/(args.image_size**2) * 100:.1f}%")


if __name__ == "__main__":
    main()
