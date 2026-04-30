#!/usr/bin/env python3
"""
Render DLA particle data into 3-channel images for diffusion model training.

Channels:
    0 - Binary presence (disc mask, same as render_disc.py)
    1 - Deposition order (normalized 0..1, seed=0, last=1)
    2 - Distance from seed (normalized 0..1)

Output: .npz files with key 'channels', shape (3, H, W), float32 in [0, 1].

Usage:
    python render_multichannel.py \\
        --particle_dir data/fixed_n1000_raw_v2/particles \\
        --output_dir data/fixed_n1000_multichannel/images \\
        --metadata_dir data/fixed_n1000_multichannel/metadata \\
        --image_size 512 \\
        --scale 2.0 \\
        --disc_radius 1 \\
        --limit 10
"""

import struct
import os
import glob
import argparse
import json
import numpy as np


def load_particles(path):
    """Load particle positions from binary file.

    Format: 4-byte int N, then N x 2 float64 (x, y) pairs in deposition order.
    """
    with open(path, 'rb') as f:
        n = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(n * 16), dtype=np.float64).reshape(n, 2)
    return data


def build_disc_mask(radius):
    """Pre-compute a disc mask of given radius."""
    r = int(radius)
    y, x = np.mgrid[-r:r+1, -r:r+1]
    return (x*x + y*y <= radius*radius).astype(np.uint8)


def render_multichannel(particles, image_size, disc_radius=2, scale=1.0,
                        centered=True):
    """
    Render particles into 3 channels at fixed scale.

    The coordinate-to-pixel mapping is identical to render_disc.py so that
    channel 0 exactly reproduces the single-channel renderer output.

    Args:
        particles: Nx2 array of (x, y) in deposition order (index 0 = seed).
        image_size: output image size in pixels.
        disc_radius: radius of each particle's disc in pixels.
        scale: pixels per simulation unit.
        centered: if True, center the cluster's COM at image center.

    Returns:
        ch0: uint16 (H, W) - binary presence (0 or 65535)
        ch1: uint16 (H, W) - deposition order (0..65535)
        ch2: uint16 (H, W) - distance from seed (0..65535)
        max_distance: float - max Euclidean distance from seed (for metadata)
    """
    N = len(particles)
    ch0 = np.zeros((image_size, image_size), dtype=np.uint16)
    ch1 = np.zeros((image_size, image_size), dtype=np.uint16)
    ch2 = np.zeros((image_size, image_size), dtype=np.uint16)

    disc_mask = build_disc_mask(disc_radius)
    r = int(disc_radius)

    if N == 0:
        return ch0, ch1, ch2, 0.0

    # --- Centering logic (verbatim from render_disc.py) ---
    if centered:
        com = particles.mean(axis=0)
        offset_x = image_size / 2.0 - com[0] * scale
        offset_y = image_size / 2.0 - com[1] * scale
    else:
        offset_x = image_size / 2.0
        offset_y = image_size / 2.0

    # --- Pre-compute per-particle values ---
    # Channel 1: deposition order normalised to [0, 65535]
    if N > 1:
        order_values = (np.arange(N, dtype=np.float64) / (N - 1) * 65535).astype(
            np.uint16)
    else:
        order_values = np.zeros(N, dtype=np.uint16)

    # Channel 2: Euclidean distance from seed (particle 0)
    seed = particles[0]
    dists = np.sqrt(np.sum((particles - seed) ** 2, axis=1))
    max_distance = float(dists.max())
    if max_distance > 0:
        dist_values = (dists / max_distance * 65535).astype(np.uint16)
    else:
        dist_values = np.zeros(N, dtype=np.uint16)

    # --- Render each particle (in deposition order) ---
    disc_white = (disc_mask * np.uint16(65535))  # for channel 0

    for idx in range(N):
        px, py = particles[idx]
        ix = int(round(px * scale + offset_x))
        iy = int(round(py * scale + offset_y))

        # Compute clipping region (verbatim from render_disc.py)
        y0 = max(0, iy - r)
        y1 = min(image_size, iy + r + 1)
        x0 = max(0, ix - r)
        x1 = min(image_size, ix + r + 1)

        # Corresponding region of disc mask
        dy0 = y0 - (iy - r)
        dy1 = dy0 + (y1 - y0)
        dx0 = x0 - (ix - r)
        dx1 = dx0 + (x1 - x0)

        if y1 <= y0 or x1 <= x0:
            continue

        mask_slice = disc_mask[dy0:dy1, dx0:dx1]

        # Channel 0: binary presence (max, same as render_disc.py)
        np.maximum(ch0[y0:y1, x0:x1], disc_white[dy0:dy1, dx0:dx1],
                   out=ch0[y0:y1, x0:x1])

        # Channel 1: deposition order – max so later particles win overlaps
        order_disc = (mask_slice * order_values[idx]).astype(np.uint16)
        np.maximum(ch1[y0:y1, x0:x1], order_disc,
                   out=ch1[y0:y1, x0:x1])

        # Channel 2: distance from seed – max at overlaps
        dist_disc = (mask_slice * dist_values[idx]).astype(np.uint16)
        np.maximum(ch2[y0:y1, x0:x1], dist_disc,
                   out=ch2[y0:y1, x0:x1])

    return ch0, ch1, ch2, max_distance


def compute_pixel_stats(ch0, threshold=32768):
    """Compute white pixel count and radius of gyration from channel 0."""
    binary = (ch0 >= threshold).astype(np.uint8)
    coords = np.argwhere(binary > 0)
    n_white = len(coords)
    if n_white == 0:
        return 0, 0.0
    com = coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - com) ** 2, axis=1)))
    return n_white, float(rg)


def main():
    p = argparse.ArgumentParser(
        description="Render DLA particles into 3-channel images (.npz)")
    p.add_argument("--particle_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--metadata_dir", default=None)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--scale", type=float, default=2.0,
                   help="Pixels per simulation unit")
    p.add_argument("--disc_radius", type=int, default=1,
                   help="Particle disc radius in pixels (default: 1)")
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of files to render")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.metadata_dir:
        os.makedirs(args.metadata_dir, exist_ok=True)

    bin_files = sorted(glob.glob(os.path.join(args.particle_dir, "*.bin")))
    if args.limit:
        bin_files = bin_files[:args.limit]

    print(f"Rendering {len(bin_files)} particle files (3-channel)")
    print(f"  image_size   = {args.image_size}")
    print(f"  scale        = {args.scale} px/unit")
    print(f"  disc_radius  = {args.disc_radius} px")
    print(f"  disc area    = {int(np.sum(build_disc_mask(args.disc_radius)))} px per particle")

    all_stats = []

    for i, bf in enumerate(bin_files):
        particles = load_particles(bf)
        ch0, ch1, ch2, max_distance = render_multichannel(
            particles, args.image_size,
            disc_radius=args.disc_radius,
            scale=args.scale,
        )

        # Stack to (3, H, W) float32 in [0, 1]
        img_3ch = np.stack([ch0, ch1, ch2], axis=0).astype(np.float32) / 65535.0

        basename = os.path.splitext(os.path.basename(bf))[0]
        np.savez_compressed(
            os.path.join(args.output_dir, basename + ".npz"),
            channels=img_3ch,
        )

        # Stats
        n_white, pixel_rg = compute_pixel_stats(ch0)
        particle_rg = float(np.sqrt(np.mean(np.sum(
            (particles - particles.mean(axis=0)) ** 2, axis=1))))

        stat = {
            "filename": basename,
            "particle_count": int(len(particles)),
            "white_pixel_count": int(n_white),
            "pixel_rg": pixel_rg,
            "particle_rg": particle_rg,
            "disc_radius": args.disc_radius,
            "max_distance": max_distance,
            "scale": args.scale,
        }
        all_stats.append(stat)

        if args.metadata_dir:
            with open(os.path.join(args.metadata_dir, basename + ".json"), "w") as f:
                json.dump(stat, f, indent=2)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(bin_files)}] N={len(particles)}, "
                  f"white_px={n_white}, pixel_Rg={pixel_rg:.1f}")

    if not all_stats:
        print("\nNo files to render.")
        return

    # --- Summary ---
    white_counts = [s["white_pixel_count"] for s in all_stats]
    pixel_rgs = [s["pixel_rg"] for s in all_stats]
    part_counts = [s["particle_count"] for s in all_stats]
    max_dists = [s["max_distance"] for s in all_stats]

    print(f"\n=== Rendering Summary (3-channel) ===")
    print(f"Total images:    {len(all_stats)}")
    print(f"Particle count:  {np.mean(part_counts):.0f} +/- {np.std(part_counts):.1f}")
    print(f"White pixels:    {np.mean(white_counts):.0f} +/- {np.std(white_counts):.1f}")
    print(f"Pixel R_g:       {np.mean(pixel_rgs):.2f} +/- {np.std(pixel_rgs):.2f}")
    print(f"Max distance:    {np.mean(max_dists):.2f} +/- {np.std(max_dists):.2f}")
    print(f"Pixels/particle: {np.mean(white_counts)/np.mean(part_counts):.2f}")
    print(f"Occupancy:       {np.mean(white_counts)/(args.image_size**2) * 100:.1f}%")

    # Channel value range sanity check
    print(f"\n--- Channel sanity check (last rendered image) ---")
    print(f"  ch0 (presence): min={ch0.min()}, max={ch0.max()}")
    print(f"  ch1 (order):    min={ch1.min()}, max={ch1.max()}")
    print(f"  ch2 (distance): min={ch2.min()}, max={ch2.max()}")


if __name__ == "__main__":
    main()
