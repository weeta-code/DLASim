#!/usr/bin/env python3
"""
Render DLA particle data into 4-channel "seed-aware" images for v5 training.

Channels (all float32 in [0, 1]):
    0 - Binary presence (disc mask, identical to render_disc.py)
    1 - REVERSE deposition order (1 = seed, 0 = last placed). Encodes growth
        causality such that the seed has the maximum value, fading to zero
        at peripheral particles.
    2 - REVERSE distance from seed (1 = at seed, 0 = farthest particle).
        Same emphasis: "deepest red" at the seed, fading outward.
    3 - Explicit seed Gaussian (small radial blob centered on seed pixel).
        Sharp attentional cue: "the seed is HERE".

Rationale:
    The previous 3-channel encoding had ch1 and ch2 both ZERO at the seed
    and MAX at the periphery. With viridis/plasma colormaps the seed read
    as the darkest point, which buried it visually and made it harder for
    the diffusion model to anchor on. Reversing both encodings + adding
    an explicit Gaussian gives the model three coincident "seed is here"
    signals at the cluster's center of growth.

Output: .npz files with key 'channels', shape (4, H, W), float32 in [0, 1].

Usage:
    python render_seedaware.py \\
        --particle_dir data/fixed_n1000_raw_v2/particles \\
        --output_dir data/fixed_n1000_seedaware/images \\
        --metadata_dir data/fixed_n1000_seedaware/metadata \\
        --image_size 512 --scale 2.0 --disc_radius 1 --seed_sigma 5.0
"""

import struct
import os
import glob
import argparse
import json

import numpy as np


def load_particles(path):
    with open(path, "rb") as f:
        n = struct.unpack("i", f.read(4))[0]
        data = np.frombuffer(f.read(n * 16), dtype=np.float64).reshape(n, 2)
    return data


def build_disc_mask(radius):
    r = int(radius)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= radius * radius).astype(np.uint8)


def make_seed_gaussian(image_size, seed_iy, seed_ix, sigma):
    """Render a 2-D isotropic Gaussian centered on (seed_iy, seed_ix), peak=1."""
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    dy = yy - seed_iy
    dx = xx - seed_ix
    g = np.exp(-(dy * dy + dx * dx) / (2.0 * sigma * sigma))
    return g.astype(np.float32)


def render_seedaware(particles, image_size, disc_radius=1, scale=2.0,
                     seed_sigma=5.0, centered=True):
    """
    Render particles into 4 seed-aware channels (uint16 internally).

    Returns:
      ch0, ch1_rev, ch2_rev: uint16 (H, W) in [0, 65535]
      ch3_gauss: float32 (H, W) in [0, 1]
      max_distance, seed_pixel: meta
    """
    N = len(particles)
    ch0 = np.zeros((image_size, image_size), dtype=np.uint16)
    ch1 = np.zeros((image_size, image_size), dtype=np.uint16)  # REVERSE order
    ch2 = np.zeros((image_size, image_size), dtype=np.uint16)  # REVERSE distance

    if N == 0:
        empty_g = np.zeros((image_size, image_size), dtype=np.float32)
        return ch0, ch1, ch2, empty_g, 0.0, (image_size // 2, image_size // 2)

    disc_mask = build_disc_mask(disc_radius)
    r = int(disc_radius)

    # --- Centering: identical to render_disc.py / render_multichannel.py ---
    if centered:
        com = particles.mean(axis=0)
        offset_x = image_size / 2.0 - com[0] * scale
        offset_y = image_size / 2.0 - com[1] * scale
    else:
        offset_x = image_size / 2.0
        offset_y = image_size / 2.0

    # --- Per-particle values ---
    # REVERSE order: particle 0 = 65535 (seed = max), particle N-1 = 0
    if N > 1:
        rev_order = ((N - 1 - np.arange(N, dtype=np.float64)) /
                     (N - 1) * 65535).astype(np.uint16)
    else:
        rev_order = np.array([65535], dtype=np.uint16)

    # REVERSE distance: at seed = 65535, farthest = 0
    seed = particles[0]
    dists = np.sqrt(np.sum((particles - seed) ** 2, axis=1))
    max_distance = float(dists.max())
    if max_distance > 0:
        rev_dist = ((1.0 - dists / max_distance) * 65535).astype(np.uint16)
    else:
        rev_dist = np.full(N, 65535, dtype=np.uint16)

    # --- Render each particle ---
    disc_white = (disc_mask * np.uint16(65535))

    # Track the seed pixel position for the Gaussian channel
    seed_iy = int(round(seed[1] * scale + offset_y))
    seed_ix = int(round(seed[0] * scale + offset_x))

    for idx in range(N):
        px, py = particles[idx]
        ix = int(round(px * scale + offset_x))
        iy = int(round(py * scale + offset_y))

        y0 = max(0, iy - r)
        y1 = min(image_size, iy + r + 1)
        x0 = max(0, ix - r)
        x1 = min(image_size, ix + r + 1)

        dy0 = y0 - (iy - r)
        dy1 = dy0 + (y1 - y0)
        dx0 = x0 - (ix - r)
        dx1 = dx0 + (x1 - x0)

        if y1 <= y0 or x1 <= x0:
            continue

        mask_slice = disc_mask[dy0:dy1, dx0:dx1]

        # Channel 0: binary presence
        np.maximum(ch0[y0:y1, x0:x1], disc_white[dy0:dy1, dx0:dx1],
                   out=ch0[y0:y1, x0:x1])

        # Channel 1: REVERSE order — seed wins overlaps (it has highest value)
        order_disc = (mask_slice * rev_order[idx]).astype(np.uint16)
        np.maximum(ch1[y0:y1, x0:x1], order_disc, out=ch1[y0:y1, x0:x1])

        # Channel 2: REVERSE distance — same logic
        dist_disc = (mask_slice * rev_dist[idx]).astype(np.uint16)
        np.maximum(ch2[y0:y1, x0:x1], dist_disc, out=ch2[y0:y1, x0:x1])

    # Channel 3: Gaussian centered on seed pixel
    ch3_gauss = make_seed_gaussian(image_size, seed_iy, seed_ix, seed_sigma)

    return ch0, ch1, ch2, ch3_gauss, max_distance, (seed_iy, seed_ix)


def compute_pixel_stats(ch0, threshold=32768):
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
        description="Render DLA particles into 4-channel seed-aware npz")
    p.add_argument("--particle_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--metadata_dir", default=None)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--disc_radius", type=int, default=1)
    p.add_argument("--seed_sigma", type=float, default=5.0,
                   help="Gaussian sigma (px) for the seed-emphasis channel")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.metadata_dir:
        os.makedirs(args.metadata_dir, exist_ok=True)

    bin_files = sorted(glob.glob(os.path.join(args.particle_dir, "*.bin")))
    if args.limit:
        bin_files = bin_files[:args.limit]

    print(f"Rendering {len(bin_files)} particle files (4-channel seed-aware)")
    print(f"  image_size  = {args.image_size}")
    print(f"  scale       = {args.scale} px/unit")
    print(f"  disc_radius = {args.disc_radius} px")
    print(f"  seed_sigma  = {args.seed_sigma} px")

    all_stats = []

    for i, bf in enumerate(bin_files):
        particles = load_particles(bf)
        ch0, ch1, ch2, ch3, max_distance, (siy, six) = render_seedaware(
            particles, args.image_size,
            disc_radius=args.disc_radius,
            scale=args.scale,
            seed_sigma=args.seed_sigma,
        )

        # Stack: (4, H, W) float32 in [0, 1]
        img_4ch = np.zeros((4, args.image_size, args.image_size),
                           dtype=np.float32)
        img_4ch[0] = ch0.astype(np.float32) / 65535.0
        img_4ch[1] = ch1.astype(np.float32) / 65535.0
        img_4ch[2] = ch2.astype(np.float32) / 65535.0
        img_4ch[3] = ch3  # already float32 in [0, 1]

        basename = os.path.splitext(os.path.basename(bf))[0]
        np.savez_compressed(
            os.path.join(args.output_dir, basename + ".npz"),
            channels=img_4ch,
        )

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
            "seed_sigma": args.seed_sigma,
            "seed_pixel": [siy, six],
        }
        all_stats.append(stat)

        if args.metadata_dir:
            with open(os.path.join(args.metadata_dir, basename + ".json"),
                      "w") as f:
                json.dump(stat, f, indent=2)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(bin_files)}] N={len(particles)}, "
                  f"white_px={n_white}, pixel_Rg={pixel_rg:.1f}")

    if not all_stats:
        print("\nNo files to render.")
        return

    white_counts = [s["white_pixel_count"] for s in all_stats]
    pixel_rgs = [s["pixel_rg"] for s in all_stats]

    print(f"\n=== Rendering summary (4-channel seed-aware) ===")
    print(f"Total images:    {len(all_stats)}")
    print(f"White pixels:    {np.mean(white_counts):.0f} +/- {np.std(white_counts):.1f}")
    print(f"Pixel R_g:       {np.mean(pixel_rgs):.2f} +/- {np.std(pixel_rgs):.2f}")
    print(f"Occupancy:       {np.mean(white_counts) / (args.image_size ** 2) * 100:.1f}%")

    print("\n--- Channel value range (last rendered image) ---")
    print(f"  ch0 (presence):     min={ch0.min()}, max={ch0.max()}  uint16")
    print(f"  ch1 (rev order):    min={ch1.min()}, max={ch1.max()}  (seed = max)")
    print(f"  ch2 (rev distance): min={ch2.min()}, max={ch2.max()}  (seed = max)")
    print(f"  ch3 (seed Gauss):   min={ch3.min():.4f}, max={ch3.max():.4f}  (peak at seed)")
    print(f"  Seed pixel: ({siy}, {six})")


if __name__ == "__main__":
    main()
