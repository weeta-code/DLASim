#!/usr/bin/env python3
"""
Render DLA particle data into 3-channel RGB images for v5 training.

Encoding:
    Seed pixel (particle 0): pure blue (0, 0, 255)         — B=255 only here
    Other particle pixels:    R = G = 255 * d / d_max
                              B = 0
    Background:               (0, 0, 0)

where d is Euclidean distance from the seed (in particle coordinates) and
d_max is the cluster's max radius from seed. Particles farther from the
seed are brighter yellow; particles near the seed are darker.

Saves as .npz with key 'channels', shape (3, H, W), float32 in [0, 1].
This format is consistent with the existing DLAMultiChannelDataset loader.

Why this encoding:
    - The model has THREE coincident "seed is here" cues — all in one pixel:
      pure blue, no R/G presence above threshold elsewhere with high B,
      and the only point that breaks the R=G symmetry.
    - Distance-from-seed is encoded as YELLOW INTENSITY which is visually
      and computationally distinct from BLUE (the seed beacon).
    - Cross-channel consistency loss can enforce R=G for non-seed
      particles, since real DLAs have this exact relationship by data
      construction.

Usage:
    python render_rgb.py \\
        --particle_dir data/fixed_n1000_raw_v2/particles \\
        --output_dir data/fixed_n1000_rgb/images \\
        --metadata_dir data/fixed_n1000_rgb/metadata \\
        --image_size 512 --scale 2.0 --disc_radius 1
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


def render_rgb(particles, image_size, disc_radius=1, scale=2.0,
               centered=True):
    """
    Render particles as RGB image with seed-special encoding.

    Returns:
      r, g, b : uint8 (H, W) channels
      max_distance, seed_pixel
    """
    N = len(particles)
    r_ch = np.zeros((image_size, image_size), dtype=np.uint8)
    g_ch = np.zeros((image_size, image_size), dtype=np.uint8)
    b_ch = np.zeros((image_size, image_size), dtype=np.uint8)

    if N == 0:
        return r_ch, g_ch, b_ch, 0.0, (image_size // 2, image_size // 2)

    disc_mask = build_disc_mask(disc_radius)
    r = int(disc_radius)

    if centered:
        com = particles.mean(axis=0)
        offset_x = image_size / 2.0 - com[0] * scale
        offset_y = image_size / 2.0 - com[1] * scale
    else:
        offset_x = image_size / 2.0
        offset_y = image_size / 2.0

    seed = particles[0]
    dists = np.sqrt(np.sum((particles - seed) ** 2, axis=1))
    max_distance = float(dists.max())

    # Per-particle R=G value (uint8). Particle 0 (seed) handled separately.
    if max_distance > 0:
        rg_values = (dists / max_distance * 255).astype(np.uint8)
    else:
        rg_values = np.zeros(N, dtype=np.uint8)

    # The seed pixel position (record for metadata)
    seed_iy = int(round(seed[1] * scale + offset_y))
    seed_ix = int(round(seed[0] * scale + offset_x))

    disc_white_u8 = (disc_mask * np.uint8(255))

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

        if idx == 0:
            # SEED: (0, 0, 255). No R, no G, full B at the disc.
            np.maximum(b_ch[y0:y1, x0:x1],
                       (mask_slice * np.uint8(255)),
                       out=b_ch[y0:y1, x0:x1])
            # We deliberately leave r_ch and g_ch at 0 here.
        else:
            # NON-SEED PARTICLE: (rg, rg, 0)
            v = rg_values[idx]
            disc_v = (mask_slice * v).astype(np.uint8)
            np.maximum(r_ch[y0:y1, x0:x1], disc_v,
                       out=r_ch[y0:y1, x0:x1])
            np.maximum(g_ch[y0:y1, x0:x1], disc_v,
                       out=g_ch[y0:y1, x0:x1])
            # B stays 0 for non-seed.

    # If a non-seed particle's disc OVERLAPS the seed disc, the seed's pure
    # blue should win (we want the seed to remain visually unambiguous).
    # Re-stamp the seed last to enforce this and zero R, G inside the seed disc.
    sy0 = max(0, seed_iy - r)
    sy1 = min(image_size, seed_iy + r + 1)
    sx0 = max(0, seed_ix - r)
    sx1 = min(image_size, seed_ix + r + 1)
    sdy0 = sy0 - (seed_iy - r); sdy1 = sdy0 + (sy1 - sy0)
    sdx0 = sx0 - (seed_ix - r); sdx1 = sdx0 + (sx1 - sx0)
    if sy1 > sy0 and sx1 > sx0:
        seed_mask_local = disc_mask[sdy0:sdy1, sdx0:sdx1].astype(bool)
        # zero R and G under the seed
        r_slice = r_ch[sy0:sy1, sx0:sx1]; r_slice[seed_mask_local] = 0
        g_slice = g_ch[sy0:sy1, sx0:sx1]; g_slice[seed_mask_local] = 0
        # ensure B is 255 under the seed
        b_slice = b_ch[sy0:sy1, sx0:sx1]; b_slice[seed_mask_local] = 255

    return r_ch, g_ch, b_ch, max_distance, (seed_iy, seed_ix)


def compute_pixel_stats(r_ch, g_ch, b_ch, threshold=128):
    """White-pixel-count proxy: any channel above threshold."""
    binary = ((r_ch >= threshold) | (g_ch >= threshold) | (b_ch >= threshold))
    binary = binary.astype(np.uint8)
    coords = np.argwhere(binary > 0)
    n_white = len(coords)
    if n_white == 0:
        return 0, 0.0
    com = coords.mean(axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - com) ** 2, axis=1)))
    return n_white, float(rg)


def main():
    p = argparse.ArgumentParser(
        description="Render DLA particles to RGB-encoded npz")
    p.add_argument("--particle_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--metadata_dir", default=None)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--scale", type=float, default=2.0)
    p.add_argument("--disc_radius", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--also_save_png", action="store_true",
                   help="Also save a PNG preview alongside each npz")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.metadata_dir:
        os.makedirs(args.metadata_dir, exist_ok=True)

    bin_files = sorted(glob.glob(os.path.join(args.particle_dir, "*.bin")))
    if args.limit:
        bin_files = bin_files[:args.limit]

    print(f"Rendering {len(bin_files)} particle files (RGB seed-blue)")
    print(f"  image_size  = {args.image_size}")
    print(f"  scale       = {args.scale} px/unit")
    print(f"  disc_radius = {args.disc_radius} px")

    all_stats = []

    for i, bf in enumerate(bin_files):
        particles = load_particles(bf)
        r_ch, g_ch, b_ch, max_distance, (siy, six) = render_rgb(
            particles, args.image_size,
            disc_radius=args.disc_radius,
            scale=args.scale,
        )

        # (3, H, W) float32 in [0, 1]
        img_3ch = np.stack([r_ch, g_ch, b_ch], axis=0).astype(np.float32) / 255.0

        basename = os.path.splitext(os.path.basename(bf))[0]
        np.savez_compressed(
            os.path.join(args.output_dir, basename + ".npz"),
            channels=img_3ch,
        )
        if args.also_save_png:
            from PIL import Image
            rgb = np.stack([r_ch, g_ch, b_ch], axis=2)
            Image.fromarray(rgb, mode="RGB").save(
                os.path.join(args.output_dir, basename + ".png"))

        n_white, pixel_rg = compute_pixel_stats(r_ch, g_ch, b_ch)
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
            "seed_pixel": [siy, six],
        }
        all_stats.append(stat)
        if args.metadata_dir:
            with open(os.path.join(args.metadata_dir, basename + ".json"),
                      "w") as f:
                json.dump(stat, f, indent=2)

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(bin_files)}] N={len(particles)}, "
                  f"white_px={n_white}, R_g={pixel_rg:.1f}")

    if not all_stats:
        print("\nNo files to render.")
        return

    print(f"\n=== Summary (RGB seed-blue) ===")
    n_w = [s["white_pixel_count"] for s in all_stats]
    rg_s = [s["pixel_rg"] for s in all_stats]
    print(f"Total: {len(all_stats)}, white_px: {np.mean(n_w):.0f}±{np.std(n_w):.1f}, "
          f"R_g: {np.mean(rg_s):.2f}±{np.std(rg_s):.2f}")

    print("\n--- Last image channels ---")
    print(f"  R:  min={r_ch.min()}, max={r_ch.max()}")
    print(f"  G:  min={g_ch.min()}, max={g_ch.max()}")
    print(f"  B:  min={b_ch.min()}, max={b_ch.max()}")
    print(f"  B=255 pixel count: {(b_ch == 255).sum()}  (expect ~{int(np.sum(build_disc_mask(args.disc_radius)))})")
    print(f"  Seed pixel: ({siy}, {six})")


if __name__ == "__main__":
    main()
