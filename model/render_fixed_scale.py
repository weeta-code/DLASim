#!/usr/bin/env python3
"""
Re-render DLA particle data with fixed pixel-to-particle ratio.

Each particle is rendered as exactly 1 pixel at the location of its center,
rounded to the nearest grid point. This ensures:
  - White pixel count ≈ particle count N (bijection)
  - Pixel R_g ≈ particle R_g  
  - Consistent rendering regardless of cluster size

Usage:
    python render_fixed_scale.py \
        --particle_dir ../data/fixed_n1000_raw/particles \
        --output_dir ../data/fixed_n1000/images \
        --image_size 256
"""

import struct
import os
import glob
import argparse
import json
import numpy as np
from PIL import Image


def load_particles(path):
    """Load particles from binary file."""
    with open(path, 'rb') as f:
        n = struct.unpack('i', f.read(4))[0]
        data = np.frombuffer(f.read(n * 16), dtype=np.float64).reshape(n, 2)
    return data


def render_fixed_scale(particles, image_size, scale=1.0):
    """
    Render particles at fixed scale: 1 simulation unit = `scale` pixels.
    Each particle becomes exactly 1 pixel.
    Cluster is centered in the image.
    
    Returns:
        img: numpy array (image_size x image_size), uint8
        n_white: number of white pixels
        pixel_rg: R_g computed from white pixel centers
    """
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # Center the cluster in the image
    cx = image_size / 2.0
    cy = image_size / 2.0
    
    # Center of mass of particles
    com = particles.mean(axis=0)
    
    white_pixels = set()
    for px, py in particles:
        # Map particle to pixel: center cluster at image center, 1 unit = scale pixels
        ix = int(round(cx + (px - com[0]) * scale))
        iy = int(round(cy + (py - com[1]) * scale))
        if 0 <= ix < image_size and 0 <= iy < image_size:
            white_pixels.add((iy, ix))
    
    # Set white pixels
    for y, x in white_pixels:
        img[y, x] = 255
    
    # Compute pixel-based stats
    if len(white_pixels) > 0:
        coords = np.array(list(white_pixels), dtype=np.float64)  # (y, x)
        n_white = len(coords)
        pix_com = coords.mean(axis=0)
        pixel_rg = np.sqrt(np.mean(np.sum((coords - pix_com)**2, axis=1)))
    else:
        n_white = 0
        pixel_rg = 0.0
    
    return img, n_white, pixel_rg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--particle_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--metadata_dir", default=None,
                   help="Also save per-image metadata JSON here")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--scale", type=float, default=1.0,
                   help="Pixels per simulation unit (1.0 = 1 pixel per unit)")
    args = p.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.metadata_dir:
        os.makedirs(args.metadata_dir, exist_ok=True)
    
    bin_files = sorted(glob.glob(os.path.join(args.particle_dir, "*.bin")))
    print(f"Rendering {len(bin_files)} particle files at scale={args.scale} "
          f"px/unit, image_size={args.image_size}")
    
    all_stats = []
    
    for i, bf in enumerate(bin_files):
        particles = load_particles(bf)
        img, n_white, pixel_rg = render_fixed_scale(
            particles, args.image_size, args.scale)
        
        basename = os.path.splitext(os.path.basename(bf))[0]
        Image.fromarray(img).save(os.path.join(args.output_dir, basename + ".png"))
        
        stat = {
            "filename": basename,
            "particle_count": len(particles),
            "white_pixel_count": n_white,
            "pixel_rg": float(pixel_rg),
            "particle_rg": float(np.sqrt(np.mean(np.sum(
                (particles - particles.mean(axis=0))**2, axis=1)))),
        }
        all_stats.append(stat)
        
        if args.metadata_dir:
            with open(os.path.join(args.metadata_dir, basename + ".json"), "w") as f:
                json.dump(stat, f, indent=2)
        
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(bin_files)}] "
                  f"N={len(particles)}, white_px={n_white}, "
                  f"pixel_Rg={pixel_rg:.1f}, particle_Rg={stat['particle_rg']:.1f}")
    
    # Summary stats
    white_counts = [s["white_pixel_count"] for s in all_stats]
    pixel_rgs = [s["pixel_rg"] for s in all_stats]
    part_counts = [s["particle_count"] for s in all_stats]
    
    print(f"\n=== Rendering Summary ===")
    print(f"Total images: {len(all_stats)}")
    print(f"Particle count: {np.mean(part_counts):.0f} +/- {np.std(part_counts):.1f}")
    print(f"White pixels:   {np.mean(white_counts):.0f} +/- {np.std(white_counts):.1f}")
    print(f"Pixel R_g:      {np.mean(pixel_rgs):.1f} +/- {np.std(pixel_rgs):.1f}")
    print(f"Pixel/Particle ratio: {np.mean(white_counts)/np.mean(part_counts):.3f}")


if __name__ == "__main__":
    main()
