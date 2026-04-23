#!/usr/bin/env python3
"""
Quick Python DLA simulator for local testing of the training pipeline.
NOT for production data generation (use the C++ simulator for that).

Generates a small set of DLA images using a simple off-lattice random walk.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from PIL import Image
from collections import defaultdict


def spatial_hash_key(x, y, cell_size):
    return (int(np.floor(x / cell_size)), int(np.floor(y / cell_size)))


def simulate_dla(num_particles, seed=42, particle_radius=1.0):
    """Simple off-lattice DLA simulation with spatial hashing."""
    rng = np.random.RandomState(seed)
    sticking_dist = 2.0 * particle_radius
    sticking_dist_sq = sticking_dist ** 2
    cell_size = sticking_dist
    fine_step = particle_radius

    particles = [(0.0, 0.0)]  # seed at origin
    grid = defaultdict(list)
    grid[spatial_hash_key(0, 0, cell_size)].append(0)
    max_radius = 0.0

    def check_sticking(x, y):
        cx, cy = spatial_hash_key(x, y, cell_size)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                key = (cx + dx, cy + dy)
                if key in grid:
                    for idx in grid[key]:
                        px, py = particles[idx]
                        if (x - px)**2 + (y - py)**2 < sticking_dist_sq:
                            return True
        return False

    for i in range(num_particles):
        spawn_r = max_radius + 20.0
        kill_r = spawn_r * 3.0

        stuck = False
        while not stuck:
            angle = rng.uniform(0, 2 * np.pi)
            x = spawn_r * np.cos(angle)
            y = spawn_r * np.sin(angle)

            while True:
                dist = np.sqrt(x*x + y*y)
                if dist > kill_r:
                    break

                gap = dist - max_radius - sticking_dist
                to_kill = kill_r - dist
                safe = min(gap, to_kill)

                if safe > 4 * fine_step:
                    hop_r = safe - fine_step
                    hop_a = rng.uniform(0, 2 * np.pi)
                    x += hop_r * np.cos(hop_a)
                    y += hop_r * np.sin(hop_a)
                else:
                    walk_a = rng.uniform(0, 2 * np.pi)
                    x += fine_step * np.cos(walk_a)
                    y += fine_step * np.sin(walk_a)

                if check_sticking(x, y):
                    idx = len(particles)
                    particles.append((x, y))
                    grid[spatial_hash_key(x, y, cell_size)].append(idx)
                    r = np.sqrt(x*x + y*y)
                    if r > max_radius:
                        max_radius = r
                    stuck = True
                    break

    return particles, max_radius


def render_image(particles, max_radius, image_size=256, particle_radius=1.0):
    """Render particles as a grayscale image."""
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    extent = max_radius + 10.0
    scale = (image_size * 0.9) / (2.0 * extent)
    cx = cy = image_size / 2.0

    for px, py in particles:
        ix = int(cx + px * scale)
        iy = int(cy + py * scale)
        r = max(1, int(particle_radius * scale))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx*dx + dy*dy <= r*r:
                    x, y = ix + dx, iy + dy
                    if 0 <= x < image_size and 0 <= y < image_size:
                        img[y, x] = 255
    return img


def main():
    p = argparse.ArgumentParser(description="Generate test DLA images (Python)")
    p.add_argument("--particles", type=int, default=1000,
                   help="Particles per cluster")
    p.add_argument("--size", type=int, default=256,
                   help="Image size")
    p.add_argument("--count", type=int, default=20,
                   help="Number of images")
    p.add_argument("--outdir", type=str, default="../data/test_output",
                   help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    img_dir = os.path.join(args.outdir, "images")
    meta_dir = os.path.join(args.outdir, "metadata")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    print(f"Generating {args.count} DLA images ({args.particles} particles, "
          f"{args.size}x{args.size})")

    for i in range(args.count):
        seed = args.seed + i
        t0 = time.time()

        particles, max_r = simulate_dla(args.particles, seed=seed)
        img = render_image(particles, max_r, args.size)

        elapsed = (time.time() - t0) * 1000

        # Save image
        Image.fromarray(img).save(os.path.join(img_dir, f"dla_{i:05d}.png"))

        # Save metadata
        coords = np.array(particles)
        com = coords.mean(axis=0)
        rg = np.sqrt(np.mean(np.sum((coords - com)**2, axis=1)))

        meta = {
            "particle_count": len(particles),
            "max_radius": float(max_r),
            "radius_of_gyration": float(rg),
            "center_of_mass": [float(com[0]), float(com[1])],
            "simulation_time_ms": elapsed,
            "seed": seed,
        }
        with open(os.path.join(meta_dir, f"dla_{i:05d}.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  [{i+1}/{args.count}] {len(particles)} particles, "
              f"R={max_r:.1f}, Rg={rg:.1f}, {elapsed:.0f}ms")

    print(f"\nDone. Output: {args.outdir}")


if __name__ == "__main__":
    main()
