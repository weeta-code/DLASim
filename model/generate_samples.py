#!/usr/bin/env python3
"""
Generate DLA samples from a trained DDPM checkpoint.

Usage:
    python generate_samples.py --checkpoint ../runs/run_XXX/checkpoints/ckpt_epoch_0299.pt \
                               --n_samples 64 --output_dir ../results/generated
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torchvision.utils import save_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from ema_pytorch import EMA


def load_model(ckpt_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    unet = Unet(
        dim=config["model_dim"],
        channels=1,
        dim_mults=tuple(config["dim_mults"]),
        flash_attn=False,
        self_condition=False,
    )

    diffusion = GaussianDiffusion(
        unet,
        image_size=config["image_size"],
        timesteps=config["timesteps"],
        sampling_timesteps=config.get("sampling_timesteps", 250),
        objective="pred_noise",
    ).to(device)

    # Load EMA weights (preferred for generation)
    if "ema" in ckpt:
        ema = EMA(diffusion, beta=config.get("ema_decay", 0.9999))
        ema.to(device)
        ema.load_state_dict(ckpt["ema"])
        model = ema.ema_model
        print("Loaded EMA model weights")
    else:
        diffusion.load_state_dict(ckpt["model"])
        model = diffusion
        print("Loaded model weights (no EMA found)")

    return model, config


def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, config = load_model(args.checkpoint, device)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = out_dir / "individual"
    individual_dir.mkdir(exist_ok=True)

    print(f"Generating {args.n_samples} samples...")

    all_samples = []
    batch_size = min(args.batch_size, args.n_samples)

    with torch.no_grad():
        remaining = args.n_samples
        idx = 0
        while remaining > 0:
            n = min(batch_size, remaining)
            samples = model.sample(batch_size=n)  # [n, 1, H, W]
            all_samples.append(samples.cpu())

            for s in samples:
                save_image(s, individual_dir / f"sample_{idx:04d}.png",
                           normalize=False)
                idx += 1

            remaining -= n
            print(f"  Generated {idx}/{args.n_samples}")

    all_samples = torch.cat(all_samples, dim=0)

    # Save grid (up to 64 images)
    grid_n = min(64, len(all_samples))
    nrow = int(grid_n ** 0.5)
    save_image(all_samples[:grid_n], out_dir / "sample_grid.png",
               nrow=nrow, normalize=False)

    # Save generation config
    gen_info = {
        "checkpoint": str(args.checkpoint),
        "n_samples": args.n_samples,
        "training_config": config,
    }
    with open(out_dir / "generation_info.json", "w") as f:
        json.dump(gen_info, f, indent=2)

    print(f"Done. Samples saved to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate DLA samples from trained DDPM")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output_dir", type=str, default="../results/generated")
    generate(p.parse_args())
