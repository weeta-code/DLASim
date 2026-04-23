#!/usr/bin/env python3
"""
Generate many samples from a trained checkpoint for robust statistics.
"""

import argparse
import os
import time
import torch
from torchvision.utils import save_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from ema_pytorch import EMA


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
        objective=config.get("objective", "pred_noise"),
        beta_schedule=config.get("beta_schedule", "linear"),
        min_snr_loss_weight=config.get("min_snr", False),
    ).to(device)

    # Load EMA
    ema = EMA(diffusion, beta=config.get("ema_decay", 0.9999))
    ema.to(device)
    ema.load_state_dict(ckpt["ema"])
    model = ema.ema_model
    model.eval()

    print(f"Generating {args.n_samples} samples (batch={args.batch_size})...")
    idx = 0
    remaining = args.n_samples
    t0 = time.time()
    with torch.no_grad():
        while remaining > 0:
            n = min(args.batch_size, remaining)
            samples = model.sample(batch_size=n)
            for s in samples:
                save_image(s, os.path.join(args.output_dir, f"sample_{idx:04d}.png"),
                           normalize=False)
                idx += 1
            remaining -= n
            elapsed = time.time() - t0
            print(f"  Generated {idx}/{args.n_samples} ({elapsed:.0f}s)")

    print(f"Done in {time.time()-t0:.0f}s. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
