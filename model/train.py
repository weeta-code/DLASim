#!/usr/bin/env python3
"""
Train a DDPM on DLA cluster images from scratch.

Uses denoising-diffusion-pytorch (lucidrains) for the U-Net and diffusion
process. Single-channel grayscale images, no text conditioning.

Usage:
    python train.py --data_dir ../data/output/images --image_size 256 --epochs 300
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from ema_pytorch import EMA

from dataset import DLADataset

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Output directory ----
    # Use fixed run_name if provided (for resume), otherwise timestamp
    if args.run_name:
        run_dir = Path(args.output_dir) / args.run_name
    else:
        run_dir = Path(args.output_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Run directory: {run_dir}")

    # Handle multichannel flag
    if args.multichannel:
        args.channels = 3

    # Auto-resume from latest checkpoint in run_dir if no explicit resume
    if not args.resume and args.run_name:
        ckpts = sorted(ckpt_dir.glob("ckpt_epoch_*.pt"))
        if ckpts:
            args.resume = str(ckpts[-1])
            print(f"Auto-resuming from {args.resume}")

    # ---- Dataset ----
    if args.channels == 3:
        from dataset import DLAMultiChannelDataset
        dataset = DLAMultiChannelDataset(
            image_dir=args.data_dir,
            image_size=args.image_size,
            augment=True,
        )
    else:
        dataset = DLADataset(
            image_dir=args.data_dir,
            image_size=args.image_size,
            augment=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    # ---- Model ----
    unet = Unet(
        dim=args.model_dim,
        channels=args.channels,
        dim_mults=tuple(args.dim_mults),
        flash_attn=False,                   # safer compatibility
        self_condition=False,
    )

    diffusion = GaussianDiffusion(
        unet,
        image_size=args.image_size,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective=args.objective,           # 'pred_noise' or 'pred_v'
        beta_schedule=args.beta_schedule,   # 'linear' or 'cosine'
        min_snr_loss_weight=args.min_snr,   # min-SNR loss weighting
    ).to(device)

    param_count = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,} ({param_count/1e6:.1f}M)")

    # torch.compile for faster training (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        diffusion = torch.compile(diffusion)
        print("Model compiled")

    # ---- EMA ----
    ema = EMA(diffusion, beta=args.ema_decay)
    ema.to(device)

    # ---- Optimizer + Scheduler ----
    optimizer = AdamW(diffusion.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.01)

    # ---- AMP Scaler ----
    use_amp = args.amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("Mixed precision (AMP) enabled")

    # ---- Resume from checkpoint ----
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        diffusion.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        ema.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # ---- Training log ----
    log_path = run_dir / "train_log.jsonl"
    log_file = open(log_path, "a")

    accum_steps = args.grad_accum
    effective_batch = args.batch_size * accum_steps
    print(f"\nStarting training: {args.epochs} epochs, "
          f"batch_size={args.batch_size} x {accum_steps} accum = {effective_batch} effective, "
          f"lr={args.lr}")
    print(f"Saving checkpoints every {args.save_every} epochs")
    print(f"Generating samples every {args.sample_every} epochs\n")

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs):
        diffusion.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t_start = time.time()

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)

            with autocast(enabled=use_amp):
                loss = diffusion(batch)
                loss = loss / accum_steps  # scale for accumulation

            scaler.scale(loss).backward()

            # Step optimizer every accum_steps
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(diffusion.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                ema.update()

            epoch_loss += loss.item() * accum_steps  # unscale for logging
            epoch_steps += 1
            global_step += 1

        # Epoch stats
        avg_loss = epoch_loss / epoch_steps
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_start

        log_entry = {
            "epoch": epoch,
            "global_step": global_step,
            "loss": avg_loss,
            "lr": lr_now,
            "time_s": elapsed,
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        print(f"Epoch {epoch:4d} | loss={avg_loss:.5f} | "
              f"lr={lr_now:.2e} | {elapsed:.1f}s")

        # ---- Save checkpoint ----
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = ckpt_dir / f"ckpt_epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model": diffusion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": ema.state_dict(),
                "config": vars(args),
            }, ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")

        # ---- Generate samples ----
        if (epoch + 1) % args.sample_every == 0 or epoch == args.epochs - 1:
            generate_samples(ema.ema_model, sample_dir, epoch, device,
                             n=args.n_samples, image_size=args.image_size)

    log_file.close()
    print(f"\nTraining complete. Output: {run_dir}")


def generate_samples(model, sample_dir, epoch, device, n=16, image_size=256):
    """Generate and save a grid of samples using EMA model."""
    model.eval()
    with torch.no_grad():
        samples = model.sample(batch_size=n)  # [n, 1, H, W]

    # Save individual samples
    epoch_dir = sample_dir / f"epoch_{epoch:04d}"
    epoch_dir.mkdir(exist_ok=True)

    from torchvision.utils import save_image
    # Save grid
    save_image(samples, sample_dir / f"grid_epoch_{epoch:04d}.png",
               nrow=int(n ** 0.5), normalize=False)

    # Save individual images
    for i, s in enumerate(samples):
        save_image(s, epoch_dir / f"sample_{i:03d}.png", normalize=False)

    print(f"  -> Generated {n} samples at epoch {epoch}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train DDPM on DLA images")

    # Data
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory containing DLA training images")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--channels", type=int, default=1,
                   help="Number of image channels (1=grayscale, 3=multichannel)")
    p.add_argument("--multichannel", action="store_true", default=False,
                   help="Use multi-channel .npz dataset (sets channels=3)")

    # Model architecture
    p.add_argument("--model_dim", type=int, default=64,
                   help="Base channel dimension for U-Net")
    p.add_argument("--dim_mults", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="Channel multipliers per resolution level")
    p.add_argument("--timesteps", type=int, default=1000,
                   help="Number of diffusion timesteps")
    p.add_argument("--sampling_timesteps", type=int, default=250,
                   help="DDIM sampling steps (fewer = faster inference)")
    p.add_argument("--beta_schedule", type=str, default="cosine",
                   choices=["linear", "cosine", "sigmoid"],
                   help="Noise schedule (cosine often better for sparse images)")
    p.add_argument("--objective", type=str, default="pred_v",
                   choices=["pred_noise", "pred_x0", "pred_v"],
                   help="Prediction objective (v-prediction often best)")
    p.add_argument("--min_snr", action="store_true", default=True,
                   help="Use min-SNR loss weighting")
    p.add_argument("--no_min_snr", action="store_false", dest="min_snr")

    # Training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    p.add_argument("--num_workers", type=int, default=4)

    # Checkpointing / sampling
    p.add_argument("--output_dir", type=str, default="../runs",
                   help="Base directory for training runs")
    p.add_argument("--save_every", type=int, default=25,
                   help="Save checkpoint every N epochs")
    p.add_argument("--sample_every", type=int, default=25,
                   help="Generate samples every N epochs")
    p.add_argument("--n_samples", type=int, default=16,
                   help="Number of samples to generate")

    # AMP
    p.add_argument("--amp", action="store_true", default=True,
                   help="Use automatic mixed precision")
    p.add_argument("--no_amp", action="store_false", dest="amp")

    # torch.compile
    p.add_argument("--compile", action="store_true", default=False,
                   help="Use torch.compile for faster training")

    # Resume
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--run_name", type=str, default=None,
                   help="Fixed run directory name (enables auto-resume from latest checkpoint)")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
