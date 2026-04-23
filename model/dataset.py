"""
DLA Image Dataset

Loads grayscale DLA images (single channel) with data augmentation
appropriate for isotropic fractal structures.

Supports in-memory caching to eliminate NFS I/O bottleneck during training.
"""

import os
import glob
import random
import json
import time
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class DLADataset(Dataset):
    """Dataset of DLA cluster images with optional metadata."""

    def __init__(self, image_dir, image_size=256, augment=True,
                 metadata_dir=None, cache_in_memory=True):
        """
        Args:
            image_dir: Path to directory containing .png DLA images
            image_size: Target image size (will resize + center crop)
            augment: Whether to apply random rotations/flips
            metadata_dir: Optional path to directory with per-image .json metadata
            cache_in_memory: If True, pre-load all images into RAM at init.
                             Eliminates disk I/O during training (~1.3GB for 5000 256x256 images).
        """
        self.paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if not self.paths:
            raise ValueError(f"No PNG images found in {image_dir}")

        self.image_size = image_size
        self.augment = augment
        self.metadata_dir = metadata_dir

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),  # [0, 255] -> [0.0, 1.0], shape [1, H, W] for grayscale
        ])

        # Pre-cache all images in memory as a single tensor
        self.cache = None
        if cache_in_memory:
            t0 = time.time()
            print(f"DLADataset: caching {len(self.paths)} images in memory...", flush=True)
            tensors = []
            for p in self.paths:
                img = Image.open(p).convert("L")
                tensors.append(self.transform(img))
            self.cache = torch.stack(tensors)  # [N, 1, H, W]
            elapsed = time.time() - t0
            mem_mb = self.cache.nbytes / 1e6
            print(f"DLADataset: cached {len(self.paths)} images ({mem_mb:.0f} MB) in {elapsed:.1f}s")
        else:
            print(f"DLADataset: {len(self.paths)} images from {image_dir}, "
                  f"size={image_size}, augment={augment}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.cache is not None:
            img = self.cache[idx].clone()  # [1, H, W], clone so augmentation doesn't mutate cache
        else:
            img = Image.open(self.paths[idx]).convert("L")
            img = self.transform(img)

        if self.augment:
            # Random 90-degree rotations (DLA is statistically isotropic)
            k = random.randint(0, 3)
            if k > 0:
                img = torch.rot90(img, k, dims=[1, 2])
            # Random horizontal flip
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])
            # Random vertical flip
            if random.random() > 0.5:
                img = torch.flip(img, dims=[1])

        return img

    def get_metadata(self, idx):
        """Load per-image metadata JSON if available."""
        if self.metadata_dir is None:
            return None
        basename = os.path.splitext(os.path.basename(self.paths[idx]))[0]
        meta_path = os.path.join(self.metadata_dir, basename + ".json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                return json.load(f)
        return None
