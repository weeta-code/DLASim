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
import numpy as np
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
            # Continuous random rotation (DLA has full SO(2) symmetry, not just D4)
            import torchvision.transforms.functional as TF
            angle = random.uniform(0, 360)
            img = TF.rotate(img, angle,
                            interpolation=TF.InterpolationMode.BILINEAR,
                            fill=0)
            # Random horizontal flip
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])

        return img


class DLAMultiChannelDataset(Dataset):
    """Dataset of multi-channel DLA images stored as .npz files.
    
    Each .npz contains key 'channels' with shape (3, H, W) float32 in [0, 1]:
      Channel 0: binary presence
      Channel 1: particle deposition order (normalized)
      Channel 2: distance from seed (normalized)
    """

    def __init__(self, image_dir, image_size=512, augment=True, cache_in_memory=True):
        self.paths = sorted(glob.glob(os.path.join(image_dir, "*.npz")))
        if not self.paths:
            raise ValueError(f"No .npz files found in {image_dir}")
        
        self.image_size = image_size
        self.augment = augment
        
        self.cache = None
        if cache_in_memory:
            t0 = time.time()
            print(f"DLAMultiChannelDataset: caching {len(self.paths)} files in memory...", flush=True)
            tensors = []
            for p in self.paths:
                arr = np.load(p)['channels']  # (3, H, W) float32
                t = torch.from_numpy(arr)     # (3, H, W)
                # Resize if needed
                if t.shape[1] != image_size or t.shape[2] != image_size:
                    t = torch.nn.functional.interpolate(
                        t.unsqueeze(0), size=(image_size, image_size),
                        mode='bilinear', align_corners=False
                    ).squeeze(0)
                tensors.append(t)
            self.cache = torch.stack(tensors)  # (N, 3, H, W)
            elapsed = time.time() - t0
            mem_mb = self.cache.nbytes / 1e6
            print(f"DLAMultiChannelDataset: cached {len(self.paths)} files ({mem_mb:.0f} MB) in {elapsed:.1f}s")
        else:
            print(f"DLAMultiChannelDataset: {len(self.paths)} files from {image_dir}, "
                  f"size={image_size}, augment={augment}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if self.cache is not None:
            img = self.cache[idx].clone()  # (3, H, W)
        else:
            arr = np.load(self.paths[idx])['channels']
            img = torch.from_numpy(arr)
            if img.shape[1] != self.image_size or img.shape[2] != self.image_size:
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(self.image_size, self.image_size),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
        
        if self.augment:
            # Continuous random rotation (SO(2) symmetry)
            import torchvision.transforms.functional as TF
            angle = random.uniform(0, 360)
            img = TF.rotate(img, angle,
                           interpolation=TF.InterpolationMode.BILINEAR,
                           fill=0)
            # Random horizontal flip
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])
        
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
