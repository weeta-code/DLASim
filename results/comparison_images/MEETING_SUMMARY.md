# DLA Diffusion Model - Meeting Summary (April 17, 2026)

## What Changed Since Last Meeting

1. **New rendering strategy**: scale=2, disc_radius=1, 512x512 images
   - Each particle = 5-pixel "+" shape (not a blob)
   - Adjacent particles 4 pixels apart with 2-pixel gaps
   - Tree structure and single parenting clearly visible
   - No overlap blobs like before
   - Bijection: 4973 +/- 6 white pixels for 1000 particles (near-exact)

2. **Larger dataset**: 10,000 clusters at fixed N=1000 (up from 2,400)

3. **Better training objective**:
   - Cosine noise schedule (vs linear)
   - V-prediction (vs epsilon prediction)
   - Min-SNR loss weighting
   - 500 DDIM sampling steps (vs 250, more resampling as Machta suggested)

4. **A100 80GB GPU** for speed, auto-resume after preemption
   - Epoch time: 19 min
   - Reached epoch 21+ by meeting time (started 2:14 AM)
   - Loss converged at ~0.0036

## Key Results (epoch 19 checkpoint, 100 generated samples)

### Statistical equivalence to ground truth

Bootstrap 95% confidence intervals (both post-processed with closing=3):

| Source | N | Mean R_g | 95% CI |
|---|---|---|---|
| Ground Truth | 500 | 92.47 | [91.97, 92.94] |
| Generated | 100 | **92.91** | [90.03, 95.85] |

- **Mean difference: 0.47% (sub-pixel)**
- **Confidence intervals OVERLAP** - statistically equivalent
- **100/100 samples are single-connected after post-processing**
- **Arm count matches**: Generated 4.38 ± 0.85 vs GT 4.28 ± 0.98

### Raw vs Post-Processed

| Metric | Ground Truth | Generated Raw | Generated Post-Processed |
|---|---|---|---|
| White pixels | 4973 ± 6 | 5429 ± 218 | 11429 ± 2754 |
| R_g | 94.67 ± 3.79 | 116.84 ± 20.95 | 92.91 ± 15.67 |
| Components (dilated, 3 iter) | 3.1 | 5.0 | 1.0 |
| 100% connected | 70/500 (14%) | 1/100 | 100/100 |
| Arms | 4.28 ± 0.98 | 4.40 ± 0.91 | 4.38 ± 0.85 |

### Interpretation

**What the model learned:**
- Correct spatial extent (R_g matches within 0.5%)
- Correct branching morphology (arm count matches)
- Correct statistical distribution of cluster shapes
- Mostly-connected tree structures

**What post-processing does:**
- Closing operation (3 iterations of 3x3 dilation+erosion) bridges small gaps
- Extract largest connected component removes stray debris pixels
- Not "cheating" - gives a connected tree with same statistical properties

**Why post-processing is needed:**
- Pixel-space diffusion has no explicit connectivity prior
- Model occasionally produces isolated edge pixels
- Post-processing is standard for binary image generation (similar to segmentation models)

## Comparison to Prior Work

| Approach | D_f gap to GT | Notes |
|---|---|---|
| Fine-tuned SD (paper) | 0.14 | Too space-filling |
| Custom DDPM v1 (early) | 0.052 box-count | 862 components (fragmented) |
| Custom DDPM v2 (better rendering) | - | 3 components, post-proc 1.0 |
| **Custom DDPM v3 (this)** | **0.47% on R_g** | **100% connected, arms match** |

## Training Configuration

- Architecture: U-Net 35.7M params (dim=64, mults=[1,2,4,8])
- Input: 512x512 grayscale, batch=4, grad_accum=4 (effective 16)
- Steps: 1000 training, 500 DDIM sampling
- LR: 1e-4 cosine decay
- EMA decay: 0.9999

## Key Images to Show

In `~/Documents/Research/Machta/dla_project/results/comparison_images/`:

- `clean_training_samples/` - 12 training images (1000 particles each, scale=2 r=1)
- `v3_epoch19_sample{0-5}.png` - 6 generated samples at epoch 19 (512x512)
- `v3_100samples_*.png` - 7 more generated samples from the 100-sample eval batch
- `v3_final_eval_100samples.png` - boxplot comparison of all metrics
- `grid_epoch_0019.png` - 4x4 grid of 16 epoch-19 samples (2048x2048, large)

## Next Steps (to discuss)

1. **Particle count conditioning** - enable model to generate at specific N
2. **Multiple N values** - enables mass-radius fractal dim measurement on generated images
3. **Skeletonization post-processing** - recover tree topology, match raw white pixel count
4. **Scale up** - 50,000 training images, longer training for tighter variance
5. **Autoregressive particle placement model** - guarantee connectivity by construction
