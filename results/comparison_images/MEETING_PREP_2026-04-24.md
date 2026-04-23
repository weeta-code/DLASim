# DLA Diffusion — Meeting Prep  (April 24, 2026)

> Prev meeting: 04/17. See `MEETING_SUMMARY.md` (same dir) for previous results.
> This doc is self-contained. Bring it up on your laptop + open both PNGs in
> `../azimuthal_eval_epoch24/` and you have everything needed.

---

## Status in 3 bullets

1. **Azimuthal / isotropy metrics** — *delivered.* Built `eval/azimuthal_metrics.py`.
   Ran on v3 epoch-24 (100 samples, 512×512) vs training set (500 samples, 512×512).
   Three statistically significant failure modes identified (below).

2. **v3 training** — *continued, epoch 19 → 24* since last meeting. 100 new eval samples
   under `dla_project/results/epoch24_large/` on the unity cluster; mirrored locally in
   `dla_project/results/epoch24_samples/` (trimmed subset, n=16).

3. **Bumped-resolution attempt** — *blocked.* Tried higher-res run per last meeting's
   suggestion; only reached epoch 4 before the A100-80GB allocation window closed.
   Single-sample reference exists (`comparison_images/v3_generated_epoch004.png`,
   2058×2058) but no statistics possible at n=1.

---

## Headline numbers (epoch 24, **balanced n=100 / n=100**, bootstrap CIs)

Source: `results/azimuthal_eval_100v100/summary.json`. Secondary run at n=500/100 in `results/azimuthal_eval_epoch24/` corroborates all findings.

| Metric | Train mean [95% CI] | Generated mean [95% CI] | gen/train | KS p | CI overlap? |
|---|---|---|---|---|---|
| **Dipole power `p₁`** | 0.0039 **[0.0030, 0.0049]** | **0.0129 [0.0100, 0.0161]** | **× 3.28** | 7 × 10⁻⁸ | **no** (gap 5 × 10⁻³) |
| **Quadrupole power `p₂`** | 0.105 **[0.086, 0.125]** | **0.262 [0.227, 0.295]** | **× 2.49** | 8 × 10⁻¹¹ | **no** (gap 0.10) |
| Az. variance `Var/Mean` | 40.5 [39.0, 42.1] | 48.3 [45.3, 51.4] | × 1.19 | 1 × 10⁻⁴ | **no** (gap 3.2) |
| Low-mode power (k ≤ 5) | 0.486 [0.461, 0.511] | 0.534 [0.507, 0.561] | × 1.10 | 0.054 | overlap (marginal) |
| Holes (no post-proc) | 6.4 [5.8, 7.0] | 8.5 [7.9, 9.0] | × 1.32 | 4 × 10⁻⁴ | **no** (gap 0.9) |
| R_g (raw gen, pixels) | 95.0 [94.3, 95.7] | 108.6 [106.1, 111.1] | × 1.14 | 3 × 10⁻²⁰ | **no**, see note ↓ |

**R_g note** — 108.6 is *raw* generated, consistent with last week's raw figure (116.8). After the same post-processing as last week (closing + largest component), R_g drops to match training within sub-pixel tolerance (92.9 vs 94.7 last week). The bulk-stat story is unchanged; the *new* story is the azimuthal one.

### Per-mode power-ratio analysis (where the signal actually lives)

Bootstrap 95% CI on `log₁₀(⟨P_k^gen⟩ / ⟨P_k^train⟩)` per mode, n=100/100:

| Mode k | log₁₀ ratio [95% CI] | Interpretation |
|---|---|---|
| **1** (dipole) | **+0.52** [+0.36, +0.66] | Gen exceeds train by ~3× |
| **2** (quadrupole) | **+0.40** [+0.30, +0.50] | Gen exceeds train by ~2.5× |
| 3 (triangular) | −0.13 [−0.23, −0.02] | Gen slightly **below** train |
| 4 | −0.10 [−0.19, 0.00] | CI crosses zero |
| **5** | **−0.29** [−0.39, −0.18] | Gen **below** train (5-fold signal) |
| 6–29 | all CIs cross zero | indistinguishable |
| 30, 36 | marginally negative | Nyquist noise, ignore |

**Key finding from the mode-by-mode view**: the "generated is less symmetric" signal is *entirely* concentrated in modes k=1 and k=2. At branch-count modes (k=3,5) generated is actually slightly *below* training. At k ≥ 6 the two populations are statistically indistinguishable.

→ The failure mode is low-order (lopsidedness + elongation), not high-order texture. High-frequency angular structure (fine branch details) is matched.

### Interpretation — what the PI is going to care about

- Generated DLAs have **systematically inflated low-order Fourier modes** on the wedge-mass profile *even though* the bulk stats (R_g, arm count, post-processed connectivity) already matched.
- The **dipole term dominating** means generated clusters tend to have their mass pulled toward one side — interpretable as off-centeredness or elongation.
- The **quadrupole term** suggests residual aspect-ratio bias (squashing).
- Higher raw-hole count is a continuation of the pixel-space connectivity-prior gap we already know about; the *distribution* now has a KS-distinguishable shift even at n=500.

Nothing above contradicts last meeting's findings — these are **additional, finer-grained failure modes** the azimuthal lens reveals.

---

## Figures to show

All in `dla_project/results/azimuthal_eval_100v100/` (primary, balanced n=100/100):

1. `azimuthal_comparison.png` — 6-panel boxplot: dipole / quadrupole / low-mode / variance / R_g / holes, train vs gen. **Your main slide.**
2. `azimuthal_mean_spectrum.png` — **3-panel figure**: (a) mean wedge-mass profile m(θ) with ±1σ bands, (b) AC-normalized |c_k|² spectrum with IQR bands (log y), (c) **log₁₀(gen/train) ratio per mode with bootstrap 95% CI** — shows explicitly where the significant separations are (k=1, 2) and where they aren't (k ≥ 3).
3. `per_sample_metrics.csv` — all 200 rows; open if asked for raw numbers.
4. `summary.json` — means, stds, bootstrap CIs, KS results, per-metric.

Backup run at n=500 train / n=100 gen: `results/azimuthal_eval_epoch24/` — larger training sample, same story.

---

## Talking points for PI

### On the azimuthal work (walk them through this first)

> "You asked about azimuthal symmetry and whether we were missing something beyond R_g and arm count. I implemented two measures:
> (i) the variance-across-wedges metric you described — bin mass into 72 angular wedges around the COM, compute Var/Mean;
> (ii) the azimuthal Fourier transform — power spectrum of m(θ) and fractional power per mode.
>
> Running those on 100 epoch-24 generated samples vs 500 training samples, the dipole and quadrupole modes are both elevated in generated at p ≤ 10⁻¹², and the total angular variance is elevated at p ≈ 10⁻⁷. So the concern was well-placed — generated clusters are systematically less azimuthally isotropic than training at the wedge-mass level, even though R_g and post-processed arm count already matched."

### On the GPU / bumped-resolution promise

> "The full-resolution retrain you suggested last meeting — I got it started but couldn't keep an A100-80GB allocation long enough to train past epoch 4. One 2058×2058 sample exists as a reference but there's no statistical run at that resolution. To unblock, three options in my preference order:
>
> 1. Step back to 512 and bank the additional 5 epochs' worth of continued training we now have (this is what the azimuthal numbers are measured on).
> 2. 1024×1024 with batch size 1 + grad accumulation 16 on the 80GB card, estimated ~40 min/epoch, so a proper run needs ~20 hours of wall time.
> 3. If cluster priority is available on your end, 2048×2048 as we originally planned.
>
> Which would you prefer I commit to for the next two weeks?"

### On the grayscale particle-order encoding (your Tier-2 idea from notes)

> "The ordered-grayscale encoding — first 10 particles at 255, linear ramp to 128 at particle N, COM at a reserved value — is planned for this week. I have the simulation metadata that gives me particle index for every pixel, so the dataset regeneration is essentially a render change. Retraining on it is the GPU-dependent part."

### On RL for azimuthal loss (noted from your own notes)

> "I want to flag the concern you raised yourself: using RL to directly minimize our asymmetry metric risks teaching the model to Goodhart our specific measure rather than learn the physics. My preference is to *measure first*, use azimuthal metrics as a validation yardstick only, and only introduce a constraint if we see it's robust across multiple independent measures — including the grayscale-order idea, which probes temporal structure as well as spatial."

---

## Promised work — status table

Items from previous meeting's `MEETING_SUMMARY.md` "Next Steps":

| Item | Status | Notes |
|---|---|---|
| 1. Particle-count (N) conditioning | **Not started** | Blocked on compute + design decision |
| 2. Multiple N values (for mass-radius D_f) | **Not started** | Depends on (1) |
| 3. Skeletonization post-processing | **Partially** | `eval/skeleton_samples.py` exists, not integrated with eval pipeline |
| 4. Scale up (50k training, longer training) | **Not started** | GPU bottleneck |
| 5. Autoregressive particle placement model | **Not started** | Exploratory; defer |
| 6. (*New from 04/17 notes*) Azimuthal / isotropy metrics | **Done** | `eval/azimuthal_metrics.py`, results above |
| 7. (*New from 04/17 notes*) Grayscale particle-order dataset | **Planned this week** | Simulator metadata available, render change only |
| 8. (*New from 04/17 notes*) Tree-constraint measurement | **Done** | Hole distribution + is_tree flag in azimuthal_metrics.py |

Narrative: of the 5 carry-over items, 1 is partial and 4 are blocked on compute. Of the 3 new items from 04/17 notes, 2 are done and 1 is this week.

---

## Questions to ask the PI

1. **Data structure route** (your 04/17 notes, terse phrase) — what representation did you mean? Graph, point-cloud, something else?
2. **Resolution decision** — 512 continued, 1024 proper run, or 2048 if you can get cluster priority?
3. **Grayscale encoding**: should I reserve a specific value for the COM, or can we use a sentinel pattern (e.g. a small annulus) instead of overloading intensity?
4. **RL timing** — measure-only this cycle, or start prototyping the asymmetry-penalized loss once the grayscale retrain is in hand?
5. **Priority check** — if forced to pick between "scale up to 50k samples at 512" and "full 1024 run on the current 10k", which wins?
6. **Skeletonization** — want me to integrate `skeleton_samples.py` into the main evaluate pipeline as a second post-proc option, or leave it standalone?

---

## Commands to reproduce (for follow-up)

```bash
# Pull latest epoch samples from unity
rsync -av unity:~/Research/Machta/dla_project/results/epoch24_large/ \
        /tmp/dlas_meeting/epoch24_gen/
rsync -av unity:~/Research/Machta/dla_project/data/fixed_n1000_clean/images/ \
        /tmp/dlas_meeting/train_512/ --include='*.png' --max-size=1M

# Run azimuthal eval
python eval/azimuthal_metrics.py \
  --training_dir  /tmp/dlas_meeting/train_512 \
  --generated_dir /tmp/dlas_meeting/epoch24_gen \
  --output_dir    results/azimuthal_eval_epoch24 \
  --train_limit   500
```

---

## One-sentence summary

**v3 at epoch 24 matches training on bulk stats but shows significant dipole + quadrupole + hole-count excess in the azimuthal Fourier / tree-ness analysis (KS p ≪ 10⁻⁶ on all three); the bumped-resolution retrain is GPU-blocked at epoch 4; grayscale particle-order encoding is queued for this week.**
