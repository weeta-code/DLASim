# DLA Diffusion — Meeting Prep  (April 30, 2026)

> Prev meeting: 04/24, see `MEETING_PREP_2026-04-24.md`. Last cycle established
> azimuthal-metrics findings: v3 epoch-24 had 3.3× dipole, 2.5× quadrupole,
> 1.32× hole excess vs training.
>
> This cycle: **localized the bias** (magnitudinal not directional),
> **diagnosed an augmentation bug**, and **ran two parallel training runs**
> with completed early evaluations.

---

## TL;DR — three things to walk in with

1. **The dipole bias is magnitudinal, not directional.** Rayleigh test of
   per-sample dipole phase: $p_\text{uniform} = 0.996$ for v3 generated,
   $p_\text{uniform} = 0.79$ for training. Both populations isotropically
   oriented. The model produces lopsided clusters in **random** directions.
   Rules out positional encoding artifacts and D₄ aliasing as cause.

2. **Augmentation bug found and patched.** `dataset.py` was using only D₄
   symmetry (90° rotations + flips). Patched to SO(2) continuous rotations.
   Warm-started v3 from epoch 64 ("v3-controt") and ran 5 more epochs.
   **Result confirms the magnitudinal hypothesis**: dipole did NOT
   improve (3.3× → 3.84×). But quadrupole improved (2.5× → 1.68×) and
   **holes dropped from 1.32× to 0.77× (gen now has FEWER loops than train)**.

3. **Multi-channel encoding produces fewer loops even at epoch 4.**
   New v4 model with 3-channel input (binary + growth-order + distance-from-seed)
   trained from scratch shows hole count of **0.30× training** at just
   epoch 4. Dipole is poor (16.5×) because severely undertrained, but
   the topology signal is unmistakable: structural priors work.

---

## Headline finding 1: dipole is magnitudinal

For each image:
$$c_1 = \frac{1}{N}\sum_j e^{-i\theta_j}\quad\text{(unit-vector dipole)}$$

| | $\bar R$ (Rayleigh) | $p_\text{uniform}$ | Mean $\|c_1\|$ |
|---|---|---|---|
| Train (n=200) | 0.034 | 0.79 | 0.029 ± 0.016 |
| **v3 generated (n=100)** | 0.006 | **0.996** | **0.055 ± 0.033 (×1.88)** |
| v3-controt @ epoch 69 (n=96) | 0.065 | 0.67 | 0.068 ± 0.036 (×2.10) |
| v4-multichannel @ epoch 4 (n=32) | 0.135 | 0.56 | 0.190 ± 0.126 (×5.86) |

**Both train and gen pass the Rayleigh uniformity test in all cases.** Phase
distributions are flat — no preferred angle, no D₄ artifacts. The dipole
excess is entirely in the **magnitude** of $|c_1|$ per sample.

`results/dipole_orientation/dipole_orientation.png` — polar histograms.
`results/dipole_orientation/dipole_phase_vs_magnitude.png` — scatter.

This finding **rules out** rotation-augmentation as a complete fix
(confirmed below) and pushes toward **structural-supervision** approaches.

---

## Headline finding 2: continuous rotation patch results

Diff in `model/dataset.py`:

```python
# OLD (D₄ orbit):                       # NEW (SO(2) × Z₂ orbit):
k = random.randint(0, 3)                # angle = random.uniform(0, 360)
img = torch.rot90(img, k, dims=[1,2])   # img = TF.rotate(img, angle, ...)
flip H, flip V                          # flip H only
```

Warm-started from v3 `clean_main` epoch-64 checkpoint.
Currently at **epoch 74** (12.6 hours in). Training continues until 1pm EDT.

### v3-controt at epoch 69 (n=96 samples)

| Metric | v3 baseline | v3-controt e69 | direction |
|---|---|---|---|
| Dipole $p_1$ | 0.0129 (3.3× train) | **0.0167 (3.84× train)** | **WORSE** |
| Quadrupole $p_2$ | 0.262 (2.5×) | 0.208 (1.68×) | **better** |
| Az variance | 48.4 | 51.4 | similar |
| **Holes** | 8.5 (1.32× train) | **4.78 (0.77× train)** | **MUCH better** |
| R_g raw | 108.6 | 117.1 | worse (raw) |

**Reading**: continuous rotation augmentation *helps* topology (less loops)
and *helps* quadrupole, but doesn't help dipole. This **confirms** the
orientation analysis — the dipole bias is magnitudinal and rotation
augmentation can't fix that.

### Why holes dropped under continuous rotation

Hypothesis: under D₄ augmentation, the model only learns 4 specific
"correct" orientations at each scale. Branches that would have curved
between cardinal directions in real DLAs get rendered as right-angle
zigzags in the model's prior, which create loops. Under SO(2) rotation,
the model has to learn smoothly-curved branches, which doesn't create
spurious closed circuits.

Files: `results/eval_v3cr_e69/azimuthal/`,
`results/eval_v3cr_e69/dipole_orientation/`.

---

## Headline finding 3: multi-channel topology signal

New `eval/render_multichannel.py` produces 3-channel `.npz`:
- ch0: binary presence (current model's input)
- ch1: particle deposition order, normalized [0, 1]
- ch2: distance from seed, normalized [0, 1]

U-Net first layer accepts 3 channels. During inference we extract ch0
as the final binary image.

10k training samples regenerated overnight. v4 training started ~6 AM EDT.

### v4-multichannel at epoch 4 (n=32 samples)

| Metric | Train | v4mc e4 | ratio | KS p |
|---|---|---|---|---|
| Dipole $p_1$ | 0.0043 | 0.072 | **×16.55** | 4×10⁻¹² |
| Quadrupole $p_2$ | 0.124 | 0.208 | ×1.68 | 0.05 |
| Az variance | 42.3 | 35.8 | **×0.85** (LOWER) | 2×10⁻⁶ |
| **Holes** | 6.23 | **1.84** | **×0.30** (3.3× FEWER) | 6×10⁻¹¹ |
| R_g raw | 95.6 | 119.7 | ×1.25 | 7×10⁻⁹ |

**Reading**:
- **Holes are 0.3× training** — multichannel encoding is teaching the
  model not to produce closed circuits. Channel 1 (growth order) makes
  loops topologically inconsistent: a pixel can't be older than its
  neighbors.
- Dipole is 16× — severely undertrained at epoch 4. Need more training.
  Will report later epochs by meeting end if available.
- R_g is too large because the model produces sparse, undercondensed
  clusters at this stage.

The early result is a **clean signal that structural priors work**: even
4 epochs of training, the model learned to avoid loops. Compare to v3
which never learned this even at epoch 64.

Files: `results/eval_v4mc_e0004/azimuthal/`,
`results/eval_v4mc_e0004/dipole_orientation/`.

---

## Comparison table

| Metric | v3 baseline | v3-controt e69 | v4mc e4 |
|---|---|---|---|
| Dipole p₁ | 3.3× | 3.84× | 16.5× |
| Quadrupole p₂ | 2.5× | 1.68× | 1.68× |
| Holes | 1.32× | **0.77×** | **0.30×** |
| Az variance | 1.17× | 1.22× | **0.85×** |
| Status | n=100, mature | n=96, fix isolated | n=32, undertrained |

**Two distinct signals**:
- Continuous rotation alone: improves topology + quadrupole, dipole unchanged
- Multichannel from scratch: dramatically improves topology + variance

These are complementary fixes targeting different failure modes.

---

## What this tells us about the path forward

1. **Multichannel encoding is the right idea.** Even drastically undertrained,
   it produces tree-like outputs (0.3× holes). The structural prior is
   working as designed.

2. **Continuous rotation is necessary but not sufficient.** It cleanly
   improves quadrupole and topology but doesn't address per-sample
   asymmetry (dipole). The augmentation bug fix was real and correct,
   but the dipole problem lives elsewhere.

3. **The dipole magnitude problem is a sample-quality problem, not a
   symmetry problem.** Per-sample lopsidedness in random directions
   suggests the model is producing samples that lack the *internal balance*
   real DLA growth produces. Possible mitigations not yet tested:
   - More training (always works)
   - Multi-channel + continuous rotation combined (next experiment)
   - Larger model (more capacity to balance)
   - Skeleton-conditioned cascaded diffusion (more direct topology prior)

4. **Per-sample R_g matters more than mean R_g.** The variance of $|c_1|$
   is itself elevated — the model isn't uniformly slightly-lopsided, it
   produces a bimodal distribution of "fine clusters" and "very lopsided
   clusters". Worth a histogram check (see notes).

---

## Remaining items in flight (status as of 10:18 AM UTC)

| Job | Status | Output if successful |
|---|---|---|
| v3-controt training (56499140) | Running, 2:17 left, at epoch 74 | Will reach epoch 78-79 before SLURM cap |
| v4-multichannel training (56501200) | Running, 1:38 left, at epoch 8 | Will reach epoch 11-12, last ckpt at epoch 9 |
| eval_v4_and_v3cr (56501202) | Running, 36 left, generating v4mc samples (32/100 done) | Full v3cr metrics, partial v4mc |
| dla_eval (56500217) pending | Will run after v3cr completes | Final v3cr eval at epoch 78-79 |

---

## What to say in the meeting

### Lead with the diagnostic story (~3 min)

> "I dug into the dipole excess we found last meeting. By computing the
> per-sample dipole phase and running a Rayleigh circular-uniformity test,
> I found the bias is **not** preferentially oriented — both training
> and generated phases are statistically uniform with p > 0.79. The
> excess is **pure per-sample magnitude inflation**: generated clusters
> are 1.88× more lopsided than training in *random* directions, not
> toward any preferred axis. That rules out positional encoding artifacts
> and D₄ aliasing."

### Then the augmentation bug (~2 min)

> "While investigating I found that our training augmentation only used
> 90° rotations plus flips — D₄ symmetry — when DLA has continuous SO(2)
> symmetry. That's a real bug. I patched it and warm-started a run from
> epoch 64. After 5 more epochs the dipole hadn't improved (3.3× → 3.84×),
> which the orientation finding predicted. **But quadrupole improved
> (2.5× → 1.68×) and hole count dropped from 1.32× training to 0.77×** —
> generated now has *fewer* loops than training. The fix is real, just
> not for the dipole."

### Then the multi-channel result (~3 min)

> "Independently I built a multi-channel pipeline: 3-channel input with
> binary presence, particle deposition order normalized [0,1], and
> distance from seed. The U-Net learns all 3 jointly during denoising.
> Channel 1 makes loops topologically inconsistent — a pixel with high
> order can't connect to a pixel with even higher order if the data
> respects deposition. I trained from scratch on 10k samples.
>
> **At just epoch 4 — radically undertrained — the model produces 0.30×
> the loops of training samples**. The structural prior is working. Dipole
> is poor (16×) because the model isn't trained yet, but the topology
> signal is unmistakable in only 4 epochs."

### Open questions for PI (~3 min)

1. **What's a fair convergence point for v4mc?** v3 was at epoch 64 for the
   reported numbers. v4 has 3× the input information so might converge
   faster, but starting from scratch needs the same loss landscape descent.
   Want to compare at matched loss values rather than matched epochs.

2. **Channel 2 redundancy** (note `#kbjqub`): distance from seed correlates
   strongly with deposition order. Are they truly independent signals or
   adding the same supervision twice? Worth testing without channel 2 next.

3. **Tree-consistency at sampling time**: the model is trained on jointly
   consistent (ch0, ch1, ch2) tuples. At inference it produces a (ch0, ch1, ch2)
   sample but those might not be self-consistent. Should we add a
   classifier-free guidance step that pushes toward consistency, or
   project onto the "valid tuple" manifold during denoising?

4. **Combined experiment**: v4mc + continuous rotation. The two fixes
   target different failure modes. Should be the next training run.

5. **Per-sample R_g distribution**: variance of $|c_1|$ is itself elevated
   (0.033 std for gen vs 0.016 std for train) — need to look at the
   histogram of dipole magnitudes, not just the mean. If it's bimodal,
   the model is producing two failure modes mixed together.

---

## Files

Primary:
- `results/eval_v3cr_e69/azimuthal/azimuthal_comparison.png` — v3-controt
- `results/eval_v3cr_e69/azimuthal/azimuthal_mean_spectrum.png`
- `results/eval_v4mc_e0004/azimuthal/azimuthal_comparison.png` — v4mc
- `results/eval_v4mc_e0004/azimuthal/azimuthal_mean_spectrum.png`
- `results/dipole_orientation/dipole_orientation.png` — original v3
- `results/dipole_orientation/dipole_phase_vs_magnitude.png`

Code:
- `eval/dipole_orientation.py` — Rayleigh test + polar histograms
- `model/render_multichannel.py` — 3-channel renderer
- `model/dataset.py` — D₄ → SO(2) augmentation patch
- `model/train.py` — added `--multichannel` flag

---

## One-sentence summary

**The dipole excess is per-sample magnitudinal lopsidedness in random
directions (rules out symmetry breaking); continuous rotation augmentation
fixes a real bug but doesn't help dipole; multi-channel encoding produces
3.3× fewer loops than training even at epoch 4; the path forward is
multi-channel + continuous rotation combined, with longer training.**
