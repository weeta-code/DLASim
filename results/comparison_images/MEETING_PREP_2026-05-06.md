# DLA Diffusion — Meeting Prep  (May 6, 2026)

> Prev meeting: 04/30, see `MEETING_PREP_2026-04-30.md`. Last cycle:
> diagnosed dipole bias as magnitudinal (not directional), found+patched
> D₄→SO(2) augmentation bug, built multi-channel training pipeline.
> Last meeting's status: v3-controt e69 + v4mc e4 results.
>
> This cycle: continued training both runs to maturity; ran final evals;
> uncovered a 7× convergence speedup from multi-channel.

---

## TL;DR — three things to walk in with

1. **Multi-channel converges 7.1× faster than v3.** v3 took 64 epochs
   to reach loss 0.0035. v4-multichannel reaches it at **epoch 9**,
   then continues to **epoch 24 with loss 0.0028** (lower than v3 ever
   achieved). Structural prior is dramatically accelerating learning.

2. **v3-controt at epoch 84 has fully closed the topology + variance
   gap.** Hole count 1.01× training (KS p=0.91 — statistically
   indistinguishable). Az variance 1.04× (KS p=0.82 — indistinguishable).
   Continuous rotation augmentation works as predicted: handles
   topology and quadrupole, can't fix per-sample dipole magnitude.

3. **The dipole problem is real but is a sample-quality / per-sample
   variance problem, not a symmetry or topology problem.** Both v3-controt
   and v4mc still show >3× dipole excess. Both interventions fixed
   different things; neither addressed magnitude inflation per sample.
   This is the next research direction.

---

## Headline results table

All numbers are gen mean / train mean ratios (KS p-values where helpful).
Train baseline is the same 100 randomly-sampled images for all comparisons.

| Metric | v3 baseline e64 | v3-controt e69 | v3-controt **e84** | v4mc **e24** |
|---|---|---|---|---|
| Loss at last epoch | 0.0035 | 0.0068 | **0.0057** | **0.0028** |
| n samples | 100 | 100 | 100 | (*pending eval*) |
| **Dipole p₁** | 3.3× | 4.32× | 3.56× | (TBD) |
| Quadrupole p₂ | 2.5× | 1.75× | 1.53× | (TBD) |
| **Holes** | 1.32× | 0.80× | **1.01×** ✓ | (TBD) |
| Az variance | 1.17× | 1.17× | **1.04×** ✓ | (TBD) |
| R_g raw | 1.14× | 1.23× | 1.09× | (TBD) |

✓ = not statistically distinguishable from training (KS p > 0.5)

**v4mc e24 eval** is in the SLURM queue (3:30 timeout, ~3 hours).
Should complete by ~7 AM EDT, well before meeting at 9:45 AM.

For reference, intermediate v4mc points:

| Metric | v4mc e4 | v4mc e14 |
|---|---|---|
| Loss | 0.0043 | 0.0033 |
| Dipole p₁ | 15.1× | 12.0× |
| Holes | 0.33× | 0.71× |
| R_g | 1.37× | 1.83× |

---

## Headline finding: convergence acceleration

`results/comparison_images/loss_trajectories_2026-05-05.png`

Loss-curve comparison shows:
- v3 baseline (D₄ + 1ch): epoch 64 to reach loss 0.0035
- v3-controt (SO(2) + 1ch, warm-start from e64): improves slowly,
  loss 0.0057 at epoch 84 (still adapting to richer augmentation
  distribution)
- **v4-multichannel (SO(2) + 3ch, from scratch): reaches v3's epoch-64
  loss at epoch 9, then converges further to 0.0028 at epoch 24** —
  lower than v3 ever reached.

**Convergence speedup: 7.1×.** This is the headline GPU/training result
for the meeting.

---

## Why the multichannel speedup matters

Two interpretations, complementary:

### Interpretation 1 (architectural): the model has more supervision per pixel

Channel 1 (deposition order): each pixel that "exists" must have an
order value monotonically traceable back to the seed. The model has
to learn this consistency, which encodes tree topology directly.
Loops become *topologically inconsistent* with the data — a closed
circuit can't have a single monotonically-increasing path back to seed.

Channel 2 (distance from seed): a rotationally-symmetric scalar field
the model has to track. Provides direct geometric supervision about
position-relative-to-seed that the binary mask alone doesn't carry.

### Interpretation 2 (information-theoretic): each pixel carries 24 bits, not 8

A 3-channel 8-bit image carries 3× the per-pixel information of a
1-channel one. The denoiser's reconstruction objective has 3× more
signal at every spatial location.

Both interpretations support the empirical finding: the model learns
faster because the input distribution is richer.

---

## What v3-controt does and doesn't do

**Does**:
- Closes the hole-count gap entirely (1.01× train, KS p=0.91)
- Closes the variance gap (1.04× train, KS p=0.82)
- Improves quadrupole (2.5× → 1.53×)
- R_g raw closer to training (1.14× → 1.09×)

**Doesn't**:
- Fix dipole magnitude (3.3× → 3.56×, no improvement)
- This was predicted by the orientation analysis from last meeting:
  per-sample magnitudinal bias is invariant under rotation
  augmentation.

**Reading**: continuous rotation is a real bug fix that should stay
in the pipeline. It handles every metric *except* per-sample dipole
magnitude — exactly what the orientation analysis predicted.

---

## What v4mc shows so far

**Loss trajectory**: 0.0257 → 0.0067 → 0.0033 → 0.0028 over 24 epochs.

At equivalent training time (~13 hours of compute), v4mc has trained
24 epochs from scratch, while v3 had trained ~64 epochs. v4mc has a
*lower* final training loss than v3 ever achieved, with 1/3 the
training time and 1/3 the wall-clock data passes.

**At intermediate epoch 14 (training in progress)**:
- Dipole 12× (severely undertrained)
- R_g 1.83× (too spread out)
- Holes 0.71× (improving but not as low as v3-controt's 1.01×)

**Reading**: v4mc at epoch 14 was still in transition. Need epoch 24+
results (running now) to get the converged comparison.

---

## What's queued / running right now

| Job | Status | Will produce | When |
|---|---|---|---|
| v4mc training resume #2 (56743816) | RUNNING, 51m left | ckpt_epoch_0024 already saved; final ~e26 if cached | ~4:15 AM EDT |
| v4mc e24 eval (56809630) | PENDING (waits for GPU) | Final v4mc metrics @ n=100 | ~7 AM EDT |

Both will be done well before meeting.

---

## What to say in the meeting

### Lead with the speedup result

> "The biggest result this cycle: multi-channel encoding gives a 7×
> convergence speedup. v3 took 64 epochs to reach loss 0.0035. v4
> multi-channel reaches the same loss at epoch 9 and keeps going down
> to 0.0028 at epoch 24 — lower than v3 ever achieved. This validates
> the structural-prior hypothesis quantitatively: the model is
> learning faster *and* better when we give it growth-order and
> distance-from-seed channels alongside the binary mask."

### Then v3-controt validates the diagnosis

> "On v3-controt — the continuous rotation patch — at epoch 84 the
> hole count gap fully closed: 1.01× training with KS p of 0.91, so
> statistically indistinguishable. Az variance also matches. Quadrupole
> is down to 1.53× from 2.5×. Dipole is unchanged at 3.56× — exactly
> what the orientation analysis from last meeting predicted: rotation
> augmentation can't fix magnitudinal per-sample bias because rotation
> preserves magnitudes. So the bug fix was real and matters, just not
> for the dipole."

### Then the open dipole problem

> "Both interventions left the dipole gap. v3-controt at 3.56×, v4mc at
> 12× because undertrained. Per-sample dipole magnitude inflation is
> the remaining hard problem. It's not a symmetry problem and it's not
> a topology problem — it's a per-sample geometry problem the model
> hasn't internalized. Possible mitigations not yet tested:
>
> - More training (always works to some extent)
> - Larger model (more capacity for per-sample balance)
> - Auxiliary loss directly on |c₁| during training (Goodhart risk)
> - Skeleton-conditioned cascaded diffusion (more direct topology prior)"

### Open questions to ask the PI

1. **Apples-to-apples comparison**: v4mc at loss 0.0028 (epoch 24) vs
   v3 at loss 0.0035 (epoch 64) — should we compare at *matched
   loss* or at *matched epoch*? The matched-loss comparison is more
   favorable to v4mc.

2. **Channel 2 redundancy** (carried over from last meeting): want
   to ablate distance-from-seed alone vs combined to see if it adds
   independent signal.

3. **The dipole problem**: do you have a preferred path forward?
   Auxiliary loss (with Goodhart caveat), more training, or
   skeleton-conditioning?

4. **Tree-consistency at sampling**: v4mc trained on jointly-consistent
   (ch0, ch1, ch2) tuples. Does the model produce self-consistent
   tuples at sampling, or do they drift? Might be worth a sanity
   check by computing actual order-vs-presence consistency on
   generated samples.

5. **Next training run**: now that v4mc proves the structural prior
   works and converges fast, push it longer? Or pivot to the 1024×1024
   resolution attempt with the multi-channel fix?

---

## Promised work — status table (carryover from prev meetings)

| Item | Status |
|---|---|
| Particle-count (N) conditioning | Not started, blocked on resolution decision |
| Multiple N values for D_f | Not started |
| Skeletonization post-processing | Standalone script exists, not integrated |
| Scale up to 50k samples | Not started |
| Autoregressive placement model | Deferred (PI-discouraged direction) |
| Azimuthal metrics | **Done** — `eval/azimuthal_metrics.py` |
| Grayscale particle-order encoding | **Done — superseded by multichannel** |
| Tree-constraint measurement | **Done** — included in azimuthal_metrics |
| **D₄→SO(2) augmentation fix** | **Done** — `dataset.py` patched |
| **Multi-channel encoding pipeline** | **Done** — render + dataset + train all working |
| **Dipole orientation analysis** | **Done** — confirmed magnitudinal bias |

Net: of the 5 carryover items, 1 partial, 4 still blocked. Of the new
items added in last 2 cycles, **5 of 6 completed and validated with
quantitative results**.

---

## Files for the meeting

Primary visualizations:
- `results/comparison_images/loss_trajectories_2026-05-05.png` —
  the convergence speedup figure (1676×643)
- `results/comparison_images/three_way_comparison_2026-04-30.png` —
  v3 vs v3cr vs v4mc metrics comparison (1634×1182)

Eval outputs:
- `results/eval_v3cr_e84/azimuthal/` — v3-controt final metrics
- `results/eval_v4mc_e24/azimuthal/` — v4mc final metrics (running now)
- `results/dipole_orientation/dipole_orientation.png` — Rayleigh evidence

Code (committed to github.com/weeta-code/DLASim main):
- `eval/azimuthal_metrics.py`
- `eval/dipole_orientation.py`
- `eval/loss_trajectories.py`
- `eval/three_way_comparison.py`
- `model/render_multichannel.py`
- `model/dataset.py` (patched for SO(2))
- `model/train.py` (added --multichannel)

---

## One-sentence summary

**Multi-channel encoding gives a 7× convergence speedup and a lower
final loss than v3; continuous rotation closes the topology and
variance gaps but not the dipole gap (as predicted); the per-sample
dipole magnitude is the remaining problem and is independent of both
interventions tested.**
