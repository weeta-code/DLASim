#!/usr/bin/env python3
"""
v4mc training trajectory: how dipole, holes, R_g evolve with epoch.

Pulls metrics from results/eval_v4mc_e{4,14,24}/ and plots the
trajectory alongside v3 baseline reference lines.

Output: results/comparison_images/v4mc_trajectory_2026-05-06.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return json.load(f)


base = Path("/home/vectors/Documents/Research/Machta/dla_project/results")

epochs = [4, 14, 24]
runs = []
for e in epochs:
    p = base / f"eval_v4mc_e{e:02d}" / "azimuthal" / "summary.json"
    if not p.exists():
        # try other naming conventions
        alt = base / f"eval_v4mc_e{e:04d}" / "azimuthal" / "summary.json"
        if alt.exists():
            p = alt
        else:
            alt2 = base / f"eval_v4mc_e{e}" / "azimuthal" / "summary.json"
            if alt2.exists():
                p = alt2
            else:
                # Nothing found, skip
                print(f"WARN: no summary.json for epoch {e}")
                continue
    runs.append((e, load(p)))

# v3 baseline reference
v3 = load(base / "azimuthal_eval_100v100" / "summary.json")
v3cr = load(base / "eval_v3cr_e84" / "azimuthal" / "summary.json")

train = v3["training"]

metrics = [
    ("p1_fraction", "Dipole p₁  (× train)"),
    ("p2_fraction", "Quadrupole p₂  (× train)"),
    ("n_holes",     "Hole count  (× train)"),
    ("rg",          "R_g (raw)  (× train)"),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

for ax, (key, title) in zip(axes.flat, metrics):
    train_mean = train[key]["mean"]
    es = []
    ratios = []
    cilo = []
    cihi = []
    for e, s in runs:
        gen = s["generated"][key]
        es.append(e)
        ratios.append(gen["mean"] / train_mean)
        cilo.append(gen["ci95"][0] / train_mean)
        cihi.append(gen["ci95"][1] / train_mean)
    es = np.array(es)
    ratios = np.array(ratios)

    yerr_lo = ratios - np.array(cilo)
    yerr_hi = np.array(cihi) - ratios
    ax.errorbar(es, ratios, yerr=[yerr_lo, yerr_hi],
                fmt="o-", color="#E53935", linewidth=2, markersize=8,
                capsize=5, label="v4-multichannel")

    # v3 baseline ratio
    v3_ratio = v3["generated"][key]["mean"] / train_mean
    ax.axhline(v3_ratio, color="#5C6BC0", linestyle="--", linewidth=1.5,
               label=f"v3 baseline e64 ({v3_ratio:.2f}×)")

    # v3-controt ratio
    v3cr_ratio = v3cr["generated"][key]["mean"] / train_mean
    ax.axhline(v3cr_ratio, color="#43A047", linestyle="--", linewidth=1.5,
               label=f"v3-controt e84 ({v3cr_ratio:.2f}×)")

    # train reference (1.0)
    ax.axhline(1.0, color="#9E9E9E", linestyle=":", linewidth=1, label="train")

    ax.set_xlabel("v4mc epoch", fontsize=10)
    ax.set_ylabel("ratio (gen mean / train mean)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(es)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    # annotate values
    for e, r in zip(es, ratios):
        ax.text(e, r * 1.06 if r > 1.5 else r + 0.1, f"×{r:.2f}",
                ha="center", fontsize=9, fontweight="bold")

fig.suptitle("v4mc training trajectory: convergence of metrics with epoch",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = base / "comparison_images" / "v4mc_trajectory_2026-05-06.png"
# 11x8 at dpi=130 -> 1430x1040
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print(f"Wrote {out}")
