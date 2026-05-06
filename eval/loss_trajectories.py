#!/usr/bin/env python3
"""
Loss-trajectory comparison: v3 (1-channel, D4 augmentation) vs
v3-controt (1-channel, SO(2) augmentation, warm-start from v3 epoch 64) vs
v4-multichannel (3-channel, SO(2) augmentation, from scratch).

Shows the convergence-acceleration story of multi-channel encoding:
v4mc reaches v3-quality loss in ~13 epochs vs v3's 64.

Output: results/comparison_images/loss_trajectories_2026-05-05.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


v3 = load("/tmp/v3_log.jsonl")          # v3 baseline (D4 aug, 1ch)
v3cr = load("/tmp/v3cr_log.jsonl")      # v3 controt (SO(2) aug, 1ch, warm from e64)
v4mc = load("/tmp/v4mc_log.jsonl")      # v4 multichannel (SO(2) aug, 3ch, from scratch)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# === Panel 1: full loss curves ===
ax = axes[0]
ax.plot([r["epoch"] for r in v3], [r["loss"] for r in v3],
        color="#5C6BC0", linewidth=1.6, marker="o", markersize=2.5,
        label="v3 baseline (D₄, 1ch)")
ax.plot([r["epoch"] for r in v3cr], [r["loss"] for r in v3cr],
        color="#43A047", linewidth=1.6, marker="s", markersize=3.5,
        label="v3-controt (SO(2), 1ch, warm e64)")
ax.plot([r["epoch"] for r in v4mc], [r["loss"] for r in v4mc],
        color="#E53935", linewidth=1.6, marker="^", markersize=3.5,
        label="v4-multichannel (SO(2), 3ch, scratch)")

# annotate v3 epoch 64 baseline level
v3_e64 = next(r for r in v3 if r["epoch"] == 64)
ax.axhline(v3_e64["loss"], color="gray", linestyle=":", linewidth=0.8,
           label=f"v3 epoch 64 ({v3_e64['loss']:.4f})")

# v4mc reaches v3 level
v4_match = next((r for r in v4mc if r["loss"] <= v3_e64["loss"]), None)
if v4_match:
    ax.axvline(v4_match["epoch"], color="#E53935", linestyle=":", linewidth=0.8)
    ax.text(v4_match["epoch"] + 1, v3_e64["loss"] * 1.5,
            f"v4 matches v3@e64\nat epoch {v4_match['epoch']}\n"
            f"({v4_match['loss']:.4f})",
            fontsize=8, color="#C62828", ha="left", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#E53935"))

ax.set_yscale("log")
ax.set_xlabel("epoch", fontsize=10)
ax.set_ylabel("training loss (log)", fontsize=10)
ax.set_title("Training-loss trajectories", fontsize=11)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3, which="both")

# === Panel 2: log-loss vs effective epoch (for fair convergence comparison) ===
ax = axes[1]

# zoomed view of the converged regime
ax.plot([r["epoch"] for r in v3 if r["epoch"] >= 5],
        [r["loss"] for r in v3 if r["epoch"] >= 5],
        color="#5C6BC0", linewidth=1.6, marker="o", markersize=3,
        label="v3 baseline")
ax.plot([r["epoch"] for r in v3cr if r["epoch"] >= 5],
        [r["loss"] for r in v3cr if r["epoch"] >= 5],
        color="#43A047", linewidth=1.6, marker="s", markersize=4,
        label="v3-controt")
ax.plot([r["epoch"] for r in v4mc if r["epoch"] >= 1],
        [r["loss"] for r in v4mc if r["epoch"] >= 1],
        color="#E53935", linewidth=1.6, marker="^", markersize=4,
        label="v4-multichannel")
ax.axhline(v3_e64["loss"], color="gray", linestyle=":", linewidth=0.8,
           label=f"v3 e64 baseline ({v3_e64['loss']:.4f})")

ax.set_xlabel("epoch", fontsize=10)
ax.set_ylabel("loss", fontsize=10)
ax.set_title("Converged regime  (epoch ≥ 5)", fontsize=11)
ax.set_ylim(0.003, 0.012)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

# Speedup annotation
v3_e64_loss = v3_e64["loss"]
v4_match_epoch = v4_match["epoch"] if v4_match else None
if v4_match_epoch:
    speedup = 64 / v4_match_epoch
    ax.text(0.5, 0.95,
            f"Multi-channel convergence speedup: {speedup:.1f}× "
            f"({v4_match_epoch} epochs vs 64)",
            transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#FFF9C4",
                      edgecolor="#F57F17", alpha=0.9))

fig.suptitle(
    "DLA diffusion training trajectories — D₄ vs SO(2) vs multi-channel",
    fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out = Path("/home/vectors/Documents/Research/Machta/dla_project/"
           "results/comparison_images/loss_trajectories_2026-05-05.png")
# 13x5 at dpi=130 -> 1690x650, under 2000
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print(f"Wrote {out}")
print()
print(f"v3 reached loss {v3_e64_loss:.4f} at epoch 64")
if v4_match:
    print(f"v4mc reached same loss at epoch {v4_match['epoch']} ({v4_match['loss']:.4f})")
    print(f"Convergence speedup: {64 / v4_match['epoch']:.1f}x")
print(f"v3cr loss at epoch 84: {v3cr[-1]['loss']:.4f}")
print(f"v4mc loss at epoch 13: {v4mc[-1]['loss']:.4f}")
