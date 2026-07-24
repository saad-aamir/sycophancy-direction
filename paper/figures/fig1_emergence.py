"""Figure 1: emergence of a usable linear capitulation direction.

Canonical deterministic numbers (sorted shuffle order, regenerated 22 July;
sources: regen_thresholds.log, src/analysis_v2.py). Each point is matched
LOQO AUROC at the best layer; short ticks mark each model's own
within-question permutation threshold (mean + 2sd).
"""
from pathlib import Path

import matplotlib.pyplot as plt

# family -> [(params_B, auroc, perm_threshold, gate_pass)]
DATA = {
    "Qwen2.5": [
        (0.5, 0.678, 0.686, False),
        (1.5, 0.632, 0.580, False),
        (3.0, 0.617, 0.572, False),
        (7.0, 0.778, 0.574, True),
        (14.0, 0.720, 0.618, True),
    ],
    "Llama-3.x": [
        (1.0, 0.569, 0.559, False),
        (3.0, 0.655, 0.603, False),
        (8.0, 0.710, 0.602, True),
    ],
}
COLORS = {"Qwen2.5": "#1f77b4", "Llama-3.x": "#d62728"}

fig, ax = plt.subplots(figsize=(5.2, 3.4))
for fam, rows in DATA.items():
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    ax.plot(xs, ys, "-", color=COLORS[fam], lw=1.4, label=fam, zorder=2)
    for x, y, thr, ok in rows:
        ax.scatter([x], [y], s=46, zorder=3, color=COLORS[fam],
                   facecolors=COLORS[fam] if ok else "white",
                   edgecolors=COLORS[fam], linewidths=1.4)
        ax.plot([x], [thr], marker="_", ms=11, mew=1.6,
                color=COLORS[fam], alpha=0.45, zorder=1)

ax.axhline(0.70, ls="--", lw=1.0, color="gray", zorder=0)
ax.text(0.52, 0.703, "pre-registered usability gate (0.70)",
        fontsize=7.5, color="gray", va="bottom")
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 1.5, 3, 7, 14])
ax.set_xticklabels(["0.5B", "1B", "1.5B", "3B", "7B", "14B"])
ax.minorticks_off()
ax.set_xlabel("Parameters")
ax.set_ylabel("Matched LOQO AUROC (residual stream)")
ax.set_ylim(0.45, 0.85)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

out = Path(__file__).parent
for ext in ("pdf", "png"):
    fig.savefig(out / f"fig1_emergence.{ext}", dpi=300)
print("wrote", out / "fig1_emergence.pdf")