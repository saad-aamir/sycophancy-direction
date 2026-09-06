"""Figure 3: ablation effect on flip rate, ordered by how far each direction
cleared the 0.70 usability bar.

Numbers from `python -m src.analysis_v2` at commit 9325963.
Negative values mean ablation reduced capitulation.
"""
from pathlib import Path

import matplotlib.pyplot as plt

# (label, gate AUROC, verdict, change_pp, lo, hi)
ROWS = [
    ("Qwen-1.5B", 0.632, "gate fail",      -2.1, -5.9, +2.1),
    ("Llama-8B",  0.710, "pass, +0.010",   +1.4, -2.0, +4.8),
    ("Qwen-7B",   0.778, "pass, +0.078",   -4.8, -8.3, -1.2),
]

fig, ax = plt.subplots(figsize=(5.4, 2.9))
ys = list(range(len(ROWS)))[::-1]

for y, (label, auroc, verdict, mid, lo, hi) in zip(ys, ROWS):
    sig = hi < 0 or lo > 0
    color = "#1f77b4" if sig else "#888888"
    ax.plot([lo, hi], [y, y], color=color, lw=2.0, solid_capstyle="round")
    ax.plot([mid], [y], "o", color=color, ms=7,
            markerfacecolor=color if sig else "white",
            markeredgecolor=color, markeredgewidth=1.6)
    ax.text(-11.4, y + 0.20, label, fontsize=9, va="center", ha="left")
    ax.text(-11.4, y - 0.22, f"AUROC {auroc:.3f}, {verdict}",
            fontsize=7, va="center", ha="left", color="#666666")

ax.axvline(0, ls="--", lw=1.0, color="gray", zorder=0)
ax.set_yticks([])
ax.set_ylim(-0.7, len(ROWS) - 0.3)
ax.set_xlim(-11.5, 6.5)
ax.set_xlabel("Change in flip rate after ablation (percentage points)")
ax.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout()

out = Path(__file__).parent
for ext in ("pdf", "png"):
    fig.savefig(out / f"fig3_ablation_forest.{ext}", dpi=300)
print("wrote", out / "fig3_ablation_forest.pdf")
