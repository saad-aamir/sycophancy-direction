"""Figure 2: the flip-to-recovery ratio declines with scale in both families.

Numbers from `python -m src.analysis_v2` at commit 9325963.
Ratio = (flip rate | initially correct) / (recovery rate | initially incorrect).
Values above 1 mean pushback destroys more truth than it repairs.
"""
from pathlib import Path

import matplotlib.pyplot as plt

# family -> [(params_B, flip_pct, recovery_pct)]
DATA = {
    "Qwen2.5": [
        (0.5, 17.9, 3.6),
        (1.5, 31.0, 7.9),
        (3.0, 23.7, 13.3),
        (7.0, 18.3, 19.4),
        (14.0, 15.1, 22.9),
    ],
    "Llama-3.x": [
        (1.0, 34.6, 9.7),
        (3.0, 44.2, 17.8),
        (8.0, 31.9, 33.3),
    ],
}
COLORS = {"Qwen2.5": "#1f77b4", "Llama-3.x": "#d62728"}

fig, ax = plt.subplots(figsize=(5.2, 3.4))
for fam, rows in DATA.items():
    xs = [r[0] for r in rows]
    ys = [r[1] / r[2] for r in rows]
    ax.plot(xs, ys, "-o", color=COLORS[fam], lw=1.5, ms=6, label=fam)

ax.axhline(1.0, ls="--", lw=1.0, color="gray", zorder=0)
ax.text(0.52, 1.06, "parity: pushback repairs as much as it destroys",
        fontsize=7.5, color="gray", va="bottom")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([0.5, 1, 3, 7, 14])
ax.set_xticklabels(["0.5B", "1B", "3B", "7B", "14B"])
ax.set_yticks([0.5, 1, 2, 5])
ax.set_yticklabels(["0.5", "1", "2", "5"])
ax.minorticks_off()
ax.set_xlabel("Parameters")
ax.set_ylabel("Flip rate / recovery rate")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

out = Path(__file__).parent
for ext in ("pdf", "png"):
    fig.savefig(out / f"fig2_recovery_ratio.{ext}", dpi=300)
print("wrote", out / "fig2_recovery_ratio.pdf")
