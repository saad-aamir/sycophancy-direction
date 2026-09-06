"""Figure 4: the sign-reversing crossover between pushback types.

Odds ratios from McNemar tests on within-question discordant pairs, flips only,
Bonferroni corrected across four pre-registered type comparisons.
Numbers from `python -m src.analysis_v2` at commit 9325963.

OR is emotional over simple: below 1 means simple pushback destabilises more.
Filled markers are the four confirmatory models named in the pre-registration;
open markers are exploratory additions.
"""
from pathlib import Path

import matplotlib.pyplot as plt

# (label, params_B, family, OR, p_bonf, confirmatory)
ROWS = [
    ("Qwen-0.5B", 0.5,  "Qwen2.5",   1.00, 1.000,  False),
    ("Qwen-1.5B", 1.5,  "Qwen2.5",   0.41, 0.013,  True),
    ("Qwen-3B",   3.0,  "Qwen2.5",   0.80, 1.000,  False),
    ("Qwen-7B",   7.0,  "Qwen2.5",   0.31, 0.0001, True),
    ("Qwen-14B",  14.0, "Qwen2.5",   0.30, 0.002,  False),
    ("Llama-1B",  1.0,  "Llama-3.x", 0.86, 1.000,  True),
    ("Llama-3B",  3.0,  "Llama-3.x", 1.76, 0.073,  False),
    ("Llama-8B",  8.0,  "Llama-3.x", 2.37, 0.006,  True),
]
COLORS = {"Qwen2.5": "#1f77b4", "Llama-3.x": "#d62728"}

fig, ax = plt.subplots(figsize=(5.6, 3.6))
seen = set()
for label, x, fam, orr, p, conf in ROWS:
    c = COLORS[fam]
    ax.scatter([x], [orr], s=64, zorder=3, color=c,
               facecolors=c if conf else "white",
               edgecolors=c, linewidths=1.6,
               label=fam if fam not in seen else None)
    seen.add(fam)
    star = "*" if p < 0.05 else ""
    ax.annotate(f"{label}{star}", (x, orr), textcoords="offset points",
                xytext=(0, 9 if orr < 1 else -16), ha="center", fontsize=7,
                color="#444444")

ax.axhline(1.0, ls="--", lw=1.0, color="gray", zorder=0)
ax.text(0.46, 1.03, "no difference between types", fontsize=7.5,
        color="gray", va="bottom")
ax.text(0.46, 0.27, "simple pushback destabilises more", fontsize=7.5,
        color="#1f77b4", va="bottom")
ax.text(0.46, 2.75, "emotional pushback destabilises more", fontsize=7.5,
        color="#d62728", va="bottom")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([0.5, 1, 3, 7, 14])
ax.set_xticklabels(["0.5B", "1B", "3B", "7B", "14B"])
ax.set_yticks([0.25, 0.5, 1, 2, 3])
ax.set_yticklabels(["0.25", "0.5", "1", "2", "3"])
ax.minorticks_off()
ax.set_xlabel("Parameters")
ax.set_ylabel("Odds ratio, emotional / simple")
ax.set_ylim(0.22, 3.4)
ax.set_xlim(0.38, 20)
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.02, 0.86))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

out = Path(__file__).parent
for ext in ("pdf", "png"):
    fig.savefig(out / f"fig4_crossover_or.{ext}", dpi=300)
print("wrote", out / "fig4_crossover_or.pdf  (filled = confirmatory, * = pBonf < 0.05)")
