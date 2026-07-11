"""Step 4 — Difference-in-means directions + layer selection.

For each contrast and each layer l:
    v_l = mean(acts[pos, l]) - mean(acts[neg, l]);   v_hat = v / ||v||

Layer selection: AUROC of the projection (acts @ v_hat) as a pos/neg
separator, computed per layer. Best layer -> saved as the primary direction.
Also extracts per-pushback-type directions at the best layer and prints
their pairwise cosine similarities (the decomposition analysis).

Outputs:
  outputs/directions/{pair_type}_directions.pt
     dirs [n_layers, d_model], auroc [n_layers], best_layer int,
     per_type {ptype: [d_model]}
  outputs/figures/{pair_type}_layer_auroc.png

    python -m src.extract_directions
"""
from __future__ import annotations

import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score

from .common import ROOT, load_config


def diff_in_means(acts: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """acts [n, L, d], labels [n] -> unit directions [L, d]."""
    pos = acts[labels == 1].mean(0)
    neg = acts[labels == 0].mean(0)
    v = pos - neg
    norms = v.norm(dim=-1, keepdim=True)
    return v / norms.clamp_min(1e-8)


def layer_aurocs(acts, labels, dirs) -> torch.Tensor:
    scores = torch.einsum("nld,ld->nl", acts, dirs)
    return torch.tensor([
        roc_auc_score(labels.numpy(), scores[:, l].numpy())
        for l in range(acts.shape[1])
    ])


def process(pair_type: str, cfg):
    path = ROOT / cfg["paths"]["activations_dir"] / f"{pair_type}_acts.pt"
    if not path.exists():
        print(f"[dirs] {path} missing, skipping")
        return
    blob = torch.load(path)
    acts, labels, ptypes = blob["acts"], blob["labels"], blob["ptypes"]
    if labels.sum() < 5 or (1 - labels).sum() < 5:
        print(f"[dirs] {pair_type}: <5 examples in a class — unstable, "
              f"collect more data before trusting this.")

    dirs = diff_in_means(acts, labels)
    auroc = layer_aurocs(acts, labels, dirs)
    best = int(auroc.argmax())
    print(f"[dirs] {pair_type}: best layer {best}  AUROC={auroc[best]:.3f}")

    # per-pushback-type directions at the best layer
    per_type = {}
    for t in sorted(set(ptypes)):
        mask = torch.tensor([p == t for p in ptypes])
        a, y = acts[mask], labels[mask]
        if y.sum() >= 3 and (1 - y).sum() >= 3:
            per_type[t] = diff_in_means(a[:, best:best + 1, :], y)[0]

    if len(per_type) >= 2:
        print(f"[dirs] {pair_type}: per-type cosine sims @ layer {best}")
        for a, b in itertools.combinations(sorted(per_type), 2):
            cos = torch.dot(per_type[a], per_type[b]).item()
            print(f"    {a:15s} x {b:15s}  {cos:+.3f}")

    out = ROOT / cfg["paths"]["directions_dir"] / f"{pair_type}_directions.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dirs": dirs, "auroc": auroc, "best_layer": best,
                "per_type": per_type}, out)
    print(f"[dirs] saved {out}")

    fig_dir = ROOT / cfg["paths"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 3.2))
    plt.plot(auroc.numpy(), marker="o", ms=3)
    plt.axhline(0.5, ls="--", c="gray", lw=1)
    plt.scatter([best], [auroc[best]], c="crimson", zorder=5,
                label=f"best: L{best} ({auroc[best]:.2f})")
    plt.xlabel("layer"); plt.ylabel("AUROC"); plt.title(f"{pair_type} direction")
    plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / f"{pair_type}_layer_auroc.png", dpi=150)
    plt.close()


def main():
    cfg = load_config()
    for pair_type in ("capitulation", "pushback"):
        process(pair_type, cfg)


if __name__ == "__main__":
    main()
