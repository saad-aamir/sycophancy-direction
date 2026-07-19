"""Step 4 (v3) — Direction extraction with matched pairs and clustered CV.

PRIMARY analysis — MATCHED within-question contrast:
    For each question with >=1 flip and >=1 hold episode, compute
    (mean of its pos activations − mean of its neg activations); the
    direction is the mean of these per-question differences. Question
    identity (topic, difficulty, answer frequency) cancels exactly.

Validation — LEAVE-ONE-QUESTION-OUT CV:
    Hold out ALL episodes of one question; fit the direction on the rest;
    score the held-out episodes. Episodes are siblings (shared prefix), so
    per-episode LOO leaks; per-question folds are the honest unit.

Null — WITHIN-QUESTION LABEL SHUFFLE (20 perms):
    Permute labels among each question's own episodes, preserving cluster
    structure; recompute LOQO AUROC. Real curve must clear mean + 2sd.

GATE: PASS iff matched LOQO AUROC(best) >= 0.70 and > shuffle mean + 2sd.

Pooled (unmatched) analysis is kept as a secondary curve. The saved
primary direction per layer uses the FULL matched data (CV is for
selection/validation only). qids are read from the cached blob if present,
else recovered from the pairs jsonl (order is preserved by caching).

    python -m src.extract_directions
"""
from __future__ import annotations

import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
torch.set_num_threads(1)
from sklearn.metrics import roc_auc_score

from .common import ROOT, load_config, read_jsonl

N_PERMS = 20
GATE_MIN_CV = 0.70


# ------------------------------------------------------------- directions

def pooled_direction(acts: torch.Tensor, labels: torch.Tensor):
    """acts [n, L, d] -> unit dirs [L, d] via pooled diff-in-means."""
    if labels.sum() < 1 or (1 - labels).sum() < 1:
        return None
    v = acts[labels == 1].mean(0) - acts[labels == 0].mean(0)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def matched_direction(acts: torch.Tensor, labels: torch.Tensor,
                      qids: list[str]):
    """Mean over questions of (pos_mean_q − neg_mean_q). Returns
    (unit dirs [L, d] | None, n_matched_questions)."""
    diffs = []
    for q in sorted(set(qids)):
        idx = torch.tensor([i for i, x in enumerate(qids) if x == q])
        y = labels[idx]
        if y.sum() >= 1 and (1 - y).sum() >= 1:
            a = acts[idx]
            diffs.append(a[y == 1].mean(0) - a[y == 0].mean(0))
    if len(diffs) < 3:
        return None, len(diffs)
    v = torch.stack(diffs).mean(0)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8), len(diffs)


# -------------------------------------------------------------------- CV

def loqo_auroc(acts, labels, qids, mode: str) -> torch.Tensor:
    """Leave-one-question-out CV AUROC per layer. mode: matched|pooled."""
    n, L, _ = acts.shape
    scores = torch.full((n, L), float("nan"))
    uniq = sorted(set(qids))
    for q in uniq:
        test = torch.tensor([i for i, x in enumerate(qids) if x == q])
        train = torch.tensor([i for i, x in enumerate(qids) if x != q])
        a_tr, y_tr = acts[train], labels[train]
        if mode == "matched":
            dirs, _ = matched_direction(a_tr, y_tr,
                                        [qids[i] for i in train.tolist()])
        else:
            dirs = pooled_direction(a_tr, y_tr)
        if dirs is None:
            continue
        scores[test] = torch.einsum("nld,ld->nl", acts[test], dirs)

    out = torch.full((L,), 0.5)
    mask = ~scores[:, 0].isnan()
    y = labels[mask].numpy()
    if mask.sum() >= 4 and 0 < y.sum() < len(y):
        for l in range(L):
            out[l] = roc_auc_score(y, scores[mask, l].numpy())
    return out


def within_q_shuffle(labels: torch.Tensor, qids: list[str], g) -> torch.Tensor:
    shuffled = labels.clone()
    for q in set(qids):
        idx = torch.tensor([i for i, x in enumerate(qids) if x == q])
        perm = idx[torch.randperm(len(idx), generator=g)]
        shuffled[idx] = labels[perm]
    return shuffled


def insample_curve(acts, labels) -> torch.Tensor:
    dirs = pooled_direction(acts, labels)
    scores = torch.einsum("nld,ld->nl", acts, dirs)
    return torch.tensor([
        roc_auc_score(labels.numpy(), scores[:, l].numpy())
        for l in range(acts.shape[1])
    ])


# ------------------------------------------------------------------ main

def load_qids(pair_type: str, blob: dict, cfg, n: int) -> list[str] | None:
    if "qids" in blob:
        return list(blob["qids"])
    pairs_path = (ROOT / cfg["paths"]["activations_dir"]
                  / f"pairs_{pair_type}.jsonl")
    if pairs_path.exists():
        pairs = read_jsonl(pairs_path)
        if len(pairs) == n:
            return [p["qid"] for p in pairs]
    return None


def process(pair_type: str, cfg):
    path = ROOT / cfg["paths"]["activations_dir"] / f"{pair_type}_acts.pt"
    if not path.exists():
        print(f"[dirs] {path} missing, skipping")
        return
    blob = torch.load(path)
    acts, labels, ptypes = blob["acts"], blob["labels"], blob["ptypes"]
    n = len(labels)
    qids = load_qids(pair_type, blob, cfg, n)
    if qids is None:
        print(f"[dirs] {pair_type}: cannot recover qids "
              f"(pairs file changed?) — re-run build_contrast_pairs + "
              f"cache_activations")
        return

    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    dirs_matched, n_matched = matched_direction(acts, labels, qids)
    print(f"[dirs] {pair_type}: n={n_pos} pos / {n_neg} neg, "
          f"{len(set(qids))} questions, {n_matched} matched (mixed-outcome)")
    if dirs_matched is None:
        print(f"[dirs]   <3 matched questions — matched analysis "
              f"impossible; collect more data")
        return

    auroc_matched = loqo_auroc(acts, labels, qids, "matched")
    auroc_pooled = loqo_auroc(acts, labels, qids, "pooled")
    auroc_in = insample_curve(acts, labels)

    g = torch.Generator().manual_seed(0)
    perms = torch.stack([
        loqo_auroc(acts, within_q_shuffle(labels, qids, g), qids, "matched")
        for _ in range(N_PERMS)])
    shuf_mean, shuf_sd = perms.mean(0), perms.std(0)

    best = int(auroc_matched.argmax())
    threshold = shuf_mean[best] + 2 * shuf_sd[best]
    gate = bool(auroc_matched[best] >= GATE_MIN_CV
                and auroc_matched[best] > threshold)

    print(f"[dirs]   MATCHED LOQO best layer {best}: "
          f"AUROC {auroc_matched[best]:.3f}  "
          f"(pooled LOQO {auroc_pooled[best]:.3f}, "
          f"in-sample {auroc_in[best]:.3f})")
    print(f"[dirs]   within-q shuffle @ L{best}: "
          f"{shuf_mean[best]:.3f} ± {shuf_sd[best]:.3f}  "
          f"(threshold {threshold:.3f})")
    print(f"[dirs]   GATE: {'PASS' if gate else 'FAIL'} "
          f"(need matched LOQO >= {GATE_MIN_CV} and > shuffle mean + 2sd)")

    # per-type matched directions at the chosen layer
    per_type, per_type_n = {}, {}
    for t in sorted(set(ptypes)):
        mask = [i for i, p in enumerate(ptypes) if p == t]
        idx = torch.tensor(mask)
        d_t, n_t = matched_direction(acts[idx][:, best:best + 1, :],
                                     labels[idx],
                                     [qids[i] for i in mask])
        if d_t is not None:
            per_type[t], per_type_n[t] = d_t[0], n_t
    if len(per_type) >= 2:
        low = any(v < 15 for v in per_type_n.values())
        note = "  [NOISE: <15 matched q/type — do not interpret]" if low else ""
        print(f"[dirs]   per-type cosines @ L{best} "
              f"(matched q per type: {per_type_n}){note}")
        for a_, b_ in itertools.combinations(sorted(per_type), 2):
            print(f"      {a_:15s} x {b_:15s}  "
                  f"{torch.dot(per_type[a_], per_type[b_]).item():+.3f}")

    out = ROOT / cfg["paths"]["directions_dir"] / f"{pair_type}_directions.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dirs": dirs_matched,          # primary, read by ablation.py
                "auroc": auroc_matched,        # alias, read by ablation.py
                "dirs_pooled": pooled_direction(acts, labels),
                "auroc_matched": auroc_matched,
                "auroc_pooled": auroc_pooled,
                "auroc_insample": auroc_in,
                "shuffle_mean": shuf_mean, "shuffle_sd": shuf_sd,
                "best_layer": best, "gate_pass": gate,
                "n_matched_questions": n_matched,
                "per_type": per_type, "per_type_n": per_type_n}, out)
    print(f"[dirs]   saved {out}")

    fig_dir = ROOT / cfg["paths"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    L = torch.arange(len(auroc_matched))
    plt.figure(figsize=(7.5, 3.4))
    plt.fill_between(L, (shuf_mean - 2 * shuf_sd).numpy(),
                     (shuf_mean + 2 * shuf_sd).numpy(),
                     alpha=0.25, color="gray",
                     label="within-q shuffle ±2sd")
    plt.plot(L, auroc_in.numpy(), lw=1, ls=":", color="tab:blue",
             label="in-sample pooled (inflated)")
    plt.plot(L, auroc_pooled.numpy(), lw=1, color="tab:orange",
             label="pooled LOQO")
    plt.plot(L, auroc_matched.numpy(), lw=2, marker="o", ms=3,
             color="tab:blue", label="matched LOQO (primary)")
    plt.axhline(0.5, ls="--", c="gray", lw=1)
    plt.scatter([best], [auroc_matched[best]], c="crimson", zorder=5,
                label=f"best: L{best} ({auroc_matched[best]:.2f})")
    plt.xlabel("layer"); plt.ylabel("AUROC")
    plt.title(f"{pair_type}  (n={n_pos}+{n_neg}, {n_matched} matched q, "
              f"gate {'PASS' if gate else 'FAIL'})")
    plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(fig_dir / f"{pair_type}_layer_auroc.png", dpi=150)
    plt.close()


def main():
    cfg = load_config()
    for pair_type in ("capitulation", "pushback"):
        process(pair_type, cfg)


if __name__ == "__main__":
    main()