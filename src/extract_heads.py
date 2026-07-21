"""H2: attention-head locus analysis.

Pre-reg section 6 names two read-outs: the last-token residual stream
(extract_directions) and attention-head outputs (here). Section 8 sets the
head-level gate: matched LOQO AUROC >= 0.70, above the within-question
shuffle mean + 2sd, AND above the 95th percentile of the max-statistic
permutation null across every head in the model. That last term is the
family-wise correction; scanning ~800 heads without it guarantees false
positives.

    python -m src.run_sweep --set-active qwen7b
    python -m src.extract_heads --perms 200
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import torch

torch.set_num_threads(1)  # tiny per-question ops: thread teams cost more than the math

from .common import ROOT, load_config


def matched(labels, qids):
    by_q = defaultdict(lambda: [[], []])
    for i, (y, q) in enumerate(zip(labels.tolist(), qids)):
        by_q[q][int(y)].append(i)
    return {q: v for q, v in by_q.items() if v[0] and v[1]}


def loqo_auroc(acts, mq, labels):
    """Matched difference-in-means direction, leave-one-question-out.
    acts [N,P,D] -> AUROC per position [P]."""
    qs = sorted(mq)
    D = torch.stack([acts[mq[q][1]].mean(0) - acts[mq[q][0]].mean(0) for q in qs])
    S, Q = D.sum(0), len(qs)
    chunks, order = [], []
    for i, q in enumerate(qs):
        d = (S - D[i]) / max(Q - 1, 1)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        idx = mq[q][0] + mq[q][1]
        chunks.append(torch.einsum("epd,pd->ep", acts[idx], d))
        order.extend(idx)
    return auroc(torch.cat(chunks), labels[order])


def auroc(scores, y):
    N, P = scores.shape
    ranks = torch.empty(N, P)
    ranks.scatter_(0, scores.argsort(dim=0),
                   torch.arange(1.0, N + 1).unsqueeze(1).expand(N, P))
    pos = y.bool()
    npos = int(pos.sum())
    nneg = N - npos
    if npos == 0 or nneg == 0:
        return torch.full((P,), float("nan"))
    return (ranks[pos].sum(0) - npos * (npos + 1) / 2) / (npos * nneg)


def shuffled(mq, g):
    out = {}
    for q, (neg, pos) in mq.items():
        idx = neg + pos
        perm = [idx[i] for i in torch.randperm(len(idx), generator=g).tolist()]
        out[q] = [perm[: len(neg)], perm[len(neg) :]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-type", default="capitulation")
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--file", default=None,
                    help="default: <activations_dir>/<pair>_v2_attn_z_last.pt")
    args = ap.parse_args()

    cfg = load_config()
    slug = cfg.get("_slug", "model")
    path = ROOT / (args.file or
                   f"{cfg['paths']['activations_dir']}/{args.pair_type}_v2_attn_z_last.pt")
    blob = torch.load(path, map_location="cpu")

    a = blob["acts"].float()                      # [N, L, H, D]
    N, L, H, Dh = a.shape
    acts = a.reshape(N, L * H, Dh)
    labels, qids = blob["labels"], blob["qids"]
    mq = matched(labels, qids)
    print(f"[heads:{slug}] {args.pair_type}: {N} eps, {L} layers x {H} heads "
          f"= {L*H} positions, {len(mq)} matched questions")
    if len(mq) < 10:
        print("[heads] too few matched questions; aborting")
        return

    obs = loqo_auroc(acts, mq, labels)
    g = torch.Generator().manual_seed(0)
    null = torch.stack([loqo_auroc(acts, shuffled(mq, g), labels)
                        for _ in range(args.perms)])          # [perms, P]

    per_pos = null.mean(0) + 2 * null.std(0)                  # pointwise
    max_null = null.max(dim=1).values                         # [perms]
    fwer = torch.quantile(max_null, 0.95).item()

    best = int(obs.argmax())
    layer, head = divmod(best, H)
    val = float(obs[best])
    passes = val >= 0.70 and val > float(per_pos[best]) and val > fwer
    print(f"[heads:{slug}]   best L{layer} H{head}: AUROC {val:.3f}")
    print(f"[heads:{slug}]   pointwise shuffle+2sd {float(per_pos[best]):.3f}; "
          f"max-null 95th pct {fwer:.3f} (perms={args.perms})")
    print(f"[heads:{slug}]   GATE: {'PASS' if passes else 'FAIL'} "
          f"(need >=0.70, > pointwise, > max-null)")

    top = torch.topk(obs, 5)
    print(f"[heads:{slug}]   top 5:")
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        l, h = divmod(i, H)
        print(f"      L{l:>2} H{h:>2}  {v:.3f}  {'*' if v > fwer else ''}")
    print(f"[heads:{slug}]   heads above max-null: "
          f"{int((obs > fwer).sum())} of {L*H}")

    out = ROOT / cfg["paths"]["directions_dir"] / f"{args.pair_type}_head_scan.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"auroc": obs.reshape(L, H), "max_null": max_null,
                "pointwise": per_pos.reshape(L, H), "fwer_95": fwer,
                "best": (layer, head), "gate_pass": passes,
                "n_matched": len(mq), "perms": args.perms}, out)
    print(f"[heads:{slug}]   saved {out}")


if __name__ == "__main__":
    main()