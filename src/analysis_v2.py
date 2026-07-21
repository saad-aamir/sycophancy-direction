"""Consolidated v2 analysis tables: H1, H3, H5, H6.

Reads committed transcripts and direction files, prints every number
that appears in the paper. Run from the repo root:

    python -m src.analysis_v2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen0.5b", "qwen1.5b", "qwen3b", "qwen7b",
          "llama1b", "llama3b", "llama8b")
TYPES = ("authoritative", "emotional", "simple", "social")


def rows(slug, condition="baseline"):
    return [json.loads(l) for l in
            open(ROOT / f"outputs/{slug}/transcripts/{condition}.jsonl")]


def h1():
    print("=== H1: matched LOQO AUROC, residual stream ===")
    for s in MODELS:
        p = ROOT / f"outputs/{s}/directions/capitulation_directions.pt"
        if not p.exists():
            print(f"{s:9s} missing"); continue
        b = torch.load(p, map_location="cpu")
        L = b["best_layer"]
        print(f"{s:9s} best L{L:>2}  AUROC {float(b['auroc'][L]):.3f}  "
              f"gate {'PASS' if b.get('gate_pass') else 'FAIL'}")


def h3():
    print("\n=== H3: McNemar emotional vs simple, flip-only, Bonferroni x4 ===")
    for s in MODELS:
        try:
            R = rows(s)
        except FileNotFoundError:
            print(f"{s:9s} missing"); continue
        d = {}
        for r in R:
            if r.get("pushback_type") not in TYPES or not r.get("initially_correct"):
                continue
            v = str(r.get("final_verdict", "")).upper()
            if v in ("RETRACTED", "UNCLEAR"):
                continue
            d[(r["qid"], r["pushback_type"])] = v == "INCORRECT"
        qs = ({q for (q, t) in d if t == "emotional"}
              & {q for (q, t) in d if t == "simple"})
        ab = sum(1 for q in qs if d[(q, "emotional")] and not d[(q, "simple")])
        ba = sum(1 for q in qs if d[(q, "simple")] and not d[(q, "emotional")])
        p = binomtest(ab, ab + ba, 0.5).pvalue if ab + ba else 1.0
        orr = ab / ba if ba else float("inf")
        print(f"{s:9s} emo {ab:3d} vs simple {ba:3d}  OR {orr:5.2f}  "
              f"pBonf={min(p * 4, 1):.4f}  (paired q={len(qs)})")


def h5():
    print("\n=== H5: flip vs recovery ===")
    for s in MODELS:
        try:
            R = rows(s)
        except FileNotFoundError:
            print(f"{s:9s} missing"); continue
        init = {r["qid"]: r.get("initially_correct")
                for r in R if (r.get("pushback_type") or "none") == "none"}
        flip = [str(r.get("final_verdict", "")).upper() == "INCORRECT"
                for r in R if r.get("pushback_type") in TYPES
                and init.get(r["qid"]) is True
                and str(r.get("final_verdict", "")).upper()
                not in ("RETRACTED", "UNCLEAR")]
        rec = [str(r.get("final_verdict", "")).upper() == "CORRECT"
               for r in R if r.get("pushback_type") in TYPES
               and init.get(r["qid"]) is False]
        f, rc = np.mean(flip), np.mean(rec)
        print(f"{s:9s} flip {100 * f:5.1f}% (n={len(flip)})   "
              f"recovery {100 * rc:5.1f}% (n={len(rec)})   "
              f"ratio {f / rc if rc else float('inf'):.1f}x")


def h6(slug):
    def load(cond):
        out = {}
        for r in rows(slug, cond):
            out.setdefault(r["qid"], {})[r.get("pushback_type") or "none"] = r
        return out
    try:
        A, B = load("baseline_tl"), load("ablated")
    except FileNotFoundError:
        print(f"{slug:9s} no H6 arms"); return
    qs = [q for q in A if q in B
          and A[q]["none"].get("initially_correct")
          and B[q]["none"].get("initially_correct")]
    flip = lambda D, q, t: str(D[q][t].get("final_verdict", "")).upper() == "INCORRECT"
    aban = lambda D, q, t: str(D[q][t].get("final_verdict", "")).upper() in ("RETRACTED", "UNCLEAR")

    def rate(D, sample, fn):
        ep = [(q, t) for q in sample for t in TYPES if t in D[q]]
        return np.mean([fn(D, q, t) for q, t in ep])

    rng = np.random.default_rng(0)
    d = [rate(B, s, flip) - rate(A, s, flip)
         for s in (list(rng.choice(qs, len(qs), replace=True))
                   for _ in range(2000))]
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"{slug:9s} paired q={len(qs)}  "
          f"baseline flip {100 * rate(A, qs, flip):.1f}% abandon {100 * rate(A, qs, aban):.1f}%   "
          f"ablated flip {100 * rate(B, qs, flip):.1f}% abandon {100 * rate(B, qs, aban):.1f}%   "
          f"change {100 * np.mean(d):+.1f}pp CI [{100 * lo:+.1f}, {100 * hi:+.1f}]")


if __name__ == "__main__":
    h1()
    h3()
    h5()
    print("\n=== H6: ablation, paired on questions correct in both arms ===")
    for s in ("qwen1.5b", "qwen7b", "llama8b"):
        h6(s)