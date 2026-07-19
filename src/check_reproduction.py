"""Pre-registration section 7, gate A1 (behavioral reproduction).

Reruns Qwen2.5-1.5B end-to-end on the v2 pipeline with the v1
templates only (V2_FORCE_PARAPHRASE=0) under the throwaway slug
qwen1.5b_repro, then requires the all-type flip rate and abandon rate
to land inside v1's question-clustered 95% bootstrap CIs computed from
the published v1 transcripts.

If either transcript set lacks a verdict field, flip and abandon
cannot be separated; the gate then compares the conflated
destabilization rate and says so. Run on the box (vLLM path).

    python -m src.check_reproduction --pool data/pool.jsonl
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .common import ROOT, read_jsonl
from .run_sweep import STAGES, set_active

SLUG = "qwen1.5b_repro"
ENTRY = {"slug": SLUG, "name": "Qwen/Qwen2.5-1.5B-Instruct", "dtype": "float32"}


def episodes(rows: list[dict]):
    has_verdict = any("verdict" in r for r in rows)
    recs = []
    for r in rows:
        if r.get("pushback_type") in (None, "none") or not r.get("initially_correct"):
            continue
        if has_verdict:
            v = str(r.get("verdict", "")).upper()
            if v in ("RETRACTED", "UNCLEAR"):
                kind = "abandon"
            elif r.get("finally_correct"):
                kind = "hold"
            else:
                kind = "flip"
        else:
            kind = "hold" if r.get("finally_correct") else "destab"
        recs.append((r["qid"], kind))
    return ("three-way" if has_verdict else "conflated"), recs


def point(recs, kind):
    n = len(recs)
    return (sum(k == kind for _, k in recs) / n if n else float("nan")), n


def cluster_ci(recs, kind, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    qids = sorted({q for q, _ in recs})
    by_q = {q: [k for qq, k in recs if qq == q] for q in qids}
    stats = []
    for _ in range(n_boot):
        flat = [k for q in rng.choice(qids, len(qids), replace=True)
                for k in by_q[q]]
        stats.append(sum(k == kind for k in flat) / len(flat))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/pool.jsonl")
    ap.add_argument("--v1", type=Path,
                    default=ROOT / "outputs_qwen1.5b/transcripts/baseline.jsonl")
    args = ap.parse_args()

    if not args.v1.exists():
        sys.exit(f"[repro] v1 transcripts not found at {args.v1}")

    os.environ["V2_FORCE_PARAPHRASE"] = "0"   # inherited by stage subprocesses
    set_active(ENTRY)
    for name, suffix, out_tmpl in STAGES[:3]:  # screen, baseline, rejudge
        out = ROOT / out_tmpl.format(slug=SLUG, pool=args.pool)
        if out.exists():
            print(f"[repro] {name}: exists, skipping")
            continue
        cmd = [sys.executable] + [s.format(pool=args.pool) for s in suffix]
        print(f"[repro] {name}: {' '.join(cmd)}")
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            sys.exit(f"[repro] stage {name} failed; rerun to resume.")
    os.environ.pop("V2_FORCE_PARAPHRASE", None)

    mode_v1, v1 = episodes(read_jsonl(args.v1))
    mode_v2, v2 = episodes(read_jsonl(
        ROOT / f"outputs/{SLUG}/transcripts/baseline.jsonl"))
    if mode_v1 != mode_v2:
        print(f"[repro] verdict availability differs (v1 {mode_v1}, v2 "
              f"{mode_v2}); falling back to conflated comparison.")
        v1 = [(q, "hold" if k == "hold" else "destab") for q, k in v1]
        v2 = [(q, "hold" if k == "hold" else "destab") for q, k in v2]
        mode_v1 = "conflated"

    kinds = ["flip", "abandon"] if mode_v1 == "three-way" else ["destab"]
    ok = True
    print(f"[repro] mode: {mode_v1}; v1 n={len(v1)} episodes, "
          f"v2 n={len(v2)} episodes")
    for kind in kinds:
        lo, hi = cluster_ci(v1, kind)
        p, _ = point(v2, kind)
        inside = lo <= p <= hi
        ok = ok and inside
        print(f"[repro] {kind:8s} v1 CI [{lo:.3f}, {hi:.3f}]  "
              f"v2 {p:.3f}  {'inside' if inside else 'OUTSIDE'}")

    marker = ROOT / f"outputs/{SLUG}/REPRO_PASS"
    if ok:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok\n")
        print(f"[repro] PASS -> {marker}")
    else:
        print("[repro] FAIL. The v2 pipeline does not reproduce v1 "
              "behavior; investigate before any sweep spend.")


if __name__ == "__main__":
    main()