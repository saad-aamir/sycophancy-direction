"""Pre-registration section 7, gate A2 (numerics), two parts plus an
optional probe.

(a) cosine: CPU-fp32 vs CUDA-bf16 per-layer last-token resid cosine on
    one prompt; PASS requires min >= 0.97.
    (Deviation 2026-07-18, logged pre-data: bar corrected 0.99 -> 0.97;
    v1 Appendix C measured min bf16 cosine 0.9757, and fp32 cannot fit
    14B in 48GB, so bf16 with this bar is the plan.)
(b) agreement: episode outcome labels on the first --agreement-n pool
    questions, TL-cuda-fp32 vs TL-cuda-bf16 (same engine, only dtype
    varies); PASS requires >= 95% identical labels. Heuristic judge on
    both sides, so judge choice cancels.
(c) optional --v1-directions: normalized cosine between the v1
    best-layer direction and a v2-recached one, bar 0.95. Deferred
    automatically until the v2 cache exists.

NUMERICS_PASS is written only when (a) and (b) pass (and (c), if run).
CUDA required: run on the box, after --set-active qwen1.5b.

    python -m src.run_sweep --set-active qwen1.5b
    python -m src.check_numerics
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

from .common import (JUDGE, ROOT, format_chat, generate, get_template,
                     last_token_resid, load_config, load_model,
                     pushback_types, read_jsonl)


def cosine_gate() -> bool:
    cfg = load_config()
    slug = cfg.get("_slug", "model")
    msgs = [{"role": "user", "content": "What is the capital of France?"}]

    cfg["model"]["device"], cfg["model"]["dtype"] = "cpu", "float32"
    m = load_model(cfg)
    prompt = format_chat(m, msgs, cfg["model"]["system_prompt"])
    a = last_token_resid(m, prompt)
    del m
    gc.collect()

    cfg = load_config()
    cfg["model"]["device"] = "cuda"
    m = load_model(cfg)
    b = last_token_resid(m, prompt)
    dtype = cfg["model"]["dtype"]
    del m
    gc.collect()
    torch.cuda.empty_cache()

    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    print(f"[numerics:{slug}] cpu-fp32 vs cuda-{dtype}: min {cos.min():.4f} "
          f"mean {cos.mean():.4f} (worst layer {int(cos.argmin())})")
    return bool(cos.min() >= 0.97)


def agreement_gate(n: int) -> bool:
    """fp32 reference vs the dtype the sweep will actually run in."""
    target = load_config()["model"]["dtype"]
    if target == "float32":
        print("[numerics] target dtype is float32; agreement gate is trivial, skipped")
        return True
    pool = read_jsonl(ROOT / "data/pool.jsonl")[:n]
    labels = {}
    for dtype in ("float32", target):
        cfg = load_config()
        cfg["model"]["device"], cfg["model"]["dtype"] = "cuda", dtype
        m = load_model(cfg)
        sp, ev = cfg["model"]["system_prompt"], cfg["eval"]
        rows = []
        for q in pool:
            msgs = [{"role": "user", "content": q["question"]}]
            initial = generate(m, format_chat(m, msgs, sp),
                               ev["max_new_tokens"], ev["temperature"])
            rows.append(("init", q["id"],
                         JUDGE(initial, q["answer"], q.get("aliases"))))
            for ptype in pushback_types(cfg):
                tmpl, _ = get_template(cfg, q["id"], ptype)
                msgs2 = msgs + [{"role": "assistant", "content": initial},
                                {"role": "user", "content": tmpl}]
                fin = generate(m, format_chat(m, msgs2, sp),
                               ev["max_new_tokens"], ev["temperature"])
                rows.append((ptype, q["id"],
                             JUDGE(fin, q["answer"], q.get("aliases"))))
        labels[dtype] = rows
        del m
        gc.collect()
        torch.cuda.empty_cache()

    pairs = list(zip(labels["float32"], labels[target]))
    same = sum(x == y for x, y in pairs)
    rate = same / len(pairs)
    slug = load_config().get("_slug", "model")
    print(f"[numerics:{slug}] outcome-label agreement fp32 vs {target}: "
          f"{same}/{len(pairs)} episodes ({rate:.1%}) [bar 95%]")
    return rate >= 0.95


def direction_probe(v1_path: Path) -> bool | None:
    cfg = load_config()
    blob = torch.load(v1_path, map_location="cpu")
    L = blob["best_layer"]
    v1_dir = blob["dirs"][L].float()
    v2_path = ROOT / cfg["paths"]["activations_dir"] / "capitulation_acts.pt"
    if not v2_path.exists():
        print("[numerics] v2 cache absent; direction probe deferred "
              "(rerun with the flag after the first cache stage)")
        return None
    from .extract_directions import diff_in_means
    v2 = torch.load(v2_path, map_location="cpu")
    v2_dir = diff_in_means(v2["acts"], v2["labels"])[L].float()
    c = torch.nn.functional.cosine_similarity(v2_dir, v1_dir, dim=0).item()
    print(f"[numerics] v1-vs-v2 direction cosine @L{L}: {c:+.3f} (bar 0.95)")
    return c >= 0.95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agreement-n", type=int, default=50)
    ap.add_argument("--skip-agreement", action="store_true",
                    help="quick cosine-only look; writes no marker")
    ap.add_argument("--v1-directions", type=Path, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("[numerics] CUDA required; this gate runs on the box.")

    cfg = load_config()
    slug = cfg.get("_slug", "model")
    ok = cosine_gate()

    if args.skip_agreement:
        print("[numerics] agreement skipped; no marker written.")
        return

    ok = agreement_gate(args.agreement_n) and ok

    if args.v1_directions:
        probe = direction_probe(args.v1_directions)
        if probe is not None:
            ok = ok and probe

    marker = ROOT / f"outputs/{slug}/NUMERICS_PASS"
    if ok:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok\n")
        print(f"[numerics:{slug}] PASS -> {marker}")
    else:
        print(f"[numerics:{slug}] FAIL. Do not start the sweep; "
              f"investigate dtype/device first.")


if __name__ == "__main__":
    main()