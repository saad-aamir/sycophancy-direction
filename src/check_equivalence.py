"""Pre-registration acceptance test 6.1 — generation equivalence.

vLLM-bf16 greedy vs TransformerLens greedy on 20 turn-1 prompts.
PASS iff >= 90% of continuations are token-identical for the first 50
tokens (compared as normalized text prefixes). Writes
outputs/{slug}/EQUIV_PASS on success; run_sweep refuses to start
without it.

    python -m src.check_equivalence --pool data/pool.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import ROOT, format_chat, generate, load_config, load_model, read_jsonl


def norm_prefix(text: str, tokenizer, k: int = 50) -> tuple:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:k]
    return tuple(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=ROOT / "data/pool.jsonl")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config()
    slug = cfg.get("_slug", "model")
    sp = cfg["model"]["system_prompt"]
    ev = cfg["eval"]
    questions = read_jsonl(args.pool)[: args.n]

    # vLLM side
    from .generation import VLLMGenerator
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    prompts = [format_chat(tok, [{"role": "user", "content": q["question"]}],
                           sp) for q in questions]
    vgen = VLLMGenerator(cfg)
    v_out = vgen.generate_batch(prompts)
    vgen.close()

    # TransformerLens side
    model = load_model(cfg)
    t_out = [generate(model, p, ev["max_new_tokens"], ev["temperature"])
             for p in prompts]

    same = sum(norm_prefix(a, tok) == norm_prefix(b, tok)
               for a, b in zip(v_out, t_out))
    rate = same / len(prompts)
    print(f"[equiv:{slug}] identical 50-token prefixes: "
          f"{same}/{len(prompts)} ({rate:.0%})")
    for i, (a, b) in enumerate(zip(v_out, t_out)):
        if norm_prefix(a, tok) != norm_prefix(b, tok):
            print(f"  MISMATCH q{i}:\n    vllm: {a[:100]!r}\n    tl:   {b[:100]!r}")

    marker = ROOT / f"outputs/{slug}/EQUIV_PASS"
    if rate >= 0.90:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"{rate:.2f}\n")
        print(f"[equiv:{slug}] PASS -> {marker}")
    else:
        print(f"[equiv:{slug}] FAIL (< 90%) — investigate before spending "
              f"(dtype kernels, stop-token handling, chat template).")


if __name__ == "__main__":
    main()
