"""v2 behavioral eval — batched, two-phase, with integrated screening.

SCREEN (turn-1 only, over a big pool; writes the per-model question set):
    python -m src.run_behavioral_eval_v2 --mode screen --pool data/pool.jsonl

MAIN (phase 1: all turn-1s in one batch; phase 2: all pushback turns in
one batch; templates via frozen paraphrase assignment):
    python -m src.run_behavioral_eval_v2 --mode main

ABLATED (hooked TransformerLens path — NOT batched; only runs when a
gate has passed, per pre-registration H6b):
    python -m src.run_behavioral_eval_v2 --mode ablated

Outputs use the active-model overlay: outputs/{slug}/transcripts/...
and data/questions_{slug}.jsonl.
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from .common import (ROOT, JUDGE, Transcript, format_chat, get_template,
                     load_config, pushback_types, read_jsonl, write_jsonl)


def get_tokenizer(cfg):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg["model"]["name"])


def chat1(tok, sp, q):
    return format_chat(tok, [{"role": "user", "content": q["question"]}], sp)


def chat2(tok, sp, q, initial, template):
    msgs = [
        {"role": "user", "content": q["question"]},
        {"role": "assistant", "content": initial},
        {"role": "user", "content": template},
    ]
    return format_chat(tok, msgs, sp)


# ------------------------------------------------------------------ screen

def run_screen(cfg, pool_path: Path):
    from .generation import VLLMGenerator
    slug = cfg.get("_slug", "model")
    tok = get_tokenizer(cfg)
    sp = cfg["model"]["system_prompt"]
    ev = cfg["eval"]

    pool = read_jsonl(pool_path)
    print(f"[screen:{slug}] {len(pool)} pool questions, one batch")
    gen = VLLMGenerator(cfg)
    prompts = [chat1(tok, sp, q) for q in pool]
    answers = gen.generate_batch(prompts)
    gen.close()

    correct, wrong, audit = [], [], []
    for q, a in zip(pool, answers):
        ok = JUDGE(a, q["answer"], q.get("aliases"))
        (correct if ok else wrong).append(q)
        audit.append({"qid": q["id"], "initial_response": a,
                      "initially_correct": ok})
    write_jsonl(ROOT / cfg["paths"]["transcripts_dir"] / "screen.jsonl", audit)

    cap = ev.get("eligible_cap", 10 ** 9)
    if len(correct) > cap:
        random.Random(0).shuffle(correct)
        correct = correct[:cap]
        print(f"[screen:{slug}] eligible capped at {cap} (seed 0)")
    qset = correct + wrong[: ev.get("keep_wrong", 30)]
    out = ROOT / f"data/questions_{slug}.jsonl"
    write_jsonl(out, qset)
    acc = (len([a for a in audit if a['initially_correct']])) / len(pool)
    print(f"[screen:{slug}] accuracy {acc:.1%}; question set = "
          f"{len(qset)} -> {out}")


def run_main(cfg, condition: str):
    slug = cfg.get("_slug", "model")
    sp = cfg["model"]["system_prompt"]
    ev = cfg["eval"]
    types = pushback_types(cfg)
    questions = read_jsonl(ROOT / cfg["paths"]["questions"])[: ev["n_questions"]]

    if condition in ("baseline_tl", "ablated"):
        from .make_questions import split_of
        questions = [q for q in questions if split_of(q["id"]) == "heldout"]
        print(f"[eval:{slug}:{condition}] held-out only: {len(questions)} questions")

    if condition == "baseline":
        from .generation import VLLMGenerator
        tok = get_tokenizer(cfg)
        gen = VLLMGenerator(cfg)
        batch = gen.generate_batch
    else: 
        from .common import generate, load_model
        model = load_model(cfg)
        tok = model
        hooks = None
        if condition == "ablated":
            from .ablation import load_ablation_hooks
            hooks = load_ablation_hooks(model, cfg)
            print(f"[eval:{slug}] ablated with {len(hooks)} hooks (sequential)")
        else:
            print(f"[eval:{slug}] TL baseline, no hooks (sequential)")

        def batch(prompts):
            return [generate(model, p, ev["max_new_tokens"],
                             ev["temperature"], fwd_hooks=hooks)
                    for p in prompts]
        gen = None

    p1 = [chat1(tok, sp, q) for q in questions]
    print(f"[eval:{slug}:{condition}] phase 1: {len(p1)} prompts")
    initials = batch(p1)
    init_ok = [JUDGE(a, q["answer"], q.get("aliases"))
               for q, a in zip(questions, initials)]

    episodes = []
    for qi, q in enumerate(questions):
        for t in types:
            tmpl, pidx = get_template(cfg, q["id"], t)
            episodes.append((qi, t, tmpl, pidx))
    p2 = [chat2(tok, sp, questions[qi], initials[qi], tmpl)
          for qi, _, tmpl, _ in episodes]
    print(f"[eval:{slug}:{condition}] phase 2: {len(p2)} prompts")
    finals = batch(p2)
    if gen is not None:
        gen.close()

    rows: list[Transcript] = []
    for qi, q in enumerate(questions):
        rows.append(Transcript(
            qid=q["id"], question=q["question"], correct_answer=q["answer"],
            pushback_type="none", condition=condition,
            initial_response=initials[qi], initially_correct=init_ok[qi]))
    for (qi, t, tmpl, pidx), final in zip(episodes, finals):
        q = questions[qi]
        fin_ok = JUDGE(final, q["answer"], q.get("aliases"))
        rows.append(Transcript(
            qid=q["id"], question=q["question"], correct_answer=q["answer"],
            pushback_type=t, condition=condition,
            initial_response=initials[qi], final_response=final,
            initially_correct=init_ok[qi], finally_correct=fin_ok,
            capitulated=(init_ok[qi] and not fin_ok),
            paraphrase_idx=pidx))

    out = ROOT / cfg["paths"]["transcripts_dir"] / f"{condition}.jsonl"
    write_jsonl(out, rows)
    summarize(rows, f"{slug}:{condition}")


def summarize(rows, label):
    stats = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.pushback_type == "none" or not r.initially_correct:
            continue
        stats[r.pushback_type][1] += 1
        stats[r.pushback_type][0] += int(bool(r.capitulated))
    print(f"\n== {label}: destabilization | initially correct (heuristic) ==")
    for t, (c, n) in sorted(stats.items()):
        print(f"  {t:15s}  {c:3d}/{n:<3d}  {c / n if n else float('nan'):6.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["screen", "main", "ablated", "baseline_tl"],
                    required=True)
    ap.add_argument("--pool", type=Path, default=ROOT / "data/pool.jsonl")
    args = ap.parse_args()
    cfg = load_config()
    if args.mode == "screen":
        run_screen(cfg, args.pool)
    else:
        run_main(cfg, "baseline" if args.mode == "main" else args.mode)
