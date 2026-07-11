"""Steps 1 & 5 — Run the multi-turn sycophancy eval on the local model.

Baseline (step 1):
    python -m src.run_behavioral_eval --condition baseline

Ablated (step 5, after extract_directions.py):
    python -m src.run_behavioral_eval --condition ablated

For every question:
  turn 1: ask question               -> initial_response, initially_correct
  turn 2: apply each pushback type   -> final_response, capitulated?
Also records a pushback_type="none" row per question (turn 1 only) — these
are the negative examples for the "pushback" contrast pairs.

Output: outputs/transcripts/{condition}.jsonl
Headline metric printed at the end:
  destabilization rate = P(final wrong | initially correct), per pushback type.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from .common import (ROOT, JUDGE, Transcript, format_chat, generate,
                     load_config, load_model, read_jsonl, write_jsonl)


def run(condition: str):
    cfg = load_config()
    model = load_model(cfg)
    sp = cfg["model"]["system_prompt"]
    ev = cfg["eval"]

    hooks = None
    if condition == "ablated":
        from .ablation import load_ablation_hooks
        hooks = load_ablation_hooks(model, cfg)
        print(f"[eval] running with {len(hooks)} ablation hooks")

    questions = read_jsonl(ROOT / cfg["paths"]["questions"])[: ev["n_questions"]]
    rows: list[Transcript] = []

    for q in tqdm(questions, desc=f"eval:{condition}"):
        # ---- turn 1
        msgs = [{"role": "user", "content": q["question"]}]
        prompt1 = format_chat(model, msgs, sp)
        initial = generate(model, prompt1, ev["max_new_tokens"],
                           ev["temperature"], fwd_hooks=hooks)
        init_ok = JUDGE(initial, q["answer"], q.get("aliases"))

        rows.append(Transcript(
            qid=q["id"], question=q["question"], correct_answer=q["answer"],
            pushback_type="none", condition=condition,
            initial_response=initial, initially_correct=init_ok,
        ))

        # ---- turn 2, one episode per pushback type
        for ptype, template in ev["pushback_templates"].items():
            msgs2 = msgs + [
                {"role": "assistant", "content": initial},
                {"role": "user", "content": template},
            ]
            prompt2 = format_chat(model, msgs2, sp)
            final = generate(model, prompt2, ev["max_new_tokens"],
                             ev["temperature"], fwd_hooks=hooks)
            fin_ok = JUDGE(final, q["answer"], q.get("aliases"))
            rows.append(Transcript(
                qid=q["id"], question=q["question"], correct_answer=q["answer"],
                pushback_type=ptype, condition=condition,
                initial_response=initial, final_response=final,
                initially_correct=init_ok, finally_correct=fin_ok,
                capitulated=(init_ok and not fin_ok),
            ))

    out = ROOT / cfg["paths"]["transcripts_dir"] / f"{condition}.jsonl"
    write_jsonl(out, rows)
    summarize(rows, condition)


def summarize(rows: list[Transcript], condition: str):
    """Destabilization rate conditioned on initially-correct, per pushback
    type. (Point estimates only — clustered CIs live in your Inspect harness;
    port them over for the paper numbers.)"""
    stats = defaultdict(lambda: [0, 0])  # ptype -> [capitulated, initially_correct]
    for r in rows:
        if r.pushback_type == "none" or not r.initially_correct:
            continue
        stats[r.pushback_type][1] += 1
        stats[r.pushback_type][0] += int(bool(r.capitulated))

    print(f"\n== {condition}: destabilization | initially correct ==")
    for ptype, (cap, n) in sorted(stats.items()):
        rate = cap / n if n else float("nan")
        print(f"  {ptype:15s}  {cap:3d}/{n:<3d}  {rate:6.1%}")
    if not stats:
        print("  (no initially-correct episodes — check the judge or model size)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["baseline", "ablated"],
                    default="baseline")
    run(ap.parse_args().condition)
