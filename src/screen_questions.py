"""Screen a large question pool with turn-1 generations only.

Full protocol costs 5 generations/question; only turn 1 decides
eligibility. This runs turn 1 over a big pool and emits:

  data/questions_batch2.jsonl  -> NEW questions to run the full protocol on
                                  (screened-correct + up to K screened-wrong
                                  as the valid-correction control)
  data/questions_all.jsonl     -> batch2 + carried-forward eligible questions
                                  from the previous baseline (for the
                                  ABLATED run, which must cover everything)

Also writes outputs/transcripts/screen.jsonl as an audit trail.

    python -m src.screen_questions data/questions_pool2.jsonl \
        --exclude outputs/transcripts/baseline_b1.jsonl \
        --old-questions data/questions.jsonl \
        --keep-wrong 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .analyze import split_of
from .common import (ROOT, JUDGE, format_chat, generate, load_config,
                     load_model, read_jsonl, write_jsonl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", type=Path)
    ap.add_argument("--exclude", type=Path, default=None,
                    help="previous baseline transcripts: skip its qids, "
                         "carry forward its initially-correct questions")
    ap.add_argument("--old-questions", type=Path, default=None,
                    help="question file matching --exclude, for carry-forward")
    ap.add_argument("--keep-wrong", type=int, default=30)
    args = ap.parse_args()

    cfg = load_config()
    ev = cfg["eval"]
    model = load_model(cfg)
    sp = cfg["model"]["system_prompt"]

    done_qids, old_eligible = set(), set()
    if args.exclude and args.exclude.exists():
        prev = read_jsonl(args.exclude)
        done_qids = {r["qid"] for r in prev}
        old_eligible = {r["qid"] for r in prev if r["initially_correct"]}
        print(f"[screen] excluding {len(done_qids)} already-run qids; "
              f"{len(old_eligible)} eligible carried forward")

    pool = [q for q in read_jsonl(args.pool) if q["id"] not in done_qids]
    print(f"[screen] screening {len(pool)} new questions (turn 1 only)")

    correct, wrong, audit = [], [], []
    for q in tqdm(pool, desc="screen"):
        prompt = format_chat(model, [{"role": "user", "content": q["question"]}], sp)
        resp = generate(model, prompt, ev["max_new_tokens"], ev["temperature"])
        ok = JUDGE(resp, q["answer"], q.get("aliases"))
        (correct if ok else wrong).append(q)
        audit.append({"qid": q["id"], "initial_response": resp,
                      "initially_correct": ok})

    write_jsonl(ROOT / cfg["paths"]["transcripts_dir"] / "screen.jsonl", audit)

    batch2 = correct + wrong[: args.keep_wrong]
    write_jsonl(ROOT / "data/questions_batch2.jsonl", batch2)

    carried = []
    if args.old_questions and args.old_questions.exists():
        carried = [q for q in read_jsonl(args.old_questions)
                   if q["id"] in old_eligible]
    write_jsonl(ROOT / "data/questions_all.jsonl", batch2 + carried)

    acc = len(correct) / len(pool) if pool else float("nan")
    n_ex = sum(split_of(q["id"]) == "extract" for q in correct)
    print(f"\n[screen] initial accuracy: {len(correct)}/{len(pool)} ({acc:.1%})")
    print(f"[screen] batch2: {len(correct)} eligible + "
          f"{min(len(wrong), args.keep_wrong)} control-wrong")
    print(f"[screen] eligible split: {n_ex} extract / {len(correct) - n_ex} heldout")
    print(f"[screen] questions_all.jsonl: {len(batch2) + len(carried)} total "
          f"(for the ablated run)")
    print("\nnext: set questions: data/questions_batch2.jsonl in config, "
          "run baseline; before the ABLATED run switch it to "
          "data/questions_all.jsonl")


if __name__ == "__main__":
    main()