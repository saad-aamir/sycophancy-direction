"""Step 2 — Build contrast pairs from the baseline transcripts.

Two contrasts, both saved:

1. "capitulation" (primary): among episodes where the model was initially
   correct and received pushback, contrast the *prompts at the moment before
   the post-pushback response* for episodes where it capitulated (pos) vs
   held firm (neg). Direction ~ "about to capitulate".

2. "pushback": same question, prompt with pushback (pos) vs the turn-1
   prompt without (neg). Direction ~ "perceiving pushback". No labels
   needed; useful control/comparison.

Each pair row stores the full prompt string whose last-token activations we
cache in step 3, plus pushback_type for the per-type decomposition.

    python -m src.build_contrast_pairs
"""
from __future__ import annotations

from .common import ROOT, format_chat, load_config, load_model, read_jsonl, write_jsonl

from .analyze import split_of


def prompt_after_pushback(model, sp, r: dict, template: str) -> str:
    msgs = [
        {"role": "user", "content": r["question"]},
        {"role": "assistant", "content": r["initial_response"]},
        {"role": "user", "content": template},
    ]
    return format_chat(model, msgs, sp)


def main():
    cfg = load_config()
    # model needed only for its tokenizer/chat template
    model = load_model(cfg)
    sp = cfg["model"]["system_prompt"]
    templates = cfg["eval"]["pushback_templates"]

    rows = read_jsonl(ROOT / cfg["paths"]["transcripts_dir"] / "baseline.jsonl")

    cap_pairs, push_pairs = [], []
    for r in rows:
        if split_of(r["qid"]) != "extract":
            continue
        if r["pushback_type"] == "none":
            continue
        template = templates[r["pushback_type"]]
        p2 = prompt_after_pushback(model, sp, r, template)
        p1 = format_chat(model, [{"role": "user", "content": r["question"]}], sp)

        # contrast 2: pushback vs none (all episodes)
        push_pairs.append({"qid": r["qid"], "pushback_type": r["pushback_type"],
                           "label": "pos", "prompt": p2})
        push_pairs.append({"qid": r["qid"], "pushback_type": r["pushback_type"],
                           "label": "neg", "prompt": p1})

        # contrast 1: capitulated vs held-firm (initially-correct only)
        if r["initially_correct"]:
            label = "pos" if r["capitulated"] else "neg"
            cap_pairs.append({"qid": r["qid"],
                              "pushback_type": r["pushback_type"],
                              "label": label, "prompt": p2})

    n_max = cfg["extraction"]["n_pairs_max"]
    out_dir = ROOT / cfg["paths"]["activations_dir"]
    write_jsonl(out_dir / "pairs_capitulation.jsonl", cap_pairs[:n_max])
    write_jsonl(out_dir / "pairs_pushback.jsonl", push_pairs[:n_max])

    pos = sum(p["label"] == "pos" for p in cap_pairs)
    print(f"[pairs] capitulation: {pos} pos / {len(cap_pairs) - pos} neg "
          f"(need both >~15 for a stable mean — if not, raise n_questions)")


if __name__ == "__main__":
    main()
