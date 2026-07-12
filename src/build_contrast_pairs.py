"""Step 2 — Build contrast pairs from the baseline transcripts.

Class definitions (D12 in RESEARCH_LOG):
  positive ("capitulation") = final_verdict INCORRECT — the model flipped
      from a correct answer to a wrong one under pushback.
  negative ("held firm")    = final_verdict CORRECT among initially-correct
      episodes (recommitted to its answer).
  EXCLUDED = RETRACTED / UNCLEAR ("abandon": drops its answer without
      committing to a new one) — behaviorally distinct from sycophantic
      flipping; reported separately, never used for direction extraction.
Rows without final_verdict (not claude-judged) fall back to the boolean
capitulated label.

Only questions in the "extract" split are used (held-out protocol, D6).
Also builds "pushback" pairs (pushback-present vs absent prompts) as a
control contrast; outcome verdicts are irrelevant to that contrast.

    python -m src.build_contrast_pairs
"""
from __future__ import annotations

from .analyze import split_of
from .common import ROOT, format_chat, load_config, load_model, read_jsonl, write_jsonl


def prompt_after_pushback(model, sp, r: dict, template: str) -> str:
    msgs = [
        {"role": "user", "content": r["question"]},
        {"role": "assistant", "content": r["initial_response"]},
        {"role": "user", "content": template},
    ]
    return format_chat(model, msgs, sp)


def classify(r: dict) -> str | None:
    """'pos' | 'neg' | None (excluded)."""
    if not r["initially_correct"]:
        return None
    fv = r.get("final_verdict")
    if fv == "INCORRECT":
        return "pos"
    if fv == "CORRECT":
        return "neg"
    if fv is None:  # heuristic-only rows
        return "pos" if r["capitulated"] else "neg"
    return None     # RETRACTED / UNCLEAR -> abandon


def main():
    cfg = load_config()
    # model needed only for its tokenizer/chat template
    model = load_model(cfg)
    sp = cfg["model"]["system_prompt"]
    templates = cfg["eval"]["pushback_templates"]

    rows = read_jsonl(ROOT / cfg["paths"]["transcripts_dir"] / "baseline.jsonl")

    cap_pairs, push_pairs, n_abandon = [], [], 0
    for r in rows:
        if r["pushback_type"] == "none":
            continue
        if split_of(r["qid"]) != "extract":
            continue
        template = templates[r["pushback_type"]]
        p2 = prompt_after_pushback(model, sp, r, template)
        p1 = format_chat(model, [{"role": "user", "content": r["question"]}], sp)

        # control contrast: pushback vs none (all episodes)
        push_pairs.append({"qid": r["qid"], "pushback_type": r["pushback_type"],
                           "label": "pos", "prompt": p2})
        push_pairs.append({"qid": r["qid"], "pushback_type": r["pushback_type"],
                           "label": "neg", "prompt": p1})

        # primary contrast: flipped vs held-firm
        label = classify(r)
        if label is None:
            n_abandon += int(bool(r["initially_correct"]))
            continue
        cap_pairs.append({"qid": r["qid"], "pushback_type": r["pushback_type"],
                          "label": label, "prompt": p2})

    n_max = cfg["extraction"]["n_pairs_max"]
    out_dir = ROOT / cfg["paths"]["activations_dir"]
    write_jsonl(out_dir / "pairs_capitulation.jsonl", cap_pairs[:n_max])
    write_jsonl(out_dir / "pairs_pushback.jsonl", push_pairs[:n_max])

    pos = sum(p["label"] == "pos" for p in cap_pairs)
    print(f"[pairs] capitulation (extract split): {pos} flips (pos) / "
          f"{len(cap_pairs) - pos} held-firm (neg) / {n_abandon} abandons excluded")
    if pos < 15:
        print("[pairs] WARNING: <15 positives — direction will be noisy; "
              "wait for batch 2 before trusting the AUROC")


if __name__ == "__main__":
    main()