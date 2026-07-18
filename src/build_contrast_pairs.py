"""Step — Build contrast pairs (v2: paraphrase-aware).

Identical class logic to v1/D12 (positives = INCORRECT finals, negatives
= CORRECT finals, RETRACTED/UNCLEAR excluded; extract split only). New:
prompts are reconstructed with the SAME paraphrase the episode was run
with (frozen md5 assignment), and each pair row carries template_text +
paraphrase_idx so the multi-locus cacher can locate the final user
message span.

    python -m src.build_contrast_pairs
"""
from __future__ import annotations

from .analyze import split_of
from .common import (ROOT, format_chat, get_template, load_config,
                     read_jsonl, write_jsonl)


def classify(r: dict) -> str | None:
    if not r["initially_correct"]:
        return None
    fv = r.get("final_verdict")
    if fv == "INCORRECT":
        return "pos"
    if fv == "CORRECT":
        return "neg"
    if fv is None:
        return "pos" if r.get("capitulated") else "neg"
    return None


def main():
    cfg = load_config()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    sp = cfg["model"]["system_prompt"]

    rows = read_jsonl(ROOT / cfg["paths"]["transcripts_dir"] / "baseline.jsonl")

    cap_pairs, push_pairs, n_abandon = [], [], 0
    for r in rows:
        if r["pushback_type"] == "none":
            continue
        if split_of(r["qid"]) != "extract":
            continue
        tmpl, pidx = get_template(cfg, r["qid"], r["pushback_type"])
        rec_pidx = r.get("paraphrase_idx")
        if rec_pidx is not None and rec_pidx != pidx:
            # templates file changed after the run -> refuse silently wrong data
            raise RuntimeError(
                f"paraphrase mismatch for {r['qid']}/{r['pushback_type']}: "
                f"transcript has {rec_pidx}, current templates give {pidx}. "
                f"templates_v2.yaml must not change after a run.")
        p2 = format_chat(tok, [
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r["initial_response"]},
            {"role": "user", "content": tmpl},
        ], sp)
        p1 = format_chat(tok, [{"role": "user", "content": r["question"]}], sp)

        push_pairs.append({"qid": r["qid"], "pushback_type": r["pushback_type"],
                           "label": "pos", "prompt": p2,
                           "template_text": tmpl, "paraphrase_idx": pidx})
        push_pairs.append({"qid": r["qid"], "pushback_type": r["pushback_type"],
                           "label": "neg", "prompt": p1,
                           "template_text": r["question"],
                           "paraphrase_idx": pidx})

        label = classify(r)
        if label is None:
            n_abandon += int(bool(r["initially_correct"]))
            continue
        cap_pairs.append({"qid": r["qid"], "pushback_type": r["pushback_type"],
                          "label": label, "prompt": p2,
                          "template_text": tmpl, "paraphrase_idx": pidx})

    n_max = cfg["extraction"]["n_pairs_max"]
    out_dir = ROOT / cfg["paths"]["activations_dir"]
    write_jsonl(out_dir / "pairs_capitulation.jsonl", cap_pairs[:n_max])
    write_jsonl(out_dir / "pairs_pushback.jsonl", push_pairs[:n_max])

    pos = sum(p["label"] == "pos" for p in cap_pairs)
    print(f"[pairs] capitulation (extract split): {pos} flips / "
          f"{len(cap_pairs) - pos} held-firm / {n_abandon} abandons excluded")


if __name__ == "__main__":
    main()
