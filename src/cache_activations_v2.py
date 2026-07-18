"""v2 caching — multi-locus, batch-1 (single forward passes are cheap
on GPU; batching mattered for generation, not here).

Per pair file, per episode prompt, captures at ALL layers:
  resid_pre  [L, d_model]        x 2 positions
  mlp_out    [L, d_model]        x 2 positions
  attn_z     [L, n_heads, d_head] x 2 positions
Positions: `last` (final prompt token) and `user_msg_mean` (mean over
the tokens of the final user message, located via the tokenizer's
offset mapping against the message text's character span).

Outputs (fp16 on disk), per pair_type:
  {pair_type}_v2_{component}_{position}.pt   with acts/labels/ptypes/
                                             qids/paraphrase_idx
Also writes v1-COMPATIBLE {pair_type}_acts.pt (resid_pre @ last, fp32)
so the unmodified v1 extract_directions.py produces the primary-locus
gate during the sweep.

    python -m src.cache_activations_v2
"""
from __future__ import annotations

import torch
from tqdm import tqdm

from .common import ROOT, load_config, load_model, read_jsonl


def find_span_tokens(tokenizer, prompt: str, needle: str) -> tuple[int, int] | None:
    """Token index span [a, b) covering the last occurrence of `needle`
    in `prompt`, via fast-tokenizer offset mapping."""
    start = prompt.rfind(needle)
    if start < 0:
        return None
    end = start + len(needle)
    enc = tokenizer(prompt, add_special_tokens=False,
                    return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    idx = [i for i, (a, b) in enumerate(offs) if a < end and b > start]
    if not idx:
        return None
    return idx[0], idx[-1] + 1


@torch.no_grad()
def capture(model, tokenizer, prompt: str, needle: str):
    """Returns dict[(component, position)] -> tensor, plus token count."""
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    toks = enc["input_ids"].to(model.cfg.device)
    span = find_span_tokens(tokenizer, prompt, needle)

    def names(n):
        return (n.endswith("hook_resid_pre") or n.endswith("hook_mlp_out")
                or n.endswith("attn.hook_z"))

    _, cache = model.run_with_cache(toks, names_filter=names)
    L = model.cfg.n_layers
    out = {}
    for comp, key in (("resid_pre", "blocks.{l}.hook_resid_pre"),
                      ("mlp_out", "blocks.{l}.hook_mlp_out"),
                      ("attn_z", "blocks.{l}.attn.hook_z")):
        stacked = torch.stack([cache[key.format(l=l)][0]
                               for l in range(L)])          # [L, pos, ...]
        out[(comp, "last")] = stacked[:, -1].half().cpu()
        if span is not None:
            a, b = span
            out[(comp, "user_msg_mean")] = stacked[:, a:b].mean(1).half().cpu()
        else:
            out[(comp, "user_msg_mean")] = stacked[:, -1].half().cpu()
    return out, span is not None


def cache(pair_type: str, model, tokenizer, cfg):
    act_dir = ROOT / cfg["paths"]["activations_dir"]
    in_path = act_dir / f"pairs_{pair_type}.jsonl"
    if not in_path.exists():
        print(f"[cache] {in_path} missing, skipping")
        return
    pairs = read_jsonl(in_path)
    if not pairs:
        return

    store = {}
    labels, ptypes, qids, pidx = [], [], [], []
    n_span_miss = 0
    for p in tqdm(pairs, desc=f"cache:{pair_type}"):
        # needle = the final user message text: pushback template for p2
        # prompts, the question itself for turn-1 (neg pushback) prompts
        needle = p.get("template_text") or p.get("question") or ""
        acts, span_ok = capture(model, tokenizer, p["prompt"], needle)
        n_span_miss += int(not span_ok)
        for k, v in acts.items():
            store.setdefault(k, []).append(v)
        labels.append(1 if p["label"] == "pos" else 0)
        ptypes.append(p["pushback_type"])
        qids.append(p["qid"])
        pidx.append(p.get("paraphrase_idx"))

    labels_t = torch.tensor(labels)
    for (comp, pos), tensors in store.items():
        out = act_dir / f"{pair_type}_v2_{comp}_{pos}.pt"
        torch.save({"acts": torch.stack(tensors), "labels": labels_t,
                    "ptypes": ptypes, "qids": qids,
                    "paraphrase_idx": pidx}, out)
    # v1-compatible primary-locus file
    torch.save({"acts": torch.stack(store[("resid_pre", "last")]).float(),
                "labels": labels_t, "ptypes": ptypes, "qids": qids},
               act_dir / f"{pair_type}_acts.pt")
    print(f"[cache:{pair_type}] {len(pairs)} eps, "
          f"{len(store)} locus files + v1-compat; span misses "
          f"{n_span_miss} (fell back to last token)")


def main():
    cfg = load_config()
    model = load_model(cfg)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    for pair_type in ("capitulation", "pushback"):
        cache(pair_type, model, tokenizer, cfg)


if __name__ == "__main__":
    main()
