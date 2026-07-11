"""Step 3 — Cache last-prompt-token residual activations for every pair.

Saves, per contrast:
  outputs/activations/{pair_type}_acts.pt   dict with
    acts:   [n_prompts, n_layers, d_model]  (float32, CPU)
    labels: [n_prompts]  (1=pos, 0=neg)
    ptypes: list[str]    pushback type per prompt (for per-type directions)

    python -m src.cache_activations
"""
from __future__ import annotations

import torch
from tqdm import tqdm

from .common import ROOT, last_token_resid, load_config, load_model, read_jsonl


def cache(pair_type: str, model, cfg):
    in_path = ROOT / cfg["paths"]["activations_dir"] / f"pairs_{pair_type}.jsonl"
    if not in_path.exists():
        print(f"[cache] {in_path} missing, skipping")
        return
    pairs = read_jsonl(in_path)
    if not pairs:
        print(f"[cache] {pair_type}: no pairs, skipping")
        return

    acts, labels, ptypes = [], [], []
    for p in tqdm(pairs, desc=f"cache:{pair_type}"):
        acts.append(last_token_resid(model, p["prompt"]))
        labels.append(1 if p["label"] == "pos" else 0)
        ptypes.append(p["pushback_type"])

    out = ROOT / cfg["paths"]["activations_dir"] / f"{pair_type}_acts.pt"
    torch.save({"acts": torch.stack(acts),
                "labels": torch.tensor(labels),
                "ptypes": ptypes}, out)
    print(f"[cache] saved {out}  shape={torch.stack(acts).shape}")


def main():
    cfg = load_config()
    model = load_model(cfg)
    for pair_type in ("capitulation", "pushback"):
        cache(pair_type, model, cfg)


if __name__ == "__main__":
    main()
