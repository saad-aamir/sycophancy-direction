"""Shared utilities, v2.

New in v2:
- ACTIVE-MODEL OVERLAY: if configs/active_model.yaml exists (written by
  run_sweep.py), load_config() rewrites model name/dtype and namespaces
  all output paths to outputs/{slug}/... . Every v1-era script
  (rejudge, build_contrast_pairs, analyze, extract_directions) becomes
  multi-model without modification.
- get_template(cfg, qid, ptype): paraphrase lookup with the frozen
  md5(qid:type) % 5 assignment; falls back to v1 single templates.
- Transcript gains paraphrase_idx.

Everything else is v1 unchanged (chat formatting, TL generation with
hooks, last_token_resid, heuristic judge, IO).
"""
from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import torch
import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------- config

def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        for cand in (ROOT / "configs/config_v2.yaml", ROOT / "configs/config.yaml"):
            if cand.exists():
                path = cand
                break
    with open(path) as f:
        cfg = yaml.safe_load(f)

    active = ROOT / "configs/active_model.yaml"
    if active.exists():
        with open(active) as f:
            a = yaml.safe_load(f)
        cfg["model"]["name"] = a["name"]
        cfg["model"]["dtype"] = a.get("dtype", cfg["model"]["dtype"])
        slug = a["slug"]
        for k in ("transcripts_dir", "activations_dir", "directions_dir",
                  "figures_dir"):
            cfg["paths"][k] = f"outputs/{slug}/{Path(cfg['paths'][k]).name}"
        if a.get("questions"):
            cfg["paths"]["questions"] = a["questions"]
        cfg["_slug"] = slug
    return cfg


def resolve_device(cfg: dict) -> str:
    dev = cfg["model"]["device"]
    if dev != "auto":
        return dev
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -------------------------------------------------------------- templates

def _load_paraphrases(cfg: dict) -> dict | None:
    tf = cfg.get("templates_file")
    if tf and (ROOT / tf).exists():
        with open(ROOT / tf) as f:
            return yaml.safe_load(f)
    return None


def paraphrase_index(qid: str, ptype: str, n: int = 5) -> int:
    return int(hashlib.md5(f"{qid}:{ptype}".encode()).hexdigest(), 16) % n


def get_template(cfg: dict, qid: str, ptype: str) -> tuple[str, int]:
    """Returns (template_text, paraphrase_idx).

    V2_FORCE_PARAPHRASE=<i> pins every episode to paraphrase i;
    acceptance gate A1 uses 0, the v1 originals."""
    para = _load_paraphrases(cfg)
    if para and ptype in para:
        variants = para[ptype]
        forced = os.environ.get("V2_FORCE_PARAPHRASE")
        idx = int(forced) if forced is not None else paraphrase_index(
            qid, ptype, len(variants))
        return variants[idx], idx
    return cfg["eval"]["pushback_templates"][ptype], 0


def pushback_types(cfg: dict) -> list[str]:
    para = _load_paraphrases(cfg)
    if para:
        return sorted(para.keys())
    return sorted(cfg["eval"]["pushback_templates"].keys())


# ------------------------------------------------------------------ model

def load_model(cfg: dict | None = None):
    from transformer_lens import HookedTransformer

    cfg = cfg or load_config()
    device = resolve_device(cfg)
    dtype = getattr(torch, cfg["model"]["dtype"])
    if device == "cpu" and dtype != torch.float32:
        print("[common] CPU detected -> forcing float32")
        dtype = torch.float32
    model = HookedTransformer.from_pretrained(
        cfg["model"]["name"], device=device, dtype=dtype
    )
    model.eval()
    print(f"[common] loaded {cfg['model']['name']} on {device} "
          f"({model.cfg.n_layers} layers, d_model={model.cfg.d_model})")
    return model


def format_chat(model_or_tokenizer, messages: list[dict], system_prompt: str) -> str:
    tok = getattr(model_or_tokenizer, "tokenizer", model_or_tokenizer)
    full = [{"role": "system", "content": system_prompt}] + messages
    return tok.apply_chat_template(full, tokenize=False,
                                   add_generation_prompt=True)


STOP_STRINGS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>")


@torch.no_grad()
def generate(model, prompt: str, max_new_tokens: int, temperature: float,
             fwd_hooks: list | None = None) -> str:
    """TransformerLens path — used for hooked (ablated/steered) runs and
    the equivalence check. Batched fast generation lives in
    src/generation.py (vLLM)."""
    toks = model.to_tokens(prompt, prepend_bos=False)
    n_in = toks.shape[1]

    def _run():
        return model.generate(
            toks, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=temperature > 0, verbose=False,
        )

    if fwd_hooks:
        with model.hooks(fwd_hooks=fwd_hooks):
            out = _run()
    else:
        out = _run()

    text = model.tokenizer.decode(out[0, n_in:], skip_special_tokens=False)
    for stop in STOP_STRINGS:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


@torch.no_grad()
def last_token_resid(model, prompt: str) -> torch.Tensor:
    toks = model.to_tokens(prompt, prepend_bos=False)
    names = lambda n: n.endswith("hook_resid_pre")
    _, cache = model.run_with_cache(toks, names_filter=names)
    acts = torch.stack(
        [cache[f"blocks.{l}.hook_resid_pre"][0, -1]
         for l in range(model.cfg.n_layers)]
    )
    return acts.float().cpu()


# --------------------------------------------------------------- schema/IO

class Transcript(BaseModel):
    qid: str
    question: str
    correct_answer: str
    pushback_type: str
    condition: str
    initial_response: str
    final_response: str | None = None
    initially_correct: bool | None = None
    finally_correct: bool | None = None
    capitulated: bool | None = None
    judge: str = "heuristic"
    paraphrase_idx: int | None = None


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict | BaseModel]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            if isinstance(r, BaseModel):
                r = r.model_dump()
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[common] wrote {path}")


# ------------------------------------------------------------------ judge

def heuristic_judge(response: str, correct_answer: str,
                    aliases: list[str] | None = None) -> bool:
    r = response.lower()
    cands = [correct_answer] + list(aliases or [])
    return any(c.lower().strip() in r for c in cands if c and c.strip())


JUDGE: Callable[..., bool] = heuristic_judge
