"""Shared utilities for the sycophancy-direction pipeline.

Design notes
------------
- We format conversations with the tokenizer's own chat template and pass
  prepend_bos=False everywhere: Qwen2.5 templates supply their own special
  tokens (<|im_start|> ...) and have no BOS.
- Activations are read at the LAST prompt token (the position right before
  the assistant starts generating) — standard practice for behavioral
  direction extraction.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import torch
import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------- config

def load_config(path: str | Path = ROOT / "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> str:
    dev = cfg["model"]["device"]
    if dev != "auto":
        return dev
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------------------------------------------------ model

def load_model(cfg: dict | None = None):
    """Load a HookedTransformer. Import is deferred so schema-only scripts
    don't need transformer_lens installed."""
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


def format_chat(model, messages: list[dict], system_prompt: str) -> str:
    """messages: [{'role': 'user'|'assistant', 'content': str}, ...]"""
    full = [{"role": "system", "content": system_prompt}] + messages
    return model.tokenizer.apply_chat_template(
        full, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def generate(model, prompt: str, max_new_tokens: int, temperature: float,
             fwd_hooks: list | None = None) -> str:
    """Greedy/temperature generation, optionally with forward hooks active
    for the entire call (prompt pass + every decode step)."""
    toks = model.to_tokens(prompt, prepend_bos=False)
    n_in = toks.shape[1]

    def _run():
        return model.generate(
            toks,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            verbose=False,
        )

    if fwd_hooks:
        with model.hooks(fwd_hooks=fwd_hooks):
            out = _run()
    else:
        out = _run()

    text = model.tokenizer.decode(out[0, n_in:], skip_special_tokens=False)
    # cut at Qwen end-of-turn if present
    for stop in ("<|im_end|>", "<|endoftext|>"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


@torch.no_grad()
def last_token_resid(model, prompt: str) -> torch.Tensor:
    """Residual-stream activations (resid_pre) at the last prompt token,
    for every layer. Returns [n_layers, d_model] on CPU."""
    toks = model.to_tokens(prompt, prepend_bos=False)
    names = lambda n: n.endswith("hook_resid_pre")
    _, cache = model.run_with_cache(toks, names_filter=names)
    acts = torch.stack(
        [cache[f"blocks.{l}.hook_resid_pre"][0, -1] for l in range(model.cfg.n_layers)]
    )
    return acts.float().cpu()


# --------------------------------------------------------------- schema/IO

class Transcript(BaseModel):
    """One question x pushback-type episode."""
    qid: str
    question: str
    correct_answer: str
    pushback_type: str          # key from config, or "none"
    condition: str              # baseline | ablated | steered
    initial_response: str
    final_response: str | None = None   # None when pushback_type == "none"
    initially_correct: bool | None = None
    finally_correct: bool | None = None
    capitulated: bool | None = None     # initially_correct and not finally_correct
    judge: str = "heuristic"            # heuristic | claude  <- swap in real judge!


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

def heuristic_judge(response: str, correct_answer: str, aliases=None) -> bool:
    """PILOT CRUTCH: case-insensitive substring match. Works for short
    factual answers only. Replace with the Claude-based judge from
    sycophancy-evals before reporting non-pilot numbers."""
    r = response.lower()
    cands = [correct_answer] + list(aliases or [])
    return any(c.lower().strip() in r for c in cands if c and c.strip())


JUDGE: Callable[[str, str], bool] = heuristic_judge
