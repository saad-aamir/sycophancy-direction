"""Batched generation via vLLM — the fast path for screening and
baseline runs. TransformerLens (common.generate) remains the only path
for hooked runs (ablation/steering) and for the equivalence check.

Usage:
    gen = VLLMGenerator(cfg)              # loads the model onto GPU
    outs = gen.generate_batch(prompts)    # list[str], order-preserving
    gen.close()                           # frees GPU before next model
"""
from __future__ import annotations

from .common import STOP_STRINGS


class VLLMGenerator:
    def __init__(self, cfg: dict):
        from vllm import LLM, SamplingParams  # lazy: not needed on Mac
        self.params = SamplingParams(
            temperature=cfg["eval"]["temperature"],
            max_tokens=cfg["eval"]["max_new_tokens"],
            stop=list(STOP_STRINGS),
        )
        self.llm = LLM(
            model=cfg["model"]["name"],
            dtype=str(cfg["model"]["dtype"]),
            gpu_memory_utilization=0.90,
            max_model_len=4096,   # Llama declares 131072; its fp32 KV cache
                                  # won't fit 48GB. Prompts are <1k tokens and
                                  # max_new_tokens=150, so this cannot alter output.
            enforce_eager=False,
        )
        print(f"[vllm] loaded {cfg['model']['name']} "
              f"(dtype={cfg['model']['dtype']})")

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Order-preserving batched greedy generation."""
        outs = self.llm.generate(prompts, self.params)
        # vLLM returns in input order for list inputs; be defensive anyway
        by_prompt = {o.prompt: o for o in outs}
        results = []
        for i, p in enumerate(prompts):
            o = by_prompt.get(p, outs[i])
            results.append(o.outputs[0].text.strip())
        return results

    def close(self):
        import gc
        import torch
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
