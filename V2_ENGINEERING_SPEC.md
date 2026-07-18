# V2 ENGINEERING SPEC — Batched Multi-Model Pipeline

Goal: run the pre-registered 8-model sweep in 10–20 GPU-hours instead of
~80 laptop-hours. Three changes: (1) batched generation via vLLM,
(2) model-namespaced orchestration, (3) multi-locus batched caching.
Everything downstream of transcripts (rejudge, pairs, analyze, McNemar)
is untouched.

Written offline and untested against live vLLM/TransformerLens — the
v1 rule applies: the logic is the contract, expect API friction, run
the acceptance tests (§6 of the pre-registration) before spending.

---

## 0. Environment (GPU box)

RunPod / Lambda / Vast, A100-80GB (or 2×A6000). Ubuntu + CUDA image.

```bash
uv venv && source .venv/bin/activate
uv pip install vllm transformer-lens torch anthropic pydantic pyyaml \
    numpy scikit-learn matplotlib tqdm
export HF_TOKEN=...            # Llama gating
export ANTHROPIC_API_KEY=...   # judge runs on the box
```

## 1. Config changes (`configs/config_v2.yaml`)

```yaml
models:                       # registry replaces single model.name
  - name: Qwen/Qwen2.5-0.5B-Instruct
    slug: qwen0.5b
  - name: Qwen/Qwen2.5-1.5B-Instruct
    slug: qwen1.5b
  # ... 3B, 7B, 14B, llama1b, llama3b, llama8b
dtype: bfloat16
paths:
  outputs_root: outputs/{slug}/   # kills the mv dance permanently
templates_file: configs/templates_v2.yaml   # 4 types x 5 paraphrases
eval:
  eligible_cap: 400
  keep_wrong: 30
loci:
  positions: [last, user_msg_mean]
  components: [resid_pre, attn_z, mlp_out]
```

Stop tokens are per-family; derive from the tokenizer
(`eos_token` + chat-template end marker) instead of a hardcoded list.
Paraphrase assignment: `idx = md5(qid + ':' + ptype) % 5`, frozen.

## 2. `src/generation.py` — the Generator abstraction

```python
class Generator(Protocol):
    def generate_batch(self, prompts: list[str]) -> list[str]: ...

class VLLMGenerator:      # fast path: screening + baseline
    # vllm.LLM(model=name, dtype="bfloat16", gpu_memory_utilization=0.9)
    # SamplingParams(temperature=0.0, max_tokens=cfg.max_new_tokens,
    #                stop=family_stop_tokens)
    # generate_batch = llm.generate(prompts) -> outputs in order

class TLGenerator:        # hooked path: ablated / steered runs only
    # current common.generate, plus batching (§4 padding rules)
```

Rule: vLLM for anything without hooks; TransformerLens only where
activations are read or edited. The two must pass the §6.1 equivalence
test before any vLLM transcript is treated as data.

## 3. Two-phase batched eval (`run_behavioral_eval` refactor)

Turn 2 depends on turn 1, so batch in phases, never per-question:

```
PHASE 1: prompts = [chat(q) for q in questions]        # one batch
         initial = generate_batch(prompts)
         init_ok = [alias_judge(a, q) for ...]          # inline labels
PHASE 2: episodes = [(q, ptype) for q in questions for ptype in types]
         prompts  = [chat(q, initial[q], template(q, ptype)) ...]
         finals   = generate_batch(prompts)             # one batch
WRITE    transcripts (schema unchanged -> rejudge/analyze untouched)
```

Screening is Phase 1 alone over the 600-question pool. Eligible-cap
subsample (seed-fixed) happens between screening and Phase 1 of the
main run.

## 4. Batched multi-locus caching (`cache_activations` v2)

The two tricky parts, spelled out:

**(a) Left-padding for last-token gather.** Batch prompts sorted by
token length into buckets of ≤16; LEFT-pad each bucket to its max
length; pass the attention mask through TL
(`model(tokens, attention_mask=mask, ...)` via run_with_cache). With
left padding, position `-1` is the true last token for every row —
no per-row index gymnastics. For the `user_msg_mean` position, compute
per-row token spans of the final user message BEFORE padding, offset by
the pad amount, and mean-pool with the mask.

**(b) Storage layout per model.** One file per component:

```
resid_pre: [n_eps, n_layers, d_model]            (fp16 on disk)
mlp_out:   [n_eps, n_layers, d_model]            (fp16)
attn_z:    [n_eps, n_layers, n_heads, d_head]    (fp16)
```

Size check (worst case, Qwen-14B, ~2,000 episodes, 2 positions):
attn_z ≈ 2000×48×40×128×2B×2 ≈ 4 GB — fine on disk, stream per-layer
if RAM-tight. Save `qids`, `labels`, `ptypes`, `paraphrase_idx`
alongside, as in v1.

**Names filter:** `hook_resid_pre`, `hook_mlp_out`, `attn.hook_z`.

## 5. Multi-locus extraction (`extract_directions` v2)

Wrap the v1 matched/LOQO/shuffle machinery in a loop over
(component, position). For `attn_z`, a "direction" is per (layer, head)
in d_head space; matched construction and LOQO identical; report the
best head with Bonferroni ×(layers×heads) inside that component's gate;
the cross-locus gate additionally corrects ×n_loci per the pre-reg.
Persist a tidy results table:
`outputs/{slug}/directions/locus_grid.csv`
(component, position, layer[, head], auroc_cv, shuffle_mean, shuffle_sd,
gate_pass) — this table IS Figure/Table material for v2.

Compute note: LOQO over ~40 heads × 48 layers × 8 models is the new
bottleneck — vectorize the per-fold matched direction over heads (one
einsum per fold per layer), and drop shuffle perms to 50 exactly as
pre-registered (they parallelize the same way).

## 6. Orchestrator (`run_sweep.py`)

```
for model in registry:
    screen (vLLM)  -> eligible + 30 wrong -> cap 400
    phase-1/2 eval (vLLM)          -> transcripts
    rejudge (anthropic, threaded)  -> judged transcripts
    build_contrast_pairs           -> pairs (paraphrase_idx carried)
    cache_activations v2 (TL)      -> 3 components x 2 positions
    extract_directions v2          -> locus_grid.csv + figures
    free GPU memory (del llm; torch.cuda.empty_cache()) before next model
checkpointing: skip any stage whose output file already exists
```

Per-model wall-clock estimate (A100): 0.5B–3B ≈ 15–30 min; 7B ≈ 45 min;
14B ≈ 90 min; caching adds ~10–30 min each. Sweep ≈ 8–14 GPU-hours.

## 7. Acceptance tests before the sweep (map to pre-reg §6)

```bash
python -m src.check_equivalence --model qwen0.5b   # vLLM vs TL, 20 prompts
python -m src.check_numerics    --model qwen1.5b   # bf16 vs v1 fp32 direction
```

Both must print PASS (thresholds in the pre-registration) before
`run_sweep.py` is allowed to start; the sweep script asserts the
presence of their PASS marker files.

## 8. Explicit non-goals for v2 infra

No streaming/async judge, no multi-GPU sharding (14B fits one A100 in
bf16), no Urdu (v3), no serving. Anything not in the pre-registration
does not get built this cycle.

## Build order (first GPU session, ~half a day)

1. generation.py + config_v2 + templates_v2 (write paraphrases FIRST,
   they freeze with the pre-reg)      ~2h
2. two-phase eval refactor            ~1h
3. acceptance tests + run on 0.5B     ~1h  <- first money spent here
4. caching v2 (resid_pre only)        ~1h
5. locus extensions (attn_z, mlp_out) ~2h
6. run_sweep.py + kick off overnight
```
