# V2 STARTER KIT — README

What this is: everything needed to start the pre-registered 8-model
sweep. Generation is batched (vLLM); all v1 analysis scripts keep
working unchanged via the active-model overlay.

## Integration into the repo

REPLACES v1 files (full-file policy):
  src/common.py                 (adds overlay + paraphrase lookup)
  src/build_contrast_pairs.py   (paraphrase-aware, carries template_text)

NEW files:
  src/generation.py             src/run_behavioral_eval_v2.py
  src/cache_activations_v2.py   src/check_equivalence.py
  src/check_numerics.py         src/run_sweep.py
  configs/config_v2.yaml        configs/templates_v2.yaml

UNCHANGED and still used (overlay makes them multi-model):
  src/rejudge.py  src/analyze.py  src/extract_directions.py
  src/ablation.py  src/steer.py  src/make_questions.py

## Order of operations

1. EDIT configs/templates_v2.yaml with Awais (rules in its header).
2. COMMIT the freeze: PREREGISTRATION_V2.md + templates_v2.yaml + this
   kit, one commit. run_sweep refuses to start without that commit.
   (Optional, recommended for the single-paper path: embargoed OSF
   registration of the same two documents.)
3. Build the pool (on the Mac, cheap):
     python src/make_questions.py --n 300 --pool 3000 --seed 2 \
         --out data/pool_seed2.jsonl
     python - <<'EOF'
     import json
     seen, out = set(), []
     for f in ("data/questions_pool2.jsonl", "data/pool_seed2.jsonl"):
         for r in map(json.loads, open(f)):
             if r["id"] not in seen:
                 seen.add(r["id"]); out.append(r)
     open("data/pool.jsonl","w").write(
         "\n".join(json.dumps(r, ensure_ascii=False) for r in out))
     print(len(out), "questions -> data/pool.jsonl")
     EOF
   (PopQA replication pool comes later: same script, --source popqa;
   run it as a second sweep with '-popqa'-suffixed slugs in a config
   copy so outputs do not collide.)
4. Rent the GPU (A100-80GB class; RunPod/Lambda/Vast). On the box:
     git clone <repo> && cd sycophancy-direction
     uv venv && source .venv/bin/activate
     uv pip install vllm transformer-lens torch transformers anthropic \
         pydantic pyyaml numpy scikit-learn matplotlib tqdm
     export HF_TOKEN=...           # Llama gating
     export ANTHROPIC_API_KEY=...  # judge runs on the box
5. Acceptance tests (first money spent here, ~15 min):
     printf 'slug: qwen0.5b\nname: Qwen/Qwen2.5-0.5B-Instruct\ndtype: bfloat16\n' \
         > configs/active_model.yaml
     python -m src.check_equivalence --pool data/pool.jsonl
     printf 'slug: qwen1.5b\nname: Qwen/Qwen2.5-1.5B-Instruct\ndtype: bfloat16\n' \
         > configs/active_model.yaml
     python -m src.check_numerics
   Both must print PASS (thresholds are in the pre-registration).
6. The sweep:
     python -m src.run_sweep --pool data/pool.jsonl
   Resumable: rerun after any crash, completed stages are skipped.
   Prints the per-model gate table at the end. Sync outputs/ off the
   box (rsync or git-lfs for the .pt files) before shutting it down.

## Cost and time expectations

Sweep ≈ 8–14 GPU-hours ≈ $15–30 at ~$2/hr. Pre-registered hard stop:
$150 total, scope freezes at whatever is complete. Small models are
minutes; qwen14b dominates (~1.5–2 h incl. caching).

## Deliberately NOT in this kit (next artifacts, no GPU needed)

- Multi-locus extraction (H2 grid over attn_z / mlp_out / positions):
  the caches capture everything already; extraction runs later on any
  machine from the saved tensors.
- H4 annotation sampler + labeling sheet for you and Awais.
- H1 emergence analysis (Spearman over the gate AUROCs) — trivial once
  the sweep table exists.

## Known sharp edges

- vLLM pins its own torch; install order above lets it win. If
  transformer-lens complains, put it in a second venv for the cache
  stage — the sweep stages are subprocesses, so two venvs work by
  editing run_sweep's interpreter per stage (last resort).
- rejudge's "done" marker is the .heuristic backup; to force a re-judge
  of a model, delete outputs/{slug}/transcripts/baseline.heuristic.jsonl.
- build_contrast_pairs hard-fails on paraphrase mismatch: that error
  means templates_v2.yaml changed after a run. It is doing its job.
