# Is Sycophancy Causally Manipulable? Direction Extraction, Ablation, and Behavioral Validation

Companion mech-interp study to the [sycophancy-evals](https://github.com/saad-aamir/sycophancy-evals) behavioral harness.

**Research question:** The field disagrees about whether sycophancy is a causally
steerable linear direction (probes find it but steer poorly; persona-vector work
argues it isn't a single direction). We adjudicate by (1) extracting candidate
directions via difference-in-means, (2) validating causally via directional
ablation against a *conditioned behavioral eval with clustered CIs*, (3)
decomposing by **pushback type**, and (4) testing **cross-lingual transfer
(English → Urdu)**.

## Pipeline order (important!)

Capitulation labels are **model-specific** — a conversation where Qwen-7B
capitulated tells you nothing about Qwen-1.5B. So the pilot order is:

```
1. run_behavioral_eval.py --condition baseline   # get THIS model's capitulation labels
2. build_contrast_pairs.py                        # pairs from step-1 transcripts
3. cache_activations.py                           # residual stream acts at last prompt token
4. extract_directions.py                          # diff-in-means per layer + AUROC plots
5. run_behavioral_eval.py --condition ablated     # rerun eval with direction projected out
6. steer.py                                       # optional: induce capitulation w/o pushback
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Model: `Qwen/Qwen2.5-1.5B-Instruct` (pilot; same family as the behavioral
harness). Runs on a free Colab T4 comfortably in bf16. CPU works for activation
caching but generation will be slow — use Colab/GPU for steps 1 and 5.
Port to 7B on a rented GPU after the pipeline is validated.

## Weekend runbook

**Friday night**
- [ ] Skim: arXiv 2607.07003 (Dissociating sycophancy representations),
      arXiv 2605.21006 (persona vectors vs targeted steering), Rimsky et al.
      CAA. Goal: positioning, not depth.
- [ ] `pip install -r requirements.txt`, verify model loads:
      `python -c "from src.common import load_model; load_model()"`
- [ ] Paste your real question set into `data/questions.jsonl` (sample file
      shows the schema) and your harness's pushback templates into
      `configs/config.yaml`.

**Saturday**
- [ ] AM: Step 1 (baseline eval, 50 questions × pushback types). Sanity-check
      transcripts and the heuristic judge's labels; spot-fix with your real judge.
- [ ] PM: Steps 2–4. Look at `outputs/figures/layer_auroc.png` — you want
      AUROC well above 0.5 at mid layers for the capitulation pairs.
- [ ] Evening: write paper §3 Method while it's fresh (skeleton in `paper/`).

**Sunday**
- [ ] AM: Step 5. Headline pilot number = destabilization rate baseline vs
      ablated, conditioned on initially-correct, on this 50-question slice.
- [ ] PM: Step 6 if time. Fill pilot numbers into paper tables (clearly
      marked PILOT).
- [ ] Evening: assemble draft — abstract, intro, related work, method,
      pilot results, limitations.

## Caveats (read once)

- Code is untested scaffolding written offline — expect minor API friction
  with TransformerLens versions; the logic is the contract.
- The heuristic judge (substring match) is a pilot crutch. Swap in your
  Claude-based judge from sycophancy-evals before any number goes in the paper
  as non-pilot.
- 1.5B will get fewer questions initially correct than 7B — the conditioned
  sample shrinks. Fine for a pilot; flag it in Limitations.
- `prepend_bos=False` everywhere: Qwen chat template supplies its own special
  tokens.
