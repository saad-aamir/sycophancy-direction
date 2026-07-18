# RESEARCH LOG — Sycophancy Direction Study

Lab notebook for "No Usable Linear 'Capitulation Direction' in Two Small
LLMs." Self-contained: paste this file to any model (or read it cold in
three months) and the full project state should be reconstructible.

Repo: `sycophancy-direction` · Companion doc: PLAIN_LANGUAGE_GUIDE.md
Started: Fri 2026-07-10 · Log last updated: Mon 2026-07-13 (complete
reconstruction; supersedes all prior versions)

---

## 1. Project one-pager (final state)

**Question.** Is sycophancy (capitulation under user pushback) a causally
manipulable linear direction in the residual stream?

**Answer found.** In two small models from different families
(Qwen2.5-1.5B, Llama-3.2-1B): **no** — no linear capitulation direction
of usable strength is decodable from the pre-response residual stream
(best matched LOQO AUROC 0.582 and 0.548 vs a pre-registered 0.70 bar;
Llama's is below its own permutation null), while the identical pipeline
recovers a known control direction at AUROC 1.000 in both. Causal
ablation was pre-registered as conditional on passing this gate and was
therefore not run.

**Behavioral headline (upgraded during the study).** Pushback
susceptibility is model-specific to the point of sign reversal: the same
paired comparison (bare doubt vs emotional appeal) is
Bonferroni-significant in opposite directions across families. Failure
mode is also model-dependent (abandonment 8.2% Llama vs 1.4% Qwen).
Pushback is net epistemically destructive in both (~3x more damage to
correct answers than repair of wrong ones). Substring grading
underestimates capitulation by 18–24pp.

**Deliverable.** arXiv preprint (v2 draft complete, paper/main.tex) +
open repo. Purpose: MATS / Anthropic Fellows portfolio.

---

## 2. Pipeline map (final)

```
0. make_questions.py        TriviaQA rc.nocontext pool (streamed, aliases)
1. screen_questions.py      turn-1 only over pool -> eligible + 30 wrong
2. run_behavioral_eval.py --condition baseline   (greedy, multi-turn)
3. rejudge.py               Claude judge, final-committed-answer rubric
4. build_contrast_pairs.py  flip vs held-firm (verdict-based, D12),
                            extract split only (D6)
5. cache_activations.py     resid_pre, last prompt token, all layers,
                            stores qids
6. extract_directions.py    v3: matched within-question diff-in-means,
                            LOQO CV, within-q shuffle null, GATE (D14/D15)
7. [if GATE PASS] run_behavioral_eval --condition ablated -> rejudge
8. analyze.py               flip/abandon/recovery, clustered bootstrap
                            CIs, heldout vs extract splits
9. steer.py                 (built, unused — gate never passed)
```

Per-model outputs live in `outputs_qwen1.5b/` and `outputs_llama1b/`
(rename `outputs/` -> `outputs_llama1b/` if not yet done). Judging is
deliberately decoupled from generation: re-grading ~1,000 responses
costs minutes and cents; regeneration costs hours.

---

## 3. Decision log

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Pilot model Qwen2.5-1.5B-Instruct | Fits laptop; weekend goal was pipeline correctness. |
| D2 | Baseline eval before direction extraction | Capitulation labels are model-specific. |
| D3 | Questions = TriviaQA rc.nocontext, streamed, with aliases | Original harness questions lost; public benchmark beats bespoke (reproducible, Sharma-et-al. precedent); aliases sharpen substring judging; streaming spares bandwidth. |
| D4 | Two-stage judging: substring+aliases inline, Claude re-judge pass | Generation expensive, judging cheap/re-runnable; substring fails on self-contradictions. |
| D5 | Judge rubric = final committed answer; RETRACTED counts as not-correct | Robust to apologies/contradictions; abandoning a correct answer is abandonment. |
| D6 | Extract/heldout split by md5(qid) parity | Directions from one half; causal/headline claims on the other. In-sample ablation is circular. |
| D7 | Keep ~30 initially-wrong questions | Free recovery control: distinguishes "removed sycophancy" from "installed stubbornness." |
| D8 | Turn-1 screening for scale-up | Eligibility costs 1 generation; full protocol costs 5. |
| D9 | Everywhere-ablation (all resid writes) | Additive stream: later layers re-write an ablated direction. (Built; unused — gate.) |
| D10 | Greedy decoding | Reproducibility; screening answers identical to run answers. |
| D11 | ~15 positives floor per class (superseded by matched-cluster power targets) | Below it, class means are noise. |
| D12 | Outcome taxonomy: flip / abandon / hold; direction positives = INCORRECT finals only; abandons excluded from pairs, reported separately | Abandonment is behaviorally distinct; conflated metric misled (see S3, S5). |
| D13 | Full-file replacement for multi-line code changes | Two surgical-patch failures (rejudge hist deletion; earlier merge ambiguity). |
| D14 | Matched within-question pairs + leave-one-question-out CV + within-question shuffle null | Episodes are siblings (leakage); topic identity confounds unmatched contrast; null must respect cluster structure. |
| D15 | Objective gate: matched LOQO AUROC >= 0.70 AND > shuffle mean + 2sd; branches pre-registered (PASS -> ablation; FAIL -> null paper) | No post-hoc layer shopping; both branches publishable. |
| D16 | Precision: fp32 end-to-end, both models; single-run design on questions_all | MPS device verified exactly correct at fp32 (cosine 1.000000); the 0.9757 scare was bf16 rounding. One run, one rejudge, no config swapping. |
| D17 | Model 2 = Llama-3.2-1B-Instruct fp32 (after Qwen2.5-3B failed twice on 16GB RAM) | Cross-family replication beats same-family scale-up for a null; 3B fp32 OOM (needs ~25GB at load), 3B bf16 froze the machine at the swap line. |

---

## 4. Chronological notebook

### Session 1 — Fri 2026-07-10, evening
- Prior-work sweep: arXiv 2607.07003 (dissociating sycophancy reps; indexes
  Wang 2025, Genadi 2026, Vennemeyer 2025), arXiv 2605.21006 (persona
  vectors rival targeted steering). Multilingual sycophancy directions:
  nothing found (open gap; deferred to v2). Paper repositioned from "does
  a direction exist" (crowded) to "adjudicate causal manipulability with
  rigorous validation" (live disagreement).
- Repo scaffolded (eval, pairs, cache, extract, ablate, steer, config,
  paper skeleton). Pipeline order fixed: baseline first (D2).

### Session 2 — Fri night -> Sat ~02:00 (toy smoke test, 10 questions)
- Full chain executed. Baseline (heuristic): 7/40 capitulations (auth 3,
  simple 2, emo 1, social 1); initial accuracy 10/10.
- Manual judge audit: 1 clear substring mislabel (q003-social "Olaf
  Stapledon" case), 1 ambiguous. Observation kept for paper: held-firm
  responses ALSO apologize -> apology register cannot be the direction.
- BUG+FIX: NaN at layer 0 in extraction. Cause: identical final template
  token + rotary encoding -> identical L0 resid -> 0/0 on normalize.
  Fix: clamp_min(1e-8); L0 correctly reads AUROC 0.5.
- Ablated smoke (IN-SAMPLE, meaningless by design): 7 -> 4; auth 3 -> 0
  (auth was 3/7 of positives); initial accuracy held 10/10.
- Protocol upgrades adopted: extract/heldout split (D6), recovery control
  (D7); analyze v1 (clustered bootstrap) written.
- Question set lost -> rebuilt on TriviaQA with aliases (D3); heuristic
  judge patched for aliases. Overnight batch-1 launched (n_questions
  stayed at 50; the intended 150 was never set — caught next day).

### Session 3 — Sat 2026-07-11 (batch 1 real data; the judge saga; v1->v3)
- Batch-1 baseline (heuristic): 20/50 initially correct; destabilization
  auth 6/20, emo 5/20, simple 8/20, social 8/20 (27/80 = 33.75%).
- REJUDGE BUG #1 (Claude's): verdict parser matched "CORRECT" inside
  "INCORRECT" -> 0 INCORRECT, 100% initial accuracy. Caught by
  impossible-distribution check. Also a flip-count denominator bug.
  BUG #2: ambiguous surgical patch dropped the `hist` lines -> NameError
  AFTER file write (labels junk). -> D13 full-file replacement policy.
  Lesson recorded: always eyeball a judge's verdict distribution for
  impossible values before trusting it.
- Clean batch-1 rejudge: initial 24/50 (48.0%); verdicts C60/I129/R8/U3;
  flips 4/50 questions, 17/200 episodes. Destabilization 44/96 (45.8%)
  vs heuristic 33.75% -> substring underestimate ~12pp (batch-1 value).
- Edge-verdict audit: 10/11 RETRACTED/UNCLEAR were social-type "I can't
  verify" deflections -> "double-check" template-confound hypothesis ->
  D12 taxonomy (positives = INCORRECT only; abandons excluded).
- Batch-1 cross-tab (initially-correct, n=24/type): flips auth 41.7 /
  emo 45.8 / simple 45.8 / social 29.2; abandons social 5/24 (20.8%),
  all others 0 (exact p<0.001). Registered hypothesis "social->abandon,
  authority->flip": abandonment social-concentration CONFIRMED (then —
  see S4 — failed to replicate); authority-dominance of flips REFUTED
  (flat).
- Delivery gap discovered: analyze.py had never been shipped (cause of
  ModuleNotFoundError); analyze upgraded to flip/abandon/recovery;
  build_contrast_pairs rewritten verdict-based.
- Direction v1 (naive): capitulation in-sample 0.814 @ L2 (26/26 eps) —
  flagged (too-early layer; topic confound: pos/neg = different
  questions; template-suffix locality; in-sample inflation at n=52,
  d=1536). Pushback direction 1.000 @ L1 reframed as POSITIVE CONTROL.
- Direction v2 (LOO CV + episode-shuffle band): in-sample 0.7–0.8 at ALL
  layers vs LOO CV at/below chance (best L27 0.531 vs threshold 0.671).
  GATE FAIL. Sub-0.5 dips = small-n LOO anti-bias, not negative signal.
  Control PASS (CV 1.000). The 0.814 was pure overfitting.
- Sibling-leakage insight: 52 episodes = 13 questions; episode-LOO leaks
  siblings and STILL found nothing. -> v3 (D14): matched within-question
  pairs, LOQO CV, within-question shuffle. cache_activations stores qids.
- v3 on batch 1: 7 matched clusters, matched LOQO 0.513, FAIL (only
  honest verdict at that N); per-type correctly suppressed (<3 matched
  q/type). Control: 25 matched, 1.000, PASS. Matched yield 54% (not the
  ~85% estimated: abandon exclusions + partial coverage).

### Session 4 — Sat night -> Sun 2026-07-12 (numerics; batch 2; the gate; v0 draft)
- MPS numerics: CPU-fp32 vs MPS-bf16 min per-layer cosine 0.9757 (gray
  zone; conflated device & dtype). Isolation test fp32-vs-fp32: min
  cosine 1.000000 -> DEVICE FINE; gap was bf16 rounding. -> D16 fp32
  policy + single-run design (questions_all for baseline AND ablated;
  supersedes merge/config-swap plan). n_pairs_max raised to 600. bf16
  batch-1 archived (baseline_b1_bf16.jsonl); its 24 eligible questions
  carried forward and regenerated fresh in fp32.
- BATCH 2 (Qwen2.5-1.5B, fp32): pool 300 (seed 1) -> 298 screened
  (50 batch-1 qids excluded), initial accuracy 139/298 (46.6%).
  questions_all = 193 (139 + 30 control-wrong + 24 carried). Baseline
  4h44m; heuristic 158/193 initially correct; heuristic per-type
  auth 29.7 / emo 22.2 / simple 31.6 / social 17.1.
- Rejudge: initial 159/193 (82.4% — enriched set, expected ~82%);
  verdicts C378/I383/R4/U7; flips 13/193 questions, 134/772 episodes.
  Destabilization (incl. abandons) 25.2% -> 43.2%: substring
  underestimate ~18pp at scale.
- ABANDON NON-REPLICATION: batch-2 abandons 9/636 (1.4%), scattered
  (auth 3, simple 4, social 2, emo 0) vs batch-1's perfect social
  concentration. Batch-1 pattern = small-n noise with a seductive
  mechanism story. Exclusion policy retained; "social deflection" claim
  withdrawn; kept as Appendix-B replication-caution vignette. Also
  explains social's flip recovery (29 -> 42%: abandons stopped absorbing
  flips).
- Qwen flip-only rates (n=159/type): simple 79 (49.7) > social 67 (42.1)
  > auth 60 = emo 60 (37.7); ALL 266/636 (41.8%).
- git initialized this session: 0f7688d (pipeline + batch-1 + log),
  fe95dd2 (batch-2 judged).
- Extraction (batch 2): pairs 143 flip / 212 held / 5 abandons excluded;
  90 questions, 48 matched. THE GATE: matched LOQO 0.582 @ L5 (pooled
  0.563, in-sample 0.623); within-q shuffle 0.504 ± 0.035 (thr 0.574).
  FAIL — marginally above permutation threshold, far below 0.70.
  In-sample decay 0.79 -> 0.62 as n 52 -> 355 (overfitting diagnosis
  confirmed by its own shrinkage). Control PASS #2 (75 matched, 1.000,
  null 0.499 ± 0.021; per-type cosines 0.66–0.88, L1 format caveat).
- BRANCH TAKEN (pre-registered): null paper.
- analyze (Qwen): HELDOUT flip ALL 44.6 [35.9, 53.3] (n=276), abandon
  1.4 [0.3, 3.0], recovery 13.9 [2.5, 28.4] (n=72), initial acc 79.3;
  EXTRACT flip 39.7 [32.6, 47.4] (n=360), recovery 10.9 [0, 28.8]
  (n=64), initial acc 84.9. Recovery combined 17/136 = 12.5% ->
  ASYMMETRY ~3.3x (pushback net destructive).
- "40 unjudged" warning resolved: stale toy ablated.jsonl (10q x 4)
  auto-loaded by analyze; zero qid overlap; archived as
  ablated_smoketest_toy.jsonl. TODO: zero-overlap guard in analyze.
- McNEMAR (pre-registered; exact binomial on within-question discordant
  pairs; Bonferroni x6), Qwen: simple > auth 30v11 (OR 2.7, Bonf .026);
  simple > emo 32v13 (OR 2.5, Bonf .040); simple v social 31v19 (.714);
  auth v emo 26v26 (tie); social v auth 27v20; social v emo 25v18 (ns).
  Headline at the time: mildest pushback most destabilizing.
- Commit 6a71245. v0 paper drafted (null-framed, MODEL2 slots).

### Session 5 — Sun night -> Mon 2026-07-13 (model-2 saga; crossover; v2 paper)
- Qwen2.5-3B fp32: OOM-killed at weight load (~12.4GB weights + TL
  loading transient > 16GB RAM). Policy amended to bf16 for 3B; the bf16
  attempt then froze the laptop at the swap line (restart; brief
  lost-folder scare — nothing lost; commits and transcripts intact).
  16GB RAM confirmed -> D17: drop 3B, go cross-family small.
- GitHub private remote push instructed post-scare (verify: git remote -v).
- Llama-3.2-1B-Instruct: gated repo 401 -> license form -> briefly
  pending (0.5B fallback staged) -> ACCESS GRANTED -> `hf auth login`
  (CLI renamed from huggingface-cli) -> stop-token patch in common.py
  generate(): added "<|eot_id|>".
- LLAMA RUN (fp32, 16 layers, d_model 2048): screen 168/300 (56.0% —
  beats Qwen-1.5B's 46.6% despite fewer params); set = 198q (168 + 30).
  Baseline 11h37m wall-clock = lid-sleep artifact (14q in 7h25m, then
  ~82s/q; caffeinate -i does not block lid-close sleep). Heuristic
  per-type: emo 38.7 > simple 29.8 > auth 23.8 > social 19.0.
  PRE-REGISTERED before judging: cross-family crossover hypothesis; key
  cell = emotional-vs-simple in Llama.
- Rejudge: initial 164/198 (82.8%); verdicts C338/I372/R66/U16; flips
  16/198 questions, 170/792 episodes. Destabilization 27.8% -> 51.4%:
  substring underestimate ~24pp (larger where abandonment is common,
  as predicted).
- Llama flip-only (n=164/type): emo 86 (52.4) > social 74 (45.1) >
  simple 62 (37.8) ~ auth 61 (37.2); ALL 283/656 (43.1%). Abandons
  54/656 (8.2%): simple 24 (21 RETRACTED), auth 12, emo 12, social 6.
  -> FAILURE MODE IS MODEL-DEPENDENT (6x vs Qwen's 1.4%); taxonomy
  reinstated as a cross-model finding; a conflated metric (51.4 vs
  43.2) would smear mode into rate.
- LLAMA GATE: pairs 151/162 (+23 abandons excluded), 84 questions, 40
  matched. Matched LOQO 0.548 @ L10 (pooled 0.583, in-sample 0.708);
  shuffle 0.487 ± 0.047 (thr 0.581) -> BELOW the permutation threshold:
  FAIL, indistinguishable from meaningless labels. NULL REPLICATES
  CROSS-FAMILY (Qwen marginal-above; Llama at-null). Overfitting gap
  replicates (0.708 vs 0.548). Control PASS #3 (75 matched, 1.000 @ L2,
  null 0.467 ± 0.042; cosines 0.637–0.844).
- analyze (Llama): HELDOUT flip ALL 41.2 [33.1, 49.1] (n=320), abandon
  9.7 [5.3, 14.5], recovery 11.8 [2.9, 25.0] (n=76), initial 80.8;
  EXTRACT flip 44.9 [37.0, 52.7] (n=336), abandon 6.8, recovery 16.7
  [1.9, 34.6] (n=60). Recovery combined 19/136 = 14.0% -> asymmetry
  ~3.1x. Both behavioral findings replicate cross-family.
- McNEMAR, Llama: emo > simple 32v8 (OR 4.0, p<.001, Bonf .001 —
  largest effect in the dataset); emo > auth 38v13 (OR 2.9, Bonf .004);
  auth v simple 22v23 (tie); auth v social 22v35 (.667); emo v social
  28v16 (.577); simple v social 16v28 (.577).
- **CROSSOVER CONFIRMED**: emotional-vs-simple is Bonferroni-significant
  in OPPOSITE directions across families (Qwen simple>emo OR 2.5 Bonf
  .040; Llama emo>simple OR 4.0 Bonf .001). Behavioral headline
  upgraded: which pressure destabilizes is a property of the MODEL, not
  the pressure. Auth-vs-emo: exact tie in Qwen, OR 2.9 in Llama.
- Commit ba9705c (llama judged). v2 PAPER COMPLETE (two-model, all
  numbers real, no empty result slots).

---

## 5. Current state & TODO (the definitive list)

**DONE:** two-model behavioral study; confirmed susceptibility
crossover; model-dependent failure mode; judge-bias quantification
(18/24pp); recovery asymmetry (~3x, both models); replicated
cross-family null under the gate with 3x passing positive controls;
full pipeline + validation protocol; v2 paper draft with every number
placed; research log + plain-language guide.

### Paper (blocking arXiv)
- [ ] Voice rewrite: abstract + intro first, then full pass (Beginner's
      Mind register; numbers untouchable).
- [ ] Transcribe per-split CI tables for both models into
      Table~splits (all numbers in Session 4/5 entries above).
- [ ] Appendix A: judge validation, parser-bug vignette,
      impossible-distribution lesson, two-stage economics, FULL McNemar
      matrices both models (all 6 comparisons each; in S4/S5).
- [ ] Appendix B: replication-caution vignette (batch-1 social
      concentration p<.001 -> dissolved at 4x data -> abandonment
      relocated to Llama-simple; taxonomy survived and became finding).
- [ ] Appendix C: reproducibility (fp32 policy; cosine checks 0.9757
      bf16 / 1.000000 fp32; greedy; seeds; pool construction; model +
      judge versions; commit hashes 0f7688d/fe95dd2/6a71245/ba9705c/...).
- [ ] Bibliography: verify EVERY "CHECK" entry against arXiv (hard
      requirement before posting); add TriviaQA, TransformerLens, and a
      McNemar-test citation; final 2025–26 behavioral-lit pass for
      Related Work.
- [ ] Figures: mv outputs -> outputs_llama1b; uncomment 4 includes
      (cap + control, per model); regenerate if paths moved.
- [ ] Compile pass: pdflatex/bibtex x2/pdflatex; fix overfull hboxes.
- [ ] Final read-through with Claude on the compiled PDF.

### Repo / infra
- [ ] Replace paper/main.tex with v2; commit + push.
- [ ] VERIFY GitHub remote actually exists and is pushed
      (`git remote -v`; repo stays private until arXiv posting, then
      flip public + tag the release commit).
- [ ] README refresh to final pipeline (screen -> baseline -> rejudge ->
      pairs -> cache -> extract(gate) -> analyze; two-model layout).
- [ ] analyze.py: zero-qid-overlap guard for stale ablated files.

### arXiv / dissemination
- [ ] Create arXiv account TODAY; check cs.CL/cs.LG endorsement
      requirement for first submission; if required, request endorsement
      now (modelDNA co-author is a candidate) — handshake can take days.
- [ ] Choose license + category (cs.CL primary; cs.LG cross-list?).
- [ ] After posting: Alignment Forum writeup.
- [ ] Beginner's Mind post: "your AI probe is grading its own homework"
      (the three-curve figure as the centerpiece).
- [ ] Drop preprint link into MATS / Anthropic Fellows applications;
      check deadlines.

### v2 research ladder (post-preprint, mostly rented GPU)
- [ ] Rung 1 (~$10–30, one day): Qwen2.5-7B + Llama-3.1-8B replication
      on A100; bf16 on CUDA after a one-time cosine check; requires
      batching generation (current loop is batch-1).
- [ ] Rung 2: scale sweep within one family (0.5B -> 14B/32B): does the
      direction EMERGE with scale?
- [ ] Urdu transfer (behavioral gap; direction cosine; cross-lingual
      ablation) — still the biggest untouched differentiator.
- [ ] Negative-control ablation: run the FAILED direction with
      pre-registered no-effect prediction (exercises causal pipeline).
- [ ] steer.py exercise (built, never run).
- [ ] Reworded social template (drop "double-check") — frozen this
      study, test in v2.
- [ ] Base-vs-instruct comparison (tests the provenance account of the
      crossover).
- [ ] MMLU-slice capability control, for any future PASS branch.
- [ ] Llama-3.2 access now granted: 3B on rented GPU is a cheap extra
      scale point.

---

## 6. Glossary

- **Episode**: question -> answer -> one pushback -> answer.
- **Flip / abandon / hold**: final commits to a wrong answer / drops its
  answer without a new one (RETRACTED/UNCLEAR) / recommits (D12).
- **Flip rate**: P(turn-2 commits to wrong | turn-1 correct, type) —
  primary metric. **Recovery rate**: P(turn-2 correct | turn-1 wrong).
- **Conditioning on initially-correct**: separates sycophancy from
  ignorance.
- **Residual stream / activations / last prompt token**: per-token
  vector that layers additively write to; read at the position the reply
  is generated from.
- **Difference-in-means direction; matched variant**: arrow between
  class means; matched = mean over per-question (flip-mean − hold-mean)
  differences, cancelling topic identity.
- **AUROC**: P(random positive scores above random negative); 0.5 chance.
- **In-sample vs LOQO CV**: grading points on an arrow they helped build
  vs holding out whole questions (episodes are siblings; episode-LOO
  leaks).
- **Within-question shuffle null**: label permutation inside each
  question, 20x — the method's noise floor at this n/d/cluster
  structure.
- **Positive control**: pushback-present vs -absent direction; known to
  exist; CV 1.000 in both models.
- **Gate (D15)**: PASS iff matched LOQO >= 0.70 and > shuffle mean+2sd.
- **Extract / heldout split**: md5(qid)-parity halves.
- **Question-clustered bootstrap**: resample questions, not episodes.

---

## 7. Log conventions

Append-only, newest session last. After each milestone: what ran, exact
numbers, what was decided, what broke. This reconstruction (Mon
2026-07-13) resolves all prior FILL slots; future entries append below.


2026-07-18  Templates v2 reviewed and frozen by SA alone; MABA on leave.
2026-07-18  Compute stop tightened $150 -> $40 (prepaid, physically enforced), pre-data.
2026-07-18  Paraphrase assignment checksum over 300-q pool: 9f4f7b6fb6ac8728.
2026-07-18  DEVIATION (pre-data): A2 residual-cosine bar corrected 0.99 -> 0.97.
            Basis: v1 Appendix C measured min bf16-vs-fp32 cosine 0.9757, and fp32
            cannot fit 14B in 48GB. Label-agreement bar (>=95% on 50 questions) unchanged.
2026-07-18  ADDITION: engine-equivalence gate (vLLM vs TL, >=90% identical 50-token
            prefixes) kept as a supplementary gate alongside A1/A2.