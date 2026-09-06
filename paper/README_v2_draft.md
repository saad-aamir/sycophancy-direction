# v2 paper draft: file inventory and build order

Everything here was written against commit `9325963` of
`github.com/saad-aamir/sycophancy-direction`. Every number regenerates with
`python -m src.analysis_v2`, except the head-scan table, which comes from
`src/extract_heads.py`.

## Section files, in paper order

| File | Section | State |
|---|---|---|
| `related_and_abstract.tex` | Abstract + Related work | Abstract done. Related work needs real citation keys. |
| `introduction.tex` | 1. Introduction | Done |
| `methods.tex` | 2. Methods | One placeholder: judge validation numbers |
| `results.tex` | 3. Results (H1, H2, H6, H3, H5) | Done |
| `hazards.tex` | 4. Measurement hazards | Done |
| `discussion.tex` | 5. Discussion, Limitations, Future work | One placeholder: judge validation |

Suggested `main.tex` body:

```latex
\input{sections/abstract}      % split from related_and_abstract.tex
\input{sections/introduction}
\input{sections/related}       % split from related_and_abstract.tex
\input{sections/methods}
\input{sections/results}
\input{sections/hazards}
\input{sections/discussion}
\bibliography{refs}
```

`related_and_abstract.tex` holds two separate pieces; split them when you wire up
`main.tex`.

## Figures

| Script | Figure | Referenced in |
|---|---|---|
| `fig1_emergence.py` (already in repo at `paper/figures/`) | Emergence curve, 8 models | Results 3.1 |
| `fig2_recovery_ratio.py` | Flip-to-recovery ratio vs scale | Results 3.5 |
| `fig3_ablation_forest.py` | Ablation effects with CIs | Results 3.3 |
| `fig4_crossover_or.py` | Crossover odds ratios | Results 3.4 |

All four are standalone: `python fig2_recovery_ratio.py` writes a PDF and a PNG next to
the script. Numbers are hard-coded from `analysis_v2` output rather than recomputed, so if
any result changes, update the `DATA` block at the top of the relevant script.

Add `\usepackage{booktabs}` to the preamble for the results tables.

## Still to do

1. **Judge validation (H4).** Two placeholders, in `methods.tex` section 2.4 and
   `discussion.tex` limitations. The annotation kit already exists in the repo at
   `annotation/`: `key.jsonl` holds 200 sampled episodes with judge verdicts withheld,
   and `sheet_A.csv` / `sheet_B.csv` are blank. Filling one sheet gives judge-vs-human
   agreement; the second gives inter-annotator agreement. This is the only item that
   blocks submission rather than drafting.

2. **Citation keys.** `related_and_abstract.tex` uses placeholders: `sharma2023`,
   `perez2022`, `wei2023`, `arditi2024`, `zou2023`, `turner2023`, `li2023`,
   `hewitt2019`, `belinkov2022`, `wei2022`, `schaeffer2023`, `joshi2017triviaqa`. Reuse
   v1's `.bib` where it already defines them. `schaeffer2023` is the one to engage with
   properly, since it argues apparent emergence can be a metric artifact, and the
   paragraph as written answers that objection directly.

3. **Deviation log.** `RESEARCH_LOG.md` contains one marked DEVIATION entry (the A2
   cosine bar correction). The paper's deviations list has six. Add the other five with
   their real dates, so the repository is at least as complete as the paper claiming to
   summarise it.

4. **Reconstruct three question files.** `data/questions_qwen0.5b.jsonl`,
   `questions_qwen3b.jsonl` and `questions_llama1b.jsonl` are missing from the repo.
   Rebuild them from the baseline transcripts, not from `screen.jsonl`, which is stale
   for at least Qwen-7B (351 rows marked correct against 381 in the committed question
   file).

5. **Optional: the Qwen-14B ablation.** It passed the gate, which under the
   pre-registration makes its ablation mandatory. Currently disclosed as a compute
   deviation in Methods and flagged in Results at the point a reader would notice. If it
   ever runs, it becomes a fourth point on the gate-margin pattern, and the prediction on
   record is a null.

## Two framing choices worth revisiting

The Llama-8B ablation null sits in the main results table rather than a footnote, and the
text states that its interval excludes an effect the size of Qwen-7B's. That turns a
non-replication into a finding and carries the gate-margin argument. A more conservative
paper would relegate it.

The Qwen-14B plateau is reported with an explicit alternative explanation: it is the most
robust model, flips least often, and therefore yields the fewest matched questions, so
lower AUROC may be power loss rather than saturation. Stating that competing explanation
costs a striking claim and buys credibility.
