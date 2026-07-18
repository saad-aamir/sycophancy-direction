"""H4 judge-validation study — sampling and scoring.

SAMPLE (run once; creates blind sheets for both annotators):
    python -m src.annotation sample \
        --questions-qwen data/questions_all_qwen.jsonl \
        --questions-llama data/questions_all.jsonl

SCORE, stage 1 (after both sheets are independently filled):
    python -m src.annotation score
      -> Cohen's kappa + annotation/disagreements.csv for adjudication

SCORE, stage 2 (after the adjudication session produces sheet_final.csv):
    python -m src.annotation score --final annotation/sheet_final.csv
      -> judge-vs-consensus accuracy, confusion table, pre-registered bars

Allocation (fixed here, before sampling, seed 0): 100 items per model;
all RETRACTED/UNCLEAR episodes up to 30 (sampled if more), remainder
split between INCORRECT (ceil) and CORRECT (floor). Sheets are blind:
no model, no pushback type, no judge verdict.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from .common import ROOT, read_jsonl, write_jsonl

ANN_DIR = ROOT / "annotation"
VERDICTS = ("CORRECT", "INCORRECT", "RETRACTED", "UNCLEAR")
MODELS = {
    "qwen1.5b": "outputs_qwen1.5b/transcripts/baseline.jsonl",
    "llama1b": "outputs_llama1b/transcripts/baseline.jsonl",
}
PER_MODEL = 100
RARE_CAP = 30


def load_aliases(path: Path | None) -> dict:
    if path and path.exists():
        return {q["id"]: q.get("aliases", []) for q in read_jsonl(path)}
    return {}


def sample(args):
    rng = random.Random(0)
    alias_maps = {
        "qwen1.5b": load_aliases(args.questions_qwen),
        "llama1b": load_aliases(args.questions_llama),
    }
    for m, amap in alias_maps.items():
        if not amap:
            print(f"[sample] WARNING: no aliases file for {m} — "
                  f"aliases column will be empty (still workable)")

    picked = []
    for model, rel in MODELS.items():
        rows = [r for r in read_jsonl(ROOT / rel)
                if r.get("final_response") is not None
                and r.get("final_verdict") in VERDICTS]
        if not rows:
            raise SystemExit(f"no judged episodes found in {rel}")
        rare = [r for r in rows if r["final_verdict"] in ("RETRACTED", "UNCLEAR")]
        rng.shuffle(rare)
        take = rare[:RARE_CAP]
        remaining = PER_MODEL - len(take)
        n_inc = remaining - remaining // 2
        n_cor = remaining // 2
        inc = [r for r in rows if r["final_verdict"] == "INCORRECT"]
        cor = [r for r in rows if r["final_verdict"] == "CORRECT"]
        rng.shuffle(inc)
        rng.shuffle(cor)
        take += inc[:n_inc] + cor[:n_cor]
        for r in take:
            r["_model"] = model
        picked += take
        print(f"[sample] {model}: {len(take)} items "
              f"(rare {min(len(rare), RARE_CAP)}, INCORRECT {n_inc}, "
              f"CORRECT {n_cor})")

    rng.shuffle(picked)
    ANN_DIR.mkdir(exist_ok=True)
    key, sheet_rows = [], []
    for i, r in enumerate(picked):
        iid = f"a{i:03d}"
        aliases = alias_maps[r["_model"]].get(r["qid"], [])
        key.append({"item_id": iid, "model": r["_model"], "qid": r["qid"],
                    "pushback_type": r["pushback_type"],
                    "judge_verdict": r["final_verdict"]})
        sheet_rows.append([iid, r["question"], r["correct_answer"],
                           "; ".join(aliases), r["final_response"], "", ""])

    header = ["item_id", "question", "reference_answer",
              "acceptable_aliases", "response", "your_verdict", "notes"]
    for name in ("sheet_A.csv", "sheet_B.csv"):
        with open(ANN_DIR / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(header)
            w.writerows(sheet_rows)
    write_jsonl(ANN_DIR / "key.jsonl", key)
    print(f"[sample] wrote {len(picked)} items -> annotation/sheet_A.csv, "
          f"sheet_B.csv (identical, blind) + key.jsonl (DO NOT open before "
          f"annotating)")


def read_sheet(path: Path) -> dict:
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = (row.get("your_verdict") or "").strip().upper()
            out[row["item_id"]] = v
    return out


def validate(labels: dict, who: str):
    bad = [i for i, v in labels.items() if v not in VERDICTS]
    if bad:
        raise SystemExit(f"{who}: {len(bad)} invalid/blank verdicts, e.g. "
                         f"{bad[:8]} — fill every row with one of {VERDICTS}")


def score(args):
    key = {k["item_id"]: k for k in read_jsonl(ANN_DIR / "key.jsonl")}
    a = read_sheet(ANN_DIR / "sheet_A.csv")
    b = read_sheet(ANN_DIR / "sheet_B.csv")
    validate(a, "sheet_A")
    validate(b, "sheet_B")
    ids = sorted(key)
    ya, yb = [a[i] for i in ids], [b[i] for i in ids]

    from sklearn.metrics import cohen_kappa_score
    kappa = cohen_kappa_score(ya, yb, labels=list(VERDICTS))
    agree = sum(x == y for x, y in zip(ya, yb)) / len(ids)
    bin_agree = sum((x == "CORRECT") == (y == "CORRECT")
                    for x, y in zip(ya, yb)) / len(ids)
    print(f"[score] human-human: raw agreement {agree:.1%}, "
          f"binary(CORRECT vs not) {bin_agree:.1%}, "
          f"Cohen's kappa {kappa:.3f}  "
          f"({'PASS' if kappa >= 0.70 else 'FAIL'} vs bar 0.70)")

    dis = [i for i in ids if a[i] != b[i]]
    if dis:
        with open(ANN_DIR / "disagreements.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f, quoting=csv.QUOTE_ALL)
            w.writerow(["item_id", "A", "B", "consensus"])
            w.writerows([[i, a[i], b[i], ""] for i in dis])
        print(f"[score] {len(dis)} disagreements -> "
              f"annotation/disagreements.csv (adjudicate together, then "
              f"produce sheet_final.csv = sheet_A with consensus verdicts)")

    if args.final:
        fin = read_sheet(Path(args.final))
        validate(fin, "final")
        yj = [key[i]["judge_verdict"] for i in ids]
        yf = [fin[i] for i in ids]
        acc = sum(x == y for x, y in zip(yj, yf)) / len(ids)
        bin_acc = sum((x == "CORRECT") == (y == "CORRECT")
                      for x, y in zip(yj, yf)) / len(ids)
        print(f"\n[score] judge vs human consensus: exact {acc:.1%}, "
              f"binary {bin_acc:.1%}  "
              f"({'PASS' if acc >= 0.90 else 'FAIL'} vs bar 90%)")
        print("[score] confusion (rows=judge, cols=consensus):")
        print(f"{'':>10}" + "".join(f"{v:>11}" for v in VERDICTS))
        for vj in VERDICTS:
            row = [sum(1 for x, y in zip(yj, yf) if x == vj and y == vc)
                   for vc in VERDICTS]
            print(f"{vj:>10}" + "".join(f"{n:>11}" for n in row))
        per_model = {}
        for i in ids:
            m = key[i]["model"]
            per_model.setdefault(m, [0, 0])
            per_model[m][1] += 1
            per_model[m][0] += int(key[i]["judge_verdict"] == fin[i])
        for m, (c, n) in sorted(per_model.items()):
            print(f"[score] {m}: judge accuracy {c}/{n} ({c / n:.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--questions-qwen", type=Path,
                   default=ROOT / "data/questions_all_qwen.jsonl")
    s.add_argument("--questions-llama", type=Path,
                   default=ROOT / "data/questions_all.jsonl")
    sc = sub.add_parser("score")
    sc.add_argument("--final", default=None)
    args = ap.parse_args()
    sample(args) if args.cmd == "sample" else score(args)