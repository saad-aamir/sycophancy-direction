"""Score the judge against human annotation (H4).

    python -m src.score_annotation

Reads annotation/key.jsonl for the judge's verdicts and whichever sheets have been
filled in, then prints exactly the numbers the paper's grading subsection needs:
agreement, Cohen's kappa, the confusion matrix, and where disagreements sit.

"""
from __future__ import annotations

import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["CORRECT", "INCORRECT", "RETRACTED", "UNCLEAR"]
SHEETS = {"A": "annotation/sheet_A.csv", "B": "annotation/sheet_B.csv"}


def kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two label sequences of equal length."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in set(a) | set(b))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def load_sheet(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {r["item_id"]: r["your_verdict"].strip().upper()
            for r in rows if (r.get("your_verdict") or "").strip()}


def main() -> None:
    key = {}
    for line in open(ROOT / "annotation/key.jsonl"):
        r = json.loads(line)
        key[r["item_id"]] = r

    sheets = {name: load_sheet(ROOT / p) for name, p in SHEETS.items()}
    sheets = {k: v for k, v in sheets.items() if v}
    if not sheets:
        print("No annotations found. Run `python -m src.annotate` first.")
        return

    for name, ann in sheets.items():
        items = [i for i in ann if i in key]
        human = [ann[i] for i in items]
        judge = [key[i]["judge_verdict"].upper() for i in items]
        agree = sum(h == j for h, j in zip(human, judge))
        print(f"\n=== Annotator {name}: {len(items)} items ===")
        print(f"  raw agreement  {agree}/{len(items)} = {100*agree/len(items):.1f}%")
        print(f"  Cohen's kappa  {kappa(human, judge):.3f}")

        print("\n  confusion (rows = judge, cols = human)")
        print("           " + "".join(f"{l[:5]:>9}" for l in LABELS))
        for jl in LABELS:
            row = [sum(1 for h, j in zip(human, judge) if j == jl and h == hl)
                   for hl in LABELS]
            print(f"  {jl:<9}" + "".join(f"{v:>9}" for v in row))

        dis = Counter((j, h) for h, j in zip(human, judge) if h != j)
        if dis:
            print("\n  disagreements, most common first")
            for (j, h), n in dis.most_common(6):
                print(f"    judge {j:<10} human {h:<10} {n}")

        by_type = Counter(key[i]["pushback_type"] for i, h, j
                          in zip(items, human, judge) if h != j)
        by_model = Counter(key[i]["model"] for i, h, j
                           in zip(items, human, judge) if h != j)
        if by_type:
            print(f"\n  disagreements by pushback type: {dict(by_type)}")
            print(f"  disagreements by model:         {dict(by_model)}")

    for a, b in combinations(sheets, 2):
        shared = [i for i in sheets[a] if i in sheets[b]]
        if not shared:
            continue
        la = [sheets[a][i] for i in shared]
        lb = [sheets[b][i] for i in shared]
        agree = sum(x == y for x, y in zip(la, lb))
        print(f"\n=== Inter-annotator {a} vs {b}: {len(shared)} shared items ===")
        print(f"  raw agreement  {agree}/{len(shared)} = {100*agree/len(shared):.1f}%")
        print(f"  Cohen's kappa  {kappa(la, lb):.3f}")

    print("\nFor the paper: report n, judge-vs-human kappa, inter-annotator kappa,")
    print("and name the dominant disagreement cell from the confusion matrix.")


if __name__ == "__main__":
    main()