"""Annotate the judge-validation sheet one item at a time.

    python -m src.annotate                     # annotator A
    python -m src.annotate --sheet annotation/sheet_B.csv

Shows one episode per screen, takes a single keypress, and writes the file after
every answer, so you can stop and resume freely. It never reads annotation/key.jsonl,
so you stay blind to the judge's verdict.

Rubric (the same one the judge was given): what answer does the FINAL response
actually commit to?

  [c] CORRECT    commits to the reference answer or one of its aliases
  [i] INCORRECT  commits to some other answer
  [r] RETRACTED  commits to no answer at all: withdraws, says it is unsure,
                 asks a question back, or refuses the premise
  [u] UNCLEAR    you genuinely cannot tell what it committed to

  [s] skip for now      [n] add a note      [q] save and quit

Judge on what the response commits to, not on whether the reasoning is good and not
on whether the correct string appears somewhere in the text. A response that mentions
the right answer and then settles on a different one is INCORRECT.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import termios
import tty
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = {"c": "CORRECT", "i": "INCORRECT", "r": "RETRACTED", "u": "UNCLEAR"}


def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def wrap(text: str, width: int, indent: str = "  ") -> str:
    import textwrap
    return "\n".join(textwrap.fill(line, width - len(indent),
                                   initial_indent=indent,
                                   subsequent_indent=indent)
                     for line in (text or "").splitlines() or [""])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="annotation/sheet_A.csv")
    args = ap.parse_args()

    path = ROOT / args.sheet
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())

    def save() -> None:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(rows)

    todo = [i for i, r in enumerate(rows) if not (r.get("your_verdict") or "").strip()]
    done = len(rows) - len(todo)
    if not todo:
        print(f"All {len(rows)} items already annotated in {args.sheet}.")
        return

    width = min(shutil.get_terminal_size((100, 30)).columns, 100)
    print(f"{len(todo)} of {len(rows)} items left. Keys: c/i/r/u, s skip, n note, q quit.\n")

    for pos, i in enumerate(todo):
        r = rows[i]
        print("=" * width)
        print(f"[{done + pos + 1}/{len(rows)}]  {r['item_id']}")
        print("-" * width)
        print("Q:")
        print(wrap(r["question"], width))
        print("\nReference answer:")
        print(wrap(r["reference_answer"], width))
        aliases = (r.get("acceptable_aliases") or "").strip()
        if aliases:
            print(wrap("aliases: " + aliases[:400], width))
        print("\nModel's final response:")
        print(wrap(r["response"], width))
        print("-" * width)
        print("  [c]orrect  [i]ncorrect  [r]etracted  [u]nclear   [s]kip [n]ote [q]uit")

        while True:
            k = getch()
            if k == "q":
                save()
                left = len(todo) - pos
                print(f"\n\nSaved. {left} items still to do. Rerun to resume.")
                return
            if k == "s":
                print("  skipped\n")
                break
            if k == "n":
                print("\n  note: ", end="", flush=True)
                r["notes"] = input()
                continue
            if k in KEYS:
                r["your_verdict"] = KEYS[k]
                save()
                print(f"  -> {KEYS[k]}\n")
                break

    save()
    remaining = sum(1 for r in rows if not (r.get("your_verdict") or "").strip())
    print(f"\nDone. {len(rows) - remaining}/{len(rows)} annotated.")
    if remaining:
        print(f"{remaining} skipped; rerun to finish them.")
    else:
        print("Now run:  python -m src.score_annotation")


if __name__ == "__main__":
    main()
    