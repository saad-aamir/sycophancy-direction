"""Audit CLI for the frozen v2 pushback templates.

The assignment rule lives in exactly one place, src.common.paraphrase_index
(pre-registered: int(md5(f"{qid}:{ptype}"), 16) % n). This tool only
validates the yaml (4 types x 5 non-empty strings), echoes the v1
originals at index 0, prints the per-type assignment histogram over a
pool, and emits the mapping checksum for RESEARCH_LOG.md.

    python -m src.templates [--pool data/pool.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from .common import ROOT, paraphrase_index

TYPES = ("simple", "authoritative", "emotional", "social")
N_PARAPHRASES = 5
DEFAULT_PATH = ROOT / "configs" / "templates_v2.yaml"


def load_and_validate(path: Path = DEFAULT_PATH) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or sorted(data) != sorted(TYPES):
        raise ValueError(f"template types mismatch: {data}")
    for ptype in TYPES:
        plist = data[ptype]
        if not isinstance(plist, list) or len(plist) != N_PARAPHRASES:
            raise ValueError(f"{ptype}: expected {N_PARAPHRASES} paraphrases")
        for i, p in enumerate(plist):
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"{ptype}[{i}]: empty or non-string")
    return data


def assignment_checksum(qids) -> str:
    parts = [f"{q}:{t}:{paraphrase_index(q, t)}"
             for q in sorted(set(qids)) for t in TYPES]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT / "data" / "pool.jsonl"))
    ap.add_argument("--path", default=str(DEFAULT_PATH))
    args = ap.parse_args()

    templates = load_and_validate(Path(args.path))
    print(f"[templates] OK: {len(TYPES)} types x {N_PARAPHRASES} paraphrases")
    for ptype in TYPES:
        head = templates[ptype][0]
        print(f"  {ptype:14s} #0 (v1 original): {head[:58]}"
              f"{'...' if len(head) > 58 else ''}")

    pool = Path(args.pool)
    if not pool.exists():
        print(f"[templates] pool not found at {pool}; audit skipped")
        return 0
    qids = [json.loads(l)["id"] for l in pool.open(encoding="utf-8") if l.strip()]
    print(f"[templates] assignment audit over {len(qids)} questions")
    for ptype in TYPES:
        counts = Counter(paraphrase_index(q, ptype) for q in qids)
        print(f"  {ptype:14s} " + "  ".join(f"{i}:{counts.get(i, 0)}"
                                            for i in range(N_PARAPHRASES)))
    print(f"[templates] assignment checksum: {assignment_checksum(qids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())