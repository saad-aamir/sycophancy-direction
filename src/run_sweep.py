"""v2 sweep orchestrator.

For each model in the frozen execution order (pre-reg section 10):
screen -> baseline (batched) -> rejudge -> pairs -> cache (multi-locus)
-> extract (v1 primary-locus gate). Each stage runs as its own
subprocess (GPU memory frees between models/stages) and is skipped if
its output exists, so the sweep is resumable after any crash.

Refuses to start unless all acceptance markers exist (A1 reproduction,
A2 numerics, engine equivalence for one model per family) and
PRE_REGISTRATION.md is committed. Also refuses on a dirty src/configs
tree unless --allow-dirty, so every run is attributable to a commit.

    python -m src.run_sweep --set-active qwen1.5b     # write active_model.yaml, exit
    python -m src.run_sweep --pool data/pool.jsonl
    python -m src.run_sweep --pool data/pool.jsonl --only qwen7b,llama8b
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from .common import ROOT, load_config

# Pre-reg section 10: partial data under the stop-rule is non-selective
# only if this order is enforced. Do not reorder.
FROZEN_ORDER = ("qwen1.5b", "qwen7b", "llama8b", "qwen14b",
                "llama1b", "qwen0.5b", "qwen3b", "llama3b")

REQUIRED_MARKERS = (
    "outputs/qwen1.5b_repro/REPRO_PASS",   # A1 (pre-reg s7): behavioral reproduction
    "outputs/qwen1.5b/NUMERICS_PASS",          # engine equivalence, Llama family
)

STAGES = [
    ("screen",  ["-m", "src.run_behavioral_eval_v2", "--mode", "screen",
                 "--pool", "{pool}"],            "data/questions_{slug}.jsonl"),
    ("baseline", ["-m", "src.run_behavioral_eval_v2", "--mode", "main"],
                                                "outputs/{slug}/transcripts/baseline.jsonl"),
    ("rejudge", ["-m", "src.rejudge", "--condition", "baseline"],
                                                "outputs/{slug}/transcripts/baseline.heuristic.jsonl"),
    ("pairs",   ["-m", "src.build_contrast_pairs"],
                                                "outputs/{slug}/activations/pairs_capitulation.jsonl"),
    ("cache",   ["-m", "src.cache_activations_v2"],
                                                "outputs/{slug}/activations/capitulation_acts.pt"),
    ("extract", ["-m", "src.extract_directions"],
                                                "outputs/{slug}/directions/capitulation_directions.pt"),
]


def preflight(allow_dirty: bool = False):
    missing = [m for m in REQUIRED_MARKERS if not (ROOT / m).exists()]
    if missing:
        sys.exit("[sweep] missing acceptance markers:\n  " + "\n  ".join(missing)
                 + "\nRun check_reproduction / check_numerics / check_equivalence first.")
    r = subprocess.run(["git", "-C", str(ROOT), "log", "--oneline", "--",
                        "PRE_REGISTRATION.md"], capture_output=True, text=True)
    if not r.stdout.strip():
        sys.exit("[sweep] PRE_REGISTRATION.md is not committed. Commit the "
                 "pre-registration before collecting data. That ordering is "
                 "the whole point.")
    d = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                        "--", "src", "configs", "PRE_REGISTRATION.md"],
                       capture_output=True, text=True)
    if d.stdout.strip() and not allow_dirty:
        sys.exit("[sweep] uncommitted changes in src/ or configs/:\n"
                 + d.stdout + "Commit them (provenance) or pass --allow-dirty.")


def set_active(entry: dict):
    active = {"slug": entry["slug"], "name": entry["name"],
              "dtype": entry.get("dtype", "bfloat16"),
              "questions": f"data/questions_{entry['slug']}.jsonl"}
    with open(ROOT / "configs/active_model.yaml", "w") as f:
        yaml.safe_dump(active, f)
    print(f"[sweep] active model -> {entry['slug']} ({entry['name']})")


def run_model(entry: dict, pool: str):
    slug = entry["slug"]
    print(f"\n{'=' * 60}\n[sweep] MODEL {slug} ({entry['name']})\n{'=' * 60}")
    set_active(entry)
    for name, suffix, out_tmpl in STAGES:
        out = ROOT / out_tmpl.format(slug=slug, pool=pool)
        if out.exists():
            print(f"[sweep:{slug}] {name}: exists, skipping ({out.name})")
            continue
        cmd = [sys.executable] + [s.format(pool=pool) for s in suffix]
        print(f"[sweep:{slug}] {name}: {' '.join(cmd)}")
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"[sweep:{slug}] stage '{name}' failed (exit "
                     f"{r.returncode}). Sweep halted; rerun to resume here.")


def ordered_registry(cfg: dict) -> list[dict]:
    rank = {s: i for i, s in enumerate(FROZEN_ORDER)}
    reg = sorted(cfg["registry"], key=lambda e: rank.get(e["slug"], 999))
    for e in reg:
        if e["slug"] not in rank:
            print(f"[sweep] WARNING: {e['slug']} not in FROZEN_ORDER; runs last.")
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/pool.jsonl")
    ap.add_argument("--only", default=None, help="comma-separated slugs")
    ap.add_argument("--set-active", default=None, metavar="SLUG",
                    help="write configs/active_model.yaml for SLUG and exit")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    registry = ordered_registry(cfg)

    if args.set_active:
        entry = next((e for e in registry if e["slug"] == args.set_active), None)
        if entry is None:
            sys.exit(f"[sweep] unknown slug {args.set_active}")
        set_active(entry)
        return

    preflight(allow_dirty=args.allow_dirty)
    if args.only:
        keep = set(args.only.split(","))
        registry = [e for e in registry if e["slug"] in keep]

    for entry in registry:
        run_model(entry, args.pool)

    print("\n[sweep] complete. Gate verdicts:")
    import torch
    for entry in ordered_registry(cfg):
        d = ROOT / f"outputs/{entry['slug']}/directions/capitulation_directions.pt"
        if not d.exists():
            continue
        try:
            blob = torch.load(d, map_location="cpu")
            L = blob["best_layer"]
            print(f"  {entry['slug']:12s} best L{L:>2} "
                  f"AUROC {float(blob['auroc'][L]):.3f} "
                  f"gate {'PASS' if blob.get('gate_pass') else 'FAIL'}")
        except Exception as e:
            print(f"  {entry['slug']:12s} summary unavailable ({e})")


if __name__ == "__main__":
    main()