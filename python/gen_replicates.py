#!/usr/bin/env python3
"""
Generate N independent replicate seeds for a set of arms, and validate each
one with generate_311_corpus.py's Phase 1 gate before it is ever loaded.

This exists because a single-seed detection-lag number (the shipped
corpus/{BASE,B,C,D}) has no confidence interval - it's one draw. Reviewers
asked for multiple realizations per arm; this script produces them.

Each replicate is written to <out>/<arm>_s<NN> (e.g. corpus/B_s01), using
seed = BASE_SEED + <replicate index>, so index 1 never collides with the
shipped single-seed corpus (which uses BASE_SEED itself, unlabeled). It never
touches corpus/BASE, corpus/B, corpus/C, corpus/D.

Usage
    python gen_replicates.py                                   # 10 seeds x BASE,B,C,D, medium
    python gen_replicates.py --n 10 --arms BASE,B,C,D --magnitude medium
    python gen_replicates.py --arms D --n 5 --days 150          # longer window, arm D only

Output
    <out>/<arm>_s<NN>/day_*.jsonl   for every (arm, seed) pair that passes the gate
    <out>/replicates_manifest.csv   one row per pair: arm,seed,label,magnitude,days,gate
Exits non-zero if any replicate fails to generate or fails the validation gate -
check the manifest and do not load a failed replicate.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_311_corpus import SEED as BASE_SEED  # noqa: E402

GENERATOR = Path(__file__).resolve().parent / "generate_311_corpus.py"


def run_one(python: str, arm: str, magnitude: str, seed: int, label: str,
            days: int, out: Path) -> dict:
    gen_cmd = [
        python, str(GENERATOR),
        "--arm", arm, "--magnitude", magnitude,
        "--seed", str(seed), "--label", label,
        "--days", str(days), "--out", str(out),
    ]
    gen = subprocess.run(gen_cmd, capture_output=True, text=True)
    row = {
        "arm": arm, "seed": seed, "label": label,
        "magnitude": magnitude, "days": days,
        "generate_ok": gen.returncode == 0,
        "gate": "SKIPPED (generate failed)",
    }
    if gen.returncode != 0:
        print(f"  [{label}] GENERATE FAILED:\n{gen.stderr}", file=sys.stderr)
        return row

    val_cmd = [python, str(GENERATOR), "--validate", str(out / label)]
    val = subprocess.run(val_cmd, capture_output=True, text=True)
    passed = val.returncode == 0 and "GATE PASSED" in val.stdout
    row["gate"] = "PASSED" if passed else "FAILED"
    if not passed:
        print(f"  [{label}] GATE FAILED:\n{val.stdout}\n{val.stderr}", file=sys.stderr)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="BASE,B,C,D",
                    help="comma-separated base arms to replicate (default: %(default)s)")
    ap.add_argument("--n", type=int, default=10,
                    help="number of replicate seeds per arm (default: %(default)s)")
    ap.add_argument("--magnitude", default="medium", choices=["small", "medium", "large"])
    ap.add_argument("--days", type=int, default=90,
                    help="total days per replicate (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=Path("./corpus"))
    ap.add_argument("--python", default=sys.executable,
                    help="python interpreter to invoke the generator with")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for arm in arms:
        for i in range(1, args.n + 1):
            seed = BASE_SEED + i
            label = f"{arm}_s{i:02d}"
            print(f"generating {label} (seed={seed}) ...", file=sys.stderr)
            rows.append(run_one(args.python, arm, args.magnitude, seed, label,
                                 args.days, args.out))

    manifest_path = args.out / "replicates_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "seed", "label", "magnitude",
                                            "days", "generate_ok", "gate"])
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(1 for r in rows if r["gate"] == "PASSED")
    print(f"\n{n_ok}/{len(rows)} replicates generated and passed the gate.")
    print(f"manifest written to {manifest_path}")
    failed = [r["label"] for r in rows if r["gate"] != "PASSED"]
    if failed:
        print(f"DO NOT LOAD: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
