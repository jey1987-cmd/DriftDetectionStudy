#!/usr/bin/env python3
"""
Summarize detection results across replicate seeds: median detection lag with
a 95% bootstrap CI, fraction of seeds detected, and (for the BASE arm) the
empirical false-alarm rate.

Input is a CSV exported from DT_STUDY.RESULT.detection_summary (or an
equivalent query), one row per (arm, signal, strategy), with at least these
columns:
    arm             e.g. "B_s01", "BASE_s07" (base arm + "_s<seed index>",
                    the convention written by generate_311_corpus.py --label
                    and gen_replicates.py)
    signal          e.g. "S1".."S7a"
    strategy        "FIXED" | "SLIDING"
    fire_day        first day the signal fired in the measurement window
                    (35-89), or empty/NULL if it never fired
    false_positives count of firings during the reference window (15-34)

Rows are grouped by (base_arm, signal, strategy), where base_arm is "arm"
with the trailing "_s<digits>" stripped - so B_s01..B_s10 all fold into one
"B" row. A non-replicate arm value (plain "BASE", "B", ...) is its own group
of size 1; that's allowed but the CI/fraction numbers are meaningless with
one sample.

For arm "BASE": since BASE has no injected drift, any fire_day in the
measurement window is by definition a false alarm, not a detection. So for
BASE, "fraction_detected" IS the empirical measurement-window false-alarm
rate across replicate seeds, and false_positives gives the reference-window
rate. Both are reported; no "detection lag" is meaningful for BASE and is
reported as such alongside, not silently dropped.

Usage
    python analyze_replicates.py detections.csv --out results
        -> results.csv, results.md
    python analyze_replicates.py detections.csv --out results --latex
        -> also results.tex
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DRIFT_ONSET = 35          # must match generate_311_corpus.py
REF_DAYS = 20             # days 15-34 inclusive
REPLICATE_RE = re.compile(r"^(?P<base>.+?)_s\d+$")
N_BOOTSTRAP = 10_000


def base_arm(arm: str) -> str:
    m = REPLICATE_RE.match(arm)
    return m.group("base") if m else arm


def to_int_or_none(v: str | None) -> int | None:
    if v is None or v == "":
        return None
    return int(float(v))


def bootstrap_median_ci(values: list[float], reps: int = N_BOOTSTRAP,
                        alpha: float = 0.05) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    rng = random.Random(0)  # fixed seed - reproducible CI, not a security context
    medians = []
    n = len(values)
    for _ in range(reps):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(sample))
    medians.sort()
    lo_idx = int((alpha / 2) * reps)
    hi_idx = int((1 - alpha / 2) * reps) - 1
    return medians[lo_idx], medians[hi_idx]


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def analyze(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        key = (base_arm(r["arm"]), r["signal"], r["strategy"])
        groups[key].append(r)

    out = []
    for (arm, signal, strategy), grp in sorted(groups.items()):
        fire_days = [to_int_or_none(r.get("fire_day")) for r in grp]
        false_pos = [to_int_or_none(r.get("false_positives")) or 0 for r in grp]
        detected = [f for f in fire_days if f is not None]
        lags = [f - DRIFT_ONSET for f in detected]

        ci = bootstrap_median_ci(lags) if lags else None
        row = {
            "arm": arm,
            "signal": signal,
            "strategy": strategy,
            "n_seeds": len(grp),
            "n_detected": len(detected),
            "fraction_detected": round(len(detected) / len(grp), 3),
            "median_lag_days": statistics.median(lags) if lags else "",
            "lag_ci95_lo": ci[0] if ci else "",
            "lag_ci95_hi": ci[1] if ci else "",
            "ref_window_false_positive_rate": round(
                sum(false_pos) / (len(false_pos) * REF_DAYS), 4),
        }
        if arm == "BASE":
            # for the stationary control, "detected" == "falsely fired during
            # the measurement window" - rename for clarity in the output,
            # don't silently reuse the drift-arm column semantics.
            row["measurement_window_false_alarm_rate"] = row["fraction_detected"]
        out.append(row)
    return out


def write_csv(results: list[dict], path: Path) -> None:
    fields = ["arm", "signal", "strategy", "n_seeds", "n_detected",
              "fraction_detected", "median_lag_days", "lag_ci95_lo",
              "lag_ci95_hi", "ref_window_false_positive_rate",
              "measurement_window_false_alarm_rate"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(results)


def write_markdown(results: list[dict], path: Path) -> None:
    lines = [
        "| Arm | Signal | Strategy | Seeds | Detected | Median lag (95% CI) | "
        "Ref-window FP rate | BASE false-alarm rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["median_lag_days"] == "":
            lag = "not detected"
        elif r["lag_ci95_lo"] == "":
            lag = f"{r['median_lag_days']:.0f}d (n=1, no CI)"
        else:
            lag = f"{r['median_lag_days']:.0f}d ({r['lag_ci95_lo']:.0f}-{r['lag_ci95_hi']:.0f})"
        base_rate = r.get("measurement_window_false_alarm_rate", "")
        lines.append(
            f"| {r['arm']} | {r['signal']} | {r['strategy']} | {r['n_seeds']} | "
            f"{r['n_detected']}/{r['n_seeds']} | {lag} | "
            f"{r['ref_window_false_positive_rate']:.1%} | "
            f"{base_rate if base_rate == '' else f'{base_rate:.1%}'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(results: list[dict], path: Path) -> None:
    lines = [
        r"\begin{tabular}{llllll}",
        r"\toprule",
        r"Arm & Signal & Strategy & Detected & Median lag (95\% CI) & FP rate \\",
        r"\midrule",
    ]
    for r in results:
        if r["median_lag_days"] == "":
            lag = "not detected"
        elif r["lag_ci95_lo"] == "":
            lag = f"{r['median_lag_days']:.0f}d (n=1)"
        else:
            lag = f"{r['median_lag_days']:.0f}d ({r['lag_ci95_lo']:.0f}--{r['lag_ci95_hi']:.0f})"
        lines.append(
            f"{r['arm']} & {r['signal']} & {r['strategy']} & "
            f"{r['n_detected']}/{r['n_seeds']} & {lag} & "
            f"{r['ref_window_false_positive_rate']:.1%} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", type=Path,
                    help="CSV export of per-replicate detection results")
    ap.add_argument("--out", type=Path, default=Path("replicate_analysis"),
                    help="output path stem (default: %(default)s); "
                         "writes <out>.csv and <out>.md")
    ap.add_argument("--latex", action="store_true",
                    help="also write <out>.tex")
    args = ap.parse_args()

    rows = load_rows(args.csv_path)
    if not rows:
        raise SystemExit(f"no rows found in {args.csv_path}")
    results = analyze(rows)

    write_csv(results, args.out.with_suffix(".csv"))
    write_markdown(results, args.out.with_suffix(".md"))
    if args.latex:
        write_latex(results, args.out.with_suffix(".tex"))

    print(f"{len(results)} (arm, signal, strategy) groups from "
          f"{len(rows)} input rows", file=sys.stderr)
    print(f"wrote {args.out.with_suffix('.csv')} and {args.out.with_suffix('.md')}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
