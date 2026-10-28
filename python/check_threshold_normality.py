#!/usr/bin/env python3
"""
Test the mean+3*SD threshold's normality assumption against the reference
window, and compare it with an empirical-quantile threshold.

The firing rule (DESIGN.md §4.1, README.md §3.7) is:
    fires on day d if statistic(d) > mean(ref) + 3*stdev(ref)
"mean + 3*SD" is only the right cutoff for a ~0.135% one-sided false-alarm
rate if the reference-window statistic is approximately normal. This script
checks that assumption directly instead of asserting it, and reports what
the threshold would be if you instead used the reference window's own
empirical 99.865th percentile (the quantile that gives the same 0.135%
one-sided rate without assuming a distribution shape).

Input: a CSV export of DT_STUDY.RESULT.signal_daily restricted to the
reference window (day_index BETWEEN 15 AND 34), columns: arm, day_index,
signal, strategy, statistic. Grouped by (signal, strategy) across whatever
arms/seeds are in the file - pass in just BASE, or BASE's replicate seeds
pooled, for the cleanest normality read (drift arms' day_index 15-34 is by
construction identical to BASE's, so pooling them only inflates n, it
doesn't change what's being tested).

Requires numpy and scipy (pip install numpy scipy) - intentionally kept out
of generate_311_corpus.py's stdlib-only requirement; this is an analysis
script over exported data, not part of corpus generation.

Usage
    python check_threshold_normality.py signal_daily_ref_window.csv --out normality_report.md
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    from scipy import stats
except ImportError:
    sys.exit(
        "this script needs numpy and scipy: pip install numpy scipy\n"
        "(these are not required anywhere else in the repro package)"
    )

ONE_SIDED_3SIGMA_QUANTILE = stats.norm.cdf(3)  # ~0.99865


def load_groups(path: Path) -> dict[tuple[str, str], list[float]]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["signal"], row["strategy"])
            groups[key].append(float(row["statistic"]))
    return groups


def analyze_group(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean, sd = float(arr.mean()), float(arr.std(ddof=1))

    mean_3sd_threshold = mean + 3 * sd
    empirical_threshold = float(np.quantile(arr, ONE_SIDED_3SIGMA_QUANTILE))

    result = {
        "n": n,
        "mean": mean,
        "sd": sd,
        "skewness": float(stats.skew(arr)),
        "kurtosis_excess": float(stats.kurtosis(arr)),  # 0 for a normal dist
        "mean_3sd_threshold": mean_3sd_threshold,
        "empirical_99.865pct_threshold": empirical_threshold,
        "threshold_pct_difference": (
            (empirical_threshold - mean_3sd_threshold) / mean_3sd_threshold * 100
            if mean_3sd_threshold else float("nan")
        ),
    }

    if n < 3:
        result["shapiro_p"] = None
        result["qq_correlation"] = None
        result["verdict"] = "n<3, cannot test"
        return result

    if n <= 5000:
        _, shapiro_p = stats.shapiro(arr)
        result["shapiro_p"] = float(shapiro_p)
    else:
        result["shapiro_p"] = None  # Shapiro-Wilk is unreliable/unsupported above n=5000

    # QQ summary: correlation between sorted sample and the normal order
    # statistics it would have if it were exactly normal. 1.0 = perfectly
    # normal-looking; this is the standard quantitative stand-in for "eyeball
    # the QQ plot" when we'd rather not ship a plot per signal.
    osm, osr = stats.probplot(arr, dist="norm", fit=False)
    result["qq_correlation"] = float(np.corrcoef(osm, osr)[0, 1])

    # Shapiro-Wilk is the actual statistical test and gates the verdict; QQ
    # correlation is reported alongside as the requested "QQ summary" but, at
    # the small sample sizes here (one reference window is 20 days), it has
    # enough natural spread under a TRUE null that gating on it too produces
    # false "departs from normal" verdicts - so it's diagnostic, not a gate.
    normal_ish = result["shapiro_p"] is None or result["shapiro_p"] > 0.05
    result["verdict"] = "consistent with normal" if normal_ish else "departs from normal"
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("--out", type=Path, default=Path("normality_report.md"))
    args = ap.parse_args()

    groups = load_groups(args.csv_path)
    if not groups:
        raise SystemExit(f"no rows found in {args.csv_path}")

    lines = [
        "| Signal | Strategy | n | Shapiro p | QQ corr | Skew | Excess kurt | "
        "mean+3SD | empirical 99.865% | diff | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (signal, strategy), values in sorted(groups.items()):
        r = analyze_group(values)
        shapiro = f"{r['shapiro_p']:.4f}" if r["shapiro_p"] is not None else "n/a"
        lines.append(
            f"| {signal} | {strategy} | {r['n']} | {shapiro} | "
            f"{r['qq_correlation']:.4f} | {r['skewness']:.3f} | "
            f"{r['kurtosis_excess']:.3f} | {r['mean_3sd_threshold']:.5f} | "
            f"{r['empirical_99.865pct_threshold']:.5f} | "
            f"{r['threshold_pct_difference']:+.1f}% | {r['verdict']} |"
        )
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    print(
        "\nShapiro p <= 0.05 or QQ correlation well below 1 means the "
        "normal-threshold assumption is shaky for that signal; the paper "
        "should either justify mean+3SD on other grounds (e.g. it's a "
        "convention, not a claim of normality) or report the empirical-"
        "quantile threshold as the operating point instead.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
