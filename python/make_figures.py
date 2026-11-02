#!/usr/bin/env python3
"""
Regenerate manuscript Figures 2 (Arm B), 3 (Arm C), 4 (Arm D) from current
DT_STUDY.DRIFT data.

No plotting script for these figures existed anywhere in this repo or the
manuscript folders (confirmed by search) - the originals were made in an
earlier session that never committed one. This script was written from
scratch, matching the published figures' style by sampling exact colors and
pixel dimensions from the embedded images in
"FINAL Article Revised v2.docx" (word/media/image3.png, image4.png,
image5.png = Fig 2, 3, 4 respectively). See the per-signal PALETTE dict
below for the sampled hex values (a Material Design palette).

Known, intentional deviation from the published figures: S5 is added to all
three (it was absent from every published figure, not just Fig 3). S5 has
no published color reference anywhere - Purple 600 (#8E24AA) was chosen as
visually distinct from every sampled color. All three legends are placed
below the axes (single row), matching Fig 3's original layout rather than
Fig 2/4's original in-plot box.

Read-only against Snowflake - no paid Cortex/warehouse cost.

Usage
    python make_figures.py --connection study_v1 --outdir <Final folder>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Loader import connect

DPI = 300
ONSET_DAY = 35
DAY_RANGE = (15, 89)

# Sampled directly from the published figures' embedded images (Material
# Design palette). S5 has no published reference anywhere - Purple 600 is a
# new addition, chosen to be visually distinct from every sampled color.
PALETTE = {
    "S1": "#78909C",   # Blue Grey 400 - sampled
    "S2": "#8D6E63",   # Brown 400 - sampled
    "S3": "#6D4C41",   # Brown 600 - sampled
    "S4": "#00897B",   # Teal 600 - sampled
    "S5": "#8E24AA",   # Purple 600 - NOT in any published figure, new
    "S6": "#1565C0",   # Blue 800 - sampled
    "S7a": "#E65100",  # Orange 900 - sampled
}
ACCURACY_COLOR = "#C83030"  # sampled from Fig 4's "Accuracy (%)" line
ONSET_LINE_COLOR = "black"
CEILING_LINE_COLOR = "#666666"  # sampled from Fig 4's day-75 marker

SIGNALS_ALL = ["S1", "S2", "S3", "S4", "S5", "S6", "S7a"]  # S5 now in Figs 2, 3, 4

plt.rcParams["font.family"] = "serif"


def fetch_arm(conn, arm: str, signals: list[str]) -> dict:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT signal, day_index, statistic
            FROM DT_STUDY.DRIFT.signal_daily
            WHERE arm = %(arm)s AND strategy = 'FIXED' AND signal IN ({})
              AND day_index BETWEEN %(lo)s AND %(hi)s
            ORDER BY signal, day_index
            """.format(",".join(f"'{s}'" for s in signals)),
            {"arm": arm, "lo": DAY_RANGE[0], "hi": DAY_RANGE[1]},
        )
        series = {s: ([], []) for s in signals}
        for signal, day_index, statistic in cur.fetchall():
            series[signal][0].append(day_index)
            series[signal][1].append(float(statistic))

        cur.execute(
            """
            SELECT signal, threshold
            FROM DT_STUDY.DRIFT.signal_fired
            WHERE arm = %(arm)s AND signal IN ({})
            GROUP BY signal, threshold
            """.format(",".join(f"'{s}'" for s in signals)),
            {"arm": arm},
        )
        thresholds = {signal: float(threshold) for signal, threshold in cur.fetchall()}
    finally:
        cur.close()
    return series, thresholds


def fetch_accuracy(conn, arm: str) -> tuple[list[int], list[float]]:
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT day_index, true_accuracy * 100
            FROM DT_STUDY.DRIFT.accuracy_daily
            WHERE arm = %(arm)s AND day_index BETWEEN %(lo)s AND %(hi)s
            ORDER BY day_index
            """,
            {"arm": arm, "lo": DAY_RANGE[0], "hi": DAY_RANGE[1]},
        )
        rows = cur.fetchall()
    finally:
        cur.close()
    return [r[0] for r in rows], [float(r[1]) for r in rows]


def ratio_series(series: dict, thresholds: dict, signal: str) -> tuple[list[int], list[float]]:
    days, vals = series[signal]
    thr = thresholds.get(signal)
    if not thr:
        return days, [0.0 for _ in vals]
    return days, [v / thr for v in vals]


def plot_fig2(conn, out_path: Path) -> None:
    series, thresholds = fetch_arm(conn, "B", SIGNALS_ALL)
    fig, ax = plt.subplots(figsize=(2361 / DPI, 952 / DPI))
    for signal in SIGNALS_ALL:
        days, ratio = ratio_series(series, thresholds, signal)
        marker = "o" if signal == "S7a" else None
        ax.plot(days, ratio, color=PALETTE[signal], label=signal,
                marker=marker, markersize=3, linewidth=1.3)
    ax.axvline(ONSET_DAY, color=ONSET_LINE_COLOR, linestyle=":", linewidth=1.2)
    ax.axhline(1.0, color="dimgray", linestyle="--", linewidth=1.2)
    ax.set_title("Arm B — abrupt vocabulary shift (100% of records affected)", fontsize=15)
    ax.set_xlabel("Simulated day", fontsize=14)
    ax.set_ylabel("statistic / threshold", fontsize=14)
    ax.set_ylim(bottom=-0.3)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=7, fontsize=10)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_fig3(conn, out_path: Path) -> dict:
    series, thresholds = fetch_arm(conn, "C", SIGNALS_ALL)
    left_signals = ["S1", "S2", "S3", "S4", "S5", "S6"]

    fig, ax_l = plt.subplots(figsize=(2044 / DPI, 954 / DPI))
    for signal in left_signals:
        days, ratio = ratio_series(series, thresholds, signal)
        ax_l.plot(days, ratio, color=PALETTE[signal], label=signal, linewidth=1.3)
    ax_l.axvline(ONSET_DAY, color=ONSET_LINE_COLOR, linestyle=":", linewidth=1.2)
    ax_l.axhline(1.0, color="dimgray", linestyle="--", linewidth=1.2)
    ax_l.set_title("Arm C — novel class (≈15% of records affected)", fontsize=15)
    ax_l.set_xlabel("Simulated day", fontsize=14)
    ax_l.set_ylabel("statistic / threshold", fontsize=14)

    ax_r = ax_l.twinx()
    days7a, ratio7a = ratio_series(series, thresholds, "S7a")
    ax_r.plot(days7a, ratio7a, color=PALETTE["S7a"], label="S7a",
              marker="o", markersize=3, linewidth=1.3)
    ax_r.set_ylabel("S7a statistic / threshold", fontsize=14, color=PALETTE["S7a"])
    ax_r.tick_params(axis="y", colors=PALETTE["S7a"])
    ax_r.set_ylim(bottom=0)

    handles_l, labels_l = ax_l.get_legend_handles_labels()
    handles_r, labels_r = ax_r.get_legend_handles_labels()
    fig.legend(handles_l + handles_r, labels_l + labels_r,
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=7, fontsize=10)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")

    post_onset = [r for d, r in zip(days7a, ratio7a) if d >= ONSET_DAY]
    return {"min": min(post_onset), "max": max(post_onset)} if post_onset else {}


def plot_fig4(conn, out_path: Path) -> None:
    series, thresholds = fetch_arm(conn, "D", SIGNALS_ALL)
    acc_days, acc_vals = fetch_accuracy(conn, "D")
    # "baseline" is the mean accuracy over the calibration window (15-34),
    # not day 15 alone - verified against the manuscript's stated 99.49%.
    calib_vals = [v for d, v in zip(acc_days, acc_vals) if 15 <= d <= 34]
    baseline = sum(calib_vals) / len(calib_vals) if calib_vals else None

    fig, ax_l = plt.subplots(figsize=(2524 / DPI, 1072 / DPI))
    for signal in SIGNALS_ALL:
        days, ratio = ratio_series(series, thresholds, signal)
        marker = "o" if signal == "S7a" else None
        ax_l.plot(days, ratio, color=PALETTE[signal], label=signal,
                  marker=marker, markersize=3, linewidth=1.3)
    ax_l.axvline(ONSET_DAY, color=ONSET_LINE_COLOR, linestyle=":", linewidth=1.2)
    ax_l.axhline(1.0, color="dimgray", linestyle="--", linewidth=1.2)
    ax_l.set_title("Arm D — gradual semantic shift (up to 30% of records affected)", fontsize=15)
    ax_l.set_xlabel("Simulated day", fontsize=14)
    ax_l.set_ylabel("statistic / threshold", fontsize=14)

    ax_r = ax_l.twinx()
    ax_r.plot(acc_days, acc_vals, color=ACCURACY_COLOR, linewidth=2.2, label="Accuracy (%)")
    if baseline is not None:
        ax_r.axhline(baseline, color=ACCURACY_COLOR, linestyle=":", linewidth=1.2)
    ax_r.set_ylabel("Classification accuracy (%)", fontsize=14, color=ACCURACY_COLOR)
    ax_r.tick_params(axis="y", colors=ACCURACY_COLOR)

    handles_l, labels_l = ax_l.get_legend_handles_labels()
    handles_r, labels_r = ax_r.get_legend_handles_labels()
    fig.legend(handles_l + handles_r, labels_l + labels_r,
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=8, fontsize=10)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"  arm D accuracy baseline (mean, days 15-34): {baseline:.4g}%"
          if baseline is not None else "  no accuracy data found")
    fire69 = next((v for d, v in zip(acc_days, acc_vals) if d == 69), None)
    if fire69 is not None and baseline is not None:
        print(f"  arm D accuracy at day 69 (S7a fire day): {fire69:.4g}% "
              f"(decline: {baseline - fire69:.2f} pp)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connection", default="study_v1")
    ap.add_argument("--outdir", type=Path, required=True,
                    help="directory to write fig2.png, fig3.png, fig4.png into")
    ap.add_argument("--only", nargs="+", choices=["fig2", "fig3", "fig4"],
                    default=["fig2", "fig3", "fig4"],
                    help="regenerate only these figures, leaving the others untouched")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    conn = connect(args.connection)
    s7a_range = {}
    try:
        if "fig2" in args.only:
            plot_fig2(conn, args.outdir / "fig2.png")
        if "fig3" in args.only:
            s7a_range = plot_fig3(conn, args.outdir / "fig3.png")
        if "fig4" in args.only:
            plot_fig4(conn, args.outdir / "fig4.png")
    finally:
        conn.close()

    if s7a_range:
        print(f"\nArm C S7a statistic/threshold, post-onset (day>={ONSET_DAY}): "
              f"min={s7a_range['min']:.2f}x, max={s7a_range['max']:.2f}x")


if __name__ == "__main__":
    main()
