#!/usr/bin/env python3
"""
Cost-gated runner for the two AI_CLASSIFY steps (calibration and
predictions), plus a read-only query-history check.

Exists because this study's two most expensive steps - calibration
(classify_calibration.sql) and the measurement-window predictions
(drift_signals.sql Section 4) - were being run from hand-copied SQL with no
cost estimate and no confirmation gate. A plausible-looking $26/$74 estimate
turned out to assume the wrong classifier config (bare labels vs. full
per-label descriptions, which the paper's own Section III-C says costs
"nearly two orders of magnitude" more per call) - a 100x error that a dry
run would have caught before any money was spent.

Three modes:

  history  - read-only. Lists past AI_CLASSIFY calls from
             ACCOUNT_USAGE.QUERY_HISTORY so you can see which label
             configuration (bare vs. described) actually ran historically,
             including whatever produced the numbers in the manuscript.
             No cost, no --confirm needed.

  measure  - runs a SMALL, bounded batch of real AI_CLASSIFY calls (--limit,
             e.g. 50) to measure actual tokens/call for a chosen label
             config. This is real, billed usage - small, but real - so it
             still requires --confirm. Prints the exact follow-up query
             against CORTEX_AI_FUNCTIONS_USAGE_HISTORY to read the measured
             token count back (that view lags up to ~3 hours, so it cannot
             be read in the same run).

  run      - runs the FULL step (calibration or predictions) for one arm.
             Requires --tokens-per-call, a number you must already have in
             hand (from `measure`, from `history`, or from your own prior
             measurement) - there is no default and no way to skip this.
             Prints the row count, estimated tokens, estimated AI credits,
             and estimated USD at both Global and Regional on-demand rates,
             and refuses to execute without --confirm.

Usage
    # 1. What actually ran before?
    python run_classification.py --mode history --connection study_v1

    # 2. Measure real tokens/call for each label config (small, real cost)
    python run_classification.py --mode measure --step calibration \
        --arm BASE --labels bare --limit 50 --connection study_v1 --confirm
    python run_classification.py --mode measure --step calibration \
        --arm BASE --labels described --limit 50 --connection study_v1 --confirm

    # 3. Run the full step once you have a real tokens-per-call figure
    python run_classification.py --mode run --step calibration \
        --arm B_s01 --labels bare --tokens-per-call 205 \
        --connection study_v1 --confirm
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Loader import connect  # noqa: E402

# ---------------------------------------------------------------------------
# Rates - VERIFY against the current Snowflake Service Consumption Table
# before trusting these; AI Credit rates change over time, and a stale rate
# here is exactly the kind of silent cost error this script exists to catch.
# Last checked: Snowflake Service Consumption Table, effective 2026-10-29.
# ---------------------------------------------------------------------------
AI_CLASSIFY_CREDITS_PER_MILLION_TOKENS = 1.62
AI_CREDIT_PRICE_USD_GLOBAL = 2.00
AI_CREDIT_PRICE_USD_REGIONAL = 2.20

# RENAMED PAIR (2026-11-01): must match classify_calibration.sql /
# drift_signals.sql Section 4 exactly - those are what actually produced the
# live predictions tables (verified against QUERY_HISTORY). The original
# 'noise_complaint'/'construction_complaint' strings were never what ran.
BARE_LABELS = (
    "['pothole_road','streetlight_outage',"
    "'noise_disturbance_non_construction','construction_site_disturbance',"
    "'illegal_dumping','overflowing_bin',"
    "'graffiti','park_maintenance','water_leak','stray_animal']"
)

# predicted_label_norm maps the two renamed labels back into true_label's
# vocabulary - see classify_calibration.sql's header for the full
# explanation. Every table this script writes to has this column; every
# INSERT below must populate it.
LABEL_NORM_CASE = """
       CASE predicted_label
           WHEN 'noise_disturbance_non_construction' THEN 'noise_complaint'
           WHEN 'construction_site_disturbance' THEN 'construction_complaint'
           ELSE predicted_label
       END AS predicted_label_norm"""

# Mirrors classify_calibration.sql exactly - do not let these two drift apart.
DESCRIBED_LABELS = """[
    {'label': 'pothole_road',
     'description': 'Damage to a road surface such as a pothole, crack, or collapsed pavement'},
    {'label': 'streetlight_outage',
     'description': 'A street light, lamp, or public lighting fixture that is broken or not working'},
    {'label': 'noise_complaint',
     'description': 'Unwanted sound disturbance NOT caused by construction or building work - '
                     'for example music, shouting, or animals'},
    {'label': 'construction_complaint',
     'description': 'A complaint about a building or construction site - including noise, dust, '
                     'or disruption FROM machinery, contractors, or site works'},
    {'label': 'illegal_dumping',
     'description': 'Waste, furniture, or debris dumped illegally in a public place, not in a bin'},
    {'label': 'overflowing_bin',
     'description': 'A public waste bin or container that is full, overflowing, or not emptied'},
    {'label': 'graffiti',
     'description': 'Spray paint, tagging, or vandalism on a wall or public surface'},
    {'label': 'park_maintenance',
     'description': 'Damaged or overgrown park equipment, benches, paths, or landscaping'},
    {'label': 'water_leak',
     'description': 'Water leaking from a pipe, main, hydrant, or road surface'},
    {'label': 'stray_animal',
     'description': 'A loose, stray, or injured animal in a public place'}
]"""

STEP_CONFIG = {
    "calibration": {
        "table": "DT_STUDY.DRIFT.predictions_calibration",
        "day_filter": "day_index BETWEEN 15 AND 34",
        "create_ddl": """
            CREATE TABLE IF NOT EXISTS DT_STUDY.DRIFT.predictions_calibration (
                arm STRING, day_index INT, record_id STRING,
                true_label STRING, predicted_label STRING,
                predicted_label_norm STRING
            )""",
    },
    "predictions": {
        # FIXED (2026-11-01): DT_STUDY.FEAT never existed in this account
        # (confirmed via SHOW SCHEMAS) - the real table drift_signals.sql
        # Section 4 / ref_priors use is DT_STUDY.DRIFT.predictions.
        "table": "DT_STUDY.DRIFT.predictions",
        "day_filter": "day_index >= 15",
        "create_ddl": """
            CREATE TABLE IF NOT EXISTS DT_STUDY.DRIFT.predictions (
                arm STRING, day_index INT, record_id STRING,
                true_label STRING, predicted_label STRING,
                predicted_label_norm STRING
            )""",
    },
}


def cost_estimate(n_rows: int, tokens_per_call: float) -> dict:
    total_tokens = n_rows * tokens_per_call
    credits = total_tokens / 1_000_000 * AI_CLASSIFY_CREDITS_PER_MILLION_TOKENS
    return {
        "rows": n_rows,
        "tokens_per_call": tokens_per_call,
        "total_tokens": total_tokens,
        "credits": credits,
        "usd_global": credits * AI_CREDIT_PRICE_USD_GLOBAL,
        "usd_regional": credits * AI_CREDIT_PRICE_USD_REGIONAL,
    }


def print_estimate(label: str, est: dict) -> None:
    print(f"\n  {label}")
    print(f"    rows                {est['rows']:,}")
    print(f"    tokens/call         {est['tokens_per_call']:,.0f}")
    print(f"    estimated tokens    {est['total_tokens']:,.0f}")
    print(f"    estimated credits   {est['credits']:,.4f}")
    print(f"    estimated USD       ${est['usd_global']:,.2f} (Global) / "
          f"${est['usd_regional']:,.2f} (Regional)")


def cmd_history(args) -> None:
    conn = connect(args.connection)
    cur = conn.cursor()
    try:
        cur.execute(f"""
            SELECT start_time, query_text
            FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
            WHERE query_text ILIKE '%AI_CLASSIFY%'
              AND start_time > DATEADD(month, -{args.months}, CURRENT_TIMESTAMP())
            ORDER BY start_time
        """)
        rows = cur.fetchall()
        if not rows:
            print(f"no AI_CLASSIFY calls found in QUERY_HISTORY over the last "
                  f"{args.months} months (ACCOUNT_USAGE can lag up to ~3 hours "
                  f"for very recent runs, and only retains ~1 year of history)")
            return
        print(f"{len(rows)} AI_CLASSIFY query/queries found:\n")
        for start_time, query_text in rows:
            has_desc = "'description'" in query_text
            variant = "DESCRIBED (expensive)" if has_desc else "bare (cheap)" \
                if "AI_CLASSIFY" in query_text else "unknown"
            print(f"--- {start_time}  [{variant}] " + "-" * 40)
            print(query_text[:2000])
            print()
    finally:
        cur.close()
        conn.close()


# SCRATCH TABLE (2026-11-01): --mode measure never writes to the live
# predictions/predictions_calibration tables. An August 21 overwrite of the
# live calibration table is what destroyed the original run this study's
# published numbers came from - AI_CLASSIFY is not guaranteed to return
# identical labels on a re-run, so even a "replace 50 rows with equivalent
# rows" measurement could nudge the calibration mean/stdev. A measurement
# only needs real billed AI_CLASSIFY calls and a token count back from
# CORTEX_AI_FUNCTIONS_USAGE_HISTORY - it has no reason to touch real data,
# and now it never does.
SCRATCH_TABLE = "DT_STUDY.DRIFT.MEASURE_SCRATCH"
SCRATCH_DDL = f"""
    CREATE TABLE IF NOT EXISTS {SCRATCH_TABLE} (
        arm STRING, day_index INT, record_id STRING,
        true_label STRING, predicted_label STRING,
        predicted_label_norm STRING, query_tag STRING
    )"""


def cmd_measure(args) -> None:
    cfg = STEP_CONFIG[args.step]
    labels_literal = BARE_LABELS if args.labels == "bare" else DESCRIBED_LABELS
    worst_case_tokens_per_call = 200 if args.labels == "bare" else 20_000
    worst_case = cost_estimate(args.limit, worst_case_tokens_per_call)
    print_estimate(
        f"WORST-CASE ceiling for this {args.limit}-call test "
        f"({args.labels} labels)", worst_case)
    print(f"\n  writes to {SCRATCH_TABLE} only - {cfg['table']} (the live "
          f"table) is never touched by --mode measure.")

    if not args.confirm:
        print("\nDRY RUN - this would execute real, billed AI_CLASSIFY calls. "
              "Pass --confirm to actually run it.")
        return

    conn = connect(args.connection)
    cur = conn.cursor()
    tag = f"driftstudy-measure-{args.step}-{args.labels}-{int(time.time())}"
    try:
        cur.execute(f"ALTER SESSION SET QUERY_TAG = '{tag}'")
        cur.execute(SCRATCH_DDL)
        cur.execute(
            f"""
            INSERT INTO {SCRATCH_TABLE}
            SELECT arm, day_index, record_id, true_label, predicted_label,
                   {LABEL_NORM_CASE}, '{tag}' AS query_tag
            FROM (
                SELECT arm, day_index, record_id, true_label,
                       AI_CLASSIFY(text, {labels_literal}):labels[0]::VARCHAR AS predicted_label
                FROM (
                    SELECT arm, day_index, record_id, true_label, text
                    FROM DT_STUDY.DRIFT.daily_records
                    WHERE arm = %(arm)s AND {cfg['day_filter']}
                    ORDER BY HASH(record_id)
                    LIMIT {args.limit}
                )
            )
            """,
            {"arm": args.arm},
        )
        conn.commit()
        print(f"\nran {args.limit} real AI_CLASSIFY calls into {SCRATCH_TABLE}, "
              f"tagged '{tag}'. Live tables untouched.")
        print("ACCOUNT_USAGE lags up to ~3 hours - once it's available, run:\n")
        print(f"""  SELECT query_id,
         SUM(m.value:value::NUMBER) AS total_tokens,
         SUM(credits) AS credits
  FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AI_FUNCTIONS_USAGE_HISTORY,
       LATERAL FLATTEN(input => metrics) m
  WHERE function_name = 'AI_CLASSIFY'
    AND query_tag = '{tag}'
    AND m.value:key.metric::STRING = 'total'
  GROUP BY query_id;
""")
        print(f"divide total_tokens by {args.limit} for the real tokens/call, "
              f"then pass it to --mode run --tokens-per-call.")
    finally:
        cur.close()
        conn.close()


#   A --tokens-per-call reconciled from CORTEX_AI_FUNCTIONS_USAGE_HISTORY
#   shortly after the measure run can read back as zero (or near it) if that
#   view hasn't caught up yet (it can lag ~3 hours) - and a zero/near-zero
#   tokens-per-call makes the full run look absurdly cheap, defeating the
#   entire point of this guard. A real AI_CLASSIFY call against this
#   taxonomy's ~15-token records plus a 10-label array cannot plausibly cost
#   under this many tokens even in the bare-label case; treat anything below
#   it as a lag artifact, not a real measurement.
MIN_PLAUSIBLE_TOKENS_PER_CALL = 30

# Historical reference points from the manuscript (README.md / Section VI-B),
# measured at PAST Cortex rates - not a validation of current rates, just a
# sanity cross-check. If a dry-run estimate at current rates differs from
# these by an order of magnitude or more, stop and double check
# --tokens-per-call and --labels before confirming, rather than assuming
# pricing alone explains the gap.
HISTORICAL_REFERENCE = {
    "calibration": (40_000, 26.0),    # (calls, USD, at old rates, shared across arms)
    "predictions": (110_000, 74.0),   # (calls, USD, at old rates, per arm)
}


def cmd_run(args) -> None:
    if args.tokens_per_call is None:
        raise SystemExit(
            "refusing to run the full step without --tokens-per-call. "
            "Get a real number from --mode measure or --mode history first - "
            "this is the guard against exactly the kind of 100x cost "
            "surprise that prompted this script."
        )
    if args.tokens_per_call < MIN_PLAUSIBLE_TOKENS_PER_CALL:
        raise SystemExit(
            f"--tokens-per-call={args.tokens_per_call} is implausibly low "
            f"(floor: {MIN_PLAUSIBLE_TOKENS_PER_CALL}). This is almost "
            f"certainly a CORTEX_AI_FUNCTIONS_USAGE_HISTORY read taken "
            f"before that view caught up (it can lag ~3 hours) rather than "
            f"a real measurement - wait and re-read it, don't pass this "
            f"value through."
        )

    conn = connect(args.connection)
    cur = conn.cursor()
    cfg = STEP_CONFIG[args.step]
    try:
        cur.execute(
            f"SELECT COUNT(*) FROM DT_STUDY.DRIFT.daily_records "
            f"WHERE arm = %(arm)s AND {cfg['day_filter']}",
            {"arm": args.arm},
        )
        n_rows = cur.fetchone()[0]
        if n_rows == 0:
            raise SystemExit(f"no rows found for arm='{args.arm}' - check it's loaded")

        est = cost_estimate(n_rows, args.tokens_per_call)
        print_estimate(f"FULL RUN estimate: {args.step} / arm={args.arm} / "
                        f"{args.labels} labels", est)

        ref_calls, ref_usd = HISTORICAL_REFERENCE[args.step]
        ratio = est["usd_global"] / ref_usd if ref_usd else float("inf")
        print(f"\n  reference: manuscript reports ~${ref_usd:,.0f} for "
              f"{ref_calls:,} calls at previously-measured rates "
              f"({args.step})")
        if ratio >= 10 or ratio <= 0.1:
            print(f"  >>> this estimate is {ratio:,.1f}x the historical "
                  f"reference - stop and double-check --tokens-per-call and "
                  f"--labels before confirming; pricing drift alone is "
                  f"unlikely to explain a gap this large <<<")

        if not args.confirm:
            print("\nDRY RUN - pass --confirm to actually execute.")
            return

        labels_literal = BARE_LABELS if args.labels == "bare" else DESCRIBED_LABELS
        tag = f"driftstudy-run-{args.step}-{args.arm}-{int(time.time())}"
        cur.execute(f"ALTER SESSION SET QUERY_TAG = '{tag}'")
        cur.execute(cfg["create_ddl"])
        cur.execute(f"DELETE FROM {cfg['table']} WHERE arm = %(arm)s", {"arm": args.arm})
        cur.execute(
            f"""
            INSERT INTO {cfg['table']}
            SELECT arm, day_index, record_id, true_label, predicted_label,
                   {LABEL_NORM_CASE}
            FROM (
                SELECT arm, day_index, record_id, true_label,
                       AI_CLASSIFY(text, {labels_literal}):labels[0]::VARCHAR AS predicted_label
                FROM DT_STUDY.DRIFT.daily_records
                WHERE arm = %(arm)s AND {cfg['day_filter']}
            )
            """,
            {"arm": args.arm},
        )
        conn.commit()
        print(f"\ndone. {n_rows:,} rows classified, tagged '{tag}' for later "
              f"cost reconciliation against CORTEX_AI_FUNCTIONS_USAGE_HISTORY.")
    finally:
        cur.close()
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["history", "measure", "run"])
    ap.add_argument("--step", choices=["calibration", "predictions"])
    ap.add_argument("--arm")
    ap.add_argument("--labels", choices=["bare", "described"], default="bare")
    ap.add_argument("--limit", type=int, help="row cap for --mode measure")
    ap.add_argument("--tokens-per-call", type=float,
                    help="required for --mode run; no default on purpose")
    ap.add_argument("--months", type=int, default=6,
                    help="lookback window for --mode history (default: %(default)s)")
    ap.add_argument("--connection", default=None)
    ap.add_argument("--confirm", action="store_true",
                    help="actually execute billed Snowflake calls; omit for a dry run")
    args = ap.parse_args()

    if args.mode in ("measure", "run") and not (args.step and args.arm):
        raise SystemExit(f"--mode {args.mode} requires --step and --arm")
    if args.mode == "measure" and not args.limit:
        raise SystemExit("--mode measure requires --limit (e.g. --limit 50)")

    {"history": cmd_history, "measure": cmd_measure, "run": cmd_run}[args.mode](args)


if __name__ == "__main__":
    main()
