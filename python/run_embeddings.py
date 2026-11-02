#!/usr/bin/env python3
"""
Cost-gated runner for drift_signals.sql Section 3 (Tier 2 embeddings).

Exists for the same reason run_classification.py does: that section's own
comment calls embeddings "the single most expensive table to recompute
carelessly" (EMBED_TEXT_768/AI_EMBED is metered per row), but until now it
only existed as raw SQL with '<ARM>' to hand-substitute - no cost estimate,
no confirmation gate.

Rate is not guessed - it's read back from this account's own historical
AI_EMBED usage (arm B's embeddings run: 18,000 rows, 361,780 tokens, 0.0108534
credits - see SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AI_FUNCTIONS_USAGE_HISTORY).
That gives ~20.1 tokens/record and ~6.03e-7 credits/record for this corpus
and model (snowflake-arctic-embed-m); --tokens-per-record lets you override it.

Scope, mirroring drift_signals.sql Section 3 exactly:
  1. daily_sample   - deterministic 200/day sample, DELETE+INSERT per arm.
  2. embeddings     - EMBED_TEXT_768 over daily_sample, DELETE+INSERT per arm.
  3. ref_sample_emb - 2,000-row reference sample of embeddings, needed for S6.

Usage
    python run_embeddings.py --arm C --connection study_v1              # dry run
    python run_embeddings.py --arm C --connection study_v1 --confirm    # real run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Loader import connect  # noqa: E402

# Rates - see claude_code_cost_rates.md note in run_classification.py for why
# these need periodic re-verification against the Service Consumption Table.
AI_CREDIT_PRICE_USD_GLOBAL = 2.00
AI_CREDIT_PRICE_USD_REGIONAL = 2.20

# Measured empirically from this account's own arm B embeddings run (see
# module docstring) - not a vendor-quoted rate, a real one.
MEASURED_TOKENS_PER_RECORD = 361780 / 18000      # ~20.10
MEASURED_CREDITS_PER_RECORD = 0.0108534 / 18000  # ~6.0297e-7

DDL = {
    "daily_sample": """
        CREATE TABLE IF NOT EXISTS DT_STUDY.DRIFT.daily_sample (
            arm STRING, day_index INT, record_id STRING, text STRING
        )""",
    "embeddings": """
        CREATE TABLE IF NOT EXISTS DT_STUDY.DRIFT.embeddings (
            arm STRING, day_index INT, record_id STRING, emb ARRAY
        )""",
    "ref_sample_emb": """
        CREATE TABLE IF NOT EXISTS DT_STUDY.DRIFT.ref_sample_emb (
            arm STRING, record_id STRING, emb ARRAY
        )""",
}


def cost_estimate(n_rows: int, credits_per_record: float) -> dict:
    credits = n_rows * credits_per_record
    return {
        "rows": n_rows,
        "credits_per_record": credits_per_record,
        "credits": credits,
        "usd_global": credits * AI_CREDIT_PRICE_USD_GLOBAL,
        "usd_regional": credits * AI_CREDIT_PRICE_USD_REGIONAL,
    }


def print_estimate(label: str, est: dict) -> None:
    print(f"\n  {label}")
    print(f"    rows                 {est['rows']:,}")
    print(f"    credits/record       {est['credits_per_record']:.4e}")
    print(f"    estimated credits    {est['credits']:.6f}")
    print(f"    estimated USD        ${est['usd_global']:.4f} (Global) / "
          f"${est['usd_regional']:.4f} (Regional)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--connection", default=None)
    ap.add_argument("--credits-per-record", type=float,
                    default=MEASURED_CREDITS_PER_RECORD,
                    help="override the measured per-record rate")
    ap.add_argument("--confirm", action="store_true",
                    help="actually execute billed Snowflake calls; omit for a dry run")
    args = ap.parse_args()

    conn = connect(args.connection)
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT COUNT(*) FROM DT_STUDY.DRIFT.daily_records WHERE arm = %(arm)s",
            {"arm": args.arm},
        )
        n_records = cur.fetchone()[0]
        if n_records == 0:
            raise SystemExit(
                f"no daily_records found for arm='{args.arm}' - load its "
                f"corpus first (Loader.py --arm {args.arm})"
            )

        # k=200/day sample, same as drift_signals.sql Section 3 - this is the
        # row count that actually gets embedded, not n_records.
        cur.execute(
            "SELECT COUNT(DISTINCT day_index) FROM DT_STUDY.DRIFT.daily_records "
            "WHERE arm = %(arm)s", {"arm": args.arm},
        )
        n_days = cur.fetchone()[0]
        n_sample_rows = n_days * 200

        est = cost_estimate(n_sample_rows, args.credits_per_record)
        print_estimate(f"EMBEDDINGS estimate: arm={args.arm} "
                        f"({n_days} days x 200/day sample)", est)

        if not args.confirm:
            print("\nDRY RUN - this would execute real, billed AI_EMBED calls. "
                  "Pass --confirm to actually run it.")
            return

        cur.execute(DDL["daily_sample"])
        cur.execute("DELETE FROM DT_STUDY.DRIFT.daily_sample WHERE arm = %(arm)s",
                    {"arm": args.arm})
        cur.execute(
            """
            INSERT INTO DT_STUDY.DRIFT.daily_sample
            SELECT arm, day_index, record_id, text
            FROM (
                SELECT arm, day_index, record_id, text,
                       ROW_NUMBER() OVER (
                           PARTITION BY arm, day_index ORDER BY HASH(record_id)
                       ) AS rn
                FROM DT_STUDY.DRIFT.daily_records
                WHERE arm = %(arm)s
            )
            WHERE rn <= 200
            """,
            {"arm": args.arm},
        )

        cur.execute(DDL["embeddings"])
        cur.execute("DELETE FROM DT_STUDY.DRIFT.embeddings WHERE arm = %(arm)s",
                    {"arm": args.arm})
        cur.execute(
            """
            INSERT INTO DT_STUDY.DRIFT.embeddings
            SELECT arm, day_index, record_id,
                   SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', text)::ARRAY AS emb
            FROM DT_STUDY.DRIFT.daily_sample
            WHERE arm = %(arm)s
            """,
            {"arm": args.arm},
        )

        cur.execute(DDL["ref_sample_emb"])
        cur.execute("DELETE FROM DT_STUDY.DRIFT.ref_sample_emb WHERE arm = %(arm)s",
                    {"arm": args.arm})
        cur.execute(
            """
            INSERT INTO DT_STUDY.DRIFT.ref_sample_emb
            SELECT arm, record_id, emb
            FROM (
                SELECT arm, record_id, emb,
                       ROW_NUMBER() OVER (PARTITION BY arm ORDER BY HASH(record_id)) rn
                FROM DT_STUDY.DRIFT.embeddings
                WHERE arm = %(arm)s AND day_index BETWEEN 15 AND 34
            )
            WHERE rn <= 2000
            """,
            {"arm": args.arm},
        )
        conn.commit()
        print(f"\ndone. arm={args.arm}: {n_sample_rows:,} rows embedded "
              f"into daily_sample/embeddings, ref_sample_emb populated.")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
