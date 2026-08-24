#!/usr/bin/env python3
"""
Loader for the municipal-311 drift-detection corpus.

Reads the day_*.jsonl files written by Corpus_Generator.py - one JSON object
per line with fields record_id, arm, day_index, text, true_label - and loads
them into DT_STUDY.DRIFT.DAILY_RECORDS.

Usage
    export SNOWFLAKE_PASSWORD=...
    pip install snowflake-connector-python

    Account/user/role/warehouse/database/schema are read from a named
    profile in ~/.snowflake/connections.toml via --connection <name>
    (see study_v1 / study_v2). Password/authenticator always come from the
    environment, never from that file. Without --connection, account/user
    fall back to the SNOWFLAKE_ACCOUNT / SNOWFLAKE_USER env vars instead.

    python Loader.py --corpus-dir ./corpus --arm BASE --check                     # validate, load nothing
    python Loader.py --corpus-dir ./corpus --arm BASE --connection study_v1       # load one arm
    python Loader.py --corpus-dir ./corpus --arm all  --connection study_v2       # load every arm found
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

DATABASE = "DT_STUDY"
SCHEMA = "DRIFT"
TABLE = "DAILY_RECORDS"
STAGE = "DRIFT_LOAD_STAGE"
WAREHOUSE = "DT_TESTBED_WH"

COLUMNS = ["record_id", "arm", "day_index", "text", "true_label"]

ARMS = ["BASE", "A", "B", "C", "D", "E"]


# --------------------------------------------------------------------------
# local discovery and validation
# --------------------------------------------------------------------------

def _find_day_files(corpus_dir: Path, arm: str) -> list[Path]:
    if arm == "all":
        paths = sorted(corpus_dir.glob("*/day_*.jsonl"))
    else:
        paths = sorted((corpus_dir / arm).glob("day_*.jsonl"))
    if not paths:
        raise SystemExit(f"no day_*.jsonl files found under {corpus_dir} for arm={arm}")
    return paths


def _validate(paths: list[Path]) -> int:
    """Parse every line locally and confirm the required fields are present
    before anything is sent to Snowflake."""
    required = set(COLUMNS)
    total = 0
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"FATAL: invalid JSON at {p}:{lineno}: {e}")
                missing = required - obj.keys()
                if missing:
                    raise SystemExit(
                        f"FATAL: {p}:{lineno} missing fields {sorted(missing)}"
                    )
                total += 1
    print(f"  validated {total:,} records across {len(paths):,} files", file=sys.stderr)
    return total


# --------------------------------------------------------------------------
# snowflake load
# --------------------------------------------------------------------------

DDL_STAGE = f"""
CREATE STAGE IF NOT EXISTS {DATABASE}.{SCHEMA}.{STAGE}
  FILE_FORMAT = (
    TYPE = JSON
    COMPRESSION = AUTO
  )
"""

COPY_SQL = f"""
COPY INTO {DATABASE}.{SCHEMA}.{TABLE} ({', '.join(COLUMNS)})
FROM (
    SELECT
        $1:record_id::STRING,
        $1:arm::STRING,
        $1:day_index::INT,
        $1:text::STRING,
        $1:true_label::STRING
    FROM @{DATABASE}.{SCHEMA}.{STAGE}/{{name}}
)
ON_ERROR = ABORT_STATEMENT
PURGE = FALSE
"""


def _connection_profile(name: str) -> dict:
    """Look up a named profile in ~/.snowflake/connections.toml (or
    $SNOWFLAKE_HOME/connections.toml). Only non-secret fields belong there -
    account, user, role, warehouse, database, schema. Password/authenticator
    always come from the environment, never from this file."""
    import tomllib

    home = Path(os.environ.get("SNOWFLAKE_HOME", Path.home() / ".snowflake"))
    path = home / "connections.toml"
    if not path.exists():
        raise SystemExit(f"connections.toml not found at {path}")
    with path.open("rb") as fh:
        profiles = tomllib.load(fh)
    if name not in profiles:
        raise SystemExit(
            f"connection '{name}' not found in {path} "
            f"(available: {', '.join(sorted(profiles)) or 'none'})"
        )
    return profiles[name]


def connect(connection_name: str | None = None):
    import snowflake.connector

    profile = _connection_profile(connection_name) if connection_name else {}

    account = profile.get("account") or os.environ.get("SNOWFLAKE_ACCOUNT")
    user = profile.get("user") or os.environ.get("SNOWFLAKE_USER")

    missing = [n for n, v in (("account", account), ("user", user)) if not v]
    if missing:
        raise SystemExit(
            f"missing {', '.join(missing)}: set via SNOWFLAKE_ACCOUNT/SNOWFLAKE_USER "
            f"env vars, or add them to the '{connection_name}' profile in "
            "connections.toml"
        )

    return snowflake.connector.connect(
        account=account,
        user=user,
        password=os.environ.get("SNOWFLAKE_PASSWORD") or profile.get("password"),
        authenticator=os.environ.get("SNOWFLAKE_AUTHENTICATOR", "snowflake"),
        role=profile.get("role") or os.environ.get("SNOWFLAKE_ROLE"),
        warehouse=profile.get("warehouse", WAREHOUSE),
        database=profile.get("database", DATABASE),
        schema=profile.get("schema", SCHEMA),
        client_session_keep_alive=True,
    )


def load(paths: list[Path], pause_seconds: float, connection_name: str | None) -> None:
    import time

    conn = connect(connection_name)
    cur = conn.cursor()
    try:
        cur.execute(f"USE WAREHOUSE {WAREHOUSE}")
        cur.execute(DDL_STAGE)

        for i, p in enumerate(paths, 1):
            cur.execute(
                f"PUT 'file://{p.resolve().as_posix()}' @{DATABASE}.{SCHEMA}.{STAGE} "
                "AUTO_COMPRESS = TRUE OVERWRITE = TRUE"
            )
            cur.execute(COPY_SQL.format(name=p.name + ".gz"))
            if i % 10 == 0 or i == len(paths):
                print(f"  loaded {i:,}/{len(paths):,} files", file=sys.stderr)
            if pause_seconds:
                time.sleep(pause_seconds)

        cur.execute(f"""
            SELECT COUNT(*)             AS total_rows,
                   COUNT(DISTINCT arm)  AS distinct_arms,
                   MIN(day_index)       AS min_day,
                   MAX(day_index)       AS max_day
            FROM {DATABASE}.{SCHEMA}.{TABLE}
        """)
        total, arms, min_day, max_day = cur.fetchone()
        print("\nloaded table summary")
        print(f"  total rows     {total:,}")
        print(f"  distinct arms  {arms}")
        print(f"  day range      {min_day}-{max_day}")
    finally:
        cur.close()
        conn.close()


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path, default=Path("./corpus"),
                    help="root directory passed as --out to Corpus_Generator.py")
    ap.add_argument("--arm", default="BASE", choices=ARMS + ["all"],
                    help="arm subdirectory to load, or 'all' for every arm found")
    ap.add_argument("--check", action="store_true",
                    help="validate the jsonl files locally, do not connect to Snowflake")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds to sleep between file loads, to spread "
                         "arrivals across refresh cycles")
    ap.add_argument("--connection", default=None,
                    help="named profile from ~/.snowflake/connections.toml "
                         "(account/user/role/warehouse/database/schema); "
                         "falls back to SNOWFLAKE_* env vars if omitted")
    args = ap.parse_args()

    paths = _find_day_files(args.corpus_dir, args.arm)
    print(f"arm={args.arm}: {len(paths):,} day files found under {args.corpus_dir}",
          file=sys.stderr)
    _validate(paths)

    if args.check:
        print("check complete, nothing loaded")
        return

    load(paths, args.pause, args.connection)


if __name__ == "__main__":
    main()
