-- ============================================================================
-- CLASSIFICATION WITH BARE LABELS (RENAMED PAIR)
-- ============================================================================
-- REVERTED (verified against SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY, 2026-11-01):
-- this file previously called AI_CLASSIFY with per-label descriptions, but
-- that variant was never the one that produced the live tables. History shows
-- a 2026-08-21 debugging session that (a) tried descriptions once, as a
-- 10-row ad hoc test never written to a permanent table, then (b) settled on
-- bare labels with the confusable pair renamed below, and that is the exact
-- query that built the predictions_calibration table currently in Snowflake
-- (confirmed empirically: DISTINCT predicted_label on that table returns
-- 'noise_disturbance_non_construction' / 'construction_site_disturbance', not
-- 'noise_complaint' / 'construction_complaint'). Using descriptions here was
-- an error introduced after that session, not a verified improvement.
--
-- LABEL MAPPING - read before computing any accuracy/true_label comparison:
-- predicted_label is in a DIFFERENT vocabulary than true_label for these two
-- classes only. predicted_label = 'noise_disturbance_non_construction' means
-- true_label-space 'noise_complaint'; predicted_label =
-- 'construction_site_disturbance' means true_label-space
-- 'construction_complaint'. All eight other labels are identical strings in
-- both columns. Any query comparing predicted_label = true_label directly
-- (e.g. accuracy_daily) must map predicted_label back first or it will count
-- every correct prediction on these two classes as wrong. See
-- python/run_classification.py's LABEL_MAP for the mapping used elsewhere.
--
-- This must also be disclosed in the manuscript (Section III-C): a reviewer
-- re-running this repo against the README's original label strings will not
-- reproduce Table 4, because the published numbers came from this renamed set.
--
-- ARM FIX (reviewer-response-r1): previously hardcoded to arm = 'BASE' on the
-- theory that days 15-34 are byte-identical across arms, so one calibration
-- run could be "shared across arms" (README.md §2). That holds only when every
-- arm is generated from the SAME seed. Once replicate seeds are introduced
-- (gen_replicates.py), each seed's days 15-34 differ from every other seed's,
-- so this must be re-run once per corpus REPLICATE (i.e. once per distinct
-- seed/label such as BASE_s01, BASE_s02, ...), not once globally. Substitute
-- the target arm/label below, same convention as sql/drift_signals.sql.
--
-- IDEMPOTENCY FIX (reviewer-response-r1): the table used to be CREATE OR
-- REPLACE, which is fine for a single global run but would silently delete
-- every other replicate's calibration rows if re-run per seed/label as above.
-- Create the (empty) table once, then DELETE+INSERT per arm/label so each
-- replicate's run only touches its own rows.
-- ============================================================================

CREATE TABLE IF NOT EXISTS DT_STUDY.DRIFT.predictions_calibration (
    arm                  STRING,
    day_index            INT,
    record_id            STRING,
    true_label           STRING,
    predicted_label      STRING,
    predicted_label_norm STRING
);

DELETE FROM DT_STUDY.DRIFT.predictions_calibration WHERE arm = '<ARM>';

-- predicted_label_norm maps the two renamed labels back into true_label's
-- vocabulary; this exact CASE reproduces the PREDICTED_LABEL_NORM column
-- already live on predictions_calibration/predictions_arm_b/predictions_arm_d
-- (verified: DISTINCT predicted_label/predicted_label_norm pairs on the live
-- table show only these two labels differ from their true_label spelling).
-- Every downstream comparison against true_label (accuracy, any future
-- per-class breakdown) must use predicted_label_norm, not predicted_label -
-- S7a's chi-square already does, via ref_priors joining on this column.
INSERT INTO DT_STUDY.DRIFT.predictions_calibration
SELECT arm, day_index, record_id, true_label, predicted_label,
       CASE predicted_label
           WHEN 'noise_disturbance_non_construction' THEN 'noise_complaint'
           WHEN 'construction_site_disturbance' THEN 'construction_complaint'
           ELSE predicted_label
       END AS predicted_label_norm
FROM (
    SELECT arm, day_index, record_id, true_label,
           AI_CLASSIFY(
               text,
               ['pothole_road','streetlight_outage',
                'noise_disturbance_non_construction',
                'construction_site_disturbance',
                'illegal_dumping','overflowing_bin','graffiti',
                'park_maintenance','water_leak','stray_animal']
           ):labels[0]::VARCHAR AS predicted_label
    FROM DT_STUDY.DRIFT.daily_records
    WHERE arm = '<ARM>' AND day_index BETWEEN 15 AND 34
);

-- ============================================================================
-- CORRECTION (2026-11-01, same day): the block below this comment, in an
-- earlier version of this file, duplicated BASE's rows into
-- predictions_calibration under arm='B'/'D'. That was the wrong table and
-- didn't fix anything real - verified live:
--   * ref_priors (drift_signals.sql) reads FROM DT_STUDY.DRIFT.predictions
--     WHERE day_index BETWEEN 15 AND 34, never from predictions_calibration.
--   * predictions_calibration is written only by the INSERT above (this
--     file) and read by nothing in the pipeline - its sole purpose is
--     mirroring the legacy S7A_CALIBRATION flat table for a manual
--     cross-check, and that table has no arm column (20 rows, days 15-34,
--     arm-agnostic), so it never needed B/D-specific rows either.
--   * `predictions` already has native calibration rows for arm B/D live
--     (40,000 each) - see the migration note below for how they got there.
-- The duplication, if it had been needed, belonged on `predictions`, not
-- `predictions_calibration`. It has been removed rather than corrected
-- in place, because it turned out not to be needed going forward at all -
-- see below.
--
-- ONE-TIME MIGRATION REPAIR, NOT A RECURRING FIX (historical note):
-- Early in the reviewer-response-r1 validation gate, S7a (chi-square
-- class-prior shift) came out several orders of magnitude too large for
-- arm='B'/'D'. Root cause: the legacy flat tables migrated into the new
-- unified `predictions` table (predictions_arm_b / predictions_arm_d) only
-- ever held the measurement window (35-89) for those arms - their
-- calibration window (15-34) was never captured in a retained table. So
-- `predictions` had zero calibration-window rows for arm B/D immediately
-- after migration, ref_priors' lookup for those arms missed on every
-- label/day, and every term fell back to COALESCE(..., 1e-6), exploding the
-- statistic. This was repaired ONCE, live, by copying BASE's existing
-- `predictions` rows for day_index BETWEEN 15 AND 34 under arm labels 'B'
-- and 'D' (valid because B/D's days 15-34 are byte-identical to BASE's -
-- single-seed design, README.md Section 2) - zero additional Cortex cost.
--
-- This is NOT a block to re-run as part of this script, and it is NOT
-- needed for a from-scratch reproduction: Section 4 of drift_signals.sql
-- classifies `day_index >= 15` per arm, so a fresh run for any arm already
-- produces its own complete calibration + measurement rows in `predictions`
-- in one pass, with no gap to patch. It was only the one-time legacy-table
-- migration that created the gap this repaired.
