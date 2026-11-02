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
-- CALIBRATION-SHARING FIX FOR SINGLE-SEED ARMS (2026-11-01)
-- ============================================================================
-- Bug found during the reviewer-response-r1 validation gate: S7a (chi-square
-- class-prior shift) was coming out several orders of magnitude too large for
-- arm='B' and arm='D' (hundreds of millions instead of the expected ~10-20).
-- Root cause: ref_priors (drift_signals.sql) partitions by arm, but the block
-- above only ever populates predictions_calibration for whichever single
-- '<ARM>' it's run against. For B and D - which are NOT their own replicate
-- seed, just BASE's corpus with drift injected after day 34 - no calibration
-- rows existed under arm='B'/arm='D' at all, so every ref_priors lookup for
-- those arms missed and fell back to COALESCE(..., 1e-6) on every label/day,
-- which is what exploded the statistic.
--
-- Fix: since B/D's days 15-34 are byte-identical to BASE's (single-seed
-- design, confirmed in README.md Section 2), their calibration is BASE's
-- calibration - copy it under each arm label rather than re-running
-- AI_CLASSIFY. This costs zero additional Cortex credits. Idempotent per arm
-- via DELETE+INSERT, same convention as above.
--
-- This block is NOT needed once a given arm is a true independent replicate
-- (e.g. B_s01 with its own seed) - at that point run the AI_CLASSIFY block
-- above directly for that arm/label instead, since its days 15-34 are no
-- longer identical to BASE's.
DELETE FROM DT_STUDY.DRIFT.predictions_calibration WHERE arm = 'B';
INSERT INTO DT_STUDY.DRIFT.predictions_calibration
SELECT 'B' AS arm, day_index, record_id, true_label, predicted_label, predicted_label_norm
FROM DT_STUDY.DRIFT.predictions_calibration
WHERE arm = 'BASE';

DELETE FROM DT_STUDY.DRIFT.predictions_calibration WHERE arm = 'D';
INSERT INTO DT_STUDY.DRIFT.predictions_calibration
SELECT 'D' AS arm, day_index, record_id, true_label, predicted_label, predicted_label_norm
FROM DT_STUDY.DRIFT.predictions_calibration
WHERE arm = 'BASE';
