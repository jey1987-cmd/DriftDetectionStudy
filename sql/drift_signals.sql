-- ============================================================================
-- DRIFT DETECTION STUDY - SIGNAL IMPLEMENTATIONS IN SNOWFLAKE
-- ============================================================================
-- Every signal produces exactly one row per (day_index, arm, signal, strategy)
-- into RESULT.signal_daily.  Firing is then decided once, uniformly, in §7.
--
-- Conventions used throughout:
--   REF_DAYS  = days 15-34  (clean inference window)
--   CUR_DAY   = the day being scored
--   All statistics are "distance from reference" - larger means more drift.
-- ============================================================================


-- ============================================================================
-- 0. SETUP
-- ============================================================================
-- NAMING FIX (reviewer-response-r1): this file previously created and queried
-- a separate DRIFT_STUDY.RAW.daily_records table on a DRIFT_WH warehouse, while
-- README.md, Loader.py, and classify_calibration.sql all load into
-- DT_STUDY.DRIFT.daily_records on warehouse DT_TESTBED_WH (confirmed against
-- the actual account setup script, Account2.sql). The two names were never the
-- same table, so running this file's Tier 1/2/3 blocks against data loaded per
-- the README would have silently queried an empty table. DT_STUDY/DT_TESTBED_WH
-- is the one actually populated; this file is corrected to match it. FEAT and
-- RESULT, which this file created and nothing else references, now live under
-- DT_STUDY too, so the whole pipeline is one database.

CREATE SCHEMA   IF NOT EXISTS DT_STUDY.FEAT;
CREATE SCHEMA   IF NOT EXISTS DT_STUDY.RESULT;

USE WAREHOUSE DT_TESTBED_WH;
USE SCHEMA DT_STUDY.DRIFT;

-- daily_records itself is created once via README.md §3.2 (or Account2.sql) and
-- loaded by Loader.py - it is NOT recreated here, since doing so would silently
-- truncate whatever has already been loaded.

CREATE OR REPLACE TABLE DT_STUDY.RESULT.signal_daily (
    arm         STRING,
    day_index   INT,
    signal      STRING,
    strategy    STRING,          -- 'FIXED' | 'SLIDING'
    statistic   FLOAT
);


-- ============================================================================
-- 1. TOKENISATION  (shared by S1, S2, S3)
-- ============================================================================
-- Materialise rather than re-derive: tokenisation is the single most repeated
-- operation in the study and recomputing it per signal triples Tier 1 cost.

CREATE OR REPLACE TABLE DT_STUDY.FEAT.tokens AS
SELECT r.arm,
       r.day_index,
       r.record_id,
       LOWER(t.value::STRING) AS token
FROM DT_STUDY.DRIFT.daily_records r,
     LATERAL SPLIT_TO_TABLE(
         REGEXP_REPLACE(LOWER(r.text), '[^a-z0-9 ]', ' '), ' ') t
WHERE t.value <> '';

CREATE OR REPLACE TABLE DT_STUDY.FEAT.record_lengths AS
SELECT arm, day_index, record_id, COUNT(*) AS n_tokens
FROM DT_STUDY.FEAT.tokens
GROUP BY 1,2,3;


-- ============================================================================
-- 2. TIER 1 - STRUCTURAL SIGNALS  (pure SQL, no model calls)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- S1  LENGTH DISTRIBUTION - two-sample Kolmogorov-Smirnov statistic
-- ----------------------------------------------------------------------------
-- KS = max |CDF_ref(x) - CDF_day(x)| over the union of observed lengths.
-- Both CDFs must be evaluated on the SAME grid, hence the FULL OUTER JOIN.

CREATE OR REPLACE FUNCTION DT_STUDY.FEAT.ks_length(arm_in STRING, day_in INT)
RETURNS FLOAT
AS
$$
WITH ref AS (
    SELECT n_tokens, COUNT(*) AS c
    FROM DT_STUDY.FEAT.record_lengths
    WHERE arm = arm_in AND day_index BETWEEN 15 AND 34
    GROUP BY 1
),
cur AS (
    SELECT n_tokens, COUNT(*) AS c
    FROM DT_STUDY.FEAT.record_lengths
    WHERE arm = arm_in AND day_index = day_in
    GROUP BY 1
),
grid AS (
    SELECT COALESCE(r.n_tokens, c.n_tokens) AS n_tokens,
           COALESCE(r.c, 0) AS rc,
           COALESCE(c.c, 0) AS cc
    FROM ref r FULL OUTER JOIN cur c ON r.n_tokens = c.n_tokens
),
tot AS (SELECT SUM(rc) AS rt, SUM(cc) AS ct FROM grid),
cdf AS (
    SELECT SUM(rc) OVER (ORDER BY n_tokens) / NULLIF(tot.rt,0) AS ref_cdf,
           SUM(cc) OVER (ORDER BY n_tokens) / NULLIF(tot.ct,0) AS cur_cdf
    FROM grid, tot
)
SELECT MAX(ABS(ref_cdf - cur_cdf)) FROM cdf
$$;

-- ----------------------------------------------------------------------------
-- S2  VOCABULARY NOVELTY - fraction of today's tokens unseen in reference
-- ----------------------------------------------------------------------------
-- Computed over token OCCURRENCES, not distinct types: a single new word used
-- 500 times is a bigger distribution change than 5 words used once each.

CREATE OR REPLACE TABLE DT_STUDY.FEAT.ref_vocab AS
SELECT DISTINCT arm, token
FROM DT_STUDY.FEAT.tokens
WHERE day_index BETWEEN 15 AND 34;

CREATE OR REPLACE VIEW DT_STUDY.FEAT.s2_vocab_novelty AS
SELECT t.arm,
       t.day_index,
       COUNT_IF(v.token IS NULL) / COUNT(*)::FLOAT AS statistic
FROM DT_STUDY.FEAT.tokens t
LEFT JOIN DT_STUDY.FEAT.ref_vocab v
       ON v.arm = t.arm AND v.token = t.token
GROUP BY 1,2;

-- ----------------------------------------------------------------------------
-- S3  TYPE-TOKEN RATIO - absolute deviation from reference lexical diversity
-- ----------------------------------------------------------------------------
-- TTR is sensitive to sample size, so it is computed per DAY (equal volume by
-- design) and the reference is the MEAN of daily TTRs, not a pooled TTR.

CREATE OR REPLACE VIEW DT_STUDY.FEAT.daily_ttr AS
SELECT arm, day_index,
       COUNT(DISTINCT token) / COUNT(*)::FLOAT AS ttr
FROM DT_STUDY.FEAT.tokens
GROUP BY 1,2;

CREATE OR REPLACE VIEW DT_STUDY.FEAT.s3_ttr_shift AS
WITH ref AS (
    SELECT arm, AVG(ttr) AS ref_ttr
    FROM DT_STUDY.FEAT.daily_ttr
    WHERE day_index BETWEEN 15 AND 34
    GROUP BY 1
)
SELECT d.arm, d.day_index, ABS(d.ttr - r.ref_ttr) AS statistic
FROM DT_STUDY.FEAT.daily_ttr d JOIN ref r ON r.arm = d.arm;


-- ============================================================================
-- 3. TIER 2 - EMBEDDING SIGNALS  (metered Cortex calls)
-- ============================================================================
-- COST CONTROL: embed a FIXED sample of k=200 records per day, never the full
-- 2,000.  The sample must be deterministic so the study is reproducible, and it
-- must be the same size every day or the statistics are not comparable.

CREATE OR REPLACE TABLE DT_STUDY.FEAT.daily_sample AS
SELECT arm, day_index, record_id, text
FROM (
    SELECT arm, day_index, record_id, text,
           ROW_NUMBER() OVER (
               PARTITION BY arm, day_index
               ORDER BY HASH(record_id)          -- deterministic, not RANDOM()
           ) AS rn
    FROM DT_STUDY.DRIFT.daily_records
)
WHERE rn <= 200;

-- Embed once, reuse for S4/S5/S6.  Stored as ARRAY so element-wise aggregation
-- is possible; Snowflake has no AVG() over VECTOR.
CREATE OR REPLACE TABLE DT_STUDY.FEAT.embeddings AS
SELECT arm, day_index, record_id,
       SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', text) AS emb
FROM DT_STUDY.FEAT.daily_sample;

-- Element-wise long form, used for centroid arithmetic
CREATE OR REPLACE VIEW DT_STUDY.FEAT.emb_elements AS
SELECT arm, day_index, record_id,
       f.index AS pos, f.value::FLOAT AS v
FROM DT_STUDY.FEAT.embeddings,
     LATERAL FLATTEN(input => emb) f;

-- ----------------------------------------------------------------------------
-- S4  CENTROID DISPLACEMENT - 1 - cosine(reference centroid, day centroid)
-- ----------------------------------------------------------------------------

CREATE OR REPLACE VIEW DT_STUDY.FEAT.ref_centroid AS
SELECT arm, pos, AVG(v) AS v
FROM DT_STUDY.FEAT.emb_elements
WHERE day_index BETWEEN 15 AND 34
GROUP BY 1,2;

CREATE OR REPLACE VIEW DT_STUDY.FEAT.day_centroid AS
SELECT arm, day_index, pos, AVG(v) AS v
FROM DT_STUDY.FEAT.emb_elements
GROUP BY 1,2,3;

CREATE OR REPLACE VIEW DT_STUDY.FEAT.s4_centroid_shift AS
SELECT d.arm, d.day_index,
       1 - ( SUM(d.v * r.v)
             / NULLIF(SQRT(SUM(d.v*d.v)) * SQRT(SUM(r.v*r.v)), 0) ) AS statistic
FROM DT_STUDY.FEAT.day_centroid d
JOIN DT_STUDY.FEAT.ref_centroid r
  ON r.arm = d.arm AND r.pos = d.pos
GROUP BY 1,2;

-- ----------------------------------------------------------------------------
-- S6  NEAREST-NEIGHBOUR DISTANCE
-- ----------------------------------------------------------------------------
-- Catches a novel cluster that does not move the centroid (arm C), which is
-- precisely the case S4 is blind to.  O(k * |ref|) pairwise - the reference is
-- capped at 2,000 vectors to bound cost.

CREATE OR REPLACE TABLE DT_STUDY.FEAT.ref_sample_emb AS
SELECT arm, record_id, emb
FROM (
    SELECT arm, record_id, emb,
           ROW_NUMBER() OVER (PARTITION BY arm ORDER BY HASH(record_id)) rn
    FROM DT_STUDY.FEAT.embeddings
    WHERE day_index BETWEEN 15 AND 34
)
WHERE rn <= 2000;

CREATE OR REPLACE VIEW DT_STUDY.FEAT.s6_nn_distance AS
WITH pair AS (
    SELECT d.arm, d.day_index, d.record_id,
           MAX(VECTOR_COSINE_SIMILARITY(d.emb, r.emb)) AS best_sim
    FROM DT_STUDY.FEAT.embeddings d
    JOIN DT_STUDY.FEAT.ref_sample_emb r
      ON r.arm = d.arm
     AND r.record_id <> d.record_id  -- leave-one-out: exclude self-match.
         -- Without this, a day-d record that is ALSO in the reference
         -- sample (true for ~half of reference-window days 15-34, since
         -- ref_sample_emb is itself drawn from that window) scores a
         -- spurious cosine similarity of 1.0 against itself, deflating
         -- the reference window's calibration mean/stdev. Measurement-
         -- window days (35+) were never affected, since they cannot
         -- appear in ref_sample_emb by construction -- the artifact was
         -- confined to the calibration window and produced a false
         -- discontinuity at the reference/measurement boundary. See
         -- response to reviewers, Reviewer 1 Concern #4 / Reviewer 2
         -- Concern #3.
    GROUP BY 1,2,3
)
SELECT arm, day_index, AVG(1 - best_sim) AS statistic
FROM pair GROUP BY 1,2;

-- ----------------------------------------------------------------------------
-- S5  MAXIMUM MEAN DISCREPANCY  (linear-kernel form)
-- ----------------------------------------------------------------------------
-- Linear-kernel MMD^2 reduces to the squared L2 distance between mean
-- embeddings - computable in O(d) rather than the O(n^2) an RBF kernel needs.
-- If the RBF form is wanted, it requires a pairwise self-join within each day
-- as well as across, at ~k^2 = 40,000 similarity evaluations per day per arm.
-- Report which form was used; they are not interchangeable in sensitivity.

CREATE OR REPLACE VIEW DT_STUDY.FEAT.s5_mmd_linear AS
SELECT d.arm, d.day_index,
       SQRT(SUM(POWER(d.v - r.v, 2))) AS statistic
FROM DT_STUDY.FEAT.day_centroid d
JOIN DT_STUDY.FEAT.ref_centroid r
  ON r.arm = d.arm AND r.pos = d.pos
GROUP BY 1,2;


-- ============================================================================
-- 4. CLASSIFICATION  (produces the input to Tier 3)
-- ============================================================================
-- Runs on ALL records for the day, not the sample - Tier 3 is "free if you are
-- already inferring", and that framing only holds at full volume.

-- FUNCTION/LABEL FIX (reviewer-response-r1): this block previously called the
-- legacy SNOWFLAKE.CORTEX.CLASSIFY_TEXT against a placeholder 8-label array
-- ('class_a'..'class_h') that matches no class in the real taxonomy (see
-- generate_311_corpus.py CLASSES / classify_calibration.sql). DESIGN.md's own
-- "API currency" section says the study uses AI_CLASSIFY, not the deprecated
-- CLASSIFY_TEXT. Corrected to AI_CLASSIFY with the real bare-label array, per
-- README.md §3.3/§3.6. Arm C's measurement window needs the 11-label variant
-- adding 'sidewalk_obstruction' (the novel class) - build that as a second
-- table for arm='C' specifically, as classify_calibration.sql's header notes;
-- do not add it to this shared array or every other arm's predictions change.
CREATE OR REPLACE TABLE DT_STUDY.FEAT.predictions AS
SELECT arm, day_index, record_id, true_label,
       AI_CLASSIFY(
           text,
           ['pothole_road','streetlight_outage','noise_complaint',
            'construction_complaint','illegal_dumping','overflowing_bin',
            'graffiti','park_maintenance','water_leak','stray_animal']
       ):labels[0]::VARCHAR AS predicted_label
FROM DT_STUDY.DRIFT.daily_records
WHERE day_index >= 15;

-- NOTE ON CONFIDENCE (S7b): AI_CLASSIFY does not return a calibrated score.
-- To run S7b you need a confidence proxy.  Two workable options:
--   (a) fine-tune with AI_COMPLETE and request a self-reported confidence in
--       the prompt - cheap but poorly calibrated, and that must be stated;
--   (b) use max cosine similarity between the record embedding and the set of
--       class-centroid embeddings, which is well defined and reproducible.
-- Option (b) is recommended.  Decide before Phase 2 - it changes what you must
-- record during the clean window, and it cannot be reconstructed later.


-- ============================================================================
-- 5. TIER 3 - PREDICTION SIGNALS
-- ============================================================================

-- ----------------------------------------------------------------------------
-- S7a  CLASS PRIOR SHIFT - Pearson chi-square against reference priors
-- ----------------------------------------------------------------------------
-- ZERO-FREQUENCY BIN (reviewer-response-r1): Arm C's novel class
-- ('sidewalk_obstruction') is injected only from day 35 onward, so it never
-- appears in ref_priors, which is built from days 15-34 only. On a
-- measurement-window day where the classifier predicts that label, the LEFT
-- JOIN to ref_priors produces r.p = NULL for that bin, not zero - NULL
-- propagates through POWER()/NULLIF() and would either divide by zero
-- (NULLIF(0,0) -> NULL) or drop that term from the SUM(), i.e. silently
-- discard exactly the bin carrying the arm-C signal. COALESCE(r.p, 1e-6)
-- is the fix already in place: it substitutes a small constant expected
-- probability instead of leaving the reference count undefined, so the
-- novel class contributes a (large) finite term to the chi-square statistic
-- rather than vanishing from it or raising a divide-by-zero.
--
-- Caveat to disclose: 1e-6 is a fixed constant, not a formally derived
-- smoothing parameter - it was chosen because at RECORDS_PER_DAY=2000,
-- t.n * 1e-6 ~ 0.002 is small enough not to distort any class that DOES
-- have reference support, while still being nonzero. A more principled
-- alternative is additive (Laplace/Dirichlet) smoothing applied to every
-- class uniformly - p_smoothed = (ref_count + alpha) / (ref_total + alpha*k)
-- for k = number of distinct predicted labels including the novel one -
-- which replaces the arbitrary constant with a parameter that has a
-- standard interpretation and degrades gracefully as k grows. See
-- s7a_prior_shift_laplace below for a side-by-side sensitivity check; it is
-- NOT used by signal_daily/detection_summary, so it does not change any
-- already-reported number, but it is evidence the published result is not
-- an artifact of the specific epsilon chosen.

CREATE OR REPLACE VIEW DT_STUDY.FEAT.ref_priors AS
SELECT arm, predicted_label,
       COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY arm)::FLOAT AS p
FROM DT_STUDY.FEAT.predictions
WHERE day_index BETWEEN 15 AND 34
GROUP BY 1,2;

CREATE OR REPLACE VIEW DT_STUDY.FEAT.s7a_prior_shift AS
WITH obs AS (
    SELECT arm, day_index, predicted_label, COUNT(*) AS o
    FROM DT_STUDY.FEAT.predictions
    GROUP BY 1,2,3
),
tot AS (SELECT arm, day_index, SUM(o) AS n FROM obs GROUP BY 1,2)
SELECT o.arm, o.day_index,
       SUM( POWER(o.o - t.n * COALESCE(r.p, 1e-6), 2)
            / NULLIF(t.n * COALESCE(r.p, 1e-6), 0) ) AS statistic
FROM obs o
JOIN tot t  ON t.arm = o.arm AND t.day_index = o.day_index
LEFT JOIN DT_STUDY.FEAT.ref_priors r
       ON r.arm = o.arm AND r.predicted_label = o.predicted_label
GROUP BY 1,2;
-- COALESCE to 1e-6 handles a class predicted on day d that never appeared in
-- the reference - without it the term divides by zero and the day is lost,
-- which is exactly the arm-C case the signal should be catching.

-- ----------------------------------------------------------------------------
-- S7a SENSITIVITY CHECK - Laplace-smoothed reference priors
-- ----------------------------------------------------------------------------
-- Same statistic, but every label's reference probability (including labels
-- absent from the reference window entirely) is smoothed by alpha=1 over the
-- observed label vocabulary, instead of patching only the absent ones with a
-- fixed constant. Compare s7a_prior_shift vs s7a_prior_shift_laplace on the
-- same arm/day - if they agree closely, the published S7a numbers are not an
-- artifact of the 1e-6 choice.

CREATE OR REPLACE VIEW DT_STUDY.FEAT.ref_priors_laplace AS
WITH vocab AS (
    SELECT DISTINCT arm, predicted_label FROM DT_STUDY.FEAT.predictions
),
ref_counts AS (
    SELECT arm, predicted_label, COUNT(*) AS c
    FROM DT_STUDY.FEAT.predictions
    WHERE day_index BETWEEN 15 AND 34
    GROUP BY 1,2
),
k AS (SELECT arm, COUNT(DISTINCT predicted_label) AS n_labels
      FROM DT_STUDY.FEAT.predictions GROUP BY 1),
ref_total AS (SELECT arm, SUM(c) AS n FROM ref_counts GROUP BY 1)
SELECT v.arm, v.predicted_label,
       (COALESCE(rc.c, 0) + 1) / (rt.n + k.n_labels)::FLOAT AS p
FROM vocab v
JOIN k        ON k.arm = v.arm
JOIN ref_total rt ON rt.arm = v.arm
LEFT JOIN ref_counts rc ON rc.arm = v.arm AND rc.predicted_label = v.predicted_label;

CREATE OR REPLACE VIEW DT_STUDY.FEAT.s7a_prior_shift_laplace AS
WITH obs AS (
    SELECT arm, day_index, predicted_label, COUNT(*) AS o
    FROM DT_STUDY.FEAT.predictions
    GROUP BY 1,2,3
),
tot AS (SELECT arm, day_index, SUM(o) AS n FROM obs GROUP BY 1,2)
SELECT o.arm, o.day_index,
       SUM( POWER(o.o - t.n * r.p, 2) / (t.n * r.p) ) AS statistic
FROM obs o
JOIN tot t ON t.arm = o.arm AND t.day_index = o.day_index
JOIN DT_STUDY.FEAT.ref_priors_laplace r
  ON r.arm = o.arm AND r.predicted_label = o.predicted_label
GROUP BY 1,2;


-- ============================================================================
-- 5b. POPULATE RESULT.signal_daily  (reviewer-response-r1: this was missing)
-- ============================================================================
-- signal_fired and detection_summary below both read FROM signal_daily, but
-- nothing in this file ever inserted the Tier 1/2/3 views' output into it -
-- every view above is fully specified, this section only wires it into the
-- results table, no new statistic logic. Re-runnable: each block deletes its
-- own (signal, strategy) rows first, matching this file's existing pattern of
-- fully recomputing derived tables from whatever is currently in
-- daily_records (see tokens/daily_sample/embeddings/predictions above), so
-- re-running after loading another arm or replicate is safe, not additive.
--
-- STRATEGY GAP: DESIGN.md §4.3 calls for every signal under both FIXED and
-- SLIDING (trailing 7-day) reference strategies. Only FIXED is implemented
-- anywhere in this file - none of the views above compute a sliding
-- reference. Only 'FIXED' rows are inserted below; add the SLIDING views
-- first if the response to reviewers needs that comparison.

DELETE FROM DT_STUDY.RESULT.signal_daily WHERE signal = 'S1' AND strategy = 'FIXED';
INSERT INTO DT_STUDY.RESULT.signal_daily
SELECT arm, day_index, 'S1', 'FIXED', DT_STUDY.FEAT.ks_length(arm, day_index)
FROM (SELECT DISTINCT arm, day_index FROM DT_STUDY.FEAT.record_lengths);

DELETE FROM DT_STUDY.RESULT.signal_daily WHERE signal = 'S2' AND strategy = 'FIXED';
INSERT INTO DT_STUDY.RESULT.signal_daily
SELECT arm, day_index, 'S2', 'FIXED', statistic FROM DT_STUDY.FEAT.s2_vocab_novelty;

DELETE FROM DT_STUDY.RESULT.signal_daily WHERE signal = 'S3' AND strategy = 'FIXED';
INSERT INTO DT_STUDY.RESULT.signal_daily
SELECT arm, day_index, 'S3', 'FIXED', statistic FROM DT_STUDY.FEAT.s3_ttr_shift;

DELETE FROM DT_STUDY.RESULT.signal_daily WHERE signal = 'S4' AND strategy = 'FIXED';
INSERT INTO DT_STUDY.RESULT.signal_daily
SELECT arm, day_index, 'S4', 'FIXED', statistic FROM DT_STUDY.FEAT.s4_centroid_shift;

DELETE FROM DT_STUDY.RESULT.signal_daily WHERE signal = 'S5' AND strategy = 'FIXED';
INSERT INTO DT_STUDY.RESULT.signal_daily
SELECT arm, day_index, 'S5', 'FIXED', statistic FROM DT_STUDY.FEAT.s5_mmd_linear;

DELETE FROM DT_STUDY.RESULT.signal_daily WHERE signal = 'S6' AND strategy = 'FIXED';
INSERT INTO DT_STUDY.RESULT.signal_daily
SELECT arm, day_index, 'S6', 'FIXED', statistic FROM DT_STUDY.FEAT.s6_nn_distance;

-- S7a only - this is the paid-classification-dependent signal (README §3.6).
-- Run this block only for the arms/replicates whose predictions table you
-- actually paid to build; it is a no-op (zero rows) for any arm without one.
DELETE FROM DT_STUDY.RESULT.signal_daily WHERE signal = 'S7a' AND strategy = 'FIXED';
INSERT INTO DT_STUDY.RESULT.signal_daily
SELECT arm, day_index, 'S7a', 'FIXED', statistic FROM DT_STUDY.FEAT.s7a_prior_shift;


-- ============================================================================
-- 6. GROUND TRUTH  (analysis only - never an input to any detector)
-- ============================================================================

CREATE OR REPLACE VIEW DT_STUDY.RESULT.accuracy_daily AS
SELECT arm, day_index,
       AVG(IFF(predicted_label = true_label, 1, 0)) AS true_accuracy
FROM DT_STUDY.FEAT.predictions
GROUP BY 1,2;


-- ============================================================================
-- 7. THRESHOLD AND FIRING  (one policy, applied uniformly)
-- ============================================================================
-- Fires on day d if the statistic exceeds mean+3*stdev of the CLEAN window
-- AND also exceeded it on day d-1.  The consecutive-day requirement costs one
-- day of lag and removes most single-day noise spikes.

CREATE OR REPLACE VIEW DT_STUDY.RESULT.signal_fired AS
WITH base AS (
    SELECT arm, signal, strategy,
           AVG(statistic)    AS mu,
           STDDEV(statistic) AS sd
    FROM DT_STUDY.RESULT.signal_daily
    WHERE day_index BETWEEN 15 AND 34
    GROUP BY 1,2,3
),
scored AS (
    SELECT s.arm, s.day_index, s.signal, s.strategy, s.statistic,
           b.mu + 3 * b.sd AS threshold,
           IFF(s.statistic > b.mu + 3 * b.sd, 1, 0) AS over
    FROM DT_STUDY.RESULT.signal_daily s
    JOIN base b
      ON b.arm = s.arm AND b.signal = s.signal AND b.strategy = s.strategy
)
SELECT *,
       LAG(over) OVER (PARTITION BY arm, signal, strategy
                       ORDER BY day_index) AS over_prev,
       IFF(over = 1 AND LAG(over) OVER (PARTITION BY arm, signal, strategy
                                        ORDER BY day_index) = 1, 1, 0) AS fired
FROM scored;

-- ----------------------------------------------------------------------------
-- DETECTION LAG, FALSE POSITIVES, ACCURACY AT DETECTION
-- ----------------------------------------------------------------------------
-- Non-detection is returned as NULL, never as a large number.  A censored
-- observation is not a slow one and must not be averaged as though it were.

CREATE OR REPLACE VIEW DT_STUDY.RESULT.detection_summary AS
WITH onset AS (SELECT 35 AS onset_day),
first_fire AS (
    SELECT arm, signal, strategy, MIN(day_index) AS fire_day
    FROM DT_STUDY.RESULT.signal_fired
    WHERE fired = 1 AND day_index >= 35
    GROUP BY 1,2,3
),
fp AS (
    SELECT arm, signal, strategy, SUM(fired) AS false_positives
    FROM DT_STUDY.RESULT.signal_fired
    WHERE day_index BETWEEN 15 AND 34
    GROUP BY 1,2,3
)
SELECT f.arm, f.signal, f.strategy,
       ff.fire_day,
       ff.fire_day - o.onset_day       AS detection_lag_days,
       f.false_positives,
       a.true_accuracy                 AS accuracy_at_detection,
       base.true_accuracy              AS baseline_accuracy,
       base.true_accuracy - a.true_accuracy AS accuracy_lost_before_detection
FROM fp f
CROSS JOIN onset o
LEFT JOIN first_fire ff
       ON ff.arm = f.arm AND ff.signal = f.signal AND ff.strategy = f.strategy
LEFT JOIN DT_STUDY.RESULT.accuracy_daily a
       ON a.arm = f.arm AND a.day_index = ff.fire_day
LEFT JOIN (
    SELECT arm, AVG(true_accuracy) AS true_accuracy
    FROM DT_STUDY.RESULT.accuracy_daily
    WHERE day_index BETWEEN 15 AND 34
    GROUP BY 1
) base ON base.arm = f.arm;


-- ============================================================================
-- 8. COST ATTRIBUTION
-- ============================================================================
-- Warehouse credits are metered at warehouse granularity, so per-signal cost
-- requires running each signal in a separately timed window on an otherwise
-- idle warehouse.  Record start/end timestamps around each signal computation
-- and join to metering afterwards.  ACCOUNT_USAGE lags several hours - use it
-- for final collection only, never during the run.

CREATE OR REPLACE TABLE DT_STUDY.RESULT.signal_timing (
    arm STRING, day_index INT, signal STRING,
    started_at TIMESTAMP_LTZ, ended_at TIMESTAMP_LTZ
);

CREATE OR REPLACE VIEW DT_STUDY.RESULT.cost_daily AS
SELECT t.arm, t.day_index, t.signal,
       SUM(m.credits_used
           * DATEDIFF('millisecond', t.started_at, t.ended_at)
           / NULLIF(DATEDIFF('millisecond', m.start_time, m.end_time), 0)
       ) AS credits_attributed
FROM DT_STUDY.RESULT.signal_timing t
JOIN SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY m
  ON m.warehouse_name = 'DRIFT_WH'
 AND t.started_at < m.end_time AND t.ended_at > m.start_time
GROUP BY 1,2,3;
-- This pro-rates hourly metering across the signal's execution window.  It is
-- an approximation and must be described as one.  Cortex token consumption is
-- metered separately and should be reported alongside, not folded in.


-- ============================================================================
-- 9. VALIDATION GATE - RUN BEFORE ANY DRIFTED ARM IS GENERATED
-- ============================================================================
-- If any signal fires during the clean window, the baseline is not stable and
-- every detection-lag number afterwards is noise.  This check is nearly free.

SELECT arm, signal, strategy, SUM(fired) AS clean_window_firings
FROM DT_STUDY.RESULT.signal_fired
WHERE day_index BETWEEN 15 AND 34
GROUP BY 1,2,3
HAVING SUM(fired) > 0;
-- Expected result: zero rows.  Anything else means fix the generator or widen
-- the threshold before proceeding.
