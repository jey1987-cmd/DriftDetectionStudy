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
-- SCHEMA RECONCILIATION (2026-11-01): this file previously wrote to
-- DT_STUDY.FEAT and DT_STUDY.RESULT, schemas that were never created in the
-- study_v1 account (confirmed via SHOW SCHEMAS IN DATABASE DT_STUDY: only
-- DEDUP_SCHEMA, DRIFT, INFORMATION_SCHEMA, PUBLIC exist). This file had
-- therefore never actually been run against study_v1 - every published
-- number came from a separate, un-versioned set of ad hoc queries that wrote
-- flat per-(signal,arm) tables directly into DT_STUDY.DRIFT (S1_ARM_B,
-- S7A_ARM_D, S7A_CALIBRATION, etc., no shared signal_daily table, no arm
-- filter column on most of them). Consolidated everything here into
-- DT_STUDY.DRIFT instead, onto the single shared signal_daily/predictions
-- structure this file already intended, so there is one versioned pipeline
-- instead of two divergent ones. The flat tables are left in place as a
-- cross-check: re-running this file for arm B/D and comparing its
-- signal_daily/predictions output against them is the validation gate
-- (see the note at the bottom of this file).

USE WAREHOUSE DT_TESTBED_WH;
USE SCHEMA DT_STUDY.DRIFT;

-- daily_records itself is created once via README.md §3.2 (or Account2.sql) and
-- loaded by Loader.py - it is NOT recreated here, since doing so would silently
-- truncate whatever has already been loaded.
--
-- IDEMPOTENCY FIX (reviewer-response-r1): signal_daily was CREATE OR REPLACE,
-- which would discard every other arm's results if this whole file is re-run
-- per arm/replicate - undoing the point of making every block below
-- incremental. CREATE TABLE IF NOT EXISTS here; Section 5b's per-signal
-- DELETE+INSERT is what keeps re-runs idempotent per arm.

CREATE TABLE IF NOT EXISTS DT_STUDY.DRIFT.signal_daily (
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
--
-- INCREMENTAL FIX (reviewer-response-r1): every table in this section used to
-- be CREATE OR REPLACE TABLE ... AS SELECT ... with no WHERE arm filter, so it
-- recomputed over EVERY arm currently in daily_records on every run - loading
-- one more replicate and re-running this file re-tokenised every arm already
-- done. Tier 1 is free, so that was only wasted compute, not wasted money, but
-- the same CREATE-OR-REPLACE-everything pattern in Section 3 (embeddings) and
-- Section 4 (predictions) DOES re-bill metered API calls for every arm already
-- processed, not just the new one - see those sections. Switched the whole
-- pipeline to create-once + per-arm DELETE+INSERT uniformly, substituting
-- '<ARM>' with the arm/replicate label just loaded, so re-running this file
-- after loading one more arm only (re)processes that arm.

CREATE TABLE IF NOT EXISTS DT_STUDY.DRIFT.tokens (
    arm STRING, day_index INT, record_id STRING, token STRING
);
DELETE FROM DT_STUDY.DRIFT.tokens WHERE arm = '<ARM>';
INSERT INTO DT_STUDY.DRIFT.tokens
SELECT r.arm,
       r.day_index,
       r.record_id,
       LOWER(t.value::STRING) AS token
FROM DT_STUDY.DRIFT.daily_records r,
     LATERAL SPLIT_TO_TABLE(
         REGEXP_REPLACE(LOWER(r.text), '[^a-z0-9 ]', ' '), ' ') t
WHERE r.arm = '<ARM>' AND t.value <> '';

CREATE TABLE IF NOT EXISTS DT_STUDY.DRIFT.record_lengths (
    arm STRING, day_index INT, record_id STRING, n_tokens INT
);
DELETE FROM DT_STUDY.DRIFT.record_lengths WHERE arm = '<ARM>';
INSERT INTO DT_STUDY.DRIFT.record_lengths
SELECT arm, day_index, record_id, COUNT(*) AS n_tokens
FROM DT_STUDY.DRIFT.tokens
WHERE arm = '<ARM>'
GROUP BY 1,2,3;


-- ============================================================================
-- 2. TIER 1 - STRUCTURAL SIGNALS  (pure SQL, no model calls)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- S1  LENGTH DISTRIBUTION - two-sample Kolmogorov-Smirnov statistic
-- ----------------------------------------------------------------------------
-- KS = max |CDF_ref(x) - CDF_day(x)| over the union of observed lengths.
--
-- REWRITTEN AS A VIEW, NOT A UDF (2026-11-01): this was a scalar SQL
-- function taking (arm, day) and computing the FULL OUTER JOIN internally.
-- It failed outright when deployed: first on a declared-vs-actual return
-- type mismatch (FLOAT vs the NUMBER(37,6) the division actually produces),
-- then - after casting that - on "Unsupported subquery type cannot be
-- evaluated inside Function object": Snowflake does not support calling a
-- UDF whose body contains a multi-table/CTE subquery once per row from an
-- outer query; it only works called with literal arguments. Both failures
-- together are further confirmation this signal had never actually been
-- run - a function that cannot be invoked the way Section 5b calls it would
-- have failed the very first time anyone tried. Rewritten as a flat view
-- computing every (arm, day_index)'s KS statistic against that arm's own
-- reference window (days 15-34) in one pass: the grid of token-length
-- breakpoints is the union of each day's own lengths and its arm's
-- reference-window lengths, and both CDFs are evaluated on that union via
-- window functions instead of a per-row function call.

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.s1_ks_length AS
WITH ref_counts AS (
    SELECT arm, n_tokens, COUNT(*) AS rc
    FROM DT_STUDY.DRIFT.record_lengths
    WHERE day_index BETWEEN 15 AND 34
    GROUP BY 1,2
),
ref_total AS (SELECT arm, SUM(rc) AS rt FROM ref_counts GROUP BY 1),
cur_counts AS (
    SELECT arm, day_index, n_tokens, COUNT(*) AS cc
    FROM DT_STUDY.DRIFT.record_lengths
    GROUP BY 1,2,3
),
cur_total AS (SELECT arm, day_index, SUM(cc) AS ct FROM cur_counts GROUP BY 1,2),
days AS (SELECT DISTINCT arm, day_index FROM cur_counts),
grid AS (
    SELECT arm, day_index, n_tokens FROM cur_counts
    UNION
    SELECT d.arm, d.day_index, r.n_tokens
    FROM days d JOIN ref_counts r ON r.arm = d.arm
),
joined AS (
    SELECT g.arm, g.day_index, g.n_tokens,
           COALESCE(r.rc, 0) AS rc,
           COALESCE(c.cc, 0) AS cc
    FROM grid g
    LEFT JOIN ref_counts r ON r.arm = g.arm AND r.n_tokens = g.n_tokens
    LEFT JOIN cur_counts c ON c.arm = g.arm AND c.day_index = g.day_index AND c.n_tokens = g.n_tokens
),
cdf AS (
    SELECT j.arm, j.day_index, j.n_tokens,
           SUM(j.rc) OVER (PARTITION BY j.arm, j.day_index ORDER BY j.n_tokens)
             / NULLIF(rt.rt, 0) AS ref_cdf,
           SUM(j.cc) OVER (PARTITION BY j.arm, j.day_index ORDER BY j.n_tokens)
             / NULLIF(ct.ct, 0) AS cur_cdf
    FROM joined j
    JOIN ref_total rt ON rt.arm = j.arm
    JOIN cur_total ct ON ct.arm = j.arm AND ct.day_index = j.day_index
)
SELECT arm, day_index, MAX(ABS(ref_cdf - cur_cdf))::FLOAT AS statistic
FROM cdf
GROUP BY 1,2;

-- ----------------------------------------------------------------------------
-- S2  VOCABULARY NOVELTY - fraction of today's tokens unseen in reference
-- ----------------------------------------------------------------------------
-- Computed over token OCCURRENCES, not distinct types: a single new word used
-- 500 times is a bigger distribution change than 5 words used once each.

CREATE TABLE IF NOT EXISTS DT_STUDY.DRIFT.ref_vocab (arm STRING, token STRING);
DELETE FROM DT_STUDY.DRIFT.ref_vocab WHERE arm = '<ARM>';
INSERT INTO DT_STUDY.DRIFT.ref_vocab
SELECT DISTINCT arm, token
FROM DT_STUDY.DRIFT.tokens
WHERE arm = '<ARM>' AND day_index BETWEEN 15 AND 34;

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.s2_vocab_novelty AS
SELECT t.arm,
       t.day_index,
       COUNT_IF(v.token IS NULL) / COUNT(*)::FLOAT AS statistic
FROM DT_STUDY.DRIFT.tokens t
LEFT JOIN DT_STUDY.DRIFT.ref_vocab v
       ON v.arm = t.arm AND v.token = t.token
GROUP BY 1,2;

-- ----------------------------------------------------------------------------
-- S3  TYPE-TOKEN RATIO - absolute deviation from reference lexical diversity
-- ----------------------------------------------------------------------------
-- TTR is sensitive to sample size, so it is computed per DAY (equal volume by
-- design) and the reference is the MEAN of daily TTRs, not a pooled TTR.

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.daily_ttr AS
SELECT arm, day_index,
       COUNT(DISTINCT token) / COUNT(*)::FLOAT AS ttr
FROM DT_STUDY.DRIFT.tokens
GROUP BY 1,2;

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.s3_ttr_shift AS
WITH ref AS (
    SELECT arm, AVG(ttr) AS ref_ttr
    FROM DT_STUDY.DRIFT.daily_ttr
    WHERE day_index BETWEEN 15 AND 34
    GROUP BY 1
)
SELECT d.arm, d.day_index, ABS(d.ttr - r.ref_ttr) AS statistic
FROM DT_STUDY.DRIFT.daily_ttr d JOIN ref r ON r.arm = d.arm;


-- ============================================================================
-- 3. TIER 2 - EMBEDDING SIGNALS  (metered Cortex calls)
-- ============================================================================
-- COST CONTROL: embed a FIXED sample of k=200 records per day, never the full
-- 2,000.  The sample must be deterministic so the study is reproducible, and it
-- must be the same size every day or the statistics are not comparable.

-- COST CONTROL, INCREMENTAL (reviewer-response-r1): embeddings below is the
-- single most expensive table to recompute carelessly, since EMBED_TEXT_768 is
-- a metered call per row. The old CREATE OR REPLACE re-embedded every arm
-- already done, every time this block ran - load one new replicate, rerun,
-- and pay again for every prior arm's 200/day sample too. DELETE+INSERT scoped
-- to '<ARM>' makes re-running this block after loading one more arm cost
-- exactly that arm's embeddings, not everyone's.

CREATE TABLE IF NOT EXISTS DT_STUDY.DRIFT.daily_sample (
    arm STRING, day_index INT, record_id STRING, text STRING
);
DELETE FROM DT_STUDY.DRIFT.daily_sample WHERE arm = '<ARM>';
INSERT INTO DT_STUDY.DRIFT.daily_sample
SELECT arm, day_index, record_id, text
FROM (
    SELECT arm, day_index, record_id, text,
           ROW_NUMBER() OVER (
               PARTITION BY arm, day_index
               ORDER BY HASH(record_id)          -- deterministic, not RANDOM()
           ) AS rn
    FROM DT_STUDY.DRIFT.daily_records
    WHERE arm = '<ARM>'
)
WHERE rn <= 200;

-- Embed once, reuse for S4/S5/S6.  Stored as ARRAY so element-wise aggregation
-- is possible; Snowflake has no AVG() over VECTOR.
--
-- CAST FIX (2026-11-01): EMBED_TEXT_768 returns VECTOR(FLOAT, 768), not
-- ARRAY. Inserting it into an ARRAY column without an explicit cast fails
-- with "Expression type does not match column data type, expecting ARRAY
-- but got VECTOR(FLOAT, 768)" - confirmed live running this for arm
-- BASE/C/D. Arm B's embeddings (already in this table) predate this bug
-- being noticed; whatever query actually produced them must have had the
-- cast or an equivalent, since the table itself is schema ARRAY and already
-- holds rows.
CREATE TABLE IF NOT EXISTS DT_STUDY.DRIFT.embeddings (
    arm STRING, day_index INT, record_id STRING, emb ARRAY
);
DELETE FROM DT_STUDY.DRIFT.embeddings WHERE arm = '<ARM>';
INSERT INTO DT_STUDY.DRIFT.embeddings
SELECT arm, day_index, record_id,
       SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', text)::ARRAY AS emb
FROM DT_STUDY.DRIFT.daily_sample
WHERE arm = '<ARM>';

-- Element-wise long form, used for centroid arithmetic
CREATE OR REPLACE VIEW DT_STUDY.DRIFT.emb_elements AS
SELECT arm, day_index, record_id,
       f.index AS pos, f.value::FLOAT AS v
FROM DT_STUDY.DRIFT.embeddings,
     LATERAL FLATTEN(input => emb) f;

-- ----------------------------------------------------------------------------
-- S4  CENTROID DISPLACEMENT - 1 - cosine(reference centroid, day centroid)
-- ----------------------------------------------------------------------------

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.ref_centroid AS
SELECT arm, pos, AVG(v) AS v
FROM DT_STUDY.DRIFT.emb_elements
WHERE day_index BETWEEN 15 AND 34
GROUP BY 1,2;

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.day_centroid AS
SELECT arm, day_index, pos, AVG(v) AS v
FROM DT_STUDY.DRIFT.emb_elements
GROUP BY 1,2,3;

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.s4_centroid_shift AS
SELECT d.arm, d.day_index,
       1 - ( SUM(d.v * r.v)
             / NULLIF(SQRT(SUM(d.v*d.v)) * SQRT(SUM(r.v*r.v)), 0) ) AS statistic
FROM DT_STUDY.DRIFT.day_centroid d
JOIN DT_STUDY.DRIFT.ref_centroid r
  ON r.arm = d.arm AND r.pos = d.pos
GROUP BY 1,2;

-- ----------------------------------------------------------------------------
-- S6  NEAREST-NEIGHBOUR DISTANCE
-- ----------------------------------------------------------------------------
-- Catches a novel cluster that does not move the centroid (arm C), which is
-- precisely the case S4 is blind to.  O(k * |ref|) pairwise - the reference is
-- capped at 2,000 vectors to bound cost.

CREATE TABLE IF NOT EXISTS DT_STUDY.DRIFT.ref_sample_emb (
    arm STRING, record_id STRING, emb ARRAY
);
DELETE FROM DT_STUDY.DRIFT.ref_sample_emb WHERE arm = '<ARM>';
INSERT INTO DT_STUDY.DRIFT.ref_sample_emb
SELECT arm, record_id, emb
FROM (
    SELECT arm, record_id, emb,
           ROW_NUMBER() OVER (PARTITION BY arm ORDER BY HASH(record_id)) rn
    FROM DT_STUDY.DRIFT.embeddings
    WHERE arm = '<ARM>' AND day_index BETWEEN 15 AND 34
)
WHERE rn <= 2000;

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.s6_nn_distance AS
WITH pair AS (
    SELECT d.arm, d.day_index, d.record_id,
           MAX(VECTOR_COSINE_SIMILARITY(d.emb, r.emb)) AS best_sim
    FROM DT_STUDY.DRIFT.embeddings d
    JOIN DT_STUDY.DRIFT.ref_sample_emb r
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

-- LEGACY DIVERGENCE (2026-11-01): the validation gate (Section 10) will show
-- this view's statistic differing from the legacy S6_ARM_B flat table ONLY
-- for reference-window days (15-34), by ~0.02 on average - NOT in the
-- measurement window (35+), which matches to floating-point precision. This
-- is expected, not a bug in this view: the legacy table was computed without
-- the leave-one-out exclusion above, so its reference-window statistic is
-- deflated by the self-match artifact described above. This view is the
-- corrected one. It means the PUBLISHED reference-window calibration
-- mean/stdev for S6 (threshold-setting) was computed on deflated statistics -
-- note this alongside the Reviewer 1 #4 / Reviewer 2 #3 fix in the revision
-- text (03_section_VIII_threats_additions.md) as corroborating evidence, not
-- a new concern.

-- ----------------------------------------------------------------------------
-- S5  MAXIMUM MEAN DISCREPANCY  (linear-kernel form)
-- ----------------------------------------------------------------------------
-- Linear-kernel MMD^2 reduces to the squared L2 distance between mean
-- embeddings - computable in O(d) rather than the O(n^2) an RBF kernel needs.
-- If the RBF form is wanted, it requires a pairwise self-join within each day
-- as well as across, at ~k^2 = 40,000 similarity evaluations per day per arm.
-- Report which form was used; they are not interchangeable in sensitivity.

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.s5_mmd_linear AS
SELECT d.arm, d.day_index,
       SQRT(SUM(POWER(d.v - r.v, 2))) AS statistic
FROM DT_STUDY.DRIFT.day_centroid d
JOIN DT_STUDY.DRIFT.ref_centroid r
  ON r.arm = d.arm AND r.pos = d.pos
GROUP BY 1,2;


-- ============================================================================
-- 4. CLASSIFICATION  (produces the input to Tier 3)
-- ============================================================================
-- Runs on ALL records for the day, not the sample - Tier 3 is "free if you are
-- already inferring", and that framing only holds at full volume.

-- LABEL FIX (2026-11-01): bare labels with the confusable pair renamed -
-- same set as classify_calibration.sql, verified against QUERY_HISTORY as
-- what actually produced predictions_arm_b/predictions_arm_d. predicted_label
-- is in a different vocabulary than true_label for these two classes only;
-- predicted_label_norm maps it back (see classify_calibration.sql's header
-- for the full explanation and the manuscript-disclosure note). Arm C's
-- measurement window still needs the 11-label variant adding
-- 'sidewalk_obstruction' (the novel class) - that is intentionally NOT in
-- this shared array; it must be a separate classification pass for arm='C'
-- only, or every other arm's predictions change.
--
-- INCREMENTAL/COST FIX (reviewer-response-r1): this is the single most
-- important table to make incremental - it's the paid classification step.
-- DELETE+INSERT scoped to '<ARM>' makes this block classify exactly the arm
-- you name and nothing else - run it only for the seeds you've decided to
-- pay for.
CREATE TABLE IF NOT EXISTS DT_STUDY.DRIFT.predictions (
    arm STRING, day_index INT, record_id STRING, true_label STRING,
    predicted_label STRING, predicted_label_norm STRING
);
DELETE FROM DT_STUDY.DRIFT.predictions WHERE arm = '<ARM>';
INSERT INTO DT_STUDY.DRIFT.predictions
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
    WHERE arm = '<ARM>' AND day_index >= 15
);

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

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.ref_priors AS
SELECT arm, predicted_label_norm,
       COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY arm)::FLOAT AS p
FROM DT_STUDY.DRIFT.predictions
WHERE day_index BETWEEN 15 AND 34
GROUP BY 1,2;

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.s7a_prior_shift AS
WITH obs AS (
    SELECT arm, day_index, predicted_label_norm, COUNT(*) AS o
    FROM DT_STUDY.DRIFT.predictions
    GROUP BY 1,2,3
),
tot AS (SELECT arm, day_index, SUM(o) AS n FROM obs GROUP BY 1,2)
SELECT o.arm, o.day_index,
       SUM( POWER(o.o - t.n * COALESCE(r.p, 1e-6), 2)
            / NULLIF(t.n * COALESCE(r.p, 1e-6), 0) ) AS statistic
FROM obs o
JOIN tot t  ON t.arm = o.arm AND t.day_index = o.day_index
LEFT JOIN DT_STUDY.DRIFT.ref_priors r
       ON r.arm = o.arm AND r.predicted_label_norm = o.predicted_label_norm
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

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.ref_priors_laplace AS
WITH vocab AS (
    SELECT DISTINCT arm, predicted_label_norm FROM DT_STUDY.DRIFT.predictions
),
ref_counts AS (
    SELECT arm, predicted_label_norm, COUNT(*) AS c
    FROM DT_STUDY.DRIFT.predictions
    WHERE day_index BETWEEN 15 AND 34
    GROUP BY 1,2
),
k AS (SELECT arm, COUNT(DISTINCT predicted_label_norm) AS n_labels
      FROM DT_STUDY.DRIFT.predictions GROUP BY 1),
ref_total AS (SELECT arm, SUM(c) AS n FROM ref_counts GROUP BY 1)
SELECT v.arm, v.predicted_label_norm,
       (COALESCE(rc.c, 0) + 1) / (rt.n + k.n_labels)::FLOAT AS p
FROM vocab v
JOIN k        ON k.arm = v.arm
JOIN ref_total rt ON rt.arm = v.arm
LEFT JOIN ref_counts rc ON rc.arm = v.arm AND rc.predicted_label_norm = v.predicted_label_norm;

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.s7a_prior_shift_laplace AS
WITH obs AS (
    SELECT arm, day_index, predicted_label_norm, COUNT(*) AS o
    FROM DT_STUDY.DRIFT.predictions
    GROUP BY 1,2,3
),
tot AS (SELECT arm, day_index, SUM(o) AS n FROM obs GROUP BY 1,2)
SELECT o.arm, o.day_index,
       SUM( POWER(o.o - t.n * r.p, 2) / (t.n * r.p) ) AS statistic
FROM obs o
JOIN tot t ON t.arm = o.arm AND t.day_index = o.day_index
JOIN DT_STUDY.DRIFT.ref_priors_laplace r
  ON r.arm = o.arm AND r.predicted_label_norm = o.predicted_label_norm
GROUP BY 1,2;


-- ============================================================================
-- 5b. POPULATE signal_daily  (reviewer-response-r1: this was missing)
-- ============================================================================
-- signal_fired and detection_summary below both read FROM signal_daily, but
-- nothing in this file ever inserted the Tier 1/2/3 views' output into it -
-- every view above is fully specified, this section only wires it into the
-- results table, no new statistic logic.
--
-- RESCOPED TO '<ARM>' (2026-11-01): this was previously a single global
-- DELETE+INSERT per signal, unscoped by arm - inconsistent with every other
-- section's "'<ARM>'-scoped" convention, and it cost a live ~20-minute S6
-- recompute (cancelled) just to add one new arm (D_ext150), because it
-- recomputed S6's O(k*|ref|) pairwise distance for every arm already done,
-- not just the new one. S6 is now safe to cancel mid-run without data loss
-- (its own DELETE commits immediately - Snowflake's Python connector
-- autocommits per statement by default - so a cancelled INSERT after it
-- leaves that signal's rows empty for every arm, not just the new one,
-- until it's re-run) but there is no reason to pay that cost at all when
-- only one arm changed. Substitute '<ARM>' per this file's usual convention;
-- re-running for an arm whose underlying tables haven't changed is still
-- safe (same result), just no longer necessary.
--
-- STRATEGY GAP: DESIGN.md §4.3 calls for every signal under both FIXED and
-- SLIDING (trailing 7-day) reference strategies. Only FIXED is implemented
-- anywhere in this file - none of the views above compute a sliding
-- reference. Only 'FIXED' rows are inserted below; add the SLIDING views
-- first if the response to reviewers needs that comparison.

DELETE FROM DT_STUDY.DRIFT.signal_daily WHERE signal = 'S1' AND strategy = 'FIXED' AND arm = '<ARM>';
INSERT INTO DT_STUDY.DRIFT.signal_daily
SELECT arm, day_index, 'S1', 'FIXED', statistic FROM DT_STUDY.DRIFT.s1_ks_length WHERE arm = '<ARM>';

DELETE FROM DT_STUDY.DRIFT.signal_daily WHERE signal = 'S2' AND strategy = 'FIXED' AND arm = '<ARM>';
INSERT INTO DT_STUDY.DRIFT.signal_daily
SELECT arm, day_index, 'S2', 'FIXED', statistic FROM DT_STUDY.DRIFT.s2_vocab_novelty WHERE arm = '<ARM>';

DELETE FROM DT_STUDY.DRIFT.signal_daily WHERE signal = 'S3' AND strategy = 'FIXED' AND arm = '<ARM>';
INSERT INTO DT_STUDY.DRIFT.signal_daily
SELECT arm, day_index, 'S3', 'FIXED', statistic FROM DT_STUDY.DRIFT.s3_ttr_shift WHERE arm = '<ARM>';

DELETE FROM DT_STUDY.DRIFT.signal_daily WHERE signal = 'S4' AND strategy = 'FIXED' AND arm = '<ARM>';
INSERT INTO DT_STUDY.DRIFT.signal_daily
SELECT arm, day_index, 'S4', 'FIXED', statistic FROM DT_STUDY.DRIFT.s4_centroid_shift WHERE arm = '<ARM>';

DELETE FROM DT_STUDY.DRIFT.signal_daily WHERE signal = 'S5' AND strategy = 'FIXED' AND arm = '<ARM>';
INSERT INTO DT_STUDY.DRIFT.signal_daily
SELECT arm, day_index, 'S5', 'FIXED', statistic FROM DT_STUDY.DRIFT.s5_mmd_linear WHERE arm = '<ARM>';

DELETE FROM DT_STUDY.DRIFT.signal_daily WHERE signal = 'S6' AND strategy = 'FIXED' AND arm = '<ARM>';
INSERT INTO DT_STUDY.DRIFT.signal_daily
SELECT arm, day_index, 'S6', 'FIXED', statistic FROM DT_STUDY.DRIFT.s6_nn_distance WHERE arm = '<ARM>';

-- S7a only - this is the paid-classification-dependent signal (README §3.6).
-- Run this block only for the arms/replicates whose predictions table you
-- actually paid to build; it is a no-op (zero rows) for any arm without one.
DELETE FROM DT_STUDY.DRIFT.signal_daily WHERE signal = 'S7a' AND strategy = 'FIXED' AND arm = '<ARM>';
INSERT INTO DT_STUDY.DRIFT.signal_daily
SELECT arm, day_index, 'S7a', 'FIXED', statistic FROM DT_STUDY.DRIFT.s7a_prior_shift WHERE arm = '<ARM>';


-- ============================================================================
-- 6. GROUND TRUTH  (analysis only - never an input to any detector)
-- ============================================================================

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.accuracy_daily AS
SELECT arm, day_index,
       AVG(IFF(predicted_label_norm = true_label, 1, 0)) AS true_accuracy
FROM DT_STUDY.DRIFT.predictions
GROUP BY 1,2;


-- ============================================================================
-- 7. THRESHOLD AND FIRING  (one policy, applied uniformly)
-- ============================================================================
-- Fires on day d if the statistic exceeds mean+3*stdev of the CLEAN window
-- AND also exceeded it on day d-1.  The consecutive-day requirement costs one
-- day of lag and removes most single-day noise spikes.

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.signal_fired AS
WITH base AS (
    SELECT arm, signal, strategy,
           AVG(statistic)    AS mu,
           STDDEV(statistic) AS sd
    FROM DT_STUDY.DRIFT.signal_daily
    WHERE day_index BETWEEN 15 AND 34
    GROUP BY 1,2,3
),
scored AS (
    SELECT s.arm, s.day_index, s.signal, s.strategy, s.statistic,
           b.mu + 3 * b.sd AS threshold,
           IFF(s.statistic > b.mu + 3 * b.sd, 1, 0) AS over
    FROM DT_STUDY.DRIFT.signal_daily s
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

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.detection_summary AS
WITH onset AS (SELECT 35 AS onset_day),
first_fire AS (
    SELECT arm, signal, strategy, MIN(day_index) AS fire_day
    FROM DT_STUDY.DRIFT.signal_fired
    WHERE fired = 1 AND day_index >= 35
    GROUP BY 1,2,3
),
fp AS (
    SELECT arm, signal, strategy, SUM(fired) AS false_positives
    FROM DT_STUDY.DRIFT.signal_fired
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
LEFT JOIN DT_STUDY.DRIFT.accuracy_daily a
       ON a.arm = f.arm AND a.day_index = ff.fire_day
LEFT JOIN (
    SELECT arm, AVG(true_accuracy) AS true_accuracy
    FROM DT_STUDY.DRIFT.accuracy_daily
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

-- CREATE TABLE IF NOT EXISTS (reviewer-response-r1): same idempotency fix as
-- signal_daily above - this is an append-only log, CREATE OR REPLACE would
-- discard every prior arm's timing log on re-run.
CREATE TABLE IF NOT EXISTS DT_STUDY.DRIFT.signal_timing (
    arm STRING, day_index INT, signal STRING,
    started_at TIMESTAMP_LTZ, ended_at TIMESTAMP_LTZ
);

CREATE OR REPLACE VIEW DT_STUDY.DRIFT.cost_daily AS
SELECT t.arm, t.day_index, t.signal,
       SUM(m.credits_used
           * DATEDIFF('millisecond', t.started_at, t.ended_at)
           / NULLIF(DATEDIFF('millisecond', m.start_time, m.end_time), 0)
       ) AS credits_attributed
FROM DT_STUDY.DRIFT.signal_timing t
JOIN SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY m
  ON m.warehouse_name = 'DT_TESTBED_WH'
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
FROM DT_STUDY.DRIFT.signal_fired
WHERE day_index BETWEEN 15 AND 34
GROUP BY 1,2,3
HAVING SUM(fired) > 0;
-- Expected result: zero rows.  Anything else means fix the generator or widen
-- the threshold before proceeding.


-- ============================================================================
-- 10. CROSS-CHECK AGAINST THE LEGACY FLAT TABLES (2026-11-01)
-- ============================================================================
-- Run this ONLY for arm='B' and arm='D', the two arms with a legacy flat
-- table to compare against (S1_ARM_B, S2_ARM_B, S7A_ARM_B, S7A_ARM_D,
-- S7A_CALIBRATION - see Section 0's note). This is the gate for trusting
-- this file as the published source of Table 4: if it disagrees with the
-- legacy numbers for the arms that DO have legacy data, find out why before
-- trusting it for arm C or BASE, which have no legacy table to check against.
-- Zero cost - pure comparison of already-computed statistics.

SELECT 'S1' AS signal, 'B' AS arm,
       s.day_index, s.statistic AS new_statistic, l.statistic AS legacy_statistic,
       ABS(s.statistic - l.statistic) AS abs_diff
FROM DT_STUDY.DRIFT.signal_daily s
JOIN DT_STUDY.DRIFT.s1_arm_b l ON l.day_index = s.day_index
WHERE s.arm = 'B' AND s.signal = 'S1' AND s.strategy = 'FIXED'
UNION ALL
SELECT 'S2', 'B', s.day_index, s.statistic, l.statistic, ABS(s.statistic - l.statistic)
FROM DT_STUDY.DRIFT.signal_daily s
JOIN DT_STUDY.DRIFT.s2_arm_b l ON l.day_index = s.day_index
WHERE s.arm = 'B' AND s.signal = 'S2' AND s.strategy = 'FIXED'
UNION ALL
SELECT 'S3', 'B', s.day_index, s.statistic, l.statistic, ABS(s.statistic - l.statistic)
FROM DT_STUDY.DRIFT.signal_daily s
JOIN DT_STUDY.DRIFT.s3_arm_b l ON l.day_index = s.day_index
WHERE s.arm = 'B' AND s.signal = 'S3' AND s.strategy = 'FIXED'
UNION ALL
SELECT 'S4', 'B', s.day_index, s.statistic, l.statistic, ABS(s.statistic - l.statistic)
FROM DT_STUDY.DRIFT.signal_daily s
JOIN DT_STUDY.DRIFT.s4_arm_b l ON l.day_index = s.day_index
WHERE s.arm = 'B' AND s.signal = 'S4' AND s.strategy = 'FIXED'
UNION ALL
SELECT 'S5', 'B', s.day_index, s.statistic, l.statistic, ABS(s.statistic - l.statistic)
FROM DT_STUDY.DRIFT.signal_daily s
JOIN DT_STUDY.DRIFT.s5_arm_b l ON l.day_index = s.day_index
WHERE s.arm = 'B' AND s.signal = 'S5' AND s.strategy = 'FIXED'
UNION ALL
-- S6: expect divergence confined to the reference window (days 15-34) only -
-- see the LEGACY DIVERGENCE note on s6_nn_distance above. Measurement window
-- (35+) should match to floating-point precision; reference window will not.
SELECT 'S6', 'B', s.day_index, s.statistic, l.statistic, ABS(s.statistic - l.statistic)
FROM DT_STUDY.DRIFT.signal_daily s
JOIN DT_STUDY.DRIFT.s6_arm_b l ON l.day_index = s.day_index
WHERE s.arm = 'B' AND s.signal = 'S6' AND s.strategy = 'FIXED'
UNION ALL
-- CORRECTION (2026-11-01, same day): this previously read 'S3', 'D' joined
-- to s3_arm_d - that table never existed (confirmed via SHOW TABLES:
-- D only has legacy S1_ARM_D, S2_ARM_D, S7A_ARM_D; no S3/S4/S5/S6_ARM_D was
-- ever computed, consistent with the long-standing "Arm D has no S3-S6"
-- finding). That line would error the moment this query actually ran - it
-- never had been. Replaced with the two real, uncross-checked D tables.
SELECT 'S1', 'D', s.day_index, s.statistic, l.statistic, ABS(s.statistic - l.statistic)
FROM DT_STUDY.DRIFT.signal_daily s
JOIN DT_STUDY.DRIFT.s1_arm_d l ON l.day_index = s.day_index
WHERE s.arm = 'D' AND s.signal = 'S1' AND s.strategy = 'FIXED'
UNION ALL
SELECT 'S2', 'D', s.day_index, s.statistic, l.statistic, ABS(s.statistic - l.statistic)
FROM DT_STUDY.DRIFT.signal_daily s
JOIN DT_STUDY.DRIFT.s2_arm_d l ON l.day_index = s.day_index
WHERE s.arm = 'D' AND s.signal = 'S2' AND s.strategy = 'FIXED'
UNION ALL
SELECT 'S7a', 'B', s.day_index, s.statistic, l.statistic, ABS(s.statistic - l.statistic)
FROM DT_STUDY.DRIFT.signal_daily s
JOIN DT_STUDY.DRIFT.s7a_arm_b l ON l.day_index = s.day_index
WHERE s.arm = 'B' AND s.signal = 'S7a' AND s.strategy = 'FIXED'
UNION ALL
SELECT 'S7a', 'D', s.day_index, s.statistic, l.statistic, ABS(s.statistic - l.statistic)
FROM DT_STUDY.DRIFT.signal_daily s
JOIN DT_STUDY.DRIFT.s7a_arm_d l ON l.day_index = s.day_index
WHERE s.arm = 'D' AND s.signal = 'S7a' AND s.strategy = 'FIXED'
ORDER BY 1, 2, 3;
-- Expect abs_diff ~0 (floating-point noise only) for EVERY row EXCEPT S6's
-- reference-window days (15-34) - see the LEGACY DIVERGENCE note on
-- s6_nn_distance above; S6's measurement window (35+) matches exactly like
-- everything else. S1-S5 are pure SQL/embedding math over already-loaded
-- data, so they should match exactly once Arm D's duplicate rows are fixed
-- (Section 1 dedupe). S7a depends on predicted_label_norm matching the legacy
-- column exactly, which this file's Section 4 CASE expression is written to
-- reproduce, and on DT_STUDY.DRIFT.predictions (which ref_priors reads,
-- NOT predictions_calibration) having calibration-window rows for arm B/D -
-- see the migration-repair note in classify_calibration.sql.
--
-- VALIDATED 2026-11-01 (re-run after the S3/D correction above): ran clean
-- on live data - S1, S2, S3, S4, S5, S7a (arm B) and S1, S2, S7a (arm D)
-- matched the legacy tables to floating-point precision (diffs ~1e-15 to
-- 1e-18); S6 (arm B) diverged only in the reference window, confirming the
-- self-match artifact above rather than a new bug. Arm D has no legacy
-- S3/S4/S5/S6 table to check against (never computed historically), so
-- those signals for D rest on the same pure-SQL/embedding logic validated
-- against arm B, not an independent cross-check of their own.
