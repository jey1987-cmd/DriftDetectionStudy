<p align="center">
  <img src="assets/banner.svg" alt="DriftDetectionStudy — Reproducibility Package" width="100%">
</p>

# DriftDetectionStudy — Reproducibility Package

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-4a5261?style=flat-square" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/Snowflake-Cortex%20AI-4a5261?style=flat-square" alt="Snowflake Cortex AI">
  <img src="https://img.shields.io/badge/IEEE%20Access-submitted-4a5261?style=flat-square" alt="IEEE Access: submitted">
  <img src="https://img.shields.io/badge/reproduction-~%24250-4a5261?style=flat-square" alt="Full reproduction cost ~$250">
  <img src="https://img.shields.io/badge/corpus-synthetic-4a5261?style=flat-square" alt="Corpus: synthetic">
</p>

Companion repository for:

> **What Kind of Drift Determines Which Signal Sees It: A Cost-Aware,
> Three-Arm Study of Text Classification Monitoring**
> J. Ramalingam, submitted to IEEE Access.

This repository contains the corpus generator, detection-signal SQL, the
cost-gated runners used to reproduce every paid step, the multi-seed
replication tooling added in response to reviewer feedback, and the
scripts that regenerate the paper's result tables and figures. No employer
data, systems, or taxonomy are used anywhere in this package — the corpus
is entirely synthetic and the experiments run on a personal Snowflake
account.

---

## 1. What's in this repo

| File | Purpose |
|---|---|
| `python/generate_311_corpus.py` | Synthetic municipal-311 corpus generator with drift injection (arms B, C, D); also validates the Phase 1 reference-window gate |
| `python/gen_replicates.py` | Generates N independent replicate seeds per arm (default 10 × BASE/B/C/D = 40), gated through `generate_311_corpus.py` before anything is written |
| `python/Loader.py` | Loads generated JSONL files into the Snowflake `daily_records` table; accepts base arms, replicate labels (`B_s01`), and extended-window labels (`D_ext150`) |
| `python/run_classification.py` | Cost-gated runner for the two `AI_CLASSIFY` steps (reference-window calibration and measurement-window predictions); `history`/`measure`/`run` modes, dry-run cost estimate required before any billed call |
| `python/run_embeddings.py` | Cost-gated runner for the Tier 2 embedding pipeline (`daily_sample` → `AI_EMBED` → `ref_sample_emb`) |
| `python/analyze_replicates.py` | Summarizes detection results across replicate seeds: median detection lag with a 95% bootstrap CI, fraction of seeds detected, empirical false-alarm rate |
| `python/check_threshold_normality.py` | Tests the mean+3SD threshold's normality assumption against the reference window; reports Shapiro-Wilk, QQ correlation, and the empirical-quantile alternative threshold |
| `python/make_figures.py` | Regenerates the manuscript's Figures 2 (Arm B), 3 (Arm C), 4 (Arm D) directly from live Snowflake data |
| `sql/classify_calibration.sql` | Zero-shot classification of the reference window (days 15–34) |
| `sql/drift_signals.sql` | Tier 1 (S1–S3), Tier 2 (S4–S6), and Tier 3 (S7a) signal computation; threshold policy and detection-summary views |
| `DESIGN.md` | Mid-project experimental design doc (superseded by the paper; kept for history) |
| `corpus/` | Pre-generated corpus data: single-seed arms BASE, B, C, D (90 days each), 40 replicate seeds (`<ARM>_s01`…`<ARM>_s10`), and `replicates_manifest.csv` |
| `README.md` | This file |

---

## 2. Requirements

- Python 3.9+, no external packages beyond the standard library for corpus
  generation and loading
- `numpy` and `scipy` for `check_threshold_normality.py` only (not required
  anywhere else)
- `matplotlib` for `make_figures.py` only
- A Snowflake account with Cortex AI Functions enabled (`AI_CLASSIFY`,
  `AI_EMBED`)
- `snowflake-connector-python`, and a named profile in
  `~/.snowflake/connections.toml` (account/user/role/warehouse/database/
  schema); the Snowflake password/authenticator always comes from the
  environment, never from that file or any script
- An isolated virtual warehouse (XSMALL is sufficient) — isolation matters
  for credit attribution, not raw compute needs

**Cost estimate for a full single-run, three-arm reproduction, at the
rates measured during the original study (subject to change — verify
current Snowflake AI pricing before running):**

| Item | Approx. cost |
|---|---|
| Calibration classification (40,000 calls, shared across B/D) | ~$26 |
| Each arm's measurement-window classification (110,000 calls) | ~$74 |
| All Tier 1 (SQL) and Tier 2 (embedding) signals, all arms combined | < $1 |
| **Single-run, three-arm reproduction (B, C, D)** | **~$250** |
| + Arm C's own 11-label calibration pass (40,000 calls, required — see 3.3) | **+~$32** |

A full re-run while preparing the R1 response (BASE S7a rerun, Arm C's
11-label calibration + S7a, plus the 40-replicate Tier 1/2 population)
measured **$206.81** in actual AI credits (103.4044 credits at the
**Global** on-demand rate, $2.00/credit — confirmed via
`SHOW PARAMETERS LIKE 'CORTEX_ENABLED_CROSS_REGION' IN ACCOUNT`, which
returns `ANY_REGION`, not the default `DISABLED`; Regional would be
$2.20/credit). AI credits and warehouse (compute) credits are billed
separately — the replicate study's Tier 1/2 signal population adds
**no** AI cost (S1–S5 are free/near-free) but does consume warehouse
time: populating S6 (the O(k·|reference|) pairwise-distance signal) for
all 40 replicates measured **≈5.1 warehouse credits** on an X-Small
warehouse (≈3.7 hours wall clock).

Tier 1 and Tier 2 signals alone (no classification, no S7a) reproduce the
paper's central finding — that new-content drift is detected by most
structural/embedding signals regardless of prevalence, while reassignment
drift is caught only by nearest-neighbor distance and classifier output —
at a cost of well under $1.

---

## 3. Step-by-step reproduction

### 3.1 Generate the corpus (no cloud cost)

The `corpus/` directory already ships pre-generated data for the four
single-seed arms (BASE, B, C, D) and 40 replicate seeds — you can skip
straight to [3.2](#32-load-into-snowflake). To regenerate it yourself
instead:

```bash
python python/generate_311_corpus.py --arm BASE --out ./corpus
python python/generate_311_corpus.py --arm B --magnitude medium --out ./corpus
python python/generate_311_corpus.py --arm C --magnitude medium --out ./corpus
python python/generate_311_corpus.py --arm D --magnitude medium --out ./corpus
```

Each arm produces 90 daily JSONL files (`day_000.jsonl` … `day_089.jsonl`)
under `./corpus/<arm>/`. Days 0–34 are byte-identical across all four arms
by construction, and the default `SEED` (20260821) must stay unchanged
for this to hold.

**Validate before loading anything:**

```bash
python python/generate_311_corpus.py --validate ./corpus/BASE
```

Expect `GATE PASSED`. If any Tier 1 signal fires during the reference
window (days 15–34), the corpus is not stable and must not be used —
every downstream detection-lag result depends on this window being clean.

**Replicate seeds** (for confidence intervals across independent
realizations — see [Section 6](#6-multi-seed-replication-study)):

```bash
python python/gen_replicates.py                              # 10 seeds x BASE,B,C,D, medium -> 40 replicates
python python/gen_replicates.py --arms D --n 5 --days 150     # longer window, one arm only (D_ext150)
```

Writes `./corpus/<arm>_s<NN>/` plus `./corpus/replicates_manifest.csv`
(one row per pair, with its gate result). Never touches `corpus/BASE`,
`corpus/B`, `corpus/C`, `corpus/D`. Exits non-zero if any replicate fails
to generate or fails the Phase 1 gate — check the manifest and do not
load a failed replicate.

### 3.2 Load into Snowflake

Create the target objects:

```sql
CREATE DATABASE IF NOT EXISTS DT_STUDY;
CREATE SCHEMA IF NOT EXISTS DT_STUDY.DRIFT;
CREATE WAREHOUSE IF NOT EXISTS DRIFT_WH
  WAREHOUSE_SIZE = 'XSMALL' AUTO_SUSPEND = 60 AUTO_RESUME = TRUE;

CREATE OR REPLACE TABLE DT_STUDY.DRIFT.daily_records (
    record_id   STRING,
    arm         STRING,
    day_index   INT,
    text        STRING,
    true_label  STRING,
    load_ts     TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);
```

Load each arm with `Loader.py` (accepts a base arm, a replicate label like
`B_s01`, or an extended-window label like `D_ext150`):

```bash
python python/Loader.py --corpus-dir ./corpus --arm BASE --connection study_v1
python python/Loader.py --corpus-dir ./corpus --arm all  --connection study_v1   # every arm found
```

Confirm after each load:

```sql
SELECT arm, MIN(day_index), MAX(day_index),
       COUNT(DISTINCT day_index), COUNT(*)
FROM DT_STUDY.DRIFT.daily_records
WHERE arm = '<ARM>'
GROUP BY arm;
```

Expect 90 days, 180,000 rows, per single-seed arm (90 days, 18,000
rows/day/arm; replicate seeds are the same shape).

### 3.3 Reference-window classification (run once per arm family)

```bash
# 1. Measure real tokens/call first - never skip straight to a full run
python python/run_classification.py --mode measure --step calibration \
    --arm BASE --labels bare --limit 50 --connection study_v1 --confirm

# 2. Run the full calibration step with that measured figure
python python/run_classification.py --mode run --step calibration \
    --arm BASE --labels bare --tokens-per-call <measured> \
    --connection study_v1 --confirm
```

This uses the bare 10-label array. **Do not use per-label descriptions** —
an earlier configuration using them was measured to increase per-call
token consumption by close to two orders of magnitude with no accuracy
benefit (see Section III-D of the paper).

**Arm C needs its own 11-label calibration pass.** Arm C's measurement
window introduces a novel 11th class (`sidewalk_obstruction`) that is not
in the shared 10-label array; without a separate Arm-C-specific
calibration run using `--labels bare11`, the reference prior for the
novel class falls back to a near-zero constant for *every* label (not
just the new one), corrupting the S7a chi-square statistic entirely —
`ref_priors`/`s7a_prior_shift` are arm-partitioned, so Arm C's zero
existing calibration rows is a real gap, not a rounding detail (see
Section V-B of the paper). Use `--calibration-only` to scope the write to
days 15–34 only, leaving the measurement window untouched:

```bash
python python/run_classification.py --mode run --step predictions \
    --arm C --labels bare11 --calibration-only --tokens-per-call <measured> \
    --connection study_v1 --confirm
```

`--labels bare11` is refused for any arm other than `C`.

### 3.4 Tier 1 signals (S1, S2, S3) — free, per arm

Run the tokenization, S1 (Kolmogorov–Smirnov on length), S2 (vocabulary
novelty), and S3 (type-token ratio deviation) blocks in
`sql/drift_signals.sql`, substituting the target arm in each
`WHERE arm = '<ARM>'` clause. No Cortex cost; near-instant per arm.

### 3.5 Tier 2 signals (S4, S5, S6) — near-free, per arm

```bash
python python/run_embeddings.py --arm C --connection study_v1              # dry run
python python/run_embeddings.py --arm C --connection study_v1 --confirm    # real run
```

Populates `daily_sample` (200 records/day, deterministic), `embeddings`
(`AI_EMBED`), and `ref_sample_emb` (2,000-row reference cap). Then run the
S4/S5/S6 blocks in `sql/drift_signals.sql`. Confirm row counts at each
checkpoint before proceeding — an empty intermediate table is a common
failure mode and is cheaper to catch early than at the final signal
computation. S6 is the most expensive signal in **warehouse** (not AI)
time — an O(k·|reference|) pairwise join — budget accordingly for large
replicate batches.

### 3.6 Tier 3 signal (S7a) — the paid step

```bash
python python/run_classification.py --mode run --step predictions \
    --arm B --labels bare --measurement-only --tokens-per-call <measured> \
    --connection study_v1 --confirm
```

`--measurement-only` scopes both the row-count estimate and the
DELETE+INSERT to `day_index BETWEEN 35 AND 89`, so a re-run never
overwrites the (already-paid-for) calibration window. For Arm C, use
`--labels bare11` here too — the measurement window must be classified
with the same label configuration as its calibration pass, or the
reference prior reflects a different classifier than what it's compared
against.

### 3.7 Threshold and detection

One policy applied to every signal, every arm (`signal_fired` /
`detection_summary` views in `sql/drift_signals.sql`):

```sql
-- fires on day d if statistic(d) > mean(ref) + 3*stdev(ref)
-- AND statistic(d-1) also exceeded that threshold
```

Reference window is days 15–34 for all signals, estimated independently
**per arm** (per seed, for replicates) from that arm's own 20 reference
days — this is a real monitor's only available history, so replicate
analysis keeps per-seed thresholds as the primary result rather than
pooling across seeds (see [Section 6](#6-multi-seed-replication-study)
for the pooled-threshold robustness check, which is reported separately).
For S7a, use the true reference-window predictions, not an
early-measurement-window proxy — see Section V-B of the paper for the
failure mode this avoids.

To check the mean+3SD threshold's normality assumption instead of just
asserting it:

```bash
python python/check_threshold_normality.py signal_daily_ref_window.csv --out normality_report.md
```

Input is a CSV export of `signal_daily` restricted to `day_index BETWEEN
15 AND 34` (columns: `arm, day_index, signal, strategy, statistic`).

---

## 4. Expected results (single realization per arm)

| Signal | Arm B (100%) | Arm C (≈15%) | Arm D (≤30%) |
|---|---|---|---|
| S1 | not detected | not detected | not detected |
| S2 | 1 day | 1 day | not detected |
| S3 | 1 day | 1 day | not detected |
| S4 | 1 day | 6 days | not detected |
| S5 | 1 day | 9 days | not detected |
| S6 | 1 day | 1 day | not detected |
| S7a | not detected | 1 day | 34 days |

Arm C's S4 lag (6 days) reflects a **borderline, consecutive-day-rule
statistic**: it crosses threshold on day 37, falls back on day 38, then
fires on days 40–41. This was recalibrated during the R1 response along
with Arm C's S7a reference-window statistics, which are **regenerated and
not reproducible from the published numbers**:

| | mean | SD | threshold (mean+3SD) |
|---|---|---|---|
| Published | 7.97 | 3.46 | 18.4 |
| Current (this repo) | 7.7987 | 3.2769 | 17.6294 |

If your reproduction diverges meaningfully from this table, check first
whether the reference window (days 15–34) is genuinely stable in your
generated corpus, and second whether the classifier label array matches
exactly (bare 10 labels for B/D, bare 11 labels for C's own calibration
and measurement window, no descriptions, with the two renamed labels
documented in Section III-D).

---

## 5. Figures

```bash
python python/make_figures.py --connection study_v1 --outdir ./figures
python python/make_figures.py --connection study_v1 --outdir ./figures --only fig3   # one at a time
```

Regenerates Figures 2 (Arm B), 3 (Arm C), 4 (Arm D) — each signal's
statistic normalized to its own threshold, onset-day and threshold-crossing
markers, S7a (Fig. 3) and classification accuracy (Fig. 4) on a secondary
axis. No plotting script existed in this repo before the R1 response;
colors were sampled from the originally published figures (a Material
Design palette) to keep the two visually consistent. Pixel dimensions
match the published figures exactly (2361×952, 2044×954, 2524×1072 @ 300
dpi) so a regenerated image can be dropped straight into the manuscript.

---

## 6. Multi-seed replication study

Reviewers asked for confidence intervals rather than a single-draw lag
number. `gen_replicates.py` produces 10 independent seeds per arm (40
corpora total); after loading and running Tiers 1–2 for all 40 (Tier 3/S7a
was **not** re-run per replicate — at ~$87/arm it was judged
cost-prohibitive to run across 10 seeds × 4 arms, so S7a claims stay
single-realization), `analyze_replicates.py` reports median detection lag
with a bootstrap 95% CI, fraction of seeds detected, and the empirical
false-alarm rate:

```bash
python python/analyze_replicates.py detections.csv --out table6 --latex
```

**Summary (10 seeds per arm, S1–S6):**

| Arm | Signal | Detected | Median lag (95% CI) |
|---|---|---|---|
| BASE | S1,S2,S4,S5,S6 | 0/10 | not detected (clean control) |
| BASE | S3 | 1/10 | 7d (single event, day 42, ~2% over threshold) |
| B | S1 | 1/10 | 28d (n=1, no CI) |
| B | S2–S6 | 10/10 | 1d (1–1) |
| C | S1 | 3/10 | 10d (3–10) |
| C | S2, S3, S6 | 10/10 | 1d (1–1) |
| C | S4 | 10/10 | 1d (1–2) |
| C | S5 | 10/10 | 1d (1–4) |
| D | S1–S3 | 0/10 | not detected |
| D | S4, S5 | 1/10 | 32d (single realization, days 67–68) |
| D | S6 | 1/10 | 50d (single realization, day 85, 0.84% over threshold) |

**Stationary control vs. Arm D's occasional events.** Pooling all six
Tier 1/2 signals, the control produced 1 false alarm in 60 signal-realizations
(550 signal-days each); Arm D produced 3 (S4, S5, S6 — S4/S5 fired together
in the same realization, so they are not independent events). A two-sided
Fisher exact test on [[1,59],[3,57]] gives **p ≈ 0.62**: Arm D's occasional
S4–S6 events are not statistically distinguishable from the stationary
false-alarm rate.

**Extended window (`D_ext150`, one realization, 150 days).** Arm D's
substitution rate reaches its ceiling on day 75; a single realization was
extended 60 days past that point. None of S1–S6 met the consecutive-day
rule at any point — S1, S3, S4 each poked above threshold on isolated
days by margins of 1–15%, S2/S5/S6 never exceeded theirs at all. This
non-detection is therefore not an artifact of a truncated window.

**Threshold robustness.** Each realization's own 20-day reference window
is noisy; re-scoring with a threshold pooled across all 10 realizations
(200 reference points) — either mean+3SD or the 99.865th percentile —
removes the stationary control's S3 false alarm, Arm B's single S1 event,
and Arm D's S4/S5 event. Arm D's S6 event (day 85) is removed by the
pooled parametric threshold but persists under the percentile threshold,
so it is method-sensitive, not dismissible as pure noise. The same check
reduces Arm C's S1 detections from 3/10 (per-seed threshold, the primary
result) to 2/10 (pooled parametric) to 0/10 (pooled percentile). None of
this changes the main findings — it is reported as a robustness check
alongside the per-seed results, not a replacement for them.

---

## 7. Notes on independence

This corpus, taxonomy, and all reported results were generated
independently on personally provisioned cloud infrastructure, using
synthetic data with an invented domain and label set. No production
system, employer data, or proprietary taxonomy is represented anywhere in
this repository.

---

## 8. Citation

If you use this code or corpus generator, please cite:

```
J. Ramalingam, "What Kind of Drift Determines Which Signal Sees It:
A Cost-Aware, Three-Arm Study of Text Classification Monitoring,"
IEEE Access, [year, volume, pages once assigned].
```
