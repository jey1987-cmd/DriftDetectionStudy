# Drift Detection Study — Experimental Design (v3)

> **Note:** this is a mid-project planning document, not the final
> methodology. Some content here (e.g., a three-model fine-tuning track)
> was later descoped, and some early estimates in this file (e.g., signal
> detection lags) were superseded by measured results. The authoritative
> methodology and results are in the paper itself and in `README.md`.

**Question.** For text classification workloads running in a cloud data warehouse,
which drift signal detects an input-distribution shift earliest and most reliably,
what does each cost per day to operate, and how much accuracy has already been
lost by the time it fires?

**Secondary question, new in v3.** Does curating the fine-tuning set — filtering
out records a zero-shot pass gets wrong or is unsure about — make the resulting
model *more brittle* under drift? A tighter fit to clean data is a plausible
mechanism for faster degradation, and it has not been measured.

**Why it is worth a paper.** Drift detection for text is well covered in the ML
literature, but almost always evaluated on detection accuracy alone. Nobody has
published the cost dimension for warehouse-native execution, where an embedding
call is a metered operation and a SQL aggregate is nearly free. The practically
interesting result would be a cheap structural signal detecting shift within a day
or two of an expensive embedding-based one.

**Independence.** Everything runs on a personal Snowflake account against a
synthetic municipal-311 corpus generated from an invented taxonomy. No employer
data, systems, taxonomy, configuration, or findings are involved.

**API currency.** The study uses the AISQL family — `AI_CLASSIFY` and `AI_EMBED`.
The older `SNOWFLAKE.CORTEX.CLASSIFY_TEXT` is documented as legacy and scheduled
for deprecation by end of 2026; building on it would date the paper on arrival.

---

## 1. Domain and taxonomy

Municipal 311 service requests. Chosen for public familiarity, clean class
separability, and the presence of genuinely confusable pairs.

| Class | Adjacent to |
| --- | --- |
| `pothole_road` | — |
| `streetlight_outage` | — |
| `noise_complaint` | `construction_complaint` |
| `construction_complaint` | `noise_complaint` |
| `illegal_dumping` | `overflowing_bin` |
| `overflowing_bin` | `illegal_dumping` |
| `graffiti` | — |
| `park_maintenance` | — |
| `water_leak` | — |
| `stray_animal` | — |

The two adjacent pairs are load-bearing: they give the gradual-drift arm somewhere
realistic to go, and they are where curation is expected to do the most damage
(§3.2).

---

## 2. Timeline

| Days | Phase | Purpose |
| --- | --- | --- |
| 0–14 | Curation + fine-tuning | Zero-shot pass, curation, train two models |
| 15–34 | Clean inference | Threshold calibration, false-positive measurement, baseline accuracy |
| 35–89 | Drifted inference | Detection-lag measurement |

- Volume fixed at 2,000 records/day. Holding volume constant isolates
  distribution shift from volume shift — state as a scope limit.
- Drift injected at **day 35**. Optional compound arm adds a second, different
  shift at **day 60**.
- Training-to-horizon ratio ≈ 1:5, reproducing the common production pattern of
  training on a bounded window and inferring over a much longer one.

### 2.1 Reference windows

- **Prediction-based signals (Tier 3)** reference **days 15–34** only. Both models
  have fitted days 0–14, so baseline confidence computed there would be inflated
  by memorisation and the threshold too loose.
- **Structural and embedding signals (Tiers 1–2)** reference **days 0–34**. These
  do not involve either model, so the wider window gives a fuller vocabulary and a
  tighter length distribution.

---

## 3. Model track (new in v3)

### 3.1 Three classifiers run in parallel

All three see identical daily inputs across days 15–89. Every detector runs
against each, so the study reports drift behaviour per model as well as per signal.

| ID | Classifier | Purpose |
| --- | --- | --- |
| **M0** | Zero-shot `AI_CLASSIFY` with label descriptions and examples | Control. No training, so it cannot degrade by *fitting* — it degrades only as inputs leave the label descriptions' coverage. |
| **M1** | Fine-tuned on the **full** days 0–14 set | Standard practice baseline |
| **M2** | Fine-tuned on the **curated** subset of days 0–14 | The brittleness question |

M1 and M2 are trained to the **same set size**. If curation removes 30% of
records, M1 is downsampled to match before training. Otherwise the comparison
confounds curation with training-set size, and any difference in brittleness is
uninterpretable.

### 3.2 Curation procedure (Option B)

Curation decides **inclusion only**. Labels always come from `true_label`. The
pre-classifier never generates a training label — that would be distillation, and
it would make later accuracy loss impossible to attribute between "the world moved"
and "the pre-classifier's errors compounded".

1. Run zero-shot `AI_CLASSIFY` over all of days 0–14.
2. Compute the confidence proxy (§4.2) for each record.
3. **Include** a record in the curated set if the zero-shot prediction equals
   `true_label` **and** the confidence proxy exceeds threshold *c*.
4. Record per-class retention rate.

**The risk to watch, and to report regardless of outcome.** Curation will
preferentially drop records from the adjacent pairs — noise/construction and
dumping/bin — because those are exactly what a zero-shot pass gets wrong. If M2
never sees hard examples from those classes, its baseline accuracy there will be
poor and drift in that region becomes hard to attribute. Per-class retention is a
required reported figure, not an optional diagnostic.

*c* is a reportable parameter. Run the curation at a single threshold for the main
study; report the retention curve across *c* so a reader can see how sharp the
selection was.

### 3.3 What this buys

- **M1 vs M2** answers whether curation trades robustness for clean-data accuracy.
- **M0 vs M1** separates degradation caused by *fitting* a distribution from
  degradation caused by inputs simply becoming unfamiliar.
- Three models × seven signals gives the detectors a range of degradation
  behaviours to be judged against, rather than a single one.

---

## 4. How drift is detected

Every signal follows the same shape; only the statistic differs.

```
  reference window ──► reference artifact      (computed once)
                              │
  day d records ──► day-d statistic ──► threshold ──► fired?
```

### 4.1 Threshold policy

One policy applied uniformly, or the lag comparison is unfair:

> Signal *s* fires on day *d* if `statistic(s,d) > mean(s, days 15–34) + 3 ×
> stdev(s, days 15–34)`, **and** it also exceeded that threshold on day *d−1*.

The consecutive-day requirement costs one day of lag and removes most single-day
noise spikes. Report at both 2σ and 3σ — lag and false-positive rate trade against
each other and a single operating point hides that.

### 4.2 Confidence proxy

Neither `AI_CLASSIFY` nor `CLASSIFY_TEXT` returns a calibrated confidence score.
S7b and the curation step therefore use:

> **confidence(record) = max cosine similarity between the record's `AI_EMBED`
> vector and the ten class-centroid embeddings**, where centroids are computed
> from days 0–14 and held fixed thereafter.

This is well defined and reproducible. The alternative — prompting `AI_COMPLETE`
for a self-reported confidence — is cheaper but poorly calibrated and would have
to be disclosed as such. **Decide before Phase 2; it cannot be reconstructed
later**, because the class centroids must be built from the training window.

### 4.3 Reference strategy as a variable

Every signal runs under two strategies. One extra column in the results table, no
additional corpus generation.

| Strategy | Reference | Expected behaviour |
| --- | --- | --- |
| **Fixed** | Days 15–34, constant | Statistic accumulates as the world moves away. Catches gradual drift. |
| **Sliding** | Trailing 7 days | Catches abrupt shifts fast. Nearly blind to gradual drift — each day resembles the one before. |

Expected finding: sliding wins on step changes (arms A–C), fails on gradual drift
(arm D); fixed catches everything with longer lag on abrupt shifts. If so, the
recommendation is to run both.

### 4.4 The seven signals

**Tier 1 — structural. Pure SQL, no model calls, near-zero cost.**

| ID | Signal | Statistic |
| --- | --- | --- |
| S1 | Length distribution | Two-sample KS statistic on token counts vs reference |
| S2 | Vocabulary novelty | Fraction of today's token *occurrences* absent from reference vocabulary |
| S3 | Type-token ratio | Absolute deviation from mean daily reference TTR |

**Tier 2 — embedding-based. One `AI_EMBED` call per sampled record (k = 200/day).**

| ID | Signal | Statistic |
| --- | --- | --- |
| S4 | Centroid displacement | 1 − cosine(reference centroid, day centroid) |
| S5 | Maximum mean discrepancy | Linear-kernel form = L2 distance between mean embeddings. RBF form needs ~k² evaluations/day; report which was used. |
| S6 | Nearest-neighbour distance | Mean distance from each day-*d* record to its closest reference record. Catches novel clusters that do not move the centroid. |

**Tier 3 — prediction-based. Free if inference already runs. Computed per model.**

| ID | Signal | Statistic |
| --- | --- | --- |
| S7a | Class prior shift | Chi-square on predicted-label distribution vs reference |
| S7b | Confidence degradation | Mean and 10th-percentile confidence proxy vs reference |

Tier 3 is a lagging indicator by construction: by the time predictions shift, the
model has already been wrong. Whether it lags Tiers 1–2 by enough to matter is the
empirical question.

### 4.5 Ground truth

`true_label` is carried on every record but is **never** an input to any detector.
It is used only in analysis, to compute true daily accuracy per model. In
production this column would not exist — which is the entire reason drift
detection is needed.

---

## 5. Drift arms

| Arm | Shift type | Mechanism | Onset |
| --- | --- | --- | --- |
| A | Prior shift | Reweight class frequencies; vocabulary unchanged | Step, day 35 |
| B | Vocabulary shift | Replace *m*% of per-class entity pool with unseen terms | Step, day 35 |
| C | Novel class | Introduce an unseen 11th class at *n*% of daily volume | Step, day 35 |
| D | Gradual semantic | Interpolate templates toward the adjacent class | Ramp, days 35–75 |
| E | Compound | Arm B at day 35, then arm C at day 60 | Two steps |

Three magnitudes (small / medium / large) so detection lag can be reported as a
function of shift size, not merely its presence.

---

## 6. Steps

### Phase 1 — Corpus (no cloud spend)

1. Build template banks and entity pools for the ten classes.
2. Generate days 0–89 at 2,000 records/day, fixed seed.
3. **Validation gate.** Run all Tier 1 signals locally across days 15–34 and
   confirm none would fire under the threshold policy. If the clean period is not
   stable, every detection-lag number afterwards is noise. Free, and must pass
   before any cloud spend.
4. Generate drifted arms, each sharing days 0–34 with the baseline.

### Phase 2 — Curate, train, calibrate

5. Create database, schemas, isolated warehouse.
6. Load days 0–34.
7. Embed days 0–14; compute the ten class centroids; store them fixed.
8. Run zero-shot `AI_CLASSIFY` over days 0–14; compute confidence proxy.
9. Apply curation (§3.2). Record per-class retention. Downsample the full set to
   the curated set's size.
10. Fine-tune M1 (full, downsampled) and M2 (curated). Record training cost for each.
11. Run M0, M1, M2 inference across days 15–34.
12. Compute reference artifacts: vocabulary set, length distribution, reference
    embeddings and centroid, per-model class priors and confidence baselines.
13. Compute all seven signals across days 15–34 under both reference strategies.
    Derive thresholds. Record any firings as false positives.

### Phase 3 — Daily run (per arm)

For each arm, for each day 35–89:

14. Load the day's records.
15. Tier 1 signals in SQL. Record statistic and credits.
16. Embed the fixed k = 200 sample; Tier 2 signals. Record statistic, credits, tokens.
17. Run M0, M1, M2 inference on all records. Tier 3 signals per model.
18. Compute true accuracy per model — analysis only.
19. Evaluate threshold policy under both strategies; write to `RESULT.signal_daily`.

### Phase 4 — Analysis

20. **Detection lag** per signal × arm × magnitude × strategy × model. Non-detection
    reported as "not detected", never as a large number — a censored observation is
    not a slow one and must not be averaged as though it were.
21. **False positive rate** per signal over days 15–34.
22. **Cost per day** per signal; cost per detection.
23. **Accuracy at detection** — true accuracy at the moment each signal fired,
    per model. The paper's headline metric: a detector firing after accuracy has
    dropped 15 points is not useful regardless of its lag in days.
24. **Model brittleness comparison** — accuracy decay curves for M0, M1, M2 across
    days 35–89. Does M2 degrade faster than M1? Does M0 degrade at all?
25. **Retraining policy statement** per signal: *this signal at this threshold
    triggers retraining before accuracy falls more than N points, at a monitoring
    cost of X credits per day.* The directly citable output.
26. Plot detection lag against daily cost, one point per signal, faceted by arm.

---

## 7. Design decisions needing justification in the paper

- **Equal training-set size for M1 and M2.** Without it, curation is confounded
  with set size.
- **Tier 2 sample size** k = 200/day. Run a sensitivity check at k ∈ {50, 200, 1000}
  on one arm, or a reviewer will ask.
- **Fixed daily volume.** Real workloads have volume seasonality, itself a confound.
- **Threshold sensitivity.** Report at 2σ and 3σ.
- **Curation threshold *c*.** Report the retention curve, not just the operating point.
- **One embedding model, one engine, one point in time.** Name model and version.

---

## 8. Threats to validity

- Drift is injected, not observed. Detection lag is measured against a shift whose
  timing and magnitude are known by construction. Standard for change-detection
  evaluation, but state it plainly.
- **Synthetic text has lower lexical variety than real text, which likely favours
  structural signals.** The most important limitation, and it cuts directly against
  the paper's most interesting potential finding. Do not soften it.
- Curation using the same embedding model that later powers S4–S6 and S7b creates a
  shared dependency: if that model represents the adjacent pairs poorly, both
  curation and detection are affected in correlated ways. Acknowledge explicitly.
- Single warehouse, single engine, single embedding model, single taxonomy.
- Fine-tuning is non-deterministic; report whether M1/M2 were trained once or
  averaged over seeds.

---

## 9. Scope control

Five arms × three magnitudes × 55 days × seven signals × two strategies × three
models is a very large matrix. First pass:

- **Arms B and C only**, **medium magnitude**, **both strategies**, **all seven
  signals**, **all three models**.
- Get the pipeline working end to end across the full 90 days before generating
  anything else.
- Expand to arms A, D, E and other magnitudes only if the first pass shows
  separation worth characterising.

**Cost check before committing.** Three models × 2,000 records/day × 75 days × 2
arms = 900,000 classifications, plus fine-tuning runs, plus embeddings. Run one
full day end to end, measure credits, multiply out, and set a spend limit. This is
the largest cost risk in the study and it is easy to underestimate.

---

## 10. What makes this publishable

Two contributions, either of which would carry a paper:

1. **Cost-normalised detector comparison in a warehouse-native setting** — a table
   letting a practitioner choose a monitoring strategy given a credit budget. If
   cheap structural signals are competitive with expensive embedding-based ones,
   that is useful and mildly counterintuitive; if not, that justifies a cost many
   teams currently incur on faith.
2. **Curation-brittleness trade-off** — whether filtering a fine-tuning set to
   clean examples buys accuracy at the price of faster degradation under drift.
   Unmeasured, and directly relevant to anyone deciding how to assemble training
   data.
