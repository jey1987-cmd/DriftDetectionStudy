-- ============================================================================
-- CLASSIFICATION WITH LABEL DESCRIPTIONS
-- ============================================================================
-- Prompted by a 52% construction_complaint -> noise_complaint misclassification
-- rate observed with bare label strings.  Descriptions give the model an
-- explicit basis to distinguish "disturbance FROM construction activity"
-- (construction_complaint) from "disturbance NOT related to construction"
-- (noise_complaint) - the two were previously indistinguishable to it because
-- both are, at root, complaints about unwanted sound.
--
-- Adding descriptions raises input tokens per call (label text is now sent on
-- every classification, not just once).  Re-run the 50-call cost check after
-- this to get an updated tokens/call and credits/call before committing to the
-- full calibration + arm B run.
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
    arm             STRING,
    day_index       INT,
    record_id       STRING,
    true_label      STRING,
    predicted_label STRING
);

DELETE FROM DT_STUDY.DRIFT.predictions_calibration WHERE arm = '<ARM>';

INSERT INTO DT_STUDY.DRIFT.predictions_calibration
SELECT arm, day_index, record_id, true_label,
       AI_CLASSIFY(
           text,
           [
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
           ]
       ):labels[0]::VARCHAR AS predicted_label
FROM DT_STUDY.DRIFT.daily_records
WHERE arm = '<ARM>' AND day_index BETWEEN 15 AND 34;
