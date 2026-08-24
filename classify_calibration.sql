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
-- ============================================================================

CREATE OR REPLACE TABLE DT_STUDY.DRIFT.predictions_calibration AS
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
WHERE arm = 'BASE' AND day_index BETWEEN 15 AND 34;
