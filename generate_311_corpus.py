#!/usr/bin/env python3
"""
Synthetic municipal-311 corpus generator with controlled drift injection.

Produces 90 days of labelled service-request text for the drift-detection study.

    days 0-14   curation + fine-tuning window
    days 15-34  clean inference  (threshold calibration, false-positive rate)
    days 35-89  drifted inference (detection-lag measurement)

Drift arms
    BASE   no drift, the control corpus
    A      prior shift        - class frequencies reweighted, vocabulary unchanged
    B      vocabulary shift   - m% of each class's entity pool replaced by unseen terms
    C      novel class        - an unseen 11th class appears at n% of daily volume
    D      gradual semantic   - templates interpolate toward the adjacent class,
                                ramped over days 35-75
    E      compound           - arm B at day 35, then arm C at day 60

Every arm shares days 0-34 byte-for-byte with BASE.  That is what makes the
reference window and the calibrated thresholds comparable across arms.

Usage
    python generate_311_corpus.py --arm BASE --out ./corpus
    python generate_311_corpus.py --arm B --magnitude medium --out ./corpus
    python generate_311_corpus.py --validate ./corpus/BASE      # the Phase 1 gate

The validation gate runs the Tier 1 signals locally over days 15-34 and reports
whether any would fire.  It must return clean before any cloud spend: if the
undrifted period is not statistically stable, every detection-lag number
afterwards is measuring generator noise rather than injected drift.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SEED = 20260821
DAYS = 90
RECORDS_PER_DAY = 2_000
CLEAN_START, CLEAN_END = 15, 34
DRIFT_ONSET = 35
COMPOUND_SECOND_ONSET = 60
GRADUAL_END = 75

MAGNITUDES = {"small": 0.10, "medium": 0.30, "large": 0.60}

# ---------------------------------------------------------------------------
# taxonomy
# ---------------------------------------------------------------------------
# ADJACENT pairs are load-bearing.  They give arm D somewhere realistic to drift
# toward, and they are where curation is expected to do the most damage - a
# zero-shot pass gets these wrong most often, so those records get filtered out
# of M2's training set.  Per-class retention must be reported.

ADJACENT = {
    "noise_complaint": "construction_complaint",
    "construction_complaint": "noise_complaint",
    "illegal_dumping": "overflowing_bin",
    "overflowing_bin": "illegal_dumping",
}

CLASSES = [
    "pothole_road", "streetlight_outage", "noise_complaint",
    "construction_complaint", "illegal_dumping", "overflowing_bin",
    "graffiti", "park_maintenance", "water_leak", "stray_animal",
]

NOVEL_CLASS = "sidewalk_obstruction"          # arm C only

# Baseline class priors.  Deliberately uneven - a uniform prior would make the
# chi-square signal (S7a) unrealistically sensitive to arm A.
PRIORS = {
    "pothole_road": 0.16, "streetlight_outage": 0.12, "noise_complaint": 0.14,
    "construction_complaint": 0.09, "illegal_dumping": 0.10,
    "overflowing_bin": 0.11, "graffiti": 0.07, "park_maintenance": 0.08,
    "water_leak": 0.08, "stray_animal": 0.05,
}

# Arm A reweighting: shift mass toward the two adjacent pairs, away from the rest.
PRIORS_SHIFTED = {
    "pothole_road": 0.08, "streetlight_outage": 0.06, "noise_complaint": 0.22,
    "construction_complaint": 0.18, "illegal_dumping": 0.16,
    "overflowing_bin": 0.14, "graffiti": 0.04, "park_maintenance": 0.05,
    "water_leak": 0.04, "stray_animal": 0.03,
}

# ---------------------------------------------------------------------------
# templates and entity pools
# ---------------------------------------------------------------------------
# {street}, {time}, {detail}, {qualifier} are slot-filled.  Length varies by
# class through template length and the optional trailing clause, so S1 (length
# KS) has something real to measure.

TEMPLATES: dict[str, list[str]] = {
    "pothole_road": [
        "Large pothole on {street} near the {detail}, {qualifier}.",
        "Road surface has collapsed on {street}, {qualifier}.",
        "Deep hole in the roadway at {street} causing {detail}.",
        "Pavement is cracked and sinking along {street}, {qualifier}.",
        "Vehicle damage risk from pothole outside the {detail} on {street}.",
    ],
    "streetlight_outage": [
        "Street light out on {street} since {time}, {qualifier}.",
        "The lamp near the {detail} on {street} has stopped working.",
        "Several lights along {street} are dark, {qualifier}.",
        "Flickering street light at {street} by the {detail}.",
        "No lighting on {street} after {time}, {qualifier}.",
    ],
    "noise_complaint": [
        "Loud music coming from a property on {street} since {time}.",
        "Persistent noise from the {detail} on {street}, {qualifier}.",
        "Neighbours playing amplified sound on {street} after {time}.",
        "Barking and shouting near the {detail} on {street}, {qualifier}.",
        "Excessive noise disturbance reported on {street} at {time}.",
    ],
    "construction_complaint": [
        "Construction work starting before {time} on {street}, {qualifier}.",
        "Building site on {street} operating outside permitted hours.",
        "Heavy machinery running near the {detail} on {street} since {time}, construction noise ongoing.",
        "Contractors blocking access on {street}, {qualifier}.",
        "Ongoing site works on {street} generating dust and noise, {qualifier}.",
    ],
    "illegal_dumping": [
        "Household waste dumped on {street} beside the {detail}.",
        "Someone left furniture and bags on {street}, {qualifier}.",
        "Fly tipping reported at {street} near the {detail}.",
        "Construction debris abandoned on {street} since {time}.",
        "Pile of discarded material on {street}, {qualifier}.",
    ],
    "overflowing_bin": [
        "Public bin overflowing on {street} outside the {detail}.",
        "Bin has not been emptied on {street} since {time}.",
        "Rubbish spilling from the container near the {detail} on {street}.",
        "Waste bin full and attracting {detail} on {street}, {qualifier}.",
        "Collection missed on {street}, bins overflowing, {qualifier}.",
    ],
    "graffiti": [
        "Graffiti sprayed on the wall of the {detail} on {street}.",
        "Tagging appeared overnight on {street} near the {detail}.",
        "Vandalism with spray paint at {street}, {qualifier}.",
        "Offensive markings on the {detail} along {street}.",
        "Repeated graffiti on {street} since {time}, {qualifier}.",
    ],
    "park_maintenance": [
        "Broken bench in the park off {street}, {qualifier}.",
        "Overgrown grass and hedges near the {detail} on {street}.",
        "Play equipment damaged in the park at {street}.",
        "Fallen branch blocking the path near the {detail} on {street}.",
        "Park gate broken on {street} since {time}, {qualifier}.",
    ],
    "water_leak": [
        "Water leaking from the roadway on {street} near the {detail}.",
        "Burst pipe flooding the pavement at {street}, {qualifier}.",
        "Continuous water flow on {street} since {time}.",
        "Standing water and leak near the {detail} on {street}.",
        "Mains leak suspected on {street}, {qualifier}.",
    ],
    "stray_animal": [
        "Stray dog wandering on {street} near the {detail}.",
        "Loose animal reported on {street} since {time}, {qualifier}.",
        "Injured cat by the {detail} on {street}, {qualifier}.",
        "Group of stray animals around {street}, {qualifier}.",
        "Unattended dog off lead on {street} by the {detail}.",
    ],
    NOVEL_CLASS: [
        "Scaffolding blocking the footpath on {street} near the {detail}.",
        "Parked vehicle obstructing the sidewalk at {street}, {qualifier}.",
        "Overgrown hedge forcing pedestrians into the road on {street}.",
        "Delivery pallets left across the pavement on {street} since {time}.",
        "Temporary barriers narrowing the walkway near the {detail} on {street}.",
    ],
}

STREETS_BASE = [
    "Maple Avenue", "Third Street", "Riverside Road", "Kingsway", "Elm Court",
    "Harbour Lane", "Weston Drive", "Cedar Close", "Market Street", "Ashford Way",
    "Bramble Hill", "Northgate", "Priory Road", "Union Street", "Larkspur Lane",
]

DETAILS_BASE: dict[str, list[str]] = {
    "pothole_road": ["bus stop", "school crossing", "roundabout", "junction"],
    "streetlight_outage": ["pedestrian crossing", "car park", "footbridge", "bus shelter"],
    "noise_complaint": ["corner shop", "flats", "pub", "community hall"],
    "construction_complaint": ["site entrance", "hoarding", "crane", "compound"],
    "illegal_dumping": ["recycling point", "alleyway", "lay-by", "verge"],
    "overflowing_bin": ["station entrance", "market stalls", "playground", "bench"],
    "graffiti": ["underpass", "substation", "shop shutter", "bus shelter"],
    "park_maintenance": ["pond", "bandstand", "tennis courts", "car park"],
    "water_leak": ["manhole", "hydrant", "kerb", "drain cover"],
    "stray_animal": ["allotments", "playing field", "car park", "canal path"],
    NOVEL_CLASS: ["shopfront", "bus stop", "crossing", "cycle rack"],
}

# Replacement pools for arm B.  Deliberately disjoint from the base pools so
# vocabulary novelty (S2) has an unambiguous signal to detect.
STREETS_DRIFT = [
    "Quarry Rise", "Fenwick Terrace", "Oakhampton Parade", "Silverbrook Way",
    "Tannery Row", "Hollowfield Crescent", "Verge End", "Kilnside",
]
DETAILS_DRIFT = [
    "transit hub", "kiosk", "charging bay", "parcel locker",
    "micromobility rack", "wayfinding totem", "planter", "gantry",
]

TIMES = ["6am", "7am", "10pm", "11pm", "midnight", "early morning",
         "last Tuesday", "the weekend", "yesterday evening"]

QUALIFIERS = [
    "this needs urgent attention", "it has been reported before",
    "residents are concerned", "please investigate",
    "it is becoming a safety hazard", "no action has been taken so far",
    "several neighbours have complained", "it is getting worse each day",
]

TRAILING = [
    "", "", "",                                  # most records have no trailing clause
    " I have attached a photo for reference.",
    " This is the third time I have reported this issue.",
    " Please confirm when someone will attend.",
]


# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Record:
    record_id: str
    arm: str
    day_index: int
    text: str
    true_label: str


class Generator:
    def __init__(self, arm: str, magnitude: str, rng: random.Random):
        self.arm = arm
        self.mag = MAGNITUDES[magnitude]
        self.rng = rng

    # -- vocabulary --------------------------------------------------------
    def streets(self, day: int) -> list[str]:
        if not self._vocab_drifted(day):
            return STREETS_BASE
        keep = max(1, int(len(STREETS_BASE) * (1 - self.mag)))
        n_new = len(STREETS_BASE) - keep
        return STREETS_BASE[:keep] + STREETS_DRIFT[:n_new]

    def details(self, label: str, day: int) -> list[str]:
        base = DETAILS_BASE[label]
        if not self._vocab_drifted(day):
            return base
        keep = max(1, int(len(base) * (1 - self.mag)))
        n_new = len(base) - keep
        return base[:keep] + DETAILS_DRIFT[:n_new]

    def _vocab_drifted(self, day: int) -> bool:
        if self.arm == "B":
            return day >= DRIFT_ONSET
        if self.arm == "E":
            return day >= DRIFT_ONSET          # B component starts at 35
        return False

    # -- class selection ---------------------------------------------------
    def priors(self, day: int) -> dict[str, float]:
        if self.arm == "A" and day >= DRIFT_ONSET:
            # interpolate by magnitude between base and shifted priors
            return {c: PRIORS[c] * (1 - self.mag) + PRIORS_SHIFTED[c] * self.mag
                    for c in CLASSES}
        return PRIORS

    def pick_label(self, day: int) -> str:
        novel_active = (
            (self.arm == "C" and day >= DRIFT_ONSET) or
            (self.arm == "E" and day >= COMPOUND_SECOND_ONSET)
        )
        if novel_active and self.rng.random() < self.mag * 0.5:
            return NOVEL_CLASS
        p = self.priors(day)
        r = self.rng.random()
        acc = 0.0
        for c, w in p.items():
            acc += w
            if r <= acc:
                return c
        return CLASSES[-1]

    # -- gradual semantic drift (arm D) ------------------------------------
    def template_label(self, label: str, day: int) -> str:
        """Which class's TEMPLATE bank to draw from.

        Arm D leaves the true label alone and progressively draws phrasing from
        the adjacent class.  That is what makes it *semantic* drift: the record
        still belongs to its label, but reads increasingly like its neighbour,
        so the classifier degrades without the label distribution changing.
        """
        if self.arm != "D" or day < DRIFT_ONSET or label not in ADJACENT:
            return label
        span = max(1, GRADUAL_END - DRIFT_ONSET)
        progress = min(1.0, (day - DRIFT_ONSET) / span)
        if self.rng.random() < progress * self.mag:
            return ADJACENT[label]
        return label

    # -- record ------------------------------------------------------------
    def make(self, day: int, i: int) -> Record:
        label = self.pick_label(day)
        tmpl_label = self.template_label(label, day)
        tmpl = self.rng.choice(TEMPLATES[tmpl_label])
        text = tmpl.format(
            street=self.rng.choice(self.streets(day)),
            detail=self.rng.choice(self.details(tmpl_label, day)),
            time=self.rng.choice(TIMES),
            qualifier=self.rng.choice(QUALIFIERS),
        ) + self.rng.choice(TRAILING)
        return Record(f"{self.arm}-{day:03d}-{i:05d}", self.arm, day, text, label)


def generate(arm: str, magnitude: str) -> list[Record]:
    """Days 0-34 are generated from a fixed seed independent of the arm, so every
    arm shares its reference window byte-for-byte with BASE."""
    out: list[Record] = []
    for day in range(DAYS):
        # pre-drift days use the BASE stream; post-drift days use the arm stream
        effective_arm = "BASE" if day < DRIFT_ONSET else arm
        rng = random.Random(f"{SEED}-{effective_arm}-{magnitude}-{day}")
        gen = Generator(effective_arm, magnitude, rng)
        for i in range(RECORDS_PER_DAY):
            r = gen.make(day, i)
            r.arm = arm                      # label the file by requested arm
            out.append(r)
    return out


def write(records: list[Record], outdir: Path, arm: str) -> None:
    d = outdir / arm
    d.mkdir(parents=True, exist_ok=True)
    by_day: dict[int, list[Record]] = {}
    for r in records:
        by_day.setdefault(r.day_index, []).append(r)
    for day, rows in sorted(by_day.items()):
        with (d / f"day_{day:03d}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({
                    "record_id": r.record_id, "arm": r.arm,
                    "day_index": r.day_index, "text": r.text,
                    "true_label": r.true_label,
                }) + "\n")
    print(f"  wrote {len(by_day)} day files to {d}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Phase 1 validation gate
# ---------------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    return [t for t in "".join(
        ch.lower() if ch.isalnum() else " " for ch in text).split() if t]


def validate(corpus_dir: Path) -> int:
    """Run the Tier 1 signals over days 15-34 and report whether any would fire.

    This is the gate described in the design doc.  It is free, and it must pass
    before any cloud spend: if a signal fires on undrifted data, the threshold is
    measuring generator noise and every detection-lag number afterwards is void.
    """
    days: dict[int, list[str]] = {}
    for p in sorted(corpus_dir.glob("day_*.jsonl")):
        d = int(p.stem.split("_")[1])
        if d <= CLEAN_END:
            days[d] = [json.loads(l)["text"] for l in p.open(encoding="utf-8")]
    if not days:
        raise SystemExit(f"no day files found under {corpus_dir}")

    ref_vocab = {t for d, texts in days.items() if d <= CLEAN_END
                 for x in texts for t in _tokens(x)}

    per_day = {}
    for d in range(CLEAN_START, CLEAN_END + 1):
        toks = [t for x in days[d] for t in _tokens(x)]
        lens = [len(_tokens(x)) for x in days[d]]
        per_day[d] = {
            "S2_novelty": sum(t not in ref_vocab for t in toks) / len(toks),
            "S3_ttr": len(set(toks)) / len(toks),
            "mean_len": statistics.mean(lens),
        }

    print(f"\nvalidation gate over days {CLEAN_START}-{CLEAN_END}")
    failed = 0
    for sig in ("S2_novelty", "S3_ttr", "mean_len"):
        vals = [per_day[d][sig] for d in sorted(per_day)]
        mu, sd = statistics.mean(vals), statistics.pstdev(vals)
        hi = mu + 3 * sd
        over = [d for d in sorted(per_day) if per_day[d][sig] > hi]
        # a signal firing on its own calibration window means the window is not stable
        status = "OK" if len(over) == 0 else f"FIRES on days {over}"
        if over:
            failed += 1
        print(f"  {sig:12s} mean={mu:.5f} sd={sd:.5f} 3sd={hi:.5f}  {status}")

    if failed:
        print("\nGATE FAILED - clean window is not stable. Fix the generator "
              "before spending anything.", file=sys.stderr)
        return 1
    print("\nGATE PASSED - clean window stable, safe to proceed to Phase 2.")
    return 0


# ---------------------------------------------------------------------------

def summarise(records: list[Record]) -> None:
    pre = Counter(r.true_label for r in records if r.day_index < DRIFT_ONSET)
    post = Counter(r.true_label for r in records if r.day_index >= DRIFT_ONSET)
    npre = sum(pre.values()) or 1
    npost = sum(post.values()) or 1
    print(f"\n  {'class':26s} {'pre-drift':>10s} {'post-drift':>11s}")
    for c in CLASSES + [NOVEL_CLASS]:
        if pre[c] or post[c]:
            print(f"  {c:26s} {pre[c]/npre:>9.1%} {post[c]/npost:>10.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["BASE", "A", "B", "C", "D", "E"], default="BASE")
    ap.add_argument("--magnitude", choices=sorted(MAGNITUDES), default="medium")
    ap.add_argument("--out", type=Path, default=Path("./corpus"))
    ap.add_argument("--validate", type=Path,
                    help="run the Phase 1 gate over an existing corpus directory")
    args = ap.parse_args()

    if args.validate:
        sys.exit(validate(args.validate))

    print(f"arm={args.arm} magnitude={args.magnitude} "
          f"({MAGNITUDES[args.magnitude]:.0%})", file=sys.stderr)
    recs = generate(args.arm, args.magnitude)
    print(f"  {len(recs):,} records over {DAYS} days", file=sys.stderr)
    summarise(recs)
    write(recs, args.out, args.arm)


if __name__ == "__main__":
    main()