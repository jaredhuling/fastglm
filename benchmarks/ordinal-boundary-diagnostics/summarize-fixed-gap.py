"""Project saved 80-decimal scores onto the known row-12 boundary normal.

Run: python3 summarize-fixed-gap.py FIXED_GAP_RESULTS HIGH_PRECISION_RESULTS
"""
import csv
import json
import sys
from decimal import Decimal as D, getcontext
from pathlib import Path

getcontext().prec = 80
base, high = map(Path, sys.argv[1:])
destination = high / "projected-scores.csv"
if destination.exists():
    raise FileExistsError("Preserve the existing projection summary")
records = []
for path in sorted(base.glob("*-coefficients.csv")):
    key = path.name.removesuffix("-coefficients.csv")
    with (base / (key + "-input.csv")).open() as stream:
        row = list(csv.DictReader(stream))[11]
    z1, z2 = [D.from_float(float(value)) for value in list(row.values())[-2:]]
    normal = [-D(1), D(1), D(0), z1, z2, -z1, -z2, D(0), D(0)]
    value = json.loads((high / (key + ".json")).read_text())
    score = [D(entry) for entry in value["score"]]
    multiplier = sum(a * b for a, b in zip(normal, score)) / sum(a * a for a in normal)
    tangent = [s - multiplier * a for s, a in zip(score, normal)]
    records.append(dict(key=key, gap=value["minimum_gap"],
                        normal_multiplier=str(multiplier),
                        tangent_maximum=str(max(map(abs, tangent))),
                        raw_maximum=str(max(map(abs, score))),
                        standardized_score=value["scaled_score"]))
with destination.open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
