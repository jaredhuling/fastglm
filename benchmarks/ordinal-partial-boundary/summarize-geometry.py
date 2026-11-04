"""Describe local boundary geometry from the retained 80-decimal states.

Run: python3 summarize-geometry.py DEVELOPMENT_BASELINE HIGH_PRECISION_RESULTS
This only projects already computed adjusted scores; it does not find roots.
"""
import csv
import json
import sys
from decimal import Decimal as D, getcontext
from pathlib import Path

getcontext().prec = 80
base, high = map(Path, sys.argv[1:])
destination = high / "boundary-geometry.csv"
if destination.exists():
    raise FileExistsError("Preserve the existing geometry summary")
records = []
for link in ("logit", "probit", "cloglog"):
    key = link + "-200-dense"
    with (base / (key + "-input.csv")).open() as stream:
        rows = list(csv.DictReader(stream))
    with (base / (key + "-coefficients.csv")).open() as stream:
        beta = [D.from_float(float(value)) for value in next(csv.DictReader(stream)).values()]
    value = json.loads((high / (key + ".json")).read_text())
    score = [D(entry) for entry in value["score"]]
    gaps = []
    for i, row in enumerate(rows):
        weight, offset, category, z1, z2 = [D.from_float(float(entry)) for entry in row.values()]
        if not weight:
            continue
        eta = [beta[j] - z1 * beta[3 + 2 * j] - z2 * beta[4 + 2 * j] - offset for j in range(3)]
        gaps.extend((eta[j + 1] - eta[j], i + 1, j, int(category), z1, z2) for j in range(2))
    gap, row, j, observed, z1, z2 = min(gaps)
    normal = [D(0)] * 9
    normal[j], normal[j + 1] = -D(1), D(1)
    normal[3 + 2 * j], normal[4 + 2 * j] = z1, z2
    normal[3 + 2 * (j + 1)], normal[4 + 2 * (j + 1)] = -z1, -z2
    multiplier = sum(a * b for a, b in zip(normal, score)) / sum(a * a for a in normal)
    tangent = [s - multiplier * a for s, a in zip(score, normal)]
    records.append(dict(link=link, row=row, vanishing_category=j + 2, observed_category=observed,
                        minimum_gap=str(gap), normal_score_multiplier=str(multiplier),
                        max_absolute_raw_score=str(max(map(abs, score))),
                        max_absolute_tangent_score=str(max(map(abs, tangent)))))
with destination.open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
