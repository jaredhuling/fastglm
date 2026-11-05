"""Independent continuous boundary-score limit for the two retained branches.

Run: python3 boundary-limit.py FIXED_GAP_RESULTS HIGH_PRECISION_RESULTS NEW_OUTPUT
The algebra is in local-boundary-limit.txt. This is not a fitted estimator.
"""
import csv
import importlib.util
import json
import sys
from decimal import Decimal as D, getcontext
from fractions import Fraction as Q
from pathlib import Path

getcontext().prec = 80
oracle_path = Path("benchmarks/ordinal-partial-boundary/high-precision.py")
spec = importlib.util.spec_from_file_location("ordinal_decimal_reference", oracle_path)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
dot, matvec, inverse = oracle.dot, oracle.matvec, oracle.inverse
base, high, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=False)
ZERO, ONE = D(0), D(1)
records = []
for link in ("logit", "cloglog"):
    key = link + "-1e-08"
    data = oracle.read_matrix(base / (key + "-input.csv"))
    beta = oracle.read_matrix(base / (key + "-coefficients.csv"))[0]
    z1, z2 = data[11][-2:]
    normal = [-ONE, ONE, ZERO, z1, z2, -z1, -z2, ZERO, ZERO]
    rational_normal, rational_beta = list(map(Q, normal)), list(map(Q, beta))
    gap = sum(a * b for a, b in zip(rational_normal, rational_beta))
    square = sum(a * a for a in rational_normal)
    rational_beta = [b - a * gap / square for a, b in zip(rational_normal, rational_beta)]
    assert sum(a * b for a, b in zip(rational_normal, rational_beta)) == 0
    beta0 = [D(value.numerator) / D(value.denominator) for value in rational_beta]
    information = [[ZERO] * 9 for _ in range(9)]
    ordinary = [ZERO] * 9
    saved = []
    exact_gaps = []
    noncollapsed_patterns = []
    for index, (weight, offset, category, zz1, zz2) in enumerate(data):
        if not weight:
            continue
        design = []
        for j in range(3):
            row = [ZERO] * 9
            row[j], row[3 + 2 * j], row[4 + 2 * j] = ONE, -zz1, -zz2
            design.append(row)
        rational_eta = [sum(Q(a) * b for a, b in zip(row, rational_beta)) - Q(offset) for row in design]
        for j in range(2):
            current_gap = rational_eta[j + 1] - rational_eta[j]
            if index == 11 and j == 0:
                assert current_gap == 0
            else:
                assert current_gap > 0
                exact_gaps.append(current_gap)
        if index != 11:
            noncollapsed_patterns.append([Q(1), -Q(zz1), -Q(zz2)])
        eta = [D(value.numerator) / D(value.denominator) for value in rational_eta]
        links = [oracle.link_values(value, link) for value in eta]
        cumulative = [ZERO] + [value[0] for value in links] + [ONE]
        probabilities = [cumulative[k + 1] - cumulative[k] for k in range(4)]
        derivatives = [[ZERO] * 9] + [[links[j][1] * value for value in design[j]] for j in range(3)] + [[ZERO] * 9]
        jacobian = [[b - a for a, b in zip(derivatives[k], derivatives[k + 1])] for k in range(4)]
        for k in range(4):
            if index == 11 and k == 1:
                assert probabilities[k] == 0 and int(category) != 2
                continue
            assert probabilities[k] > 0
            scale = weight / probabilities[k]
            if k == int(category) - 1:
                ordinary = [value + scale * derivative for value, derivative in zip(ordinary, jacobian[k])]
            for r in range(9):
                for s in range(9):
                    information[r][s] += scale * jacobian[k][r] * jacobian[k][s]
        saved.append((index, weight, design, links, probabilities, jacobian, eta))
    pattern = noncollapsed_patterns[:3]
    determinant = (pattern[0][0] * (pattern[1][1] * pattern[2][2] - pattern[1][2] * pattern[2][1])
                   - pattern[0][1] * (pattern[1][0] * pattern[2][2] - pattern[1][2] * pattern[2][0])
                   + pattern[0][2] * (pattern[1][0] * pattern[2][1] - pattern[1][1] * pattern[2][0]))
    assert determinant != 0
    H = inverse(information)
    ha = matvec(H, normal)
    denominator = dot(normal, ha)
    assert denominator > 0
    u = [value / denominator for value in ha]
    V = [[H[r][s] - ha[r] * ha[s] / denominator for s in range(9)] for r in range(9)]
    adjustment = [ZERO] * 9
    for index, weight, design, links, probabilities, jacobian, eta in saved:
        correction = [ZERO] + [weight * links[j][2] * dot(design[j], matvec(V, design[j])) / 2 for j in range(3)] + [ZERO]
        for k in range(4):
            if index == 11 and k == 1:
                continue
            factor = (correction[k + 1] - correction[k]) / probabilities[k]
            adjustment = [value + factor * derivative for value, derivative in zip(adjustment, jacobian[k])]
        if index == 11:
            x = design[0]
            F, g, gp = links[0]
            if link == "logit":
                gpp = g * ((1 - 2 * F) ** 2 - 2 * g)
            else:
                exponential = eta[0].exp()
                gpp = g * (1 - 3 * exponential + exponential ** 2)
            variance = dot(x, matvec(V, x))
            C = gp / (2 * g) * (2 * dot(x, u) + 1) + weight / 2 * (gpp - 2 * gp * gp / g) * variance
    adjustment = [value + a * C for value, a in zip(adjustment, normal)]
    score = [a + b for a, b in zip(ordinary, adjustment)]
    multiplier = dot(normal, score) / dot(normal, normal)
    tangent = [value - a * multiplier for value, a in zip(score, normal)]
    previous = json.loads((high / (key + ".json")).read_text())
    previous_score = [D(value) for value in previous["score"]]
    record = dict(link=link, normal_multiplier=str(multiplier),
                  tangent_maximum=str(max(map(abs, tangent))), raw_maximum=str(max(map(abs, score))),
                  difference_from_final_interior_score=str(max(abs(a - b) for a, b in zip(score, previous_score))),
                  minimum_other_gap=float(min(exact_gaps)), other_gaps_positive_exact=True,
                  remaining_pattern_rank_exact=True, determinant=str(determinant))
    records.append(record)
    (out / (link + ".json")).write_text(json.dumps(dict(record=record, beta=beta0,
        score=score, ordinary=ordinary, adjustment=adjustment, positive_category_information=information,
        boundary_covariance=V, vanishing_count_derivative=C), default=str, indent=2) + "\n")
    print(record)
with (out / "limits.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
