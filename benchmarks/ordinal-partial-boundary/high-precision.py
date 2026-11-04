"""Independent 80-decimal cumulative likelihood and mean-bias equations.

Uses only Python's standard-library decimal arithmetic; input decimals are
first decoded as binary64 and then represented exactly at working precision.
Run: python3 high-precision.py REPRODUCTION_DIRECTORY NEW_OUTPUT_DIRECTORY
"""
import csv
import json
import sys
from decimal import Decimal as D, getcontext
from pathlib import Path

getcontext().prec = 80
ZERO, ONE, HALF = D(0), D(1), D("0.5")
PI = D("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628")


def read_matrix(path):
    with open(path) as stream:
        rows = list(csv.reader(stream))
    return [[D.from_float(float(value)) for value in row] for row in rows[1:]]


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), ZERO)


def matvec(a, b):
    return [dot(row, b) for row in a]


def inverse(a):
    n = len(a)
    work = [row[:] + [ONE if i == j else ZERO for j in range(n)] for i, row in enumerate(a)]
    for j in range(n):
        pivot = max(range(j, n), key=lambda i: abs(work[i][j]))
        work[j], work[pivot] = work[pivot], work[j]
        denominator = work[j][j]
        if not denominator:
            raise ValueError("Singular information in arbitrary precision")
        work[j] = [value / denominator for value in work[j]]
        for i in range(n):
            if i != j:
                coefficient = work[i][j]
                work[i] = [x - coefficient * y for x, y in zip(work[i], work[j])]
    return [row[n:] for row in work]


def link_values(eta, link):
    if link == "logit":
        p = ONE / (ONE + (-eta).exp())
        g = p * (ONE - p)
        return p, g, g * (ONE - 2 * p)
    if link == "cloglog":
        exponential = eta.exp()
        survival = (-exponential).exp()
        g = exponential * survival
        return ONE - survival, g, g * (ONE - exponential)
    # Phi(eta)=1/2+phi(eta)*sum_{j>=0} eta^(2j+1)/(2j+1)!!.
    # The retained states have moderate predictors. Terms are summed until
    # additional terms cannot affect the requested 80-decimal result.
    g = (-eta * eta / 2).exp() / (2 * PI).sqrt()
    term = total = eta
    for j in range(1, 2000):
        term *= eta * eta / (2 * j + 1)
        previous = total
        total += term
        if total == previous:
            break
    else:
        raise ValueError("Normal-CDF series did not converge")
    return HALF + g * total, g, -eta * g


def evaluate(data, beta, link):
    dim, q = len(beta), 3
    information = [[ZERO for _ in beta] for _ in beta]
    ordinary = [ZERO for _ in beta]
    loglik = ZERO
    rows = []
    for weight, offset, category, z1, z2 in data:
        if not weight:
            continue
        design = []
        for j in range(q):
            row = [ZERO for _ in beta]
            row[j] = ONE
            row[q + 2 * j], row[q + 2 * j + 1] = -z1, -z2
            design.append(row)
        eta = [value - offset for value in matvec(design, beta)]
        if any(eta[j + 1] <= eta[j] for j in range(q - 1)):
            raise ValueError("Outside strict positive-probability domain")
        link_result = [link_values(value, link) for value in eta]
        cumulative = [ZERO] + [value[0] for value in link_result] + [ONE]
        probabilities = [cumulative[k + 1] - cumulative[k] for k in range(q + 1)]
        derivatives = [[ZERO for _ in beta]] + [
            [link_result[j][1] * value for value in design[j]] for j in range(q)
        ] + [[ZERO for _ in beta]]
        jacobian = [[b - a for a, b in zip(derivatives[k], derivatives[k + 1])] for k in range(q + 1)]
        observed = int(category) - 1
        loglik += weight * probabilities[observed].ln()
        for k in range(q + 1):
            scale = weight / probabilities[k]
            if k == observed:
                ordinary = [value + scale * derivative for value, derivative in zip(ordinary, jacobian[k])]
            for r in range(dim):
                for s in range(dim):
                    information[r][s] += scale * jacobian[k][r] * jacobian[k][s]
        rows.append((weight, design, eta, link_result, probabilities, jacobian, observed))
    covariance = inverse(information)
    adjustment = [ZERO for _ in beta]
    for weight, design, eta, links, probabilities, jacobian, observed in rows:
        correction = [ZERO] + [
            weight * links[j][2] * dot(design[j], matvec(covariance, design[j])) / 2 for j in range(q)
        ] + [ZERO]
        for k in range(q + 1):
            factor = (correction[k + 1] - correction[k]) / probabilities[k]
            adjustment = [value + factor * derivative for value, derivative in zip(adjustment, jacobian[k])]
    score = [u + a for u, a in zip(ordinary, adjustment)]
    step = matvec(covariance, score)
    gaps = [(eta[j + 1] - eta[j], index, j, observed) for index, (_, _, eta, _, _, _, observed) in enumerate(rows)
            for j in range(q - 1)]
    minimum = min(gaps)
    scaled = max(abs(score[j]) / information[j][j].sqrt() for j in range(dim))
    predictor_step = max(abs(value) for _, design, _, _, _, _, _ in rows for value in matvec(design, step))
    return dict(loglik=loglik, ordinary=ordinary, adjustment=adjustment, score=score,
                information=information, covariance=covariance, step=step, minimum_gap=minimum[0],
                scaled_score=scaled, predictor_step=predictor_step)


def maximum_error(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def main():
    root, out = map(Path, sys.argv[1:])
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for path in sorted(root.glob("*-coefficients.csv")):
        key = path.name.removesuffix("-coefficients.csv")
        link = key.split("-")[0]
        data = read_matrix(root / (key + "-input.csv"))
        beta = read_matrix(path)[0]
        value = evaluate(data, beta, link)
        vectors = read_matrix(root / (key + "-native-vectors.csv"))
        info = read_matrix(root / (key + "-native-information.csv"))
        cov = read_matrix(root / (key + "-native-inverse.csv"))
        row = dict(key=key, minimum_gap=value["minimum_gap"], scaled_score=value["scaled_score"],
                   predictor_step=value["predictor_step"], loglik=value["loglik"])
        for name, column in (("ordinary", 0), ("adjustment", 1), ("score", 2), ("step", 3)):
            row[name + "_absolute_error"] = maximum_error(value[name], [v[column] for v in vectors])
        for name, reference in (("information", info), ("covariance", cov)):
            row[name + "_relative_error"] = max(abs(value[name][i][j] - reference[i][j]) /
                max(ONE, abs(value[name][i][j])) for i in range(len(beta)) for j in range(len(beta)))
        records.append({name: str(entry) for name, entry in row.items()})
        with open(out / (key + ".json"), "w") as stream:
            json.dump(value, stream, default=str, indent=2)
        with open(out / "comparisons.csv", "w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(key, "gap", float(value["minimum_gap"]), "score", float(value["scaled_score"]),
              "adjustment error", float(row["adjustment_absolute_error"]), flush=True)


if __name__ == "__main__":
    main()
