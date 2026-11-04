"""Independent decimal reference for the probit score/Hessian tail fixture.

The normal inverse Mills ratio is t + 1/(t + 2/(t + 3/(...))),
from the complementary-error-function fraction in NIST DLMF 7.9.1.
No third-party Python package is required. Run from any working directory.
"""
from decimal import Decimal, getcontext

getcontext().prec = 90
t = Decimal(40)
values = []
for depth in (200, 201):
    correction = Decimal(0)
    for k in range(depth, 0, -1):
        correction = Decimal(k) / (t + correction)
    score = t + correction
    values.append(score)
    print("depth:", depth)
    print("score:", score)
    print("observed curvature:", score * correction)
print("successive convergent difference:", abs(values[1] - values[0]))
