#!/usr/bin/env python3
"""
Search for fully symmetric tetrahedral quadrature rules with fewer points.

The search starts from a fully symmetric rule with positive weights, given in the text
format of src/quad_rules/tetrahedron (barycentric coordinates and weights summing to 1).
It splits the rule into orbits under the permutations of the vertices:

  S1   (1/4, 1/4, 1/4, 1/4)      1 point,   no parameters
  S4   (a, a, a, 1 - 3a)          4 points,  parameter a
  S6   (a, a, 1/2 - a, 1/2 - a)   6 points,  parameter a
  S12  (a, a, b, 1 - 2a - b)     12 points,  parameters a, b
  S24  (a, b, c, 1 - a - b - c)  24 points,  parameters a, b, c

and then repeatedly tries to remove an orbit or to replace it by an orbit with fewer
points, solving the moment equations for the weights and parameters of the resulting
orbits with a Levenberg-Marquardt method. The equations require exactness for the
polynomials up to the given degree that are invariant under the vertex permutations,
which suffices for a fully symmetric rule. Weights have to stay positive and all
barycentric coordinates at least --min-barycentric. A move is kept as soon as the
equations are solved; the search ends when no move succeeds.

The result is refined to 40 digits with mpmath and written in the same text format,
with the orbits listed in the header.

Example:

  python tools/reduce_symmetric_tet_rule.py --degree 12 \\
    src/quad_rules/tetrahedron/jaskowiec-sukumar-symmetric/jaskowiec-sukumar-n172-d13.txt \\
    reduced-symmetric-n128-d12.txt
"""

import argparse
import itertools
import math
import os
import time
from fractions import Fraction

import mpmath
import numpy as np
from scipy.special import eval_jacobi, roots_jacobi

ORBIT_SIZE = {"S1": 1, "S4": 4, "S6": 6, "S12": 12, "S24": 24}
ORBIT_PARAMETERS = {"S1": 0, "S4": 1, "S6": 1, "S12": 2, "S24": 3}
ORBIT_PATTERN = {"S1": "AAAA", "S4": "AAAB", "S6": "AABB", "S12": "AABC", "S24": "ABCD"}
ORBIT_KINDS = ("S1", "S4", "S6", "S12", "S24")


def orbit_values(kind, p, one):
    """The distinct barycentric values of an orbit, by label of its pattern."""
    if kind == "S1":
        return {"A": one / 4}
    if kind == "S4":
        return {"A": p[0], "B": one - 3 * p[0]}
    if kind == "S6":
        return {"A": p[0], "B": one / 2 - p[0]}
    if kind == "S12":
        return {"A": p[0], "B": p[1], "C": one - 2 * p[0] - p[1]}
    return {"A": p[0], "B": p[1], "C": p[2], "D": one - p[0] - p[1] - p[2]}


def orbit_points(kind, p, one=1.0):
    """Barycentric coordinates of all points of an orbit (list of 4-tuples)."""
    values = orbit_values(kind, p, one)
    labels = sorted(set(itertools.permutations(ORBIT_PATTERN[kind])))
    return [tuple(values[label] for label in labels_) for labels_ in labels]


def orbit_barycentric(kind, p):
    return np.array(orbit_points(kind, [float(v) for v in p]))


def orbit_valid(kind, p, min_barycentric):
    """All points distinct and all barycentric coordinates above the bound."""
    lam = orbit_barycentric(kind, p)
    if len({tuple(np.round(row, 13)) for row in lam}) != ORBIT_SIZE[kind]:
        return False
    return bool(np.all(lam > min_barycentric))


def number_of_points(orbits):
    return sum(ORBIT_SIZE[kind] for kind, _, _ in orbits)


def describe(orbits):
    counts = {}
    for kind, _, _ in orbits:
        counts[kind] = counts.get(kind, 0) + 1
    return " ".join(f"{kind}x{counts[kind]}" for kind in ORBIT_KINDS if kind in counts)


def smallest_barycentric(orbits):
    return min(float(orbit_barycentric(kind, p).min()) for kind, _, p in orbits)


def conical_product_rule(n):
    """Collapsed Gauss-Jacobi rule on the unit tetrahedron, exact up to degree 2n - 1."""

    def jacobi(alpha):
        t, w = roots_jacobi(n, alpha, 0.0)
        return (t + 1) / 2, w / 2 ** (alpha + 1)

    a, wa = jacobi(2.0)
    b, wb = jacobi(1.0)
    c, wc = jacobi(0.0)
    A, B, C = np.meshgrid(a, b, c, indexing="ij")
    WA, WB, WC = np.meshgrid(wa, wb, wc, indexing="ij")
    points = np.stack(
        [A.ravel(), (B * (1 - A)).ravel(), (C * (1 - A) * (1 - B)).ravel()], axis=1
    )
    return points, (WA * WB * WC).ravel()


class OrthonormalBasis:
    """Orthonormal polynomials up to degree N on the unit tetrahedron."""

    def __init__(self, N):
        self.N = N
        self.indices = np.array(
            [
                (i, j, k)
                for i in range(N + 1)
                for j in range(N + 1 - i)
                for k in range(N + 1 - i - j)
            ]
        )
        points, weights = conical_product_rule(N + 3)
        V = self.evaluate_unscaled(points)
        gram = (V * weights[:, None]).T @ V
        self.scale = 1.0 / np.sqrt(np.diag(gram))
        self.integrals = (self.evaluate(points) * weights[:, None]).sum(axis=0)

    def evaluate_unscaled(self, X):
        """Values at the points X (n, 3) of the unit tetrahedron, shape (n, number of polynomials)."""
        r, s, t = (2 * X[:, 0] - 1, 2 * X[:, 1] - 1, 2 * X[:, 2] - 1)
        u = 2 + 2 * r + s + t
        v = -s - t
        # homogeneous Legendre polynomials q_i(u, v) = v^i P_i(u / v)
        q = [np.ones_like(u), u]
        for i in range(1, self.N):
            q.append(((2 * i + 1) * u * q[i] - i * v * v * q[i - 1]) / (i + 1))
        q = np.array(q[: self.N + 1])
        one_minus_t = 1 - t
        with np.errstate(divide="ignore", invalid="ignore"):
            b = np.where(
                one_minus_t > 1e-300,
                2 * (1 + s) / np.where(one_minus_t > 1e-300, one_minus_t, 1.0) - 1,
                -1.0,
            )
        i, j, k = self.indices[:, 0], self.indices[:, 1], self.indices[:, 2]
        Pj = eval_jacobi(j[:, None], (2 * i + 1)[:, None], 0.0, b[None, :])
        Pk = eval_jacobi(k[:, None], (2 * i + 2 * j + 2)[:, None], 0.0, t[None, :])
        values = (
            q[i]
            * (0.5 ** (i + j))[:, None]
            * one_minus_t[None, :] ** j[:, None]
            * Pj
            * Pk
        )
        return values.T

    def evaluate(self, X):
        return self.evaluate_unscaled(X) * self.scale[None, :]


class MomentEquations:
    """
    Exactness of a fully symmetric rule up to a degree, as equations for the orbits.

    Weights are per point on the unit tetrahedron (volume 1/6).
    """

    def __init__(self, degree, seed=0):
        self.degree = degree
        self.basis = OrthonormalBasis(degree)
        # the invariant polynomials: span of orbit sums over generic orbits
        rng = np.random.default_rng(seed)
        sums = []
        for _ in range(240):
            p = rng.dirichlet(np.ones(4))
            sums.append(
                self.basis.evaluate(orbit_barycentric("S24", p[:3])[:, 1:]).sum(axis=0)
            )
        U, S, _ = np.linalg.svd(np.array(sums).T, full_matrices=False)
        rank = int(np.sum(S > S[0] * 1e-11))
        self.U = U[:, :rank]
        self.target = self.U.T @ self.basis.integrals

    def orbit_sum(self, kind, p):
        X = orbit_barycentric(kind, p)[:, 1:]
        return self.U.T @ self.basis.evaluate(X).sum(axis=0)

    def residual(self, orbits):
        r = -self.target.copy()
        for kind, w, p in orbits:
            r += w * self.orbit_sum(kind, p)
        return r

    def jacobian(self, orbits, h=1e-7):
        columns = []
        for kind, w, p in orbits:
            columns.append(self.orbit_sum(kind, p))
            for m in range(len(p)):
                plus = np.array(p, dtype=float)
                minus = plus.copy()
                plus[m] += h
                minus[m] -= h
                columns.append(
                    w
                    * (self.orbit_sum(kind, plus) - self.orbit_sum(kind, minus))
                    / (2 * h)
                )
        return np.array(columns).T

    @staticmethod
    def pack(orbits):
        x = []
        for _, w, p in orbits:
            x.append(w)
            x.extend(p)
        return np.array(x, dtype=float)

    @staticmethod
    def unpack(kinds, x):
        orbits = []
        i = 0
        for kind in kinds:
            n = ORBIT_PARAMETERS[kind]
            orbits.append((kind, x[i], list(x[i + 1 : i + 1 + n])))
            i += 1 + n
        return orbits

    @staticmethod
    def feasible(orbits, min_barycentric):
        return all(
            w > 0 and orbit_valid(kind, p, min_barycentric) for kind, w, p in orbits
        )

    def solve(self, orbits, min_barycentric, iterations=300, tolerance=2e-14):
        """
        Levenberg-Marquardt; steps that leave the feasible set are rejected.
        Returns the orbits and the norm of the residual.
        """
        kinds = [kind for kind, _, _ in orbits]
        x = self.pack(orbits)
        current = self.unpack(kinds, x)
        r = self.residual(current)
        f = r @ r
        mu = 1e-3
        for _ in range(iterations):
            if math.sqrt(f) < tolerance:
                break
            J = self.jacobian(current)
            g = J.T @ r
            A = J.T @ J
            diagonal = np.diag(A).copy()
            diagonal[diagonal == 0] = 1.0
            improved = False
            for _ in range(30):
                try:
                    dx = -np.linalg.solve(A + mu * np.diag(diagonal), g)
                except np.linalg.LinAlgError:
                    mu *= 10
                    continue
                candidate = self.unpack(kinds, x + dx)
                if self.feasible(candidate, min_barycentric):
                    rc = self.residual(candidate)
                    fc = rc @ rc
                    if fc < f:
                        x, current, r, f = x + dx, candidate, rc, fc
                        mu = max(mu / 3, 1e-15)
                        improved = True
                        break
                mu *= 4
            if not improved:
                break
        return current, math.sqrt(f)


def read_rule(path):
    data = np.loadtxt(path, comments="#", ndmin=2)
    # weights per point on the unit tetrahedron
    return data[:, :4], data[:, 4] / 6.0


def decompose(barycentric, weights, tolerance=1e-9):
    """Orbits of a fully symmetric rule."""
    groups = {}
    for lam, w in zip(barycentric, weights):
        key = tuple(np.round(np.sort(lam), 9))
        groups.setdefault(key, []).append((lam, w))
    orbits = []
    for members in groups.values():
        s = np.mean([np.sort(lam) for lam, _ in members], axis=0)
        w = float(np.mean([w for _, w in members]))
        clusters = [[s[0]]]
        for v in s[1:]:
            if abs(v - clusters[-1][-1]) < tolerance:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        counts = [len(c) for c in clusters]
        values = [float(np.mean(c)) for c in clusters]
        pattern = tuple(sorted(counts, reverse=True))
        if pattern == (4,):
            orbits.append(("S1", w, []))
        elif pattern == (3, 1):
            orbits.append(("S4", w, [values[counts.index(3)]]))
        elif pattern == (2, 2):
            orbits.append(("S6", w, [min(values)]))
        elif pattern == (2, 1, 1):
            single = [v for v, c in zip(values, counts) if c == 1]
            orbits.append(("S12", w, [values[counts.index(2)], single[0]]))
        else:
            orbits.append(("S24", w, [float(s[0]), float(s[1]), float(s[2])]))
        if len(members) != ORBIT_SIZE[orbits[-1][0]]:
            raise ValueError("the rule is not fully symmetric")
    return orbits


def demotions(kind, p):
    """Orbits with fewer points that an orbit could be replaced by, with start values."""
    if kind == "S24":
        values = sorted(list(p) + [1 - sum(p)])
        # merge the closest pair of values into an S12 orbit
        gaps = [values[i + 1] - values[i] for i in range(3)]
        i = int(np.argmin(gaps))
        merged = 0.5 * (values[i] + values[i + 1])
        others = [v for j, v in enumerate(values) if j not in (i, i + 1)]
        return [("S12", [merged, others[0]])]
    if kind == "S12":
        a, b = p
        c = 1 - 2 * a - b
        low, high = 1e-6, 1 / 3 - 1e-6
        return [
            ("S4", [min(max((2 * a + b) / 3, low), high)]),
            ("S4", [min(max((2 * a + c) / 3, low), high)]),
            ("S6", [min(max(0.5 * (a + 0.5 - 0.5 * (b + c)), 1e-6), 0.25 - 1e-6)]),
        ]
    return []


def remove_by_continuation(equations, orbits, index, min_barycentric, steps=8):
    """Shrink the weight of one orbit to zero in steps, solving for the others each time."""
    kind, w0, p0 = orbits[index]
    contribution = w0 * equations.orbit_sum(kind, p0)
    current = [o for i, o in enumerate(orbits) if i != index]
    target = equations.target.copy()
    try:
        for step in range(1, steps + 1):
            remaining = 1.0 - step / steps
            equations.target = target - remaining * contribution
            tolerance = 2e-14 if step == steps else 1e-12
            current, residual = equations.solve(
                current, min_barycentric, tolerance=tolerance
            )
            if not (
                residual < tolerance and equations.feasible(current, min_barycentric)
            ):
                return None
    finally:
        equations.target = target
    return current


def reduce_rule(degree, orbits, min_barycentric, budget):
    equations = MomentEquations(degree)
    solution, residual = equations.solve(orbits, min_barycentric)
    if residual >= 2e-14 or not equations.feasible(solution, min_barycentric):
        # points of the start rule may lie closer to the boundary than allowed;
        # then the first move has to remove or replace their orbits
        solution, residual = equations.solve(orbits, 0.0)
        if residual >= 2e-14 or not equations.feasible(solution, 0.0):
            raise ValueError(
                f"the start rule does not solve the equations (residual {residual:.1e})"
            )
    orbits = solution
    print(
        f"start: {number_of_points(orbits)} points [{describe(orbits)}], "
        f"smallest barycentric coordinate {smallest_barycentric(orbits):.2e}",
        flush=True,
    )
    start = time.time()
    while time.time() - start < budget:
        # candidate moves, most points saved first, then smallest total weight
        moves = []
        for i, (kind, w, p) in enumerate(orbits):
            moves.append((ORBIT_SIZE[kind], w * ORBIT_SIZE[kind], i, None))
            for new_kind, new_p in demotions(kind, p):
                saved = ORBIT_SIZE[kind] - ORBIT_SIZE[new_kind]
                moves.append((saved, w * ORBIT_SIZE[kind], i, (new_kind, new_p)))
        moves.sort(key=lambda move: (-move[0], move[1]))
        accepted = None
        for _, _, i, replacement in moves:
            if time.time() - start > budget:
                break
            if replacement is None:
                trial = [o for j, o in enumerate(orbits) if j != i]
            else:
                new_kind, new_p = replacement
                trial = list(orbits)
                kind, w, _ = orbits[i]
                trial[i] = (
                    new_kind,
                    w * ORBIT_SIZE[kind] / ORBIT_SIZE[new_kind],
                    new_p,
                )
            solution, residual = equations.solve(trial, min_barycentric)
            if residual < 2e-14 and equations.feasible(solution, min_barycentric):
                accepted = solution
            elif replacement is None:
                accepted = remove_by_continuation(equations, orbits, i, min_barycentric)
            if accepted is not None:
                break
        if accepted is None:
            break
        orbits = accepted
        print(
            f"  {number_of_points(orbits)} points [{describe(orbits)}], "
            f"smallest barycentric coordinate {smallest_barycentric(orbits):.2e}, "
            f"{time.time() - start:.0f} s",
            flush=True,
        )
    return orbits


def monomial_integral(a, b, c):
    """Integral of x^a y^b z^c over the unit tetrahedron."""
    return Fraction(
        math.factorial(a) * math.factorial(b) * math.factorial(c),
        math.factorial(a + b + c + 3),
    )


def refine(orbits, degree, digits=50, iterations=60):
    """
    Newton iteration on the monomial moments with residuals in mpmath and the
    Jacobian in double precision. Returns the orbits as mpmath numbers and the
    largest relative error of the moments.
    """
    mpmath.mp.dps = digits
    one = mpmath.mpf(1)
    monomials = [
        (a, b, d - a - b)
        for d in range(degree + 1)
        for a in range(d + 1)
        for b in range(d + 1 - a)
    ]
    exact = [
        mpmath.mpf(e.numerator) / e.denominator
        for e in (monomial_integral(*m) for m in monomials)
    ]

    def residual(orbits_mp):
        points = []
        for kind, w, p in orbits_mp:
            for lam in orbit_points(kind, p, one):
                points.append((w, lam[1], lam[2], lam[3]))
        return [
            (mpmath.fsum(w * x**a * y**b * z**c for w, x, y, z in points) - e) / e
            for (a, b, c), e in zip(monomials, exact)
        ]

    def residual_double(orbits_double):
        points = []
        for kind, w, p in orbits_double:
            for lam in orbit_barycentric(kind, p):
                points.append((w, lam[1], lam[2], lam[3]))
        P = np.array(points)
        return np.array(
            [
                (
                    np.sum(P[:, 0] * P[:, 1] ** a * P[:, 2] ** b * P[:, 3] ** c)
                    - float(e)
                )
                / float(e)
                for (a, b, c), e in zip(monomials, exact)
            ]
        )

    current = [
        (kind, mpmath.mpf(w), [mpmath.mpf(v) for v in p]) for kind, w, p in orbits
    ]
    double = [(kind, float(w), [float(v) for v in p]) for kind, w, p in current]
    columns = []
    for i, (kind, w, p) in enumerate(double):
        step = 1e-7 * w
        plus, minus = list(double), list(double)
        plus[i], minus[i] = (kind, w + step, p), (kind, w - step, p)
        columns.append((residual_double(plus) - residual_double(minus)) / (2 * step))
        for m in range(len(p)):
            step = 1e-7 * max(abs(p[m]), 1e-3)
            p_plus, p_minus = list(p), list(p)
            p_plus[m] += step
            p_minus[m] -= step
            plus, minus = list(double), list(double)
            plus[i], minus[i] = (kind, w, p_plus), (kind, w, p_minus)
            columns.append(
                (residual_double(plus) - residual_double(minus)) / (2 * step)
            )
    inverse = np.linalg.pinv(np.array(columns).T, rcond=1e-12)
    error = None
    for _ in range(iterations):
        r = residual(current)
        error = max(abs(v) for v in r)
        if error < mpmath.mpf(10) ** (-(digits - 8)):
            break
        dx = inverse @ np.array([float(v) for v in r])
        updated = []
        j = 0
        for kind, w, p in current:
            w = w - mpmath.mpf(float(dx[j]))
            j += 1
            p_new = []
            for v in p:
                p_new.append(v - mpmath.mpf(float(dx[j])))
                j += 1
            updated.append((kind, w, p_new))
        current = updated
    error = max(abs(v) for v in residual(current))
    return current, error


def write_rule(path, orbits, degree, provenance, min_barycentric):
    one = mpmath.mpf(1)

    def number(v):
        return mpmath.nstr(v, 40, min_fixed=-30, max_fixed=30)

    lines = []
    for kind, w, p in orbits:
        for lam in orbit_points(kind, p, one):
            lines.append(" ".join(number(v) for v in lam) + " " + number(6 * w))
    with open(path, "w") as f:
        f.write(
            f"# Fully symmetric rule for the tetrahedron, degree {degree}, "
            f"{len(lines)} points, positive weights, "
            f"barycentric coordinates > {min_barycentric:g}\n"
        )
        f.write(f"# {provenance}; see README.md\n")
        f.write("# orbits: kind, weight per point, parameters (see the tool)\n")
        for kind, w, p in orbits:
            f.write(
                f"# {kind} {number(6 * w)} {' '.join(number(v) for v in p)}".rstrip()
            )
            f.write("\n")
        f.write(
            "# columns: l0 l1 l2 l3 weight (barycentric coordinates; the weights sum to 1)\n"
        )
        for line in lines:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--degree", type=int, required=True, help="degree of exactness")
    parser.add_argument("--min-barycentric", type=float, default=1e-3)
    parser.add_argument(
        "--budget", type=float, default=3600, help="time limit in seconds"
    )
    parser.add_argument("start", help="fully symmetric start rule (text format)")
    parser.add_argument("output", help="file for the reduced rule")
    args = parser.parse_args()

    barycentric, weights = read_rule(args.start)
    orbits = decompose(barycentric, weights)
    orbits = reduce_rule(args.degree, orbits, args.min_barycentric, args.budget)
    if not MomentEquations.feasible(orbits, args.min_barycentric):
        raise ValueError("no rule found that satisfies the bound on the barycentrics")
    orbits, error = refine(orbits, args.degree)
    print(
        f"{number_of_points(orbits)} points [{describe(orbits)}], "
        f"largest relative moment error {mpmath.nstr(error, 3)}"
    )
    write_rule(
        args.output,
        orbits,
        args.degree,
        f"obtained with tools/reduce_symmetric_tet_rule.py "
        f"from {os.path.basename(args.start)}",
        args.min_barycentric,
    )


if __name__ == "__main__":
    main()
