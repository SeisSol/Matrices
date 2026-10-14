#!/usr/bin/env python3
"""
Search one orbit structure for a fully symmetric tetrahedral quadrature rule.

Where tools/reduce_symmetric_tet_rule.py shrinks a known rule, this tool starts from
random points: for a structure, i.e. how many orbits of each kind under the permutations
of the vertices (S1, S4, S6, S12, S24, see that tool), it solves the moment equations
from many random starts at once with a Levenberg-Marquardt method whose steps may not
leave the admissible set (positive weights, all barycentric coordinates at least
--min-barycentric). The equations require exactness for the polynomials up to the given
degree that are invariant under the vertex permutations, as polynomials in the
elementary symmetric functions e2, e3, e4 of the barycentric coordinates.

A structure with more unknowns than equations has families of solutions. Along the family
of the solution found, the tool moves by pseudo-arclength continuation to the member
whose smallest barycentric coordinate is largest. The result is refined to 40 digits with
mpmath and written in the text format of src/quad_rules/tetrahedron, with the orbits
listed in the header.

Example (the rule of degree 10 with 79 points, about ten minutes):

  python tools/search_symmetric_tet_rule.py --degree 10 \\
    --structure S1:1,S4:3,S6:1,S12:5 orbit-search-n79-d10.txt
"""

import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reduce_symmetric_tet_rule import (  # noqa: E402
    ORBIT_KINDS,
    ORBIT_PARAMETERS,
    ORBIT_SIZE,
    conical_product_rule,
    describe,
    number_of_points,
    refine,
    smallest_barycentric,
    write_rule,
)


def elementary(lam):
    """
    e2, e3, e4 of barycentric coordinates lam (..., 4), shape (..., 3), and their
    derivatives with respect to lam, shape (..., 3, 4).
    """
    values = []
    derivatives = []
    for k in (2, 3, 4):
        e = np.zeros(lam.shape[:-1])
        de = np.zeros(lam.shape)
        for combination in itertools.combinations(range(4), k):
            e = e + np.prod(lam[..., list(combination)], axis=-1)
            for c in combination:
                rest = [other for other in combination if other != c]
                de[..., c] += np.prod(lam[..., rest], axis=-1)
        values.append(e)
        derivatives.append(de)
    return np.stack(values, axis=-1), np.stack(derivatives, axis=-2)


class InvariantEquations:
    """
    The moment equations of a fully symmetric rule: exactness for the monomials in
    e2, e3, e4 of weighted degree (2, 3, 4) up to the degree, orthonormalised on the
    unit tetrahedron. Weights are per point on the unit tetrahedron (volume 1/6).
    """

    def __init__(self, degree):
        self.exponents = np.array(
            [
                e
                for e in itertools.product(*(range(degree // w + 1) for w in (2, 3, 4)))
                if 2 * e[0] + 3 * e[1] + 4 * e[2] <= degree
            ]
        )
        self.count = len(self.exponents)
        # centre and scale e2, e3, e4 around their values at the centroid
        centre, _ = elementary(np.full((1, 4), 0.25))
        self.shift = centre[0]
        self.scale = 1.0 / centre[0]
        points, weights = conical_product_rule(degree + 2)
        lam = np.column_stack([1 - points.sum(axis=1), points])
        e, _ = elementary(lam)
        F = self.monomials((e - self.shift) * self.scale)
        _, R = np.linalg.qr(np.sqrt(weights)[:, None] * F)
        R = R * np.sign(np.diag(R))[:, None]
        self.T = np.linalg.inv(R).T
        self.moments = self.T @ (F.T @ weights)
        self.volume = weights.sum()

    def monomials(self, e):
        out = np.ones(e.shape[:-1] + (self.count,))
        for k in range(3):
            out = out * e[..., k : k + 1] ** self.exponents[:, k]
        return out

    def monomial_gradient(self, e):
        gradients = []
        for k in range(3):
            g = np.ones(e.shape[:-1] + (self.count,))
            for j in range(3):
                if j == k:
                    power = self.exponents[:, j] - 1
                    g = g * np.where(
                        power >= 0,
                        self.exponents[:, j]
                        * e[..., j : j + 1] ** np.maximum(power, 0),
                        0.0,
                    )
                else:
                    g = g * e[..., j : j + 1] ** self.exponents[:, j]
            gradients.append(g)
        return np.stack(gradients, axis=-1)

    def values(self, lam):
        """The orthonormal invariants at lam (..., 4) and their derivatives."""
        e, de = elementary(lam)
        en = (e - self.shift) * self.scale
        de = de * self.scale[:, None]
        g = self.monomials(en) @ self.T.T
        dg = np.einsum("pq,...qv->...pv", self.T, self.monomial_gradient(en) @ de)
        return g, dg


def base_point(kind, p):
    """The first point of an orbit (..., 4) and its derivative by the parameters."""
    shape = p.shape[:-1]
    if kind == "S1":
        return np.full(shape + (4,), 0.25), np.zeros(shape + (0, 4))
    if kind == "S4":
        a = p[..., 0]
        derivative = np.array([[1.0, 1, 1, -3]])
        lam = np.stack([a, a, a, 1 - 3 * a], -1)
    elif kind == "S6":
        a = p[..., 0]
        derivative = np.array([[1.0, 1, -1, -1]])
        lam = np.stack([a, a, 0.5 - a, 0.5 - a], -1)
    elif kind == "S12":
        a, b = p[..., 0], p[..., 1]
        derivative = np.array([[1.0, 1, 0, -2], [0, 0, 1, -1]])
        lam = np.stack([a, a, b, 1 - 2 * a - b], -1)
    else:
        a, b, c = p[..., 0], p[..., 1], p[..., 2]
        derivative = np.array([[1.0, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1]])
        lam = np.stack([a, b, c, 1 - a - b - c], -1)
    return lam, np.broadcast_to(derivative, shape + derivative.shape)


# how many distinct barycentric coordinates the points of an orbit have
DISTINCT_VALUES = {"S4": 2, "S6": 2, "S12": 3, "S24": 4}


def random_parameters(kind, count, rng, bound):
    if kind == "S1":
        return np.zeros((count, 0))
    if kind == "S4":
        return rng.uniform(bound, (1 - bound) / 3, size=(count, 1))
    if kind == "S6":
        return rng.uniform(bound, 0.25, size=(count, 1))
    if kind == "S12":
        y = rng.dirichlet(np.ones(3), size=count)
        return np.column_stack([y[:, 0] / 2, y[:, 1]])
    return rng.dirichlet(np.ones(4), size=count)[:, :3]


class Structure:
    """The orbits of a structure, and the unknowns of a rule: weights, then parameters."""

    def __init__(self, counts):
        self.kinds = [kind for kind in ORBIT_KINDS for _ in range(counts.get(kind, 0))]
        self.size = np.array([ORBIT_SIZE[kind] for kind in self.kinds], dtype=float)
        self.parameters = [ORBIT_PARAMETERS[kind] for kind in self.kinds]
        self.orbits = len(self.kinds)
        self.offsets = np.cumsum([0] + self.parameters)[:-1] + self.orbits
        self.unknowns = self.orbits + sum(self.parameters)

    def split(self, x):
        return x[:, : self.orbits], [
            x[:, o : o + n] for o, n in zip(self.offsets, self.parameters)
        ]

    def to_orbits(self, x):
        W, P = self.split(x[None, :])
        return [
            (kind, float(W[0, o]), [float(v) for v in P[o][0]])
            for o, kind in enumerate(self.kinds)
        ]


def evaluate(equations, structure, x, bound, jacobian=True):
    """
    Residuals of the moment equations for many rules x (S, unknowns), their Jacobians,
    and whether each rule is admissible.
    """
    W, P = structure.split(x)
    S = x.shape[0]
    R = -np.broadcast_to(equations.moments, (S, equations.count)).copy()
    J = np.zeros((S, equations.count, structure.unknowns)) if jacobian else None
    admissible = np.all(W > 0, axis=1)
    for o, kind in enumerate(structure.kinds):
        lam, dlam = base_point(kind, P[o])
        admissible &= np.all(lam >= bound, axis=1)
        if kind != "S1":
            # the points of the orbit stay distinct
            gaps = np.diff(np.sort(lam, axis=-1), axis=-1)
            admissible &= 1 + (gaps > 1e-6).sum(axis=-1) >= DISTINCT_VALUES[kind]
        g, dg = equations.values(lam)
        R += (W[:, o] * structure.size[o])[:, None] * g
        if jacobian:
            J[:, :, o] = structure.size[o] * g
            n = structure.parameters[o]
            if n:
                off = structure.offsets[o]
                J[:, :, off : off + n] = (W[:, o] * structure.size[o])[
                    :, None, None
                ] * np.einsum("smv,skv->smk", dg, dlam)
    return R, J, admissible


def solve(equations, structure, x, bound, iterations=200, tolerance=1e-13):
    """Levenberg-Marquardt for many starts at once; inadmissible steps are rejected."""
    x = x.copy()
    R, J, _ = evaluate(equations, structure, x, bound)
    f = np.einsum("sm,sm->s", R, R)
    mu = np.full(x.shape[0], 1e-2)
    active = np.ones(x.shape[0], dtype=bool)
    identity = np.eye(structure.unknowns)
    for _ in range(iterations):
        active &= (np.sqrt(f) >= tolerance) & (mu < 1e12)
        if not active.any():
            break
        idx = np.where(active)[0]
        A = np.einsum("smu,smv->suv", J[idx], J[idx])
        g = np.einsum("smu,sm->su", J[idx], R[idx])
        diagonal = np.einsum("suu->su", A).copy()
        diagonal[diagonal == 0] = 1.0
        pending = np.ones(len(idx), dtype=bool)
        for _ in range(12):
            sel = np.where(pending)[0]
            if len(sel) == 0:
                break
            M = A[sel] + mu[idx[sel]][:, None, None] * (
                diagonal[sel][:, :, None] * identity[None]
            )
            try:
                dx = -np.linalg.solve(M, g[sel][:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                mu[idx[sel]] *= 10
                continue
            xn = x[idx[sel]] + dx
            Rn, Jn, ok = evaluate(equations, structure, xn, bound)
            fn = np.einsum("sm,sm->s", Rn, Rn)
            accepted = ok & (fn < f[idx[sel]])
            better = idx[sel][accepted]
            x[better], R[better], J[better], f[better] = (
                xn[accepted],
                Rn[accepted],
                Jn[accepted],
                fn[accepted],
            )
            mu[better] = np.maximum(mu[better] / 3, 1e-15)
            worse = idx[sel][~accepted]
            mu[worse] *= 4
            pending[sel[accepted]] = False
            pending[sel[~accepted]] = mu[worse] < 1e12
    R, _, ok = evaluate(equations, structure, x, bound, jacobian=False)
    return x, np.sqrt(np.einsum("sm,sm->s", R, R)), ok


def follow_family(equations, structure, x, step=5e-4, steps=4000):
    """
    Pseudo-arclength continuation along the family of solutions through x, in both
    directions until a weight or a barycentric coordinate becomes negative; returns
    the member whose smallest barycentric coordinate is largest.
    """

    def tangent(x, previous=None):
        _, J, _ = evaluate(equations, structure, x[None, :], 0.0)
        t = np.linalg.svd(J[0])[2][-1]
        return -t if previous is not None and t @ previous < 0 else t

    def correct(predicted, t):
        x = predicted.copy()
        for _ in range(40):
            R, J, _ = evaluate(equations, structure, x[None, :], 0.0)
            F = np.concatenate([R[0], [t @ (x - predicted)]])
            dx = np.linalg.solve(np.vstack([J[0], t[None, :]]), -F)
            x = x + dx
            if np.linalg.norm(dx) < 1e-15:
                break
        R, _, ok = evaluate(equations, structure, x[None, :], 0.0, jacobian=False)
        return x, float(np.linalg.norm(R)), bool(ok[0])

    best = (smallest_barycentric(structure.to_orbits(x)), x)
    for direction in (1.0, -1.0):
        current = x.copy()
        t = direction * tangent(current)
        for _ in range(steps):
            candidate, residual, ok = correct(current + step * t, t)
            if residual > 1e-11 or not ok:
                break
            t = tangent(candidate, t)
            current = candidate
            smallest = smallest_barycentric(structure.to_orbits(current))
            if smallest > best[0]:
                best = (smallest, current)
    return best[1]


def search(degree, counts, starts, seed, min_barycentric, batch=500):
    equations = InvariantEquations(degree)
    structure = Structure(counts)
    print(
        f"degree {degree}: {equations.count} equations; structure "
        f"{describe([(kind, 0.0, []) for kind in structure.kinds])}, "
        f"{int(structure.size.sum())} points, {structure.unknowns} unknowns",
        flush=True,
    )
    rng = np.random.default_rng(seed)
    start = time.time()
    found = []
    tried = 0
    while tried < starts:
        count = min(batch, starts - tried)
        x0 = np.zeros((count, structure.unknowns))
        mean = equations.volume / structure.size.sum()
        x0[:, : structure.orbits] = mean * np.exp(
            0.3 * rng.standard_normal((count, structure.orbits))
        )
        for o, kind in enumerate(structure.kinds):
            off = structure.offsets[o]
            x0[:, off : off + structure.parameters[o]] = random_parameters(
                kind, count, rng, min_barycentric
            )
        x, residual, ok = solve(equations, structure, x0, min_barycentric)
        found += [x[s] for s in np.where(ok & (residual < 1e-13))[0]]
        tried += count
        print(
            f"  {tried} starts: {len(found)} solutions, {time.time() - start:.0f} s",
            flush=True,
        )
    if not found:
        raise ValueError("no solution found; try more starts or another seed")
    x = max(found, key=lambda x: smallest_barycentric(structure.to_orbits(x)))
    if structure.unknowns > equations.count:
        x = follow_family(equations, structure, x)
    return structure.to_orbits(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--degree", type=int, required=True, help="degree of exactness")
    parser.add_argument(
        "--structure",
        required=True,
        help="orbits of each kind, e.g. S1:1,S4:3,S6:1,S12:5",
    )
    parser.add_argument("--starts", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-barycentric", type=float, default=1e-3)
    parser.add_argument("output", help="file for the rule")
    args = parser.parse_args()

    counts = {
        kind: int(count)
        for kind, count in (entry.split(":") for entry in args.structure.split(","))
    }
    orbits = search(args.degree, counts, args.starts, args.seed, args.min_barycentric)
    orbits, error = refine(orbits, args.degree)
    print(
        f"{number_of_points(orbits)} points [{describe(orbits)}], smallest barycentric "
        f"coordinate {smallest_barycentric(orbits):.3e}, largest relative moment error "
        f"{float(error):.1e}"
    )
    write_rule(
        args.output,
        orbits,
        args.degree,
        f"obtained with tools/search_symmetric_tet_rule.py for the structure "
        f"{describe(orbits)}",
        args.min_barycentric,
    )


if __name__ == "__main__":
    main()
