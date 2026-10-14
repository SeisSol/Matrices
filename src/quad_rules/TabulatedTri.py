import itertools
import os
import re
from decimal import Decimal, localcontext

import numpy as np

from quad_rules.QuadRule import QuadRule

DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "triangle")
FILENAME_PATTERN = re.compile(
    r"^(?P<name>[a-z\-]+)-n(?P<points>\d+)-d(?P<degree>\d+)\.txt$"
)


def read_rule(path):
    """
    Read a triangular quadrature rule from a text file.

    Each line holds the barycentric coordinates l0 l1 l2 of a point and its
    weight, where the weights sum to 1; lines starting with # are comments.
    The point with the barycentric coordinates (l0, l1, l2) is
    l0 * v0 + l1 * v1 + l2 * v2.

    Returns the points and weights on the reference triangle with the vertices
    v0 = (-1, -1), v1 = (1, -1), v2 = (-1, 1).
    """
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            values = [Decimal(v) for v in line.split()]
            if len(values) != 4:
                raise ValueError(f"{path}: expected 4 columns, found {len(values)}")
            rows.append(values)
    # converted in decimal arithmetic and rounded once, so that a point given on the
    # reference triangle in the source reads as the double nearest to that value
    with localcontext() as context:
        context.prec = 100
        points = np.array([[float(2 * l - 1) for l in row[1:3]] for row in rows])
        weights = np.array([float(2 * row[3]) for row in rows])
    return points, weights


def barycentric(points):
    """Barycentric coordinates of points on the reference triangle, shape (n, 3)."""
    lam = 0.5 * (points + 1.0)
    return np.column_stack([1.0 - lam.sum(axis=1), lam])


def is_fully_symmetric(points, weights, tol=1e-12):
    """
    Check whether a rule is invariant under all 6 permutations of the vertices,
    i.e. whether every permutation maps each point onto a point with the same weight.
    """
    lam = barycentric(points)
    for permutation in itertools.permutations(range(3)):
        permuted = lam[:, list(permutation)]
        distance = np.linalg.norm(permuted[:, None, :] - lam[None, :, :], axis=2)
        partner = np.argmin(distance, axis=1)
        if distance[np.arange(len(lam)), partner].max() > tol:
            return False
        if np.abs(weights[partner] - weights).max() > tol:
            return False
    return True


class TabulatedTri(QuadRule):
    """
    The triangular quadrature rules of one family, read from the files
    triangle/<family>/<name>-n<points>-d<degree>.txt
    where <points> is the number of points and <degree> is the polynomial degree
    up to which the rule is exact.
    """

    def __init__(self, family):
        directory = os.path.join(DATA_DIRECTORY, family)
        self.files = {}
        self.data = {}
        for filename in sorted(os.listdir(directory)):
            match = FILENAME_PATTERN.match(filename)
            if match is None:
                continue
            degree = int(match["degree"])
            if degree in self.data:
                raise ValueError(f"{directory}: more than one rule of degree {degree}")
            path = os.path.join(directory, filename)
            points, weights = read_rule(path)
            if points.shape[0] != int(match["points"]):
                raise ValueError(
                    f"{path}: expected {match['points']} points, found {points.shape[0]}"
                )
            self.files[degree] = path
            self.data[degree] = [points, weights]
