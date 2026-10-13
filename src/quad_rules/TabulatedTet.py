import itertools
import os
import re
from decimal import Decimal, localcontext

import numpy as np

from quad_rules.QuadRule import QuadRule

DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tetrahedron")
FILENAME_PATTERN = re.compile(
    r"^(?P<name>[a-z\-]+)-n(?P<points>\d+)-d(?P<degree>\d+)\.txt$"
)


def read_rule(path):
    """
    Read a tetrahedral quadrature rule from a text file.

    Each line holds the barycentric coordinates l0 l1 l2 l3 of a point and its
    weight, where the weights sum to 1; lines starting with # are comments.
    The point with the barycentric coordinates (l0, l1, l2, l3) is
    l0 * v0 + l1 * v1 + l2 * v2 + l3 * v3.

    Returns the points and weights on the reference tetrahedron with the vertices
    v0 = (-1, -1, -1), v1 = (1, -1, -1), v2 = (-1, 1, -1), v3 = (-1, -1, 1).
    """
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            values = [Decimal(v) for v in line.split()]
            if len(values) != 5:
                raise ValueError(f"{path}: expected 5 columns, found {len(values)}")
            rows.append(values)
    # converted in decimal arithmetic and rounded once, so that a point given on the
    # reference tetrahedron in the source reads as the double nearest to that value
    with localcontext() as context:
        context.prec = 100
        points = np.array([[float(2 * l - 1) for l in row[1:4]] for row in rows])
        weights = np.array([float(Decimal(4) / 3 * row[4]) for row in rows])
    return points, weights


def barycentric(points):
    """Barycentric coordinates of points on the reference tetrahedron, shape (n, 4)."""
    lam = 0.5 * (points + 1.0)
    return np.column_stack([1.0 - lam.sum(axis=1), lam])


def is_fully_symmetric(points, weights, tol=1e-12):
    """
    Check whether a rule is invariant under all 24 permutations of the vertices,
    i.e. whether every permutation maps each point onto a point with the same weight.
    """
    lam = barycentric(points)
    for permutation in itertools.permutations(range(4)):
        permuted = lam[:, list(permutation)]
        distance = np.linalg.norm(permuted[:, None, :] - lam[None, :, :], axis=2)
        partner = np.argmin(distance, axis=1)
        if distance[np.arange(len(lam)), partner].max() > tol:
            return False
        if np.abs(weights[partner] - weights).max() > tol:
            return False
    return True


class TabulatedTet(QuadRule):
    """
    The tetrahedral quadrature rules of one family, read from the files
    tetrahedron/<family>/<name>-n<points>-d<degree>.txt
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
