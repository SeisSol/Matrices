import itertools
import os

import numpy as np

from quad_rules.QuadRule import QuadRule
from quad_rules.TabulatedTet import DATA_DIRECTORY, FILENAME_PATTERN, read_rule

# Smallest barycentric coordinate allowed for the points of a selected rule.
MIN_BARYCENTRIC = 1e-3

# For each degree, the fully symmetric rule with positive weights and all barycentric
# coordinates above MIN_BARYCENTRIC that has the fewest points among the tabulated
# families (a rule of a higher degree serves the degrees in between).
# tests/test_tet_rules.py checks this choice against all families.
RULES = [
    "witherden-vincent/witherden-vincent-n1-d1.txt",
    "witherden-vincent/witherden-vincent-n4-d2.txt",
    "witherden-vincent/witherden-vincent-n8-d3.txt",
    "witherden-vincent/witherden-vincent-n14-d5.txt",
    "witherden-vincent/witherden-vincent-n24-d6.txt",
    "witherden-vincent/witherden-vincent-n35-d7.txt",
    "witherden-vincent/witherden-vincent-n46-d8.txt",
    "orbit-search/orbit-search-n79-d10.txt",
    "worku-hicken-zingg/worku-hicken-zingg-n100-d11.txt",
    "worku-hicken-zingg/worku-hicken-zingg-n124-d12.txt",
    "witherden/witherden-n145-d13.txt",
    "worku-hicken-zingg/worku-hicken-zingg-n181-d14.txt",
    "witherden/witherden-n213-d15.txt",
    "witherden/witherden-n251-d16.txt",
    "witherden/witherden-n290-d17.txt",
    "witherden/witherden-n352-d18.txt",
    "worku-hicken-zingg/worku-hicken-zingg-n408-d19.txt",
    "worku-hicken-zingg/worku-hicken-zingg-n469-d20.txt",
    "worku-hicken-zingg/worku-hicken-zingg-n540-d21.txt",
]

# Orbits under the permutations of the vertices, as patterns of barycentric coordinates.
# The values of the letters are, with parameters a, b, c:
#   S1   AAAA  A = 1/4
#   S4   AAAB  A = a, B = 1 - 3a
#   S6   AABB  A = a, B = 1/2 - a          with a < 1/4
#   S12  AABC  A = a, B = b, C = 1 - 2a - b    with b < C
#   S24  ABCD  A = a, B = b, C = c, D = 1 - a - b - c    with a < b < c < D
ORBIT_KINDS = ["S1", "S4", "S6", "S12", "S24"]
ORBIT_PATTERNS = {
    "S1": "AAAA",
    "S4": "AAAB",
    "S6": "AABB",
    "S12": "AABC",
    "S24": "ABCD",
}


def orbit_barycentric(kind, parameters):
    """
    Barycentric coordinates of the points of an orbit, shape (size, 4).

    The points are the distinct permutations of the pattern of the orbit, in
    lexicographic order of the letters.
    """
    p = list(parameters)
    if kind == "S1":
        values = {"A": 0.25}
    elif kind == "S4":
        values = {"A": p[0], "B": 1.0 - 3.0 * p[0]}
    elif kind == "S6":
        values = {"A": p[0], "B": 0.5 - p[0]}
    elif kind == "S12":
        values = {"A": p[0], "B": p[1], "C": 1.0 - 2.0 * p[0] - p[1]}
    elif kind == "S24":
        values = {"A": p[0], "B": p[1], "C": p[2], "D": 1.0 - p[0] - p[1] - p[2]}
    else:
        raise ValueError(f"unknown orbit {kind}")
    permutations = sorted(set(itertools.permutations(ORBIT_PATTERNS[kind])))
    return np.array(
        [[values[letter] for letter in letters] for letters in permutations]
    )


def decompose(barycentric, weights, tolerance=1e-8):
    """
    Orbits (kind, weight per point, parameters) of a fully symmetric rule given by
    the barycentric coordinates and weights of its points, sorted by kind and then by
    parameters.
    """
    groups = {}
    for lam, weight in zip(barycentric, weights):
        values = np.sort(lam)
        clusters = [[values[0]]]
        for v in values[1:]:
            if v - clusters[-1][-1] < tolerance:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        counts = [len(c) for c in clusters]
        means = [float(np.mean(c)) for c in clusters]
        pattern = sorted(counts, reverse=True)
        if pattern == [4]:
            kind, parameters = "S1", []
        elif pattern == [3, 1]:
            kind, parameters = "S4", [means[counts.index(3)]]
        elif pattern == [2, 2]:
            kind, parameters = "S6", [means[0]]
        elif pattern == [2, 1, 1]:
            singles = [m for m, c in zip(means, counts) if c == 1]
            kind, parameters = "S12", [means[counts.index(2)], singles[0]]
        else:
            kind, parameters = "S24", means[:3]
        key = (kind,) + tuple(np.round(parameters, 9))
        groups.setdefault(key, []).append((parameters, weight))
    orbits = []
    for key, members in groups.items():
        kind = key[0]
        size = len(orbit_barycentric(kind, members[0][0]))
        if len(members) != size:
            raise ValueError(
                f"the rule is not fully symmetric: {kind} with {len(members)} points"
            )
        parameters = list(np.mean([m[0] for m in members], axis=0)) if size > 1 else []
        weight = float(np.mean([m[1] for m in members]))
        orbits.append((kind, weight, [float(v) for v in parameters]))
    orbits.sort(key=lambda o: (ORBIT_KINDS.index(o[0]), o[2]))
    return orbits


def expand(orbits):
    """Barycentric coordinates and weights of all points of the orbits, in order."""
    barycentric = np.vstack([orbit_barycentric(kind, p) for kind, _, p in orbits])
    weights = np.concatenate(
        [np.full(len(orbit_barycentric(kind, p)), w) for kind, w, p in orbits]
    )
    return barycentric, weights


class SymmetricTet(QuadRule):
    """
    For each degree, the cheapest tabulated tetrahedral rule that is invariant under
    all permutations of the vertices, has positive weights and keeps all barycentric
    coordinates above MIN_BARYCENTRIC (see RULES).

    The points are ordered by orbits as given by decompose() and orbit_barycentric(),
    which SeisSol reproduces for its plasticity points.
    """

    def __init__(self):
        self.files = {}
        self.orbits = {}
        self.data = {}
        for name in RULES:
            path = os.path.join(DATA_DIRECTORY, name)
            degree = int(FILENAME_PATTERN.match(os.path.basename(name))["degree"])
            points, weights = read_rule(path)
            lam = 0.5 * (points + 1.0)
            barycentric = np.column_stack([1.0 - lam.sum(axis=1), lam])
            # weights relative to the volume
            orbits = decompose(barycentric, 0.75 * weights)
            barycentric, relative_weights = expand(orbits)
            self.files[degree] = path
            self.orbits[degree] = orbits
            self.data[degree] = [
                -1.0 + 2.0 * barycentric[:, 1:],
                4.0 / 3.0 * relative_weights,
            ]
