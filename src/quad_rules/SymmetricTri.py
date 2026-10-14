import itertools
import os

import numpy as np

from quad_rules.QuadRule import QuadRule
from quad_rules.TabulatedTri import DATA_DIRECTORY, FILENAME_PATTERN, read_rule

# Smallest barycentric coordinate allowed for the points of a selected rule.
MIN_BARYCENTRIC = 1e-3

# For each degree, the fully symmetric rule with positive weights and all barycentric
# coordinates above MIN_BARYCENTRIC that has the fewest points among the tabulated
# families, and of those the one whose smallest barycentric coordinate is largest,
# Witherden-Vincent where the two families differ by less than 1e-6 there
# (a rule of a higher degree serves the degrees in between).
# tests/test_tri_rules.py checks this choice against all families.
RULES = [
    "witherden-vincent/witherden-vincent-n1-d1.txt",
    "witherden-vincent/witherden-vincent-n3-d2.txt",
    "witherden-vincent/witherden-vincent-n6-d4.txt",
    "witherden-vincent/witherden-vincent-n7-d5.txt",
    "witherden-vincent/witherden-vincent-n12-d6.txt",
    "xiao-gimbutas/xiao-gimbutas-n15-d7.txt",
    "witherden-vincent/witherden-vincent-n16-d8.txt",
    "witherden-vincent/witherden-vincent-n19-d9.txt",
    "witherden-vincent/witherden-vincent-n25-d10.txt",
    "witherden-vincent/witherden-vincent-n28-d11.txt",
    "witherden-vincent/witherden-vincent-n33-d12.txt",
    "xiao-gimbutas/xiao-gimbutas-n37-d13.txt",
    "witherden-vincent/witherden-vincent-n42-d14.txt",
    "witherden-vincent/witherden-vincent-n49-d15.txt",
    "xiao-gimbutas/xiao-gimbutas-n55-d16.txt",
    "witherden-vincent/witherden-vincent-n60-d17.txt",
    "xiao-gimbutas/xiao-gimbutas-n67-d18.txt",
    "witherden-vincent/witherden-vincent-n73-d19.txt",
    "witherden-vincent/witherden-vincent-n79-d20.txt",
]

# Orbits under the permutations of the vertices, as patterns of barycentric coordinates.
# The values of the letters are, with parameters a, b:
#   S1   AAA  A = 1/3
#   S3   AAB  A = a, B = 1 - 2a
#   S6   ABC  A = a, B = b, C = 1 - a - b    with a < b < C
ORBIT_KINDS = ["S1", "S3", "S6"]
ORBIT_PATTERNS = {
    "S1": "AAA",
    "S3": "AAB",
    "S6": "ABC",
}


def orbit_barycentric(kind, parameters):
    """
    Barycentric coordinates of the points of an orbit, shape (size, 3).

    The points are the distinct permutations of the pattern of the orbit, in
    lexicographic order of the letters.
    """
    p = list(parameters)
    if kind == "S1":
        values = {"A": 1.0 / 3.0}
    elif kind == "S3":
        values = {"A": p[0], "B": 1.0 - 2.0 * p[0]}
    elif kind == "S6":
        values = {"A": p[0], "B": p[1], "C": 1.0 - p[0] - p[1]}
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
        if pattern == [3]:
            kind, parameters = "S1", []
        elif pattern == [2, 1]:
            kind, parameters = "S3", [means[counts.index(2)]]
        else:
            kind, parameters = "S6", means[:2]
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


class SymmetricTri(QuadRule):
    """
    For each degree, the cheapest tabulated triangular rule that is invariant under
    all permutations of the vertices, has positive weights and keeps all barycentric
    coordinates above MIN_BARYCENTRIC (see RULES).

    The points are ordered by orbits as given by decompose() and orbit_barycentric().
    Since every permutation of the vertices maps the set onto itself, the two cells
    sharing a face see the same points, only numbered differently.
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
            # weights relative to the area
            orbits = decompose(barycentric, 0.5 * weights)
            barycentric, relative_weights = expand(orbits)
            self.files[degree] = path
            self.orbits[degree] = orbits
            self.data[degree] = [
                -1.0 + 2.0 * barycentric[:, 1:],
                2.0 * relative_weights,
            ]
