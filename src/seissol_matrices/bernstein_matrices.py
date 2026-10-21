"""
The operators of the Bernstein-Bezier basis (basis_functions.BernsteinGenerator)
in the sparse, exact form a kernel applies them in, instead of the dense
matrices the generic generators give.

A polynomial of degree m has the coefficients c_a, one per multi-index a of
degree m in the order of basis_functions.bernstein_indices. All entries below are
rational and stated exactly.

- derivative: d / d_x_k maps degree m to m - 1 with
  (D c)_b = m (c_(b + e_(k+1)) - c_(b + e_0)): two entries per row.
- elevation: the same polynomial with the coefficients of degree m,
  (E c)_a = sum_i a_i / m c_(a - e_i): up to dim + 1 entries per row.
- trace: the coefficients of a face are the coefficients of the volume whose
  index at the opposite vertex is zero, a gather.
- face_permutation: a reparametrisation of a face permutes the coefficients.
- lift: M^-1 rT^T M_face, the map from the coefficients of a face to the cell,
  factorises (Chan and Warburton, 2017) as

      L = sum_j l_j S_j (E^m_(m-j))^T L_0,
      L_0 = (2m + 3) I + m^2 E^m_(m-1) (E^m_(m-1))^T,
      l_j = (-1)^j binom(m, j) / (j + 1),

  with the elevations E on the face, and S_j the scatter of the face
  coefficients of degree m - j to the volume coefficients whose index at the
  opposite vertex is j. (E^m_(m-j))^T is the chain of single-degree steps
  (E^(m-j+1)_(m-j))^T ... (E^m_(m-1))^T.
"""

import math
from fractions import Fraction

import numpy as np

from seissol_matrices import basis_functions, dg_matrices


def count(degree, dim):
    """The number of polynomials of a degree in dim variables, 0 below degree 0."""
    return math.comb(degree + dim, dim) if degree >= 0 else 0


def _positions(degree, dim):
    return {a: i for i, a in enumerate(basis_functions.bernstein_indices(degree, dim))}


def _full(tail, degree):
    """(a_0, a_1, ..., a_dim) from (a_1, ..., a_dim)."""
    return (degree - sum(tail),) + tuple(tail)


def _to_array(entries, shape):
    matrix = np.zeros(shape)
    for (row, column), value in entries.items():
        matrix[row, column] = float(value)
    return matrix


def derivative_entries(degree, dim, direction):
    """The entries {(row, column): value} of the derivative d / d_x_direction from
    degree to degree - 1."""
    m = degree
    lower = basis_functions.bernstein_indices(m - 1, dim)
    upper = _positions(m, dim)
    entries = {}
    for row, b in enumerate(lower):
        raised = list(b)
        raised[direction] += 1
        entries[(row, upper[tuple(raised)])] = Fraction(m)
        entries[(row, upper[tuple(b)])] = Fraction(-m)
    return entries


def derivative(degree, dim, direction):
    """The derivative d / d_x_direction as a matrix from degree to degree - 1."""
    return _to_array(
        derivative_entries(degree, dim, direction),
        (count(degree - 1, dim), count(degree, dim)),
    )


def elevation_entries(degree, dim):
    """The entries of the elevation from degree - 1 to degree."""
    m = degree
    lower = _positions(m - 1, dim)
    entries = {}
    for row, tail in enumerate(basis_functions.bernstein_indices(m, dim)):
        a = _full(tail, m)
        for i in range(dim + 1):
            if a[i] > 0:
                b = list(a)
                b[i] -= 1
                entries[(row, lower[tuple(b[1:])])] = Fraction(a[i], m)
    return entries


def elevation(degree, dim):
    """The elevation from degree - 1 to degree, as a matrix."""
    return _to_array(
        elevation_entries(degree, dim), (count(degree, dim), count(degree - 1, dim))
    )


def _volume_index(face, face_index, degree, level=0, ranks=(0, 1, 2)):
    """
    The full volume multi-index of the face multi-index (b_1, b_2) of degree
    `degree`, with `level` at the opposite vertex. The face index is in the frame
    in which the cell's face has the orientation `ranks`
    (dg_matrices.canonical_to_face_parametrisation); (0, 1, 2) is the frame of the
    cell's own parametrisation.
    """
    b = _full(face_index, degree)
    a = [0, 0, 0, 0]
    for slot, vertex in enumerate(dg_matrices.face_vertices(face)):
        a[vertex] = b[ranks[slot]]
    a[dg_matrices.opposite_vertex(face)] = level
    return tuple(a)


def trace_indices(degree, face, ranks=(0, 1, 2)):
    """For every face coefficient (in face order), the volume coefficient it is:
    in the cell's parametrisation of the face, or in its canonical frame for the
    orientation `ranks`."""
    volume = _positions(degree, 3)
    return [
        volume[_volume_index(face, b, degree, 0, ranks)[1:]]
        for b in basis_functions.bernstein_indices(degree, 2)
    ]


def trace(degree, face, ranks=(0, 1, 2)):
    """The trace on a face as a matrix of zeros and ones, one per row."""
    indices = trace_indices(degree, face, ranks)
    matrix = np.zeros((len(indices), count(degree, 3)))
    matrix[np.arange(len(indices)), indices] = 1.0
    return matrix


def face_permutation_indices(degree, reparametrisation):
    """
    For the face coefficients of f(y) with y = face_to_face_parametrisation(x,
    reparametrisation), as a function of x: the coefficient of f each one is.
    """
    corners = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    mapped = dg_matrices.face_to_face_parametrisation(corners, reparametrisation)
    reference = corners.T
    # the corner of the face y lands on for each corner x
    image = [
        int(np.argmin(np.linalg.norm(reference - mapped[:, s], axis=1)))
        for s in range(3)
    ]
    positions = _positions(degree, 2)
    indices = []
    for tail in basis_functions.bernstein_indices(degree, 2):
        a = _full(tail, degree)
        # the barycentric coordinate of y at corner image[s] is the one of x at s
        b = [0, 0, 0]
        for s in range(3):
            b[image[s]] = a[s]
        indices.append(positions[tuple(b[1:])])
    return indices


def face_permutation(degree, reparametrisation):
    indices = face_permutation_indices(degree, reparametrisation)
    matrix = np.zeros((len(indices), len(indices)))
    matrix[np.arange(len(indices)), indices] = 1.0
    return matrix


def lift_coefficients(degree):
    """l_j = (-1)^j binom(m, j) / (j + 1) for the levels j = 0, ..., m."""
    m = degree
    return [Fraction((-1) ** j * math.comb(m, j), j + 1) for j in range(m + 1)]


def lift_face_entries(degree):
    """The entries of L_0 = (2m + 3) I + m^2 E E^T on the face."""
    m = degree
    n = count(m, 2)
    entries = {(i, i): Fraction(2 * m + 3) for i in range(n)}
    if m > 0:
        E = elevation_entries(m, 2)
        columns = {}
        for (row, column), value in E.items():
            columns.setdefault(column, []).append((row, value))
        for terms in columns.values():
            for row_a, value_a in terms:
                for row_b, value_b in terms:
                    key = (row_a, row_b)
                    entries[key] = entries.get(key, 0) + m * m * value_a * value_b
    return entries


def lift_face(degree):
    n = count(degree, 2)
    return _to_array(lift_face_entries(degree), (n, n))


def lift_scatter_indices(degree, face, level, ranks=(0, 1, 2)):
    """For every face coefficient of degree m - level, the volume coefficient of
    degree m with index `level` at the opposite vertex that it lands on, for face
    coefficients in the frame of the orientation `ranks`."""
    m = degree
    volume = _positions(m, 3)
    return [
        volume[_volume_index(face, b, m - level, level, ranks)[1:]]
        for b in basis_functions.bernstein_indices(m - level, 2)
    ]


def lift(degree, face, ranks=(0, 1, 2)):
    """
    The lift M^-1 rT^T M_face as a dense matrix, assembled from its factors, for
    face coefficients in the frame of the orientation `ranks`. L_0 and the
    elevations commute with the permutations of a face, so a frame changes the
    scatters alone.
    """
    m = degree
    L0 = lift_face(m)
    result = np.zeros((count(m, 3), count(m, 2)))
    chain = L0
    for level, scale in enumerate(lift_coefficients(m)):
        if level > 0:
            chain = elevation(m - level + 1, 2).T @ chain
        indices = lift_scatter_indices(m, face, level, ranks)
        result[indices, :] += float(scale) * chain
    return result
