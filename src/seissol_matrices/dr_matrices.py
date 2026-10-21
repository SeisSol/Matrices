#!/usr/bin/env python

import numpy as np
from seissol_matrices import base
from seissol_matrices import basis_functions
from seissol_matrices import dg_matrices
from seissol_matrices import quad_points


def fault_points(a, b, points):
    """
    The points (shape points x 2) of a fault face in the coordinates of the
    tetrahedron, shape 3 x points: on its side a, with the face parametrised as
    the cell sees it (b = 0) or as its neighbour does with the orientation b - 1
    (see dg_matrices.dg_generator.face_to_face_parametrisation).
    """
    p0 = points[:, 0]
    p1 = points[:, 1]
    p2 = 1.0 - p0 - p1
    zero = np.zeros_like(p0)
    table = {
        (0, 0): (p1, p0, zero),
        (0, 1): (p0, p1, zero),
        (0, 2): (p1, p2, zero),
        (0, 3): (p2, p0, zero),
        (1, 0): (p0, zero, p1),
        (1, 1): (p1, zero, p0),
        (1, 2): (p2, zero, p1),
        (1, 3): (p0, zero, p2),
        (2, 0): (zero, p1, p0),
        (2, 1): (zero, p0, p1),
        (2, 2): (zero, p1, p2),
        (2, 3): (zero, p2, p0),
        (3, 0): (p2, p0, p1),
        (3, 1): (p2, p1, p0),
        (3, 2): (p0, p2, p1),
        (3, 3): (p1, p0, p2),
    }
    return np.stack(table[(a, b)])


class dr_generator:
    """
    The matrices of a fault face for the quadrature rule q, in the basis `family`.

    A cell reads its field at the points of the fault either from the volume
    (V3mTo2n, for its side a and the orientation b of the face) or from its trace:
    with the trace in the canonical frame of the face
    (dg_matrices.dg_generator.canonical_rT) and the points given in that frame,
    V2mTo2Quad reads the trace at the points for both sides alike, and the fault's
    flux goes back through V2QuadTo2m and the canonical lift
    (canonical_rDivM M_face, or bernstein_matrices.lift). In a basis that is not
    hierarchical, the time derivatives of the predictor have lower degrees and
    their traces are read with V2mTo2Quad of that degree.
    """

    def __init__(self, o, q, family="dubiner"):
        self.order = o
        self.quadrule = q
        self.family = family
        self.bf3_generator = basis_functions.generator(family, 3, self.order)
        self.bf2_generator = basis_functions.generator(family, 2, self.order)
        self.dg3_generator = dg_matrices.dg_generator(self.order, 3, family)
        self.dg2_generator = dg_matrices.dg_generator(self.order, 2, family)

    def V3mTo2n(self, a, b, matorder=None):
        points = fault_points(a, b, self.quadrule.points())
        values = self.bf3_generator.eval_all(points)
        if matorder is None:
            return values
        materialbasis = basis_functions.generator(self.family, 3, matorder)
        material = materialbasis.eval_all(points)
        return np.einsum("ij,ik->kij", values, material)

    def V3mTo2nTWDivM(self, a, b, matorder=None):
        matrix = self.V3mTo2n(a, b, matorder)
        mass = self.dg3_generator.mass_matrix()
        weights = self.quadrule.weights()
        n = weights.shape[0]
        W = np.eye(n)
        for i in range(n):
            W[i, i] = weights[i]

        if matorder is None:
            return np.linalg.solve(mass, np.dot(matrix.T, W))

        matrixT = matrix.transpose((0, 2, 1))
        return base.material_index_last(np.linalg.solve(mass, np.dot(matrixT, W)))

    def quadpoints(self):
        points = self.quadrule.points()
        return points

    def quadweights(self):
        weights = self.quadrule.weights()
        return weights

    def V2QuadTo2m(self):
        points = self.quadrule.points()
        weights = self.quadrule.weights()
        E = self.bf2_generator.eval_all(points.T).T * weights[None, :]
        mass = self.dg2_generator.mass_matrix()
        return np.linalg.solve(mass, E)

    def V2mTo2Quad(self, degree=None):
        """The face basis of the scheme, or of `degree`, at the points of the fault,
        shape (points, face basis functions)."""
        basis = self.bf2_generator
        if degree is not None:
            basis = basis_functions.generator(self.family, 2, degree + 1)
        return basis.eval_all(self.quadrule.points().T)

    def resample(self):
        return self.V2mTo2Quad() @ self.V2QuadTo2m()


if __name__ == "__main__":
    from seissol_matrices import json_io

    for order in range(2, 9):
        for quadrule in quad_points.dr_rules(order).values():
            generator = dr_generator(order, quadrule)

            filename = f"dr_{quadrule.name}_matrices_{order}.json"
            quadpoints = generator.quadpoints()
            quadweights = generator.quadweights()
            resample = generator.resample()
            V2QuadTo2m = generator.V2QuadTo2m()
            V2mTo2Quad = generator.V2mTo2Quad()

            json_io.write_matrix(quadpoints, "quadpoints", filename)
            json_io.write_matrix(quadweights, "quadweights", filename)
            json_io.write_matrix(resample, "resample", filename)
            json_io.write_matrix(V2QuadTo2m, "V2QuadTo2m", filename)
            json_io.write_matrix(V2mTo2Quad, "V2mTo2Quad", filename)
            for a in range(0, 4):
                for b in range(0, 4):
                    V3mTo2n = generator.V3mTo2n(a, b)
                    V3mTo2nTWDivM = generator.V3mTo2nTWDivM(a, b)
                    json_io.write_matrix(V3mTo2n, f"V3mTo2n({a},{b})", filename)
                    json_io.write_matrix(
                        V3mTo2nTWDivM, f"V3mTo2nTWDivM({a},{b})", filename
                    )
