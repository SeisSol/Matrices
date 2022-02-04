#!/usr/bin/env python

import numpy as np
from seissol_matrices import basis_functions
from seissol_matrices import dg_matrices
from seissol_matrices import quadrature


class dr_generator:
    def __init__(self, o, q):
        self.order = o
        self.quadrule = q
        self.bf3_generator = basis_functions.BasisFunctionGenerator3D(self.order)
        self.bf2_generator = basis_functions.BasisFunctionGenerator2D(self.order)
        self.dg3_generator = dg_matrices.dg_generator(self.order, 3)
        self.dg2_generator = dg_matrices.dg_generator(self.order, 2)

    def V3mTo2n(self, a, b):
        m = self.bf3_generator.number_of_basis_functions()
        points = self.quadrule.points()
        n = points.shape[0]

        matrix = np.zeros((n, m))
        for i, p in enumerate(points):
            for j in range(m):
                if a == 0 and b == 0:
                    matrix[i, j] = self.bf3_generator.eval_basis([p[1], p[0], 0], j)
                if a == 0 and b == 1:
                    matrix[i, j] = self.bf3_generator.eval_basis([p[0], p[1], 0], j)
                if a == 0 and b == 2:
                    matrix[i, j] = self.bf3_generator.eval_basis(
                        [p[1], 1 - p[0] - p[1], 0], j
                    )
                if a == 0 and b == 3:
                    matrix[i, j] = self.bf3_generator.eval_basis(
                        [1 - p[0] - p[1], p[0], 0], j
                    )
                if a == 1 and b == 0:
                    matrix[i, j] = self.bf3_generator.eval_basis([p[0], 0, p[1]], j)
                if a == 1 and b == 1:
                    matrix[i, j] = self.bf3_generator.eval_basis([p[1], 0, p[0]], j)
                if a == 1 and b == 2:
                    matrix[i, j] = self.bf3_generator.eval_basis(
                        [1 - p[0] - p[1], 0, p[1]], j
                    )
                if a == 1 and b == 3:
                    matrix[i, j] = self.bf3_generator.eval_basis(
                        [p[0], 0, 1 - p[0] - p[1]], j
                    )
                if a == 2 and b == 0:
                    matrix[i, j] = self.bf3_generator.eval_basis([0, p[1], p[0]], j)
                if a == 2 and b == 1:
                    matrix[i, j] = self.bf3_generator.eval_basis([0, p[0], p[1]], j)
                if a == 2 and b == 2:
                    matrix[i, j] = self.bf3_generator.eval_basis(
                        [0, p[1], 1 - p[0] - p[1]], j
                    )
                if a == 2 and b == 3:
                    matrix[i, j] = self.bf3_generator.eval_basis(
                        [0, 1 - p[0] - p[1], p[0]], j
                    )
                if a == 3 and b == 0:
                    matrix[i, j] = self.bf3_generator.eval_basis(
                        [1 - p[0] - p[1], p[0], p[1]], j
                    )
                if a == 3 and b == 1:
                    matrix[i, j] = self.bf3_generator.eval_basis(
                        [1 - p[0] - p[1], p[1], p[0]], j
                    )
                if a == 3 and b == 2:
                    matrix[i, j] = self.bf3_generator.eval_basis(
                        [p[0], 1 - p[0] - p[1], p[1]], j
                    )
                if a == 3 and b == 3:
                    matrix[i, j] = self.bf3_generator.eval_basis(
                        [p[1], p[0], 1 - p[0] - p[1]], j
                    )
        return matrix

    def V3mTo2nTWDivM(self, a, b):
        matrix = self.V3mTo2n(a, b)
        mass = self.dg3_generator.mass_matrix()
        weights = self.quadrule.weights()
        n = weights.shape[0]
        W = np.eye(n)
        for i in range(n):
            W[i, i] = weights[i]

        return np.linalg.solve(mass, np.dot(matrix.T, W))

    def quadpoints(self):
        points = self.quadrule.points()
        return points

    def quadweights(self):
        weights = self.quadrule.weights()
        return weights

    def resample(self):
        m = self.bf2_generator.number_of_basis_functions()
        points = self.quadrule.points()
        weights = self.quadrule.weights()
        n = weights.shape[0]

        E = np.zeros((n, m))
        for i, p in enumerate(points):
            for j in range(m):
                E[i, j] = self.bf2_generator.eval_basis(p, j)
        W = np.eye(n)
        for i in range(n):
            W[i, i] = weights[i]

        mass = self.dg2_generator.mass_matrix()

        return np.dot(E, np.linalg.solve(mass, np.dot(E.T, W)))


if __name__ == "__main__":
    from seissol_matrices import json_io

    for order in range(2, 8):
        for quadrule in [
            quadrature.gauss_jacobi(order + 1),
            quadrature.dunavant(order + 1),
        ]:
            generator = dr_generator(order, quadrule)

            filename = f"dr_{quadrule.name}_matrices_{order}.json"
            print(generator.resample())

            quadpoints = generator.quadpoints()
            quadweights = generator.quadweights()
            resample = generator.resample()
            json_io.write_matrix(quadpoints, "quadpoints", filename)
            json_io.write_matrix(quadweights, "quadweights", filename)
            json_io.write_matrix(resample, "resample", filename)
            for a in range(0, 4):
                for b in range(0, 4):
                    V3mTo2n = generator.V3mTo2n(a, b)
                    V3mTo2nTWDivM = generator.V3mTo2nTWDivM(a, b)
                    json_io.write_matrix(V3mTo2n, f"V3mTo2n({a},{b})", filename)
                    json_io.write_matrix(
                        V3mTo2nTWDivM, f"V3mTo2nTWDivM({a},{b})", filename
                    )
