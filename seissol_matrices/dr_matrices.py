#!/usr/bin/env python

import numpy as np
from seissol_matrices import basis_functions
from seissol_matrices import dg_matrices
from seissol_matrices import quadrature


class dr_generator:
    def __init__(self, o, q):
        self.order = o
        self.quadrule = q
        self.bf_generator = basis_functions.BasisFunctionGenerator3D(self.order)
        self.dg_generator = dg_matrices.dg_generator(self.order, 3)

    def V3mTo2n(self, a, b):
        m = self.bf_generator.number_of_basis_functions()
        points = self.quadrule.points(self.order + 1)
        n = points.shape[0]

        matrix = np.zeros((n * n, m))
        for k in range(n):
            for l in range(n):
                p = points[k, l, :]
                i = k * n + l
                for j in range(m):
                    if a == 0 and b == 0:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [p[1], p[0], 0], j
                        )
                    if a == 0 and b == 1:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [p[0], p[1], 0], j
                        )
                    if a == 0 and b == 2:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [p[1], 1 - p[0] - p[1], 0], j
                        )
                    if a == 0 and b == 3:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [1 - p[0] - p[1], p[0], 0], j
                        )
                    if a == 1 and b == 0:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [p[0], 0, p[1]], j
                        )
                    if a == 1 and b == 1:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [p[1], 0, p[0]], j
                        )
                    if a == 1 and b == 2:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [1 - p[0] - p[1], 0, p[1]], j
                        )
                    if a == 1 and b == 3:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [p[0], 0, 1 - p[0] - p[1]], j
                        )
                    if a == 2 and b == 0:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [0, p[1], p[0]], j
                        )
                    if a == 2 and b == 1:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [0, p[0], p[1]], j
                        )
                    if a == 2 and b == 2:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [0, p[1], 1 - p[0] - p[1]], j
                        )
                    if a == 2 and b == 3:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [0, 1 - p[0] - p[1], p[0]], j
                        )
                    if a == 3 and b == 0:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [1 - p[0] - p[1], p[0], p[1]], j
                        )
                    if a == 3 and b == 1:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [1 - p[0] - p[1], p[1], p[0]], j
                        )
                    if a == 3 and b == 2:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [p[0], 1 - p[0] - p[1], p[1]], j
                        )
                    if a == 3 and b == 3:
                        matrix[n * n - i - 1, j] = self.bf_generator.eval_basis(
                            [p[1], p[0], 1 - p[0] - p[1]], j
                        )
        return matrix

    def V3mTo2nTWDivM(self, a, b):
        matrix = self.V3mTo2n(a, b)
        mass = self.dg_generator.mass_matrix()
        weights = self.quadrule.weights(self.order + 1)
        n = weights.shape[0]
        W = np.eye(n * n)
        for k in range(n):
            for l in range(n):
                i = k * n + l
                W[n * n - 1 - i, n * n - 1 - i] = weights[k, l]

        return np.linalg.solve(mass, np.dot(matrix.T, W))

    def quadpoints(self):
        points = self.quadrule.points(self.order + 1)
        points = points.reshape(((self.order + 1) ** 2, 2))
        points = points[::-1, :]
        return points

    def quadweights(self):
        weights = self.quadrule.weights(self.order + 1)
        weights = weights.reshape(((self.order + 1) ** 2, 1))
        weights = weights[::-1, :]
        return weights


if __name__ == "__main__":
    from seissol_matrices import json_io

    quadrule = quadrature.gauss_jacobi()
    for order in range(2, 8):

        generator = dr_generator(order, quadrule)

        filename = f"dr_jacobi_matrices_{order}.json"

        quadpoints = generator.quadpoints()
        quadweights = generator.quadweights()
        json_io.write_matrix(quadpoints, "quadpoints", filename)
        json_io.write_matrix(quadweights, "quadweights", filename)
        for a in range(0, 4):
            for b in range(0, 4):
                V3mTo2n = generator.V3mTo2n(a, b)
                V3mTo2nTWDivM = generator.V3mTo2nTWDivM(a, b)
                json_io.write_matrix(V3mTo2n, f"V3mTo2n({a},{b})", filename)
                json_io.write_matrix(V3mTo2nTWDivM, f"V3mTo2nTWDivM({a},{b})", filename)
