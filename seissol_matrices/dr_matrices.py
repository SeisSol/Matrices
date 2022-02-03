#!/usr/bin/env python

import numpy as np
import scipy.special
import quadpy as qp
from seissol_matrices import basis_functions
from seissol_matrices import dg_matrices


def gauss_jacobi_quadrature_1d(n, a, b):
    gauss_jacobi_rule = qp.c1.gauss_jacobi(n, a, b)
    return gauss_jacobi_rule.points, gauss_jacobi_rule.weights


def gauss_jacobi_quadrature_2d(n):
    points0, weights0 = gauss_jacobi_quadrature_1d(n, 0, 0)
    points1, weights1 = gauss_jacobi_quadrature_1d(n, 1, 0)
    points = np.zeros((n, n, 2))
    weights = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            points[i, j, 0] = 0.5 * (1.0 + points1[i])
            points[i, j, 1] = 0.25 * (1.0 + points0[j]) * (1.0 - points1[i])
            weights[i, j] = weights1[i] * weights0[j] * 0.125
    return points, weights


class dr_generator:
    def __init__(self, o):
        self.order = o
        self.bf_generator = basis_functions.BasisFunctionGenerator3D(self.order)
        self.dg_generator = dg_matrices.dg_generator(self.order, 3)

    def V3mTo2n(self, a, b):
        m = self.bf_generator.number_of_basis_functions()
        points, _ = gauss_jacobi_quadrature_2d(self.order + 1)
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
        _, weights = gauss_jacobi_quadrature_2d(self.order + 1)
        n = weights.shape[0]
        W = np.eye(n * n)
        for k in range(n):
            for l in range(n):
                i = k * n + l
                W[n * n - 1 - i, n * n - 1 - i] = weights[k, l]

        return np.linalg.solve(mass, np.dot(matrix.T, W))

    def quadpoints(self):
        points, _ = gauss_jacobi_quadrature_2d(self.order + 1)
        return points.reshape(((self.order+1)**2, 2))

    def quadweights(self):
        _, weights = gauss_jacobi_quadrature_2d(self.order + 1)
        return weights.reshape((1, (self.order+1)**2))


if __name__ == "__main__":
    generator = dr_generator(2)
    from seissol_matrices import json_io

    filename = "dr_jacobi_matrices_2.json"

    for a in range(0, 4):
        for b in range(0, 4):
            V3mTo2n = generator.V3mTo2n(a, b)
            V3mTo2nTWDivM = generator.V3mTo2nTWDivM(a, b)
            quadpoints = generator.quadpoints()
            quadweights = generator.quadweights()
            json_io.write_matrix(V3mTo2n, f"V3mTo2n({a},{b})", filename)
            json_io.write_matrix(V3mTo2nTWDivM, f"V3mTo2nTWDivM({a},{b})", filename)
            json_io.write_matrix(quadpoints, "quadpoints", filename)
            json_io.write_matrix(quadweights, "quadweights", filename)


