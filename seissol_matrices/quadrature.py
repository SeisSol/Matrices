from abc import ABC
import quadpy as qp
import numpy as np


class quadrature(ABC):
    def points(n):
        pass

    def weights(n):
        pass


class gauss_jacobi(quadrature):
    def gauss_jacobi_quadrature_1d(self, n, a, b):
        gauss_jacobi_rule = qp.c1.gauss_jacobi(n, a, b)
        return gauss_jacobi_rule.points, gauss_jacobi_rule.weights

    def points(self, n):
        points0, weights0 = self.gauss_jacobi_quadrature_1d(n, 0, 0)
        points1, weights1 = self.gauss_jacobi_quadrature_1d(n, 1, 0)
        points = np.zeros((n, n, 2))
        for i in range(n):
            for j in range(n):
                points[i, j, 0] = 0.5 * (1.0 + points1[i])
                points[i, j, 1] = 0.25 * (1.0 + points0[j]) * (1.0 - points1[i])
        return points

    def weights(self, n):
        points0, weights0 = self.gauss_jacobi_quadrature_1d(n, 0, 0)
        points1, weights1 = self.gauss_jacobi_quadrature_1d(n, 1, 0)
        weights = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                weights[i, j] = weights1[i] * weights0[j] * 0.125
        return weights
