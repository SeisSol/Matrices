#!/usr/bin/env python

import numpy as np
from seissol_matrices import basis_functions
from seissol_matrices import dg_matrices
from seissol_matrices import quad_points


class dr_generator:
    def __init__(self, o, q):
        self.order = o
        self.quadrule = q
        self.bf3_generator = basis_functions.BasisFunctionGenerator3D(self.order)
        self.bf2_generator = basis_functions.BasisFunctionGenerator2D(self.order)
        self.dg3_generator = dg_matrices.dg_generator(self.order, 3)
        self.dg2_generator = dg_matrices.dg_generator(self.order, 2)

    def V3mTo2n(self, a, b, matorder=None):
        m = self.bf3_generator.number_of_basis_functions()
        points = self.quadrule.points()
        n = points.shape[0]

        if matorder is None:
            evalfun = lambda q, j, k: self.bf3_generator.eval_basis(q, j)
            materialdim = 1
        else:
            materialbasis = basis_functions.BasisFunctionGenerator3D(matorder)
            evalfun = lambda q, j, k: self.bf3_generator.eval_basis(
                q, j
            ) * materialbasis.eval_basis(q, k)
            materialdim = materialbasis.number_of_basis_functions()

        matrix = np.zeros((materialdim, n, m))
        for i, p in enumerate(points):
            for j in range(m):
                for k in range(materialdim):
                    if a == 0 and b == 0:
                        matrix[k, i, j] = evalfun([p[1], p[0], 0], j, k)
                    if a == 0 and b == 1:
                        matrix[k, i, j] = evalfun([p[0], p[1], 0], j, k)
                    if a == 0 and b == 2:
                        matrix[k, i, j] = evalfun([p[1], 1 - p[0] - p[1], 0], j, k)
                    if a == 0 and b == 3:
                        matrix[k, i, j] = evalfun([1 - p[0] - p[1], p[0], 0], j, k)
                    if a == 1 and b == 0:
                        matrix[k, i, j] = evalfun([p[0], 0, p[1]], j, k)
                    if a == 1 and b == 1:
                        matrix[k, i, j] = evalfun([p[1], 0, p[0]], j, k)
                    if a == 1 and b == 2:
                        matrix[k, i, j] = evalfun([1 - p[0] - p[1], 0, p[1]], j, k)
                    if a == 1 and b == 3:
                        matrix[k, i, j] = evalfun([p[0], 0, 1 - p[0] - p[1]], j, k)
                    if a == 2 and b == 0:
                        matrix[k, i, j] = evalfun([0, p[1], p[0]], j, k)
                    if a == 2 and b == 1:
                        matrix[k, i, j] = evalfun([0, p[0], p[1]], j, k)
                    if a == 2 and b == 2:
                        matrix[k, i, j] = evalfun([0, p[1], 1 - p[0] - p[1]], j, k)
                    if a == 2 and b == 3:
                        matrix[k, i, j] = evalfun([0, 1 - p[0] - p[1], p[0]], j, k)
                    if a == 3 and b == 0:
                        matrix[k, i, j] = evalfun([1 - p[0] - p[1], p[0], p[1]], j, k)
                    if a == 3 and b == 1:
                        matrix[k, i, j] = evalfun([1 - p[0] - p[1], p[1], p[0]], j, k)
                    if a == 3 and b == 2:
                        matrix[k, i, j] = evalfun([p[0], 1 - p[0] - p[1], p[1]], j, k)
                    if a == 3 and b == 3:
                        matrix[k, i, j] = evalfun([p[1], p[0], 1 - p[0] - p[1]], j, k)
        if matorder is None:
            matrix = np.squeeze(matrix)
        return matrix

    def V3mTo2nTWDivM(self, a, b, matorder=None):
        matrix = self.V3mTo2n(a, b, matorder)
        mass = self.dg3_generator.mass_matrix()
        weights = self.quadrule.weights()
        n = weights.shape[0]
        W = np.eye(n)
        for i in range(n):
            W[i, i] = weights[i]

        matrixT = matrix.T if matorder is None else matrix.transpose((0, 2, 1))

        return np.linalg.solve(mass, np.dot(matrixT, W))

    def quadpoints(self):
        points = self.quadrule.points()
        return points

    def quadweights(self):
        weights = self.quadrule.weights()
        return weights

    def V2QuadTo2m(self):
        n = self.bf2_generator.number_of_basis_functions()
        points = self.quadrule.points()
        weights = self.quadrule.weights()
        m = weights.shape[0]

        E = np.zeros((n, m))
        for i in range(n):
            for j, p in enumerate(points):
                E[i, j] = self.bf2_generator.eval_basis(p, i) * weights[j]

        mass = self.dg2_generator.mass_matrix()

        V = np.linalg.solve(mass, E)

        return V

    def V2mTo2Quad(self):
        n = self.bf2_generator.number_of_basis_functions()
        points = self.quadrule.points()
        m = points.shape[0]

        E = np.zeros((m, n))
        for i in range(n):
            for j, p in enumerate(points):
                E[j, i] = self.bf2_generator.eval_basis(p, i)

        return E

    def resample(self):
        return self.V2mTo2Quad() @ self.V2QuadTo2m()


if __name__ == "__main__":
    from seissol_matrices import json_io

    for order in range(1, 8):
        for quadrule in [
            quad_points.stroud(order + 1),
            quad_points.dunavant(order + 1),
            quad_points.witherden_vincent(order + 1),
        ]:
            generator = dr_generator(order, quadrule)

            filename = f"dr_{quadrule.name}_matrices_{order}.json"
            quadpoints = generator.quadpoints()
            quadweights = generator.quadweights()
            resample = generator.resample()
            V2QuadTo2m = generator.V2QuadTo2m()
            V2mTo2Quad = generator.V2mTo2Quad()

            json_io.write_matrix(quadpoints, "quadpoints", filename)
            json_io.write_matrix(quadweights.reshape(-1, 1), "quadweights", filename)
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
