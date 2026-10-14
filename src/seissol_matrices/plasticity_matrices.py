#!/usr/bin/env python3

import numpy as np
import os
import quad_rules.Quadrature
import quad_rules.SymmetricTet
from seissol_matrices import basis_functions, dg_matrices, json_io
from seissol_matrices.plasticity import stroud

UNIT_TETRAHEDRON = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)


def parse_nodes(filename):
    with open(filename) as f:
        lines = [[float(e) for e in l.rstrip().split()] for l in f]
        return np.array(lines)


class PlasticityGenerator:
    def __init__(self, order):
        self.bf3_generator = basis_functions.BasisFunctionGenerator3D(order)
        self.dg3_generator = dg_matrices.dg_generator(order, d=3)
        self.order = order

    def quadrature_rule(self, mode):
        """
        The points and weights (on the unit tetrahedron) of the quadrature rule
        a mode projects with:
        ip: the conical-product (Stroud) rule with order + 1 points per direction
        sq: the fully symmetric rule with the fewest points that is exact for the
            product of two basis functions, i.e. up to degree 2 * (order - 1)
        """
        assert mode in ["ip", "sq"]
        if mode == "ip":
            return stroud.stroud(self.order + 1)
        points, weights = quad_rules.SymmetricTet.SymmetricTet().find_best_rule(
            2 * (self.order - 1)
        )
        return quad_rules.Quadrature.transform(points, weights, UNIT_TETRAHEDRON)

    def nodes(self, mode):
        assert mode in ["ip", "nb", "sq"]
        if mode == "nb":
            nodes = parse_nodes(
                f"{os.path.dirname(__file__)}/plasticity/nb_{self.order}.txt"
            )
        else:
            nodes, _ = self.quadrature_rule(mode)

        return nodes

    def generate_Vandermonde(self, mode):
        nodes = self.nodes(mode)
        m = self.bf3_generator.number_of_basis_functions()
        n = nodes.shape[0]

        vandermonde = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                node = nodes[i]
                vandermonde[i, j] = self.bf3_generator.eval_basis(
                    [node[0], node[1], node[2]], j
                )

        return vandermonde

    def generate_VandermondeDerivative(self, mode):
        nodes = self.nodes(mode)
        m = self.bf3_generator.number_of_basis_functions()
        n = nodes.shape[0]

        vandermonde = np.zeros((3, n, m))

        for k in range(3):
            for i in range(n):
                for j in range(m):
                    node = nodes[i]
                    vandermonde[k, i, j] = self.bf3_generator.eval_diff_basis(
                        [node[0], node[1], node[2]], j, k
                    )

        return vandermonde

    def generate_Vandermonde_inv(self, mode):
        vandermonde = self.generate_Vandermonde(mode)
        if mode == "nb":
            vandermonde_inv = np.linalg.solve(vandermonde, np.eye(vandermonde.shape[0]))
        else:
            mass = self.dg3_generator.mass_matrix()
            nodes, weights = self.quadrature_rule(mode)
            vandermonde_inv = np.zeros(vandermonde.T.shape)
            for l in range(vandermonde_inv.shape[0]):
                for i in range(vandermonde_inv.shape[1]):
                    vandermonde_inv[l, i] = vandermonde[i, l] * weights[i] / mass[l, l]

        return vandermonde_inv


if __name__ == "__main__":
    for mode in ["nb", "ip", "sq"]:
        for order in range(2, 9):
            generator = PlasticityGenerator(order)
            nodes = generator.nodes(mode)
            vandermonde = generator.generate_Vandermonde(mode)
            vandermondeDerivative = generator.generate_VandermondeDerivative(mode)
            vandermonde_inv = generator.generate_Vandermonde_inv(mode)
            filename = f"output/plasticity-{mode}-matrices-{order}.json"
            json_io.write_matrix(nodes, "vNodes", filename)
            json_io.write_matrix(vandermonde, "v", filename)
            json_io.write_matrix(vandermondeDerivative, "vD", filename)
            json_io.write_matrix(vandermonde_inv, "vInv", filename)
