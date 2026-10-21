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
    def __init__(self, order, family="dubiner"):
        self.bf3_generator = basis_functions.generator(family, 3, order)
        self.dg3_generator = dg_matrices.dg_generator(order, d=3, family=family)
        self.order = order

    def quadrature_rule(self, mode):
        """
        The points and weights (on the unit tetrahedron) of the quadrature rule
        a mode projects with:
        ip:  the conical-product (Stroud) rule with order + 1 points per direction
        sq:  the fully symmetric rule with the fewest points that is exact for the
             product of two basis functions, i.e. up to degree 2 * (order - 1)
        sq3: the same for the product of three basis functions, i.e. up to degree
             3 * (order - 1), so that the projection of a source quadratic in the
             state is exact, and at least up to the degree of ip, 2 * order + 1,
             which is the higher one below order 4
        """
        assert mode in ["ip", "sq", "sq3"]
        if mode == "ip":
            return stroud.stroud(self.order + 1)
        if mode == "sq":
            degree = 2 * (self.order - 1)
        else:
            degree = max(3 * (self.order - 1), 2 * self.order + 1)
        points, weights = quad_rules.SymmetricTet.SymmetricTet().find_best_rule(degree)
        return quad_rules.Quadrature.transform(points, weights, UNIT_TETRAHEDRON)

    def nodes(self, mode):
        assert mode in ["ip", "nb", "sq", "sq3"]
        if mode == "nb":
            nodes = parse_nodes(
                f"{os.path.dirname(__file__)}/plasticity/nb_{self.order}.txt"
            )
        else:
            nodes, _ = self.quadrature_rule(mode)

        return nodes

    def generate_Vandermonde(self, mode):
        return self.bf3_generator.eval_all(self.nodes(mode).T)

    def generate_VandermondeDerivative(self, mode):
        nodes = self.nodes(mode).T
        return np.stack([self.bf3_generator.eval_diff_all(nodes, k) for k in range(3)])

    def generate_Vandermonde_inv(self, mode):
        """
        The map from values at the points back to the basis: the inverse of the
        Vandermonde matrix for the nodal points (nb), and the L2 projection with
        the quadrature rule otherwise, M^-1 V^T W.
        """
        vandermonde = self.generate_Vandermonde(mode)
        if mode == "nb":
            return np.linalg.solve(vandermonde, np.eye(vandermonde.shape[0]))
        mass = self.dg3_generator.mass_matrix()
        _, weights = self.quadrature_rule(mode)
        return np.linalg.solve(mass, vandermonde.T * weights[None, :])


if __name__ == "__main__":
    for mode in ["nb", "ip", "sq", "sq3"]:
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
