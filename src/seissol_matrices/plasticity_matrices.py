#!/usr/bin/env python3

import numpy as np
import os
from seissol_matrices import basis_functions, dg_matrices, json_io
from seissol_matrices.plasticity import stroud


def parse_nodes(filename):
    with open(filename) as f:
        lines = [[float(e) for e in l.rstrip().split()] for l in f]
        return np.array(lines)


class PlasticityGenerator:
    def __init__(self, order):
        self.bf3_generator = basis_functions.BasisFunctionGenerator3D(order)
        self.dg3_generator = dg_matrices.dg_generator(order, d=3)
        self.order = order

    def nodes(self, mode):
        assert mode in ["ip", "nb"]
        if mode == "nb":
            nodes = parse_nodes(
                f"{os.path.dirname(__file__)}/plasticity/nb_{self.order}.txt"
            )
        else:
            nodes, _ = stroud.stroud(self.order + 1)

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
            nodes, weights = stroud.stroud(self.order + 1)
            vandermonde_inv = np.zeros(vandermonde.T.shape)
            for l in range(vandermonde_inv.shape[0]):
                for i in range(vandermonde_inv.shape[1]):
                    vandermonde_inv[l, i] = vandermonde[i, l] * weights[i] / mass[l, l]

        return vandermonde_inv


if __name__ == "__main__":
    for mode in ["nb", "ip"]:
        for order in range(2, 9):
            generator = PlasticityGenerator(order)
            vandermonde = generator.generate_Vandermonde(mode)
            vandermondeDerivative = generator.generate_VandermondeDerivative(mode)
            vandermonde_inv = generator.generate_Vandermonde_inv(mode)
            filename = f"output/plasticity_{mode}_matrices_{order}.json"
            json_io.write_matrix(vandermonde, "v", filename)
            json_io.write_matrix(vandermondeDerivative, "vD", filename)
            json_io.write_matrix(vandermonde_inv, "vInv", filename)
