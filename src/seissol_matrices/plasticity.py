#!/usr/bin/env python3

import numpy as np
import os
from seissol_matrices import basis_functions, dg_matrices, json_io


def parse_nodes(filename):
    with open(filename) as f:
        lines = [[float(e) for e in l.rstrip().split()] for l in f]
        return np.array(lines)


class PlasticityGenerator:
    def __init__(self, order):
        self.bf3_generator = basis_functions.BasisFunctionGenerator3D(order)
        self.dg3_generator = dg_matrices.dg_generator(order, d=3)
        self.order = order

    def generate_Vandermonde(self, mode):
        assert mode in ["ip", "nb"]
        nodes = parse_nodes(
            f"{os.path.dirname(__file__)}/plasticity/{mode}_{self.order}.txt"
        )
        m = self.bf3_generator.number_of_basis_functions()
        assert nodes.shape[0] == m

        vandermonde = np.zeros((m, m))

        for i in range(m):
            for j in range(m):
                n = nodes[i]
                vandermonde[i, j] = self.bf3_generator.eval_basis([n[0], n[1], n[2]], j)

        return vandermonde

    def generate_Vandermonde_inv(self, mode):
        vandermonde = self.generate_Vandermonde(mode)
        vandermonde_inv = np.linalg.solve(vandermonde, np.eye(vandermonde.shape[0]))
        return vandermonde_inv


if __name__ == "__main__":
    for mode in ["nb"]:
        for order in range(2, 8):
            generator = PlasticityGenerator(order)
            nb_vandermonde = generator.generate_Vandermonde(mode)
            nb_vandermonde_inv = generator.generate_Vandermonde_inv(mode)
            filename = f"output/plasticity_{mode}_matrices_{order}.json"
            xml_io.write_matrix(nb_vandermonde, "v", filename)
            xml_io.write_matrix(nb_vandermonde_inv, "vInv", filename)
