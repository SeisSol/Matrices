#!/usr/bin/env python3

import numpy as np
import os
from seissol_matrices import basis_functions, dg_matrices, quadrature, json_io


def parse_nodes(filename):
    with open(filename) as f:
        lines = [[float(e) for e in l.rstrip().split()] for l in f]
        return np.array(lines)


class NodalBoundaryGenerator:
    def __init__(self, order):
        self.bf2_generator = basis_functions.BasisFunctionGenerator2D(order)
        self.dg2_generator = dg_matrices.dg_generator(order, d=2)
        self.dg3_generator = dg_matrices.dg_generator(order, d=3)

        self.nodes = parse_nodes(f"{os.path.dirname(__file__)}/nodal/nodes_{order}.txt")
        self.vandermonde = self.generate_Vandermonde()
        self.mass_matrix_2d = self.dg2_generator.mass_matrix()
        self.rTs = [self.dg3_generator.rT(faceid) for faceid in range(4)]

    def generate_Vandermonde(self):
        m = self.bf2_generator.number_of_basis_functions()
        assert self.nodes.shape[0] == m

        vandermonde = np.zeros((m, m))

        for i in range(m):
            for j in range(m):
                n = self.nodes[i]
                vandermonde[i, j] = self.bf2_generator.eval_basis([n[0], n[1]], j)

        return vandermonde

    def generate_V2mTo2n(self):
        return self.vandermonde

    def generate_V2nTo2m(self):
        return self.mass_matrix_2d @ np.linalg.inv(self.vandermonde)

    def generate_MV2nTo2m(self):
        return np.linalg.inv(self.vandermonde)

    def generate_V3mTo2nFace(self, faceid):
        return self.vandermonde @ self.rTs[faceid]


def main():
    for order in range(2, 8):
        nb_generator = NodalBoundaryGenerator(order)
        filename = f"output/nodalBoundary_matrices_{order}.json"
        json_io.write_matrix(nb_generator.nodes, "nodes2D", filename)
        json_io.write_matrix(nb_generator.generate_V2mTo2n(), "V2mTo2n", filename)
        for faceid in range(4):
            json_io.write_matrix(
                nb_generator.generate_V3mTo2nFace(faceid),
                f"V3mTo2nFace({faceid})",
                filename,
            )


if __name__ == "__main__":
    main()
