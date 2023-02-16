#!/bin/env python3
import quadpy as qp
import numpy as np
from seissol_matrices import basis_functions, quadrature, xml_io

import json
import sys

def generate_V2mTo2n(order, quadrule):
    bf2_generator = basis_functions.BasisFunctionGenerator2D(order)
    m = bf2_generator.number_of_basis_functions()

    points = quadrule.points()

    n = points.shape[0]
    matrix = np.zeros((n, m))
    for i, p in enumerate(points):
        for j in range(m):
            matrix[i, j] = bf2_generator.eval_basis([p[0], p[1]], j)

    return matrix


def main():
    for order in range(2, 8):
        output_file = f"output/gravitational_energy_matrices_{order}.xml"

        quadrule = quadrature.gauss_jacobi(order + 1)
        V2mTo2n = generate_V2mTo2n(order=order, quadrule=quadrule)
        xml_io.write_matrix(V2mTo2n, "V2mTo2JacobiQuad", output_file)


if __name__ == "__main__":
    main()
