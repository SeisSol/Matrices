#!/bin/env python3
from seissol_matrices import basis_functions, quad_points, xml_io

import json
import sys


def generate_V2mTo2n(order, quadrule, family="dubiner"):
    bf2_generator = basis_functions.generator(family, 2, order)
    return bf2_generator.eval_all(quadrule.points().T)


def main():
    for order in range(2, 8):
        output_file = f"output/gravitational_energy_matrices_{order}.xml"

        quadrule = quad_points.stroud(order + 1)
        V2mTo2n = generate_V2mTo2n(order=order, quadrule=quadrule)
        xml_io.write_matrix(V2mTo2n, "V2mTo2JacobiQuad", output_file)


if __name__ == "__main__":
    main()
