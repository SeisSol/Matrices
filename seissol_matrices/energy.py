#!/bin/env python3
import quadpy as qp
import numpy as np
from seissol_matrices import basis_functions, quadrature, xml_io

import json

def parse_seissol_json(s):
    rows = s["rows"]
    columns = s["columns"]
    matrix = np.zeros((rows, columns))
    for entry in s["entries"]:
        i = entry[0] - 1
        j = entry[1] - 1
        val = entry[2]
        matrix[i,j] = val

    print(matrix)
    return matrix

def generate_V2mTo2n(order, quadrule):
    #order = 4
    #path = f"/home/lukas/src/SeisSol/generated_code/matrices/nodal/nodalBoundary_matrices_{order}.json"

    #with open(path) as f:
        #matrices = json.load(f)
        #
        #nodes_2d = None
        #reference = None
        #for matrix in matrices:
            #name = matrix["name"]
            #print(name)
            #if name == "nodes2D":
                #nodes_2d = parse_seissol_json(matrix)
            #if name == "V2mTo2n":
                #reference = parse_seissol_json(matrix)

    bf2_generator = basis_functions.BasisFunctionGenerator2D(order)
    m = bf2_generator.number_of_basis_functions()

    points = quadrule.points()
    #points = nodes_2d

    n = points.shape[0]
    matrix = np.zeros((n, m))
    for i, p in enumerate(points):
        for j in range(m):
            matrix[i,j] = bf2_generator.eval_basis([p[0], p[1]], j)

    #print(reference - matrix)
    #print(reference)
    #print(matrix)
    return matrix

def main():
    order = 2
    quadrule = quadrature.gauss_jacobi(order + 1)
    V2mTo2n = generate_V2mTo2n(order=order, quadrule=quadrule)
    xml_io.write_matrix(V2mTo2n, "V2nTo2m", "tmp.xml")

if __name__ == '__main__':
    main()