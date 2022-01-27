import numpy as np
import json
import os


def read_json(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    file_txt = " ".join([line for line in lines])
    return json.loads(file_txt)


def read_matrix(matrixname, filename):
    matrix_list = read_json(filename)
    for raw_matrix in matrix_list:
        if raw_matrix["name"] == matrixname:
            rows = int(raw_matrix["rows"])
            columns = int(raw_matrix["columns"])
            matrix = np.zeros((rows, columns))
            for entry in raw_matrix["entries"]:
                i = entry[0] - 1
                j = entry[1] - 1
                value = entry[2]
                matrix[i, j] = value
            return matrix
    raise ValueError(f"Matrix with name {matrixname} not found in {filename}.")
