import numpy as np
import json
import os


def read_json(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    file_txt = " ".join([line for line in lines])
    return json.loads(file_txt)

def write_json(content, filename):
    with open(filename, "w+") as f:
        json.dump(content, f)


def create_empty_json_file(filename):
    write_json([], filename)


def write_matrix(matrix, matrixname, filename):
    if not os.path.exists(filename):
        create_empty_json_file(filename)

    matrix_list = read_json(filename)
    assert(isinstance(matrix_list, list))

    rows, columns = matrix.shape
    entries = []
    for r in range(rows):
        for c in range(columns):
            if np.abs(matrix[r, c]) > 1e-14:
                entries.append([r+1, c+1, str(matrix[r, c])])
    m = {"name": matrixname, "rows": rows, "columns": columns, "entries": entries}
    matrix_list.append(m)
    write_json(matrix_list, filename)


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
