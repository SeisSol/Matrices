import numpy as np
import xml.etree.ElementTree as ET


def write_matrix(matrix, matrixname, filename):
    rows, columns = matrix.shape
    root = ET.Element(
        "matrix", {"name": matrixname, "columns": str(columns), "rows": str(rows)}
    )
    for r in range(rows):
        for c in range(columns):
            if np.abs(matrix[r, c]) > 1e-14:
                ET.SubElement(
                    root,
                    "entry",
                    {
                        "column": str(c + 1),
                        "row": str(r + 1),
                        "value": str(matrix[r, c]),
                    },
                )
    ET.indent(root, "  ")
    tree = ET.ElementTree(root)
    tree.write(filename)


def read_matrix(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        file_txt = " ".join([line for line in lines])
    tree = ET.ElementTree(ET.fromstring(file_txt))
    root = tree.getroot()
    if not root.tag == "matrix":
        raise ValueError("Root element is not `matrix`")

    rows = int(root.attrib["rows"])
    columns = int(root.attrib["columns"])
    matrix = np.zeros((rows, columns))

    for child in root:
        if child.tag == "entry":
            row = int(child.attrib["row"]) - 1
            column = int(child.attrib["column"]) - 1
            value = float(child.attrib["value"])
            matrix[row, column] = value

    return matrix, root.attrib["name"]
