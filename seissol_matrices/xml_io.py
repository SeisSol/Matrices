import numpy as np
import xml.etree.ElementTree as ET
import os


def read_xml(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    file_txt = " ".join([line for line in lines])
    tree = ET.ElementTree(ET.fromstring(file_txt))
    root = tree.getroot()
    return root

def write_xml(root, filename):
    ET.indent(root, "  ")
    tree = ET.ElementTree(root)
    xmlstr = ET.tostring(root, encoding="utf8", method="xml", xml_declaration=True).decode("utf-8")
    with open(filename, "w+") as f:
        f.write(xmlstr)
        f.write("\n")


def create_empty_xml_file(filename):
    root = ET.Element( "matrices")
    write_xml(root, filename)



def write_matrix(matrix, matrixname, filename):
    if not os.path.exists(filename):
        create_empty_xml_file(filename)

    root = read_xml(filename)
    if not root.tag == "matrices":
        raise ValueError("Try to append matrix to an XML file, which does not have <matrices> as root tag.")
    
    rows, columns = matrix.shape
    matrix_root = ET.SubElement(root,
        "matrix", {"name": matrixname, "columns": str(columns), "rows": str(rows)}
    )
    for r in range(rows):
        for c in range(columns):
            if np.abs(matrix[r, c]) > 1e-14:
                ET.SubElement(
                    matrix_root,
                    "entry",
                    {
                        "column": str(c + 1),
                        "row": str(r + 1),
                        "value": str(matrix[r, c]),
                    },
                )
    write_xml(root, filename)



def read_matrix(matrixname, filename):
    root = read_xml(filename)
    if not root.tag == "matrices":
        raise ValueError("Root element is not `matrices`")

    for matrix_root in root:

        if matrix_root.tag == "matrix" and matrix_root.attrib["name"] == matrixname: 
            rows = int(matrix_root.attrib["rows"])
            columns = int(matrix_root.attrib["columns"])
            matrix = np.zeros((rows, columns))
        
            for child in matrix_root:
                if child.tag == "entry":
                    row = int(child.attrib["row"]) - 1
                    column = int(child.attrib["column"]) - 1
                    value = float(child.attrib["value"])
                    matrix[row, column] = value

    return matrix
