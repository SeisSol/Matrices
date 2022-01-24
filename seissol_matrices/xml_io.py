import numpy as np

def write_matrix_to_xml(matrix, matrixname, filename):
    with open(filename, 'w+') as f:
        rows, columns = matrix.shape
        f.write(f'<matrix columns="{columns}" name="{matrixname}" rows="{rows}">\n')
        for r in range(rows):
            for c in range(columns):
                if np.abs(matrix[r,c]) > 1e-14:
                    f.write(f'  <entry column="{c+1}" row="{r+1}" value="{matrix[r,c]}"/>\n')
        f.write('</matrix>\n')

