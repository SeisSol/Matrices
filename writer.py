import numpy as np

def write_matrix_to_xml(M, name):
    rows, columns = M.shape
    print('<matrix columns="{}" name="{}" rows="{}">'.format(columns, name, rows))
    for r in range(rows):
        for c in range(columns):
            if np.abs(M[r,c]) > 1e-14:
                print('  <entry column="{}" row="{}" value="{}"/>'.format(c+1, r+1, M[r, c]))
    print('</matrix>')
