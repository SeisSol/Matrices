import numpy as np
import quadpy as qp

import basis_functions

def mass_matrix(order):
    #TODO: Choose a quadrature formula according to the order
    scheme = qp.tetrahedron.xiao_gimbutas_15()
    number_of_3d_basis_functions = basis_functions.number_of_3d_basis_functions(order)
    M = np.zeros((number_of_3d_basis_functions, number_of_3d_basis_functions))
    for i in range(number_of_3d_basis_functions):
        for j in range(number_of_3d_basis_functions):
            prod = lambda x: basis_functions.eval_basis_3d(x, basis_functions.number_to_index_3d(i)) * \
                    basis_functions.eval_basis_3d(x, basis_functions.number_to_index_3d(j))
            M[i,j] = scheme.integrate(prod, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return M

def stiffness_matrix(order, k):
    #TODO: Choose a quadrature formula according to the order
    scheme = qp.tetrahedron.xiao_gimbutas_15()
    number_of_3d_basis_functions = basis_functions.number_of_3d_basis_functions(order)
    K = np.zeros((number_of_3d_basis_functions, number_of_3d_basis_functions))
    for i in range(number_of_3d_basis_functions):
        for j in range(number_of_3d_basis_functions):
            prod = lambda x: basis_functions.eval_basis_3d(x, basis_functions.number_to_index_3d(i)) * \
                    basis_functions.eval_diff_basis_3d(x, basis_functions.number_to_index_3d(j), k)
            K[i,j] = scheme.integrate(prod, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return K

