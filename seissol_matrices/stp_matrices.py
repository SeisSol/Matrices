import numpy as np
import quadpy as qp

from seissol_matrices import basis_functions
from seissol_matrices import dg_matrices

### generates matrices which are needed for the space time predictor:
### Z
### value at 0
### time integrated

class stp_generator:
    def __init__(self, o):
        self.order = o

        self.generator = basis_functions.BasisFunctionGenerator1D(self.order)
        self.scheme = qp.c1.gauss_legendre(self.order+1)
        self.geometry = [0.0, 1.0]
        self.dg_generator = dg_matrices.dg_generator(self.order, 1)
        self.S = self.dg_generator.mass_matrix()


    def w(self):
        number_of_basis_functions = self.generator.number_of_basis_functions()
        w = np.zeros((number_of_basis_functions,))
        for i in range(number_of_basis_functions):
            w[i] = self.generator.eval_basis(0, i)
        return w

    def S(self):
        return self.S

    def K(self):
        return self.dg_generator.stiffness_matrix(0)

    def S_hat(self):
        W = self.W()
        K_t = self.dg_generator.stiffness_matrix(0)
        return np.linalg.solve(W - K_t, self.S)

    def W(self):
        number_of_basis_functions = self.generator.number_of_basis_functions()
        W = np.zeros((number_of_basis_functions,number_of_basis_functions))
    
        for i in range(number_of_basis_functions):
            for j in range(number_of_basis_functions):
                W[i,j] = self.generator.eval_basis(1, i) * \
                    self.generator.eval_basis(1, j)

        return W

    def Z(self):
        W = self.W()
        K_t = self.dg_generator.stiffness_matrix(0)

        return np.linalg.solve(self.S, W-K_t)

    def time_int(self):
        number_of_basis_functions = self.generator.number_of_basis_functions()
        t = np.zeros((number_of_basis_functions,))
        t[0] = 1

        return t

if __name__ == '__main__':
    from seissol_matrices import xml_io
    generator = stp_generator(5)
    xml_io.write_matrix_to_xml(generator.Z(), "Z", "z.xml")


    
