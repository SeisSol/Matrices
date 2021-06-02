import numpy as np
import quadpy as qp

from . import basis_functions
from . import writer

### Generates matrices which are of general interest for DG methods: 
### Mass and Stiffness matrices
### kDivM and kDivMT which are SeisSol specific
class dg_generator:
    def __init__(self, o, d):
        self.order = o
        self.dim = d

        if self.dim == 3:
            self.generator = basis_functions.BasisFunctionGenerator3D(self.order)
            self.scheme = qp.t3.get_good_scheme(self.order+1)
            self.geometry = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        elif self.dim == 2:
            self.generator = basis_functions.BasisFunctionGenerator2D(self.order)
            self.scheme = qp.t2.get_good_scheme(self.order+1)
            self.geometry = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        elif self.dim == 1:
            self.generator = basis_functions.BasisFunctionGenerator1D(self.order)
            self.scheme = qp.c1.gauss_legendre(self.order+1)
            self.geometry = [0.0, 1.0]
        else:
            raise Execption('Can only generate 1D, 2D or 2D basis functions')

        self.M = None
        self.K = self.dim * [None]

    def mass_matrix(self):
        number_of_basis_functions = self.generator.number_of_basis_functions()
        M = np.zeros((number_of_basis_functions, number_of_basis_functions))
    
        for i in range(number_of_basis_functions):
            for j in range(number_of_basis_functions):
                prod = lambda x: self.generator.eval_basis(x, i) * \
                        self.generator.eval_basis(x, j)
                M[i,j] = self.scheme.integrate(prod, self.geometry)
        return M

    def stiffness_matrix(self, k):
        number_of_basis_functions = self.generator.number_of_basis_functions()
        K = np.zeros((number_of_basis_functions, number_of_basis_functions))

        for i in range(number_of_basis_functions):
            for j in range(number_of_basis_functions):
                prod = lambda x: self.generator.eval_diff_basis(x, i, k) * \
                        self.generator.eval_basis(x, j)
                K[i,j] = self.scheme.integrate(prod, self.geometry)
        return K

    def kDivM(self, dim):
        if np.any(self.K[dim] == None):
            self.K[dim] = self.stiffness_matrix(dim)
        if np.any(self.M == None):
            self.M = self.mass_matrix()

        return np.linalg.solve(self.M, self.K[dim])

    def kDivMT(self, dim):
        if np.any(self.K[dim] == None):
            self.K[dim] = self.stiffness_matrix(dim)
        if np.any(self.M == None):
            self.M = self.mass_matrix()

        return np.linalg.solve(self.M, self.K[dim].T)

if __name__ == '__main__':
    generator = dg_generator(3, 3)

    writer.write_matrix_to_xml(generator.kDivM(2), "kDivM(2)")

    
